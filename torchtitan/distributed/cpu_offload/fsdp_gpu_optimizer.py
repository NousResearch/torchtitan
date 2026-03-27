# Copyright (c) 2025 Nous Research. All rights reserved.

"""FSDP CPU offload with GPU optimizer.

FSDP's CPUOffloadPolicy saves massive memory by keeping params+optimizer on CPU.
But the optimizer step runs on CPU — which is 10-90x slower than GPU.

This module wraps a GPU optimizer to work WITH FSDP CPU offload:
1. FSDP handles weight lifecycle (CPU storage, H2D before allgather, D2H after reshard)
2. Before optimizer.step(): copy sharded grads GPU→GPU (they're already on GPU from reduce-scatter)
3. Run optimizer.step() on GPU (fast)
4. After optimizer.step(): copy updated sharded params back to CPU (for next forward's H2D)

The key insight: FSDP's CPUOffloadPolicy does D2H of grads in reduce-scatter stream.
If we intercept BEFORE the D2H and keep grads on GPU, we can run the optimizer on GPU.
"""

import torch
import torch.nn as nn
from typing import Optional


class FSDPGPUOptimizerWrapper:
    """Wraps an optimizer to run on GPU even when FSDP uses CPUOffloadPolicy.

    After FSDP's backward, sharded gradients end up on CPU (via D2H in
    reduce-scatter stream). This wrapper:
    1. Copies CPU grads → GPU before step
    2. Runs optimizer.step() on GPU
    3. Copies updated GPU params → CPU after step

    Args:
        optimizer: A torch.optim.Optimizer whose params are on CPU (from FSDP offload).
        device: GPU device for optimizer step.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, device: torch.device = None):
        self.optimizer = optimizer
        self.device = device or torch.device("cuda")
        self._gpu_param_map = {}  # cpu_param → gpu_copy
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        self._initialized = False

    def _lazy_init(self):
        """Create GPU copies of all CPU parameters on first step."""
        if self._initialized:
            return
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.device.type == "cpu" and param.grad is not None:
                    gpu_copy = param.data.to(self.device, non_blocking=True)
                    self._gpu_param_map[param] = gpu_copy
        torch.cuda.synchronize()
        self._initialized = True

    def step(self, closure=None):
        """Run optimizer step on GPU, then sync back to CPU."""
        # Copy CPU grads → GPU and run step on GPU copies
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param not in self._gpu_param_map:
                    self._gpu_param_map[param] = param.data.to(self.device)

                gpu_param = self._gpu_param_map[param]

                # Copy grad to GPU
                if param.grad.device.type == "cpu":
                    gpu_grad = param.grad.to(self.device, non_blocking=True)
                else:
                    gpu_grad = param.grad

                # Set GPU copies for optimizer
                gpu_param.grad = gpu_grad

        torch.cuda.synchronize()

        # Temporarily swap params to GPU copies for optimizer step
        original_params = {}
        for group in self.optimizer.param_groups:
            original_params[id(group)] = group["params"]
            gpu_params = []
            for param in group["params"]:
                if param in self._gpu_param_map:
                    gpu_p = self._gpu_param_map[param]
                    gpu_p.requires_grad_(True)
                    gpu_params.append(gpu_p)
                else:
                    gpu_params.append(param)
            group["params"] = gpu_params

        # Run optimizer on GPU
        self.optimizer.step(closure)

        # Restore original CPU params and copy updated values back
        for group in self.optimizer.param_groups:
            gpu_params = group["params"]
            group["params"] = original_params[id(group)]

            for cpu_param, gpu_param in zip(group["params"], gpu_params):
                if cpu_param in self._gpu_param_map:
                    # Async copy updated params back to CPU
                    cpu_param.data.copy_(gpu_param.data, non_blocking=True)
                    self._gpu_param_map[cpu_param] = gpu_param.data

        torch.cuda.synchronize()

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none)
        for gpu_param in self._gpu_param_map.values():
            if set_to_none:
                gpu_param.grad = None
            elif gpu_param.grad is not None:
                gpu_param.grad.zero_()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
