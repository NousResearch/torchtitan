# Copyright (c) 2025, NVIDIA CORPORATION and Alibaba PAI. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
# Modifications: Extracted as standalone module — no Megatron dependencies.

"""HybridDeviceOptimizer — split optimizer across GPU and CPU.

Offloads a configurable fraction of parameters to CPU, runs their optimizer
steps on CPU, and uses dedicated D2H/H2D CUDA streams to overlap gradient
sync and parameter copy-back with GPU optimizer execution.

Usage:
    optimizer = HybridDeviceOptimizer(
        model.parameters(),
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        offload_fraction=0.5,
    )
    optimizer.step()
"""

from collections import defaultdict
from typing import Dict, List, Optional, Type

import torch


def _param_generator(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            yield param


class HybridDeviceOptimizer(torch.optim.Optimizer):
    """Optimizer that splits parameter updates between GPU and CPU.

    Args:
        params: Model parameters.
        offload_fraction: Fraction of GPU parameters to offload to CPU (0.0–1.0).
        cpu_optimizer_cls: Optimizer class for CPU parameters (e.g., torch.optim.AdamW).
        gpu_optimizer_cls: Optimizer class for GPU parameters.
        param_update_in_fp32: Cast parameters to FP32 for optimizer step.
        pin_cpu_grads: Pin CPU gradient buffers for async D2H.
        pin_cpu_params: Pin CPU parameter copies.
        overlap_cpu_optimizer_d2h_h2d: Use separate streams to overlap transfers.
        **kwargs: Additional kwargs passed to both optimizer classes.
    """

    def __init__(
        self,
        params,
        offload_fraction: float = 0.5,
        cpu_optimizer_cls: Optional[Type[torch.optim.Optimizer]] = None,
        gpu_optimizer_cls: Optional[Type[torch.optim.Optimizer]] = None,
        param_update_in_fp32: bool = False,
        pin_cpu_grads: bool = True,
        pin_cpu_params: bool = True,
        overlap_cpu_optimizer_d2h_h2d: bool = True,
        **kwargs,
    ):
        super().__init__(
            params,
            defaults={
                "offload_fraction": offload_fraction,
                "cpu_optimizer_cls": cpu_optimizer_cls,
                "gpu_optimizer_cls": gpu_optimizer_cls,
                "param_update_in_fp32": param_update_in_fp32,
                "pin_cpu_grads": pin_cpu_grads,
                "pin_cpu_params": pin_cpu_params,
                "overlap_cpu_optimizer_d2h_h2d": overlap_cpu_optimizer_d2h_h2d,
                **kwargs,
            },
        )

        self.offload_fraction = offload_fraction
        self.cpu_optimizer_cls = cpu_optimizer_cls
        self.gpu_optimizer_cls = gpu_optimizer_cls
        self.pin_cpu_grads = pin_cpu_grads
        self.pin_cpu_params = pin_cpu_params
        self.overlap_cpu_optimizer_d2h_h2d = overlap_cpu_optimizer_d2h_h2d
        self.param_update_in_fp32 = param_update_in_fp32
        self.sub_optimizer_kwargs = kwargs

        self._init_sub_optimizers()
        self._register_load_state_dict_hooks()

    # ── Gradient sync (GPU → CPU) ──────────────────────────────────────

    def _set_sub_optimizer_grads(self):
        if self.param_update_in_fp32:
            for param in self.param_to_fp32_param:
                if param in self.gpu_params_map_cpu_copy:
                    continue
                fp32_param = self.param_to_fp32_param[param]
                grad = getattr(param, "decoupled_grad", param.grad)
                if grad is not None:
                    fp32_param.grad = grad.to(fp32_param.dtype)
                    fp32_param.requires_grad = True
                else:
                    fp32_param.requires_grad = False

        for optimizer in self.cpu_optimizers:
            for param in _param_generator(optimizer):
                gpu_param = self.cpu_copys_map_gpu_param[param]
                grad = getattr(gpu_param, "decoupled_grad", gpu_param.grad)
                if grad is None:
                    param.requires_grad = False
                    continue

                param.requires_grad = False
                if param not in self.cpu_copy_map_grad:
                    self.cpu_copy_map_grad[param] = torch.empty(
                        param.shape,
                        dtype=param.dtype,
                        pin_memory=self.pin_cpu_grads,
                        device="cpu",
                    )
                    param.grad = self.cpu_copy_map_grad[param]

                self.cpu_copy_map_grad[param].data.copy_(grad, non_blocking=True)
            self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()

    # ── Parameter copy-back (CPU → GPU) ────────────────────────────────

    def _register_param_copy_back_gpu_hook(self):
        def param_copy_back_hook_closure():
            def hook(optimizer, args, kwargs):
                self._h2d_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._h2d_stream):
                    for param in _param_generator(optimizer):
                        gpu_param = self.cpu_copys_map_gpu_param[param]
                        gpu_param.data.copy_(param.data, non_blocking=True)
                self._h2d_stream.record_event().wait(torch.cuda.current_stream())

            return hook

        def fp32_copy_back_hook_closure():
            def hook(optimizer, args, kwargs):
                for group in self.param_groups:
                    for param in group["params"]:
                        if param in self.gpu_params_map_cpu_copy:
                            continue
                        if param in self.param_to_fp32_param:
                            fp32_param = self.param_to_fp32_param[param]
                            param.data.copy_(fp32_param.data)

            return hook

        for optimizer in self.sub_optimizers:
            if optimizer is not self.gpu_optimizer:
                optimizer.register_step_post_hook(param_copy_back_hook_closure())
            elif self.param_update_in_fp32:
                optimizer.register_step_post_hook(fp32_copy_back_hook_closure())

    # ── Optimizer step ─────────────────────────────────────────────────

    def step(self, closure=None):
        self._sync_hdo_param_groups_to_sub_optimizers()

        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            self._set_sub_optimizer_grads()

        if self.gpu_optimizer:
            self.gpu_optimizer.step(closure)

        for cpu_opt in self.cpu_optimizers:
            event = self._cpu_optimizer_map_data_event.pop(cpu_opt, None)
            if event is not None:
                event.synchronize()
            cpu_opt.step(closure)

        self._sync_sub_optimizers_state_to_hdo()

    # ── Sub-optimizer initialization ───────────────────────────────────

    def _init_sub_optimizers(self):
        (
            self.cpu_param_groups,
            self.gpu_param_groups,
            self.gpu_params_map_cpu_copy,
            self.cpu_copys_map_gpu_param,
            self.param_to_fp32_param,
        ) = self._get_sub_optimizer_param_groups(self.offload_fraction)

        self.param_to_inner_param = {}
        self.inner_param_to_orig_param = {}
        for group in self.param_groups:
            for param in group["params"]:
                if param in self.param_to_fp32_param:
                    inner = self.param_to_fp32_param[param]
                elif param in self.gpu_params_map_cpu_copy:
                    inner = self.gpu_params_map_cpu_copy[param]
                else:
                    inner = param
                self.param_to_inner_param[param] = inner
                self.inner_param_to_orig_param[inner] = param
        self.fp32_param_to_orig_param = {v: k for k, v in self.param_to_fp32_param.items()}

        self.cpu_optimizers = []
        if self.overlap_cpu_optimizer_d2h_h2d:
            self.cpu_optimizers = self._build_cpu_optimizer_list(
                self.cpu_optimizer_cls, self.cpu_param_groups
            )
        elif self.cpu_param_groups:
            self.cpu_optimizers = [self.cpu_optimizer_cls(self.cpu_param_groups)]

        self.gpu_optimizer = (
            self.gpu_optimizer_cls(self.gpu_param_groups)
            if self.gpu_param_groups
            else None
        )

        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)

        # Create dedicated streams if overlapping
        self._d2h_stream = torch.cuda.current_stream()
        self._h2d_stream = torch.cuda.current_stream()
        if self.overlap_cpu_optimizer_d2h_h2d:
            self._d2h_stream = torch.cuda.Stream()
            self._h2d_stream = torch.cuda.Stream()
        self._cpu_optimizer_map_data_event = {}

        self._register_param_copy_back_gpu_hook()

    @staticmethod
    def _build_cpu_optimizer_list(cpu_optimizer_cls, cpu_param_groups):
        """Build one CPU optimizer per parameter for maximum overlap."""
        optimizers = []
        for group in cpu_param_groups:
            defaults = group.copy()
            params = defaults.pop("params")
            if isinstance(params, torch.Tensor):
                params = [params]
            for param in params:
                pg = defaults.copy()
                pg["params"] = [param]
                optimizers.append(cpu_optimizer_cls([pg]))
        return optimizers

    def _get_sub_optimizer_param_groups(self, offload_fraction: float):
        params = [p for g in self.param_groups for p in g["params"]]
        gpu_numel = sum(p.numel() for p in params if p.is_cuda)
        threshold = gpu_numel * offload_fraction
        offloaded = 0

        cpu_groups, gpu_groups = [], []
        gpu_map_cpu, cpu_map_gpu = {}, {}
        fp32_map = {}

        for group in self.param_groups:
            gpu_g = {**group, "params": []}
            cpu_g = {**group, "params": []}
            for param in group["params"]:
                orig = param
                is_cpu_copy = False
                if offloaded < threshold and param.is_cuda:
                    param = param.detach().clone().cpu().pin_memory()
                    offloaded += param.numel()
                    is_cpu_copy = True
                if self.param_update_in_fp32 and param.dtype != torch.float32:
                    param = param.detach().clone().float()
                    fp32_map[orig] = param
                if is_cpu_copy:
                    gpu_map_cpu[orig] = param
                    cpu_map_gpu[param] = orig
                if param.is_cuda:
                    gpu_g["params"].append(param)
                else:
                    cpu_g["params"].append(param)
            if gpu_g["params"]:
                gpu_groups.append(gpu_g)
            if cpu_g["params"]:
                cpu_groups.append(cpu_g)

        return cpu_groups, gpu_groups, gpu_map_cpu, cpu_map_gpu, fp32_map

    # ── State sync helpers ─────────────────────────────────────────────

    def _sync_sub_optimizers_state_to_hdo(self):
        new_state = defaultdict(dict)
        for opt in self.sub_optimizers:
            for param in opt.state:
                orig = self.inner_param_to_orig_param[param]
                new_state[orig] = opt.state[param]
                if self.param_update_in_fp32:
                    new_state[orig]["master_param"] = param
        self.state = new_state

    def _sync_hdo_state_to_sub_optimizers(self):
        for opt in self.sub_optimizers:
            new_state = defaultdict(dict)
            for group in opt.param_groups:
                for param in group["params"]:
                    orig = self.inner_param_to_orig_param[param]
                    new_state[param] = self.state[orig]
            opt.state = new_state
        self._update_fp32_params_by_new_state()
        self._move_new_state_to_right_device()

    def _sync_hdo_param_groups_to_sub_optimizers(self):
        idx_map = {}
        for i, group in enumerate(self.param_groups):
            for param in group["params"]:
                idx_map[self.param_to_inner_param[param]] = i

        for opt in self.sub_optimizers:
            new_groups = []
            for group in opt.param_groups:
                ng = group.copy()
                gi = idx_map[group["params"][0]]
                updates = self.param_groups[gi].copy()
                del updates["params"]
                ng.update(updates)
                new_groups.append(ng)
            opt.param_groups = new_groups

    def _move_new_state_to_right_device(self):
        for opt in self.sub_optimizers:
            for param, state in opt.state.items():
                for k, v in state.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    orig = self.inner_param_to_orig_param.get(param, param)
                    target = "cpu" if isinstance(opt, self.defaults["cpu_optimizer_cls"]) else "cuda"
                    self.state[orig][k] = state[k] = v.to(target)

    def _update_fp32_params_by_new_state(self):
        if not self.param_update_in_fp32:
            return
        for param, v in self.state.items():
            fp32 = self.param_to_fp32_param[param]
            fp32.data.copy_(v["master_param"])

    def update_fp32_param_by_new_param(self):
        for param, fp32 in self.param_to_fp32_param.items():
            fp32.data.copy_(param)

    # ── State dict hooks ───────────────────────────────────────────────

    def _register_load_state_dict_hooks(self):
        def pre_hook(self_opt, state_dict):
            if not self_opt.param_update_in_fp32:
                return state_dict
            new_state = {}
            for param, v in self_opt.state.items():
                param = self_opt.param_to_fp32_param.get(param, param)
                new_state[param] = v
            self_opt.state = new_state
            for group in self_opt.param_groups:
                for i, param in enumerate(group["params"]):
                    group["params"][i] = self_opt.param_to_fp32_param.get(param, param)
            return state_dict

        self.register_load_state_dict_pre_hook(pre_hook)

        def post_hook(self_opt):
            if self_opt.param_update_in_fp32:
                new_state = {}
                for param, v in self_opt.state.items():
                    orig = self_opt.fp32_param_to_orig_param.get(param, param)
                    new_state[orig] = v
                self_opt.state = new_state
                for group in self_opt.param_groups:
                    for i, param in enumerate(group["params"]):
                        group["params"][i] = self_opt.fp32_param_to_orig_param.get(param, param)
            self_opt._init_sub_optimizers()
            self_opt._sync_hdo_param_groups_to_sub_optimizers()
            self_opt._sync_hdo_state_to_sub_optimizers()

        self.register_load_state_dict_post_hook(post_hook)

    # ── Utilities ──────────────────────────────────────────────────────

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none)
        for group in self.param_groups:
            for param in group["params"]:
                if hasattr(param, "decoupled_grad"):
                    if set_to_none:
                        param.decoupled_grad = None
                    else:
                        param.decoupled_grad.zero_()

    def dummy_step(self):
        """Initialize optimizer state by running a dummy step."""
        for group in self.param_groups:
            for param in group["params"]:
                param.grad = torch.randn_like(param)
        self.step()
        self.zero_grad()

    @property
    def sub_optimizers(self):
        if self.gpu_optimizer is not None:
            return self.cpu_optimizers + [self.gpu_optimizer]
        return self.cpu_optimizers
