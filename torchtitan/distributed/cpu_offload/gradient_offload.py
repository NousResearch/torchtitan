# Copyright (c) 2025 Nous Research. All rights reserved.

"""Gradient offloading via autograd hooks.

Moves gradients to pinned CPU memory after backward computation.
Useful when optimizer runs on CPU or to free GPU memory for the
next forward pass during gradient accumulation.

Usage:
    offloader = TensorOffloader()
    grad_offloader = GradientOffloadHook(offloader)

    # Register parameters whose gradients should be offloaded
    grad_offloader.register(model.parameters())

    # Training — gradients auto-offload after backward
    loss.backward()

    # Before optimizer step, reload gradients if needed
    grad_offloader.reload_all()
    optimizer.step()

    # Cleanup
    grad_offloader.clear()
    offloader.reset_pool()
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .tensor_offloader import TensorOffloader, OffloadHandle


class GradientOffloadHook:
    """Offloads parameter gradients to CPU after backward via post-accumulate hooks.

    Args:
        offloader: Shared TensorOffloader for async transfers.
        release_gpu_grad: If True, set param.grad = None after offloading
            to free GPU memory immediately.
    """

    def __init__(
        self,
        offloader: TensorOffloader,
        release_gpu_grad: bool = True,
    ):
        self.offloader = offloader
        self.release_gpu_grad = release_gpu_grad
        self._handles: Dict[torch.nn.Parameter, OffloadHandle] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        self._registered_params: List[torch.nn.Parameter] = []

    def register(self, params) -> None:
        """Register parameters for gradient offloading.

        Args:
            params: Iterable of nn.Parameter (e.g., model.parameters()).
        """
        for param in params:
            if not param.requires_grad:
                continue
            self._registered_params.append(param)
            handle = param.register_post_accumulate_grad_hook(self._grad_hook)
            self._hook_handles.append(handle)

    def _grad_hook(self, param: torch.nn.Parameter) -> None:
        """Called after gradient is accumulated for this parameter."""
        if param.grad is None:
            return
        if param.grad.device.type != "cuda":
            return

        # Async offload gradient to CPU
        handle = self.offloader.offload(param.grad, release_storage=False)
        self._handles[param] = handle

        if self.release_gpu_grad:
            self.offloader.sync_offload()
            # Can't assign CPU tensor to param.grad when param is on CUDA.
            # Set grad to None to free GPU memory; CPU copy lives in _handles.
            param.grad = None

    def reload_all(self) -> None:
        """Reload all offloaded gradients back to GPU.

        Call before optimizer.step() if optimizer runs on GPU.
        """
        for param, handle in self._handles.items():
            gpu_grad = self.offloader.reload(handle)
            self.offloader.sync_reload()
            param.grad = gpu_grad
        self._handles.clear()

    def has_offloaded_grads(self) -> bool:
        """Check if any gradients are currently on CPU."""
        return len(self._handles) > 0

    @property
    def offloaded_params(self) -> List[torch.nn.Parameter]:
        """List of parameters with offloaded gradients."""
        return list(self._handles.keys())

    def clear(self) -> None:
        """Discard all offloaded gradient handles."""
        self._handles.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._registered_params.clear()
        self._handles.clear()
