# Copyright (c) 2025 Nous Research. All rights reserved.

"""Clean offload API — model code never touches streams, events, or resize_().

Two use cases, same engine:

1. ACTIVATION OFFLOADING (like Megatron's off_interface):
   Offloads tensors saved by autograd between forward and backward.

   with offload_activation(should_offload, input_tensor, "expert_fc1") as x:
       output = self.linear(x)
   output = offload_commit(output, "expert_fc1", release=[input_tensor])

2. WEIGHT OFFLOADING (new):
   Offloads module weights to CPU after forward, reloads before backward.
   No manual tensor movement — engine handles everything.

   offload_weights(module, "expert_weights")  # registers hooks
   # That's it. Engine automatically:
   #   - After module.forward(): D2H weights to CPU, free GPU
   #   - Before module.backward(): H2D reload from CPU
   #   - Overlap with surrounding compute

Both use TensorOffloader under the hood for async D2H/H2D with pinned memory.
"""

from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.nn as nn

from .tensor_offloader import TensorOffloader, OffloadHandle


# Shared offloader singleton
_OFFLOADER: Optional[TensorOffloader] = None


def _get_offloader() -> TensorOffloader:
    global _OFFLOADER
    if _OFFLOADER is None:
        _OFFLOADER = TensorOffloader(pin_memory=True, use_pool=False)
    return _OFFLOADER


def reset_offloader() -> None:
    """Reset the shared offloader (call between iterations)."""
    global _OFFLOADER
    if _OFFLOADER is not None:
        _OFFLOADER.reset_pool()


# ═══════════════════════════════════════════════════════════════════════
# WEIGHT OFFLOADING
# ═══════════════════════════════════════════════════════════════════════


class WeightOffloadState:
    """Tracks offloaded weight handles for a module."""

    def __init__(self):
        self.handles: dict[str, OffloadHandle] = {}
        self.offloaded = False


def offload_weights(
    module: nn.Module,
    name: str = "weights",
) -> None:
    """Register hooks to automatically offload module weights after forward.

    After forward: async D2H all parameters to CPU, free GPU storage.
    Before backward: async H2D reload from CPU, restore GPU storage.

    The D2H overlaps with whatever compute follows the module's forward.
    The H2D overlaps with whatever backward precedes the module's backward.

    Args:
        module: The module whose weights to offload.
        name: Label for debugging/profiling.

    Usage:
        offload_weights(self.experts)
        # That's it. Forward/backward work automatically.
    """
    state = WeightOffloadState()
    offloader = _get_offloader()

    def _post_forward_hook(mod, args, output):
        """After forward: offload weights to CPU, free GPU storage."""
        state.handles.clear()
        for pname, param in mod.named_parameters(recurse=True):
            data = param.data
            if hasattr(data, "to_local"):
                data = data.to_local()
            if data.device.type != "cuda" or data.untyped_storage().size() == 0:
                continue
            state.handles[pname] = offloader.offload(data, release_storage=True)
        state.offloaded = True

        # Register backward hook on output to reload before backward
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(_make_backward_hook(mod, state))
        elif isinstance(output, tuple):
            for t in output:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    t.register_hook(_make_backward_hook(mod, state))
                    break
        return output

    def _make_backward_hook(mod, state):
        def _backward_hook(grad):
            if not state.offloaded:
                return grad
            for pname, param in mod.named_parameters(recurse=True):
                if pname not in state.handles:
                    continue
                data = param.data
                if hasattr(data, "to_local"):
                    data = data.to_local()
                offloader.reload_into(state.handles[pname], data)
            offloader.sync_reload()
            state.offloaded = False
            state.handles.clear()
            return grad
        return _backward_hook

    module.register_forward_hook(_post_forward_hook)


# ═══════════════════════════════════════════════════════════════════════
# ACTIVATION OFFLOADING (Megatron-style off_interface)
# ═══════════════════════════════════════════════════════════════════════


class offload_activation:
    """Context manager for activation offloading.

    Wraps a computation and marks the input tensor for offloading.
    Inside the context, autograd saved-tensor hooks intercept
    save_for_backward and queue tensors for D2H.

    Usage:
        with offload_activation(should_offload, x, "expert_fc1") as x:
            output = self.linear(x)
        output = offload_commit(output, "expert_fc1", release=[x])
    """

    def __init__(self, offload: bool, tensor: torch.Tensor, name: str):
        self.offload = offload
        self.tensor = tensor
        self.name = name

    def __enter__(self) -> torch.Tensor:
        if not self.offload:
            return self.tensor
        # Use save_on_cpu for FSDP-compatible activation offloading
        self._ctx = torch.autograd.graph.save_on_cpu(pin_memory=True)
        self._ctx.__enter__()
        return self.tensor

    def __exit__(self, *args) -> None:
        if self.offload:
            self._ctx.__exit__(*args)


def offload_commit(
    tensor,
    name: str,
    release: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Mark the end of an offload group.

    Frees the GPU storage of tensors in `release` list.

    Args:
        tensor: Output tensor from the offloaded computation.
        name: Group name matching the offload_activation context.
        release: Tensors to free from GPU immediately.

    Returns:
        The tensor (pass-through).
    """
    if release:
        for t in release:
            if (
                isinstance(t, torch.Tensor)
                and t.device.type == "cuda"
                and t.untyped_storage().size() > 0
            ):
                t.untyped_storage().resize_(0)
    return tensor
