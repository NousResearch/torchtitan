# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
# Modifications: Refactored to be framework-agnostic — no Megatron/TE dependencies.

"""Autograd functions and context manager for fine-grained activation offloading.

These are the user-facing primitives:
- ActivationOffloadContext: wraps a module's forward to intercept save_for_backward
- group_start(): marks the beginning of an offload group (inserts into autograd graph)
- group_commit(): marks the end — triggers D2H and optional tensor release
- flush_delayed_groups(): execute any deferred offload operations

Usage example in a custom module forward:

    from torchtitan.distributed.cpu_offload import offload_ctx, group_commit

    with offload_ctx(should_offload, input_tensor, "my_module") as x:
        out = self.linear(x)
    out = group_commit(out, "my_module", forced_released_tensors=[input_tensor])
"""

from contextlib import nullcontext
from typing import Any, List, Optional

import torch

from .offload_manager import OffloadManager
from .utils import is_graph_capturing


# ── Autograd Functions ─────────────────────────────────────────────────


class _GroupCommitFunction(torch.autograd.Function):
    """Identity op that triggers offload in forward and sync in backward."""

    @staticmethod
    def forward(ctx, tensor, chunk_handler, name, forced_released_tensors, delay_offload):
        mgr = OffloadManager.get_instance()
        if delay_offload:
            mgr.push_offload_groups(
                lambda frt: chunk_handler.on_group_commit_forward(frt, mgr.front_backward_chunk),
                forced_released_tensors,
            )
        else:
            chunk_handler.on_group_commit_forward(
                forced_released_tensors, mgr.front_backward_chunk
            )
        ctx.cpu_offload_handler = chunk_handler
        ctx.name = name
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        mgr = OffloadManager.get_instance()
        ctx.cpu_offload_handler.on_group_commit_backward(
            ctx.name, mgr.pop_backward_chunk, mgr.cur_backward_chunk
        )
        return grad_output + (None, None, None, None)


class _GroupStartFunction(torch.autograd.Function):
    """Identity op that prepares a new offload group in forward, triggers reload in backward."""

    @staticmethod
    def forward(ctx, tensor, chunk_handler, name):
        ctx.cpu_offload_handler = chunk_handler
        chunk_handler.on_group_start_forward(name)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        mgr = OffloadManager.get_instance()
        ctx.cpu_offload_handler.on_group_start_backward(mgr.front_backward_chunk)
        return grad_output, None, None, None


class _ForwardRecordFunction(torch.autograd.Function):
    """Record backward event for CUDA graph capture compatibility."""

    @staticmethod
    def forward(ctx, tensor, event: torch.cuda.Event) -> torch.Tensor:
        ctx.event = event
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        h2d_stream = OffloadManager.get_instance().h2d_stream
        torch.cuda.current_stream().record_event(ctx.event)
        torch.cuda.current_stream().wait_stream(h2d_stream)
        return grad_output, None


# ── Public API functions ───────────────────────────────────────────────


def group_start(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Mark the start of an offload group in the autograd graph.

    Args:
        tensor: Input tensor to the module being offloaded.
        name: Group name (e.g., "expert_fc1", "core_attn").

    Returns:
        The same tensor (identity), now tracked in the autograd graph.
    """
    chunk = OffloadManager.get_instance().pop_forward_chunk(name=name)
    if chunk is None:
        return tensor
    return _GroupStartFunction.apply(tensor, chunk, name)


def group_commit(
    tensor,
    name: str,
    forced_released_tensors: Optional[List[torch.Tensor]] = None,
    delay_offload: bool = False,
):
    """Mark the end of an offload group — triggers D2H copy.

    Args:
        tensor: Output tensor from the module. Can be a single tensor,
            tuple, or list. Only the first element is tracked.
        name: Group name matching the corresponding group_start().
        forced_released_tensors: GPU tensors to free immediately after
            offloading (bypasses GC delay).
        delay_offload: If True, defer the actual D2H copy to a later
            flush_delayed_groups() call.

    Returns:
        The tensor (identity pass-through).
    """
    if forced_released_tensors is None:
        forced_released_tensors = []

    # Handle tuple/list outputs — wrap only the first element
    if isinstance(tensor, tuple):
        if len(tensor) == 0:
            return tensor
        c0 = group_commit(tensor[0], name, forced_released_tensors, delay_offload)
        return (c0,) + tensor[1:]
    if isinstance(tensor, list):
        if len(tensor) == 0:
            return tensor
        c0 = group_commit(tensor[0], name, forced_released_tensors, delay_offload)
        return [c0] + tensor[1:]

    chunk = OffloadManager.get_instance().cur_forward_chunk()
    if chunk is None:
        return tensor
    return _GroupCommitFunction.apply(tensor, chunk, name, forced_released_tensors, delay_offload)


def flush_delayed_groups() -> None:
    """Execute all deferred offload operations."""
    OffloadManager.get_instance().flush_delayed_groups()


def disable_offload() -> None:
    """Temporarily disable all offloading."""
    OffloadManager.get_instance().disable_offload()


def enable_offload() -> None:
    """Re-enable offloading after a disable_offload() call."""
    OffloadManager.get_instance().enable_offload()


def forward_record(event: torch.cuda.Event) -> None:
    """Record a forward event for CUDA graph capture compatibility."""
    d2h_stream = OffloadManager.get_instance().d2h_stream
    torch.cuda.current_stream().record_event(event)
    torch.cuda.current_stream().wait_stream(d2h_stream)


def backward_record(tensor: torch.Tensor, event: torch.cuda.Event) -> torch.Tensor:
    """Record a backward event for CUDA graph capture compatibility."""
    return _ForwardRecordFunction.apply(tensor, event)


# ── Context Manager ────────────────────────────────────────────────────


class ActivationOffloadContext:
    """Context manager that wraps a module's forward for activation offloading.

    Calls group_start() on enter and installs autograd saved-tensor hooks
    so that tensors saved for backward are transparently offloaded to CPU.

    Usage:
        with ActivationOffloadContext(should_offload, input_tensor, "expert_fc1") as x:
            output = self.linear(x)
        output = group_commit(output, "expert_fc1", forced_released_tensors=[input_tensor])
    """

    def __init__(self, offload: bool, tensor: torch.Tensor, name: str):
        self.offload = offload
        self.tensor = tensor
        self.name = name

    def __enter__(self) -> torch.Tensor:
        if self.offload:
            self.tensor = group_start(self.tensor, self.name)
            OffloadManager.get_instance().__enter__()
        return self.tensor

    def __exit__(self, *args) -> None:
        if self.offload:
            OffloadManager.get_instance().__exit__()

    @staticmethod
    def init_chunk_handler(
        vp_size: Optional[int] = None,
        vp_stage: Optional[int] = None,
        min_offloaded_tensor_size: int = 1024 * 1024,
    ) -> None:
        """Initialize a chunk handler for a new microbatch."""
        OffloadManager.get_instance().init_chunk_handler(
            vp_size, vp_stage, min_offloaded_tensor_size
        )

    @staticmethod
    def get_context(flag: bool):
        """Get either the offload manager context or a no-op context."""
        return OffloadManager.get_instance() if flag else nullcontext()

    @staticmethod
    def mark_not_offloadable(tensor: torch.Tensor) -> None:
        """Mark a tensor as never offloadable (e.g., model parameters)."""
        OffloadManager.get_instance().mark_not_offloadable(tensor)

    @staticmethod
    def reset() -> None:
        """Reset the offload manager for a new iteration."""
        OffloadManager.get_instance().reset()

    @staticmethod
    def reset_instance() -> None:
        """Reset the singleton manager instance."""
        OffloadManager.reset_instance()
