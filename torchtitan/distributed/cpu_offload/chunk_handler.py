# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
# Modifications: Refactored to be standalone — no Megatron imports.

"""ChunkOffloadHandler — manages activation offload/reload for one microbatch.

This is the core D2H/H2D engine. It owns the low-level copy logic:
- offload(): GPU tensor -> pinned CPU tensor (async via D2H stream)
- reload():  pinned CPU tensor -> GPU tensor (async via H2D stream)
- bulk_offload_group() / bulk_reload_group(): batch operations with event sync

The handler is agnostic to pipeline parallelism — the OffloadManager above
handles VP/PP chunk scheduling.
"""

from typing import List, Optional, Tuple

import torch

from .offload_group import OffloadTensorGroup
from .tensor_pool import TensorPool
from .utils import debug_rank, is_graph_capturing


# Type alias for the offloaded state: (original_device, cpu_backup, used_pool)
OffloadState = Tuple[torch.device, torch.Tensor, bool]


class ChunkOffloadHandler:
    """Handles activation offloading/reloading for a single microbatch.

    Args:
        min_offloaded_tensor_size: Minimum numel to offload (skip small tensors).
        cpu_tensor_pool: Shared TensorPool for pinned CPU buffers.
        d2h_stream: CUDA stream for GPU->CPU transfers.
        h2d_stream: CUDA stream for CPU->GPU transfers.
    """

    def __init__(
        self,
        min_offloaded_tensor_size: int,
        cpu_tensor_pool: TensorPool,
        d2h_stream: torch.cuda.Stream,
        h2d_stream: torch.cuda.Stream,
    ):
        self.min_offloaded_tensor_size = min_offloaded_tensor_size
        self.cpu_tensor_pool = cpu_tensor_pool
        self.d2h_stream = d2h_stream
        self.h2d_stream = h2d_stream

        self.do_offload = True
        self.is_warmup = True
        self.vpp_rank = 0

        # Group management
        self.offload_groups: List[OffloadTensorGroup] = []
        self._offloaded_group_index = 0
        self._groups_to_offload: List[OffloadTensorGroup] = []
        self._groups_to_reload: List[OffloadTensorGroup] = []
        self._tensor_count_current_group = 0
        self._max_group_size = 0
        self._reloading_group: List[OffloadTensorGroup] = []
        self.torch_tensor_count = 0

    def reset(self) -> None:
        """Reset handler state for a new iteration."""
        self._offloaded_group_index = 0
        self._groups_to_offload = []
        self._groups_to_reload = []
        self._tensor_count_current_group = 0
        self._reloading_group = []

    # ── Low-level copy operations ──────────────────────────────────────

    def offload(self, src_tensor: torch.Tensor, use_cpu_pool: bool = True) -> OffloadState:
        """Copy a GPU tensor to pinned CPU memory (called inside D2H stream context).

        Returns:
            Tuple of (original_device, cpu_backup_tensor, used_pool_flag).
        """
        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

        if use_cpu_pool:
            cpu_backup = self.cpu_tensor_pool.allocate(
                src_tensor.shape, dtype=src_tensor.dtype
            )
        else:
            cpu_backup = torch.empty(
                src_tensor.shape, dtype=src_tensor.dtype, device="cpu", pin_memory=True
            )

        cpu_backup.copy_(src_tensor, non_blocking=True)
        return (src_tensor.device, cpu_backup, use_cpu_pool)

    def reload(self, state: OffloadState, non_blocking: Optional[bool] = None) -> torch.Tensor:
        """Copy a tensor from pinned CPU memory back to GPU.

        Args:
            state: The OffloadState tuple returned by offload().
            non_blocking: Override for non_blocking flag. Defaults to
                True if cpu_backup is pinned.

        Returns:
            New GPU tensor with the restored data.
        """
        dev, cpu_backup, use_cpu_pool = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        gpu_tensor = torch.empty(
            cpu_backup.size(), dtype=cpu_backup.dtype, layout=cpu_backup.layout, device=dev
        )
        gpu_tensor.copy_(cpu_backup, non_blocking=non_blocking)
        if use_cpu_pool:
            self.cpu_tensor_pool.free(cpu_backup)
        return gpu_tensor

    # ── Tensor filtering ───────────────────────────────────────────────

    def should_offload_tensor(self, tensor: torch.Tensor) -> bool:
        """Decide whether a tensor should be offloaded."""
        if tensor.numel() < self.min_offloaded_tensor_size:
            return False
        if hasattr(tensor, "offloading_activation") and not tensor.offloading_activation:
            return False
        return True

    # ── Group management ───────────────────────────────────────────────

    def find_group_with_name(self, name: str, start_index: int = 0):
        """Find the first group with the given name starting from start_index."""
        return next(
            (g for g in self.offload_groups[start_index:] if g._name == name), None
        )

    def is_empty_chunk(self, name=None) -> bool:
        """Check if this chunk has no tensors to manage."""
        if name is not None:
            return self.find_group_with_name(name) is None
        return self._max_group_size == 0

    def finish_all_groups(self, name=None) -> bool:
        """Check if all groups have been processed."""
        if (
            len(self._groups_to_reload) == 0
            and len(self._groups_to_offload) == 0
            and self._offloaded_group_index > 0
        ):
            return True
        assert name is not None, "Name is required"
        return self.find_group_with_name(name, self._offloaded_group_index) is None

    def find_next_group(self, name=None):
        """Find the next group with the given name from current index."""
        assert name is not None, "Name is required"
        return self.find_group_with_name(name, self._offloaded_group_index)

    def get_max_deduplicated_groups(self) -> int:
        """Count distinct group names."""
        seen = []
        for group in self.offload_groups:
            if group._name not in seen:
                seen.append(group._name)
        return len(seen)

    # ── Tensor push/pop (autograd hook interface) ──────────────────────

    def tensor_push(self, tensor: torch.Tensor):
        """Register a tensor for potential offloading. Returns a tag for retrieval."""
        torch_stray = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )
        assert not torch_stray, "FakeTensor/FunctionalTensor should not be offloaded"

        tensor_tag = (self._offloaded_group_index, self._tensor_count_current_group)
        self._tensor_count_current_group += 1
        self.offload_groups[self._offloaded_group_index - 1].push_tensor(tensor_tag, tensor)
        debug_rank(f"tensor_push {tensor_tag}")
        return tensor_tag

    def tensor_pop(self, tensor_tag):
        """Retrieve a tensor by tag, reloading from CPU if it was offloaded."""
        group_id, _ = tensor_tag
        tensor = self.offload_groups[group_id - 1].pop_tensor(tensor_tag)
        if isinstance(tensor, tuple):
            tensor = self.reload(tensor)
        return tensor

    # ── Bulk offload/reload ────────────────────────────────────────────

    def bulk_offload_group(self) -> None:
        """Offload all tensors in the current group via the D2H stream."""
        group = self._groups_to_offload[-1]
        torch.cuda.nvtx.range_push(f"activation_offload_{group._name}")
        with torch.cuda.stream(self.d2h_stream):
            for tag, tensor_on_device in group._tensors.items():
                if self.should_offload_tensor(tensor_on_device):
                    state = self.offload(tensor_on_device, use_cpu_pool=group.use_cpu_pool)
                    if self.is_warmup:
                        group.update_offload_info(tensor_on_device)
                    tensor_on_device.record_stream(self.d2h_stream)
                    group.push_tensor(tag, state)
            group.record_offload_event(self.d2h_stream)
        self._groups_to_offload.pop()
        torch.cuda.nvtx.range_pop()

    def bulk_reload_group(self) -> None:
        """Reload all offloaded tensors in the current group via the H2D stream."""
        group = self._groups_to_reload[-1]
        torch.cuda.nvtx.range_push(f"activation_reload_{group._name}")
        with torch.cuda.stream(self.h2d_stream):
            if not is_graph_capturing():
                group.wait_offload_event(self.h2d_stream)
            for tag, state in group._tensors.items():
                if isinstance(state, tuple):
                    recovered = self.reload(state)
                    group.push_tensor(tag, recovered)
            group.record_reload_event(self.h2d_stream)
        self._groups_to_reload.pop()
        self._reloading_group.append(group)
        torch.cuda.nvtx.range_pop()

    # ── Forward/backward callbacks ─────────────────────────────────────

    def should_bulk_offload(self, front_backward_chunk_fn) -> bool:
        """Decide whether the current group should be offloaded."""
        assert len(self._groups_to_offload) > 0
        group = self._groups_to_offload[-1]
        if self.is_warmup:
            return True
        if not group.offload:
            return False
        # Don't offload if this is the last group and it's about to be used
        next_bwd = front_backward_chunk_fn(group._name)
        if next_bwd is not None and next_bwd is self:
            if self.find_next_group(group._name) is None:
                return False
        return True

    def bulk_offload(self, forced_released_tensors, front_backward_chunk_fn) -> None:
        """Offload current group and optionally release GPU tensors."""
        if self.should_bulk_offload(front_backward_chunk_fn):
            self._groups_to_reload.append(self._groups_to_offload[-1])
            self.bulk_offload_group()
            if forced_released_tensors:
                cur_stream = torch.cuda.current_stream()
                for t in forced_released_tensors:
                    if self.should_offload_tensor(t):
                        t.record_stream(cur_stream)
                        t.untyped_storage().resize_(0)

    def on_group_commit_forward(self, forced_released_tensors, front_backward_chunk_fn) -> None:
        """Called at the end of a module's forward to trigger offloading."""
        if not self.do_offload:
            return
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_offload(forced_released_tensors, front_backward_chunk_fn)

    def on_group_start_forward(self, name: str) -> None:
        """Called at the start of a module's forward to prepare a new group."""
        if not self.do_offload:
            return
        self._offloaded_group_index += 1
        if self.is_warmup:
            # Determine pool usage: dynamic-shape modules (MoE) don't use pool
            use_pool = name not in ("expert_fc1", "moe_act")
            self.offload_groups.append(OffloadTensorGroup(name, use_cpu_pool=use_pool))
            self._max_group_size = max(self._max_group_size, self._offloaded_group_index)
        else:
            for group in self.offload_groups[self._offloaded_group_index - 1:]:
                if group._name == name:
                    break
                self._offloaded_group_index += 1
        self._tensor_count_current_group = 0
        self._groups_to_offload.append(self.offload_groups[self._offloaded_group_index - 1])

    def bulk_reload(self, front_backward_chunk_fn) -> None:
        """Reload the next group, or pre-reload the next chunk's last layer."""
        if len(self._groups_to_reload) > 0:
            self.bulk_reload_group()
        else:
            next_bwd = front_backward_chunk_fn()
            if (
                next_bwd is not None
                and next_bwd._offloaded_group_index == next_bwd._max_group_size
            ):
                next_bwd.pre_reload_last_layer()

    def pre_reload_last_layer(self) -> None:
        """Pre-reload the last layer to hide reload latency."""
        if len(self._groups_to_reload) > 0:
            self.bulk_reload_group()

    def on_group_start_backward(self, front_backward_chunk_fn) -> None:
        """Called at the start of a module's backward to trigger reloading."""
        if not self.do_offload:
            return
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_reload(front_backward_chunk_fn)

    def on_group_commit_backward(self, name: str, pop_backward_chunk_fn, cur_backward_chunk_fn) -> None:
        """Called at the end of a module's backward. Ensures correct chunk and sync."""
        if not self.do_offload:
            return
        if cur_backward_chunk_fn() is not self:
            pop_backward_chunk_fn(name)
        assert cur_backward_chunk_fn() is self, "Chunk mismatch in backward"
        if not is_graph_capturing() and self._reloading_group:
            for rg in self._reloading_group:
                if rg._name == name:
                    rg.wait_reload_event(torch.cuda.current_stream())
                    self._reloading_group.remove(rg)
                    break
