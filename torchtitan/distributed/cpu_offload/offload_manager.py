# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
# Modifications: Refactored to be framework-agnostic — no Megatron/TE dependencies.

"""OffloadManager — singleton orchestrator for activation offloading.

Coordinates chunk handlers across microbatches and virtual pipeline stages.
Owns the shared D2H/H2D CUDA streams and the pinned-memory tensor pool.

This module has ZERO dependencies on Megatron, TransformerEngine, or any
specific training framework. It only requires PyTorch.
"""

from collections import deque
from typing import Dict, List, Optional

import torch

from .chunk_handler import ChunkOffloadHandler
from .offload_group import OffloadTensorGroup
from .tensor_pool import TensorPool
from .utils import debug_rank, print_offload_summary_table


class OffloadManager:
    """Singleton manager coordinating activation offloading.

    Manages ChunkOffloadHandlers across microbatches and VP stages,
    owns CUDA streams for async transfers, and provides the autograd
    hook entry points.
    """

    _INSTANCE: Optional["OffloadManager"] = None

    @classmethod
    def get_instance(cls) -> "OffloadManager":
        if cls._INSTANCE is None:
            cls._INSTANCE = OffloadManager()
        return cls._INSTANCE

    @classmethod
    def reset_instance(cls) -> None:
        cls._INSTANCE = None
        cls._INSTANCE = OffloadManager()

    def __init__(self):
        self._queue: deque = deque()
        self._stages: Optional[List[List]] = None

        # Dedicated CUDA streams for async D2H/H2D transfers
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()

        # Shared pinned-memory pool for all chunks
        self._cpu_tensor_pool = TensorPool(device="cpu", pin_memory=True)

        # Warmup tracking
        self._is_warmup = True
        self._cached_chunks_forward: List[ChunkOffloadHandler] = []
        self._cached_chunks_backward: List[ChunkOffloadHandler] = []
        self._cached_chunks_index_backward = 0
        self._cached_chunks_index_forward = 0

        self.do_offload = True
        self._offload_margin = 0
        self._delayed_offload_groups = []

        # Summary collected after warmup
        self._offload_summary_bytes: Dict[str, int] = {}
        self._offload_summary_total_bytes: int = 0

        self.reset()

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def d2h_stream(self) -> torch.cuda.Stream:
        return self._d2h_stream

    @property
    def h2d_stream(self) -> torch.cuda.Stream:
        return self._h2d_stream

    @property
    def cpu_tensor_pool(self) -> TensorPool:
        return self._cpu_tensor_pool

    @property
    def offload_summary_bytes(self) -> Dict[str, int]:
        return self._offload_summary_bytes

    @property
    def offload_summary_total_bytes(self) -> int:
        return self._offload_summary_total_bytes

    # ── Delayed offload ────────────────────────────────────────────────

    def push_offload_groups(self, group_hook, forced_released_tensors) -> None:
        self._delayed_offload_groups.append((group_hook, forced_released_tensors))

    def flush_delayed_groups(self) -> None:
        for hook, tensors in reversed(self._delayed_offload_groups):
            hook(tensors)
        self._delayed_offload_groups = []

    # ── Lifecycle ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset manager state for a new training iteration."""
        self._inside_context = False
        self._cur_forward_chunk: Optional[ChunkOffloadHandler] = None
        self._cur_backward_chunk: Optional[ChunkOffloadHandler] = None

        if hasattr(self, "_cpu_tensor_pool"):
            self._cpu_tensor_pool.reset()

        if self._is_warmup and len(self._cached_chunks_forward) > 0:
            self._post_warmup()

        self._cached_chunks_index_backward = 0
        self._cached_chunks_index_forward = 0

        for chunk in self._cached_chunks_forward:
            chunk.reset()
        self._delayed_offload_groups = []

    def disable_offload(self) -> None:
        self.do_offload = False
        for chunk in self._cached_chunks_forward:
            chunk.do_offload = False

    def enable_offload(self) -> None:
        self.do_offload = True
        for chunk in self._cached_chunks_forward:
            chunk.do_offload = True

    # ── Warmup ─────────────────────────────────────────────────────────

    def _post_warmup(self) -> None:
        """Collect statistics and mark last groups as non-offloadable."""
        self._is_warmup = False
        assert len(self._cached_chunks_forward) == len(self._cached_chunks_backward)

        for chunk in self._cached_chunks_forward:
            chunk.is_warmup = False
            self._offload_margin = max(
                self._offload_margin, chunk.get_max_deduplicated_groups()
            )

        # Mark last groups as non-offloadable so reload doesn't block compute
        last_group_by_name = {}
        for chunk in reversed(self._cached_chunks_backward):
            for group in chunk.offload_groups:
                last_group_by_name[group._name] = group

        for name, group in last_group_by_name.items():
            if self._offload_margin > 0:
                group.offload = False
                self._offload_margin -= 1
            else:
                break

        # Collect statistics
        total_bytes: Dict[str, int] = {}
        for chunk in self._cached_chunks_forward:
            for group in chunk.offload_groups:
                if group.offload:
                    total_bytes.setdefault(group._name, 0)
                    total_bytes[group._name] += group.total_offload_bytes
            if chunk is self._cached_chunks_backward[0]:
                break

        self._offload_summary_bytes = total_bytes
        self._offload_summary_total_bytes = sum(total_bytes.values())
        print_offload_summary_table(total_bytes)

    # ── VP/PP chunk scheduling ─────────────────────────────────────────

    def flush(self) -> None:
        """Flush all staged VP chunks to the backward queue."""
        if len(self._stages[0]) == len(self._stages[-1]):
            lens = [len(e) for e in self._stages]
            assert min(lens) == max(lens), "All VP stages must have same chunk count"
            self._stages[-1] = []
            for chunks in reversed(self._stages):
                for chunk in chunks:
                    self._push(chunk)
            for i in range(self._vpp):
                self._stages[i] = []

    def _push(self, handler: ChunkOffloadHandler) -> None:
        self._queue.append(handler)
        if self._is_warmup:
            self._cached_chunks_backward.append(handler)

    def pop_backward_chunk(self, name=None) -> None:
        """Advance to the next non-empty backward chunk."""
        self._cur_backward_chunk = None
        for handler in self._cached_chunks_backward[self._cached_chunks_index_backward:]:
            self._cached_chunks_index_backward += 1
            if not handler.is_empty_chunk(name):
                self._cur_backward_chunk = handler
                break
        assert self._cur_backward_chunk is not None, "No non-empty backward chunk found"

    def front_backward_chunk(self, name=None):
        """Peek at the next non-empty backward chunk without consuming it."""
        for handler in self._cached_chunks_backward[self._cached_chunks_index_backward:]:
            if not handler.is_empty_chunk(name):
                return handler
        return None

    def cur_forward_chunk(self) -> Optional[ChunkOffloadHandler]:
        return self._cur_forward_chunk

    def cur_backward_chunk(self) -> Optional[ChunkOffloadHandler]:
        return self._cur_backward_chunk

    # ── Chunk initialization ───────────────────────────────────────────

    def init_chunk_handler(
        self,
        vp_size: Optional[int] = None,
        vp_stage: Optional[int] = None,
        min_offloaded_tensor_size: int = 1024 * 1024,
    ) -> None:
        """Initialize a chunk handler for a new microbatch forward pass.

        Args:
            vp_size: Virtual pipeline size (None or 1 for no VP).
            vp_stage: Current VP stage index.
            min_offloaded_tensor_size: Min numel to offload.
        """
        if not self._is_warmup:
            return

        vp_size = 1 if vp_size is None else vp_size
        if self._stages is None:
            self._vpp = vp_size
            self._stages = [[] for _ in range(vp_size)]

        cur_vpp_rank = 0 if vp_stage is None else vp_stage

        if cur_vpp_rank == self._vpp - 1:
            self.flush()

        chunk = ChunkOffloadHandler(
            min_offloaded_tensor_size=min_offloaded_tensor_size,
            cpu_tensor_pool=self._cpu_tensor_pool,
            d2h_stream=self._d2h_stream,
            h2d_stream=self._h2d_stream,
        )
        self._stages[cur_vpp_rank].append(chunk)
        if cur_vpp_rank == self._vpp - 1:
            self._push(chunk)
            self.flush()
        self._cur_forward_chunk = chunk
        chunk.vpp_rank = cur_vpp_rank
        self._cached_chunks_forward.append(chunk)

    def pop_forward_chunk(self, name=None) -> Optional[ChunkOffloadHandler]:
        """Get the next forward chunk handler."""
        if not self.do_offload:
            return self._cur_forward_chunk
        while not self._is_warmup and (
            self._cur_forward_chunk is None
            or self._cur_forward_chunk.finish_all_groups(name)
        ):
            if self._cached_chunks_index_forward >= len(self._cached_chunks_forward):
                self._cur_forward_chunk = None
                break
            self._cur_forward_chunk = self._cached_chunks_forward[
                self._cached_chunks_index_forward
            ]
            self._cached_chunks_index_forward += 1
        return self._cur_forward_chunk

    # ── Autograd hook entry points ─────────────────────────────────────

    def mark_not_offloadable(self, tensor: torch.Tensor) -> None:
        """Mark a tensor (e.g., model parameter) as never offloadable."""
        if tensor is not None:
            tensor.offloading_activation = False

    @property
    def inside_context(self) -> bool:
        return self._inside_context

    @inside_context.setter
    def inside_context(self, value: bool) -> None:
        self._inside_context = value

    def __enter__(self):
        """Enter autograd saved-tensor hook context."""
        if self._cur_forward_chunk is None or not self._cur_forward_chunk.do_offload:
            return
        self._inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(
            self._on_save_for_backward, self._on_get_saved_tensor
        )

    def __exit__(self, *args):
        """Exit autograd saved-tensor hook context."""
        if self._cur_forward_chunk is None or not self._cur_forward_chunk.do_offload:
            return
        self._inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def _on_save_for_backward(self, tensor: torch.Tensor):
        """Autograd hook: intercept save_for_backward, return a tag."""
        return self._cur_forward_chunk.tensor_push(tensor)

    def _on_get_saved_tensor(self, saved_state):
        """Autograd hook: retrieve tensor by tag, reloading from CPU if needed."""
        return self._cur_backward_chunk.tensor_pop(saved_state)
