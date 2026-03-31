# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
# Modifications: Extracted as standalone module, no framework dependencies.

"""Pinned CPU tensor pool for efficient D2H/H2D transfers.

Pools pre-allocated pinned-memory tensors by (shape, dtype) to avoid
repeated cudaMallocHost/cudaFreeHost overhead after warmup.
"""

from collections import deque
from typing import Any, Dict, Tuple

import torch

from .utils import debug_rank


class TensorPool:
    """Reusable tensor pool with optional pinned memory.

    Maintains separate pools keyed by (shape, dtype). After warmup,
    all allocations are O(1) deque pops — no new CUDA pinned allocations.

    Args:
        device: Target device for tensors (typically "cpu" for offload pools).
        pin_memory: Whether to pin allocated tensors. Required for
            non-blocking D2H/H2D copies.
    """

    def __init__(self, device: str = "cpu", pin_memory: bool = True):
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        # {(shape, dtype): {'free': deque, 'all': list, 'allocated_count': int}}
        self._pools: Dict[Tuple, Dict[str, Any]] = {}
        self._stats = {
            "total_allocated": 0,
            "current_in_use": 0,
            "allocation_requests": 0,
            "free_requests": 0,
            "pool_hits": 0,
            "pool_misses": 0,
        }

    def allocate(self, shape: Tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get a tensor from the pool, allocating a new one if needed."""
        self._stats["allocation_requests"] += 1
        pool_key = (shape, dtype)

        if pool_key not in self._pools:
            self._pools[pool_key] = {
                "free": deque(),
                "all": [],
                "allocated_count": 0,
            }

        pool = self._pools[pool_key]

        if len(pool["free"]) > 0:
            tensor = pool["free"].popleft()
            self._stats["pool_hits"] += 1
        else:
            tensor = torch.empty(
                shape, dtype=dtype, device=self.device, pin_memory=self.pin_memory
            )
            pool["all"].append(tensor)
            self._stats["total_allocated"] += 1
            self._stats["pool_misses"] += 1
            debug_rank(
                f"TensorPool: new tensor shape={shape} dtype={dtype} "
                f"size={tensor.numel() * tensor.element_size() / (1024**2):.2f}MB"
            )

        pool["allocated_count"] += 1
        self._stats["current_in_use"] += 1
        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool for reuse."""
        self._stats["free_requests"] += 1
        pool_key = (tensor.shape, tensor.dtype)

        if pool_key not in self._pools:
            raise ValueError(
                f"No pool for shape={tensor.shape}, dtype={tensor.dtype}. "
                f"Available: {list(self._pools.keys())}"
            )

        pool = self._pools[pool_key]
        if not any(tensor is t for t in pool["all"]):
            raise ValueError("Tensor does not belong to this pool")

        pool["free"].append(tensor)
        pool["allocated_count"] -= 1
        self._stats["current_in_use"] -= 1

    def reset(self) -> None:
        """Mark all tensors as available for the next iteration."""
        for pool in self._pools.values():
            pool["free"].clear()
            for tensor in pool["all"]:
                pool["free"].append(tensor)
            pool["allocated_count"] = 0
        self._stats["current_in_use"] = 0

    def clear(self) -> None:
        """Release all tensors and free memory."""
        for pool in self._pools.values():
            pool["free"].clear()
            pool["all"].clear()
        self._pools.clear()
        self._stats["current_in_use"] = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def stats(self) -> Dict[str, int]:
        return self._stats.copy()

    def __del__(self):
        self.clear()
