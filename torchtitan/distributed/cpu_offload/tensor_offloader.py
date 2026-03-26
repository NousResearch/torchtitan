# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.

"""General-purpose async tensor offloader.

Low-level building block that moves ANY tensor between GPU and CPU using
dedicated CUDA streams with pinned memory. This is the foundation for
activation, weight, and gradient offloading.

Usage:
    offloader = TensorOffloader()

    # Async GPU → CPU
    handle = offloader.offload(gpu_tensor)

    # ... do other GPU work while copy runs ...

    # Async CPU → GPU
    gpu_tensor = offloader.reload(handle)

    # Sync to ensure reload is done before using
    offloader.sync_reload()
"""

from typing import Optional, Tuple

import torch

from .tensor_pool import TensorPool


# Opaque handle returned by offload(), consumed by reload()
OffloadHandle = Tuple[torch.device, torch.Tensor, bool]


class TensorOffloader:
    """Async GPU ↔ CPU tensor transfer engine.

    Owns dedicated D2H/H2D CUDA streams and a pinned-memory tensor pool.
    All copies are non-blocking when using pinned memory.

    Args:
        pin_memory: Use pinned CPU memory for async transfers.
        use_pool: Reuse CPU buffers across calls (O(1) after warmup).
    """

    def __init__(self, pin_memory: bool = True, use_pool: bool = True):
        self.pin_memory = pin_memory
        self.use_pool = use_pool
        self._pool = TensorPool(device="cpu", pin_memory=pin_memory) if use_pool else None
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        self._last_offload_event: Optional[torch.cuda.Event] = None
        self._last_reload_event: Optional[torch.cuda.Event] = None

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def d2h_stream(self) -> torch.cuda.Stream:
        return self._d2h_stream

    @property
    def h2d_stream(self) -> torch.cuda.Stream:
        return self._h2d_stream

    @property
    def pool(self) -> Optional[TensorPool]:
        return self._pool

    # ── Core operations ────────────────────────────────────────────────

    def offload(
        self,
        tensor: torch.Tensor,
        release_storage: bool = False,
    ) -> OffloadHandle:
        """Async copy GPU tensor to pinned CPU memory.

        Args:
            tensor: GPU tensor to offload.
            release_storage: If True, free the GPU tensor's storage after
                queuing the copy (saves GPU memory immediately).

        Returns:
            Opaque handle to pass to reload().
        """
        assert tensor.device.type == "cuda", f"Expected CUDA tensor, got {tensor.device}"

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Allocate CPU buffer
        if self._pool is not None:
            cpu_buf = self._pool.allocate(tensor.shape, dtype=tensor.dtype)
        else:
            cpu_buf = torch.empty(
                tensor.shape, dtype=tensor.dtype, device="cpu",
                pin_memory=self.pin_memory,
            )

        # Async D2H copy
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            cpu_buf.copy_(tensor, non_blocking=self.pin_memory)

        # Record event for synchronization
        event = torch.cuda.Event()
        event.record(self._d2h_stream)
        self._last_offload_event = event

        # Optionally free GPU storage immediately
        if release_storage:
            tensor.record_stream(self._d2h_stream)
            tensor.untyped_storage().resize_(0)

        return (tensor.device, cpu_buf, self._pool is not None)

    def reload(
        self,
        handle: OffloadHandle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Async copy tensor from CPU back to GPU.

        Args:
            handle: The OffloadHandle from offload().
            device: Target GPU device (defaults to original device).

        Returns:
            New GPU tensor with the restored data.
        """
        orig_device, cpu_buf, used_pool = handle
        target = device or orig_device

        # Wait for offload to finish before reloading
        if self._last_offload_event is not None:
            self._h2d_stream.wait_event(self._last_offload_event)

        # Allocate GPU buffer and async H2D copy
        gpu_tensor = torch.empty(
            cpu_buf.shape, dtype=cpu_buf.dtype, device=target,
        )
        with torch.cuda.stream(self._h2d_stream):
            gpu_tensor.copy_(cpu_buf, non_blocking=cpu_buf.is_pinned())

        # Record reload event
        event = torch.cuda.Event()
        event.record(self._h2d_stream)
        self._last_reload_event = event

        # Return CPU buffer to pool
        if used_pool and self._pool is not None:
            self._pool.free(cpu_buf)

        return gpu_tensor

    def sync_offload(self) -> None:
        """Block until the last offload (D2H) completes."""
        if self._last_offload_event is not None:
            self._last_offload_event.synchronize()

    def sync_reload(self) -> None:
        """Block current stream until the last reload (H2D) completes."""
        if self._last_reload_event is not None:
            torch.cuda.current_stream().wait_event(self._last_reload_event)

    def sync_all(self) -> None:
        """Block until all pending transfers complete."""
        torch.cuda.synchronize()

    def reset_pool(self) -> None:
        """Reset the tensor pool for a new iteration."""
        if self._pool is not None:
            self._pool.reset()

    def clear(self) -> None:
        """Release all pool resources."""
        if self._pool is not None:
            self._pool.clear()
