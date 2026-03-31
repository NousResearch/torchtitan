# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
# Modifications: Extracted as standalone module.

"""OffloadTensorGroup — a batch of tensors offloaded/reloaded together.

Each group corresponds to one "module boundary" in the model (e.g., expert_fc1,
moe_act, core_attn). Groups own CUDA events for synchronizing D2H and H2D
transfers with the compute stream.
"""

from typing import Any, Dict, Tuple

import torch


class OffloadTensorGroup:
    """A named group of tensors that are offloaded and reloaded as a unit.

    Args:
        name: Identifier for this group (e.g., "expert_fc1", "core_attn").
        use_cpu_pool: Whether to use the shared TensorPool for CPU buffers.
            Set to False when tensor shapes are dynamic (e.g., MoE expert
            tensors whose shapes depend on routing).
    """

    def __init__(self, name: str, use_cpu_pool: bool = True):
        self._name = name
        self._tensors: Dict[Tuple[int, int], Any] = {}
        self._offload_event = torch.cuda.Event()
        self._reload_event = torch.cuda.Event()
        self.offload = True
        self.use_cpu_pool = use_cpu_pool
        # Warmup statistics
        self.total_offload_bytes = 0
        self.total_tensor_count = 0

    @property
    def name(self) -> str:
        return self._name

    def push_tensor(self, tag: Tuple[int, int], tensor: Any) -> None:
        """Store a tensor (or offload state tuple) under the given tag."""
        self._tensors[tag] = tensor

    def pop_tensor(self, tag: Tuple[int, int]) -> Any:
        """Remove and return the tensor/state stored under tag."""
        return self._tensors.pop(tag)

    def record_offload_event(self, stream: torch.cuda.Stream) -> None:
        """Record completion of D2H copies on the given stream."""
        self._offload_event.record(stream)

    def wait_offload_event(self, stream: torch.cuda.Stream) -> None:
        """Make stream wait until all D2H copies for this group are done."""
        stream.wait_event(self._offload_event)

    def record_reload_event(self, stream: torch.cuda.Stream) -> None:
        """Record completion of H2D copies on the given stream."""
        self._reload_event.record(stream)

    def wait_reload_event(self, stream: torch.cuda.Stream) -> None:
        """Make stream wait until all H2D copies for this group are done."""
        stream.wait_event(self._reload_event)

    def update_offload_info(self, tensor: torch.Tensor) -> None:
        """Accumulate offload statistics during warmup."""
        self.total_offload_bytes += tensor.numel() * tensor.element_size()
        self.total_tensor_count += 1
