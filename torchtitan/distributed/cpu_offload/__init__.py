# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
#
# CPU offload engine for large-scale MoE training.
# Forked from NVIDIA Megatron-LM, refactored to be framework-agnostic.

"""CPU offload engine for activation and weight offloading.

Core engine (Megatron-style, low-level):
  - TensorPool: pinned CPU memory pool with O(1) reuse
  - OffloadTensorGroup: tensor batch with CUDA event sync
  - ChunkOffloadHandler: async D2H/H2D copy engine
  - OffloadManager: singleton orchestrator
  - ActivationOffloadContext: autograd hooks for activation offload
  - TensorOffloader: general-purpose async GPU↔CPU transfer

Clean API (high-level, for model integration):
  - offload_activation: context manager for activation offloading
  - offload_commit: mark end of offload group, free GPU tensors
  - offload_weights: register hooks for automatic weight offloading
"""

# Core engine (Megatron-style)
from .tensor_pool import TensorPool
from .offload_group import OffloadTensorGroup
from .chunk_handler import ChunkOffloadHandler
from .offload_manager import OffloadManager
from .tensor_offloader import TensorOffloader, OffloadHandle

# Activation offloading (autograd hooks)
from .autograd_hooks import (
    ActivationOffloadContext,
    group_start,
    group_commit,
    flush_delayed_groups,
    disable_offload,
    enable_offload,
    forward_record,
    backward_record,
)

# Clean API (for model integration)
from .offload_api import (
    offload_activation,
    offload_commit,
    offload_weights,
    reset_offloader,
)

# Optimizer offloading
from .hybrid_optimizer import HybridDeviceOptimizer

__all__ = [
    # Core engine
    "TensorPool",
    "OffloadTensorGroup",
    "ChunkOffloadHandler",
    "OffloadManager",
    "TensorOffloader",
    "OffloadHandle",
    # Activation offloading
    "ActivationOffloadContext",
    "group_start",
    "group_commit",
    "flush_delayed_groups",
    "disable_offload",
    "enable_offload",
    "forward_record",
    "backward_record",
    # Clean API
    "offload_activation",
    "offload_commit",
    "offload_weights",
    "reset_offloader",
    # Optimizer
    "HybridDeviceOptimizer",
]
