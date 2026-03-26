# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
#
# General-purpose CPU offload engine for large-scale model training.
# Supports activation, weight, gradient, and optimizer state offloading.

"""General-purpose CPU offload engine.

Moves tensors between GPU and pinned CPU memory using dedicated CUDA streams
for fully asynchronous, overlapped transfers. Supports:

1. **Activation Offloading** — per-module granularity, autograd hooks
2. **Weight Offloading** — module hooks for prefetch/offload around forward
3. **Gradient Offloading** — post-accumulate hooks, CPU optimizer compatible
4. **Optimizer State Offloading** — HybridDeviceOptimizer (GPU/CPU split)

All built on the same core: TensorOffloader (async D2H/H2D with pinned pool).
"""

# General-purpose tensor offloader (foundation for everything)
from .tensor_offloader import TensorOffloader, OffloadHandle

# Activation offloading (fine-grained, per-module)
from .offload_manager import OffloadManager
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

# Weight offloading (module hooks)
from .weight_offload import WeightOffloadHook, offload_module_weights

# Gradient offloading (post-accumulate hooks)
from .gradient_offload import GradientOffloadHook

# Optimizer offloading (GPU/CPU split)
from .hybrid_optimizer import HybridDeviceOptimizer

# Building blocks
from .tensor_pool import TensorPool
from .offload_group import OffloadTensorGroup
from .chunk_handler import ChunkOffloadHandler

__all__ = [
    # General-purpose
    "TensorOffloader",
    "OffloadHandle",
    # Activation offloading
    "OffloadManager",
    "ActivationOffloadContext",
    "group_start",
    "group_commit",
    "flush_delayed_groups",
    "disable_offload",
    "enable_offload",
    "forward_record",
    "backward_record",
    # Weight offloading
    "WeightOffloadHook",
    "offload_module_weights",
    # Gradient offloading
    "GradientOffloadHook",
    # Optimizer offloading
    "HybridDeviceOptimizer",
    # Building blocks
    "TensorPool",
    "OffloadTensorGroup",
    "ChunkOffloadHandler",
]
