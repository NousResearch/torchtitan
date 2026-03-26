# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
#
# Originally forked from NVIDIA Megatron-LM:
#   megatron/core/pipeline_parallel/fine_grained_activation_offload.py
#   megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py
# Refactored to be framework-agnostic and portable.

"""Fine-grained CPU offload engine for activation and optimizer state offloading.

This package provides two independent mechanisms:

1. **Activation Offloading** — offloads intermediate activations to pinned CPU
   memory during forward, reloads them during backward. Uses dedicated D2H/H2D
   CUDA streams with event-based synchronization. Per-module granularity via
   ActivationOffloadContext / group_start / group_commit.

2. **Optimizer State Offloading** (optional) — HybridDeviceOptimizer splits
   optimizer state across GPU and CPU with async gradient sync.

Quick start:

    from torchtitan.distributed.cpu_offload import (
        ActivationOffloadContext,
        group_commit,
        OffloadManager,
    )

    # At the start of each microbatch forward:
    ActivationOffloadContext.init_chunk_handler()

    # Wrap each module's forward:
    with ActivationOffloadContext(should_offload, input_tensor, "my_module") as x:
        output = self.linear(x)
    output = group_commit(output, "my_module", forced_released_tensors=[input_tensor])

    # At iteration boundary:
    ActivationOffloadContext.reset()
"""

# Core manager
from .offload_manager import OffloadManager

# User-facing context manager and functions
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

# Building blocks (for advanced usage)
from .tensor_pool import TensorPool
from .offload_group import OffloadTensorGroup
from .chunk_handler import ChunkOffloadHandler

# Optimizer offloading (optional, independent from activation offloading)
from .hybrid_optimizer import HybridDeviceOptimizer

__all__ = [
    # Core
    "OffloadManager",
    # User API
    "ActivationOffloadContext",
    "group_start",
    "group_commit",
    "flush_delayed_groups",
    "disable_offload",
    "enable_offload",
    "forward_record",
    "backward_record",
    # Building blocks
    "TensorPool",
    "OffloadTensorGroup",
    "ChunkOffloadHandler",
    # Optimizer
    "HybridDeviceOptimizer",
]
