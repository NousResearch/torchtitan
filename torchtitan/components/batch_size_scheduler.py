# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch Size Scheduler - Orthogonal to Data Stages and LR Scheduler.

Handles dynamic batch size scheduling based on consumed_samples only.
No knowledge of data stages, datasets, or learning rate.

Supports three modes:
- constant: Fixed batch size (default, backward compatible)
- linear: Smooth interpolation (DeepSeek-V3 style)
- increment: Step-wise increments (Megatron style)
"""

from dataclasses import dataclass, field

from torchtitan.tools.logging import logger


@dataclass
class BatchSizeState:
    """State for batch size computation. Owned by training loop."""

    consumed_samples: int = 0


class BatchSizeScheduler:
    """Base class for batch size schedulers. Stateless - all state in BatchSizeState."""

    def get_batch_size(self, state: BatchSizeState) -> int:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


@dataclass
class ConstantBatchSize(BatchSizeScheduler):
    """Fixed batch size throughout training."""

    batch_size: int

    def get_batch_size(self, state: BatchSizeState) -> int:
        return self.batch_size

    def get_name(self) -> str:
        return "constant"


@dataclass
class LinearRampup(BatchSizeScheduler):
    """
    Smooth linear interpolation from start to end.
    Used by: DeepSeek-V3 (3072 -> 15360 over 469B tokens)
    """

    start_batch_size: int
    end_batch_size: int
    rampup_samples: int

    def get_batch_size(self, state: BatchSizeState) -> int:
        if state.consumed_samples >= self.rampup_samples:
            return self.end_batch_size
        progress = state.consumed_samples / self.rampup_samples
        return int(
            self.start_batch_size
            + progress * (self.end_batch_size - self.start_batch_size)
        )

    def get_name(self) -> str:
        return "linear"


@dataclass
class IncrementRampup(BatchSizeScheduler):
    """
    Megatron-style step-wise increments at regular intervals.
    """

    start_batch_size: int
    end_batch_size: int
    increment: int
    rampup_samples: int
    _samples_per_increment: float = field(init=False, default=0.0)

    def __post_init__(self):
        diff = self.end_batch_size - self.start_batch_size
        num_increments = diff // self.increment if self.increment > 0 else 0
        self._samples_per_increment = (
            self.rampup_samples / num_increments if num_increments > 0 else float("inf")
        )

    def get_batch_size(self, state: BatchSizeState) -> int:
        if state.consumed_samples >= self.rampup_samples:
            return self.end_batch_size
        steps = int(state.consumed_samples / self._samples_per_increment)
        return min(self.start_batch_size + steps * self.increment, self.end_batch_size)

    def get_name(self) -> str:
        return "increment"


class BatchSizeManager:
    """
    Wraps scheduler. Handles alignment and grad accum computation.

    NOT responsible for: tracking consumed_samples, data loading, LR.
    """

    def __init__(
        self,
        scheduler: BatchSizeScheduler,
        micro_batch_size: int,
        data_parallel_size: int,
    ):
        self.scheduler = scheduler
        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size
        self._unit = micro_batch_size * data_parallel_size
        self._last_batch_size: int | None = None

    def get_batch_size(self, state: BatchSizeState) -> int:
        """Get aligned global batch size."""
        raw = self.scheduler.get_batch_size(state)
        return (raw // self._unit) * self._unit

    def get_grad_accum_steps(self, state: BatchSizeState) -> int:
        """Get gradient accumulation steps."""
        return self.get_batch_size(state) // self._unit

    def did_change(self, state: BatchSizeState) -> tuple[bool, int, int]:
        """Check if batch size changed. Returns (changed, old, new)."""
        current = self.get_batch_size(state)
        old = self._last_batch_size
        changed = old is not None and current != old
        self._last_batch_size = current
        return changed, old if old is not None else current, current


def build_batch_size_manager(
    job_config,
    dp_degree: int,
) -> BatchSizeManager:
    """Factory to build manager from config."""
    bs_cfg = job_config.batch_size_scheduler
    training = job_config.training

    # Compute target global batch size (same logic as train.py)
    target = training.global_batch_size
    if target < 0:
        target = training.local_batch_size * dp_degree

    micro_batch_size = training.local_batch_size

    # Build appropriate scheduler based on mode
    # Default to constant if no rampup configured (backward compatible)
    if (
        bs_cfg.mode == "constant"
        or bs_cfg.start_batch_size <= 0
        or bs_cfg.rampup_samples <= 0
    ):
        scheduler = ConstantBatchSize(batch_size=target)
        logger.info(f"Batch size scheduler: constant at {target}")
    elif bs_cfg.mode == "linear":
        scheduler = LinearRampup(
            start_batch_size=bs_cfg.start_batch_size,
            end_batch_size=target,
            rampup_samples=bs_cfg.rampup_samples,
        )
        logger.info(
            f"Batch size scheduler: linear rampup {bs_cfg.start_batch_size} -> {target} "
            f"over {bs_cfg.rampup_samples} samples"
        )
    elif bs_cfg.mode == "increment":
        increment = (
            bs_cfg.increment if bs_cfg.increment > 0 else bs_cfg.start_batch_size
        )
        scheduler = IncrementRampup(
            start_batch_size=bs_cfg.start_batch_size,
            end_batch_size=target,
            increment=increment,
            rampup_samples=bs_cfg.rampup_samples,
        )
        logger.info(
            f"Batch size scheduler: increment rampup {bs_cfg.start_batch_size} -> {target} "
            f"(increment={increment}) over {bs_cfg.rampup_samples} samples"
        )
    else:
        raise ValueError(f"Unknown batch_size_scheduler.mode: {bs_cfg.mode}")

    return BatchSizeManager(scheduler, micro_batch_size, dp_degree)
