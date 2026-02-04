# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive memory profiler for initialization stages.

Tracks memory allocation at every stage of model initialization,
parallelization, and optimizer creation to identify memory regressions.
"""

import gc
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """A snapshot of memory state at a specific point."""

    stage: str
    pytorch_reserved_gb: float
    pytorch_allocated_gb: float
    pytorch_active_gb: float
    nvidia_smi_used_gb: float
    peak_reserved_gb: float
    num_tensors: int
    tensor_breakdown: dict[str, float] = field(default_factory=dict)


class InitMemoryProfiler:
    """
    Profile memory usage throughout model initialization.

    Tracks memory at each stage to pinpoint where memory is allocated.
    Useful for debugging OOM issues and comparing memory between implementations.
    """

    def __init__(self, enabled: bool = True, verbose: bool = True):
        self.enabled = enabled
        self.verbose = verbose
        self.snapshots: list[MemorySnapshot] = []
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None

        if self.enabled and self.device is not None:
            # Reset peak stats for accurate tracking
            torch.cuda.reset_peak_memory_stats(self.device)
            logger.info("=" * 100)
            logger.info("INIT MEMORY PROFILER ENABLED")
            logger.info("=" * 100)

    def _get_nvidia_smi_memory(self) -> float:
        """Get memory from nvidia-smi in GB."""
        if self.device is None:
            return 0.0
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.device),
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / 1024  # MB to GB
        except Exception:
            pass
        return 0.0

    def _count_tensors_by_device(self) -> dict[str, int]:
        """Count tensors by device location."""
        counts = {"cuda": 0, "cpu": 0, "meta": 0, "other": 0}
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor):
                    device_type = str(obj.device).split(":")[0]
                    if device_type in counts:
                        counts[device_type] += 1
                    else:
                        counts["other"] += 1
            except Exception:
                pass
        return counts

    def _get_tensor_memory_breakdown(self) -> dict[str, float]:
        """Get memory breakdown by tensor size categories."""
        breakdown = {
            "tiny_(<1MB)": 0.0,
            "small_(1-10MB)": 0.0,
            "medium_(10-100MB)": 0.0,
            "large_(100MB-1GB)": 0.0,
            "huge_(>1GB)": 0.0,
        }

        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    size_mb = obj.element_size() * obj.nelement() / 1e6
                    if size_mb < 1:
                        breakdown["tiny_(<1MB)"] += size_mb
                    elif size_mb < 10:
                        breakdown["small_(1-10MB)"] += size_mb
                    elif size_mb < 100:
                        breakdown["medium_(10-100MB)"] += size_mb
                    elif size_mb < 1000:
                        breakdown["large_(100MB-1GB)"] += size_mb
                    else:
                        breakdown["huge_(>1GB)"] += size_mb
            except Exception:
                pass

        # Convert to GB
        return {k: v / 1000 for k, v in breakdown.items()}

    def snapshot(self, stage: str, sync: bool = True, clear_cache: bool = False):
        """
        Take a memory snapshot at the current stage.

        Args:
            stage: Name of the current stage (e.g., "after_mesh_creation")
            sync: Whether to synchronize CUDA before measuring
            clear_cache: Whether to clear cache before measuring
        """
        if not self.enabled or self.device is None:
            return

        if sync:
            torch.cuda.synchronize(self.device)

        if clear_cache:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

        stats = torch.cuda.memory_stats(self.device)
        tensor_counts = self._count_tensors_by_device()

        snapshot = MemorySnapshot(
            stage=stage,
            pytorch_reserved_gb=torch.cuda.memory_reserved(self.device) / 1e9,
            pytorch_allocated_gb=torch.cuda.memory_allocated(self.device) / 1e9,
            pytorch_active_gb=stats.get("active_bytes.all.current", 0) / 1e9,
            nvidia_smi_used_gb=self._get_nvidia_smi_memory(),
            peak_reserved_gb=stats.get("reserved_bytes.all.peak", 0) / 1e9,
            num_tensors=tensor_counts["cuda"],
            tensor_breakdown=self._get_tensor_memory_breakdown()
            if self.verbose
            else {},
        )

        self.snapshots.append(snapshot)

        # Log immediately
        self._log_snapshot(snapshot, len(self.snapshots) - 1)

        return snapshot

    def _log_snapshot(self, snap: MemorySnapshot, idx: int):
        """Log a single snapshot."""
        prev_reserved = self.snapshots[idx - 1].pytorch_reserved_gb if idx > 0 else 0
        delta = snap.pytorch_reserved_gb - prev_reserved
        delta_str = f"({delta:+.2f} GB)" if idx > 0 else ""

        logger.info("-" * 100)
        logger.info(f"[INIT_MEM] Stage {idx}: {snap.stage}")
        logger.info(
            f"[INIT_MEM]   PyTorch Reserved:  {snap.pytorch_reserved_gb:8.3f} GB {delta_str}"
        )
        logger.info(
            f"[INIT_MEM]   PyTorch Allocated: {snap.pytorch_allocated_gb:8.3f} GB"
        )
        logger.info(f"[INIT_MEM]   PyTorch Active:    {snap.pytorch_active_gb:8.3f} GB")
        logger.info(
            f"[INIT_MEM]   nvidia-smi Used:   {snap.nvidia_smi_used_gb:8.3f} GB"
        )
        logger.info(f"[INIT_MEM]   Peak Reserved:     {snap.peak_reserved_gb:8.3f} GB")
        logger.info(f"[INIT_MEM]   CUDA Tensors:      {snap.num_tensors:8d}")

        if self.verbose and snap.tensor_breakdown:
            logger.info(f"[INIT_MEM]   Tensor breakdown by size:")
            for category, size_gb in snap.tensor_breakdown.items():
                if size_gb > 0.001:  # Only show if > 1MB
                    logger.info(f"[INIT_MEM]     {category}: {size_gb:.3f} GB")

    def snapshot_with_gc(self, stage: str):
        """Take snapshot after forcing garbage collection."""
        if not self.enabled:
            return
        gc.collect()
        torch.cuda.empty_cache()
        self.snapshot(f"{stage}_after_gc", sync=True, clear_cache=True)

    def compare_with_baseline(self, baseline_snapshots: list[MemorySnapshot]):
        """Compare current snapshots with a baseline (from another repo/run)."""
        logger.info("=" * 100)
        logger.info("MEMORY COMPARISON WITH BASELINE")
        logger.info("=" * 100)

        for i, (current, baseline) in enumerate(
            zip(self.snapshots, baseline_snapshots)
        ):
            if current.stage != baseline.stage:
                logger.warning(f"Stage mismatch: {current.stage} vs {baseline.stage}")
                continue

            diff = current.pytorch_reserved_gb - baseline.pytorch_reserved_gb
            logger.info(
                f"Stage: {current.stage:40s} | "
                f"Current: {current.pytorch_reserved_gb:6.2f} GB | "
                f"Baseline: {baseline.pytorch_reserved_gb:6.2f} GB | "
                f"Diff: {diff:+6.2f} GB"
            )

    def get_summary(self) -> str:
        """Get a summary of all snapshots."""
        if not self.snapshots:
            return "No snapshots recorded"

        lines = [
            "",
            "=" * 100,
            "INITIALIZATION MEMORY PROFILER SUMMARY",
            "=" * 100,
            "",
            f"{'Stage':<50} {'Reserved':>12} {'Delta':>12} {'Allocated':>12} {'Tensors':>10}",
            "-" * 100,
        ]

        prev_reserved = 0
        for snap in self.snapshots:
            delta = snap.pytorch_reserved_gb - prev_reserved
            delta_str = f"{delta:+.2f} GB" if prev_reserved > 0 else "---"
            lines.append(
                f"{snap.stage:<50} "
                f"{snap.pytorch_reserved_gb:10.2f} GB "
                f"{delta_str:>12} "
                f"{snap.pytorch_allocated_gb:10.2f} GB "
                f"{snap.num_tensors:>10d}"
            )
            prev_reserved = snap.pytorch_reserved_gb

        # Find biggest memory increases
        lines.append("")
        lines.append("TOP 5 MEMORY INCREASES:")
        lines.append("-" * 50)

        deltas = []
        for i in range(1, len(self.snapshots)):
            delta = (
                self.snapshots[i].pytorch_reserved_gb
                - self.snapshots[i - 1].pytorch_reserved_gb
            )
            deltas.append((self.snapshots[i].stage, delta))

        deltas.sort(key=lambda x: x[1], reverse=True)
        for stage, delta in deltas[:5]:
            if delta > 0:
                lines.append(f"  {stage}: +{delta:.2f} GB")

        lines.append("")
        lines.append(
            f"FINAL MEMORY: {self.snapshots[-1].pytorch_reserved_gb:.2f} GB reserved"
        )
        lines.append(
            f"PEAK MEMORY:  {max(s.peak_reserved_gb for s in self.snapshots):.2f} GB"
        )
        lines.append("=" * 100)

        return "\n".join(lines)

    def dump_to_file(self, filepath: str):
        """Dump snapshots to a JSON file for later comparison."""
        import json

        data = []
        for snap in self.snapshots:
            data.append(
                {
                    "stage": snap.stage,
                    "pytorch_reserved_gb": snap.pytorch_reserved_gb,
                    "pytorch_allocated_gb": snap.pytorch_allocated_gb,
                    "pytorch_active_gb": snap.pytorch_active_gb,
                    "nvidia_smi_used_gb": snap.nvidia_smi_used_gb,
                    "peak_reserved_gb": snap.peak_reserved_gb,
                    "num_tensors": snap.num_tensors,
                    "tensor_breakdown": snap.tensor_breakdown,
                }
            )

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Memory profile dumped to {filepath}")


# Global profiler instance for easy access
_global_profiler: InitMemoryProfiler | None = None


def get_init_memory_profiler(
    enabled: bool = True, verbose: bool = True
) -> InitMemoryProfiler:
    """Get or create the global init memory profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = InitMemoryProfiler(enabled=enabled, verbose=verbose)
    return _global_profiler


def mem_snapshot(stage: str, sync: bool = True, clear_cache: bool = False):
    """Convenience function to take a memory snapshot."""
    if _global_profiler is not None:
        _global_profiler.snapshot(stage, sync=sync, clear_cache=clear_cache)
