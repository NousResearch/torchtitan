#!/usr/bin/env python3
"""
Comprehensive CPU-GPU boundary analysis for PyTorch profiler traces.

Analyzes the complete lifecycle of operations:
1. CPU launch overhead (Python/C++ overhead, cudaLaunchKernel calls)
2. GPU kernel execution time
3. CPU-GPU synchronization overhead (cudaStreamSynchronize, cudaEventSynchronize)
4. Pipeline bubbles and idle time
"""

import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class OperationStats:
    """Statistics for a single operation type."""

    name: str
    count: int = 0

    # CPU-side times (microseconds)
    cpu_time_total: float = 0
    cpu_time_self: float = 0  # Excluding children

    # GPU kernel times (microseconds)
    kernel_time: float = 0
    kernel_count: int = 0

    # CUDA API overhead (microseconds)
    cuda_launch_time: float = 0  # cudaLaunchKernel
    cuda_memcpy_time: float = 0  # cudaMemcpy*
    cuda_sync_time: float = 0  # cudaStreamSynchronize, cudaEventSynchronize

    # NCCL communication (microseconds)
    nccl_time: float = 0
    nccl_count: int = 0

    # Individual samples for percentile analysis
    cpu_samples: List[float] = field(default_factory=list)
    kernel_samples: List[float] = field(default_factory=list)

    def total_time(self) -> float:
        """Total wall-clock time including all overhead."""
        return self.cpu_time_total + self.kernel_time + self.cuda_sync_time


@dataclass
class TimelineEvent:
    """A single event in the timeline."""

    name: str
    category: str
    start_us: float
    end_us: float
    tid: int
    pid: int

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us


def parse_trace(trace_path: str) -> Tuple[List[TimelineEvent], float, float]:
    """Parse PyTorch profiler JSON trace."""
    print(f"Parsing trace: {trace_path}")

    with open(trace_path, "r") as f:
        data = json.load(f)

    events = []
    min_ts = float("inf")
    max_ts = 0

    for event in data.get("traceEvents", []):
        # Skip metadata events
        if event.get("ph") not in ["X", "B", "E"]:
            continue

        name = event.get("name", "")
        cat = event.get("cat", "")
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)

        if event["ph"] == "X":  # Complete event
            start_us = ts
            end_us = ts + dur
        else:
            continue  # Skip begin/end pairs for now

        if dur > 0:  # Only include events with duration
            events.append(
                TimelineEvent(
                    name=name,
                    category=cat,
                    start_us=start_us,
                    end_us=end_us,
                    tid=event.get("tid", 0),
                    pid=event.get("pid", 0),
                )
            )

            min_ts = min(min_ts, start_us)
            max_ts = max(max_ts, end_us)

    total_duration_us = max_ts - min_ts
    print(f"Parsed {len(events)} events")
    print(f"Trace duration: {total_duration_us / 1e6:.2f} seconds\n")

    return events, min_ts, total_duration_us


def categorize_events(events: List[TimelineEvent]) -> Dict[str, List[TimelineEvent]]:
    """Categorize events by type."""
    categorized = defaultdict(list)

    for event in events:
        cat = event.category.lower()
        name = event.name.lower()

        # Categorize by event type
        if "kernel" in cat:
            categorized["gpu_kernels"].append(event)
        elif "nccl" in name or "ncclKernel" in event.name:
            categorized["nccl"].append(event)
        elif "cuda" in cat or "cuda_runtime" in cat:
            if "cudaLaunchKernel" in event.name:
                categorized["cuda_launch"].append(event)
            elif "cudaMemcpy" in event.name or "cudaMemset" in event.name:
                categorized["cuda_memcpy"].append(event)
            elif "Synchronize" in event.name or "cudaDeviceSynchronize" in event.name:
                categorized["cuda_sync"].append(event)
            else:
                categorized["cuda_other"].append(event)
        elif "cpu_op" in cat or "python" in cat:
            categorized["cpu_ops"].append(event)
        else:
            categorized["other"].append(event)

    return categorized


def analyze_operations(
    events: List[TimelineEvent], categorized: Dict[str, List[TimelineEvent]]
) -> Dict[str, OperationStats]:
    """Analyze operations and compute statistics."""
    stats = defaultdict(OperationStats)

    # Group CPU operations by name
    cpu_ops_by_name = defaultdict(list)
    for event in categorized.get("cpu_ops", []):
        # Clean up operation name
        op_name = event.name.split("(")[0].strip()
        cpu_ops_by_name[op_name].append(event)

    # Analyze each operation type
    for op_name, op_events in cpu_ops_by_name.items():
        stat = OperationStats(name=op_name, count=len(op_events))

        for event in op_events:
            stat.cpu_time_total += event.duration_us
            stat.cpu_samples.append(event.duration_us)

        stats[op_name] = stat

    # Add GPU kernel stats
    kernel_by_name = defaultdict(list)
    for event in categorized.get("gpu_kernels", []):
        kernel_name = event.name
        kernel_by_name[kernel_name].append(event)

    for kernel_name, kernel_events in kernel_by_name.items():
        # Try to map kernel to CPU op (simplified)
        base_name = kernel_name.split("<")[0].strip()

        if base_name not in stats:
            stats[base_name] = OperationStats(name=base_name)

        stat = stats[base_name]
        stat.kernel_count += len(kernel_events)

        for event in kernel_events:
            stat.kernel_time += event.duration_us
            stat.kernel_samples.append(event.duration_us)

    # Add CUDA API overhead
    for event in categorized.get("cuda_launch", []):
        # Generic launch overhead
        if "cuda_launch" not in stats:
            stats["cuda_launch"] = OperationStats(name="cuda_launch")
        stats["cuda_launch"].cuda_launch_time += event.duration_us
        stats["cuda_launch"].count += 1

    for event in categorized.get("cuda_memcpy", []):
        memcpy_type = event.name
        if memcpy_type not in stats:
            stats[memcpy_type] = OperationStats(name=memcpy_type)
        stats[memcpy_type].cuda_memcpy_time += event.duration_us
        stats[memcpy_type].count += 1

    for event in categorized.get("cuda_sync", []):
        sync_type = event.name
        if sync_type not in stats:
            stats[sync_type] = OperationStats(name=sync_type)
        stats[sync_type].cuda_sync_time += event.duration_us
        stats[sync_type].count += 1

    # NCCL operations
    for event in categorized.get("nccl", []):
        nccl_op = event.name
        if nccl_op not in stats:
            stats[nccl_op] = OperationStats(name=nccl_op)
        stats[nccl_op].nccl_time += event.duration_us
        stats[nccl_op].nccl_count += 1

    return stats


def print_summary(stats: Dict[str, OperationStats], total_duration_us: float):
    """Print comprehensive summary."""
    print("=" * 100)
    print("CPU-GPU BOUNDARY ANALYSIS")
    print("=" * 100)

    total_duration_s = total_duration_us / 1e6

    # Sort by total time
    sorted_ops = sorted(stats.values(), key=lambda x: x.total_time(), reverse=True)

    # === SECTION 1: Top Operations by Total Time ===
    print("\n" + "=" * 100)
    print("1. TOP OPERATIONS BY TOTAL TIME (CPU + GPU + Sync)")
    print("=" * 100)
    print(
        f"{'Operation':<50} {'Count':>8} {'Total (s)':>12} {'% Step':>8} {'CPU (s)':>12} {'Kernel (s)':>12} {'Sync (s)':>12}"
    )
    print("-" * 100)

    for stat in sorted_ops[:30]:
        total_s = stat.total_time() / 1e6
        pct = (stat.total_time() / total_duration_us) * 100
        cpu_s = stat.cpu_time_total / 1e6
        kernel_s = stat.kernel_time / 1e6
        sync_s = stat.cuda_sync_time / 1e6

        if total_s < 0.01:  # Skip very small operations
            continue

        print(
            f"{stat.name:<50} {stat.count:>8} {total_s:>12.3f} {pct:>7.2f}% {cpu_s:>12.3f} {kernel_s:>12.3f} {sync_s:>12.3f}"
        )

    # === SECTION 2: CPU Operations Analysis ===
    print("\n" + "=" * 100)
    print("2. CPU OPERATIONS (Launch Overhead & Python/C++)")
    print("=" * 100)
    print(
        f"{'Operation':<50} {'Count':>8} {'Total (s)':>12} {'% Step':>8} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10}"
    )
    print("-" * 100)

    cpu_ops = [s for s in sorted_ops if s.cpu_time_total > 0]
    cpu_ops.sort(key=lambda x: x.cpu_time_total, reverse=True)

    for stat in cpu_ops[:30]:
        cpu_s = stat.cpu_time_total / 1e6
        pct = (stat.cpu_time_total / total_duration_us) * 100
        avg_ms = (stat.cpu_time_total / stat.count / 1000) if stat.count > 0 else 0

        if stat.cpu_samples:
            p50_ms = statistics.median(stat.cpu_samples) / 1000
            p95_ms = (
                statistics.quantiles(stat.cpu_samples, n=20)[18] / 1000
                if len(stat.cpu_samples) > 1
                else p50_ms
            )
        else:
            p50_ms = p95_ms = 0

        if cpu_s < 0.01:
            continue

        print(
            f"{stat.name:<50} {stat.count:>8} {cpu_s:>12.3f} {pct:>7.2f}% {avg_ms:>10.3f} {p50_ms:>10.3f} {p95_ms:>10.3f}"
        )

    # === SECTION 3: GPU Kernel Execution ===
    print("\n" + "=" * 100)
    print("3. GPU KERNEL EXECUTION TIME")
    print("=" * 100)
    print(
        f"{'Kernel':<50} {'Count':>8} {'Total (s)':>12} {'% Step':>8} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10}"
    )
    print("-" * 100)

    kernel_ops = [s for s in sorted_ops if s.kernel_time > 0]
    kernel_ops.sort(key=lambda x: x.kernel_time, reverse=True)

    for stat in kernel_ops[:30]:
        kernel_s = stat.kernel_time / 1e6
        pct = (stat.kernel_time / total_duration_us) * 100
        avg_ms = (
            (stat.kernel_time / stat.kernel_count / 1000)
            if stat.kernel_count > 0
            else 0
        )

        if stat.kernel_samples:
            p50_ms = statistics.median(stat.kernel_samples) / 1000
            p95_ms = (
                statistics.quantiles(stat.kernel_samples, n=20)[18] / 1000
                if len(stat.kernel_samples) > 1
                else p50_ms
            )
        else:
            p50_ms = p95_ms = 0

        if kernel_s < 0.01:
            continue

        print(
            f"{stat.name:<50} {stat.kernel_count:>8} {kernel_s:>12.3f} {pct:>7.2f}% {avg_ms:>10.3f} {p50_ms:>10.3f} {p95_ms:>10.3f}"
        )

    # === SECTION 4: CUDA Synchronization Overhead ===
    print("\n" + "=" * 100)
    print("4. CUDA SYNCHRONIZATION OVERHEAD")
    print("=" * 100)
    print(
        f"{'Sync Operation':<50} {'Count':>8} {'Total (s)':>12} {'% Step':>8} {'Avg (ms)':>10}"
    )
    print("-" * 100)

    sync_ops = [s for s in sorted_ops if s.cuda_sync_time > 0]
    sync_ops.sort(key=lambda x: x.cuda_sync_time, reverse=True)

    for stat in sync_ops:
        sync_s = stat.cuda_sync_time / 1e6
        pct = (stat.cuda_sync_time / total_duration_us) * 100
        avg_ms = (stat.cuda_sync_time / stat.count / 1000) if stat.count > 0 else 0

        print(
            f"{stat.name:<50} {stat.count:>8} {sync_s:>12.3f} {pct:>7.2f}% {avg_ms:>10.3f}"
        )

    # === SECTION 5: Memory Operations ===
    print("\n" + "=" * 100)
    print("5. MEMORY OPERATIONS (cudaMemcpy, cudaMemset)")
    print("=" * 100)
    print(
        f"{'Operation':<50} {'Count':>8} {'Total (s)':>12} {'% Step':>8} {'Avg (ms)':>10}"
    )
    print("-" * 100)

    memcpy_ops = [s for s in sorted_ops if s.cuda_memcpy_time > 0]
    memcpy_ops.sort(key=lambda x: x.cuda_memcpy_time, reverse=True)

    for stat in memcpy_ops:
        memcpy_s = stat.cuda_memcpy_time / 1e6
        pct = (stat.cuda_memcpy_time / total_duration_us) * 100
        avg_ms = (stat.cuda_memcpy_time / stat.count / 1000) if stat.count > 0 else 0

        print(
            f"{stat.name:<50} {stat.count:>8} {memcpy_s:>12.3f} {pct:>7.2f}% {avg_ms:>10.3f}"
        )

    # === SECTION 6: NCCL Communication ===
    print("\n" + "=" * 100)
    print("6. NCCL COLLECTIVE COMMUNICATION")
    print("=" * 100)
    print(
        f"{'NCCL Operation':<50} {'Count':>8} {'Total (s)':>12} {'% Step':>8} {'Avg (ms)':>10}"
    )
    print("-" * 100)

    nccl_ops = [s for s in sorted_ops if s.nccl_time > 0]
    nccl_ops.sort(key=lambda x: x.nccl_time, reverse=True)

    for stat in nccl_ops:
        nccl_s = stat.nccl_time / 1e6
        pct = (stat.nccl_time / total_duration_us) * 100
        avg_ms = (stat.nccl_time / stat.nccl_count / 1000) if stat.nccl_count > 0 else 0

        print(
            f"{stat.name:<50} {stat.nccl_count:>8} {nccl_s:>12.3f} {pct:>7.2f}% {avg_ms:>10.3f}"
        )

    # === SECTION 7: Time Breakdown ===
    print("\n" + "=" * 100)
    print("7. OVERALL TIME BREAKDOWN")
    print("=" * 100)

    total_cpu = sum(s.cpu_time_total for s in stats.values()) / 1e6
    total_kernel = sum(s.kernel_time for s in stats.values()) / 1e6
    total_sync = sum(s.cuda_sync_time for s in stats.values()) / 1e6
    total_memcpy = sum(s.cuda_memcpy_time for s in stats.values()) / 1e6
    total_nccl = sum(s.nccl_time for s in stats.values()) / 1e6

    print(f"Total trace duration:          {total_duration_s:>10.2f} seconds")
    print(
        f"CPU operations:                {total_cpu:>10.2f} seconds ({total_cpu/total_duration_s*100:>5.1f}%)"
    )
    print(
        f"GPU kernel execution:          {total_kernel:>10.2f} seconds ({total_kernel/total_duration_s*100:>5.1f}%)"
    )
    print(
        f"CUDA synchronization:          {total_sync:>10.2f} seconds ({total_sync/total_duration_s*100:>5.1f}%)"
    )
    print(
        f"Memory operations (memcpy):    {total_memcpy:>10.2f} seconds ({total_memcpy/total_duration_s*100:>5.1f}%)"
    )
    print(
        f"NCCL communication:            {total_nccl:>10.2f} seconds ({total_nccl/total_duration_s*100:>5.1f}%)"
    )

    print(
        "\nNOTE: Times may overlap due to concurrent execution on CPU and GPU streams."
    )
    print("=" * 100)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_cpu_gpu_boundary.py <trace.json>")
        sys.exit(1)

    trace_path = sys.argv[1]

    # Parse trace
    events, min_ts, total_duration_us = parse_trace(trace_path)

    # Categorize events
    categorized = categorize_events(events)

    print(f"Event categories:")
    for cat, evt_list in categorized.items():
        print(f"  {cat}: {len(evt_list)} events")
    print()

    # Analyze operations
    stats = analyze_operations(events, categorized)

    # Print comprehensive summary
    print_summary(stats, total_duration_us)


if __name__ == "__main__":
    main()
