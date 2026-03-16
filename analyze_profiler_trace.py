#!/usr/bin/env python3
"""
Analyze PyTorch profiler trace JSON for CPU-boundary bottleneck identification.

The chrome trace only has CPU-side events. Each cpu_op's duration is the CPU wall-clock
time spent in that operation (dispatch + waiting for GPU + nested ops).

Key insight: CPU ops are NESTED. We need to find TOP-LEVEL operations to avoid
double-counting. An op's "self time" = its duration minus all child ops' durations.
"""

import json
import sys
from collections import defaultdict


def analyze_trace(trace_path):
    print(f"Loading trace: {trace_path}")
    with open(trace_path, 'r') as f:
        data = json.load(f)

    events = data.get('traceEvents', data)

    # Collect all cpu_op X events with their timing
    cpu_ops = []
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get('ph') != 'X':
            continue
        cat = e.get('cat', '')
        if cat not in ('cpu_op', 'user_annotation'):
            continue
        name = e.get('name', '')
        dur = e.get('dur', 0)  # microseconds
        ts = e.get('ts', 0)
        tid = e.get('tid', 0)
        cpu_ops.append({
            'name': name,
            'dur_us': dur,
            'ts': ts,
            'end': ts + dur,
            'tid': tid,
            'cat': cat,
            'args': e.get('args', {}),
        })

    # Sort by start time per thread
    cpu_ops.sort(key=lambda x: (x['tid'], x['ts']))

    print(f"  Total cpu_op events: {len(cpu_ops)}")

    # =========================================================================
    # Compute SELF TIME for each operation (excluding nested children)
    # =========================================================================
    # Group by thread
    by_thread = defaultdict(list)
    for op in cpu_ops:
        by_thread[op['tid']].append(op)

    # For each thread, compute self time using a stack-based approach
    for tid, ops in by_thread.items():
        # ops are sorted by ts
        stack = []  # stack of (op, children_time)
        for op in ops:
            # Pop ops that ended before this one started
            while stack and stack[-1][0]['end'] <= op['ts']:
                stack.pop()

            # If stack is not empty, this op is a child of the top
            if stack:
                stack[-1][1].append(op['dur_us'])

            op['_children_dur'] = []
            stack.append((op, op['_children_dur']))

        # Compute self time
        for op in ops:
            children_total = sum(op['_children_dur'])
            op['self_us'] = max(0, op['dur_us'] - children_total)

    # =========================================================================
    # 1. TOP OPERATIONS BY INCLUSIVE TIME (wall clock including children)
    # =========================================================================
    print("\n" + "=" * 110)
    print("TOP 30 OPERATIONS BY INCLUSIVE WALL TIME (CPU boundary)")
    print("Inclusive = total wall clock for this op (includes all nested/child operations)")
    print("=" * 110)

    incl_stats = defaultdict(lambda: {'count': 0, 'total_us': 0})
    for op in cpu_ops:
        if op['cat'] == 'cpu_op':
            incl_stats[op['name']]['count'] += 1
            incl_stats[op['name']]['total_us'] += op['dur_us']

    sorted_incl = sorted(incl_stats.items(), key=lambda x: -x[1]['total_us'])
    total_time = sum(v['total_us'] for v in incl_stats.values())

    print(f"\n{'#':<4} {'Operation':<65} {'Incl (ms)':>10} {'Count':>7} {'Avg (ms)':>10} {'%':>6}")
    print("-" * 110)
    for i, (name, st) in enumerate(sorted_incl[:30]):
        t = st['total_us'] / 1000
        avg = t / st['count']
        pct = st['total_us'] / total_time * 100
        print(f"{i+1:<4} {name[:64]:<65} {t:>10.1f} {st['count']:>7} {avg:>10.3f} {pct:>5.1f}%")

    # =========================================================================
    # 2. TOP OPERATIONS BY SELF TIME (actual time spent in this op, not children)
    # =========================================================================
    print("\n" + "=" * 110)
    print("TOP 30 OPERATIONS BY SELF TIME (excludes nested children)")
    print("Self time = the ACTUAL CPU cost of this specific op, not its children")
    print("=" * 110)

    self_stats = defaultdict(lambda: {'count': 0, 'total_self_us': 0})
    for op in cpu_ops:
        if op['cat'] == 'cpu_op':
            self_stats[op['name']]['count'] += 1
            self_stats[op['name']]['total_self_us'] += op.get('self_us', 0)

    sorted_self = sorted(self_stats.items(), key=lambda x: -x[1]['total_self_us'])
    total_self = sum(v['total_self_us'] for v in self_stats.values())

    print(f"\n{'#':<4} {'Operation':<65} {'Self (ms)':>10} {'Count':>7} {'Avg (ms)':>10} {'%':>6}")
    print("-" * 110)
    for i, (name, st) in enumerate(sorted_self[:30]):
        t = st['total_self_us'] / 1000
        avg = t / st['count'] if st['count'] > 0 else 0
        pct = st['total_self_us'] / total_self * 100 if total_self > 0 else 0
        print(f"{i+1:<4} {name[:64]:<65} {t:>10.1f} {st['count']:>7} {avg:>10.3f} {pct:>5.1f}%")

    # =========================================================================
    # 3. CATEGORY BREAKDOWN BY SELF TIME
    # =========================================================================
    print("\n" + "=" * 110)
    print("CATEGORY BREAKDOWN BY SELF TIME")
    print("=" * 110)

    categories = defaultdict(float)
    for name, st in self_stats.items():
        t = st['total_self_us']
        nl = name.lower()
        if 'nccl' in nl or 'all_reduce' in nl or 'reduce_scatter' in nl or 'all_gather' in nl or 'all_to_all' in nl or 'c10d' in nl:
            categories['Communication (NCCL/c10d)'] += t
        elif 'bmm' in nl or 'batch_mm' in nl:
            categories['Attention BMM'] += t
        elif 'flex_attention' in nl or 'sdpa' in nl:
            categories['FlexAttention'] += t
        elif 'grouped_mm' in nl:
            categories['Expert FFN (grouped_mm)'] += t
        elif ('mm' in nl or 'matmul' in nl or 'linear' in nl or 'addmm' in nl or 'gemm' in nl) and 'bmm' not in nl:
            categories['Dense MatMul/Linear'] += t
        elif 'adam' in nl or 'optimizer' in nl or 'fused_adam' in nl or 'foreach' in nl:
            categories['Optimizer'] += t
        elif 'copy_' in nl or '_to_copy' in nl:
            categories['Copy/DType Conversion'] += t
        elif 'index' in nl or 'scatter' in nl or 'gather' in nl or 'topk' in nl or 'sort' in nl or 'permute' in nl:
            categories['Index/Scatter/Gather/Permute'] += t
        elif 'norm' in nl or 'layer_norm' in nl or 'rms_norm' in nl:
            categories['Normalization'] += t
        elif 'cat' in nl or 'chunk' in nl or 'split' in nl or 'stack' in nl:
            categories['Concat/Split/Chunk'] += t
        elif 'add' in nl or 'mul' in nl or 'silu' in nl or 'gelu' in nl or 'elementwise' in nl:
            categories['ElementWise Ops'] += t
        elif 'empty' in nl or 'zeros' in nl or 'fill' in nl or 'resize' in nl:
            categories['Memory Allocation'] += t
        elif 'backward' in nl or 'autograd' in nl:
            categories['Autograd Engine'] += t
        elif 'checkpoint' in nl or 'recompute' in nl:
            categories['Activation Checkpoint'] += t
        elif 'item' in nl or 'scalar_dense' in nl:
            categories['GPU Sync (item/scalar)'] += t
        else:
            categories['Other'] += t

    sorted_cats = sorted(categories.items(), key=lambda x: -x[1])

    print(f"\n{'Category':<45} {'Self Time (ms)':>14} {'%':>6}")
    print("-" * 70)
    for cat, t in sorted_cats:
        if t > 0:
            pct = t / total_self * 100 if total_self > 0 else 0
            print(f"  {cat:<43} {t/1000:>14.1f} {pct:>5.1f}%")
    print(f"  {'TOTAL':<43} {total_self/1000:>14.1f}")

    # =========================================================================
    # 4. LARGEST INDIVIDUAL OPERATIONS (single invocations)
    # =========================================================================
    print("\n" + "=" * 110)
    print("TOP 20 LARGEST INDIVIDUAL OP INVOCATIONS")
    print("(Single calls that took the longest wall time)")
    print("=" * 110)

    big_ops = sorted(cpu_ops, key=lambda x: -x['dur_us'])[:20]
    print(f"\n{'#':<4} {'Operation':<65} {'Duration (ms)':>14} {'Self (ms)':>10}")
    print("-" * 100)
    for i, op in enumerate(big_ops):
        dur_ms = op['dur_us'] / 1000
        self_ms = op.get('self_us', 0) / 1000
        print(f"{i+1:<4} {op['name'][:64]:<65} {dur_ms:>14.1f} {self_ms:>10.1f}")

    # =========================================================================
    # 5. NCCL OPERATIONS DETAIL
    # =========================================================================
    print("\n" + "=" * 110)
    print("NCCL / COMMUNICATION OPERATIONS")
    print("=" * 110)

    nccl_ops = [op for op in cpu_ops if any(k in op['name'].lower() for k in ['nccl', 'all_to_all', 'all_reduce', 'reduce_scatter', 'all_gather', 'c10d'])]
    nccl_agg = defaultdict(lambda: {'count': 0, 'total_us': 0, 'self_us': 0})
    for op in nccl_ops:
        nccl_agg[op['name']]['count'] += 1
        nccl_agg[op['name']]['total_us'] += op['dur_us']
        nccl_agg[op['name']]['self_us'] += op.get('self_us', 0)

    sorted_nccl = sorted(nccl_agg.items(), key=lambda x: -x[1]['total_us'])
    print(f"\n{'Operation':<65} {'Total (ms)':>10} {'Self (ms)':>10} {'Count':>7}")
    print("-" * 100)
    for name, st in sorted_nccl:
        print(f"  {name[:63]:<63} {st['total_us']/1000:>10.1f} {st['self_us']/1000:>10.1f} {st['count']:>7}")

    total_nccl = sum(v['total_us'] for v in nccl_agg.values())
    total_nccl_self = sum(v['self_us'] for v in nccl_agg.values())
    print(f"\n  Total NCCL inclusive: {total_nccl/1000:.1f} ms, self: {total_nccl_self/1000:.1f} ms")

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"  Total CPU op time (inclusive, all ops): {total_time/1000:.1f} ms")
    print(f"  Total self time (excluding nesting):    {total_self/1000:.1f} ms")
    print()


if __name__ == '__main__':
    trace_path = sys.argv[1] if len(sys.argv) > 1 else \
        "./outputs/profiling_lbs2_726tps/profile_traces/iteration_5/rank0_trace.json"
    analyze_trace(trace_path)
