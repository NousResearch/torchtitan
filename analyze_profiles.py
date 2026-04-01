#!/usr/bin/env python3
"""Comprehensive pairwise profiling comparison: Baseline vs ScMoE.

Usage:
    python analyze_profiles.py <baseline.nsys-rep> <scmoe.nsys-rep>
    python analyze_profiles.py --torch <baseline_trace.json> <scmoe_trace.json>

Produces:
1. GPU kernel breakdown (GEMM, NCCL, DeepEP, attention, elementwise, norm)
2. CPU operator breakdown (PyTorch ops)
3. CUDA stream utilization (overlap analysis)
4. Pairwise delta with percentage changes
5. Bottleneck identification
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict


def analyze_nsys_rep(rep_path, label):
    """Analyze .nsys-rep file using nsys stats."""
    print(f"\n{'='*80}")
    print(f"  NSYS Analysis: {label}")
    print(f"  File: {rep_path}")
    print(f"{'='*80}")

    # Export to sqlite for detailed analysis
    sqlite_path = rep_path.replace('.nsys-rep', '.sqlite')
    if not os.path.exists(sqlite_path):
        print("Exporting to sqlite...")
        subprocess.run(
            ['nsys', 'export', '--type=sqlite', '--output=' + sqlite_path, rep_path],
            capture_output=True, text=True
        )

    if not os.path.exists(sqlite_path):
        print(f"ERROR: Could not create {sqlite_path}")
        # Fallback: use nsys stats
        print("\nFallback: using nsys stats CLI...")
        for report in ['cuda_gpu_kern_sum', 'cuda_api_sum', 'nvtx_sum']:
            print(f"\n--- {report} ---")
            result = subprocess.run(
                ['nsys', 'stats', '--report', report, '--format', 'csv', rep_path],
                capture_output=True, text=True, timeout=120
            )
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[:25]:
                    print(f"  {line}")
        return {}

    # Query sqlite directly for detailed analysis
    conn = sqlite3.connect(sqlite_path)
    results = {}

    # 1. GPU Kernel Summary
    print("\n--- GPU KERNEL SUMMARY (top 30 by total time) ---")
    try:
        cursor = conn.execute("""
            SELECT
                shortName as name,
                COUNT(*) as count,
                SUM(end - start) as total_ns,
                AVG(end - start) as avg_ns,
                MAX(end - start) as max_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY shortName
            ORDER BY total_ns DESC
            LIMIT 30
        """)
        kernel_rows = cursor.fetchall()
        print(f"  {'Kernel':<60} {'Count':>6} {'Total(ms)':>10} {'Avg(ms)':>8} {'Max(ms)':>8}")
        print(f"  {'-'*60} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")
        for name, count, total_ns, avg_ns, max_ns in kernel_rows:
            total_ms = total_ns / 1e6
            avg_ms = avg_ns / 1e6
            max_ms = max_ns / 1e6
            print(f"  {str(name)[:60]:<60} {count:>6} {total_ms:>10.2f} {avg_ms:>8.3f} {max_ms:>8.2f}")
            results[str(name)] = {'count': count, 'total_ms': total_ms, 'avg_ms': avg_ms}
    except Exception as e:
        print(f"  Error querying kernels: {e}")

    # 2. Kernel category breakdown
    print("\n--- KERNEL CATEGORY BREAKDOWN ---")
    try:
        cursor = conn.execute("""
            SELECT shortName, SUM(end - start) as total_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY shortName
        """)
        categories = defaultdict(lambda: {'total_ns': 0, 'count': 0})
        for name, total_ns in cursor.fetchall():
            name = str(name).lower()
            if 'nccl' in name:
                cat = 'NCCL'
            elif 'nvshmem' in name or 'deep_ep' in name or 'deepep' in name:
                cat = 'DeepEP/NVSHMEM'
            elif 'gemm' in name or 'cutlass' in name or 'sm90' in name or 'grouped' in name:
                cat = 'GEMM'
            elif 'flash' in name or 'fmha' in name or 'attention' in name:
                cat = 'Attention'
            elif 'rms' in name or 'norm' in name or 'layer_norm' in name:
                cat = 'Norm'
            elif 'elementwise' in name or 'vectorized' in name or 'pointwise' in name:
                cat = 'Elementwise'
            elif 'reduce' in name or 'allreduce' in name:
                cat = 'Reduce'
            else:
                cat = 'Other'
            categories[cat]['total_ns'] += total_ns
            categories[cat]['count'] += 1

        total_ns = sum(v['total_ns'] for v in categories.values())
        print(f"  {'Category':<25} {'Total(ms)':>10} {'%':>6} {'Count':>6}")
        print(f"  {'-'*25} {'-'*10} {'-'*6} {'-'*6}")
        for cat, v in sorted(categories.items(), key=lambda x: -x[1]['total_ns']):
            ms = v['total_ns'] / 1e6
            pct = v['total_ns'] / total_ns * 100 if total_ns > 0 else 0
            print(f"  {cat:<25} {ms:>10.2f} {pct:>5.1f}% {v['count']:>6}")
        results['_categories'] = {cat: v['total_ns'] / 1e6 for cat, v in categories.items()}
    except Exception as e:
        print(f"  Error: {e}")

    # 3. CUDA memcpy
    print("\n--- CUDA MEMCPY ---")
    try:
        cursor = conn.execute("""
            SELECT
                copyKind,
                COUNT(*) as count,
                SUM(end - start) as total_ns,
                SUM(bytes) as total_bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY copyKind
            ORDER BY total_ns DESC
        """)
        for kind, count, total_ns, total_bytes in cursor.fetchall():
            print(f"  Kind={kind}: count={count}, total={total_ns/1e6:.2f}ms, bytes={total_bytes/1e6:.1f}MB")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. CUDA stream utilization
    print("\n--- CUDA STREAMS (kernel count per stream) ---")
    try:
        cursor = conn.execute("""
            SELECT
                streamId,
                COUNT(*) as kernel_count,
                SUM(end - start) as total_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY streamId
            ORDER BY total_ns DESC
        """)
        for stream_id, count, total_ns in cursor.fetchall():
            print(f"  Stream {stream_id}: {count} kernels, {total_ns/1e6:.2f}ms total")
    except Exception as e:
        print(f"  Error: {e}")

    # 5. NVTX ranges (if available)
    print("\n--- NVTX RANGES (top 20) ---")
    try:
        cursor = conn.execute("""
            SELECT
                text,
                COUNT(*) as count,
                SUM(end - start) as total_ns
            FROM NVTX_EVENTS
            WHERE end > start
            GROUP BY text
            ORDER BY total_ns DESC
            LIMIT 20
        """)
        for text, count, total_ns in cursor.fetchall():
            print(f"  {str(text)[:60]:<60} count={count:>5} total={total_ns/1e6:>10.2f}ms")
    except Exception as e:
        print(f"  NVTX not available: {e}")

    conn.close()
    return results


def analyze_torch_trace(trace_path, label):
    """Analyze PyTorch profiler JSON trace."""
    print(f"\n{'='*80}")
    print(f"  PyTorch Trace Analysis: {label}")
    print(f"  File: {trace_path}")
    print(f"{'='*80}")

    with open(trace_path) as f:
        data = json.load(f)
    events = data.get('traceEvents', data) if isinstance(data, dict) else data

    # CPU ops
    cpu_ops = defaultdict(lambda: {'count': 0, 'total_us': 0, 'max_us': 0})
    # GPU kernels
    gpu_kernels = defaultdict(lambda: {'count': 0, 'total_us': 0})

    for e in events:
        if not isinstance(e, dict) or 'dur' not in e:
            continue
        cat = e.get('cat', '')
        name = e.get('name', '')
        dur = e['dur']

        if cat == 'cpu_op':
            cpu_ops[name]['count'] += 1
            cpu_ops[name]['total_us'] += dur
            cpu_ops[name]['max_us'] = max(cpu_ops[name]['max_us'], dur)
        elif cat == 'kernel':
            gpu_kernels[name]['count'] += 1
            gpu_kernels[name]['total_us'] += dur

    # Print CPU ops
    print("\n--- TOP 30 CPU OPS ---")
    total_cpu = sum(v['total_us'] for v in cpu_ops.values())
    print(f"  Total CPU op time: {total_cpu/1000:.1f}ms")
    print(f"  {'Op':<60} {'Count':>6} {'Total(ms)':>10} {'Avg(ms)':>8} {'Max(ms)':>8}")
    print(f"  {'-'*60} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")
    for name, v in sorted(cpu_ops.items(), key=lambda x: -x[1]['total_us'])[:30]:
        total_ms = v['total_us'] / 1000
        avg_ms = v['total_us'] / v['count'] / 1000
        max_ms = v['max_us'] / 1000
        print(f"  {name[:60]:<60} {v['count']:>6} {total_ms:>10.2f} {avg_ms:>8.3f} {max_ms:>8.2f}")

    return {'cpu_ops': dict(cpu_ops), 'gpu_kernels': dict(gpu_kernels)}


def pairwise_compare(baseline_results, scmoe_results, level='cpu_ops'):
    """Pairwise comparison between baseline and ScMoE."""
    print(f"\n{'='*80}")
    print(f"  PAIRWISE COMPARISON: {level}")
    print(f"{'='*80}")

    b = baseline_results.get(level, baseline_results)
    s = scmoe_results.get(level, scmoe_results)

    all_ops = set(list(b.keys()) + list(s.keys()))

    diffs = []
    for op in all_ops:
        if op.startswith('_'):
            continue
        b_time = b.get(op, {}).get('total_us', b.get(op, {}).get('total_ms', 0) * 1000) / 1000 if op in b else 0
        s_time = s.get(op, {}).get('total_us', s.get(op, {}).get('total_ms', 0) * 1000) / 1000 if op in s else 0
        b_count = b.get(op, {}).get('count', 0)
        s_count = s.get(op, {}).get('count', 0)

        if b_time == 0 and s_time == 0:
            continue
        delta = s_time - b_time
        pct = (delta / b_time * 100) if b_time > 0 else float('inf')
        diffs.append((abs(delta), delta, pct, op, b_count, s_count, b_time, s_time))

    diffs.sort(key=lambda x: -x[0])

    print(f"  {'Op':<55} {'Base(ms)':>9} {'ScMoE(ms)':>9} {'Delta':>9} {'%':>7} {'B_cnt':>6} {'S_cnt':>6}")
    print(f"  {'-'*55} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*6} {'-'*6}")

    for _, delta, pct, op, bc, sc, bt, st in diffs[:30]:
        sign = '+' if delta > 0 else ''
        pct_str = f"{pct:+.1f}%" if pct != float('inf') else "NEW"
        print(f"  {op[:55]:<55} {bt:>9.1f} {st:>9.1f} {sign}{delta:>8.1f} {pct_str:>7} {bc:>6} {sc:>6}")

    # Summary
    total_b = sum(v.get('total_us', v.get('total_ms', 0)*1000)/1000 for v in b.values() if isinstance(v, dict) and not isinstance(list(v.values())[0] if v else None, dict))
    total_s = sum(v.get('total_us', v.get('total_ms', 0)*1000)/1000 for v in s.values() if isinstance(v, dict) and not isinstance(list(v.values())[0] if v else None, dict))
    print(f"\n  TOTAL: Baseline={total_b:.1f}ms  ScMoE={total_s:.1f}ms  Delta={total_s-total_b:+.1f}ms ({(total_s/total_b-1)*100:+.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch', action='store_true', help='Analyze PyTorch traces instead of nsys')
    parser.add_argument('baseline', help='Baseline trace file')
    parser.add_argument('scmoe', help='ScMoE trace file')
    args = parser.parse_args()

    if args.torch:
        b = analyze_torch_trace(args.baseline, 'BASELINE')
        s = analyze_torch_trace(args.scmoe, 'SCMOE')
        pairwise_compare(b, s, level='cpu_ops')
    else:
        b = analyze_nsys_rep(args.baseline, 'BASELINE')
        s = analyze_nsys_rep(args.scmoe, 'SCMOE')
        if b and s:
            # Compare kernel categories
            if '_categories' in b and '_categories' in s:
                print(f"\n{'='*80}")
                print(f"  KERNEL CATEGORY COMPARISON")
                print(f"{'='*80}")
                all_cats = set(list(b['_categories'].keys()) + list(s['_categories'].keys()))
                for cat in sorted(all_cats):
                    bv = b['_categories'].get(cat, 0)
                    sv = s['_categories'].get(cat, 0)
                    delta = sv - bv
                    pct = (delta / bv * 100) if bv > 0 else 0
                    print(f"  {cat:<25} Base={bv:>8.1f}ms  ScMoE={sv:>8.1f}ms  Delta={delta:+.1f}ms ({pct:+.1f}%)")
