#!/usr/bin/env python3
"""Detailed profiling trace analysis with specific operation breakdown"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

def analyze_trace_detailed(trace_file: str) -> Dict[str, Dict]:
    """Analyze trace file and return detailed operation statistics"""
    with open(trace_file, 'r') as f:
        data = json.load(f)

    # Collect detailed timing statistics by exact operation name
    stats = defaultdict(lambda: {'count': 0, 'total_time_us': 0, 'min_us': float('inf'), 'max_us': 0})

    # Parse trace events
    for event in data.get('traceEvents', []):
        if event.get('ph') == 'X':  # Duration event
            name = event.get('name', '')
            dur = event.get('dur', 0)  # Duration in microseconds

            if dur > 0:  # Only count events with positive duration
                stats[name]['count'] += 1
                stats[name]['total_time_us'] += dur
                stats[name]['min_us'] = min(stats[name]['min_us'], dur)
                stats[name]['max_us'] = max(stats[name]['max_us'], dur)

    return dict(stats)

def compute_differences(ep2_stats: Dict, ep1_stats: Dict) -> List[Tuple]:
    """Compute differences between EP=2 and EP=1"""
    differences = []

    # Get all operation names
    all_ops = set(ep2_stats.keys()) | set(ep1_stats.keys())

    for op_name in all_ops:
        ep2_time = ep2_stats.get(op_name, {}).get('total_time_us', 0) / 1000.0  # Convert to ms
        ep1_time = ep1_stats.get(op_name, {}).get('total_time_us', 0) / 1000.0

        ep2_count = ep2_stats.get(op_name, {}).get('count', 0)
        ep1_count = ep1_stats.get(op_name, {}).get('count', 0)

        time_diff = ep2_time - ep1_time
        count_diff = ep2_count - ep1_count

        # Calculate percentage difference
        if ep1_time > 0:
            pct_diff = (time_diff / ep1_time) * 100
        elif ep2_time > 0:
            pct_diff = float('inf')  # New operation in EP=2
        else:
            continue  # Both zero, skip

        differences.append({
            'operation': op_name,
            'ep2_time_ms': ep2_time,
            'ep1_time_ms': ep1_time,
            'time_diff_ms': time_diff,
            'pct_diff': pct_diff,
            'ep2_count': ep2_count,
            'ep1_count': ep1_count,
            'count_diff': count_diff,
            'ep2_avg_us': ep2_time * 1000 / ep2_count if ep2_count > 0 else 0,
            'ep1_avg_us': ep1_time * 1000 / ep1_count if ep1_count > 0 else 0,
        })

    # Sort by absolute time difference (most significant first)
    differences.sort(key=lambda x: abs(x['time_diff_ms']), reverse=True)

    return differences

def print_comparison_table(differences: List[Tuple], top_n: int = 20):
    """Print detailed comparison table"""
    print("\n" + "="*150)
    print(f"TOP {top_n} OPERATIONS BY TIME DIFFERENCE (EP=2 vs EP=1)")
    print("="*150)
    print(f"{'#':<3} {'Operation Name':<60} {'EP=2 Time':>12} {'EP=1 Time':>12} {'Diff':>12} {'% Change':>10} {'EP=2 Cnt':>9} {'EP=1 Cnt':>9}")
    print("-"*150)

    for i, diff in enumerate(differences[:top_n], 1):
        op_name = diff['operation']
        if len(op_name) > 58:
            op_name = op_name[:55] + "..."

        ep2_time = diff['ep2_time_ms']
        ep1_time = diff['ep1_time_ms']
        time_diff = diff['time_diff_ms']
        pct_diff = diff['pct_diff']
        ep2_count = diff['ep2_count']
        ep1_count = diff['ep1_count']

        if pct_diff == float('inf'):
            pct_str = "NEW"
        else:
            pct_str = f"{pct_diff:+.1f}%"

        print(f"{i:<3} {op_name:<60} {ep2_time:>10.2f}ms {ep1_time:>10.2f}ms {time_diff:>+10.2f}ms {pct_str:>10} {ep2_count:>9} {ep1_count:>9}")

def print_category_summary(differences: List[Tuple]):
    """Print summary by operation category"""
    categories = {
        'NCCL Communication': ['nccl', 'all_to_all', 'all_reduce', 'all_gather'],
        'Memory Transfer': ['cudamemcpy', 'memcpy', 'd2h', 'h2d', 'dtoh', 'htod'],
        'Synchronization': ['wait', 'synchronize', 'stream'],
        'MoE Operations': ['moe', 'expert', 'router', 'gate'],
        'Attention': ['attention', 'attn', 'sdpa'],
        'Linear/GEMM': ['linear', 'mm', 'gemm', 'matmul', 'addmm'],
        'Elementwise': ['add', 'mul', 'div', 'relu', 'gelu', 'silu'],
    }

    category_stats = defaultdict(lambda: {'ep2_time': 0, 'ep1_time': 0, 'count': 0})

    for diff in differences:
        op_name_lower = diff['operation'].lower()
        matched = False

        for cat_name, keywords in categories.items():
            if any(kw in op_name_lower for kw in keywords):
                category_stats[cat_name]['ep2_time'] += diff['ep2_time_ms']
                category_stats[cat_name]['ep1_time'] += diff['ep1_time_ms']
                category_stats[cat_name]['count'] += 1
                matched = True
                break

        if not matched:
            category_stats['Other']['ep2_time'] += diff['ep2_time_ms']
            category_stats['Other']['ep1_time'] += diff['ep1_time_ms']
            category_stats['Other']['count'] += 1

    print("\n" + "="*100)
    print("SUMMARY BY OPERATION CATEGORY")
    print("="*100)
    print(f"{'Category':<25} {'EP=2 Time':>15} {'EP=1 Time':>15} {'Difference':>15} {'% Change':>12} {'Ops':>8}")
    print("-"*100)

    for cat_name, stats in sorted(category_stats.items(), key=lambda x: abs(x[1]['ep2_time'] - x[1]['ep1_time']), reverse=True):
        ep2_time = stats['ep2_time']
        ep1_time = stats['ep1_time']
        diff = ep2_time - ep1_time
        pct = (diff / ep1_time * 100) if ep1_time > 0 else float('inf')
        count = stats['count']

        if pct == float('inf'):
            pct_str = "NEW"
        else:
            pct_str = f"{pct:+.1f}%"

        print(f"{cat_name:<25} {ep2_time:>13.2f}ms {ep1_time:>13.2f}ms {diff:>+13.2f}ms {pct_str:>12} {count:>8}")

def main():
    print("Loading and analyzing traces...")

    # Analyze both traces
    ep2_stats = analyze_trace_detailed('./outputs_profile_ep2/profile_trace/iteration_5/rank0_trace.json')
    ep1_stats = analyze_trace_detailed('./outputs_profile_ep1/profile_trace/iteration_5/rank0_trace.json')

    print(f"EP=2: Found {len(ep2_stats)} unique operations")
    print(f"EP=1: Found {len(ep1_stats)} unique operations")

    # Compute differences
    differences = compute_differences(ep2_stats, ep1_stats)

    # Print detailed comparison
    print_comparison_table(differences, top_n=30)

    # Print category summary
    print_category_summary(differences)

    # Print top operations that only exist in EP=2
    print("\n" + "="*100)
    print("NEW OPERATIONS IN EP=2 (Top 15 by Time)")
    print("="*100)
    new_ops = [d for d in differences if d['ep1_time_ms'] == 0]
    for i, diff in enumerate(new_ops[:15], 1):
        print(f"{i:2}. {diff['operation']:<70} {diff['ep2_time_ms']:>10.2f}ms (count={diff['ep2_count']})")

if __name__ == '__main__':
    main()
