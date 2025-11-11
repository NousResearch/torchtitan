#!/usr/bin/env python3
"""Analyze and compare profiling traces between EP=1 and EP=2"""

import json
import sys
from collections import defaultdict

def analyze_trace(trace_file):
    """Analyze a single trace file"""
    with open(trace_file, 'r') as f:
        data = json.load(f)

    # Collect timing statistics
    stats = defaultdict(lambda: {'count': 0, 'total_time': 0})

    # Parse trace events
    for event in data.get('traceEvents', []):
        if event.get('ph') == 'X':  # Duration event
            name = event.get('name', '')
            dur = event.get('dur', 0)  # Duration in microseconds

            # Categorize key operations
            if 'all_to_all' in name.lower():
                stats['all_to_all']['count'] += 1
                stats['all_to_all']['total_time'] += dur
            elif 'cudamemcpy' in name.lower() or 'd2h' in name.lower() or 'dtoh' in name.lower():
                stats['d2h_memcpy']['count'] += 1
                stats['d2h_memcpy']['total_time'] += dur
            elif 'wait' in name.lower() and 'tensor' in name.lower():
                stats['wait_tensor']['count'] += 1
                stats['wait_tensor']['total_time'] += dur
            elif 'nccl' in name.lower():
                stats['nccl']['count'] += 1
                stats['nccl']['total_time'] += dur
            elif 'permute' in name.lower() or '_permute' in name:
                stats['permute']['count'] += 1
                stats['permute']['total_time'] += dur
            elif 'forward' in name.lower() or 'backward' in name.lower():
                if 'moe' in name.lower() or 'expert' in name.lower():
                    stats['moe_forward_backward']['count'] += 1
                    stats['moe_forward_backward']['total_time'] += dur

    return stats

def print_stats(stats, name):
    """Print statistics"""
    print(f"\n{name} Statistics:")
    print("=" * 80)
    for op_name, op_stats in sorted(stats.items()):
        count = op_stats['count']
        total_ms = op_stats['total_time'] / 1000.0  # Convert to milliseconds
        if count > 0:
            avg_ms = total_ms / count
            print(f"  {op_name:30s}: count={count:6d}, total={total_ms:10.2f}ms, avg={avg_ms:8.4f}ms")

def main():
    # Analyze EP=2 rank 0
    print("\nAnalyzing EP=2 (iteration 5, rank 0)...")
    ep2_stats = analyze_trace('./outputs_profile_ep2/profile_trace/iteration_5/rank0_trace.json')
    print_stats(ep2_stats, "EP=2")

    # Analyze EP=1 rank 0
    print("\nAnalyzing EP=1 (iteration 5, rank 0)...")
    ep1_stats = analyze_trace('./outputs_profile_ep1/profile_trace/iteration_5/rank0_trace.json')
    print_stats(ep1_stats, "EP=1")

    # Compare key metrics
    print("\n" + "="*80)
    print("COMPARISON (EP=2 vs EP=1):")
    print("="*80)

    for op_name in set(ep2_stats.keys()) | set(ep1_stats.keys()):
        ep2_time = ep2_stats[op_name]['total_time'] / 1000.0
        ep1_time = ep1_stats[op_name]['total_time'] / 1000.0
        diff = ep2_time - ep1_time
        if ep1_time > 0:
            pct = (diff / ep1_time) * 100
            print(f"  {op_name:30s}: EP=2={ep2_time:8.2f}ms, EP=1={ep1_time:8.2f}ms, diff={diff:+8.2f}ms ({pct:+6.1f}%)")
        elif ep2_time > 0:
            print(f"  {op_name:30s}: EP=2={ep2_time:8.2f}ms, EP=1={ep1_time:8.2f}ms (NEW IN EP=2)")

if __name__ == '__main__':
    main()
