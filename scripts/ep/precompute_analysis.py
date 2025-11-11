#!/usr/bin/env python3
"""
Pre-compute and cache analysis data for ultra-fast dashboard loading.

This script does all the heavy lifting once (loads 2.4GB traces, parses, aggregates)
and saves the results to a pickle file. The dashboard then loads this cached file
in <5 seconds instead of 60-90 seconds.

Usage:
    ./scripts/ep/precompute_analysis.py

Output:
    scripts/ep/.analysis_cache.pkl (contains all pre-computed data)
"""

import os
import sys
import pickle
import time
from pathlib import Path
from collections import defaultdict
import statistics

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from advanced_analysis import (
    analyze_all_traces,
    aggregate_statistics,
    compute_contribution_analysis,
    analyze_rank_differences,
    analyze_communication_patterns,
    analyze_memory_patterns,
    analyze_module_performance,
)

CACHE_FILE = SCRIPT_DIR / ".analysis_cache.pkl"


def get_trace_files_mtime(base_dir):
    """Get the latest modification time of all trace files"""
    ep2_pattern = Path(base_dir) / "outputs_profile_ep2/profile_trace/iteration_*/rank*_trace.json"
    ep1_pattern = Path(base_dir) / "outputs_profile_ep1/profile_trace/iteration_*/rank*_trace.json"

    import glob
    all_traces = glob.glob(str(ep2_pattern)) + glob.glob(str(ep1_pattern))

    if not all_traces:
        return 0

    return max(os.path.getmtime(f) for f in all_traces)


def is_cache_valid():
    """Check if cached data is still valid (traces haven't changed)"""
    if not CACHE_FILE.exists():
        return False

    cache_mtime = os.path.getmtime(CACHE_FILE)
    traces_mtime = get_trace_files_mtime(SCRIPT_DIR.parent.parent)

    return cache_mtime > traces_mtime


def precompute_and_cache():
    """Load, parse, aggregate all data and save to cache"""

    print("="*80)
    print("🔬 Pre-computing Ultra-Deep Analysis Data")
    print("="*80)
    print()

    start_time = time.time()

    # Change to repo root
    os.chdir(SCRIPT_DIR.parent.parent)

    # Step 1: Load traces
    print("📊 [1/7] Loading EP=2 traces...")
    t0 = time.time()
    ep2_data = analyze_all_traces('.', 'ep2')
    print(f"   ✓ Loaded in {time.time() - t0:.1f}s")

    print("📊 [2/7] Loading EP=1 traces...")
    t0 = time.time()
    ep1_data = analyze_all_traces('.', 'ep1')
    print(f"   ✓ Loaded in {time.time() - t0:.1f}s")

    if not ep2_data or not ep1_data:
        print("❌ ERROR: Failed to load trace data")
        return False

    # Step 2: Aggregate statistics
    print("📊 [3/7] Aggregating EP=2 statistics...")
    t0 = time.time()
    ep2_summary = aggregate_statistics(ep2_data)
    print(f"   ✓ Aggregated {len(ep2_summary)} operations in {time.time() - t0:.1f}s")

    print("📊 [4/7] Aggregating EP=1 statistics...")
    t0 = time.time()
    ep1_summary = aggregate_statistics(ep1_data)
    print(f"   ✓ Aggregated {len(ep1_summary)} operations in {time.time() - t0:.1f}s")

    # Step 3: Compute step times
    print("📊 [5/7] Computing step time statistics...")
    t0 = time.time()

    ep2_step_times = []
    ep2_step_by_rank = defaultdict(list)
    ep2_step_by_iter = defaultdict(list)

    for trace in ep2_data['traces']:
        rank = trace['rank']
        iteration = trace['iteration']
        for step in trace['profiler_steps']:
            step_ms = step['duration'] / 1000.0
            ep2_step_times.append(step_ms)
            ep2_step_by_rank[rank].append(step_ms)
            ep2_step_by_iter[iteration].append(step_ms)

    ep1_step_times = []
    ep1_step_by_rank = defaultdict(list)
    ep1_step_by_iter = defaultdict(list)

    for trace in ep1_data['traces']:
        rank = trace['rank']
        iteration = trace['iteration']
        for step in trace['profiler_steps']:
            step_ms = step['duration'] / 1000.0
            ep1_step_times.append(step_ms)
            ep1_step_by_rank[rank].append(step_ms)
            ep1_step_by_iter[iteration].append(step_ms)

    avg_ep2_step = statistics.mean(ep2_step_times) if ep2_step_times else 0
    avg_ep1_step = statistics.mean(ep1_step_times) if ep1_step_times else 0
    total_step_diff = avg_ep2_step - avg_ep1_step

    print(f"   ✓ Computed in {time.time() - t0:.1f}s")
    print(f"      EP=2 avg: {avg_ep2_step:.2f}ms | EP=1 avg: {avg_ep1_step:.2f}ms | Diff: +{total_step_diff:.2f}ms")

    # Step 4: Compute contributions
    print("📊 [6/7] Computing contribution analysis...")
    t0 = time.time()
    contributions = compute_contribution_analysis(ep2_summary, ep1_summary, total_step_diff)

    normalized_contribs = []
    for c in contributions:
        normalized_contribs.append({
            'op_name': c['operation'],
            'ep2_avg': c['ep2_avg_ms'],
            'ep1_avg': c['ep1_avg_ms'],
            'diff_ms': c['diff_ms'],
            'contrib_pct': c['contribution_pct'],
            'ep2_std': c['ep2_std'],
            'ep1_std': c['ep1_std'],
        })
    print(f"   ✓ Computed {len(normalized_contribs)} contributions in {time.time() - t0:.1f}s")

    # Step 5: Ultra-deep analysis
    print("📊 [7/7] Computing ultra-deep analysis...")
    t0 = time.time()

    rank_diffs = analyze_rank_differences(ep2_data, ep1_data)
    comm_analysis = analyze_communication_patterns(ep2_data, ep1_data)
    memory_analysis = analyze_memory_patterns(ep2_data, ep1_data)
    module_analysis = analyze_module_performance(ep2_summary, ep1_summary)

    print(f"   ✓ Computed in {time.time() - t0:.1f}s")
    print(f"      Communication ops: {len(comm_analysis)}")
    print(f"      Modules: {len(module_analysis)}")

    # Step 6: Package all data (OPTIMIZED - don't store raw trace data!)
    # Remove bulky raw trace events, keep only aggregated stats
    print("📦 Optimizing data for cache (removing raw trace events)...")

    # Don't store raw trace events - they're huge!
    ep2_data_lite = {
        'traces': [
            {
                'rank': t['rank'],
                'iteration': t['iteration'],
                # Don't include 'stats' with all raw times - too big!
            }
            for t in ep2_data.get('traces', [])
        ]
    }

    ep1_data_lite = {
        'traces': [
            {
                'rank': t['rank'],
                'iteration': t['iteration'],
            }
            for t in ep1_data.get('traces', [])
        ]
    }

    cached_data = {
        'ep2_data': ep2_data_lite,  # LITE version
        'ep1_data': ep1_data_lite,  # LITE version
        'ep2_summary': ep2_summary,
        'ep1_summary': ep1_summary,
        'ep2_step_times': ep2_step_times,
        'ep1_step_times': ep1_step_times,
        'ep2_step_by_rank': dict(ep2_step_by_rank),
        'ep1_step_by_rank': dict(ep1_step_by_rank),
        'ep2_step_by_iter': dict(ep2_step_by_iter),
        'ep1_step_by_iter': dict(ep1_step_by_iter),
        'avg_ep2_step': avg_ep2_step,
        'avg_ep1_step': avg_ep1_step,
        'total_step_diff': total_step_diff,
        'contributions': normalized_contribs,
        'rank_diffs': rank_diffs,
        'comm_analysis': comm_analysis,
        'memory_analysis': memory_analysis,
        'module_analysis': module_analysis,
        'cache_timestamp': time.time(),
        'traces_mtime': get_trace_files_mtime('.'),
    }

    # Step 7: Save to cache
    print()
    print(f"💾 Saving cached data to {CACHE_FILE}...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    cache_size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
    total_time = time.time() - start_time

    print()
    print("="*80)
    print("✅ PRE-COMPUTATION COMPLETE")
    print("="*80)
    print(f"📦 Cache file: {CACHE_FILE}")
    print(f"📊 Cache size: {cache_size_mb:.1f} MB")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"🚀 Dashboard will now load in <5 seconds!")
    print()
    print("Next step: Launch dashboard with ./scripts/ep/START_DASHBOARD.sh")
    print("="*80)

    return True


def main():
    # Check if cache is valid
    if is_cache_valid():
        print("="*80)
        print("✅ Cached analysis data is up to date!")
        print("="*80)
        cache_age = time.time() - os.path.getmtime(CACHE_FILE)
        cache_size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
        print(f"📦 Cache file: {CACHE_FILE}")
        print(f"📊 Cache size: {cache_size_mb:.1f} MB")
        print(f"⏱️  Cache age: {cache_age / 60:.1f} minutes")
        print()
        print("To force recomputation, delete the cache file:")
        print(f"   rm {CACHE_FILE}")
        print("="*80)
        return

    # Pre-compute
    success = precompute_and_cache()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
