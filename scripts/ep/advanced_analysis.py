#!/usr/bin/env python3
"""Advanced profiling analysis with averaging, contribution analysis, and visualization"""

import json
import sys
import glob
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Try to use faster JSON library
try:
    import orjson
    def load_json(file_path):
        with open(file_path, 'rb') as f:
            return orjson.loads(f.read())
    JSON_LIBRARY = "orjson (fast)"
except ImportError:
    try:
        import ujson
        def load_json(file_path):
            with open(file_path, 'r') as f:
                return ujson.load(f)
        JSON_LIBRARY = "ujson (fast)"
    except ImportError:
        def load_json(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        JSON_LIBRARY = "json (standard)"

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")
    print("Continuing with text-only analysis...\n")


# ============================================================================
# SOURCE LOCATION EXTRACTION
# ============================================================================

def extract_source_location_from_stack(stack_trace: str) -> Optional[Dict[str, str]]:
    """
    Extract source file and line number from stack trace.

    Filters out torch internals, returns first user code frame.

    Returns:
        {'file': 'expert_parallel.py', 'line': 104, 'full_path': '/path/to/file.py'}
        or None if no user code found
    """
    if not stack_trace:
        return None

    # Stack trace format examples:
    # 1. "File \"/path/to/file.py\", line 104, in function_name"
    # 2. "/path/to/file.py:104:function_name"

    # Try to extract file:line patterns
    patterns = [
        r'File\s+"([^"]+)",\s+line\s+(\d+)',  # Python format
        r'([^:\s]+):(\d+):',  # Simple format
        r'at\s+([^:]+):(\d+)',  # AT format
    ]

    lines = stack_trace.split('\n') if isinstance(stack_trace, str) else [stack_trace]

    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                full_path = match.group(1)
                line_num = int(match.group(2))

                # Filter out torch/Python internals
                if any(exclude in full_path for exclude in [
                    '/torch/', '/python', 'site-packages',
                    '/lib/', '/bin/', 'built-in', '<built-in>'
                ]):
                    continue

                # Extract filename only
                filename = Path(full_path).name

                # Shorten path if it's from torchtitan
                if 'torchtitan' in full_path:
                    parts = full_path.split('torchtitan/')
                    if len(parts) > 1:
                        short_path = 'torchtitan/' + parts[-1]
                    else:
                        short_path = filename
                else:
                    short_path = filename

                return {
                    'file': filename,
                    'line': line_num,
                    'short_path': short_path,
                    'full_path': full_path
                }

    return None


def categorize_operation_type(op_name: str) -> str:
    """
    Categorize operation by type for filtering/grouping.

    Returns: 'Communication', 'Memory', 'Synchronization', 'Compute', 'Other'
    """
    op_lower = op_name.lower()

    # Communication operations
    if any(keyword in op_lower for keyword in [
        'nccl', 'all_to_all', 'alltoall', 'all_reduce', 'allreduce',
        'all_gather', 'allgather', 'broadcast', 'reduce_scatter',
        'send', 'recv', 'p2p'
    ]):
        return 'Communication'

    # Memory operations
    if any(keyword in op_lower for keyword in [
        '_to_copy', 'memcpy', 'memory', 'alloc', 'free',
        'cuda_malloc', 'cuda_free', 'copy_'
    ]):
        return 'Memory'

    # Synchronization operations
    if any(keyword in op_lower for keyword in [
        'synchronize', 'barrier', 'lock', 'wait', 'event_record',
        'event_wait', 'stream_sync', 'device_sync'
    ]):
        return 'Synchronization'

    # Compute operations
    if any(keyword in op_lower for keyword in [
        'mm', 'gemm', 'conv', 'matmul', 'linear', 'grouped_mm',
        'bmm', 'addmm', 'baddbmm', 'fft', 'softmax', 'relu',
        'gelu', 'layernorm', 'attention'
    ]):
        return 'Compute'

    return 'Other'


def format_operation_with_source(op_name: str, source_info: Optional[Dict] = None) -> str:
    """
    Format operation name with source location.

    Examples:
        "all_to_all" → "all_to_all @ expert_parallel.py:104"
        "grouped_mm" → "grouped_mm @ moe.py:89"
    """
    if not source_info:
        return op_name

    # Truncate long operation names
    if len(op_name) > 40:
        op_name = op_name[:37] + "..."

    return f"{op_name} @ {source_info['file']}:{source_info['line']}"


def analyze_trace_with_profiler_steps(trace_file: str) -> Dict:
    """Analyze trace file and extract profiler step timings + ULTRA-DEEP DATA"""
    data = load_json(trace_file)  # Use faster JSON loading

    # Collect detailed timing statistics by exact operation name
    stats = defaultdict(lambda: {
        'count': 0,
        'total_time_us': 0,
        'min_us': float('inf'),
        'max_us': 0,
        'times': [],  # Store all individual times for statistics
        'flops': [],  # Store FLOP counts
        'modules': set(),  # Module hierarchy
        'stacks': [],  # Stack traces
        'args': [],  # Event arguments
        'source_locations': [],  # Source file:line from stack traces
        'op_type': None,  # Operation category (Comm/Memory/Sync/Compute)
    })

    profiler_steps = []
    memory_events = []
    communication_events = []
    cuda_sync_events = []
    python_gc_events = []

    # Parse trace events - ULTRA-DEEP MODE
    for event in data.get('traceEvents', []):
        ph = event.get('ph', '')
        name = event.get('name', '')
        cat = event.get('cat', '')
        dur = event.get('dur', 0)  # Duration in microseconds
        ts = event.get('ts', 0)  # Timestamp
        args = event.get('args', {})

        # Duration events (operations)
        if ph == 'X' and dur > 0:
            stats[name]['count'] += 1
            stats[name]['total_time_us'] += dur
            stats[name]['min_us'] = min(stats[name]['min_us'], dur)
            stats[name]['max_us'] = max(stats[name]['max_us'], dur)
            stats[name]['times'].append(dur)
            stats[name]['args'].append(args)

            # Extract FLOPs if available
            if 'Flops' in args or 'flops' in args:
                flops = args.get('Flops', args.get('flops', 0))
                if flops:
                    stats[name]['flops'].append(flops)

            # Extract module hierarchy
            if 'Module Hierarchy' in args or 'module' in args:
                module = args.get('Module Hierarchy', args.get('module', ''))
                if module:
                    stats[name]['modules'].add(module)

            # Extract stack trace
            if 'Python call stack' in args or 'callstack' in args:
                stack = args.get('Python call stack', args.get('callstack', ''))
                if stack:
                    stats[name]['stacks'].append(stack)

                    # Extract source location from stack trace
                    source_loc = extract_source_location_from_stack(stack)
                    if source_loc:
                        stats[name]['source_locations'].append(source_loc)

            # Categorize operation type (do once per operation)
            if stats[name]['op_type'] is None:
                stats[name]['op_type'] = categorize_operation_type(name)

            # Track profiler steps
            if 'ProfilerStep#' in name:
                profiler_steps.append({
                    'name': name,
                    'start': ts,
                    'end': ts + dur,
                    'duration': dur
                })

            # Identify communication operations
            if 'nccl' in name.lower() or 'all_to_all' in name.lower() or 'alltoall' in name.lower():
                communication_events.append({
                    'name': name,
                    'ts': ts,
                    'dur': dur,
                    'args': args
                })

            # Identify CUDA sync events
            if 'cudaDeviceSynchronize' in name or 'cudaStreamSynchronize' in name:
                cuda_sync_events.append({
                    'name': name,
                    'ts': ts,
                    'dur': dur
                })

        # Memory allocation/free events
        if 'memory' in cat.lower() or name.startswith('[memory]'):
            memory_events.append({
                'name': name,
                'ts': ts,
                'dur': dur,
                'type': ph,
                'args': args
            })

        # Python GC events
        if 'python_gc' in name.lower() or 'garbage' in name.lower():
            python_gc_events.append({
                'name': name,
                'ts': ts,
                'dur': dur,
                'args': args
            })

    # Compute statistics
    for op_name in stats:
        times = stats[op_name]['times']
        if len(times) > 0:
            stats[op_name]['mean_us'] = statistics.mean(times)
            stats[op_name]['median_us'] = statistics.median(times)
            if len(times) > 1:
                stats[op_name]['std_us'] = statistics.stdev(times)
            else:
                stats[op_name]['std_us'] = 0

        # Aggregate FLOPs
        if stats[op_name]['flops']:
            stats[op_name]['total_flops'] = sum(stats[op_name]['flops'])
            stats[op_name]['avg_flops'] = statistics.mean(stats[op_name]['flops'])

        # Convert modules set to list for JSON serialization
        stats[op_name]['modules'] = list(stats[op_name]['modules'])

    return {
        'stats': dict(stats),
        'profiler_steps': profiler_steps,
        'memory_events': memory_events,
        'communication_events': communication_events,
        'cuda_sync_events': cuda_sync_events,
        'python_gc_events': python_gc_events,
    }


def _process_single_trace(trace_file: str) -> Dict:
    """Helper function to process a single trace file (for parallel execution)"""
    # Extract iteration and rank from filename
    parts = trace_file.split('/')
    iteration = [p for p in parts if 'iteration_' in p][0].split('_')[1]
    rank = [p for p in parts if 'rank' in p][0].split('_')[0].replace('rank', '')

    result = analyze_trace_with_profiler_steps(trace_file)
    result['iteration'] = iteration
    result['rank'] = rank
    result['file'] = trace_file
    return result


def analyze_all_traces(base_path: str, ep_name: str) -> Dict:
    """Analyze all trace files for a given EP configuration (PARALLEL)"""
    pattern = f"{base_path}/outputs_profile_{ep_name}/profile_trace/iteration_*/rank*_trace.json"
    trace_files = sorted(glob.glob(pattern))

    if not trace_files:
        print(f"Warning: No trace files found for {ep_name} at {pattern}")
        return None

    num_files = len(trace_files)
    print(f"Found {num_files} trace files for {ep_name}")

    # Use parallel processing for faster loading
    num_workers = min(num_files, multiprocessing.cpu_count())
    print(f"   Using {num_workers} parallel workers (JSON lib: {JSON_LIBRARY})")

    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all trace file processing tasks
        future_to_file = {executor.submit(_process_single_trace, tf): tf for tf in trace_files}

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            trace_file = future_to_file[future]
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                if completed % 2 == 0 or completed == num_files:
                    print(f"   Progress: {completed}/{num_files} files loaded", end='\r')
            except Exception as e:
                print(f"\n   Warning: Failed to process {trace_file}: {e}")

        print()  # New line after progress

    return {
        'traces': all_results,
        'ep_name': ep_name
    }


def aggregate_statistics(all_data: Dict) -> Dict:
    """Aggregate statistics across all traces - ULTRA-DEEP"""
    aggregated = defaultdict(lambda: {
        'counts': [],
        'total_times': [],
        'mean_times': [],
        'min_times': [],
        'max_times': [],
        'by_rank': defaultdict(list),
        'by_iteration': defaultdict(list),
        'all_flops': [],  # NEW
        'all_modules': set(),  # NEW
        'stack_samples': [],  # NEW
        'source_locations': [],  # NEW: Source file:line
        'op_type': None,  # NEW: Operation category
    })

    for trace in all_data['traces']:
        rank = trace['rank']
        iteration = trace['iteration']

        for op_name, op_stats in trace['stats'].items():
            total_time_ms = op_stats['total_time_us'] / 1000.0
            mean_time_us = op_stats.get('mean_us', 0)

            aggregated[op_name]['counts'].append(op_stats['count'])
            aggregated[op_name]['total_times'].append(total_time_ms)
            aggregated[op_name]['mean_times'].append(mean_time_us / 1000.0)
            aggregated[op_name]['min_times'].append(op_stats['min_us'] / 1000.0)
            aggregated[op_name]['max_times'].append(op_stats['max_us'] / 1000.0)

            aggregated[op_name]['by_rank'][rank].append(total_time_ms)
            aggregated[op_name]['by_iteration'][iteration].append(total_time_ms)

            # NEW: Aggregate FLOPs
            if 'flops' in op_stats and op_stats['flops']:
                aggregated[op_name]['all_flops'].extend(op_stats['flops'])

            # NEW: Aggregate modules
            if 'modules' in op_stats and op_stats['modules']:
                aggregated[op_name]['all_modules'].update(op_stats['modules'])

            # NEW: Sample stack traces (limit to avoid memory issues)
            if 'stacks' in op_stats and op_stats['stacks'] and len(aggregated[op_name]['stack_samples']) < 3:
                aggregated[op_name]['stack_samples'].extend(op_stats['stacks'][:3 - len(aggregated[op_name]['stack_samples'])])

            # NEW: Aggregate source locations
            if 'source_locations' in op_stats and op_stats['source_locations']:
                aggregated[op_name]['source_locations'].extend(op_stats['source_locations'])

            # NEW: Store operation type (should be same across all traces)
            if aggregated[op_name]['op_type'] is None and 'op_type' in op_stats:
                aggregated[op_name]['op_type'] = op_stats['op_type']

    # Compute summary statistics
    summary = {}
    for op_name, data in aggregated.items():
        if len(data['total_times']) > 0:
            # Get most common source location
            source_info = None
            if data['source_locations']:
                # Find most common source location (by file:line)
                loc_counts = defaultdict(int)
                for loc in data['source_locations']:
                    key = f"{loc['file']}:{loc['line']}"
                    loc_counts[key] += 1
                most_common = max(loc_counts.items(), key=lambda x: x[1])
                # Find the full location dict for the most common
                for loc in data['source_locations']:
                    if f"{loc['file']}:{loc['line']}" == most_common[0]:
                        source_info = loc
                        break

            summary[op_name] = {
                'avg_total_ms': statistics.mean(data['total_times']),
                'std_total_ms': statistics.stdev(data['total_times']) if len(data['total_times']) > 1 else 0,
                'min_total_ms': min(data['total_times']),
                'max_total_ms': max(data['total_times']),
                'avg_count': statistics.mean(data['counts']),
                'by_rank': {r: statistics.mean(times) for r, times in data['by_rank'].items()},
                'by_iteration': {i: statistics.mean(times) for i, times in data['by_iteration'].items()},
                'all_times': data['total_times'],
                # NEW FIELDS
                'total_flops': sum(data['all_flops']) if data['all_flops'] else 0,
                'avg_flops': statistics.mean(data['all_flops']) if data['all_flops'] else 0,
                'modules': list(data['all_modules']),
                'stack_sample': data['stack_samples'][0] if data['stack_samples'] else None,
                'source_info': source_info,  # NEW: Most common source location
                'op_type': data['op_type'],  # NEW: Operation category
            }

    return summary


def compute_contribution_analysis(ep2_summary: Dict, ep1_summary: Dict, total_step_diff_ms: float):
    """Compute contribution of each operation to the slowdown"""
    contributions = []

    for op_name in set(ep2_summary.keys()) | set(ep1_summary.keys()):
        ep2_time = ep2_summary.get(op_name, {}).get('avg_total_ms', 0)
        ep1_time = ep1_summary.get(op_name, {}).get('avg_total_ms', 0)

        diff_ms = ep2_time - ep1_time

        if total_step_diff_ms > 0:
            contribution_pct = (diff_ms / total_step_diff_ms) * 100
        else:
            contribution_pct = 0

        contributions.append({
            'operation': op_name,
            'ep2_avg_ms': ep2_time,
            'ep1_avg_ms': ep1_time,
            'diff_ms': diff_ms,
            'contribution_pct': contribution_pct,
            'ep2_std': ep2_summary.get(op_name, {}).get('std_total_ms', 0),
            'ep1_std': ep1_summary.get(op_name, {}).get('std_total_ms', 0),
        })

    contributions.sort(key=lambda x: abs(x['diff_ms']), reverse=True)
    return contributions


def print_contribution_analysis(contributions: List[Dict], top_n: int = 30):
    """Print contribution analysis table"""
    print("\n" + "="*160)
    print(f"TOP {top_n} OPERATIONS BY CONTRIBUTION TO SLOWDOWN")
    print("="*160)
    print(f"{'#':<3} {'Operation':<55} {'EP=2 Avg':>12} {'EP=1 Avg':>12} {'Diff':>12} {'Contrib %':>10} {'EP=2 Std':>10} {'EP=1 Std':>10}")
    print("-"*160)

    for i, contrib in enumerate(contributions[:top_n], 1):
        op_name = contrib['operation']
        if len(op_name) > 53:
            op_name = op_name[:50] + "..."

        ep2_avg = contrib['ep2_avg_ms']
        ep1_avg = contrib['ep1_avg_ms']
        diff = contrib['diff_ms']
        contrib_pct = contrib['contribution_pct']
        ep2_std = contrib['ep2_std']
        ep1_std = contrib['ep1_std']

        print(f"{i:<3} {op_name:<55} {ep2_avg:>10.2f}ms {ep1_avg:>10.2f}ms {diff:>+10.2f}ms {contrib_pct:>9.1f}% {ep2_std:>8.2f}ms {ep1_std:>8.2f}ms")


def analyze_rank_differences(ep2_data: Dict, ep1_data: Dict):
    """Analyze differences by rank"""
    print("\n" + "="*100)
    print("RANK-LEVEL ANALYSIS")
    print("="*100)

    # Get all ranks
    ep2_ranks = set()
    ep1_ranks = set()

    for trace in ep2_data['traces']:
        ep2_ranks.add(trace['rank'])
    for trace in ep1_data['traces']:
        ep1_ranks.add(trace['rank'])

    all_ranks = sorted(ep2_ranks | ep1_ranks, key=lambda x: int(x))

    # Compute total time per rank
    rank_totals_ep2 = defaultdict(lambda: {'total': 0, 'count': 0})
    rank_totals_ep1 = defaultdict(lambda: {'total': 0, 'count': 0})

    for trace in ep2_data['traces']:
        rank = trace['rank']
        for profiler_step in trace['profiler_steps']:
            rank_totals_ep2[rank]['total'] += profiler_step['duration'] / 1000.0  # Convert to ms
            rank_totals_ep2[rank]['count'] += 1

    for trace in ep1_data['traces']:
        rank = trace['rank']
        for profiler_step in trace['profiler_steps']:
            rank_totals_ep1[rank]['total'] += profiler_step['duration'] / 1000.0
            rank_totals_ep1[rank]['count'] += 1

    print(f"\n{'Rank':<6} {'EP=2 Avg Step':>15} {'EP=1 Avg Step':>15} {'Difference':>15} {'% Slower':>10}")
    print("-"*100)

    rank_diffs = []
    for rank in all_ranks:
        ep2_avg = rank_totals_ep2[rank]['total'] / max(rank_totals_ep2[rank]['count'], 1)
        ep1_avg = rank_totals_ep1[rank]['total'] / max(rank_totals_ep1[rank]['count'], 1)
        diff = ep2_avg - ep1_avg
        pct = (diff / ep1_avg * 100) if ep1_avg > 0 else 0

        rank_diffs.append({
            'rank': rank,
            'ep2_avg': ep2_avg,
            'ep1_avg': ep1_avg,
            'diff': diff,
            'pct': pct
        })

        print(f"Rank {rank:<3} {ep2_avg:>13.2f}ms {ep1_avg:>13.2f}ms {diff:>+13.2f}ms {pct:>9.1f}%")

    return rank_diffs


def analyze_communication_patterns(ep2_data: Dict, ep1_data: Dict) -> Dict:
    """Analyze all-to-all and communication patterns - NEW"""
    ep2_comm = defaultdict(list)
    ep1_comm = defaultdict(list)

    for trace in ep2_data['traces']:
        for event in trace.get('communication_events', []):
            ep2_comm[event['name']].append(event['dur'] / 1000.0)  # Convert to ms

    for trace in ep1_data['traces']:
        for event in trace.get('communication_events', []):
            ep1_comm[event['name']].append(event['dur'] / 1000.0)

    return {
        'ep2': {k: {'mean': statistics.mean(v), 'count': len(v), 'total': sum(v)} for k, v in ep2_comm.items() if v},
        'ep1': {k: {'mean': statistics.mean(v), 'count': len(v), 'total': sum(v)} for k, v in ep1_comm.items() if v},
    }


def analyze_memory_patterns(ep2_data: Dict, ep1_data: Dict) -> Dict:
    """Analyze memory allocation patterns - NEW"""
    ep2_mem_total = sum(len(t.get('memory_events', [])) for t in ep2_data['traces'])
    ep1_mem_total = sum(len(t.get('memory_events', [])) for t in ep1_data['traces'])

    return {
        'ep2_total_events': ep2_mem_total,
        'ep1_total_events': ep1_mem_total,
        'diff': ep2_mem_total - ep1_mem_total,
        'pct_increase': ((ep2_mem_total - ep1_mem_total) / ep1_mem_total * 100) if ep1_mem_total > 0 else 0
    }


def analyze_by_source_location(ep2_summary: Dict, ep1_summary: Dict) -> Dict:
    """
    Group operations by source file location.

    Returns structure:
    {
        'expert_parallel.py': {
            'operations': [
                {'op': 'all_to_all', 'line': 104, 'ep2_time': 567, 'ep1_time': 50, 'diff': 517, 'type': 'Communication'},
                ...
            ],
            'total_ep2_time': 801,
            'total_ep1_time': 150,
            'total_diff': 651,
        },
        ...
    }
    """
    by_file = defaultdict(lambda: {
        'operations': [],
        'total_ep2_time': 0,
        'total_ep1_time': 0,
        'total_diff': 0,
    })

    # Collect all operations with source info
    all_ops = set(ep2_summary.keys()) | set(ep1_summary.keys())

    for op_name in all_ops:
        ep2_stats = ep2_summary.get(op_name, {})
        ep1_stats = ep1_summary.get(op_name, {})

        ep2_time = ep2_stats.get('avg_total_ms', 0)
        ep1_time = ep1_stats.get('avg_total_ms', 0)
        diff = ep2_time - ep1_time

        # Get source info (prefer ep2, fallback to ep1)
        source_info = ep2_stats.get('source_info') or ep1_stats.get('source_info')
        op_type = ep2_stats.get('op_type') or ep1_stats.get('op_type') or 'Other'

        if source_info:
            filename = source_info['file']
            line = source_info['line']
            short_path = source_info.get('short_path', filename)

            by_file[short_path]['operations'].append({
                'op': op_name,
                'line': line,
                'ep2_time': ep2_time,
                'ep1_time': ep1_time,
                'diff': diff,
                'type': op_type,
            })

            by_file[short_path]['total_ep2_time'] += ep2_time
            by_file[short_path]['total_ep1_time'] += ep1_time
            by_file[short_path]['total_diff'] += diff

    # Sort operations within each file by diff
    for file_data in by_file.values():
        file_data['operations'].sort(key=lambda x: abs(x['diff']), reverse=True)

    return dict(by_file)


def analyze_module_performance(ep2_summary: Dict, ep1_summary: Dict) -> Dict:
    """Aggregate performance by module hierarchy - NEW"""
    module_perf = defaultdict(lambda: {'ep2_ms': 0, 'ep1_ms': 0, 'ops': []})

    for op_name, stats in ep2_summary.items():
        modules = stats.get('modules', [])
        for module in modules:
            if module:
                module_perf[module]['ep2_ms'] += stats.get('avg_total_ms', 0)
                module_perf[module]['ops'].append(op_name)

    for op_name, stats in ep1_summary.items():
        modules = stats.get('modules', [])
        for module in modules:
            if module:
                module_perf[module]['ep1_ms'] += stats.get('avg_total_ms', 0)

    # Compute differences
    for module in module_perf:
        module_perf[module]['diff_ms'] = module_perf[module]['ep2_ms'] - module_perf[module]['ep1_ms']

    return dict(module_perf)


def create_box_plot(ep2_summary: Dict, ep1_summary: Dict, top_ops: List[str], output_file: str = 'ep_comparison_boxplot.html'):
    """Create interactive box plot comparing EP=2 vs EP=1"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping box plot generation")
        return

    fig = make_subplots(
        rows=len(top_ops),
        cols=1,
        subplot_titles=[f"{op[:60]}..." if len(op) > 60 else op for op in top_ops],
        vertical_spacing=0.02
    )

    for idx, op in enumerate(top_ops, 1):
        ep2_times = ep2_summary.get(op, {}).get('all_times', [])
        ep1_times = ep1_summary.get(op, {}).get('all_times', [])

        # Add EP=2 box
        fig.add_trace(
            go.Box(
                y=ep2_times,
                name='EP=2',
                marker_color='indianred',
                boxmean='sd'
            ),
            row=idx, col=1
        )

        # Add EP=1 box
        fig.add_trace(
            go.Box(
                y=ep1_times,
                name='EP=1',
                marker_color='lightseagreen',
                boxmean='sd'
            ),
            row=idx, col=1
        )

    fig.update_layout(
        height=300 * len(top_ops),
        title_text="EP=2 vs EP=1 Operation Time Distributions",
        showlegend=True
    )

    fig.update_yaxes(title_text="Time (ms)")

    fig.write_html(output_file)
    print(f"\nBox plot saved to: {output_file}")


def create_contribution_waterfall(contributions: List[Dict], top_n: int = 20, output_file: str = 'ep_contribution_waterfall.html'):
    """Create waterfall chart showing contribution to slowdown"""
    if not PLOTLY_AVAILABLE:
        return

    top_contribs = contributions[:top_n]

    operations = [c['operation'][:50] for c in top_contribs]
    diffs = [c['diff_ms'] for c in top_contribs]
    contribs = [c['contribution_pct'] for c in top_contribs]

    fig = go.Figure()

    # Waterfall chart
    fig.add_trace(go.Waterfall(
        name = "Contribution",
        orientation = "v",
        measure = ["relative"] * len(operations),
        x = operations,
        y = diffs,
        text = [f"{c:.1f}%" for c in contribs],
        textposition = "outside",
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title = f"Top {top_n} Contributors to EP=2 Slowdown",
        showlegend = False,
        height = 600,
        xaxis = dict(tickangle=-45)
    )

    fig.write_html(output_file)
    print(f"Waterfall chart saved to: {output_file}")


def create_rank_comparison_plot(rank_diffs: List[Dict], output_file: str = 'ep_rank_comparison.html'):
    """Create plot comparing ranks"""
    if not PLOTLY_AVAILABLE:
        return

    ranks = [r['rank'] for r in rank_diffs]
    ep2_avgs = [r['ep2_avg'] for r in rank_diffs]
    ep1_avgs = [r['ep1_avg'] for r in rank_diffs]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=ranks,
        y=ep2_avgs,
        name='EP=2',
        marker_color='indianred'
    ))

    fig.add_trace(go.Bar(
        x=ranks,
        y=ep1_avgs,
        name='EP=1',
        marker_color='lightseagreen'
    ))

    fig.update_layout(
        title='Average Step Time by Rank',
        xaxis_title='Rank',
        yaxis_title='Time (ms)',
        barmode='group',
        height=500
    )

    fig.write_html(output_file)
    print(f"Rank comparison plot saved to: {output_file}")


def main():
    print("="*100)
    print("ADVANCED EP PERFORMANCE ANALYSIS")
    print("="*100)

    # Analyze all traces
    print("\nAnalyzing EP=2 traces...")
    ep2_data = analyze_all_traces('.', 'ep2')

    print("\nAnalyzing EP=1 traces...")
    ep1_data = analyze_all_traces('.', 'ep1')

    if not ep2_data or not ep1_data:
        print("Error: Missing trace data. Please run profiling first.")
        return

    # Aggregate statistics
    print("\nAggregating statistics across all traces...")
    ep2_summary = aggregate_statistics(ep2_data)
    ep1_summary = aggregate_statistics(ep1_data)

    # Compute total step time difference
    ep2_step_times = []
    for trace in ep2_data['traces']:
        for step in trace['profiler_steps']:
            ep2_step_times.append(step['duration'] / 1000.0)

    ep1_step_times = []
    for trace in ep1_data['traces']:
        for step in trace['profiler_steps']:
            ep1_step_times.append(step['duration'] / 1000.0)

    avg_ep2_step = statistics.mean(ep2_step_times) if ep2_step_times else 0
    avg_ep1_step = statistics.mean(ep1_step_times) if ep1_step_times else 0
    total_step_diff = avg_ep2_step - avg_ep1_step

    print(f"\nAverage Profiler Step Time:")
    print(f"  EP=2: {avg_ep2_step:.2f}ms (std: {statistics.stdev(ep2_step_times) if len(ep2_step_times) > 1 else 0:.2f}ms)")
    print(f"  EP=1: {avg_ep1_step:.2f}ms (std: {statistics.stdev(ep1_step_times) if len(ep1_step_times) > 1 else 0:.2f}ms)")
    print(f"  Difference: {total_step_diff:+.2f}ms ({(total_step_diff/avg_ep1_step*100):+.1f}%)")

    # Contribution analysis
    contributions = compute_contribution_analysis(ep2_summary, ep1_summary, total_step_diff)
    print_contribution_analysis(contributions, top_n=30)

    # Rank analysis
    rank_diffs = analyze_rank_differences(ep2_data, ep1_data)

    # Create visualizations
    if PLOTLY_AVAILABLE:
        print("\n" + "="*100)
        print("GENERATING INTERACTIVE VISUALIZATIONS")
        print("="*100)

        # Get top operations for box plot
        top_ops = [c['operation'] for c in contributions[:10] if c['diff_ms'] > 0]

        create_box_plot(ep2_summary, ep1_summary, top_ops, 'scripts/ep/boxplot.html')
        create_contribution_waterfall(contributions, top_n=20, output_file='scripts/ep/waterfall.html')
        create_rank_comparison_plot(rank_diffs, output_file='scripts/ep/rank_comparison.html')

        print("\nOpen these files in your browser:")
        print("  - scripts/ep/boxplot.html")
        print("  - scripts/ep/waterfall.html")
        print("  - scripts/ep/rank_comparison.html")
    else:
        print("\nTo generate interactive plots, install plotly:")
        print("  pip install plotly")


if __name__ == '__main__':
    main()
