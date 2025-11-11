#!/usr/bin/env python3
"""
Ultra-Deep Interactive Dashboard for EP Performance Analysis
Leverages enhanced profiling data: memory, FLOPs, stack traces, modules, per-kernel timing
Goal: Identify exact bottlenecks when scaling from EP=1 to EP=2
"""

import os
import sys
import json
import statistics
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_enhanced_trace(trace_file: str) -> Dict:
    """Parse trace with ALL available profiling data"""
    with open(trace_file, 'r') as f:
        data = json.load(f)

    # Extract metadata
    parts = trace_file.split('/')
    iteration = [p for p in parts if 'iteration_' in p][0].split('_')[1]
    rank = [p for p in parts if 'rank' in p][0].split('_')[0].replace('rank', '')

    # Storage for different event types
    operations = defaultdict(lambda: {
        'count': 0,
        'total_us': 0,
        'times': [],
        'args': [],  # Store args for each occurrence
        'stacks': [],  # Store stack traces
        'flops': [],  # Store FLOP counts
        'module': None,  # Module hierarchy
    })

    memory_events = []
    cuda_sync_events = []
    python_gc_events = []
    communication_events = []

    profiler_steps = []

    # Parse all events
    for event in data.get('traceEvents', []):
        ph = event.get('ph', '')
        name = event.get('name', '')
        cat = event.get('cat', '')
        dur = event.get('dur', 0)
        ts = event.get('ts', 0)
        args = event.get('args', {})

        # Duration events (operations)
        if ph == 'X' and dur > 0:
            operations[name]['count'] += 1
            operations[name]['total_us'] += dur
            operations[name]['times'].append(dur)
            operations[name]['args'].append(args)

            # Extract stack trace if available
            if 'Python call stack' in args or 'callstack' in args:
                stack = args.get('Python call stack', args.get('callstack', ''))
                operations[name]['stacks'].append(stack)

            # Extract module info
            if 'Module Hierarchy' in args or 'module' in args:
                module = args.get('Module Hierarchy', args.get('module', ''))
                if module and not operations[name]['module']:
                    operations[name]['module'] = module

            # Extract FLOPs if available
            if 'Flops' in args or 'flops' in args:
                flops = args.get('Flops', args.get('flops', 0))
                if flops:
                    operations[name]['flops'].append(flops)

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

    # Compute statistics for operations
    for op_name in operations:
        times = operations[op_name]['times']
        if times:
            operations[op_name]['mean_us'] = statistics.mean(times)
            operations[op_name]['median_us'] = statistics.median(times)
            operations[op_name]['min_us'] = min(times)
            operations[op_name]['max_us'] = max(times)
            operations[op_name]['std_us'] = statistics.stdev(times) if len(times) > 1 else 0

        # Aggregate FLOPs
        if operations[op_name]['flops']:
            operations[op_name]['total_flops'] = sum(operations[op_name]['flops'])
            operations[op_name]['avg_flops'] = statistics.mean(operations[op_name]['flops'])

    return {
        'iteration': iteration,
        'rank': rank,
        'file': trace_file,
        'operations': dict(operations),
        'profiler_steps': profiler_steps,
        'memory_events': memory_events,
        'communication_events': communication_events,
        'cuda_sync_events': cuda_sync_events,
        'python_gc_events': python_gc_events,
    }


def analyze_all_traces_ultra_deep(base_path: str, ep_name: str) -> Optional[Dict]:
    """Analyze all traces with ultra-deep parsing"""
    pattern = f"{base_path}/outputs_profile_{ep_name}/profile_trace/iteration_*/rank*_trace.json"
    trace_files = sorted(glob.glob(pattern))

    if not trace_files:
        print(f"No trace files found for {ep_name} at {pattern}")
        return None

    print(f"Analyzing {len(trace_files)} trace files for {ep_name}...")

    all_traces = []
    for trace_file in trace_files:
        try:
            trace_data = parse_enhanced_trace(trace_file)
            all_traces.append(trace_data)
        except Exception as e:
            print(f"Error parsing {trace_file}: {e}")
            continue

    return {'traces': all_traces, 'ep_name': ep_name}


def aggregate_ultra_deep_stats(ep_data: Dict) -> Dict:
    """Aggregate statistics across all traces"""
    aggregated = defaultdict(lambda: {
        'all_times': [],
        'all_counts': [],
        'all_flops': [],
        'modules': set(),
        'stack_traces': [],
    })

    for trace in ep_data['traces']:
        for op_name, op_data in trace['operations'].items():
            aggregated[op_name]['all_times'].extend(op_data['times'])
            aggregated[op_name]['all_counts'].append(op_data['count'])

            if 'total_flops' in op_data:
                aggregated[op_name]['all_flops'].append(op_data['total_flops'])

            if op_data.get('module'):
                aggregated[op_name]['modules'].add(op_data['module'])

            if op_data.get('stacks'):
                aggregated[op_name]['stack_traces'].extend(op_data['stacks'])

    # Compute final statistics
    for op_name in aggregated:
        times = aggregated[op_name]['all_times']
        if times:
            aggregated[op_name]['avg_ms'] = statistics.mean(times) / 1000.0
            aggregated[op_name]['std_ms'] = statistics.stdev(times) / 1000.0 if len(times) > 1 else 0
            aggregated[op_name]['min_ms'] = min(times) / 1000.0
            aggregated[op_name]['max_ms'] = max(times) / 1000.0
            aggregated[op_name]['total_ms'] = sum(times) / 1000.0

        counts = aggregated[op_name]['all_counts']
        if counts:
            aggregated[op_name]['avg_count'] = statistics.mean(counts)

        flops = aggregated[op_name]['all_flops']
        if flops:
            aggregated[op_name]['total_flops'] = sum(flops)
            aggregated[op_name]['avg_flops'] = statistics.mean(flops)

    return dict(aggregated)


def load_ultra_deep_data():
    """Load and process all ultra-deep analysis data"""
    os.chdir(SCRIPT_DIR.parent.parent)

    ep2_data = analyze_all_traces_ultra_deep('.', 'ep2')
    ep1_data = analyze_all_traces_ultra_deep('.', 'ep1')

    if not ep2_data or not ep1_data:
        return None

    ep2_summary = aggregate_ultra_deep_stats(ep2_data)
    ep1_summary = aggregate_ultra_deep_stats(ep1_data)

    # Compute step times by rank
    ep2_step_by_rank = defaultdict(list)
    ep1_step_by_rank = defaultdict(list)

    for trace in ep2_data['traces']:
        rank = trace['rank']
        for step in trace['profiler_steps']:
            ep2_step_by_rank[rank].append(step['duration'] / 1000.0)

    for trace in ep1_data['traces']:
        rank = trace['rank']
        for step in trace['profiler_steps']:
            ep1_step_by_rank[rank].append(step['duration'] / 1000.0)

    # Compute average step times
    all_ep2_steps = [t for times in ep2_step_by_rank.values() for t in times]
    all_ep1_steps = [t for times in ep1_step_by_rank.values() for t in times]

    avg_ep2_step = statistics.mean(all_ep2_steps) if all_ep2_steps else 0
    avg_ep1_step = statistics.mean(all_ep1_steps) if all_ep1_steps else 0
    total_step_diff = avg_ep2_step - avg_ep1_step

    # Compute contributions
    contributions = []
    for op_name in set(list(ep2_summary.keys()) + list(ep1_summary.keys())):
        ep2_avg = ep2_summary.get(op_name, {}).get('avg_ms', 0)
        ep1_avg = ep1_summary.get(op_name, {}).get('avg_ms', 0)
        diff = ep2_avg - ep1_avg

        if abs(diff) > 0.1:  # Only include significant differences
            contrib_pct = (diff / total_step_diff * 100) if total_step_diff != 0 else 0
            contributions.append({
                'op_name': op_name,
                'ep2_avg': ep2_avg,
                'ep1_avg': ep1_avg,
                'diff_ms': diff,
                'contrib_pct': contrib_pct,
                'ep2_std': ep2_summary.get(op_name, {}).get('std_ms', 0),
                'ep1_std': ep1_summary.get(op_name, {}).get('std_ms', 0),
                'ep2_total_flops': ep2_summary.get(op_name, {}).get('total_flops', 0),
                'ep1_total_flops': ep1_summary.get(op_name, {}).get('total_flops', 0),
                'modules': list(ep2_summary.get(op_name, {}).get('modules', set())),
            })

    contributions.sort(key=lambda x: abs(x['diff_ms']), reverse=True)

    # Analyze communication patterns
    comm_analysis = analyze_communication_patterns(ep2_data, ep1_data)

    # Analyze memory patterns
    memory_analysis = analyze_memory_patterns(ep2_data, ep1_data)

    # Module-level analysis
    module_analysis = analyze_module_performance(ep2_summary, ep1_summary)

    return {
        'ep2_data': ep2_data,
        'ep1_data': ep1_data,
        'ep2_summary': ep2_summary,
        'ep1_summary': ep1_summary,
        'ep2_step_by_rank': ep2_step_by_rank,
        'ep1_step_by_rank': ep1_step_by_rank,
        'avg_ep2_step': avg_ep2_step,
        'avg_ep1_step': avg_ep1_step,
        'total_step_diff': total_step_diff,
        'contributions': contributions,
        'comm_analysis': comm_analysis,
        'memory_analysis': memory_analysis,
        'module_analysis': module_analysis,
    }


def analyze_communication_patterns(ep2_data, ep1_data):
    """Analyze all-to-all and communication patterns"""
    ep2_comm = defaultdict(list)
    ep1_comm = defaultdict(list)

    for trace in ep2_data['traces']:
        for event in trace['communication_events']:
            ep2_comm[event['name']].append(event['dur'] / 1000.0)

    for trace in ep1_data['traces']:
        for event in trace['communication_events']:
            ep1_comm[event['name']].append(event['dur'] / 1000.0)

    return {
        'ep2': {k: {'mean': statistics.mean(v), 'count': len(v), 'total': sum(v)} for k, v in ep2_comm.items()},
        'ep1': {k: {'mean': statistics.mean(v), 'count': len(v), 'total': sum(v)} for k, v in ep1_comm.items()},
    }


def analyze_memory_patterns(ep2_data, ep1_data):
    """Analyze memory allocation patterns"""
    ep2_mem_total = sum(len(t['memory_events']) for t in ep2_data['traces'])
    ep1_mem_total = sum(len(t['memory_events']) for t in ep1_data['traces'])

    return {
        'ep2_total_events': ep2_mem_total,
        'ep1_total_events': ep1_mem_total,
        'diff': ep2_mem_total - ep1_mem_total
    }


def analyze_module_performance(ep2_summary, ep1_summary):
    """Aggregate performance by module hierarchy"""
    module_perf = defaultdict(lambda: {'ep2_ms': 0, 'ep1_ms': 0, 'ops': []})

    for op_name, stats in ep2_summary.items():
        modules = stats.get('modules', set())
        for module in modules:
            if module:
                module_perf[module]['ep2_ms'] += stats.get('total_ms', 0)
                module_perf[module]['ops'].append(op_name)

    for op_name, stats in ep1_summary.items():
        modules = stats.get('modules', set())
        for module in modules:
            if module:
                module_perf[module]['ep1_ms'] += stats.get('total_ms', 0)

    # Compute differences
    for module in module_perf:
        module_perf[module]['diff_ms'] = module_perf[module]['ep2_ms'] - module_perf[module]['ep1_ms']

    return dict(module_perf)


# Global data cache
ANALYSIS_DATA = None

def get_data():
    global ANALYSIS_DATA
    if ANALYSIS_DATA is None:
        ANALYSIS_DATA = load_ultra_deep_data()
    return ANALYSIS_DATA


def create_communication_analysis_plot():
    """Deep dive into communication overhead"""
    data = get_data()
    if not data:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    comm = data['comm_analysis']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Communication Operations - Time Comparison',
            'Communication Count (EP=2 vs EP=1)',
            'Total Communication Time by Operation',
            'Average Communication Latency'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Get all operation names
    all_ops = set(list(comm['ep2'].keys()) + list(comm['ep1'].keys()))

    for op in all_ops:
        ep2_mean = comm['ep2'].get(op, {}).get('mean', 0)
        ep1_mean = comm['ep1'].get(op, {}).get('mean', 0)
        ep2_count = comm['ep2'].get(op, {}).get('count', 0)
        ep1_count = comm['ep1'].get(op, {}).get('count', 0)
        ep2_total = comm['ep2'].get(op, {}).get('total', 0)
        ep1_total = comm['ep1'].get(op, {}).get('total', 0)

        # Plot 1: Mean time
        fig.add_trace(go.Bar(x=[op], y=[ep1_mean], name='EP=1', marker_color='lightseagreen', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=[op], y=[ep2_mean], name='EP=2', marker_color='indianred', showlegend=False), row=1, col=1)

        # Plot 2: Count
        fig.add_trace(go.Bar(x=[op], y=[ep1_count], name='EP=1', marker_color='lightseagreen', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=[op], y=[ep2_count], name='EP=2', marker_color='indianred', showlegend=False), row=1, col=2)

        # Plot 3: Total
        fig.add_trace(go.Bar(x=[op], y=[ep1_total], name='EP=1', marker_color='lightseagreen', showlegend=False), row=2, col=1)
        fig.add_trace(go.Bar(x=[op], y=[ep2_total], name='EP=2', marker_color='indianred', showlegend=False), row=2, col=1)

        # Plot 4: Latency
        fig.add_trace(go.Bar(x=[op], y=[ep2_mean - ep1_mean], marker_color='coral', showlegend=False), row=2, col=2)

    fig.update_layout(height=900, title_text="Communication Analysis - EP Overhead Breakdown", title_x=0.5)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Total Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Latency Diff (ms)", row=2, col=2)

    return fig


def create_module_performance_plot():
    """Module-level performance breakdown"""
    data = get_data()
    if not data:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    modules = data['module_analysis']

    # Sort by diff
    sorted_modules = sorted(modules.items(), key=lambda x: abs(x[1]['diff_ms']), reverse=True)[:15]

    fig = go.Figure()

    module_names = [m[0][:40] + '...' if len(m[0]) > 40 else m[0] for m in sorted_modules]
    diffs = [m[1]['diff_ms'] for m in sorted_modules]
    ep2_times = [m[1]['ep2_ms'] for m in sorted_modules]
    ep1_times = [m[1]['ep1_ms'] for m in sorted_modules]

    fig.add_trace(go.Bar(
        y=module_names,
        x=diffs,
        orientation='h',
        marker_color='coral',
        text=[f"+{d:.1f}ms" for d in diffs],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Diff: %{x:.1f}ms<br>EP=2: %{customdata[0]:.1f}ms<br>EP=1: %{customdata[1]:.1f}ms<extra></extra>',
        customdata=[[e2, e1] for e2, e1 in zip(ep2_times, ep1_times)]
    ))

    fig.update_layout(
        height=700,
        title_text="Module-Level Performance Analysis - Top 15 Slowdowns",
        xaxis_title="Time Difference (ms)",
        yaxis_title="Module"
    )

    return fig


def create_flops_efficiency_plot():
    """FLOPs efficiency analysis"""
    data = get_data()
    if not data:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # Find operations with FLOP data
    flop_ops = []
    for contrib in data['contributions']:
        if contrib['ep2_total_flops'] > 0 or contrib['ep1_total_flops'] > 0:
            flop_ops.append(contrib)

    if not flop_ops:
        return go.Figure().add_annotation(text="No FLOP data available in traces", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    flop_ops = sorted(flop_ops, key=lambda x: x['ep2_total_flops'], reverse=True)[:10]

    fig = go.Figure()

    op_names = [op['op_name'][:35] + '...' if len(op['op_name']) > 35 else op['op_name'] for op in flop_ops]
    ep2_flops = [op['ep2_total_flops'] / 1e9 for op in flop_ops]  # Convert to GFLOPs
    ep1_flops = [op['ep1_total_flops'] / 1e9 for op in flop_ops]

    fig.add_trace(go.Bar(x=op_names, y=ep1_flops, name='EP=1', marker_color='lightseagreen'))
    fig.add_trace(go.Bar(x=op_names, y=ep2_flops, name='EP=2', marker_color='indianred'))

    fig.update_layout(
        height=600,
        title_text="FLOPs Comparison - Top 10 Compute Operations",
        xaxis_title="Operation",
        yaxis_title="GFLOPs",
        barmode='group'
    )

    return fig


def create_rank_load_balance_plot():
    """Enhanced rank load balance analysis"""
    data = get_data()
    if not data:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    ranks = sorted(data['ep2_step_by_rank'].keys())

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Step Time by Rank (EP=2)',
            'Step Time by Rank (EP=1)',
            'Per-Rank Slowdown Distribution'
        ),
        row_heights=[0.35, 0.35, 0.30]
    )

    # EP=2 box plots
    for r in ranks:
        fig.add_trace(
            go.Box(
                y=data['ep2_step_by_rank'][r],
                name=f"Rank {r}",
                boxmean='sd',
                marker_color='indianred',
                showlegend=False
            ),
            row=1, col=1
        )

    # EP=1 box plots
    for r in ranks:
        fig.add_trace(
            go.Box(
                y=data['ep1_step_by_rank'][r],
                name=f"Rank {r}",
                boxmean='sd',
                marker_color='lightseagreen',
                showlegend=False
            ),
            row=2, col=1
        )

    # Slowdown analysis
    ep1_avg = [statistics.mean(data['ep1_step_by_rank'][r]) for r in ranks]
    ep2_avg = [statistics.mean(data['ep2_step_by_rank'][r]) for r in ranks]
    slowdown_pct = [(e2 - e1) / e1 * 100 for e1, e2 in zip(ep1_avg, ep2_avg)]

    fig.add_trace(
        go.Bar(
            x=[f"Rank {r}" for r in ranks],
            y=slowdown_pct,
            text=[f"{pct:.1f}%" for pct in slowdown_pct],
            textposition='outside',
            marker=dict(
                color=slowdown_pct,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Slowdown %", y=0.15, len=0.3)
            ),
            showlegend=False
        ),
        row=3, col=1
    )

    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Slowdown %", row=3, col=1)

    # Check if load imbalance
    variance = statistics.stdev(slowdown_pct) if len(slowdown_pct) > 1 else 0
    balance_status = "BALANCED ✓" if variance < 2.0 else "IMBALANCED ⚠"

    fig.update_layout(
        height=1200,
        title_text=f"Rank Load Balance Analysis - {balance_status} (±{variance:.2f}% variance)",
        title_x=0.5,
        showlegend=False
    )

    return fig


def create_memory_analysis_plot():
    """Memory allocation analysis"""
    data = get_data()
    if not data:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    mem = data['memory_analysis']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['EP=1', 'EP=2'],
        y=[mem['ep1_total_events'], mem['ep2_total_events']],
        text=[f"{mem['ep1_total_events']}", f"{mem['ep2_total_events']}"],
        textposition='outside',
        marker_color=['lightseagreen', 'indianred']
    ))

    fig.update_layout(
        height=500,
        title_text=f"Memory Events Comparison - EP=2 has {mem['diff']} more events ({mem['diff']/mem['ep1_total_events']*100:.1f}% increase)",
        yaxis_title="Total Memory Events",
        xaxis_title="Configuration"
    )

    return fig


def create_ultra_summary_text():
    """Ultra-detailed summary"""
    data = get_data()
    if not data:
        return "No data available. Run profiling first."

    summary = f"""# 🔬 Ultra-Deep EP Performance Analysis

## 📊 Overall Performance

- **EP=2 Average Step**: {data['avg_ep2_step']:.2f}ms
- **EP=1 Average Step**: {data['avg_ep1_step']:.2f}ms
- **Slowdown**: +{data['total_step_diff']:.2f}ms (+{(data['total_step_diff']/data['avg_ep1_step']*100):.1f}%)

## 🎯 Top 10 Bottlenecks

"""

    for i, c in enumerate(data['contributions'][:10], 1):
        summary += f"""
### {i}. `{c['op_name']}`
- **Time Diff**: +{c['diff_ms']:.1f}ms ({c['contrib_pct']:.1f}% of slowdown)
- **EP=2**: {c['ep2_avg']:.1f}ms ± {c['ep2_std']:.1f}ms
- **EP=1**: {c['ep1_avg']:.1f}ms ± {c['ep1_std']:.1f}ms
"""
        if c['modules']:
            summary += f"- **Module**: `{c['modules'][0]}`\n"
        if c['ep2_total_flops'] > 0:
            summary += f"- **FLOPs (EP=2)**: {c['ep2_total_flops']/1e9:.2f} GFLOPs\n"

    # Communication analysis
    comm = data['comm_analysis']
    if comm['ep2']:
        total_comm_ep2 = sum(v['total'] for v in comm['ep2'].values())
        total_comm_ep1 = sum(v['total'] for v in comm['ep1'].values())
        summary += f"""
## 📡 Communication Overhead

- **Total Communication Time (EP=2)**: {total_comm_ep2:.1f}ms
- **Total Communication Time (EP=1)**: {total_comm_ep1:.1f}ms
- **Communication Overhead**: +{total_comm_ep2 - total_comm_ep1:.1f}ms
- **% of Slowdown from Comms**: {(total_comm_ep2 - total_comm_ep1)/data['total_step_diff']*100:.1f}%

"""
        for op_name, stats in sorted(comm['ep2'].items(), key=lambda x: x[1]['total'], reverse=True)[:3]:
            summary += f"- `{op_name}`: {stats['mean']:.1f}ms avg × {stats['count']} calls = {stats['total']:.1f}ms total\n"

    # Memory analysis
    mem = data['memory_analysis']
    summary += f"""
## 💾 Memory Analysis

- **Memory Events (EP=2)**: {mem['ep2_total_events']}
- **Memory Events (EP=1)**: {mem['ep1_total_events']}
- **Increase**: +{mem['diff']} events (+{mem['diff']/mem['ep1_total_events']*100:.1f}%)

"""

    # Rank balance
    ranks = sorted(data['ep2_step_by_rank'].keys())
    ep1_avg = [statistics.mean(data['ep1_step_by_rank'][r]) for r in ranks]
    ep2_avg = [statistics.mean(data['ep2_step_by_rank'][r]) for r in ranks]
    slowdown_pct = [(e2 - e1) / e1 * 100 for e1, e2 in zip(ep1_avg, ep2_avg)]
    variance = statistics.stdev(slowdown_pct) if len(slowdown_pct) > 1 else 0

    summary += f"""
## ⚖️ Load Balance Analysis

- **Variance across ranks**: ±{variance:.2f}%
- **Status**: {"✓ BALANCED - not a load imbalance issue" if variance < 2.0 else "⚠ IMBALANCED - investigate rank differences"}

"""

    for r, pct in zip(ranks, slowdown_pct):
        summary += f"- Rank {r}: {pct:.1f}% slower\n"

    summary += f"""
## 💡 Key Findings

1. **Primary Bottleneck**: {data['contributions'][0]['op_name']} accounts for {data['contributions'][0]['contrib_pct']:.1f}% of slowdown
2. **Communication vs Compute**: {'Communication-bound' if total_comm_ep2 > data['total_step_diff'] * 0.5 else 'Compute-bound'} scaling issue
3. **Load Balance**: {'Uniform slowdown across ranks indicates systemic overhead, not imbalance' if variance < 2.0 else 'Significant rank variance suggests load imbalance'}

## 🚀 Optimization Recommendations

"""

    if 'all_to_all' in data['contributions'][0]['op_name'].lower():
        summary += """
1. **Primary Issue**: All-to-all communication overhead is inherent to EP=2
   - Consider overlapping communication with computation
   - Investigate token distribution to minimize data transfer

"""

    if any('copy' in c['op_name'].lower() or 'memcpy' in c['op_name'].lower() for c in data['contributions'][:5]):
        summary += """
2. **Memory Transfer Bottleneck Detected**:
   - Apply `non_blocking=True` to CPU tensor transfers in `expert_parallel.py:104`
   - Use pinned memory for faster H2D/D2H transfers

"""

    if mem['diff'] > mem['ep1_total_events'] * 0.2:
        summary += """
3. **Memory Allocation Overhead**:
   - EP=2 has {:.1f}% more memory events
   - Consider memory pooling or pre-allocation strategies

""".format(mem['diff']/mem['ep1_total_events']*100)

    return summary


def create_detailed_contrib_table():
    """Enhanced contribution table with all data"""
    data = get_data()
    if not data:
        return pd.DataFrame({"Error": ["No data available"]})

    rows = []
    for c in data['contributions'][:50]:
        rows.append({
            'Operation': c['op_name'][:60],
            'Diff (ms)': f"{c['diff_ms']:.2f}",
            'Contrib %': f"{c['contrib_pct']:.1f}%",
            'EP=2 (ms)': f"{c['ep2_avg']:.2f}",
            'EP=1 (ms)': f"{c['ep1_avg']:.2f}",
            'EP=2 Std': f"{c['ep2_std']:.2f}",
            'EP=1 Std': f"{c['ep1_std']:.2f}",
            'EP=2 GFLOPs': f"{c['ep2_total_flops']/1e9:.2f}" if c['ep2_total_flops'] > 0 else "N/A",
            'EP=1 GFLOPs': f"{c['ep1_total_flops']/1e9:.2f}" if c['ep1_total_flops'] > 0 else "N/A",
            'Module': c['modules'][0][:40] if c['modules'] else "N/A",
        })

    return pd.DataFrame(rows)


def create_ultra_dashboard():
    """Create the ultra-deep dashboard"""

    with gr.Blocks(title="Ultra-Deep EP Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔬 Ultra-Deep Expert Parallelism Performance Analysis")
        gr.Markdown("Leveraging enhanced profiling: memory, FLOPs, stack traces, modules, per-kernel timing")

        with gr.Row():
            load_btn = gr.Button("🔄 Load Ultra-Deep Analysis", variant="primary", size="lg")
            status = gr.Markdown("Click to load...")

        load_btn.click(
            fn=lambda: ("✅ Ultra-deep data loaded!" if load_ultra_deep_data() else "❌ Failed"),
            outputs=status
        )

        with gr.Tabs():
            with gr.Tab("📊 Summary"):
                summary_md = gr.Markdown()
                load_btn.click(fn=create_ultra_summary_text, outputs=summary_md)

            with gr.Tab("📡 Communication Analysis"):
                gr.Markdown("### Deep Dive into Communication Overhead (all-to-all, nccl, etc.)")
                comm_plot = gr.Plot()
                load_btn.click(fn=create_communication_analysis_plot, outputs=comm_plot)

            with gr.Tab("🧩 Module Performance"):
                gr.Markdown("### Performance Breakdown by Module Hierarchy")
                module_plot = gr.Plot()
                load_btn.click(fn=create_module_performance_plot, outputs=module_plot)

            with gr.Tab("⚡ FLOPs Efficiency"):
                gr.Markdown("### Compute Efficiency Analysis")
                flops_plot = gr.Plot()
                load_btn.click(fn=create_flops_efficiency_plot, outputs=flops_plot)

            with gr.Tab("⚖️ Rank Load Balance"):
                gr.Markdown("### Detailed Per-Rank Analysis - Detect False Negatives")
                rank_plot = gr.Plot()
                load_btn.click(fn=create_rank_load_balance_plot, outputs=rank_plot)

            with gr.Tab("💾 Memory Analysis"):
                gr.Markdown("### Memory Allocation Patterns")
                mem_plot = gr.Plot()
                load_btn.click(fn=create_memory_analysis_plot, outputs=mem_plot)

            with gr.Tab("📋 Full Data Table"):
                gr.Markdown("### Complete Operation Statistics")
                data_table = gr.Dataframe(wrap=True, max_height=700)
                load_btn.click(fn=create_detailed_contrib_table, outputs=data_table)

        gr.Markdown("""
        ---
        ## 🎯 Dashboard Features

        - **Communication Analysis**: Detailed breakdown of all-to-all and NCCL operations
        - **Module Performance**: Aggregated timing by MoE module (Router, GroupedExperts, etc.)
        - **FLOPs Efficiency**: Compare actual compute vs theoretical
        - **Rank Balance**: Detect false negatives from rank variance
        - **Memory Patterns**: Track memory allocation overhead
        - **Full Stack Traces**: (when available in trace data)

        ## 🔍 How to Identify Bottlenecks

        1. Check **Summary** tab for top bottlenecks
        2. Look at **Communication** to see if EP overhead is from data transfer
        3. Check **Rank Balance** to rule out load imbalance (false negative)
        4. Review **Module Performance** to identify which MoE component is slow
        5. Use **FLOPs** to determine if operations are compute-bound
        """)

    return demo


if __name__ == "__main__":
    dashboard = create_ultra_dashboard()

    print("="*80)
    print("🔬 Ultra-Deep EP Performance Analysis Dashboard")
    print("="*80)
    print("\n📡 Launching with public link...")
    print("="*80)

    dashboard.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7861,
        inbrowser=False,
        show_error=True,
    )
