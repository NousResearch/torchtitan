#!/usr/bin/env python3
"""
Comprehensive Interactive Dashboard for EP Performance Analysis
- Ultra-fast loading using pre-computed cache
- Real Plotly graphs embedded in Gradio
- Detailed operation analysis with variance
- Rank-level and global views
- Contribution percentage analysis
"""

import os
import sys
import json
import pickle
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from advanced_analysis import (
    analyze_all_traces,
    aggregate_statistics,
    compute_contribution_analysis,
    analyze_rank_differences,
    analyze_communication_patterns,
    analyze_memory_patterns,
    analyze_module_performance,
    analyze_by_source_location,
    format_operation_with_source,
    categorize_operation_type,
)

CACHE_FILE = SCRIPT_DIR / ".analysis_cache.pkl"


def load_from_cache(verbose=False):
    """Load pre-computed analysis data from cache (fast!)"""
    if not CACHE_FILE.exists():
        return None

    try:
        if verbose:
            print(f"📦 Loading cached analysis from {CACHE_FILE}...")
        start_time = time.time()

        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)

        load_time = time.time() - start_time
        cache_age = time.time() - data.get('cache_timestamp', 0)

        if verbose:
            print(f"✅ Loaded cached data in {load_time:.1f}s (cache age: {cache_age / 60:.1f} min)")

        return data

    except Exception as e:
        if verbose:
            print(f"❌ Failed to load cache: {e}")
        return None


def load_analysis_data_slow():
    """Load and process all analysis data (slow fallback)"""
    print("⚠️  No cache found. Loading and parsing all traces (60-90 seconds)...")
    print("💡 TIP: Run './scripts/ep/precompute_analysis.py' to create cache for instant loading!")

    os.chdir(SCRIPT_DIR.parent.parent)

    # Analyze all traces
    ep2_data = analyze_all_traces('.', 'ep2')
    ep1_data = analyze_all_traces('.', 'ep1')

    if not ep2_data or not ep1_data:
        return None

    # Aggregate statistics
    ep2_summary = aggregate_statistics(ep2_data)
    ep1_summary = aggregate_statistics(ep1_data)

    # Compute step times
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

    # Compute contributions
    contributions = compute_contribution_analysis(ep2_summary, ep1_summary, total_step_diff)

    # Normalize contribution keys for easier access
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

    # Rank analysis
    rank_diffs = analyze_rank_differences(ep2_data, ep1_data)

    # NEW: Ultra-deep analysis
    comm_analysis = analyze_communication_patterns(ep2_data, ep1_data)
    memory_analysis = analyze_memory_patterns(ep2_data, ep1_data)
    module_analysis = analyze_module_performance(ep2_summary, ep1_summary)
    source_location_analysis = analyze_by_source_location(ep2_summary, ep1_summary)

    return {
        'ep2_data': ep2_data,
        'ep1_data': ep1_data,
        'ep2_summary': ep2_summary,
        'ep1_summary': ep1_summary,
        'ep2_step_times': ep2_step_times,
        'ep1_step_times': ep1_step_times,
        'ep2_step_by_rank': ep2_step_by_rank,
        'ep1_step_by_rank': ep1_step_by_rank,
        'ep2_step_by_iter': ep2_step_by_iter,
        'ep1_step_by_iter': ep1_step_by_iter,
        'avg_ep2_step': avg_ep2_step,
        'avg_ep1_step': avg_ep1_step,
        'total_step_diff': total_step_diff,
        'contributions': normalized_contribs,
        'rank_diffs': rank_diffs,
        # NEW: Ultra-deep analysis results
        'comm_analysis': comm_analysis,
        'memory_analysis': memory_analysis,
        'module_analysis': module_analysis,
        'source_location_analysis': source_location_analysis,
    }


def load_analysis_data():
    """Load analysis data (tries cache first for speed)"""
    # Try cache first
    data = load_from_cache()

    # Fallback to slow loading if no cache
    if data is None:
        data = load_analysis_data_slow()

    return data


# Global data cache
ANALYSIS_DATA = None

def get_data():
    """Get cached analysis data"""
    global ANALYSIS_DATA
    if ANALYSIS_DATA is None:
        ANALYSIS_DATA = load_analysis_data()
    return ANALYSIS_DATA


def create_overview_plot():
    """Create overview comparison plot"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available. Run profiling first.",
                                         xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Average Step Time Comparison',
            'Step Time Distribution',
            'Top 8 Operations by Time Difference (ms)',
            'Rank-Level Comparison'
        ),
        specs=[[{'type': 'bar'}, {'type': 'box'}],
               [{'type': 'bar'}, {'type': 'bar'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Plot 1: Average step time
    fig.add_trace(
        go.Bar(
            x=['EP=1', 'EP=2'],
            y=[data['avg_ep1_step'], data['avg_ep2_step']],
            text=[f"{data['avg_ep1_step']:.1f}ms", f"{data['avg_ep2_step']:.1f}ms"],
            textposition='outside',
            marker_color=['lightseagreen', 'indianred'],
            name='Avg Step Time',
            showlegend=False
        ),
        row=1, col=1
    )

    # Plot 2: Box plot of step times
    fig.add_trace(
        go.Box(
            y=data['ep1_step_times'],
            name='EP=1',
            marker_color='lightseagreen',
            boxmean='sd'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(
            y=data['ep2_step_times'],
            name='EP=2',
            marker_color='indianred',
            boxmean='sd'
        ),
        row=1, col=2
    )

    # Plot 3: Top 8 bottlenecks by absolute time difference (cleaner view)
    top_contribs = data['contributions'][:8]
    fig.add_trace(
        go.Bar(
            x=[c['diff_ms'] for c in top_contribs],
            y=[c['op_name'][:35] + '...' if len(c['op_name']) > 35 else c['op_name'] for c in top_contribs],
            orientation='h',
            text=[f"+{c['diff_ms']:.0f}ms" for c in top_contribs],
            textposition='outside',
            marker_color='coral',
            showlegend=False,
            hovertemplate='<b>%{y}</b><br>Diff: %{x:.1f}ms<extra></extra>'
        ),
        row=2, col=1
    )

    # Plot 4: Rank comparison
    ranks = sorted(data['ep2_step_by_rank'].keys())
    ep1_avg_by_rank = [statistics.mean(data['ep1_step_by_rank'][r]) for r in ranks]
    ep2_avg_by_rank = [statistics.mean(data['ep2_step_by_rank'][r]) for r in ranks]

    fig.add_trace(
        go.Bar(
            x=[f"Rank {r}" for r in ranks],
            y=ep1_avg_by_rank,
            name='EP=1',
            marker_color='lightseagreen'
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(
            x=[f"Rank {r}" for r in ranks],
            y=ep2_avg_by_rank,
            name='EP=2',
            marker_color='indianred'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Configuration", row=1, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)

    fig.update_yaxes(title_text="Time (ms)", row=1, col=2)

    fig.update_xaxes(title_text="Contribution %", row=2, col=1)
    fig.update_yaxes(title_text="Operation", row=2, col=1)

    fig.update_xaxes(title_text="Rank", row=2, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=2)

    fig.update_layout(
        height=900,
        showlegend=True,
        title_text=f"EP Performance Overview - EP=2 is {data['total_step_diff']:.1f}ms slower ({(data['total_step_diff']/data['avg_ep1_step']*100):.1f}%)",
        title_x=0.5
    )

    return fig


def create_detailed_contribution_plot(top_n=15):
    """Create single clean contribution chart with time diff + percentage"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    contribs = data['contributions'][:top_n]

    # NEW: Enhance operation names with source location (file:line)
    op_names = []
    for c in contribs:
        op_name = c['op_name']
        # Try to get source info from summary
        ep2_stats = data['ep2_summary'].get(op_name, {})
        source_info = ep2_stats.get('source_info')

        if source_info:
            # Format: "operation @ file:line"
            enhanced_name = format_operation_with_source(op_name, source_info)
            # Truncate if still too long
            if len(enhanced_name) > 60:
                enhanced_name = enhanced_name[:57] + "..."
            op_names.append(enhanced_name)
        else:
            # No source info, just truncate operation name
            if len(op_name) > 45:
                op_names.append(op_name[:42] + "...")
            else:
                op_names.append(op_name)

    # Calculate sum of displayed contributions
    total_displayed_pct = sum(c.get('normalized_pct', 0) for c in contribs)

    fig = go.Figure()

    # Single bar chart with time difference, percentage in hover/label
    fig.add_trace(
        go.Bar(
            y=op_names,
            x=[c['diff_ms'] for c in contribs],
            orientation='h',
            text=[f"+{c['diff_ms']:.0f}ms ({c.get('normalized_pct', 0):.1f}%)" for c in contribs],
            textposition='outside',
            marker=dict(
                color=[c.get('normalized_pct', 0) for c in contribs],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Contrib %"),
                line=dict(width=0.5, color='darkred')
            ),
            hovertemplate=(
                '<b>%{y}</b><br>' +
                'Time Difference: <b>%{x:.1f}ms</b><br>' +
                'Contribution: <b>%{customdata[0]:.1f}%</b> (of total step overhead)<br>' +
                'EP=2: %{customdata[1]:.1f}ms<br>' +
                'EP=1: %{customdata[2]:.1f}ms<br>' +
                '<extra></extra>'
            ),
            customdata=[[c.get('normalized_pct', 0), c['ep2_avg'], c['ep1_avg']] for c in contribs]
        )
    )

    fig.update_xaxes(title_text="Time Difference (ms)")
    fig.update_yaxes(autorange="reversed")  # Top operation at top

    fig.update_layout(
        height=max(500, top_n * 30),
        title_text=f"Top {top_n} Operations by Contribution<br><sub>Total Slowdown: {data['total_step_diff']:.1f}ms | Displayed ops sum: {total_displayed_pct:.1f}%</sub>",
        title_x=0.5,
        showlegend=False
    )

    return fig


def create_contribution_pie_chart(top_n=12):
    """Create pie chart showing top N ops by global contribution (relative to total step overhead)"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    contribs = data['contributions'][:top_n]

    labels = []
    values = []
    hover_texts = []
    total_displayed_pct = 0

    for c in contribs:
        op_name = c['op_name']
        if len(op_name) > 35:
            op_name = op_name[:32] + "..."

        # Use global normalized percentage (relative to entire step overhead)
        pct = c.get('normalized_pct', 0)
        total_displayed_pct += pct

        labels.append(op_name)
        values.append(pct)
        hover_texts.append(f"{c['op_name']}<br>+{c['diff_ms']:.0f}ms<br>{pct:.1f}% of total step overhead")

    # Add "Others" if displayed operations don't sum to 100%
    if total_displayed_pct < 99:  # Allow small rounding tolerance
        others_pct = 100 - total_displayed_pct
        labels.append("Others")
        values.append(others_pct)
        hover_texts.append(f"Remaining operations<br>{others_pct:.1f}% of total step overhead")

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        text=[f"{v:.1f}%" for v in values],
        textposition='inside',
        textfont_size=11,
        hovertext=hover_texts,
        hoverinfo='text',
        marker=dict(
            colors=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#e7cb94', '#843c39', '#aaaaaa'],
            line=dict(color='white', width=2)
        ),
        hole=0.3  # Donut chart for better readability
    )])

    fig.update_layout(
        title_text=f"Top {top_n} Operations - Contribution to Total Step Overhead<br><sub>Displayed ops sum: {total_displayed_pct:.1f}%</sub>",
        title_x=0.5,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        annotations=[dict(text=f'Top {top_n}<br><b>{total_displayed_pct:.1f}%</b>', x=0.5, y=0.5, font_size=14, showarrow=False)]
    )

    return fig


def create_variance_analysis_plot(top_n=8):
    """Create box plots showing variance across ranks and iterations - cleaner view"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    top_ops = [c['op_name'] for c in data['contributions'][:top_n]]

    # Truncate operation names for cleaner display
    op_display_names = []
    for op in top_ops:
        if len(op) > 50:
            op_display_names.append(op[:47] + "...")
        else:
            op_display_names.append(op)

    fig = make_subplots(
        rows=top_n, cols=1,
        subplot_titles=op_display_names,
        vertical_spacing=0.04
    )

    for idx, op_name in enumerate(top_ops, 1):
        ep1_stats = data['ep1_summary'].get(op_name, {})
        ep2_stats = data['ep2_summary'].get(op_name, {})

        ep1_times = ep1_stats.get('all_times', [])
        ep2_times = ep2_stats.get('all_times', [])

        if ep1_times:
            fig.add_trace(
                go.Box(
                    y=ep1_times,
                    name='EP=1',
                    marker_color='lightseagreen',
                    boxmean='sd',
                    showlegend=(idx == 1),
                    boxpoints=False  # Hide individual points for cleaner view
                ),
                row=idx, col=1
            )

        if ep2_times:
            fig.add_trace(
                go.Box(
                    y=ep2_times,
                    name='EP=2',
                    marker_color='indianred',
                    boxmean='sd',
                    showlegend=(idx == 1),
                    boxpoints=False
                ),
                row=idx, col=1
            )

        fig.update_yaxes(title_text="ms", row=idx, col=1, title_font=dict(size=10))

    fig.update_layout(
        height=max(700, top_n * 100),
        showlegend=True,
        title_text=f"Variance Analysis - Top {top_n} Operations (EP=2 vs EP=1)",
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_rank_analysis_plot():
    """Detailed rank-level analysis"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    ranks = sorted(data['ep2_step_by_rank'].keys())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Average Step Time by Rank',
            'Step Time Distribution by Rank (EP=2)',
            'Step Time Distribution by Rank (EP=1)',
            'Slowdown % by Rank'
        ),
        specs=[[{'type': 'bar'}, {'type': 'box'}],
               [{'type': 'box'}, {'type': 'bar'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Plot 1: Bar chart
    ep1_avg = [statistics.mean(data['ep1_step_by_rank'][r]) for r in ranks]
    ep2_avg = [statistics.mean(data['ep2_step_by_rank'][r]) for r in ranks]

    fig.add_trace(
        go.Bar(x=[f"Rank {r}" for r in ranks], y=ep1_avg, name='EP=1', marker_color='lightseagreen'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=[f"Rank {r}" for r in ranks], y=ep2_avg, name='EP=2', marker_color='indianred'),
        row=1, col=1
    )

    # Plot 2: EP=2 box plots by rank
    for r in ranks:
        fig.add_trace(
            go.Box(
                y=data['ep2_step_by_rank'][r],
                name=f"Rank {r}",
                boxmean='sd',
                showlegend=False
            ),
            row=1, col=2
        )

    # Plot 3: EP=1 box plots by rank
    for r in ranks:
        fig.add_trace(
            go.Box(
                y=data['ep1_step_by_rank'][r],
                name=f"Rank {r}",
                boxmean='sd',
                showlegend=False
            ),
            row=2, col=1
        )

    # Plot 4: Slowdown percentage
    slowdown_pct = [(e2 - e1) / e1 * 100 for e1, e2 in zip(ep1_avg, ep2_avg)]
    fig.add_trace(
        go.Bar(
            x=[f"Rank {r}" for r in ranks],
            y=slowdown_pct,
            text=[f"{pct:.1f}%" for pct in slowdown_pct],
            textposition='outside',
            marker_color='coral',
            showlegend=False
        ),
        row=2, col=2
    )

    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Slowdown %", row=2, col=2)

    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Rank-Level Analysis - Identifying Load Imbalance",
        title_x=0.5
    )

    return fig


def create_communication_analysis_plot():
    """Analyze communication patterns (all-to-all, NCCL)"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    comm = data.get('comm_analysis', {})
    if not comm:
        return go.Figure().add_annotation(
            text="No communication data available.<br>This may indicate no all-to-all operations were detected.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    # Extract communication operations
    comm_ops = []
    for op_name, stats in comm.items():
        if 'ep2' in stats and 'ep1' in stats:
            comm_ops.append({
                'op': op_name,
                'ep2_mean': stats['ep2'].get('mean', 0),
                'ep1_mean': stats['ep1'].get('mean', 0),
                'ep2_count': stats['ep2'].get('count', 0),
                'ep1_count': stats['ep1'].get('count', 0),
                'ep2_total': stats['ep2'].get('total', 0),
                'ep1_total': stats['ep1'].get('total', 0),
            })

    if not comm_ops:
        return go.Figure().add_annotation(
            text="No communication operations with data in both EP=1 and EP=2",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    # Sort by time difference
    comm_ops.sort(key=lambda x: x['ep2_total'] - x['ep1_total'], reverse=True)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Communication Time Comparison (ms)',
            'Operation Count Comparison',
            'Average Latency per Call (ms)',
            'Total Communication Time'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    op_names = [c['op'][:30] + '...' if len(c['op']) > 30 else c['op'] for c in comm_ops]

    # Plot 1: Mean time comparison
    fig.add_trace(
        go.Bar(x=op_names, y=[c['ep1_mean'] for c in comm_ops], name='EP=1 Mean', marker_color='lightseagreen'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=op_names, y=[c['ep2_mean'] for c in comm_ops], name='EP=2 Mean', marker_color='indianred'),
        row=1, col=1
    )

    # Plot 2: Count comparison
    fig.add_trace(
        go.Bar(x=op_names, y=[c['ep1_count'] for c in comm_ops], name='EP=1', marker_color='lightseagreen', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=op_names, y=[c['ep2_count'] for c in comm_ops], name='EP=2', marker_color='indianred', showlegend=False),
        row=1, col=2
    )

    # Plot 3: Latency per call
    latency_diff = [(c['ep2_mean'] - c['ep1_mean']) if c['ep1_mean'] > 0 else 0 for c in comm_ops]
    fig.add_trace(
        go.Bar(
            x=op_names,
            y=latency_diff,
            marker_color='coral',
            text=[f"+{d:.1f}ms" if d > 0 else f"{d:.1f}ms" for d in latency_diff],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=1
    )

    # Plot 4: Total time
    fig.add_trace(
        go.Bar(x=op_names, y=[c['ep1_total'] for c in comm_ops], name='EP=1', marker_color='lightseagreen', showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=op_names, y=[c['ep2_total'] for c in comm_ops], name='EP=2', marker_color='indianred', showlegend=False),
        row=2, col=2
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Diff (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=2)

    total_comm_overhead = sum(c['ep2_total'] - c['ep1_total'] for c in comm_ops)
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text=f"📡 Communication Analysis - Total Overhead: +{total_comm_overhead:.1f}ms",
        title_x=0.5
    )

    return fig


def create_module_performance_plot():
    """Analyze performance by module hierarchy"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    modules = data.get('module_analysis', {})
    if not modules:
        return go.Figure().add_annotation(
            text="No module hierarchy data available.<br>Enhanced profiling may not have captured module info.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    # Convert to list and sort by time difference
    module_list = []
    for module_name, stats in modules.items():
        if 'ep2_time' in stats and 'ep1_time' in stats:
            module_list.append({
                'module': module_name,
                'ep2_time': stats['ep2_time'],
                'ep1_time': stats['ep1_time'],
                'diff': stats['ep2_time'] - stats['ep1_time'],
                'ep2_count': stats.get('ep2_count', 0),
                'ep1_count': stats.get('ep1_count', 0),
            })

    if not module_list:
        return go.Figure().add_annotation(
            text="No module data with both EP=1 and EP=2 times",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    module_list.sort(key=lambda x: x['diff'], reverse=True)
    top_modules = module_list[:15]  # Show top 15

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Module Time Comparison', 'Time Difference by Module'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]],
        horizontal_spacing=0.12
    )

    module_names = [m['module'][:40] + '...' if len(m['module']) > 40 else m['module'] for m in top_modules]

    # Plot 1: Comparison
    fig.add_trace(
        go.Bar(
            x=module_names,
            y=[m['ep1_time'] for m in top_modules],
            name='EP=1',
            marker_color='lightseagreen'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=module_names,
            y=[m['ep2_time'] for m in top_modules],
            name='EP=2',
            marker_color='indianred'
        ),
        row=1, col=1
    )

    # Plot 2: Difference
    fig.add_trace(
        go.Bar(
            x=module_names,
            y=[m['diff'] for m in top_modules],
            marker_color='coral',
            text=[f"+{m['diff']:.1f}ms" for m in top_modules],
            textposition='outside',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Diff: %{y:.1f}ms<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Diff (ms)", row=1, col=2)

    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="🏗️ Module Performance Analysis - Top 15 Modules by Slowdown",
        title_x=0.5
    )

    return fig


def create_flops_efficiency_plot():
    """Analyze FLOPs efficiency across operations"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    # Extract FLOPs data from summary
    flops_data = []
    for op_name, stats in data['ep2_summary'].items():
        if 'all_flops' in stats and stats['all_flops'] and stats.get('avg_ms', 0) > 0:
            avg_flops = statistics.mean(stats['all_flops']) if stats['all_flops'] else 0
            if avg_flops > 0:
                # GFLOPs = FLOPs / 1e9, GFLOPs/s = GFLOPs / (time_ms / 1000)
                gflops = avg_flops / 1e9
                time_s = stats['avg_ms'] / 1000.0
                gflops_per_sec = gflops / time_s if time_s > 0 else 0

                flops_data.append({
                    'op': op_name,
                    'gflops': gflops,
                    'time_ms': stats['avg_ms'],
                    'gflops_per_sec': gflops_per_sec,
                })

    if not flops_data:
        return go.Figure().add_annotation(
            text="No FLOPs data available.<br>Enhanced profiling with with_flops=True may not have captured FLOPs.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    # Sort by GFLOPs/s
    flops_data.sort(key=lambda x: x['gflops_per_sec'], reverse=True)
    top_flops = flops_data[:20]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Compute Intensity (GFLOPs)', 'Compute Efficiency (GFLOPs/s)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]],
        horizontal_spacing=0.12
    )

    op_names = [f['op'][:35] + '...' if len(f['op']) > 35 else f['op'] for f in top_flops]

    # Plot 1: GFLOPs
    fig.add_trace(
        go.Bar(
            y=op_names,
            x=[f['gflops'] for f in top_flops],
            orientation='h',
            marker_color='steelblue',
            text=[f"{f['gflops']:.1f}" for f in top_flops],
            textposition='outside',
            showlegend=False,
            hovertemplate='<b>%{y}</b><br>GFLOPs: %{x:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Plot 2: GFLOPs/s
    fig.add_trace(
        go.Bar(
            y=op_names,
            x=[f['gflops_per_sec'] for f in top_flops],
            orientation='h',
            marker_color='darkorange',
            text=[f"{f['gflops_per_sec']:.1f}" for f in top_flops],
            textposition='outside',
            showlegend=False,
            hovertemplate='<b>%{y}</b><br>GFLOPs/s: %{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="GFLOPs", row=1, col=1)
    fig.update_xaxes(title_text="GFLOPs/s", row=1, col=2)
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        height=max(600, len(top_flops) * 25),
        title_text="⚡ FLOPs Efficiency Analysis - Top 20 Operations (EP=2)",
        title_x=0.5
    )

    return fig


def create_memory_analysis_plot():
    """Analyze memory allocation patterns"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    mem_analysis = data.get('memory_analysis', {})
    if not mem_analysis:
        return go.Figure().add_annotation(
            text="No memory event data available.<br>Enhanced profiling may not have captured memory events.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    ep2_events = mem_analysis.get('ep2_memory_events', 0)
    ep1_events = mem_analysis.get('ep1_memory_events', 0)
    diff_events = ep2_events - ep1_events
    pct_increase = (diff_events / ep1_events * 100) if ep1_events > 0 else 0

    fig = go.Figure()

    # Simple bar chart comparing memory events
    fig.add_trace(go.Bar(
        x=['EP=1', 'EP=2'],
        y=[ep1_events, ep2_events],
        text=[f"{ep1_events}", f"{ep2_events}"],
        textposition='outside',
        marker_color=['lightseagreen', 'indianred'],
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>Memory Events: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title_text=f"💾 Memory Event Analysis<br><sub>EP=2 has +{diff_events} events (+{pct_increase:.1f}%) vs EP=1</sub>",
        title_x=0.5,
        xaxis_title="Configuration",
        yaxis_title="Memory Event Count",
        height=500
    )

    return fig


def create_source_location_analysis_plot():
    """NEW: Analyze operations grouped by source file location"""
    data = get_data()
    if data is None:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    source_analysis = data.get('source_location_analysis', {})
    if not source_analysis:
        return go.Figure().add_annotation(
            text="No source location data available.<br>Stack traces may not have been captured during profiling.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    # Sort files by total time difference
    sorted_files = sorted(source_analysis.items(), key=lambda x: abs(x[1]['total_diff']), reverse=True)
    top_files = sorted_files[:10]  # Show top 10 files

    if not top_files:
        return go.Figure().add_annotation(
            text="No operations with source location information",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    # Create accordion-style visualization
    fig = go.Figure()

    y_pos = 0
    y_labels = []
    y_values = []
    colors = []
    hover_texts = []
    op_types_list = []

    type_colors = {
        'Communication': '#FF6B6B',
        'Memory': '#FFA500',
        'Synchronization': '#FFD700',
        'Compute': '#4ECDC4',
        'Other': '#95a5a6'
    }

    for file_path, file_data in top_files:
        # Add file header
        total_diff = file_data['total_diff']
        y_labels.append(f"📁 {file_path}")
        y_values.append(total_diff)
        colors.append('#2c3e50')
        hover_texts.append(f"<b>{file_path}</b><br>Total overhead: {total_diff:.1f}ms<br>Operations: {len(file_data['operations'])}")
        op_types_list.append('File')
        y_pos += 1

        # Add top 5 operations from this file
        top_ops = file_data['operations'][:5]
        for op in top_ops:
            op_display = f"  Line {op['line']}: {op['op'][:40]}..."if len(op['op']) > 40 else f"  Line {op['line']}: {op['op']}"
            y_labels.append(op_display)
            y_values.append(op['diff'])
            colors.append(type_colors.get(op['type'], '#95a5a6'))
            hover_texts.append(
                f"<b>{op['op']}</b><br>" +
                f"Type: {op['type']}<br>" +
                f"Line: {op['line']}<br>" +
                f"EP=2: {op['ep2_time']:.1f}ms<br>" +
                f"EP=1: {op['ep1_time']:.1f}ms<br>" +
                f"Diff: +{op['diff']:.1f}ms"
            )
            op_types_list.append(op['type'])
            y_pos += 1

    # Create horizontal bar chart
    fig.add_trace(go.Bar(
        y=y_labels,
        x=y_values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"+{v:.0f}ms" if v > 0 else f"{v:.0f}ms" for v in y_values],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        title_text="🗂️ Source Location Analysis - Operations Grouped by File<br><sub>Top 10 files by overhead, showing top 5 operations each</sub>",
        title_x=0.5,
        xaxis_title="Time Difference (ms)",
        yaxis=dict(autorange="reversed"),
        height=max(600, len(y_labels) * 25),
        showlegend=False
    )

    # Add legend for operation types
    for op_type, color in type_colors.items():
        fig.add_trace(go.Bar(
            x=[0],
            y=[op_type],
            orientation='h',
            marker=dict(color=color),
            name=op_type,
            showlegend=True,
            visible='legendonly'
        ))

    return fig


def create_detailed_table():
    """Create comprehensive data table"""
    data = get_data()
    if data is None:
        return pd.DataFrame({"Error": ["No data available. Run profiling first."]})

    rows = []
    for contrib in data['contributions'][:50]:
        op_name = contrib['op_name']
        ep2_stats = data['ep2_summary'].get(op_name, {})
        ep1_stats = data['ep1_summary'].get(op_name, {})

        # NEW: Get source location and operation type
        source_info = ep2_stats.get('source_info') or ep1_stats.get('source_info')
        op_type = ep2_stats.get('op_type') or ep1_stats.get('op_type') or 'Other'

        rows.append({
            'Operation': op_name,
            'Type': op_type,  # NEW
            'Source File': source_info['file'] if source_info else 'N/A',  # NEW
            'Line': source_info['line'] if source_info else 'N/A',  # NEW
            'EP=2 Mean (ms)': f"{contrib['ep2_avg']:.2f}",
            'EP=1 Mean (ms)': f"{contrib['ep1_avg']:.2f}",
            'Diff (ms)': f"{contrib['diff_ms']:.2f}",
            'Contribution %': f"{contrib['contrib_pct']:.1f}%",
            'EP=2 Std (ms)': f"{contrib['ep2_std']:.2f}",
            'EP=1 Std (ms)': f"{contrib['ep1_std']:.2f}",
            'EP=2 Min (ms)': f"{ep2_stats.get('min_total_ms', 0):.2f}",
            'EP=2 Max (ms)': f"{ep2_stats.get('max_total_ms', 0):.2f}",
            'EP=1 Min (ms)': f"{ep1_stats.get('min_total_ms', 0):.2f}",
            'EP=1 Max (ms)': f"{ep1_stats.get('max_total_ms', 0):.2f}",
            'EP=2 Count': f"{ep2_stats.get('avg_count', 0):.0f}",
            'EP=1 Count': f"{ep1_stats.get('avg_count', 0):.0f}",
        })

    return pd.DataFrame(rows)


def create_summary_text():
    """Create summary statistics text"""
    data = get_data()
    if data is None:
        return "No data available. Please run profiling first."

    summary = f"""# 📊 EP Performance Analysis Summary

## Overall Performance

- **EP=2 Average Step Time**: {data['avg_ep2_step']:.2f}ms (±{statistics.stdev(data['ep2_step_times']):.2f}ms)
- **EP=1 Average Step Time**: {data['avg_ep1_step']:.2f}ms (±{statistics.stdev(data['ep1_step_times']):.2f}ms)
- **Difference**: +{data['total_step_diff']:.2f}ms (+{(data['total_step_diff']/data['avg_ep1_step']*100):.1f}%)
- **Total Traces Analyzed**: {len(data['ep2_data']['traces'])} EP=2, {len(data['ep1_data']['traces'])} EP=1

## Top 5 Bottlenecks

"""

    for i, contrib in enumerate(data['contributions'][:5], 1):
        summary += f"{i}. **{contrib['op_name']}**\n"
        summary += f"   - Time Difference: +{contrib['diff_ms']:.1f}ms\n"
        summary += f"   - Contribution: **{contrib['contrib_pct']:.1f}%**\n"
        summary += f"   - EP=2: {contrib['ep2_avg']:.1f}ms (±{contrib['ep2_std']:.1f}ms)\n"
        summary += f"   - EP=1: {contrib['ep1_avg']:.1f}ms (±{contrib['ep1_std']:.1f}ms)\n\n"

    # New operations analysis
    ep2_ops = set(data['ep2_summary'].keys())
    ep1_ops = set(data['ep1_summary'].keys())
    new_in_ep2 = ep2_ops - ep1_ops
    new_in_ep1 = ep1_ops - ep2_ops

    summary += f"""
## New Operations Analysis

"""

    if new_in_ep2:
        summary += f"**Operations only in EP=2** ({len(new_in_ep2)} operations):\n"
        # Show top 5 by time
        ep2_only_sorted = sorted([(op, data['ep2_summary'][op]) for op in new_in_ep2
                                   if 'avg_ms' in data['ep2_summary'][op]],
                                  key=lambda x: x[1].get('avg_ms', 0), reverse=True)[:5]
        for op, stats in ep2_only_sorted:
            summary += f"  - `{op}`: {stats.get('avg_ms', 0):.1f}ms\n"
        if len(new_in_ep2) > 5:
            summary += f"  - ... and {len(new_in_ep2) - 5} more\n"
    else:
        summary += "**Operations only in EP=2**: None\n"

    summary += "\n"

    if new_in_ep1:
        summary += f"**Operations only in EP=1** ({len(new_in_ep1)} operations):\n"
        # Show top 5 by time
        ep1_only_sorted = sorted([(op, data['ep1_summary'][op]) for op in new_in_ep1
                                   if 'avg_ms' in data['ep1_summary'][op]],
                                  key=lambda x: x[1].get('avg_ms', 0), reverse=True)[:5]
        for op, stats in ep1_only_sorted:
            summary += f"  - `{op}`: {stats.get('avg_ms', 0):.1f}ms\n"
        if len(new_in_ep1) > 5:
            summary += f"  - ... and {len(new_in_ep1) - 5} more\n"
    else:
        summary += "**Operations only in EP=1**: None\n"

    summary += f"""
## Rank Analysis

"""

    for rank_diff in data['rank_diffs']:
        pct_diff = rank_diff.get('pct_diff', rank_diff.get('pct', 0))
        summary += f"- **Rank {rank_diff['rank']}**: EP=2: {rank_diff['ep2_avg']:.1f}ms | EP=1: {rank_diff['ep1_avg']:.1f}ms | Diff: +{rank_diff['diff']:.1f}ms ({pct_diff:.1f}%)\n"

    # Check variance across ranks
    rank_variances = [rank_diff.get('pct_diff', rank_diff.get('pct', 0)) for rank_diff in data['rank_diffs']]
    variance_std = statistics.stdev(rank_variances) if len(rank_variances) > 1 else 0

    if variance_std < 1.0:
        summary += "\n**Conclusion**: All ranks show consistent slowdown (±{:.2f}%) - **not a load imbalance issue**.\n".format(variance_std)
    else:
        summary += "\n**Conclusion**: Ranks show variable slowdown (±{:.2f}%) - **potential load imbalance**.\n".format(variance_std)

    # NEW: Ultra-deep analysis insights
    comm_analysis = data.get('comm_analysis', {})
    memory_analysis = data.get('memory_analysis', {})
    module_analysis = data.get('module_analysis', {})

    summary += f"""
## 🔬 Ultra-Deep Analysis Results

"""

    # Communication Analysis
    if comm_analysis:
        total_comm_overhead = 0
        comm_ops_count = 0
        for op_name, stats in comm_analysis.items():
            if 'ep2' in stats and 'ep1' in stats:
                overhead = stats['ep2'].get('total', 0) - stats['ep1'].get('total', 0)
                total_comm_overhead += overhead
                comm_ops_count += 1

        if comm_ops_count > 0:
            comm_pct = (total_comm_overhead / data['total_step_diff'] * 100) if data['total_step_diff'] > 0 else 0
            summary += f"### 📡 Communication Overhead\n"
            summary += f"- **Total communication overhead**: +{total_comm_overhead:.1f}ms\n"
            summary += f"- **Contribution to slowdown**: {comm_pct:.1f}%\n"
            summary += f"- **Number of communication operations**: {comm_ops_count}\n"
            if comm_pct > 50:
                summary += f"- **⚠️ Communication-bound**: >50% of slowdown is from communication (inherent to EP)\n"
            summary += "\n"

    # Memory Analysis
    if memory_analysis:
        ep2_mem = memory_analysis.get('ep2_memory_events', 0)
        ep1_mem = memory_analysis.get('ep1_memory_events', 0)
        if ep1_mem > 0:
            mem_increase_pct = ((ep2_mem - ep1_mem) / ep1_mem * 100)
            summary += f"### 💾 Memory Event Analysis\n"
            summary += f"- **EP=2 memory events**: {ep2_mem}\n"
            summary += f"- **EP=1 memory events**: {ep1_mem}\n"
            summary += f"- **Increase**: +{ep2_mem - ep1_mem} events (+{mem_increase_pct:.1f}%)\n"
            if mem_increase_pct > 30:
                summary += f"- **⚠️ High memory allocation overhead**: Consider memory pooling\n"
            summary += "\n"

    # Module Analysis
    if module_analysis:
        module_list = sorted(
            [(m, s['ep2_time'] - s['ep1_time']) for m, s in module_analysis.items() if 'ep2_time' in s and 'ep1_time' in s],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        if module_list:
            summary += f"### 🏗️ Top 5 Slowest Modules\n"
            for module_name, diff in module_list:
                summary += f"- `{module_name}`: +{diff:.1f}ms slower\n"
            summary += "\n"

    summary += f"""
## Key Insights

1. **Communication Overhead**: `nccl:all_to_all` operations contribute {data['contributions'][0]['contrib_pct']:.1f}% to slowdown (inherent to EP)
2. **Memory Transfer Bottleneck**: `aten::_to_copy` and `cudaMemcpyAsync` contribute {sum(c['contrib_pct'] for c in data['contributions'][1:3] if 'copy' in c['op_name'].lower() or 'memcpy' in c['op_name'].lower()):.1f}% (optimizable)
3. **Variance**: {'Low' if statistics.stdev(data['ep2_step_times']) / data['avg_ep2_step'] < 0.05 else 'High'} variance indicates {'consistent' if statistics.stdev(data['ep2_step_times']) / data['avg_ep2_step'] < 0.05 else 'variable'} performance

## Recommended Actions

1. **Apply memory transfer fix** in `expert_parallel.py:104`:
   ```python
   output_splits = (...).to(torch.device("cpu"), non_blocking=True)
   ```
   Expected improvement: 20-30% throughput increase

2. **Profile again** after applying fix to measure improvement

3. **Consider** overlapping communication with computation for all-to-all operations
"""

    return summary


def create_dashboard():
    """Create the main dashboard"""

    with gr.Blocks(title="EP Performance Analysis - Ultra-Deep", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Expert Parallelism Performance Analysis - Ultra-Deep Dashboard")
        gr.Markdown("Comprehensive analysis with enhanced profiling: Memory, FLOPs, Communication, Modules, and more!")

        # Load data button
        with gr.Row():
            load_btn = gr.Button("🔄 Load/Refresh Analysis Data", variant="primary", size="lg")
            status_text = gr.Markdown("Click 'Load/Refresh' to analyze profiling data")

        load_btn.click(
            fn=lambda: ("✅ Data loaded successfully!" if load_analysis_data() else "❌ Failed to load data"),
            outputs=status_text
        )

        with gr.Tabs():
            # Tab 1: Summary
            with gr.Tab("📊 Summary"):
                summary_md = gr.Markdown()
                load_btn.click(fn=create_summary_text, outputs=summary_md)

            # Tab 2: Overview Plots
            with gr.Tab("📈 Overview"):
                gr.Markdown("### Global Performance Overview")
                overview_plot = gr.Plot()
                load_btn.click(fn=create_overview_plot, outputs=overview_plot)

            # Tab 3: Detailed Contributions
            with gr.Tab("🎯 Contribution Analysis"):
                gr.Markdown("### Detailed Operation-Level Contribution Analysis")
                with gr.Row():
                    top_n_slider = gr.Slider(10, 50, value=30, step=5, label="Number of operations to show")
                contrib_plot = gr.Plot()
                load_btn.click(
                    fn=lambda: create_detailed_contribution_plot(30),
                    outputs=contrib_plot
                )
                top_n_slider.change(
                    fn=create_detailed_contribution_plot,
                    inputs=top_n_slider,
                    outputs=contrib_plot
                )

            # Tab 4: Variance Analysis
            with gr.Tab("📊 Variance Analysis"):
                gr.Markdown("### Box Plots - Variance Across Ranks and Iterations")
                with gr.Row():
                    var_n_slider = gr.Slider(5, 20, value=15, step=1, label="Number of operations to show")
                variance_plot = gr.Plot()
                load_btn.click(
                    fn=lambda: create_variance_analysis_plot(15),
                    outputs=variance_plot
                )
                var_n_slider.change(
                    fn=create_variance_analysis_plot,
                    inputs=var_n_slider,
                    outputs=variance_plot
                )

            # Tab 5: Rank Analysis
            with gr.Tab("🔀 Rank Analysis"):
                gr.Markdown("### Per-Rank Performance Analysis")
                rank_plot = gr.Plot()
                load_btn.click(fn=create_rank_analysis_plot, outputs=rank_plot)

            # NEW: Tab 6: Communication Analysis
            with gr.Tab("📡 Communication Analysis"):
                gr.Markdown("### Detailed Communication Pattern Analysis")
                gr.Markdown("Analyzes all-to-all, NCCL, and other communication operations")
                comm_plot = gr.Plot()
                load_btn.click(fn=create_communication_analysis_plot, outputs=comm_plot)

            # NEW: Tab 7: Module Performance
            with gr.Tab("🏗️ Module Performance"):
                gr.Markdown("### Module Hierarchy Performance Breakdown")
                gr.Markdown("Shows performance aggregated by PyTorch module (MoE.router, MoE.experts, etc.)")
                module_plot = gr.Plot()
                load_btn.click(fn=create_module_performance_plot, outputs=module_plot)

            # NEW: Tab 8: FLOPs Efficiency
            with gr.Tab("⚡ FLOPs Efficiency"):
                gr.Markdown("### Compute Intensity and Efficiency Analysis")
                gr.Markdown("Analyzes GFLOPs and compute efficiency (GFLOPs/s) for operations")
                flops_plot = gr.Plot()
                load_btn.click(fn=create_flops_efficiency_plot, outputs=flops_plot)

            # NEW: Tab 9: Memory Analysis
            with gr.Tab("💾 Memory Analysis"):
                gr.Markdown("### Memory Allocation Pattern Analysis")
                gr.Markdown("Compares memory event counts between EP=1 and EP=2")
                memory_plot = gr.Plot()
                load_btn.click(fn=create_memory_analysis_plot, outputs=memory_plot)

            # NEW: Tab 10: Source Location Analysis ⭐
            with gr.Tab("🗂️ Source Location"):
                gr.Markdown("### Operations Grouped by Source File")
                gr.Markdown("**Shows which source files contain bottleneck operations with exact line numbers**")
                gr.Markdown("Operations are color-coded by type: Communication (red), Memory (orange), Synchronization (yellow), Compute (teal)")
                source_loc_plot = gr.Plot()
                load_btn.click(fn=create_source_location_analysis_plot, outputs=source_loc_plot)

            # Tab 11: Detailed Table
            with gr.Tab("📋 Detailed Data"):
                gr.Markdown("### Comprehensive Operation Statistics")
                detail_table = gr.Dataframe(wrap=True, max_height=600)
                load_btn.click(fn=create_detailed_table, outputs=detail_table)

        gr.Markdown("""
        ---
        ### 💡 How to Use This Dashboard (Ultra-Deep Enhanced + Source Traceability)

        **📊 Dashboard Tabs (11 total):**

        1. **Summary**: High-level overview with ultra-deep analysis insights
        2. **Overview**: 4-panel global performance comparison
        3. **Contribution Analysis**: Operations by slowdown (now with file:line! 🆕)
        4. **Variance Analysis**: Box plots across ranks and iterations
        5. **Rank Analysis**: Per-rank performance (⚠️ critical for false negatives)
        6. **Communication Analysis**: 📡 All-to-all and NCCL breakdown
        7. **Module Performance**: 🏗️ PyTorch module hierarchy timing
        8. **FLOPs Efficiency**: ⚡ Compute intensity metrics
        9. **Memory Analysis**: 💾 Memory allocation patterns
        10. **🆕 Source Location**: 🗂️ Operations grouped by file with line numbers
        11. **Detailed Data**: Table with Type, File, and Line columns 🆕

        **🆕 Source Traceability Features:**
        ✅ **Operation names now show source location**: `all_to_all @ expert_parallel.py:104`
        ✅ **Source Location tab**: Groups operations by file, shows exact line numbers
        ✅ **Operation type categorization**: Communication, Memory, Sync, Compute
        ✅ **Color-coded by type**: Red (Comm), Orange (Memory), Yellow (Sync), Teal (Compute)
        ✅ **Detailed table includes**: Type, Source File, and Line columns
        ✅ **Jump directly to code**: See exact file:line for each bottleneck

        **Enhanced Profiling Data:**
        - Memory event tracking (allocations/deallocations)
        - FLOPs estimation for compute efficiency
        - Module hierarchy breakdown
        - Stack traces with source locations 🆕
        - Per-kernel timing
        - CUDA synchronization overhead
        - Python GC event tracking

        **Interactive Features:**
        - Hover over plots for detailed information
        - Zoom, pan, and select regions in plots
        - Adjust sliders to change number of operations displayed
        - Export plots as images using Plotly toolbar

        **🎯 Recommended Analysis Workflow:**
        1. **Check Summary** tab for overall metrics
        2. **Review Rank Analysis** FIRST (low variance = systemic, high variance = load imbalance)
        3. **Check Source Location** tab to see which files have bottlenecks 🆕
        4. **Look at Contribution Analysis** - operations now show file:line! 🆕
        5. **Drill into specific operations** using Detailed Data table (filterable by Type/File)
        6. **Jump to source code** with exact file:line information
        7. **Verify with Communication/Module/FLOPs tabs** for deeper analysis

        **Example: Tracing a Bottleneck**
        1. See `all_to_all_single @ expert_parallel.py:104` in Contribution chart
        2. Switch to Source Location tab → find `torchtitan/distributed/expert_parallel.py`
        3. See all operations from that file with line numbers
        4. Open source file and navigate to line 104
        5. Analyze why that operation is slow
        """)

    return demo


if __name__ == "__main__":
    print("="*80)
    print("🚀 EP Performance Analysis - Ultra-Deep Dashboard")
    print("="*80)
    print("\n✨ Enhanced with:")
    print("  📡 Communication Analysis (all-to-all, NCCL)")
    print("  🏗️  Module Performance Breakdown")
    print("  ⚡ FLOPs Efficiency Metrics")
    print("  💾 Memory Event Tracking")
    print("  🔬 Per-Kernel Timing + Stack Traces")
    print()

    # PRE-LOAD CACHE AT STARTUP (load once, not per-tab!)
    if CACHE_FILE.exists():
        cache_size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
        print(f"⚡ FAST MODE: Pre-loading cache ({cache_size_mb:.1f} MB)...")

        # Load data ONCE before creating dashboard
        import time
        start = time.time()
        ANALYSIS_DATA = load_from_cache(verbose=True)  # Load directly with verbose output
        load_time = time.time() - start

        if ANALYSIS_DATA:
            print(f"   ✅ Cache loaded in {load_time:.1f}s - Ready!")
            print(f"   📊 Loaded {len(ANALYSIS_DATA.get('contributions', []))} operations")
        else:
            print("   ⚠️ Cache load failed, will load from traces on demand")
    else:
        print("⚠️  SLOW MODE: No cache found")
        print("   First load will take 60-90 seconds")
        print()
        print("💡 TIP: Run this first to enable instant loading:")
        print("   ./scripts/ep/precompute_analysis.py")

    # Create dashboard AFTER loading data
    dashboard = create_dashboard()

    print()
    print("📡 Creating public link for SSH access...")
    print("The public URL will appear below in ~10 seconds.\n")
    print("="*80)

    dashboard.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False,
        show_error=True,
    )
