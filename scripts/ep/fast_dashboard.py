#!/usr/bin/env python3
"""
Fast Interactive Dashboard - Pre-loads data at startup, shows progress
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

import gradio as gr

print("=" * 80)
print("🚀 FAST EP Performance Analysis Dashboard")
print("=" * 80)
print("\n📊 Loading analysis data (this takes ~60-90 seconds for 2.4GB of traces)...")
print("Progress:")

# Change to repo root
os.chdir(SCRIPT_DIR.parent.parent)

# Import and run analysis
print("  [1/4] Importing analysis modules...")
from interactive_dashboard import (
    load_analysis_data,
    create_overview_plot,
    create_detailed_contribution_plot,
    create_contribution_pie_chart,
    create_variance_analysis_plot,
    create_rank_analysis_plot,
    create_detailed_table,
    create_summary_text,
)

print("  [2/4] Loading EP=2 traces (8 files, ~1.3GB)...")
print("  [3/4] Loading EP=1 traces (8 files, ~1.1GB)...")
print("  [4/4] Computing statistics and filtering operations...")

# Pre-load data
DATA = load_analysis_data()

if DATA:
    # Filter out ProfilerStep from contributions
    print("  [5/5] Filtering out profiler overhead (keeping only real operations)...")
    original_count = len(DATA['contributions'])
    DATA['contributions'] = [
        c for c in DATA['contributions']
        if 'ProfilerStep' not in c['op_name'] and 'profiler' not in c['op_name'].lower()
    ]
    filtered_count = len(DATA['contributions'])
    print(f"         Filtered: {original_count} → {filtered_count} operations (removed {original_count - filtered_count} profiler overhead entries)")

    # Normalize contribution percentages to sum to 100% for positive diffs
    print("  [6/6] Normalizing contribution percentages...")
    total_positive_diff = sum(c['diff_ms'] for c in DATA['contributions'] if c['diff_ms'] > 0)
    for c in DATA['contributions']:
        c['normalized_pct'] = (c['diff_ms'] / total_positive_diff * 100) if total_positive_diff > 0 and c['diff_ms'] > 0 else 0
    print(f"         Top 10 ops now sum to: {sum(c['normalized_pct'] for c in DATA['contributions'][:10]):.1f}%")

if DATA is None:
    print("\n❌ ERROR: Could not load profiling data!")
    print("Please run: ./scripts/ep/run_profiling.sh both")
    sys.exit(1)

print("\n✅ Data loaded successfully!")
print(f"   - EP=2 traces: {len(DATA['ep2_data']['traces'])}")
print(f"   - EP=1 traces: {len(DATA['ep1_data']['traces'])}")
print(f"   - Operations analyzed: {len(DATA['contributions'])}")
print(f"   - Average slowdown: +{DATA['total_step_diff']:.1f}ms (+{(DATA['total_step_diff']/DATA['avg_ep1_step']*100):.1f}%)")
print("\n" + "=" * 80)
print("🌐 Creating web interface...")


def get_cached_data():
    """Return pre-loaded data"""
    return DATA


# Monkey-patch the get_data function to use pre-loaded data
import interactive_dashboard
interactive_dashboard.get_data = get_cached_data
interactive_dashboard.ANALYSIS_DATA = DATA


def create_dashboard():
    """Create dashboard with pre-loaded data"""

    with gr.Blocks(title="EP Performance Analysis (Fast)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Expert Parallelism Performance Analysis - Fast Dashboard")
        gr.Markdown(f"**Data pre-loaded**: {len(DATA['ep2_data']['traces']) + len(DATA['ep1_data']['traces'])} traces analyzed")

        with gr.Tabs():
            # Tab 1: Summary
            with gr.Tab("📊 Summary"):
                gr.Markdown(create_summary_text())

            # Tab 2: Overview
            with gr.Tab("📈 Overview"):
                gr.Markdown("### Global Performance Overview")
                gr.Plot(create_overview_plot())

            # Tab 3: Contribution Analysis
            with gr.Tab("🎯 Contribution Analysis"):
                gr.Markdown("""
                ### Operation-Level Contribution Analysis

                **Bar Chart**: Shows time difference (ms) with contribution % displayed.
                - Bar color intensity = contribution percentage
                - Hover to see full details
                - Percentages show contribution to total step overhead (global normalization)
                - Sum of displayed ops shown in chart title
                """)

                top_n = gr.Slider(5, 50, value=15, step=1, label="Number of operations to show (bar chart)")
                contrib_plot = gr.Plot(create_detailed_contribution_plot(15))

                top_n.change(
                    fn=create_detailed_contribution_plot,
                    inputs=top_n,
                    outputs=contrib_plot
                )

                gr.Markdown("""
                ---
                ### Pie Chart: Contribution Breakdown

                Visual breakdown showing contribution to total step overhead (global normalization).
                - Each operation shows its % of total step overhead
                - "Others" represents remaining operations
                - Sum of displayed ops shown in chart center
                """)

                pie_n = gr.Slider(5, 50, value=12, step=1, label="Number of operations to show (pie chart)")
                pie_plot = gr.Plot(create_contribution_pie_chart(12))

                pie_n.change(
                    fn=create_contribution_pie_chart,
                    inputs=pie_n,
                    outputs=pie_plot
                )

            # Tab 4: Variance Analysis
            with gr.Tab("📊 Variance Analysis"):
                gr.Markdown("""
                ### Box Plots - Variance Across Ranks and Iterations

                Shows the distribution of operation times across all profiling runs.
                - **Box**: 25th-75th percentile
                - **Line in box**: Median
                - **Diamond**: Mean
                - **Whiskers**: Min/Max range

                Narrow boxes = consistent performance. Wide boxes = high variance.
                """)
                var_n = gr.Slider(5, 10, value=8, step=1, label="Number of operations to show")
                var_plot = gr.Plot(create_variance_analysis_plot(8))

                var_n.change(
                    fn=create_variance_analysis_plot,
                    inputs=var_n,
                    outputs=var_plot
                )

            # Tab 5: Rank Analysis
            with gr.Tab("🔀 Rank Analysis"):
                gr.Markdown("### Per-Rank Performance Analysis")
                gr.Plot(create_rank_analysis_plot())

            # Tab 6: Detailed Table
            with gr.Tab("📋 Detailed Data"):
                gr.Markdown("### Comprehensive Operation Statistics (Top 50)")
                gr.Dataframe(create_detailed_table(), wrap=True, max_height=600)

        gr.Markdown("""
        ---
        ### 💡 Dashboard Features

        **Interactive Plotly Graphs:**
        - **Zoom**: Click and drag on plot
        - **Pan**: Shift + drag
        - **Hover**: See exact values
        - **Export**: Camera icon to save as PNG

        **Key Insights:**
        - **Summary Tab**: High-level findings and recommendations
        - **Contribution Analysis**: Shows % each operation contributes to slowdown
        - **Variance Analysis**: Box plots show consistency across ranks/iterations
        - **Rank Analysis**: Identifies if specific ranks are slower

        **All data is pre-loaded** - switching tabs and adjusting sliders is instant!
        """)

    return demo


if __name__ == "__main__":
    dashboard = create_dashboard()

    print("✅ Dashboard ready!")
    print("\n📡 Creating public link...")
    print("=" * 80)

    dashboard.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False,
        show_error=True,
    )
