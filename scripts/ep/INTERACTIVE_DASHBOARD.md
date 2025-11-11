# 🎯 Interactive Dashboard - Complete Guide

## Overview

The **Interactive Dashboard** is a comprehensive web-based interface for analyzing EP performance with real-time Plotly visualizations. It provides everything you need to identify bottlenecks, understand variance, and pinpoint the root cause of performance issues.

## 🚀 Launch the Dashboard

```bash
./scripts/ep/START_DASHBOARD.sh
```

After ~10 seconds, you'll see:
```
Running on public URL: https://xxxxx.gradio.live
```

**Copy that URL** and open it in your local browser! (Works from SSH sessions)

## ✨ Key Features

### 1. **Real-Time Interactive Plotly Graphs**
- **Zoom**: Click and drag to zoom into specific regions
- **Pan**: Shift + drag to pan around
- **Hover**: Get detailed information for each data point
- **Export**: Save plots as PNG images
- **Responsive**: Graphs update instantly when you adjust sliders

### 2. **Global View with Detail**
Six comprehensive tabs provide both overview and deep-dive analysis:

#### 📊 Tab 1: Summary
- High-level performance metrics
- Top 5 bottlenecks with contribution percentages
- Rank-by-rank slowdown analysis
- Key insights and recommendations
- Variance analysis across ranks

**What You'll See:**
- Average step times: EP=2 vs EP=1
- Total slowdown in ms and percentage
- Number of traces analyzed
- Whether load imbalance exists

#### 📈 Tab 2: Overview
Four key visualizations in one view:

1. **Average Step Time Comparison** (Bar Chart)
   - Direct comparison of EP=2 vs EP=1
   - Shows exact times in milliseconds

2. **Step Time Distribution** (Box Plots)
   - Shows min, max, median, mean, and quartiles
   - Variance across all profiling steps
   - Identifies consistency of performance

3. **Top 10 Bottlenecks by Contribution %** (Horizontal Bar)
   - Operations sorted by impact on slowdown
   - Percentage contribution clearly labeled

4. **Rank-Level Comparison** (Grouped Bars)
   - EP=2 vs EP=1 for each rank
   - Identifies if specific ranks are slower

#### 🎯 Tab 3: Contribution Analysis
**Most Important Tab for Identifying Bottlenecks**

Two side-by-side visualizations:

1. **Time Difference** (Left)
   - Absolute time difference in ms (EP=2 - EP=1)
   - Color-coded by magnitude
   - Sorted by impact
   - Adjustable: Show top 10-50 operations (slider)

2. **Contribution Percentage** (Right)
   - Shows what % each operation contributes to overall slowdown
   - Formula: `(operation_diff / total_step_diff) × 100`
   - **>100% possible**: Operations can overlap in time
   - Helps prioritize optimization efforts

**Interactive Features:**
- Slider to adjust number of operations (10-50)
- Hover for exact values
- Click legend to hide/show series

#### 📊 Tab 4: Variance Analysis
**Box Plots Showing Consistency Across Ranks and Iterations**

- One box plot per operation (top 15 by default)
- Side-by-side comparison: EP=2 vs EP=1
- Each box shows:
  - **25th-75th percentile** (box)
  - **Median** (line in box)
  - **Mean** (diamond marker)
  - **Min/Max** (whiskers)
  - **Standard deviation** (visible when hovering)

**What It Tells You:**
- **Narrow boxes** = Consistent performance across ranks/iterations
- **Wide boxes** = High variability (potential issue)
- **Outliers** = Specific ranks or iterations behaving differently
- **EP=2 wider than EP=1** = EP introduces variability

**Interactive Features:**
- Slider: Show top 5-20 operations
- Zoom into specific operation
- See exact values on hover

#### 🔀 Tab 5: Rank Analysis
**Detailed Per-Rank Performance Investigation**

Four visualizations:

1. **Average Step Time by Rank** (Grouped Bars)
   - EP=2 vs EP=1 for each rank
   - Quickly spot if one rank is slower

2. **Step Time Distribution by Rank - EP=2** (Box Plots)
   - Variance within each rank for EP=2
   - Identifies if specific ranks have high variance

3. **Step Time Distribution by Rank - EP=1** (Box Plots)
   - Variance within each rank for EP=1
   - Baseline comparison

4. **Slowdown % by Rank** (Bar Chart)
   - Percentage slowdown for each rank
   - **Consistent percentages** = Global issue (not load imbalance)
   - **Variable percentages** = Specific ranks affected more

**Diagnostic Value:**
- If all ranks show ~36% slowdown → **Systemic issue** (as in our case)
- If ranks vary significantly → **Load imbalance** or **specific rank bottleneck**

#### 📋 Tab 6: Detailed Data
**Comprehensive Statistics Table**

Shows top 50 operations with:
- Operation name
- EP=2 Mean (ms)
- EP=1 Mean (ms)
- Difference (ms)
- **Contribution %** ← Key metric
- EP=2 Std Dev (ms)
- EP=1 Std Dev (ms)
- EP=2 Min (ms)
- EP=2 Max (ms)
- EP=1 Min (ms)
- EP=1 Max (ms)
- EP=2 Count (average across traces)
- EP=1 Count (average across traces)

**Use Cases:**
- Export data for external analysis
- Sort by any column
- Filter operations by name
- See complete statistics at a glance

## 🔬 How to Use for Root Cause Analysis

### Step 1: Load Data
1. Click "Load/Refresh Analysis Data" button
2. Wait 5-10 seconds for analysis to complete
3. Status message will confirm success

### Step 2: Quick Assessment (Summary Tab)
- Check overall slowdown percentage
- Review top 5 bottlenecks
- Note contribution percentages
- See if ranks are equally affected

### Step 3: Identify Top Bottlenecks (Contribution Tab)
- Adjust slider to show top 20-30 operations
- Look for operations with highest contribution %
- Separate:
  - **Inherent overhead** (e.g., `nccl:all_to_all` at 226%)
  - **Optimizable overhead** (e.g., `aten::_to_copy` at 195%)

### Step 4: Check Variance (Variance Tab)
- Look at box plots for top bottlenecks
- Identify if variance is consistent or problematic
- Wide boxes = investigate further
- Outliers = specific rank/iteration issue

### Step 5: Verify Load Balance (Rank Analysis Tab)
- Check if all ranks show similar slowdown %
- If yes → Global issue (optimize bottlenecks from Step 3)
- If no → Investigate specific ranks

### Step 6: Deep Dive (Detailed Data Tab)
- Export table for detailed analysis
- Cross-reference min/max values with box plots
- Look at operation counts (more calls = cumulative impact)

## 📊 Understanding Contribution Percentage

**Formula:**
```
Contribution % = (Operation_Diff_ms / Total_Step_Diff_ms) × 100
```

**Example from our data:**
- Total step difference: +1098ms
- `nccl:all_to_all` difference: +2490ms
- Contribution: (2490 / 1098) × 100 = **226.7%**

**Why > 100%?**
1. Operations can **overlap in time** (run in parallel)
2. Some operations are **faster in EP=2** (negative contribution)
3. The sum of positive contributions exceeds 100%

**How to use it:**
- **>100%**: Major bottleneck, but may include overlapped time
- **50-100%**: Significant contributor
- **20-50%**: Moderate contributor
- **<20%**: Minor contributor

**Prioritization:**
Focus on operations with:
1. High contribution % **AND**
2. Known optimization opportunities (e.g., memory transfers)

## 🎨 Interactive Features

### Plotly Toolbar (Top Right of Each Plot)
- 📷 **Camera**: Download plot as PNG
- 🔍 **Zoom**: Click and drag to zoom
- 📐 **Pan**: Move around zoomed area
- 📊 **Box Select**: Select specific data points
- 🏠 **Home**: Reset to original view
- ⚙️ **Autoscale**: Fit data to axes

### Sliders
- **Contribution Analysis**: 10-50 operations
- **Variance Analysis**: 5-20 operations
- Real-time updates (no page reload)

### Hover Information
Each plot shows different info on hover:
- **Bar charts**: Operation name, value
- **Box plots**: Min, Q1, median, Q3, max, mean, std dev
- **Interactive legends**: Click to hide/show series

## 🔧 Troubleshooting

### Dashboard shows "No data available"
**Fix:**
```bash
# Run profiling first
./scripts/ep/run_profiling.sh both

# Then launch dashboard
./scripts/ep/START_DASHBOARD.sh
```

### Plots are blank after clicking "Load/Refresh"
**Fix:**
1. Check that profiling generated trace files:
   ```bash
   ls -la outputs_profile_ep*/profile_trace/iteration_5/
   ```
2. If missing, run profiling again

### Dashboard is slow to load
**Cause:** Loading and processing all trace data
**Solution:** This is normal for first load (~10-15 seconds)
**Subsequent operations** (changing sliders, switching tabs) are instant

### Public URL expired
**Solution:** Restart the dashboard to get a new URL:
```bash
# Stop dashboard (Ctrl+C in terminal where it's running)
# Or kill process:
pkill -f "interactive_dashboard.py"

# Restart:
./scripts/ep/START_DASHBOARD.sh
```

### Plots don't update when I change sliders
**Fix:** Click "Load/Refresh Analysis Data" first

## 🎯 Real Example: Our EP Analysis

### What the Dashboard Revealed:

1. **Summary Tab**:
   - EP=2: 4117ms (±81ms)
   - EP=1: 3019ms (±68ms)
   - Slowdown: +1098ms (+36.4%)

2. **Contribution Tab**:
   - `nccl:all_to_all`: **226.7%** → Inherent to EP (communication)
   - `aten::_to_copy`: **195.0%** → Optimizable (memory transfer)
   - `cudaMemcpyAsync`: **69.9%** → Optimizable (memory transfer)

3. **Variance Tab**:
   - Low variance across ranks/iterations
   - Consistent performance
   - Results are reproducible

4. **Rank Analysis Tab**:
   - All ranks: ~36.4% slowdown
   - **Conclusion**: Not a load imbalance issue
   - **Conclusion**: Systemic bottleneck in EP implementation

5. **Root Cause**:
   - Synchronous device-to-host transfer at `expert_parallel.py:104`
   - `non_blocking=False` causing blocking memcpy
   - Fix: Change to `non_blocking=True`
   - Expected improvement: 20-30%

## 📈 Advanced Usage

### Export Data for External Analysis

From the **Detailed Data Tab**:
1. View the comprehensive table
2. Copy-paste into Excel/Google Sheets
3. Or use browser's "Inspect Element" to extract table HTML

### Compare Multiple Configurations

To compare different EP degrees:
1. Run profiling with different configs
2. Update trace file paths in `interactive_dashboard.py`
3. Relaunch dashboard

### Custom Analysis

Modify `interactive_dashboard.py` to:
- Filter specific operation patterns
- Add custom visualizations
- Change color schemes
- Adjust plot sizes

## 🎓 Best Practices

1. **Always load data first**: Click "Load/Refresh" when you open the dashboard
2. **Start with Summary**: Get high-level understanding
3. **Use Contribution tab**: Identify real bottlenecks
4. **Check variance**: Ensure results are reliable
5. **Verify with Rank Analysis**: Rule out load imbalance
6. **Cross-reference tabs**: Consistent story across views

## 🌐 SSH Access

The public Gradio URL works from anywhere:
- No VPN needed
- No port forwarding needed
- Valid for 72 hours
- Can share with team (if data not sensitive)

**Alternative**: SSH port forwarding
```bash
# On local machine:
ssh -L 7860:localhost:7860 user@remote-server

# Then access:
http://localhost:7860
```

## 📚 Additional Resources

- **Basic Analysis**: `./scripts/ep/compare_profiles.sh`
- **Technical Report**: `scripts/ep/EP_PERFORMANCE_ANALYSIS.md`
- **Advanced Features**: `scripts/ep/ADVANCED_ANALYSIS.md`
- **Quick Reference**: `scripts/ep/COMMANDS.txt`

## 🎉 Summary

The Interactive Dashboard provides:

✅ **Real-time interactive Plotly graphs** (not static HTML)
✅ **Global view** (overview) and **detailed view** (per-operation)
✅ **Variance analysis** across ranks and iterations (box plots)
✅ **Contribution percentages** for identifying real bottlenecks
✅ **Rank-level comparison** for load imbalance detection
✅ **Comprehensive operation logs** with all statistics
✅ **Interactive sliders** for adjusting display
✅ **Public URL** accessible from local browser via SSH

**Result**: Complete visibility into EP performance with ability to identify, understand, and prioritize optimization opportunities.
