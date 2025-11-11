# Advanced EP Performance Analysis

## New Features

The advanced analysis provides:

1. **Averaging across all traces** (multiple iterations and ranks)
2. **Profiler step timing comparison** with statistics
3. **Contribution percentage** - shows % of each operation to total slowdown
4. **Rank-level analysis** - identifies if specific ranks are slower
5. **Interactive visualizations**:
   - Box plots showing distribution of operation times
   - Waterfall chart showing contribution to slowdown
   - Rank comparison bar charts

## Quick Start

```bash
# Run advanced analysis (generates tables + interactive plots)
env/bin/python scripts/ep/advanced_analysis.py

# Or use the wrapper
./scripts/ep/run_advanced_analysis.sh
```

Then open the generated HTML files in your browser:
- `scripts/ep/boxplot.html` - Distribution of top operation times
- `scripts/ep/waterfall.html` - Contribution waterfall chart
- `scripts/ep/rank_comparison.html` - Rank-by-rank comparison

## Key Results

### Average Profiler Step Time (Across All Traces)

```
EP=2: 4117.37ms (std: 80.67ms)
EP=1: 3018.89ms (std: 68.12ms)
Difference: +1098.48ms (+36.4%)
```

### Top Contributors to Slowdown

| # | Operation | EP=2 Avg | EP=1 Avg | Diff | **Contrib %** |
|---|-----------|----------|----------|------|---------------|
| 1 | **nccl:all_to_all** | 2,490ms | 0ms | +2,490ms | **226.7%** |
| 2 | **ncclDevKernel_SendRecv** | 2,428ms | 0ms | +2,428ms | **221.0%** |
| 3 | **aten::_to_copy** | 2,445ms | 303ms | +2,142ms | **195.0%** |
| 4 | **aten::to** | 1,583ms | 286ms | +1,296ms | **118.0%** |
| 5 | **cudaMemcpyAsync** | 829ms | 61ms | +768ms | **69.9%** |

**Note:** Contribution % > 100% means operations overlap or the total is computed differently.

### Rank Analysis

All ranks show consistent slowdown (~36.4%):

| Rank | EP=2 Avg Step | EP=1 Avg Step | Difference | % Slower |
|------|---------------|---------------|------------|----------|
| 0 | 4,118ms | 3,019ms | +1,099ms | 36.4% |
| 1 | 4,117ms | 3,018ms | +1,098ms | 36.4% |
| 2 | 4,117ms | 3,019ms | +1,098ms | 36.4% |
| 3 | 4,118ms | 3,019ms | +1,099ms | 36.4% |

**Conclusion:** All ranks are equally affected - not a rank imbalance issue.

## Understanding the Results

### 1. Contribution Percentage

Shows what percentage each operation contributes to the total slowdown:

```
Total Step Diff: +1098ms
aten::_to_copy Diff: +2142ms
Contribution: (2142 / 1098) × 100 = 195%
```

**Why > 100%?** Operations can overlap in time, or some operations are faster in EP=2 (negative contribution), so the sum of positive contributions exceeds 100%.

### 2. Box Plots (boxplot.html)

Shows the **distribution** of operation times across all traces:
- Box = 25th to 75th percentile
- Line in box = median
- Whiskers = min/max (or 1.5×IQR)
- Diamond = mean
- Allows you to see variability and outliers

**Interpretation:**
- Narrow box = consistent timing
- Wide box = high variability
- Compare EP=2 vs EP=1 side-by-side

### 3. Waterfall Chart (waterfall.html)

Shows cumulative contribution of operations to slowdown:
- Each bar represents one operation's contribution
- Bars stack to show cumulative effect
- Percentage labels show relative contribution
- Helps prioritize optimization targets

### 4. Rank Comparison (rank_comparison.html)

Bar chart comparing average step times across ranks:
- Grouped bars (EP=2 vs EP=1) for each rank
- Identifies if specific ranks are slower
- Useful for debugging load imbalance

## Averaging Methodology

### Traces Analyzed

The script analyzes **all available traces**:
- Multiple iterations (e.g., iteration_5, iteration_10)
- Multiple ranks (rank0, rank1, rank2, rank3)
- Total: 8 traces for EP=2, 8 traces for EP=1

### Statistics Computed

For each operation:
- **Mean** - average time across all traces
- **Std Dev** - standard deviation (variability)
- **Min/Max** - extremes
- **Median** - middle value

### Profiler Step Timing

Extracts timing from `ProfilerStep#N` events:
- Each step represents one training iteration
- Compares average step duration across all traces
- Provides overall slowdown percentage

## Advanced Usage

### Analyzing Specific Ranks

Edit `advanced_analysis.py` to filter ranks:

```python
# Line ~115, in analyze_all_traces
for trace_file in trace_files:
    # Add filter
    if 'rank0' not in trace_file:
        continue
    # ... rest of code
```

### Analyzing Specific Operations

To focus on specific operation patterns:

```python
# Line ~280, in compute_contribution_analysis
contributions = []
for op_name in set(ep2_summary.keys()) | set(ep1_summary.keys()):
    # Add filter
    if 'nccl' not in op_name.lower():
        continue
    # ... rest of code
```

### Exporting Data for External Analysis

The script stores all raw data. You can export it:

```python
# Add at end of main()
import json
with open('ep_analysis_data.json', 'w') as f:
    json.dump({
        'ep2_summary': ep2_summary,
        'ep1_summary': ep1_summary,
        'contributions': contributions,
        'rank_diffs': rank_diffs
    }, f, indent=2)
```

## Troubleshooting

### "No trace files found"

Ensure you have profiling data:
```bash
ls -la outputs_profile_ep*/profile_trace/*/rank*.json
```

If missing, run profiling:
```bash
./scripts/ep/run_profiling.sh both
```

### "Plotly not available"

Install plotly:
```bash
env/bin/pip install plotly kaleido
```

### Plots not opening

Plots are HTML files. Open them with:
```bash
# On Linux
xdg-open scripts/ep/boxplot.html

# On macOS
open scripts/ep/boxplot.html

# Or copy to local machine and open in browser
```

### Memory issues

If analysis fails due to memory:
1. Reduce number of traces analyzed
2. Process one rank at a time
3. Skip plot generation (comment out create_*_plot calls)

## Key Insights from Advanced Analysis

### 1. All-to-All Dominates

`nccl:all_to_all` + `ncclDevKernel_SendRecv` = **4,918ms overhead**
- Represents 448% of total slowdown
- This is the core EP communication overhead
- **Not easily optimizable** - inherent to EP design

### 2. Memory Transfer Is Second Biggest

`aten::_to_copy` + `cudaMemcpyAsync` = **2,910ms overhead**
- Represents 265% of total slowdown
- **This CAN be optimized** - see proposed fix in EP_PERFORMANCE_ANALYSIS.md
- Line 104 in expert_parallel.py

### 3. No Rank Imbalance

All ranks show identical slowdown (±0.5ms variance)
- Problem is not load imbalance
- Problem is systemic to EP implementation

### 4. High Consistency

Low standard deviations indicate:
- Results are reproducible
- No significant performance variability
- Profiling data is reliable

## Next Steps

1. **Apply the fix** in `expert_parallel.py:104`
   ```python
   .to(torch.device("cpu"), non_blocking=True)
   ```

2. **Re-run profiling** after the fix:
   ```bash
   ./scripts/ep/run_profiling.sh both
   env/bin/python scripts/ep/advanced_analysis.py
   ```

3. **Compare before/after** using the plots

4. **Expected improvement:** 20-30% throughput increase

## References

- Basic analysis: `./scripts/ep/compare_profiles.sh`
- Technical report: `scripts/ep/EP_PERFORMANCE_ANALYSIS.md`
- Source code: `scripts/ep/advanced_analysis.py`
