# Dashboard Merge Complete - Ultra-Deep Analysis Integrated

## Date: November 11, 2025

## Summary

Successfully merged all ultra-deep analysis features from `ultra_deep_dashboard.py` into the existing `interactive_dashboard.py`, creating a single comprehensive dashboard with enhanced profiling capabilities.

---

## What Was Changed

### 1. Backend Analysis (advanced_analysis.py)

**Enhanced trace parsing** (`analyze_trace_with_profiler_steps`):
- Extracts memory events (allocations/deallocations)
- Identifies communication events (all-to-all, NCCL)
- Tracks CUDA synchronization events
- Records Python GC events
- Captures FLOPs data per operation
- Extracts module hierarchy information
- Samples stack traces

**New helper functions added**:
- `analyze_communication_patterns(ep2_data, ep1_data)` - All-to-all overhead breakdown
- `analyze_memory_patterns(ep2_data, ep1_data)` - Memory allocation analysis
- `analyze_module_performance(ep2_summary, ep1_summary)` - MoE component-level timing

### 2. Frontend Dashboard (interactive_dashboard.py)

**Updated imports**:
```python
from advanced_analysis import (
    analyze_all_traces,
    aggregate_statistics,
    compute_contribution_analysis,
    analyze_rank_differences,
    analyze_communication_patterns,  # NEW
    analyze_memory_patterns,         # NEW
    analyze_module_performance,      # NEW
)
```

**Enhanced data loading** (`load_analysis_data`):
- Calls new analysis functions
- Returns ultra-deep analysis results (comm_analysis, memory_analysis, module_analysis)

**New plotting functions**:
1. `create_communication_analysis_plot()` - 4-panel communication breakdown:
   - Communication time comparison
   - Operation count comparison
   - Average latency per call
   - Total communication time

2. `create_module_performance_plot()` - 2-panel module analysis:
   - Module time comparison (EP=1 vs EP=2)
   - Time difference by module

3. `create_flops_efficiency_plot()` - 2-panel compute efficiency:
   - Compute intensity (GFLOPs)
   - Compute efficiency (GFLOPs/s)

4. `create_memory_analysis_plot()` - Simple memory event comparison:
   - Bar chart comparing memory event counts
   - Shows percentage increase

**Enhanced summary** (`create_summary_text`):
- Added "Ultra-Deep Analysis Results" section
- Communication overhead analysis with percentage contribution
- Memory event analysis with increase percentage
- Top 5 slowest modules breakdown

**New Gradio tabs**:
- Tab 6: 📡 Communication Analysis
- Tab 7: 🏗️ Module Performance
- Tab 8: ⚡ FLOPs Efficiency
- Tab 9: 💾 Memory Analysis

**Updated UI elements**:
- Dashboard title: "Expert Parallelism Performance Analysis - Ultra-Deep Dashboard"
- Enhanced "How to Use" section with workflow guidance
- Startup banner showing enhanced features

---

## Dashboard Features (11 Tabs Total)

### Original Tabs (Enhanced)
1. **📊 Summary** - Now includes ultra-deep analysis insights
2. **📈 Overview** - 4-panel global comparison
3. **🎯 Contribution Analysis** - Top operations by slowdown
4. **📊 Variance Analysis** - Box plots across ranks/iterations
5. **🔀 Rank Analysis** - Per-rank performance (critical for false negatives)

### New Ultra-Deep Tabs
6. **📡 Communication Analysis** - All-to-all and NCCL breakdown
7. **🏗️ Module Performance** - PyTorch module hierarchy timing
8. **⚡ FLOPs Efficiency** - Compute intensity metrics
9. **💾 Memory Analysis** - Memory event comparison

### Data Tab
10. **📋 Detailed Data** - Comprehensive statistics table

---

## Enhanced Profiling Data Available

Thanks to the enhanced PyTorch profiler configuration in `torchtitan/tools/profiling.py`:

1. **Memory Profiling** (`profile_memory=True`)
   - CUDA memory allocations and deallocations
   - Memory timeline visualization

2. **Stack Traces** (`with_stack=True`)
   - Full Python call stacks
   - Source file and line number tracking

3. **FLOPs Tracking** (`with_flops=True`)
   - Floating-point operations per kernel
   - Compute efficiency metrics

4. **Module Hierarchy** (`with_modules=True`)
   - Groups operations by PyTorch module
   - MoE component-level breakdown

5. **Per-Kernel Timing** (`profiler_measure_per_kernel=True`)
   - Detailed per-CUDA-kernel performance
   - Kernel launch overhead

6. **CUDA Synchronization** (`enable_cuda_sync_events=True`)
   - Tracks synchronization overhead
   - Better async operation timing

7. **Python GC Info** (`record_python_gc_info=True`)
   - Garbage collection events
   - Memory fragmentation analysis

---

## Key Analysis Workflow

When using the ultra-deep dashboard:

1. **Load Data**: Click "Load/Refresh Analysis Data"

2. **Check Summary Tab First**:
   - Review overall performance metrics
   - Identify top 5 bottlenecks
   - Read ultra-deep analysis insights

3. **Review Rank Analysis Tab** ⚠️ **CRITICAL**:
   - Check slowdown variance across ranks
   - **Low variance (<2%)**: Uniform slowdown → systemic EP overhead (NOT load imbalance)
   - **High variance (>5%)**: Uneven slowdown → load imbalance issue

4. **Check Communication Analysis**:
   - Look for all-to-all operations
   - Calculate % contribution to slowdown
   - If >50% → Communication-bound (inherent to EP)

5. **Review Module Performance**:
   - See which MoE component is slowest
   - Examples: MoE.router, MoE.experts, MoE.reorderer

6. **Check FLOPs Efficiency**:
   - Determine if operations are compute-bound or memory-bound
   - Low GFLOPs/s → Memory-bound (optimize memory access)
   - High GFLOPs/s → Compute-bound (good GPU utilization)

7. **Review Memory Analysis**:
   - Check memory event increase
   - >30% increase → Memory allocation overhead

---

## Comparison: Before vs After

| Feature | Before | After (Ultra-Deep) |
|---------|--------|-------------------|
| **Tabs** | 6 tabs | 10 tabs |
| **Profiling Data Types** | Basic timing | 7 data types (memory, FLOPs, modules, stacks, per-kernel, CUDA sync, GC) |
| **Communication Analysis** | None | Detailed 4-panel breakdown |
| **Module Breakdown** | None | Full hierarchy analysis |
| **FLOPs Tracking** | None | Compute intensity + efficiency |
| **Memory Events** | None | Full tracking + comparison |
| **Summary Insights** | Basic | Ultra-deep with recommendations |

---

## Profiling Results Available

Both EP=1 and EP=2 profiling runs completed successfully:

**EP=2 Performance:**
- Average TPS: ~5,872-5,964 tokens/second
- Memory usage: 174.89 GiB (98.06%)
- Traces: `outputs_profile_ep2/profile_trace/`

**EP=1 Performance:**
- Average TPS: ~7,858-8,211 tokens/second
- Memory usage: 175.98 GiB (98.66%)
- Traces: `outputs_profile_ep1/profile_trace/`

**Initial Finding:** EP=1 is **~35-40% faster** than EP=2

---

## How to Launch the Dashboard

```bash
cd /home/phuc/workspace/moe/reference_repos/torchtitan-nous
./scripts/ep/interactive_dashboard.py
```

The dashboard will:
1. Auto-generate a public Gradio link (for SSH access)
2. Parse all enhanced profiling data
3. Generate interactive Plotly visualizations
4. Display comprehensive analysis across 10 tabs

---

## Expected Bottlenecks to Identify

Based on MoE architecture, common bottlenecks:

1. **All-to-all communication** (EP=2 only)
   - Token shuffle between EP ranks
   - Inherent cost of EP

2. **Memory transfers** (`aten::_to_copy`, `cudaMemcpyAsync`)
   - Blocking CPU tensor copies
   - **Fix**: `non_blocking=True` in `expert_parallel.py:104`

3. **Token reordering** (`torch::histc`, `torch::argsort`)
   - Called twice (router + reorderer)
   - **Optimization**: Single histc call

4. **Expert computation imbalance** (if variance >5%)
   - Uneven expert load
   - **Fix**: Improve load balancing

5. **Memory allocation overhead** (if >30% increase)
   - Frequent alloc/free in EP=2
   - **Fix**: Memory pooling

---

## Files Modified

1. `/home/phuc/workspace/moe/reference_repos/torchtitan-nous/scripts/ep/advanced_analysis.py`
   - Enhanced trace parsing
   - Added 3 new analysis functions
   - Updated aggregate_statistics

2. `/home/phuc/workspace/moe/reference_repos/torchtitan-nous/scripts/ep/interactive_dashboard.py`
   - Added 4 new plotting functions
   - Enhanced summary with ultra-deep insights
   - Added 4 new Gradio tabs
   - Updated UI elements and documentation

3. `/home/phuc/workspace/moe/reference_repos/torchtitan-nous/torchtitan/tools/profiling.py`
   - Enhanced PyTorch profiler configuration (done earlier)

---

## Key Advantages of Merged Dashboard

1. **Single Tool**: No need to switch between dashboards
2. **Comprehensive**: All analysis in one place
3. **Progressive Disclosure**: Start with summary, drill down as needed
4. **Enhanced Data**: Leverages all 7 profiling data types
5. **Workflow Guidance**: Clear step-by-step analysis path
6. **False Negative Protection**: Rank variance analysis prevents misdiagnosis

---

## Next Steps

1. **Launch the dashboard**:
   ```bash
   ./scripts/ep/interactive_dashboard.py
   ```

2. **Load the data** using the "Load/Refresh Analysis Data" button

3. **Follow the analysis workflow** (Summary → Rank Analysis → Communication → Module → FLOPs)

4. **Identify top 3 bottlenecks**

5. **Implement fixes** based on recommendations

6. **Re-profile** to measure improvement

---

## Success Metrics

✅ **Dashboard merge complete**: All features integrated into single tool
✅ **Enhanced profiling enabled**: 7 data types captured
✅ **Profiling runs successful**: Both EP=1 and EP=2 completed
✅ **Analysis tools ready**: 10 tabs with comprehensive visualizations
✅ **Workflow documented**: Clear step-by-step guidance provided

**Status**: Ready for analysis! 🚀

---

**Last Updated**: November 11, 2025
**Merge Completed By**: Claude Code (Sonnet 4.5)
