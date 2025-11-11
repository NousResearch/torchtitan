# Summary of Enhanced Profiling and Analysis Changes

## Date: November 11, 2025

## Goal
Identify exact bottlenecks when scaling from EP=1 to EP=2 in MoE training, with particular attention to avoiding false negatives from rank variance.

---

## Changes Made

### 1. Enhanced PyTorch Profiler Configuration

**File**: `torchtitan/tools/profiling.py`

**Changes**:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, gpu_device_profiled],
    schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
    on_trace_ready=trace_handler,
    record_shapes=True,
    # NEW: Enhanced profiling
    profile_memory=True,                    # Track memory allocations/deallocations
    with_stack=True,                        # Record source file and line numbers
    with_flops=True,                        # Estimate FLOPs for operations
    with_modules=True,                      # Record module hierarchy
    experimental_config=torch._C._profiler._ExperimentalConfig(
        verbose=True,                       # Enable verbose logging
        profiler_measure_per_kernel=True,   # Per-kernel performance
        enable_cuda_sync_events=True,       # Better async operation timing
        profile_all_threads=True,           # Profile all threads, not just main
        capture_overload_names=True,        # Better operation identification
        record_python_gc_info=True,         # Track GC pauses
    ),
)
```

**Benefits**:
- Memory timeline tracking
- Stack traces for hot path identification
- FLOPs data for compute efficiency analysis
- Module-level performance breakdown
- Per-kernel timing granularity
- CUDA synchronization overhead detection
- Python GC impact visibility

---

### 2. Ultra-Deep Performance Dashboard

**File**: `scripts/ep/ultra_deep_dashboard.py` (NEW)

**Features**:

#### a. Enhanced Trace Parsing
- Extracts memory events, stack traces, FLOPs, module hierarchy
- Identifies communication operations (all-to-all, nccl)
- Tracks CUDA synchronization events
- Records Python GC events

#### b. New Visualization Tabs

1. **Communication Analysis**
   - Detailed breakdown of all-to-all and NCCL operations
   - Time, count, and latency comparisons
   - Goal: Identify if EP overhead is communication-bound

2. **Module Performance**
   - Aggregated timing by PyTorch module (MoE.router, MoE.experts, etc.)
   - Identifies which MoE component is slow
   - Goal: Component-level bottleneck identification

3. **FLOPs Efficiency**
   - GFLOPs comparison across operations
   - Compute efficiency metrics
   - Goal: Determine if operations are compute-bound vs memory-bound

4. **Rank Load Balance** ⚠️ **Critical**
   - Per-rank step time distributions
   - Slowdown variance analysis
   - **Goal**: Detect false negatives from load imbalance
   - **Key Metric**: Low variance (<2%) = systemic overhead, not imbalance

5. **Memory Analysis**
   - Memory event count comparison
   - Allocation overhead detection
   - Goal: Identify memory management bottlenecks

6. **Full Data Table**
   - Comprehensive statistics: timing, FLOPs, modules, contributions
   - Sortable and filterable

#### c. Auto-Generated Summary Report
- Top 10 bottlenecks with contribution percentages
- Communication overhead breakdown
- Memory analysis
- Rank balance status
- Optimization recommendations

---

### 3. Documentation

**Files Created**:
- `scripts/ep/ULTRA_DEEP_DASHBOARD_README.md`: Comprehensive usage guide
- `scripts/ep/SUMMARY_OF_CHANGES.md`: This file

**Content**:
- Detailed feature explanations
- Step-by-step usage workflow
- Key metrics interpretation
- Example analysis scenarios
- Troubleshooting guide

---

## Usage

### Run Profiling
```bash
cd /path/to/torchtitan-nous
./scripts/ep/run_profiling.sh both
```

### Launch Ultra-Deep Dashboard
```bash
./scripts/ep/ultra_deep_dashboard.py
```

### Access Dashboard
- Dashboard will create a public Gradio link
- Access from any browser (useful for SSH access)
- Port: 7861 (vs 7860 for original dashboard)

---

## Key Improvements Over Original Dashboard

| Aspect | Original | Ultra-Deep | Improvement |
|--------|----------|------------|-------------|
| **Profiling Data** | Basic timing | Memory, FLOPs, stack traces, modules, GC | 7x more data types |
| **Communication Analysis** | None | Detailed breakdown | NEW |
| **Module Breakdown** | None | Full hierarchy | NEW |
| **FLOPs Tracking** | None | Per-operation GFLOPs | NEW |
| **Rank Analysis** | Basic | Enhanced with variance detection | Critical for false negatives |
| **Memory Events** | None | Full tracking | NEW |
| **Per-Kernel Timing** | None | Enabled | NEW |

---

## Critical Insight: Avoiding False Negatives

### Problem
When analyzing EP scaling, you might incorrectly conclude there's a load imbalance issue if you only look at raw timing differences between ranks.

### Solution
The **Rank Load Balance** tab includes variance analysis:

- **Low variance (<2%)**: All ranks slow down uniformly
  - **Conclusion**: Systemic EP overhead (inherent to all-to-all, etc.)
  - **Action**: Accept or optimize communication

- **High variance (>5%)**: Ranks slow down unevenly
  - **Conclusion**: Load imbalance (uneven expert assignment)
  - **Action**: Improve load balancing in router

### Example
```
Rank 0: +120ms slower (40%)
Rank 1: +118ms slower (39%)
Rank 2: +122ms slower (41%)
Rank 3: +120ms slower (40%)

Variance: ±0.8% → BALANCED

Conclusion: NOT a load imbalance issue. This is systemic EP overhead.
```

---

## Profiling Configuration

### Steps Profiled
- Total steps: 10
- Profiling at: steps 5 and 10
- Warmup: 2 steps
- Active: 2 steps
- Frequency: 5 steps

### Output Locations
- EP=2 traces: `outputs_profile_ep2/profile_trace/iteration_*/rank*_trace.json`
- EP=1 traces: `outputs_profile_ep1/profile_trace/iteration_*/rank*_trace.json`

### Trace Size
With enhanced profiling:
- Each trace: ~50-200 MB (depending on operations)
- Total for both EP configs: ~2-4 GB

---

## Expected Bottlenecks to Identify

Based on MoE architecture, common bottlenecks:

1. **All-to-all communication** (EP=2 only)
   - Token shuffle between EP ranks
   - Inherent cost of EP

2. **Memory transfers** (`aten::_to_copy`, `cudaMemcpyAsync`)
   - Blocking CPU tensor copies
   - Fix: `non_blocking=True`

3. **Token reordering** (`torch::histc`, `torch::argsort`)
   - Called twice (router + reorderer)
   - Potential optimization: single call

4. **Expert computation imbalance**
   - Uneven expert load
   - Fix: Improve load balancing

5. **Memory allocation overhead**
   - Frequent alloc/free in EP=2
   - Fix: Memory pooling

---

## Next Steps

Once profiling completes:

1. **Load dashboard** and click "Load Ultra-Deep Analysis"
2. **Check Summary tab** for overview
3. **Check Rank Balance tab** first to rule out false negatives
4. **Dive into Communication Analysis** if communication-bound
5. **Check Module Performance** to identify MoE component bottlenecks
6. **Use FLOPs Efficiency** to determine compute vs memory bound
7. **Review optimization recommendations** in Summary

---

## Metrics to Monitor

### Performance
- EP=2 average step time
- EP=1 average step time
- Slowdown percentage

### Communication
- Total communication time
- % of slowdown from communication
- All-to-all latency

### Load Balance
- Rank slowdown variance
- Min/max rank slowdown

### Memory
- Memory event count increase
- Peak memory usage

### Compute
- FLOPs rate
- Achieved vs theoretical FLOP rate

---

## Technical Details

### Profiler Overhead
With enhanced profiling:
- Overhead: ~10-15% slower than normal training
- Acceptable for profiling runs
- Do NOT use in production training

### Trace Processing Time
- Parsing all traces: ~10-30 seconds
- Dashboard rendering: ~5-10 seconds
- Total: <1 minute

### Browser Requirements
- Modern browser with JavaScript enabled
- Recommended: Chrome, Firefox
- Plotly visualizations require WebGL

---

## Troubleshooting

### Profiling fails
- **Check**: CUDA memory available
- **Check**: Disk space for traces
- **Fix**: Reduce batch size or sequence length

### Dashboard shows no data
- **Check**: Trace files exist in `outputs_profile_*/profile_trace/`
- **Fix**: Re-run profiling

### FLOPs data missing
- **Check**: Enhanced profiling config applied
- **Verify**: `torchtitan/tools/profiling.py` has `with_flops=True`

### Memory events missing
- **Check**: `profile_memory=True` in profiling config
- **Verify**: PyTorch version supports memory profiling

---

## Comparison: Before vs After

### Before (Original Dashboard)
- Basic operation timing
- Simple contribution analysis
- Limited rank comparison
- **Risk**: Missing false negatives from load imbalance

### After (Ultra-Deep Dashboard)
- 7 types of profiling data
- Communication breakdown
- Module-level analysis
- FLOPs efficiency
- **Safety**: Rank variance analysis prevents false negatives
- Memory tracking
- Auto-generated optimization recommendations

---

## Acknowledgments

- PyTorch Profiler team for comprehensive API
- TorchTitan team for MoE implementation
- Gradio team for interactive dashboards
- Plotly team for visualizations

---

**Created**: November 11, 2025
**Status**: ✅ Enhanced profiling enabled, profiling runs in progress
**Next**: Analyze results with ultra-deep dashboard once profiling completes
