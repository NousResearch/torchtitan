# 🔬 Ultra-Deep EP Performance Analysis Dashboard

## Overview

The **Ultra-Deep Dashboard** (`ultra_deep_dashboard.py`) leverages the enhanced PyTorch profiler configuration to provide comprehensive, in-depth analysis of Expert Parallelism (EP) performance bottlenecks when scaling from EP=1 to EP=2.

## What's New in Enhanced Profiling?

The profiling configuration in `torchtitan/tools/profiling.py` has been significantly enhanced to capture:

1. **Memory Profiling** (`profile_memory=True`)
   - Tracks all CUDA memory allocations and deallocations
   - Memory timeline visualization
   - Peak memory usage analysis

2. **Stack Traces** (`with_stack=True`)
   - Full Python call stacks for every operation
   - Source file and line number tracking
   - Hot path identification

3. **FLOPs Tracking** (`with_flops=True`)
   - Estimates floating-point operations for each kernel
   - Compute efficiency metrics
   - Identifies compute-bound vs memory-bound operations

4. **Module Hierarchy** (`with_modules=True`)
   - Groups operations by PyTorch module
   - MoE component-level breakdown (Router, GroupedExperts, etc.)
   - Hierarchical performance analysis

5. **Per-Kernel Timing** (`profiler_measure_per_kernel=True`)
   - Detailed per-CUDA-kernel performance
   - Kernel launch overhead tracking
   - Identifies inefficient kernel launches

6. **CUDA Synchronization Events** (`enable_cuda_sync_events=True`)
   - Tracks cudaDeviceSynchronize calls
   - Identifies unnecessary synchronization overhead
   - Better accuracy for async operations

7. **Python GC Info** (`record_python_gc_info=True`)
   - Garbage collection event tracking
   - Memory fragmentation analysis
   - GC-induced pauses

## Dashboard Features

### 🎯 Core Visualizations

#### 1. **Communication Analysis Tab**
Deep dive into all-to-all and NCCL operations:
- Time comparison (EP=2 vs EP=1)
- Operation count and frequency
- Total communication time breakdown
- Average latency differences
- **Goal**: Identify if EP scaling overhead is communication-bound

#### 2. **Module Performance Tab**
Performance aggregated by PyTorch module:
- MoE.router timing
- MoE.experts (GroupedExperts) timing
- MoE.reorderer timing
- Token dispatch/combine analysis
- **Goal**: Identify which MoE component is the bottleneck

#### 3. **FLOPs Efficiency Tab**
Compute intensity analysis:
- GFLOPs comparison across operations
- Achieved vs theoretical FLOP rate
- Identify compute-bound vs memory-bound ops
- **Goal**: Determine if operations are utilizing GPU compute efficiently

#### 4. **Rank Load Balance Tab** ⚠️ **Critical for False Negatives**
Detailed per-rank analysis:
- Step time distribution per rank (box plots)
- Slowdown percentage per rank
- Variance analysis across ranks
- **Goal**: Detect if slowdown is uniform (systemic) or imbalanced (workload distribution)
- **Why Critical**: If all ranks show uniform slowdown (low variance), it's NOT a load imbalance issue—it's inherent EP overhead!

#### 5. **Memory Analysis Tab**
Memory allocation pattern comparison:
- Total memory events (EP=2 vs EP=1)
- Memory allocation overhead
- Fragmentation detection
- **Goal**: Identify if memory management is a bottleneck

#### 6. **Full Data Table Tab**
Comprehensive operation statistics:
- Operation name, timings, std dev
- FLOPs data
- Module hierarchy
- Contribution percentage
- **Goal**: Deep dive into specific operations

### 📊 Summary Tab

Auto-generated analysis report including:
- Overall performance metrics
- Top 10 bottlenecks with contribution percentages
- Communication overhead breakdown
- Memory analysis
- Rank balance status
- Optimization recommendations

## How to Use

### 1. Run Profiling (Both EP Configs)

```bash
cd /path/to/torchtitan-nous
./scripts/ep/run_profiling.sh both
```

This will:
- Run EP=2 profiling (saves to `outputs_profile_ep2/profile_trace/`)
- Run EP=1 profiling (saves to `outputs_profile_ep1/profile_trace/`)
- Each runs 10 steps with profiling at steps 5 and 10

### 2. Launch Ultra-Deep Dashboard

```bash
cd /path/to/torchtitan-nous
./scripts/ep/ultra_deep_dashboard.py
```

Or use the original dashboard:

```bash
./scripts/ep/interactive_dashboard.py
```

The dashboard will:
- Parse all trace JSON files
- Extract memory, FLOPs, stack traces, module info
- Compute aggregated statistics
- Generate interactive Plotly visualizations
- Create a public Gradio link (for SSH access)

### 3. Analyze Results

**Step-by-step workflow:**

1. **Click "Load Ultra-Deep Analysis"** button
   - This parses all traces and computes statistics
   - Takes 10-30 seconds depending on trace size

2. **Check Summary Tab**
   - Review overall slowdown percentage
   - Identify top 5-10 bottlenecks
   - Note if communication or compute-bound

3. **Communication Analysis Tab**
   - Look for `nccl:all_to_all`, `all_to_all_single`, etc.
   - Check if communication time >> compute time
   - If yes → communication-bound (inherent to EP)

4. **Rank Balance Tab** ⚠️ **Most Important**
   - Check slowdown variance across ranks
   - **Low variance (<2%)**: Uniform slowdown → systemic EP overhead (not fixable by load balancing)
   - **High variance (>5%)**: Load imbalance → investigate token distribution, expert assignment

5. **Module Performance Tab**
   - See which MoE component contributes most
   - Example findings:
     - `MoE.experts` slow → Expert computation bottleneck
     - `MoE.router` slow → Routing overhead
     - `all_to_all` slow → Token shuffle overhead

6. **FLOPs Efficiency Tab**
   - Check if compute ops are achieving high FLOP rates
   - Low FLOP rate + high time → memory-bound
   - High FLOP rate → compute-bound (good GPU utilization)

7. **Memory Analysis Tab**
   - Check if EP=2 has significantly more memory events
   - High increase → memory allocation overhead

## Key Metrics to Watch

### 🚨 Critical Indicators

| Metric | What It Means | Action |
|--------|---------------|--------|
| **Communication time > 50% of slowdown** | EP is communication-bound | Overlap comm with compute, reduce data transfer |
| **Rank variance < 2%** | NOT a load imbalance issue | Focus on systemic optimizations (kernel fusion, overlap) |
| **Rank variance > 5%** | Load imbalance detected | Investigate expert assignment, token distribution |
| **Memory events +30%** | Memory allocation overhead | Use memory pooling, pre-allocation |
| **FLOPs rate < 30% of peak** | Memory-bound operations | Optimize memory access patterns |

### 💡 Optimization Targets

Based on dashboard findings:

1. **If `nccl:all_to_all` is #1 bottleneck**:
   - **Root cause**: Token shuffle between EP ranks
   - **Fix**: Overlap all-to-all with other computation (hard)
   - **Alternative**: Accept as inherent cost of EP

2. **If `aten::_to_copy` or `cudaMemcpyAsync` is high**:
   - **Root cause**: Blocking CPU tensor copies in `expert_parallel.py:104`
   - **Fix**: Change to `non_blocking=True`
   - **Expected gain**: 20-30% throughput increase

3. **If `torch::histc` is high**:
   - **Root cause**: Token counting happens twice (router + reorderer)
   - **Fix**: Optimize to single histc call
   - **Expected gain**: 5-10% reduction

4. **If rank variance > 5%**:
   - **Root cause**: Uneven expert load distribution
   - **Fix**: Improve load balancing in router
   - **Expected gain**: Reduce slowest rank time

## Comparing with Original Dashboard

| Feature | Original (`interactive_dashboard.py`) | Ultra-Deep (`ultra_deep_dashboard.py`) |
|---------|---------------------------------------|----------------------------------------|
| Operation timing | ✅ | ✅ |
| Contribution analysis | ✅ | ✅ |
| Rank analysis | ✅ | ✅ Enhanced |
| Communication breakdown | ❌ | ✅ Detailed |
| Module hierarchy | ❌ | ✅ New |
| FLOPs analysis | ❌ | ✅ New |
| Memory events | ❌ | ✅ New |
| Stack traces | ❌ | ✅ Parsed (in data) |
| Per-kernel timing | ❌ | ✅ New |

## Technical Details

### Trace Parsing

The dashboard parses Chrome Trace JSON files with:
- **Duration events** (`ph: 'X'`): Operation timing
- **Memory events**: Allocation/free events
- **Stack info**: From `args['Python call stack']`
- **FLOPs**: From `args['Flops']`
- **Module**: From `args['Module Hierarchy']`

### Data Aggregation

For each operation:
- Mean, median, std dev, min, max across all occurrences
- Aggregated across ranks and iterations
- Per-rank statistics for load balance analysis

### Contribution Calculation

```python
contribution_pct = (ep2_time - ep1_time) / total_step_diff * 100
```

This shows what % of the slowdown each operation contributes.

## Troubleshooting

### No data appears

**Cause**: Trace files not found
**Fix**: Check that profiling completed:
```bash
ls outputs_profile_ep2/profile_trace/iteration_*/rank*_trace.json
ls outputs_profile_ep1/profile_trace/iteration_*/rank*_trace.json
```

### Dashboard crashes on load

**Cause**: Corrupted trace JSON
**Fix**: Re-run profiling or exclude corrupted files

### FLOPs/Memory data missing

**Cause**: Enhanced profiling not enabled
**Fix**: Verify `torchtitan/tools/profiling.py` has the enhanced config (should be already applied)

### Communication events not detected

**Cause**: NCCL operations may have different names
**Fix**: Check trace files manually for operation names containing "nccl", "all_to_all", etc.

## Examples of Insights

### Example 1: Communication-Bound

```
Summary Tab shows:
- Total slowdown: +500ms
- Top bottleneck: nccl:all_to_all → +400ms (80% contribution)
- Rank variance: ±1.2% → Uniform across ranks

Communication Tab shows:
- all_to_all: EP=2: 450ms, EP=1: 50ms

Conclusion: EP=2 is communication-bound. 80% of slowdown is inherent all-to-all overhead.
Action: Accept as cost of EP or try to overlap communication.
```

### Example 2: Memory Transfer Bottleneck

```
Summary Tab shows:
- Total slowdown: +300ms
- Top bottleneck: aten::_to_copy → +180ms (60% contribution)
- Rank variance: ±0.8% → Uniform

Module Tab shows:
- This operation is in expert_parallel.py token dispatch

Memory Tab shows:
- +500 more memory events in EP=2

Conclusion: Blocking CPU tensor copies causing slowdown.
Action: Apply non_blocking=True fix.
```

### Example 3: Load Imbalance (False Negative Avoided!)

```
Summary Tab shows:
- Total slowdown: +400ms
- Rank variance: ±8.5% → HIGH variance!

Rank Balance Tab shows:
- Rank 0: +250ms slower
- Rank 1: +320ms slower
- Rank 2: +450ms slower ← SLOWEST
- Rank 3: +350ms slower

Conclusion: Load imbalance! Rank 2 is bottleneck.
Action: Investigate expert assignment for Rank 2, check token distribution.
```

**Note**: Without the Rank Balance tab, you might miss that Rank 2 is the problem!

## Future Enhancements

Potential additions:
- Timeline visualization (Gantt chart of operations)
- Kernel-level heatmap (which kernels are slow)
- Stack trace frequency analysis (hot paths)
- Memory timeline plot
- Iteration-to-iteration variance
- Automated bottleneck classification (comm vs compute vs memory)

## References

- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan)
- [Expert Parallelism Paper](https://arxiv.org/abs/2103.16690)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Plotly Documentation](https://plotly.com/python/)

---

**Dashboard developed for ultra-deep profiling analysis of MoE Expert Parallelism scaling bottlenecks.**
