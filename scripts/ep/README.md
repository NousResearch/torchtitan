# Expert Parallelism Performance Analysis Tools

This directory contains tools for profiling and analyzing Expert Parallelism (EP) performance in TorchTitan.

## Quick Start

### 1. View existing analysis results

```bash
# Show the detailed comparison table
./scripts/ep/compare_profiles.sh
```

This will display the top 30 operations by time difference between EP=2 and EP=1.

### 2. Re-run profiling (optional)

If you want to generate fresh profiling data:

```bash
# Run both EP=1 and EP=2 profiling
./scripts/ep/run_profiling.sh both

# Or run individually
./scripts/ep/run_profiling.sh ep1
./scripts/ep/run_profiling.sh ep2
```

### 3. Analyze results

```bash
# After profiling completes, run analysis
./scripts/ep/compare_profiles.sh
```

## Files

### Scripts

- **`compare_profiles.sh`** - Main analysis script (run this to see the table)
- **`run_profiling.sh`** - Run profiling for EP=1/EP=2
- **`analyze_traces.py`** - Basic profiling analysis tool
- **`detailed_trace_analysis.py`** - Detailed operation-level comparison

### Config Files

- **`profile_ep1_config.toml`** - Profiling config for EP=1
- **`profile_ep2_config.toml`** - Profiling config for EP=2

### Documentation

- **`EP_PERFORMANCE_ANALYSIS.md`** - Comprehensive analysis report with:
  - Root cause analysis
  - Profiling results breakdown
  - Proposed solutions
  - Technical deep dive

## Output Files (Generated)

After running profiling, you'll find:

```
./outputs_profile_ep1/
  └── profile_trace/
      ├── iteration_5/rank0_trace.json   # Main trace file
      └── iteration_10/rank0_trace.json

./outputs_profile_ep2/
  └── profile_trace/
      ├── iteration_5/rank0_trace.json
      └── iteration_10/rank0_trace.json

./ep1_profile_run.log  # Full training log for EP=1
./ep2_profile_run.log  # Full training log for EP=2
```

## Understanding the Output

### Comparison Table

The `compare_profiles.sh` script shows:

1. **Top 30 Operations by Time Difference**
   - Operation name
   - Time spent in EP=2 vs EP=1
   - Absolute and percentage difference
   - Number of calls for each

2. **Category Summary**
   - NCCL Communication
   - Memory Transfer
   - MoE Operations
   - Linear/GEMM
   - Synchronization
   - etc.

3. **New Operations in EP=2**
   - Operations that don't exist in EP=1
   - Primarily all-to-all communication

### Key Metrics to Look For

- **aten::_to_copy** - Device-to-host transfers (should be high in EP=2)
- **cudaMemcpyAsync** - Async memory copies (blocking in EP=2)
- **nccl:all_to_all** - All-to-all collectives (new in EP=2)
- **wait_tensor** - Synchronization points

## Key Findings

From the analysis:

1. **EP=2 is 35% slower than EP=1** (5,900 vs 7,961 TPS)

2. **Root cause**: Synchronous device-to-host transfers
   - Location: `torchtitan/distributed/expert_parallel.py:104`
   - `aten::_to_copy`: +2,314ms (+837%)
   - `cudaMemcpyAsync`: +781ms (+1,350%)

3. **Secondary bottlenecks**:
   - All-to-all communication: +1,115ms (inherent to EP)
   - NCCL overhead: +2,920ms (2D mesh coordination)

## Proposed Fix

Quick fix in `expert_parallel.py:104`:

```python
# Change from:
output_splits = (...).to(torch.device("cpu"), non_blocking=False)

# To:
output_splits = (...).to(torch.device("cpu"), non_blocking=True)
```

Expected improvement: 20-30% throughput increase.

## Advanced Usage

### Custom Analysis

You can modify `detailed_trace_analysis.py` to:

- Change the number of operations shown (default: 30)
- Filter specific operation categories
- Analyze different iterations
- Compare multiple ranks

Example:

```python
# In detailed_trace_analysis.py, line ~143
print_comparison_table(differences, top_n=50)  # Show top 50 instead
```

### Compare Different Configurations

To compare other EP configurations:

1. Copy and modify `profile_ep2_config.toml`
2. Change `expert_parallel_degree` to your desired value
3. Run profiling: `./scripts/ep/run_profiling.sh`
4. Update `detailed_trace_analysis.py` to point to new trace files

## Troubleshooting

### "Trace file not found"

Make sure you've run profiling first:
```bash
./scripts/ep/run_profiling.sh both
```

### "Permission denied"

Make scripts executable:
```bash
chmod +x scripts/ep/*.sh
```

### Out of Memory

Reduce `local_batch_size` in the config files:
```toml
[training]
local_batch_size = 4  # Reduce from 6
```

## Further Reading

- **EP_PERFORMANCE_ANALYSIS.md** - Full technical analysis
- PyTorch Profiler docs: https://pytorch.org/docs/stable/profiler.html
- TorchTitan Expert Parallel: `../../torchtitan/distributed/expert_parallel.py`

## Contact

For questions or issues, refer to the main TorchTitan repository.
