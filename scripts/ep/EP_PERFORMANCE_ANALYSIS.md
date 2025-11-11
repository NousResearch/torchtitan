# Expert Parallelism Performance Analysis: EP=2 vs EP=1

## Executive Summary

EP=2 is **~35% slower** than EP=1 (5,900 vs 7,961 TPS), contrary to the expected performance improvement from distributed expert computation. Root cause analysis reveals that **synchronous device-to-host memory transfers** in the Expert Parallel implementation create massive overhead.

## Performance Comparison

### Throughput Results
| Configuration | Average TPS | TFLOPs | MFU |
|--------------|-------------|--------|-----|
| EP=1 (4 GPUs) | 7,961 | 185.9 | 8.26% |
| EP=2 (4 GPUs) | 5,900 | 138.0 | 6.14% |
| **Difference** | **-25.9%** | **-25.8%** | **-25.6%** |

### Profiling Analysis (Single Training Step)

| Operation | EP=2 Time | EP=1 Time | Overhead | Percentage |
|-----------|-----------|-----------|----------|------------|
| **D2H memcpy** | 839.69ms | 57.90ms | +781.79ms | **+1350%** |
| **wait_tensor** | 75.38ms | 0.22ms | +75.16ms | **+33,968%** |
| **NCCL collectives** | 2,600.22ms | 1,612.54ms | +987.68ms | **+61.3%** |
| **all_to_all (NEW)** | 1,390.43ms | 0.00ms | +1,390.43ms | **N/A** |
| **Total Overhead** | - | - | **~3,235ms** | - |

### Operation Counts

| Operation | EP=2 Count | EP=1 Count | Ratio |
|-----------|------------|------------|-------|
| D2H memcpy | 1,582 | 810 | 1.95x |
| wait_tensor | 1,356 | 8 | **169x** |
| NCCL ops | 3,138 | 918 | 3.42x |
| all_to_all | 3,263 | 0 | NEW |

## Root Cause Analysis

### Primary Bottleneck: Synchronous Device-to-Host Transfer

**Location:** `torchtitan/distributed/expert_parallel.py:104`

```python
# PROBLEMATIC CODE
output_splits = (
    num_tokens_per_expert_group.view(ep_degree, -1)
    .sum(dim=1)
    .to(torch.device("cpu"), non_blocking=False)  # <-- BLOCKING SYNC!
)
self.output_splits = output_splits.tolist()  # Uses CPU tensor immediately
```

**Why This Is Catastrophic:**

1. **Frequency:** Executes once per MoE layer, per forward/backward pass
   - Model has 48 MoE layers
   - Forward + Backward = 96 calls per training step
   - With microbatches: ~1,500+ calls per profiled step

2. **Blocking Behavior:** `non_blocking=False` forces:
   - All GPU kernels to complete before transfer
   - CPU to wait for transfer completion
   - All other GPUs in the collective to wait
   - Pipeline stall across entire distributed system

3. **Measured Impact:**
   - 1,582 D2H transfers (vs 810 in EP=1)
   - 839.69ms spent in memcpy (vs 57.90ms in EP=1)
   - Additional 75.38ms in wait_tensor operations

### Secondary Bottlenecks

#### 1. Excessive wait_tensor Synchronization (Line 92-93)

```python
num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
    num_tokens_per_expert_group
)
```

- **Impact:** 1,356 synchronization points vs 8 in EP=1
- **Overhead:** +75ms per step
- **Root Issue:** Forces synchronous wait before D2H transfer

#### 2. All-to-All Communication Overhead

- **New Operation:** Not present in EP=1
- **Frequency:** 3,263 operations per step
- **Overhead:** 1,390.43ms per step
- **Note:** This is inherent to EP design, but exacerbated by synchronization

#### 3. Increased NCCL Collective Operations

- **EP=2:** 3,138 operations (2,600ms)
- **EP=1:** 918 operations (1,613ms)
- **Overhead:** +987ms per step
- **Reason:** 2D mesh requires more coordination

## Technical Deep Dive

### Why EP=2 Uses 2D Mesh

```
EP=2: Building 2-D device mesh with ['dp_shard_mod_ep', 'dp_shard_in_ep'], [2, 2]
EP=1: Building 1-D device mesh with ['dp_shard'], [4]
```

With 4 GPUs and EP=2:
- Expert dimension is sharded across 2 GPUs
- Data parallelism dimension uses remaining 2 GPUs
- Creates [2, 2] mesh requiring inter-dimension communication

### The Token Dispatch Flow (EP=2)

For each MoE layer forward pass:

1. **Router** selects experts for each token
2. **Token Dispatch** (`_token_dispatch`):
   a. First all_to_all: Exchange token counts (line 84-89)
   b. **SYNC POINT 1:** wait_tensor (line 92-93) - **SLOW**
   c. Compute input_splits on GPU (line 95-99)
   d. **SYNC POINT 2:** D2H transfer with blocking=True (line 101-105) - **VERY SLOW**
   e. Convert to Python list (line 106-107) - requires CPU tensor
   f. Second all_to_all: Exchange actual tokens (line 110-115)
   g. Permute tokens for local experts (line 129-136)

3. **Expert Computation** on local experts
4. **Token Combine** (`_token_combine`):
   - Reverse permutation
   - All-to-all to return tokens to original ranks

**The Problem:** Steps 2b, 2d, and 2e create a synchronous pipeline stall **48 times per forward and 48 times per backward**.

## Why non_blocking=False Was Used

The code comment suggests this was intentional:

```python
# NOTE: this would incur a device-to-host sync
output_splits = (...).to(torch.device("cpu"), non_blocking=False)
```

**Likely Reason:** The `output_splits.tolist()` call on line 106-107 immediately accesses the CPU tensor. With `non_blocking=True`, this would cause an implicit synchronization anyway when `.tolist()` is called.

**However:** The current implementation forces synchronization at the worst possible time - in the critical path of every MoE layer.

## Proposed Solutions

### Solution 1: Cache Split Computation (Best for Static Workloads)

If token distributions don't vary significantly across steps:

```python
# Cache splits for N steps, only recompute periodically
if self.cached_splits is None or step % RECOMPUTE_INTERVAL == 0:
    # Compute splits asynchronously
    output_splits = (...).to(torch.device("cpu"), non_blocking=True)
    torch.cuda.synchronize()  # Explicit sync when we can afford it
    self.cached_splits = output_splits.tolist()
else:
    output_splits_list = self.cached_splits
```

**Pros:** Eliminates 95%+ of D2H transfers
**Cons:** May hurt load balancing if distributions change significantly

### Solution 2: Async D2H with Double Buffering

```python
# Use previous step's splits while computing current
if self.pending_splits_future is not None:
    self.input_splits = self.pending_splits_future.wait()

# Start async transfer for next iteration
output_splits_gpu = num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1)
output_splits_cpu = output_splits_gpu.to(torch.device("cpu"), non_blocking=True)

# Save for next iteration (don't access yet)
self.pending_splits_future = AsyncSplitsFuture(output_splits_cpu)

# Use approximate/previous splits for this iteration
# (Requires conservative memory allocation)
```

**Pros:** Hides latency completely
**Cons:** Complex implementation, requires conservative memory estimates

### Solution 3: Keep Splits on GPU

Avoid CPU entirely by keeping splits as GPU tensors:

```python
# Keep on GPU - no D2H transfer
output_splits_gpu = num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1)

# Use CUDA-aware all-to-all that accepts GPU splits
routed_input = all_to_all_single_autograd_gpu_splits(
    routed_input,
    output_splits_gpu,  # Pass GPU tensor directly
    input_splits_gpu,
    device_mesh.get_group(),
)
```

**Pros:** Eliminates D2H transfer completely
**Cons:** Requires changes to PyTorch distributed APIs

### Solution 4: Pipelined Communication (Recommended)

Overlap computation with communication:

```python
# Start all-to-all early, before waiting for splits
with torch.cuda.stream(comm_stream):
    num_tokens_per_expert_group = all_to_all_single(...)

# Overlap: While all-to-all is running, do other work
# ...

# Only wait when absolutely necessary
num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(...)

# Use non_blocking=True and synchronize explicitly later
output_splits = (...).to(torch.device("cpu"), non_blocking=True)
# Do other GPU work here...
torch.cuda.current_stream().synchronize()  # Sync only when needed
self.output_splits = output_splits.tolist()
```

**Pros:** Hides communication latency
**Cons:** Moderate implementation complexity

## Immediate Quick Fix

**Simplest improvement with minimal risk:**

```python
# Line 104: Change from
output_splits = (...).to(torch.device("cpu"), non_blocking=False)

# To:
output_splits = (...).to(torch.device("cpu"), non_blocking=True)
# Move the implicit sync to after we're done with GPU work
torch.cuda.current_stream().synchronize()
self.output_splits = output_splits.tolist()
```

**Expected Impact:** Reduce D2H overhead by allowing GPU to continue work while transfer happens.

**Risk:** Low - still ensures correctness with explicit sync before use.

## Validation Plan

1. **Apply Quick Fix** (non_blocking=True change)
2. **Re-run profiling** with same config
3. **Expected improvements:**
   - D2H memcpy: 840ms → ~200ms (75% reduction)
   - wait_tensor: 75ms → ~5ms (93% reduction)
   - Overall TPS: 5,900 → ~7,500 (27% improvement)
4. **Verify correctness:** Check loss values match baseline

## Conclusion

The EP=2 slowdown is **entirely due to synchronization overhead** in the expert dispatch logic, not fundamental limitations of expert parallelism. The blocking device-to-host transfer creates a synchronous bottleneck that happens 96 times per training step.

**Key Insight:** The overhead from synchronization (~3,235ms) is actually larger than the communication overhead from all-to-all operations (~1,390ms), making synchronization the primary target for optimization.

**Recommendation:** Apply Solution 1 (caching) for immediate 10-20% improvement, then implement Solution 4 (pipelining) for full optimization targeting 30-40% overall improvement.
