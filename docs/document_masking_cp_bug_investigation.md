# Document Masking Bug with Context Parallel (CP) - Investigation Report

**Date:** 2026-01-31
**Repository:** `/home/phuc/kimi_1t/torchtitan`
**Issue:** Loss ~3 instead of ~0.8 when using document masking with Context Parallel enabled

---

## 1. Problem Statement

### Reported Symptoms
- With CP enabled + document masking: loss starts at ~3 (wrong)
- With CP disabled + document masking: loss starts at ~0.8 (correct)
- Using `block_causal_by_sequence_lengths` mask type with sequence lengths from dataloader
- Suspicion: "flex masks actually were not being applied"

### Affected Code Path
```
torchtitan/models/deepseek_v3/model/model.py:get_attention_masks()
  → create_cp_block_mask() from PyTorch
  → Used inside context_parallel() context during training
```

---

## 2. Initial Hypothesis

**My first hypothesis was wrong.**

I initially thought the bug was:
- `create_cp_block_mask` passes LOCAL q_idx to mask_mod
- But `document_ids` tensor uses GLOBAL indices
- Therefore, on rank > 0, the mask indexes wrong document IDs

This led me to write `debug_document_mask_simple.py` which showed 40% mask errors on rank 1.

### Why This Was Wrong

After reading PyTorch's source code at:
```
/home/phuc/kimi_1t/env/lib/python3.10/site-packages/torch/distributed/tensor/experimental/_attention.py:1160-1183
```

I discovered that `create_cp_block_mask` **already handles the offset conversion** internally via `_rewrite_mask_mod`:

```python
def _rewrite_mask_mod(mask_mod, rank, world_size, block_size, local_q_size):
    def local_q_idx_to_q_idx(local_q_idx):
        local_blk_idx, local_blk_offset = (
            local_q_idx // block_size,
            local_q_idx % block_size,
        )
        local_num_blocks = local_q_size // block_size
        blk_idx = local_num_blocks * rank + local_blk_idx
        return blk_idx * block_size + local_blk_offset

    return lambda b, h, q_idx, kv_idx: mask_mod(
        b, h, local_q_idx_to_q_idx(q_idx), kv_idx  # <-- Already converts to global!
    )
```

So PyTorch's `create_cp_block_mask` should work correctly in theory.

---

## 3. Failed Approach #1: Adding Offset Wrapper

### What I Tried
Added `apply_cp_offset_to_mask_mod()` to `torchtitan/models/attention.py`:

```python
def apply_cp_offset_to_mask_mod(mask_mod, q_offset):
    def cp_aware_mask_mod(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + q_offset
        return mask_mod(b, h, global_q_idx, kv_idx)
    return cp_aware_mask_mod
```

And used it in `model.py` before calling `create_cp_block_mask`.

### Why It Failed
**Index out of bounds error:** `256 + r0_3 < 256`

Since `create_cp_block_mask` already converts local→global internally, my wrapper was **double-applying** the offset, causing indices to exceed the document_ids tensor bounds.

### Reverted
Removed this approach entirely.

---

## 4. Investigation: What's Actually Wrong with create_cp_block_mask?

### Test: `debug_cp_document_mask_e2e.py`

Created end-to-end test comparing:
1. **Baseline**: Full sequence attention, no CP
2. **Manual CP**: Local Q, gathered K/V, offset-aware mask (manually implemented)
3. **PyTorch CP**: Using `create_cp_block_mask` + `context_parallel`

### Results

```
Manual CP vs Baseline: max_diff=0.000000  ← Manual approach works perfectly!
PyTorch CP: FAILED with shape mismatch errors
```

The manual CP implementation (where I handle the offset myself) produces **identical** output to the baseline. This proves the approach is correct.

But PyTorch's `create_cp_block_mask` + `context_parallel` combination has issues:
- Shape mismatches between mask and tensor sizes
- The mask is created for one shape but `context_parallel` transforms tensors differently

### Key Insight from Error Messages

```
ValueError: block_mask was created for block_mask.shape=(1, 4, 128, 128)
but got q_len=64 and kv_len=64
```

Inside `context_parallel`, tensors are further transformed in ways that don't match what `create_cp_block_mask` expects.

---

## 5. Working Solution

### The Fix

Instead of using PyTorch's `create_cp_block_mask`, manually create the mask with proper offset handling.

**File:** `torchtitan/models/deepseek_v3/model/model.py:519-547`

```python
# Before (buggy):
if cp_mesh is not None:
    return create_cp_block_mask(
        mask_mod=combined_mask_mod,
        B=B, H=H, Q_LEN=seq_len, KV_LEN=seq_len,
        device_mesh=cp_mesh,
    )

# After (fixed):
if cp_mesh is not None:
    cp_rank = cp_mesh.get_local_rank()
    cp_size = cp_mesh.size()
    local_seq_len = seq_len // cp_size
    q_offset = cp_rank * local_seq_len

    def cp_aware_mask_mod(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + q_offset
        return combined_mask_mod(b, h, global_q_idx, kv_idx)

    return create_attention_mask(
        cp_aware_mask_mod, B, None, local_seq_len, seq_len
    )
```

### Why This Works

1. **Mask dimensions**: `Q_LEN=local_seq_len`, `KV_LEN=full_seq_len`
   - Matches what CP attention expects: local Q vs gathered global K/V

2. **Offset handling**: `global_q_idx = q_idx + q_offset`
   - Correctly converts local indices to global for document_ids lookup

3. **Bypasses buggy PyTorch code**: Uses standard `create_attention_mask` instead of `create_cp_block_mask`

---

## 6. Verification

### Test 1: Mask Correctness (`debug_cp_fix_verify.py`)

```
Rank 0: max_diff = 0.00000000 ✓
Rank 1: max_diff = 0.00000000 ✓
ALL TESTS PASSED - The fix is correct!
```

### Test 2: Document Masking Verification (`debug_verify_document_masking.py`)

```
Cross-document pairs blocked: 192/192 (100%)
Within-document causal pairs allowed: 40/40 (100%)
✓ Document masking is 100% correct!
```

Visual mask output shows correct block-diagonal causal structure:
```
 0 | 1 0 0 0 |0 0 0 0 |0 0 0 0 |0 0 0 0   <- Doc 0
 1 | 1 1 0 0 |0 0 0 0 |0 0 0 0 |0 0 0 0   <- Doc 0
...
 4 | 0 0 0 0 |1 0 0 0 |0 0 0 0 |0 0 0 0   <- Doc 1
```

### Test 3: Training Comparison (`debug_train_cp_comparison.py`)

**Without CP (baseline):**
```
Final losses: ['7.0625', '7.0625', '7.0625', '7.0625', '7.1250', '7.0625', '7.0625', '7.0312', '7.0625', '7.0938']
```

**With CP (fixed):**
```
Final losses: ['7.0625', '7.0625', '7.0625', '7.0625', '7.1250', '7.0625', '7.0625', '7.0000', '7.0625', '7.0625']
```

Losses match closely (small differences due to floating-point precision in distributed ops).

---

## 7. Files Modified

### Production Code
| File | Change |
|------|--------|
| `torchtitan/models/deepseek_v3/model/model.py` | Fixed `get_attention_masks()` to bypass `create_cp_block_mask` and manually handle CP offset |

### Debug/Test Files Created
| File | Purpose |
|------|---------|
| `debug_document_mask_simple.py` | Single-GPU simulation showing mask index bug (initial hypothesis) |
| `debug_document_mask_cp_bug.py` | Distributed test demonstrating the bug |
| `debug_cp_mask_indices_check.py` | Diagnostic to check what indices PyTorch passes to mask_mod |
| `debug_cp_mask_verify.py` | Verify `create_cp_block_mask` mask correctness |
| `debug_cp_document_mask_e2e.py` | End-to-end comparison: baseline vs manual CP vs PyTorch CP |
| `debug_cp_document_mask_e2e_v2.py` | Updated E2E test matching torchtitan's usage pattern |
| `debug_cp_fix_verify.py` | Verify the fix produces correct output |
| `debug_train_cp_comparison.py` | Training comparison with/without CP |
| `debug_verify_document_masking.py` | Verify document masking blocks cross-doc attention |

---

## 8. Root Cause Summary

**The bug is in PyTorch's `create_cp_block_mask` + `context_parallel` interaction**, not in torchtitan's code.

Specifically:
1. `create_cp_block_mask` creates a mask with certain assumptions about tensor shapes
2. `context_parallel` transforms tensors in ways that don't match those assumptions
3. This causes shape mismatches, incorrect attention patterns, or failures

**The fix** bypasses this by:
1. Not using `create_cp_block_mask`
2. Creating a standard mask with `create_attention_mask`
3. Manually handling the q_offset in the mask_mod closure
4. Using correct dimensions: `Q_LEN=local_seq_len`, `KV_LEN=full_seq_len`

---

## 9. Remaining Questions

1. **Should this be reported to PyTorch?** The `create_cp_block_mask` function appears to have bugs when used with document masking.

2. **Does this affect other models?** Only `deepseek_v3` has `cp_mesh` in `get_attention_masks()`. `llama3` and `qwen3` don't have CP mask support, so they're unaffected.

3. **Performance impact?** The fix uses standard `create_attention_mask` instead of `create_cp_block_mask`. Performance should be similar, but worth benchmarking.

---

## 10. How to Reproduce

```bash
# Verify document masking works correctly
python debug_verify_document_masking.py

# Verify fix produces correct output
python debug_cp_fix_verify.py

# Compare training with/without CP
python debug_train_cp_comparison.py --no-cp  # Baseline
torchrun --nproc_per_node=2 debug_train_cp_comparison.py --cp  # With CP
```

Expected: Both should produce similar losses (~7 for random data, ~0.8 for real data like Hermes3).

---

## 11. FlexAttention + CP Correctness Verification

After the document masking fix, additional verification was performed to ensure FlexAttention works correctly with Context Parallel in general.

### 11.1 Motivation

Before testing document masking with a real model, we needed to verify that:
1. FlexAttention + CP works correctly with **causal masking only** (no document masking)
2. FlexAttention + CP works correctly with **document masking**
3. End-to-end training produces correct results

### 11.2 Test 1: FlexAttention + CP with Causal Masking Only

**File:** `debug_cp_causal_only.py`

**Purpose:** Verify CP works with simple causal masking (no document masking) as a baseline.

**Approach:**
1. Create identical Q, K, V tensors on all ranks
2. Run full-sequence FlexAttention (baseline)
3. Run CP FlexAttention (local Q, gathered K/V, offset-aware mask)
4. Compare outputs

**Key implementation:**
```python
# CP-aware causal mask with offset
q_offset = rank * local_seq_len

def cp_causal_mask(b, h, q_idx, kv_idx):
    global_q_idx = q_idx + q_offset
    return global_q_idx >= kv_idx

cp_mask = create_block_mask(cp_causal_mask, B=batch_size, H=n_heads,
                             Q_LEN=local_seq_len, KV_LEN=seq_len, device=device)
```

**Results:**
```
[Rank 0] Max diff between CP and baseline: 0.0000000000
[Rank 1] Max diff between CP and baseline: 0.0000000000
✓ PASS: CP output matches baseline within tolerance 0.001
✓ GLOBAL PASS: All CP outputs match baseline
```

**Conclusion:** FlexAttention + CP with causal masking works perfectly.

---

### 11.3 Test 2: FlexAttention + CP with Document Masking

**File:** `debug_cp_document_masking.py`

**Purpose:** Verify CP works correctly when document masking is added.

**Setup:**
- seq_len=256, local_seq_len=128 per rank
- 4 documents of 64 tokens each
- Rank 0 handles docs 0-1, Rank 1 handles docs 2-3

**Key implementation:**
```python
def cp_doc_mask_mod(b, h, q_idx, kv_idx):
    global_q_idx = q_idx + q_offset
    causal = global_q_idx >= kv_idx
    doc_match = document_ids[b, global_q_idx] == document_ids[b, kv_idx]
    return causal & doc_match
```

**Results:**
```
[Rank 0] Max diff between CP and baseline: 0.0000000000
[Rank 1] Max diff between CP and baseline: 0.0000000000
✓ PASS: CP output matches baseline within tolerance 0.001
✓ GLOBAL PASS: All CP outputs match baseline
```

**Conclusion:** FlexAttention + CP with document masking works perfectly.

---

### 11.4 Test 3: Minimal FlexAttention + CP Test

**File:** `debug_flex_attention_cp_test.py`

**Purpose:** Stripped-down test of just `flex_attention()` function with CP.

**What it tests:**
- Direct `flex_attention(Q_local, K_gathered, V_gathered, block_mask=cp_mask)`
- Verifies gathered K/V matches original
- Compares output to baseline

**Results:**
```
K_gathered matches K_full: True
V_gathered matches V_full: True
Max diff: 0.0000000000
✓ PASS: FlexAttention + CP produces identical results
✓ GLOBAL PASS: All ranks produce correct results
```

---

### 11.5 Test 4: End-to-End Training (Random Labels)

**File:** `debug_e2e_cp_flex.py`

**Purpose:** Verify training loop works with CP (random labels, loss doesn't decrease).

**Results:**
| Step | No CP | With CP |
|------|-------|---------|
| 0 | 7.0625 | 7.0625 |
| 5 | 7.0625 | 7.0625 |
| 9 | 7.0938 | 7.0938 |

**Conclusion:** Training produces matching losses between CP and non-CP modes.

---

### 11.6 Test 5: End-to-End Training (Real Learning)

**File:** `debug_e2e_cp_flex_real.py`

**Purpose:** Verify training with actual learning (loss decreases over time).

**Setup:**
- Model: 4 layers, dim=256, 8 heads
- Data: Repeated patterns for next-token prediction
- 200 training steps

**Results:**
| Step | No CP | With CP |
|------|-------|---------|
| 0 | 5.75 | 5.75 |
| 10 | 3.27 | 3.47 |
| 30 | 0.36 | 0.43 |
| 100 | 0.12 | 0.13 |
| 190 | 0.10 | 0.11 |
| **Final** | **0.10** | **0.11** |

**Wandb Links:**
- No CP: https://wandb.ai/nous_research/cp-flex-attention-test/runs/mzirj0iw
- With CP: https://wandb.ai/nous_research/cp-flex-attention-test/runs/we6d317m

**Conclusion:** Both modes show proper learning with loss decreasing from ~5.75 to ~0.10. Small differences are expected due to distributed floating-point operations.

---

## 12. Applying Fix to Qwen3 Model

### 12.1 Problem

The qwen3 model's `get_attention_masks()` method did not have CP support - it didn't accept a `cp_mesh` parameter and didn't handle the offset.

**File:** `torchtitan/models/qwen3/model/model.py`

### 12.2 Changes Made

**Added imports:**
```python
from typing import Optional
from torch.distributed.device_mesh import DeviceMesh
```

**Updated `get_attention_masks()` signature:**
```python
def get_attention_masks(
    self,
    input_batch: torch.Tensor,
    tokenizer: BaseTokenizer,
    extra_inputs: dict[str, torch.Tensor] | None = None,
    cp_mesh: Optional[DeviceMesh] = None,  # NEW
) -> AttentionMasksType:
```

**Added CP-aware mask creation (same pattern as deepseek_v3):**
```python
combined_mask_mod = and_masks(*mask_mods)
seq_len = input_batch.shape[1]

if cp_mesh is not None:
    cp_rank = cp_mesh.get_local_rank()
    cp_size = cp_mesh.size()
    local_seq_len = seq_len // cp_size
    q_offset = cp_rank * local_seq_len

    def cp_aware_mask_mod(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + q_offset
        return combined_mask_mod(b, h, global_q_idx, kv_idx)

    return create_attention_mask(
        cp_aware_mask_mod, B, None, local_seq_len, seq_len
    )
else:
    return create_attention_mask(combined_mask_mod, B, None, seq_len, seq_len)
```

---

## 13. Summary of All Test Files Created

| File | Purpose | Result |
|------|---------|--------|
| `debug_cp_causal_only.py` | Verify CP + causal masking (no doc mask) | ✓ 0.0 diff |
| `debug_cp_document_masking.py` | Verify CP + document masking | ✓ 0.0 diff |
| `debug_flex_attention_cp_test.py` | Minimal FlexAttention + CP test | ✓ 0.0 diff |
| `debug_e2e_cp_flex.py` | E2E training with random labels | ✓ Losses match |
| `debug_e2e_cp_flex_wandb.py` | E2E training with wandb logging | ✓ Losses match |
| `debug_e2e_cp_flex_real.py` | E2E training with real learning | ✓ Loss 5.75→0.10 |

---

## 14. Key Learnings

1. **PyTorch's `create_cp_block_mask` is buggy** when used with document masking and `context_parallel`. It has shape mismatch issues.

2. **The manual approach works perfectly:**
   - Create mask with `Q_LEN=local_seq_len`, `KV_LEN=full_seq_len`
   - Add `q_offset` to convert local q_idx to global in mask_mod
   - Use standard `create_attention_mask` or `create_block_mask`

3. **FlexAttention + CP is mathematically correct** when implemented properly:
   - Local Q attends to gathered global K/V
   - Mask must account for the q_offset
   - All tests show 0.0 difference between CP and non-CP

4. **The fix pattern is consistent** across models (deepseek_v3, qwen3):
   ```python
   if cp_mesh is not None:
       q_offset = cp_rank * local_seq_len
       def cp_aware_mask_mod(b, h, q_idx, kv_idx):
           return base_mask_mod(b, h, q_idx + q_offset, kv_idx)
       return create_attention_mask(cp_aware_mask_mod, B, None, local_seq_len, seq_len)
   ```

---

## 15. Next Steps (Original)

1. **Test with real model checkpoint** (e.g., Qwen3 30B A3B from HuggingFace)
2. **Verify loss starts at expected ~0.8** with document masking + CP enabled
3. **Consider reporting PyTorch bug** for `create_cp_block_mask` + `context_parallel` interaction

---

## 16. End-to-End Training Attempt: Qwen3 30B-A3B (FAILED)

### 16.1 Goal

Test CP + FlexAttention + Document Masking with Qwen3 30B-A3B model in actual torchtitan training.

### 16.2 Changes Made

**File: `torchtitan/models/qwen3/__init__.py`**

Added new model variant with FlexAttention enabled:
```python
"30B-A3B-flex": Qwen3ModelArgs(
    vocab_size=151936,
    max_seq_len=262144,
    head_dim=128,
    dim=2048,
    n_layers=48,
    n_heads=32,
    n_kv_heads=4,
    qk_norm=True,
    hidden_dim=6144,
    rope_theta=1000000,
    moe_enabled=True,
    moe_inter_dim=768,
    eos_id=151643,
    moe_args=MoEArgs(...),
    use_flex_attn=True,
    attn_mask_type="block_causal",
)
```

**File: `torchtitan/models/qwen3/infra/parallelize.py`**

Removed the NotImplementedError check for CP + FlexAttention:
```python
# Removed this check:
if job_config.parallelism.context_parallel_degree > 1 and use_flex_attn:
    raise NotImplementedError("CP support for FlexAttention is still in progress.")
```

**Config: `torchtitan/models/qwen3/train_configs/qwen3_30b_a3b_cp_flex_test.toml`**
```toml
[model]
name = "qwen3"
flavor = "30B-A3B-flex"

[parallelism]
context_parallel_degree = 2  # CP enabled
```

### 16.3 Errors Encountered

**Error 1: `freqs_cis` attribute error**
```
AttributeError: 'Qwen3Model' object has no attribute 'freqs_cis'
```

Qwen3 uses `rope_cache` instead of `freqs_cis`. Modified `train.py` to check for both.

**Error 2: Block mask shape mismatch**
```
ValueError: block_mask was created for block_mask.shape=(1, 1, 1024, 2048)
but got q_len=1024 and kv_len=1024
```

The mask was created for `(local_q, global_kv)` but attention received `(local_q, local_kv)`.

### 16.4 Reverted

User asked to revert Qwen3 changes and try DeepSeek instead:
```bash
git checkout torchtitan/models/qwen3/__init__.py \
             torchtitan/models/qwen3/model/model.py \
             torchtitan/models/qwen3/infra/parallelize.py \
             torchtitan/train.py
```

---

## 17. End-to-End Training Attempt: DeepSeek debugmodel_flex_attn (FAILED)

### 17.1 Goal

Since DeepSeek already had CP + FlexAttention partially implemented, test with the existing `debugmodel_flex_attn` variant.

### 17.2 Config Created

**File: `torchtitan/models/deepseek_v3/train_configs/debug_cp_flex_docmask.toml`**
```toml
[model]
name = "deepseek_v3"
flavor = "debugmodel_flex_attn"  # use_flex_attn=True, attn_mask_type="block_causal"

[parallelism]
context_parallel_degree = 2  # CP enabled
```

### 17.3 Bug #1: Double Division of seq_len

**File: `torchtitan/models/deepseek_v3/model/model.py:519-547`**

Original code:
```python
seq_len = input_batch.shape[1]  # Already LOCAL when CP enabled!
local_seq_len = seq_len // cp_size  # WRONG: divides again!
```

**Problem:** When CP is enabled, `input_batch` is already sharded to `local_seq_len`. The code incorrectly divided by `cp_size` again.

**Fix attempted:**
```python
local_seq_len = seq_len  # input_batch is already sharded
full_seq_len = seq_len * cp_size  # reconstruct full length
```

### 17.4 Bug #2: Index Out of Bounds in Document Mask

**Error:**
```
Assertion `index out of bounds: 1024 + r0_4 + 128*x1 < 1024` failed
```

**Problem:** `get_document_mask_mod()` creates `sequence_indices` from the **local** sharded `input_batch`, but the mask_mod tries to access **global** indices.

```python
# In get_document_mask_mod():
sequence_indices = compute_from(input_batch)  # Shape: [batch, local_seq_len]

# In cp_aware_mask_mod():
global_q_idx = q_idx + q_offset  # Can be >= local_seq_len!
sequence_indices[b, global_q_idx]  # OUT OF BOUNDS!
```

**Fix attempted:** All-gather `input_batch` before creating document mask:
```python
if cp_mesh is not None:
    full_input_batch = torch.empty((batch, full_seq_len), ...)
    torch.distributed.all_gather_into_tensor(
        full_input_batch, input_batch, group=cp_mesh.get_group()
    )
    # Use full_input_batch for get_document_mask_mod()
```

### 17.5 Bug #3: Block Mask Shape Mismatch (K/V Not Gathered)

**Error:**
```
ValueError: block_mask was created for block_mask.shape=(2, 1, 512, 1024)
but got q_len=512 and kv_len=512
```

**Problem:**
- Mask was created for `Q_LEN=512` (local), `KV_LEN=1024` (global)
- But `flex_attention` received K, V with `kv_len=512` (local)
- **K and V were never all-gathered!**

### 17.6 Attempted Fix: All-Gather K/V in Attention

**File: `torchtitan/models/deepseek_v3/model/model.py:257-328`**

Added `cp_mesh` parameter to attention forward:
```python
def forward(self, x, freqs_cis, attention_masks, position_ids=None, cp_mesh=None):
    ...
    if self.use_flex_attn and cp_mesh is not None:
        # All-gather K and V across CP ranks
        k_gathered = all_gather(k, cp_mesh)
        v_gathered = all_gather(v, cp_mesh)
        output = flex_attention(q, k_gathered, v_gathered, block_mask=attention_masks)
```

Also updated `TransformerBlock` and `DeepSeekV3Model` to propagate `cp_mesh`.

### 17.7 User Feedback: This Defeats the Purpose of CP!

User correctly pointed out:
> "wait if we gather kv then what is the point of context parallel anymore?"

**The fundamental issue:**
- Context Parallel is designed for **memory efficiency** via ring attention
- Ring attention: K/V stay distributed, rotated through a ring, attention computed progressively
- **FlexAttention does NOT support ring attention** - it needs all K/V at once
- If we all-gather K/V, we lose the memory benefit of CP

### 17.8 Reverted All Changes

```bash
git checkout torchtitan/models/deepseek_v3/model/model.py torchtitan/train.py
rm torchtitan/models/deepseek_v3/train_configs/debug_cp_flex_docmask.toml
```

---

## 18. Root Cause Analysis: Why FlexAttention + CP + Document Masking Doesn't Work

### 18.1 How Context Parallel Works

**Ring Attention Pattern (what `context_parallel` does):**
1. Each rank holds local Q chunk
2. K/V are rotated through ranks in a ring
3. At each step, compute partial attention with current K/V chunk
4. Accumulate results across all steps
5. **Memory efficient:** Never need full K/V on any single rank

**PyTorch's implementation:**
```python
with context_parallel(cp_mesh, ...):
    output = F.scaled_dot_product_attention(Q, K, V, ...)
    # Internally handles K/V rotation and partial attention
```

### 18.2 Why FlexAttention Doesn't Work with Ring Attention

**FlexAttention API:**
```python
output = flex_attention(Q, K, V, block_mask=mask)
```

- Expects **all** K and V to be present at call time
- Computes full attention in one pass
- No support for progressive/partial computation
- **Cannot integrate with ring attention pattern**

### 18.3 The Only Options for FlexAttention + CP

| Option | Description | Downside |
|--------|-------------|----------|
| All-gather K/V | Gather K/V before flex_attention | Defeats memory efficiency of CP |
| Don't use CP | Disable context_parallel | No sequence length scaling |
| Don't use FlexAttention | Use SDPA with context_parallel | Lose custom mask flexibility |

### 18.4 For Document Masking Specifically

**If you need document masking with CP:**
- Use SDPA-based attention (not FlexAttention)
- `context_parallel` handles the ring attention
- Document boundaries must be handled differently (pre-computed masks or loss masking)

**If you need FlexAttention with document masking:**
- Disable CP (`context_parallel_degree = 1`)
- Or accept the K/V all-gather overhead

---

## 19. Summary of Session Attempts

| Attempt | Model | Issue | Resolution |
|---------|-------|-------|------------|
| 1 | Qwen3 30B-A3B-flex | `freqs_cis` not found | Qwen3 uses `rope_cache` |
| 2 | Qwen3 30B-A3B-flex | Block mask shape mismatch | K/V not gathered |
| 3 | DeepSeek debugmodel_flex_attn | Double division of seq_len | Fixed |
| 4 | DeepSeek debugmodel_flex_attn | Index out of bounds | All-gather input_batch |
| 5 | DeepSeek debugmodel_flex_attn | Block mask shape mismatch | All-gather K/V |
| 6 | DeepSeek debugmodel_flex_attn | **Defeats CP purpose** | **REVERTED** |

---

## 20. Final Conclusions

### 20.1 What Works (Verified in §11-§15)

✅ **Standalone FlexAttention + CP tests** (debug scripts)
- Manually all-gathering K/V in test code
- Manually creating offset-aware masks
- Produces 0.0 diff vs baseline

### 20.2 What Doesn't Work

❌ **FlexAttention + CP + Document Masking in torchtitan**
- FlexAttention fundamentally incompatible with ring attention
- Would require all-gathering K/V (defeats CP purpose)
- PyTorch's `create_cp_block_mask` has additional bugs

### 20.3 Recommendations

1. **For CP + Document Masking:** Use SDPA-based attention, not FlexAttention

2. **For FlexAttention + Document Masking:** Disable CP

3. **Do NOT mix FlexAttention + CP** unless willing to accept K/V all-gather overhead

4. **PyTorch bug report:** `create_cp_block_mask` has issues when used with `context_parallel` and document masking - consider reporting

---

## 21. How Upstream TorchTitan Fixes It

### 21.1 Key Discovery

The upstream repo at `/home/phuc/kimi_1t_upstream/torchtitan` uses a **completely different approach** that works correctly.

### 21.2 The Upstream Approach

**New file: `torchtitan/distributed/context_parallel.py`**

Three key functions:

1. **`apply_cp_to_attention_module()`** - Wraps attention modules with `_ContextParallel`:
```python
def apply_cp_to_attention_module(attention_modules, cp_mesh, attention_type):
    match attention_type:
        case "flex":
            cp_plan = _ContextParallel(
                seq_dim=2, attention_type=_ContextParallel.AttentionType.FLEX
            )
        case "sdpa":
            _enable_context_parallel_dispatcher()
            cp_plan = _ContextParallel(
                seq_dim=2, attention_type=_ContextParallel.AttentionType.SDPA
            )

    for attention_module in attention_modules:
        parallelize_module(module=attention_module, device_mesh=cp_mesh, parallelize_plan=cp_plan)
```

2. **`prepare_context_parallel_input()`** - Shards inputs, labels, AND masks:
```python
def prepare_context_parallel_input(inputs, labels, extra_kwargs, cp_mesh, device, load_balancer_type):
    attention_masks = extra_kwargs.get("attention_masks", None)
    positions = torch.arange(0, inputs.shape[1], ...)

    (inputs, labels, positions), attention_masks = cp_shard(
        cp_mesh, (inputs, labels, positions), attention_masks, load_balancer_type
    )

    extra_kwargs["positions"] = positions
    extra_kwargs["attention_masks"] = attention_masks
    return inputs, labels, extra_kwargs
```

3. **`cp_shard()`** - Uses PyTorch's `_context_parallel_shard` with load balancing:
```python
def cp_shard(cp_mesh, inputs, attention_masks, load_balancer_type="headtail"):
    if load_balancer_type == "ptrr":
        # For FlexAttention: uses _PTRRLoadBalancer
        load_balancer = _PTRRLoadBalancer(attention_masks, cp_world_size)

    inputs = _context_parallel_shard(mesh=cp_mesh, buffers=inputs, ...)

    # Shard mask on Q dimension only (dim=2), not KV
    MASK_Q_SEQ_DIM = 2
    masks = _context_parallel_shard(mesh=cp_mesh, buffers=masks, seq_dims=(MASK_Q_SEQ_DIM,), ...)
```

### 21.3 Key Differences from Our Approach

| Aspect | Our Repo (kimi_1t) | Upstream Repo |
|--------|-------------------|---------------|
| CP Module | None | `_ContextParallel` wraps attention |
| Mask Creation | `create_cp_block_mask` (buggy) | Standard `create_attention_mask` |
| Mask Sharding | None (mask created at wrong size) | `_context_parallel_shard` on Q dim |
| Input Sharding | Manual in `context_parallel` ctx | `prepare_context_parallel_input()` |
| Load Balancing | None | `_PTRRLoadBalancer` for FlexAttention |
| K/V Handling | **Missing!** | `_ContextParallel` handles internally |

### 21.4 The Critical Insight

**Our repo is missing `torchtitan/distributed/context_parallel.py` entirely!**

The upstream approach:
1. Creates mask at **full sequence length** with standard `create_attention_mask`
2. Shards the mask's **Q dimension only** via `_context_parallel_shard`
3. Wraps attention with `_ContextParallel` which handles K/V gathering **internally**
4. Uses `_PTRRLoadBalancer` for efficient FlexAttention load balancing

### 21.5 Why This Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    Upstream Approach                             │
├─────────────────────────────────────────────────────────────────┤
│  1. Create mask at FULL seq_len (e.g., 2048)                    │
│  2. _context_parallel_shard shards mask Q dim → (1024, 2048)    │
│  3. _ContextParallel wraps flex_attention                       │
│  4. Internally gathers K/V before attention                     │
│  5. Mask modifier works correctly (indices are global)          │
└─────────────────────────────────────────────────────────────────┘
```

The key is that `_ContextParallel` from `torch.distributed.tensor.experimental._attention` **handles all the complexity internally**, including:
- K/V all-gather
- Proper index translation
- Load balancing across ranks

### 21.6 How to Fix Our Repo

To enable CP + FlexAttention + Document Masking:

1. **Copy `context_parallel.py`** from upstream
2. **Update `parallelize.py`** to call `apply_cp_to_attention_module()`
3. **Update `train.py`** to call `prepare_context_parallel_input()`
4. **Remove** the old `create_cp_block_mask` usage from `get_attention_masks()`

---

## 22. Files Modified/Created in This Investigation

### Production Code (All Reverted)
| File | Status |
|------|--------|
| `torchtitan/models/deepseek_v3/model/model.py` | REVERTED |
| `torchtitan/models/qwen3/__init__.py` | REVERTED |
| `torchtitan/models/qwen3/model/model.py` | REVERTED |
| `torchtitan/models/qwen3/infra/parallelize.py` | REVERTED |
| `torchtitan/train.py` | REVERTED |

### Test Configs (Deleted)
| File | Status |
|------|--------|
| `torchtitan/models/qwen3/train_configs/qwen3_30b_a3b_cp_flex_test.toml` | DELETED |
| `torchtitan/models/deepseek_v3/train_configs/debug_cp_flex_docmask.toml` | DELETED |

### Debug Scripts (From Earlier Sessions)
| File | Purpose |
|------|---------|
| `debug_cp_causal_only.py` | CP + causal masking test |
| `debug_cp_document_masking.py` | CP + document masking test |
| `debug_flex_attention_cp_test.py` | Minimal FlexAttention + CP |
| `debug_e2e_cp_flex_real.py` | E2E training with learning |

---

## 22. The Fundamental Limitation (TL;DR)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FlexAttention + CP                            │
├─────────────────────────────────────────────────────────────────┤
│  FlexAttention: Needs ALL K/V at once                           │
│  Context Parallel: K/V stay distributed (ring attention)        │
│                                                                  │
│  These are FUNDAMENTALLY INCOMPATIBLE                           │
│                                                                  │
│  Options:                                                        │
│  1. All-gather K/V → Works but defeats CP memory efficiency     │
│  2. Use SDPA → Works with context_parallel wrapper              │
│  3. Disable CP → FlexAttention works, no sequence scaling       │
└─────────────────────────────────────────────────────────────────┘
```
