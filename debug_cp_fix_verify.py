#!/usr/bin/env python3
"""
Verify the CP document masking fix.

This test:
1. Creates a mask using the fixed approach (manual offset handling)
2. Compares with the correct reference mask
3. Runs end-to-end attention comparison

Run with: python debug_cp_fix_verify.py (single GPU, simulates CP)
"""

import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
)


def _get_document_ids_from_seq_lens(seq_lens, device):
    batch_document_ids = []
    for sample_lens in seq_lens:
        doc_ids = torch.cat(
            [
                torch.full((l.item(),), i, dtype=torch.long, device=device)
                for i, l in enumerate(sample_lens)
            ]
        )
        batch_document_ids.append(doc_ids)
    return torch.stack(batch_document_ids)


def get_causal_mask_mod():
    def _causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return _causal_mask


def get_block_causal_mask_mod_by_seq_lens(seq_lens, device):
    document_ids = _get_document_ids_from_seq_lens(seq_lens, device)

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal_mask & document_mask

    return mask_mod


def main():
    print("=" * 70)
    print("Verifying CP Document Masking Fix")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Config
    batch_size = 1
    n_heads = 4
    head_dim = 32
    dim = n_heads * head_dim
    full_seq_len = 256
    cp_degree = 2
    local_seq_len = full_seq_len // cp_degree

    # Documents: 4 docs of 64 tokens each
    doc_sizes = [64, 64, 64, 64]
    seq_lens = [[torch.tensor(s, device=device) for s in doc_sizes]]

    print(f"\nConfig:")
    print(
        f"  full_seq_len={full_seq_len}, cp_degree={cp_degree}, local_seq_len={local_seq_len}"
    )
    print(f"  Documents: {doc_sizes}")

    # Create mask_mods
    causal_mask = get_causal_mask_mod()
    doc_mask = get_block_causal_mask_mod_by_seq_lens(seq_lens, device)
    mask_mods = [causal_mask, doc_mask]

    # Create input
    torch.manual_seed(42)
    x_full = torch.randn(
        batch_size, full_seq_len, dim, device=device, dtype=torch.bfloat16
    )

    # Simple model
    torch.manual_seed(123)
    wq = torch.randn(dim, dim, device=device, dtype=torch.bfloat16) * 0.02
    wk = torch.randn(dim, dim, device=device, dtype=torch.bfloat16) * 0.02
    wv = torch.randn(dim, dim, device=device, dtype=torch.bfloat16) * 0.02

    # Compute Q, K, V for full sequence
    q_full = (
        (x_full @ wq).view(batch_size, full_seq_len, n_heads, head_dim).transpose(1, 2)
    )
    k_full = (
        (x_full @ wk).view(batch_size, full_seq_len, n_heads, head_dim).transpose(1, 2)
    )
    v_full = (
        (x_full @ wv).view(batch_size, full_seq_len, n_heads, head_dim).transpose(1, 2)
    )

    print("\n" + "=" * 50)
    print("TEST 1: Baseline (full sequence, no CP)")
    print("=" * 50)

    baseline_mask = create_block_mask(
        and_masks(*mask_mods),
        B=batch_size,
        H=n_heads,
        Q_LEN=full_seq_len,
        KV_LEN=full_seq_len,
        device=device,
    )
    out_baseline = flex_attention(q_full, k_full, v_full, block_mask=baseline_mask)
    print(f"Baseline output shape: {out_baseline.shape}")

    all_passed = True

    for rank in range(cp_degree):
        print(f"\n{'='*50}")
        print(f"TEST 2.{rank}: Simulated CP rank {rank}")
        print("=" * 50)

        local_start = rank * local_seq_len
        q_offset = local_start

        # Local Q (sharded)
        q_local = q_full[:, :, local_start : local_start + local_seq_len, :]

        # Global K, V (all-gathered)
        k_global = k_full
        v_global = v_full

        print(f"Rank {rank}: q_local shape = {q_local.shape}")
        print(f"Rank {rank}: k_global shape = {k_global.shape}")
        print(f"Rank {rank}: q_offset = {q_offset}")

        # THE FIX: Create mask with offset-aware mask_mod
        def cp_aware_mask_mod(b, h, q_idx, kv_idx):
            global_q_idx = q_idx + q_offset
            combined = and_masks(*mask_mods)
            return combined(b, h, global_q_idx, kv_idx)

        # Mask for local Q vs global KV
        fixed_mask = create_block_mask(
            cp_aware_mask_mod,
            B=batch_size,
            H=n_heads,
            Q_LEN=local_seq_len,
            KV_LEN=full_seq_len,
            device=device,
        )

        print(f"Rank {rank}: Fixed mask shape: {fixed_mask.shape}")

        # Run attention
        out_fixed = flex_attention(q_local, k_global, v_global, block_mask=fixed_mask)
        print(f"Rank {rank}: Fixed output shape: {out_fixed.shape}")

        # Compare with baseline's local portion
        out_baseline_local = out_baseline[
            :, :, local_start : local_start + local_seq_len, :
        ]
        diff = (out_fixed - out_baseline_local).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Rank {rank}: max_diff = {max_diff:.8f}, mean_diff = {mean_diff:.8f}")

        if max_diff < 1e-5:
            print(f"Rank {rank}: ✓ PASSED - Fix produces correct output!")
        else:
            print(f"Rank {rank}: ✗ FAILED - Output differs from baseline!")
            all_passed = False

        # Also verify some mask values
        try:
            fixed_dense = fixed_mask.to_dense()
            test_cases = [
                (0, 0, "first to first"),
                (
                    local_seq_len // 2,
                    local_start + local_seq_len // 2,
                    "middle to middle",
                ),
            ]
            for lq, kv, desc in test_cases:
                if lq < fixed_dense.shape[2] and kv < fixed_dense.shape[3]:
                    val = fixed_dense[0, 0, lq, kv].item()
                    gq = lq + local_start
                    expected_causal = gq >= kv
                    # Get document IDs
                    doc_ids = _get_document_ids_from_seq_lens(seq_lens, device)
                    expected_doc = (doc_ids[0, gq] == doc_ids[0, kv]).item()
                    expected = expected_causal and expected_doc
                    status = "✓" if val == expected else "✗"
                    print(
                        f"  Mask[lq={lq}, kv={kv}] ({desc}): val={int(val)}, expected={int(expected)} {status}"
                    )
        except Exception as e:
            print(f"  Could not verify mask: {e}")

    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL TESTS PASSED - The fix is correct!")
    else:
        print("✗ SOME TESTS FAILED - Check the output above")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
