#!/usr/bin/env python3
"""
Distributed test for document masking with Context Parallel.
Verifies the fix produces correct attention outputs.

Run with: torchrun --nproc_per_node=2 debug_document_mask_cp_distributed.py
"""

import os
import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import and_masks, flex_attention

# Import the fixed functions
from torchtitan.models.attention import (
    _get_document_ids_from_seq_lens,
    apply_cp_offset_to_mask_mod,
)

try:
    from torch.distributed.tensor.experimental._attention import (
        context_parallel,
        create_cp_block_mask,
    )
except ImportError:
    print("PyTorch version doesn't support create_cp_block_mask")
    sys.exit(1)


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def get_causal_mask_mod():
    def _causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return _causal_mask


def get_block_causal_mask_mod_by_seq_lens(seq_lens):
    document_ids = _get_document_ids_from_seq_lens(seq_lens)

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal_mask & document_mask

    return mask_mod


def check_tensor(name, tensor, rank):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.float().min().item()
    max_val = tensor.float().max().item()
    status = "OK" if not (has_nan or has_inf) else "NaN!" if has_nan else "Inf!"
    print(
        f"[Rank {rank}] {name}: shape={tuple(tensor.shape)}, min={min_val:.4f}, max={max_val:.4f} {status}"
    )
    return has_nan or has_inf


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    print(f"\n{'='*70}")
    print(f"[Rank {rank}] Document Mask + CP Distributed Test")
    print(f"{'='*70}\n")

    # Configuration
    batch_size = 1
    full_seq_len = 256
    n_heads = 4
    head_dim = 64

    local_seq_len = full_seq_len // world_size

    # Create CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )

    # Create document structure: 4 documents of varying sizes
    doc_sizes = [64, 64, 64, 64]  # 4 equal docs for simplicity
    seq_lens = [[torch.tensor(s, device=device) for s in doc_sizes]]

    print(f"[Rank {rank}] full_seq_len={full_seq_len}, local_seq_len={local_seq_len}")
    print(f"[Rank {rank}] Document sizes: {doc_sizes}")

    # Create Q, K, V - same seed across ranks for reproducibility
    torch.manual_seed(42)
    q_full = (
        torch.randn(
            batch_size,
            n_heads,
            full_seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    k_full = (
        torch.randn(
            batch_size,
            n_heads,
            full_seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    v_full = (
        torch.randn(
            batch_size,
            n_heads,
            full_seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )

    # Shard Q for this rank
    local_start = rank * local_seq_len
    q_local = q_full[:, :, local_start : local_start + local_seq_len, :].clone()
    k_local = k_full[:, :, local_start : local_start + local_seq_len, :].clone()
    v_local = v_full[:, :, local_start : local_start + local_seq_len, :].clone()

    # Create mask_mods
    mask_mods = [get_causal_mask_mod()]
    mask_mods.append(get_block_causal_mask_mod_by_seq_lens(seq_lens))
    combined_mask_mod = and_masks(*mask_mods)

    # ========== Test 1: BUGGY - without CP offset fix ==========
    print(f"\n[Rank {rank}] === Test 1: BUGGY (no offset fix) ===")

    buggy_cp_mask = create_cp_block_mask(
        mask_mod=combined_mask_mod,  # Not fixed!
        B=batch_size,
        H=n_heads,
        Q_LEN=full_seq_len,
        KV_LEN=full_seq_len,
        device_mesh=cp_mesh,
    )

    try:
        with context_parallel(cp_mesh):
            buggy_out = flex_attention(
                q_local.clone(),
                k_local.clone(),
                v_local.clone(),
                block_mask=buggy_cp_mask,
            )
        check_tensor("Buggy output", buggy_out, rank)
    except Exception as e:
        print(f"[Rank {rank}] Buggy test error: {e}")
        buggy_out = None

    # ========== Test 2: FIXED - with CP offset fix ==========
    print(f"\n[Rank {rank}] === Test 2: FIXED (with offset fix) ===")

    # Apply the fix
    cp_rank = cp_mesh.get_local_rank()
    cp_size = cp_mesh.size()
    q_offset = cp_rank * local_seq_len
    fixed_mask_mod = apply_cp_offset_to_mask_mod(combined_mask_mod, q_offset)

    fixed_cp_mask = create_cp_block_mask(
        mask_mod=fixed_mask_mod,  # Fixed!
        B=batch_size,
        H=n_heads,
        Q_LEN=full_seq_len,
        KV_LEN=full_seq_len,
        device_mesh=cp_mesh,
    )

    try:
        with context_parallel(cp_mesh):
            fixed_out = flex_attention(
                q_local.clone(),
                k_local.clone(),
                v_local.clone(),
                block_mask=fixed_cp_mask,
            )
        check_tensor("Fixed output", fixed_out, rank)
    except Exception as e:
        print(f"[Rank {rank}] Fixed test error: {e}")
        fixed_out = None

    # ========== Test 3: Reference - non-CP baseline ==========
    print(f"\n[Rank {rank}] === Test 3: Reference (non-CP baseline) ===")

    # Create reference output without CP - each rank computes for its local Q against full K,V
    from torch.nn.attention.flex_attention import create_block_mask

    # For local Q vs full KV, we need a mask that handles the offset
    def reference_mask_mod(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + local_start
        document_ids = _get_document_ids_from_seq_lens(seq_lens)
        causal_mask = global_q_idx >= kv_idx
        document_mask = document_ids[b, global_q_idx] == document_ids[b, kv_idx]
        return causal_mask & document_mask

    ref_mask = create_block_mask(
        reference_mask_mod,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=full_seq_len,
        device=device,
    )

    try:
        ref_out = flex_attention(
            q_local.clone(), k_full.clone(), v_full.clone(), block_mask=ref_mask
        )
        check_tensor("Reference output", ref_out, rank)
    except Exception as e:
        print(f"[Rank {rank}] Reference test error: {e}")
        ref_out = None

    # ========== Compare outputs ==========
    print(f"\n[Rank {rank}] === Comparison ===")

    if fixed_out is not None and ref_out is not None:
        diff = (fixed_out - ref_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(
            f"[Rank {rank}] Fixed vs Reference: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        )

        if max_diff < 0.01:
            print(f"[Rank {rank}] ✓ Fixed output matches reference!")
        else:
            print(f"[Rank {rank}] ✗ Fixed output differs from reference")

    if buggy_out is not None and ref_out is not None:
        diff = (buggy_out - ref_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(
            f"[Rank {rank}] Buggy vs Reference: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        )

    dist.barrier()

    if rank == 0:
        print(f"\n{'='*70}")
        print("Test complete. If Fixed matches Reference, the fix is working!")
        print("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
