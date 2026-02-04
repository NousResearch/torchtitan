#!/usr/bin/env python3
"""
Verify create_cp_block_mask produces correct masks for document masking.

This test checks if PyTorch's create_cp_block_mask correctly converts
local indices to global indices for document masking.

Run with: torchrun --nproc_per_node=2 debug_cp_mask_verify.py
"""

import os
import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask

try:
    from torch.distributed.tensor.experimental._attention import create_cp_block_mask
except ImportError:
    print("PyTorch version doesn't support create_cp_block_mask")
    sys.exit(1)


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def _get_document_ids_from_seq_lens(seq_lens, device):
    """Same as torchtitan implementation."""
    batch_size = len(seq_lens)
    batch_document_ids = []
    for sample_idx in range(batch_size):
        document_ids = torch.cat(
            [
                torch.full((seq_len,), i, dtype=torch.long, device=device)
                for i, seq_len in enumerate(seq_lens[sample_idx])
            ]
        )
        batch_document_ids.append(document_ids)
    return torch.stack(batch_document_ids)


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    print(f"\n{'='*70}")
    print(f"[Rank {rank}] Verifying create_cp_block_mask for document masking")
    print(f"{'='*70}\n")

    # Configuration - small enough to visualize
    batch_size = 1
    n_heads = 1
    full_seq_len = 256
    local_seq_len = full_seq_len // world_size

    # Create CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )

    # Document structure: 4 documents of equal size
    doc_sizes = [64, 64, 64, 64]
    seq_lens = [[torch.tensor(s, device=device) for s in doc_sizes]]
    document_ids = _get_document_ids_from_seq_lens(seq_lens, device)

    print(f"[Rank {rank}] full_seq_len={full_seq_len}, local_seq_len={local_seq_len}")
    print(f"[Rank {rank}] Document sizes: {doc_sizes}")
    print(f"[Rank {rank}] Document IDs shape: {document_ids.shape}")

    # Create mask_mod for document masking
    def document_causal_mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        doc_match = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal & doc_match

    # ========== Test 1: Create mask with create_cp_block_mask ==========
    print(f"\n[Rank {rank}] === Creating mask with create_cp_block_mask ===")

    cp_mask = create_cp_block_mask(
        mask_mod=document_causal_mask_mod,
        B=batch_size,
        H=n_heads,
        Q_LEN=full_seq_len,
        KV_LEN=full_seq_len,
        device_mesh=cp_mesh,
    )

    print(f"[Rank {rank}] CP mask shape: {cp_mask.shape}")
    print(f"[Rank {rank}] CP mask seq_lengths: {cp_mask.seq_lengths}")

    # Get dense mask for verification
    try:
        cp_dense = cp_mask.to_dense()
        print(f"[Rank {rank}] CP dense mask shape: {cp_dense.shape}")
    except Exception as e:
        print(f"[Rank {rank}] Could not get dense CP mask: {e}")
        cp_dense = None

    # ========== Test 2: Create reference mask without CP ==========
    print(f"\n[Rank {rank}] === Creating reference mask (non-CP) ===")

    # For reference, create a mask for local Q vs global KV
    local_start = rank * local_seq_len

    def reference_mask_mod(b, h, q_idx, kv_idx):
        # q_idx is local (0 to local_seq_len-1), need to add offset
        global_q_idx = q_idx + local_start
        causal = global_q_idx >= kv_idx
        doc_match = document_ids[b, global_q_idx] == document_ids[b, kv_idx]
        return causal & doc_match

    ref_mask = create_block_mask(
        mask_mod=reference_mask_mod,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=full_seq_len,
        device=device,
    )

    print(f"[Rank {rank}] Reference mask shape: {ref_mask.shape}")

    try:
        ref_dense = ref_mask.to_dense()
        print(f"[Rank {rank}] Reference dense mask shape: {ref_dense.shape}")
    except Exception as e:
        print(f"[Rank {rank}] Could not get dense reference mask: {e}")
        ref_dense = None

    # ========== Test 3: Compare masks ==========
    print(f"\n[Rank {rank}] === Comparing masks ===")

    if cp_dense is not None and ref_dense is not None:
        # Both should have shape [B, H, local_seq_len, full_seq_len] for comparison
        # But cp_dense might have different shape
        print(f"[Rank {rank}] CP dense shape: {cp_dense.shape}")
        print(f"[Rank {rank}] Ref dense shape: {ref_dense.shape}")

        if cp_dense.shape == ref_dense.shape:
            diff = (cp_dense != ref_dense).sum().item()
            total = cp_dense.numel()
            print(
                f"[Rank {rank}] Differences: {diff} / {total} ({100*diff/total:.2f}%)"
            )

            if diff > 0:
                print(f"[Rank {rank}] ✗ MASKS DIFFER!")
                # Find first difference
                diff_mask = cp_dense != ref_dense
                diff_indices = torch.nonzero(diff_mask, as_tuple=True)
                if len(diff_indices[0]) > 0:
                    b, h, q, kv = [idx[0].item() for idx in diff_indices]
                    print(
                        f"[Rank {rank}] First diff at: b={b}, h={h}, q={q} (global={q+local_start}), kv={kv}"
                    )
                    print(
                        f"[Rank {rank}]   CP mask value: {cp_dense[b, h, q, kv].item()}"
                    )
                    print(
                        f"[Rank {rank}]   Ref mask value: {ref_dense[b, h, q, kv].item()}"
                    )
                    # Check what document they belong to
                    global_q = q + local_start
                    q_doc = document_ids[b, global_q].item()
                    kv_doc = document_ids[b, kv].item()
                    print(
                        f"[Rank {rank}]   q_doc={q_doc}, kv_doc={kv_doc}, global_q >= kv: {global_q >= kv}"
                    )
            else:
                print(f"[Rank {rank}] ✓ MASKS MATCH!")
        else:
            print(f"[Rank {rank}] Shapes differ, cannot directly compare")

            # Try to understand the CP mask structure
            print(f"\n[Rank {rank}] Analyzing CP mask structure:")

            # Check a few specific positions
            test_positions = [
                (0, 0),
                (local_seq_len // 2, local_seq_len // 2),
                (local_seq_len - 1, full_seq_len - 1),
            ]
            for local_q, kv in test_positions:
                global_q = local_q + local_start
                expected_causal = global_q >= kv
                expected_doc = document_ids[0, global_q] == document_ids[0, kv]
                expected = expected_causal and expected_doc

                # Try to get CP mask value
                try:
                    if cp_dense.shape[2] > local_q and cp_dense.shape[3] > kv:
                        cp_val = cp_dense[0, 0, local_q, kv].item()
                    else:
                        cp_val = "OOB"
                except Exception:
                    cp_val = "ERR"

                ref_val = (
                    ref_dense[0, 0, local_q, kv].item()
                    if local_q < ref_dense.shape[2] and kv < ref_dense.shape[3]
                    else "OOB"
                )

                print(
                    f"[Rank {rank}] local_q={local_q}, global_q={global_q}, kv={kv}: "
                    f"expected={int(expected)}, cp={cp_val}, ref={ref_val}"
                )

    # ========== Visualize small portion ==========
    print(f"\n[Rank {rank}] === Visualizing first 16x16 of masks ===")

    def visualize(dense, name, q_offset=0):
        if dense is None:
            print(f"[Rank {rank}] {name}: None")
            return
        h = min(16, dense.shape[2])
        w = min(16, dense.shape[3])
        print(
            f"[Rank {rank}] {name} (local_q={0}-{h-1}, global_q={q_offset}-{q_offset+h-1}, kv=0-{w-1}):"
        )
        for i in range(h):
            row = "".join(["█" if dense[0, 0, i, j].item() else "·" for j in range(w)])
            print(f"[Rank {rank}]   {row}")

    visualize(cp_dense, "CP mask", local_start)
    visualize(ref_dense, "Reference mask", local_start)

    dist.barrier()

    if rank == 0:
        print(f"\n{'='*70}")
        print("Test complete. Check if CP mask matches Reference mask.")
        print("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
