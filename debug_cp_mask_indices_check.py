#!/usr/bin/env python3
"""
Diagnose what indices create_cp_block_mask passes to mask_mod.

Run with: torchrun --nproc_per_node=2 debug_cp_mask_indices_check.py
"""

import os

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor.experimental._attention import create_cp_block_mask
except ImportError:
    print("PyTorch version doesn't support create_cp_block_mask")
    exit(1)


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    full_seq_len = 16
    local_seq_len = full_seq_len // world_size

    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )

    print(f"\n[Rank {rank}] full_seq_len={full_seq_len}, local_seq_len={local_seq_len}")

    # Track what indices are passed to mask_mod
    q_indices_seen = []
    kv_indices_seen = []

    def diagnostic_mask_mod(b, h, q_idx, kv_idx):
        # Record the indices (only on first call pattern)
        if len(q_indices_seen) < 100:
            q_indices_seen.append(
                q_idx.item() if q_idx.numel() == 1 else q_idx.tolist()
            )
            kv_indices_seen.append(
                kv_idx.item() if kv_idx.numel() == 1 else kv_idx.tolist()
            )
        return q_idx >= kv_idx

    try:
        mask = create_cp_block_mask(
            mask_mod=diagnostic_mask_mod,
            B=1,
            H=1,
            Q_LEN=full_seq_len,
            KV_LEN=full_seq_len,
            device_mesh=cp_mesh,
        )

        print(f"[Rank {rank}] Mask created successfully")
        print(f"[Rank {rank}] Mask shape: {mask.shape}")

        # Print observed indices
        if q_indices_seen:
            print(
                f"[Rank {rank}] Sample q_indices seen (first 20): {q_indices_seen[:20]}"
            )
            print(
                f"[Rank {rank}] Sample kv_indices seen (first 20): {kv_indices_seen[:20]}"
            )

            # Check ranges
            flat_q = []
            for x in q_indices_seen[:50]:
                if isinstance(x, list):
                    flat_q.extend(x)
                else:
                    flat_q.append(x)

            if flat_q:
                print(
                    f"[Rank {rank}] q_idx range: min={min(flat_q)}, max={max(flat_q)}"
                )

            flat_kv = []
            for x in kv_indices_seen[:50]:
                if isinstance(x, list):
                    flat_kv.extend(x)
                else:
                    flat_kv.append(x)

            if flat_kv:
                print(
                    f"[Rank {rank}] kv_idx range: min={min(flat_kv)}, max={max(flat_kv)}"
                )

        # Try to get dense mask
        try:
            dense = mask.to_dense()
            print(f"[Rank {rank}] Dense mask shape: {dense.shape}")
            print(f"[Rank {rank}] Dense mask [0,0]:\n{dense[0,0].int()}")
        except Exception as e:
            print(f"[Rank {rank}] Could not get dense mask: {e}")

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
