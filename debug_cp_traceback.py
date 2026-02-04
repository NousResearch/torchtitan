#!/usr/bin/env python3
"""
Get full traceback for the "list index out of range" error in context_parallel.
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental._attention import (
    context_parallel,
    create_cp_block_mask,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n[Rank {rank}] Testing context_parallel with flex_attention\n")

    # Configuration
    batch_size = 1
    seq_len = 256
    n_heads = 4
    head_dim = 64

    # Create CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )
    local_seq_len = seq_len // world_size

    # Create inputs
    torch.manual_seed(42 + rank)
    q = (
        torch.randn(
            batch_size,
            n_heads,
            local_seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    k = (
        torch.randn(
            batch_size,
            n_heads,
            local_seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    v = (
        torch.randn(
            batch_size,
            n_heads,
            local_seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )

    # Create CP block mask
    block_mask = create_cp_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device_mesh=cp_mesh,
    )
    print(f"[Rank {rank}] BlockMask shape: {block_mask.shape}")

    # Test with context_parallel - full traceback
    print(f"\n[Rank {rank}] === Calling context_parallel ===")
    try:
        with context_parallel(cp_mesh):
            out = flex_attention(q, k, v, block_mask=block_mask)
        print(f"[Rank {rank}] Output shape: {out.shape}")
    except Exception as e:
        print(f"\n[Rank {rank}] ======= FULL TRACEBACK =======")
        import traceback

        traceback.print_exc()
        print(f"[Rank {rank}] ======= END TRACEBACK =======\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
