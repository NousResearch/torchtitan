#!/usr/bin/env python3
"""
Minimal debug to find exact NaN source in CP + flex_attention.
"""

import os

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


def check_tensor(name, tensor, rank):
    if tensor is None:
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.float().min().item()
    max_val = tensor.float().max().item()
    status = "✓" if not (has_nan or has_inf) else "✗ NaN!" if has_nan else "✗ Inf!"
    print(
        f"[Rank {rank}] {name}: shape={tuple(tensor.shape)}, min={min_val:.6f}, max={max_val:.6f} {status}"
    )
    return has_nan or has_inf


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Minimal CP + FlexAttention Debug")
    print(f"{'='*60}\n")

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

    check_tensor("Input Q", q, rank)
    check_tensor("Input K", k, rank)
    check_tensor("Input V", v, rank)

    # Create CP block mask
    block_mask = create_cp_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device_mesh=cp_mesh,
    )
    print(f"[Rank {rank}] BlockMask created")

    # Use buffers=None to let context_parallel use default behavior
    print(f"\n[Rank {rank}] === TEST: flex_attention with context_parallel ===")

    # Test non-compiled first
    print(f"\n[Rank {rank}] --- Non-compiled ---")
    try:
        with context_parallel(cp_mesh):
            out_nc = flex_attention(q, k, v, block_mask=block_mask)
        check_tensor("Non-compiled output", out_nc, rank)
    except Exception as e:
        print(f"[Rank {rank}] Non-compiled error: {e}")

    # Test compiled with dynamic=True
    print(f"\n[Rank {rank}] --- Compiled dynamic=True ---")
    compiled_flex = torch.compile(flex_attention, dynamic=True)
    try:
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        with context_parallel(cp_mesh):
            out_c = compiled_flex(q2, k2, v2, block_mask=block_mask)
        check_tensor("Compiled output", out_c, rank)
    except Exception as e:
        print(f"[Rank {rank}] Compiled error: {e}")

    # Test: manually gather K,V and call flex_attention directly
    print(f"\n[Rank {rank}] --- Manual gather + flex_attention ---")
    import torch.distributed._functional_collectives as ft_c

    try:
        global_k = ft_c.all_gather_tensor(
            k.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )
        global_v = ft_c.all_gather_tensor(
            v.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )

        check_tensor("Global K", global_k, rank)
        check_tensor("Global V", global_v, rank)

        # Create a proper block mask for local Q vs global KV
        # This mask should map local q_idx to global q_idx
        local_start = rank * local_seq_len

        def cp_causal_mask(b, h, q_idx, kv_idx):
            # q_idx is local [0, local_seq_len), need to map to global
            global_q_idx = q_idx + local_start
            return global_q_idx >= kv_idx

        manual_mask = create_block_mask(
            cp_causal_mask,
            B=batch_size,
            H=n_heads,
            Q_LEN=local_seq_len,
            KV_LEN=seq_len,
            device=device,
        )
        print(f"[Rank {rank}] Manual mask shape: {manual_mask.shape}")

        # Call flex_attention directly (non-compiled)
        out_manual = flex_attention(q, global_k, global_v, block_mask=manual_mask)
        check_tensor("Manual gather output (non-compiled)", out_manual, rank)

        # Call flex_attention (compiled)
        out_manual_c = compiled_flex(
            q.clone(), global_k.clone(), global_v.clone(), block_mask=manual_mask
        )
        check_tensor("Manual gather output (compiled)", out_manual_c, rank)

    except Exception as e:
        print(f"[Rank {rank}] Manual gather error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n[Rank {rank}] === Debug Complete ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
