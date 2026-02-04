#!/usr/bin/env python3
"""
Debug the block_mask handling in context_parallel for flex_attention.
"""

import os

import torch
import torch._higher_order_ops.flex_attention as flex_hop
import torch.distributed as dist
from torch.distributed.tensor.experimental._attention import create_cp_block_mask
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


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


def print_block_mask_tuple(name, bm_tuple, rank):
    """Print block_mask tuple contents."""
    print(f"\n[Rank {rank}] === {name} ===")
    print(f"[Rank {rank}] Tuple length: {len(bm_tuple)}")
    for i, item in enumerate(bm_tuple):
        if isinstance(item, torch.Tensor):
            print(
                f"[Rank {rank}] [{i}] Tensor: shape={tuple(item.shape)}, dtype={item.dtype}, "
                f"min={item.float().min().item():.4f}, max={item.float().max().item():.4f}"
            )
        elif item is None:
            print(f"[Rank {rank}] [{i}] None")
        elif callable(item):
            print(f"[Rank {rank}] [{i}] Callable: {item}")
        else:
            print(f"[Rank {rank}] [{i}] {type(item).__name__}: {item}")


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Block Mask Debug for CP + FlexAttention")
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
    local_start = rank * local_seq_len

    print(
        f"[Rank {rank}] seq_len={seq_len}, local_seq_len={local_seq_len}, local_start={local_start}"
    )

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
    cp_block_mask = create_cp_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device_mesh=cp_mesh,
    )

    print(f"\n[Rank {rank}] CP BlockMask object: {type(cp_block_mask)}")
    print(f"[Rank {rank}] CP BlockMask shape: {cp_block_mask.shape}")

    # Convert to tuple (this is what gets passed to the HOP)
    cp_mask_tuple = cp_block_mask.as_tuple()
    print_block_mask_tuple("CP BlockMask tuple", cp_mask_tuple, rank)

    # Now manually gather K, V
    import torch.distributed._functional_collectives as ft_c

    global_k = ft_c.all_gather_tensor(
        k.contiguous(), gather_dim=2, group=cp_mesh.get_group()
    )
    global_v = ft_c.all_gather_tensor(
        v.contiguous(), gather_dim=2, group=cp_mesh.get_group()
    )

    print(f"\n[Rank {rank}] Global K shape: {global_k.shape}")
    print(f"[Rank {rank}] Global V shape: {global_v.shape}")

    # This is what context_parallel does to the block_mask:
    # if block_mask[1] != global_key.size(-2):
    #     block_mask = (block_mask[0], global_key.size(-2), *block_mask[2:])
    print(f"\n[Rank {rank}] Original block_mask[1]: {cp_mask_tuple[1]}")
    print(f"[Rank {rank}] global_k.size(-2): {global_k.size(-2)}")

    if cp_mask_tuple[1] != global_k.size(-2):
        modified_mask_tuple = (cp_mask_tuple[0], global_k.size(-2), *cp_mask_tuple[2:])
        print(f"[Rank {rank}] Block mask tuple MODIFIED")
    else:
        modified_mask_tuple = cp_mask_tuple
        print(f"[Rank {rank}] Block mask tuple NOT modified (already correct)")

    print_block_mask_tuple("Modified BlockMask tuple", modified_mask_tuple, rank)

    # Skip HOP direct calls, just test high-level API

    # Now test with the high-level flex_attention API
    print(f"\n[Rank {rank}] === Testing high-level flex_attention API ===")

    # Create a proper BlockMask for local Q vs global KV
    def cp_causal_mask(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + local_start
        return global_q_idx >= kv_idx

    proper_mask = create_block_mask(
        cp_causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=seq_len,
        device=device,
    )
    print(f"[Rank {rank}] Proper mask shape: {proper_mask.shape}")

    print(f"\n[Rank {rank}] --- Non-compiled high-level API ---")
    try:
        out1 = flex_attention(q, global_k, global_v, block_mask=proper_mask)
        check_tensor("High-level non-compiled", out1, rank)
    except Exception as e:
        print(f"[Rank {rank}] High-level non-compiled error: {e}")

    print(f"\n[Rank {rank}] --- Compiled high-level API (dynamic=True) ---")
    compiled_flex = torch.compile(flex_attention, dynamic=True)
    try:
        out2 = compiled_flex(
            q.clone(), global_k.clone(), global_v.clone(), block_mask=proper_mask
        )
        check_tensor("High-level compiled", out2, rank)
    except Exception as e:
        print(f"[Rank {rank}] High-level compiled error: {e}")

    print(f"\n[Rank {rank}] === Debug Complete ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
