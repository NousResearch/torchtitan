#!/usr/bin/env python3
"""
Compare CP block_mask created by create_cp_block_mask() vs manual create_block_mask()
to find why one produces NaN in torchtitan while the other works.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental._attention import (
    context_parallel,
    create_cp_block_mask,
)
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
    status = "OK" if not (has_nan or has_inf) else "NaN!" if has_nan else "Inf!"
    print(
        f"[Rank {rank}] {name}: shape={tuple(tensor.shape)}, min={min_val:.6f}, max={max_val:.6f} {status}"
    )
    return has_nan or has_inf


def print_block_mask_details(name, bm, rank):
    """Print detailed BlockMask information."""
    print(f"\n[Rank {rank}] === {name} ===")
    print(f"[Rank {rank}] shape: {bm.shape}")

    # Get tuple representation
    bm_tuple = bm.as_tuple()
    print(f"[Rank {rank}] Tuple length: {len(bm_tuple)}")

    # Key elements
    print(f"[Rank {rank}] [0] Q_LEN: {bm_tuple[0]}")
    print(f"[Rank {rank}] [1] KV_LEN: {bm_tuple[1]}")

    # kv_num_blocks and kv_indices
    if isinstance(bm_tuple[2], torch.Tensor):
        print(
            f"[Rank {rank}] [2] kv_num_blocks: shape={tuple(bm_tuple[2].shape)}, values={bm_tuple[2].flatten()[:10].tolist()}"
        )
    if isinstance(bm_tuple[3], torch.Tensor):
        print(
            f"[Rank {rank}] [3] kv_indices: shape={tuple(bm_tuple[3].shape)}, sample={bm_tuple[3][0,0,:5,:5] if bm_tuple[3].dim() == 4 else bm_tuple[3].flatten()[:10].tolist()}"
        )

    # full_kv_num_blocks and full_kv_indices
    if isinstance(bm_tuple[4], torch.Tensor):
        print(f"[Rank {rank}] [4] full_kv_num_blocks: shape={tuple(bm_tuple[4].shape)}")
    elif bm_tuple[4] is None:
        print(f"[Rank {rank}] [4] full_kv_num_blocks: None")

    if isinstance(bm_tuple[5], torch.Tensor):
        print(f"[Rank {rank}] [5] full_kv_indices: shape={tuple(bm_tuple[5].shape)}")
    elif bm_tuple[5] is None:
        print(f"[Rank {rank}] [5] full_kv_indices: None")

    # q_num_blocks and q_indices
    if isinstance(bm_tuple[6], torch.Tensor):
        print(
            f"[Rank {rank}] [6] q_num_blocks: shape={tuple(bm_tuple[6].shape)}, values={bm_tuple[6].flatten()[:10].tolist()}"
        )
    if isinstance(bm_tuple[7], torch.Tensor):
        print(f"[Rank {rank}] [7] q_indices: shape={tuple(bm_tuple[7].shape)}")

    # Block sizes
    print(f"[Rank {rank}] [10] Q_BLOCK_SIZE: {bm_tuple[10]}")
    print(f"[Rank {rank}] [11] KV_BLOCK_SIZE: {bm_tuple[11]}")

    # Mask mod
    print(f"[Rank {rank}] [12] mask_mod: {bm_tuple[12]}")

    # Try to get dense representation
    try:
        dense = bm.to_dense()
        print(f"[Rank {rank}] Dense shape: {dense.shape}")
        print(f"[Rank {rank}] Dense sum: {dense.sum().item()}")
        print(f"[Rank {rank}] Dense[0,0,:8,:8]:\n{dense[0,0,:8,:8].int()}")
    except Exception as e:
        print(f"[Rank {rank}] Could not get dense: {e}")


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*70}")
    print(f"[Rank {rank}] Comparing CP BlockMask vs Manual BlockMask")
    print(f"{'='*70}\n")

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

    # ============ Create the two masks ============

    # 1. CP block mask (what torchtitan uses)
    cp_block_mask = create_cp_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device_mesh=cp_mesh,
    )
    print_block_mask_details("CP BlockMask (create_cp_block_mask)", cp_block_mask, rank)

    # 2. Manual "proper" mask (what worked in debug script)
    def cp_causal_mask_manual(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + local_start
        return global_q_idx >= kv_idx

    proper_mask = create_block_mask(
        cp_causal_mask_manual,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=seq_len,
        device=device,
    )
    print_block_mask_details(
        "Proper Mask (manual create_block_mask)", proper_mask, rank
    )

    # ============ Test 1: CP mask with context_parallel ============
    print(f"\n[Rank {rank}] === TEST 1: CP block_mask + context_parallel ===")
    try:
        with context_parallel(cp_mesh):
            out1 = flex_attention(
                q.clone(), k.clone(), v.clone(), block_mask=cp_block_mask
            )
        check_tensor("CP mask + context_parallel (non-compiled)", out1, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST1 Error: {e}")

    # ============ Test 2: Proper mask with manual gather ============
    print(f"\n[Rank {rank}] === TEST 2: Proper mask + manual gather ===")
    import torch.distributed._functional_collectives as ft_c

    try:
        global_k = ft_c.all_gather_tensor(
            k.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )
        global_v = ft_c.all_gather_tensor(
            v.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )

        out2 = flex_attention(q.clone(), global_k, global_v, block_mask=proper_mask)
        check_tensor("Proper mask + manual gather (non-compiled)", out2, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST2 Error: {e}")

    # ============ Test 3: CP mask with manual gather (no context_parallel) ============
    print(
        f"\n[Rank {rank}] === TEST 3: CP mask + manual gather (NO context_parallel) ==="
    )
    try:
        # This tests if the CP mask itself is the problem
        out3 = flex_attention(q.clone(), global_k, global_v, block_mask=cp_block_mask)
        check_tensor("CP mask + manual gather (non-compiled)", out3, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST3 Error: {e}")

    # ============ Test 4: Compiled versions ============
    print(f"\n[Rank {rank}] === TEST 4: Compiled versions (dynamic=True) ===")
    compiled_flex = torch.compile(flex_attention, dynamic=True)

    # 4a: CP mask + context_parallel + compiled
    try:
        with context_parallel(cp_mesh):
            out4a = compiled_flex(
                q.clone(), k.clone(), v.clone(), block_mask=cp_block_mask
            )
        check_tensor("CP mask + context_parallel (compiled)", out4a, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST4a Error: {e}")

    # 4b: Proper mask + manual gather + compiled
    try:
        out4b = compiled_flex(q.clone(), global_k, global_v, block_mask=proper_mask)
        check_tensor("Proper mask + manual gather (compiled)", out4b, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST4b Error: {e}")

    # 4c: CP mask + manual gather + compiled (no context_parallel)
    try:
        out4c = compiled_flex(q.clone(), global_k, global_v, block_mask=cp_block_mask)
        check_tensor("CP mask + manual gather (compiled)", out4c, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST4c Error: {e}")

    # ============ Test 5: Check what context_parallel does to the mask ============
    print(f"\n[Rank {rank}] === TEST 5: Trace context_parallel mask modification ===")

    # Get the mask tuple before and see what would happen
    cp_tuple_before = cp_block_mask.as_tuple()
    print(f"[Rank {rank}] CP mask tuple[1] (KV_LEN) before: {cp_tuple_before[1]}")
    print(f"[Rank {rank}] global_k.size(-2): {global_k.size(-2)}")

    # This is what context_parallel does:
    if cp_tuple_before[1] != global_k.size(-2):
        modified_tuple = (cp_tuple_before[0], global_k.size(-2), *cp_tuple_before[2:])
        print(
            f"[Rank {rank}] MODIFIED: tuple[1] changed from {cp_tuple_before[1]} to {global_k.size(-2)}"
        )

        # Compare kv_indices between CP mask and proper mask
        print(f"\n[Rank {rank}] --- Comparing kv_indices ---")
        cp_kv_indices = cp_tuple_before[3]
        proper_kv_indices = proper_mask.as_tuple()[3]

        if isinstance(cp_kv_indices, torch.Tensor) and isinstance(
            proper_kv_indices, torch.Tensor
        ):
            print(f"[Rank {rank}] CP kv_indices shape: {cp_kv_indices.shape}")
            print(f"[Rank {rank}] Proper kv_indices shape: {proper_kv_indices.shape}")
            print(
                f"[Rank {rank}] CP kv_indices[0,0,0]: {cp_kv_indices[0,0,0] if cp_kv_indices.dim() >= 3 else cp_kv_indices}"
            )
            print(
                f"[Rank {rank}] Proper kv_indices[0,0,0]: {proper_kv_indices[0,0,0] if proper_kv_indices.dim() >= 3 else proper_kv_indices}"
            )
    else:
        print(f"[Rank {rank}] NOT MODIFIED: tuple[1] already equals global_k.size(-2)")

    # ============ Test 6: Compare dense masks ============
    print(f"\n[Rank {rank}] === TEST 6: Dense mask comparison ===")
    try:
        cp_dense = cp_block_mask.to_dense()
        proper_dense = proper_mask.to_dense()

        print(f"[Rank {rank}] CP dense shape: {cp_dense.shape}")
        print(f"[Rank {rank}] Proper dense shape: {proper_dense.shape}")

        # Check if they match where they should
        if cp_dense.shape == proper_dense.shape:
            diff = (cp_dense != proper_dense).sum().item()
            print(f"[Rank {rank}] Number of differing elements: {diff}")
        else:
            print(f"[Rank {rank}] Shapes differ, comparing overlapping region...")
            min_q = min(cp_dense.shape[2], proper_dense.shape[2])
            min_kv = min(cp_dense.shape[3], proper_dense.shape[3])
            diff = (
                (cp_dense[:, :, :min_q, :min_kv] != proper_dense[:, :, :min_q, :min_kv])
                .sum()
                .item()
            )
            print(
                f"[Rank {rank}] Differences in overlapping region [{min_q}x{min_kv}]: {diff}"
            )

        # Print some samples
        print(f"\n[Rank {rank}] CP dense mask [0,0, :8, :16]:")
        print(cp_dense[0, 0, :8, :16].int())
        print(f"\n[Rank {rank}] Proper dense mask [0,0, :8, :16]:")
        print(proper_dense[0, 0, :8, :16].int())

    except Exception as e:
        print(f"[Rank {rank}] TEST6 Error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n[Rank {rank}] === Debug Complete ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
