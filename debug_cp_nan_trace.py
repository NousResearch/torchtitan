#!/usr/bin/env python3
"""
Deep debug script to trace NaN in flex_attention + Context Parallel.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor.experimental._attention import (
    context_parallel,
    create_cp_block_mask,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Enable anomaly detection for gradient debugging
torch.autograd.set_detect_anomaly(True)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def check_tensor(name, tensor, rank):
    """Check tensor for NaN/Inf."""
    if tensor is None:
        print(f"[Rank {rank}] {name}: None")
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
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
    print(f"[Rank {rank}] Deep Debug: CP + FlexAttention NaN")
    print(f"[Rank {rank}] World size: {world_size}")
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
    cp_rank = cp_mesh.get_local_rank()
    cp_size = cp_mesh.size()
    local_seq_len = seq_len // cp_size

    print(
        f"[Rank {rank}] seq_len={seq_len}, local_seq_len={local_seq_len}, cp_rank={cp_rank}"
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

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    check_tensor("Input Q", q, rank)
    check_tensor("Input K", k, rank)
    check_tensor("Input V", v, rank)

    # Create CP block mask
    print(f"\n[Rank {rank}] Creating CP block mask...")
    block_mask = create_cp_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device_mesh=cp_mesh,
    )
    print(f"[Rank {rank}] BlockMask created: shape={block_mask.shape}")

    # Check mask internals
    mask_tuple = block_mask.as_tuple()
    print(f"[Rank {rank}] BlockMask as_tuple has {len(mask_tuple)} elements")

    # ============== TEST 1: Non-compiled flex_attention ==============
    print(f"\n[Rank {rank}] === TEST 1: Non-compiled flex_attention ===")
    try:
        with context_parallel(cp_mesh, buffers={}):
            out1 = flex_attention(q, k, v, block_mask=block_mask)
        check_tensor("TEST1 output", out1, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST1 Error: {e}")

    # ============== TEST 2: Compiled with dynamic=True ==============
    print(f"\n[Rank {rank}] === TEST 2: Compiled dynamic=True ===")
    compiled_flex = torch.compile(flex_attention, dynamic=True)
    try:
        q2 = q.detach().clone()
        k2 = k.detach().clone()
        v2 = v.detach().clone()
        with context_parallel(cp_mesh, buffers={}):
            out2 = compiled_flex(q2, k2, v2, block_mask=block_mask)
        check_tensor("TEST2 output", out2, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST2 Error: {e}")

    # ============== TEST 3: Step through context_parallel internals ==============
    print(f"\n[Rank {rank}] === TEST 3: Manual CP ring attention trace ===")

    # Get the _RingAttention module to understand what happens
    from torch.distributed.tensor.experimental._attention import (
        _templated_ring_attention,
    )

    # Trace what context_parallel does
    print(f"[Rank {rank}] Checking what context_parallel modifies...")

    # Let's manually do what ring attention does to find where NaN comes from
    # Ring attention rotates K,V around the ring and accumulates attention

    # Initial attention scores check
    print(f"\n[Rank {rank}] Computing local attention scores...")
    with torch.no_grad():
        # Local Q @ K^T
        local_scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        local_scores = local_scores / (head_dim**0.5)
        check_tensor("Local attention scores (Q @ K^T / sqrt(d))", local_scores, rank)

        # After softmax (with causal mask applied manually)
        # For simplicity, just check raw softmax
        local_probs = torch.softmax(local_scores, dim=-1)
        check_tensor("Local softmax(scores)", local_probs, rank)

        # Local output
        local_out = torch.matmul(local_probs, v.float())
        check_tensor("Local attention output", local_out, rank)

    # ============== TEST 4: Trace through the actual flex_attention internals ==============
    print(f"\n[Rank {rank}] === TEST 4: Flex attention with score_mod tracing ===")

    # Create a score_mod that logs values
    nan_detected = [False]

    def debug_score_mod(score, b, h, q_idx, kv_idx):
        if torch.isnan(score).any():
            nan_detected[0] = True
        return score  # identity

    # Create block mask with debug score mod
    from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks

    def causal_with_debug(b, h, q_idx, kv_idx):
        result = q_idx >= kv_idx
        return result

    try:
        # Regular create_block_mask (not CP) for comparison
        regular_mask = create_block_mask(
            causal_mask,
            B=batch_size,
            H=n_heads,
            Q_LEN=local_seq_len,
            KV_LEN=local_seq_len,
            device=device,
        )
        print(f"[Rank {rank}] Regular mask shape: {regular_mask.shape}")

        # Test with regular mask (no CP)
        q3 = q.detach().clone()
        k3 = k.detach().clone()
        v3 = v.detach().clone()
        out3 = flex_attention(q3, k3, v3, block_mask=regular_mask)
        check_tensor("Regular (non-CP) flex_attention output", out3, rank)

    except Exception as e:
        print(f"[Rank {rank}] TEST4 Error: {e}")
        import traceback

        traceback.print_exc()

    # ============== TEST 5: Compare CP mask vs regular mask ==============
    print(f"\n[Rank {rank}] === TEST 5: Comparing masks ===")
    try:
        # Convert masks to dense for comparison
        cp_dense = block_mask.to_dense()
        regular_dense = regular_mask.to_dense()

        print(f"[Rank {rank}] CP mask dense shape: {cp_dense.shape}")
        print(f"[Rank {rank}] Regular mask dense shape: {regular_dense.shape}")

        # Check for differences
        print(f"[Rank {rank}] CP mask sum: {cp_dense.sum().item()}")
        print(f"[Rank {rank}] Regular mask sum: {regular_dense.sum().item()}")

        # Sample some values
        print(f"[Rank {rank}] CP mask[0,0,:5,:5]:\n{cp_dense[0,0,:5,:5]}")
        print(f"[Rank {rank}] Regular mask[0,0,:5,:5]:\n{regular_dense[0,0,:5,:5]}")

    except Exception as e:
        print(f"[Rank {rank}] TEST5 Error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n[Rank {rank}] === Debug Complete ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
