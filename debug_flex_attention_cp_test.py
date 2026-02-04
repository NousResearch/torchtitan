#!/usr/bin/env python3
"""
Test FlexAttention with CP - minimal, explicit test.

Run with 2 GPUs:
  torchrun --nproc_per_node=2 debug_flex_attention_cp_test.py
"""

import os
import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(rank, msg):
    print(f"[Rank {rank}] {msg}", flush=True)


def main():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    log(rank, "=" * 60)
    log(rank, "FlexAttention + CP Test")
    log(rank, "=" * 60)

    # Config
    batch_size = 2
    n_heads = 4
    head_dim = 32
    seq_len = 256
    local_seq_len = seq_len // world_size

    log(
        rank,
        f"seq_len={seq_len}, local_seq_len={local_seq_len}, n_heads={n_heads}, head_dim={head_dim}",
    )

    # Create Q, K, V tensors (same on all ranks for baseline comparison)
    torch.manual_seed(42)
    Q_full = torch.randn(
        batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
    )
    K_full = torch.randn(
        batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
    )
    V_full = torch.randn(
        batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
    )

    # CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )
    cp_group = cp_mesh.get_group()

    # ========================================
    # Causal mask function
    # ========================================
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # ========================================
    # Test 1: Full FlexAttention (baseline)
    # ========================================
    log(rank, "\n--- Baseline: Full FlexAttention ---")

    full_mask = create_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    with torch.no_grad():
        output_baseline = flex_attention(Q_full, K_full, V_full, block_mask=full_mask)

    log(rank, f"Baseline output shape: {output_baseline.shape}")

    # ========================================
    # Test 2: CP FlexAttention (local Q, global K/V)
    # ========================================
    log(rank, "\n--- CP: Local Q, Gathered K/V ---")

    # Get local Q
    local_start = rank * local_seq_len
    Q_local = Q_full[:, :, local_start : local_start + local_seq_len, :]
    K_local = K_full[:, :, local_start : local_start + local_seq_len, :]
    V_local = V_full[:, :, local_start : local_start + local_seq_len, :]

    # All-gather K, V
    K_gathered = ft_c.all_gather_tensor(
        K_local.contiguous(), gather_dim=2, group=cp_group
    )
    V_gathered = ft_c.all_gather_tensor(
        V_local.contiguous(), gather_dim=2, group=cp_group
    )

    log(rank, f"Q_local shape: {Q_local.shape}")
    log(rank, f"K_gathered shape: {K_gathered.shape}")
    log(rank, f"V_gathered shape: {V_gathered.shape}")

    # Verify gathered K/V matches original
    k_match = torch.allclose(K_gathered, K_full, atol=1e-6)
    v_match = torch.allclose(V_gathered, V_full, atol=1e-6)
    log(rank, f"K_gathered matches K_full: {k_match}")
    log(rank, f"V_gathered matches V_full: {v_match}")

    # CP-aware causal mask
    q_offset = local_start

    def cp_causal_mask(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + q_offset
        return global_q_idx >= kv_idx

    cp_mask = create_block_mask(
        cp_causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    with torch.no_grad():
        output_cp = flex_attention(Q_local, K_gathered, V_gathered, block_mask=cp_mask)

    log(rank, f"CP output shape: {output_cp.shape}")

    # ========================================
    # Compare
    # ========================================
    log(rank, "\n--- Comparison ---")

    baseline_local = output_baseline[:, :, local_start : local_start + local_seq_len, :]

    max_diff = (output_cp - baseline_local).abs().max().item()
    mean_diff = (output_cp - baseline_local).abs().mean().item()

    log(rank, f"Max diff: {max_diff:.10f}")
    log(rank, f"Mean diff: {mean_diff:.10f}")

    if max_diff < 1e-5:
        log(rank, "✓ PASS: FlexAttention + CP produces identical results")
    else:
        log(rank, f"✗ FAIL: Difference of {max_diff}")

    # ========================================
    # Global verification
    # ========================================
    output_cp_gathered = ft_c.all_gather_tensor(
        output_cp.contiguous(), gather_dim=2, group=cp_group
    )

    if rank == 0:
        global_max_diff = (output_cp_gathered - output_baseline).abs().max().item()
        log(rank, f"\nGlobal max diff: {global_max_diff:.10f}")
        if global_max_diff < 1e-5:
            log(rank, "✓ GLOBAL PASS: All ranks produce correct results")
        else:
            log(rank, f"✗ GLOBAL FAIL: Difference of {global_max_diff}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
