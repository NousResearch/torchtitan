#!/usr/bin/env python3
"""
Verify Context Parallel (CP) works correctly with simple causal masking (no document masking).

This test:
1. Creates a small transformer model
2. Runs forward pass WITHOUT CP (baseline)
3. Runs forward pass WITH CP (using manual all-gather)
4. Compares outputs - they should be identical

Run with 2 GPUs:
  torchrun --nproc_per_node=2 debug_cp_causal_only.py
"""

import os
import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(rank, msg):
    print(f"[Rank {rank}] {msg}", flush=True)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim**-0.5

    def forward(self, x, mask):
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        out = flex_attention(q, k, v, block_mask=mask, scale=self.scale)
        return self.wo(out.transpose(1, 2).contiguous().view(bsz, seqlen, -1))

    def forward_cp(self, x, mask, cp_group):
        """Forward with Context Parallel: local Q, gathered K/V."""
        bsz, local_seqlen, _ = x.shape

        q = (
            self.wq(x)
            .view(bsz, local_seqlen, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k_local = (
            self.wk(x)
            .view(bsz, local_seqlen, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_local = (
            self.wv(x)
            .view(bsz, local_seqlen, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # All-gather K, V across CP ranks
        k_global = ft_c.all_gather_tensor(
            k_local.contiguous(), gather_dim=2, group=cp_group
        )
        v_global = ft_c.all_gather_tensor(
            v_local.contiguous(), gather_dim=2, group=cp_group
        )

        out = flex_attention(q, k_global, v_global, block_mask=mask, scale=self.scale)
        return self.wo(out.transpose(1, 2).contiguous().view(bsz, local_seqlen, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, mask):
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def forward_cp(self, x, mask, cp_group):
        x = x + self.attention.forward_cp(self.attention_norm(x), mask, cp_group)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, hidden_dim):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, hidden_dim) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens, mask):
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        return self.output(h)

    def forward_cp(self, tokens, mask, cp_group):
        """Forward with Context Parallel."""
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer.forward_cp(h, mask, cp_group)
        h = self.norm(h)
        return self.output(h)


def get_causal_mask_mod():
    def mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return mask


def main():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    log(rank, "=" * 60)
    log(rank, f"Testing CP correctness with CAUSAL MASKING ONLY (no document masking)")
    log(rank, f"World size: {world_size}")
    log(rank, "=" * 60)

    # Config
    vocab_size = 1000
    dim = 128
    n_layers = 2
    n_heads = 4
    hidden_dim = 256
    batch_size = 2
    seq_len = 256
    local_seq_len = seq_len // world_size

    log(rank, f"Config: dim={dim}, n_layers={n_layers}, n_heads={n_heads}")
    log(rank, f"seq_len={seq_len}, local_seq_len={local_seq_len}")

    # Create model (same seed = same weights on all ranks)
    torch.manual_seed(42)
    model = (
        SmallTransformer(vocab_size, dim, n_layers, n_heads, hidden_dim)
        .to(device)
        .to(torch.bfloat16)
    )

    # Create input (same on all ranks)
    torch.manual_seed(123)
    tokens_full = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )
    cp_group = cp_mesh.get_group()

    # ========================================
    # Test 1: Full sequence forward (baseline)
    # ========================================
    log(rank, "\n--- Test 1: Full sequence forward (baseline) ---")

    # Create full causal mask
    causal_mask_mod = get_causal_mask_mod()
    full_mask = create_block_mask(
        causal_mask_mod,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    with torch.no_grad():
        output_baseline = model(tokens_full, full_mask)

    log(rank, f"Baseline output shape: {output_baseline.shape}")

    # ========================================
    # Test 2: CP forward (local Q, global K/V)
    # ========================================
    log(rank, "\n--- Test 2: CP forward (local Q, gathered K/V) ---")

    # Get local portion of tokens
    local_start = rank * local_seq_len
    tokens_local = tokens_full[:, local_start : local_start + local_seq_len]

    # Create CP-aware causal mask
    # Q is local (0 to local_seq_len-1), KV is global (0 to seq_len-1)
    # We need to offset q_idx to get correct causal masking
    q_offset = local_start

    def cp_causal_mask_mod(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + q_offset
        return global_q_idx >= kv_idx

    cp_mask = create_block_mask(
        cp_causal_mask_mod,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    with torch.no_grad():
        output_cp = model.forward_cp(tokens_local, cp_mask, cp_group)

    log(rank, f"CP output shape: {output_cp.shape}")

    # ========================================
    # Compare outputs
    # ========================================
    log(rank, "\n--- Comparing outputs ---")

    # Extract the portion of baseline output that corresponds to this rank
    baseline_local = output_baseline[:, local_start : local_start + local_seq_len, :]

    # Compare
    max_diff = (output_cp - baseline_local).abs().max().item()
    mean_diff = (output_cp - baseline_local).abs().mean().item()

    log(rank, f"Max diff between CP and baseline: {max_diff:.10f}")
    log(rank, f"Mean diff between CP and baseline: {mean_diff:.10f}")

    # Check if they match (allowing for small floating point differences)
    tolerance = 1e-3  # bfloat16 has lower precision
    if max_diff < tolerance:
        log(rank, f"✓ PASS: CP output matches baseline within tolerance {tolerance}")
    else:
        log(rank, f"✗ FAIL: CP output differs from baseline by {max_diff}")

        # Debug: print some values
        log(rank, f"Baseline local[0,0,:5]: {baseline_local[0,0,:5]}")
        log(rank, f"CP output[0,0,:5]: {output_cp[0,0,:5]}")

    # ========================================
    # Gather all outputs to verify global consistency
    # ========================================
    if rank == 0:
        log(rank, "\n--- Gathering all CP outputs to verify global consistency ---")

    # Gather CP outputs from all ranks
    output_cp_gathered = ft_c.all_gather_tensor(
        output_cp.contiguous(), gather_dim=1, group=cp_group
    )

    if rank == 0:
        global_max_diff = (output_cp_gathered - output_baseline).abs().max().item()
        global_mean_diff = (output_cp_gathered - output_baseline).abs().mean().item()

        log(rank, f"Global max diff (gathered CP vs baseline): {global_max_diff:.10f}")
        log(
            rank, f"Global mean diff (gathered CP vs baseline): {global_mean_diff:.10f}"
        )

        if global_max_diff < tolerance:
            log(
                rank,
                f"✓ GLOBAL PASS: All CP outputs match baseline within tolerance {tolerance}",
            )
        else:
            log(
                rank,
                f"✗ GLOBAL FAIL: CP outputs differ from baseline by {global_max_diff}",
            )

    dist.barrier()

    if rank == 0:
        log(rank, "\n" + "=" * 60)
        log(rank, "CP CAUSAL-ONLY TEST COMPLETE")
        log(rank, "=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
