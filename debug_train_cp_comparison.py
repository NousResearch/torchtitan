#!/usr/bin/env python3
"""
Minimal training comparison: Document masking with and without CP.

This script:
1. Creates a small transformer model
2. Trains with document masking (no CP) - baseline
3. Trains with document masking + CP - should match baseline

Run WITHOUT CP (single GPU baseline):
  python debug_train_cp_comparison.py --no-cp

Run WITH CP (2 GPUs):
  torchrun --nproc_per_node=2 debug_train_cp_comparison.py --cp

Compare the losses - they should be very close!
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
)

try:
    from torch.distributed.tensor.experimental._attention import context_parallel

    HAS_CP = True
except ImportError:
    HAS_CP = False
    print("Warning: context_parallel not available")


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


def _get_document_ids(seq_lens, device):
    batch_document_ids = []
    for sample_lens in seq_lens:
        doc_ids = torch.cat(
            [
                torch.full((l,), i, dtype=torch.long, device=device)
                for i, l in enumerate(sample_lens)
            ]
        )
        batch_document_ids.append(doc_ids)
    return torch.stack(batch_document_ids)


def get_document_causal_mask_mod(document_ids):
    def mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        doc_match = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal & doc_match

    return mask


def train_no_cp(args):
    """Train without context parallel (single GPU baseline)."""
    device = "cuda"
    rank = 0

    log(rank, "=" * 60)
    log(rank, "Training WITHOUT Context Parallel (baseline)")
    log(rank, "=" * 60)

    # Config
    vocab_size = 1000
    dim = 128
    n_layers = 2
    n_heads = 4
    hidden_dim = 256
    batch_size = 2
    seq_len = 256
    n_steps = 10

    # Document structure: 4 documents per sequence
    doc_sizes = [64, 64, 64, 64]

    log(rank, f"Config: dim={dim}, n_layers={n_layers}, n_heads={n_heads}")
    log(rank, f"Training: batch_size={batch_size}, seq_len={seq_len}, steps={n_steps}")
    log(rank, f"Documents per sequence: {doc_sizes}")

    # Create model
    torch.manual_seed(42)
    model = (
        SmallTransformer(vocab_size, dim, n_layers, n_heads, hidden_dim)
        .to(device)
        .to(torch.bfloat16)
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create document IDs for mask
    seq_lens_batch = [[s for s in doc_sizes] for _ in range(batch_size)]
    document_ids = _get_document_ids(seq_lens_batch, device)

    # Create mask
    mask_mod = get_document_causal_mask_mod(document_ids)
    mask = create_block_mask(
        mask_mod, B=batch_size, H=n_heads, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )

    losses = []
    for step in range(n_steps):
        # Generate random input (same seed for reproducibility)
        torch.manual_seed(1000 + step)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()
        logits = model(tokens, mask)
        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        log(rank, f"Step {step}: loss = {loss.item():.6f}")

    log(rank, f"\nFinal losses: {[f'{l:.4f}' for l in losses]}")
    return losses


def train_with_cp(args):
    """Train with context parallel."""
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    log(rank, "=" * 60)
    log(rank, f"Training WITH Context Parallel (CP={world_size})")
    log(rank, "=" * 60)

    # Config (same as no-cp)
    vocab_size = 1000
    dim = 128
    n_layers = 2
    n_heads = 4
    hidden_dim = 256
    batch_size = 2
    seq_len = 256
    n_steps = 10

    local_seq_len = seq_len // world_size
    local_start = rank * local_seq_len

    # Document structure
    doc_sizes = [64, 64, 64, 64]

    log(rank, f"Config: dim={dim}, n_layers={n_layers}, n_heads={n_heads}")
    log(
        rank,
        f"Training: batch_size={batch_size}, seq_len={seq_len}, local_seq_len={local_seq_len}",
    )
    log(
        rank,
        f"Rank {rank} handles positions {local_start}-{local_start+local_seq_len-1}",
    )

    # Create model (same seed = same weights)
    torch.manual_seed(42)
    model = (
        SmallTransformer(vocab_size, dim, n_layers, n_heads, hidden_dim)
        .to(device)
        .to(torch.bfloat16)
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )

    # Document IDs (full sequence)
    seq_lens_batch = [[s for s in doc_sizes] for _ in range(batch_size)]
    document_ids = _get_document_ids(seq_lens_batch, device)

    # Create CP-aware mask (THE FIX)
    q_offset = local_start
    base_mask_mod = get_document_causal_mask_mod(document_ids)

    def cp_aware_mask_mod(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + q_offset
        return base_mask_mod(b, h, global_q_idx, kv_idx)

    # Mask for local Q vs global KV
    mask = create_block_mask(
        cp_aware_mask_mod,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    losses = []
    for step in range(n_steps):
        # Generate random input (same seed = same data across ranks)
        torch.manual_seed(1000 + step)
        tokens_full = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels_full = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Get local portion
        tokens_local = tokens_full[:, local_start : local_start + local_seq_len]
        labels_local = labels_full[:, local_start : local_start + local_seq_len]

        optimizer.zero_grad()

        # Manual CP forward pass
        # 1. Compute embeddings for local tokens
        h = model.tok_embeddings(tokens_local)

        # 2. For each layer, we need to:
        #    - Compute local Q
        #    - All-gather K, V
        #    - Attention with local Q vs global K, V
        for layer in model.layers:
            # Attention with CP
            h_normed = layer.attention_norm(h)
            bsz, seqlen, _ = h_normed.shape

            q = (
                layer.attention.wq(h_normed)
                .view(bsz, seqlen, n_heads, dim // n_heads)
                .transpose(1, 2)
            )
            k_local = (
                layer.attention.wk(h_normed)
                .view(bsz, seqlen, n_heads, dim // n_heads)
                .transpose(1, 2)
            )
            v_local = (
                layer.attention.wv(h_normed)
                .view(bsz, seqlen, n_heads, dim // n_heads)
                .transpose(1, 2)
            )

            # All-gather K, V
            k_global = ft_c.all_gather_tensor(
                k_local.contiguous(), gather_dim=2, group=cp_mesh.get_group()
            )
            v_global = ft_c.all_gather_tensor(
                v_local.contiguous(), gather_dim=2, group=cp_mesh.get_group()
            )

            attn_out = flex_attention(
                q, k_global, v_global, block_mask=mask, scale=layer.attention.scale
            )
            attn_out = layer.attention.wo(
                attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            )
            h = h + attn_out

            # FFN (local, no communication needed)
            h = h + layer.feed_forward(layer.ffn_norm(h))

        # Output
        h = model.norm(h)
        logits = model.output(h)

        # Loss (local)
        loss_local = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), labels_local.reshape(-1)
        )

        # Average loss across ranks
        loss_tensor = loss_local.detach().clone()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        loss_local.backward()

        # Sync gradients
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        optimizer.step()

        losses.append(loss_tensor.item())
        log(rank, f"Step {step}: loss = {loss_tensor.item():.6f}")

    if rank == 0:
        log(rank, f"\nFinal losses: {[f'{l:.4f}' for l in losses]}")

    dist.barrier()
    dist.destroy_process_group()
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", action="store_true", help="Run with context parallel")
    parser.add_argument(
        "--no-cp", action="store_true", help="Run without context parallel"
    )
    args = parser.parse_args()

    if args.cp:
        if not HAS_CP:
            print("Error: context_parallel not available in this PyTorch version")
            sys.exit(1)
        train_with_cp(args)
    elif args.no_cp:
        train_no_cp(args)
    else:
        print("Please specify --cp or --no-cp")
        print("\nUsage:")
        print("  Baseline (no CP):  python debug_train_cp_comparison.py --no-cp")
        print(
            "  With CP:           torchrun --nproc_per_node=2 debug_train_cp_comparison.py --cp"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
