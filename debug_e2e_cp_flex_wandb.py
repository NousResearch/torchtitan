#!/usr/bin/env python3
"""
End-to-end training test with FlexAttention + CP + wandb logging.

Run baseline (single GPU):
  python debug_e2e_cp_flex_wandb.py --no-cp

Run with CP (2 GPUs):
  torchrun --nproc_per_node=2 debug_e2e_cp_flex_wandb.py --cp
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.nn as nn

import wandb
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


class FlexAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim**-0.5

    def forward(self, x, mask, cp_group=None):
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        if cp_group is not None:
            # CP mode: gather K, V
            k = ft_c.all_gather_tensor(k.contiguous(), gather_dim=2, group=cp_group)
            v = ft_c.all_gather_tensor(v.contiguous(), gather_dim=2, group=cp_group)

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
        self.attention = FlexAttention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, mask, cp_group=None):
        x = x + self.attention(self.attention_norm(x), mask, cp_group)
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

    def forward(self, tokens, mask, cp_group=None):
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, mask, cp_group)
        h = self.norm(h)
        return self.output(h)


def train_no_cp():
    """Train without CP (baseline)."""
    device = "cuda"
    rank = 0

    # Init wandb
    wandb.init(
        project="cp-flex-attention-test",
        name="no-cp-baseline",
        config={
            "cp_enabled": False,
            "world_size": 1,
            "dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "seq_len": 256,
            "batch_size": 2,
        },
    )

    log(rank, "=" * 60)
    log(rank, "Training WITHOUT CP (baseline) - FlexAttention")
    log(rank, "=" * 60)

    # Config
    vocab_size = 1000
    dim = 128
    n_layers = 2
    n_heads = 4
    hidden_dim = 256
    batch_size = 2
    seq_len = 256
    n_steps = 50

    log(rank, f"Config: dim={dim}, n_layers={n_layers}, n_heads={n_heads}")
    log(rank, f"Training: batch_size={batch_size}, seq_len={seq_len}, steps={n_steps}")

    # Create model
    torch.manual_seed(42)
    model = (
        SmallTransformer(vocab_size, dim, n_layers, n_heads, hidden_dim)
        .to(device)
        .to(torch.bfloat16)
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Causal mask
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    mask = create_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    for step in range(n_steps):
        torch.manual_seed(1000 + step)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()
        logits = model(tokens, mask)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), labels.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item(), "step": step})
        log(rank, f"Step {step}: loss = {loss.item():.6f}")

    wandb.finish()
    log(rank, f"Wandb run finished")


def train_with_cp():
    """Train with CP."""
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    # Only rank 0 logs to wandb
    if rank == 0:
        wandb.init(
            project="cp-flex-attention-test",
            name=f"with-cp-{world_size}gpu",
            config={
                "cp_enabled": True,
                "world_size": world_size,
                "dim": 128,
                "n_layers": 2,
                "n_heads": 4,
                "seq_len": 256,
                "batch_size": 2,
            },
        )

    log(rank, "=" * 60)
    log(rank, f"Training WITH CP (world_size={world_size}) - FlexAttention")
    log(rank, "=" * 60)

    # Config (same as no-cp)
    vocab_size = 1000
    dim = 128
    n_layers = 2
    n_heads = 4
    hidden_dim = 256
    batch_size = 2
    seq_len = 256
    n_steps = 50

    local_seq_len = seq_len // world_size
    local_start = rank * local_seq_len

    log(rank, f"Config: dim={dim}, n_layers={n_layers}, n_heads={n_heads}")
    log(rank, f"seq_len={seq_len}, local_seq_len={local_seq_len}")
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
    cp_group = cp_mesh.get_group()

    # CP-aware causal mask
    q_offset = local_start

    def cp_causal_mask(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + q_offset
        return global_q_idx >= kv_idx

    mask = create_block_mask(
        cp_causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    for step in range(n_steps):
        # Generate same data as no-cp
        torch.manual_seed(1000 + step)
        tokens_full = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels_full = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Get local portion
        tokens_local = tokens_full[:, local_start : local_start + local_seq_len]
        labels_local = labels_full[:, local_start : local_start + local_seq_len]

        optimizer.zero_grad()
        logits = model(tokens_local, mask, cp_group)
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

        if rank == 0:
            wandb.log({"loss": loss_tensor.item(), "step": step})
        log(rank, f"Step {step}: loss = {loss_tensor.item():.6f}")

    if rank == 0:
        wandb.finish()
        log(rank, f"Wandb run finished")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", action="store_true", help="Run with CP")
    parser.add_argument("--no-cp", action="store_true", help="Run without CP")
    args = parser.parse_args()

    if args.cp:
        train_with_cp()
    elif args.no_cp:
        train_no_cp()
    else:
        print("Usage:")
        print("  Baseline (no CP):  python debug_e2e_cp_flex_wandb.py --no-cp")
        print(
            "  With CP:           torchrun --nproc_per_node=2 debug_e2e_cp_flex_wandb.py --cp"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
