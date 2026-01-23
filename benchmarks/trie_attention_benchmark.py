# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script for comparing standard causal attention vs trie-based attention.

This script measures the performance difference between:
1. Standard causal attention (baseline)
2. Trie causal attention with shared prefix structure

Usage:
    python benchmarks/trie_attention_benchmark.py [--batch_size=4] [--seq_len=2048] [--prefix_ratio=0.5]
"""

import argparse
import time
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    and_masks,
    BlockMask,
    create_block_mask,
    flex_attention,
)


def get_causal_mask_mod() -> _mask_mod_signature:
    """Standard causal mask."""

    def causal_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return q_idx >= kv_idx

    return causal_mask


def get_trie_causal_mask_mod(
    tin: torch.Tensor,
    tout: torch.Tensor,
) -> _mask_mod_signature:
    """Trie-based causal mask using DFS interval containment + causal ordering."""

    def trie_causal_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # Ancestor check via DFS interval containment
        is_ancestor = (tin[b, kv_idx] <= tin[b, q_idx]) & (
            tout[b, q_idx] <= tout[b, kv_idx]
        )
        # Causal ordering within same node
        is_causal = q_idx >= kv_idx
        return is_ancestor & is_causal

    return trie_causal_mask


def generate_trie_data(
    batch_size: int,
    seq_len: int,
    prefix_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic trie data with shared prefix.

    Creates tin/tout tensors where:
    - First `prefix_len` tokens are shared prefix (same tin/tout across batch)
    - Remaining tokens are unique branches per batch item

    Args:
        batch_size: Number of samples in the batch
        seq_len: Total sequence length
        prefix_ratio: Fraction of sequence that is shared prefix (0.0 to 1.0)
        device: Device to create tensors on

    Returns:
        tin: DFS entry times [B, S]
        tout: DFS exit times [B, S]
    """
    prefix_len = int(seq_len * prefix_ratio)
    branch_len = seq_len - prefix_len

    tin = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    tout = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    # Shared prefix: linear chain structure
    # All batch items have same tin/tout for prefix tokens
    # tin = [0, 1, 2, ..., prefix_len-1]
    # tout = [2*seq_len, 2*seq_len-1, ..., 2*seq_len-prefix_len+1]
    for i in range(prefix_len):
        tin[:, i] = i
        tout[:, i] = 2 * seq_len - i

    # Branch tokens: each batch item has unique tin/tout
    # Simulates different branches from the end of prefix
    for b in range(batch_size):
        for i in range(branch_len):
            pos = prefix_len + i
            # Branch starts from end of prefix, with offset per batch item
            base_tin = prefix_len + b * branch_len + i
            tin[b, pos] = base_tin
            # tout creates proper nesting: later tokens have smaller intervals
            tout[b, pos] = base_tin + 1

    return tin, tout


def generate_linear_chain_trie(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate trie data for a linear chain (equivalent to causal).

    This is used to verify correctness - trie attention on a linear chain
    should produce the same results as standard causal attention.
    """
    tin = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    tout = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for i in range(seq_len):
        tin[:, i] = i
        tout[:, i] = 2 * seq_len - i

    return tin, tout


class SimplifiedTransformerBlock(nn.Module):
    """Simplified transformer block for benchmarking."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        use_flex_attn: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.use_flex_attn = use_flex_attn

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

        self._compiled_flex_attn = torch.compile(
            flex_attention, mode="max-autotune-no-cudagraphs"
        )

    def forward(
        self,
        x: torch.Tensor,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # Attention
        h = self.norm1(x)
        q = self.wq(h).view(bs, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(bs, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(bs, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_flex_attn and block_mask is not None:
            attn_out = self._compiled_flex_attn(q, k, v, block_mask=block_mask)
        else:
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        x = x + self.wo(attn_out)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class BenchmarkModel(nn.Module):
    """Simple model for benchmarking attention."""

    def __init__(
        self,
        vocab_size: int = 2048,
        dim: int = 256,
        n_layers: int = 8,
        n_heads: int = 16,
        head_dim: int = 128,
        use_flex_attn: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                SimplifiedTransformerBlock(dim, n_heads, head_dim, use_flex_attn)
                for _ in range(n_layers)
            ]
        )
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x, block_mask)
        return self.output(x)


def benchmark_forward_backward(
    model: nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    block_mask: BlockMask | None,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> tuple[float, float]:
    """Benchmark forward and backward pass.

    Returns:
        Tuple of (forward_time_ms, backward_time_ms)
    """
    # Warmup
    for _ in range(warmup_iters):
        logits = model(tokens, block_mask)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # Benchmark forward
    forward_times = []
    for _ in range(benchmark_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(tokens, block_mask)
        torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)

    # Benchmark backward
    backward_times = []
    for _ in range(benchmark_iters):
        logits = model(tokens, block_mask)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)
        model.zero_grad()

    return sum(forward_times) / len(forward_times), sum(backward_times) / len(
        backward_times
    )


def verify_correctness(
    model: nn.Module,
    tokens: torch.Tensor,
    causal_mask: BlockMask,
    trie_mask: BlockMask,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Verify trie attention matches causal for linear chain."""
    with torch.no_grad():
        causal_out = model(tokens, causal_mask)
        trie_out = model(tokens, trie_mask)

    max_diff = (causal_out - trie_out).abs().max().item()
    is_close = torch.allclose(causal_out, trie_out, rtol=rtol, atol=atol)

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Outputs match: {is_close}")

    return is_close


def main():
    parser = argparse.ArgumentParser(description="Benchmark trie attention")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--prefix_ratio",
        type=float,
        default=0.5,
        help="Ratio of shared prefix (0.0 to 1.0)",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=3, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--benchmark_iters", type=int, default=10, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--skip_correctness",
        action="store_true",
        help="Skip correctness verification",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 60)
    print("TrieAttention Benchmark")
    print("=" * 60)
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Prefix ratio: {args.prefix_ratio}")
    print(f"Prefix length: {int(args.seq_len * args.prefix_ratio)}")
    print(f"Branch length: {args.seq_len - int(args.seq_len * args.prefix_ratio)}")
    print()

    # Create model
    print("Creating model...")
    model = BenchmarkModel(
        vocab_size=2048,
        dim=256,
        n_layers=8,
        n_heads=16,
        head_dim=128,
        use_flex_attn=True,
    ).to(device=device, dtype=dtype)

    # Use SGD without momentum (minimal optimizer state)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Generate data
    print("Generating data...")
    tokens = torch.randint(0, 2048, (args.batch_size, args.seq_len), device=device)
    labels = torch.randint(0, 2048, (args.batch_size, args.seq_len), device=device)

    # Create causal mask
    print("Creating causal attention mask...")
    _compiled_create_block_mask = torch.compile(create_block_mask)
    causal_mask = _compiled_create_block_mask(
        get_causal_mask_mod(),
        1,  # B=1 for broadcasting
        None,
        args.seq_len,
        args.seq_len,
    )

    # Generate trie data with shared prefix
    print("Creating trie attention mask (with shared prefix)...")
    tin, tout = generate_trie_data(
        args.batch_size, args.seq_len, args.prefix_ratio, device
    )
    trie_mask = _compiled_create_block_mask(
        get_trie_causal_mask_mod(tin, tout),
        args.batch_size,
        None,
        args.seq_len,
        args.seq_len,
    )

    # Correctness verification
    if not args.skip_correctness:
        print()
        print("-" * 60)
        print("Correctness Verification (linear chain trie vs causal)")
        print("-" * 60)

        tin_linear, tout_linear = generate_linear_chain_trie(
            args.batch_size, args.seq_len, device
        )
        linear_trie_mask = _compiled_create_block_mask(
            get_trie_causal_mask_mod(tin_linear, tout_linear),
            args.batch_size,
            None,
            args.seq_len,
            args.seq_len,
        )
        verify_correctness(model, tokens, causal_mask, linear_trie_mask)

    # Benchmark
    print()
    print("-" * 60)
    print("Performance Benchmark")
    print("-" * 60)

    print("\nBenchmarking standard causal attention...")
    causal_fwd, causal_bwd = benchmark_forward_backward(
        model,
        tokens,
        labels,
        causal_mask,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )
    print(f"  Forward:  {causal_fwd:.2f} ms")
    print(f"  Backward: {causal_bwd:.2f} ms")
    print(f"  Total:    {causal_fwd + causal_bwd:.2f} ms")

    print(f"\nBenchmarking trie attention ({args.prefix_ratio*100:.0f}% shared prefix)...")
    trie_fwd, trie_bwd = benchmark_forward_backward(
        model,
        tokens,
        labels,
        trie_mask,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )
    print(f"  Forward:  {trie_fwd:.2f} ms")
    print(f"  Backward: {trie_bwd:.2f} ms")
    print(f"  Total:    {trie_fwd + trie_bwd:.2f} ms")

    # Summary
    print()
    print("-" * 60)
    print("Summary")
    print("-" * 60)
    speedup_fwd = causal_fwd / trie_fwd if trie_fwd > 0 else float("inf")
    speedup_bwd = causal_bwd / trie_bwd if trie_bwd > 0 else float("inf")
    speedup_total = (causal_fwd + causal_bwd) / (trie_fwd + trie_bwd)

    print(f"Forward speedup:  {speedup_fwd:.2f}x")
    print(f"Backward speedup: {speedup_bwd:.2f}x")
    print(f"Total speedup:    {speedup_total:.2f}x")

    # Memory stats
    print()
    print("-" * 60)
    print("Memory Usage")
    print("-" * 60)
    print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Peak GPU memory reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

    print()
    print("=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
