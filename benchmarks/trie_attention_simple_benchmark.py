# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple single-GPU benchmark for trie attention overhead.

Usage:
    python benchmarks/trie_attention_simple_benchmark.py [--model=4B] [--seq_len=2048]
"""

import argparse
import time
from dataclasses import replace

import torch

from torchtitan.config.job_config import PEFT
from torchtitan.models.qwen3 import qwen3_args
from torchtitan.models.qwen3.model.model import Qwen3Model


def generate_trie_data(
    batch_size: int,
    seq_len: int,
    prefix_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic trie data with shared prefix structure."""
    prefix_len = int(seq_len * prefix_ratio)

    tin = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    tout = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    # Build a simple tree: shared prefix + branching
    for i in range(prefix_len):
        tin[:, i] = i
        tout[:, i] = 2 * seq_len - i

    for i in range(seq_len - prefix_len):
        pos = prefix_len + i
        tin[:, pos] = prefix_len + i
        tout[:, pos] = prefix_len + i + 1

    return tin, tout


def benchmark_model(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    attention_masks,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
) -> tuple[float, float]:
    """Benchmark forward and backward pass."""

    # Warmup
    for _ in range(warmup_iters):
        output = model(tokens, attention_masks=attention_masks)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), labels.view(-1)
        )
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # Benchmark forward
    forward_times = []
    for _ in range(benchmark_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(tokens, attention_masks=attention_masks)
        torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)

    # Benchmark backward
    backward_times = []
    for _ in range(benchmark_iters):
        output = model(tokens, attention_masks=attention_masks)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), labels.view(-1)
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)
        model.zero_grad()

    avg_fwd = sum(forward_times) / len(forward_times)
    avg_bwd = sum(backward_times) / len(backward_times)
    return avg_fwd, avg_bwd


def main():
    parser = argparse.ArgumentParser(description="Benchmark trie attention")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--prefix_ratio", type=float, default=0.5)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--benchmark_iters", type=int, default=20)
    parser.add_argument("--model", type=str, default="4B",
                        choices=["debugmodel", "0.6B", "1.7B", "4B", "8B", "30B-A3B"])
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 60)
    print(f"Trie Attention Benchmark - Qwen3 {args.model}")
    print("=" * 60)
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Prefix ratio: {args.prefix_ratio}")
    print()

    # Get base config
    base_config = qwen3_args[args.model]

    print(f"Model config ({args.model}):")
    print(f"  dim: {base_config.dim}")
    print(f"  n_layers: {base_config.n_layers}")
    print(f"  n_heads: {base_config.n_heads}")
    print(f"  hidden_dim: {base_config.hidden_dim}")
    print(f"  moe_enabled: {base_config.moe_enabled}")
    if base_config.moe_enabled:
        print(f"  num_experts: {base_config.moe_args.num_experts}")
        print(f"  top_k: {base_config.moe_args.top_k}")
    print()

    peft_config = PEFT()

    # Generate data
    torch.manual_seed(42)
    vocab_size = base_config.vocab_size
    tokens = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
    tin, tout = generate_trie_data(args.batch_size, args.seq_len, args.prefix_ratio, device)

    # ========== Benchmark Causal Attention ==========
    print("Creating causal model...")
    causal_config = replace(
        base_config,
        use_flex_attn=True,
        attn_mask_type="causal",
        max_seq_len=args.seq_len,
    )

    causal_model = Qwen3Model(causal_config, peft_config).to(device=device, dtype=dtype)
    total_params = sum(p.numel() for p in causal_model.parameters())
    print(f"Total parameters: {total_params / 1e9:.2f}B")

    # Create attention masks
    causal_masks = causal_model.get_attention_masks(tokens, tokenizer=None, extra_inputs={})

    print("\nBenchmarking causal attention...")
    causal_fwd, causal_bwd = benchmark_model(
        causal_model, tokens, labels, causal_masks,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )
    print(f"  Forward:  {causal_fwd:.2f} ms")
    print(f"  Backward: {causal_bwd:.2f} ms")
    print(f"  Total:    {causal_fwd + causal_bwd:.2f} ms")

    del causal_model, causal_masks
    torch.cuda.empty_cache()

    # ========== Benchmark Trie Attention ==========
    print("\nCreating trie attention model...")
    trie_config = replace(
        base_config,
        use_flex_attn=True,
        attn_mask_type="trie_causal",
        max_seq_len=args.seq_len,
    )

    trie_model = Qwen3Model(trie_config, peft_config).to(device=device, dtype=dtype)

    # Create attention masks for trie
    trie_masks = trie_model.get_attention_masks(
        tokens, tokenizer=None, extra_inputs={"tin": tin, "tout": tout}
    )

    print(f"Benchmarking trie attention ({args.prefix_ratio*100:.0f}% shared prefix)...")
    trie_fwd, trie_bwd = benchmark_model(
        trie_model, tokens, labels, trie_masks,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )
    print(f"  Forward:  {trie_fwd:.2f} ms")
    print(f"  Backward: {trie_bwd:.2f} ms")
    print(f"  Total:    {trie_fwd + trie_bwd:.2f} ms")

    # ========== Summary ==========
    print()
    print("-" * 60)
    print("Summary")
    print("-" * 60)
    overhead_fwd = (trie_fwd / causal_fwd - 1) * 100
    overhead_bwd = (trie_bwd / causal_bwd - 1) * 100
    overhead_total = ((trie_fwd + trie_bwd) / (causal_fwd + causal_bwd) - 1) * 100

    print(f"Forward overhead:  {overhead_fwd:+.1f}%")
    print(f"Backward overhead: {overhead_bwd:+.1f}%")
    print(f"Total overhead:    {overhead_total:+.1f}%")

    print()
    print("With typical tree data (6.8x duplication ratio):")
    effective_speedup = 6.8 / (1 + overhead_total / 100)
    print(f"  Effective speedup from zero-redundancy: {effective_speedup:.2f}x")

    print()
    print("=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
