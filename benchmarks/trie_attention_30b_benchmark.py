# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark trie attention on Qwen3 30B-A3B MoE model.

Usage:
    python benchmarks/trie_attention_30b_benchmark.py [--seq_len=2048] [--batch_size=1]
"""

import argparse
import time
from dataclasses import replace

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.config.job_config import PEFT
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.models.qwen3.model.model import Qwen3Model
from torchtitan.models.attention import get_causal_mask_mod, get_trie_causal_mask_mod


def generate_trie_data(
    batch_size: int,
    seq_len: int,
    prefix_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic trie data with shared prefix."""
    prefix_len = int(seq_len * prefix_ratio)
    branch_len = seq_len - prefix_len

    tin = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    tout = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    # Shared prefix: linear chain structure
    for i in range(prefix_len):
        tin[:, i] = i
        tout[:, i] = 2 * seq_len - i

    # Branch tokens: each batch item has unique tin/tout
    for b in range(batch_size):
        for i in range(branch_len):
            pos = prefix_len + i
            base_tin = prefix_len + b * branch_len + i
            tin[b, pos] = base_tin
            tout[b, pos] = base_tin + 1

    return tin, tout


def benchmark_forward_backward(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    mask_fn,
    mask_args: dict,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> tuple[float, float]:
    """Benchmark forward and backward pass."""
    B, S = tokens.shape
    _compiled_create_block_mask = torch.compile(create_block_mask)

    # Create block mask
    block_mask = _compiled_create_block_mask(
        mask_fn,
        mask_args.get("B", 1),
        None,
        S,
        S,
    )

    # Prepare extra inputs for trie attention
    extra_inputs = mask_args.get("extra_inputs", {})

    # Warmup
    for _ in range(warmup_iters):
        output = model(tokens, **extra_inputs)
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
        output = model(tokens, **extra_inputs)
        torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)

    # Benchmark backward
    backward_times = []
    for _ in range(benchmark_iters):
        output = model(tokens, **extra_inputs)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), labels.view(-1)
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)
        model.zero_grad()

    return sum(forward_times) / len(forward_times), sum(backward_times) / len(backward_times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark trie attention on 30B-A3B")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--prefix_ratio", type=float, default=0.5, help="Ratio of shared prefix")
    parser.add_argument("--warmup_iters", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--benchmark_iters", type=int, default=10, help="Benchmark iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 60)
    print("Trie Attention Benchmark - Qwen3 30B-A3B MoE")
    print("=" * 60)
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Prefix ratio: {args.prefix_ratio}")
    print()

    # Get 30B-A3B config and modify for trie attention
    base_config = qwen3_configs["30B-A3B"]

    # Create causal version
    causal_config = replace(
        base_config,
        use_flex_attn=True,
        attn_mask_type="causal",
        max_seq_len=args.seq_len,
    )

    # Create trie version
    trie_config = replace(
        base_config,
        use_flex_attn=True,
        attn_mask_type="trie_causal",
        max_seq_len=args.seq_len,
    )

    print("Model config (30B-A3B MoE):")
    print(f"  dim: {base_config.dim}")
    print(f"  n_layers: {base_config.n_layers}")
    print(f"  n_heads: {base_config.n_heads}")
    print(f"  hidden_dim: {base_config.hidden_dim}")
    print(f"  moe_enabled: {base_config.moe_enabled}")
    print(f"  num_experts: {base_config.moe_args.num_experts}")
    print(f"  top_k: {base_config.moe_args.top_k}")
    print()

    # Create model (single GPU for now)
    print("Creating model...")
    peft_config = PEFT()

    # Use trie config for the model
    model = Qwen3Model(trie_config, peft_config).to(device=device, dtype=dtype)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e9:.2f}B")
    print()

    # Generate data
    print("Generating data...")
    vocab_size = base_config.vocab_size
    tokens = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)

    # Generate trie data
    tin, tout = generate_trie_data(args.batch_size, args.seq_len, args.prefix_ratio, device)

    print("-" * 60)
    print("Performance Benchmark")
    print("-" * 60)

    # Benchmark causal attention
    print("\nBenchmarking standard causal attention...")
    try:
        causal_model = Qwen3Model(causal_config, peft_config).to(device=device, dtype=dtype)
        causal_fwd, causal_bwd = benchmark_forward_backward(
            causal_model,
            tokens,
            labels,
            get_causal_mask_mod(),
            {"B": 1, "extra_inputs": {}},
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
        )
        print(f"  Forward:  {causal_fwd:.2f} ms")
        print(f"  Backward: {causal_bwd:.2f} ms")
        print(f"  Total:    {causal_fwd + causal_bwd:.2f} ms")
        del causal_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")
        causal_fwd, causal_bwd = None, None

    # Benchmark trie attention
    print(f"\nBenchmarking trie attention ({args.prefix_ratio*100:.0f}% shared prefix)...")
    try:
        trie_fwd, trie_bwd = benchmark_forward_backward(
            model,
            tokens,
            labels,
            get_trie_causal_mask_mod(tin, tout),
            {"B": args.batch_size, "extra_inputs": {"tin": tin, "tout": tout}},
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
        )
        print(f"  Forward:  {trie_fwd:.2f} ms")
        print(f"  Backward: {trie_bwd:.2f} ms")
        print(f"  Total:    {trie_fwd + trie_bwd:.2f} ms")
    except Exception as e:
        print(f"  Error: {e}")
        trie_fwd, trie_bwd = None, None

    # Summary
    if causal_fwd and trie_fwd:
        print()
        print("-" * 60)
        print("Summary")
        print("-" * 60)
        speedup_fwd = causal_fwd / trie_fwd
        speedup_bwd = causal_bwd / trie_bwd
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
