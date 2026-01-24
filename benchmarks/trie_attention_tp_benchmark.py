# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark trie attention on Qwen3 30B-A3B MoE model with Tensor Parallelism.

Usage:
    torchrun --nproc_per_node=8 benchmarks/trie_attention_tp_benchmark.py
"""

import argparse
import os
import time
from dataclasses import replace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    SequenceParallel,
)
from torchtitan.config.job_config import PEFT
from torchtitan.models.qwen3 import qwen3_args
from torchtitan.models.qwen3.model.model import Qwen3Model
from torchtitan.models.attention import get_causal_mask_mod, get_trie_causal_mask_mod


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
    # Prefix tokens: linear chain (each is parent of next)
    for i in range(prefix_len):
        tin[:, i] = i
        tout[:, i] = 2 * seq_len - i  # Exit time decreases for ancestors

    # Branch tokens: each is a leaf of the prefix
    for i in range(seq_len - prefix_len):
        pos = prefix_len + i
        tin[:, pos] = prefix_len + i
        tout[:, pos] = prefix_len + i + 1  # Leaf nodes have tight intervals

    return tin, tout


def apply_tp(model, tp_mesh):
    """Apply tensor parallelism to the model."""
    # Parallelize embeddings, norm, and output
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    # Apply TP to each transformer block
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate(), None),
                desired_input_layouts=(Replicate(), Replicate(), None),
            ),
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.q_norm": SequenceParallel(sequence_dim=2),
            "attention.k_norm": SequenceParallel(sequence_dim=2),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }

        # Non-MoE FFN layers
        if not transformer_block.moe_enabled:
            layer_plan.update({
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
            })

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    return model


# MoE TP is handled separately via expert parallelism in production
# For this benchmark, we skip MoE-specific TP and just run with replicated experts


def benchmark_step(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    attention_masks,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
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

    return sum(forward_times) / len(forward_times), sum(backward_times) / len(backward_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--prefix_ratio", type=float, default=0.5)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--benchmark_iters", type=int, default=10)
    parser.add_argument("--model", type=str, default="30B-A3B",
                        choices=["30B-A3B", "10B-A1B", "8B", "4B"])
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create device mesh for TP
    tp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

    if rank == 0:
        print("=" * 60)
        print(f"Trie Attention Benchmark - Qwen3 {args.model} with TP={world_size}")
        print("=" * 60)
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.seq_len}")
        print(f"Prefix ratio: {args.prefix_ratio}")
        print()

    # Get model config
    base_config = qwen3_args[args.model]

    if rank == 0:
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

    dtype = torch.bfloat16
    peft_config = PEFT()

    # Generate data (same on all ranks)
    torch.manual_seed(42)
    vocab_size = base_config.vocab_size
    tokens = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
    tin, tout = generate_trie_data(args.batch_size, args.seq_len, args.prefix_ratio, device)

    # ========== Benchmark Causal Attention ==========
    if rank == 0:
        print("Creating causal model...")

    causal_config = replace(
        base_config,
        use_flex_attn=True,
        attn_mask_type="causal",
        max_seq_len=args.seq_len,
    )

    causal_model = Qwen3Model(causal_config, peft_config).to(device=device, dtype=dtype)
    apply_tp(causal_model, tp_mesh)

    if rank == 0:
        total_params = sum(p.numel() for p in causal_model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")
        print()
        print("Benchmarking causal attention...")

    # Create attention masks for causal model
    causal_masks = causal_model.get_attention_masks(tokens, tokenizer=None, extra_inputs={})

    dist.barrier()
    causal_fwd, causal_bwd = benchmark_step(
        causal_model, tokens, labels, causal_masks,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )

    if rank == 0:
        print(f"  Forward:  {causal_fwd:.2f} ms")
        print(f"  Backward: {causal_bwd:.2f} ms")
        print(f"  Total:    {causal_fwd + causal_bwd:.2f} ms")

    del causal_model
    torch.cuda.empty_cache()

    # ========== Benchmark Trie Attention ==========
    if rank == 0:
        print()
        print("Creating trie attention model...")

    trie_config = replace(
        base_config,
        use_flex_attn=True,
        attn_mask_type="trie_causal",
        max_seq_len=args.seq_len,
    )

    trie_model = Qwen3Model(trie_config, peft_config).to(device=device, dtype=dtype)
    apply_tp(trie_model, tp_mesh)

    if rank == 0:
        print(f"Benchmarking trie attention ({args.prefix_ratio*100:.0f}% shared prefix)...")

    # Create attention masks for trie model
    trie_masks = trie_model.get_attention_masks(
        tokens, tokenizer=None, extra_inputs={"tin": tin, "tout": tout}
    )

    dist.barrier()
    trie_fwd, trie_bwd = benchmark_step(
        trie_model, tokens, labels, trie_masks,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )

    if rank == 0:
        print(f"  Forward:  {trie_fwd:.2f} ms")
        print(f"  Backward: {trie_bwd:.2f} ms")
        print(f"  Total:    {trie_fwd + trie_bwd:.2f} ms")

    # ========== Summary ==========
    if rank == 0:
        print()
        print("-" * 60)
        print("Summary (same sequence length comparison)")
        print("-" * 60)
        overhead_fwd = (trie_fwd / causal_fwd - 1) * 100
        overhead_bwd = (trie_bwd / causal_bwd - 1) * 100
        overhead_total = ((trie_fwd + trie_bwd) / (causal_fwd + causal_bwd) - 1) * 100

        print(f"Forward overhead:  {overhead_fwd:+.1f}%")
        print(f"Backward overhead: {overhead_bwd:+.1f}%")
        print(f"Total overhead:    {overhead_total:+.1f}%")

        print()
        print("Note: Trie attention overhead is offset by token reduction.")
        print("With 6.8x duplication ratio (typical for tree data):")
        effective_speedup = 6.8 / (1 + overhead_total/100)
        print(f"  Effective speedup: {effective_speedup:.2f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
