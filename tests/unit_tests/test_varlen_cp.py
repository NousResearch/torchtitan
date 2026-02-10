#!/usr/bin/env python3
"""
Test varlen attention with Context Parallelism.
Run with: torchrun --nproc_per_node=2 tests/unit_tests/test_varlen_cp.py
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.models.attention import (
    VarlenMetadata,
    create_varlen_metadata_for_document,
)
from torchtitan.models.deepseek_v3 import DeepSeekV3Model, deepseekv3_args
from torchtitan.config.job_config import PEFT


def test_varlen_cp():
    """Test varlen attention with Context Parallelism."""
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("=" * 60)
        print(f"Testing Varlen Attention with CP (world_size={world_size})")
        print("=" * 60)

    # Create CP mesh
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))

    # Get mini kimi k2 varlen config
    model_args = deepseekv3_args["mini_kimi_k2_varlen"]

    if rank == 0:
        print(f"\nModel: mini_kimi_k2_varlen")
        print(f"  attn_mask_type: {model_args.attn_mask_type}")
        print(f"  dim={model_args.dim}, n_heads={model_args.n_heads}")

    # Create model
    peft_config = PEFT()
    torch.manual_seed(42)
    with device:
        model = DeepSeekV3Model(model_args, peft_config)
    model = model.to(device).to(torch.bfloat16)
    model.init_weights(device)

    # Create input
    batch_size = 2
    seq_len = 64  # Must be divisible by CP world size
    eos_id = 1

    torch.manual_seed(42)  # Same input across ranks
    tokens = torch.randint(2, model_args.vocab_size, (batch_size, seq_len), device=device)
    tokens[:, 31] = eos_id  # Doc boundary
    tokens[:, 63] = eos_id  # End of sequence

    # Create a mock tokenizer
    class MockTokenizer:
        eos_id = 1

    tokenizer = MockTokenizer()

    # Get attention masks with CP mesh
    attention_masks = model.get_attention_masks(tokens, tokenizer, cp_mesh=cp_mesh)

    if rank == 0:
        print(f"\nAttention masks type: {type(attention_masks).__name__}")
        if isinstance(attention_masks, VarlenMetadata):
            print(f"  cu_seq_q: {attention_masks.cu_seq_q}")
            print(f"  max_q: {attention_masks.max_q}")

    # Forward pass
    if rank == 0:
        print("\nRunning forward pass with CP...")

    with torch.no_grad():
        output = model(tokens, attention_masks=attention_masks)

    if rank == 0:
        print(f"Output shape: {output.shape}")

    # Verify output
    assert output.shape == (batch_size, seq_len, model_args.vocab_size), \
        f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), f"Output contains NaN on rank {rank}!"
    assert not torch.isinf(output).any(), f"Output contains Inf on rank {rank}!"

    # Synchronize all ranks
    dist.barrier()

    if rank == 0:
        print("\n✓ Varlen attention with CP test passed!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    test_varlen_cp()
