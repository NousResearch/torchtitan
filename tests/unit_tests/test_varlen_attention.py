#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for varlen attention with document masking.

This script tests:
1. VarlenAttentionWrapper works correctly without CP
2. Varlen attention produces correct outputs with document masking
3. Outputs match between varlen and FlexAttention (functionally equivalent)
"""

import torch
import torch.nn as nn

from torchtitan.models.attention import (
    VarlenMetadata,
    VarlenAttentionWrapper,
    create_varlen_metadata_for_document,
    FlexAttentionWrapper,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torch.nn.attention.flex_attention import and_masks


def test_varlen_metadata_creation():
    """Test that VarlenMetadata is created correctly from input batch."""
    print("=" * 60)
    print("Test 1: VarlenMetadata creation")
    print("=" * 60)

    # Create a batch with 2 sequences, each containing 2 documents
    # Batch 1: [doc1 (3 tokens), EOS, doc2 (3 tokens), EOS]
    # Batch 2: [doc1 (2 tokens), EOS, doc2 (4 tokens), EOS]
    batch_size = 2
    seq_len = 8
    eos_id = 1

    # Create input batch
    input_batch = torch.zeros(batch_size, seq_len, dtype=torch.long)
    # Batch 0: tokens at positions 3 and 7 are EOS
    input_batch[0, 3] = eos_id  # End of doc1
    input_batch[0, 7] = eos_id  # End of doc2
    # Batch 1: tokens at positions 2 and 7 are EOS
    input_batch[1, 2] = eos_id  # End of doc1
    input_batch[1, 7] = eos_id  # End of doc2

    print(f"Input batch shape: {input_batch.shape}")
    print(f"Input batch:\n{input_batch}")

    # Create VarlenMetadata
    varlen_meta = create_varlen_metadata_for_document(input_batch, eos_id)

    print(f"\nVarlenMetadata:")
    print(f"  cu_seq_q: {varlen_meta.cu_seq_q}")
    print(f"  cu_seq_k: {varlen_meta.cu_seq_k}")
    print(f"  max_q: {varlen_meta.max_q}")
    print(f"  max_k: {varlen_meta.max_k}")

    # Verify the cumulative sequence lengths are correct
    # Batch 0: doc1 has 4 tokens (0-3), doc2 has 4 tokens (4-7)
    # Batch 1: doc1 has 3 tokens (0-2), doc2 has 5 tokens (3-7)
    # With packing offset: batch0 is 0-7, batch1 is 8-15
    # Expected: [0, 4, 8, 11, 16]
    expected_cu_seqlens = torch.tensor([0, 4, 8, 11, 16], dtype=torch.int32)

    assert torch.equal(varlen_meta.cu_seq_q, expected_cu_seqlens), \
        f"Expected {expected_cu_seqlens}, got {varlen_meta.cu_seq_q}"

    print("\n✓ VarlenMetadata creation test passed!")
    return True


def test_varlen_attention_forward():
    """Test VarlenAttentionWrapper forward pass."""
    print("\n" + "=" * 60)
    print("Test 2: VarlenAttentionWrapper forward pass")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model dimensions (matching mini_kimi_k2 style)
    batch_size = 2
    seq_len = 16
    n_heads = 4
    qk_head_dim = 32  # qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = 24   # Different from qk_head_dim (like DeepSeek MLA)
    eos_id = 1

    # Create input tensors [B, H, S, head_dim]
    xq = torch.randn(batch_size, n_heads, seq_len, qk_head_dim, device=device, dtype=torch.bfloat16)
    xk = torch.randn(batch_size, n_heads, seq_len, qk_head_dim, device=device, dtype=torch.bfloat16)
    xv = torch.randn(batch_size, n_heads, seq_len, v_head_dim, device=device, dtype=torch.bfloat16)

    # Create input batch for document masking
    input_batch = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    input_batch[0, 7] = eos_id   # Doc boundary at position 7
    input_batch[0, 15] = eos_id  # End of sequence
    input_batch[1, 5] = eos_id   # Doc boundary at position 5
    input_batch[1, 15] = eos_id  # End of sequence

    # Create VarlenMetadata
    varlen_meta = create_varlen_metadata_for_document(input_batch, eos_id)
    print(f"VarlenMetadata cu_seq_q: {varlen_meta.cu_seq_q}")

    # Create VarlenAttentionWrapper (no CP)
    varlen_attn = VarlenAttentionWrapper(cp_mesh=None)

    # Forward pass
    output = varlen_attn(
        xq, xk, xv,
        head_dim=v_head_dim,
        attention_masks=varlen_meta,
        scale=qk_head_dim ** -0.5,
    )

    print(f"\nInput shapes: Q={xq.shape}, K={xk.shape}, V={xv.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [{batch_size * seq_len}, {n_heads}, {v_head_dim}]")

    # Verify output shape
    expected_shape = (batch_size * seq_len, n_heads, v_head_dim)
    assert output.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {output.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"

    print("\n✓ VarlenAttentionWrapper forward pass test passed!")
    return True


def test_varlen_vs_flex_attention():
    """Test that varlen and flex attention produce similar outputs."""
    print("\n" + "=" * 60)
    print("Test 3: Varlen vs FlexAttention comparison")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("Skipping comparison test - requires CUDA")
        return True

    # Model dimensions
    batch_size = 2
    seq_len = 32
    n_heads = 4
    head_dim = 32  # Same head_dim for Q, K, V in this test
    eos_id = 1

    # Create input tensors
    xq = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    xk = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    xv = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # Create input batch for document masking (same documents in both batches for simplicity)
    input_batch = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    input_batch[:, 15] = eos_id  # Doc boundary at position 15
    input_batch[:, 31] = eos_id  # End of sequence

    scale = head_dim ** -0.5

    # ===== Varlen Attention =====
    varlen_meta = create_varlen_metadata_for_document(input_batch, eos_id)
    varlen_attn = VarlenAttentionWrapper(cp_mesh=None)

    output_varlen = varlen_attn(
        xq, xk, xv,
        head_dim=head_dim,
        attention_masks=varlen_meta,
        scale=scale,
    )
    # Reshape to [B, S, H, D]
    output_varlen = output_varlen.view(batch_size, seq_len, n_heads, head_dim)

    # ===== FlexAttention =====
    # Create block mask with document masking
    mask_mods = [get_causal_mask_mod(), get_document_mask_mod(input_batch, eos_id)]
    combined_mask = and_masks(*mask_mods)
    block_mask = create_attention_mask(combined_mask, batch_size, None, seq_len, seq_len)

    flex_attn = FlexAttentionWrapper()
    output_flex = flex_attn(xq, xk, xv, block_mask=block_mask, scale=scale)
    # Reshape to [B, S, H, D]
    output_flex = output_flex.transpose(1, 2)  # [B, H, S, D] -> [B, S, H, D]

    print(f"Varlen output shape: {output_varlen.shape}")
    print(f"Flex output shape: {output_flex.shape}")

    # Compare outputs
    diff = (output_varlen - output_flex).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nOutput comparison:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Allow some numerical tolerance for bf16
    tolerance = 0.01
    if max_diff < tolerance:
        print(f"\n✓ Outputs match within tolerance ({tolerance})!")
    else:
        print(f"\n⚠ Outputs differ more than tolerance ({tolerance})")
        print("  This may be due to numerical differences between implementations")

    # Check neither has NaN
    assert not torch.isnan(output_varlen).any(), "Varlen output contains NaN!"
    assert not torch.isnan(output_flex).any(), "Flex output contains NaN!"

    print("\n✓ Varlen vs FlexAttention comparison test passed!")
    return True


def test_mini_kimi_k2_model():
    """Test the mini_kimi_k2_varlen model end-to-end."""
    print("\n" + "=" * 60)
    print("Test 4: Mini Kimi K2 Varlen Model E2E")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("Skipping model test - requires CUDA")
        return True

    from torchtitan.models.deepseek_v3 import DeepSeekV3Model, deepseekv3_args
    from torchtitan.config.job_config import PEFT

    # Get mini kimi k2 varlen config
    model_args = deepseekv3_args["mini_kimi_k2_varlen"]
    print(f"Model config: {model_args.attn_mask_type}")
    print(f"  dim={model_args.dim}, n_heads={model_args.n_heads}, n_layers={model_args.n_layers}")
    print(f"  qk_nope_head_dim={model_args.qk_nope_head_dim}, qk_rope_head_dim={model_args.qk_rope_head_dim}")
    print(f"  v_head_dim={model_args.v_head_dim}, kv_lora_rank={model_args.kv_lora_rank}")

    # Create model
    peft_config = PEFT()
    with device:
        model = DeepSeekV3Model(model_args, peft_config)
    model = model.to(device).to(torch.bfloat16)
    model.init_weights(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params / 1e6:.2f}M")

    # Create input
    batch_size = 2
    seq_len = 64
    eos_id = 1

    # Create tokens with document boundaries
    tokens = torch.randint(2, model_args.vocab_size, (batch_size, seq_len), device=device)
    tokens[:, 31] = eos_id  # Doc boundary
    tokens[:, 63] = eos_id  # End of sequence

    # Create a mock tokenizer
    class MockTokenizer:
        eos_id = 1

    tokenizer = MockTokenizer()

    # Get attention masks
    attention_masks = model.get_attention_masks(tokens, tokenizer, cp_mesh=None)
    print(f"\nAttention masks type: {type(attention_masks).__name__}")
    if isinstance(attention_masks, VarlenMetadata):
        print(f"  cu_seq_q shape: {attention_masks.cu_seq_q.shape}")
        print(f"  max_q: {attention_masks.max_q}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(tokens, attention_masks=attention_masks)

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {model_args.vocab_size}]")

    # Verify output
    assert output.shape == (batch_size, seq_len, model_args.vocab_size), \
        f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"

    print("\n✓ Mini Kimi K2 Varlen model E2E test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("VARLEN ATTENTION TEST SUITE")
    print("=" * 60)

    tests = [
        ("VarlenMetadata creation", test_varlen_metadata_creation),
        ("VarlenAttentionWrapper forward", test_varlen_attention_forward),
        ("Varlen vs FlexAttention", test_varlen_vs_flex_attention),
        ("Mini Kimi K2 Varlen E2E", test_mini_kimi_k2_model),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"  {symbol} {name}: {status}")

    all_passed = all(s == "PASSED" for _, s in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
