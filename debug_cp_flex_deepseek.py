#!/usr/bin/env python
"""
Debug script for DeepSeek V3 + CP + FlexAttention.
Properly simulates the training flow.
"""

import argparse
import os

# Import DeepSeek model components
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

from torch.nn.attention.flex_attention import and_masks, BlockMask, flex_attention
from torchtitan.models.attention import create_attention_mask, get_causal_mask_mod
from torchtitan.models.deepseek_v3 import deepseekv3_args, DeepSeekV3Model

# CP imports
try:
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import (
        create_cp_block_mask,
        set_rotate_method,
    )

    HAS_CP_SUPPORT = True
except ImportError:
    HAS_CP_SUPPORT = False
    print("Warning: PyTorch version does not support CP + FlexAttention")


def run_debug(args):
    """Run the debug test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"\n{'='*60}")
    print(f"DeepSeek V3 + CP + FlexAttention Debug (v2)")
    print(f"{'='*60}")
    print(f"Rank: {rank}, World Size: {world_size}")
    print(f"CP Degree: {args.cp_degree}")
    print(f"Device: {device}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Has CP Support: {HAS_CP_SUPPORT}")
    print(f"{'='*60}\n")

    # Initialize distributed if needed
    cp_mesh = None
    if args.cp_degree > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Create device mesh for CP
        cp_mesh = init_device_mesh("cuda", (args.cp_degree,), mesh_dim_names=("cp",))
        print(f"[Rank {rank}] Created CP mesh: {cp_mesh}")

    # Get model config
    config_name = args.model_flavor
    model_args = deepseekv3_args[config_name]

    print(f"\nModel Config ({config_name}):")
    print(f"  n_heads={model_args.n_heads}")
    print(f"  qk_nope_head_dim={model_args.qk_nope_head_dim}")
    print(f"  qk_rope_head_dim={model_args.qk_rope_head_dim}")
    print(f"  v_head_dim={model_args.v_head_dim}")
    print(f"  use_flex_attn={model_args.use_flex_attn}")
    print(f"  attn_mask_type={model_args.attn_mask_type}")

    # FULL sequence length (before CP split)
    full_seq_len = args.seq_len
    batch_size = args.batch_size

    print(f"  full_seq_len={full_seq_len}, batch_size={batch_size}")

    # Create attention mask with FULL sequence length
    print(f"\nCreating attention mask...")
    B = 1 if model_args.attn_mask_type == "causal" else batch_size
    H = model_args.n_heads

    mask_mod = get_causal_mask_mod()

    if cp_mesh is not None and HAS_CP_SUPPORT:
        print(f"  Creating CP block mask with:")
        print(f"    B={B}, H={H}, Q_LEN={full_seq_len}, KV_LEN={full_seq_len}")
        print(f"    device_mesh={cp_mesh}")

        block_mask = create_cp_block_mask(
            mask_mod=mask_mod,
            B=B,
            H=H,
            Q_LEN=full_seq_len,
            KV_LEN=full_seq_len,
            device_mesh=cp_mesh,
        )
    else:
        print(f"  Creating standard block mask")
        block_mask = create_attention_mask(
            mask_mod, B, None, full_seq_len, full_seq_len
        )

    print(f"  Block mask type: {type(block_mask)}")
    if isinstance(block_mask, BlockMask):
        print(f"  Block mask shape: {block_mask.shape}")
        print(f"  Block mask BLOCK_SIZE: {block_mask.BLOCK_SIZE}")
        print(f"  kv_num_blocks: {block_mask.kv_num_blocks.shape}")
        print(f"  q_num_blocks: {block_mask.q_num_blocks.shape}")

    # Create test Q, K, V tensors with FULL sequence length
    torch.manual_seed(42 + rank)

    qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
    v_head_dim = model_args.v_head_dim

    # Create with FULL sequence length (CP context will split them)
    q = torch.randn(
        batch_size,
        H,
        full_seq_len,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        H,
        full_seq_len,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        H,
        full_seq_len,
        v_head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    print(f"\nInitial tensor shapes (FULL seq_len):")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")

    # Try to run FlexAttention
    print(f"\nRunning FlexAttention...")
    softmax_scale = qk_head_dim**-0.5

    try:
        # Setup CP context if needed
        if cp_mesh is not None and HAS_CP_SUPPORT:
            print(f"  Setting up CP context...")
            set_rotate_method("allgather")

            # Pass the tensors to context_parallel for splitting
            # Note: We need to use detached tensors and create new ones inside
            buffers = [q.detach(), k.detach(), v.detach()]
            buffer_seq_dims = [2, 2, 2]  # sequence dimension is dim 2

            cp_context = context_parallel(
                cp_mesh,
                buffers=buffers,
                buffer_seq_dims=buffer_seq_dims,
                no_restore_buffers=set(buffers),
            )

            with cp_context:
                # Get the split tensors
                q_split = buffers[0].requires_grad_(True)
                k_split = buffers[1].requires_grad_(True)
                v_split = buffers[2].requires_grad_(True)

                print(f"  Inside CP context:")
                print(f"    Q shape: {q_split.shape}")
                print(f"    K shape: {k_split.shape}")
                print(f"    V shape: {v_split.shape}")
                print(f"    Block mask shape: {block_mask.shape}")

                # Verify dimensions match
                local_seq_len = q_split.shape[2]
                mask_seq_len = block_mask.shape[2]
                print(f"    local_seq_len from tensor: {local_seq_len}")
                print(f"    mask_seq_len from block_mask: {mask_seq_len}")

                # Check if they match
                if local_seq_len != mask_seq_len:
                    print(f"\n  MISMATCH DETECTED!")
                    print(
                        f"  Tensor seq_len={local_seq_len} vs Block mask seq_len={mask_seq_len}"
                    )
                    print(f"\n  This is the root cause of the bug!")
                    raise ValueError(
                        f"Sequence length mismatch: tensor has {local_seq_len}, mask has {mask_seq_len}"
                    )

                # Use compiled version like the actual code does
                compiled_flex_attn = torch.compile(
                    flex_attention, mode="max-autotune-no-cudagraphs"
                )

                print(f"  Calling flex_attention...")
                output = compiled_flex_attn(
                    q_split,
                    k_split,
                    v_split,
                    block_mask=block_mask,
                    scale=softmax_scale,
                )
                print(f"  Output shape: {output.shape}")
                print(f"  Output has NaN: {torch.isnan(output).any().item()}")

                # Try backward
                print(f"\nRunning backward...")
                loss = output.sum()
                loss.backward()

                print(f"  Backward completed successfully")

        else:
            output = flex_attention(q, k, v, block_mask=block_mask, scale=softmax_scale)
            print(f"  Output shape: {output.shape}")
            print(f"  Output has NaN: {torch.isnan(output).any().item()}")

            # Try backward
            print(f"\nRunning backward...")
            loss = output.sum()
            loss.backward()

        # Final verdict
        print(f"\n{'='*60}")
        print("RESULT: Test passed!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        print(f"\n{'='*60}")
        print("RESULT: Test FAILED!")
        print(f"{'='*60}\n")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Debug DeepSeek V3 CP + FlexAttention")
    parser.add_argument(
        "--cp_degree", type=int, default=1, help="Context parallel degree"
    )
    parser.add_argument("--seq_len", type=int, default=256, help="Full sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--model_flavor",
        type=str,
        default="debugmodel_flex_attn_causal",
        help="Model flavor to use",
    )

    args = parser.parse_args()
    run_debug(args)


if __name__ == "__main__":
    main()
