#!/usr/bin/env python3
"""
Deep trace of flex_attention + CP to find NaN source.
Hooks into the actual flex_attention HOP to trace values.
"""

import os

import torch
import torch.distributed as dist
from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch.distributed.tensor.experimental._attention import (
    context_parallel,
    create_cp_block_mask,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Store original __call__
_original_flex_attention_call = flex_attention_hop.__call__


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def check_tensor(name, tensor, rank):
    if tensor is None:
        print(f"[Rank {rank}] {name}: None")
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.float().min().item()
    max_val = tensor.float().max().item()
    status = "✓" if not (has_nan or has_inf) else "✗ NaN!" if has_nan else "✗ Inf!"
    print(
        f"[Rank {rank}] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={min_val:.6f}, max={max_val:.6f} {status}"
    )
    return has_nan or has_inf


# Global rank for debugging
_debug_rank = 0


def debug_flex_attention_wrapper(self, *args, **kwargs):
    """Wrapper to trace flex_attention HOP calls."""
    global _debug_rank
    rank = _debug_rank

    print(f"\n[Rank {rank}] === FLEX_ATTENTION HOP CALLED ===")
    print(f"[Rank {rank}] Number of args: {len(args)}")

    if len(args) >= 3:
        query, key, value = args[0], args[1], args[2]
        check_tensor("HOP Query", query, rank)
        check_tensor("HOP Key", key, rank)
        check_tensor("HOP Value", value, rank)

    if len(args) >= 5:
        block_mask = args[4]
        print(f"[Rank {rank}] block_mask type: {type(block_mask)}")
        if isinstance(block_mask, tuple):
            print(f"[Rank {rank}] block_mask tuple length: {len(block_mask)}")
            for i, item in enumerate(block_mask[:5]):
                if isinstance(item, torch.Tensor):
                    check_tensor(f"block_mask[{i}]", item, rank)
                else:
                    print(f"[Rank {rank}] block_mask[{i}]: {item}")

    # Call original
    result = _original_flex_attention_call(*args, **kwargs)

    if isinstance(result, tuple):
        check_tensor("HOP Output[0]", result[0], rank)
        if len(result) > 1 and result[1] is not None:
            check_tensor("HOP Output[1] (logsumexp)", result[1], rank)
    else:
        check_tensor("HOP Output", result, rank)

    print(f"[Rank {rank}] === FLEX_ATTENTION HOP DONE ===\n")
    return result


def main():
    global _debug_rank
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    _debug_rank = rank
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Deep Debug: Flex Attention HOP Tracing")
    print(f"{'='*60}\n")

    # Configuration
    batch_size = 1
    seq_len = 256
    n_heads = 4
    head_dim = 64

    # Create CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )
    cp_rank = cp_mesh.get_local_rank()
    cp_size = cp_mesh.size()
    local_seq_len = seq_len // cp_size

    print(f"[Rank {rank}] seq_len={seq_len}, local_seq_len={local_seq_len}")

    # Create inputs
    torch.manual_seed(42 + rank)
    q = (
        torch.randn(
            batch_size,
            n_heads,
            local_seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    k = (
        torch.randn(
            batch_size,
            n_heads,
            local_seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    v = (
        torch.randn(
            batch_size,
            n_heads,
            local_seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )

    check_tensor("Input Q", q, rank)
    check_tensor("Input K", k, rank)
    check_tensor("Input V", v, rank)

    # Create CP block mask
    block_mask = create_cp_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device_mesh=cp_mesh,
    )
    print(f"[Rank {rank}] BlockMask shape: {block_mask.shape}")

    # Monkey-patch the flex_attention HOP to trace calls
    print(f"\n[Rank {rank}] Installing debug hook on flex_attention HOP...")
    import types

    flex_attention_hop.__call__ = types.MethodType(
        debug_flex_attention_wrapper, flex_attention_hop
    )

    # Test 1: Non-compiled flex_attention with CP
    print(f"\n[Rank {rank}] === TEST 1: Non-compiled with CP ===")
    try:
        with context_parallel(cp_mesh, buffers={}):
            out1 = flex_attention(q, k, v, block_mask=block_mask)
        check_tensor("TEST1 Final Output", out1, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST1 Error: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: Compiled with dynamic=True and CP
    print(f"\n[Rank {rank}] === TEST 2: Compiled dynamic=True with CP ===")
    compiled_flex = torch.compile(flex_attention, dynamic=True)
    try:
        q2 = q.detach().clone()
        k2 = k.detach().clone()
        v2 = v.detach().clone()
        with context_parallel(cp_mesh, buffers={}):
            out2 = compiled_flex(q2, k2, v2, block_mask=block_mask)
        check_tensor("TEST2 Final Output", out2, rank)
    except Exception as e:
        print(f"[Rank {rank}] TEST2 Error: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Check what global K,V look like after all_gather
    print(f"\n[Rank {rank}] === TEST 3: Manual all_gather trace ===")
    import torch.distributed._functional_collectives as ft_c

    try:
        # Manually do what context_parallel does
        global_k = ft_c.all_gather_tensor(
            k.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )
        global_v = ft_c.all_gather_tensor(
            v.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )

        check_tensor("Gathered K (global)", global_k, rank)
        check_tensor("Gathered V (global)", global_v, rank)

        # Now compute attention manually with local Q, global K,V
        print(f"\n[Rank {rank}] Computing attention manually with gathered KV...")

        # Attention scores: Q @ K^T / sqrt(d)
        scores = torch.matmul(q.float(), global_k.float().transpose(-2, -1)) / (
            head_dim**0.5
        )
        check_tensor("Manual scores (Q @ K_global^T)", scores, rank)

        # Apply causal mask based on global positions
        # For rank 0: q_idx 0-127 can attend to kv_idx 0-127 (causal)
        # For rank 1: q_idx 128-255 can attend to kv_idx 0-255 (full first half, causal second half)
        q_start = cp_rank * local_seq_len
        q_end = q_start + local_seq_len
        print(f"[Rank {rank}] Q position range: [{q_start}, {q_end})")

        # Create causal mask for this rank
        q_pos = torch.arange(q_start, q_end, device=device).unsqueeze(
            1
        )  # [local_seq_len, 1]
        kv_pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        causal_mask_manual = (
            (q_pos >= kv_pos).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, local_seq_len, seq_len]

        print(f"[Rank {rank}] Causal mask shape: {causal_mask_manual.shape}")
        print(f"[Rank {rank}] Causal mask sum: {causal_mask_manual.sum().item()}")

        # Apply mask
        masked_scores = scores.masked_fill(~causal_mask_manual, float("-inf"))
        check_tensor("Masked scores", masked_scores, rank)

        # Softmax
        attn_probs = torch.softmax(masked_scores, dim=-1)
        check_tensor("Attention probs", attn_probs, rank)

        # Output
        manual_out = torch.matmul(attn_probs, global_v.float())
        check_tensor("Manual output", manual_out, rank)

    except Exception as e:
        print(f"[Rank {rank}] TEST3 Error: {e}")
        import traceback

        traceback.print_exc()

    # Restore original function
    flex_attention_hop.__call__ = _original_flex_attention_call

    print(f"\n[Rank {rank}] === Debug Complete ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
