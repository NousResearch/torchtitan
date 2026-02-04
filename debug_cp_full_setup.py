#!/usr/bin/env python3
"""
Full setup debug for CP + flex_attention, mimicking torchtitan's actual usage.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental._attention import (
    context_parallel,
    create_cp_block_mask,
)
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def check_tensor(name, tensor, rank):
    if tensor is None:
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.float().min().item()
    max_val = tensor.float().max().item()
    status = "OK" if not (has_nan or has_inf) else "NaN!" if has_nan else "Inf!"
    print(
        f"[Rank {rank}] {name}: shape={tuple(tensor.shape)}, min={min_val:.6f}, max={max_val:.6f} {status}"
    )
    return has_nan or has_inf


def precompute_freqs_cis(
    dim: int, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32
):
    """Simplified RoPE frequencies computation."""
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis.to(dtype)


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*70}")
    print(f"[Rank {rank}] Full CP + FlexAttention Setup Test")
    print(f"{'='*70}\n")

    # Configuration (similar to LLaMA 8B-like)
    batch_size = 1
    seq_len = 256
    n_heads = 4
    head_dim = 64

    # Create CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )
    local_seq_len = seq_len // world_size
    local_start = rank * local_seq_len

    print(
        f"[Rank {rank}] seq_len={seq_len}, local_seq_len={local_seq_len}, local_start={local_start}"
    )

    # Create inputs (pre-sharded, as they would be after data loading)
    torch.manual_seed(42)  # Same seed across ranks for consistent data
    # Full tensors first, then shard
    full_input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    full_labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Shard along sequence dimension
    input_ids = full_input_ids[:, local_start : local_start + local_seq_len].clone()
    labels = full_labels[:, local_start : local_start + local_seq_len].clone()

    # Create freqs_cis (this is what gets passed to context_parallel)
    # freqs_cis is computed for the FULL sequence, then sharded by context_parallel
    full_freqs_cis = precompute_freqs_cis(head_dim, seq_len, device)
    print(f"[Rank {rank}] full_freqs_cis shape: {full_freqs_cis.shape}")

    # Create QKV (as they would be after linear projections)
    # These are local tensors for each rank
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

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

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

    # ============ Test 1: Full torchtitan-style setup with buffers ============
    print(f"\n[Rank {rank}] === TEST 1: Full setup with buffers ===")

    # Prepare buffers like torchtitan does
    # cp_buffers = list(input_dict.values()) + [labels] + [m.freqs_cis for m in model_parts]
    # cp_seq_dims = [1] * len(input_dict) + [1] + [0 for _ in model_parts]

    # For our test: input_ids (seq_dim=1), labels (seq_dim=1), freqs_cis (seq_dim=0)
    cp_buffers = [input_ids, labels, full_freqs_cis]
    cp_seq_dims = [
        1,
        1,
        0,
    ]  # input_ids is [B, S], labels is [B, S], freqs_cis is [S, ...]
    no_restore_buffers = {input_ids, labels}

    try:
        with context_parallel(
            cp_mesh,
            buffers=cp_buffers,
            buffer_seq_dims=cp_seq_dims,
            no_restore_buffers=no_restore_buffers,
        ):
            # Inside context_parallel, freqs_cis gets sharded
            print(
                f"[Rank {rank}] Inside CP context, freqs_cis shape: {full_freqs_cis.shape}"
            )

            # Non-compiled flex_attention
            out1 = flex_attention(q, k, v, block_mask=block_mask)
            check_tensor("Non-compiled output", out1, rank)

    except Exception as e:
        print(f"[Rank {rank}] TEST1 Error: {e}")
        import traceback

        traceback.print_exc()

    # ============ Test 2: Compiled flex_attention with full setup ============
    print(f"\n[Rank {rank}] === TEST 2: Compiled (dynamic=True) with full setup ===")

    compiled_flex = torch.compile(flex_attention, dynamic=True)

    # Reset freqs_cis to full size
    full_freqs_cis = precompute_freqs_cis(head_dim, seq_len, device)
    cp_buffers = [input_ids.clone(), labels.clone(), full_freqs_cis]

    try:
        q2, k2, v2 = q.detach().clone(), k.detach().clone(), v.detach().clone()
        q2.requires_grad_(True)
        k2.requires_grad_(True)
        v2.requires_grad_(True)

        with context_parallel(
            cp_mesh,
            buffers=cp_buffers,
            buffer_seq_dims=cp_seq_dims,
            no_restore_buffers=set(cp_buffers[:2]),
        ):
            out2 = compiled_flex(q2, k2, v2, block_mask=block_mask)
            check_tensor("Compiled output (forward)", out2, rank)

            # Test backward pass
            loss = out2.sum()
            loss.backward()
            check_tensor("Q gradient", q2.grad, rank)
            check_tensor("K gradient", k2.grad, rank)
            check_tensor("V gradient", v2.grad, rank)

    except Exception as e:
        print(f"[Rank {rank}] TEST2 Error: {e}")
        import traceback

        traceback.print_exc()

    # ============ Test 3: Manual gather approach (known working) ============
    print(f"\n[Rank {rank}] === TEST 3: Manual gather (known working baseline) ===")
    import torch.distributed._functional_collectives as ft_c

    def cp_causal_mask_manual(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + local_start
        return global_q_idx >= kv_idx

    proper_mask = create_block_mask(
        cp_causal_mask_manual,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    try:
        q3, k3, v3 = q.detach().clone(), k.detach().clone(), v.detach().clone()
        q3.requires_grad_(True)
        k3.requires_grad_(True)
        v3.requires_grad_(True)

        global_k = ft_c.all_gather_tensor(
            k3.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )
        global_v = ft_c.all_gather_tensor(
            v3.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )

        out3 = compiled_flex(q3, global_k, global_v, block_mask=proper_mask)
        check_tensor("Manual gather output (forward)", out3, rank)

        loss3 = out3.sum()
        loss3.backward()
        check_tensor("Q gradient (manual)", q3.grad, rank)

    except Exception as e:
        print(f"[Rank {rank}] TEST3 Error: {e}")
        import traceback

        traceback.print_exc()

    # ============ Compare outputs ============
    print(f"\n[Rank {rank}] === OUTPUT COMPARISON ===")
    try:
        if (
            "out2" in dir()
            and "out3" in dir()
            and out2 is not None
            and out3 is not None
        ):
            diff = (out2 - out3).abs().max().item()
            print(
                f"[Rank {rank}] Max diff between context_parallel and manual: {diff:.6e}"
            )
    except Exception as e:
        print(f"[Rank {rank}] Comparison error: {e}")

    print(f"\n[Rank {rank}] === Debug Complete ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
