#!/usr/bin/env python3
"""
Test known NaN conditions from upstream issues:
1. Issue #153799: NaN if seq_len NOT multiple of 128 AND both block_mask + score_mod used
2. Issue #146377: Garbage K.grad when K_sliced.is_contiguous() is false
3. Issue #158212: Wrong gradients with compiled flex attention + custom masks
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
        return "None"
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.float().min().item()
    max_val = tensor.float().max().item()
    status = "OK" if not (has_nan or has_inf) else "NaN!" if has_nan else "Inf!"
    print(
        f"[Rank {rank}] {name}: shape={tuple(tensor.shape)}, min={min_val:.6f}, max={max_val:.6f} {status}"
    )
    return status


def noop_score_mod(score, b, h, q_idx, kv_idx):
    """No-op score modifier to test issue #153799."""
    return score


def precompute_freqs_cis(dim: int, seq_len: int, device: torch.device):
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs).float()


def test_condition(
    name,
    seq_len,
    n_heads,
    head_dim,
    use_score_mod,
    make_k_noncontiguous,
    rank,
    device,
    cp_mesh,
):
    """Test a specific condition and return whether NaN occurred."""
    print(f"\n[Rank {rank}] === TEST: {name} ===")
    print(
        f"[Rank {rank}]   seq_len={seq_len} (mod 128 = {seq_len % 128}), use_score_mod={use_score_mod}, k_noncontiguous={make_k_noncontiguous}"
    )

    world_size = dist.get_world_size()
    batch_size = 1
    local_seq_len = seq_len // world_size
    local_start = rank * local_seq_len

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

    # Make K non-contiguous if requested (issue #146377)
    if make_k_noncontiguous:
        # Create non-contiguous K by slicing from a larger tensor
        k_large = (
            torch.randn(
                batch_size,
                n_heads,
                local_seq_len * 2,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        k = k_large[:, :, ::2, :]  # Take every other element - non-contiguous
        print(f"[Rank {rank}]   K.is_contiguous() = {k.is_contiguous()}")

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    # Create CP block mask
    block_mask = create_cp_block_mask(
        causal_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device_mesh=cp_mesh,
    )

    # Create buffers
    input_ids = torch.randint(0, 1000, (batch_size, local_seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, local_seq_len), device=device)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len, device)

    cp_buffers = [input_ids, labels, freqs_cis]
    cp_seq_dims = [1, 1, 0]

    # Compile flex_attention
    if use_score_mod:
        compiled_flex = torch.compile(
            lambda q, k, v, bm: flex_attention(
                q, k, v, block_mask=bm, score_mod=noop_score_mod
            ),
            dynamic=True,
        )
    else:
        compiled_flex = torch.compile(flex_attention, dynamic=True)

    try:
        with context_parallel(
            cp_mesh,
            buffers=cp_buffers,
            buffer_seq_dims=cp_seq_dims,
            no_restore_buffers={input_ids, labels},
        ):
            if use_score_mod:
                out = compiled_flex(q, k, v, block_mask)
            else:
                out = compiled_flex(q, k, v, block_mask=block_mask)

            fwd_status = check_tensor("Forward output", out, rank)

            # Backward
            loss = out.sum()
            loss.backward()

            q_status = check_tensor("Q gradient", q.grad, rank)
            k_status = check_tensor("K gradient", k.grad, rank)
            v_status = check_tensor("V gradient", v.grad, rank)

            has_nan = any(
                s == "NaN!" for s in [fwd_status, q_status, k_status, v_status]
            )
            return has_nan

    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback

        traceback.print_exc()
        return True  # Count errors as failures


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*80}")
    print(f"[Rank {rank}] Testing Known NaN Conditions")
    print(f"{'='*80}")

    # Create CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )

    n_heads = 4
    head_dim = 64
    results = {}

    # Test 1: Sequence length multiple of 128 (should work)
    results["seq256_no_scoremod"] = test_condition(
        "seq_len=256 (mult of 128), NO score_mod",
        seq_len=256,
        n_heads=n_heads,
        head_dim=head_dim,
        use_score_mod=False,
        make_k_noncontiguous=False,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 2: Sequence length NOT multiple of 128 (issue #153799)
    results["seq250_no_scoremod"] = test_condition(
        "seq_len=250 (NOT mult of 128), NO score_mod",
        seq_len=250,
        n_heads=n_heads,
        head_dim=head_dim,
        use_score_mod=False,
        make_k_noncontiguous=False,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 3: Sequence length NOT multiple of 128 WITH score_mod (issue #153799 trigger)
    results["seq250_with_scoremod"] = test_condition(
        "seq_len=250 (NOT mult of 128), WITH score_mod",
        seq_len=250,
        n_heads=n_heads,
        head_dim=head_dim,
        use_score_mod=True,
        make_k_noncontiguous=False,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 4: Multiple of 128 WITH score_mod
    results["seq256_with_scoremod"] = test_condition(
        "seq_len=256 (mult of 128), WITH score_mod",
        seq_len=256,
        n_heads=n_heads,
        head_dim=head_dim,
        use_score_mod=True,
        make_k_noncontiguous=False,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 5: Non-contiguous K (issue #146377)
    results["seq256_noncontig_k"] = test_condition(
        "seq_len=256, non-contiguous K",
        seq_len=256,
        n_heads=n_heads,
        head_dim=head_dim,
        use_score_mod=False,
        make_k_noncontiguous=True,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 6: Larger sequence (typical LLM training)
    results["seq2048_no_scoremod"] = test_condition(
        "seq_len=2048 (large, mult of 128)",
        seq_len=2048,
        n_heads=n_heads,
        head_dim=head_dim,
        use_score_mod=False,
        make_k_noncontiguous=False,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 7: 1025 (issue #153799 specific example)
    results["seq2050_with_scoremod"] = test_condition(
        "seq_len=2050 (NOT mult of 128), WITH score_mod",
        seq_len=2050,
        n_heads=n_heads,
        head_dim=head_dim,
        use_score_mod=True,
        make_k_noncontiguous=False,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Summary
    print(f"\n{'='*80}")
    print(f"[Rank {rank}] === RESULTS SUMMARY ===")
    print(f"{'='*80}")
    for test_name, had_nan in results.items():
        status = "FAIL (NaN/Error)" if had_nan else "PASS"
        print(f"[Rank {rank}] {test_name}: {status}")

    print(f"\n[Rank {rank}] === Debug Complete ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
