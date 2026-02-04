#!/usr/bin/env python3
"""
E2E test V2: Match exactly how torchtitan uses CP + document masking.

Key insight from V1: Manual CP works perfectly (diff=0).
The issue is in how PyTorch's context_parallel interacts with create_cp_block_mask.

Run with: torchrun --nproc_per_node=2 debug_cp_document_mask_e2e_v2.py
"""

import os
import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    and_masks,
    BlockMask,
    create_block_mask,
    flex_attention,
)

try:
    from torch.distributed.tensor.experimental._attention import (
        context_parallel,
        create_cp_block_mask,
    )
except ImportError:
    print("PyTorch version doesn't support create_cp_block_mask")
    sys.exit(1)


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(rank, msg):
    print(f"[Rank {rank}] {msg}", flush=True)


class FlexAttentionWrapper(nn.Module):
    """Same as torchtitan's FlexAttentionWrapper."""

    def forward(self, q, k, v, *, block_mask, scale=None):
        return flex_attention(q, k, v, block_mask=block_mask, scale=scale)


class SimpleModel(nn.Module):
    """Simple model matching torchtitan structure."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.inner_attention = FlexAttentionWrapper()
        self.scale = self.head_dim**-0.5

    def forward(self, x, attention_masks):
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        out = self.inner_attention(
            q, k, v, block_mask=attention_masks, scale=self.scale
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


def _get_document_ids(seq_lens, device):
    batch_document_ids = []
    for sample_lens in seq_lens:
        doc_ids = torch.cat(
            [
                torch.full((l.item(),), i, dtype=torch.long, device=device)
                for i, l in enumerate(sample_lens)
            ]
        )
        batch_document_ids.append(doc_ids)
    return torch.stack(batch_document_ids)


def get_document_causal_mask(document_ids):
    def mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        doc_match = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal & doc_match

    return mask


def check_tensor(name, t, rank):
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    log(
        rank,
        f"{name}: shape={tuple(t.shape)}, min={t.min().item():.4f}, max={t.max().item():.4f}, nan={has_nan}, inf={has_inf}",
    )
    return has_nan or has_inf


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    log(rank, "=" * 70)
    log(rank, "E2E Test V2: Matching torchtitan's CP usage")
    log(rank, "=" * 70)

    # Config
    dim = 128
    n_heads = 4
    batch_size = 1
    full_seq_len = 256
    local_seq_len = full_seq_len // world_size
    local_start = rank * local_seq_len

    # Documents
    doc_sizes = [64, 64, 64, 64]
    seq_lens = [[torch.tensor(s, device=device) for s in doc_sizes]]
    document_ids = _get_document_ids(seq_lens, device)

    log(rank, f"full_seq_len={full_seq_len}, local_seq_len={local_seq_len}")
    log(
        rank,
        f"Rank {rank} handles global positions {local_start}-{local_start+local_seq_len-1}",
    )

    # CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )

    # Model (same on all ranks)
    torch.manual_seed(42)
    model = SimpleModel(dim, n_heads).to(device).to(torch.bfloat16)
    model.eval()

    # Input (FULL sequence - same as dataloader would provide)
    torch.manual_seed(123)
    x_full = torch.randn(
        batch_size, full_seq_len, dim, device=device, dtype=torch.bfloat16
    )

    # Create mask_mod
    mask_mod = get_document_causal_mask(document_ids)

    log(rank, "")
    log(rank, "=" * 50)
    log(rank, "TEST 1: Non-CP Baseline")
    log(rank, "=" * 50)

    if rank == 0:
        baseline_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=n_heads,
            Q_LEN=full_seq_len,
            KV_LEN=full_seq_len,
            device=device,
        )
        with torch.no_grad():
            out_baseline = model(x_full, baseline_mask)
        check_tensor("out_baseline", out_baseline, rank)
    else:
        out_baseline = None

    dist.barrier()

    log(rank, "")
    log(rank, "=" * 50)
    log(rank, "TEST 2: With CP - create_cp_block_mask OUTSIDE context")
    log(rank, "=" * 50)

    # This matches torchtitan: mask created BEFORE entering context_parallel
    cp_mask = create_cp_block_mask(
        mask_mod=mask_mod,
        B=batch_size,
        H=n_heads,
        Q_LEN=full_seq_len,
        KV_LEN=full_seq_len,
        device_mesh=cp_mesh,
    )

    log(rank, f"CP mask shape: {cp_mask.shape}")
    log(rank, f"CP mask seq_lengths: {cp_mask.seq_lengths}")

    # Check CP mask dense representation
    try:
        cp_dense = cp_mask.to_dense()
        log(rank, f"CP mask dense shape: {cp_dense.shape}")

        # Verify some mask values
        test_cases = [
            (0, 0, "first token to first token"),
            (local_seq_len // 2, local_seq_len // 2, "middle token to itself"),
            (local_seq_len - 1, 0, "last local to first global"),
        ]
        for local_q, kv, desc in test_cases:
            if local_q < cp_dense.shape[2] and kv < cp_dense.shape[3]:
                global_q = local_q + local_start
                val = cp_dense[0, 0, local_q, kv].item()
                expected_causal = global_q >= kv
                expected_doc = (document_ids[0, global_q] == document_ids[0, kv]).item()
                expected = expected_causal and expected_doc
                status = "✓" if val == expected else "✗"
                log(
                    rank,
                    f"  {desc}: mask[{local_q},{kv}]={int(val)}, expected={int(expected)} {status}",
                )
    except Exception as e:
        log(rank, f"Could not analyze CP mask: {e}")

    # Run with context_parallel - PASSING FULL x_full, context_parallel will shard it
    log(rank, "")
    log(rank, "Running with context_parallel...")

    try:
        with torch.no_grad():
            with context_parallel(
                cp_mesh,
                buffers=[x_full],
                buffer_seq_dims=[1],
                no_restore_buffers=set([x_full]),
            ):
                log(rank, f"Inside context_parallel, x_full.shape = {x_full.shape}")
                out_cp = model(x_full, cp_mask)

        check_tensor("out_cp", out_cp, rank)
        log(rank, f"CP output shape: {out_cp.shape}")

        # The output should be local (sharded)
        # Compare with baseline's local portion
        if rank == 0 and out_baseline is not None:
            out_baseline_local = out_baseline[
                :, local_start : local_start + local_seq_len, :
            ]
            diff = (out_cp - out_baseline_local).abs()
            log(rank, f"CP vs Baseline local: max_diff={diff.max().item():.6f}")
    except Exception as e:
        log(rank, f"context_parallel FAILED: {e}")
        import traceback

        traceback.print_exc()

    log(rank, "")
    log(rank, "=" * 50)
    log(rank, "TEST 3: Manual CP (proven correct in V1)")
    log(rank, "=" * 50)

    # Manual CP: local Q, gathered K/V, offset-aware mask
    x_local = x_full[:, local_start : local_start + local_seq_len, :].clone()

    def manual_mask_mod(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + local_start
        causal = global_q_idx >= kv_idx
        doc = document_ids[b, global_q_idx] == document_ids[b, kv_idx]
        return causal & doc

    manual_mask = create_block_mask(
        manual_mask_mod,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=full_seq_len,
        device=device,
    )

    with torch.no_grad():
        q = (
            model.wq(x_local)
            .view(batch_size, local_seq_len, n_heads, dim // n_heads)
            .transpose(1, 2)
        )
        k_local = (
            model.wk(x_local)
            .view(batch_size, local_seq_len, n_heads, dim // n_heads)
            .transpose(1, 2)
        )
        v_local = (
            model.wv(x_local)
            .view(batch_size, local_seq_len, n_heads, dim // n_heads)
            .transpose(1, 2)
        )

        # Gather K, V
        k_global = ft_c.all_gather_tensor(
            k_local.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )
        v_global = ft_c.all_gather_tensor(
            v_local.contiguous(), gather_dim=2, group=cp_mesh.get_group()
        )

        attn_out = flex_attention(
            q, k_global, v_global, block_mask=manual_mask, scale=model.scale
        )
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, local_seq_len, -1)
        )
        out_manual = model.wo(attn_out)

    check_tensor("out_manual", out_manual, rank)

    # Compare manual with baseline
    if rank == 0 and out_baseline is not None:
        out_baseline_local = out_baseline[
            :, local_start : local_start + local_seq_len, :
        ]
        diff = (out_manual - out_baseline_local).abs()
        log(rank, f"Manual CP vs Baseline: max_diff={diff.max().item():.6f}")

    dist.barrier()

    if rank == 0:
        log(rank, "")
        log(rank, "=" * 70)
        log(rank, "CONCLUSION:")
        log(rank, "- Manual CP with offset-aware mask works correctly")
        log(
            rank,
            "- If PyTorch CP fails or differs, the bug is in PyTorch's CP/mask interaction",
        )
        log(rank, "=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
