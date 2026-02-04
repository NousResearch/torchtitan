#!/usr/bin/env python3
"""
End-to-end test for document masking with Context Parallel.

This test:
1. Creates a simple transformer model
2. Runs forward pass with document masking (CP enabled)
3. Runs forward pass with document masking (CP disabled, but simulated by gathering)
4. Compares outputs/losses

Run with: torchrun --nproc_per_node=2 debug_cp_document_mask_e2e.py
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


class SimpleAttention(nn.Module):
    """Minimal attention for testing."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim**-0.5

    def forward(self, x, block_mask):
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        out = flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


def _get_document_ids(seq_lens, device):
    """Create document IDs from sequence lengths."""
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


def get_causal_mask():
    def mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return mask


def get_document_mask(document_ids):
    def mask(b, h, q_idx, kv_idx):
        return document_ids[b, q_idx] == document_ids[b, kv_idx]

    return mask


def check_tensor(name, t, rank):
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    log(
        rank,
        f"{name}: shape={tuple(t.shape)}, min={t.min().item():.4f}, max={t.max().item():.4f}, "
        f"mean={t.mean().item():.4f}, nan={has_nan}, inf={has_inf}",
    )
    return has_nan or has_inf


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    log(rank, "=" * 70)
    log(rank, "E2E Test: Document Masking with Context Parallel")
    log(rank, "=" * 70)

    # Model config
    dim = 128
    n_heads = 4
    head_dim = dim // n_heads
    batch_size = 1
    full_seq_len = 256
    local_seq_len = full_seq_len // world_size
    local_start = rank * local_seq_len

    # Document structure: 4 documents
    doc_sizes = [64, 64, 64, 64]
    seq_lens = [[torch.tensor(s, device=device) for s in doc_sizes]]
    document_ids = _get_document_ids(seq_lens, device)

    log(
        rank,
        f"Config: dim={dim}, n_heads={n_heads}, full_seq_len={full_seq_len}, local_seq_len={local_seq_len}",
    )
    log(rank, f"Documents: {doc_sizes}")
    log(
        rank,
        f"This rank handles global positions {local_start} to {local_start + local_seq_len - 1}",
    )

    # Create CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )

    # Create model (same weights on all ranks)
    torch.manual_seed(42)
    model = SimpleAttention(dim, n_heads).to(device).to(torch.bfloat16)
    model.eval()

    # Create input (same on all ranks, will be sharded for CP)
    torch.manual_seed(123)
    x_full = torch.randn(
        batch_size, full_seq_len, dim, device=device, dtype=torch.bfloat16
    )
    x_local = x_full[:, local_start : local_start + local_seq_len, :].clone()

    check_tensor("x_full", x_full, rank)
    check_tensor("x_local", x_local, rank)

    # Create combined mask mod
    causal_mask = get_causal_mask()
    doc_mask = get_document_mask(document_ids)
    combined_mask = and_masks(causal_mask, doc_mask)

    log(rank, "")
    log(rank, "=" * 50)
    log(rank, "TEST 1: Non-CP baseline (full sequence)")
    log(rank, "=" * 50)

    # Test 1: Non-CP baseline - use full sequence on rank 0 only
    if rank == 0:
        full_mask = create_block_mask(
            combined_mask,
            B=batch_size,
            H=n_heads,
            Q_LEN=full_seq_len,
            KV_LEN=full_seq_len,
            device=device,
        )
        with torch.no_grad():
            out_baseline = model(x_full, full_mask)
        check_tensor("out_baseline (full)", out_baseline, rank)
        # Extract the local portion for comparison
        out_baseline_local = out_baseline[
            :, local_start : local_start + local_seq_len, :
        ].clone()
    else:
        out_baseline_local = None

    dist.barrier()

    log(rank, "")
    log(rank, "=" * 50)
    log(rank, "TEST 2: Manual CP (gather K,V, local Q)")
    log(rank, "=" * 50)

    # Test 2: Manual CP simulation
    # - Local Q
    # - Gather K, V globally
    # - Use mask that accounts for local Q offset

    def manual_cp_mask_mod(b, h, q_idx, kv_idx):
        global_q_idx = q_idx + local_start
        causal = global_q_idx >= kv_idx
        doc = document_ids[b, global_q_idx] == document_ids[b, kv_idx]
        return causal & doc

    manual_mask = create_block_mask(
        manual_cp_mask_mod,
        B=batch_size,
        H=n_heads,
        Q_LEN=local_seq_len,
        KV_LEN=full_seq_len,
        device=device,
    )

    # Manually compute Q, K, V
    with torch.no_grad():
        q_local = (
            model.wq(x_local)
            .view(batch_size, local_seq_len, n_heads, head_dim)
            .transpose(1, 2)
        )
        k_local = (
            model.wk(x_local)
            .view(batch_size, local_seq_len, n_heads, head_dim)
            .transpose(1, 2)
        )
        v_local = (
            model.wv(x_local)
            .view(batch_size, local_seq_len, n_heads, head_dim)
            .transpose(1, 2)
        )

    check_tensor("q_local", q_local, rank)
    check_tensor("k_local", k_local, rank)
    check_tensor("v_local", v_local, rank)

    # Gather K, V
    k_global = ft_c.all_gather_tensor(
        k_local.contiguous(), gather_dim=2, group=cp_mesh.get_group()
    )
    v_global = ft_c.all_gather_tensor(
        v_local.contiguous(), gather_dim=2, group=cp_mesh.get_group()
    )

    check_tensor("k_global (gathered)", k_global, rank)
    check_tensor("v_global (gathered)", v_global, rank)

    # Run attention
    with torch.no_grad():
        attn_out = flex_attention(
            q_local, k_global, v_global, block_mask=manual_mask, scale=model.scale
        )
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, local_seq_len, -1)

    with torch.no_grad():
        out_manual_cp = model.wo(attn_out)

    check_tensor("out_manual_cp", out_manual_cp, rank)

    log(rank, "")
    log(rank, "=" * 50)
    log(rank, "TEST 3: PyTorch create_cp_block_mask + context_parallel")
    log(rank, "=" * 50)

    # Test 3: Using PyTorch's CP mechanism
    cp_mask = create_cp_block_mask(
        mask_mod=combined_mask,
        B=batch_size,
        H=n_heads,
        Q_LEN=full_seq_len,
        KV_LEN=full_seq_len,
        device_mesh=cp_mesh,
    )

    log(rank, f"CP mask shape: {cp_mask.shape}")
    log(rank, f"CP mask seq_lengths: {cp_mask.seq_lengths}")

    try:
        cp_dense = cp_mask.to_dense()
        log(rank, f"CP mask dense shape: {cp_dense.shape}")
        # Check some values
        for lq in [0, local_seq_len // 2, local_seq_len - 1]:
            gq = lq + local_start
            for kv in [0, gq, full_seq_len - 1]:
                if (
                    kv < full_seq_len
                    and lq < cp_dense.shape[2]
                    and kv < cp_dense.shape[3]
                ):
                    val = cp_dense[0, 0, lq, kv].item()
                    expected_causal = gq >= kv
                    expected_doc = document_ids[0, gq] == document_ids[0, kv]
                    expected = expected_causal and expected_doc.item()
                    status = "✓" if val == expected else "✗"
                    log(
                        rank,
                        f"  Mask[lq={lq}, gq={gq}, kv={kv}] = {int(val)}, expected={int(expected)} {status}",
                    )
    except Exception as e:
        log(rank, f"Could not analyze CP mask: {e}")

    # Run with context_parallel
    with torch.no_grad():
        try:
            with context_parallel(
                cp_mesh,
                buffers=[x_local],
                buffer_seq_dims=[1],
                no_restore_buffers=set(),
            ):
                # Inside context_parallel, flex_attention is wrapped
                out_pytorch_cp = model(x_local.clone(), cp_mask)
            check_tensor("out_pytorch_cp", out_pytorch_cp, rank)
        except Exception as e:
            log(rank, f"PyTorch CP failed: {e}")
            import traceback

            traceback.print_exc()
            out_pytorch_cp = None

    log(rank, "")
    log(rank, "=" * 50)
    log(rank, "COMPARISON")
    log(rank, "=" * 50)

    # Gather baseline from rank 0 for comparison
    if rank == 0 and out_baseline_local is not None:
        baseline_to_compare = out_baseline_local
    else:
        baseline_to_compare = None

    # Compare manual CP vs baseline
    if out_baseline_local is not None:
        diff = (out_manual_cp - out_baseline_local).abs()
        log(
            rank,
            f"Manual CP vs Baseline: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}",
        )

    # Compare PyTorch CP vs manual CP
    if out_pytorch_cp is not None:
        diff = (out_pytorch_cp - out_manual_cp).abs()
        log(
            rank,
            f"PyTorch CP vs Manual CP: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}",
        )

    # Compare PyTorch CP vs baseline (if on rank 0)
    if out_pytorch_cp is not None and out_baseline_local is not None:
        diff = (out_pytorch_cp - out_baseline_local).abs()
        log(
            rank,
            f"PyTorch CP vs Baseline: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}",
        )

    dist.barrier()

    if rank == 0:
        log(rank, "")
        log(rank, "=" * 70)
        log(rank, "Summary:")
        log(rank, "- If 'Manual CP vs Baseline' is small, manual CP is correct")
        log(rank, "- If 'PyTorch CP vs Manual CP' is small, PyTorch CP works correctly")
        log(
            rank, "- If 'PyTorch CP vs Manual CP' is large, there's a bug in PyTorch CP"
        )
        log(rank, "=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
