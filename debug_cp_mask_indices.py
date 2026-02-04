#!/usr/bin/env python3
"""
Debug script to check if create_cp_block_mask passes local or global indices.

Run with: torchrun --nproc_per_node=4 debug_cp_mask_indices.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# Try to import create_cp_block_mask
try:
    from torch.distributed.tensor.experimental._attention import create_cp_block_mask

    HAS_CP_BLOCK_MASK = True
except ImportError:
    HAS_CP_BLOCK_MASK = False
    print("create_cp_block_mask not available")


def setup_distributed():
    """Initialize distributed environment."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def test_cp_block_mask_indices():
    """Test what indices create_cp_block_mask passes to mask_mod."""

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    # Create CP mesh
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))

    # Simulate document packing with sequence lengths
    # Full sequence length = 1024, with 4 documents of varying lengths
    FULL_SEQ_LEN = 1024
    LOCAL_SEQ_LEN = FULL_SEQ_LEN // world_size  # 256 per rank with 4 GPUs

    # Document lengths: [300, 250, 274, 200] = 1024 total
    seq_lens = [torch.tensor([300, 250, 274, 200], device=device)]

    # Create document_ids for FULL sequence (like the real code does)
    def get_document_ids_from_seq_lens(seq_lens):
        batch_document_ids = []
        for sample_idx in range(len(seq_lens)):
            document_ids = torch.cat(
                [
                    torch.full((seq_len.item(),), i, dtype=torch.long, device=device)
                    for i, seq_len in enumerate(seq_lens[sample_idx])
                ]
            )
            batch_document_ids.append(document_ids)
        return torch.stack(batch_document_ids)

    document_ids = get_document_ids_from_seq_lens(seq_lens)  # Shape: [1, 1024]

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Configuration:")
        print(f"  World size (CP): {world_size}")
        print(f"  Full seq len: {FULL_SEQ_LEN}")
        print(f"  Local seq len per rank: {LOCAL_SEQ_LEN}")
        print(
            f"  Document boundaries: 0-299 (doc0), 300-549 (doc1), 550-823 (doc2), 824-1023 (doc3)"
        )
        print(f"  document_ids shape: {document_ids.shape}")
        print(f"{'='*60}\n")

    dist.barrier()

    # Track indices passed to mask_mod
    observed_q_indices = []
    observed_kv_indices = []
    call_count = [0]

    def debug_mask_mod(b, h, q_idx, kv_idx):
        """Mask mod that logs the indices it receives."""
        # Only log first few calls to avoid spam
        if call_count[0] < 20:
            # Store for analysis
            if isinstance(q_idx, torch.Tensor):
                observed_q_indices.append((q_idx.min().item(), q_idx.max().item()))
            else:
                observed_q_indices.append((q_idx, q_idx))

            if isinstance(kv_idx, torch.Tensor):
                observed_kv_indices.append((kv_idx.min().item(), kv_idx.max().item()))
            else:
                observed_kv_indices.append((kv_idx, kv_idx))

        call_count[0] += 1

        # Actual mask logic (causal + document)
        causal_mask = q_idx >= kv_idx

        # This is where the bug would manifest:
        # If q_idx is LOCAL (0-255), indexing document_ids[b, q_idx] gets wrong values
        # If q_idx is GLOBAL (rank*256 to (rank+1)*256-1), it works correctly
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]

        return causal_mask & document_mask

    # Create CP block mask
    if rank == 0:
        print(f"Creating CP block mask...")

    block_mask = create_cp_block_mask(
        mask_mod=debug_mask_mod,
        B=1,  # batch size
        H=None,  # heads (None = broadcast)
        Q_LEN=FULL_SEQ_LEN,
        KV_LEN=FULL_SEQ_LEN,
        device_mesh=cp_mesh,
    )

    dist.barrier()

    # Report findings
    print(f"\n[Rank {rank}] Results:")
    print(f"  mask_mod was called {call_count[0]} times")
    print(f"  block_mask type: {type(block_mask)}")
    if hasattr(block_mask, "shape"):
        print(f"  block_mask shape: {block_mask.shape}")

    if observed_q_indices:
        q_min = min(x[0] for x in observed_q_indices)
        q_max = max(x[1] for x in observed_q_indices)
        kv_min = min(x[0] for x in observed_kv_indices)
        kv_max = max(x[1] for x in observed_kv_indices)

        # Expected ranges
        expected_q_start_global = rank * LOCAL_SEQ_LEN
        expected_q_end_global = (rank + 1) * LOCAL_SEQ_LEN - 1

        print(f"\n  Q indices observed: min={q_min}, max={q_max}")
        print(f"  KV indices observed: min={kv_min}, max={kv_max}")
        print(
            f"\n  Expected Q range if GLOBAL: [{expected_q_start_global}, {expected_q_end_global}]"
        )
        print(f"  Expected Q range if LOCAL:  [0, {LOCAL_SEQ_LEN - 1}]")

        if (
            q_min >= expected_q_start_global
            and q_max <= expected_q_end_global
            and rank > 0
        ):
            print(
                f"\n  >>> GLOBAL indices - mask_mod receives correct global positions"
            )
        elif q_max < LOCAL_SEQ_LEN:
            print(f"\n  >>> LOCAL indices - BUG! mask_mod receives local positions")
            print(f"      document_ids indexing will be WRONG for rank > 0")
        else:
            print(f"\n  >>> UNCLEAR - need more analysis")

    dist.barrier()

    # Additional test: manually check what document_ids values we'd get
    if rank > 0:
        print(f"\n[Rank {rank}] Document ID check:")
        print(
            f"  If q_idx=0 (local), document_ids[0, 0] = {document_ids[0, 0].item()} (doc 0)"
        )
        global_q_start = rank * LOCAL_SEQ_LEN
        print(
            f"  If q_idx={global_q_start} (global), document_ids[0, {global_q_start}] = {document_ids[0, global_q_start].item()}"
        )
        print(
            f"  These should be DIFFERENT if rank > 0 and we cross document boundaries"
        )

    dist.barrier()
    dist.destroy_process_group()


def test_without_cp():
    """Test mask_mod indices without CP for comparison."""
    from torch.nn.attention.flex_attention import create_block_mask

    device = torch.device("cuda:0")
    SEQ_LEN = 1024

    seq_lens = [torch.tensor([300, 250, 274, 200], device=device)]

    def get_document_ids_from_seq_lens(seq_lens):
        batch_document_ids = []
        for sample_idx in range(len(seq_lens)):
            document_ids = torch.cat(
                [
                    torch.full((seq_len.item(),), i, dtype=torch.long, device=device)
                    for i, seq_len in enumerate(seq_lens[sample_idx])
                ]
            )
            batch_document_ids.append(document_ids)
        return torch.stack(batch_document_ids)

    document_ids = get_document_ids_from_seq_lens(seq_lens)

    observed_q = []
    call_count = [0]

    def debug_mask_mod(b, h, q_idx, kv_idx):
        if call_count[0] < 10:
            if isinstance(q_idx, torch.Tensor):
                observed_q.append((q_idx.min().item(), q_idx.max().item()))
        call_count[0] += 1
        return (q_idx >= kv_idx) & (document_ids[b, q_idx] == document_ids[b, kv_idx])

    print("Testing WITHOUT CP:")
    block_mask = create_block_mask(
        debug_mask_mod, B=1, H=None, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN
    )

    if observed_q:
        print(
            f"  Q indices range: {min(x[0] for x in observed_q)} to {max(x[1] for x in observed_q)}"
        )
        print(f"  (Should be 0 to {SEQ_LEN-1} for full sequence)")


if __name__ == "__main__":
    if not HAS_CP_BLOCK_MASK:
        print("create_cp_block_mask not available, skipping CP test")
        exit(1)

    # Check if we're in distributed mode
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        test_cp_block_mask_indices()
    else:
        print("Run with: torchrun --nproc_per_node=4 debug_cp_mask_indices.py")
        print("\nRunning non-distributed test first...\n")
        torch.cuda.set_device(0)
        test_without_cp()
