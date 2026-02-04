#!/usr/bin/env python3
"""
Minimal test to demonstrate the document masking bug with Context Parallel.

The bug: document_ids tensor uses GLOBAL indices, but create_cp_block_mask
passes LOCAL indices to mask_mod functions.

Run with: torchrun --nproc_per_node=2 debug_document_mask_cp_bug.py
"""

import os

import torch
import torch.distributed as dist


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def _get_document_ids_from_seq_lens(seq_lens):
    """Exact copy from torchtitan/models/attention.py:190-216"""
    batch_size = len(seq_lens)
    batch_document_ids = []
    for sample_idx in range(batch_size):
        document_ids = torch.cat(
            [
                torch.full(
                    (seq_len,),
                    i,
                    dtype=torch.long,
                    device=seq_lens[sample_idx][0].device,
                )
                for i, seq_len in enumerate(seq_lens[sample_idx])
            ]
        )
        batch_document_ids.append(document_ids)
    return torch.stack(batch_document_ids)


def get_block_causal_mask_mod_by_seq_lens(seq_lens):
    """Exact copy from torchtitan/models/attention.py:219-235"""
    document_ids = _get_document_ids_from_seq_lens(seq_lens)

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal_mask & document_mask

    return mask_mod, document_ids


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    print(f"\n{'='*70}")
    print(f"[Rank {rank}] Document Mask + CP Bug Demonstration")
    print(f"{'='*70}\n")

    # Simulate a packed sequence with 3 documents
    # Total seq_len = 16, documents of length [4, 6, 6]
    full_seq_len = 16
    local_seq_len = full_seq_len // world_size  # 8 per rank with CP=2

    # Document boundaries: [0-3]=doc0, [4-9]=doc1, [10-15]=doc2
    seq_lens = [
        [
            torch.tensor(4, device=device),
            torch.tensor(6, device=device),
            torch.tensor(6, device=device),
        ]
    ]

    mask_mod, document_ids = get_block_causal_mask_mod_by_seq_lens(seq_lens)

    print(f"[Rank {rank}] Full seq_len: {full_seq_len}")
    print(f"[Rank {rank}] Local seq_len (with CP={world_size}): {local_seq_len}")
    print(f"[Rank {rank}] Document IDs shape: {document_ids.shape}")
    print(f"[Rank {rank}] Document IDs: {document_ids[0].tolist()}")
    print(f"[Rank {rank}]              positions: {list(range(full_seq_len))}")

    # What global positions does this rank handle?
    local_start = rank * local_seq_len
    local_end = local_start + local_seq_len
    print(f"\n[Rank {rank}] Handles GLOBAL positions: {local_start} to {local_end-1}")

    # But create_cp_block_mask passes LOCAL indices (0 to local_seq_len-1)
    print(f"[Rank {rank}] But mask_mod receives LOCAL q_idx: 0 to {local_seq_len-1}")

    print(f"\n[Rank {rank}] === Testing mask_mod with sample indices ===")

    # Test some positions
    b, h = torch.tensor(0), torch.tensor(0)

    for local_q in [0, 4, 7]:
        global_q = local_q + local_start

        # What the mask_mod ACTUALLY computes (BUGGY - uses local index)
        actual_doc_id = document_ids[0, local_q].item()

        # What it SHOULD compute (correct - uses global index)
        if global_q < full_seq_len:
            expected_doc_id = document_ids[0, global_q].item()
        else:
            expected_doc_id = "OUT_OF_BOUNDS"

        status = "✓" if actual_doc_id == expected_doc_id else "✗ BUG!"
        print(
            f"[Rank {rank}] local_q={local_q}, global_q={global_q}: "
            f"actual_doc_id={actual_doc_id}, expected_doc_id={expected_doc_id} {status}"
        )

    # Demonstrate concrete masking errors
    print(f"\n[Rank {rank}] === Concrete Masking Errors ===")

    if rank == 1:
        # Rank 1 handles global positions 8-15 (doc1[4:6] and doc2[0:6])
        # But mask_mod sees local indices 0-7

        print(f"[Rank {rank}] Rank 1 handles global positions 8-15")
        print(f"[Rank {rank}] Global pos 8-9 are doc1, global pos 10-15 are doc2")
        print()

        # Local position 0 on rank 1 = global position 8 = doc1
        # Local position 2 on rank 1 = global position 10 = doc2
        local_q, local_kv = torch.tensor(2), torch.tensor(0)
        global_q, global_kv = 10, 8  # doc2 querying doc1

        # What mask_mod computes (buggy)
        buggy_q_doc = document_ids[0, local_q].item()  # Uses local=2 -> doc0!
        buggy_kv_doc = document_ids[0, local_kv].item()  # Uses local=0 -> doc0!
        buggy_allows = buggy_q_doc == buggy_kv_doc

        # What it should compute
        correct_q_doc = document_ids[0, global_q].item()  # global=10 -> doc2
        correct_kv_doc = document_ids[0, global_kv].item()  # global=8 -> doc1
        correct_allows = correct_q_doc == correct_kv_doc

        print(f"[Rank {rank}] Query: local={local_q.item()}, global={global_q}")
        print(f"[Rank {rank}] Key:   local={local_kv.item()}, global={global_kv}")
        print(
            f"[Rank {rank}] BUGGY:   q_doc={buggy_q_doc}, kv_doc={buggy_kv_doc}, allows={buggy_allows}"
        )
        print(
            f"[Rank {rank}] CORRECT: q_doc={correct_q_doc}, kv_doc={correct_kv_doc}, allows={correct_allows}"
        )
        print(f"[Rank {rank}] --> BUG: doc2 token incorrectly attends to doc1 token!")

    if rank == 0:
        print(f"[Rank {rank}] Rank 0 handles global positions 0-7")
        print(f"[Rank {rank}] Global pos 0-3 are doc0, global pos 4-7 are doc1")
        print(f"[Rank {rank}] For rank 0, local==global, so no bug visible here")

    # Show full mask comparison
    print(f"\n[Rank {rank}] === Full Mask Visualization (local 8x8) ===")

    buggy_mask = torch.zeros(local_seq_len, full_seq_len, dtype=torch.bool)
    correct_mask = torch.zeros(local_seq_len, full_seq_len, dtype=torch.bool)

    for lq in range(local_seq_len):
        gq = lq + local_start
        for kv in range(full_seq_len):
            # Buggy: uses local q index
            buggy_causal = lq >= kv  # Also wrong for causal!
            buggy_doc = document_ids[0, lq] == document_ids[0, kv]
            buggy_mask[lq, kv] = buggy_causal and buggy_doc

            # Correct: uses global q index
            correct_causal = gq >= kv
            correct_doc = document_ids[0, gq] == document_ids[0, kv]
            correct_mask[lq, kv] = correct_causal and correct_doc

    def mask_to_str(mask):
        return "\n".join(
            ["".join(["1" if x else "." for x in row]) for row in mask.int().tolist()]
        )

    print(
        f"[Rank {rank}] BUGGY mask (rows=local_q 0-{local_seq_len-1}, cols=global_kv 0-{full_seq_len-1}):"
    )
    print(mask_to_str(buggy_mask))
    print()
    print(f"[Rank {rank}] CORRECT mask:")
    print(mask_to_str(correct_mask))
    print()

    diff = (buggy_mask != correct_mask).sum().item()
    print(
        f"[Rank {rank}] Number of incorrect mask entries: {diff} / {local_seq_len * full_seq_len}"
    )

    dist.barrier()

    if rank == 0:
        print(f"\n{'='*70}")
        print("CONCLUSION: The document mask uses local indices but document_ids")
        print("expects global indices. On rank > 0, this causes wrong document")
        print("boundaries, allowing cross-document attention.")
        print(f"{'='*70}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
