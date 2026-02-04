#!/usr/bin/env python3
"""
Simple single-GPU test to demonstrate the document masking bug with CP.
No distributed setup needed - just simulates what happens.

Run with: python debug_document_mask_simple.py
"""

import torch


def _get_document_ids_from_seq_lens(seq_lens):
    """Exact copy from torchtitan/models/attention.py"""
    batch_size = len(seq_lens)
    batch_document_ids = []
    for sample_idx in range(batch_size):
        document_ids = torch.cat(
            [
                torch.full((seq_len,), i, dtype=torch.long)
                for i, seq_len in enumerate(seq_lens[sample_idx])
            ]
        )
        batch_document_ids.append(document_ids)
    return torch.stack(batch_document_ids)


def main():
    print("=" * 70)
    print("Document Mask + CP Bug Demonstration (Single GPU)")
    print("=" * 70)

    # Simulate: seq_len=16, CP=2, 3 documents [4, 6, 6]
    full_seq_len = 16
    cp_degree = 2
    local_seq_len = full_seq_len // cp_degree

    seq_lens = [[torch.tensor(4), torch.tensor(6), torch.tensor(6)]]
    document_ids = _get_document_ids_from_seq_lens(seq_lens)

    print(f"\nSetup:")
    print(f"  Full sequence length: {full_seq_len}")
    print(f"  CP degree: {cp_degree}")
    print(f"  Local sequence length per rank: {local_seq_len}")
    print(f"\nDocument layout:")
    print(f"  Position:    {list(range(full_seq_len))}")
    print(f"  Document ID: {document_ids[0].tolist()}")
    print(f"  (doc0=pos 0-3, doc1=pos 4-9, doc2=pos 10-15)")

    for rank in range(cp_degree):
        local_start = rank * local_seq_len
        print(f"\n{'='*70}")
        print(
            f"RANK {rank}: handles global positions {local_start} to {local_start + local_seq_len - 1}"
        )
        print("=" * 70)

        # Build buggy mask (what the code does) and correct mask
        buggy_mask = torch.zeros(local_seq_len, full_seq_len, dtype=torch.int)
        correct_mask = torch.zeros(local_seq_len, full_seq_len, dtype=torch.int)

        for local_q in range(local_seq_len):
            global_q = local_q + local_start

            for kv in range(full_seq_len):
                # BUGGY: mask_mod receives local_q but indexes document_ids with it
                buggy_causal = local_q >= kv  # Wrong! Should be global_q >= kv
                buggy_doc = (
                    document_ids[0, local_q] == document_ids[0, kv]
                )  # Wrong index!
                buggy_mask[local_q, kv] = int(buggy_causal and buggy_doc)

                # CORRECT: should use global indices
                correct_causal = global_q >= kv
                correct_doc = document_ids[0, global_q] == document_ids[0, kv]
                correct_mask[local_q, kv] = int(correct_causal and correct_doc)

        # Visualize
        def mask_to_str(mask, label_start=0):
            header = "     " + "".join([f"{i%10}" for i in range(mask.shape[1])])
            rows = []
            for i, row in enumerate(mask.tolist()):
                local_idx = i
                global_idx = i + label_start
                rows.append(
                    f"L{local_idx}G{global_idx} "
                    + "".join(["█" if x else "·" for x in row])
                )
            return header + "\n" + "\n".join(rows)

        print(f"\nBUGGY mask (rows=local_q, cols=global_kv):")
        print(mask_to_str(buggy_mask, local_start))

        print(f"\nCORRECT mask:")
        print(mask_to_str(correct_mask, local_start))

        diff = (buggy_mask != correct_mask).sum().item()
        total = local_seq_len * full_seq_len
        print(f"\nDifferences: {diff} / {total} entries ({100*diff/total:.1f}%)")

        if diff > 0:
            print("\nSpecific errors:")
            for local_q in range(local_seq_len):
                global_q = local_q + local_start
                for kv in range(full_seq_len):
                    if buggy_mask[local_q, kv] != correct_mask[local_q, kv]:
                        buggy_val = "ALLOW" if buggy_mask[local_q, kv] else "BLOCK"
                        correct_val = "ALLOW" if correct_mask[local_q, kv] else "BLOCK"
                        q_doc_buggy = document_ids[0, local_q].item()
                        q_doc_correct = document_ids[0, global_q].item()
                        kv_doc = document_ids[0, kv].item()
                        print(
                            f"  local_q={local_q} (global={global_q}, doc={q_doc_correct}) -> kv={kv} (doc={kv_doc}): "
                            f"buggy={buggy_val} (used doc={q_doc_buggy}), correct={correct_val}"
                        )

    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print("  The mask_mod function receives LOCAL q_idx from create_cp_block_mask,")
    print("  but document_ids[b, q_idx] expects GLOBAL indices.")
    print("  On rank > 0, this causes:")
    print("    1. Wrong document lookups (local 0 maps to doc0, not the actual doc)")
    print("    2. Cross-document attention leakage")
    print("    3. Loss ~3 instead of ~0.8")
    print("=" * 70)


if __name__ == "__main__":
    main()
