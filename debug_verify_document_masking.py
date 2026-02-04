#!/usr/bin/env python3
"""
Verify document masking is actually working.

This script checks that:
1. Tokens CAN attend within their document
2. Tokens CANNOT attend across document boundaries
"""

import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask


def _get_document_ids(seq_lens, device):
    batch_document_ids = []
    for sample_lens in seq_lens:
        doc_ids = torch.cat(
            [
                torch.full((l,), i, dtype=torch.long, device=device)
                for i, l in enumerate(sample_lens)
            ]
        )
        batch_document_ids.append(doc_ids)
    return torch.stack(batch_document_ids)


def get_document_causal_mask_mod(document_ids):
    def mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        doc_match = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal & doc_match

    return mask


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Verifying Document Masking")
    print("=" * 60)

    # Small example: 16 tokens, 4 documents of 4 tokens each
    seq_len = 16
    doc_sizes = [4, 4, 4, 4]  # 4 documents
    batch_size = 1
    n_heads = 1

    print(f"\nSequence length: {seq_len}")
    print(f"Documents: {doc_sizes}")
    print("Document layout:")
    print("  Positions 0-3:   Doc 0")
    print("  Positions 4-7:   Doc 1")
    print("  Positions 8-11:  Doc 2")
    print("  Positions 12-15: Doc 3")

    # Create document IDs
    seq_lens_batch = [[s for s in doc_sizes]]
    document_ids = _get_document_ids(seq_lens_batch, device)

    print(f"\nDocument IDs: {document_ids[0].tolist()}")

    # Create mask using create_mask (dense) instead of create_block_mask (sparse)
    mask_mod = get_document_causal_mask_mod(document_ids)
    dense = create_mask(
        mask_mod, B=batch_size, H=n_heads, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )
    dense = dense[0, 0]  # [seq_len, seq_len]

    print("\n" + "=" * 60)
    print("Attention Mask (1=can attend, 0=blocked):")
    print("=" * 60)
    print("\nKV positions ->")
    print("     ", end="")
    for i in range(seq_len):
        print(f"{i:2d}", end=" ")
    print("\n     " + "-" * (seq_len * 3))

    for q in range(seq_len):
        print(f"{q:2d} | ", end="")
        for kv in range(seq_len):
            val = int(dense[q, kv].item())
            # Highlight document boundaries
            if kv > 0 and kv % 4 == 0:
                print("|", end="")
            print(f"{val} ", end="")
        q_doc = q // 4
        print(f"  <- Q Doc {q_doc}")
        if (q + 1) % 4 == 0 and q < seq_len - 1:
            print("     " + "-" * (seq_len * 3))

    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)

    errors = []

    # Check specific cases
    test_cases = [
        # (q_pos, kv_pos, expected, description)
        (0, 0, 1, "Doc0[0] -> Doc0[0] (self-attention)"),
        (3, 0, 1, "Doc0[3] -> Doc0[0] (within doc, causal OK)"),
        (3, 3, 1, "Doc0[3] -> Doc0[3] (self-attention)"),
        (0, 3, 0, "Doc0[0] -> Doc0[3] (within doc, but FUTURE - blocked by causal)"),
        (4, 3, 0, "Doc1[0] -> Doc0[3] (CROSS-DOC - must be BLOCKED)"),
        (4, 4, 1, "Doc1[0] -> Doc1[0] (self-attention)"),
        (7, 4, 1, "Doc1[3] -> Doc1[0] (within doc, causal OK)"),
        (8, 7, 0, "Doc2[0] -> Doc1[3] (CROSS-DOC - must be BLOCKED)"),
        (8, 0, 0, "Doc2[0] -> Doc0[0] (CROSS-DOC - must be BLOCKED)"),
        (15, 12, 1, "Doc3[3] -> Doc3[0] (within doc, causal OK)"),
        (15, 11, 0, "Doc3[3] -> Doc2[3] (CROSS-DOC - must be BLOCKED)"),
        (15, 0, 0, "Doc3[3] -> Doc0[0] (CROSS-DOC - must be BLOCKED)"),
    ]

    for q, kv, expected, desc in test_cases:
        actual = int(dense[q, kv].item())
        status = "✓" if actual == expected else "✗ FAIL"
        print(f"  [{status}] {desc}")
        print(f"       mask[{q},{kv}] = {actual}, expected = {expected}")
        if actual != expected:
            errors.append((q, kv, expected, actual, desc))

    print("\n" + "=" * 60)
    if not errors:
        print("✓ ALL CHECKS PASSED - Document masking is working correctly!")
    else:
        print(f"✗ {len(errors)} CHECKS FAILED - Document masking is NOT working!")
        for q, kv, expected, actual, desc in errors:
            print(f"  - {desc}: got {actual}, expected {expected}")
    print("=" * 60)

    # Count cross-document attentions that are blocked
    cross_doc_blocked = 0
    cross_doc_total = 0
    within_doc_allowed = 0
    within_doc_total = 0

    for q in range(seq_len):
        q_doc = q // 4
        for kv in range(seq_len):
            kv_doc = kv // 4
            val = int(dense[q, kv].item())

            if q_doc != kv_doc:
                # Cross-document - should ALL be blocked
                cross_doc_total += 1
                if val == 0:
                    cross_doc_blocked += 1
            else:
                # Within document - causal should be allowed
                if q >= kv:  # Causal
                    within_doc_total += 1
                    if val == 1:
                        within_doc_allowed += 1

    print(f"\nStatistics:")
    print(
        f"  Cross-document pairs blocked: {cross_doc_blocked}/{cross_doc_total} ({100*cross_doc_blocked/cross_doc_total:.0f}%)"
    )
    print(
        f"  Within-document causal pairs allowed: {within_doc_allowed}/{within_doc_total} ({100*within_doc_allowed/within_doc_total:.0f}%)"
    )

    if cross_doc_blocked == cross_doc_total and within_doc_allowed == within_doc_total:
        print("\n✓ Document masking is 100% correct!")
    else:
        print("\n✗ Document masking has errors!")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
