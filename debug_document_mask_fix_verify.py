#!/usr/bin/env python3
"""
Verify the document masking CP fix works correctly.

Run with: python debug_document_mask_fix_verify.py
"""

import sys

import torch

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

from torch.nn.attention.flex_attention import and_masks
from torchtitan.models.attention import (
    _get_document_ids_from_seq_lens,
    apply_cp_offset_to_mask_mod,
)


def get_causal_mask_mod():
    def _causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return _causal_mask


def get_block_causal_mask_mod_by_seq_lens(seq_lens):
    """Original buggy version for comparison."""
    document_ids = _get_document_ids_from_seq_lens(seq_lens)

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal_mask & document_mask

    return mask_mod


def main():
    print("=" * 70)
    print("Verifying Document Mask + CP Fix")
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

    # Create mask_mods
    mask_mods = [get_causal_mask_mod()]
    mask_mods.append(get_block_causal_mask_mod_by_seq_lens(seq_lens))
    combined_mask_mod = and_masks(*mask_mods)

    all_passed = True

    for rank in range(cp_degree):
        local_start = rank * local_seq_len
        q_offset = rank * local_seq_len

        print(f"\n{'='*70}")
        print(
            f"RANK {rank}: handles global positions {local_start} to {local_start + local_seq_len - 1}"
        )
        print(f"  q_offset = {q_offset}")
        print("=" * 70)

        # Apply the fix
        fixed_mask_mod = apply_cp_offset_to_mask_mod(combined_mask_mod, q_offset)

        # Build masks
        buggy_mask = torch.zeros(local_seq_len, full_seq_len, dtype=torch.int)
        fixed_mask = torch.zeros(local_seq_len, full_seq_len, dtype=torch.int)
        expected_mask = torch.zeros(local_seq_len, full_seq_len, dtype=torch.int)

        b, h = torch.tensor(0), torch.tensor(0)

        for local_q in range(local_seq_len):
            global_q = local_q + local_start

            for kv in range(full_seq_len):
                # Buggy: passes local q_idx directly
                buggy_result = combined_mask_mod(
                    b, h, torch.tensor(local_q), torch.tensor(kv)
                )
                buggy_mask[local_q, kv] = int(buggy_result)

                # Fixed: apply_cp_offset_to_mask_mod adds offset to q_idx
                fixed_result = fixed_mask_mod(
                    b, h, torch.tensor(local_q), torch.tensor(kv)
                )
                fixed_mask[local_q, kv] = int(fixed_result)

                # Expected: manually compute with global indices
                q_doc = document_ids[0, global_q]
                kv_doc = document_ids[0, kv]
                causal_ok = global_q >= kv
                doc_ok = q_doc == kv_doc
                expected_mask[local_q, kv] = int(causal_ok and doc_ok)

        # Visualize
        def mask_to_str(mask, label_start=0):
            header = "     " + "".join([f"{i%10}" for i in range(mask.shape[1])])
            rows = []
            for i, row in enumerate(mask.tolist()):
                global_idx = i + label_start
                rows.append(
                    f"L{i}G{global_idx:02d} "
                    + "".join(["█" if x else "·" for x in row])
                )
            return header + "\n" + "\n".join(rows)

        print(f"\nBUGGY mask (before fix):")
        print(mask_to_str(buggy_mask, local_start))

        print(f"\nFIXED mask (after fix):")
        print(mask_to_str(fixed_mask, local_start))

        print(f"\nEXPECTED mask:")
        print(mask_to_str(expected_mask, local_start))

        buggy_diff = (buggy_mask != expected_mask).sum().item()
        fixed_diff = (fixed_mask != expected_mask).sum().item()
        total = local_seq_len * full_seq_len

        print(
            f"\nBuggy differences:  {buggy_diff} / {total} ({100*buggy_diff/total:.1f}%)"
        )
        print(
            f"Fixed differences:  {fixed_diff} / {total} ({100*fixed_diff/total:.1f}%)"
        )

        if fixed_diff == 0:
            print(f"✓ RANK {rank}: FIX VERIFIED - mask matches expected!")
        else:
            print(f"✗ RANK {rank}: FIX FAILED - {fixed_diff} differences remain!")
            all_passed = False

    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL RANKS PASSED - The fix is correct!")
    else:
        print("✗ SOME RANKS FAILED - The fix has issues!")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
