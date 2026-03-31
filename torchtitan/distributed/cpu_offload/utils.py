# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Nous Research. All rights reserved.
# Modifications: Refactored to be framework-agnostic.

from typing import Dict, Optional

import torch

DEBUG = False
DEBUG_RANK = 0


def debug_rank(message: str) -> None:
    """Print debug message for a specific rank when DEBUG is enabled."""
    if not DEBUG:
        return
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == DEBUG_RANK:
            print(message)
    else:
        print(message)


def is_graph_capturing() -> bool:
    """Check if CUDA graph capture is in progress.

    Standalone replacement for megatron.core.transformer.cuda_graphs.is_graph_capturing.
    Uses torch.cuda.is_current_stream_capturing() which is available in PyTorch >= 2.0.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def print_offload_summary_table(
    total_offload_bytes: Dict[str, int],
    distributed: bool = True,
) -> None:
    """Print an ASCII table summarizing offload bytes.

    If distributed is True and torch.distributed is initialized, gathers data
    from all ranks and prints a formatted table on rank 0.
    Otherwise, prints a single-rank table.

    Args:
        total_offload_bytes: Dict mapping group names to offload bytes for this rank.
        distributed: Whether to gather data from all ranks.
    """
    if distributed and torch.distributed.is_initialized():
        _print_distributed_summary(total_offload_bytes)
    else:
        _print_local_summary(total_offload_bytes)


def _print_local_summary(total_offload_bytes: Dict[str, int]) -> None:
    """Print offload summary for a single process."""
    if not total_offload_bytes:
        return
    all_group_names = sorted(total_offload_bytes.keys())
    col_width = max(12, max(len(name) for name in all_group_names) + 2)

    header = "Group".ljust(col_width) + "Offloaded (MB)".rjust(col_width)
    separator = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("Activation Offload Summary (MB)".center(len(header)))
    print(f"{'=' * len(header)}")
    print(header)
    print(separator)

    total = 0
    for name in all_group_names:
        b = total_offload_bytes[name]
        total += b
        print(f"{name.ljust(col_width)}{(b / (1024 * 1024)):.2f}".rjust(col_width))

    print(separator)
    print(f"{'Total'.ljust(col_width)}{(total / (1024 * 1024)):.2f}".rjust(col_width))
    print(f"{'=' * len(header)}\n")


def _print_distributed_summary(total_offload_bytes: Dict[str, int]) -> None:
    """Print offload summary gathered from all ranks."""
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Gather all group names across ranks
    local_names = list(total_offload_bytes.keys())
    all_names_list = [None] * world_size
    torch.distributed.all_gather_object(all_names_list, local_names)
    all_group_names = sorted(set(name for names in all_names_list for name in names))

    # Gather offload bytes from all ranks
    local_bytes = [total_offload_bytes.get(name, 0) for name in all_group_names]
    all_bytes_list = [None] * world_size
    torch.distributed.all_gather_object(all_bytes_list, local_bytes)

    if rank == 0:
        col_width = max(12, max((len(name) for name in all_group_names), default=8) + 2)
        rank_col_width = max(6, len(f"Rank {world_size - 1}") + 2)

        header = "Rank".ljust(rank_col_width)
        header += "".join(name.rjust(col_width) for name in all_group_names)
        header += "Total".rjust(col_width)
        separator = "-" * len(header)

        print(f"\n{'=' * len(header)}")
        print("Activation Offload Summary (MB)".center(len(header)))
        print(f"{'=' * len(header)}")
        print(header)
        print(separator)

        grand_total = 0
        col_totals = [0] * len(all_group_names)
        for r in range(world_size):
            row_bytes = all_bytes_list[r]
            row_total = sum(row_bytes)
            grand_total += row_total
            for i, b in enumerate(row_bytes):
                col_totals[i] += b
            row_str = f"Rank {r}".ljust(rank_col_width)
            for b in row_bytes:
                row_str += f"{b / (1024 * 1024):.2f}".rjust(col_width)
            row_str += f"{row_total / (1024 * 1024):.2f}".rjust(col_width)
            print(row_str)

        print(separator)
        totals_row = "Total".ljust(rank_col_width)
        for ct in col_totals:
            totals_row += f"{ct / (1024 * 1024):.2f}".rjust(col_width)
        totals_row += f"{grand_total / (1024 * 1024):.2f}".rjust(col_width)
        print(totals_row)
        print(f"{'=' * len(header)}\n")

    torch.distributed.barrier()
