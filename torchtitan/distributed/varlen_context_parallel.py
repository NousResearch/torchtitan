# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Varlen Context Parallel implementation for document masking with ring attention.

This module provides memory-efficient context parallelism that supports document
masking via cu_seqlens, avoiding the memory overhead of FlexAttention's BlockMask.

Based on nanotron's llama3_ring_attention implementation.
"""

import os
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

# Enable timing instrumentation via environment variable
_VARLEN_CP_TIMING = os.environ.get("VARLEN_CP_TIMING", "0") == "1"
_VARLEN_CP_TIMING_VERBOSE = os.environ.get("VARLEN_CP_TIMING_VERBOSE", "0") == "1"
_timing_stats = {
    "v_padding": [],
    "buffer_alloc": [],
    "k_contiguous": [],
    "v_contiguous": [],
    "k_allgather": [],
    "v_allgather": [],
    "k_slice_contiguous": [],
    "v_slice_contiguous": [],
    "attention": [],
    "output_unpad": [],
    "total": [],
}
_call_count = 0
_step_call_count = 0  # Calls within current step
_tensor_info = {}  # Store tensor shape/size info


def _sync_and_time():
    """Synchronize CUDA and return current time."""
    torch.cuda.synchronize()
    return time.perf_counter()


def reset_varlen_cp_timing_stats():
    """Reset timing stats for a new step."""
    global _step_call_count
    for key in _timing_stats:
        _timing_stats[key].clear()
    _step_call_count = 0
    _tensor_info.clear()


def get_varlen_cp_timing_summary():
    """Get timing summary as a dict for the current step."""
    summary = {}
    for key, values in _timing_stats.items():
        if values:
            summary[key] = {
                "avg_ms": sum(values) / len(values) * 1000,
                "total_ms": sum(values) * 1000,
                "count": len(values),
            }
    summary["tensor_info"] = _tensor_info.copy()
    return summary


def print_varlen_cp_timing_stats():
    """Print timing statistics for varlen CP attention."""
    if not _timing_stats["total"]:
        print("No timing stats collected. Set VARLEN_CP_TIMING=1 to enable.")
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    n_calls = len(_timing_stats["total"])
    total_time_ms = sum(_timing_stats["total"]) * 1000

    print("\n" + "=" * 80)
    print(f"VARLEN CP TIMING STATS ({n_calls} calls, total={total_time_ms:.1f}ms)")
    print("=" * 80)

    # Calculate percentages
    total_sum = sum(_timing_stats["total"]) if _timing_stats["total"] else 1

    for key, values in _timing_stats.items():
        if values and key != "total":
            avg_ms = sum(values) / len(values) * 1000
            total_ms = sum(values) * 1000
            pct = (sum(values) / total_sum) * 100
            print(f"{key:25s}: avg={avg_ms:8.3f}ms  total={total_ms:8.1f}ms  ({pct:5.1f}%)")

    print("-" * 80)
    print(f"{'TOTAL':25s}: avg={total_time_ms/n_calls:8.3f}ms  total={total_time_ms:8.1f}ms  (100.0%)")

    # Print tensor info
    if _tensor_info:
        print("-" * 80)
        print("Tensor shapes/sizes:")
        for k, v in _tensor_info.items():
            print(f"  {k}: {v}")

    print("=" * 80 + "\n")


def prepare_cu_seqlens_for_cp(
    cu_seqlens: torch.Tensor,
    cp_rank: int,
    cp_world_size: int,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, slice]:
    """
    Prepare cu_seqlens for context parallel varlen attention.

    This function splits the global cu_seqlens across CP ranks, computing
    the local cu_seqlens for Q and the cu_seqlens for K that each rank needs.

    Args:
        cu_seqlens: Global cumulative sequence lengths tensor.
            Shape: [num_sequences + 1], e.g., [0, 512, 1024, 2048] for 3 docs.
        cp_rank: Current CP rank (0 to cp_world_size - 1).
        cp_world_size: Total number of CP ranks.
        causal: Whether using causal attention. When True, each position only
            attends to previous positions within the same document.

    Returns:
        cu_seqlens_q: Local cu_seqlens for Q on this rank.
        cu_seqlens_k: cu_seqlens for K that this rank needs to attend to.
        max_seqlen_q: Maximum sequence length for Q on this rank.
        max_seqlen_k: Maximum sequence length for K this rank attends to.
        local_k_slice: Slice of global K/V tensor this rank needs.

    Example:
        >>> cu_seqlens = torch.tensor([0, 512, 1024, 2048])  # 3 docs
        >>> # With cp_world_size=2, rank 0 gets positions 0-1023, rank 1 gets 1024-2047
        >>> cu_seqlens_q, cu_seqlens_k, max_q, max_k, k_slice = prepare_cu_seqlens_for_cp(
        ...     cu_seqlens, cp_rank=0, cp_world_size=2, causal=True
        ... )
    """
    total_length = cu_seqlens[-1].item()
    assert total_length % cp_world_size == 0, (
        f"Total sequence length {total_length} must be divisible by "
        f"cp_world_size {cp_world_size}"
    )
    length_per_rank = total_length // cp_world_size

    # Find document boundaries within this rank's slice
    # left: first document that starts at or before this rank's start
    # right: first document that starts at or after this rank's end
    rank_start = cp_rank * length_per_rank
    rank_end = (cp_rank + 1) * length_per_rank

    left = torch.searchsorted(cu_seqlens, rank_start, right=False)
    right = torch.searchsorted(cu_seqlens, rank_end, right=False)

    # Adjust left if the boundary doesn't align exactly
    if left > 0 and cu_seqlens[left].item() > rank_start:
        left -= 1

    left = left.item() if isinstance(left, torch.Tensor) else left
    right = right.item() if isinstance(right, torch.Tensor) else right

    # Compute local cu_seqlens_q (for Q on this rank)
    cu_seqlens_q = cu_seqlens[left:right + 1].clone()
    cu_seqlens_q = cu_seqlens_q - rank_start
    cu_seqlens_q[0] = 0
    cu_seqlens_q[-1] = length_per_rank

    # Compute cu_seqlens_k (for K that this rank needs)
    cu_seqlens_k = cu_seqlens[left:right + 1].clone()

    if causal:
        # For causal attention, we only need K up to the current position
        # The last document's K ends at the rank's end position
        slice_right = rank_end
        cu_seqlens_k[-1] = slice_right
    else:
        # For non-causal, we need the full last document
        slice_right = cu_seqlens[right].item()

    slice_left = cu_seqlens[left].item()
    cu_seqlens_k = cu_seqlens_k - slice_left

    # Compute max sequence lengths
    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

    max_seqlen_q = q_lengths.max().item() if len(q_lengths) > 0 else 0
    max_seqlen_k = k_lengths.max().item() if len(k_lengths) > 0 else 0

    local_k_slice = slice(slice_left, slice_right)

    return cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice


def _get_varlen_attn_func():
    """Get the best available varlen attention function."""
    # Try PyTorch native varlen first
    try:
        from torch.nn.attention.varlen import varlen_attn
        return varlen_attn, "torch"
    except ImportError:
        pass

    # Fall back to flash_attn
    try:
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func, "flash_attn"
    except ImportError:
        pass

    raise ImportError(
        "No varlen attention implementation found. "
        "Requires either PyTorch >= 2.5 (torch.nn.attention.varlen) "
        "or flash_attn (pip install flash-attn)"
    )


def varlen_attention_with_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_k_slice: slice,
    cp_group: dist.ProcessGroup,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """
    Varlen attention with context parallelism support.

    This function performs varlen Flash Attention with K/V gathered across
    CP ranks. It uses gather_dim=0 to avoid memory copies.

    Args:
        q: Query tensor, shape [total_q, num_heads, head_dim]
        k: Key tensor (local), shape [total_k_local, num_heads_k, head_dim]
        v: Value tensor (local), shape [total_k_local, num_heads_k, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for Q (prepared for this rank)
        cu_seqlens_k: Cumulative sequence lengths for K (prepared for this rank)
        max_seqlen_q: Maximum Q sequence length
        max_seqlen_k: Maximum K sequence length
        local_k_slice: Slice of gathered K/V this rank needs
        cp_group: Process group for context parallelism
        softmax_scale: Scale for softmax (default: 1/sqrt(head_dim))
        causal: Whether to use causal attention
        window_size: Sliding window size (left, right), -1 means infinite

    Returns:
        Attention output tensor, shape [total_q, num_heads, head_dim]
    """
    global _call_count, _step_call_count

    varlen_attn_func, backend = _get_varlen_attn_func()

    world_size = dist.get_world_size(cp_group)
    total_k_local = k.shape[0]
    num_heads_k = k.shape[1]
    k_head_dim = k.shape[2]
    v_head_dim_orig = v.shape[2]

    if softmax_scale is None:
        softmax_scale = k_head_dim ** -0.5

    # Start total timing
    if _VARLEN_CP_TIMING:
        t_total_start = _sync_and_time()
        _step_call_count += 1

        # Capture tensor info on first call
        if _step_call_count == 1:
            _tensor_info["q_shape"] = str(tuple(q.shape))
            _tensor_info["k_shape"] = str(tuple(k.shape))
            _tensor_info["v_shape"] = str(tuple(v.shape))
            _tensor_info["k_head_dim"] = k_head_dim
            _tensor_info["v_head_dim"] = v_head_dim_orig
            _tensor_info["cp_world_size"] = world_size
            _tensor_info["need_v_padding"] = v_head_dim_orig != k_head_dim
            # Memory sizes
            k_size_mb = k.numel() * k.element_size() / 1024 / 1024
            v_size_mb = v.numel() * v.element_size() / 1024 / 1024
            k_buffer_size_mb = k_size_mb * world_size
            v_buffer_size_mb = v_size_mb * world_size * (k_head_dim / v_head_dim_orig if v_head_dim_orig != k_head_dim else 1)
            _tensor_info["k_local_size_mb"] = f"{k_size_mb:.1f}"
            _tensor_info["v_local_size_mb"] = f"{v_size_mb:.1f}"
            _tensor_info["k_buffer_size_mb"] = f"{k_buffer_size_mb:.1f}"
            _tensor_info["v_buffer_size_mb"] = f"{v_buffer_size_mb:.1f}"

    # Flash Attention requires K and V to have the same head dimension.
    # If they differ (e.g., DeepSeek MLA), we pad V to match K's dimension.
    need_v_padding = v_head_dim_orig != k_head_dim
    if need_v_padding:
        if _VARLEN_CP_TIMING:
            t0 = _sync_and_time()
        # Pad V to match K's head dimension
        v_padded = torch.nn.functional.pad(
            v, (0, k_head_dim - v_head_dim_orig), mode='constant', value=0
        )
        if _VARLEN_CP_TIMING:
            _timing_stats["v_padding"].append(_sync_and_time() - t0)
    else:
        v_padded = v

    # Pre-allocate buffers for gathered K and V (now same head_dim)
    if _VARLEN_CP_TIMING:
        t0 = _sync_and_time()
    k_buffer = torch.empty(
        (total_k_local * world_size, num_heads_k, k_head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    v_buffer = torch.empty(
        (total_k_local * world_size, num_heads_k, k_head_dim),  # Use k_head_dim for padded V
        dtype=v_padded.dtype,
        device=v_padded.device,
    )
    if _VARLEN_CP_TIMING:
        _timing_stats["buffer_alloc"].append(_sync_and_time() - t0)

    # Make K contiguous
    if _VARLEN_CP_TIMING:
        t0 = _sync_and_time()
    k_contiguous = k.contiguous()
    if _VARLEN_CP_TIMING:
        _timing_stats["k_contiguous"].append(_sync_and_time() - t0)

    # Make V contiguous
    if _VARLEN_CP_TIMING:
        t0 = _sync_and_time()
    v_contiguous = v_padded.contiguous()
    if _VARLEN_CP_TIMING:
        _timing_stats["v_contiguous"].append(_sync_and_time() - t0)

    # All-gather K
    if _VARLEN_CP_TIMING:
        t0 = _sync_and_time()
    dist.all_gather_into_tensor(k_buffer, k_contiguous, group=cp_group)
    if _VARLEN_CP_TIMING:
        _timing_stats["k_allgather"].append(_sync_and_time() - t0)

    # All-gather V
    if _VARLEN_CP_TIMING:
        t0 = _sync_and_time()
    dist.all_gather_into_tensor(v_buffer, v_contiguous, group=cp_group)
    if _VARLEN_CP_TIMING:
        _timing_stats["v_allgather"].append(_sync_and_time() - t0)

    # Slice K and make contiguous
    if _VARLEN_CP_TIMING:
        t0 = _sync_and_time()
    k_full = k_buffer[local_k_slice].contiguous()
    if _VARLEN_CP_TIMING:
        _timing_stats["k_slice_contiguous"].append(_sync_and_time() - t0)

    # Slice V and make contiguous
    if _VARLEN_CP_TIMING:
        t0 = _sync_and_time()
    v_full = v_buffer[local_k_slice].contiguous()
    if _VARLEN_CP_TIMING:
        _timing_stats["v_slice_contiguous"].append(_sync_and_time() - t0)

    # Ensure cu_seqlens are int32 on the correct device
    cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32, device=q.device)
    cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32, device=q.device)

    # Call varlen attention based on backend
    if _VARLEN_CP_TIMING:
        t0 = _sync_and_time()
    if backend == "torch":
        # PyTorch native varlen_attn uses window_size for causal:
        # (-1, -1) for full attention, (-1, 0) for causal attention
        causal_window = (-1, 0) if causal else (-1, -1)
        out = varlen_attn_func(
            q,
            k_full,
            v_full,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale=softmax_scale,
            window_size=causal_window,
        )
    else:
        # flash_attn backend
        out = varlen_attn_func(
            q,
            k_full,
            v_full,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    if _VARLEN_CP_TIMING:
        _timing_stats["attention"].append(_sync_and_time() - t0)

    # If V was padded, unpad the output to get original v_head_dim
    if need_v_padding:
        if _VARLEN_CP_TIMING:
            t0 = _sync_and_time()
        out = out[..., :v_head_dim_orig].contiguous()
        if _VARLEN_CP_TIMING:
            _timing_stats["output_unpad"].append(_sync_and_time() - t0)

    if _VARLEN_CP_TIMING:
        _timing_stats["total"].append(_sync_and_time() - t_total_start)
        _call_count += 1

        # Verbose: print every call
        if _VARLEN_CP_TIMING_VERBOSE:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                t = _timing_stats["total"][-1] * 1000
                print(f"[varlen_cp] call {_call_count}: total={t:.2f}ms")

    return out


class VarlenContextParallelAttention(torch.nn.Module):
    """
    Module wrapper for varlen attention with context parallelism.

    This module handles the preparation of cu_seqlens and the attention
    computation with CP support.
    """

    def __init__(
        self,
        cp_mesh: Optional[DeviceMesh] = None,
        causal: bool = True,
    ):
        super().__init__()
        self.cp_mesh = cp_mesh
        self.causal = causal
        self._cp_group = None
        self._cp_rank = None
        self._cp_world_size = None

    def _get_cp_info(self):
        """Lazily initialize CP info."""
        if self._cp_group is None and self.cp_mesh is not None:
            self._cp_group = self.cp_mesh.get_group()
            self._cp_rank = dist.get_rank(self._cp_group)
            self._cp_world_size = dist.get_world_size(self._cp_group)
        return self._cp_group, self._cp_rank, self._cp_world_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional context parallelism.

        Args:
            q: Query tensor [total_seq, num_heads, head_dim]
            k: Key tensor [total_seq, num_heads_k, head_dim]
            v: Value tensor [total_seq, num_heads_k, head_dim]
            cu_seqlens: Global cumulative sequence lengths
            max_seqlen: Maximum sequence length
            softmax_scale: Optional softmax scale

        Returns:
            Attention output [total_seq, num_heads, head_dim]
        """
        cp_group, cp_rank, cp_world_size = self._get_cp_info()

        if cp_group is None or cp_world_size == 1:
            # No CP, use regular varlen attention
            varlen_attn_func, backend = _get_varlen_attn_func()

            cu_seqlens = cu_seqlens.to(dtype=torch.int32, device=q.device)

            if backend == "torch":
                # PyTorch varlen_attn uses window_size for causal:
                # (-1, -1) for full attention, (-1, 0) for causal attention
                causal_window = (-1, 0) if self.causal else (-1, -1)
                return varlen_attn_func(
                    q, k, v,
                    cu_seqlens, cu_seqlens,
                    max_seqlen, max_seqlen,
                    scale=softmax_scale,
                    window_size=causal_window,
                )
            else:
                # flash_attn
                return varlen_attn_func(
                    q, k, v,
                    cu_seqlens, cu_seqlens,
                    max_seqlen, max_seqlen,
                    softmax_scale=softmax_scale,
                    causal=self.causal,
                )

        # With CP: prepare cu_seqlens and run distributed attention
        cu_seqlens_q, cu_seqlens_k, max_q, max_k, local_k_slice = prepare_cu_seqlens_for_cp(
            cu_seqlens, cp_rank, cp_world_size, self.causal
        )

        return varlen_attention_with_cp(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_q, max_k,
            local_k_slice,
            cp_group,
            softmax_scale=softmax_scale,
            causal=self.causal,
        )
