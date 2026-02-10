# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import functools
from collections.abc import Callable
from typing import ClassVar, NamedTuple

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None  # type: ignore[assignment]

from torch.types import Number


__all__ = [
    "FlexAttentionWrapper",
    "ScaledDotProductAttentionWrapper",
    "VarlenAttentionWrapper",
    "VarlenMetadata",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_sliding_window_mask_mod",
    "get_fixed_block_mask_mod",
    "get_block_causal_mask_mod_by_seq_lens",
    "create_attention_mask",
    "create_varlen_metadata_for_document",
    "create_varlen_metadata_from_sequence_lengths",
]


class VarlenMetadata(NamedTuple):
    """
    Cumulative sequence positions for queries and keys/values.
    Used for variable-length attention with document masking.
    """

    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: Number
    max_k: Number


class VarlenAttentionWrapper(torch.nn.Module):
    """Wrapper for varlen attention with optional Context Parallelism support.

    When cp_mesh is provided, this uses all_gather to collect K/V across CP ranks
    and computes attention with proper document masking via cu_seqlens.
    """

    _compiled_varlen_attn: ClassVar[Callable] = None

    @classmethod
    def _get_compiled_varlen_attn(cls):
        if cls._compiled_varlen_attn is None and varlen_attn is not None:
            cls._compiled_varlen_attn = torch.compile(
                varlen_attn, mode="max-autotune-no-cudagraphs"
            )
        return cls._compiled_varlen_attn

    def __init__(self, cp_mesh=None):
        super().__init__()
        self.cp_mesh = cp_mesh
        self._cp_group = None
        self._cp_rank = None
        self._cp_world_size = None

    def _get_cp_info(self):
        """Lazily initialize CP info from mesh."""
        if self._cp_group is None and self.cp_mesh is not None:
            import torch.distributed as dist
            self._cp_group = self.cp_mesh.get_group()
            self._cp_rank = dist.get_rank(self._cp_group)
            self._cp_world_size = dist.get_world_size(self._cp_group)
        return self._cp_group, self._cp_rank, self._cp_world_size

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        head_dim: int,  # This is v_head_dim for output reshaping
        attention_masks: VarlenMetadata,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for varlen attention.

        Note: Q, K may have different head_dim than V (e.g., DeepSeek MLA uses
        qk_head_dim=192 vs v_head_dim=128). We infer dimensions from tensor shapes.
        The `head_dim` parameter is used for output reshaping (should be v_head_dim).
        """
        cu_seq_q = attention_masks.cu_seq_q
        cu_seq_k = attention_masks.cu_seq_k
        max_q = attention_masks.max_q
        max_k = attention_masks.max_k

        # Get actual head dimensions from input tensors
        # xq: [B, H, S, qk_head_dim], xv: [B, H, S, v_head_dim]
        n_local_heads = xq.shape[1]
        qk_head_dim = xq.shape[-1]  # Q and K have same head_dim
        v_head_dim = xv.shape[-1]

        # Reshape to varlen format: [B*S, H, head_dim]
        xq_packed = xq.transpose(1, 2).reshape(-1, n_local_heads, qk_head_dim)
        xk_packed = xk.transpose(1, 2).reshape(-1, n_local_heads, qk_head_dim)
        xv_packed = xv.transpose(1, 2).reshape(-1, n_local_heads, v_head_dim)

        cp_group, cp_rank, cp_world_size = self._get_cp_info()

        if cp_group is not None and cp_world_size > 1:
            # Use CP-aware varlen attention with all_gather
            from torchtitan.distributed.varlen_context_parallel import (
                prepare_cu_seqlens_for_cp,
                varlen_attention_with_cp,
            )

            # Prepare cu_seqlens for this CP rank
            cu_seqlens_q_cp, cu_seqlens_k_cp, max_q_cp, max_k_cp, local_k_slice = (
                prepare_cu_seqlens_for_cp(cu_seq_q, cp_rank, cp_world_size, causal=True)
            )

            output_packed = varlen_attention_with_cp(
                xq_packed,
                xk_packed,
                xv_packed,
                cu_seqlens_q_cp,
                cu_seqlens_k_cp,
                max_q_cp,
                max_k_cp,
                local_k_slice,
                cp_group,
                softmax_scale=scale,
                causal=True,
            )
        else:
            # No CP, use regular varlen attention
            # PyTorch varlen_attn requires K and V to have the same head dimension.
            # If they differ (e.g., DeepSeek MLA), pad V to match K's dimension.
            need_v_padding = v_head_dim != qk_head_dim
            if need_v_padding:
                xv_packed = torch.nn.functional.pad(
                    xv_packed, (0, qk_head_dim - v_head_dim), mode='constant', value=0
                )

            compiled_attn = self._get_compiled_varlen_attn()
            if compiled_attn is None:
                raise ImportError(
                    "varlen_attn not available. Requires PyTorch >= 2.5"
                )

            # PyTorch varlen_attn uses window_size for causal attention:
            # (-1, -1) for full attention, (-1, 0) for causal attention
            output_packed = compiled_attn(
                xq_packed,
                xk_packed,
                xv_packed,
                cu_seq_q,
                cu_seq_k,
                max_q,
                max_k,
                scale=scale,
                window_size=(-1, 0),  # Causal attention
            )

            # Unpad output if V was padded
            if need_v_padding:
                output_packed = output_packed[..., :v_head_dim].contiguous()

        # Output has shape [B*S, H, v_head_dim]
        return output_packed


class FlexAttentionWrapper(torch.nn.Module):
    """Wrapper around `flex_attention` to make it torch.compile and CP compatible.

    This wrapper serves two purposes:
    1) Invoke `torch.compile` with a valid mode "max-autotune-no-cudagraphs" to
       achieve good performance.
    2) Being a wrapper allows us to apply _ContextParallel to it.

    Note:
        The forward function must have q, k, v as the first three arguments, and
        block_mask as a keyword argument to be compatible with _ContextParallel.
    """

    # Using dynamic=True to avoid CUDA crashes with CP, but need to debug NaN issue
    _compiled_flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention,
        dynamic=True,
    )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        block_mask: BlockMask,
        scale: float | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 1. _compiled_flex_attn has to be a class variable, otherwise there will
        #    be multiple compiled flex_attention instances, which can be slow.
        # 2. `self._compiled_flex_attn` is not correct, `self` will be passed in
        #    as the first argument, which will cause an error.
        #    `FlexAttentionWrapper._compiled_flex_attn` is correct.
        # 3. Used `return_lse` instead of `return_aux` because of easier TP module notation
        #    to convert `lse` to be DTensor.

        return FlexAttentionWrapper._compiled_flex_attn(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=scale,
            return_lse=return_lse,
        )


class ScaledDotProductAttentionWrapper(torch.nn.Module):
    """Wrapper around `F.scaled_dot_product_attention` to make it CP compatible.

    This wrapper is needed because `F.scaled_dot_product_attention` is not
    a torch.nn.Module, and thus cannot be applied with _ContextParallel.
    We need to wrap it into a torch.nn.Module.

    Note:
        The forward function must have q, k, v as the first three arguments to be
        compatible with _ContextParallel.
    """

    # TODO: remove sdpa_backends after PyTorch 2.9 is released.
    sdpa_backends: ClassVar[list[SDPBackend]] = []

    def __init__(self) -> None:
        super().__init__()
        if not self.sdpa_backends:
            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)


# We cannot do inner function/closure because we won't be able to cache it --
# if we an inner function, a new closure will be created every time
# `get_causal_mask_mod` is called.
def _causal_mask(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
) -> torch.Tensor:
    """Causal mask that prevents attention to future tokens."""
    return q_idx >= kv_idx


def get_causal_mask_mod() -> _mask_mod_signature:
    """Returns a causal mask modifier for flex attention.

    Returns:
        A mask modifier function that implements causal masking.
    """
    return _causal_mask


def get_document_mask_mod(batch: torch.Tensor, eos_id: int) -> _mask_mod_signature:
    """Creates a document mask that prevents attention across document boundaries.

    Args:
        batch: Input batch tensor with shape [b, s, h, d]
        eos_id: End-of-sequence token ID that marks document boundaries

    Returns:
        A mask modifier function that implements document-level masking.
    """
    # batch is [b, s, h, d] shape
    eos_mask = batch == eos_id
    eos_mask[:, -1] = True
    cumulative_mask = torch.cumsum(torch.where(eos_mask, 1, 0), dim=1)
    sequence_indices = torch.zeros_like(cumulative_mask, dtype=torch.int32)
    sequence_indices[:, 1:] = cumulative_mask[:, :-1]

    def document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]

    return document_mask


def get_fixed_block_mask_mod(fixed_block_size: int) -> _mask_mod_signature:
    """
    Divide the input sequence into blocks and only allow attention within the same block.

    Args:
        fixed_block_size: The number of tokens in each block.

    Returns:
        A mask modifier function that implements block-wise attention masking.
    """

    # Credit to @drisspg.
    def blocked_mask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # Get the block index of the query and key
        q_block = q_idx // fixed_block_size
        kv_block = kv_idx // fixed_block_size
        # Only allow attention within the same block
        return q_block == kv_block

    blocked_mask_mod.__name__ = f"blocked_mask_mod_fixed_block_size_{fixed_block_size}"

    return blocked_mask_mod


def _get_document_ids_from_seq_lens(
    seq_lens: list[torch.Tensor],
) -> torch.Tensor:
    """
    Convert a batch tensor of seq lens into integer IDs denoting sample ownership.
    For example, seq_lens = [2, 3, 1] would return [0, 0, 1, 1, 1, 2].
    Args:
        seq_lens (list[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.
    Returns:
        Tensor: Document IDs of shape (batch_size, max_seq_len).
    """
    batch_size = len(seq_lens)
    batch_document_ids = []
    for sample_idx in range(batch_size):
        # We assume seq lens sum to max seq lens, so document_ids should be of
        # shape (max_seq_len, )
        document_ids = torch.cat(
            [
                torch.full((seq_len,), i, dtype=torch.long, device=seq_len.device)
                for i, seq_len in enumerate(seq_lens[sample_idx])
            ]
        )
        batch_document_ids.append(document_ids)
    batch_document_ids = torch.stack(batch_document_ids)
    return batch_document_ids


def get_block_causal_mask_mod_by_seq_lens(
    seq_lens: list[torch.Tensor],
) -> _mask_mod_signature:
    document_ids = _get_document_ids_from_seq_lens(seq_lens)

    def mask_mod(b, h, q_idx, kv_idx):
        """
        Defines the logic of a block causal mask by combining both a standard causal mask
        and a block diagonal document mask.
        See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
        for an illustration.
        """
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal_mask & document_mask

    return mask_mod


def get_sliding_window_mask_mod(window_size: int) -> _mask_mod_signature:
    """Creates a sliding window mask that only attends to tokens within a fixed window size.

    This implements causal sliding window attention where each token can only attend to:
    - Itself (current token)
    - Up to `window_size - 1` previous tokens
    Args:
        window_size: The maximum number of tokens to attend to (including current token).
                    Must be >= 1. A window_size of 1 means attend only to self.

    Returns:
        A mask modifier function that implements causal sliding window masking.
    """

    if window_size < 1:
        raise ValueError(
            f"window_size must be >= 1 for sliding window attention mask, got {window_size}"
        )

    def sliding_window_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # Window mask: can only attend within the window
        # q_idx - kv_idx < window_size ensures we look at most window_size-1 tokens back
        return (kv_idx <= q_idx) & (q_idx - kv_idx < window_size)

    sliding_window_mod.__name__ = f"sliding_window_mod_window_size_{window_size}"

    return sliding_window_mod


_compiled_create_block_mask = torch.compile(create_block_mask)


@functools.lru_cache(4)
def create_attention_mask(*args, **kwargs):
    """Create an attention mask using compiled create_block_mask.

    This function is cached to avoid recreating BlockMasks for the same
    arguments.
    """
    return _compiled_create_block_mask(*args, **kwargs)


def create_varlen_metadata_for_document(
    input_batch: torch.Tensor, eos_id: int
) -> VarlenMetadata:
    """
    Creates cumulative sequence length indices needed for variable length attention.

    Args:
        input_batch: Input batch tensor with shape [batch_size, seq_len]
        eos_id: the EOS id marker

    Returns:
        VarlenMetadata containing cumulative sequence length indices for q, k, and max_seq_len
    """
    batch_size, seq_len = input_batch.shape
    device = input_batch.device
    cu_seqlens_list, all_seq_lengths = [], []
    offset = 0

    for b in range(batch_size):
        tokens = input_batch[b]
        eos_positions = (tokens == eos_id).nonzero(as_tuple=True)[0].to(torch.int32)
        sample_cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                eos_positions + 1,
                torch.tensor([seq_len], dtype=torch.int32, device=device),
            ]
        )
        sample_cu_seqlens = torch.unique_consecutive(sample_cu_seqlens)

        seq_lengths = torch.diff(sample_cu_seqlens)
        all_seq_lengths.append(seq_lengths)

        cu_seqlens_adjusted = sample_cu_seqlens[:-1] + offset
        cu_seqlens_list.append(cu_seqlens_adjusted)

        offset += seq_len

    packed_cu_seqlens = torch.cat(
        cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)]
    )

    max_seqlen = 0
    if len(all_seq_lengths) > 0:
        all_seq_lengths = torch.cat(all_seq_lengths)
        # device to host sync but only done once per model forward
        max_seqlen = all_seq_lengths.max().item()

    return VarlenMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )


def create_varlen_metadata_from_sequence_lengths(
    sequence_lengths: list[torch.Tensor],
    seq_len: int,
    device: torch.device,
) -> VarlenMetadata:
    """
    Creates cumulative sequence length indices needed for variable length attention
    from explicit sequence lengths provided by the data loader.

    This is an alternative to `create_varlen_metadata_for_document` that doesn't
    rely on EOS token detection, making it suitable for multi-turn chat data
    that has multiple EOS tokens per sample.

    Args:
        sequence_lengths: List of tensors, one per batch element, containing
            the lengths of each document/turn within that batch element.
        seq_len: The sequence length dimension of the batch.
        device: The device to place the output tensors on.

    Returns:
        VarlenMetadata containing cumulative sequence length indices for q, k, and max_seq_len
    """
    batch_size = len(sequence_lengths)
    cu_seqlens_list = []
    all_seq_lengths = []
    offset = 0

    for b in range(batch_size):
        sample_seq_lens = sequence_lengths[b]
        # Compute cumulative sequence lengths for this sample
        sample_cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.cumsum(sample_seq_lens.to(torch.int32), dim=0),
            ]
        )

        all_seq_lengths.append(sample_seq_lens)

        # Adjust for batch offset (excluding the final cumulative sum)
        cu_seqlens_adjusted = (sample_cu_seqlens[:-1] + offset).to(torch.int32)
        cu_seqlens_list.append(cu_seqlens_adjusted)

        offset += seq_len

    packed_cu_seqlens = torch.cat(
        cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)]
    ).to(torch.int32)

    max_seqlen = 0
    if len(all_seq_lengths) > 0:
        all_seq_lengths_cat = torch.cat(all_seq_lengths)
        # device to host sync but only done once per model forward
        max_seqlen = all_seq_lengths_cat.max().item()

    return VarlenMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )
