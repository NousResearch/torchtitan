# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Nemotron Super (NemotronH) model.

Each layer = (Mamba2, [Attention], MoE).
Mamba2 via FLA, attention is vanilla GQA + RoPE, MoE from torchtitan.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.models.moe import MoE
from torchtitan.models.utils import trunc_normal_
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.tools.logging import logger

from .args import NemotronSuperModelArgs

from fla.models.mamba2.modeling_mamba2 import Mamba2


# -- RoPE --

def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 10000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seqlen = xq.shape[1]
    head_dim = xq.shape[-1]
    rope = rope_cache[:seqlen].view(1, seqlen, 1, head_dim * 2)
    cos = rope[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope[..., head_dim:].to(dtype=xq.dtype, device=xq.device)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# -- Attention (vanilla GQA + RoPE, no QK-norm) --

class Attention(nn.Module):
    def __init__(self, model_args: NemotronSuperModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads or model_args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim ** -0.5
        self.attn_type = model_args.attn_type
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.wq = nn.Linear(model_args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, model_args.dim, bias=False)

        self.inner_attention: nn.Module
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            trunc_normal_(linear.weight, mean=0.0, std=0.02)
        trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        bs, seqlen, _ = x.shape
        xq = self.wq(x).view(bs, seqlen, -1, self.head_dim)
        xk = self.wk(x).view(bs, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, rope_cache)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask)
                output = (
                    self.inner_attention(
                        xq, xk, xv,
                        block_mask=attention_masks,
                        scale=self.scaling,
                        enable_gqa=self.enable_gqa,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata)
                output = self.inner_attention(
                    xq, xk, xv,
                    attention_masks,
                    scale=self.scaling,
                )
            case "sdpa":
                assert attention_masks is None
                output = (
                    self.inner_attention(
                        xq, xk, xv,
                        scale=self.scaling,
                        enable_gqa=self.enable_gqa,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        return self.wo(output.view(bs, seqlen, -1))


# -- Layer: (Mamba2, [Attention], MoE) --

class NemotronSuperLayer(nn.Module):
    def __init__(self, layer_id: int, model_args: NemotronSuperModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = model_args.dim
        self.n_layers = model_args.n_layers
        self.has_attn = layer_id in set(model_args.attn_layer_idxs)

        # Mamba2 (always present)
        self.mamba_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.mamba = Mamba2(
            num_heads=model_args.mamba_num_heads,
            head_dim=model_args.mamba_head_dim,
            hidden_size=model_args.dim,
            state_size=model_args.ssm_state_size,
            expand=model_args.mamba_expand,
            n_groups=model_args.n_groups,
            conv_kernel=model_args.conv_kernel,
            use_conv_bias=model_args.use_conv_bias,
            hidden_act=model_args.mamba_hidden_act,
            chunk_size=model_args.chunk_size,
            time_step_min=model_args.time_step_min,
            time_step_max=model_args.time_step_max,
            use_bias=model_args.use_mamba_proj_bias,
            norm_eps=model_args.norm_eps,
            layer_idx=layer_id,
        )

        # Attention (only on selected layers)
        if self.has_attn:
            self.attn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
            self.attention = Attention(model_args)

        # MoE (always present)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        expert_intermediate = (
            model_args.moe_args.expert_intermediate_size
            if model_args.moe_args.expert_intermediate_size is not None
            else model_args.hidden_dim
        )
        self.moe = MoE(
            model_args.moe_args,
            dim=model_args.dim,
            hidden_dim=expert_intermediate,
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        # Mamba2
        mamba_out, _, _ = self.mamba(self.mamba_norm(x))
        x = x + mamba_out

        # Attention (optional)
        if self.has_attn:
            x = x + self.attention(self.attn_norm(x), rope_cache, attention_masks)

        # MoE
        x = x + self.moe(self.ffn_norm(x))

        return x

    def init_weights(self, buffer_device: torch.device):
        self.mamba_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        # FLA's Mamba2 initializes its own weights internally
        if self.has_attn:
            self.attn_norm.reset_parameters()
            self.attention.init_weights(self.weight_init_std)
        self.moe.init_weights(self.weight_init_std, buffer_device, self.n_layers)


# -- MTP (multi-token prediction) --

class MTPBlock(nn.Module):
    """
    Multi-token prediction block.

    Fuses token embeddings with previous hidden states, runs through
    attention + MoE, produces hidden states for the shared lm_head.

    HF flat layout for pattern "*E":
      mtp.layers.0: attention (*) + fusion (enorm, hnorm, eh_proj)
      mtp.layers.1: MoE (E) + final_layernorm
    """

    def __init__(self, model_args: NemotronSuperModelArgs):
        super().__init__()
        dim = model_args.dim

        # Fusion: concat(norm(embed), norm(hidden)) -> project to dim
        self.enorm = nn.RMSNorm(dim, eps=model_args.norm_eps)
        self.hnorm = nn.RMSNorm(dim, eps=model_args.norm_eps)
        self.eh_proj = nn.Linear(dim * 2, dim, bias=False)

        # Attention
        self.attn_norm = nn.RMSNorm(dim, eps=model_args.norm_eps)
        self.attention = Attention(model_args)

        # MoE
        self.ffn_norm = nn.RMSNorm(dim, eps=model_args.norm_eps)
        expert_intermediate = (
            model_args.moe_args.expert_intermediate_size
            if model_args.moe_args.expert_intermediate_size is not None
            else model_args.hidden_dim
        )
        self.moe = MoE(
            model_args.moe_args,
            dim=dim,
            hidden_dim=expert_intermediate,
        )

        # Final norm before lm_head
        self.final_layernorm = nn.RMSNorm(dim, eps=model_args.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        tok_embeddings: nn.Embedding,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ) -> torch.Tensor:
        # Fuse: previous hidden states + shifted token embeddings
        embeds = tok_embeddings(input_ids)
        fused = torch.cat(
            [self.enorm(embeds), self.hnorm(hidden_states)],
            dim=-1,
        )
        h = self.eh_proj(fused)

        # Attention + MoE
        h = h + self.attention(self.attn_norm(h), rope_cache, attention_masks)
        h = h + self.moe(self.ffn_norm(h))

        return self.final_layernorm(h)

    def init_weights(self, init_std: float, buffer_device: torch.device, n_layers: int):
        for norm in (self.enorm, self.hnorm, self.attn_norm, self.ffn_norm, self.final_layernorm):
            norm.reset_parameters()
        trunc_normal_(self.eh_proj.weight, mean=0.0, std=0.02)
        self.attention.init_weights(init_std)
        self.moe.init_weights(init_std, buffer_device, n_layers)


# -- Full model --

class NemotronSuperModel(ModelProtocol):
    def __init__(self, model_args: NemotronSuperModelArgs):
        super().__init__(model_args)

        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.head_dim = model_args.head_dim

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = NemotronSuperLayer(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # MTP head (shared lm_head, separate fusion + attn + moe)
        if model_args.num_nextn_predict_layers > 0:
            self.mtp = MTPBlock(model_args)
        else:
            self.mtp = None

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        trunc_normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        for layer in self.layers.values():
            layer.init_weights(buffer_device)
        self.norm.reset_parameters()
        final_out_std = self.model_args.dim ** -0.5
        trunc_normal_(self.output.weight, mean=0.0, std=final_out_std)
        if self.mtp is not None:
            init_std = 0.02 / (2 * self.model_args.n_layers) ** 0.5
            self.mtp.init_weights(init_std, buffer_device, self.model_args.n_layers)

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.model_args.attn_mask_type:
            case "causal":
                match self.model_args.attn_type:
                    case "sdpa":
                        return None
                    case "flex":
                        return create_attention_mask(
                            get_causal_mask_mod(),
                            1,
                            None,
                            input_batch.shape[1],
                            input_batch.shape[1],
                        )
                    case _:
                        raise ValueError(
                            f"Unsupported attn_type for causal mask: {self.model_args.attn_type}"
                        )
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h, self.rope_cache, attention_masks)
        h = self.norm(h)
        logits = self.output(h)

        if self.mtp is not None and self.training:
            # MTP: predict next+1 token using backbone hidden states + shifted embeddings
            # Shift input_ids by 1: MTP fuses embed(token[t+1]) with hidden[t]
            mtp_input_ids = tokens[:, 1:]
            mtp_hidden = h[:, :-1]
            mtp_h = self.mtp(
                mtp_hidden, mtp_input_ids, self.tok_embeddings,
                self.rope_cache, attention_masks,
            )
            mtp_logits = self.output(mtp_h)  # reuse lm_head
            return logits, mtp_logits

        return logits
