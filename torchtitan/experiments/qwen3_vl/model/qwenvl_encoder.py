# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import einops as E
import torch
import torch.nn.functional as F
from torch import nn

from typing import Optional, Tuple, Union, List, Dict, Any

from torchtitan.models.attention import build_attention, init_attention_mask

from .args import Qwen3VLEncoderArgs

VISION_ACT_FN = nn.SiLU 

# helpers from huggingface

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MLP(nn.Module):
    def __init__(self, args: Qwen3VLEncoderArgs, bias: bool = False):
        super().__init__()
        self.hidden_size = args.dim
        self.intermediate_size = args.ffn_dim
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = VISION_ACT_FN()

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
        #return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):
    def __init__(self, args: Qwen3VLEncoderArgs, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = args.dim * (args.spatial_merge_size**2)
        self.norm_dim = args.dim
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else args.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, args.out_dim)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class VisionPatchEmbed(nn.Module):
    """
    Qwen2.5-VL uses Conv3D(T, P, P) with stride (T, P, P). Here we implement an equivalent Linear on flattened 3D patches.
    """
    def __init__(self, args: Qwen3VLEncoderArgs):
        super().__init__()
        self.in_channels = args.n_channels
        self.temporal_patch_size = args.temporal_patch_size
        self.patch_size = args.patch_size
        self.embed_dim = args.dim
        flat = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        self.proj = nn.Linear(flat, self.embed_dim, bias=False)

    def forward(self, x_NLD: torch.Tensor) -> torch.Tensor:
        # x_NLD: [N, L, flat3D]
        return self.proj(x_NLD)  # [N, L, E]

class VisionAttention(nn.Module):
    """
    MHA over variable-length segments (windows or full) defined by `cu_seqlens`.
    Shapes follow HF reference:
      - We flatten batch and run attention over each contiguous segment.
    """
    def __init__(self, args: Qwen3VLEncoderArgs, attn_impl: str = "eager"):
        super().__init__()
        self.dim = args.dim
        self.num_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        assert self.head_dim * args.n_heads == args.dim, "dim must be divisible by n_heads"
        self.qkv = nn.Linear(args.dim, args.dim * 3, bias=True)
        self.proj = nn.Linear(args.dim, args.dim, bias=True)
        self.scaling = self.head_dim ** -0.5
        self.attn_impl = attn_impl
        self.attn_drop = 0.0

        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def _attn_segment(
        self,
        q_seg: torch.Tensor,  # [L, H, Dh]
        k_seg: torch.Tensor,  # [L, H, Dh]
        v_seg: torch.Tensor,  # [L, H, Dh]
    ) -> torch.Tensor:
        # compute attn per segment
        # -> [H, L, Dh] for batched matmul convenience
        q = q_seg.transpose(0, 1)  # [H, L, Dh]
        k = k_seg.transpose(0, 1)  # [H, L, Dh]
        v = v_seg.transpose(0, 1)  # [H, L, Dh]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [H, L, L]
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = F.dropout(attn, p=self.attn_drop, training=self.training)
        out = torch.matmul(attn, v)  # [H, L, Dh]
        return out.transpose(0, 1)   # [L, H, Dh]

    def forward(
        self,
        hidden_states: torch.Tensor,           # [S, D]
        cu_seqlens: torch.Tensor,              # [num_segments+1], int32 prefix sums
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) each [S, Dh]
    ) -> torch.Tensor:
        S, D = hidden_states.shape
        qkv = self.qkv(hidden_states)          # [S, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to [S, H, Dh]
        q = q.view(S, self.num_heads, self.head_dim)
        k = k.view(S, self.num_heads, self.head_dim)
        v = v.view(S, self.num_heads, self.head_dim)

        cos, sin = position_embeddings
        # apply rotary on q,k
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # segment-wise attention
        outs = []
        starts = cu_seqlens[:-1].tolist()
        ends = cu_seqlens[1:].tolist()
        for s, e in zip(starts, ends):
            if e <= s:
                continue
            q_seg = q[s:e]  # [L, H, Dh]
            k_seg = k[s:e]
            v_seg = v[s:e]
            outs.append(self._attn_segment(q_seg, k_seg, v_seg))
        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]  # [S, H, Dh]
        out = out.reshape(S, D)  # [S, D]
        return self.proj(out)    # [S, D]

class VisionBlock(nn.Module):
    def __init__(self, args: Qwen3VLEncoderArgs, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(args.dim, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(args.dim, eps=1e-6)
        self.attn = VisionAttention(args=args)
        self.mlp = MLP(args=args, bias=True)
    
    def init_weights(self):
        pass
        #self.norm1.reset_parameters()
        #self.norm2.reset_parameters()
        #self.attn.init_weights()
        #self.mlp.init_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class QwenVisionTransformer(nn.Module):
    def __init__(self, args: Qwen3VLEncoderArgs) -> None:
        super().__init__()

        self.patch_embed = VisionPatchEmbed(args)

        self.spatial_merge_size = args.spatial_merge_size
        self.patch_size = args.patch_size
        self.fullatt_block_indexes = args.fullatt_block_indexes
        self.window_size = args.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        head_dim = args.dim // args.n_heads
        self.rotary_pos_emb = RotaryEmbedding(head_dim // 2)

        #self.layers = nn.ModuleList([VisionBlock(args) for _ in range(args.n_layers)])
        self.layers = nn.ModuleDict(
            {str(idx): VisionBlock(args) for idx in range(args.n_layers)}
        )
        self.merger = PatchMerger(args)
        self.gradient_checkpointing = False

        self.deepstack_visual_indexes = args.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                PatchMerger(
                    args=args,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

    

    def init_weights(self):
        for layer in self.layers.values():
            layer.init_weights()

    def rot_pos_emb(self, grid_thw):

        pos_ids = []
        for t, h, w in grid_thw:

            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, pixel_masks_NL: torch.BoolTensor,  grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        # qwen's grid thw format 
        mod_grid_thw = torch.tensor([[1, grid_thw[:, :, 1].max() + 1, grid_thw[:, :, 2].max() + 1]])
        grid_thw = mod_grid_thw
        rotary_pos_emb = self.rot_pos_emb(mod_grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        hidden_states = hidden_states.squeeze(0)
        old_hidden_states = hidden_states.clone()

        seq_len, _ = hidden_states.size()
        unpadded_seq_len = grid_thw[0][1] * grid_thw[0][2]
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(unpadded_seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(unpadded_seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(unpadded_seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []


        for layer_num, blk in enumerate(self.layers.values()):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        hidden_states = hidden_states.unsqueeze(0)

        # pad back to  (1, 12000, dim)
        hidden_states = F.pad(hidden_states, (0, 0, 0, 12000 - hidden_states.shape[1]), value=0)

        return hidden_states, deepstack_feature_lists