# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import einops as E
import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.attention import build_attention, init_attention_mask

from .args import Qwen2_5VLEncoderArgs


def resize_positional_embeddings(
    pos_embs_HWD: torch.Tensor,
    spatial_shapes_N2: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    Resize the learned 2D positional embeddings to image-specific size and pad to a fixed size.

    Args:
        pos_embs_HWD (`torch.Tensor`):
            Position embeddings of shape (height, width, embed_dim)
        spatial_shapes (`torch.LongTensor`):
            Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        max_length (`int`):
            Maximum length of the positional embeddings to pad resized positional embeddings to

    Returns:
        `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
    """
    _, _, D = pos_embs_HWD.shape
    B, _ = spatial_shapes_N2.shape

    resized_embs_BLD = torch.empty(
        (B, max_length, D),
        device=pos_embs_HWD.device,
        dtype=pos_embs_HWD.dtype,
    )

    # TODO: group images by size, and do interpolate,
    # or cache the interpolate output so we do this once per size
    for i in range(B):
        height, width = spatial_shapes_N2[i].tolist()
        if (height + width) == 0:  # Skip empty padding images
            continue

        resized_emb = F.interpolate(
            E.rearrange(pos_embs_HWD, "h w d -> 1 d h w"),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        resized_emb_LD = E.rearrange(resized_emb, "1 d h w -> (h w) d")
        resized_embs_BLD[i, : int(height * width)] = resized_emb_LD

    return resized_embs_BLD


class VisionEmbeddings(nn.Module):
    def __init__(self, args: Qwen2_5VLEncoderArgs):
        super().__init__()
        self.patch_embedding = nn.Linear(
            in_features=args.n_channels * args.patch_size * args.patch_size,
            out_features=args.dim,
        )
        #self.position_embedding = nn.Embedding(args.n_pos_embs**2, args.dim)
        #self.n_pos_embs = args.n_pos_embs

    def init_weights(self):
        nn.init.trunc_normal_(self.patch_embedding.weight, mean=0.0, std=0.02)
        #nn.init.normal_(self.position_embedding.weight)

    def forward(self, pixels_NLD: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        # Apply patch embeddings to already patchified pixel values
        patch_embeds_NLD = self.patch_embedding(pixels_NLD)

        # Get positional resized and padded positional embeddings
        #pos_emb_HWD = self.position_embedding.weight.reshape(
        #    self.n_pos_embs, self.n_pos_embs, -1
        #)
        #spatial_h = E.reduce(grid_hw[:, :, 0], "n l -> n", reduction="max") + 1
        #spatial_w = E.reduce(grid_hw[:, :, 1], "n l -> n", reduction="max") + 1
        #spatial_shapes = torch.stack([spatial_h, spatial_w], dim=-1).long()
        #resized_positional_embeddings = resize_positional_embeddings(
        #    pos_emb_HWD,
        #    spatial_shapes,
        #    max_length=pixels_NLD.shape[1],
        #)
        ## Add positional embeddings to patch embeddings
        #embeddings = patch_embeds_NLD + resized_positional_embeddings
        return embeddings

class VisionPatchEmbed(nn.Module):
    """
    Qwen2.5-VL uses Conv3D(T, P, P) with stride (T, P, P). Here we implement an equivalent Linear on flattened 3D patches.
    """
    def __init__(self, in_channels: int, temporal_patch_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        flat = in_channels * temporal_patch_size * patch_size * patch_size
        self.proj = nn.Linear(flat, embed_dim, bias=False)

    @torch.no_grad()
    def load_conv3d_weight(self, conv3d_weight: torch.Tensor):
        """
        Load HF-style Conv3D kernel into this Linear:
            conv3d.weight: [out_dim, in_channels, T, P, P]
        """
        out_dim, C, T, P1, P2 = conv3d_weight.shape
        assert C == self.in_channels and T == self.temporal_patch_size and P1 == self.patch_size and P2 == self.patch_size
        flat = conv3d_weight.reshape(out_dim, -1)  # [E, C*T*P*P]
        self.proj.weight.copy_(flat)

    def forward(self, x_NLD: torch.Tensor) -> torch.Tensor:
        # x_NLD: [N, L, flat3D]
        return self.proj(x_NLD)  # [N, L, E]
class VisionAttention(nn.Module):
    """
    MHA over variable-length segments (windows or full) defined by `cu_seqlens`.
    Shapes follow HF reference:
      - We flatten batch and run attention over each contiguous segment.
    """
    def __init__(self, dim: int, num_heads: int, attn_impl: str = "eager"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "hidden_size must be divisible by num_heads"
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
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

class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act = nn.GELU()

        for lin in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.trunc_normal_(lin.weight, mean=0.0, std=0.02)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

class VisionBlock(nn.Module):
    def __init__(self, args: Qwen25VLVisionArgs):
        super().__init__()
        self.norm1 = RMSNorm(args.hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(args.hidden_size, eps=1e-6)
        self.attn  = VisionAttention(args.hidden_size, args.num_heads, attn_impl=args.attn_implementation)
        self.mlp   = GatedMLP(args.hidden_size, intermediate_size=4 * args.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [S, D]
        cu_seqlens: torch.Tensor,             # [num_segments+1]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
    ) -> torch.Tensor:
        x = hidden_states
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerger(nn.Module):
    """
    Concatenate s^2 neighboring tokens (already laid out contiguously) and project to out_hidden_size.
    """
    def __init__(self, context_dim: int, out_dim: int, spatial_merge_size: int):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.unit = spatial_merge_size * spatial_merge_size
        self.norm = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim * self.unit, context_dim * self.unit, bias=True),
            nn.GELU(),
            nn.Linear(context_dim * self.unit, out_dim, bias=True),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, D] where S is multiple of (s^2), contiguous grouping by design
        D = x.shape[-1]
        s2 = self.unit
        assert x.shape[0] % s2 == 0, "Sequence length must be divisible by s^2 for merger."
        x = self.norm(x)
        x = x.view(-1, s2 * D)     # group s^2 tokens
        return self.mlp(x)         # [S / s^2, out_dim]



class Qwen25VLVisionTransformer(nn.Module):
    """
    Pure-PyTorch re-implementation of the Qwen-2.5-VL visual encoder:
      * Linear(Conv3D-equivalent) patch-embed on flattened 3D patches
      * Rotary 2D pos-emb (cos/sin) applied to q/k
      * Windowed attention vs full attention via cu_seqlens
      * s^2 spatial patch-merger → out_hidden_size tokens for LLM

    Forward I/O:
      - pixel_values_NLD: [N, L, P] flattened 3D patches per sample
      - grid_thw:         [N, 3]    (T, H, W) per sample in *patch* units (before spatial merge)
      - returns FloatTensor [sum_i T_i * (H_i/s)^2, out_hidden_size]
    """
    def __init__(self, args: Qwen25VLVisionArgs):
        super().__init__()
        self.args = args
        self.spatial_merge_unit = args.spatial_merge_size * args.spatial_merge_size

        # embed
        self.patch_embed = VisionPatchEmbed(
            in_channels=args.in_channels,
            temporal_patch_size=args.temporal_patch_size,
            patch_size=args.patch_size,
            embed_dim=args.hidden_size,
        )

        # RoPE over (H, W) positions (head_dim/2)
        head_dim = args.hidden_size // args.num_heads
        self.rotary = VisionRotaryEmbedding(dim=head_dim // 2, theta=args.rope_theta)

        # blocks
        self.blocks = nn.ModuleList([VisionBlock(args) for _ in range(args.depth)])

        # merger
        self.merger = PatchMerger(context_dim=args.hidden_size,
                                  out_dim=args.out_hidden_size,
                                  spatial_merge_size=args.spatial_merge_size)

    # ---- helpers ----

    @torch.no_grad()
    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Build per-token rotary phase (cos/sin later) for all tokens pre-merge, but
        constructed from (H//s, W//s) "LLM-grid" positions and then broadcast to s^2.
        Returns: rotary_phase [S, head_dim] where S = sum_i T_i * H_i * W_i (pre-merge length).
        """
        device = grid_thw.device
        s = self.args.spatial_merge_size
        # per-sample (T, H, W) -> per-sample llm grid (T, H//s, W//s)
        pos_ids_list = []
        for (t, h, w) in grid_thw.tolist():
            llm_h, llm_w = h // s, w // s
            # 2D indices (H, W) flattened in "window-friendly" order
            h_ids = torch.arange(llm_h, device=device).unsqueeze(1).expand(llm_h, llm_w)
            h_ids = h_ids.reshape(llm_h // 1, 1, llm_w // 1, 1)  # trivial blocks; kept for symmetry
            h_ids = h_ids.permute(0, 2, 1, 3).reshape(-1)

            w_ids = torch.arange(llm_w, device=device).unsqueeze(0).expand(llm_h, llm_w)
            w_ids = w_ids.reshape(llm_h // 1, 1, llm_w // 1, 1)
            w_ids = w_ids.permute(0, 2, 1, 3).reshape(-1)

            # repeat across temporal grid
            pos_ids_list.append(torch.stack([h_ids, w_ids], dim=-1).repeat(t, 1))  # [t*llm_h*llm_w, 2]

        pos_ids = torch.cat(pos_ids_list, dim=0)  # [N_llm, 2], indexing into max(H//s, W//s)
        max_grid = int(grid_thw[:, 1:].max().item()) // s
        rotary_full = self.rotary(max_grid)  # [max_llm, Dh/2]
        # gather per-axis, then concat (Dh)
        rotary_phase = rotary_full[pos_ids]  # [N_llm, 2, Dh/2]
        rotary_phase = rotary_phase.flatten(1)  # [N_llm, Dh]

        # broadcast to pre-merge tokens by repeating s^2 for each (H//s, W//s) position
        s2 = self.spatial_merge_unit
        rotary_phase = E.repeat(rotary_phase, "n d -> (n r) d", r=s2)  # [N_llm * s^2, Dh]
        return rotary_phase

    @torch.no_grad()
    def _window_index_and_cu_seqlens(self, grid_thw: torch.Tensor):
        """
        Windowize LLM-grid positions and build prefix sums (in *pre-merge token units*).
        Returns:
          window_index:      LongTensor [N_llm]  indices that permute groups of s^2 tokens
          cu_window_seqlens: Int32Tensor [num_windows+1]  prefix sums in pre-merge token units
          cu_full_seqlens:   Int32Tensor [num_samples*T + 1]  full attention segments (one per frame)
        """
        device = grid_thw.device
        s = self.args.spatial_merge_size
        P = self.args.patch_size
        vit_win = self.args.window_size // s // P  # window size on LLM grid

        window_index_parts = []
        cu_win_psums: List[int] = [0]
        running = 0
        # Build cu_seqlens for "full attention" (per frame) too
        # Each frame has H*W llm-grid positions; pre-merge token count per llm position = s^2
        full_psums: List[int] = [0]

        window_index_id = 0
        for (t, h, w) in grid_thw.tolist():
            llm_h, llm_w = h // s, w // s
            # frame-level for full attention (per-frame segments)
            frame_len_llm = llm_h * llm_w
            for _ in range(t):
                running += frame_len_llm * (s * s)
                full_psums.append(running)

            # tile LLM grid into (vit_win x vit_win) windows with padding sentinel
            index = torch.arange(t * llm_h * llm_w, device=device).reshape(t, llm_h, llm_w)
            pad_h = (vit_win - llm_h % vit_win) % vit_win
            pad_w = (vit_win - llm_w % vit_win) % vit_win
            num_h = (llm_h + pad_h) // vit_win
            num_w = (llm_w + pad_w) // vit_win

            index_padded = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
            index_padded = index_padded.reshape(t, num_h, vit_win, num_w, vit_win)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                t, num_h * num_w, vit_win, vit_win
            )  # [t, WN, vh, vw]

            # per-window counts (in llm positions)
            seqlens_llm = (index_padded != -100).sum(dim=[2, 3]).reshape(-1)  # [t * WN]
            flat = index_padded.reshape(-1)
            keep = flat[flat != -100]  # [t * llm_h * llm_w] ascending
            window_index_parts.append(keep + window_index_id)  # shift

            # convert llm counts -> pre-merge token counts (x s^2), then cumsum
            c = (seqlens_llm * (s * s)).cumsum(0)
            cu_win_psums.extend((c + cu_win_psums[-1]).tolist())

            window_index_id += (t * llm_h * llm_w)

        window_index = torch.cat(window_index_parts, dim=0)  # [N_llm]
        cu_window_seqlens = torch.tensor(cu_win_psums, device=device, dtype=torch.int32)
        cu_full_seqlens = torch.tensor(full_psums, device=device, dtype=torch.int32)
        return window_index, cu_window_seqlens, cu_full_seqlens

    # ---- forward ----

    def forward(
        self,
        pixel_values_NLD: torch.FloatTensor,   # [N, L, flat3D], pre-patchified tokens
        grid_thw: torch.LongTensor,            # [N, 3] T,H,W (patch units) per sample
    ) -> torch.FloatTensor:
        """
        Returns:
            hidden_states: FloatTensor [sum_i T_i * (H_i/s)^2, out_hidden_size]
        """
        assert pixel_values_NLD.dim() == 3, "pixel_values_NLD must be [N, L, P]"
        N, L, Pflat = pixel_values_NLD.shape
        x = self.patch_embed(pixel_values_NLD.to(self.patch_embed.proj.weight.dtype))  # [N, L, D]
        x = x.reshape(-1, x.shape[-1])  # [S, D], S = N*L (pre-merge token length)

        # rotary positions constructed from LLM grid then expanded to pre-merge tokens
        rotary_phase = self._rot_pos_emb(grid_thw)  # [S, Dh]
        # double to full Dh: (cos,sin) expects Dh
        emb = torch.cat([rotary_phase, rotary_phase], dim=-1)  # [S, Dh*2 == head_dim]
        cos, sin = emb.cos(), emb.sin()

        # prepare window reindex + cu_seqlens for windowed and full attention
        window_index, cu_win, cu_full = self._window_index_and_cu_seqlens(grid_thw)

        # Re-arrange tokens so that tokens from each LLM-grid position window are contiguous,
        # and within each position, the s^2 pre-merge tokens are contiguous.
        s2 = self.spatial_merge_unit
        S, D = x.shape
        assert S % s2 == 0, "Total tokens must be divisible by s^2."
        x = x.view(S // s2, s2, D)              # [N_llm, s^2, D]
        x = x[window_index, :, :]               # window order
        x = x.reshape(S, D)                     # back to [S, D]

        cos = cos.view(S // s2, s2, -1)[window_index, :, :].reshape(S, -1)
        sin = sin.view(S // s2, s2, -1)[window_index, :, :].reshape(S, -1)

        position_embeddings = (cos, sin)

        # run transformer
        for li, blk in enumerate(self.blocks):
            cu_now = cu_full if li in self.args.fullatt_block_indexes else cu_win
            x = blk(x, cu_seqlens=cu_now, position_embeddings=position_embeddings)

        # spatial merger: collapse each s^2 group -> 1, then undo window permutation
        x_merged = self.merger(x)  # [N_llm, out_dim]
        # invert window_index
        inv = torch.argsort(window_index)
        x_merged = x_merged[inv, :]  # restore (per-sample) llm-grid order

        # Done. Shape is per LLM-grid position across all frames and samples:
        # sum_i T_i * (H_i/s) * (W_i/s) rows, width out_hidden_size
        return x_merged