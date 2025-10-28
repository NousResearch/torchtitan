# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.components.tokenizer import HuggingFaceTokenizer

#from torchtitan.models.llama3 import TransformerModelArgs as Llama3Args
from torchtitan.experiments.qwen3.model.args import Qwen3ModelArgs


@dataclass
class SpecialTokens:
    img_token: str
    img_id: int
    boi_token: str
    boi_id: int
    eoi_token: str
    eoi_id: int
    pad_token: str
    pad_id: int
    ignore_id: int = -100  # Pytorch F.cross_entropy default

    @classmethod
    def from_tokenizer(cls, tokenizer: HuggingFaceTokenizer):
        SPECIAL_TOKENS_MAP = {
            "img": "<|image|>",
            "boi": "<|begin_of_image|>",
            "eoi": "<|end_of_image|>",
            "pad": "<|pad|>",
        }
        """
        SPECIAL_TOKENS_MAP = {
            "img": "<|pad|>",
            "boi": "<|pad|>",
            "eoi": "<|pad|>",
            "pad": "<|pad|>",
        }

        SPECIAL_TOKENS_MAP = {
            "img": "<image>",
            "boi": "<unk>",
            "eoi": "<unk>",
            "pad": "<|pad|>" }
        """
        added_tokens = tokenizer.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}
        special_tokens_dict = {}
        for prefix, tok in SPECIAL_TOKENS_MAP.items():
            special_tokens_dict[f"{prefix}_token"] = tok
            special_tokens_dict[f"{prefix}_id"] = token_to_id[tok]
        return cls(**special_tokens_dict)

@dataclass
class Qwen3VLEncoderArgs:
    dim: int = 768
    ffn_dim: int = 3072
    n_layers: int = 32
    n_heads: int = 16
    hidden_size: int = 1280 
    out_dim: int = 3584
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8,16,24])

    n_pos_embs: int = 16  # Number of positional embeddings per h&w
    n_channels: int = 3  # RGB channels
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2

    layer_norm_eps: float = 1e-6
    use_flex_attn: bool = True
    attn_mask_type: str = "causal"
    window_size: int = 112 
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [7,15,23,31])

@dataclass
class Qwen3VLModelArgs(Qwen3ModelArgs):
    encoder: Qwen3VLEncoderArgs = field(default_factory=Qwen3VLEncoderArgs)