# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

logger = logging.getLogger()

from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from .args import Qwen3VLModelArgs


class Qwen3VLStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_args: Qwen3VLModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = {

            # LANGUAGE MODEL WEIGHTS MAP 

            # Attention module
            "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.language_model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # MLP module for non-MoE
            "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Transformer layer
            "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.language_model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.language_model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.language_model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            "model.language_model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.language_model.norm.weight": "norm.weight",
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            "lm_head.weight": "output.weight",

            # MULTIMODAL ADAPTER WEIGHTS MAP
            "model.visual.merger.linear_fc1.bias": "encoder.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc1.weight": "encoder.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc2.bias": "encoder.merger.linear_fc2.bias",
            "model.visual.merger.linear_fc2.weight": "encoder.merger.linear_fc2.weight",
            "model.visual.merger.norm.weight": "encoder.merger.norm.weight",
            "model.visual.merger.norm.bias": "encoder.merger.norm.bias",

            "model.visual.deepstack_merger_list.{}.linear_fc1.bias": "encoder.deepstack_merger_list.{}.linear_fc1.bias",
            "model.visual.deepstack_merger_list.{}.linear_fc2.bias": "encoder.deepstack_merger_list.{}.linear_fc2.bias",
            "model.visual.deepstack_merger_list.{}.linear_fc1.weight": "encoder.deepstack_merger_list.{}.linear_fc1.weight",
            "model.visual.deepstack_merger_list.{}.linear_fc2.weight": "encoder.deepstack_merger_list.{}.linear_fc2.weight",
            "model.visual.deepstack_merger_list.{}.norm.weight": "encoder.deepstack_merger_list.{}.norm.weight",
            "model.visual.deepstack_merger_list.{}.norm.bias": "encoder.deepstack_merger_list.{}.norm.bias",

            "model.visual.blocks.{}.norm1.weight": "encoder.layers.{}.norm1.weight",
            "model.visual.blocks.{}.norm1.bias":   "encoder.layers.{}.norm1.bias",
            "model.visual.blocks.{}.norm2.weight": "encoder.layers.{}.norm2.weight",
            "model.visual.blocks.{}.norm2.bias":   "encoder.layers.{}.norm2.bias",

            "model.visual.patch_embed.proj.bias": "encoder.patch_embed.proj.bias",
            "model.visual.patch_embed.proj.weight": "encoder.patch_embed.proj.weight",
            "model.visual.pos_embed.weight": "encoder.pos_embed.weight",

            "model.visual.blocks.{}.attn.qkv.weight": "encoder.layers.{}.attn.qkv.weight",
            "model.visual.blocks.{}.attn.qkv.bias": "encoder.layers.{}.attn.qkv.bias",
            "model.visual.blocks.{}.attn.proj.weight": "encoder.layers.{}.attn.proj.weight",
            "model.visual.blocks.{}.attn.proj.bias": "encoder.layers.{}.attn.proj.bias",
            "model.visual.blocks.{}.mlp.linear_fc1.weight": "encoder.layers.{}.mlp.linear_fc1.weight",
            "model.visual.blocks.{}.mlp.linear_fc1.bias": "encoder.layers.{}.mlp.linear_fc1.bias",
            "model.visual.blocks.{}.mlp.linear_fc2.weight": "encoder.layers.{}.mlp.linear_fc2.weight",
            "model.visual.blocks.{}.mlp.linear_fc2.bias": "encoder.layers.{}.mlp.linear_fc2.bias",

            "model.visual.blocks.{}.norm1.weight": "encoder.layers.{}.norm1.weight",
            "model.visual.blocks.{}.norm1.bias":   "encoder.layers.{}.norm1.bias",
            "model.visual.blocks.{}.norm2.weight": "encoder.layers.{}.norm2.weight",
            "model.visual.blocks.{}.norm2.bias":   "encoder.layers.{}.norm2.bias",

        }

    # HuggingFace permutation function (exact copy from their conversion script)
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        hf_state_dict = {}


        for key, value in state_dict.items():
            if "layers" in key or 'blocks' in key or 'deepstack_merger_list' in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.

                if abstract_key == "layers.{}.attention.wq.weight":
                    value = self._permute(value, n_heads)
                elif abstract_key == "layers.{}.attention.wk.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            elif "patch_embed.proj.weight" in key: 
                value = value.view(-1, 3, 2, 16, 16)
                new_key = to_hf_map[key]
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        state_dict = {}

        for key, value in hf_state_dict.items():
            print(key, value.shape)
            if "layers" in key or 'blocks' in key or 'deepstack_merger_list' in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]

                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "model.language_model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                if abstract_key == "model.language_model.layers.{}.self_attn.k_proj.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)


                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            elif "patch_embed.proj.weight" in key:
                a, b, c, d, e = value.shape
                value = value.reshape(a, b * c * d * e)
                new_key = self.from_hf_map[key]
            else:
                new_key = self.from_hf_map[key]

            state_dict[new_key] = value
        
        return state_dict
