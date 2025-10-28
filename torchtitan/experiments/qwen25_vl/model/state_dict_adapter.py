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
from .args import Llama3Siglip2ModelArgs


class Llama3Siglip2StateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_args: Llama3Siglip2ModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = {

            # LANGUAGE MODEL WEIGHTS MAP 
            "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
            "language_model.model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "language_model.model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "language_model.model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "language_model.model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "language_model.model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "language_model.model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "language_model.model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "language_model.model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "language_model.model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "language_model.model.norm.weight": "norm.weight",
            "language_model.lm_head.weight": "output.weight",

            # MULTIMODAL ADAPTER WEIGHTS MAP
            "multi_modal_projector.linear_1.weight": "projector.w1.weight",
            "multi_modal_projector.linear_1.bias": "projector.w1.bias",
            "multi_modal_projector.linear_2.weight": "projector.w2.weight",
            "multi_modal_projector.linear_2.bias": "projector.w2.bias",

            # ENCODER/VISION ADAPTER WEIGHTS MAP 
            "vision_tower.vision_model.embeddings.patch_embedding.weight": "encoder.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias": "encoder.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight": "encoder.embeddings.position_embedding.weight",
            "vision_tower.vision_model.encoder.layers.{}.layer_norm1.weight": "encoder.layers.{}.layer_norm1.weight",
            "vision_tower.vision_model.encoder.layers.{}.layer_norm1.bias": "encoder.layers.{}.layer_norm1.bias",
            "vision_tower.vision_model.encoder.layers.{}.layer_norm2.weight": "encoder.layers.{}.layer_norm2.weight",
            "vision_tower.vision_model.encoder.layers.{}.layer_norm2.bias": "encoder.layers.{}.layer_norm2.bias",            
            "vision_tower.vision_model.encoder.layers.{}.self_attn.q_proj.weight": "encoder.layers.{}.self_attn.q_proj.weight",
            "vision_tower.vision_model.encoder.layers.{}.self_attn.k_proj.weight": "encoder.layers.{}.self_attn.k_proj.weight",
            "vision_tower.vision_model.encoder.layers.{}.self_attn.v_proj.weight": "encoder.layers.{}.self_attn.v_proj.weight",
            "vision_tower.vision_model.encoder.layers.{}.self_attn.out_proj.weight": "encoder.layers.{}.self_attn.out_proj.weight",
            "vision_tower.vision_model.encoder.layers.{}.self_attn.q_proj.bias": "encoder.layers.{}.self_attn.q_proj.bias",
            "vision_tower.vision_model.encoder.layers.{}.self_attn.k_proj.bias": "encoder.layers.{}.self_attn.k_proj.bias",
            "vision_tower.vision_model.encoder.layers.{}.self_attn.v_proj.bias": "encoder.layers.{}.self_attn.v_proj.bias",
            "vision_tower.vision_model.encoder.layers.{}.self_attn.out_proj.bias": "encoder.layers.{}.self_attn.out_proj.bias",
            "vision_tower.vision_model.encoder.layers.{}.mlp.fc1.weight": "encoder.layers.{}.mlp.fc1.weight",
            "vision_tower.vision_model.encoder.layers.{}.mlp.fc1.bias": "encoder.layers.{}.mlp.fc1.bias",
            "vision_tower.vision_model.encoder.layers.{}.mlp.fc2.weight": "encoder.layers.{}.mlp.fc2.weight",
            "vision_tower.vision_model.encoder.layers.{}.mlp.fc2.bias": "encoder.layers.{}.mlp.fc2.bias",
            "vision_tower.vision_model.post_layernorm.weight": "encoder.post_layernorm.weight",
            "vision_tower.vision_model.post_layernorm.bias": "encoder.post_layernorm.bias",
            "vision_tower.vision_model.head.attention.in_proj_weight": None,
            "vision_tower.vision_model.head.attention.in_proj_bias": None,
            "vision_tower.vision_model.head.attention.out_proj.weight": None, 
            "vision_tower.vision_model.head.attention.out_proj.bias": None,
            "vision_tower.vision_model.head.layernorm.weight": None,
            "vision_tower.vision_model.head.layernorm.bias": None,
            "vision_tower.vision_model.head.mlp.fc1.weight": None,
            "vision_tower.vision_model.head.mlp.fc1.bias": None,
            "vision_tower.vision_model.head.mlp.fc2.weight": None,
            "vision_tower.vision_model.head.mlp.fc2.bias": None,
            "vision_tower.vision_model.head.probe": None
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
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.

                if abstract_key == "layers.{}.attention.wq.weight":
                    value = self._permute(value, n_heads)
                if abstract_key == "layers.{}.attention.wk.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            elif "patch_embedding.weight" in key: 
                value = value.view(-1, 3, 14, 14)
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
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]

                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)


                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            elif "patch_embedding.weight" in key:
                a, b, c, d = value.shape
                value = value.reshape(a, b * c * d)
                new_key = self.from_hf_map[key]
            else:
                new_key = self.from_hf_map[key]

            state_dict[new_key] = value
        
        return state_dict
