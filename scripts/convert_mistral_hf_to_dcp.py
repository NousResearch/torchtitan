# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP

from torchtitan.tools.logging import logger
from torchtitan.models.mistral3.model.model import precompute_freqs_cis

from transformers import AutoTokenizer, Mistral3Config


# permute for sliced rotary
def permute(w, n_heads, dim1, dim2):
    return (
        w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


# And reversed
def reverse_permute(w, n_heads, dim1, dim2):
    return (
        w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


@torch.inference_mode()
def convert_mistral3_weights(mistral_model, output_dir, max_seq_len: int):
    # Loading the model directly might be too large, so we'll use safetensors to load the weights
    from safetensors import safe_open
    import os
    import json
    
    # Load the config
    config_path = os.path.join(mistral_model, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = Mistral3Config.from_dict(config_dict)
    tok = AutoTokenizer.from_pretrained(mistral_model)
    
    # Find all safetensors files
    index_path = os.path.join(mistral_model, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        all_files = set(weight_map.values())
    else:
        # If no index, look for a single safetensors file
        safetensors_files = [f for f in os.listdir(mistral_model) if f.endswith(".safetensors")]
        if len(safetensors_files) == 1:
            all_files = [safetensors_files[0]]
        else:
            raise ValueError("Multiple safetensors files found without an index file")
    
    # Extract language model parameters from the text config
    text_config = config.text_config
    n_layers = text_config.num_hidden_layers
    n_heads = text_config.num_attention_heads
    dim = text_config.hidden_size
    dims_per_head = dim // n_heads

    logger.info(f"Loading original Mistral3 weights from {mistral_model}")
    
    state_dict = {}
    n_heads_per_shard = n_heads
    num_key_value_heads = text_config.num_key_value_heads
    n_kv_heads_per_shard = num_key_value_heads
    key_value_dim = dims_per_head * num_key_value_heads
    
    # Load and process weights
    hf_state_dict = {}
    for filename in all_files:
        filepath = os.path.join(mistral_model, filename)
        with safe_open(filepath, framework="pt") as f:
            for key in f.keys():
                hf_state_dict[key] = f.get_tensor(key)
    
    # Process language model layers
    for layer in range(n_layers):
        # Map from HF to torchtitan structure
        # Based on the keys we found in the checkpoint

        # Norm layers
        state_dict[f"language_model.layers.{layer}.ln_attn.weight"] = hf_state_dict[
            f"language_model.model.layers.{layer}.input_layernorm.weight"
        ]
        state_dict[f"language_model.layers.{layer}.ln_mlp.weight"] = hf_state_dict[
            f"language_model.model.layers.{layer}.post_attention_layernorm.weight"
        ]

        # Attention layers
        for wn, hn, nh in [
            ("wq", "q_proj", n_heads_per_shard),
            ("wk", "k_proj", n_kv_heads_per_shard),
            ("wv", "v_proj", n_kv_heads_per_shard),
        ]:
            if wn != "wv":
                # Need to reverse the permutation for sliced rotary
                
                state_dict[f"language_model.layers.{layer}.attn.{wn}.weight"] = reverse_permute(
                    hf_state_dict[f"language_model.model.layers.{layer}.self_attn.{hn}.weight"],
                    n_heads if wn == "wq" else num_key_value_heads,
                    dim1=4096 if wn == "wq" else int(key_value_dim*0.8),
                    dim2=dim,
                )
            else:

                state_dict[f"language_model.layers.{layer}.attn.{wn}.weight"] = hf_state_dict[
                    f"language_model.model.layers.{layer}.self_attn.{hn}.weight"
                ]

        state_dict[f"language_model.layers.{layer}.attn.wo.weight"] = hf_state_dict[
            f"language_model.model.layers.{layer}.self_attn.o_proj.weight"
        ]

        # Feed-forward layers
        state_dict[f"language_model.layers.{layer}.mlp.w1.weight"] = hf_state_dict[
            f"language_model.model.layers.{layer}.mlp.gate_proj.weight"
        ]
        state_dict[f"language_model.layers.{layer}.mlp.w2.weight"] = hf_state_dict[
            f"language_model.model.layers.{layer}.mlp.down_proj.weight"
        ]
        state_dict[f"language_model.layers.{layer}.mlp.w3.weight"] = hf_state_dict[
            f"language_model.model.layers.{layer}.mlp.up_proj.weight"
        ]

    # Language model norm and embeddings
    state_dict["language_model.norm.weight"] = hf_state_dict["language_model.model.norm.weight"]
    
    # Handling embeddings
    state_dict["language_model.tok_embeddings.weight"] = hf_state_dict["language_model.model.embed_tokens.weight"]
    # If fusion embedding exists in the HF model
        
    state_dict["language_model.output.weight"] = hf_state_dict["language_model.lm_head.weight"]
    
    # Vision tower components
    if "vision_tower.ln_pre.weight" in hf_state_dict:
        # Copy over vision tower weights, restructuring to put them under model.vision_encoder.pixtral_vision
        vision_keys = [k for k in hf_state_dict.keys() if k.startswith("vision_tower.")]
        for key in vision_keys:
            # Replace vision_tower with vision_encoder.pixtral_vision in the key path
            new_key = key.replace("vision_tower", "vision_encoder.pixtral_vision")
            state_dict[new_key] = hf_state_dict[key]
    
    # Multi-modal projector
    mm_keys = [k for k in hf_state_dict.keys() if k.startswith("multi_modal_projector.")]
    for key in mm_keys:
        state_dict["vision_encoder." + key] = hf_state_dict[key]

    # TODO figure out how to not hardcode
    dims_per_head = 128

    # NOTE: precompute freqs_cis because must be persisted by default in torchtitan
    state_dict["language_model.freqs_cis"] = precompute_freqs_cis(
        dims_per_head,
        max_seq_len,
        text_config.rope_theta,
    )

    print(state_dict.keys())

    logger.info(f"Writing to DCP at '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir, thread_count=8)

    DCP.save(state_dict, storage_writer=storage_writer)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(tokenizer_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mistral3 weights to DCP format.")
    parser.add_argument("mistral_model", type=Path, help="HF Model in Mistral3 format")
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=131072,
        help="The maximum sequence length of the model.",
    )
    args = parser.parse_args()

    convert_mistral3_weights(
        args.mistral_model, args.output_dir, max_seq_len=args.max_seq_len
    )
