# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP

from torchtitan.tools.logging import init_logger, logger

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa F401


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


def reverse_permute_1d(w, n_heads, dim1):
    return w.view(n_heads, 2, dim1 // n_heads // 2).transpose(1, 2).reshape(dim1)


@torch.inference_mode()
def convert_llama_weights(llama_model, output_dir):
    hf_model = AutoModelForCausalLM.from_pretrained(
        llama_model, torch_dtype=torch.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(llama_model)
    config = hf_model.config
    hf_state_dict = hf_model.state_dict()
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    dim = config.hidden_size
    dims_per_head = config.head_dim

    logger.info(f"Loading original Llama weights from {llama_model}")

    state_dict = {}
    n_heads_per_shard = n_heads
    num_key_value_heads = config.num_key_value_heads
    n_kv_heads_per_shard = num_key_value_heads
    query_dim = dims_per_head * n_heads
    key_value_dim = dims_per_head * num_key_value_heads
    for layer in range(n_layers):
        state_dict[f"layers.{layer}.attention_norm.weight"] = hf_state_dict[
            f"model.layers.{layer}.input_layernorm.weight"
        ]
        state_dict[f"layers.{layer}.ffn_norm.weight"] = hf_state_dict[
            f"model.layers.{layer}.post_attention_layernorm.weight"
        ]

        for wn, hn, nh in [
            ("wq", "q_proj", n_heads_per_shard),
            ("wk", "k_proj", n_kv_heads_per_shard),
            ("wv", "v_proj", n_kv_heads_per_shard),
        ]:
            if wn != "wv":
                # Need to reverse the permutation for sliced rotary
                state_dict[f"layers.{layer}.attention.{wn}.weight"] = reverse_permute(
                    hf_state_dict[f"model.layers.{layer}.self_attn.{hn}.weight"],
                    n_heads if wn == "wq" else num_key_value_heads,
                    dim1=query_dim if wn == "wq" else key_value_dim,
                    dim2=dim,
                )
                if f"model.layers.{layer}.self_attn.{hn}.bias" in hf_state_dict:
                    # bias is used in Qwen
                    state_dict[
                        f"layers.{layer}.attention.{wn}.bias"
                    ] = reverse_permute_1d(
                        hf_state_dict[f"model.layers.{layer}.self_attn.{hn}.bias"],
                        n_heads if wn == "wq" else num_key_value_heads,
                        dim1=query_dim if wn == "wq" else key_value_dim,
                    )
            else:
                state_dict[f"layers.{layer}.attention.{wn}.weight"] = hf_state_dict[
                    f"model.layers.{layer}.self_attn.{hn}.weight"
                ]
                if f"model.layers.{layer}.self_attn.{hn}.bias" in hf_state_dict:
                    # bias is used in Qwen
                    state_dict[f"layers.{layer}.attention.{wn}.bias"] = hf_state_dict[
                        f"model.layers.{layer}.self_attn.{hn}.bias"
                    ]

        state_dict[f"layers.{layer}.attention.wo.weight"] = hf_state_dict[
            f"model.layers.{layer}.self_attn.o_proj.weight"
        ]

        # Add q_norm and k_norm if they exist (for models like Qwen3)
        # These operate on head_dim but still need permutation due to sliced rotary
        if f"model.layers.{layer}.self_attn.q_norm.weight" in hf_state_dict:
            # q_norm and k_norm weights are of size head_dim
            state_dict[f"layers.{layer}.attention.q_norm.weight"] = reverse_permute_1d(
                hf_state_dict[f"model.layers.{layer}.self_attn.q_norm.weight"],
                1,  # Single head dimension
                dims_per_head,  # head_dim
            )
        if f"model.layers.{layer}.self_attn.k_norm.weight" in hf_state_dict:
            state_dict[f"layers.{layer}.attention.k_norm.weight"] = reverse_permute_1d(
                hf_state_dict[f"model.layers.{layer}.self_attn.k_norm.weight"],
                1,  # Single head dimension
                dims_per_head,  # head_dim
            )

        state_dict[f"layers.{layer}.feed_forward.w1.weight"] = hf_state_dict[
            f"model.layers.{layer}.mlp.gate_proj.weight"
        ]

        state_dict[f"layers.{layer}.feed_forward.w2.weight"] = hf_state_dict[
            f"model.layers.{layer}.mlp.down_proj.weight"
        ]

        state_dict[f"layers.{layer}.feed_forward.w3.weight"] = hf_state_dict[
            f"model.layers.{layer}.mlp.up_proj.weight"
        ]

    state_dict["norm.weight"] = hf_state_dict["model.norm.weight"]
    state_dict["tok_embeddings.weight"] = hf_state_dict["model.embed_tokens.weight"]
    state_dict["output.weight"] = hf_state_dict["lm_head.weight"]

    logger.info(f"Writing to DCP at '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir, thread_count=1)
    DCP.save(state_dict, storage_writer=storage_writer, no_dist=True)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(tokenizer_dir)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert Llama weights to DCP format.")
    parser.add_argument("llama_model", type=str, help="HF Model in llama format")
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    args = parser.parse_args()

    convert_llama_weights(args.llama_model, args.output_dir)
