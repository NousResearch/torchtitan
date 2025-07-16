# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint
import torch.distributed.checkpoint.format_utils

from torchtitan.tools.logging import init_logger, logger

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM  # noqa F401


# permute for sliced rotary
def permute(w, n_heads, dim1, dim2):
    return (
        w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def permute_1d(w, n_heads, dim1):
    return w.view(n_heads, dim1 // n_heads // 2, 2).transpose(1, 2).reshape(dim1)


def param_to_hf_processing(name, param, num_heads, dim1, dim2, num_kv_heads=None):
    if "attention.w" in name:
        # In the attention layer!
        if ("wq.weight" in name) or ("wk.weight" in name):
            # need to permute the param...
            out_param = permute(
                param.detach(),
                num_heads if "wq" in name else 8,
                dim1 if "wq" in name else dim2,
                dim1,
            )
        else:
            out_param = param
        out_name = "model." + name.split("attention")[0] + "self_attn."
        mapping = {"wq": "q_proj", "wk": "k_proj", "wv": "v_proj", "wo": "o_proj"}
        out_name += mapping[name.split("attention.")[1].split(".")[0]] + ".weight"
    elif "attention.q_norm" in name or "attention.k_norm" in name:
        # Handle q_norm and k_norm weights for models like Qwen3
        out_param = permute_1d(param.detach(), 1, param.shape[0])
        out_name = "model." + name.split("attention")[0] + "self_attn."
        if "q_norm" in name:
            out_name += "q_norm.weight"
        else:
            out_name += "k_norm.weight"
    elif "feed_forward" in name:
        out_name = "model." + name.replace("feed_forward", "mlp")
        mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for key, val in mapping.items():
            out_name = out_name.replace(key, val)
        out_param = param
    elif "_norm" in name:
        out_name = "model." + name.replace("attention_norm", "input_layernorm").replace(
            "ffn_norm", "post_attention_layernorm"
        )
        out_param = param
    elif "output.weight" == name:
        out_name = "lm_head.weight"
        out_param = param
    else:
        # emb/out layer
        out_name = "model." + name.replace("tok_embeddings", "embed_tokens")
        out_param = param
    out_name = out_name.replace("._checkpoint_wrapped_module.", ".")
    return out_name, out_param


@torch.inference_mode()
def convert_llama_weights(input_dir, output_dir, model_base, tokenizer):
    hf_model = LlamaForCausalLM.from_pretrained(model_base)
    tok = AutoTokenizer.from_pretrained(tokenizer if tokenizer else model_base)
    config = hf_model.config  # type: LlamaConfig
    dim = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    dims_per_head = dim // n_heads
    kv_dim = dims_per_head * config.num_key_value_heads
    hf_state_dict = hf_model.state_dict()
    print("Starting keys and shapes...")
    for key, val in hf_state_dict.items():
        print(f"{key}: {val.shape}")
    logger.info(f"Loading finetuned Llama weights from {input_dir}")
    sd = {}
    torch.distributed.checkpoint.format_utils._load_state_dict(
        sd,
        torch.distributed.checkpoint.filesystem.FileSystemReader(input_dir),
        planner=torch.distributed.checkpoint.format_utils._EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    # now to convert...
    if "model" in sd.keys():
        sd = sd["model"]
    for name, param in sd.items():
        if name == "freqs_cis" or name == "train_state" or name == "optimizer" or name == "dataloader" or name == "lr_scheduler":
            continue
        hf_name, hf_param = param_to_hf_processing(
            name, param, n_heads, dim, kv_dim, n_kv_heads
        )
        hf_state_dict[hf_name] = hf_param
        print(f"Converted {name} to {hf_name}")
    print("Ending keys and shapes...")
    for key, val in hf_state_dict.items():
        try:
            print(f"{key}: {val.shape}")
        except AttributeError as e:
            pass
    # now update the state dict
    hf_model.load_state_dict(hf_state_dict)
    # save in bf16 because it's a nice format :)
    hf_model = hf_model.to(dtype=torch.bfloat16)
    hf_model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert Torchtitan DCP weights to HF")
    parser.add_argument("input_dir", type=Path, help="Input directory for DCP.")
    parser.add_argument("output_dir", type=Path, help="Output directory for HF.")
    parser.add_argument("llama_model", type=str, help="The base model used in hf.")
    parser.add_argument("--tokenizer", type=str, help="The base model used in hf.")
    args = parser.parse_args()

    convert_llama_weights(args.input_dir, args.output_dir, args.llama_model, args.tokenizer)
