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

from transformers import AutoTokenizer, AutoConfig, Mistral3ForConditionalGeneration# noqa F401





def permute_1d(w: torch.Tensor, n_heads: int, dim1: int) -> torch.Tensor:
    return w.view(n_heads, dim1 // n_heads // 2, 2).transpose(1, 2).reshape(dim1)




# permute for sliced rotary
def permute(w, n_heads, dim1, dim2):
    print("n_heads ", n_heads)
    print("dim1 ", dim1)
    print("dim2 ", dim2)
    return (
        w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


# And reversed
def reverse_permute(w, n_heads, dim1, dim2):
    print("n_heads ", n_heads)
    print("dim1 ", dim1)
    print("dim2 ", dim2)
    return (
        w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def param_to_hf_processing(
    name: str,
    param: torch.Tensor,
    num_heads: int,
    hidden_size: int,
    kv_dim: int,
    num_kv_heads: int,
):
    # language model layers
    if name.startswith("language_model.layers."):
        # Example: language_model.layers.{L}.ln_attn.weight
        if name.endswith(".ln_attn.weight"):
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".ln_attn.weight", ".input_layernorm.weight")
            return out_name, param
        if name.endswith(".ln_mlp.weight"):
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".ln_mlp.weight", ".post_attention_layernorm.weight")
            return out_name, param

        # Attention projections
        if ".attn.wq.weight" in name:
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".attn.wq.weight", ".self_attn.q_proj.weight")
            #out_param = permute(param.detach(), num_heads, 4096, hidden_size)
            out_param = reverse_permute(param.detach(), 32, 4096, 5120)
            return out_name, out_param
        if ".attn.wk.weight" in name:
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".attn.wk.weight", ".self_attn.k_proj.weight")
            #out_param = permute(param.detach(), num_kv_heads, kv_dim, hidden_size)
            out_param = reverse_permute(param.detach(), 8, 1024, 5120)
            return out_name, out_param
        if ".attn.wv.weight" in name:
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".attn.wv.weight", ".self_attn.v_proj.weight")
            return out_name, param
        if ".attn.wo.weight" in name:
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".attn.wo.weight", ".self_attn.o_proj.weight")
            return out_name, param

        # Optional q_norm / k_norm 1D weights (not present in current Mistral3 export, but safe to handle)
        if ".attn.q_norm.weight" in name:
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".attn.q_norm.weight", ".self_attn.q_norm.weight")
            out_param = permute_1d(param.detach(), 1, param.shape[0])
            return out_name, out_param
        if ".attn.k_norm.weight" in name:
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".attn.k_norm.weight", ".self_attn.k_norm.weight")
            out_param = permute_1d(param.detach(), 1, param.shape[0])
            return out_name, out_param

        # MLP
        if ".mlp.w1.weight" in name:
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".mlp.w1.weight", ".mlp.gate_proj.weight")
            return out_name, param
        if ".mlp.w2.weight" in name:
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".mlp.w2.weight", ".mlp.down_proj.weight")
            return out_name, param
        if ".mlp.w3.weight" in name:
            out_name = name.replace(
                "language_model.layers.", "model.language_model.layers."
            ).replace(".mlp.w3.weight", ".mlp.up_proj.weight")
            return out_name, param

    # language model top-level components
    if name == "language_model.norm.weight":
        return "model.language_model.norm.weight", param
    if name == "language_model.tok_embeddings.weight":
        return "model.language_model.embed_tokens.weight", param
    if name == "language_model.output.weight":
        return "lm_head.weight", param

    # Vision tower and multimodal projector: keep the same keys (HF uses the same prefix here in our export)
    if name.startswith("vision_tower."):
        return name.replace("vision_tower.", "model.vision_tower."), param
    if name.startswith("multi_modal_projector."):
        return name.replace("multi_modal_projector.", "model.multi_modal_projector."), param

    # Unknown/unhandled: return None to skip
    return None, None


@torch.inference_mode()
def convert_mistral_weights(input_dir: Path, output_dir: Path, model_base: str, tokenizer: str | None) -> None:
    hf_model = Mistral3ForConditionalGeneration.from_pretrained(model_base)
    tok = AutoTokenizer.from_pretrained(tokenizer if tokenizer else model_base)
    config: AutoConfig = hf_model.config

    # Mistral3 models store text config under text_config; fall back to top-level for generic models
    text_cfg = getattr(config, "text_config", config)
    hidden_size: int = int(text_cfg.hidden_size)
    num_heads: int = int(text_cfg.num_attention_heads)
    num_kv_heads: int = int(getattr(text_cfg, "num_key_value_heads", num_heads))
    dims_per_head: int = 128 #hidden_size // num_heads
    kv_dim: int = dims_per_head * num_kv_heads

    hf_state_dict = hf_model.state_dict()

    logger.info(f"Loading TorchTitan Mistral DCP weights from {input_dir}")
    sd: dict[str, torch.Tensor] = {}
    torch.distributed.checkpoint.format_utils._load_state_dict(
        sd,
        torch.distributed.checkpoint.filesystem.FileSystemReader(input_dir),
        planner=torch.distributed.checkpoint.format_utils._EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    # Some checkpoints might nest parameters under 'model'
    if "model" in sd:
        sd = sd["model"]

    skipped = {"language_model.freqs_cis", "train_state", "optimizer", "dataloader", "lr_scheduler"}

    for name, param in sd.items():
        if any(name == s or name.startswith(s + ".") for s in skipped):
            continue
        out_name, out_param = param_to_hf_processing(
            name, param, num_heads, hidden_size, kv_dim, num_kv_heads
        )
        if out_name is None:
            logger.debug(f"Skipping unrecognized key: {name}")
            continue
        hf_state_dict[out_name] = out_param
        logger.info(f"Converted {name} -> {out_name}")

    # Load updated weights into the HF model

    print("my state dict", hf_state_dict.keys())
    print("model state dict", hf_model.state_dict().keys())
    hf_model.load_state_dict(hf_state_dict)

    # Save in bf16
    hf_model = hf_model.to(dtype=torch.bfloat16)
    hf_model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert TorchTitan Mistral DCP weights to HF format.")
    parser.add_argument("input_dir", type=Path, help="Input directory containing DCP checkpoint")
    parser.add_argument("output_dir", type=Path, help="Output directory for HF model")
    parser.add_argument("mistral_model", type=str, help="Base HF model to load config and architecture from")
    parser.add_argument("--tokenizer", type=str, help="Optional tokenizer source; defaults to base model", default=None)
    args = parser.parse_args()

    convert_mistral_weights(args.input_dir, args.output_dir, args.mistral_model, args.tokenizer)


