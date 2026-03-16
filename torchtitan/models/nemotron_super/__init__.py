# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.hf_datasets.dataloader import build_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_qwen3
from .model.args import NemotronSuperModelArgs
from .model.model import NemotronSuperModel
from .model.state_dict_adapter import NemotronSuperStateDictAdapter

__all__ = [
    "parallelize_qwen3",
    "NemotronSuperModelArgs",
    "NemotronSuperModel",
    "nemotron_super_args",
]


# ----------------------------------------------------------------------
# Nemotron Super (NemotronH) model configs
# ----------------------------------------------------------------------

# HF config reference for 120B-A12B:
# https://huggingface.co/nvidia/Nemotron-H-120B-A12B/blob/main/config.json
#
# Each layer = (Mamba2, [Attention], MoE). attn_layer_idxs controls which
# layers include attention between mamba and moe.
#
# Architecture notes:
#   - MoE experts are NON-GATED relu^2 (up+down only, no gate_proj)
#   - Latent MoE: bottleneck projection before/after experts (4096->1024->experts->1024->4096)
#   - 512 experts, top-22 sigmoid routing with e_score_correction_bias
#   - Mamba2 via FLA

nemotron_super_args = {
    "120B-A12B": NemotronSuperModelArgs(
        dim=4096,
        n_layers=40,
        n_heads=32,
        n_kv_heads=2,
        head_dim=128,
        vocab_size=131072,
        norm_eps=1e-5,
        max_seq_len=262144,
        rope_theta=10000.0,
        num_nextn_predict_layers=0,
        # 8 of 40 layers get attention, roughly every 4-5
        attn_layer_idxs=[3, 7, 11, 16, 21, 26, 31, 35],
        # mamba2
        mamba_num_heads=128,
        mamba_head_dim=64,
        ssm_state_size=128,
        conv_kernel=4,
        chunk_size=128,
        n_groups=8,
        mamba_expand=2,
        mamba_hidden_act="silu",
        use_conv_bias=True,
        use_mamba_proj_bias=False,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=0.0001,
        # MoE
        moe_args=MoEArgs(
            num_experts=512,
            num_shared_experts=1,
            top_k=22,
            score_func="sigmoid",
            route_norm=True,
            route_scale=5.0,
            gate_bias=False,
            score_before_experts=False,
            num_expert_groups=1,
            num_limited_groups=1,
            gated_experts=False,
            expert_act="relu2",
            expert_intermediate_size=2688,
            shared_expert_intermediate_size=5376,
            latent_size=1024,
            expert_bias=False,
        ),
    ),
    "120B-A12B-mtp": NemotronSuperModelArgs(
        dim=4096,
        n_layers=40,
        n_heads=32,
        n_kv_heads=2,
        head_dim=128,
        vocab_size=131072,
        norm_eps=1e-5,
        max_seq_len=262144,
        rope_theta=10000.0,
        num_nextn_predict_layers=1,
        # 8 of 40 layers get attention, roughly every 4-5
        attn_layer_idxs=[3, 7, 11, 16, 21, 26, 31, 35],
        # mamba2
        mamba_num_heads=128,
        mamba_head_dim=64,
        ssm_state_size=128,
        conv_kernel=4,
        chunk_size=128,
        n_groups=8,
        mamba_expand=2,
        mamba_hidden_act="silu",
        use_conv_bias=True,
        use_mamba_proj_bias=False,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=0.0001,
        # MoE
        moe_args=MoEArgs(
            num_experts=512,
            num_shared_experts=1,
            top_k=22,
            score_func="sigmoid",
            route_norm=True,
            route_scale=5.0,
            gate_bias=False,
            score_before_experts=False,
            num_expert_groups=1,
            num_limited_groups=1,
            gated_experts=False,
            expert_act="relu2",
            expert_intermediate_size=2688,
            shared_expert_intermediate_size=5376,
            latent_size=1024,
            expert_bias=False,
        ),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=NemotronSuperModel,
        model_args=nemotron_super_args,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=NemotronSuperStateDictAdapter,
    )
