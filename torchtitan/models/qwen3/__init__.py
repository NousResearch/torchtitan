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
from torchtitan.datasets.dataloader import build_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_qwen3
from .infra.pipeline import pipeline_qwen3
from .model.args import TransformerModelArgs
from .model.model import Transformer

__all__ = [
    "parallelize_qwen3",
    "pipeline_qwen3",
    "TransformerModelArgs",
    "Transformer",
    "qwen3_configs",
]


qwen3_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, qk_norm=True, rope_theta=500000
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        qk_norm=True,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "4B": TransformerModelArgs(
        dim=2560,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        rope_theta=1000000,
        head_dim=128,
        norm_eps=1e-06,
        vocab_size=151936,
        ffn_dim_multiplier=1.425,  # To get intermediate_size=17408
        multiple_of=128,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        rope_theta=1000000,
        norm_eps=1e-06,
        vocab_size=151936,
        ffn_dim_multiplier=1.125,  # To get intermediate_size=12288
        multiple_of=256,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "14B": TransformerModelArgs(
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=8,
        qk_norm=True,
        rope_theta=1000000,
        norm_eps=1e-06,
        vocab_size=151936,
        ffn_dim_multiplier=1.275,  # To get intermediate_size=17408
        multiple_of=256,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "4B_finetuning": TransformerModelArgs(
        dim=2560,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        rope_theta=1000000,
        head_dim=128,
        norm_eps=1e-06,
        vocab_size=151936,
        ffn_dim_multiplier=1.425,  # To get intermediate_size=17408
        multiple_of=128,
        use_flex_attn=True,
        attn_mask_type="block_causal_by_sequence_lengths",
    ),
    "8B_finetuning": TransformerModelArgs(
        dim=4096,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        rope_theta=1000000,
        norm_eps=1e-06,
        vocab_size=151936,
        ffn_dim_multiplier=1.125,  # To get intermediate_size=12288
        multiple_of=256,
        use_flex_attn=True,
        attn_mask_type="block_causal_by_sequence_lengths",
    ),
    "14B_finetuning": TransformerModelArgs(
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=8,
        qk_norm=True,
        rope_theta=1000000,
        norm_eps=1e-06,
        vocab_size=151936,
        ffn_dim_multiplier=1.275,  # To get intermediate_size=17408
        multiple_of=256,
        use_flex_attn=True,
        attn_mask_type="block_causal_by_sequence_lengths",
    ),
}

register_train_spec(
    TrainSpec(
        name="qwen3",
        model_cls=Transformer,
        model_args=qwen3_configs,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_qwen3,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
)
