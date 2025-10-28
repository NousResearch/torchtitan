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
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_qwen3
from .model.args import Qwen25ModelArgs
from .model.model import Qwen25Model
from .model.state_dict_adapter import Qwen25StateDictAdapter

__all__ = [
    "parallelize_qwen3",
    "Qwen3ModelArgs",
    "Qwen3Model",
    "qwen3_configs",
]

# Adding different variants of the model

qwen25_configs = {
    "7B": Qwen25ModelArgs(
        vocab_size=152064,
        max_seq_len=4096,
        head_dim=128,
        dim=3584,
        n_layers=28,
        n_heads=28,
        n_kv_heads=4,
        qk_norm=True,
        hidden_dim=18944,
        rope_theta=1000000,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        name="qwen3",
        model_cls=Qwen25Model,
        model_args=qwen25_configs,  # Change from dict to Mapping
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen25StateDictAdapter,
    )
