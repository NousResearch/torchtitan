# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, replace

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.experiments.qwen25.model.model import Qwen25Model
from torchtitan.protocols.train_spec import TrainSpec

from .datasets.mm_datasets import build_mm_dataloader
from .infra.parallelize import parallelize_vlm
from .model.args import Qwen3VLModelArgs, Qwen3VLEncoderArgs
from .model.model import Qwen3VLTransformer
from .model.state_dict_adapter import Qwen3VLStateDictAdapter
# import qwen25_configs
from ..qwen3 import qwen3_configs


qwen3_vl_configs = {
    "8B": Qwen3VLModelArgs(
        **asdict(replace(qwen3_configs["8B"])),
        encoder=Qwen3VLEncoderArgs(
            dim=1152,
            hidden_size=1152,
            out_dim=4096,
            ffn_dim=4304,
            n_layers=27,
            n_heads=16,
            patch_size=16,
            spatial_merge_size=2,
            n_pos_embs=27

        ),
    ),
}

def get_train_spec() -> TrainSpec:
    return TrainSpec(
        name="qwen3_vl",
        model_cls=Qwen3VLTransformer,
        model_args=qwen3_vl_configs,
        parallelize_fn=parallelize_vlm,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3VLStateDictAdapter,
    )