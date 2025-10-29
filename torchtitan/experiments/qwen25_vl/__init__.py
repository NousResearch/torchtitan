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
from .model.args import Qwen25VLModelArgs, Qwen25VLEncoderArgs
from .model.model import Qwen25VLTransformer
from .model.state_dict_adapter import Qwen25VLStateDictAdapter
#from .model.state_dict_adapter import Llama3Siglip2StateDictAdapter
# import qwen25_configs
from ..qwen25 import qwen25_configs


qwen25_vl_configs = {
    "7B": Qwen25VLModelArgs(
        **asdict(replace(qwen25_configs["7B"])),
        encoder=Qwen25VLEncoderArgs(
            dim=1280,
            out_dim=3584,
            ffn_dim=3420,
            n_layers=32,
            n_heads=16,
            patch_size=14,
            spatial_merge_size=2,
            n_pos_embs=27
        ),
    ),
}

def get_train_spec() -> TrainSpec:
    return TrainSpec(
        name="qwen25_vl",
        model_cls=Qwen25VLTransformer,
        model_args=qwen25_vl_configs,
        parallelize_fn=parallelize_vlm,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen25VLStateDictAdapter,
    )