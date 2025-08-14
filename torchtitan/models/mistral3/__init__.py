# Copyright (c) 2025, Anthropic Research Labs
# All rights reserved.

from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.dataloader import build_dataloader

#from .model.configuration_pixtral import PixtralVisionConfig
from .model.model import VLMArgs, VLM

from .infra.parallelize import parallelize_mistral3
from .infra.pipeline import pipeline_mistral3


from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_mistral3
from .infra.pipeline import pipeline_mistral3
from .model.model import VLM


__all__ = [
    "parallelize_mistral3",
    "pipeline_mistral3",
    "VLMArgs",
    "VLM",
    "mistral3_configs",
]

# Define model configurations
mistral3_configs = {
    "24B": VLMArgs(
        # vision encoder part
        vision_embed_dim=1024,
        vision_num_layers=24,
        vision_num_heads=16,
        vision_feature_layer=-2,
        patch_size=14,
        image_size=1540,
        in_channels=3,
        spatial_merge_size=2,
        
        # projection part
        num_layers_projection=8,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,

        # decoder part
        decoder_embed_dim=5120,
        decoder_num_layers=40,
        decoder_num_heads=32,
        decoder_num_kv_heads=8,
        fusion_interval=8,
        image_token_index=10,
        
        # common part
        vocab_size=131072,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        rope_theta=1000000000.0,
        max_seq_len=131072,
        use_flex_attn=True,
        attn_mask_type="block_causal_by_sequence_lengths",
    ),
}


# Register the model
register_train_spec(
    TrainSpec(
        name="mistral3",
        parallelize_fn=parallelize_mistral3,
        model_cls=VLM,
        model_args=mistral3_configs,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        pipelining_fn=pipeline_mistral3,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
    )
)