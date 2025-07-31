
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.models.attention import build_attention, init_attention_mask

from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config_manager import JobConfig


from dataclasses import dataclass

from dataclasses import dataclass

from torch import nn

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs

@dataclass
class VLMArgs(BaseModelArgs):
    # vision encoder part
    vision_embed_dim: int = 1024
    vision_num_layers: int = 24
    vision_num_heads: int = 16
    vision_feature_layer: int = -1
    patch_size: int = 14
    image_size: int = 1540
    in_channels: int = 3
    # For merging patches
    spatial_merge_size: int = 2
    
    # projection part
    num_layers_projection: int = 8
    projector_hidden_act: str = "gelu"
    multimodal_projector_bias: bool = False

    # decoder part
    decoder_embed_dim: int = 5120
    decoder_num_layers: int = 40
    decoder_num_heads: int = 32
    decoder_num_kv_heads: int = 8
    fusion_interval: int = 8  # Interval for fusion of vision features into text model
    image_token_index: int = 10  # Token ID representing an image in the text
    
    # common part
    vocab_size: int = 131072
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 1000000000.0
    max_seq_len: int = 131072
    activation: nn.Module = nn.GELU()
    depth_init: bool = True

    n_layers: int = 40
    n_heads: int = 32
    n_embd: int = 5120
    dim: int = 4096

    use_flex_attn: bool = False
    attn_mask_type: str = "block_causal_by_sequence_lengths"
    eos_id: int = 0

    def update_from_config(
        self, job_config: JobConfig, tokenizer: BaseTokenizer
    ) -> None:
        self.vocab_size = tokenizer.get_vocab_size()
        self.max_seq_len = job_config.training.seq_len
        self.eos_id = tokenizer.eos_id

        if job_config.activation_checkpoint.mode == "selective" and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with selective AC yet. "
                "See https://github.com/pytorch/pytorch/issues/147879"
            )

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with CP yet. "
                "We are still working on this."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

        return nparams, num_flops_per_token