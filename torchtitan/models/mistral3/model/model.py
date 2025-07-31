# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

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

from .args import VLMArgs

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    #logger.info(freqs_cis.shape)
    #logger.info(x.shape)
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.
        position_ids (torch.Tensor, optional): Custom position IDs of shape [batch_size, seq_len].
                                              If provided, will use these to index into freqs_cis.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    if position_ids is not None:
        gathered_freqs = freqs_cis[position_ids]  # [bs, seqlen, head_dim/2]
        gathered_freqs = gathered_freqs.unsqueeze(2)  # [bs, seqlen, 1, head_dim/2]

        xq_out = torch.view_as_real(xq_ * gathered_freqs).flatten(3)
        xk_out = torch.view_as_real(xk_ * gathered_freqs).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)
    else:
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, num_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=num_rep)"""
    bsz, seq_len, num_kv_heads, head_dim = x.shape
    if num_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bsz, seq_len, num_kv_heads, num_rep, head_dim)
        .reshape(bsz, seq_len, num_kv_heads * num_rep, head_dim)
    )


class Attention(nn.Module):

    def __init__(self, config: VLMArgs, is_vision=True):
        super().__init__()
        if is_vision:
            self.num_heads = config.vision_num_heads
            self.num_kv_heads = config.vision_num_heads
            self.head_dim = config.vision_embed_dim // config.vision_num_heads
            self.embed_dim = config.vision_embed_dim
            self.is_causal = False
        else:
            self.num_heads = config.decoder_num_heads
            self.num_kv_heads = (
                config.decoder_num_heads if config.decoder_num_kv_heads is None else config.decoder_num_kv_heads
            )
            self.head_dim = config.decoder_embed_dim // config.decoder_num_heads
            self.embed_dim = config.decoder_embed_dim
            self.is_causal = True
            
        self.num_rep = self.num_heads // self.num_kv_heads


        self.wq = nn.Linear(self.embed_dim, int(self.num_heads * self.head_dim * 0.8), bias=False)
        self.wk = nn.Linear(self.embed_dim, int(self.num_kv_heads * self.head_dim * 0.8), bias=False)
        self.wv = nn.Linear(self.embed_dim, int(self.num_kv_heads * self.head_dim * 0.8), bias=False)
        self.wo = nn.Linear(int(self.num_heads * self.head_dim * 0.8), self.embed_dim, bias=False)

        self.sdpa = build_attention(True, config.attn_mask_type)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor, optional): Precomputed frequency tensor.
            position_ids (torch.Tensor, optional): Custom position ids tensor of shape [batch, seq_len].

        Returns:
            torch.Tensor: Output tensor after attention.
        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `num_heads` (or `num_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, 128)
        xk = xk.view(bs, seqlen, -1, 128)
        xv = xv.view(bs, seqlen, -1, 128)

        if freqs_cis is not None:
            # Apply RoPE with position_ids if provided
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, position_ids=position_ids)


        # repeat k/v heads if num_kv_heads < num_heads
        keys = repeat_kv(xk, self.num_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.num_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        #output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv)
        output = self.sdpa(xq, xk, xv)

        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module for the decoder. It's different from the one in the encoder.
    This is the component which is originally used in Mistral3/Llama3.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        hidden_dim = 32768

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: VLMArgs,
    ):
        super().__init__()
        self.attn = Attention(config, is_vision=False)
        #self.ln_attn = build_norm("rmsnorm", config.decoder_embed_dim, config.norm_eps)
        self.ln_attn = nn.RMSNorm(config.decoder_embed_dim, config.norm_eps)
        self.mlp = FeedForward(
            dim=config.decoder_embed_dim,
            hidden_dim=4 * config.decoder_embed_dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        #self.ln_mlp = build_norm("rmsnorm", config.decoder_embed_dim, config.norm_eps)
        self.ln_mlp = nn.RMSNorm(config.decoder_embed_dim, config.norm_eps)

    def init_weights(self):
        """
        Initialize weights following the Llama3 pattern.
        """
        # Initialize attention and feedforward components
        self.attn.init_weights(0.02)  # Use standard init_std for attention
        self.mlp.init_weights(0.02)   # Use standard init_std for feedforward
        
        # Initialize norm layers
        for norm in (self.ln_attn, self.ln_mlp):
            norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ):
        # Handle custom position_ids if provided
        if position_ids is not None:
            # Custom handling for position_ids
            # We need to index into freqs_cis with the position_ids
            # First, we'll do a custom reshape_for_broadcast implementation that uses position_ids
            x_norm = self.ln_attn(x)
            # Get the appropriate freqs_cis based on position_ids
            x = x + self.attn(x_norm, freqs_cis, position_ids=position_ids)
        else:
            # Standard forwarding without custom position_ids
            x = x + self.attn(self.ln_attn(x), freqs_cis)
            
        x = x + self.mlp(self.ln_mlp(x))
        return x


class VisionEncoder(nn.Module):
    """Vision encoder model using Pixtral Vision for Mistral3. This integrates the Pixtral vision encoder
    with a projection to connect to the multimodal decoder.

    Args:
        config (VLMArgs): configs for the vision encoder.
    """

    def __init__(self, config: VLMArgs) -> None:
        super().__init__()
        
        # Import Pixtral components here to avoid circular imports
        from .modeling_pixtral import PixtralVisionModel, PixtralVisionConfig
        
        # Create a PixtralVisionConfig based on the ModelArgs 
        pixtral_config = PixtralVisionConfig(
            hidden_size=config.vision_embed_dim,
            intermediate_size=4 * config.vision_embed_dim,  # Standard multiplier
            num_hidden_layers=config.vision_num_layers,
            num_attention_heads=config.vision_num_heads,
            num_channels=config.in_channels,
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_act="gelu",  # Standard activation
            attention_dropout=0.0,  # No dropout by default
            rope_theta=config.rope_theta,
            initializer_range=0.02  # Standard initialization
        )
        
        # Initialize the Pixtral vision model
        self.pixtral_vision = PixtralVisionModel(pixtral_config)
        
        # Add projection to connect to the decoder
        #self.multi_modal_projector = Mistral3MultiModalProjector(config)

    def init_weights(self):
        """
        Initialize weights for the vision encoder components.
        """
        # Initialize pixtral vision model if it has init_weights
        if hasattr(self.pixtral_vision, 'init_weights'):
            self.pixtral_vision.init_weights()
        
        # Initialize multimodal projector if it has init_weights
        if hasattr(self.multi_modal_projector, 'init_weights'):
            self.multi_modal_projector.init_weights()

    def forward(self, pixel_values: torch.Tensor, image_sizes: torch.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> torch.Tensor:

        # Pass through Pixtral vision model
        vision_outputs = self.pixtral_vision(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return vision_outputs

        
        # Get the last hidden state
        #image_features = vision_outputs.last_hidden_state
        
        # Project to decoder dimension
        #return self.multi_modal_projector(image_features, image_sizes)

class Transformer(nn.Module):
    """Decoder multimodal model for Mistral3.

    Args:
        config (VLMArgs): configs for the model.
    """

    def __init__(self, config: VLMArgs):
        super().__init__()

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(config), persistent=False
        )

        self.layers = nn.ModuleDict()
        for idx in range(config.decoder_num_layers):
            # define a llama3-like decoder layer
            decoder_layer = TransformerBlock(config)
            self.layers[str(idx)] = decoder_layer

        self.tok_embeddings = nn.Embedding(131072, config.decoder_embed_dim)
        #self.norm = build_norm(
        #    config.norm_type, dim=config.decoder_embed_dim, eps=config.norm_eps
        #self.norm=nn.LayerNorm(config.decoder_embed_dim, eps=config.norm_eps)
        self.norm = nn.RMSNorm(config.decoder_embed_dim, eps=config.norm_eps)
        self.output = nn.Linear(
            config.decoder_embed_dim, 131072, bias=False
        )

    def init_weights(self):
        """
        Initialize weights following the Llama3 pattern.
        """
        # Initialize token embeddings
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        
        # Initialize all layers
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        
        # Initialize norm layer
        if self.norm is not None:
            self.norm.reset_parameters()
        
        # Initialize output layer with truncated normal
        if self.output is not None:
            final_out_std = self.output.in_features**-0.5
            cutoff_factor = 3
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_freqs_cis(self, config) -> torch.Tensor:
        return precompute_freqs_cis(
            int(config.decoder_embed_dim // config.decoder_num_heads * 0.8),
            # Need to compute until at least the max token limit for generation
            # (use 2x max sequence length to be safe)
            config.max_seq_len,
            config.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        if inputs_embeds is None:
            h = self.tok_embeddings(tokens)
        else:
            h = inputs_embeds

        # Setup freqs_cis based on position_ids or sequence length
        if position_ids is not None:
            # Use custom position_ids to index into freqs_cis
            # We still need freqs_cis with the right device/dtype
            freqs_cis = self.freqs_cis
        else:
            # Default: use standard positions based on sequence length
            freqs_cis = self.freqs_cis

        for layer in self.layers.values():
            # shape: [b, s, d]
            h = layer(
                h,
                freqs_cis=freqs_cis,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                position_ids=position_ids,
            )

        # shape: [b, s, d]
        h = self.norm(h)
        output = self.output(h)

        return output


class VLM(nn.Module, ModelProtocol):
    """
    Mistral3 model which consists of a vision backbone and a language model.
    
    Args:
        config (VLMArgs): Configuration for the model.
    """
    
    def __init__(self, config: VLMArgs):
        super().__init__()
        self.config = config

        # Language model decoder
        self.language_model = Transformer(config)
        
        # Special token for representing images in the text
        self.image_token_index = config.image_token_index

        self.vision_model_initialized = False


    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):

        buffer_device = buffer_device or self.language_model.freqs_cis.device
        with torch.device(buffer_device):
            self.language_model._precompute_freqs_cis(self.config)
        
        # Initialize language model components
        if hasattr(self.language_model, 'init_weights'):
            self.language_model.init_weights()
        
        ## Initialize vision tower if it exists
        if hasattr(self, 'vision_tower') and self.vision_tower is not None:
            if hasattr(self.vision_tower, 'init_weights'):
                self.vision_tower.init_weights()
        
        ## Initialize multimodal projector if it exists
        if hasattr(self, 'multi_modal_projector') and self.multi_modal_projector is not None:
            if hasattr(self.multi_modal_projector, 'init_weights'):
                self.multi_modal_projector.init_weights()
        
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        image_sizes: torch.Tensor,
        **kwargs,
    ):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        with torch.no_grad():
            image_outputs = self.vision_tower(pixel_values, image_sizes=image_sizes, output_hidden_states=False)
            # If we have one vision feature layer, return the corresponding hidden states,
            # otherwise, select the hidden states of each feature layer and concatenate them
            if isinstance(vision_feature_layer, int):
                selected_image_feature = image_outputs.last_hidden_state #[vision_feature_layer]
            else:
                hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
                selected_image_feature = torch.cat(hs_pool, dim=-1)

            image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
            return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        sequence_lengths: list[torch.Tensor] | None = None,
    ):

        if self.config.use_flex_attn:
            init_attention_mask(input_ids, eos_id=self.config.eos_id, sequence_lengths=sequence_lengths)

        if position_ids is not None:
            # for the case where we want to do sequence packing, we need to pass the nonstandard position_ids
            logits = self.language_model(
                        tokens=input_ids,
                        encoder_mask=None,
                        position_ids=position_ids
                    )
        else:
            logits = self.language_model(
                        tokens=input_ids,
                        encoder_mask=None,
                    )
                
        return logits

    @classmethod
    def from_model_args(cls, model_args: VLMArgs) -> "Transformer":
        return cls(model_args)
