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
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformers.image_utils import load_image

from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.models.attention import build_attention, init_attention_mask

from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config_manager import JobConfig

from .args import VLMArgs


def build_norm(norm_type: str, dim: int, eps: float = 1e-6, device: torch.device = None):
    """
    Builds the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to build.
            Supported types: layernorm, np_layernorm, rmsnorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The built normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps, device=device)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

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


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: VLMArgs):
        super().__init__()
        self.config = config

        hidden_size = config.vision_embed_dim
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.merging_layer = nn.Linear(hidden_size * self.spatial_merge_size**2, hidden_size, bias=False)
        
    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid, kernel_size=self.spatial_merge_size, stride=self.spatial_merge_size
            )
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)

        return image_features.unsqueeze(0)
    


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: VLMArgs):
        super().__init__()
        self.norm = nn.RMSNorm(config.vision_embed_dim, eps=config.norm_eps, device=torch.cuda.current_device())
        self.patch_merger = Mistral3PatchMerger(config)
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_embed_dim * num_feature_layers,
            config.decoder_embed_dim,
            bias=config.multimodal_projector_bias,
        )
        self.act = nn.GELU()#config.projector_hidden_act #activation
        self.linear_2 = nn.Linear(
            config.decoder_embed_dim, config.decoder_embed_dim, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor):
        image_features = self.norm(image_features)

        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states



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
        #output = self.sdpa(xq, xk, xv)
        output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, is_causal=True)

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
        self.ln_attn = nn.RMSNorm(config.decoder_embed_dim, config.norm_eps, device=torch.cuda.current_device())
        self.mlp = FeedForward(
            dim=config.decoder_embed_dim,
            hidden_dim=4 * config.decoder_embed_dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        #self.ln_mlp = build_norm("rmsnorm", config.decoder_embed_dim, config.norm_eps)
        self.ln_mlp = nn.RMSNorm(config.decoder_embed_dim, config.norm_eps, device=torch.cuda.current_device())

        self.image_token_id = config.image_token_id

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

class Transformer(nn.Module):
    """Decoder multimodal model for Mistral3.

    Args:
        config (VLMArgs): configs for the model.
    """

    def __init__(self, config: VLMArgs):
        super().__init__()

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(config), persistent=True
        )

        self.layers = nn.ModuleDict()
        for idx in range(config.decoder_num_layers):
            # define a llama3-like decoder layer
            decoder_layer = TransformerBlock(config)
            self.layers[str(idx)] = decoder_layer

        self.tok_embeddings = nn.Embedding(131072, config.decoder_embed_dim)
        self.norm = nn.RMSNorm(config.decoder_embed_dim, eps=config.norm_eps, device=torch.cuda.current_device())
        self.output = nn.Linear(
            config.decoder_embed_dim, 131072, bias=False
        )

        self.image_token_id = config.image_token_id

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
    
    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ) -> torch.BoolTensor:
        """
        Obtains multimodal placeholdr mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_features: Optional[list] = None,
    ) -> torch.Tensor:

        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        if inputs_embeds is None:
            h = self.tok_embeddings(tokens)
        else:
            h = inputs_embeds
        

        #print(h)


        if image_features is not None:

            if isinstance(h, DTensor):
                h_full = h.redistribute(h.device_mesh, [Replicate()]).to_local()
                new_h_full = h_full.clone()  # Create a copy to modify

                for i, i_image_features in enumerate(image_features):
                    if i_image_features is not None:

                        image_feat = i_image_features
                        #image_feat = i_image_features.unsqueeze(0)
                        print(tokens.shape)
                        print(h_full[i].shape)
                        print(image_feat.shape)

 
                        special_image_mask = self.get_placeholder_mask(
                            tokens[i].unsqueeze(0), h_full[i].unsqueeze(0), image_feat
                        )
                        # Use torch.where instead of masked_scatter
                        new_h_full[i] = h_full[i].masked_scatter(special_image_mask, image_feat)
             
                
                # Convert back to DTensor
                new_h_full = new_h_full.to(h.device)
                #h_replicated = distribute_tensor(new_h_full, h.device_mesh, [Replicate()])
                h_replicated = DTensor.from_local(new_h_full, h.device_mesh, placements=[Replicate()])

                h = h_replicated.redistribute(h.device_mesh, [Shard(1)])
            else:
                for i, i_image_features in enumerate(image_features):
                    if i_image_features is not None:

                        #image_features = i_image_features
                        image_features = i_image_features.unsqueeze(0)
                        special_image_mask = self.get_placeholder_mask(
                            input_ids=tokens[i].unsqueeze(0), inputs_embeds=h[i].unsqueeze(0), image_features=image_features
                        )

                        h[i] = h[i].masked_scatter(special_image_mask, image_features.to(h[i].device, dtype=h[i].dtype))


            #print(h)

        if image_features is None:
            print("image features is None")
        

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

        from .modeling_pixtral import PixtralVisionModel, PixtralVisionConfig

        # Create a PixtralVisionConfig based on the ModelArgs 
        pixtral_config = PixtralVisionConfig(
            hidden_size=config.vision_embed_dim,
            intermediate_size=4 * config.vision_embed_dim,  # Standard multiplier
            num_hidden_layers=config.vision_num_layers,
            num_attention_heads=config.vision_num_heads,
            num_channels=3,
            image_size=1540,
            patch_size=14,
            hidden_act="silu",  # Standard activation
            attention_dropout=0.0,  # No dropout by default
            rope_theta=10000.0,
            initializer_range=0.02  # Standard initialization
        )

        self.vision_tower = PixtralVisionModel(pixtral_config)

        # Add projection to connect to the decoder
        self.multi_modal_projector = Mistral3MultiModalProjector(config)

        from transformers import AutoProcessor
        self.preprocessor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503", use_fast=True)

        self.initialized_vision=False

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):

        buffer_device = buffer_device or self.language_model.freqs_cis.device
        with torch.device(buffer_device):
            self.language_model._precompute_freqs_cis(self.config)



    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, list[int]]`, *optional*):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            image_sizes (`torch.Tensor`, *optional*):
                Tensor containing the image sizes as returned by the processor.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        # Debug prints for inputs

        
        # Debug print for config var before assignment
        #print("Config - self.config.vision_feature_layer:", self.config.vision_feature_layer)
        
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        
        # Debug print for vision_feature_layer after assignment
        #print("Assigned - vision_feature_layer:", vision_feature_layer)
        
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(pixel_values, image_sizes=image_sizes, output_hidden_states=True, **kwargs)
        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)
        
        #print("selected_image_feature", selected_image_feature.shape)
        #print("image_outputs.hidden_states", selected_image_feature)



        image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)

        #print("image_features", image_features.shape)
        #print("image_features", image_features)
        # Debug print for config var
        #print("Config - self.config.spatial_merge_size:", self.config.spatial_merge_size)
        #print("Config - self.vision_tower.patch_size:", self.vision_tower.patch_size)

        
        #downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
        #split_sizes = [(height // downsample_ratio) * (width // downsample_ratio) for height, width in image_sizes]
        #image_features = torch.split(image_features.squeeze(0), split_sizes)
        return image_features

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        sequence_lengths: list[torch.Tensor] | None = None,
        image_features: Optional[torch.FloatTensor] = None,
        images: Optional[list] = None
    ):

        if not self.initialized_vision:
            from transformers import AutoModelForImageTextToText
            hf_model = AutoModelForImageTextToText.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503", device_map='cpu', torch_dtype=torch.bfloat16)

            vision_tower_device = self.vision_tower.device


            hf_model.vision_tower = self.vision_tower.to(vision_tower_device, dtype=torch.bfloat16)
            hf_model.multi_modal_projector = self.multi_modal_projector.to(vision_tower_device, dtype=torch.bfloat16)

            self.vision_tower = hf_model.vision_tower
            self.multi_modal_projector = hf_model.multi_modal_projector

            print("did the thing")
            self.initialized_vision=True


        #image_features = None
        all_image_features = []
        #image_features = None

        if image_features is not None:
            all_image_features = image_features
        else:

            for i, batch in enumerate(images):
                i_image_features = None

                if batch is not None:
                    i_image_features = None
                    image_features_batch = []
                    images = [load_image(im) if isinstance(im, str) else im for im in batch]

                    image_inputs = self.preprocessor.image_processor(images, patch_size=self.config.patch_size * 2)

                    #print("my model")
                    #print(self.vision_tower.config)
                    #print(self.vision_tower.patch_conv.weight)
                    #print(self.multi_modal_projector.linear_1.weight)
                    #exit(0)


                    #image_encoder_outputs = self.get_image_features(image_inputs["pixel_values"].to(self.vision_tower.device, dtype=torch.float16), 2, image_inputs["image_sizes"])
                    image_encoder_outputs = self.get_image_features(pixel_values=image_inputs["pixel_values"].to(self.vision_tower.device, dtype=self.vision_tower.dtype), image_sizes=image_inputs["image_sizes"], vision_feature_layer=-1)[0].unsqueeze(0)

                    # Collect image features from all images in the batch
                    i_image_features = image_encoder_outputs


                    all_image_features.append(i_image_features)
        

        #else:
        #    return NotImplementedError("Position IDs are required for multimodal input.")

        if self.config.use_flex_attn:
            init_attention_mask(input_ids, eos_id=self.config.eos_id, sequence_lengths=sequence_lengths)

        if position_ids is not None:
            if all_image_features:
                logits = self.language_model(
                        tokens=input_ids,
                        encoder_mask=None,
                        position_ids=position_ids, 
                        image_features=all_image_features,
                    )
            else:
                logits = self.language_model(
                        tokens=input_ids,
                        encoder_mask=None,
                        position_ids=position_ids, 
                    )
        else:
            if all_image_features:
                logits = self.language_model(
                            tokens=input_ids,
                            encoder_mask=None,
                            image_features=all_image_features,
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
