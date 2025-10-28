# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import einops as E
import torch
from torch import nn

from torchtitan.models.llama3 import Transformer as Llama3

from ..datasets.mm_datasets import SpecialTokens

from .args import Qwen3VLModelArgs
from .qwenvl_encoder import QwenVisionTransformer
from ...qwen3.model.model import Qwen3Model

class Qwen3VLTransformer(Qwen3Model):

    def __init__(self, model_args: Qwen3VLModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.encoder = QwenVisionTransformer(model_args.encoder)


    def init_weights(self, buffer_device=None):
        super().init_weights(buffer_device=buffer_device)
        if self.encoder is not None:
            self.encoder.init_weights()
        #if self.projector is not None:
        #    self.projector.init_weights()

    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        special_tokens: SpecialTokens,
        input_batch: torch.Tensor | None = None,
    ):

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h_BSD = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        deepstack_visual_embeds = None
        visual_pos_masks = None

        if self.encoder is not None:
            #grid_hw = grid_thw[:, :, 1:]  # Siglip2 only support image hw
            #pixel_masks = E.reduce(grid_hw != -1, "n l hw -> n l", reduction="all")
            i_NLD, deepstack_feature_lists = self.encoder(pixel_values, None, grid_thw)

            deepstack_visual_embeds = deepstack_feature_lists


            pixel_masks = E.repeat(tokens == 1998, "b s -> b s 1")

            visual_pos_masks = pixel_masks
            #i_NLD = self.projector(i_NLD)
            h_BSD = h_BSD.masked_scatter(mask=pixel_masks, source=i_NLD.float())
            #h_BSD = _scatter_img_tokens(
            #    h_BSD, tokens, i_NLD, pixel_masks, special_tokens.img_id
            #)

        for layer_idx, layer in enumerate(self.layers.values()):
            h_BSD = layer(h_BSD, self.rope_cache)


            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                h_BSD = self._deepstack_process(
                    h_BSD,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        h_BSD = self.norm(h_BSD) if self.norm else h_BSD
        output = self.output(h_BSD) if self.output else h_BSD
        return output