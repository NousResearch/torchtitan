# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, TypeAlias

import torch

from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def chunked_cross_entropy_loss(
    logits: torch.Tensor, labels: torch.Tensor, num_output_chunks: int = 8
) -> torch.Tensor:
    # Adapted from torchtune
    # https://github.com/pytorch/torchtune/blob/c3703482bde72e572b535d3f7c43c81e94164ebc/torchtune/modules/loss/ce_chunked_output_loss.py

    labels = [target_chunk for target_chunk in labels.chunk(num_output_chunks, dim=1)]
    logits = [logit_chunk for logit_chunk in logits.chunk(num_output_chunks, dim=1)]

    # compute one chunk at a time
    total_loss = 0.0
    for logits_chunk, labels_chunk in zip(logits, labels):
        total_loss += cross_entropy_loss(logits_chunk, labels_chunk)

        return total_loss / num_output_chunks


def build_cross_entropy_loss(job_config: JobConfig):
    loss_fn = (
        chunked_cross_entropy_loss
        if job_config.training.chunked_loss
        else cross_entropy_loss
    )
    if job_config.training.compile:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """

    @functools.wraps(unwrapped_loss_fn)
    def accumulated_loss_fn(*args, **kwargs):
        loss = unwrapped_loss_fn(*args, **kwargs)
        return loss / accumulation_steps

    return accumulated_loss_fn
