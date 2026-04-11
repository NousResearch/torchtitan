# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.cuda
import torch.distributed.tensor

import math

import wandb
from torch import nn
from torch.autograd import Function

from torch.distributed.tensor import DTensor

from torch.distributed.tensor.experimental import register_sharding


IS_HISTC_SUPPORTED = None


def log_tensor_stats(tensor: torch.Tensor, name: str, num_bins: int = 64):  # noqa: C901
    global IS_HISTC_SUPPORTED
    """Add distribution statistics on a tensor's elements to the current History entry."""
    # TODO Handle the case of duplicate names.
    tensor = tensor.detach()
    flat = tensor.reshape(-1)

    if flat.is_cuda:
        if IS_HISTC_SUPPORTED is None:
            try:
                flat.histc(bins=num_bins)
            except RuntimeError:
                IS_HISTC_SUPPORTED = False
            else:
                IS_HISTC_SUPPORTED = True

        # As of torch 1.0.1.post2+nightly, float16 cuda summary ops are not supported (convert to float32)
        if not IS_HISTC_SUPPORTED:
            flat = flat.cpu()
        elif not isinstance(flat, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
            flat = flat.type(torch.cuda.FloatTensor)

    # Since we use histc, we need to make sure that torch supports the operation on CPU,
    # otherwise we'll get a runtime error. Hence, we need to upcast to float32.
    if not flat.is_cuda and not isinstance(
        flat, (torch.FloatTensor, torch.DoubleTensor)
    ):
        flat = flat.type(torch.FloatTensor)

    # Skip logging if all values are nan or inf or the tensor is empty.
    if flat.shape == torch.Size([0]) or (~torch.isfinite(flat)).all().item():
        return

    # Remove nans and infs if present. There's no good way to represent that in histograms.
    if not torch.isfinite(flat).all():
        flat = flat[torch.isfinite(flat)]

    tmin = flat.min().item()
    tmax = flat.max().item()
    # Handle precision errors where min/max might be inverted or equal
    if tmin > tmax:
        tmin, tmax = tmax, tmin
    if tmin == tmax:
        tensor = torch.Tensor([flat.numel()])
        tensor = tensor.cpu().clone().detach()
        bins = torch.Tensor([tmin, tmax])
    else:
        tensor = flat.histc(bins=num_bins, min=tmin, max=tmax)
        tensor = tensor.cpu().detach().clone()
        bins = torch.linspace(tmin, tmax, steps=num_bins + 1)

    wandb.run._log(
        {name: wandb.Histogram(np_histogram=(tensor.tolist(), bins.tolist()))},
        commit=False,
    )


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None, keepdim: bool = False, per_seq: bool = False) -> torch.Tensor:
    """Compute the mean of a tensor, with a mask applied."""
    if per_seq:
        # Special case: mean per sequence then average across sequences
        # tensor/mask assumed to be [B, S, ...]
        seq_sum = (tensor * mask).sum(dim=1)
        seq_count = mask.sum(dim=1).clamp(min=1.0)
        seq_mean = seq_sum / seq_count
        return seq_mean.mean()
    
    masked_tensor = tensor * mask
    total_sum = masked_tensor.sum(dim=dim, keepdim=keepdim)
    total_count = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1.0)
    return total_sum / total_count


def masked_sum(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None, keepdim: bool = False) -> torch.Tensor:
    """Compute the sum of a tensor, with a mask applied."""
    return (tensor * mask).sum(dim=dim, keepdim=keepdim)


def normalize_rewards_distributed(
    reward: torch.Tensor,
    mask: torch.Tensor,
    mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
    pg: Optional[torch.distributed.ProcessGroup] = None
) -> torch.Tensor:
    from torchtitan.distributed import utils as dist_utils
    # Local values
    local_sum = (reward * mask).sum()
    local_count = mask.sum()
    
    # Global values
    global_sum = dist_utils.dist_sum(local_sum, mesh, pg)
    global_count = dist_utils.dist_sum(local_count, mesh, pg)
    
    global_mean = global_sum / max(global_count, 1.0)
    
    # Variance
    local_var_sum = (((reward - global_mean) ** 2) * mask).sum()
    global_var_sum = dist_utils.dist_sum(local_var_sum, mesh, pg)
    
    global_var = global_var_sum / max(global_count, 1.0)
    global_std = max(math.sqrt(global_var), 1e-4)
    
    normalized_reward = (reward - global_mean) / global_std
    return normalized_reward * mask


@register_sharding(torch.ops.aten.amax.default)
def custom_amax_sharding(x, dim, keepdim):
    if isinstance(dim, list):
        if len(dim) == 1:
            dim = dim[0]
        else:
            raise ValueError(f"dim must be a single integer, got {dim}")
    amax_dim = dim if dim >= 0 else dim + x.ndim
    out_sharding = [torch.distributed.tensor.Partial(reduce_op="max"), None, None]
    in_sharding = [torch.distributed.tensor.Shard(amax_dim)]
    return [(out_sharding, in_sharding)]


@register_sharding(torch.ops.aten.amin.default)
def custom_amin_sharding(x, dim, keepdim):
    if isinstance(dim, list):
        if len(dim) == 1:
            dim = dim[0]
        else:
            raise ValueError(f"dim must be a single integer, got {dim}")
    amax_dim = dim if dim >= 0 else dim + x.ndim
    out_sharding = [torch.distributed.tensor.Partial(reduce_op="min"), None, None]
    in_sharding = [torch.distributed.tensor.Shard(amax_dim)]
    return [(out_sharding, in_sharding)]


def local_std(x: DTensor, dim: Optional[int] = None, keepdim: bool = False) -> DTensor:
    """
    Compute the local standard deviation of a tensor.
    """
    local_x = x.to_local()
    return DTensor.from_local(
        torch.std(local_x, dim=dim, keepdim=keepdim),
        device_mesh=x.device_mesh,
        placements=[torch.distributed.tensor.Partial(reduce_op="avg")],
    ).redistribute(
        device_mesh=x.device_mesh,
        placements=[torch.distributed.tensor.Replicate()],
    )


class VocabParallelEntropyFunction(Function):
    """
    Fused entropy loss computation with efficient backward pass.
    Saves only necessary tensors for gradient computation.
    """

    @staticmethod
    def forward(ctx, logits):
        """
        Forward pass computing entropy loss with mixed precision for stability.

        Args:
            logits: Local tensor [B*S, local_vocab]

        Returns:
            entropy_loss: Per-token entropy loss [B*S]
        """
        input_dtype = logits.dtype

        # Find global max for numerical stability
        if isinstance(logits, DTensor):
            logit_max = torch.amax(logits, dim=-1, keepdim=True).redistribute(
                device_mesh=logits.device_mesh,
                placements=[torch.distributed.tensor.Replicate()],
            )
        else:
            logit_max = torch.amax(logits, dim=-1, keepdim=True)

        # Compute stable softmax
        shifted_logits = logits - logit_max
        exp_logits = shifted_logits.exp()

        global_sum_exp = exp_logits.sum(dim=-1, keepdim=True)

        # Probabilities in original dtype
        probs = exp_logits / global_sum_exp

        # Compute log_probs in fp32
        log_probs = shifted_logits - global_sum_exp.log()

        # Entropy loss in fp32
        entropy_loss = torch.sum(probs * log_probs, dim=-1)

        # Save in original dtype to save memory
        ctx.save_for_backward(probs, log_probs)

        return entropy_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for entropy loss with mixed precision handling.
        """
        probs, log_probs = ctx.saved_tensors

        entropy_per_sample = torch.sum(probs * log_probs, dim=-1, keepdim=True)
        grad_logits = probs * (log_probs - entropy_per_sample)
        grad_logits = grad_logits * grad_output.unsqueeze(-1)

        return grad_logits


class VocabParallelEntropyLoss(nn.Module):
    """
    Entropy loss for vocabulary parallel outputs in TorchTitan.
    Handles distributed softmax computation with numerical stability.
    """

    def __init__(self, process_group=None):
        super().__init__()
        self.process_group = process_group

    def forward(self, logits: DTensor):
        """
        Compute entropy loss for vocabulary parallel logits.

        Args:
            logits: Local logits tensor [batch_size, seq_len, local_vocab_size]

        Returns:
            entropy_loss: Scalar entropy loss
        """
        batch_size, seq_len, local_vocab_size = logits.shape

        # Step 1: Find global max for numerical stability
        logit_max = torch.amax(logits, dim=-1, keepdim=True).redistribute(
            device_mesh=logits.device_mesh,
            placements=[torch.distributed.tensor.Replicate()],
        )

        # Step 2: Compute shifted logits and global sum of exponentials
        shifted_logits = logits - logit_max
        exp_logits = shifted_logits.exp()
        global_sum_exp = exp_logits.sum(dim=-1, keepdim=True)

        # Step 3: Compute probabilities and log-probabilities
        probs = exp_logits / global_sum_exp
        logp = shifted_logits - global_sum_exp.log()

        # Step 4: Compute entropy contribution: p * log(p)
        entropy = torch.sum(probs * logp, dim=-1)
        return entropy
