import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.nn import init


class Lora(nn.Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b + ((xB^T)C^T)`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        r: rank of the low-rank approximation
        lora_alpha: alpha parameter for the lora scaling
        lora_dropout: dropout probability for the lora scaling

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
        lora_a: learnable matrix of shape :math:`(\text{in\_features}, \text{r})`
        lora_b: learnable matrix of shape :math:`(\text{r}, \text{out\_features})`
        scaling: scaling factor for the lora scaling :math:`\text{lora\_alpha} / \text{r}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 8,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.weight.requires_grad = False
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias.requires_grad = False
        else:
            self.register_parameter("bias", None)
        self.lora_a = nn.Parameter(torch.empty((r, in_features), **factory_kwargs))
        self.lora_b = nn.Parameter(torch.empty((out_features, r), **factory_kwargs))
        self.scaling = lora_alpha / r
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        init.zeros_(self.lora_b)

    def forward(self, input: Tensor) -> Tensor:
        return (
            F.linear(input, self.weight, self.bias)
            + F.linear(F.linear(input, self.lora_a), self.lora_b) * self.scaling
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"r={self.r}, "
            f"lora_alpha={self.lora_alpha}, "
            f"lora_dropout={self.lora_dropout}"
        )


class LoraColwiseParallel(ColwiseParallel):
    def _partition_lora_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    module.weight,
                    device_mesh,
                    [Shard(0)],
                    src_data_rank=self.src_data_rank,
                )
            ),
        )
        if getattr(module, "bias", None) is not None:
            module.register_parameter(
                "bias",
                nn.Parameter(
                    distribute_tensor(
                        module.bias,
                        device_mesh,
                        [Shard(0)],
                        src_data_rank=self.src_data_rank,
                    )
                ),
            )
        # lora_a should not be sharded, as it gets used immediately by lora_b to compute the output
        # and is not sharded on the input dimension
        module.register_parameter(
            "lora_a",
            nn.Parameter(
                distribute_tensor(
                    module.lora_a,
                    device_mesh,
                    [Replicate()],
                    src_data_rank=self.src_data_rank,
                )
            ),
        )
        module.register_parameter(
            "lora_b",
            nn.Parameter(
                distribute_tensor(
                    module.lora_b,
                    device_mesh,
                    [Shard(0)],
                    src_data_rank=self.src_data_rank,
                )
            ),
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        elif isinstance(module, Lora):
            partition_fn = self._partition_lora_fn
        else:
            raise NotImplementedError(
                "ColwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


class LoraRowwiseParallel(RowwiseParallel):
    def _partition_lora_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    module.weight,
                    device_mesh,
                    [Shard(1)],
                    src_data_rank=self.src_data_rank,
                )
            ),
        )
        if getattr(module, "bias", None) is not None:
            # The Linear module has bias
            module.register_parameter(
                "bias",
                nn.Parameter(
                    distribute_tensor(
                        module.bias,
                        device_mesh,
                        [Replicate()],
                        src_data_rank=self.src_data_rank,
                    )
                ),
            )
        # lora_a should be sharded as it's operating on the input dimension
        # lora_b, however, is operating on the output dimension and should not be sharded
        module.register_parameter(
            "lora_a",
            nn.Parameter(
                distribute_tensor(
                    module.lora_a,
                    device_mesh,
                    [Shard(1)],
                    src_data_rank=self.src_data_rank,
                )
            ),
        )
        module.register_parameter(
            "lora_b",
            nn.Parameter(
                distribute_tensor(
                    module.lora_b,
                    device_mesh,
                    [Replicate()],
                    src_data_rank=self.src_data_rank,
                )
            ),
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, Lora):
            partition_fn = self._partition_linear_fn
            # rowwise linear runtime sharding requires input tensor shard on last dim
            self.desired_input_layouts: tuple[Placement, ...] = (Shard(-1),)
        elif isinstance(module, Lora):
            partition_fn = self._partition_lora_fn
            self.desired_input_layouts: tuple[Placement, ...] = (Shard(1),)
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
            # rowwise embedding runtime sharding requires input tensor replicated
            self.desired_input_layouts = (Replicate(),)
        else:
            raise NotImplementedError(
                "RowwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )
