# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import version

# Import to register quantization modules.
import torchtitan.components.quantization  # noqa: F401

try:
    __version__ = version("torchtitan")
except Exception as e:
    __version__ = "0.0.0+unknown"

# Distributed checkpoint shims for PyTorch 2.6.0 stability
import torch
import sys
import types

try:
    from torch.distributed.checkpoint.format_utils import HuggingFaceStorageReader
except ImportError:
    dist_checkpoint = types.ModuleType("torch.distributed.checkpoint")
    format_utils = types.ModuleType("torch.distributed.checkpoint.format_utils")
    
    class FakeReader:
        def __init__(self, *args, **kwargs): pass
    
    format_utils.HuggingFaceStorageReader = FakeReader
    dist_checkpoint.format_utils = format_utils
    sys.modules["torch.distributed.checkpoint.format_utils"] = format_utils
    if "torch.distributed.checkpoint" not in sys.modules:
        sys.modules["torch.distributed.checkpoint"] = dist_checkpoint
