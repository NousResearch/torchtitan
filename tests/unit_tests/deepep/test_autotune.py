# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for DeepEP autotuning.

Tests _create_uniform_routing, _State config management, _bench_fn,
_detect_internode, _get_gpu_sm_range, and run_deepep_autotune_if_enabled.

Run:
    python -m pytest tests/unit_tests/deepep/test_autotune.py -v
"""

import sys
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import torch

# Mock deep_ep module if not available
DEEP_EP_AVAILABLE = False
try:
    import deep_ep  # noqa: F401

    DEEP_EP_AVAILABLE = True
except ImportError:
    pass

if not DEEP_EP_AVAILABLE:
    mock_deep_ep = MagicMock()
    mock_deep_ep.Config = MagicMock
    mock_deep_ep.Buffer = MagicMock
    mock_deep_ep_utils = MagicMock()
    mock_deep_ep_utils.EventOverlap = MagicMock
    mock_deep_ep_utils.EventHandle = MagicMock
    sys.modules["deep_ep"] = mock_deep_ep
    sys.modules["deep_ep.utils"] = mock_deep_ep_utils

from torchtitan.distributed.deepep.deepep import (
    _bench_fn,
    _create_uniform_routing,
    _detect_internode,
    _get_gpu_sm_range,
    _State,
    _state,
    get_tuned_configs,
    run_deepep_autotune_if_enabled,
    set_tuned_configs,
)


class TestCreateUniformRouting(unittest.TestCase):
    """Tests for _create_uniform_routing."""

    def test_output_shapes(self):
        num_tokens, hidden, num_experts, num_topk = 128, 256, 64, 8
        x, topk_idx, topk_weights = _create_uniform_routing(
            num_tokens, hidden, num_experts, num_topk
        )
        self.assertEqual(x.shape, (num_tokens, hidden))
        self.assertEqual(topk_idx.shape, (num_tokens, num_topk))
        self.assertEqual(topk_weights.shape, (num_tokens, num_topk))

    def test_output_dtypes(self):
        x, topk_idx, topk_weights = _create_uniform_routing(32, 64, 8, 2)
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertEqual(topk_idx.dtype, torch.int64)
        self.assertEqual(topk_weights.dtype, torch.float32)

    def test_output_device(self):
        x, topk_idx, topk_weights = _create_uniform_routing(32, 64, 8, 2)
        self.assertEqual(x.device.type, "cuda")
        self.assertEqual(topk_idx.device.type, "cuda")
        self.assertEqual(topk_weights.device.type, "cuda")

    def test_uniform_weights(self):
        """Weights should be 1/topk for all entries."""
        num_topk = 4
        _, _, topk_weights = _create_uniform_routing(64, 32, 16, num_topk)
        expected = 1.0 / num_topk
        self.assertTrue(torch.allclose(topk_weights, torch.full_like(topk_weights, expected)))

    def test_round_robin_expert_assignment(self):
        """Each token's experts should be assigned round-robin."""
        num_tokens, num_experts, num_topk = 16, 8, 2
        _, topk_idx, _ = _create_uniform_routing(num_tokens, 32, num_experts, num_topk)
        topk_idx_cpu = topk_idx.cpu()
        for i in range(num_tokens):
            for k in range(num_topk):
                expected = (i * num_topk + k) % num_experts
                self.assertEqual(topk_idx_cpu[i, k].item(), expected)

    def test_expert_indices_in_range(self):
        """All expert indices should be in [0, num_experts)."""
        num_experts = 32
        _, topk_idx, _ = _create_uniform_routing(256, 64, num_experts, 4)
        self.assertTrue((topk_idx >= 0).all())
        self.assertTrue((topk_idx < num_experts).all())

    def test_balanced_load(self):
        """With round-robin, each expert should get equal tokens."""
        num_tokens, num_experts, num_topk = 128, 8, 2
        _, topk_idx, _ = _create_uniform_routing(num_tokens, 32, num_experts, num_topk)
        # Count assignments per expert
        counts = torch.zeros(num_experts, dtype=torch.int64)
        for e in range(num_experts):
            counts[e] = (topk_idx == e).sum().item()
        expected_per_expert = num_tokens * num_topk // num_experts
        for e in range(num_experts):
            self.assertEqual(counts[e].item(), expected_per_expert)


class TestStateConfigManagement(unittest.TestCase):
    """Tests for _State config getter/setter."""

    def setUp(self):
        # Reset state before each test
        _state.tuned_dispatch_config = None
        _state.tuned_combine_config = None

    def tearDown(self):
        _state.tuned_dispatch_config = None
        _state.tuned_combine_config = None

    def test_initial_configs_are_none(self):
        d, c = get_tuned_configs()
        self.assertIsNone(d)
        self.assertIsNone(c)

    def test_set_and_get_configs(self):
        mock_dispatch = MagicMock()
        mock_combine = MagicMock()
        set_tuned_configs(mock_dispatch, mock_combine)
        d, c = get_tuned_configs()
        self.assertIs(d, mock_dispatch)
        self.assertIs(c, mock_combine)

    def test_set_partial_configs(self):
        mock_dispatch = MagicMock()
        set_tuned_configs(dispatch_config=mock_dispatch)
        d, c = get_tuned_configs()
        self.assertIs(d, mock_dispatch)
        self.assertIsNone(c)

    def test_overwrite_configs(self):
        set_tuned_configs(MagicMock(), MagicMock())
        new_dispatch = MagicMock()
        new_combine = MagicMock()
        set_tuned_configs(new_dispatch, new_combine)
        d, c = get_tuned_configs()
        self.assertIs(d, new_dispatch)
        self.assertIs(c, new_combine)


class TestBenchFn(unittest.TestCase):
    """Tests for _bench_fn."""

    def test_returns_positive_time(self):
        counter = [0]
        def fn():
            counter[0] += 1
        with patch("torch.cuda.synchronize"):
            t = _bench_fn(fn, warmup=1, repeat=2)
        self.assertGreater(t, 0)
        # warmup=1 + repeat=2 = 3 calls
        self.assertEqual(counter[0], 3)

    def test_warmup_and_repeat_counts(self):
        counter = [0]
        def fn():
            counter[0] += 1
        with patch("torch.cuda.synchronize"):
            _bench_fn(fn, warmup=5, repeat=10)
        self.assertEqual(counter[0], 15)

    def test_exception_propagates(self):
        def bad_fn():
            raise RuntimeError("CUDA error")
        with patch("torch.cuda.synchronize"):
            with self.assertRaises(RuntimeError):
                _bench_fn(bad_fn, warmup=1, repeat=1)


class TestDetectInternode(unittest.TestCase):
    """Tests for _detect_internode."""

    def test_intranode_single_rdma(self):
        mock_buffer = MagicMock()
        mock_buffer.group_size = 8
        mock_buffer.runtime.get_num_rdma_ranks.return_value = 1
        with patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "8"}):
            is_inter, local_ranks, num_nodes = _detect_internode(mock_buffer)
        self.assertFalse(is_inter)
        self.assertEqual(local_ranks, 8)
        self.assertEqual(num_nodes, 1)

    def test_internode_multi_rdma(self):
        mock_buffer = MagicMock()
        mock_buffer.group_size = 16
        mock_buffer.runtime.get_num_rdma_ranks.return_value = 2
        is_inter, local_ranks, num_nodes = _detect_internode(mock_buffer)
        self.assertTrue(is_inter)
        self.assertEqual(local_ranks, 8)
        self.assertEqual(num_nodes, 2)

    def test_internode_four_nodes(self):
        mock_buffer = MagicMock()
        mock_buffer.group_size = 32
        mock_buffer.runtime.get_num_rdma_ranks.return_value = 4
        is_inter, local_ranks, num_nodes = _detect_internode(mock_buffer)
        self.assertTrue(is_inter)
        self.assertEqual(local_ranks, 8)
        self.assertEqual(num_nodes, 4)


class TestGetGpuSmRange(unittest.TestCase):
    """Tests for _get_gpu_sm_range."""

    def test_b200(self):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA B200"):
            result = _get_gpu_sm_range()
        self.assertEqual(result, [24, 32, 48, 64])

    def test_h100(self):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA H100 SXM"):
            result = _get_gpu_sm_range()
        self.assertEqual(result, [16, 20, 24, 28, 32])

    def test_a100(self):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-80GB"):
            result = _get_gpu_sm_range()
        self.assertEqual(result, [16, 20, 24, 28, 32])

    def test_unknown_gpu(self):
        with patch("torch.cuda.get_device_name", return_value="Unknown GPU"):
            result = _get_gpu_sm_range()
        self.assertEqual(result, [24])

    def test_exception_returns_default(self):
        with patch("torch.cuda.get_device_name", side_effect=RuntimeError):
            result = _get_gpu_sm_range()
        self.assertEqual(result, [24])


class TestRunDeepepAutotuneIfEnabled(unittest.TestCase):
    """Tests for run_deepep_autotune_if_enabled."""

    def setUp(self):
        _state.tuned_dispatch_config = None
        _state.tuned_combine_config = None

    def tearDown(self):
        _state.tuned_dispatch_config = None
        _state.tuned_combine_config = None

    def test_disabled_sets_default_configs(self):
        """When autotune=False, should set default configs."""
        @dataclass
        class FakeConfig:
            autotune: bool = False
            num_sms: int = 24
            nvl_buffer_size: int = 256
            rdma_buffer_size: int = 128

        mock_group = MagicMock()
        mock_group.size.return_value = 8

        mock_buffer = MagicMock()
        mock_buffer.runtime.get_num_rdma_ranks.return_value = 1
        mock_buffer.group_size = 8

        with patch(
            "torchtitan.distributed.deepep.deepep.get_buffer",
            return_value=mock_buffer,
        ), patch(
            "torchtitan.distributed.deepep.deepep.torch.distributed.get_rank",
            return_value=0,
        ), patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "8"}):
            result = run_deepep_autotune_if_enabled(
                deepep_config=FakeConfig(),
                ep_group=mock_group,
                num_tokens=16384,
                hidden=256,
                num_experts=64,
                num_topk=8,
            )

        self.assertIsNone(result)
        d, c = get_tuned_configs()
        self.assertIsNotNone(d)
        self.assertIsNotNone(c)

    def test_none_config_sets_defaults(self):
        """When deepep_config is None, should set default configs."""
        mock_group = MagicMock()
        mock_group.size.return_value = 8

        mock_buffer = MagicMock()
        mock_buffer.runtime.get_num_rdma_ranks.return_value = 1
        mock_buffer.group_size = 8

        with patch(
            "torchtitan.distributed.deepep.deepep.get_buffer",
            return_value=mock_buffer,
        ), patch(
            "torchtitan.distributed.deepep.deepep.torch.distributed.get_rank",
            return_value=0,
        ), patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "8"}):
            result = run_deepep_autotune_if_enabled(
                deepep_config=None,
                ep_group=mock_group,
                num_tokens=16384,
                hidden=256,
                num_experts=64,
                num_topk=8,
            )

        self.assertIsNone(result)
        d, c = get_tuned_configs()
        self.assertIsNotNone(d)
        self.assertIsNotNone(c)


if __name__ == "__main__":
    unittest.main()
