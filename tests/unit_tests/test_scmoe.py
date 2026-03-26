# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Shortcut Connected MoE (ScMoE) implementation.

Tests cover:
1. Forward pass correctness (output shape, dtype, determinism)
2. Backward pass correctness (gradients flow through all paths)
3. Shortcut connection correctness (routed uses shortcut, shared uses current)
4. Numerical precision
5. Edge cases
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.moe.moe import (
    FeedForward,
    GroupedExperts,
    MoE,
    MoEArgs,
    ScMoEConfig,
    TokenChoiceTopKRouter,
)
from torchtitan.models.moe.scmoe import (
    ScMoE,
    ScMoEStreamManager,
    ScMoETransformerBlock,
)


class TestScMoEConfig:
    """Tests for ScMoE configuration."""

    def test_default_config(self):
        """Test default ScMoE configuration values."""
        config = ScMoEConfig()
        assert config.shortcut_position == "pos1"
        assert config.use_separate_streams is True
        assert config.sync_before_combine is False

    def test_moe_args_with_scmoe(self):
        """Test MoEArgs includes ScMoE config."""
        args = MoEArgs(use_scmoe=True)
        assert args.use_scmoe is True
        assert isinstance(args.scmoe, ScMoEConfig)


class TestScMoEStreamManager:
    """Tests for ScMoE stream manager."""

    def test_singleton_instance(self):
        """Test that stream manager is a singleton."""
        manager1 = ScMoEStreamManager.get_instance()
        manager2 = ScMoEStreamManager.get_instance()
        assert manager1 is manager2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_comm_stream_creation(self):
        """Test communication stream is created correctly."""
        manager = ScMoEStreamManager.get_instance()
        manager.comm_stream = None  # Reset for test
        device = torch.device("cuda:0")
        stream = manager.get_comm_stream(device)
        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_event_recording_and_waiting(self):
        """Test event recording and waiting mechanisms."""
        manager = ScMoEStreamManager.get_instance()
        manager._events = {}  # Reset events

        manager.record_event("test_event")
        assert "test_event" in manager._events

        manager.wait_event("test_event")
        manager.wait_event("non_existent")  # Should be a no-op


def _make_scmoe_args(**kwargs):
    """Helper to create MoEArgs with ScMoE defaults."""
    defaults = dict(
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        use_scmoe=True,
        use_grouped_mm=True,
        score_func="softmax",
        score_before_experts=True,
        scmoe=ScMoEConfig(),
    )
    defaults.update(kwargs)
    return MoEArgs(**defaults)


def _init_scmoe(module: ScMoE) -> ScMoE:
    """Helper to initialize ScMoE weights for tests."""
    with torch.no_grad():
        module.init_weights(init_std=0.02, buffer_device=torch.device("cuda"), n_layers=1)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoE requires CUDA for torch.histc")
class TestScMoEForwardPass:
    """Tests for ScMoE forward pass correctness."""

    @pytest.fixture
    def scmoe_module(self):
        """Create ScMoE module on CUDA with initialized weights."""
        return _init_scmoe(
            ScMoE(moe_args=_make_scmoe_args(), dim=32, hidden_dim=64).cuda()
        )

    def test_output_shape(self, scmoe_module):
        """Test that output shape matches input shape."""
        bs, slen, dim = 2, 16, 32
        x_current = torch.randn(bs, slen, dim, device="cuda")
        x_shortcut = torch.randn(bs, slen, dim, device="cuda")
        output = scmoe_module(x_current, x_shortcut)
        assert output.shape == (bs, slen, dim)

    def test_output_dtype(self, scmoe_module):
        """Test that output dtype matches input dtype."""
        bs, slen, dim = 2, 16, 32
        for dtype in [torch.float32, torch.bfloat16]:
            x_current = torch.randn(bs, slen, dim, dtype=dtype, device="cuda")
            x_shortcut = torch.randn(bs, slen, dim, dtype=dtype, device="cuda")
            output = scmoe_module.to(dtype)(x_current, x_shortcut)
            assert output.dtype == dtype

    def test_deterministic_output(self, scmoe_module):
        """Test that output is deterministic with same input."""
        bs, slen, dim = 2, 16, 32
        torch.manual_seed(42)
        x_current = torch.randn(bs, slen, dim, device="cuda")
        x_shortcut = torch.randn(bs, slen, dim, device="cuda")
        output1 = scmoe_module(x_current, x_shortcut)
        output2 = scmoe_module(x_current, x_shortcut)
        torch.testing.assert_close(output1, output2)

    def test_different_inputs_different_outputs(self, scmoe_module):
        """Test that different inputs produce different outputs."""
        bs, slen, dim = 2, 16, 32
        output1 = scmoe_module(
            torch.randn(bs, slen, dim, device="cuda"),
            torch.randn(bs, slen, dim, device="cuda"),
        )
        output2 = scmoe_module(
            torch.randn(bs, slen, dim, device="cuda"),
            torch.randn(bs, slen, dim, device="cuda"),
        )
        assert not torch.allclose(output1, output2)

    def test_routed_experts_use_shortcut_input(self):
        """Test that routed experts (router) use shortcut input, not current input."""
        dim = 32
        scmoe = _init_scmoe(
            ScMoE(moe_args=_make_scmoe_args(), dim=dim, hidden_dim=64).cuda()
        )

        router_inputs = []
        scmoe.router.register_forward_hook(
            lambda mod, inp, out: router_inputs.append(inp[0].clone())
        )

        bs, slen = 2, 16
        x_current = torch.ones(bs, slen, dim, device="cuda")   # All ones
        x_shortcut = torch.zeros(bs, slen, dim, device="cuda")  # All zeros
        scmoe(x_current, x_shortcut)

        # Router should receive normalized shortcut (zeros), not current (ones)
        assert len(router_inputs) == 1
        assert router_inputs[0].mean().abs() < 0.5

    def test_shared_experts_use_current_input(self):
        """Test that shared experts use current input, not shortcut."""
        dim = 32
        scmoe = _init_scmoe(
            ScMoE(moe_args=_make_scmoe_args(), dim=dim, hidden_dim=64).cuda()
        )

        shared_inputs = []
        scmoe.shared_experts.register_forward_hook(
            lambda mod, inp, out: shared_inputs.append(inp[0].clone())
        )

        bs, slen = 2, 16
        x_current = torch.ones(bs, slen, dim, device="cuda") * 5
        x_shortcut = torch.ones(bs, slen, dim, device="cuda") * -5
        scmoe(x_current, x_shortcut)

        assert len(shared_inputs) == 1
        assert shared_inputs[0].mean() > 0  # Positive from x_current


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoE requires CUDA for torch.histc")
class TestScMoEBackwardPass:
    """Tests for ScMoE backward pass correctness."""

    @pytest.fixture
    def scmoe_module(self):
        return _init_scmoe(
            ScMoE(moe_args=_make_scmoe_args(), dim=32, hidden_dim=64).cuda()
        )

    def test_gradients_exist(self, scmoe_module):
        """Test that gradients exist for all parameters after backward."""
        bs, slen, dim = 2, 16, 32
        x_current = torch.randn(bs, slen, dim, device="cuda", requires_grad=True)
        x_shortcut = torch.randn(bs, slen, dim, device="cuda", requires_grad=True)

        output = scmoe_module(x_current, x_shortcut)
        output.sum().backward()

        assert x_current.grad is not None
        assert x_shortcut.grad is not None
        for name, param in scmoe_module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_shapes(self, scmoe_module):
        """Test that gradient shapes match parameter shapes."""
        bs, slen, dim = 2, 16, 32
        x_current = torch.randn(bs, slen, dim, device="cuda", requires_grad=True)
        x_shortcut = torch.randn(bs, slen, dim, device="cuda", requires_grad=True)

        scmoe_module(x_current, x_shortcut).sum().backward()

        for name, param in scmoe_module.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert param.grad.shape == param.shape, f"Shape mismatch for {name}"

    def test_gradient_flow_to_shortcut(self, scmoe_module):
        """Test that gradients flow through shortcut path (routed experts)."""
        bs, slen, dim = 2, 16, 32
        x_current = torch.randn(bs, slen, dim, device="cuda", requires_grad=True)
        x_shortcut = torch.randn(bs, slen, dim, device="cuda", requires_grad=True)

        scmoe_module(x_current, x_shortcut).sum().backward()
        assert x_shortcut.grad is not None
        assert x_shortcut.grad.abs().sum() > 0

    def test_gradient_flow_to_current(self, scmoe_module):
        """Test that gradients flow through current path (shared experts)."""
        bs, slen, dim = 2, 16, 32
        x_current = torch.randn(bs, slen, dim, device="cuda", requires_grad=True)
        x_shortcut = torch.randn(bs, slen, dim, device="cuda", requires_grad=True)

        scmoe_module(x_current, x_shortcut).sum().backward()
        assert x_current.grad is not None
        assert x_current.grad.abs().sum() > 0

    def test_gradient_accumulation(self, scmoe_module):
        """Test gradient accumulation over multiple forward passes."""
        bs, slen, dim = 2, 16, 32
        scmoe_module.zero_grad()

        for _ in range(2):
            x_current = torch.randn(bs, slen, dim, device="cuda")
            x_shortcut = torch.randn(bs, slen, dim, device="cuda")
            scmoe_module(x_current, x_shortcut).sum().backward()

        for name, param in scmoe_module.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoE requires CUDA for torch.histc")
class TestScMoENumericalPrecision:
    """Tests for numerical precision and stability."""

    def test_fp32_bf16_close(self):
        """Test FP32 and BF16 outputs are close."""
        dim, hidden_dim = 32, 64
        args = _make_scmoe_args(use_grouped_mm=False)

        scmoe_fp32 = _init_scmoe(ScMoE(moe_args=args, dim=dim, hidden_dim=hidden_dim).cuda())
        scmoe_bf16 = ScMoE(moe_args=args, dim=dim, hidden_dim=hidden_dim).cuda()
        scmoe_bf16.load_state_dict(scmoe_fp32.state_dict())
        scmoe_bf16 = scmoe_bf16.to(torch.bfloat16)

        bs, slen = 2, 16
        torch.manual_seed(42)
        x_current_fp32 = torch.randn(bs, slen, dim, device="cuda")
        x_shortcut_fp32 = torch.randn(bs, slen, dim, device="cuda")

        output_fp32 = scmoe_fp32(x_current_fp32, x_shortcut_fp32)
        output_bf16 = scmoe_bf16(
            x_current_fp32.to(torch.bfloat16), x_shortcut_fp32.to(torch.bfloat16)
        )

        relative_error = (output_fp32 - output_bf16.float()).abs() / (output_fp32.abs() + 1e-6)
        assert relative_error.mean() < 0.1

    def test_no_nan_or_inf(self):
        """Test that outputs don't contain NaN or Inf."""
        args = _make_scmoe_args(use_grouped_mm=False)
        scmoe = _init_scmoe(ScMoE(moe_args=args, dim=32, hidden_dim=64).cuda())

        for _ in range(10):
            output = scmoe(
                torch.randn(2, 16, 32, device="cuda"),
                torch.randn(2, 16, 32, device="cuda"),
            )
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoE requires CUDA for torch.histc")
class TestScMoETransformerBlock:
    """Tests for ScMoE TransformerBlock."""

    @pytest.fixture
    def mock_attention(self):
        class MockAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, x, freqs_cis, attention_masks, positions=None):
                return self.linear(x)

            def init_weights(self, init_std):
                nn.init.xavier_uniform_(self.linear.weight)

        return MockAttention

    def test_transformer_block_output_shape(self, mock_attention):
        """Test transformer block output shape."""
        dim, hidden_dim = 32, 64
        block = ScMoETransformerBlock(
            layer_id=0, dim=dim, n_heads=4, n_kv_heads=4, head_dim=8,
            hidden_dim=hidden_dim,
            moe_args=_make_scmoe_args(),
            attention_module=mock_attention(dim),
        ).cuda()
        with torch.no_grad():
            block.init_weights(buffer_device=torch.device("cuda"))

        bs, slen = 2, 16
        x = torch.randn(bs, slen, dim, device="cuda")
        shortcut = torch.randn(bs, slen, dim, device="cuda")
        freqs_cis = torch.randn(slen, 8, device="cuda")

        output, next_shortcut = block(x, shortcut, freqs_cis, None)
        assert output.shape == (bs, slen, dim)
        assert next_shortcut.shape == (bs, slen, dim)

    def test_transformer_block_shortcut_propagation(self, mock_attention):
        """Test that shortcut is correctly passed between layers."""
        dim, hidden_dim = 32, 64
        block = ScMoETransformerBlock(
            layer_id=0, dim=dim, n_heads=4, n_kv_heads=4, head_dim=8,
            hidden_dim=hidden_dim,
            moe_args=_make_scmoe_args(),
            attention_module=mock_attention(dim),
        ).cuda()
        with torch.no_grad():
            block.init_weights(buffer_device=torch.device("cuda"))

        bs, slen = 2, 16
        x = torch.randn(bs, slen, dim, device="cuda")
        freqs_cis = torch.randn(slen, 8, device="cuda")

        output1, shortcut1 = block(x, None, freqs_cis, None)
        output2, shortcut2 = block(output1, shortcut1, freqs_cis, None)
        assert not torch.allclose(output1, output2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoE requires CUDA for torch.histc")
class TestScMoEEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_token(self):
        scmoe = _init_scmoe(ScMoE(moe_args=_make_scmoe_args(), dim=32, hidden_dim=64).cuda())
        output = scmoe(
            torch.randn(1, 1, 32, device="cuda"),
            torch.randn(1, 1, 32, device="cuda"),
        )
        assert output.shape == (1, 1, 32)

    def test_large_batch(self):
        scmoe = _init_scmoe(ScMoE(moe_args=_make_scmoe_args(), dim=32, hidden_dim=64).cuda())
        output = scmoe(
            torch.randn(64, 128, 32, device="cuda"),
            torch.randn(64, 128, 32, device="cuda"),
        )
        assert output.shape == (64, 128, 32)

    def test_no_shared_experts(self):
        """Test with no shared experts (routed-only, no overlap benefit)."""
        args = _make_scmoe_args(num_shared_experts=0)
        scmoe = _init_scmoe(ScMoE(moe_args=args, dim=32, hidden_dim=64).cuda())
        assert scmoe.shared_experts is None
        output = scmoe(
            torch.randn(2, 16, 32, device="cuda"),
            torch.randn(2, 16, 32, device="cuda"),
        )
        assert output.shape == (2, 16, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
