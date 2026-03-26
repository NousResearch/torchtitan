# Copyright (c) 2025 Nous Research. All rights reserved.

"""Tests for HybridDeviceOptimizer — GPU/CPU split optimizer.

Ported and adapted from Megatron-LM:
  tests/unit_tests/test_optimizer_cpu_offloading.py
"""

import random

import pytest
import torch
import torch.nn as nn

from torchtitan.distributed.cpu_offload.hybrid_optimizer import HybridDeviceOptimizer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ── Test Models ────────────────────────────────────────────────────────


class SmallNet(nn.Module):
    """Small model for basic optimizer tests."""

    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class BigNet(nn.Module):
    """Larger model to exercise offloading with many parameters."""

    def __init__(self, dim=256, hidden=1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ── Helpers ────────────────────────────────────────────────────────────


def _set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_steps(model, optimizer, n_steps, dim):
    """Run n training steps and return final parameters."""
    for _ in range(n_steps):
        _set_seed()
        x = torch.randn(4, dim, device="cuda")
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return {n: p.data.clone() for n, p in model.named_parameters()}


# ── Tests: Basic Construction ──────────────────────────────────────────


class TestHybridOptimizerConstruction:

    def test_create_with_defaults(self):
        model = SmallNet().cuda()
        opt = HybridDeviceOptimizer(
            model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=0.5,
            lr=1e-3,
        )
        assert opt.offload_fraction == 0.5
        assert len(opt.sub_optimizers) > 0

    def test_zero_offload_fraction(self):
        """offload_fraction=0 keeps everything on GPU."""
        model = SmallNet().cuda()
        opt = HybridDeviceOptimizer(
            model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=0.0,
            lr=1e-3,
        )
        assert len(opt.cpu_optimizers) == 0
        assert opt.gpu_optimizer is not None

    def test_full_offload_fraction(self):
        """offload_fraction=1.0 puts everything on CPU."""
        model = SmallNet().cuda()
        opt = HybridDeviceOptimizer(
            model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=1.0,
            lr=1e-3,
        )
        assert len(opt.cpu_optimizers) > 0


# ── Tests: Training Correctness ────────────────────────────────────────


class TestHybridOptimizerCorrectness:

    @pytest.mark.parametrize("offload_fraction", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("n_steps", [1, 5])
    def test_step_runs_without_error(self, offload_fraction, n_steps):
        """Basic smoke test that step() executes."""
        _set_seed()
        model = SmallNet().cuda()
        opt = HybridDeviceOptimizer(
            model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=offload_fraction,
            lr=1e-3,
        )
        for _ in range(n_steps):
            x = torch.randn(4, 64, device="cuda")
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

    @pytest.mark.parametrize("offload_fraction", [0.0, 0.5, 1.0])
    def test_parameters_update(self, offload_fraction):
        """Parameters should change after optimizer step."""
        _set_seed()
        model = SmallNet().cuda()
        initial_params = {n: p.data.clone() for n, p in model.named_parameters()}

        opt = HybridDeviceOptimizer(
            model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=offload_fraction,
            lr=1e-2,  # large LR to see changes
        )

        x = torch.randn(4, 64, device="cuda")
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for name, param in model.named_parameters():
            assert not torch.equal(param.data, initial_params[name]), \
                f"Parameter {name} didn't change after step"

    def test_zero_offload_matches_pure_gpu(self):
        """offload_fraction=0 should produce identical results to a pure GPU optimizer."""
        _set_seed()
        dim = 64

        # Pure GPU reference
        _set_seed()
        ref_model = SmallNet(dim).cuda()
        ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-3)
        ref_params = _train_steps(ref_model, ref_opt, 5, dim)

        # Hybrid with 0 offload
        _set_seed()
        hybrid_model = SmallNet(dim).cuda()
        hybrid_opt = HybridDeviceOptimizer(
            hybrid_model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=0.0,
            lr=1e-3,
        )
        hybrid_params = _train_steps(hybrid_model, hybrid_opt, 5, dim)

        for name in ref_params:
            assert torch.allclose(hybrid_params[name], ref_params[name], atol=1e-5), \
                f"Mismatch at {name}"


# ── Tests: Overlap D2H/H2D ────────────────────────────────────────────


class TestOverlapD2HH2D:

    @pytest.mark.parametrize("overlap", [True, False])
    def test_overlap_flag(self, overlap):
        _set_seed()
        model = BigNet().cuda()
        opt = HybridDeviceOptimizer(
            model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=0.5,
            overlap_cpu_optimizer_d2h_h2d=overlap,
            lr=1e-3,
        )
        # Just verify it runs
        x = torch.randn(4, 256, device="cuda")
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_overlap_and_non_overlap_produce_similar_results(self):
        """Both modes should converge to similar parameter values."""
        _set_seed()
        dim = 64
        n_steps = 10

        # No overlap
        _set_seed()
        m1 = SmallNet(dim).cuda()
        o1 = HybridDeviceOptimizer(
            m1.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=0.5,
            overlap_cpu_optimizer_d2h_h2d=False,
            lr=1e-3,
        )
        p1 = _train_steps(m1, o1, n_steps, dim)

        # With overlap
        _set_seed()
        m2 = SmallNet(dim).cuda()
        o2 = HybridDeviceOptimizer(
            m2.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=0.5,
            overlap_cpu_optimizer_d2h_h2d=True,
            lr=1e-3,
        )
        p2 = _train_steps(m2, o2, n_steps, dim)

        for name in p1:
            assert torch.allclose(p1[name], p2[name], atol=1e-3), \
                f"Overlap mismatch at {name}: {(p1[name] - p2[name]).abs().max()}"


# ── Tests: FP32 Mixed Precision ────────────────────────────────────────


class TestFP32MixedPrecision:

    def test_param_update_in_fp32(self):
        _set_seed()
        model = SmallNet().cuda().half()  # FP16 model
        opt = HybridDeviceOptimizer(
            model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=0.5,
            param_update_in_fp32=True,
            lr=1e-3,
        )
        x = torch.randn(4, 64, device="cuda", dtype=torch.float16)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        # Verify model params are still fp16
        for p in model.parameters():
            assert p.dtype == torch.float16


# ── Tests: Zero Grad ───────────────────────────────────────────────────


class TestZeroGrad:

    @pytest.mark.parametrize("set_to_none", [True, False])
    def test_zero_grad(self, set_to_none):
        model = SmallNet().cuda()
        opt = HybridDeviceOptimizer(
            model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=0.5,
            lr=1e-3,
        )
        x = torch.randn(4, 64, device="cuda")
        model(x).sum().backward()
        opt.zero_grad(set_to_none=set_to_none)

        for p in model.parameters():
            if set_to_none:
                assert p.grad is None
            else:
                assert (p.grad == 0).all()


# ── Tests: Dummy Step ──────────────────────────────────────────────────


class TestDummyStep:

    def test_dummy_step_initializes_state(self):
        model = SmallNet().cuda()
        opt = HybridDeviceOptimizer(
            model.parameters(),
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            offload_fraction=0.5,
            lr=1e-3,
        )
        opt.dummy_step()
        # After dummy step, optimizer state should be populated
        assert len(opt.state) > 0
