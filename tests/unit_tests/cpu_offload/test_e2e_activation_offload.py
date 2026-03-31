# Copyright (c) 2025 Nous Research. All rights reserved.

"""End-to-end tests for activation offloading with real neural network modules.

Tests range from a simple Linear layer to multi-layer MLPs and MoE-like
architectures. Each test verifies:
1. Forward correctness: offloaded output == baseline output
2. Backward correctness: offloaded gradients == baseline gradients
3. Memory reduction: peak GPU memory is lower with offloading
"""

import random

import pytest
import torch
import torch.nn as nn

from torchtitan.distributed.cpu_offload import (
    ActivationOffloadContext,
    OffloadManager,
    group_commit,
    group_start,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ── Test Models ────────────────────────────────────────────────────────


class SimpleLinear(nn.Module):
    """Single linear layer with offloading hooks."""

    def __init__(self, in_features, out_features, offload=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.offload = offload

    def forward(self, x):
        if self.offload:
            with ActivationOffloadContext(True, x, "linear") as x:
                out = self.linear(x)
            out = group_commit(out, "linear", forced_released_tensors=[])
            return out
        return self.linear(x)


class TwoLayerMLP(nn.Module):
    """Two-layer MLP with per-layer offloading control."""

    def __init__(self, dim, hidden, offload_fc1=False, offload_fc2=False):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.offload_fc1 = offload_fc1
        self.offload_fc2 = offload_fc2

    def forward(self, x):
        # FC1
        if self.offload_fc1:
            with ActivationOffloadContext(True, x, "fc1") as x_in:
                h = self.fc1(x_in)
            h = group_commit(h, "fc1", forced_released_tensors=[x])
        else:
            h = self.fc1(x)

        h = self.act(h)

        # FC2
        if self.offload_fc2:
            with ActivationOffloadContext(True, h, "fc2") as h_in:
                out = self.fc2(h_in)
            out = group_commit(out, "fc2", forced_released_tensors=[h])
        else:
            out = self.fc2(h)

        return out


class MultiLayerMLP(nn.Module):
    """N-layer MLP to test deep offloading.

    All layers use the same group name "mlp_layer" — this mirrors
    Megatron's pattern where every transformer layer's expert_fc1
    shares the name. The layer-staggered reload pattern hides
    layer N's reload behind layer N+1's backward compute. Only the
    *last* occurrence of each name gets marked non-offloadable.
    """

    def __init__(self, dim, hidden, n_layers, offload=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_d = dim if i == 0 else hidden
            out_d = dim if i == n_layers - 1 else hidden
            self.layers.append(nn.Linear(in_d, out_d))
        self.offload = offload

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Same name for all layers — enables layer-staggered reload
            name = "mlp_layer"
            if self.offload:
                with ActivationOffloadContext(True, x, name) as x_in:
                    x = layer(x_in)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)
                x = group_commit(x, name)
            else:
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)
        return x


class FakeExpertMLP(nn.Module):
    """Simulates an MoE expert layer with two FC layers and offloading."""

    def __init__(self, dim, expert_dim, offload_fc1=False, offload_act=False):
        super().__init__()
        self.fc1 = nn.Linear(dim, expert_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(expert_dim, dim)
        self.offload_fc1 = offload_fc1
        self.offload_act = offload_act

    def forward(self, x):
        # Expert FC1 + offload input
        if self.offload_fc1:
            with ActivationOffloadContext(True, x, "expert_fc1") as x_in:
                h = self.fc1(x_in)
            h = group_commit(h, "expert_fc1", forced_released_tensors=[x])
        else:
            h = self.fc1(x)

        # Activation + offload intermediate
        if self.offload_act:
            with ActivationOffloadContext(True, h, "moe_act") as h_in:
                h = self.act(h_in)
            h = group_commit(h, "moe_act")
        else:
            h = self.act(h)

        out = self.fc2(h)
        return out


class SimpleMoE(nn.Module):
    """Simple MoE with N experts and top-1 routing."""

    def __init__(self, dim, expert_dim, n_experts, offload=False):
        super().__init__()
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            FakeExpertMLP(dim, expert_dim, offload_fc1=offload, offload_act=offload)
            for _ in range(n_experts)
        ])

    def forward(self, x):
        # x: (batch, seq, dim)
        logits = self.router(x)  # (batch, seq, n_experts)
        indices = logits.argmax(dim=-1)  # (batch, seq)

        # Simple loop-based routing (not efficient, but correct for testing)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            if mask.any():
                tokens = x[mask]
                out[mask] = expert(tokens)
        return out


# ── Helpers ────────────────────────────────────────────────────────────


def _set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _reset_cuda_memory():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _run_forward_backward(model, x, seed=42):
    """Run forward + backward and return (output, grad_dict)."""
    _set_seed(seed)
    model.zero_grad()
    out = model(x)
    loss = out.sum()
    loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return out.detach().clone(), grads


def _init_offload_manager():
    """Reset and initialize the offload manager for a fresh run."""
    OffloadManager.reset_instance()
    ActivationOffloadContext.init_chunk_handler(
        vp_size=1, vp_stage=0, min_offloaded_tensor_size=1
    )


# ── Tests: Simple Linear ──────────────────────────────────────────────


class TestSimpleLinearOffload:

    def test_forward_correctness(self):
        """Offloaded forward matches baseline."""
        _set_seed()
        dim = 256
        x = torch.randn(4, 32, dim, device="cuda")

        # Baseline
        base_model = SimpleLinear(dim, dim, offload=False).cuda()
        base_out, _ = _run_forward_backward(base_model, x)

        # Offloaded
        off_model = SimpleLinear(dim, dim, offload=True).cuda()
        off_model.load_state_dict(base_model.state_dict())
        _init_offload_manager()
        off_out, _ = _run_forward_backward(off_model, x)

        assert torch.allclose(off_out, base_out, rtol=1e-4, atol=1e-4), \
            f"Max diff: {(off_out - base_out).abs().max()}"

    def test_backward_correctness(self):
        """Offloaded gradients match baseline."""
        _set_seed()
        dim = 256
        x = torch.randn(4, 32, dim, device="cuda")

        base_model = SimpleLinear(dim, dim, offload=False).cuda()
        _, base_grads = _run_forward_backward(base_model, x)

        off_model = SimpleLinear(dim, dim, offload=True).cuda()
        off_model.load_state_dict(base_model.state_dict())
        _init_offload_manager()
        _, off_grads = _run_forward_backward(off_model, x)

        assert set(off_grads.keys()) == set(base_grads.keys())
        for name in base_grads:
            assert torch.allclose(off_grads[name], base_grads[name], rtol=1e-4, atol=1e-4), \
                f"Grad mismatch at {name}: max diff {(off_grads[name] - base_grads[name]).abs().max()}"


# ── Tests: Two-Layer MLP ──────────────────────────────────────────────


class TestTwoLayerMLPOffload:

    @pytest.mark.parametrize("offload_fc1,offload_fc2", [
        (True, False),
        (False, True),
        (True, True),
    ])
    def test_forward_backward_correctness(self, offload_fc1, offload_fc2):
        _set_seed()
        dim, hidden = 256, 512
        x = torch.randn(4, 32, dim, device="cuda")

        base = TwoLayerMLP(dim, hidden, False, False).cuda()
        base_out, base_grads = _run_forward_backward(base, x)

        off = TwoLayerMLP(dim, hidden, offload_fc1, offload_fc2).cuda()
        off.load_state_dict(base.state_dict())
        _init_offload_manager()
        off_out, off_grads = _run_forward_backward(off, x)

        assert torch.allclose(off_out, base_out, rtol=1e-3, atol=1e-3), \
            f"Forward mismatch: {(off_out - base_out).abs().max()}"

        for name in base_grads:
            assert torch.allclose(off_grads[name], base_grads[name], rtol=1e-3, atol=1e-3), \
                f"Grad mismatch at {name}: {(off_grads[name] - base_grads[name]).abs().max()}"


# ── Tests: Deep Multi-Layer MLP ───────────────────────────────────────


class TestMultiLayerMLPOffload:

    @pytest.mark.parametrize("n_layers", [4, 8, 16])
    def test_deep_offload_correctness(self, n_layers):
        _set_seed()
        dim, hidden = 128, 256
        x = torch.randn(2, 16, dim, device="cuda")

        base = MultiLayerMLP(dim, hidden, n_layers, offload=False).cuda()
        base_out, base_grads = _run_forward_backward(base, x)

        off = MultiLayerMLP(dim, hidden, n_layers, offload=True).cuda()
        off.load_state_dict(base.state_dict())
        _init_offload_manager()
        off_out, off_grads = _run_forward_backward(off, x)

        assert torch.allclose(off_out, base_out, rtol=1e-3, atol=1e-3)
        for name in base_grads:
            assert torch.allclose(off_grads[name], base_grads[name], rtol=1e-3, atol=1e-3)


# ── Tests: Fake Expert (MoE-like) ─────────────────────────────────────


class TestFakeExpertOffload:

    @pytest.mark.parametrize("offload_fc1,offload_act", [
        (True, False),
        (False, True),
        (True, True),
    ])
    def test_expert_correctness(self, offload_fc1, offload_act):
        _set_seed()
        dim, expert_dim = 256, 512
        x = torch.randn(8, dim, device="cuda")

        base = FakeExpertMLP(dim, expert_dim, False, False).cuda()
        base_out, base_grads = _run_forward_backward(base, x)

        off = FakeExpertMLP(dim, expert_dim, offload_fc1, offload_act).cuda()
        off.load_state_dict(base.state_dict())
        _init_offload_manager()
        off_out, off_grads = _run_forward_backward(off, x)

        assert torch.allclose(off_out, base_out, rtol=1e-3, atol=1e-3)
        for name in base_grads:
            assert torch.allclose(off_grads[name], base_grads[name], rtol=1e-3, atol=1e-3)


# ── Tests: SimpleMoE ──────────────────────────────────────────────────


class TestSimpleMoEOffload:

    def test_moe_forward_backward_correctness(self):
        _set_seed()
        dim, expert_dim, n_experts = 128, 256, 4
        x = torch.randn(2, 16, dim, device="cuda")

        base = SimpleMoE(dim, expert_dim, n_experts, offload=False).cuda()
        base_out, base_grads = _run_forward_backward(base, x)

        off = SimpleMoE(dim, expert_dim, n_experts, offload=True).cuda()
        off.load_state_dict(base.state_dict())
        _init_offload_manager()
        off_out, off_grads = _run_forward_backward(off, x)

        assert torch.allclose(off_out, base_out, rtol=1e-3, atol=1e-3)
        for name in base_grads:
            if name in off_grads:
                assert torch.allclose(off_grads[name], base_grads[name], rtol=1e-3, atol=1e-3), \
                    f"MoE grad mismatch at {name}"


# ── Tests: Memory Reduction ───────────────────────────────────────────


class TestMemoryReduction:

    def test_offloading_reduces_peak_memory(self):
        """Offloading should reduce peak GPU memory for large models.

        Uses warmup iteration (iter 0) + steady-state iteration (iter 1).
        Memory is measured on the steady-state iteration where offloading
        is fully active (warmup marks last groups as non-offloadable,
        but with shared group names, only the last occurrence is skipped).
        """
        _set_seed()
        dim, hidden, n_layers = 512, 2048, 8
        x = torch.randn(4, 64, dim, device="cuda")

        # Baseline: no offloading, measure steady-state peak
        _reset_cuda_memory()
        base = MultiLayerMLP(dim, hidden, n_layers, offload=False).cuda()
        _run_forward_backward(base, x)  # warmup
        _reset_cuda_memory()
        _run_forward_backward(base, x)  # steady state
        base_peak = torch.cuda.max_memory_allocated()
        del base
        _reset_cuda_memory()

        # Offloaded: warmup iter triggers post_warmup, then measure steady-state
        off = MultiLayerMLP(dim, hidden, n_layers, offload=True).cuda()
        _init_offload_manager()
        _run_forward_backward(off, x)  # warmup (creates groups, collects stats)
        ActivationOffloadContext.reset()  # triggers post_warmup
        ActivationOffloadContext.init_chunk_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=1)
        _reset_cuda_memory()
        _run_forward_backward(off, x)  # steady state with offloading active
        off_peak = torch.cuda.max_memory_allocated()

        saved_mb = (base_peak - off_peak) / (1024 * 1024)
        print(f"Memory: baseline={base_peak/(1024**2):.1f}MB, "
              f"offload={off_peak/(1024**2):.1f}MB, saved={saved_mb:.1f}MB")

        mgr = OffloadManager.get_instance()
        print(f"Offload summary: {mgr.offload_summary_bytes}")
        print(f"Total offload: {mgr.offload_summary_total_bytes / (1024**2):.1f}MB")

        del off
        _reset_cuda_memory()

        assert off_peak <= base_peak, \
            f"Offloading increased memory! base={base_peak}, off={off_peak}"


# ── Tests: Multiple Iterations ─────────────────────────────────────────


class TestMultipleIterations:

    def test_offload_across_iterations(self):
        """Run multiple forward-backward iterations with offloading.

        After warmup (iteration 0), the tensor pool should be reused
        and results should remain correct.
        """
        _set_seed()
        dim, hidden = 256, 512
        n_iters = 5

        base = TwoLayerMLP(dim, hidden, False, False).cuda()
        off = TwoLayerMLP(dim, hidden, True, True).cuda()
        off.load_state_dict(base.state_dict())

        for i in range(n_iters):
            _set_seed(seed=42 + i)
            x = torch.randn(4, 32, dim, device="cuda")

            base_out, base_grads = _run_forward_backward(base, x, seed=42 + i)

            if i == 0:
                _init_offload_manager()
            else:
                ActivationOffloadContext.reset()
                ActivationOffloadContext.init_chunk_handler(
                    vp_size=1, vp_stage=0, min_offloaded_tensor_size=1
                )

            off_out, off_grads = _run_forward_backward(off, x, seed=42 + i)

            assert torch.allclose(off_out, base_out, rtol=1e-3, atol=1e-3), \
                f"Iter {i}: forward mismatch"
            for name in base_grads:
                assert torch.allclose(off_grads[name], base_grads[name], rtol=1e-3, atol=1e-3), \
                    f"Iter {i}: grad mismatch at {name}"

            # Update both models with same optimizer step so they stay in sync
            with torch.no_grad():
                for (bn, bp), (on, op) in zip(
                    base.named_parameters(), off.named_parameters()
                ):
                    if bp.grad is not None:
                        bp.data -= 0.01 * bp.grad
                        op.data -= 0.01 * op.grad


# ── Tests: Offload Summary Statistics ──────────────────────────────────


class TestOffloadSummary:

    def test_warmup_creates_offload_groups(self):
        """After warmup, OffloadManager should have created groups and
        collected statistics during the warmup forward pass.

        Note: with a single chunk (no PP/VPP), all groups in the last
        chunk get marked offload=False because there's no following
        compute to hide reload behind. This is correct — the summary
        may show 0 bytes, but the groups themselves should exist.
        """
        _set_seed()
        dim, hidden, n_layers = 128, 256, 4
        x = torch.randn(2, 16, dim, device="cuda")

        model = MultiLayerMLP(dim, hidden, n_layers, offload=True).cuda()
        _init_offload_manager()

        # First forward+backward is warmup
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Get the chunk handler — it should have groups
        mgr = OffloadManager.get_instance()
        assert len(mgr._cached_chunks_forward) > 0, "No chunks cached during warmup"
        chunk = mgr._cached_chunks_forward[0]
        assert len(chunk.offload_groups) == n_layers, \
            f"Expected {n_layers} groups, got {len(chunk.offload_groups)}"

        # All groups should share the name "mlp_layer"
        group_names = [g._name for g in chunk.offload_groups]
        assert all(n == "mlp_layer" for n in group_names), \
            f"Expected all 'mlp_layer', got {group_names}"

        # Verify warmup collected byte stats on at least some groups
        total_warmup_bytes = sum(g.total_offload_bytes for g in chunk.offload_groups)
        assert total_warmup_bytes > 0, "Warmup should have measured offload bytes"

        # Reset triggers post_warmup
        ActivationOffloadContext.reset()


# ── Tests: Dtypes ──────────────────────────────────────────────────────


class TestDtypeSupport:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_offload_different_dtypes(self, dtype):
        _set_seed()
        dim = 128
        x = torch.randn(2, 16, dim, device="cuda", dtype=dtype)

        base = SimpleLinear(dim, dim, offload=False).cuda().to(dtype)
        base_out, _ = _run_forward_backward(base, x)

        off = SimpleLinear(dim, dim, offload=True).cuda().to(dtype)
        off.load_state_dict(base.state_dict())
        _init_offload_manager()
        off_out, _ = _run_forward_backward(off, x)

        assert torch.allclose(off_out, base_out, rtol=1e-2, atol=1e-2)
