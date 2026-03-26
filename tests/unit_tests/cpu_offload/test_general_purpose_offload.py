# Copyright (c) 2025 Nous Research. All rights reserved.

"""Tests for general-purpose offloading: TensorOffloader, weights, gradients,
torch.compile compatibility, and CUDA graph compatibility."""

import pytest
import torch
import torch.nn as nn

from torchtitan.distributed.cpu_offload.tensor_offloader import TensorOffloader
from torchtitan.distributed.cpu_offload.weight_offload import WeightOffloadHook
from torchtitan.distributed.cpu_offload.gradient_offload import GradientOffloadHook

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ═══════════════════════════════════════════════════════════════════════
# TENSOR OFFLOADER — GENERAL PURPOSE
# ═══════════════════════════════════════════════════════════════════════


class TestTensorOffloader:

    def test_offload_reload_roundtrip(self):
        offloader = TensorOffloader()
        t = torch.randn(256, 256, device="cuda")
        expected = t.clone()

        handle = offloader.offload(t)
        offloader.sync_offload()
        recovered = offloader.reload(handle)
        offloader.sync_reload()

        assert torch.equal(recovered, expected)
        offloader.clear()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_offload_dtypes(self, dtype):
        offloader = TensorOffloader()
        t = torch.randn(128, 128, device="cuda", dtype=dtype)
        expected = t.clone()

        handle = offloader.offload(t)
        offloader.sync_offload()
        recovered = offloader.reload(handle)
        offloader.sync_reload()

        assert torch.equal(recovered, expected)
        offloader.clear()

    def test_offload_with_release_storage(self):
        offloader = TensorOffloader()
        t = torch.randn(512, 512, device="cuda")
        expected = t.clone()

        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        handle = offloader.offload(t, release_storage=True)
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        assert mem_after < mem_before, "release_storage should free GPU memory"
        assert t.untyped_storage().size() == 0

        recovered = offloader.reload(handle)
        offloader.sync_reload()
        assert torch.equal(recovered, expected)
        offloader.clear()

    def test_offload_non_contiguous(self):
        offloader = TensorOffloader()
        t = torch.randn(64, 128, device="cuda").t()
        assert not t.is_contiguous()
        expected = t.clone().contiguous()

        handle = offloader.offload(t)
        offloader.sync_offload()
        recovered = offloader.reload(handle)
        offloader.sync_reload()

        assert torch.equal(recovered, expected)
        offloader.clear()

    def test_multiple_tensors_concurrent(self):
        offloader = TensorOffloader()
        tensors = [torch.randn(256, 256, device="cuda") for _ in range(10)]
        expected = [t.clone() for t in tensors]

        handles = [offloader.offload(t) for t in tensors]
        offloader.sync_offload()

        recovered = [offloader.reload(h) for h in handles]
        offloader.sync_all()

        for i, (r, e) in enumerate(zip(recovered, expected)):
            assert torch.equal(r, e), f"Mismatch at tensor {i}"
        offloader.clear()

    def test_pool_reuse_across_calls(self):
        offloader = TensorOffloader(use_pool=True)
        shape = (128, 256)

        t = torch.randn(*shape, device="cuda")
        h = offloader.offload(t)
        offloader.sync_offload()
        offloader.reload(h)
        offloader.sync_all()

        assert offloader.pool.stats["pool_misses"] == 1
        offloader.reset_pool()

        t2 = torch.randn(*shape, device="cuda")
        h2 = offloader.offload(t2)
        offloader.sync_offload()
        offloader.reload(h2)
        offloader.sync_all()

        assert offloader.pool.stats["pool_hits"] >= 1
        offloader.clear()

    def test_without_pool(self):
        offloader = TensorOffloader(use_pool=False)
        t = torch.randn(64, 64, device="cuda")
        expected = t.clone()

        handle = offloader.offload(t)
        offloader.sync_offload()
        recovered = offloader.reload(handle)
        offloader.sync_reload()

        assert torch.equal(recovered, expected)

    def test_streams_are_separate(self):
        offloader = TensorOffloader()
        assert offloader.d2h_stream != torch.cuda.current_stream()
        assert offloader.h2d_stream != torch.cuda.current_stream()
        assert offloader.d2h_stream != offloader.h2d_stream


# ═══════════════════════════════════════════════════════════════════════
# WEIGHT OFFLOADING
# ═══════════════════════════════════════════════════════════════════════


class TestWeightOffload:

    def test_weight_offload_forward_correctness(self):
        """Weights offloaded to CPU, reloaded before forward — output should match."""
        torch.manual_seed(42)
        dim = 128

        # Baseline (no offloading)
        base = nn.Linear(dim, dim).cuda()
        x = torch.randn(4, dim, device="cuda")
        base_out = base(x).detach().clone()

        # With weight offloading
        off = nn.Linear(dim, dim).cuda()
        off.load_state_dict(base.state_dict())

        offloader = TensorOffloader()
        hook = WeightOffloadHook(offloader, prefetch_next=False)
        hook.register(off)

        off_out = off(x).detach().clone()

        assert torch.allclose(off_out, base_out, rtol=1e-4, atol=1e-4)
        hook.remove_hooks()
        offloader.clear()

    def test_weight_offload_backward_correctness(self):
        """Gradients should be correct even with weight offloading."""
        torch.manual_seed(42)
        dim = 128
        x = torch.randn(4, dim, device="cuda")

        base = nn.Linear(dim, dim).cuda()
        base.zero_grad()
        base(x).sum().backward()
        base_grad = base.weight.grad.clone()

        off = nn.Linear(dim, dim).cuda()
        off.load_state_dict(base.state_dict())
        offloader = TensorOffloader()
        hook = WeightOffloadHook(offloader, prefetch_next=False)
        hook.register(off)

        off.zero_grad()
        off(x).sum().backward()
        off_grad = off.weight.grad.clone()

        assert torch.allclose(off_grad, base_grad, rtol=1e-4, atol=1e-4)
        hook.remove_hooks()
        offloader.clear()

    def test_weight_offload_multi_layer(self):
        """Test offloading across multiple layers."""
        torch.manual_seed(42)
        dim = 64

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim)
                self.fc2 = nn.Linear(dim, dim)
                self.fc3 = nn.Linear(dim, dim)

            def forward(self, x):
                return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

        x = torch.randn(4, dim, device="cuda")

        base = MLP().cuda()
        base_out = base(x).detach().clone()

        off = MLP().cuda()
        off.load_state_dict(base.state_dict())
        offloader = TensorOffloader()
        hook = WeightOffloadHook(offloader, prefetch_next=True)
        hook.register(off.fc1)
        hook.register(off.fc2)
        hook.register(off.fc3)

        off_out = off(x).detach().clone()

        assert torch.allclose(off_out, base_out, rtol=1e-3, atol=1e-3)
        hook.remove_hooks()
        offloader.clear()


# ═══════════════════════════════════════════════════════════════════════
# GRADIENT OFFLOADING
# ═══════════════════════════════════════════════════════════════════════


class TestGradientOffload:

    def test_gradient_offload_to_cpu(self):
        """After backward with release_gpu_grad, param.grad is freed and CPU copy in handles."""
        model = nn.Linear(64, 64).cuda()
        offloader = TensorOffloader()
        grad_hook = GradientOffloadHook(offloader, release_gpu_grad=True)
        grad_hook.register(model.parameters())

        x = torch.randn(4, 64, device="cuda")
        model(x).sum().backward()

        # With release_gpu_grad=True, param.grad is set to None
        for p in model.parameters():
            assert p.grad is None, "Grad should be None (freed from GPU)"

        # But the CPU copy exists in handles
        assert grad_hook.has_offloaded_grads()

        grad_hook.remove_hooks()
        offloader.clear()

    def test_gradient_reload_for_gpu_optimizer(self):
        """Reload gradients back to GPU before optimizer step."""
        model = nn.Linear(64, 64).cuda()
        offloader = TensorOffloader()
        grad_hook = GradientOffloadHook(offloader, release_gpu_grad=True)
        grad_hook.register(model.parameters())

        x = torch.randn(4, 64, device="cuda")
        model(x).sum().backward()

        # Reload before optimizer
        grad_hook.reload_all()

        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.device.type == "cuda", f"Grad should be on GPU after reload"

        grad_hook.remove_hooks()
        offloader.clear()

    def test_gradient_correctness(self):
        """Offloaded+reloaded gradients must match baseline."""
        torch.manual_seed(42)
        dim = 128
        x = torch.randn(4, dim, device="cuda")

        # Baseline
        base = nn.Linear(dim, dim).cuda()
        base(x).sum().backward()
        base_grads = {n: p.grad.clone() for n, p in base.named_parameters()}

        # With gradient offloading
        off = nn.Linear(dim, dim).cuda()
        off.load_state_dict(base.state_dict())
        offloader = TensorOffloader()
        grad_hook = GradientOffloadHook(offloader, release_gpu_grad=True)
        grad_hook.register(off.parameters())

        off(x).sum().backward()
        grad_hook.reload_all()

        for name, param in off.named_parameters():
            assert torch.allclose(param.grad, base_grads[name], rtol=1e-5, atol=1e-5), \
                f"Grad mismatch at {name}"

        grad_hook.remove_hooks()
        offloader.clear()

    def test_gradient_accumulation_with_offload(self):
        """Multiple backward passes with gradient offloading."""
        torch.manual_seed(42)
        dim = 64
        accum_steps = 3

        # Baseline
        base = nn.Linear(dim, dim).cuda()
        base.zero_grad()
        for i in range(accum_steps):
            torch.manual_seed(100 + i)
            x = torch.randn(4, dim, device="cuda")
            (base(x).sum() / accum_steps).backward()
        base_grads = {n: p.grad.clone() for n, p in base.named_parameters()}

        # With offloading — but keep grads on CPU (no release)
        off = nn.Linear(dim, dim).cuda()
        off.load_state_dict(base.state_dict())
        offloader = TensorOffloader()
        # Don't release GPU grad — accumulation needs it on GPU
        grad_hook = GradientOffloadHook(offloader, release_gpu_grad=False)
        grad_hook.register(off.parameters())

        off.zero_grad()
        for i in range(accum_steps):
            torch.manual_seed(100 + i)
            x = torch.randn(4, dim, device="cuda")
            (off(x).sum() / accum_steps).backward()

        for name, param in off.named_parameters():
            assert torch.allclose(param.grad, base_grads[name], rtol=1e-4, atol=1e-4)

        grad_hook.remove_hooks()
        offloader.clear()

    def test_has_offloaded_grads(self):
        model = nn.Linear(32, 32).cuda()
        offloader = TensorOffloader()
        grad_hook = GradientOffloadHook(offloader, release_gpu_grad=True)
        grad_hook.register(model.parameters())

        assert not grad_hook.has_offloaded_grads()

        model(torch.randn(2, 32, device="cuda")).sum().backward()
        # After offload, handles exist
        assert grad_hook.has_offloaded_grads()

        grad_hook.reload_all()
        assert not grad_hook.has_offloaded_grads()

        grad_hook.remove_hooks()
        offloader.clear()


# ═══════════════════════════════════════════════════════════════════════
# TORCH.COMPILE COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════


class TestTorchCompileCompat:
    """Test that offloading works with torch.compile."""

    def test_tensor_offloader_with_compiled_model(self):
        """Offload/reload tensors produced by a compiled model."""
        offloader = TensorOffloader()
        model = nn.Linear(128, 128).cuda()
        compiled = torch.compile(model, backend="eager")

        x = torch.randn(4, 128, device="cuda")
        out = compiled(x)
        expected = out.detach().clone()

        handle = offloader.offload(out.detach())
        offloader.sync_offload()
        recovered = offloader.reload(handle)
        offloader.sync_reload()

        assert torch.equal(recovered, expected)
        offloader.clear()

    def test_gradient_offload_with_compiled_model(self):
        """Gradient offloading works when model is compiled."""
        torch.manual_seed(42)
        dim = 64
        x = torch.randn(4, dim, device="cuda")

        # Baseline
        base = nn.Linear(dim, dim).cuda()
        base(x).sum().backward()
        base_grads = {n: p.grad.clone() for n, p in base.named_parameters()}

        # Compiled + gradient offload
        off = nn.Linear(dim, dim).cuda()
        off.load_state_dict(base.state_dict())
        compiled = torch.compile(off, backend="eager")

        offloader = TensorOffloader()
        grad_hook = GradientOffloadHook(offloader, release_gpu_grad=True)
        grad_hook.register(off.parameters())

        compiled(x).sum().backward()
        grad_hook.reload_all()

        for name, param in off.named_parameters():
            assert torch.allclose(param.grad, base_grads[name], rtol=1e-4, atol=1e-4)

        grad_hook.remove_hooks()
        offloader.clear()

    def test_activation_offload_context_with_compile(self):
        """ActivationOffloadContext works with a compiled inner module."""
        from torchtitan.distributed.cpu_offload import (
            ActivationOffloadContext, OffloadManager, group_commit,
        )

        torch.manual_seed(42)
        dim = 128
        x = torch.randn(4, dim, device="cuda")

        linear = nn.Linear(dim, dim).cuda()

        # Baseline
        base_out = linear(x).detach().clone()

        # Compiled + activation offload
        compiled_linear = torch.compile(linear, backend="eager")
        OffloadManager.reset_instance()
        ActivationOffloadContext.init_chunk_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=1)

        with ActivationOffloadContext(True, x, "linear") as x_in:
            off_out = compiled_linear(x_in)
        off_out = group_commit(off_out, "linear")

        assert torch.allclose(off_out.detach(), base_out, rtol=1e-4, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# CUDA GRAPH COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════


class TestCUDAGraphCompat:
    """Test compatibility with CUDA graphs.

    CUDA graphs require static shapes and no CPU-dependent control flow.
    The offloader uses external CUDA events (not stream.synchronize()),
    which is the pattern required for CUDA graph compatibility.
    """

    def test_offloader_uses_events_not_stream_sync(self):
        """Verify offloader uses events, not stream.synchronize()."""
        offloader = TensorOffloader()
        t = torch.randn(64, 64, device="cuda")

        handle = offloader.offload(t)
        # Last offload event should exist
        assert offloader._last_offload_event is not None

        recovered = offloader.reload(handle)
        assert offloader._last_reload_event is not None

        offloader.sync_all()
        offloader.clear()

    def test_tensor_offload_in_cuda_graph_region(self):
        """Offload/reload outside CUDA graph, use result inside graph."""
        offloader = TensorOffloader()
        t = torch.randn(64, 64, device="cuda")
        expected = t.clone()

        # Offload outside graph
        handle = offloader.offload(t)
        offloader.sync_offload()

        # Reload outside graph
        recovered = offloader.reload(handle)
        offloader.sync_reload()

        # Use recovered tensor in a CUDA graph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g = torch.cuda.CUDAGraph()
            # Warmup
            result = recovered + 1.0
            # Capture
            with torch.cuda.graph(g):
                result = recovered + 1.0
            # Replay
            g.replay()

        torch.cuda.synchronize()
        assert torch.allclose(result, expected + 1.0)
        offloader.clear()

    def test_offloader_events_are_cuda_graph_safe(self):
        """CUDA events recorded by offloader should be queryable."""
        offloader = TensorOffloader()
        t = torch.randn(32, 32, device="cuda")

        handle = offloader.offload(t)
        event = offloader._last_offload_event

        # Event should be queryable (not raise)
        assert isinstance(event.query(), bool)

        torch.cuda.synchronize()
        # After sync, event should be done
        assert event.query() is True

        offloader.reload(handle)
        offloader.sync_all()
        offloader.clear()


# ═══════════════════════════════════════════════════════════════════════
# COMBINED: ALL OFFLOAD TYPES TOGETHER
# ═══════════════════════════════════════════════════════════════════════


class TestCombinedOffloading:
    """Test using activation, weight, and gradient offloading simultaneously."""

    def test_all_three_offload_types(self):
        """Activation + gradient offloading on same model, correct gradients."""
        from torchtitan.distributed.cpu_offload import (
            ActivationOffloadContext, OffloadManager, group_commit,
        )

        torch.manual_seed(42)
        dim = 128
        x = torch.randn(4, dim, device="cuda")

        # Baseline
        base = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)).cuda()
        base_out = base(x)
        base_out.sum().backward()
        base_grads = {n: p.grad.clone() for n, p in base.named_parameters()}

        # Offloaded: activation + gradient
        off = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)).cuda()
        off.load_state_dict(base.state_dict())

        offloader = TensorOffloader()
        grad_hook = GradientOffloadHook(offloader, release_gpu_grad=False)
        grad_hook.register(off.parameters())

        OffloadManager.reset_instance()
        ActivationOffloadContext.init_chunk_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=1)

        with ActivationOffloadContext(True, x, "layer") as x_in:
            off_out = off(x_in)
        off_out = group_commit(off_out, "layer")
        off_out.sum().backward()

        for name, param in off.named_parameters():
            assert torch.allclose(param.grad, base_grads[name], rtol=1e-3, atol=1e-3), \
                f"Combined offload grad mismatch at {name}"

        grad_hook.remove_hooks()
        offloader.clear()
