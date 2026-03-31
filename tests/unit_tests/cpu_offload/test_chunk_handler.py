# Copyright (c) 2025 Nous Research. All rights reserved.

"""Tests for ChunkOffloadHandler — core D2H/H2D copy engine."""

import pytest
import torch

from torchtitan.distributed.cpu_offload.chunk_handler import ChunkOffloadHandler
from torchtitan.distributed.cpu_offload.tensor_pool import TensorPool

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture
def handler():
    """Create a ChunkOffloadHandler with its own streams and pool."""
    pool = TensorPool(device="cpu", pin_memory=True)
    d2h = torch.cuda.Stream()
    h2d = torch.cuda.Stream()
    h = ChunkOffloadHandler(
        min_offloaded_tensor_size=1,
        cpu_tensor_pool=pool,
        d2h_stream=d2h,
        h2d_stream=h2d,
    )
    yield h
    pool.clear()


class TestOffloadReload:
    """Test the low-level offload/reload copy operations."""

    def test_offload_returns_correct_state(self, handler):
        gpu_tensor = torch.randn(64, 128, device="cuda")
        state = handler.offload(gpu_tensor, use_cpu_pool=True)
        dev, cpu_backup, used_pool = state
        assert dev == torch.device("cuda", torch.cuda.current_device())
        assert cpu_backup.device == torch.device("cpu")
        assert cpu_backup.is_pinned()
        assert used_pool is True
        assert cpu_backup.shape == (64, 128)
        torch.cuda.synchronize()

    def test_offload_preserves_data(self, handler):
        gpu_tensor = torch.randn(32, 64, device="cuda")
        expected = gpu_tensor.clone().cpu()
        state = handler.offload(gpu_tensor, use_cpu_pool=True)
        torch.cuda.synchronize()
        _, cpu_backup, _ = state
        assert torch.allclose(cpu_backup, expected)

    def test_offload_non_contiguous_tensor(self, handler):
        gpu_tensor = torch.randn(64, 128, device="cuda").t()  # non-contiguous
        assert not gpu_tensor.is_contiguous()
        state = handler.offload(gpu_tensor)
        torch.cuda.synchronize()
        _, cpu_backup, _ = state
        assert cpu_backup.shape == gpu_tensor.shape
        assert torch.allclose(cpu_backup, gpu_tensor.cpu())

    def test_offload_without_pool(self, handler):
        gpu_tensor = torch.randn(16, 16, device="cuda")
        state = handler.offload(gpu_tensor, use_cpu_pool=False)
        _, cpu_backup, used_pool = state
        assert used_pool is False
        assert cpu_backup.is_pinned()
        torch.cuda.synchronize()

    def test_reload_restores_to_gpu(self, handler):
        gpu_tensor = torch.randn(32, 64, device="cuda")
        expected = gpu_tensor.clone()
        state = handler.offload(gpu_tensor, use_cpu_pool=True)
        torch.cuda.synchronize()
        recovered = handler.reload(state)
        assert recovered.device.type == "cuda"
        assert recovered.shape == (32, 64)
        assert torch.allclose(recovered, expected)

    def test_offload_reload_roundtrip_exact(self, handler):
        """Full roundtrip: GPU -> CPU -> GPU, verify bitwise equality."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            gpu_tensor = torch.randn(128, 256, device="cuda", dtype=dtype)
            original = gpu_tensor.clone()
            state = handler.offload(gpu_tensor, use_cpu_pool=False)
            torch.cuda.synchronize()
            recovered = handler.reload(state)
            assert torch.equal(recovered, original), f"Roundtrip failed for {dtype}"

    def test_offload_reload_various_sizes(self, handler):
        """Test with different tensor sizes."""
        for size in [(1,), (8,), (64, 64), (128, 256), (1024, 512)]:
            t = torch.randn(*size, device="cuda")
            state = handler.offload(t, use_cpu_pool=False)
            torch.cuda.synchronize()
            r = handler.reload(state)
            assert torch.equal(r, t), f"Failed for size {size}"


class TestShouldOffloadTensor:
    """Test the tensor filtering logic."""

    def test_large_tensor_offloaded(self, handler):
        handler.min_offloaded_tensor_size = 100
        t = torch.randn(200, device="cuda")
        assert handler.should_offload_tensor(t) is True

    def test_small_tensor_skipped(self, handler):
        handler.min_offloaded_tensor_size = 1000
        t = torch.randn(100, device="cuda")
        assert handler.should_offload_tensor(t) is False

    def test_marked_not_offloadable(self, handler):
        handler.min_offloaded_tensor_size = 1
        t = torch.randn(1000, device="cuda")
        t.offloading_activation = False
        assert handler.should_offload_tensor(t) is False

    def test_no_attribute_defaults_to_offload(self, handler):
        handler.min_offloaded_tensor_size = 1
        t = torch.randn(1000, device="cuda")
        assert handler.should_offload_tensor(t) is True


class TestBulkOffloadReload:
    """Test bulk group offload/reload with proper stream synchronization."""

    def test_bulk_offload_and_reload_group(self, handler):
        """Full cycle: group_start -> tensor_push -> bulk_offload -> bulk_reload."""
        from torchtitan.distributed.cpu_offload.offload_group import OffloadTensorGroup

        handler.is_warmup = True
        handler.do_offload = True

        # Simulate on_group_start_forward
        handler._offloaded_group_index = 1
        group = OffloadTensorGroup("test_module", use_cpu_pool=False)
        handler.offload_groups.append(group)
        handler._max_group_size = 1
        handler._tensor_count_current_group = 0

        # Push tensors
        t1 = torch.randn(64, 128, device="cuda")
        t2 = torch.randn(32, 64, device="cuda")
        t1_clone = t1.clone()
        t2_clone = t2.clone()

        tag1 = (1, 0)
        tag2 = (1, 1)
        group.push_tensor(tag1, t1)
        group.push_tensor(tag2, t2)

        # Bulk offload
        handler._groups_to_offload.append(group)
        handler.d2h_stream.wait_stream(torch.cuda.current_stream())
        handler.bulk_offload_group()
        torch.cuda.synchronize()

        # Verify tensors were replaced with state tuples
        state1 = group._tensors.get(tag1) or group.pop_tensor(tag1)
        state2 = group._tensors.get(tag2) or group.pop_tensor(tag2)
        # Tensors should be on CPU now (stored as tuples)
        # Re-push for reload
        group.push_tensor(tag1, state1)
        group.push_tensor(tag2, state2)

        # Bulk reload
        handler._groups_to_reload.append(group)
        handler.h2d_stream.wait_stream(torch.cuda.current_stream())
        handler.bulk_reload_group()
        torch.cuda.synchronize()

        # Verify recovered tensors match originals
        r1 = group.pop_tensor(tag1)
        r2 = group.pop_tensor(tag2)
        assert torch.allclose(r1, t1_clone, atol=1e-6)
        assert torch.allclose(r2, t2_clone, atol=1e-6)

    def test_bulk_offload_skips_small_tensors(self, handler):
        from torchtitan.distributed.cpu_offload.offload_group import OffloadTensorGroup

        handler.is_warmup = True
        handler.min_offloaded_tensor_size = 1000  # large threshold

        handler._offloaded_group_index = 1
        group = OffloadTensorGroup("test", use_cpu_pool=False)
        handler.offload_groups.append(group)

        small_t = torch.randn(10, device="cuda")  # 10 elements < 1000
        tag = (1, 0)
        group.push_tensor(tag, small_t)

        handler._groups_to_offload.append(group)
        handler.d2h_stream.wait_stream(torch.cuda.current_stream())
        handler.bulk_offload_group()
        torch.cuda.synchronize()

        # Small tensor should NOT have been offloaded (still a tensor, not tuple)
        result = group.pop_tensor(tag)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cuda"
