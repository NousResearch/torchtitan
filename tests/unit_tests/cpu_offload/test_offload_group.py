# Copyright (c) 2025 Nous Research. All rights reserved.

"""Tests for OffloadTensorGroup — tensor batching with CUDA events."""

import pytest
import torch

from torchtitan.distributed.cpu_offload.offload_group import OffloadTensorGroup


class TestOffloadTensorGroupBasic:
    """Tests that don't require CUDA."""

    def test_init_defaults(self):
        g = OffloadTensorGroup("test_group")
        assert g.name == "test_group"
        assert g.offload is True
        assert g.use_cpu_pool is True
        assert g.total_offload_bytes == 0
        assert g.total_tensor_count == 0

    def test_init_no_pool(self):
        g = OffloadTensorGroup("test", use_cpu_pool=False)
        assert g.use_cpu_pool is False

    def test_push_pop_tensor(self):
        g = OffloadTensorGroup("test")
        t = torch.randn(4, 8)
        tag = (1, 0)
        g.push_tensor(tag, t)
        recovered = g.pop_tensor(tag)
        assert recovered is t

    def test_push_pop_multiple(self):
        g = OffloadTensorGroup("test")
        tensors = {}
        for i in range(5):
            tag = (1, i)
            t = torch.randn(2, 3)
            g.push_tensor(tag, t)
            tensors[tag] = t

        for tag, expected in tensors.items():
            recovered = g.pop_tensor(tag)
            assert recovered is expected

    def test_push_overwrites(self):
        g = OffloadTensorGroup("test")
        tag = (1, 0)
        t1 = torch.randn(2, 2)
        t2 = torch.randn(2, 2)
        g.push_tensor(tag, t1)
        g.push_tensor(tag, t2)
        assert g.pop_tensor(tag) is t2

    def test_update_offload_info(self):
        g = OffloadTensorGroup("test")
        t1 = torch.randn(100, dtype=torch.float32)  # 400 bytes
        t2 = torch.randn(200, dtype=torch.float16)  # 400 bytes
        g.update_offload_info(t1)
        g.update_offload_info(t2)
        assert g.total_tensor_count == 2
        assert g.total_offload_bytes == 100 * 4 + 200 * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestOffloadTensorGroupCUDA:
    """Tests requiring CUDA for event operations."""

    def test_record_and_wait_offload_event(self):
        g = OffloadTensorGroup("test")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            _ = torch.randn(1000, 1000, device="cuda")
        g.record_offload_event(stream)
        # Default stream waits on the event
        g.wait_offload_event(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def test_record_and_wait_reload_event(self):
        g = OffloadTensorGroup("test")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            _ = torch.randn(1000, 1000, device="cuda")
        g.record_reload_event(stream)
        g.wait_reload_event(torch.cuda.current_stream())
        torch.cuda.synchronize()
