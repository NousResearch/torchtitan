#!/usr/bin/env python3
"""
Unit tests for batch size scheduler.

Usage:
    python docs/batch_size_scheduler/test_unit.py
"""

import sys


def test_constant_batch_size():
    """Test constant batch size scheduler."""
    from torchtitan.components.batch_size_scheduler import (
        BatchSizeState,
        ConstantBatchSize,
    )

    scheduler = ConstantBatchSize(batch_size=4096)

    # Should always return the same batch size
    for samples in [0, 1000, 1000000, 1000000000]:
        state = BatchSizeState(consumed_samples=samples)
        assert scheduler.get_batch_size(state) == 4096, f"Failed at {samples}"

    print("  ConstantBatchSize: PASSED")


def test_linear_rampup():
    """Test linear rampup scheduler."""
    from torchtitan.components.batch_size_scheduler import BatchSizeState, LinearRampup

    scheduler = LinearRampup(
        start_batch_size=1024, end_batch_size=4096, rampup_samples=1000000
    )

    # At 0%: should be start_batch_size
    state = BatchSizeState(consumed_samples=0)
    assert scheduler.get_batch_size(state) == 1024, "Failed at 0%"

    # At 50%: should be halfway
    state = BatchSizeState(consumed_samples=500000)
    expected = 1024 + 0.5 * (4096 - 1024)  # 2560
    assert scheduler.get_batch_size(state) == int(
        expected
    ), f"Failed at 50%: got {scheduler.get_batch_size(state)}, expected {int(expected)}"

    # At 100%: should be end_batch_size
    state = BatchSizeState(consumed_samples=1000000)
    assert scheduler.get_batch_size(state) == 4096, "Failed at 100%"

    # After 100%: should stay at end_batch_size
    state = BatchSizeState(consumed_samples=2000000)
    assert scheduler.get_batch_size(state) == 4096, "Failed after 100%"

    print("  LinearRampup: PASSED")


def test_increment_rampup():
    """Test increment rampup scheduler."""
    from torchtitan.components.batch_size_scheduler import (
        BatchSizeState,
        IncrementRampup,
    )

    scheduler = IncrementRampup(
        start_batch_size=1024,
        end_batch_size=4096,
        increment=1024,
        rampup_samples=1000000,
    )

    # 3 increments needed: 1024 -> 2048 -> 3072 -> 4096
    # samples_per_increment = 1000000 / 3 = 333333.33

    # At 0%: should be start_batch_size
    state = BatchSizeState(consumed_samples=0)
    assert scheduler.get_batch_size(state) == 1024, "Failed at 0%"

    # Just before first increment
    state = BatchSizeState(consumed_samples=333332)
    assert scheduler.get_batch_size(state) == 1024, "Failed just before first increment"

    # After first increment
    state = BatchSizeState(consumed_samples=333334)
    assert scheduler.get_batch_size(state) == 2048, "Failed after first increment"

    # At 100%: should be end_batch_size
    state = BatchSizeState(consumed_samples=1000000)
    assert scheduler.get_batch_size(state) == 4096, "Failed at 100%"

    # After 100%: should stay at end_batch_size
    state = BatchSizeState(consumed_samples=2000000)
    assert scheduler.get_batch_size(state) == 4096, "Failed after 100%"

    print("  IncrementRampup: PASSED")


def test_batch_size_manager():
    """Test batch size manager alignment and grad accum."""
    from torchtitan.components.batch_size_scheduler import (
        BatchSizeManager,
        BatchSizeState,
        LinearRampup,
    )

    scheduler = LinearRampup(
        start_batch_size=1024, end_batch_size=4096, rampup_samples=1000000
    )
    manager = BatchSizeManager(
        scheduler=scheduler, micro_batch_size=4, data_parallel_size=2
    )
    # unit = 4 * 2 = 8

    # Test alignment
    state = BatchSizeState(consumed_samples=0)
    assert manager.get_batch_size(state) == 1024, "Alignment failed"
    assert manager.get_grad_accum_steps(state) == 1024 // 8, "Grad accum failed"

    # Test did_change
    changed, old, new = manager.did_change(state)
    assert not changed, "Should not detect change on first call"
    assert old == new == 1024, "Old and new should be equal"

    # Move to 50%
    state = BatchSizeState(consumed_samples=500000)
    changed, old, new = manager.did_change(state)
    assert changed, "Should detect change"
    assert old == 1024, "Old should be 1024"
    assert new == 2560, "New should be 2560"

    print("  BatchSizeManager: PASSED")


def test_build_batch_size_manager():
    """Test factory function with config."""
    from dataclasses import dataclass

    from torchtitan.components.batch_size_scheduler import (
        BatchSizeState,
        build_batch_size_manager,
    )

    # Mock config
    @dataclass
    class MockBSConfig:
        mode: str = "linear"
        start_batch_size: int = 1024
        rampup_samples: int = 1000000
        increment: int = 0

    @dataclass
    class MockTrainingConfig:
        global_batch_size: int = 4096
        local_batch_size: int = 4

    @dataclass
    class MockJobConfig:
        batch_size_scheduler: MockBSConfig = None
        training: MockTrainingConfig = None

        def __post_init__(self):
            if self.batch_size_scheduler is None:
                self.batch_size_scheduler = MockBSConfig()
            if self.training is None:
                self.training = MockTrainingConfig()

    config = MockJobConfig()
    manager = build_batch_size_manager(config, dp_degree=2)

    state = BatchSizeState(consumed_samples=0)
    assert manager.get_batch_size(state) == 1024, "Factory failed"

    # Test constant mode (default)
    config = MockJobConfig(
        batch_size_scheduler=MockBSConfig(mode="constant", start_batch_size=0)
    )
    manager = build_batch_size_manager(config, dp_degree=2)

    state = BatchSizeState(consumed_samples=500000)
    assert manager.get_batch_size(state) == 4096, "Constant mode failed"

    print("  build_batch_size_manager: PASSED")


def main():
    print("=" * 50)
    print("Batch Size Scheduler Unit Tests")
    print("=" * 50)
    print()

    try:
        test_constant_batch_size()
        test_linear_rampup()
        test_increment_rampup()
        test_batch_size_manager()
        test_build_batch_size_manager()

        print()
        print("=" * 50)
        print("All tests PASSED!")
        print("=" * 50)
        return 0

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
