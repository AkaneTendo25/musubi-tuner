"""Tests for PinnedSlabPool acquire/release/warmup logic."""
import torch
import torch.nn as nn

from musubi_tuner.ltx_2.model.ltx2_custom_offloading_utils import PinnedSlabPool


def test_acquire_release_roundtrip():
    pool = PinnedSlabPool()
    shape = (4, 8)
    dtype = torch.float32
    # Manually insert a buffer
    buf = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    pool._pools[(shape, dtype)] = [buf]
    pool._total_bytes = buf.numel() * buf.element_size()

    # Acquire should return the buffer
    acquired = pool.acquire(shape, dtype)
    assert acquired is not None
    assert acquired.shape == torch.Size(shape)
    assert id(acquired) in pool._in_use

    # Pool should now be empty for this key
    assert pool.acquire(shape, dtype) is None

    # Release should return it
    pool.release(acquired)
    assert id(acquired) not in pool._in_use
    assert len(pool._pools[(shape, dtype)]) == 1


def test_acquire_or_alloc_fallback():
    pool = PinnedSlabPool()
    shape = (2, 3)
    dtype = torch.bfloat16

    # Pool is empty — should allocate on demand
    t = pool.acquire_or_alloc(shape, dtype)
    assert t.shape == torch.Size(shape)
    assert t.dtype == dtype
    assert t.is_pinned()


def test_warmup_with_simple_module():
    pool = PinnedSlabPool()

    # Create a minimal module to scan
    block = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 8))
    pool.warmup([block], num_buffers_per_shape=3)

    # Should have created entries for the unique parameter shapes
    assert len(pool._pools) > 0
    assert pool._total_bytes > 0

    # Each unique shape should have 3 buffers
    for key, buffers in pool._pools.items():
        assert len(buffers) == 3
        for buf in buffers:
            assert buf.is_pinned()
            shape, dtype = key
            assert buf.shape == torch.Size(shape)
            assert buf.dtype == dtype


def test_stats_property():
    pool = PinnedSlabPool()
    block = nn.Linear(4, 4)
    pool.warmup([block], num_buffers_per_shape=2)

    stats = pool.stats
    assert "PinnedSlabPool" in stats
    assert "free" in stats
    assert "in-use" in stats

    # Acquire one and check counts change
    shape = next(iter(pool._pools.keys()))
    t = pool.acquire(*shape)
    assert t is not None

    stats2 = pool.stats
    assert "1 in-use" in stats2


def test_warmup_empty_block_list():
    pool = PinnedSlabPool()
    pool.warmup([], num_buffers_per_shape=2)
    assert len(pool._pools) == 0
    assert pool._total_bytes == 0


def test_release_unknown_tensor_is_noop():
    pool = PinnedSlabPool()
    t = torch.randn(2, 2)
    # Should not raise
    pool.release(t)
    assert len(pool._in_use) == 0
