"""Correctness of the block-sparse attention against dense attention."""

import pytest
import torch

from musubi_tuner.minimax_h3.block_sparse_attention import BlockSparseConfig, block_sparse_attention, select_blocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs CUDA")


def _qkv(batch=1, heads=4, rows=512, dim=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, heads, rows, dim)
    return tuple(torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) for _ in range(3))


def test_full_fraction_matches_dense():
    """Keeping every block must reproduce dense attention, or the mask is wrong."""
    q, k, v = _qkv()
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    sparse = block_sparse_attention(q, k, v, BlockSparseConfig(block=128, kv_fraction=1.0))
    assert torch.allclose(dense, sparse, atol=2e-3, rtol=2e-3), (dense - sparse).abs().max().item()


def test_sparse_differs_and_stays_finite():
    q, k, v = _qkv()
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    sparse = block_sparse_attention(q, k, v, BlockSparseConfig(block=128, kv_fraction=0.25))
    assert torch.isfinite(sparse).all()
    assert not torch.allclose(dense, sparse, atol=1e-2)


def test_gradients_flow_through_retained_blocks():
    q, k, v = _qkv()
    q, k, v = (t.requires_grad_(True) for t in (q, k, v))
    block_sparse_attention(q, k, v, BlockSparseConfig(block=128, kv_fraction=0.5)).square().mean().backward()
    for name, t in (("query", q), ("key", k), ("value", v)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert t.grad.abs().sum() > 0, name


def test_rows_not_multiple_of_block():
    q, k, v = _qkv(rows=500)
    out = block_sparse_attention(q, k, v, BlockSparseConfig(block=128, kv_fraction=1.0))
    assert out.shape == q.shape
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(dense, out, atol=2e-3, rtol=2e-3)


def test_own_block_always_retained():
    q, k, v = _qkv(rows=1024)
    mask = select_blocks(q, k, BlockSparseConfig(block=128, kv_fraction=0.01, min_blocks=1))
    n = mask.shape[-2]
    diag = mask[..., torch.arange(n), torch.arange(n)]
    assert diag.all()


def test_share_heads_gives_one_selection_for_all_heads():
    q, k, v = _qkv()
    mask = select_blocks(q, k, BlockSparseConfig(block=128, kv_fraction=0.25, share_heads=True))
    assert (mask[:, 0] == mask[:, 1]).all()


def test_threshold_one_keeps_every_block():
    """Full score mass means dense attention, or the threshold rule is wrong."""
    q, k, v = _qkv()
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    sparse = block_sparse_attention(q, k, v, BlockSparseConfig(block=128, threshold=1.0))
    assert torch.allclose(dense, sparse, atol=2e-3, rtol=2e-3), (dense - sparse).abs().max().item()


def test_threshold_keeps_fewer_blocks_as_it_falls():
    q, k, v = _qkv(rows=2048)
    counts = []
    for threshold in (0.9, 0.5, 0.2):
        mask = select_blocks(q, k, BlockSparseConfig(block=128, threshold=threshold))
        counts.append(mask.sum(-1).float().mean().item())
    assert counts[0] > counts[1] > counts[2], counts


def test_threshold_adapts_per_query_block():
    """A fixed share cannot vary per query block; the threshold rule must."""
    q, k, v = _qkv(rows=2048)
    mask = select_blocks(q, k, BlockSparseConfig(block=128, threshold=0.5))
    per_block = mask.sum(-1).flatten()
    assert per_block.min() != per_block.max(), per_block.unique()
