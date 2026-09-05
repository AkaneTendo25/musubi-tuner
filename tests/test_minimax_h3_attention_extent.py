from __future__ import annotations

import torch

from musubi_tuner.minimax_h3.model import SDPA_INT32_EXTENT, materialize_wide_view, view_element_extent


def _fused_value_view(rows: int, inner: int) -> torch.Tensor:
    # The real path: value is the last third of a [B, S, 3 * inner] projection.
    fused = torch.empty(1, rows, 3 * inner)
    return fused.chunk(3, dim=-1)[2]


def test_view_extent_counts_the_fused_stride_not_the_element_count() -> None:
    value = _fused_value_view(rows=8, inner=4)
    assert value.numel() == 32
    # rows times the fused row stride, the span the SDPA backend indexes
    assert view_element_extent(value) == 8 * 12
    assert view_element_extent(torch.empty(0, 4)) == 0


def test_h3_scale_boundary_matches_the_upstream_threshold() -> None:
    inner = 56 * 128
    for batch, rows, wide in ((1, 99_864, False), (1, 99_865, True), (2, 49_932, False), (2, 49_933, True)):
        # meta tensors carry shape and strides without allocating the multi-GiB storage
        fused = torch.empty_strided((batch, rows, inner), (rows * 3 * inner, 3 * inner, 1), device="meta")
        assert (view_element_extent(fused) > SDPA_INT32_EXTENT) is wide


def test_materialize_copies_only_a_wide_view(monkeypatch) -> None:
    value = _fused_value_view(rows=8, inner=4)
    assert materialize_wide_view(value) is value
    monkeypatch.setattr("musubi_tuner.minimax_h3.model.SDPA_INT32_EXTENT", 16)
    copied = materialize_wide_view(value)
    assert copied is not value
    assert copied.is_contiguous()
    assert torch.equal(copied, value)
    assert materialize_wide_view(copied) is copied
