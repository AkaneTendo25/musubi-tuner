from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from musubi_tuner.minimax_h3 import residual_norm_native
from musubi_tuner.minimax_h3.model import _use_native_residual_boundary
from musubi_tuner.modules import convrot_int8_kernels, convrot_int8_native


cuda_device_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
requires_cuda_native = pytest.mark.skipif(
    not cuda_device_available or not convrot_int8_native.HAS_TRITON,
    reason="requires CUDA and Triton",
)
requires_cuda_residual = pytest.mark.skipif(
    not cuda_device_available or not residual_norm_native.HAS_TRITON or residual_norm_native._native_rms_ops() is None,
    reason="requires CUDA, Triton, and native fused RMSNorm ops",
)


def test_fast_transpose_cpu_uses_compatible_fallback():
    source = torch.arange(12, dtype=torch.int8).reshape(3, 4)

    result = convrot_int8_native.transpose_int8_contiguous_or_fallback(source)

    assert result.is_contiguous()
    assert torch.equal(result, source.t())


def test_scaled_quant_cpu_preserves_materialized_fallback(monkeypatch):
    calls = {}

    def fake_int8_lora_linear(x, weight, weight_scale, down, up, out_dtype):
        calls["x"] = x
        return torch.zeros((x.shape[0], weight.shape[0]), dtype=out_dtype)

    from musubi_tuner.modules import convrot_int8_kernels

    monkeypatch.setattr(convrot_int8_kernels, "int8_lora_linear", fake_int8_lora_linear)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    column_scale = torch.tensor([2.0, 0.5], dtype=torch.float32)
    weight = torch.zeros((3, 2), dtype=torch.int8)

    result = convrot_int8_native.scaled_int8_lora_linear_or_fallback(
        x,
        column_scale,
        weight,
        torch.ones(3),
        torch.zeros((2, 1), dtype=torch.bfloat16),
        torch.zeros((1, 3), dtype=torch.bfloat16),
        torch.bfloat16,
    )

    assert result.shape == (2, 3)
    assert torch.equal(calls["x"], x * column_scale.to(torch.bfloat16))


def test_residual_fusion_falls_back_for_shared_and_batched_cpu_indices():
    hidden = torch.zeros((2, 3, 4), dtype=torch.bfloat16)
    branch = torch.zeros_like(hidden)
    table = torch.zeros((5, 4), dtype=torch.bfloat16)
    weight = torch.ones(4, dtype=torch.bfloat16)

    for indices in (torch.tensor([0, 1, 2]), torch.tensor([[0, 1, 2], [2, 3, 4]])):
        assert (
            residual_norm_native.try_residual_norm_adaln_gated(hidden, branch, table, weight, table, table, indices, 1e-6) is None
        )


def test_flat_indices_supports_shared_and_batched_layouts():
    shared = torch.tensor([1, 2, 3])
    batched = torch.tensor([[1, 2, 3], [4, 5, 6]])

    assert torch.equal(residual_norm_native._flat_indices(shared, 6), torch.tensor([1, 2, 3, 1, 2, 3]))
    assert torch.equal(residual_norm_native._flat_indices(batched, 6), batched.reshape(-1))


def test_index_backward_reduces_repeated_rows_like_advanced_indexing():
    table = torch.zeros((3, 2), dtype=torch.bfloat16)
    indices = torch.tensor([0, 1, 0, 2])
    selected_grad = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.bfloat16)

    result = residual_norm_native._index_backward(table, indices, selected_grad)

    expected = torch.tensor([[6, 8], [3, 4], [7, 8]], dtype=torch.bfloat16)
    assert torch.equal(result, expected)


def test_index_backward_matches_shared_index_broadcast_reduction():
    table = torch.randn((4, 3), dtype=torch.bfloat16, requires_grad=True)
    indices = torch.tensor([1, 1, 2])
    selected_grad = torch.randn((2, 3, 3), dtype=torch.bfloat16)

    (table[indices].unsqueeze(0) * selected_grad).sum().backward()
    result = residual_norm_native._index_backward(table.detach(), indices, selected_grad.reshape(-1, 3))

    assert torch.equal(result, table.grad)


@requires_cuda_native
def test_fast_transpose_cuda_is_bitwise_equal_to_contiguous_transpose():
    source = torch.randint(-127, 128, (37, 53), device="cuda", dtype=torch.int8)

    result = convrot_int8_native.transpose_int8_contiguous_or_fallback(source)

    assert result.is_contiguous()
    assert torch.equal(result, source.t().contiguous())


@requires_cuda_native
def test_scaled_quant_cuda_matches_existing_quantizer_bitwise():
    torch.manual_seed(1234)
    value = torch.randn((19, 256), device="cuda", dtype=torch.bfloat16)
    column_scale = torch.randn(256, device="cuda", dtype=torch.float32).abs().add_(0.01)
    materialized = (value * column_scale.to(torch.bfloat16)).to(torch.bfloat16)
    expected_codes, expected_scales = convrot_int8_kernels.triton_quantize_rowwise(materialized)

    codes, scales = convrot_int8_native.scaled_quantize_rowwise_bf16(value, column_scale)

    assert torch.equal(codes, expected_codes)
    assert torch.equal(scales, expected_scales)


def _residual_reference(hidden, branch, gate, weight, shift, scale, indices, eps):
    residual = torch.addcmul(hidden, gate[indices], branch)
    normalized = F.rms_norm(residual, (hidden.shape[-1],), weight, eps)
    modulated = torch.addcmul(shift[indices], normalized, 1.0 + scale[indices])
    return residual, modulated


@requires_cuda_residual
@pytest.mark.parametrize("width", [5376, 7168])
@pytest.mark.parametrize("batched_indices", [False, True])
@pytest.mark.parametrize(
    "grad_mask",
    ["010000", "110000", "111111", "001000", "000100", "000010", "000001"],
)
def test_residual_native_cuda_matches_eager_outputs_and_required_gradients(width, batched_indices, grad_mask):
    torch.manual_seed(2026)
    batch, sequence, table_rows = 2, 3, 4
    shared = torch.tensor([1, 1, 2], device="cuda", dtype=torch.int64)
    indices = torch.stack((shared, torch.tensor([2, 1, 2], device="cuda"))) if batched_indices else shared

    def inputs():
        tensors = [
            torch.randn((batch, sequence, width), device="cuda", dtype=torch.bfloat16),
            torch.randn((batch, sequence, width), device="cuda", dtype=torch.bfloat16),
            torch.randn((table_rows, width), device="cuda", dtype=torch.bfloat16),
            torch.randn(width, device="cuda", dtype=torch.bfloat16),
            torch.randn((table_rows, width), device="cuda", dtype=torch.bfloat16),
            torch.randn((table_rows, width), device="cuda", dtype=torch.bfloat16),
        ]
        for tensor, required in zip(tensors, grad_mask, strict=True):
            tensor.requires_grad_(required == "1")
        return tensors

    candidate_inputs = inputs()
    reference_inputs = [value.detach().clone().requires_grad_(value.requires_grad) for value in candidate_inputs]
    candidate = residual_norm_native.try_residual_norm_adaln_gated(*candidate_inputs, indices, 1e-6)
    assert candidate is not None
    reference = _residual_reference(*reference_inputs, indices, 1e-6)
    grad_residual = torch.randn_like(candidate[0])
    grad_modulated = torch.randn_like(candidate[1])

    active_outputs = [index for index, output in enumerate(reference) if output.requires_grad]
    output_grads = (grad_residual, grad_modulated)
    torch.autograd.backward(
        [candidate[index] for index in active_outputs],
        [output_grads[index] for index in active_outputs],
    )
    torch.autograd.backward(
        [reference[index] for index in active_outputs],
        [output_grads[index] for index in active_outputs],
    )

    for actual, expected in zip(candidate, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    for required, actual, expected in zip(grad_mask, candidate_inputs, reference_inputs, strict=True):
        if required == "1":
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0.01, atol=0.02)
        else:
            assert actual.grad is None


def test_residual_boundary_compile_fullgraph_uses_fallback():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    hidden = torch.randn((2, 3, 4), dtype=torch.float32)
    branch = torch.randn_like(hidden)
    table = torch.randn((5, 4), dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    indices = torch.tensor([[0, 1, 2], [2, 3, 4]])

    def boundary(hidden, branch, gate, weight, shift, scale, indices):
        fused = residual_norm_native.try_residual_norm_adaln_gated(hidden, branch, gate, weight, shift, scale, indices, 1e-6)
        if fused is not None:
            return fused
        return _residual_reference(hidden, branch, gate, weight, shift, scale, indices, 1e-6)

    expected = boundary(hidden, branch, table, weight, table, table, indices)
    compiled = torch.compile(boundary, backend="eager", fullgraph=True)
    actual = compiled(hidden, branch, table, weight, table, table, indices)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize(
    "fused_elementwise,fused_indexed_adaln,index_ndim,expected",
    [
        (False, False, 2, False),
        (False, True, 1, False),
        (True, False, 1, True),
        (True, False, 2, True),
        (True, True, 1, False),
        (True, True, 2, True),
    ],
)
def test_native_residual_boundary_honors_existing_fusion_flags(fused_elementwise, fused_indexed_adaln, index_ndim, expected):
    indices = torch.zeros((3,) if index_ndim == 1 else (2, 3), dtype=torch.int64)

    assert _use_native_residual_boundary(fused_elementwise, fused_indexed_adaln, indices) is expected
