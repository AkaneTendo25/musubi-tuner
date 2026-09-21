"""Q/K normalization and RoPE must preserve the attention backend's dtype contract."""

import copy

import pytest
import torch

from musubi_tuner.minimax_h3 import model as h3_model
from musubi_tuner.modules import attention as shared_attention


class PromoteToFloat(torch.nn.Module):
    def forward(self, tensor):
        return tensor.float()


@pytest.mark.parametrize("rotary", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_attention_restores_qk_dtype_before_backend(monkeypatch, rotary, fused):
    layer = h3_model.MiniMaxH3Attention(32, 2, 16, 1e-6, attention_mode="flash").bfloat16()
    # Keep real RMSNorm parameters so the fused-path eligibility checks still work.
    monkeypatch.setattr(layer.q_norm, "forward", PromoteToFloat().forward)
    monkeypatch.setattr(layer.k_norm, "forward", PromoteToFloat().forward)
    layer.fused_qk_norm_rope = fused
    observed = []

    def capture(qkv, *, attn_params):
        observed.append(tuple(tensor.dtype for tensor in qkv))
        return qkv[2].flatten(2, 3)

    monkeypatch.setattr(h3_model, "musubi_attention", capture)
    embedding = (torch.ones(3, 12), torch.zeros(3, 12)) if rotary else None
    result = layer(torch.randn(1, 3, 32, dtype=torch.bfloat16), embedding)
    assert observed == [(torch.bfloat16,) * 3]
    assert result.dtype == torch.bfloat16


@pytest.mark.skipif(
    not torch.cuda.is_available() or shared_attention.flash_attn_func is None,
    reason="requires CUDA and FlashAttention 2",
)
def test_flash_bf16_forward_backward_matches_torch():
    torch.manual_seed(7)
    layer = h3_model.MiniMaxH3Attention(256, 2, 128, 1e-6, attention_mode="flash").cuda().bfloat16()
    layer.q_norm = PromoteToFloat()
    layer.k_norm = PromoteToFloat()
    reference = copy.deepcopy(layer)
    reference.attention_mode = "torch"
    inputs = torch.randn(1, 16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    angles = torch.randn(16, 96, device="cuda", dtype=torch.float32)
    rotary = angles.cos(), angles.sin()
    actual = layer(inputs, rotary)
    expected = reference(reference_inputs, rotary)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0.02, atol=0.02)
    torch.testing.assert_close(layer.qkv_proj.weight.grad, reference.qkv_proj.weight.grad, rtol=0.03, atol=0.03)
