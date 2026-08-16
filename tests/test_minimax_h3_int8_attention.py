from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

import musubi_tuner.minimax_h3.int8_attention as int8_attention_module
from musubi_tuner.minimax_h3.int8_attention import HAS_TRITON, int8_attention
from musubi_tuner.minimax_h3.model import MiniMaxH3Attention, MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser


def _tiny_config() -> MiniMaxH3TransformerConfig:
    return MiniMaxH3TransformerConfig(
        hidden_size=16,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        num_layers=1,
        num_refiner_layers=1,
        patch_size=(1, 1, 1),
        in_channels=2,
        audio_in_channels=2,
        text_dim=16,
        rope_freq_dim=1,
    )


def test_int8_attention_is_disabled_by_default_and_context_restores_state(monkeypatch):
    monkeypatch.setattr("musubi_tuner.minimax_h3.model.HAS_TRITON", True)
    model = MiniMaxH3Transformer(_tiny_config())
    modules = [module for module in model.modules() if isinstance(module, MiniMaxH3Attention)]
    assert modules and not any(module.int8_attention for module in modules)

    model.set_int8_attention_mode("aux")
    assert not any(module.int8_attention for module in modules)
    with model.int8_attention_context(auxiliary=True):
        assert all(module.int8_attention for module in modules)
    assert not any(module.int8_attention for module in modules)

    model.set_int8_attention_mode("train")
    assert all(module.int8_attention for module in modules)
    with model.int8_attention_context(auxiliary=False):
        assert all(module.int8_attention for module in modules)
    assert all(module.int8_attention for module in modules)


def test_int8_attention_cli_is_explicitly_opt_in():
    assert create_parser().parse_args(["--sdpa"]).h3_int8_attention == "off"
    assert create_parser().parse_args(["--sdpa", "--h3_int8_attention", "aux"]).h3_int8_attention == "aux"
    assert create_parser().parse_args(["--sdpa", "--h3_int8_attention", "train"]).h3_int8_attention == "train"


def test_int8_attention_rejects_compile_until_the_custom_autograd_op_is_traceable():
    args = create_parser().parse_args(["--sdpa", "--h3_int8_attention", "train", "--compile"])
    with pytest.raises(ValueError, match="cannot currently be combined with --compile"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
@pytest.mark.parametrize("sequence", [31, 37, 65, 257])
def test_int8_attention_forward_and_backward_are_close_to_sdpa(sequence: int):
    torch.manual_seed(1234)
    shape = (1, 2, sequence, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    actual = int8_attention(query, key, value)
    actual.backward(grad)
    actual_gradients = tuple(tensor.grad.detach().float().clone() for tensor in (query, key, value))

    references = tuple(tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value))
    expected = F.scaled_dot_product_attention(*references, dropout_p=0.0, is_causal=False)
    expected.backward(grad)
    expected_gradients = tuple(tensor.grad.detach().float() for tensor in references)

    assert torch.testing.assert_close(actual.float(), expected.float(), rtol=4e-2, atol=4e-2) is None
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        cosine = F.cosine_similarity(actual_gradient.flatten(), expected_gradient.flatten(), dim=0)
        assert float(cosine) > 0.995


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
def test_h3_attention_module_propagates_projection_and_input_gradients():
    torch.manual_seed(4321)
    reference = MiniMaxH3Attention(256, 2, 128, 1.0e-5).cuda().bfloat16()
    quantized = copy.deepcopy(reference)
    quantized.int8_attention = True
    reference_input = torch.randn(1, 37, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    quantized_input = reference_input.detach().clone().requires_grad_(True)
    gradient = torch.randn_like(reference_input)

    reference_output = reference(reference_input)
    quantized_output = quantized(quantized_input)
    reference_output.backward(gradient)
    quantized_output.backward(gradient)

    assert torch.testing.assert_close(quantized_output.float(), reference_output.float(), rtol=5e-2, atol=5e-2) is None
    assert float(F.cosine_similarity(quantized_input.grad.float().flatten(), reference_input.grad.float().flatten(), dim=0)) > 0.99
    assert quantized.qkv_proj.weight.grad is not None
    assert quantized.out_proj.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
def test_native_backward_fallback_propagates_gradients(monkeypatch):
    monkeypatch.setattr(int8_attention_module, "_flash_backward_supported", lambda _query: False)
    tensors = tuple(torch.randn(1, 2, 37, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    int8_attention(*tensors).square().mean().backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in tensors)
