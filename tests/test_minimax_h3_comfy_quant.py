import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from musubi_tuner.minimax_h3.comfy_quant import (
    ComfyInt8Embedding,
    ComfyNvfp4Linear,
    _encode_e2m1,
    has_comfy_quantized_layers,
    load_comfy_quantized_state_dict,
    quantize_nvfp4_activations,
    swizzle_nvfp4_scales,
    unswizzle_nvfp4_scales,
)


def _marker(quant_format: str) -> torch.Tensor:
    return torch.tensor(list(json.dumps({"format": quant_format}).encode("utf-8")), dtype=torch.uint8)


def _swizzle_nvfp4_scales(values: torch.Tensor) -> torch.Tensor:
    rows, columns = values.shape
    assert rows % 128 == 0 and columns % 4 == 0
    row_blocks = rows // 128
    column_blocks = columns // 4
    blocked = values.reshape(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    blocked = blocked.reshape(row_blocks, column_blocks, 4, 32, 4)
    blocked = blocked.reshape(-1, 4, 32, 4).transpose(1, 2)
    return blocked.reshape(-1)


def test_unswizzle_nvfp4_scales_restores_row_major_order():
    row_major = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8)
    blocked = _swizzle_nvfp4_scales(row_major)

    restored = unswizzle_nvfp4_scales(blocked, 256, 8)

    assert torch.equal(restored, row_major)


def test_swizzle_nvfp4_scales_round_trips_with_padding():
    row_major = torch.arange(131 * 7, dtype=torch.float32).reshape(131, 7)

    blocked = swizzle_nvfp4_scales(row_major)
    restored = unswizzle_nvfp4_scales(blocked, 131, 7)

    assert blocked.shape == (256, 8)
    assert torch.equal(restored, row_major)


def test_e2m1_encoder_uses_expected_codes_and_saturates():
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0])

    positive = _encode_e2m1(values)
    negative = _encode_e2m1(-values)

    assert torch.equal(positive, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 7], dtype=torch.uint8))
    assert torch.equal(negative, positive | 8)

    ties = _encode_e2m1(torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]))
    assert torch.equal(ties, torch.tensor([0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8))


def test_nvfp4_activation_quantization_pads_rows_and_handles_zero():
    inputs = torch.zeros((17, 16), dtype=torch.bfloat16)

    packed, blocked_scales, tensor_scale, original_rows = quantize_nvfp4_activations(inputs)

    assert packed.shape == (32, 8)
    assert blocked_scales.shape == (128, 4)
    assert original_rows == 17
    assert tensor_scale.item() == 0.0
    assert not packed.any()
    assert not blocked_scales.float().any()


def test_int8_embedding_dequantizes_only_selected_rows():
    embedding = ComfyInt8Embedding(
        torch.tensor([[1, -2], [3, 4]], dtype=torch.int8),
        torch.tensor([[0.5], [2.0]], dtype=torch.float32),
        torch.bfloat16,
    )

    output = embedding(torch.tensor([[1, 0]]))

    assert output.dtype is torch.bfloat16
    assert torch.equal(output.float(), torch.tensor([[[6.0, 8.0], [0.5, -1.0]]]))


def test_nvfp4_linear_decodes_e2m1_codes_and_applies_awq_scale():
    source = nn.Linear(16, 2, bias=False)
    packed = torch.full((2, 8), 0x12, dtype=torch.uint8)
    blocked_scales = torch.ones(512, dtype=torch.float8_e4m3fn)
    linear = ComfyNvfp4Linear(
        source,
        packed,
        blocked_scales,
        torch.tensor(0.5, dtype=torch.float32),
        torch.full((16,), 2.0, dtype=torch.bfloat16),
        torch.bfloat16,
    )

    weight = linear.dequantize_weight(torch.float32)
    output = linear(torch.ones(1, 16, dtype=torch.bfloat16))

    assert torch.equal(weight[0], torch.tensor([0.25, 0.5] * 8))
    assert torch.equal(output.float(), torch.tensor([[12.0, 12.0]]))


def test_nvfp4_scaled_mm_path_is_opt_in_and_preserves_leading_dimensions(monkeypatch):
    source = nn.Linear(16, 2, bias=True)
    packed = torch.full((2, 8), 0x12, dtype=torch.uint8)
    blocked_scales = torch.ones(512, dtype=torch.float8_e4m3fn)
    calls = []

    def fake_scaled_mm(inputs, packed_weight, scales, tensor_scale, bias):
        calls.append((inputs.clone(), packed_weight, scales, tensor_scale, bias))
        return torch.full((inputs.shape[0], 2), 3.0, dtype=inputs.dtype)

    monkeypatch.setattr("musubi_tuner.minimax_h3.comfy_quant._nvfp4_scaled_mm", fake_scaled_mm)
    linear = ComfyNvfp4Linear(
        source,
        packed,
        blocked_scales,
        torch.tensor(0.5),
        torch.full((16,), 2.0, dtype=torch.bfloat16),
        torch.bfloat16,
        scaled_mm=True,
    )

    output = linear(torch.ones((2, 3, 16), dtype=torch.bfloat16))

    assert output.shape == (2, 3, 2)
    assert torch.equal(output, torch.full_like(output, 3.0))
    assert len(calls) == 1
    assert calls[0][0].shape == (6, 16)
    assert torch.equal(calls[0][0], torch.full((6, 16), 2.0, dtype=torch.bfloat16))


class _ToyConditioner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(2, 16)
        self.language_model.proj = nn.Linear(16, 2, bias=False)
        self.visual = nn.Linear(2, 2, bias=False)


def _write_toy_checkpoint(path: Path) -> None:
    save_file(
        {
            "model.embed_tokens.comfy_quant": _marker("int8_tensorwise"),
            "model.embed_tokens.weight": torch.tensor([[1] * 16, [2] * 16], dtype=torch.int8),
            "model.embed_tokens.weight_scale": torch.tensor([[0.5], [0.25]], dtype=torch.float32),
            "model.proj.comfy_quant": _marker("nvfp4"),
            "model.proj.weight": torch.full((2, 8), 0x12, dtype=torch.uint8),
            "model.proj.weight_scale": torch.ones(512, dtype=torch.float8_e4m3fn),
            "model.proj.weight_scale_2": torch.tensor(1.0, dtype=torch.float32),
            "visual.weight": torch.eye(2, dtype=torch.bfloat16),
        },
        str(path),
    )


def test_comfy_quantized_checkpoint_load_is_strict_and_runnable(tmp_path):
    checkpoint = tmp_path / "conditioner.safetensors"
    _write_toy_checkpoint(checkpoint)
    model = _ToyConditioner()

    count = load_comfy_quantized_state_dict(
        model,
        checkpoint,
        key_map=lambda prefix: "language_model" if prefix == "model" else prefix.replace("model.", "language_model.", 1),
        output_dtype=torch.bfloat16,
    )

    assert count == 2
    assert has_comfy_quantized_layers(checkpoint)
    assert isinstance(model.language_model.embed_tokens, ComfyInt8Embedding)
    assert isinstance(model.language_model.proj, ComfyNvfp4Linear)
    hidden = model.language_model.embed_tokens(torch.tensor([[0, 1]]))
    output = model.language_model.proj(hidden)
    assert output.shape == (1, 2, 2)
    assert torch.isfinite(output).all()
    assert torch.equal(model.visual.weight.float(), torch.eye(2))


def test_non_quantized_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "conditioner.safetensors"
    save_file({"visual.weight": torch.eye(2, dtype=torch.bfloat16)}, str(checkpoint))

    assert not has_comfy_quantized_layers(checkpoint)
    with pytest.raises(ValueError, match="does not contain Comfy quantized layers"):
        load_comfy_quantized_state_dict(
            _ToyConditioner(),
            checkpoint,
            key_map=lambda prefix: prefix,
            output_dtype=torch.bfloat16,
        )
