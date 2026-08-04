import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from musubi_tuner.minimax_h3.comfy_quant import (
    ComfyInt8Embedding,
    ComfyNvfp4Linear,
    has_comfy_quantized_layers,
    load_comfy_quantized_state_dict,
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
