import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner import kandinsky6_prequantize
from musubi_tuner.kandinsky6_prequantize import export_prequantized_checkpoint
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer


def _save_source(path):
    tensors = {
        "visual_embeddings.in_layer.weight": torch.arange(32, dtype=torch.bfloat16).reshape(2, 16),
        "out_layer.out_layer.weight": torch.arange(64, dtype=torch.bfloat16).reshape(64, 1),
        "visual_transformer_blocks.0.proj.weight": torch.arange(512, dtype=torch.float32).reshape(2, 256),
        "audio_text_transformer_blocks.0.proj.weight": torch.arange(128, dtype=torch.float16).reshape(2, 64),
        "visual_transformer_blocks.0.norm.weight": torch.arange(256, dtype=torch.float32),
    }
    save_file(tensors, str(path))
    return tensors


def test_export_roundtrip_uses_loader_policy_and_preserves_float_values(tmp_path, monkeypatch):
    source = tmp_path / "source.safetensors"
    original = _save_source(source)
    output = tmp_path / "prequantized.safetensors"
    monkeypatch.setattr(kandinsky6_prequantize, "inspect_checkpoint", lambda _path: ("pro", None))

    report = export_prequantized_checkpoint(source, output, quant_device="cpu")

    with safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata()
        assert handle.get_tensor("visual_transformer_blocks.0.proj.weight").dtype is torch.int8
        assert handle.get_tensor("visual_transformer_blocks.0.proj.weight_scale").dtype is torch.float32
        assert handle.get_tensor("audio_text_transformer_blocks.0.proj.weight").dtype is torch.int8
        assert torch.equal(
            handle.get_tensor("visual_transformer_blocks.0.norm.weight"),
            original["visual_transformer_blocks.0.norm.weight"].bfloat16(),
        )
        assert metadata["kandinsky6.variant"] == "pro"
        assert metadata["kandinsky6.checkpoint_format"] == "regular"
    assert "visual_transformer_blocks.0.proj.comfy_quant" in keys
    assert "audio_text_transformer_blocks.0.proj.comfy_quant" in keys
    assert report.groupsize_counts == {256: 1, 64: 1}
    assert report.output_bytes >= report.estimated_tensor_bytes

    quantizer = ConvRotInt8Quantizer(target_layer_keys=[])
    reloaded = quantizer.load_and_quantize([str(output)], calc_device=None)
    assert quantizer.module_groupsizes == {
        "audio_text_transformer_blocks.0.proj": 64,
        "visual_transformer_blocks.0.proj": 256,
    }
    with safe_open(output, framework="pt", device="cpu") as handle:
        assert torch.equal(
            reloaded["visual_transformer_blocks.0.proj.weight"], handle.get_tensor("visual_transformer_blocks.0.proj.weight")
        )
        assert torch.equal(
            reloaded["visual_transformer_blocks.0.proj.scale_weight"],
            handle.get_tensor("visual_transformer_blocks.0.proj.weight_scale"),
        )


def test_export_does_not_requantize_existing_comfy_triple(tmp_path, monkeypatch):
    source = tmp_path / "existing.safetensors"
    weight = torch.arange(512, dtype=torch.int16).reshape(2, 256).to(torch.int8)
    scale = torch.tensor([[1.25], [2.5]], dtype=torch.float32)
    payload = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()), dtype=torch.uint8
    )
    save_file(
        {
            "visual_embeddings.in_layer.weight": torch.zeros(2, 16, dtype=torch.bfloat16),
            "out_layer.out_layer.weight": torch.zeros(64, 1, dtype=torch.bfloat16),
            "visual_transformer_blocks.0.proj.weight": weight,
            "visual_transformer_blocks.0.proj.weight_scale": scale,
            "visual_transformer_blocks.0.proj.comfy_quant": payload,
        },
        str(source),
    )
    output = tmp_path / "copy.safetensors"
    monkeypatch.setattr(kandinsky6_prequantize, "inspect_checkpoint", lambda _path: ("pro", None))

    export_prequantized_checkpoint(source, output, quant_device="cpu")

    with safe_open(output, framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor("visual_transformer_blocks.0.proj.weight"), weight)
        assert torch.equal(handle.get_tensor("visual_transformer_blocks.0.proj.weight_scale"), scale)


def test_export_refuses_existing_destination_by_default(tmp_path, monkeypatch):
    source = tmp_path / "source.safetensors"
    _save_source(source)
    output = tmp_path / "exists.safetensors"
    output.write_bytes(b"keep")
    monkeypatch.setattr(kandinsky6_prequantize, "inspect_checkpoint", lambda _path: ("lite", None))

    with pytest.raises(FileExistsError):
        export_prequantized_checkpoint(source, output, quant_device="cpu")

    assert output.read_bytes() == b"keep"


def test_export_rejects_distilled_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "source.safetensors"
    _save_source(source)
    monkeypatch.setattr(kandinsky6_prequantize, "inspect_checkpoint", lambda _path: ("pro", 4))

    with pytest.raises(ValueError, match="regular.*PiFlow"):
        export_prequantized_checkpoint(source, tmp_path / "output.safetensors", quant_device="cpu")
