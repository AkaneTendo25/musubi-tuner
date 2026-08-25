from __future__ import annotations

import copy

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from musubi_tuner.minimax_h3 import backend as h3_backend
from musubi_tuner.minimax_h3 import integration as h3_integration
from musubi_tuner.minimax_h3.comfy_quant import ComfyNvfp4Linear
from musubi_tuner.minimax_h3.conditioning import _load_online_nvfp4_text_conditioner, load_text_conditioner
from musubi_tuner.minimax_h3_cache_text_encoder_outputs import create_parser as create_cache_parser
from musubi_tuner.minimax_h3_generate_video import create_parser as create_generate_parser
from musubi_tuner.modules.custom_offloading_utils import ForwardOnlyBlockStreamer


class _FrozenBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width)
        self.register_buffer("scale", torch.ones((), dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.projection(self.norm(hidden_states)) * self.scale


class _ToyTextConditioner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(4, 16)
        self.language_model.proj = nn.Linear(16, 8, bias=True)
        self.visual = nn.Linear(16, 4, bias=True)


def test_text_encoder_streaming_cli_defaults_are_disabled() -> None:
    cache_args = create_cache_parser().parse_args(["--dataset_config", "dataset.toml", "--text_encoder", "qwen.safetensors"])
    generate_args = create_generate_parser().parse_args(["--model", "h3.safetensors", "--prompt", "test", "--output", "output.mp4"])
    assert cache_args.h3_text_encoder_blocks_to_stream == 0
    assert generate_args.h3_text_encoder_blocks_to_stream == 0
    assert not cache_args.h3_nvfp4_scaled_mm
    assert not generate_args.h3_nvfp4_scaled_mm


def test_online_nvfp4_is_available_to_cache_and_generation_clis() -> None:
    cache_args = create_cache_parser().parse_args(
        ["--dataset_config", "dataset.toml", "--text_encoder", "qwen.safetensors", "--text_encoder_quantization", "nvfp4"]
    )
    generate_args = create_generate_parser().parse_args(
        [
            "--model",
            "h3.safetensors",
            "--prompt",
            "test",
            "--output",
            "output.mp4",
            "--text_encoder_quantization",
            "nvfp4",
        ]
    )

    assert cache_args.text_encoder_quantization == "nvfp4"
    assert generate_args.text_encoder_quantization == "nvfp4"


def test_online_nvfp4_loader_quantizes_linears_and_preserves_other_parameters(tmp_path) -> None:
    model = _ToyTextConditioner()
    source = {
        key.replace("language_model.", "model."): value.detach().to(torch.bfloat16) for key, value in model.state_dict().items()
    }
    checkpoint = tmp_path / "qwen.safetensors"
    save_file(source, str(checkpoint))

    count = _load_online_nvfp4_text_conditioner(
        model,
        checkpoint,
        output_dtype=torch.bfloat16,
        quantize_device=torch.device("cpu"),
    )

    assert count == 2
    assert isinstance(model.language_model.proj, ComfyNvfp4Linear)
    assert isinstance(model.visual, ComfyNvfp4Linear)
    assert model.language_model.embed_tokens.weight.dtype is torch.bfloat16
    output = model.language_model.proj(model.language_model.embed_tokens(torch.tensor([[0, 1]])))
    assert output.shape == (1, 2, 8)
    assert torch.isfinite(output).all()


def test_nvfp4_scaled_mm_rejects_incompatible_quantization_before_loading() -> None:
    with pytest.raises(ValueError, match="requires --text_encoder_quantization nvfp4_awq"):
        load_text_conditioner(
            "missing",
            "missing",
            device="cpu",
            dtype=torch.bfloat16,
            quantization="none",
            nvfp4_scaled_mm=True,
        )


def test_nvfp4_scaled_mm_rejects_unsupported_device_before_loading() -> None:
    with pytest.raises(ValueError, match=r"requires PyTorch 2.10\+ and a Blackwell CUDA GPU"):
        load_text_conditioner(
            "missing",
            "missing",
            device="cpu",
            dtype=torch.bfloat16,
            quantization="nvfp4_awq",
            nvfp4_scaled_mm=True,
        )


def test_conditioning_backend_routes_explicit_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def create_encoder(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(h3_integration, "create_conditioning_encoder", create_encoder)
    h3_backend.create_conditioning_encoder(
        text_encoder="qwen.safetensors",
        tokenizer="tokenizer",
        task="t2va",
        device="cuda",
        dtype="bfloat16",
        blocks_to_stream=50,
        nvfp4_scaled_mm=True,
    )

    assert captured["blocks_to_stream"] == 50
    assert captured["nvfp4_scaled_mm"] is True


@pytest.mark.parametrize("blocks", [-1, 51])
def test_text_encoder_streaming_rejects_invalid_block_count(blocks: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 50"):
        load_text_conditioner("missing", "missing", device="cuda", dtype=torch.bfloat16, blocks_to_stream=blocks)


@pytest.mark.parametrize("quantization", ["int8", "nf4"])
def test_text_encoder_streaming_rejects_bitsandbytes_payloads(quantization: str) -> None:
    with pytest.raises(ValueError, match="does not support bitsandbytes"):
        load_text_conditioner(
            "missing",
            "missing",
            device="cuda",
            dtype=torch.bfloat16,
            quantization=quantization,
            blocks_to_stream=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA streams")
@pytest.mark.parametrize("blocks_to_stream", [2, 6])
def test_forward_only_block_streamer_matches_resident_forward(blocks_to_stream: int) -> None:
    torch.manual_seed(17)
    resident = nn.Sequential(*[_FrozenBlock(32) for _ in range(6)]).requires_grad_(False).eval().cuda()
    streamed = copy.deepcopy(resident).cpu()
    inputs = torch.randn(2, 5, 32, device="cuda")
    expected = resident(inputs)

    streamer = ForwardOnlyBlockStreamer(
        "test",
        list(streamed),
        blocks_to_stream,
        torch.device("cuda"),
        ring_size=2,
        use_pinned_memory=False,
    )
    streamer.prepare()
    actual_first = streamed(inputs)
    actual_second = streamed(inputs)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_first, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_second, expected, rtol=0, atol=0)
    assert len(streamer.ring_flat) == min(2, blocks_to_stream)
    streamer.close()
    for block_index in streamer.stream_indices:
        assert all(tensor.device.type == "cpu" for tensor in streamed[block_index].parameters())
        assert all(tensor.device.type == "cpu" for tensor in streamed[block_index].buffers())
