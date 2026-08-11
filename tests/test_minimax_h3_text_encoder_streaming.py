from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from musubi_tuner.minimax_h3 import backend as h3_backend
from musubi_tuner.minimax_h3 import integration as h3_integration
from musubi_tuner.minimax_h3.conditioning import load_text_conditioner
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


def test_text_encoder_streaming_cli_defaults_are_disabled() -> None:
    cache_args = create_cache_parser().parse_args(["--dataset_config", "dataset.toml", "--text_encoder", "qwen.safetensors"])
    generate_args = create_generate_parser().parse_args(["--model", "h3.safetensors", "--prompt", "test", "--output", "output.mp4"])
    assert cache_args.h3_text_encoder_blocks_to_stream == 0
    assert generate_args.h3_text_encoder_blocks_to_stream == 0


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
    )

    assert captured["blocks_to_stream"] == 50


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
