import torch
from torch import nn
from torch.nn import functional as F

from musubi_tuner.minimax_h3 import video_vae
from musubi_tuner.minimax_h3.video_vae import (
    MiniMaxH3VideoDecoderModel,
    MiniMaxH3VideoDecoderTransformerBlock,
    MiniMaxH3VideoEncoderModel,
)


def test_decoder_norm_fp32_weight_tracks_same_device_updates():
    norm = nn.RMSNorm(4).to(dtype=torch.bfloat16).requires_grad_(False)
    first = MiniMaxH3VideoDecoderTransformerBlock._f32_weight(norm)
    assert MiniMaxH3VideoDecoderTransformerBlock._f32_weight(norm) is first
    with torch.no_grad():
        norm.weight.fill_(2)
    second = MiniMaxH3VideoDecoderTransformerBlock._f32_weight(norm)
    torch.testing.assert_close(first, torch.ones(4))
    torch.testing.assert_close(second, torch.full((4,), 2.0))

    norm.load_state_dict({"weight": torch.full((4,), 3.0, dtype=torch.bfloat16)})
    third = MiniMaxH3VideoDecoderTransformerBlock._f32_weight(norm)
    torch.testing.assert_close(third, torch.full((4,), 3.0))


def test_decoder_norm_fp32_weight_keeps_trainable_gradient():
    norm = nn.RMSNorm(4).to(dtype=torch.bfloat16)
    MiniMaxH3VideoDecoderTransformerBlock._f32_weight(norm).sum().backward()
    torch.testing.assert_close(norm.weight.grad, torch.ones(4, dtype=torch.bfloat16))


class _Capture(nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.batch_sizes = []

    def forward(self, tensor):
        self.batch_sizes.append(tensor.shape[0])
        return self.operation(tensor)


def _model(cls):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.tile_sample_min_height = 32
    model.tile_sample_min_width = 32
    model.tile_sample_min_overlap_height = 16
    model.tile_sample_min_overlap_width = 16
    return model


def _single_tile(tiles, process, _batch_size):
    return [process(tile) for tile in tiles]


def test_encoder_spatial_tile_batch_matches_single_tile_for_multiple_samples(monkeypatch):
    model = _model(MiniMaxH3VideoEncoderModel)
    model.encoder = _Capture(lambda x: F.avg_pool3d(x, (1, 16, 16)))
    model.quant_conv = nn.Identity()
    pixels = torch.arange(2 * 1 * 1 * 64 * 64, dtype=torch.float32).reshape(2, 1, 1, 64, 64)

    original = video_vae._process_tiles
    monkeypatch.setattr(video_vae, "_process_tiles", _single_tile)
    reference = model._encode_clip(pixels)
    assert set(model.encoder.batch_sizes) == {2}
    assert len(model.encoder.batch_sizes) > 3

    monkeypatch.setattr(video_vae, "_process_tiles", original)
    model.encoder.batch_sizes.clear()
    actual = model._encode_clip(pixels)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert max(model.encoder.batch_sizes) == 4
    assert model.encoder.batch_sizes[-1] == 2


def test_decoder_spatial_tile_batch_matches_single_tile_for_multiple_samples(monkeypatch):
    model = _model(MiniMaxH3VideoDecoderModel)
    model.post_quant_conv = nn.Identity()
    model.decoder = _Capture(lambda x: x.repeat_interleave(16, dim=-2).repeat_interleave(16, dim=-1))
    latents = torch.arange(2 * 1 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 1, 4, 4)

    original = video_vae._process_tiles
    monkeypatch.setattr(video_vae, "_process_tiles", _single_tile)
    reference = model._decode_clip(latents)
    assert set(model.decoder.batch_sizes) == {2}
    assert len(model.decoder.batch_sizes) > 3

    monkeypatch.setattr(video_vae, "_process_tiles", original)
    model.decoder.batch_sizes.clear()
    actual = model._decode_clip(latents)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert max(model.decoder.batch_sizes) == 4
    assert model.decoder.batch_sizes[-1] == 2


def test_tile_batch_retries_one_tile_after_oom():
    class Limited(nn.Module):
        def __init__(self):
            super().__init__()
            self.attempts = []

        def forward(self, tensor):
            self.attempts.append(tensor.shape[0])
            if tensor.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("simulated tile batch OOM")
            return tensor * 2

    process = Limited()
    tiles = [torch.full((2, 1), index) for index in range(3)]
    actual = video_vae._process_tiles(tiles, process, 2)
    assert process.attempts == [4, 2, 2, 2]
    for index, output in enumerate(actual):
        torch.testing.assert_close(output, tiles[index] * 2)
