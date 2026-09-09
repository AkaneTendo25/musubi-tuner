from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LOSS_MASK_KEY,
    H3_ONE_FRAME_CONTROL_INDICES_KEY,
    H3_ONE_FRAME_TARGET_INDEX_KEY,
    logical_cache_key,
)
from musubi_tuner.minimax_h3.conditioning import MiniMaxH3ConditioningEncoder
from musubi_tuner.minimax_h3.integration import _NativeLatentEncoder, _NativeTrainingBackend
from musubi_tuner.minimax_h3.media import MediaAsset, MediaModality
from musubi_tuner.minimax_h3.video_vae import MiniMaxH3VideoDecoderModel


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def encode_image(self, pixels):
        return torch.full((1, 24, 1, 2, 2), float(pixels.mean()))

    def encode_reference(self, pixels, *, image):
        assert image
        return self.encode_image(pixels)


class SilenceEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0

    def encode(self, waveform):
        assert waveform.shape == (2, 1, 1600) and not waveform.any()
        self.calls += 1
        return torch.full((2, 32, 2), 0.125)


def test_two_token_image_decode_runs_one_temporal_chunk():
    calls = []

    def decode_clip(latents):
        calls.append(latents.shape[2])
        return torch.zeros(1, 3, latents.shape[2] * 4, 32, 32)

    decoder = SimpleNamespace(
        tokens_chunk_size=5,
        temporal_compression_ratio=4,
        token_drop=3,
        token_overlap=2,
        frame_pre_padding=3,
        frame_overlap=5,
        clip_length=17,
        _decode_clip=decode_clip,
    )
    pixels = MiniMaxH3VideoDecoderModel._decode_temporal(decoder, torch.zeros(1, 24, 2, 2, 2))
    assert calls == [7]
    assert pixels.shape == (1, 3, 5, 32, 32)


def test_one_frame_encoder_preserves_three_control_slots_and_silent_audio(tmp_path):
    paths = []
    for index, value in enumerate((20, 100, 230)):
        path = tmp_path / f"control_{index}.png"
        Image.new("RGBA", (48, 64), (value, value, value, 10)).save(path)
        paths.append(path)
    item = SimpleNamespace(
        item_key="target.png",
        content=np.zeros((32, 32, 3), dtype=np.uint8),
        bucket_size=(32, 32),
        h3_target_mode="video",
        h3_one_frame=True,
        h3_condition_paths=tuple(paths),
        h3_media_assets=(MediaAsset(Path("target.png"), MediaModality.IMAGE, "target"),),
    )
    silence_encoder = SilenceEncoder()
    encoder = _NativeLatentEncoder(ImageEncoder(), silence_encoder, torch.float32)
    encoder._encode_references = lambda item: {}
    (tensors,) = encoder.encode_latents([item])
    batch = {re.sub(r"_\d+x\d+x\d+$", "", logical_cache_key(key)): value for key, value in tensors.items()}
    assert batch["latents"].shape == (24, 1, 2, 2)
    assert batch["latents_audio"].shape == (2, 32, 2)
    assert (batch["latents_audio"] == 0.125).all()
    encoder.encode_latents([item])
    assert silence_encoder.calls == 1
    assert not batch[H3_AUDIO_LOSS_MASK_KEY].any()
    assert [float(batch[f"latents_cond_{i:03d}"].mean()) for i in range(3)] == sorted(
        float(batch[f"latents_cond_{i:03d}"].mean()) for i in range(3)
    )
    batch[H3_ONE_FRAME_TARGET_INDEX_KEY] = torch.tensor(24)
    batch[H3_ONE_FRAME_CONTROL_INDICES_KEY] = torch.tensor([96, 0, 48])
    backend = _NativeTrainingBackend(SimpleNamespace())
    target, indices, rows = backend._one_frame_cache(
        batch,
        latent_frames=1,
        latent_height=2,
        latent_width=2,
        patch_size=(1, 2, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert target == 24 and indices == (96, 0, 48)
    assert rows.shape == (3, 96)
    assert torch.equal(rows[:, 0], torch.tensor([batch[f"latents_cond_{i:03d}"].flatten()[0] for i in range(3)]))
    text_encoder = object.__new__(MiniMaxH3ConditioningEncoder)
    text_encoder.task = "fl2va"
    images = text_encoder._images_for_item(item)
    assert [image.size for image in images] == [(32, 32)] * 3
    assert [image.getpixel((0, 0)) for image in images] == [(value,) * 3 for value in (20, 100, 230)]


@pytest.mark.parametrize("change", ["gap", "legacy", "missing_time", "negative", "float_time", "video_target"])
def test_one_frame_cache_rejects_invalid_controls(change):
    batch = {
        H3_ONE_FRAME_TARGET_INDEX_KEY: torch.tensor(24),
        H3_ONE_FRAME_CONTROL_INDICES_KEY: torch.tensor([0, 48, 96]),
        **{f"latents_cond_{i:03d}": torch.zeros(24, 1, 2, 2) for i in range(3)},
    }
    frames = 1
    if change == "gap":
        batch["latents_cond_004"] = batch.pop("latents_cond_001")
    elif change == "legacy":
        batch["latents_first"] = torch.zeros(24, 1, 2, 2)
    elif change == "missing_time":
        del batch[H3_ONE_FRAME_TARGET_INDEX_KEY]
    elif change == "negative":
        batch[H3_ONE_FRAME_CONTROL_INDICES_KEY][0] = -1
    elif change == "float_time":
        batch[H3_ONE_FRAME_TARGET_INDEX_KEY] = torch.tensor(24.0)
    else:
        frames = 2
    with pytest.raises(ValueError):
        _NativeTrainingBackend(SimpleNamespace())._one_frame_cache(
            batch,
            latent_frames=frames,
            latent_height=2,
            latent_width=2,
            patch_size=(1, 2, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
