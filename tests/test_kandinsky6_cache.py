import numpy as np
import pytest
import torch
from torch import nn

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.kandinsky6_cache_latents import (
    audio_latent_frames_for_video,
    audio_samples_for_frames,
    encode_batch,
    first_control_frame,
    prepare_video_pixels,
)
from musubi_tuner.kandinsky6_cache_text_encoder_outputs import split_text_batch
from musubi_tuner.kandinsky6_generate_video import resolve_config


@pytest.mark.parametrize(
    "module",
    [
        "musubi_tuner.kandinsky6_cache_latents",
        "musubi_tuner.kandinsky6_cache_text_encoder_outputs",
        "musubi_tuner.kandinsky6_generate_video",
    ],
)
def test_cli_help_builds_without_argument_conflicts(module) -> None:
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize(
    "name",
    ["lite", "lite-distill", "lite-pretrain", "pro", "pro-distill", "pro-pretrain"],
)
def test_generate_resolves_bundled_config_names(name) -> None:
    path = resolve_config(name)
    assert path.name == f"{name}.yaml"
    assert path.parent.name == "checkpoints"
    assert path.is_file()


def test_generate_preserves_explicit_config_path(tmp_path) -> None:
    path = tmp_path / "custom.yaml"
    assert resolve_config(path) == path


def test_audio_window_is_aligned_to_audio_vae_grid() -> None:
    assert audio_samples_for_frames(1) == 2 * 1024
    assert audio_samples_for_frames(125) == 225 * 1024
    assert audio_samples_for_frames(125) % 1024 == 0
    assert audio_latent_frames_for_video(121) == 218
    with pytest.raises(ValueError, match="positive"):
        audio_samples_for_frames(0)


def test_prepare_video_pixels_uses_cthw_and_upstream_normalization() -> None:
    frames = np.zeros((2, 4, 6, 4), dtype=np.uint8)
    frames[..., 0] = 255
    result = prepare_video_pixels(frames)
    assert result.shape == (3, 2, 4, 6)
    assert torch.all(result[0] == 1)
    assert torch.all(result[1:] == -1)


def test_first_control_frame_collapses_repeated_video_dataset_image() -> None:
    repeated = np.stack(
        [np.full((4, 6, 3), value, dtype=np.uint8) for value in (17, 18, 19)],
        axis=0,
    )
    frame = first_control_frame(repeated)
    assert frame.shape == (4, 6, 3)
    assert np.all(frame == 17)


def test_first_control_frame_accepts_list_control_layout() -> None:
    first = np.full((4, 6, 3), 23, dtype=np.uint8)
    second = np.full((4, 6, 3), 42, dtype=np.uint8)
    frame = first_control_frame([first, second])
    assert frame.shape == (4, 6, 3)
    assert np.all(frame == 23)


def test_split_packed_text_batch_preserves_item_boundaries() -> None:
    text = torch.arange(5 * 3).reshape(5, 3)
    pooled = torch.randn(2, 4)
    rows = split_text_batch(text, pooled, torch.tensor([0, 2, 5]), None)
    assert [row[0].shape for row in rows] == [(2, 3), (3, 3)]
    assert all(row[2].dtype == torch.bool and row[2].all() for row in rows)


def test_split_padded_text_batch_drops_padding() -> None:
    text = torch.randn(2, 4, 3)
    pooled = torch.randn(2, 4)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    rows = split_text_batch(text, pooled, torch.tensor([0, 4, 8]), mask)
    assert [row[0].shape[0] for row in rows] == [2, 3]


def test_encode_batch_scales_audio_and_encodes_condition_separately(monkeypatch, tmp_path) -> None:
    class Posterior:
        def __init__(self, value):
            self.value = value

        def sample(self):
            return self.value

    class VideoVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = type("Config", (), {"scaling_factor": 2.0})()

        def encode(self, pixels):
            value = pixels.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1, 1)
            return type("Encoded", (), {"latent_dist": Posterior(value)})()

    class AudioVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.scaling_factor = 0.5

        @property
        def device(self):
            return self.anchor.device

        @property
        def dtype(self):
            return self.anchor.dtype

        def encode_audio(self, waveform):
            return type("Distribution", (), {"mean": torch.full((1, 3, 2), 4.0)})()

    saved = {}
    monkeypatch.setattr("musubi_tuner.dataset.cache_io.save_latent_cache_kandinsky6", lambda item, **kw: saved.update(kw))
    item = ItemInfo("x", "caption", (8, 8), frame_count=1, content=np.zeros((2, 8, 8, 3), dtype=np.uint8))
    item.control_content = np.stack(
        [np.full((8, 8, 3), 255, dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]
    )
    item.audio_content = torch.zeros(1, 2048)
    item.audio_present = True
    item.latent_cache_path = str(tmp_path / "x.safetensors")

    encode_batch(VideoVAE(), AudioVAE(), [item])

    assert saved["video_latent"].shape == (4, 2, 8, 8)
    assert torch.all(saved["image_latent"] == 2.0)
    assert torch.all(saved["audio_latent"] == 2.0)
    assert saved["metadata"]["audio_vae_scaling_factor"] == "0.5"


def test_encode_batch_image_adds_masked_one_frame_silence(monkeypatch, tmp_path) -> None:
    class Posterior:
        def sample(self):
            return torch.ones(1, 4, 1, 1, 1)

    class VideoVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = type("Config", (), {"scaling_factor": 2.0})()

        def encode(self, pixels):
            assert pixels.shape == (1, 3, 1, 8, 8)
            return type("Encoded", (), {"latent_dist": Posterior()})()

    class AudioVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.scaling_factor = 0.5
            self.waveform = None

        @property
        def device(self):
            return self.anchor.device

        @property
        def dtype(self):
            return self.anchor.dtype

        def encode_audio(self, waveform):
            self.waveform = waveform.detach().clone()
            return type("Distribution", (), {"mean": torch.ones(1, 3, 2)})()

    saved = {}
    monkeypatch.setattr("musubi_tuner.dataset.cache_io.save_latent_cache_kandinsky6", lambda item, **kw: saved.update(kw))
    item = ItemInfo("image", "caption", (8, 8), frame_count=None, content=np.zeros((8, 8, 3), dtype=np.uint8))
    item.latent_cache_path = str(tmp_path / "image.safetensors")
    audio_vae = AudioVAE()

    encode_batch(VideoVAE(), audio_vae, [item])

    assert saved["video_latent"].shape == (4, 1, 1, 1)
    assert saved["audio_latent"].shape == (3, 2)
    assert saved["audio_present"] is False
    assert audio_vae.waveform.shape == (1, 2048)
    assert not torch.any(audio_vae.waveform)


def test_encode_batch_video_still_requires_dataset_audio(monkeypatch, tmp_path) -> None:
    class VideoVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = type("Config", (), {"scaling_factor": 1.0})()

        def encode(self, pixels):
            posterior = type("Posterior", (), {"sample": lambda self: torch.ones(1, 4, 2, 1, 1)})()
            return type("Encoded", (), {"latent_dist": posterior})()

    class AudioVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.scaling_factor = 1.0

    item = ItemInfo("video", "caption", (8, 8), frame_count=5, content=np.zeros((5, 8, 8, 3), dtype=np.uint8))
    item.latent_cache_path = str(tmp_path / "video.safetensors")
    with pytest.raises(ValueError, match="audio-enabled video item"):
        encode_batch(VideoVAE(), AudioVAE(), [item])
