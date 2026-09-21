"""Regression checks for inference condition-noise stream aliasing."""

from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.inference import (
    H3EncodedReferences,
    _augment_keyframe_rows,
    _augment_reference_video_rows,
    denoise_ref2va,
)
from musubi_tuner.minimax_h3.packing import MiniMaxH3ReferenceGeometry, patchify_video_latents
from musubi_tuner.minimax_h3.training import H3ModelPrediction


def test_condition_noise_continues_initial_noise_stream() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    initial_noise = torch.randn((2, 3), generator=generator)
    conditions = _augment_keyframe_rows(torch.zeros(2, 3), rows_per_anchor=1, generator=generator)

    expected_generator = torch.Generator(device="cpu").manual_seed(17)
    torch.testing.assert_close(initial_noise, torch.randn((2, 3), generator=expected_generator))
    torch.testing.assert_close(conditions, 0.001 * torch.randn((2, 3), generator=expected_generator))
    assert not torch.equal(conditions, 0.001 * initial_noise)

    repeated = torch.Generator(device="cpu").manual_seed(17)
    torch.randn((2, 3), generator=repeated)
    torch.testing.assert_close(conditions, _augment_keyframe_rows(torch.zeros(2, 3), rows_per_anchor=1, generator=repeated))


def test_equal_shaped_references_receive_distinct_sequential_noise() -> None:
    references = (
        MiniMaxH3ReferenceGeometry(kind=0, num_latent_frames=1, latent_height=2, latent_width=2),
        MiniMaxH3ReferenceGeometry(kind=0, num_latent_frames=1, latent_height=2, latent_width=2),
    )
    rows = torch.zeros(2, 16)
    generator = torch.Generator(device="cpu").manual_seed(9)

    actual = _augment_reference_video_rows(rows, references, (1, 2, 2), generator)
    expected_generator = torch.Generator(device="cpu").manual_seed(9)
    expected = 0.001 * torch.cat(
        [torch.randn((1, 16), generator=expected_generator), torch.randn((1, 16), generator=expected_generator)]
    )

    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual[0], actual[1])


def test_no_reference_noise_does_not_advance_stream() -> None:
    generator = torch.Generator(device="cpu").manual_seed(5)
    empty = torch.empty(0, 16)

    assert _augment_reference_video_rows(empty, (), (1, 2, 2), generator) is empty
    torch.testing.assert_close(
        torch.randn((2, 3), generator=generator),
        torch.randn((2, 3), generator=torch.Generator(device="cpu").manual_seed(5)),
    )


class _SpyTransformer:
    def __init__(self) -> None:
        self.config = SimpleNamespace(in_channels=4, audio_in_channels=8, patch_size=(1, 2, 2), text_dim=12)
        self.video_rows: torch.Tensor | None = None

    def __call__(self, **kwargs) -> H3ModelPrediction:
        video_rows = kwargs["video_hidden_states"]
        audio_rows = kwargs["audio_hidden_states"]
        self.video_rows = video_rows.detach().cpu().clone()
        return H3ModelPrediction(torch.zeros_like(video_rows), torch.zeros_like(audio_rows))


def _run_ref2va_spy(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spy = _SpyTransformer()
    reference_geometry = MiniMaxH3ReferenceGeometry(kind=0, num_latent_frames=1, latent_height=2, latent_width=2)
    references = H3EncodedReferences(
        geometries=(reference_geometry, reference_geometry),
        video_rows=torch.zeros(2, 16),  # Encoded conditions stay on CPU, including for CUDA sampling.
        audio_rows=torch.empty(0, 8),
    )
    video, audio = denoise_ref2va(
        spy,
        {
            H3_TEXT_HIDDEN_KEY: torch.zeros(2, 12),
            H3_TEXT_TOKEN_TAGS_KEY: torch.ones(2, dtype=torch.long),
        },
        references,
        height=32,
        width=32,
        frame_count=5,
        num_inference_steps=2,
        generator=torch.Generator(device=device).manual_seed(31),
        device=device,
        show_progress=False,
    )
    assert spy.video_rows is not None
    return spy.video_rows, video.cpu(), audio.cpu()


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(torch.device("cuda"), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_denoise_routes_one_stream_through_initial_and_distinct_reference_noise(device: torch.device) -> None:
    observed_rows, video, audio = _run_ref2va_spy(device)

    expected_generator = torch.Generator(device=device).manual_seed(31)
    initial_video = torch.randn((1, 4, 2, 2, 2), generator=expected_generator, device=device)
    torch.randn((1, 2, 8, 8), generator=expected_generator, device=device)
    expected_conditions = (
        torch.cat([torch.randn((1, 16), generator=expected_generator, device=device).cpu() for _ in range(2)]) * 0.001
    )

    torch.testing.assert_close(observed_rows[0, :2], expected_conditions)
    torch.testing.assert_close(observed_rows[:, 2:], patchify_video_latents(initial_video, (1, 2, 2)).cpu())
    assert not torch.equal(observed_rows[0, 0], observed_rows[0, 1])
    assert not torch.equal(observed_rows[0, 0], observed_rows[0, 2] * 0.001)

    repeated_rows, repeated_video, repeated_audio = _run_ref2va_spy(device)
    torch.testing.assert_close(repeated_rows, observed_rows)
    torch.testing.assert_close(repeated_video, video)
    torch.testing.assert_close(repeated_audio, audio)
