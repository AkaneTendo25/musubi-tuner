from __future__ import annotations

from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_CONDITIONING_TASK_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    H3_VIDEO_GEOMETRY_KEY,
)
from musubi_tuner.minimax_h3.slider import (
    H3SliderConfig,
    H3SliderDataset,
    H3SliderTarget,
    load_h3_slider_config,
    slider_direction_targets,
)
from musubi_tuner.minimax_h3.training import H3JointNoisyInputs, H3ModelPrediction, contrastive_guidance_target
from musubi_tuner.minimax_h3_train_slider import MiniMaxH3SliderTrainer, create_parser
from musubi_tuner.training.resume_utils import _restore_dataset_seed


def _write_cache(directory, stem: str, *, frames: int = 1, audio_frames: int | None = None, value: float = 0.0):
    directory.mkdir(parents=True, exist_ok=True)
    latent = {f"latents_{frames}x4x6_float32": torch.full((24, frames, 4, 6), value)}
    if audio_frames is not None:
        latent[f"latents_audio_2x32x{audio_frames}_float32"] = torch.full((2, 32, audio_frames), value)
        latent["audio_loss_mask"] = torch.ones(audio_frames, dtype=torch.bool)
    save_file(latent, directory / f"{stem}_0064x0096_mmh3.safetensors")
    save_file(
        {
            f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16": torch.zeros(3, 5120, dtype=torch.bfloat16),
            f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": torch.full((3,), 3, dtype=torch.long),
            H3_CONDITIONING_TASK_KEY: torch.tensor(0, dtype=torch.long),
        },
        directory / f"{stem}_mmh3_te.safetensors",
    )


def test_slider_direction_targets_are_symmetric_and_normalized():
    positive = torch.tensor([[[[4.0, 0.0]]]])
    neutral = torch.tensor([[[[10.0, 14.0]]]])
    negative = torch.tensor([[[[0.0, 4.0]]]])
    enhance, erase = slider_direction_targets(positive, neutral, negative, 2.0)
    assert torch.allclose((enhance + erase) / 2, torch.full_like(neutral, neutral.mean()))
    assert enhance.mean() == pytest.approx(neutral.mean())
    assert erase.mean() == pytest.approx(neutral.mean())
    assert enhance.std(correction=0) == pytest.approx(neutral.std(correction=0))
    assert erase.std(correction=0) == pytest.approx(neutral.std(correction=0))


def test_load_text_slider_config(tmp_path):
    path = tmp_path / "slider.toml"
    path.write_text(
        """
mode = "text"
target_modality = "video"
latent_frames = 1
latent_height = 4
latent_width = 6
sample_slider_range = [-1, 0, 1]

[[targets]]
positive = "detailed"
negative = "blurry"
target_class = "a landscape"
weight = 0.5

[[anchors]]
prompt = "a plain white wall"
""",
        encoding="utf-8",
    )
    config = load_h3_slider_config(path)
    assert config.mode == "text"
    assert config.target_modality == "video"
    assert config.targets == (H3SliderTarget("detailed", "blurry", "a landscape", 0.5),)
    assert config.anchors[0].prompt == "a plain white wall"
    assert config.sample_slider_range == (-1.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("modality", "audio_frames", "expected"),
    [
        ("video", None, {"latents"}),
        ("audio", 5, {H3_AUDIO_LATENTS_KEY}),
        ("av", 5, {"latents", H3_AUDIO_LATENTS_KEY}),
    ],
)
def test_paired_slider_dataset_supports_video_audio_and_av(tmp_path, modality, audio_frames, expected):
    positive = tmp_path / "positive"
    negative = tmp_path / "negative"
    _write_cache(positive, "clip", frames=2, audio_frames=audio_frames, value=1.0)
    _write_cache(negative, "clip", frames=2, audio_frames=audio_frames, value=-1.0)
    dataset = H3SliderDataset(
        H3SliderConfig(
            mode="reference",
            target_modality=modality,
            positive_cache_dir=str(positive),
            negative_cache_dir=str(negative),
        )
    )
    item = dataset[0]
    assert expected <= set(item["positive"])
    assert expected <= set(item["negative"])
    assert (set(item["positive"]) & {"latents", H3_AUDIO_LATENTS_KEY}) == expected
    assert torch.all(item["positive"][next(iter(expected))] >= 0)
    assert torch.all(item["negative"][next(iter(expected))] <= 0)
    assert item["positive"][H3_TEXT_HIDDEN_KEY].data_ptr() == item["negative"][H3_TEXT_HIDDEN_KEY].data_ptr()
    if modality == "audio":
        assert torch.equal(item["positive"][H3_VIDEO_GEOMETRY_KEY], torch.tensor([[4, 6]]))


def test_paired_image_slider_uses_single_video_latent_frame(tmp_path):
    positive = tmp_path / "positive"
    negative = tmp_path / "negative"
    _write_cache(positive, "image", frames=1, value=1.0)
    _write_cache(negative, "image", frames=1, value=-1.0)
    dataset = H3SliderDataset(
        H3SliderConfig(
            mode="reference",
            target_modality="video",
            positive_cache_dir=str(positive),
            negative_cache_dir=str(negative),
        )
    )
    assert dataset[0]["positive"]["latents"].shape == (1, 24, 1, 4, 6)


def test_ref2va_slider_takes_shared_conditioning_from_third_cache(tmp_path):
    positive = tmp_path / "positive"
    negative = tmp_path / "negative"
    shared = tmp_path / "shared"
    _write_cache(positive, "clip", frames=2, value=1.0)
    _write_cache(negative, "clip", frames=2, value=-1.0)
    _write_cache(shared, "clip", frames=2, value=7.0)
    dataset = H3SliderDataset(
        H3SliderConfig(
            mode="ref2va",
            target_modality="video",
            positive_cache_dir=str(positive),
            negative_cache_dir=str(negative),
            conditioning_cache_dir=str(shared),
        )
    )
    item = dataset[0]
    assert torch.all(item["positive"]["latents"] == 1)
    assert torch.all(item["negative"]["latents"] == -1)
    assert item["positive"][H3_TEXT_HIDDEN_KEY].data_ptr() == item["negative"][H3_TEXT_HIDDEN_KEY].data_ptr()


def test_text_slider_dataset_synthesizes_requested_modalities():
    dataset = H3SliderDataset(
        H3SliderConfig(
            mode="text",
            target_modality="av",
            targets=(H3SliderTarget("bright", "dark"),),
            latent_frames=2,
            latent_height=4,
            latent_width=6,
            audio_latent_frames=5,
        )
    )
    item = dataset[0]
    assert item["latents"].shape == (1, 24, 2, 4, 6)
    assert item[H3_AUDIO_LATENTS_KEY].shape == (1, 2, 32, 5)


def test_slider_parser_is_dedicated_and_does_not_change_normal_h3_parser(tmp_path):
    slider = tmp_path / "slider.toml"
    slider.write_text('mode="text"\n[[targets]]\npositive="a"\nnegative="b"\n', encoding="utf-8")
    args = create_parser().parse_args(["--slider_config", str(slider)])
    assert args.slider_config == str(slider)


def test_slider_dataset_group_seed_walk_terminates():
    dataset = H3SliderDataset(
        H3SliderConfig(
            mode="text",
            target_modality="video",
            targets=(H3SliderTarget("warm", "cold"),),
        )
    )
    _restore_dataset_seed(dataset, 123)
    assert dataset.datasets[0].get_metadata() == dataset.get_metadata()


def test_text_slider_plain_prediction_reset_does_not_assign_derived_properties():
    trainer = MiniMaxH3SliderTrainer()
    trainer._prepare_plain_prediction()
    assert trainer._active_extension_video_frames == 0
    assert trainer._active_extension_audio_latents == 0


@pytest.mark.parametrize("probability", [1.0, 0.5])
def test_text_slider_reuses_regular_contrastive_guidance_formula(probability):
    trainer = MiniMaxH3SliderTrainer()
    trainer._guidance_scale_range = None
    args = Namespace(
        h3_loss_balance="token",
        h3_video_loss_weight=1.0,
        h3_audio_loss_weight=1.0,
        h3_guidance_distillation_scale=3.5,
        h3_guidance_distillation_probability=probability,
        h3_guidance_loss_schedule="constant",
        h3_guidance_loss_form="contrastive",
        h3_guidance_cfg_zero=False,
    )
    sigma = torch.tensor([0.5])
    zeros = torch.zeros(1, 1, 1, 1, 2)
    inputs = H3JointNoisyInputs(zeros, None, zeros, None, sigma, sigma, sigma, sigma)
    prediction = H3ModelPrediction(torch.tensor([[[[[1.0, 2.0]]]]]), None)
    empty = H3ModelPrediction(torch.tensor([[[[[0.25, 0.5]]]]]), None)
    target = H3ModelPrediction(torch.tensor([[[[[2.0, 4.0]]]]]), None)

    actual = trainer._guided_slider_loss(args, None, prediction, target, empty, inputs)
    guided_target = contrastive_guidance_target(target, empty, 3.5)
    plain = trainer._slider_loss(args, prediction, target)
    guided = trainer._slider_loss(args, prediction, guided_target)
    expected = guided if probability == 1.0 else plain + (guided - plain) / probability

    assert torch.allclose(actual, expected)


class _FakeNetwork:
    def __init__(self):
        self.multipliers = []

    def set_multiplier(self, value):
        self.multipliers.append(value)


class _FakeAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(value):
        return value

    def backward(self, loss):
        loss.backward()


def test_paired_step_uses_opposite_multipliers_and_shared_rng(monkeypatch):
    trainer = MiniMaxH3SliderTrainer()
    trainer.slider_config = H3SliderConfig(
        mode="reference", target_modality="video", positive_cache_dir="p", negative_cache_dir="n"
    )
    network = _FakeNetwork()
    draws = []

    def fake_step(*args, **kwargs):
        draws.append(float(torch.rand(())))
        parameter = torch.tensor(1.0, requires_grad=True)
        return parameter.square(), {"loss/video": 1.0}

    monkeypatch.setattr(trainer, "_process_single_batch", fake_step)
    positive = {"latents": torch.zeros(1, 24, 2, 4, 6)}
    negative = {"latents": torch.ones(1, 24, 2, 4, 6)}
    loss, metrics = trainer._process_paired_batch(
        Namespace(),
        _FakeAccelerator(),
        object(),
        network,
        {"positive": positive, "negative": negative},
        torch.zeros_like(positive["latents"]),
        object(),
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )
    assert network.multipliers == [1.0, -1.0, 1.0]
    assert draws[0] == draws[1]
    assert loss.item() == pytest.approx(1.0)
    assert metrics["loss/slider_positive"] == pytest.approx(1.0)
    assert trainer._batch_backward_performed


def test_paired_step_forwards_shared_guidance_and_preservation_draws(monkeypatch):
    trainer = MiniMaxH3SliderTrainer()
    trainer.slider_config = H3SliderConfig(
        mode="reference", target_modality="video", positive_cache_dir="p", negative_cache_dir="n"
    )
    network = _FakeNetwork()
    overrides = []

    monkeypatch.setattr(trainer, "_base_preservation_active", lambda *_: True)
    monkeypatch.setattr(trainer, "_guidance_distillation_active", lambda *_: True)

    def fake_step(*args, **kwargs):
        overrides.append((kwargs["guidance_active_override"], kwargs["preservation_active_override"]))
        parameter = torch.tensor(1.0, requires_grad=True)
        return parameter.square(), {"loss/video": 1.0}

    monkeypatch.setattr(trainer, "_process_single_batch", fake_step)
    positive = {"latents": torch.zeros(1, 24, 2, 4, 6)}
    negative = {"latents": torch.ones(1, 24, 2, 4, 6)}
    args = Namespace(
        h3_guidance_distillation_scale=3.5,
        h3_guidance_distillation_probability=0.5,
        h3_base_preservation_loss_weight=0.02,
        h3_base_preservation_probability=0.25,
    )

    trainer._process_paired_batch(
        args,
        _FakeAccelerator(),
        object(),
        network,
        {"positive": positive, "negative": negative},
        torch.zeros_like(positive["latents"]),
        object(),
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    assert overrides == [(True, True), (True, True)]
