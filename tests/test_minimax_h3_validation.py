import math
import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import musubi_tuner.minimax_h3_train_network as h3_train_network
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_AUDIO_LOSS_MASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_REFERENCE_MODALITY_PROBABILITIES_KEY,
)
from musubi_tuner.minimax_h3.training import H3ModelPrediction, shift_sigma
from musubi_tuner.minimax_h3.validation import (
    H3ValidationAccumulator,
    image_flow_shift,
    image_validation_sigma,
    masked_squared_error_sum,
    preserve_rng_state,
    validation_sigma_bins,
)
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
from musubi_tuner.training.validation import derive_validation_seed


def test_validation_sigma_bins_are_base_coordinate_midpoints_with_exact_modality_shifts():
    bins = validation_sigma_bins(4, minimum=0.2, maximum=0.6)
    base = torch.tensor([0.25, 0.35, 0.45, 0.55], dtype=torch.float64)

    assert [item.index for item in bins] == [0, 1, 2, 3]
    torch.testing.assert_close(torch.tensor([item.base_sigma for item in bins], dtype=torch.float64), base)
    torch.testing.assert_close(torch.tensor([item.video_sigma for item in bins], dtype=torch.float64), shift_sigma(base, 12.0))
    torch.testing.assert_close(torch.tensor([item.audio_sigma for item in bins], dtype=torch.float64), shift_sigma(base, 3.0))


def test_image_validation_sigma_matches_resolution_aware_and_explicit_training_math():
    base = torch.tensor([0.25, 0.75], dtype=torch.float64)
    expected_shift = torch.exp(torch.tensor(0.5 + (4 - 256) * (1.15 - 0.5) / (6400 - 256), dtype=torch.float64))

    assert image_flow_shift(4, 4) == pytest.approx(float(expected_shift))
    torch.testing.assert_close(
        image_validation_sigma(base, latent_height=4, latent_width=4), shift_sigma(base, float(expected_shift))
    )
    torch.testing.assert_close(
        image_validation_sigma(base, latent_height=4, latent_width=4, flow_shift=2.5), shift_sigma(base, 2.5)
    )


def test_validation_seed_requires_explicit_identity_and_is_stable_per_stream_and_bin():
    item_seed = derive_validation_seed(123, bin_index=2, item_key="dataset/item-7", stream="video")
    assert item_seed == derive_validation_seed(123, bin_index=2, item_key="dataset/item-7", stream="video")
    assert item_seed != derive_validation_seed(123, bin_index=3, item_key="dataset/item-7", stream="video")
    assert item_seed != derive_validation_seed(123, bin_index=2, item_key="dataset/item-7", stream="audio")
    assert item_seed != derive_validation_seed(123, bin_index=2, dataset_index=7, stream="video")
    assert 0 <= item_seed < 2**63

    with pytest.raises(ValueError, match="exactly one"):
        derive_validation_seed(123, bin_index=0)
    with pytest.raises(ValueError, match="exactly one"):
        derive_validation_seed(123, bin_index=0, item_key="item", dataset_index=0)


def test_validation_accumulator_token_balance_uses_global_weighted_sums_and_counts():
    accumulator = H3ValidationAccumulator(2, balance="token", video_weight=2.0, audio_weight=0.5)
    accumulator.add(0, "video", 8.0, 4)
    accumulator.add(0, "audio", 9.0, 3)
    accumulator.add(1, "video", 20.0, 5)

    metrics = accumulator.metrics()
    assert metrics["loss"] == pytest.approx((2 * 28 + 0.5 * 9) / (2 * 9 + 0.5 * 3))
    assert metrics["loss/video"] == pytest.approx(28 / 9)
    assert metrics["loss/audio"] == pytest.approx(3)
    assert metrics["loss/bin_00"] == pytest.approx((2 * 8 + 0.5 * 9) / (2 * 4 + 0.5 * 3))
    assert metrics["loss/bin_01"] == pytest.approx(4)


def test_validation_accumulator_modality_balance_omits_zero_element_modalities_and_bins():
    accumulator = H3ValidationAccumulator(3, balance="modality", video_weight=1.0, audio_weight=3.0)
    accumulator.add(0, "video", 8.0, 4)
    accumulator.add(0, "audio", 9.0, 3)
    accumulator.add(1, "video", 20.0, 5)
    accumulator.add(2, "audio", 0.0, 0)

    metrics = accumulator.metrics()
    assert metrics["loss"] == pytest.approx((1 * (28 / 9) + 3 * 3) / 4)
    assert metrics["loss/bin_00"] == pytest.approx((1 * 2 + 3 * 3) / 4)
    assert metrics["loss/bin_01"] == pytest.approx(4)
    assert "loss/bin_02" not in metrics


def test_validation_accumulator_serializes_for_sum_reduction_and_validates_shape():
    left = H3ValidationAccumulator(2)
    right = H3ValidationAccumulator(2)
    left.add(0, "video", torch.tensor(4.0), 2)
    right.add(0, "video", 6.0, 3)
    right.add(1, "audio", 8.0, 4)

    reduced = left.reduction_tensor() + right.reduction_tensor()
    combined = H3ValidationAccumulator(2)
    combined.load_reduced_tensor(reduced)
    assert combined.metrics() == {
        "loss": pytest.approx(18 / 9),
        "loss/video": pytest.approx(2),
        "loss/audio": pytest.approx(2),
        "loss/bin_00": pytest.approx(2),
        "loss/bin_01": pytest.approx(2),
    }
    with pytest.raises(ValueError, match="shape"):
        combined.load_reduced_tensor(torch.zeros(2, 2))


def test_masked_squared_error_sum_matches_training_broadcast_and_handles_fully_masked_modality():
    prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.zeros_like(prediction)
    total, count = masked_squared_error_sum(prediction, target, torch.tensor([[True, False]]), sample_weight=torch.tensor([2.0]))
    # Weighting scales the numerator, so it must scale the denominator too: the
    # accumulator's ratio has to stay the plain mean squared error (1 and 9).
    assert count == pytest.approx(4.0)
    assert float(total) == pytest.approx(20.0)
    assert float(total) / count == pytest.approx(5.0)
    unweighted_total, unweighted_count = masked_squared_error_sum(prediction, target, torch.tensor([[True, False]]))
    assert unweighted_count == pytest.approx(2.0)
    assert float(unweighted_total) / unweighted_count == pytest.approx(5.0)
    empty_total, empty_count = masked_squared_error_sum(prediction, target, torch.zeros(1, 2, dtype=torch.bool))
    assert empty_count == 0
    assert float(empty_total) == 0.0


def test_validation_accumulator_mean_is_a_weighted_mean_across_differently_weighted_items():
    # Two batch items with the same per-element error but different sample
    # weights must average to that error, not to a weight-scaled value.
    prediction = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    target = torch.zeros_like(prediction)
    accumulator = H3ValidationAccumulator(1)

    for weight in (0.25, 4.0):
        total, count = masked_squared_error_sum(prediction[:1], target[:1], None, sample_weight=torch.tensor([weight]))
        accumulator.add(0, "video", total, count)

    assert accumulator.metrics()["loss/video"] == pytest.approx(4.0)
    # Fractional denominators must survive the distributed round trip.
    combined = H3ValidationAccumulator(1)
    combined.load_reduced_tensor(accumulator.reduction_tensor())
    assert combined.metrics() == accumulator.metrics()


def test_preserve_rng_state_restores_python_numpy_and_torch_streams():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    with preserve_rng_state():
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        _ = (random.random(), np.random.rand(), torch.rand(()))
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


class _ValidationTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.swap_mode = "training"
        self.swap_events = []

    def switch_block_swap_for_inference(self):
        self.swap_mode = "inference"
        self.swap_events.append("inference")

    def switch_block_swap_for_training(self):
        self.swap_mode = "training"
        self.swap_events.append("training")


class _ValidationBackend:
    def __init__(self):
        self.calls = []
        self.random_draws = []
        self.swap_modes = []

    def predict_training(
        self,
        transformer,
        batch,
        video_hidden_states,
        audio_hidden_states,
        video_timestep,
        audio_timestep,
        *,
        conditioning="prompt",
        **_kwargs,
    ):
        del batch, video_timestep, audio_timestep
        self.calls.append((conditioning, torch.is_grad_enabled()))
        self.random_draws.append(float(torch.rand(())))
        self.swap_modes.append(transformer.swap_mode)
        scale = 0.0 if conditioning == "empty" else transformer.scale
        return H3ModelPrediction(
            video_hidden_states * scale if video_hidden_states is not None else None,
            audio_hidden_states * scale if audio_hidden_states is not None else None,
        )


class _ValidationAccelerator:
    device = torch.device("cpu")
    process_index = 0
    num_processes = 1

    def __init__(self):
        self.trackers = [object()]
        self.logged = []

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def reduce(value, reduction):
        assert reduction == "sum"
        return value

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    @staticmethod
    def print(*_args, **_kwargs):
        pass


def test_h3_validation_dataloader_is_rank_partitioned_and_preserves_construction_rng(monkeypatch, tmp_path):
    class Group(torch.utils.data.Dataset):
        num_train_items = 5

        def __len__(self):
            return 5

        def __getitem__(self, index):
            return {"index": index}

    def create_group(*_args, **_kwargs):
        _ = (random.random(), np.random.rand(), torch.rand(()))
        return Group(), object()

    monkeypatch.setattr(h3_train_network.config_utils, "load_user_config", lambda _path: {"datasets": []})
    monkeypatch.setattr(h3_train_network, "create_h3_dataset_group", create_group)
    args = _validation_args()
    args.validation_dataset_config = tmp_path / "validation.toml"
    args.max_validation_items = 4
    accelerator = _ValidationAccelerator()
    accelerator.process_index = 1
    accelerator.num_processes = 2
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    loader = MiniMaxH3NetworkTrainer()._build_validation_dataloader(args, accelerator)

    assert list(loader) == [[1, {"index": 1}], [3, {"index": 3}]]
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


def _validation_args(*, mode="fl2va", guidance=False):
    args = create_parser().parse_args([])
    args.seed = 123
    args.validation_seed = 456
    args.validation_timestep_bins = 2
    args.validation_min_timestep = 100
    args.validation_max_timestep = 900
    args.h3_training_mode = mode
    if guidance:
        args.h3_guidance_distillation_scale = 3.0
    return args


@pytest.mark.parametrize(
    ("mode", "batch", "expected_modalities"),
    [
        ("fl2va", {"latents": torch.zeros(1, 24, 2, 2, 2)}, {"video"}),
        ("fl2va", {"latents": torch.zeros(1, 24, 1, 4, 4)}, {"video"}),
        ("fl2va", {H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3)}, {"audio"}),
        (
            "ref2va",
            {
                "latents": torch.zeros(1, 24, 2, 2, 2),
                H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
            },
            {"video", "audio"},
        ),
        (
            "ref2va_omni",
            {
                "latents": torch.zeros(1, 24, 2, 2, 2),
                H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
            },
            {"video", "audio"},
        ),
    ],
)
def test_h3_validation_runs_all_target_shapes_and_modes_without_gradients(mode, batch, expected_modalities):
    args = _validation_args(mode=mode)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [(0, {**batch, "timesteps": None})]
    accelerator = _ValidationAccelerator()
    trainer.validate(accelerator, args, _ValidationTransformer(), None, 7, None)

    metrics, step = accelerator.logged[-1]
    assert step == 7
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert {name for name in ("video", "audio") if f"val/loss/{name}" in metrics} == expected_modalities
    assert all(not grad_enabled for _, grad_enabled in trainer.backend.calls)


def test_h3_validation_uses_forward_only_block_swap_and_restores_training_mode():
    args = _validation_args()
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.blocks_to_swap = 2
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [(0, {"latents": torch.zeros(1, 24, 2, 2, 2), "timesteps": None})]
    transformer = _ValidationTransformer()

    trainer.validate(_ValidationAccelerator(), args, transformer, None, 7, None)

    assert transformer.swap_events == ["inference", "training"]
    assert transformer.swap_mode == "training"
    assert trainer.backend.swap_modes == ["inference", "inference"]


def test_h3_validation_disables_network_dropout_and_restores_module_modes():
    args = _validation_args()
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [(0, {"latents": torch.zeros(1, 24, 2, 2, 2), "timesteps": None})]
    transformer = _ValidationTransformer().train()
    network = nn.Sequential(nn.Dropout(0.5)).train()
    modes = []
    original_predict = trainer._predict

    def record_modes(*call_args, **call_kwargs):
        modes.append((transformer.training, network.training))
        return original_predict(*call_args, **call_kwargs)

    trainer._predict = record_modes

    trainer.validate(_ValidationAccelerator(), args, transformer, network, 7, None)

    assert modes == [(False, False), (False, False)]
    assert transformer.training and network.training


def test_h3_validation_recreates_task_conditioning_and_scores_only_generated_region(monkeypatch):
    args = _validation_args()
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [
        (
            0,
            {
                "latents": torch.zeros(1, 24, 2, 2, 2),
                "video_loss_mask": torch.ones(1, 2, 2, 2, dtype=torch.bool),
                "timesteps": None,
            },
        )
    ]
    generated = torch.zeros(2, 2, 2, dtype=torch.bool)
    generated[1] = True
    step_mask = SimpleNamespace(video_rows=None, audio_rows=None, video_latent=generated, audio_latent=None)
    trainer._resolve_keyframe_anchors = lambda _video: (("first",), (0,))
    trainer._draw_step_mask = lambda _inputs, _patch, _batch=None: step_mask
    observed_state = []
    original_predict = trainer._predict

    def record_predict(*call_args, **call_kwargs):
        observed_state.append((trainer._step_keyframes, trainer._step_mask))
        return original_predict(*call_args, **call_kwargs)

    trainer._predict = record_predict
    captured_masks = []
    original_loss = h3_train_network.masked_squared_error_sum

    def capture_mask(prediction, target, mask, **kwargs):
        if prediction.ndim == 5:
            captured_masks.append(mask.clone())
        return original_loss(prediction, target, mask, **kwargs)

    monkeypatch.setattr(h3_train_network, "masked_squared_error_sum", capture_mask)

    trainer.validate(_ValidationAccelerator(), args, _ValidationTransformer(), None, 7, None)

    assert observed_state == [((("first",), (0,)), step_mask)] * 2
    assert len(captured_masks) == 2
    for mask in captured_masks:
        assert not bool(mask[:, :, 0].any())
        assert bool(mask[:, :, 1].all())
    assert trainer._step_mask is None and trainer._step_keyframes is None


def test_h3_validation_omits_fully_masked_modality_and_uses_guidance_primary_objective_only():
    args = _validation_args(guidance=True)
    batch = {
        "latents": torch.zeros(1, 24, 2, 2, 2),
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        H3_AUDIO_LOSS_MASK_KEY: torch.zeros(1, 3, dtype=torch.bool),
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 5120)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
        "timesteps": None,
    }
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [(4, batch)]
    accelerator = _ValidationAccelerator()

    torch.manual_seed(987)
    expected_next = float(torch.rand(()))
    torch.manual_seed(987)
    trainer.validate(accelerator, args, _ValidationTransformer(), None, 3, None)

    metrics = accelerator.logged[-1][0]
    assert "val/loss" in metrics and "val/loss/video" in metrics
    assert "val/loss/audio" not in metrics
    assert not any("crepa" in key or "preservation" in key for key in metrics)
    assert trainer.backend.calls == [("empty", False), ("prompt", False)] * 2
    assert trainer.backend.random_draws[0] == trainer.backend.random_draws[1]
    assert trainer.backend.random_draws[2] == trainer.backend.random_draws[3]
    assert float(torch.rand(())) == expected_next
    trainer.backend.calls.clear()
    trainer.validate(accelerator, args, _ValidationTransformer(), None, 4, None)
    assert accelerator.logged[-1][0] == metrics

    unguided_args = _validation_args()
    unguided = MiniMaxH3NetworkTrainer()
    unguided.dit_dtype = torch.float32
    unguided.backend = _ValidationBackend()
    unguided._validation_dataloader = [(4, batch)]
    unguided_accelerator = _ValidationAccelerator()
    unguided.validate(
        unguided_accelerator,
        unguided_args,
        _ValidationTransformer(),
        None,
        3,
        None,
    )
    assert unguided_accelerator.logged[-1][0]["val/loss"] != pytest.approx(metrics["val/loss"])


@pytest.mark.parametrize(("observed", "remaining"), [("video", "audio"), ("audio", "video")])
def test_h3_validation_observed_modality_is_conditioning_not_a_metric(observed, remaining):
    args = _validation_args(mode="ref2va_omni")
    args.h3_observed_modality = observed
    batch = {
        "latents": torch.zeros(1, 24, 2, 2, 2),
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        "timesteps": None,
    }
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [(0, batch)]
    accelerator = _ValidationAccelerator()

    trainer.validate(accelerator, args, _ValidationTransformer(), None, 1, None)

    metrics = accelerator.logged[-1][0]
    assert f"val/loss/{observed}" not in metrics
    assert f"val/loss/{remaining}" in metrics


def test_h3_validation_random_observed_mode_reports_every_training_direction_deterministically():
    args = _validation_args(mode="ref2va_omni")
    args.h3_observed_modality = "random"
    batch = {
        "latents": torch.zeros(1, 24, 2, 2, 2),
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        "timesteps": None,
    }
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [(0, batch)]
    accelerator = _ValidationAccelerator()

    trainer.validate(accelerator, args, _ValidationTransformer(), None, 1, None)

    metrics = accelerator.logged[-1][0]
    assert "val/joint/loss" in metrics
    assert "val/joint/loss/video" in metrics
    assert "val/joint/loss/audio" in metrics
    assert "val/v2a/loss" in metrics
    assert "val/v2a/loss/video" not in metrics
    assert "val/v2a/loss/audio" in metrics
    assert "val/a2v/loss" in metrics
    assert "val/a2v/loss/video" in metrics
    assert "val/a2v/loss/audio" not in metrics
    first = dict(metrics)

    trainer.validate(accelerator, args, _ValidationTransformer(), None, 2, None)

    assert accelerator.logged[-1][0] == first
    assert trainer.backend.calls == [("prompt", False)] * 12


def test_h3_validation_random_observed_mode_accepts_mixed_single_modality_items():
    args = _validation_args(mode="ref2va_omni")
    args.h3_observed_modality = "random"
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [
        (0, {"latents": torch.zeros(1, 24, 2, 2, 2), "timesteps": None}),
        (1, {H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3), "timesteps": None}),
    ]
    accelerator = _ValidationAccelerator()

    trainer.validate(accelerator, args, _ValidationTransformer(), None, 1, None)

    metrics = accelerator.logged[-1][0]
    assert "val/loss/video" in metrics
    assert "val/loss/audio" in metrics
    assert not any(key.startswith("val/v2a/") or key.startswith("val/a2v/") for key in metrics)


def test_h3_validation_reports_each_enabled_reference_modality_variant():
    args = _validation_args(mode="ref2va_omni")
    batch = {
        "latents": torch.zeros(1, 24, 2, 2, 2),
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        H3_REFERENCE_MODALITY_PROBABILITIES_KEY: torch.tensor([[0.5, 0.5, 0.0]]),
        "timesteps": None,
    }
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [(0, batch)]
    accelerator = _ValidationAccelerator()

    trainer.validate(accelerator, args, _ValidationTransformer(), None, 1, None)

    metrics = accelerator.logged[-1][0]
    assert "val/joint/ref_av/loss" in metrics
    assert "val/joint/ref_video/loss" in metrics
    assert not any("ref_audio" in key for key in metrics)


def test_h3_validation_fully_masked_batch_produces_no_nan_or_fake_zero_metric():
    args = _validation_args(mode="ref2va")
    batch = {
        "latents": torch.zeros(1, 24, 2, 2, 2),
        "video_loss_mask": torch.zeros(1, 2, 2, 2, dtype=torch.bool),
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        H3_AUDIO_LOSS_MASK_KEY: torch.zeros(1, 3, dtype=torch.bool),
        "timesteps": None,
    }
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [(0, batch)]
    accelerator = _ValidationAccelerator()

    trainer.validate(accelerator, args, _ValidationTransformer(), None, 1, None)

    assert accelerator.logged == []


def test_h3_validation_weights_each_modality_from_its_own_sigma_and_clears_keyframes():
    args = _validation_args(mode="ref2va")
    batch = {
        "latents": torch.zeros(1, 24, 2, 2, 2),
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        "timesteps": None,
    }
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ValidationBackend()
    trainer._validation_dataloader = [(0, batch)]
    trainer._step_keyframes = (("last",), (1,))
    sampled_sigmas = []

    def record_weight(_args, sigma, modality="video"):
        sampled_sigmas.append(float(sigma))
        return torch.ones_like(sigma)

    trainer._sample_weight = record_weight
    trainer.validate(_ValidationAccelerator(), args, _ValidationTransformer(), None, 1, None)

    assert trainer._step_keyframes is None
    assert len(sampled_sigmas) == 4
    assert sampled_sigmas[0] != pytest.approx(sampled_sigmas[1])
    assert sampled_sigmas[2] != pytest.approx(sampled_sigmas[3])


def test_h3_field_cosine_is_masked_and_scale_invariant():
    """The angle must ignore length and must ignore elements the mask excludes.

    Scale invariance is what makes the cosine independent of the ratio reported
    beside it; without it the two numbers would move together and the pair would
    say no more than either alone. The mask matters because padded rows carry no
    authored content, and letting them into the inner product would report the
    padding's agreement rather than the field's.
    """
    cosine = h3_train_network.MiniMaxH3NetworkTrainer._masked_cosine
    field = torch.tensor([[3.0, 4.0, 0.0]])
    assert cosine(field, field * 7.0, None) == pytest.approx(1.0, abs=1e-6)
    assert cosine(field, -field, None) == pytest.approx(-1.0, abs=1e-6)

    # Third element disagrees violently, and the mask excludes it.
    left = torch.tensor([[1.0, 0.0, 99.0]])
    right = torch.tensor([[1.0, 0.0, -99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert cosine(left, right, mask) == pytest.approx(1.0, abs=1e-6)
    assert cosine(left, right, None) < 0.0

    # A field that vanished has no direction to report, and 0 is the honest answer
    # rather than a division that would raise or return a nan the average swallows.
    assert cosine(torch.zeros(1, 3), field, None) == 0.0


def test_h3_field_distance_orders_a_rotated_field_below_a_shortened_one():
    """Length and angle folded into one number, and why that number is needed.

    Read as a length alone, an adapter that keeps its field long looks like the one
    that preserved it best. This is the case that showed otherwise on real runs: a
    field at 0.74 of the base turned 45 degrees sits FURTHER from the base's field
    than one shortened to 0.55 while staying aligned, and the ratio ranks them the
    other way round.
    """

    def distance(ratio: float, cosine: float) -> float:
        return math.sqrt(max(0.0, 1.0 + ratio * ratio - 2.0 * ratio * cosine))

    assert distance(1.0, 1.0) == pytest.approx(0.0, abs=1e-9)
    # Same direction, half the length: exactly half the field is missing.
    assert distance(0.5, 1.0) == pytest.approx(0.5, abs=1e-9)
    long_but_turned = distance(0.74, 0.702)
    short_but_aligned = distance(0.55, 0.94)
    assert long_but_turned > short_but_aligned
    # And the ratio on its own would have called the rotated one better.
    assert 0.74 > 0.55


def test_h3_relative_velocity_error_reads_1_when_an_adapter_learned_nothing():
    """The ratio that makes a preservation score readable.

    Every preservation metric here is flattered by an adapter that barely moved, and
    on real runs the arm that disturbed the base least had covered about a seventh of
    the distance ordinary training covers. Dividing the adapter's error by the
    untouched checkpoint's turns that into a number: 1.0 says the run learned nothing,
    and only a value well below 1 makes a low field distance mean anything.

    Exercised on the same masked reduction the metric is built from, so the mask
    exclusion the rest of the probe relies on is covered here too.
    """
    target = torch.tensor([[2.0, 2.0, 50.0]])
    base = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    def ratio(adapted: torch.Tensor) -> float:
        error, _ = masked_squared_error_sum(adapted, target, mask)
        base_error, _ = masked_squared_error_sum(base, target, mask)
        return float(error) / float(base_error)

    # An adapter that reproduces the base exactly has learned nothing.
    assert ratio(base) == pytest.approx(1.0, abs=1e-6)
    # One that reaches the target has learned everything.
    assert ratio(target) == pytest.approx(0.0, abs=1e-9)
    # Halving the residual quarters the ratio, because both sides are energies.
    assert ratio(target / 2.0) == pytest.approx(0.25, abs=1e-6)
    # The masked-out element disagrees enormously and must not enter either side.
    assert ratio(torch.tensor([[2.0, 2.0, -999.0]])) == pytest.approx(0.0, abs=1e-9)


def test_h3_field_probe_refuses_a_checkpoint_with_merged_base_weights():
    """The probe's reference must be the checkpoint, and a merge destroys it.

    Every number the probe reports is a ratio against the checkpoint, formed by
    disabling the trainable network for one pair of forwards. ``--base_weights`` is
    folded into the transformer at load time and cannot be switched off again, so the
    reference silently becomes "the checkpoint plus that adapter" and every ratio lands
    on a scale no other run shares -- while looking completely ordinary in the log.

    That is not hypothetical: an arm measured this way reported a field of 1.0 at step
    zero, meaning "untouched", when the adapter it was being compared against had
    already lost two thirds of the base's field.
    """
    trainer = MiniMaxH3NetworkTrainer.__new__(MiniMaxH3NetworkTrainer)
    trainer._validation_network = nn.Linear(1, 1)
    trainer._merged_base_weight_paths = ["collapsed.safetensors"]
    inputs = SimpleNamespace(video=torch.zeros(1, 1))
    batch = {H3_EMPTY_TEXT_HIDDEN_KEY: torch.zeros(1), H3_EMPTY_TEXT_TOKEN_TAGS_KEY: torch.zeros(1)}

    with pytest.raises(ValueError, match="merged"):
        MiniMaxH3NetworkTrainer._probe_guidance_field(
            trainer,
            SimpleNamespace(device=torch.device("cpu")),
            SimpleNamespace(),
            nn.Linear(1, 1),
            batch,
            inputs,
            None,
            dataset_index=0,
            sigma_bin=SimpleNamespace(index=0),
            observed="video",
        )


def test_h3_null_anchor_weight_must_not_be_negative():
    """A negative weight would configure the anchor and then train without it.

    Every gate on the feature reads ``> 0``, so a negative value silently disables the
    thing the launch line asked for. That failure mode has already cost this codebase a
    long run: an arm trained for an hour and a half believing it was anchored, and only
    the missing ``loss/guidance_null_anchor`` tag gave it away afterwards.
    """
    trainer = MiniMaxH3NetworkTrainer.__new__(MiniMaxH3NetworkTrainer)
    args = create_parser().parse_args([])

    assert args.h3_guidance_null_anchor_weight == 0.0

    args.h3_guidance_null_anchor_weight = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        trainer.handle_model_specific_args(args)

    args.h3_guidance_null_anchor_weight = float("nan")
    with pytest.raises(ValueError, match="finite"):
        trainer.handle_model_specific_args(args)


def _dense_probe_trainer(step: int):
    """A trainer whose validation network is the dense module: nothing to switch off."""
    trainer = MiniMaxH3NetworkTrainer.__new__(MiniMaxH3NetworkTrainer)
    trainer._validation_network = nn.Linear(1, 1)  # no set_enabled(): the full fine-tune shape
    trainer._validation_global_step = step
    trainer._merged_base_weight_paths = None
    trainer._step_reference_modality = "av"
    for name in (
        "_field_base_gaps",
        "_field_base_branches",
        "_velocity_errors",
        "_velocity_error_ratios",
        "_branch_drift",
        "_prompted_drift_ratios",
        "_field_ratios",
        "_field_cosines",
        "_field_distances",
        "_null_field_ratios",
    ):
        setattr(trainer, name, {})
    calls = []

    def predict(accelerator, transformer, batch, inputs, conditioning):
        calls.append(conditioning)
        return SimpleNamespace(video=torch.full((1, 4), 2.0 if conditioning == "prompt" else 1.0))

    trainer._predict = predict
    return trainer, calls


def _run_dense_probe(trainer, dataset_index=0):
    inputs = SimpleNamespace(video=torch.zeros(1, 4), video_target=None)
    batch = {H3_EMPTY_TEXT_HIDDEN_KEY: torch.zeros(1), H3_EMPTY_TEXT_TOKEN_TAGS_KEY: torch.zeros(1)}
    MiniMaxH3NetworkTrainer._probe_guidance_field(
        trainer,
        SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda module: module),
        SimpleNamespace(),
        nn.Linear(1, 1),
        batch,
        inputs,
        None,
        dataset_index=dataset_index,
        sigma_bin=SimpleNamespace(index=0),
        observed="video",
    )


def test_h3_field_probe_on_a_full_finetune_takes_its_reference_at_step_zero():
    """With no network to switch off, step zero is the only time the weights are the
    base: the pair taken then is stored, and is also the adapted pair of that step."""
    trainer, calls = _dense_probe_trainer(step=0)

    _run_dense_probe(trainer)

    key = (0, 0, "video", "av")
    assert key in trainer._field_base_branches
    assert calls == ["prompt", "empty"], "one pair of forwards, not two"
    assert trainer._field_ratios[0] == [pytest.approx(1.0)]
    assert trainer._field_distances[0] == [pytest.approx(0.0)]


def test_h3_field_probe_on_a_full_finetune_refuses_an_item_without_a_reference():
    trainer, _ = _dense_probe_trainer(step=250)

    with pytest.raises(ValueError, match="reference taken at step 0"):
        _run_dense_probe(trainer)


def test_h3_field_probe_reference_survives_a_round_trip_through_its_file(tmp_path):
    source, _ = _dense_probe_trainer(step=0)
    _run_dense_probe(source)
    path = str(tmp_path / "run_field_probe_base.pt")
    fingerprint = {"dit": "base.safetensors", "validation_seed": "7"}
    MiniMaxH3NetworkTrainer._save_field_probe_snapshot(source, path, fingerprint)

    resumed, calls = _dense_probe_trainer(step=250)
    resumed._field_probe_snapshot_loaded = False
    MiniMaxH3NetworkTrainer._load_field_probe_snapshot(resumed, path, dict(fingerprint))
    _run_dense_probe(resumed)

    # The resumed run measured its live weights against the stored base, with
    # exactly one pair of forwards and no refusal.
    assert calls == ["prompt", "empty"]
    assert resumed._field_ratios[0] == [pytest.approx(1.0)]


def test_h3_field_probe_reference_is_refused_under_another_configuration(tmp_path):
    """The stored pairs are indexed by validation position and sigma bin, which name
    nothing under a different validation set, seed or checkpoint."""
    source, _ = _dense_probe_trainer(step=0)
    _run_dense_probe(source)
    path = str(tmp_path / "run_field_probe_base.pt")
    MiniMaxH3NetworkTrainer._save_field_probe_snapshot(source, path, {"dit": "base.safetensors", "validation_seed": "7"})

    resumed, _ = _dense_probe_trainer(step=250)
    resumed._field_probe_snapshot_loaded = False
    with pytest.raises(ValueError, match="different validation_seed"):
        MiniMaxH3NetworkTrainer._load_field_probe_snapshot(resumed, path, {"dit": "base.safetensors", "validation_seed": "8"})
    assert not resumed._field_base_gaps


def test_h3_field_probe_resume_without_its_reference_file_is_refused(tmp_path):
    resumed, _ = _dense_probe_trainer(step=250)
    resumed._field_probe_snapshot_loaded = False
    with pytest.raises(FileNotFoundError, match="validate_at_start"):
        MiniMaxH3NetworkTrainer._load_field_probe_snapshot(resumed, str(tmp_path / "missing.pt"), {})


def test_h3_field_probe_fingerprint_reads_the_validation_config_content(tmp_path):
    config = tmp_path / "holdout.toml"
    config.write_text("[general]\n", encoding="utf-8")
    args = SimpleNamespace(
        dit="base.safetensors",
        validation_dataset_config=str(config),
        validation_seed=None,
        seed=3,
        max_validation_items=None,
        h3_training_mode="fl2va",
    )
    first = MiniMaxH3NetworkTrainer._field_probe_fingerprint(args)
    config.write_text("[general]\nresolution = [1, 1]\n", encoding="utf-8")
    second = MiniMaxH3NetworkTrainer._field_probe_fingerprint(args)

    assert first["validation_seed"] == "3"
    assert first["validation_dataset_config"] != second["validation_dataset_config"]


class _ConditionalBackend(_ValidationBackend):
    """An adapter whose effect exists only on triggered items: with the adapter on and
    the batch not marked bare, the prompted prediction is doubled; otherwise it is the
    base's. The empty branch is always the base's."""

    def predict_training(
        self,
        transformer,
        batch,
        video_hidden_states,
        audio_hidden_states,
        video_timestep,
        audio_timestep,
        *,
        conditioning="prompt",
        **_kwargs,
    ):
        del video_timestep, audio_timestep
        self.calls.append((conditioning, torch.is_grad_enabled()))
        adapter_on = getattr(transformer, "adapter_enabled", True)
        bare = bool(batch.get("bare", False))
        if conditioning == "empty":
            scale = 0.25
        elif adapter_on and not bare:
            scale = 2.0
        else:
            scale = 1.0
        return H3ModelPrediction(
            video_hidden_states * scale if video_hidden_states is not None else None,
            audio_hidden_states * scale if audio_hidden_states is not None else None,
        )


class _TriggerNetwork:
    def __init__(self, transformer):
        self.transformer = transformer

    def set_enabled(self, enabled):
        self.transformer.adapter_enabled = enabled

    def is_enabled(self):
        return getattr(self.transformer, "adapter_enabled", True)

    def named_parameters(self):
        return self.transformer.named_parameters()

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    @property
    def training(self):
        return False


class _TriggerAccelerator(_ValidationAccelerator):
    @staticmethod
    def unwrap_model(model):
        return model


def _trigger_batches(bare: bool):
    from musubi_tuner.minimax_h3.cache import (
        H3_EMPTY_TEXT_HIDDEN_KEY,
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
        H3_TEXT_HIDDEN_KEY,
        H3_TEXT_TOKEN_TAGS_KEY,
    )
    from musubi_tuner.minimax_h3.rollout import H3_ROLLOUT_ITEM_KEYS_BATCH_KEY

    batches = []
    for index, name in enumerate(("clip_a", "clip_b")):
        torch.manual_seed(index)
        batch = {
            "latents": torch.randn(1, 24, 2, 2, 2),
            "timesteps": None,
            H3_TEXT_HIDDEN_KEY: [torch.full((4, 8), 0.25)],
            H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
            H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(4, 8)],
            H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
            H3_ROLLOUT_ITEM_KEYS_BATCH_KEY: [name],
        }
        if bare:
            batch["bare"] = True
        batches.append((index, batch))
    return batches


def test_trigger_probe_reports_the_conditional_gain_and_leaves_the_main_metrics_alone():
    args = _validation_args()
    args.h3_validation_field_probe = True
    args.h3_validation_bare_dataset_config = "bare.toml"
    args.h3_validation_std = True
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ConditionalBackend()
    trainer._validation_dataloader = _trigger_batches(bare=False)
    trainer._bare_validation_dataloader = _trigger_batches(bare=True)
    transformer = _ValidationTransformer()
    network = _TriggerNetwork(transformer)
    accelerator = _TriggerAccelerator()

    trainer.validate(accelerator, args, transformer, network, 7, None)
    metrics, _ = accelerator.logged[-1]

    assert metrics["val/trigger/pairs"] == 2.0
    # Triggered: the prompted prediction moved from 1x to 2x the latents against a
    # base field of 0.75x, so the relative drift is 1/0.75; bare: no movement.
    assert metrics["val/drift/prompted_rel"] == pytest.approx(1.0 / 0.75, abs=1e-5)
    assert metrics["val/trigger/drift_bare"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["val/trigger/drift_gain"] == pytest.approx(1.0 / 0.75, abs=1e-5)
    assert "val/trigger/drift_gain_std" in metrics
    # The bare prompt did not move at all, so none of the trigger's update leaked to it.
    assert metrics["val/trigger/leak_share"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["val/trigger/leak_cos"] == pytest.approx(0.0, abs=1e-6)
    assert any(key.startswith("val/trigger/leak_share/bin") for key in metrics)
    # Paired STATES (item x sigma bin), so at least one per paired item.
    assert metrics["val/trigger/leak_pairs"] >= metrics["val/trigger/pairs"]
    assert trainer._trigger_drift is None
    # The fit ratio: the adapter is worse than the base on the triggered items (2x vs
    # 1x of a unit-velocity target) and identical on the bare ones, so the gain is
    # negative here and exactly zero on the bare side.
    assert metrics["val/trigger/err_rel_bare"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["val/trigger/err_gain"] == pytest.approx(1.0 - metrics["val/velocity_err_rel"], abs=1e-5)
    # The bare pass never touched the main pools.
    assert metrics["val/field"] == pytest.approx((2.0 - 0.25) / 0.75, abs=1e-5)
    # The base alone: prompted 1x, empty 0.25x of the latents, so the field is 0.75 of the
    # prompted RMS and the two branches are parallel.
    assert metrics["val/base/field_rel"] == pytest.approx(0.75, abs=1e-5)
    assert metrics["val/base/cos_prompted_empty"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["val/base/empty_rms"] == pytest.approx(0.25 * metrics["val/base/prompted_rms"], rel=1e-4)
    for name in ("empty_rms", "prompted_rms", "field_rel", "cos_prompted_empty"):
        assert any(key.startswith(f"val/base/{name}/bin") for key in metrics), name


def test_trigger_probe_needs_the_field_probe():
    args = _validation_args()
    args.validation_dataset_config = "holdout.toml"
    args.h3_validation_bare_dataset_config = "bare.toml"
    with pytest.raises(ValueError, match="needs --h3_validation_field_probe"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_trigger_probe_rejects_a_full_finetune_before_any_validation_work():
    args = _validation_args()
    args.h3_validation_field_probe = True
    args.h3_validation_bare_dataset_config = "bare.toml"
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _ConditionalBackend()
    trainer._validation_dataloader = None
    trainer._build_validation_dataloader = lambda *a, **k: (_ for _ in ()).throw(AssertionError("loader built"))
    with pytest.raises(ValueError, match="full fine-tune"):
        trainer.validate(_TriggerAccelerator(), args, _ValidationTransformer(), None, 7, None)
    assert trainer.backend.calls == []
