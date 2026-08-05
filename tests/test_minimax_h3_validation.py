import random
from contextlib import nullcontext

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
)
from musubi_tuner.minimax_h3.training import H3ModelPrediction
from musubi_tuner.minimax_h3.training import shift_sigma
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
    torch.testing.assert_close(
        torch.tensor([item.video_sigma for item in bins], dtype=torch.float64), shift_sigma(base, 12.0)
    )
    torch.testing.assert_close(
        torch.tensor([item.audio_sigma for item in bins], dtype=torch.float64), shift_sigma(base, 3.0)
    )


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
    total, count = masked_squared_error_sum(
        prediction, target, torch.tensor([[True, False]]), sample_weight=torch.tensor([2.0])
    )
    assert count == 2
    assert float(total) == pytest.approx(20.0)
    empty_total, empty_count = masked_squared_error_sum(prediction, target, torch.zeros(1, 2, dtype=torch.bool))
    assert empty_count == 0
    assert float(empty_total) == 0.0


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


class _ValidationBackend:
    def __init__(self):
        self.calls = []

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
    ):
        del batch, video_timestep, audio_timestep
        self.calls.append((conditioning, torch.is_grad_enabled()))
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
