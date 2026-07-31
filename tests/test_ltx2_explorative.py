from __future__ import annotations

import argparse
import random
from types import SimpleNamespace

import pytest
import torch

import musubi_tuner.ltx2_train_network as ltx2_train_network
from musubi_tuner.gui_dashboard.command_builder import _append_ltx2_performance_args
from musubi_tuner.gui_dashboard.project_schema import TrainingConfig
from musubi_tuner.ltx2_args import ltx2_setup_parser
from musubi_tuner.ltx2_explorative import (
    RNGSnapshot,
    per_sample_reconstruction_loss,
    score_ltx2_candidate,
    seeded_randn_like,
    update_streaming_winner,
    validate_ltx2_xm_args,
)
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer
from musubi_tuner.training.losses import reduce_masked_loss


def test_per_sample_masked_reconstruction_loss_normalizes_each_sample() -> None:
    pred = torch.tensor([[[1.0], [3.0]], [[2.0], [100.0]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, True], [True, False]])
    loss = per_sample_reconstruction_loss(pred, target, mask, loss_type="mse", huber_delta=1.0)
    torch.testing.assert_close(loss, torch.tensor([5.0, 4.0]))


def test_batch_global_reconstruction_loss_decomposes_exact_scalar() -> None:
    pred = torch.tensor([[[1.0], [3.0]], [[2.0], [100.0]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, True], [True, False]])
    contributions = per_sample_reconstruction_loss(
        pred,
        target,
        mask,
        loss_type="mse",
        huber_delta=1.0,
        batch_global=True,
    )
    torch.testing.assert_close(contributions, torch.tensor([10.0 / 3.0, 4.0 / 3.0]))
    torch.testing.assert_close(contributions.sum(), torch.tensor(14.0 / 3.0))


@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize(
    "mask",
    [
        None,
        torch.tensor([[True, True], [True, False]]),
        torch.zeros((2, 2), dtype=torch.bool),
    ],
)
def test_xm_per_sample_scores_match_ltx_production_reducer(mask, per_sample: bool) -> None:
    pred = torch.tensor([[[1.0], [3.0]], [[2.0], [100.0]]])
    target = torch.zeros_like(pred)
    contributions = per_sample_reconstruction_loss(
        pred,
        target,
        mask,
        loss_type="mse",
        huber_delta=1.0,
        batch_global=not per_sample,
    )
    production_loss, _ = reduce_masked_loss((pred - target).square(), mask, per_sample=per_sample)
    xm_reduction = contributions.mean() if per_sample else contributions.sum()
    torch.testing.assert_close(xm_reduction, production_loss)


def test_joint_av_score_uses_one_weighted_candidate() -> None:
    output = {
        "video_pred": torch.tensor([[[1.0]], [[3.0]]]),
        "video_target": torch.zeros((2, 1, 1)),
        "video_loss_weight": 2.0,
        "audio_pred": torch.tensor([[[[4.0]]], [[[1.0]]]]),
        "audio_target": torch.zeros((2, 1, 1, 1)),
        "audio_loss_weight": 0.5,
    }
    score = score_ltx2_candidate(output, loss_type="mse", huber_delta=1.0, per_sample_loss=True)
    torch.testing.assert_close(score, torch.tensor([10.0, 18.5]))


def test_streaming_winner_is_per_sample_and_ties_keep_earlier_candidate() -> None:
    best_score = torch.tensor([4.0, 1.0])
    candidate_score = torch.tensor([0.25, 1.0])
    best_index = torch.zeros(2, dtype=torch.long)
    best_video = torch.tensor([[[2.0]], [[1.0]]])
    candidate_video = torch.tensor([[[0.5]], [[9.0]]])
    score, index, video, audio = update_streaming_winner(
        best_score,
        candidate_score,
        1,
        best_index,
        best_video,
        candidate_video,
        None,
        None,
    )
    torch.testing.assert_close(score, torch.tensor([0.25, 1.0]))
    torch.testing.assert_close(index, torch.tensor([1, 0]))
    torch.testing.assert_close(video, torch.tensor([[[0.5]], [[1.0]]]))
    assert audio is None


def test_streaming_joint_av_winner_keeps_modalities_from_same_candidate() -> None:
    _, index, video, audio = update_streaming_winner(
        torch.tensor([4.0, 1.0]),
        torch.tensor([0.25, 1.0]),
        1,
        torch.zeros(2, dtype=torch.long),
        torch.tensor([[[2.0]], [[1.0]]]),
        torch.tensor([[[0.5]], [[9.0]]]),
        torch.tensor([[[20.0]], [[10.0]]]),
        torch.tensor([[[5.0]], [[90.0]]]),
    )
    torch.testing.assert_close(index, torch.tensor([1, 0]))
    torch.testing.assert_close(video, torch.tensor([[[0.5]], [[1.0]]]))
    torch.testing.assert_close(audio, torch.tensor([[[5.0]], [[10.0]]]))


@pytest.mark.parametrize("value", [0, 33, -1])
def test_xm_k_validation_rejects_out_of_range(value: int) -> None:
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        validate_ltx2_xm_args(SimpleNamespace(ltx2_xm_k=value))


def test_xm_validation_rejects_stateful_objective_combinations() -> None:
    with pytest.raises(ValueError, match="--self_flow"):
        validate_ltx2_xm_args(SimpleNamespace(ltx2_xm_k=2, self_flow=True, audio_loss_balance_mode="none"))
    with pytest.raises(ValueError, match="audio_loss_balance_mode none"):
        validate_ltx2_xm_args(SimpleNamespace(ltx2_xm_k=2, audio_loss_balance_mode="ema_mag"))
    with pytest.raises(ValueError, match="--blank_preservation"):
        validate_ltx2_xm_args(SimpleNamespace(ltx2_xm_k=2, blank_preservation=True, audio_loss_balance_mode="none"))
    with pytest.raises(ValueError, match="--cts_lambda_video_driven"):
        validate_ltx2_xm_args(SimpleNamespace(ltx2_xm_k=2, cts_lambda_video_driven=0.3, audio_loss_balance_mode="none"))


def test_parser_defaults_to_disabled_xm() -> None:
    parser = ltx2_setup_parser(argparse.ArgumentParser())
    args = parser.parse_args([])
    assert args.ltx2_xm_k == 1


def test_dashboard_suppresses_default_and_emits_enabled_xm() -> None:
    cmd: list[str] = []
    _append_ltx2_performance_args(cmd, TrainingConfig())
    assert "--ltx2_xm_k" not in cmd

    cmd = []
    _append_ltx2_performance_args(cmd, TrainingConfig(ltx2_xm_k=3))
    assert cmd[-2:] == ["--ltx2_xm_k", "3"]


class _ForwardHarness:
    training = True
    call_dit = LTX2NetworkTrainer.call_dit
    _call_dit_forward_xm = LTX2NetworkTrainer._call_dit_forward_xm

    def __init__(self) -> None:
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[tuple[bool, float, float]] = []

    def _call_dit_once(self, args, accelerator, transformer, latents, batch, noise, noisy, timesteps, dtype, **kwargs):
        self.calls.append((torch.is_grad_enabled(), random.random(), float(torch.rand(()).item())))
        pred = noise.to(device=accelerator.device, dtype=torch.float32) * self.scale
        return {
            "video_pred": pred,
            "video_target": torch.zeros_like(pred),
            "video_loss_weight": 1.0,
        }, torch.tensor(0.0)


class _ReplayMismatchHarness(_ForwardHarness):
    def _call_dit_once(self, args, accelerator, transformer, latents, batch, noise, noisy, timesteps, dtype, **kwargs):
        pred = noise.to(device=accelerator.device, dtype=torch.float32) * self.scale
        if torch.is_grad_enabled():
            pred = pred + 0.25
        return {
            "video_pred": pred,
            "video_target": torch.zeros_like(pred),
            "video_loss_weight": 1.0,
        }, torch.tensor(0.0)


def test_k1_calls_original_forward_once_without_exploration(monkeypatch) -> None:
    harness = _ForwardHarness()
    monkeypatch.setattr(ltx2_train_network, "seeded_randn_like", lambda *_args, **_kwargs: pytest.fail("XM noise used"))
    noise = torch.tensor([[[[[2.0]]]]])
    output, _ = harness.call_dit(
        SimpleNamespace(ltx2_xm_k=1),
        SimpleNamespace(device=torch.device("cpu")),
        None,
        torch.zeros_like(noise),
        {},
        noise,
        noise,
        torch.tensor([500.0]),
        torch.float32,
    )
    assert len(harness.calls) == 1
    assert "_xm_metrics" not in output
    torch.testing.assert_close(output["video_pred"], noise)


def test_forward_xm_selects_per_sample_winners_and_replays_condition_rng(monkeypatch) -> None:
    harness = _ForwardHarness()
    candidate_noises = iter(
        [
            torch.tensor([[[[[2.0]]]], [[[[4.0]]]]]),
            torch.tensor([[[[[0.5]]]], [[[[2.0]]]]]),
        ]
    )
    monkeypatch.setattr(ltx2_train_network, "seeded_randn_like", lambda _tensor, _seed: next(candidate_noises))
    initial_noise = torch.tensor([[[[[3.0]]]], [[[[1.0]]]]])
    latents = torch.zeros_like(initial_noise)
    output, _ = harness.call_dit(
        SimpleNamespace(ltx2_xm_k=3, loss_type="mse", huber_delta=1.0),
        SimpleNamespace(device=torch.device("cpu")),
        None,
        latents,
        {},
        initial_noise,
        initial_noise * 0.5,
        torch.tensor([500.0, 500.0]),
        torch.float32,
    )

    # Three no-gradient score forwards and one gradient-bearing winner replay.
    assert [call[0] for call in harness.calls] == [False, False, False, True]
    for call in harness.calls[1:]:
        assert call[1:] == harness.calls[0][1:]
    torch.testing.assert_close(output["video_pred"], torch.tensor([[[[[0.5]]]], [[[[1.0]]]]]))
    assert output["video_pred"].requires_grad
    assert output["_xm_metrics"]["xm/winner_0_fraction"] == 0.5
    assert output["_xm_metrics"]["xm/winner_2_fraction"] == 0.5


def test_forward_xm_rejects_multi_process_execution() -> None:
    harness = _ForwardHarness()
    noise = torch.ones((1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="single-process"):
        harness.call_dit(
            SimpleNamespace(ltx2_xm_k=2),
            SimpleNamespace(device=torch.device("cpu"), num_processes=2),
            None,
            torch.zeros_like(noise),
            {},
            noise,
            noise * 0.5,
            torch.tensor([500.0]),
            torch.float32,
        )


def test_forward_xm_rejects_nonreproducible_winner_replay(monkeypatch) -> None:
    harness = _ReplayMismatchHarness()
    noise = torch.ones((1, 1, 1, 1, 1))
    monkeypatch.setattr(ltx2_train_network, "seeded_randn_like", lambda _tensor, _seed: noise * 2.0)
    with pytest.raises(RuntimeError, match="winner replay changed"):
        harness.call_dit(
            SimpleNamespace(ltx2_xm_k=2, loss_type="mse", huber_delta=1.0),
            SimpleNamespace(device=torch.device("cpu")),
            None,
            torch.zeros_like(noise),
            {},
            noise,
            noise * 0.5,
            torch.tensor([500.0]),
            torch.float32,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_private_noise_and_condition_rng_replay() -> None:
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(1234)
    snapshot = RNGSnapshot.capture(device)
    first_condition_draw = torch.rand((8,), device=device)
    snapshot.restore()
    torch.testing.assert_close(torch.rand((8,), device=device), first_condition_draw)

    global_state = torch.cuda.get_rng_state(device).clone()
    template = torch.empty((2, 3), device=device)
    first_noise = seeded_randn_like(template, 99)
    torch.testing.assert_close(seeded_randn_like(template, 99), first_noise)
    assert not torch.equal(seeded_randn_like(template, 100), first_noise)
    torch.testing.assert_close(torch.cuda.get_rng_state(device), global_state)
