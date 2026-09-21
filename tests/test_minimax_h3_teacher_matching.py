from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.cache import (
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.training import H3ModelPrediction, joint_prediction_loss
from musubi_tuner.minimax_h3_train_network import (
    MiniMaxH3NetworkTrainer,
    _teacher_anchor_compensation,
    _teacher_conditioned,
    _teacher_flow_loss,
    _teacher_joint_loss,
    create_parser,
)


def test_teacher_loss_value_and_independent_gradient():
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    target = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]])
    loss = _teacher_flow_loss(pred, target, mag_weight=0.4, dc_weight=0.25)
    residual = pred - target
    adjusted = pred - 0.5 * residual.mean(dim=-1, keepdim=True)
    pnorm, tnorm = adjusted.flatten().norm(), target.flatten().norm()
    reference = (
        0.4 * (pnorm - tnorm).square() + 2 * pnorm.detach() * tnorm * (1 - (adjusted * target).sum() / (pnorm * tnorm + 1e-12))
    ) / pred.numel()
    assert torch.allclose(loss, reference)
    assert torch.allclose(torch.autograd.grad(loss, pred)[0], torch.autograd.grad(reference, pred)[0])


def test_teacher_loss_numerical_contracts():
    target = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    pred = (target + 0.7).clone().requires_grad_()
    assert torch.allclose(_teacher_flow_loss(pred, target, mag_weight=1.0), torch.nn.functional.mse_loss(pred, target))
    dc_loss = _teacher_flow_loss(pred, target, mag_weight=1.0, dc_weight=0.0)
    assert dc_loss.abs() < 1e-6
    assert torch.autograd.grad(dc_loss, pred)[0].abs().max() < 1e-6
    radial = (2.0 * target.double()).clone().requires_grad_()
    direction_only = _teacher_flow_loss(radial, target, mag_weight=0.0)
    assert direction_only.abs() < 1e-6
    assert torch.autograd.grad(direction_only, radial)[0].abs().max() < 1e-6


def test_teacher_sigma_gates_and_anchor_focus_compensation():
    args = SimpleNamespace(
        h3_teacher_condition_sigma_min=0.2,
        h3_teacher_condition_sigma_max=0.7,
        h3_timestep_focus_probability=0.5,
        h3_timestep_focus_min=0.3,
        h3_timestep_focus_max=0.6,
    )
    assert [_teacher_conditioned(args, x) for x in (0.1, 0.2, 0.7, 0.8)] == [False, True, True, False]
    assert _teacher_anchor_compensation(args) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "conditions,identity,task",
    [
        ("first,last", 1, "fl2va"),
        ("ref", 2, "ref2va"),
        ("subject_ref", 3, "ref2va"),
    ],
)
def test_teacher_batch_is_separate_from_student(conditions, identity, task):
    trainer = MiniMaxH3NetworkTrainer()
    trainer.backend = SimpleNamespace(mode="fl2va", _one_conditioning_item=lambda batch, key, expected_ndim: batch[key][0])
    student_hidden = torch.zeros(1, 2, 3)
    teacher_hidden = torch.ones(1, 3, 3)
    batch = {
        H3_CONDITIONING_TASK_KEY: torch.tensor([H3_CONDITIONING_TASK_IDS["t2va"]]),
        H3_TEXT_HIDDEN_KEY: student_hidden,
        H3_TEXT_TOKEN_TAGS_KEY: torch.zeros(1, 2, dtype=torch.long),
        "mmh3_teacher_hidden_states": teacher_hidden,
        "mmh3_teacher_token_tags": torch.ones(1, 3, dtype=torch.long),
        "mmh3_teacher_conditions": torch.tensor([identity]),
    }
    inputs = SimpleNamespace(
        video=torch.ones(1, 1, 1, 1, 1), video_target=torch.zeros(1, 1, 1, 1, 1), video_sigma=torch.ones(1, 1, 1, 1, 1), audio=None
    )
    teacher, mode = trainer._teacher_matching_batch(SimpleNamespace(h3_teacher_conditions=conditions), batch, inputs, True)
    assert mode == task
    assert teacher[H3_TEXT_HIDDEN_KEY] is teacher_hidden
    assert int(teacher[H3_CONDITIONING_TASK_KEY][0]) == H3_CONDITIONING_TASK_IDS[task]
    assert batch[H3_TEXT_HIDDEN_KEY] is student_hidden
    anchor, anchor_mode = trainer._teacher_matching_batch(SimpleNamespace(h3_teacher_conditions=conditions), batch, inputs, False)
    assert anchor is batch and anchor_mode == "fl2va"


def test_teacher_parser_default_off():
    args = create_parser().parse_args([])
    assert args.h3_teacher_matching is False


@pytest.mark.parametrize("balance", ["modality", "token"])
@pytest.mark.parametrize("normalization", ["weighted", "full"])
@pytest.mark.parametrize("audio_present", [False, True])
def test_teacher_joint_unit_weights_match_fork_loss(balance, normalization, audio_present):
    student = H3ModelPrediction(
        torch.tensor([[[[[0.5, -1.0], [1.0, 2.0]]]]]),
        torch.tensor([[[[0.5, 1.5, -2.0]], [[2.0, -1.0, 0.0]]]]) if audio_present else None,
    )
    teacher = H3ModelPrediction(
        torch.tensor([[[[[1.0, 0.5], [-0.5, 1.0]]]]]),
        torch.zeros_like(student.audio) if audio_present else None,
    )
    video_mask = torch.tensor([[[[0.0, 0.25], [1.0, 0.5]]]])
    audio_mask = torch.tensor([[[0.0, 0.5, 1.0]]]) if audio_present else None
    video_sample_weight = torch.tensor([1.7])
    audio_sample_weight = torch.tensor([0.6]) if audio_present else None
    expected = joint_prediction_loss(
        student,
        teacher,
        video_mask=video_mask,
        audio_mask=audio_mask,
        video_sample_weight=video_sample_weight,
        audio_sample_weight=audio_sample_weight,
        balance=balance,
        mask_normalization=normalization,
        video_weight=1.3,
        audio_weight=0.7,
    )
    actual, video, audio = _teacher_joint_loss(
        student,
        teacher,
        video_mask=video_mask,
        audio_mask=audio_mask,
        video_sample_weight=video_sample_weight,
        audio_sample_weight=audio_sample_weight,
        balance=balance,
        mask_normalization=normalization,
        video_weight=1.3,
        audio_weight=0.7,
        mag_weight=1.0,
        dc_weight=1.0,
    )
    torch.testing.assert_close(actual, expected.loss)
    torch.testing.assert_close(video, expected.video_loss)
    torch.testing.assert_close(audio, expected.audio_loss)
