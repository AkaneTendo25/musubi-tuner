import pytest
import torch

from musubi_tuner.minimax_h3.training import (
    H3ModelPrediction,
    joint_velocity_loss,
    prepare_joint_noisy_inputs,
    prepare_soar_auxiliary_inputs,
    shift_sigma,
)
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser


def _joint_example(*, rollout_requires_grad: bool = False):
    video = torch.tensor([[[[[2.0, -1.0]]]]])
    video_noise = torch.tensor([[[[[10.0, 5.0]]]]])
    audio = torch.tensor([[[[3.0, -2.0]]]])
    audio_noise = torch.tensor([[[[9.0, 6.0]]]])
    current_base = torch.tensor([0.4])
    inputs = prepare_joint_noisy_inputs(video, audio, video_noise, audio_noise, current_base)
    rollout = H3ModelPrediction(
        torch.tensor([[[[[0.75, -0.25]]]]], requires_grad=rollout_requires_grad),
        torch.tensor([[[[0.5, -1.0]]]], requires_grad=rollout_requires_grad),
    )
    return video, audio, video_noise, audio_noise, inputs, rollout


def test_soar_auxiliary_uses_h3_data_ward_sign_and_shared_shifted_schedule():
    video, audio, video_noise, audio_noise, inputs, rollout = _joint_example()
    next_base = torch.tensor([0.35])
    auxiliary_base = torch.tensor([0.7])

    result = prepare_soar_auxiliary_inputs(
        inputs,
        video,
        audio,
        video_noise,
        audio_noise,
        rollout,
        next_base,
        auxiliary_base,
        video_shift=12.0,
        audio_shift=3.0,
    )

    video_next = shift_sigma(next_base, 12.0)
    audio_next = shift_sigma(next_base, 3.0)
    video_aux = shift_sigma(auxiliary_base, 12.0)
    audio_aux = shift_sigma(auxiliary_base, 3.0)
    torch.testing.assert_close(result.video_sigma, video_aux)
    torch.testing.assert_close(result.audio_sigma, audio_aux)
    assert not torch.equal(result.video_sigma, result.audio_sigma)

    def expected(clean, state, noise, velocity, sigma_now, sigma_next, sigma_aux):
        rolled = state + (sigma_now - sigma_next) * velocity
        alpha = (sigma_aux - sigma_next) / (1.0 - sigma_next)
        auxiliary = (1.0 - alpha) * rolled + alpha * noise
        # H3 predicts data-ward velocity, the negative of the SOAR paper's
        # noise-ward convention.
        target = (clean - auxiliary) / sigma_aux
        return auxiliary, target

    expected_video, expected_video_target = expected(
        video, inputs.video, video_noise, rollout.video, inputs.video_sigma, video_next, video_aux
    )
    expected_audio, expected_audio_target = expected(
        audio, inputs.audio, audio_noise, rollout.audio, inputs.audio_sigma, audio_next, audio_aux
    )
    torch.testing.assert_close(result.video, expected_video)
    torch.testing.assert_close(result.audio, expected_audio)
    torch.testing.assert_close(result.video_target, expected_video_target)
    torch.testing.assert_close(result.audio_target, expected_audio_target)
    torch.testing.assert_close(result.video + video_aux.view(-1, 1, 1, 1, 1) * result.video_target, video)
    torch.testing.assert_close(result.audio + audio_aux.view(-1, 1, 1, 1) * result.audio_target, audio)


def test_soar_same_noise_renoise_reaches_original_noise_endpoint_and_stops_rollout_gradient():
    video, audio, video_noise, audio_noise, inputs, rollout = _joint_example(rollout_requires_grad=True)
    ones = torch.ones(1)
    result = prepare_soar_auxiliary_inputs(
        inputs,
        video,
        audio,
        video_noise,
        audio_noise,
        rollout,
        torch.tensor([0.35]),
        ones,
        video_shift=12.0,
        audio_shift=3.0,
    )

    torch.testing.assert_close(result.video, video_noise)
    torch.testing.assert_close(result.audio, audio_noise)
    torch.testing.assert_close(result.video_target, video - video_noise)
    torch.testing.assert_close(result.audio_target, audio - audio_noise)
    assert not result.video.requires_grad
    assert not result.audio.requires_grad


def test_soar_correction_loss_has_finite_nonzero_student_gradient():
    video, audio, video_noise, audio_noise, inputs, rollout = _joint_example()
    result = prepare_soar_auxiliary_inputs(
        inputs,
        video,
        audio,
        video_noise,
        audio_noise,
        rollout,
        torch.tensor([0.35]),
        torch.tensor([0.7]),
        video_shift=12.0,
        audio_shift=3.0,
    )
    video_prediction = torch.zeros_like(result.video_target, requires_grad=True)
    audio_prediction = torch.zeros_like(result.audio_target, requires_grad=True)
    loss = joint_velocity_loss(H3ModelPrediction(video_prediction, audio_prediction), result).loss
    loss.backward()

    assert torch.isfinite(loss)
    assert video_prediction.grad is not None and torch.isfinite(video_prediction.grad).all()
    assert audio_prediction.grad is not None and torch.isfinite(audio_prediction.grad).all()
    assert video_prediction.grad.abs().sum() > 0
    assert audio_prediction.grad.abs().sum() > 0


def test_h3_soar_parser_defaults_disabled_and_accepts_explicit_configuration():
    defaults = create_parser().parse_args([])
    assert defaults.h3_soar_weight == 0.0
    assert defaults.h3_soar_aux_points == 1
    assert defaults.h3_soar_rollout_steps == 20

    configured = create_parser().parse_args(
        ["--h3_soar_weight", "0.5", "--h3_soar_aux_points", "2", "--h3_soar_rollout_steps", "30"]
    )
    assert configured.h3_soar_weight == 0.5
    assert configured.h3_soar_aux_points == 2
    assert configured.h3_soar_rollout_steps == 30


def test_h3_soar_validation_is_inert_at_zero_and_rejects_guidance_distillation_when_enabled():
    trainer = MiniMaxH3NetworkTrainer()
    disabled = create_parser().parse_args(["--h3_guidance_distillation_scale", "4"])
    trainer._validate_soar_args(disabled)

    enabled = create_parser().parse_args(["--h3_soar_weight", "0.5", "--h3_guidance_distillation_scale", "4"])
    with pytest.raises(ValueError, match="guidance distillation"):
        trainer._validate_soar_args(enabled)

    ranged = create_parser().parse_args(["--h3_soar_weight", "0.5", "--h3_guidance_scale_range", "2.5,3.5"])
    with pytest.raises(ValueError, match="guidance scale range"):
        trainer._validate_soar_args(ranged)


def test_soar_rejects_zero_auxiliary_sigma():
    video, audio, video_noise, audio_noise, inputs, rollout = _joint_example()
    with pytest.raises(ValueError, match="positive"):
        prepare_soar_auxiliary_inputs(
            inputs,
            video,
            audio,
            video_noise,
            audio_noise,
            rollout,
            torch.zeros(1),
            torch.zeros(1),
        )
