import shutil
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

import musubi_tuner.minimax_h3.model as h3_model
import musubi_tuner.minimax_h3_train_network as h3_train_network
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_AUDIO_LOSS_MASK_KEY,
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_KEYFRAME_VIDEO_ROWS_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
    H3_REFERENCE_IMAGE_SHORT_EDGE_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_MAX_CAPTION_TOKENS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_VISUAL_MAX_PIXELS_KEY,
    H3_VIDEO_GEOMETRY_KEY,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.crepa import H3CREPA, H3CREPAConfig, parse_crepa_config
from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
from musubi_tuner.minimax_h3.masking import (
    audio_mask_to_rows,
    sample_video_mask,
    video_mask_to_rows,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3Attention, MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.packing import (
    MiniMaxH3ReferenceGeometry,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    build_t2va_packed_sequence,
    pack_audio_latents,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from musubi_tuner.minimax_h3 import training as h3_training
from musubi_tuner.minimax_h3.training import (
    OBSERVED_AUDIO_SIGMA,
    OBSERVED_VIDEO_SIGMA,
    H3ModelPrediction,
    contrastive_guidance_target,
    guidance_consistent_prediction,
    guidance_scale_for_sigma,
    joint_prediction_loss,
    joint_velocity_loss,
    map_sigma_between_shifts,
    prepare_joint_noisy_inputs,
    shift_sigma,
    unshift_sigma,
)
from musubi_tuner.minimax_h3.validation import masked_squared_error_sum
from musubi_tuner.minimax_h3_cache_dino_features import (
    _dino_cache_is_current,
    _latent_cache_identity,
    _save_features,
    dino_cache_path,
)
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
from musubi_tuner.networks import lora_minimax_h3


def test_ref2va_reference_cache_selects_matching_video_and_audio_variants():
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="ref2va")
    batch = {
        H3_REFERENCE_KINDS_KEY: [torch.tensor([0, 1, 2])],
        H3_REFERENCE_VIDEO_SHAPES_KEY: [torch.tensor([[1, 2, 2], [1, 2, 2], [0, 0, 0]])],
        H3_REFERENCE_AUDIO_LENGTHS_KEY: [torch.tensor([0, 1, 1])],
        H3_REFERENCE_VIDEO_ROWS_KEY: [torch.arange(8, dtype=torch.float32).reshape(2, 4)],
        H3_REFERENCE_AUDIO_ROWS_KEY: [torch.arange(8, dtype=torch.float32).reshape(4, 2)],
    }

    video_refs, video_rows, video_audio = backend._reference_cache(
        batch,
        patch_size=(1, 2, 2),
        video_width=4,
        audio_width=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        reference_modality="video",
    )
    audio_refs, audio_video, audio_rows = backend._reference_cache(
        batch,
        patch_size=(1, 2, 2),
        video_width=4,
        audio_width=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        reference_modality="audio",
    )

    assert [(ref.kind, ref.num_audio_latents) for ref in video_refs] == [(0, 0), (1, 0)]
    assert video_rows.shape == (2, 4) and video_audio.shape == (0, 2)
    assert [(ref.kind, ref.num_audio_latents) for ref in audio_refs] == [(0, 0), (2, 1), (2, 1)]
    assert audio_video.shape == (1, 4) and audio_rows.shape == (4, 2)


class _CREPABlock(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states):
        return hidden_states * self.scale


class _CREPATransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_CREPABlock(1.0), _CREPABlock(2.0), _CREPABlock(3.0)])

    def forward(self, hidden_states, checkpoint_blocks=False):
        for block in self.blocks:
            if checkpoint_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(block, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(hidden_states)
        return hidden_states


def test_h3_crepa_single_flag_parses_defaults_and_overrides():
    assert create_parser().parse_args([]).crepa is None
    assert create_parser().parse_args(["--crepa"]).crepa == []
    values = create_parser().parse_args(["--crepa", "student_block=4", "weight=0.02"]).crepa
    config = parse_crepa_config(values)
    assert config.student_block == 4
    assert config.teacher_block == 33
    assert config.weight == pytest.approx(0.02)


def test_h3_dino_cache_path_follows_the_native_architecture_suffix(tmp_path):
    assert dino_cache_path(tmp_path / "clip_mmh3.safetensors").name == "clip_mmh3_dino.safetensors"
    with pytest.raises(ValueError, match="latent-cache name"):
        dino_cache_path(tmp_path / "clip.safetensors")


@pytest.mark.parametrize("atomic", [False, True])
def test_h3_dino_cache_write_is_complete_and_metadata_bearing(tmp_path, atomic):
    path = tmp_path / "clip_mmh3_dino.safetensors"
    expected = torch.randn(2, 3, 4, dtype=torch.float16)
    _save_features(path, expected, {"dino_model": "probe"}, atomic=atomic)
    torch.testing.assert_close(load_file(path)["h3_dino_features"], expected)
    with safe_open(path, framework="pt") as handle:
        assert handle.metadata() == {"dino_model": "probe"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_h3_dino_cache_identity_detects_rebuilt_latents(tmp_path):
    latent = tmp_path / "clip_mmh3.safetensors"
    dino = dino_cache_path(latent)
    save_file({"latents": torch.zeros(1)}, latent)
    _save_features(dino, torch.zeros(1, 1, 1), {"dino_model": "probe", **_latent_cache_identity(latent)}, atomic=False)
    assert _dino_cache_is_current(dino, latent, "probe")

    save_file({"latents": torch.zeros(2)}, latent)
    assert not _dino_cache_is_current(dino, latent, "probe")


@pytest.mark.parametrize(
    "values",
    [
        ["unknown=1"],
        ["weight=0"],
        ["weight=0.1", "weight=0.2"],
        ["student_block=3", "teacher_block=2"],
        ["neighbors=-1"],
        ["weight"],
    ],
)
def test_h3_crepa_rejects_invalid_configuration(values):
    with pytest.raises(ValueError):
        parse_crepa_config(values)


def test_h3_crepa_extracts_only_target_video_rows_and_backpropagates(tmp_path):
    config = H3CREPAConfig(student_block=0, teacher_block=2, weight=0.1, tau=1.0, neighbors=1)
    transformer = _CREPATransformer()
    crepa = H3CREPA(hidden_size=4, config=config)
    crepa.install(transformer)
    # Packed rows 0/1 are text or references. Target video is frame-major at
    # rows 2..5, with two spatial rows per latent frame.
    crepa.set_layout(torch.tensor([2, 3, 4, 5]), frames=2, rows_per_frame=2)
    crepa.begin_step(True)
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4).requires_grad_()
    transformer(hidden)

    expected_student = hidden[:, 2:].reshape(1, 2, 2, 4)
    torch.testing.assert_close(crepa._student, expected_student)
    assert not crepa._teacher.requires_grad
    loss, metrics = crepa.loss()
    loss.backward()
    assert metrics["crepa/alignment"] <= 1.0
    assert metrics["crepa/similarity_self"] <= 1.0
    assert hidden.grad is not None and hidden.grad[:, 2:].abs().sum() > 0
    assert hidden.grad[:, :2].abs().sum() == 0
    assert all(parameter.grad is not None for parameter in crepa.projector.parameters())

    crepa.save_state(tmp_path)
    restored = H3CREPA(hidden_size=4, config=config)
    assert restored.load_state(tmp_path)
    for expected, actual in zip(crepa.projector.parameters(), restored.projector.parameters()):
        torch.testing.assert_close(actual, expected)


def test_h3_crepa_can_defer_one_ema_update_across_dataset_batch():
    config = H3CREPAConfig(
        student_block=0,
        teacher_block=2,
        weight=0.1,
        similarity_threshold=0.5,
        similarity_ema_decay=0.9,
    )
    transformer = _CREPATransformer()
    crepa = H3CREPA(hidden_size=4, config=config)
    crepa.install(transformer)
    crepa.set_layout(torch.tensor([0, 1, 2, 3]), frames=2, rows_per_frame=2)
    alignments = []
    for _ in range(2):
        crepa.begin_step(True)
        transformer(torch.randn(1, 6, 4))
        _, metrics = crepa.loss(update_similarity_threshold=False)
        alignments.append(metrics["crepa/alignment"])
    assert crepa._similarity_ema is None
    crepa.update_similarity_threshold(sum(alignments) / len(alignments))
    assert crepa._similarity_ema == pytest.approx(sum(alignments) / len(alignments))


def test_h3_crepa_is_inert_when_disabled():
    transformer = _CREPATransformer()
    crepa = H3CREPA(4, H3CREPAConfig(student_block=0, teacher_block=2))
    crepa.install(transformer)
    crepa.set_layout(torch.tensor([0, 1]), frames=1, rows_per_frame=2)
    crepa.begin_step(False)
    transformer(torch.ones(1, 2, 4, requires_grad=True))
    assert crepa._student is None
    assert crepa._teacher is None


def test_h3_crepa_hooks_are_symmetric_during_checkpoint_recomputation():
    transformer = _CREPATransformer()
    crepa = H3CREPA(4, H3CREPAConfig(student_block=0, teacher_block=2))
    crepa.install(transformer)
    crepa.set_layout(torch.tensor([0, 1, 2, 3]), frames=2, rows_per_frame=2)
    crepa.begin_step(True)
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    transformer(hidden, checkpoint_blocks=True)
    loss, _ = crepa.loss()
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_h3_crepa_uses_existing_optimizer_group():
    trainer = MiniMaxH3NetworkTrainer()
    trainer._crepa = H3CREPA(4, H3CREPAConfig(student_block=0, teacher_block=2))
    original = nn.Parameter(torch.ones(1))
    groups = trainer.extra_trainable_params(None, SimpleNamespace(device=torch.device("cpu")), None, None, [{"params": [original]}])
    assert len(groups) == 1
    assert groups[0]["params"][0] is original
    assert len(groups[0]["params"]) == 1 + len(list(trainer._crepa.projector.parameters()))


def test_h3_crepa_schedule_cutoff_and_resume_state(tmp_path):
    config = H3CREPAConfig(
        student_block=0,
        teacher_block=2,
        weight=0.2,
        schedule="linear",
        warmup_steps=2,
        max_steps=10,
        cutoff_step=9,
        similarity_threshold=0.5,
        similarity_ema_decay=0.0,
    )
    crepa = H3CREPA(4, config)
    crepa.begin_step(True, 1)
    assert crepa.active
    assert crepa._effective_weight == pytest.approx(0.1)
    assert crepa.status_metrics()["crepa/cutoff"] == 0
    crepa.begin_step(True, 6)
    assert crepa._effective_weight == pytest.approx(0.1)
    crepa._similarity_ema = 0.75
    crepa._cutoff_triggered = True
    crepa.save_state(tmp_path)
    restored = H3CREPA(4, config)
    assert restored.load_state(tmp_path)
    assert restored._similarity_ema == pytest.approx(0.75)
    restored.begin_step(True, 7)
    assert not restored.active
    assert restored.status_metrics()["crepa/cutoff"] == 1


def test_h3_crepa_zero_warmup_weight_is_not_reported_as_cutoff():
    config = H3CREPAConfig(
        student_block=0,
        teacher_block=2,
        schedule="cosine",
        warmup_steps=2,
        max_steps=10,
    )
    crepa = H3CREPA(4, config)
    crepa.begin_step(True, 0)
    assert not crepa.active
    assert crepa.status_metrics() == {"crepa/weight": 0.0, "crepa/cutoff": 0.0}


def test_h3_crepa_dino_mode_aligns_cached_frame_features():
    config = H3CREPAConfig(mode="dino", dino_model="dinov2_vits14", student_block=0, teacher_block=2)
    transformer = _CREPATransformer()
    crepa = H3CREPA(4, config)
    assert crepa.projector[0].weight.shape == (4, 4)
    assert crepa.projector[2].weight.shape == (384, 4)
    crepa.install(transformer)
    crepa.set_layout(torch.tensor([1, 2, 3, 4]), frames=2, rows_per_frame=2)
    crepa.begin_step(True)
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    transformer(hidden)
    dino = torch.randn(1, 4, 3, 384)
    loss, metrics = crepa.loss(dino)
    loss.backward()
    assert torch.isfinite(loss)
    assert -1 <= metrics["crepa/alignment"] <= 1
    assert hidden.grad is not None and hidden.grad[:, 1:].abs().sum() > 0


def test_h3_crepa_neighbor_objective_matches_valid_comparison_normalization():
    config = H3CREPAConfig(
        student_block=0,
        teacher_block=2,
        weight=0.2,
        tau=1.0,
        neighbors=1,
        normalize=False,
    )
    crepa = H3CREPA(1, config)
    crepa.projector = nn.Identity()
    crepa._student = torch.tensor([[[[1.0]], [[2.0]]]])
    crepa._teacher = torch.tensor([[[3.0], [4.0]]])
    crepa._effective_weight = config.weight
    loss, metrics = crepa.loss()
    # The numerator is temporally weighted, so the denominator is the sum of the
    # same weights (2 self pairs at 1.0 plus 2 neighbour pairs at exp(-1/tau)),
    # not the plain comparison count.
    expected = -config.weight * (11.0 + 10.0 / torch.e) / (2.0 + 2.0 / torch.e)
    torch.testing.assert_close(loss, loss.new_tensor(expected))
    expected_alignment = ((3.0 + 4.0 / torch.e) + (8.0 + 6.0 / torch.e)) / (2.0 * (1.0 + 1.0 / torch.e))
    assert metrics["crepa/alignment"] == pytest.approx(float(expected_alignment))
    assert metrics["crepa/similarity_self"] == pytest.approx(5.5)
    assert "crepa/alignment_ema" not in metrics


def test_h3_shift_round_trip_and_cross_modality_mapping():
    base = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
    video = shift_sigma(base, 12.0)
    audio = shift_sigma(base, 3.0)

    torch.testing.assert_close(unshift_sigma(video, 12.0), base)
    torch.testing.assert_close(map_sigma_between_shifts(video, source_shift=12.0, target_shift=3.0), audio)
    assert video[0] == audio[0] == 0
    assert video[-1] == audio[-1] == 1


def test_h3_joint_noising_uses_data_ward_velocity_and_model_time():
    video = torch.tensor([[[[[2.0, 4.0]]]]])
    video_noise = torch.tensor([[[[[10.0, 20.0]]]]])
    audio = torch.tensor([[[[3.0, 6.0]]]])
    audio_noise = torch.tensor([[[[9.0, 12.0]]]])
    base_sigma = torch.tensor([0.25])
    video_sigma = shift_sigma(base_sigma, 12.0)
    audio_sigma = shift_sigma(base_sigma, 3.0)

    result = prepare_joint_noisy_inputs(video, audio, video_noise, audio_noise, base_sigma)

    torch.testing.assert_close(result.video_sigma, video_sigma)
    torch.testing.assert_close(result.audio_sigma, audio_sigma)
    torch.testing.assert_close(result.video, (1 - video_sigma.item()) * video + video_sigma.item() * video_noise)
    torch.testing.assert_close(result.audio, (1 - audio_sigma.item()) * audio + audio_sigma.item() * audio_noise)
    torch.testing.assert_close(result.video_target, video - video_noise)
    torch.testing.assert_close(result.audio_target, audio - audio_noise)
    torch.testing.assert_close(result.video_timestep, 1 - video_sigma)
    torch.testing.assert_close(result.audio_timestep, 1 - audio_sigma)


@pytest.mark.parametrize("video_shift", [12.0, 6.0, 3.0, 1.0])
def test_h3_joint_noising_keeps_audio_synchronized_at_any_video_shift(video_shift):
    # Both modalities sit at the same underlying schedule position, so each sigma
    # is the shared base coordinate shifted by that modality's own shift.
    video = torch.zeros(1, 1, 1, 1, 2)
    audio = torch.zeros(1, 1, 1, 2)
    base = torch.tensor([0.5])

    result = prepare_joint_noisy_inputs(
        video,
        audio,
        torch.ones_like(video),
        torch.ones_like(audio),
        base,
        video_shift=video_shift,
        audio_shift=3.0,
    )

    torch.testing.assert_close(result.video_sigma, shift_sigma(base, video_shift))
    torch.testing.assert_close(result.audio_sigma, shift_sigma(base, 3.0))


@pytest.mark.parametrize(
    ("observed", "pinned_sigma"),
    [("video", OBSERVED_VIDEO_SIGMA), ("audio", OBSERVED_AUDIO_SIGMA)],
)
def test_h3_observed_modality_is_pinned_to_the_released_conditioning_level(observed, pinned_sigma):
    # The observed modality is read, not generated, so it leaves the sampled
    # schedule and sits at the noise level the released transformer uses for
    # conditioning. The generated modality keeps its own shifted schedule.
    video = torch.zeros(1, 1, 1, 1, 2)
    audio = torch.zeros(1, 1, 1, 2)
    base = torch.tensor([0.5])

    result = prepare_joint_noisy_inputs(
        video,
        audio,
        torch.ones_like(video),
        torch.ones_like(audio),
        base,
        video_shift=12.0,
        audio_shift=3.0,
        observed=observed,
    )

    if observed == "video":
        torch.testing.assert_close(result.video_sigma, torch.full_like(base, pinned_sigma))
        torch.testing.assert_close(result.audio_sigma, shift_sigma(base, 3.0))
    else:
        torch.testing.assert_close(result.audio_sigma, torch.full_like(base, pinned_sigma))
        torch.testing.assert_close(result.video_sigma, shift_sigma(base, 12.0))


def test_h3_observed_audio_passes_through_untouched():
    # Audio conditioning rides sigma 0, so the packed rows must be the clean
    # latents themselves and the timestep must be exactly 1.0.
    video = torch.zeros(1, 1, 1, 1, 2)
    audio = torch.full((1, 1, 1, 2), 0.25)
    base = torch.tensor([0.5])

    result = prepare_joint_noisy_inputs(
        video,
        audio,
        torch.ones_like(video),
        torch.ones_like(audio),
        base,
        observed="audio",
    )

    torch.testing.assert_close(result.audio, audio)
    torch.testing.assert_close(result.audio_timestep, torch.ones_like(base))


def test_h3_observed_modality_requires_both_modalities():
    video = torch.zeros(1, 1, 1, 1, 2)
    base = torch.tensor([0.5])

    with pytest.raises(ValueError, match="requires both video and audio"):
        prepare_joint_noisy_inputs(video, None, torch.ones_like(video), None, base, observed="audio")


def test_h3_joint_loss_ignores_the_observed_modality():
    # Zeroing the observed modality's weight must remove it from the objective
    # entirely, leaving the generated modality's own mean.
    prediction = H3ModelPrediction(video=torch.ones(1, 4), audio=torch.zeros(1, 4))
    inputs = SimpleNamespace(video_target=torch.zeros(1, 4), audio_target=torch.ones(1, 4))

    result = joint_velocity_loss(prediction, inputs, video_weight=0.0, audio_weight=1.0)

    torch.testing.assert_close(result.loss, result.audio_loss)
    # The zero-weight modality's squared error is never materialized: it is the
    # largest allocation in a v2a/a2v step and multiplying it by zero afterwards
    # also reported a meaningless loss/video on the divergence panel.
    assert result.video_elements == 0
    assert float(result.video_loss) == 0.0


def test_h3_joint_loss_skips_the_zero_weight_modality_without_changing_the_total():
    torch.manual_seed(0)
    prediction = H3ModelPrediction(video=torch.randn(2, 5), audio=torch.randn(2, 3))
    inputs = SimpleNamespace(video_target=torch.randn(2, 5), audio_target=torch.randn(2, 3))
    weights = torch.tensor([0.25, 1.5])

    calls: list[str] = []
    original = h3_training._modality_loss

    def counting(prediction_tensor, target_tensor, mask, sample_weight):
        calls.append(tuple(prediction_tensor.shape))
        return original(prediction_tensor, target_tensor, mask, sample_weight)

    for balance in ("token", "modality"):
        for video_weight, audio_weight in ((0.0, 1.0), (1.0, 0.0)):
            expected = joint_velocity_loss(
                prediction,
                inputs,
                balance=balance,
                video_sample_weight=weights,
                audio_sample_weight=weights,
                video_weight=video_weight,
                audio_weight=audio_weight,
            )
            calls.clear()
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(h3_training, "_modality_loss", counting)
                actual = joint_velocity_loss(
                    prediction,
                    inputs,
                    balance=balance,
                    video_sample_weight=weights,
                    audio_sample_weight=weights,
                    video_weight=video_weight,
                    audio_weight=audio_weight,
                )
            # Only the nonzero-weight modality is evaluated, and the total the
            # optimizer sees is unchanged.
            assert calls == [(2, 3) if video_weight == 0.0 else (2, 5)]
            torch.testing.assert_close(actual.loss, expected.loss)


def test_h3_prediction_preservation_detaches_reference_and_ignores_observed_modality():
    video = torch.ones(1, 2, requires_grad=True)
    audio = torch.ones(1, 3, requires_grad=True)
    reference_video = torch.zeros_like(video, requires_grad=True)
    reference_audio = torch.zeros_like(audio, requires_grad=True)

    result = joint_prediction_loss(
        H3ModelPrediction(video, audio),
        H3ModelPrediction(reference_video, reference_audio),
        balance="modality",
        video_weight=0.0,
        audio_weight=1.0,
    )
    result.loss.backward()

    torch.testing.assert_close(result.loss, result.audio_loss)
    assert video.grad is not None and torch.count_nonzero(video.grad) == 0
    assert audio.grad is not None and torch.count_nonzero(audio.grad) > 0
    assert reference_video.grad is None
    assert reference_audio.grad is None


def test_h3_joint_noising_gives_equal_sigmas_for_equal_shifts():
    # At video_shift == audio_shift the two schedules are the same schedule, so
    # the two sigmas must coincide for any base coordinate.
    video = torch.zeros(1, 1, 1, 1, 2)
    audio = torch.zeros(1, 1, 1, 2)
    base = torch.tensor([0.5])

    result = prepare_joint_noisy_inputs(
        video,
        audio,
        torch.ones_like(video),
        torch.ones_like(audio),
        base,
        video_shift=3.0,
        audio_shift=3.0,
    )

    torch.testing.assert_close(result.video_sigma, result.audio_sigma)


def test_h3_t2va_packing_matches_released_row_order_rope_clock_and_inverses():
    video = torch.arange(1 * 4 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 4, 2, 4, 4)
    audio = torch.arange(1 * 2 * 6 * 3, dtype=torch.float32).reshape(1, 2, 6, 3)
    video_rows = patchify_video_latents(video, (1, 2, 2))
    audio_rows = pack_audio_latents(audio)
    layout = build_t2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=(1, 2, 2),
    )

    assert video_rows.shape == (1, 8, 16)
    assert audio_rows.shape == (1, 6, 6)
    assert layout.sequence_length == 18
    torch.testing.assert_close(layout.text_indices, torch.arange(4))
    torch.testing.assert_close(layout.audio_indices, torch.arange(4, 10))
    torch.testing.assert_close(layout.video_indices, torch.arange(10, 18))
    torch.testing.assert_close(layout.position_ids[layout.audio_indices, 0], torch.tensor([4, 5, 6, 4, 5, 6], dtype=torch.float64))
    torch.testing.assert_close(
        layout.position_ids[layout.audio_indices, 2], torch.tensor([0, 0, 0, 16, 16, 16], dtype=torch.float64)
    )
    torch.testing.assert_close(layout.position_ids[layout.video_indices[:4], 0], torch.full((4,), 4.0, dtype=torch.float64))
    torch.testing.assert_close(
        layout.position_ids[layout.video_indices[4:], 0],
        torch.full((4,), 4.0 + 5.0 / 3.0, dtype=torch.float64),
    )

    timesteps, timestep_indices = build_row_timesteps(layout, torch.tensor([0.3]), torch.tensor([0.7]))
    torch.testing.assert_close(timesteps, torch.tensor([0.3, 0.7]))
    assert bool((timestep_indices[layout.audio_indices] == 1).all())
    assert bool((timestep_indices[layout.text_indices] == 0).all())
    assert bool((timestep_indices[layout.video_indices] == 0).all())
    torch.testing.assert_close(
        unpatchify_video_tokens(video_rows, latent_shape=(4, 2, 4, 4), patch_size=(1, 2, 2)),
        video,
    )
    torch.testing.assert_close(unpack_audio_tokens(audio_rows, num_audio_latents=3), audio)


def _anchor_layout(anchors=(), *, condition_audio=0, frames=7):
    return build_t2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        num_latent_frames=frames,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=(1, 2, 2),
        keyframe_anchors=anchors,
        num_condition_audio_latents=condition_audio,
    )


def _anchor_times(layout, count, rows_per_frame=4):
    return [float(layout.position_ids[layout.video_indices[index * rows_per_frame], 0]) for index in range(count)]


def test_h3_released_keyframe_anchors_keep_their_exact_positions():
    # "first" sits at the text boundary and "last" one latent window past the
    # final window's start, which is the released FL2VA contract.
    layout = _anchor_layout(("first", "last"))
    span = sum(5.0 / 3.0 * (1, 4, 4, 4, 4)[index % 5] for index in range(7))

    assert _anchor_times(layout, 2) == [4.0, 4.0 + span - 5.0 / 3.0]


def test_h3_integer_keyframe_anchor_lands_on_its_target_latent_window():
    # An integer anchor names a latent window, so it must coincide with the
    # position of that window's own target rows.
    layout = _anchor_layout((3,))
    target_start = layout.video_indices[layout.num_condition_video_rows :]

    assert _anchor_times(layout, 1)[0] == float(layout.position_ids[target_start[3 * 4], 0])


def test_h3_last_anchor_is_not_the_final_latent_index():
    # "last" is the final pixel frame; index N-1 is the final latent window's
    # start. Collapsing them would silently move the released anchor.
    string_layout = _anchor_layout(("last",))
    index_layout = _anchor_layout((6,))

    assert _anchor_times(string_layout, 1) != _anchor_times(index_layout, 1)


def test_h3_keyframe_anchors_reject_duplicate_first_and_zero():
    with pytest.raises(ValueError, match="unique"):
        _anchor_layout(("first", 0))


@pytest.mark.parametrize("anchor", [7, -1, True, 1.5])
def test_h3_keyframe_anchors_reject_invalid_indices(anchor):
    with pytest.raises(ValueError):
        _anchor_layout((anchor,))


def test_h3_condition_audio_shares_the_opening_target_coordinates():
    # Condition audio duplicates the opening target span, so both use the same
    # rotary coordinates and remain aligned with the video timeline.
    layout = _anchor_layout(condition_audio=2)
    audio = layout.audio_indices

    assert layout.num_condition_audio_rows == 4
    condition_times = layout.position_ids[audio[: layout.num_condition_audio_rows], 0]
    target_times = layout.position_ids[audio[layout.num_condition_audio_rows :], 0]
    torch.testing.assert_close(condition_times, torch.tensor([4.0, 5.0, 4.0, 5.0], dtype=torch.float64))
    torch.testing.assert_close(target_times, torch.tensor([4.0, 5.0, 6.0, 4.0, 5.0, 6.0], dtype=torch.float64))
    torch.testing.assert_close(condition_times[:2], target_times[:2])
    torch.testing.assert_close(condition_times[2:], target_times[3:5])


def test_h3_condition_audio_is_absent_by_default():
    # The default layout must be untouched: no condition rows, and the target
    # audio still opens at the text boundary.
    layout = _anchor_layout()

    assert layout.num_condition_audio_rows == 0
    torch.testing.assert_close(
        layout.position_ids[layout.audio_indices, 0],
        torch.tensor([4.0, 5.0, 6.0, 4.0, 5.0, 6.0], dtype=torch.float64),
    )


def test_h3_condition_audio_rows_take_the_condition_timestep():
    layout = _anchor_layout(condition_audio=2)

    timesteps, indices = build_row_timesteps(
        layout, torch.tensor([0.3]), torch.tensor([0.7]), condition_audio_timestep=torch.tensor([1.0])
    )

    audio = layout.audio_indices
    assert bool((timesteps[indices[audio[: layout.num_condition_audio_rows]]] == 1.0).all())
    assert bool((timesteps[indices[audio[layout.num_condition_audio_rows :]]] == 0.7).all())


def test_h3_condition_audio_requires_an_audio_target():
    with pytest.raises(ValueError, match="audio conditioning requires an audio target"):
        build_t2va_packed_sequence(
            torch.ones(4, dtype=torch.long),
            num_latent_frames=7,
            latent_height=4,
            latent_width=4,
            num_audio_latents=0,
            patch_size=(1, 2, 2),
            num_condition_audio_latents=2,
        )


def _row_timestep_layout():
    return build_t2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=(1, 2, 2),
    )


def test_h3_row_timesteps_accept_one_value_per_target_video_row():
    # A packed sequence may carry a different noise level on every video row.
    # The transformer selects modulation through timestep_indices, so the only
    # requirement here is that each row recovers the value it was given.
    layout = _row_timestep_layout()
    per_row = torch.linspace(0.1, 0.8, int(layout.video_indices.numel()))

    timesteps, indices = build_row_timesteps(layout, per_row, torch.tensor([0.9]), per_row_timesteps=True)

    torch.testing.assert_close(timesteps[indices[layout.video_indices]], per_row)
    assert bool((timesteps[indices[layout.audio_indices]] == 0.9).all())
    # Text follows the first video value unless told otherwise.
    assert bool((timesteps[indices[layout.text_indices]] == per_row[0]).all())


def test_h3_row_timesteps_accept_one_value_per_target_audio_row():
    layout = _row_timestep_layout()
    per_row = torch.linspace(0.2, 0.6, int(layout.audio_indices.numel()))

    timesteps, indices = build_row_timesteps(layout, torch.tensor([0.4]), per_row, per_row_timesteps=True)

    torch.testing.assert_close(timesteps[indices[layout.audio_indices]], per_row)
    assert bool((timesteps[indices[layout.video_indices]] == 0.4).all())


def test_h3_row_timesteps_can_pin_the_text_prefix_independently():
    layout = _row_timestep_layout()
    per_row = torch.linspace(0.1, 0.8, int(layout.video_indices.numel()))

    timesteps, indices = build_row_timesteps(
        layout, per_row, torch.tensor([0.9]), per_row_timesteps=True, text_timestep=torch.tensor([0.5])
    )

    assert bool((timesteps[indices[layout.text_indices]] == 0.5).all())


def test_h3_row_timesteps_reject_vectors_without_the_opt_in():
    # Without per_row_timesteps the function keeps its original contract, so a
    # vector cannot reach the per-row path by accident even when its length
    # happens to match a target block.
    layout = _row_timestep_layout()
    per_row = torch.linspace(0.1, 0.8, int(layout.video_indices.numel()))

    with pytest.raises(ValueError, match="requires one video and audio timestep"):
        build_row_timesteps(layout, per_row, torch.tensor([0.9]))


def test_h3_row_timesteps_reject_a_length_that_matches_no_target_block():
    layout = _row_timestep_layout()

    with pytest.raises(ValueError, match="one value per target video row"):
        build_row_timesteps(layout, torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.9]), per_row_timesteps=True)


def test_h3_row_timesteps_are_unchanged_by_a_scalar_that_repeats_per_row():
    # Passing the same value once or once per row must produce the same packing,
    # so the generalization cannot alter any existing caller.
    layout = _row_timestep_layout()
    scalar_timesteps, scalar_indices = build_row_timesteps(layout, torch.tensor([0.3]), torch.tensor([0.7]))
    expanded = torch.full((int(layout.video_indices.numel()),), 0.3)
    vector_timesteps, vector_indices = build_row_timesteps(layout, expanded, torch.tensor([0.7]), per_row_timesteps=True)

    torch.testing.assert_close(scalar_timesteps, vector_timesteps)
    torch.testing.assert_close(scalar_indices, vector_indices)


def test_h3_image_packing_has_no_audio_rows():
    layout = build_t2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        num_latent_frames=1,
        latent_height=4,
        latent_width=4,
        num_audio_latents=0,
        patch_size=(1, 2, 2),
    )

    assert layout.audio_indices.numel() == 0
    assert layout.num_condition_audio_rows == 0
    assert layout.video_indices.numel() == 4
    assert layout.sequence_length == 8


def test_h3_audio_only_packing_has_no_video_rows():
    layout = build_t2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        num_latent_frames=0,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=(1, 2, 2),
    )

    assert layout.video_indices.numel() == 0
    assert layout.audio_indices.numel() == 6
    assert layout.sequence_length == 10


@pytest.mark.parametrize("target", ["video", "audio"])
def test_h3_modality_only_noising_and_loss(target):
    video = torch.zeros(1, 4, 1, 2, 2) if target == "video" else None
    audio = torch.zeros(1, 2, 6, 3) if target == "audio" else None
    inputs = prepare_joint_noisy_inputs(
        video,
        audio,
        torch.ones_like(video) if video is not None else None,
        torch.ones_like(audio) if audio is not None else None,
        torch.tensor([0.5]),
    )
    prediction = H3ModelPrediction(
        video=inputs.video_target.clone() if inputs.video_target is not None else None,
        audio=inputs.audio_target.clone() if inputs.audio_target is not None else None,
    )
    result = joint_velocity_loss(prediction, inputs)

    assert result.loss == 0
    assert (result.video_elements > 0) == (target == "video")
    assert (result.audio_elements > 0) == (target == "audio")


def test_h3_fl2va_keyframes_precede_targets_and_share_target_rotary_anchors():
    layout = build_t2va_packed_sequence(
        torch.tensor([1, 0, 1, 0]),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=(1, 2, 2),
        keyframe_anchors=("first", "last"),
    )

    assert layout.sequence_length == 26
    assert layout.num_condition_video_rows == 8
    torch.testing.assert_close(layout.video_indices[:8], torch.arange(4, 12))
    torch.testing.assert_close(layout.audio_indices, torch.arange(12, 18))
    torch.testing.assert_close(layout.video_indices[8:], torch.arange(18, 26))
    torch.testing.assert_close(layout.position_ids[4:8, 0], torch.full((4,), 4.0, dtype=torch.float64))
    torch.testing.assert_close(
        layout.position_ids[8:12, 0],
        torch.full((4,), 4.0 + 20.0 / 3.0, dtype=torch.float64),
    )

    timesteps, indices = build_row_timesteps(
        layout,
        torch.tensor([0.3]),
        torch.tensor([0.7]),
        torch.tensor([0.999]),
    )
    torch.testing.assert_close(timesteps, torch.tensor([0.3, 0.7, 0.999]))
    assert bool((indices[layout.video_indices[:8]] == 2).all())
    assert bool((indices[layout.video_indices[8:]] == 0).all())
    assert bool((indices[layout.audio_indices] == 1).all())


def test_h3_ref2va_packing_preserves_reference_order_and_shared_rotary_clock():
    references = (
        MiniMaxH3ReferenceGeometry(kind=0, num_latent_frames=1, latent_height=4, latent_width=4),
        MiniMaxH3ReferenceGeometry(kind=2, num_audio_latents=2),
        MiniMaxH3ReferenceGeometry(
            kind=1,
            num_latent_frames=2,
            latent_height=4,
            latent_width=4,
            num_audio_latents=3,
        ),
    )
    layout = build_ref2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        references,
        num_latent_frames=1,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
    )

    assert layout.sequence_length == 34
    assert layout.num_condition_video_rows == 12
    assert layout.num_condition_audio_rows == 10
    torch.testing.assert_close(layout.video_indices, torch.tensor([4, 5, 6, 7, *range(18, 26), *range(30, 34)]))
    torch.testing.assert_close(layout.audio_indices, torch.tensor([*range(8, 18), *range(26, 30)]))
    torch.testing.assert_close(layout.position_ids[8:12, 0], torch.tensor([5, 6, 5, 6], dtype=torch.float64))
    torch.testing.assert_close(layout.position_ids[12:18, 0], torch.tensor([7, 8, 9, 7, 8, 9], dtype=torch.float64))
    target_origin = 7.0 + 25.0 / 3.0
    torch.testing.assert_close(
        layout.position_ids[26:30, 0],
        torch.tensor([target_origin, target_origin + 1] * 2, dtype=torch.float64),
    )

    timesteps, indices = build_row_timesteps(
        layout,
        torch.tensor([0.2]),
        torch.tensor([0.6]),
        torch.tensor([0.999]),
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(timesteps, torch.tensor([0.2, 0.6, 0.999, 1.0]))
    assert bool((indices[layout.video_indices[:12]] == 2).all())
    assert bool((indices[layout.audio_indices[:10]] == 3).all())
    assert bool((indices[layout.video_indices[12:]] == 0).all())
    assert bool((indices[layout.audio_indices[10:]] == 1).all())


def test_h3_ref2va_packing_accepts_text_only_presentation():
    layout = build_ref2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        (),
        num_latent_frames=1,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
    )

    assert layout.sequence_length == 12
    assert layout.num_condition_video_rows == 0
    assert layout.num_condition_audio_rows == 0
    torch.testing.assert_close(layout.audio_indices, torch.arange(4, 8))
    torch.testing.assert_close(layout.video_indices, torch.arange(8, 12))


@pytest.mark.parametrize(
    ("activation_cpu_offloading", "reusable_activation_offload"), [(False, False), (True, False), (True, True)]
)
def test_native_h3_t2va_backend_runs_joint_forward_and_backward(activation_cpu_offloading, reusable_activation_offload):
    if reusable_activation_offload and not torch.cuda.is_available():
        pytest.skip("reusable activation offload requires CUDA")
    device = torch.device("cuda" if activation_cpu_offloading and torch.cuda.is_available() else "cpu")
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=2,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer(config).to(device)
    transformer.enable_gradient_checkpointing(activation_cpu_offloading)
    if reusable_activation_offload:
        transformer.enable_reusable_activation_offload()
    backend = _NativeTrainingBackend(transformer)
    video_latents = torch.randn(1, 4, 2, 4, 4, device=device)
    audio_latents = torch.randn(1, 2, 6, 3, device=device)
    inputs = prepare_joint_noisy_inputs(
        video_latents,
        audio_latents,
        torch.randn_like(video_latents),
        torch.randn_like(audio_latents),
        torch.tensor([0.6]),
    )
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(4, 8, device=device)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long, device=device)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"], device=device)],
    }

    prediction = backend.predict_training(
        transformer,
        batch,
        inputs.video,
        inputs.audio,
        inputs.video_timestep,
        inputs.audio_timestep,
    )
    result = joint_velocity_loss(prediction, inputs)
    result.loss.backward()

    assert prediction.video.shape == video_latents.shape
    assert prediction.audio.shape == audio_latents.shape
    assert torch.isfinite(result.loss)
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None
    assert torch.isfinite(transformer.blocks[0].attn.qkv_proj.weight.grad).all()
    assert transformer.activation_cpu_offloading is activation_cpu_offloading
    if reusable_activation_offload:
        pooled_bytes = transformer.reusable_activation_offloader.pooled_bytes
        assert pooled_bytes > 0
        first_video = prediction.video.detach().clone()
        first_audio = prediction.audio.detach().clone()
        transformer.zero_grad(set_to_none=True)

        repeated = backend.predict_training(
            transformer,
            batch,
            inputs.video,
            inputs.audio,
            inputs.video_timestep,
            inputs.audio_timestep,
        )
        repeated_loss = joint_velocity_loss(repeated, inputs)
        repeated_loss.loss.backward()

        torch.testing.assert_close(repeated.video, first_video, rtol=0, atol=0)
        torch.testing.assert_close(repeated.audio, first_audio, rtol=0, atol=0)
        assert transformer.reusable_activation_offloader.pooled_bytes == pooled_bytes

        pool_entries = len(transformer.reusable_activation_offloader._pool)
        transformer.zero_grad(set_to_none=True)
        wider_video = torch.randn(1, 4, 2, 4, 6, device=device)
        wider_inputs = prepare_joint_noisy_inputs(
            wider_video,
            audio_latents,
            torch.randn_like(wider_video),
            torch.randn_like(audio_latents),
            torch.tensor([0.6]),
        )
        wider_prediction = backend.predict_training(
            transformer,
            batch,
            wider_inputs.video,
            wider_inputs.audio,
            wider_inputs.video_timestep,
            wider_inputs.audio_timestep,
        )
        joint_velocity_loss(wider_prediction, wider_inputs).loss.backward()

        grown_bytes = transformer.reusable_activation_offloader.pooled_bytes
        assert len(transformer.reusable_activation_offloader._pool) == pool_entries
        assert grown_bytes > pooled_bytes

        transformer.zero_grad(set_to_none=True)
        smaller_again = backend.predict_training(
            transformer,
            batch,
            inputs.video,
            inputs.audio,
            inputs.video_timestep,
            inputs.audio_timestep,
        )
        joint_velocity_loss(smaller_again, inputs).loss.backward()
        assert len(transformer.reusable_activation_offloader._pool) == pool_entries
        assert transformer.reusable_activation_offloader.pooled_bytes == grown_bytes


def test_native_h3_image_backend_runs_video_only_forward_and_backward():
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer)
    video = torch.randn(1, 4, 1, 4, 4)
    audio = torch.empty(1, 2, 6, 0)
    inputs = prepare_joint_noisy_inputs(
        video,
        audio,
        torch.randn_like(video),
        torch.empty_like(audio),
        torch.tensor([0.6]),
        video_shift=1.0,
        audio_shift=1.0,
    )
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(4, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
    }

    prediction = backend.predict_training(
        transformer,
        batch,
        inputs.video,
        inputs.audio,
        inputs.video_timestep,
        inputs.audio_timestep,
    )
    result = joint_velocity_loss(prediction, inputs)
    result.loss.backward()

    assert prediction.video.shape == video.shape

    assert prediction.audio.shape == audio.shape
    assert result.audio_elements == 0
    assert torch.isfinite(result.loss)
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None


def test_native_h3_t2va_backend_rejects_visual_conditioning_without_keyframe_latents():
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=8,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor([1, 0])],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
    }

    with pytest.raises(ValueError, match="--task t2va"):
        backend.predict_training(
            transformer,
            batch,
            torch.randn(1, 4, 1, 2, 2),
            torch.randn(1, 2, 6, 1),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )


@pytest.mark.parametrize(("task", "text_tags"), (("i2va", [1, 0, 1]), ("fl2va", [1, 0, 0, 1]), ("l2va", [1, 0, 1])))
@pytest.mark.parametrize("with_audio", [True, False])
def test_native_h3_fl2va_backend_runs_keyframe_conditioned_target_only_backward(task, text_tags, with_audio):
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=2,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer, mode="fl2va")
    video = torch.randn(1, 4, 2, 2, 2)
    audio = torch.randn(1, 2, 6, 1) if with_audio else None
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(len(text_tags), 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor(text_tags)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS[task])],
        # Latent caching stores both anchors so the same cache supports either task.
        H3_KEYFRAME_VIDEO_ROWS_KEY: [torch.randn(2, 16)],
    }

    prediction = backend.predict_training(
        transformer,
        batch,
        video,
        audio,
        torch.tensor([0.4]),
        torch.tensor([0.7]),
        video_row_schedule=torch.linspace(0.2, 0.8, 2),
    )
    loss = prediction.video.square().mean()
    if prediction.audio is not None:
        loss = loss + prediction.audio.square().mean()
    loss.backward()

    assert prediction.video.shape == video.shape
    assert prediction.audio is None if audio is None else prediction.audio.shape == audio.shape
    assert torch.isfinite(loss)
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None


@pytest.mark.parametrize("text_tags", ([1, 0, 1], [1, 0, 0, 1]))
def test_native_h3_fl2va_conditioned_image_runs_single_latent_target(text_tags):
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=8,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer, mode="fl2va")
    video = torch.randn(1, 4, 1, 2, 2)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(len(text_tags), 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor(text_tags)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["fl2va"])],
        H3_KEYFRAME_VIDEO_ROWS_KEY: [torch.randn(2, 16)],
    }

    prediction = backend.predict_training(
        transformer,
        batch,
        video,
        None,
        torch.tensor([0.4]),
        torch.tensor([0.7]),
    )
    loss = prediction.video.square().mean()
    loss.backward()

    assert prediction.video.shape == video.shape
    assert prediction.audio is None
    assert torch.isfinite(loss)


def test_native_h3_l2va_selects_last_cached_keyframe_rows():
    batch = {H3_KEYFRAME_VIDEO_ROWS_KEY: [torch.stack((torch.zeros(16), torch.ones(16)))]}
    backend = _NativeTrainingBackend(SimpleNamespace(config=SimpleNamespace()), mode="fl2va")

    selected = backend._keyframe_cache(
        batch,
        anchors=("last",),
        rows_per_anchor=1,
        row_width=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(selected, torch.ones(1, 16))


def test_native_h3_i2va_backend_requires_recached_keyframe_latents():
    transformer = SimpleNamespace(config=SimpleNamespace(in_channels=4, audio_in_channels=6, text_dim=8, patch_size=(1, 2, 2)))
    backend = _NativeTrainingBackend(transformer, mode="fl2va")
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(3, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor([1, 0, 1])],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["i2va"])],
    }

    with pytest.raises(KeyError, match=H3_KEYFRAME_VIDEO_ROWS_KEY):
        backend.predict_training(
            transformer,
            batch,
            torch.randn(1, 4, 1, 2, 2),
            torch.randn(1, 2, 6, 1),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )


@pytest.mark.parametrize("mode", ["ref2va", "ref2va_omni"])
@pytest.mark.parametrize("with_target_audio", [False, True])
def test_native_h3_ref2va_backend_runs_target_only_forward_and_backward(mode, with_target_audio):
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=2,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer, mode=mode)
    video = torch.randn(1, 4, 1, 2, 2)
    audio = torch.randn(1, 2, 6, 1) if with_target_audio else None
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(3, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor([1, 0, 1])],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS[mode])],
        H3_REFERENCE_KINDS_KEY: [torch.tensor([0, 2])],
        H3_REFERENCE_VIDEO_SHAPES_KEY: [torch.tensor([[1, 2, 2], [0, 0, 0]])],
        H3_REFERENCE_AUDIO_LENGTHS_KEY: [torch.tensor([0, 1])],
        H3_REFERENCE_VIDEO_ROWS_KEY: [torch.randn(1, 16)],
        H3_REFERENCE_AUDIO_ROWS_KEY: [torch.randn(2, 6)],
    }

    prediction = backend.predict_training(
        transformer,
        batch,
        video,
        audio,
        torch.tensor([0.4]),
        torch.tensor([0.7]),
        video_row_schedule=torch.tensor([0.6]),
    )
    loss = prediction.video.square().mean()
    if prediction.audio is not None:
        loss = loss + prediction.audio.square().mean()
    loss.backward()

    assert prediction.video.shape == video.shape
    assert prediction.audio is None if audio is None else prediction.audio.shape == audio.shape
    assert torch.isfinite(loss)
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None


@pytest.mark.parametrize("target", ["video", "audio"])
def test_native_h3_backend_runs_modality_only_forward_and_backward(target):
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=2,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer)
    video = torch.randn(1, 4, 1, 2, 2) if target == "video" else None
    audio = torch.randn(1, 2, 6, 2) if target == "audio" else None
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(3, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(3, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
    }
    if target == "audio":
        batch[H3_VIDEO_GEOMETRY_KEY] = [torch.tensor([2, 2])]

    prediction = backend.predict_training(
        transformer,
        batch,
        video,
        audio,
        torch.tensor([0.4]),
        torch.tensor([0.7]),
    )
    output = prediction.video if target == "video" else prediction.audio
    loss = output.square().mean()
    loss.backward()

    assert prediction.video is None if target == "audio" else prediction.video.shape == video.shape
    assert prediction.audio is None if target == "video" else prediction.audio.shape == audio.shape
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None


def test_native_h3_ref2va_omni_backend_runs_text_only_forward_and_backward():
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=2,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer, mode="ref2va_omni")
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(2, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["ref2va_omni"])],
    }

    prediction = backend.predict_training(
        transformer,
        batch,
        torch.randn(1, 4, 1, 2, 2),
        None,
        torch.tensor([0.5]),
        torch.tensor([0.5]),
    )
    loss = prediction.video.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None


def test_native_h3_strict_ref2va_still_rejects_text_only_conditioning():
    transformer = SimpleNamespace(config=SimpleNamespace(in_channels=4, audio_in_channels=6, text_dim=8))
    backend = _NativeTrainingBackend(transformer, mode="ref2va")
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(2, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["ref2va"])],
    }

    with pytest.raises(ValueError, match="reference presentation"):
        backend.predict_training(
            transformer,
            batch,
            torch.randn(1, 4, 1, 2, 2),
            torch.randn(1, 2, 6, 1),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )


def test_native_h3_ref2va_rejects_mismatched_text_cache_reference_size():
    transformer = SimpleNamespace(config=SimpleNamespace(in_channels=4, audio_in_channels=6, text_dim=8))
    backend = _NativeTrainingBackend(transformer, mode="ref2va", reference_image_short_edge=384)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(2, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["ref2va"])],
        H3_REFERENCE_IMAGE_SHORT_EDGE_KEY: [torch.tensor(2048)],
    }

    with pytest.raises(ValueError, match="text cache uses reference short edge 2048"):
        backend.predict_training(
            transformer,
            batch,
            torch.randn(1, 4, 1, 2, 2),
            None,
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )


def test_native_h3_ref2va_rejects_mismatched_qwen_visual_cap():
    transformer = SimpleNamespace(config=SimpleNamespace(in_channels=4, audio_in_channels=6, text_dim=8))
    backend = _NativeTrainingBackend(transformer, mode="ref2va", text_visual_max_pixels=65_536)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(2, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["ref2va"])],
        H3_TEXT_VISUAL_MAX_PIXELS_KEY: [torch.tensor(131_072)],
    }

    with pytest.raises(ValueError, match="different h3_text_visual_max_pixels"):
        backend.predict_training(
            transformer,
            batch,
            torch.randn(1, 4, 1, 2, 2),
            None,
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )


def test_native_h3_conditioned_image_training_checks_qwen_visual_cap():
    transformer = SimpleNamespace(config=SimpleNamespace(in_channels=4, audio_in_channels=6, text_dim=8))
    backend = _NativeTrainingBackend(transformer, mode="fl2va", text_visual_max_pixels=65_536)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(2, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["fl2va"])],
        H3_TEXT_VISUAL_MAX_PIXELS_KEY: [torch.tensor(131_072)],
    }

    with pytest.raises(ValueError, match="different h3_text_visual_max_pixels"):
        backend.predict_training(
            transformer,
            batch,
            torch.randn(1, 4, 1, 2, 2),
            None,
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )

    batch[H3_TEXT_VISUAL_MAX_PIXELS_KEY] = [torch.tensor(65_536)]
    with pytest.raises(ValueError, match="keyframe vision rows"):
        backend.predict_training(
            transformer,
            batch,
            torch.randn(1, 4, 1, 2, 2),
            None,
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )


def test_native_h3_rejects_mismatched_caption_token_cap():
    transformer = SimpleNamespace(config=SimpleNamespace(in_channels=4, audio_in_channels=6, text_dim=8))
    backend = _NativeTrainingBackend(transformer, mode="fl2va", max_caption_tokens=512)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(2, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
        H3_MAX_CAPTION_TOKENS_KEY: [torch.tensor(256)],
    }

    with pytest.raises(ValueError, match="different h3_max_caption_tokens"):
        backend.predict_training(
            transformer,
            batch,
            torch.randn(1, 4, 1, 2, 2),
            None,
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )


def test_guidance_consistent_prediction_reconstructs_conditional_and_stops_empty_gradient():
    scale = 4.0
    conditional_video = torch.tensor([2.0])
    conditional_audio = torch.tensor([5.0])
    unconditional_video = torch.tensor([-1.0], requires_grad=True)
    unconditional_audio = torch.tensor([1.0], requires_grad=True)
    guided_video = (unconditional_video.detach() + scale * (conditional_video - unconditional_video.detach())).requires_grad_()
    guided_audio = (unconditional_audio.detach() + scale * (conditional_audio - unconditional_audio.detach())).requires_grad_()

    reconstructed = guidance_consistent_prediction(
        H3ModelPrediction(guided_video, guided_audio),
        H3ModelPrediction(unconditional_video, unconditional_audio),
        scale,
    )

    torch.testing.assert_close(reconstructed.video, conditional_video)
    torch.testing.assert_close(reconstructed.audio, conditional_audio)
    (reconstructed.video.sum() + reconstructed.audio.sum()).backward()
    torch.testing.assert_close(guided_video.grad, torch.full_like(guided_video, 1 / scale))
    torch.testing.assert_close(guided_audio.grad, torch.full_like(guided_audio, 1 / scale))
    assert unconditional_video.grad is None
    assert unconditional_audio.grad is None


def test_h3_guidance_schedule_and_modality_specific_targets():
    sigma = torch.tensor([0.0, 0.25, 1.0])
    torch.testing.assert_close(guidance_scale_for_sigma(3.0, sigma, "sigma"), torch.tensor([1.0, 1.5, 3.0]))
    torch.testing.assert_close(guidance_scale_for_sigma(3.0, sigma, "constant"), torch.full_like(sigma, 3.0))
    with pytest.raises(ValueError, match="schedule"):
        guidance_scale_for_sigma(3.0, sigma, "unknown")

    target = H3ModelPrediction(torch.tensor([[2.0], [4.0]]), torch.tensor([[3.0], [5.0]]))
    empty = H3ModelPrediction(torch.tensor([[1.0], [1.0]]), torch.tensor([[2.0], [2.0]]))
    guided = contrastive_guidance_target(
        target,
        empty,
        torch.tensor([1.0, 2.0]),
        audio_guidance_scale=torch.tensor([2.0, 3.0]),
    )
    reconstructed = guidance_consistent_prediction(
        guided,
        empty,
        torch.tensor([1.0, 2.0]),
        audio_guidance_scale=torch.tensor([2.0, 3.0]),
    )
    torch.testing.assert_close(reconstructed.video, target.video)
    torch.testing.assert_close(reconstructed.audio, target.audio)


def test_joint_loss_masks_audio_padding_and_supports_both_balances():
    video = torch.zeros(1, 1, 1, 1, 2)
    audio = torch.zeros(1, 1, 1, 3)
    inputs = prepare_joint_noisy_inputs(video, audio, video.clone(), audio.clone(), torch.tensor([0.5]))
    prediction = H3ModelPrediction(
        video=torch.tensor([[[[[1.0, 3.0]]]]]),
        audio=torch.tensor([[[[2.0, 100.0, 100.0]]]]),
    )
    audio_mask = torch.tensor([[True, False, False]])

    token = joint_velocity_loss(prediction, inputs, audio_mask=audio_mask, balance="token")
    modality = joint_velocity_loss(prediction, inputs, audio_mask=audio_mask, balance="modality")

    assert token.video_elements == 2
    assert token.audio_elements == 1
    torch.testing.assert_close(token.video_loss, torch.tensor(5.0))
    torch.testing.assert_close(token.audio_loss, torch.tensor(4.0))
    torch.testing.assert_close(token.loss, torch.tensor(14.0 / 3.0))
    torch.testing.assert_close(modality.loss, torch.tensor(4.5))

    video_only = joint_velocity_loss(
        prediction,
        inputs,
        audio_mask=torch.zeros_like(audio_mask),
        balance="modality",
    )
    torch.testing.assert_close(video_only.loss, torch.tensor(5.0))
    skipped = joint_velocity_loss(
        prediction,
        inputs,
        video_mask=torch.zeros(1, 1, 1, 2, dtype=torch.bool),
        audio_mask=torch.zeros_like(audio_mask),
    )
    assert skipped.video_elements == skipped.audio_elements == 0
    torch.testing.assert_close(skipped.loss, torch.tensor(0.0))


def test_joint_loss_mask_preserves_reference_gradient():
    prediction = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[True, False, True, False]]]])

    result = joint_prediction_loss(
        H3ModelPrediction(video=prediction, audio=None),
        H3ModelPrediction(video=target, audio=None),
        video_mask=mask,
    )
    result.loss.backward()

    torch.testing.assert_close(result.loss, torch.tensor(5.0))
    torch.testing.assert_close(prediction.grad, torch.tensor([[[[[1.0, 0.0, 3.0, 0.0]]]]]))


def test_joint_loss_applies_modality_specific_sample_weights():
    video = torch.zeros(2, 1, 1, 1, 1)
    audio = torch.zeros(2, 1, 1, 1)
    inputs = prepare_joint_noisy_inputs(video, audio, video.clone(), audio.clone(), torch.full((2,), 0.5))
    prediction = H3ModelPrediction(torch.ones_like(video), torch.ones_like(audio))

    result = joint_velocity_loss(
        prediction,
        inputs,
        video_sample_weight=torch.tensor([1.0, 3.0]),
        audio_sample_weight=torch.tensor([2.0, 4.0]),
        balance="modality",
    )

    # Every squared error is 1 and every item holds one element, so a true
    # weighted mean is sum(w * 1) / sum(w) = 1 for either modality, not the
    # weighted sum over a plain element count.
    torch.testing.assert_close(result.video_loss, torch.tensor(1.0))
    torch.testing.assert_close(result.audio_loss, torch.tensor(1.0))
    torch.testing.assert_close(result.loss, torch.tensor(1.0))


def _weighted_reference(prediction, target, weights, mask=None):
    squared = (prediction - target).float().square()
    numerator = 0.0
    denominator = 0.0
    for index, weight in enumerate(weights.tolist()):
        valid = torch.ones_like(squared[index], dtype=torch.bool) if mask is None else mask[index].expand_as(squared[index])
        numerator += float((squared[index] * valid).sum()) * weight
        denominator += float(valid.sum()) * weight
    return numerator, denominator


def test_joint_loss_sample_weights_form_a_true_weighted_mean():
    torch.manual_seed(0)
    video = torch.randn(3, 1, 1, 2, 2)
    audio = torch.randn(3, 1, 1, 4)
    inputs = prepare_joint_noisy_inputs(video, audio, torch.randn_like(video), torch.randn_like(audio), torch.full((3,), 0.4))
    prediction = H3ModelPrediction(torch.randn_like(video), torch.randn_like(audio))
    video_weights = torch.tensor([0.25, 1.5, 4.0])
    audio_weights = torch.tensor([2.0, 0.5, 1.0])
    audio_mask = torch.tensor([[True, True, False, False], [True, False, False, False], [True, True, True, True]])

    video_numerator, video_denominator = _weighted_reference(prediction.video, inputs.video_target, video_weights)
    audio_numerator, audio_denominator = _weighted_reference(
        prediction.audio, inputs.audio_target, audio_weights, mask=audio_mask.view(3, 1, 1, 4)
    )

    for balance, expected_total in (
        ("modality", 0.5 * (video_numerator / video_denominator + audio_numerator / audio_denominator)),
        ("token", (video_numerator + audio_numerator) / (video_denominator + audio_denominator)),
    ):
        result = joint_velocity_loss(
            prediction,
            inputs,
            audio_mask=audio_mask,
            video_sample_weight=video_weights,
            audio_sample_weight=audio_weights,
            balance=balance,
        )
        torch.testing.assert_close(result.video_loss, torch.tensor(video_numerator / video_denominator))
        torch.testing.assert_close(result.audio_loss, torch.tensor(audio_numerator / audio_denominator))
        torch.testing.assert_close(result.loss, torch.tensor(expected_total))
        # Counts stay counts: the weighted denominator never leaks into them.
        assert result.video_elements == 12
        assert result.audio_elements == 7


def test_joint_loss_matches_validation_weighted_normalization():
    torch.manual_seed(1)
    prediction = torch.randn(2, 1, 1, 3)
    target = torch.randn(2, 1, 1, 3)
    weights = torch.tensor([0.5, 3.0])
    mask = torch.tensor([[True, True, False], [True, False, True]])

    result = joint_prediction_loss(
        H3ModelPrediction(video=None, audio=prediction),
        H3ModelPrediction(video=None, audio=target),
        audio_mask=mask,
        audio_sample_weight=weights,
    )
    error_sum, denominator = masked_squared_error_sum(prediction, target, mask.view(2, 1, 1, 3), sample_weight=weights)

    torch.testing.assert_close(result.audio_loss, error_sum / denominator)
    torch.testing.assert_close(result.loss, error_sum / denominator)


def test_joint_loss_unweighted_normalization_is_unchanged_by_weighting_support():
    torch.manual_seed(2)
    video = torch.randn(2, 1, 1, 2, 2)
    audio = torch.randn(2, 1, 1, 3)
    inputs = prepare_joint_noisy_inputs(video, audio, torch.randn_like(video), torch.randn_like(audio), torch.full((2,), 0.5))
    prediction = H3ModelPrediction(torch.randn_like(video), torch.randn_like(audio))
    audio_mask = torch.tensor([[True, True, False], [True, False, False]])

    video_error = (prediction.video - inputs.video_target).float().square()
    audio_error = ((prediction.audio - inputs.audio_target).float().square() * audio_mask.view(2, 1, 1, 3)).sum()

    result = joint_velocity_loss(prediction, inputs, audio_mask=audio_mask, balance="token")

    torch.testing.assert_close(result.video_loss, video_error.sum() / 8)
    torch.testing.assert_close(result.audio_loss, audio_error / 3)
    torch.testing.assert_close(result.loss, (video_error.sum() + audio_error) / 11)
    # A unit sample weight must reproduce the unweighted path exactly.
    unit = joint_velocity_loss(
        prediction,
        inputs,
        audio_mask=audio_mask,
        video_sample_weight=torch.ones(2),
        audio_sample_weight=torch.ones(2),
        balance="token",
    )
    torch.testing.assert_close(unit.loss, result.loss)


def test_joint_loss_zero_sample_weights_do_not_produce_nan():
    video = torch.ones(2, 1, 1, 1, 1)
    inputs = prepare_joint_noisy_inputs(video, None, torch.zeros_like(video), None, torch.full((2,), 0.5))
    prediction = H3ModelPrediction(torch.zeros_like(video), None)

    result = joint_velocity_loss(prediction, inputs, video_sample_weight=torch.zeros(2))

    assert torch.isfinite(result.loss)
    torch.testing.assert_close(result.loss, torch.tensor(0.0))


def test_h3_text_cache_contract_and_optional_empty_pair(tmp_path):
    path = tmp_path / "sample_mmh3_te.safetensors"
    item = ItemInfo("sample", "caption", (0, 0), (0, 0))
    item.text_encoder_output_cache_path = str(path)
    tensors = {
        f"varlen_{H3_TEXT_HIDDEN_KEY}_float32": torch.zeros(3, 5120),
        f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": torch.tensor([1, 1, 1]),
        H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"]),
        f"varlen_{H3_EMPTY_TEXT_HIDDEN_KEY}_float32": torch.zeros(1, 5120),
        f"varlen_{H3_EMPTY_TEXT_TOKEN_TAGS_KEY}_int64": torch.tensor([1]),
    }

    save_text_encoder_output_cache_minimax_h3(item, tensors)

    with safe_open(path, framework="pt") as handle:
        assert set(handle.keys()) == set(tensors)
        assert handle.metadata()["architecture"] == "minimax_h3"

    tensors.pop(f"varlen_{H3_EMPTY_TEXT_TOKEN_TAGS_KEY}_int64")
    with pytest.raises(ValueError, match="both hidden states and token tags"):
        save_text_encoder_output_cache_minimax_h3(item, tensors)


@pytest.mark.parametrize("load_dino", [False, True])
def test_standard_bucket_manager_loads_h3_joint_cache_without_shared_schema_changes(tmp_path, load_dino):
    latent_path = tmp_path / "sample_mmh3.safetensors"
    text_path = tmp_path / "sample_mmh3_te.safetensors"
    dino_path = tmp_path / "sample_mmh3_dino.safetensors"
    item = ItemInfo("sample", "caption", (64, 64), (64, 64), latent_cache_path=str(latent_path))
    item.text_encoder_output_cache_path = str(text_path)
    from musubi_tuner.minimax_h3.cache import save_latent_cache_minimax_h3

    save_latent_cache_minimax_h3(
        item,
        {
            "latents_2x2x2_float32": torch.zeros(24, 2, 2, 2),
            "latents_audio_2x32x3_float32": torch.zeros(2, 32, 3),
            "audio_loss_mask": torch.tensor([True, True, False]),
            f"varlen_{H3_KEYFRAME_VIDEO_ROWS_KEY}_float32": torch.zeros(2, 96),
            f"varlen_{H3_REFERENCE_KINDS_KEY}_int64": torch.tensor([0, 2]),
            f"varlen_{H3_REFERENCE_VIDEO_SHAPES_KEY}_int64": torch.tensor([[1, 2, 2], [0, 0, 0]]),
            f"varlen_{H3_REFERENCE_AUDIO_LENGTHS_KEY}_int64": torch.tensor([0, 1]),
            f"varlen_{H3_REFERENCE_VIDEO_ROWS_KEY}_float32": torch.zeros(1, 96),
            f"varlen_{H3_REFERENCE_AUDIO_ROWS_KEY}_float32": torch.zeros(2, 32),
        },
    )
    save_text_encoder_output_cache_minimax_h3(
        item,
        {
            f"varlen_{H3_TEXT_HIDDEN_KEY}_float32": torch.zeros(2, 5120),
            f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": torch.ones(2, dtype=torch.long),
            H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"]),
        },
    )
    dino_metadata = {
        "dino_model": "dinov2_vitb14",
        "frames": "2",
        "patches": "4",
        "channels": "384",
        **_latent_cache_identity(latent_path),
    }
    save_file({"h3_dino_features": torch.zeros(2, 4, 384, dtype=torch.float16)}, dino_path, metadata=dino_metadata)

    latent_path_2 = tmp_path / "sample2_mmh3.safetensors"
    text_path_2 = tmp_path / "sample2_mmh3_te.safetensors"
    dino_path_2 = tmp_path / "sample2_mmh3_dino.safetensors"
    shutil.copyfile(latent_path, latent_path_2)
    copied_metadata = {**dino_metadata, **_latent_cache_identity(latent_path_2)}
    save_file({"h3_dino_features": torch.zeros(2, 4, 384, dtype=torch.float16)}, dino_path_2, metadata=copied_metadata)
    item_2 = ItemInfo("sample2", "longer caption", (64, 64), (64, 64), latent_cache_path=str(latent_path_2))
    item_2.text_encoder_output_cache_path = str(text_path_2)
    save_text_encoder_output_cache_minimax_h3(
        item_2,
        {
            f"varlen_{H3_TEXT_HIDDEN_KEY}_float32": torch.zeros(3, 5120),
            f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": torch.ones(3, dtype=torch.long),
            H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"]),
        },
    )

    manager = BucketBatchManager({(64, 64): [item, item_2]}, batch_size=2)
    manager.load_h3_dino_features = load_dino
    batch = manager[0]

    assert batch["latents"].shape == (2, 24, 2, 2, 2)
    assert batch[H3_AUDIO_LATENTS_KEY].shape == (2, 2, 32, 3)
    assert batch[H3_AUDIO_LOSS_MASK_KEY].shape == (2, 3)
    assert isinstance(batch[H3_TEXT_HIDDEN_KEY], list)
    assert batch[H3_TEXT_HIDDEN_KEY][0].shape == (2, 5120)
    assert batch[H3_TEXT_HIDDEN_KEY][1].shape == (3, 5120)
    assert batch[H3_REFERENCE_KINDS_KEY][0].tolist() == [0, 2]
    assert batch[H3_REFERENCE_VIDEO_ROWS_KEY][0].shape == (1, 96)
    assert batch[H3_REFERENCE_AUDIO_ROWS_KEY][0].shape == (2, 32)
    assert batch[H3_KEYFRAME_VIDEO_ROWS_KEY][0].shape == (2, 96)
    if load_dino:
        assert batch["h3_dino_features"].shape == (2, 2, 4, 384)
    else:
        assert "h3_dino_features" not in batch


def test_standard_bucket_manager_requires_dino_cache_only_when_enabled(tmp_path):
    latent_path = tmp_path / "sample_mmh3.safetensors"
    text_path = tmp_path / "sample_mmh3_te.safetensors"
    save_file({"latents_float32": torch.zeros(1)}, latent_path)
    save_file({"text_float32": torch.zeros(1)}, text_path)
    item = ItemInfo("sample", "caption", (64, 64), (64, 64), latent_cache_path=str(latent_path))
    item.text_encoder_output_cache_path = str(text_path)
    manager = BucketBatchManager({(64, 64): [item]}, batch_size=1)
    manager.load_h3_dino_features = True
    with pytest.raises(FileNotFoundError, match="cache_dino_features"):
        manager[0]


def test_standard_bucket_manager_rejects_dino_cache_from_another_model(tmp_path):
    latent_path = tmp_path / "sample_mmh3.safetensors"
    text_path = tmp_path / "sample_mmh3_te.safetensors"
    dino_path = tmp_path / "sample_mmh3_dino.safetensors"
    save_file({"latents_float32": torch.zeros(1)}, latent_path)
    save_file({"text_float32": torch.zeros(1)}, text_path)
    save_file(
        {"h3_dino_features": torch.zeros(2, 4, 384, dtype=torch.float16)},
        dino_path,
        metadata={"dino_model": "dinov2_vits14", "frames": "2", "patches": "4", "channels": "384"},
    )
    item = ItemInfo("sample", "caption", (64, 64), (64, 64), latent_cache_path=str(latent_path))
    item.text_encoder_output_cache_path = str(text_path)
    manager = BucketBatchManager({(64, 64): [item]}, batch_size=1)
    manager.load_h3_dino_features = True
    manager.h3_dino_model = "dinov2_vitb14"

    with pytest.raises(ValueError, match="dinov2_vits14.*dinov2_vitb14"):
        manager[0]


def test_h3_timestep_bucket_pool_round_trips_through_accelerator_state_hooks(tmp_path):
    class Transformer:
        @staticmethod
        def set_gradient_checkpointing_blocks(_value):
            pass

        @staticmethod
        def set_activation_cpu_offload_pin_memory(_value):
            pass

    class Accelerator:
        is_main_process = True

        def __init__(self):
            self.save_hooks = []
            self.load_hooks = []

        def register_save_state_pre_hook(self, hook):
            self.save_hooks.append(hook)

        def register_load_state_pre_hook(self, hook):
            self.load_hooks.append(hook)

    args = SimpleNamespace(
        h3_gradient_checkpointing_blocks=None,
        h3_gradient_checkpointing_cpu_offload_pin_memory=False,
        h3_reusable_activation_offload=False,
        h3_fused_qk_norm_rope=False,
        h3_convrot_int8_lora_fused=False,
    )
    accelerator = Accelerator()
    trainer = MiniMaxH3NetworkTrainer()
    trainer.num_timestep_buckets = 4
    trainer.timestep_range_pool = [(0.0, 0.25), (0.5, 0.75)]
    trainer.on_transformer_loaded(args, accelerator, Transformer())

    accelerator.save_hooks[0]([], [], tmp_path)
    trainer.timestep_range_pool = []
    accelerator.load_hooks[0]([], tmp_path)

    assert trainer.timestep_range_pool == [(0.0, 0.25), (0.5, 0.75)]


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(4, 4, bias=False)
        self.ff = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))
        self.adaln_proj = nn.Linear(4, 4, bias=False)
        self.norm_probe = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states):
        return hidden_states + self.attn(hidden_states) + self.ff(hidden_states)


class MiniMaxH3TokenRefinerBlock(MiniMaxH3TransformerBlock):
    pass


class TinyH3Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([MiniMaxH3TransformerBlock()])
        self.token_refiner = nn.ModuleList([MiniMaxH3TokenRefinerBlock()])

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def test_h3_lora_targets_main_attention_and_ff_only():
    transformer = TinyH3Transformer().requires_grad_(False)
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, [], transformer)
    names = {module.lora_name for module in network.unet_loras}

    assert any(name.endswith("_attn") for name in names)
    assert sum("_ff_" in name for name in names) == 2
    assert not any("adaln" in name or "norm" in name for name in names)
    assert not any("token_refiner" in name for name in names)

    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    transformer(torch.ones(1, 4)).sum().backward()
    adapter_grads = [parameter.grad for parameter in network.parameters()]
    assert any(gradient is not None and torch.isfinite(gradient).all() and bool(gradient.abs().sum()) for gradient in adapter_grads)
    assert all(parameter.grad is None for parameter in transformer.transformer_blocks[0].adaln_proj.parameters())


def test_h3_lora_can_add_token_refiner_targets():
    transformer = TinyH3Transformer().requires_grad_(False)
    network = lora_minimax_h3.create_arch_network(
        1.0,
        2,
        2.0,
        None,
        [],
        transformer,
        h3_lora_token_refiner="true",
    )
    names = {module.lora_name for module in network.unet_loras}

    refiner_names = {name for name in names if "token_refiner" in name}
    assert len(refiner_names) == 3
    assert any(name.endswith("_attn") for name in refiner_names)
    assert sum("_ff_" in name for name in refiner_names) == 2
    assert not any("adaln" in name or "norm" in name for name in names)


def test_h3_token_refiner_lora_round_trips_from_weights(tmp_path):
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=2,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer(config).requires_grad_(False)
    network = lora_minimax_h3.create_arch_network(
        1.0,
        2,
        2.0,
        None,
        [],
        transformer,
        h3_lora_token_refiner="true",
    )
    assert len(network.unet_loras) == config.num_layers * 4 + config.num_refiner_layers * 4
    assert sum("token_refiner" in module.lora_name for module in network.unet_loras) == config.num_refiner_layers * 4
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

    checkpoint = tmp_path / "h3_token_refiner_lora.safetensors"
    network.save_weights(checkpoint, torch.float32, {"ss_h3_lora_token_refiner": "True"})
    weights = load_file(checkpoint)
    restored = lora_minimax_h3.create_arch_network_from_weights(1.0, weights, unet=transformer)

    assert len(restored.unet_loras) == len(network.unet_loras)
    assert {module.lora_name for module in restored.unet_loras} == {module.lora_name for module in network.unet_loras}


def test_h3_token_refiner_flag_is_forwarded_as_network_arg():
    args = create_parser().parse_args(["--sdpa", "--h3_lora_token_refiner"])
    trainer = MiniMaxH3NetworkTrainer()

    trainer.handle_model_specific_args(args)

    assert "h3_lora_token_refiner=true" in args.network_args
    assert trainer.extra_metadata(args)["ss_h3_lora_token_refiner"] == "True"


def test_h3_token_refiner_flag_rejects_other_network_modules():
    args = create_parser().parse_args(["--h3_lora_token_refiner", "--network_module", "networks.lora"])

    with pytest.raises(ValueError, match="networks.lora_minimax_h3"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_native_h3_lora_optimizer_step_and_save_reload_are_equivalent(tmp_path):
    torch.manual_seed(7)
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer(config).requires_grad_(False)
    base_state = {name: value.detach().clone() for name, value in transformer.state_dict().items()}
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, [], transformer)
    assert len(network.unet_loras) == config.num_layers * 4
    assert all(
        module.lora_name.endswith(("_attn_qkv_proj", "_attn_out_proj", "_mlp_fc1", "_mlp_fc2")) for module in network.unet_loras
    )
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

    optimizer_groups, descriptions = network.prepare_optimizer_params(unet_lr=1e-2)
    assert descriptions == ["unet"]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.0)
    backend = _NativeTrainingBackend(transformer)
    video_latents = torch.randn(1, 4, 2, 4, 4)
    audio_latents = torch.randn(1, 2, 6, 3)
    inputs = prepare_joint_noisy_inputs(
        video_latents,
        audio_latents,
        torch.randn_like(video_latents),
        torch.randn_like(audio_latents),
        torch.tensor([0.6]),
    )
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(4, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
    }

    prediction = backend.predict_training(
        transformer,
        batch,
        inputs.video,
        inputs.audio,
        inputs.video_timestep,
        inputs.audio_timestep,
    )
    joint_velocity_loss(prediction, inputs).loss.backward()
    adapter_gradients = [parameter.grad for parameter in network.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in adapter_gradients)
    assert any(bool(gradient.abs().sum()) for gradient in adapter_gradients)
    assert all(parameter.grad is None for parameter in transformer.parameters())

    before_step = {name: parameter.detach().clone() for name, parameter in network.named_parameters()}
    optimizer.step()
    assert any(not torch.equal(before_step[name], parameter) for name, parameter in network.named_parameters())
    for name, value in transformer.state_dict().items():
        torch.testing.assert_close(value, base_state[name])

    with torch.no_grad():
        trained_prediction = backend.predict_training(
            transformer,
            batch,
            inputs.video,
            inputs.audio,
            inputs.video_timestep,
            inputs.audio_timestep,
        )

    checkpoint = tmp_path / "h3_lora.safetensors"
    network.save_weights(checkpoint, torch.float32, {"architecture": "minimax_h3"})
    weights = load_file(checkpoint)
    assert len(weights) == len(network.unet_loras) * 3
    assert not any("adaln" in name or "norm" in name for name in weights)

    restored_transformer = MiniMaxH3Transformer(config).requires_grad_(False)
    restored_transformer.load_state_dict(base_state)
    restored_network = lora_minimax_h3.create_arch_network_from_weights(1.0, weights, unet=restored_transformer)
    restored_network.apply_to(None, restored_transformer, apply_text_encoder=False, apply_unet=True)
    load_info = restored_network.load_weights(checkpoint)
    assert not load_info.missing_keys
    assert not load_info.unexpected_keys
    with torch.no_grad():
        restored_prediction = _NativeTrainingBackend(restored_transformer).predict_training(
            restored_transformer,
            batch,
            inputs.video,
            inputs.audio,
            inputs.video_timestep,
            inputs.audio_timestep,
        )

    torch.testing.assert_close(restored_prediction.video, trained_prediction.video)
    torch.testing.assert_close(restored_prediction.audio, trained_prediction.audio)


class _FakeAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def backward(loss):
        loss.backward()


def test_h3_backward_releases_crepa_capture_after_recomputation():
    trainer = MiniMaxH3NetworkTrainer()
    cleared = []
    trainer._crepa = SimpleNamespace(clear_step=lambda: cleared.append(True))
    value = torch.tensor(2.0, requires_grad=True)

    trainer.backward_loss(_FakeAccelerator(), value.square())

    assert value.grad == 4.0
    assert cleared == [True]


class _ScaleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))


class _SwapAwareScaleTransformer(_ScaleTransformer):
    def __init__(self):
        super().__init__()
        self.swap_mode = "training"
        self.swap_events = []

    def switch_block_swap_for_inference(self):
        self.swap_mode = "inference"
        self.swap_events.append("inference")

    def switch_block_swap_for_training(self):
        self.swap_mode = "training"
        self.swap_events.append("training")


class _ToggleNetwork:
    def __init__(self, transformer):
        self.transformer = transformer
        self.events = []

    def set_enabled(self, enabled):
        self.events.append(enabled)
        self.transformer.adapter_enabled = enabled


class _StochasticPreservationBackend:
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
    ):
        del batch, video_timestep, audio_timestep
        source = video_hidden_states if video_hidden_states is not None else audio_hidden_states
        draw = torch.rand((), device=source.device)
        self.calls.append((conditioning, torch.is_grad_enabled()))
        self.random_draws.append(float(draw))
        self.swap_modes.append(getattr(transformer, "swap_mode", None))
        scale = transformer.scale if getattr(transformer, "adapter_enabled", True) else transformer.scale.detach() * 0 + 1.0
        return H3ModelPrediction(
            video_hidden_states * scale + draw if video_hidden_states is not None else None,
            audio_hidden_states * scale + draw if audio_hidden_states is not None else None,
        )


class _FakeBackend:
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
        return H3ModelPrediction(
            video_hidden_states * transformer.scale if video_hidden_states is not None else None,
            audio_hidden_states * transformer.scale if audio_hidden_states is not None else None,
        )


@pytest.mark.parametrize(
    "guidance_scale,expected_calls",
    [(None, [("prompt", True)]), (4.0, [("empty", False), ("prompt", True)])],
)
def test_h3_trainer_joint_process_batch_routes_optional_guidance(guidance_scale, expected_calls):
    args = create_parser().parse_args([])
    args.h3_guidance_distillation_scale = guidance_scale
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    backend = _FakeBackend()
    trainer.backend = backend
    transformer = _ScaleTransformer()
    video = torch.zeros(1, 24, 2, 2, 2)
    batch = {
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        H3_AUDIO_LOSS_MASK_KEY: torch.zeros(1, 3, dtype=torch.bool),
        "timesteps": [0.5],
    }
    if guidance_scale is not None:
        batch[H3_EMPTY_TEXT_HIDDEN_KEY] = [torch.zeros(1, 5120)]
        batch[H3_EMPTY_TEXT_TOKEN_TAGS_KEY] = [torch.ones(1, dtype=torch.long)]

    torch.manual_seed(0)
    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        None,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    loss.backward()

    assert backend.calls == expected_calls
    assert torch.isfinite(loss)
    assert transformer.scale.grad is not None and torch.isfinite(transformer.scale.grad)
    assert set(metrics) == {"loss/video", "loss/audio", "h3/sigma_video", "h3/sigma_audio"}
    assert metrics["loss/audio"] == 0.0


def test_h3_batch_two_matches_two_independent_items_and_averages_gradients():
    args = create_parser().parse_args([])
    batch = {
        H3_AUDIO_LATENTS_KEY: torch.zeros(2, 2, 32, 3),
        H3_AUDIO_LOSS_MASK_KEY: torch.zeros(2, 3, dtype=torch.bool),
        "timesteps": [0.5, 0.5],
        H3_TEXT_HIDDEN_KEY: [torch.zeros(1, 5120), torch.zeros(3, 5120)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long), torch.ones(3, dtype=torch.long)],
    }
    video = torch.stack((torch.zeros(24, 2, 2, 2), torch.ones(24, 2, 2, 2)))
    noise = torch.stack((torch.ones(24, 2, 2, 2), torch.full((24, 2, 2, 2), 2.0)))

    batched_trainer = MiniMaxH3NetworkTrainer()
    batched_trainer.dit_dtype = torch.float32
    batched_backend = _FakeBackend()
    batched_trainer.backend = batched_backend
    batched_transformer = _ScaleTransformer()
    torch.manual_seed(123)
    batched_loss, batched_metrics = batched_trainer.process_batch(
        args,
        _FakeAccelerator(),
        batched_transformer,
        None,
        batch,
        video,
        noise,
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    batched_trainer.backward_loss(_FakeAccelerator(), batched_loss)

    single_trainer = MiniMaxH3NetworkTrainer()
    single_trainer.dit_dtype = torch.float32
    single_backend = _FakeBackend()
    single_trainer.backend = single_backend
    single_transformer = _ScaleTransformer()
    torch.manual_seed(123)
    single_results = [
        single_trainer.process_batch(
            args,
            _FakeAccelerator(),
            single_transformer,
            None,
            single_trainer._slice_batch_item(batch, index, 2),
            video[index : index + 1],
            noise[index : index + 1],
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )
        for index in range(2)
    ]
    single_loss = torch.stack([loss for loss, _ in single_results]).mean()
    single_loss.backward()

    torch.testing.assert_close(batched_loss, single_loss)
    torch.testing.assert_close(batched_transformer.scale.grad, single_transformer.scale.grad)
    assert batched_backend.calls == [("prompt", True), ("prompt", True)]
    assert single_backend.calls == batched_backend.calls
    for key in batched_metrics:
        expected = sum(metrics.get(key, 0.0) for _, metrics in single_results) / 2
        assert batched_metrics[key] == pytest.approx(expected)


def test_h3_batch_two_wraps_forward_and_backward_in_ddp_no_sync():
    events = []

    class RecordingAccelerator(_FakeAccelerator):
        num_processes = 2

        @staticmethod
        def no_sync(_model):
            class Context:
                def __enter__(self):
                    events.append("enter")

                def __exit__(self, *_args):
                    events.append("exit")

            return Context()

        @staticmethod
        def backward(loss):
            events.append("backward")
            loss.backward()

    class RecordingBackend(_FakeBackend):
        def predict_training(self, *args, **kwargs):
            events.append("forward")
            return super().predict_training(*args, **kwargs)

    args = create_parser().parse_args([])
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = RecordingBackend()
    transformer = _ScaleTransformer()
    video = torch.zeros(2, 24, 2, 2, 2)
    batch = {
        H3_AUDIO_LATENTS_KEY: torch.zeros(2, 2, 32, 3),
        H3_AUDIO_LOSS_MASK_KEY: torch.zeros(2, 3, dtype=torch.bool),
        "timesteps": [0.5, 0.5],
    }

    trainer.process_batch(
        args,
        RecordingAccelerator(),
        transformer,
        None,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert events == ["enter", "forward", "backward", "exit", "forward", "backward"]


def test_h3_batch_metrics_average_only_items_that_report_each_metric():
    averaged = MiniMaxH3NetworkTrainer._average_batch_metrics([{"loss/total": 2.0, "crepa/alignment": 0.8}, {"loss/total": 4.0}])
    assert averaged == {"loss/total": 3.0, "crepa/alignment": 0.8}


def test_h3_validation_batch_size_fails_during_startup(tmp_path):
    config = tmp_path / "validation.toml"
    config.write_text('[general]\nbatch_size = 2\n[[datasets]]\nvideo_directory = "unused"\n', encoding="utf-8")
    args = create_parser().parse_args(["--validation_dataset_config", str(config)])
    with pytest.raises(ValueError, match="validation requires batch_size = 1"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def _caption_dropout_batch():
    return {
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        H3_AUDIO_LOSS_MASK_KEY: torch.zeros(1, 3, dtype=torch.bool),
        "timesteps": [0.5],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 5120)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
    }


def _run_caption_dropout(rate, *, guidance_scale=None, network=None, backend=None):
    args = create_parser().parse_args([])
    args.h3_caption_dropout_rate = rate
    args.h3_guidance_distillation_scale = guidance_scale
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = backend or _FakeBackend()
    transformer = _ScaleTransformer()
    video = torch.zeros(1, 24, 2, 2, 2)
    torch.manual_seed(0)
    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        network,
        _caption_dropout_batch(),
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    return trainer.backend, loss, metrics


@pytest.mark.parametrize(("rate", "expected"), [(0.0, "prompt"), (1.0, "empty")])
def test_h3_caption_dropout_selects_the_conditioning_branch(rate, expected):
    # Rate 0 must leave the prompt branch untouched; rate 1 must train the
    # unconditional branch, which is what gives inference a null prompt.
    backend, _, metrics = _run_caption_dropout(rate)

    assert backend.calls == [(expected, True)]
    if rate > 0:
        assert metrics["h3/caption_dropped"] == float(expected == "empty")
    else:
        # A run that does not use dropout keeps exactly its previous metric set.
        assert "h3/caption_dropped" not in metrics


def test_h3_caption_dropout_skips_the_guidance_correction():
    # A dropped step is already unconditional, so there is no guided field to
    # invert and the empty branch would duplicate the trainable one.
    backend, _, _ = _run_caption_dropout(1.0, guidance_scale=4.0)

    assert backend.calls == [("empty", True)]


def test_h3_caption_dropout_keeps_guidance_when_the_prompt_survives():
    backend, _, _ = _run_caption_dropout(0.0, guidance_scale=4.0)

    assert backend.calls == [("empty", False), ("prompt", True)]


def test_h3_caption_dropout_requires_the_empty_cache():
    args = create_parser().parse_args([])
    args.h3_caption_dropout_rate = 0.5
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.backend = _FakeBackend()
    video = torch.zeros(1, 24, 2, 2, 2)

    with pytest.raises(KeyError, match="cache_guidance_empty"):
        trainer.process_batch(
            args,
            _FakeAccelerator(),
            _ScaleTransformer(),
            None,
            {"timesteps": [0.5]},
            video,
            torch.ones_like(video),
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_h3_caption_dropout_rate_is_validated(rate):
    args = create_parser().parse_args([])
    args.h3_caption_dropout_rate = rate
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match=r"h3_caption_dropout_rate"):
        trainer.handle_model_specific_args(args)


def test_h3_trainer_base_preservation_replays_rng_and_restores_network():
    args = create_parser().parse_args([])
    args.h3_base_preservation_loss_weight = 0.1
    args.h3_guidance_distillation_scale = 3.0
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    backend = _StochasticPreservationBackend()
    trainer.backend = backend
    transformer = _ScaleTransformer()
    network = _ToggleNetwork(transformer)
    video = torch.zeros(1, 24, 2, 2, 2)

    torch.manual_seed(0)
    batch = {
        "timesteps": [0.5],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 5120)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
    }
    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        network,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    loss.backward()

    assert backend.calls == [("empty", False), ("prompt", False), ("prompt", True)]
    assert backend.random_draws[0] == backend.random_draws[1] == backend.random_draws[2]
    assert network.events == [False, True]
    assert transformer.adapter_enabled is True
    assert metrics["loss/base_preservation"] > 0
    assert transformer.scale.grad is not None and torch.isfinite(transformer.scale.grad)


def test_h3_auxiliary_forwards_use_forward_only_block_swap_then_restore_training():
    args = create_parser().parse_args([])
    args.h3_base_preservation_loss_weight = 0.1
    args.h3_guidance_distillation_scale = 3.0
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.blocks_to_swap = 2
    backend = _StochasticPreservationBackend()
    trainer.backend = backend
    transformer = _SwapAwareScaleTransformer()
    network = _ToggleNetwork(transformer)
    video = torch.zeros(1, 24, 2, 2, 2)
    batch = {
        "timesteps": [0.5],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 5120)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
    }

    loss, _ = trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        network,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    loss.backward()

    assert transformer.swap_events == ["inference", "training"]
    assert backend.swap_modes == ["inference", "inference", "training"]


def test_h3_h2d_only_auxiliary_forwards_avoid_per_step_mode_switches():
    args = create_parser().parse_args([])
    args.h3_guidance_distillation_scale = 3.0
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.blocks_to_swap = 2
    trainer._block_swap_h2d_only = True
    backend = _StochasticPreservationBackend()
    trainer.backend = backend
    transformer = _SwapAwareScaleTransformer()
    video = torch.zeros(1, 24, 2, 2, 2)
    batch = {
        "timesteps": [0.5],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 5120)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
    }

    trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        None,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert transformer.swap_events == []
    assert backend.swap_modes == ["training", "training"]


def test_h3_auxiliary_forward_exception_restores_training_block_swap():
    class FailingBackend(_StochasticPreservationBackend):
        def predict_training(self, *args, **kwargs):
            raise RuntimeError("teacher failed")

    args = create_parser().parse_args([])
    args.h3_guidance_distillation_scale = 3.0
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer.blocks_to_swap = 2
    trainer.backend = FailingBackend()
    transformer = _SwapAwareScaleTransformer()
    video = torch.zeros(1, 24, 2, 2, 2)
    batch = {
        "timesteps": [0.5],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 5120)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
    }

    with pytest.raises(RuntimeError, match="teacher failed"):
        trainer.process_batch(
            args,
            _FakeAccelerator(),
            transformer,
            None,
            batch,
            video,
            torch.ones_like(video),
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )

    assert transformer.swap_mode == "training"
    assert transformer.swap_events == ["inference", "training"]


def test_h3_base_preservation_uses_the_primary_generated_region_mask(monkeypatch):
    args = create_parser().parse_args([])
    args.h3_base_preservation_loss_weight = 0.1
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer._mask_mode = "box"
    generated = torch.zeros(2, 2, 2, dtype=torch.bool)
    generated[1] = True
    trainer._draw_step_mask = lambda inputs, patch_size: SimpleNamespace(
        video_rows=torch.zeros(2, dtype=torch.bool),
        audio_rows=None,
        video_latent=generated,
        audio_latent=None,
    )
    transformer = _ScaleTransformer()
    network = _ToggleNetwork(transformer)

    def predict(_accelerator, _transformer, _batch, inputs, **kwargs):
        del kwargs
        scale = transformer.scale if getattr(transformer, "adapter_enabled", True) else transformer.scale.detach() * 0 + 1.0
        return H3ModelPrediction(inputs.video * scale, None)

    trainer._predict = predict
    captured = {}
    original = h3_train_network.joint_prediction_loss

    def capture(*args, **kwargs):
        captured["video_mask"] = kwargs["video_mask"].clone()
        return original(*args, **kwargs)

    monkeypatch.setattr(h3_train_network, "joint_prediction_loss", capture)
    video = torch.zeros(1, 24, 2, 2, 2)
    trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        network,
        {"timesteps": [0.5], "video_loss_mask": torch.ones(1, 2, 2, 2, dtype=torch.bool)},
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert not captured["video_mask"][:, :, 0].any()
    assert captured["video_mask"][:, :, 1].all()


def test_h3_sparse_base_preservation_skips_teacher_forward():
    args = create_parser().parse_args([])
    args.h3_base_preservation_loss_weight = 0.1
    args.h3_base_preservation_probability = 0.25
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    trainer._base_preservation_active = lambda accelerator, probability: False
    backend = _StochasticPreservationBackend()
    trainer.backend = backend
    transformer = _ScaleTransformer()
    network = _ToggleNetwork(transformer)
    video = torch.zeros(1, 24, 2, 2, 2)

    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        network,
        {"timesteps": [0.5]},
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    loss.backward()

    assert backend.calls == [("prompt", True)]
    assert network.events == []
    assert metrics["h3/base_preservation_active"] == 0.0
    assert metrics["loss/base_preservation"] == 0.0


def test_h3_sparse_base_preservation_uses_inverse_probability_weight():
    def run(probability):
        args = create_parser().parse_args([])
        args.h3_base_preservation_loss_weight = 0.1
        args.h3_base_preservation_probability = probability
        trainer = MiniMaxH3NetworkTrainer()
        trainer.dit_dtype = torch.float32
        trainer._base_preservation_active = lambda accelerator, probability: True
        backend = _StochasticPreservationBackend()
        trainer.backend = backend
        transformer = _ScaleTransformer()
        network = _ToggleNetwork(transformer)
        video = torch.zeros(1, 24, 2, 2, 2)
        torch.manual_seed(123)
        _, metrics = trainer.process_batch(
            args,
            _FakeAccelerator(),
            transformer,
            network,
            {"timesteps": [0.5]},
            video,
            torch.ones_like(video),
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )
        assert backend.calls == [("prompt", False), ("prompt", True)]
        return metrics

    dense = run(1.0)
    sparse = run(0.25)

    assert sparse["h3/base_preservation_active"] == 1.0
    assert sparse["loss/base_preservation"] == pytest.approx(4 * dense["loss/base_preservation"])


@pytest.mark.parametrize("probability", [0.0, -0.1, 1.1, float("nan")])
def test_h3_base_preservation_probability_is_validated(probability):
    args = create_parser().parse_args([])
    args.h3_base_preservation_probability = probability

    with pytest.raises(ValueError, match="h3_base_preservation_probability"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_base_preservation_rejects_networks_without_runtime_disable_support():
    args = create_parser().parse_args([])
    args.h3_base_preservation_loss_weight = 0.1

    with pytest.raises(TypeError, match="set_enabled"):
        MiniMaxH3NetworkTrainer().extra_trainable_params(
            args,
            _FakeAccelerator(),
            object(),
            None,
            [],
        )


def test_h3_base_preservation_draw_advances_cpu_rng_and_broadcasts(monkeypatch):
    broadcasts = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda value, src: broadcasts.append((value.clone(), src)))

    torch.manual_seed(321)
    torch.rand(())
    expected_next = torch.rand(())
    torch.manual_seed(321)
    MiniMaxH3NetworkTrainer._base_preservation_active(_FakeAccelerator(), 0.5)
    actual_next = torch.rand(())

    assert actual_next == expected_next
    assert len(broadcasts) == 1 and broadcasts[0][1] == 0


def test_h3_ref2va_rejects_training_time_sample_prompts():
    args = create_parser().parse_args(["--sdpa"])
    args.h3_training_mode = "ref2va"
    args.sample_prompts = "samples.txt"

    with pytest.raises(ValueError, match="minimax_h3_generate_video.py"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


@pytest.mark.parametrize("schedule", ["constant", "sigma"])
def test_h3_contrastive_guidance_form_is_scale_squared_larger(schedule):
    def run(form):
        args = create_parser().parse_args([])
        args.h3_guidance_distillation_scale = 3.0
        args.h3_guidance_loss_form = form
        args.h3_guidance_loss_schedule = schedule
        trainer = MiniMaxH3NetworkTrainer()
        trainer.dit_dtype = torch.float32
        trainer.backend = _FakeBackend()
        transformer = _ScaleTransformer()
        video = torch.zeros(1, 24, 2, 2, 2)
        batch = {
            "timesteps": [0.5],
            H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 5120)],
            H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
        }
        torch.manual_seed(0)
        loss, metrics = trainer.process_batch(
            args,
            _FakeAccelerator(),
            transformer,
            None,
            batch,
            video,
            torch.ones_like(video),
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )
        return loss, metrics

    normalized_loss, normalized_metrics = run("normalized")
    contrastive_loss, contrastive_metrics = run("contrastive")

    effective_scale = 3.0
    if schedule == "sigma":
        effective_scale = 1.0 + 2.0 * normalized_metrics["h3/sigma_video"]
    torch.testing.assert_close(contrastive_loss, normalized_loss * effective_scale**2)
    assert contrastive_metrics["loss/video"] == pytest.approx(normalized_metrics["loss/video"] * effective_scale**2)


def test_h3_trainer_image_process_batch_uses_resolution_schedule_without_audio():
    args = create_parser().parse_args([])
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    backend = _FakeBackend()
    trainer.backend = backend
    transformer = _ScaleTransformer()
    video = torch.zeros(1, 24, 1, 4, 4)
    batch = {"timesteps": [0.5]}

    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        None,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    loss.backward()

    expected_shift = torch.exp(torch.tensor(0.5 + (4 - 256) * (1.15 - 0.5) / (6400 - 256)))
    expected_sigma = expected_shift / (1.0 + expected_shift)
    assert metrics["h3/sigma_video"] == pytest.approx(float(expected_sigma))
    assert metrics["h3/sigma_audio"] == pytest.approx(float(expected_sigma))
    assert metrics["loss/audio"] == 0.0
    assert backend.calls == [("prompt", True)]
    assert transformer.scale.grad is not None


def test_h3_trainer_build_dataset_routes_through_h3_adapter(monkeypatch):
    group = SimpleNamespace(num_train_items=1)
    captured = {}

    monkeypatch.setattr(h3_train_network.config_utils, "load_user_config", lambda path: {"source": path})

    def create_group(user_config, args, **kwargs):
        captured.update(user_config=user_config, args=args, kwargs=kwargs)
        return group, object()

    monkeypatch.setattr(h3_train_network, "create_h3_dataset_group", create_group)
    args = SimpleNamespace(
        dataset_config="dataset.toml",
        num_timestep_buckets=4,
        max_data_loader_n_workers=0,
    )

    trainer = MiniMaxH3NetworkTrainer()
    built_group, collator, current_epoch = trainer._build_dataset(args)

    assert built_group is group
    assert collator.dataset is group
    assert current_epoch.value == 0
    assert captured == {
        "user_config": {"source": "dataset.toml"},
        "args": args,
        "kwargs": {"training": True, "num_timestep_buckets": 4, "shared_epoch": current_epoch},
    }


class _TrainingBackend:
    def __init__(self, transformer):
        self.transformer = transformer

    def get_training_transformer(self):
        return self.transformer


def test_h3_trainer_loads_only_the_selected_training_transformer(monkeypatch, tmp_path):
    transformer = nn.Linear(2, 2)
    backend = _TrainingBackend(transformer)
    captured = {}

    def create_backend(**kwargs):
        captured.update(kwargs)
        return backend

    monkeypatch.setattr(h3_train_network, "create_training_backend", create_backend)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.bfloat16
    args = SimpleNamespace(
        h3_training_mode="ref2va",
        fp8_base=False,
        int8_convrot_base=False,
        h3_adaln_rank=None,
        h3_fp8_quantization_mode="block",
        h3_convrot_int8=False,
        h3_convrot_int8_bwd="bf16",
        h3_convrot_int8_fwd="int8",
        h3_attn_auto_dispatch=False,
    )
    accelerator = SimpleNamespace(device=torch.device("cuda", 0))

    loaded = trainer.load_transformer(accelerator, args, str(tmp_path), "torch", False, "cpu", torch.bfloat16)

    assert loaded is transformer
    assert captured == {
        "model": tmp_path,
        "device": "cpu",
        "dtype": "bfloat16",
        "mode": "ref2va",
        "attention_mode": "torch",
        "split_attention": False,
        "fp8_scaled": False,
        "quantization_device": "cuda:0",
        "int8_convrot": False,
        "adaln_rank": None,
        "fp8_quantization_mode": "block",
        "convrot_int8": False,
        "convrot_int8_bwd": "bf16",
        "convrot_int8_fwd": "int8",
        "target_device": "cuda:0",
        "blocks_to_swap": 0,
        "block_swap_h2d_only": False,
    }


def test_h3_trainer_routes_base_loras_through_model_loading(monkeypatch, tmp_path):
    transformer = nn.Linear(2, 2)
    captured = {}
    messages = []
    lora_state = {"lora_unet_blocks_0_mlp_fc1.lora_down.weight": torch.ones(1, 2)}

    def create_backend(**kwargs):
        captured.update(kwargs)
        return _TrainingBackend(transformer)

    monkeypatch.setattr(h3_train_network, "create_training_backend", create_backend)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.bfloat16
    monkeypatch.setattr(trainer, "load_network_weights", lambda _path, _module: lora_state)
    args = SimpleNamespace(
        h3_training_mode="fl2va",
        fp8_base=False,
        int8_convrot_base=True,
        h3_adaln_rank=None,
        h3_fp8_quantization_mode="block",
        h3_convrot_int8=False,
        h3_convrot_int8_bwd="bf16",
        h3_convrot_int8_fwd="int8",
        h3_attn_auto_dispatch=False,
        base_weights=["base.safetensors"],
        base_weights_multiplier=[0.75],
    )
    accelerator = SimpleNamespace(device=torch.device("cuda", 0), print=messages.append)

    loaded = trainer.load_transformer(accelerator, args, str(tmp_path), "torch", False, "cpu", torch.bfloat16)

    assert loaded is transformer
    assert captured["base_lora_weights"] == [lora_state]
    assert captured["base_lora_multipliers"] == [0.75]
    assert args.base_weights is None
    assert args.base_weights_multiplier is None
    assert messages == ["all H3 base weights merged during model loading: base.safetensors"]


@pytest.mark.parametrize(("option", "value", "message"), [("--sdpa", "--split_attn", "split attention")])
def test_h3_trainer_rejects_release_dependent_common_loading_modes(option, value, message):
    argv = [option] if value is None else [option, value]
    args = create_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_prequantized_convrot_base_accepts_bf16_forward_control():
    args = create_parser().parse_args(["--sdpa", "--int8_convrot_base", "--h3_convrot_int8_fwd", "bf16"])

    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)

    assert args.int8_convrot_base is True
    assert args.h3_convrot_int8_fwd == "bf16"


def test_h3_trainer_accepts_compile_and_exposes_fallback_controls():
    args = create_parser().parse_args(["--sdpa", "--compile", "--compile_fallback_to_eager", "--compile_auto_cache_size_limit"])

    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)

    assert args.compile
    assert args.compile_fallback_to_eager
    assert args.compile_auto_cache_size_limit


def test_h3_regional_compile_does_not_self_reference_transformer_root(monkeypatch):
    class Accelerator:
        def prepare(self, *objects, **_kwargs):
            return objects[0] if len(objects) == 1 else objects

        @staticmethod
        def unwrap_model(model):
            return model

    class Network(nn.Module):
        def prepare_grad_etc(self, _transformer):
            pass

    trainer = MiniMaxH3NetworkTrainer()
    monkeypatch.setattr(trainer, "compile_transformer", lambda _args, transformer: transformer)
    args = SimpleNamespace(compile=True, gradient_checkpointing=False)
    transformer = nn.Linear(2, 2)
    network = Network()

    prepared = trainer._prepare_with_accelerator(
        args,
        Accelerator(),
        transformer,
        network,
        object(),
        object(),
        object(),
        torch.float32,
        torch.float32,
        torch.float32,
    )
    prepared_transformer = prepared[0]

    assert prepared_transformer is transformer
    assert prepared_transformer.__dict__.get("_orig_mod") is not prepared_transformer
    assert prepared_transformer._modules.get("_orig_mod") is not prepared_transformer
    assert set(prepared_transformer.state_dict()) == {"weight", "bias"}


def test_h3_partial_checkpointing_requires_gradient_checkpointing():
    args = create_parser().parse_args(["--sdpa", "--h3_gradient_checkpointing_blocks", "25"])

    with pytest.raises(ValueError, match="requires --gradient_checkpointing"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_partial_checkpointing_allows_compile():
    # Each block is compiled independently and the checkpoint wrapper sits outside
    # the compiled callable, so a partially wrapped stack only adds a second
    # compiled variant of the shared block forward.
    args = create_parser().parse_args(
        ["--sdpa", "--gradient_checkpointing", "--h3_gradient_checkpointing_blocks", "25", "--compile"]
    )

    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)

    assert args.h3_gradient_checkpointing_blocks == 25


def test_h3_partial_checkpointing_rejects_block_swap():
    args = create_parser().parse_args(
        ["--sdpa", "--gradient_checkpointing", "--h3_gradient_checkpointing_blocks", "25", "--blocks_to_swap", "10"]
    )

    with pytest.raises(ValueError, match="cannot be combined with block swap"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_pinned_activation_offload_requires_cpu_checkpoint_offload():
    valid = create_parser().parse_args(
        [
            "--sdpa",
            "--gradient_checkpointing",
            "--gradient_checkpointing_cpu_offload",
            "--h3_gradient_checkpointing_cpu_offload_pin_memory",
        ]
    )
    MiniMaxH3NetworkTrainer().handle_model_specific_args(valid)

    invalid = create_parser().parse_args(["--sdpa", "--h3_gradient_checkpointing_cpu_offload_pin_memory"])
    with pytest.raises(ValueError, match="requires --gradient_checkpointing"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(invalid)


def test_h3_reusable_activation_offload_requires_cpu_checkpoint_offload():
    valid = create_parser().parse_args(
        ["--sdpa", "--gradient_checkpointing", "--gradient_checkpointing_cpu_offload", "--h3_reusable_activation_offload"]
    )
    MiniMaxH3NetworkTrainer().handle_model_specific_args(valid)

    invalid = create_parser().parse_args(["--sdpa", "--h3_reusable_activation_offload"])
    with pytest.raises(ValueError, match="requires --gradient_checkpointing"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(invalid)


@pytest.mark.parametrize("option", ["--flash_attn", "--flash3"])
def test_h3_trainer_accepts_flash_attention(option):
    args = create_parser().parse_args([option])

    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)

    assert args.flash_attn or args.flash3


@pytest.mark.parametrize("attention_mode", ["flash", "flash3"])
def test_h3_flash_attention_uses_common_maskless_backend(monkeypatch, attention_mode):
    captured = {}

    def flash_attention(qkv, *, attn_params, drop_rate=0.0):
        query, key, value = qkv
        captured.update(
            query_shape=query.shape,
            key_shape=key.shape,
            value_shape=value.shape,
            mode=attn_params.attn_mode,
            split=attn_params.split_attn,
            drop_rate=drop_rate,
        )
        return query.flatten(2, 3)

    monkeypatch.setattr(h3_model, "musubi_attention", flash_attention)
    module = MiniMaxH3Attention(hidden_size=16, heads=2, head_dim=8, qk_norm_eps=1e-5, attention_mode=attention_mode)
    hidden_states = torch.randn(1, 5, 16)

    output = module(hidden_states)

    assert output.shape == hidden_states.shape
    assert captured == {
        "query_shape": torch.Size([1, 5, 2, 8]),
        "key_shape": torch.Size([1, 5, 2, 8]),
        "value_shape": torch.Size([1, 5, 2, 8]),
        "mode": attention_mode,
        "split": False,
        "drop_rate": 0.0,
    }


@pytest.mark.parametrize("attention_mode", ["flash", "flash3"])
def test_h3_flash_attention_preserves_sdpa_padding_mask_fallback(monkeypatch, attention_mode):
    def forbidden_flash(*_args, **_kwargs):
        raise AssertionError("pairwise padding masks must not enter FlashAttention")

    monkeypatch.setattr(h3_model, "musubi_attention", forbidden_flash)
    module = MiniMaxH3Attention(hidden_size=16, heads=2, head_dim=8, qk_norm_eps=1e-5, attention_mode=attention_mode)
    hidden_states = torch.randn(1, 4, 16)
    attention_mask = torch.eye(4, dtype=torch.bool)

    output = module(hidden_states, attention_mask=attention_mask)

    assert output.shape == hidden_states.shape


def test_h3_trainer_maps_common_fp8_switch_to_scaled_loading_and_accepts_swap():
    args = create_parser().parse_args(
        [
            "--sdpa",
            "--fp8_base",
            "--blocks_to_swap",
            "2",
            "--block_swap_h2d_only",
            "--block_swap_ring_size",
            "1",
            "--block_swap_granularity",
            "layer",
            "--use_pinned_memory_for_block_swap",
            "--gradient_checkpointing",
        ]
    )

    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)

    assert args.fp8_base is True
    assert args.fp8_scaled is True
    assert args.blocks_to_swap == 2
    assert args.block_swap_h2d_only is True
    assert args.block_swap_ring_size == 1
    assert args.block_swap_granularity == "layer"
    assert args.use_pinned_memory_for_block_swap is True
    assert args.h3_attn_auto_dispatch is False


@pytest.mark.parametrize("convrot_flag", ["--h3_convrot_int8", "--int8_convrot_base"])
def test_h3_trainer_rejects_layer_granular_swap_with_convrot_int8(convrot_flag):
    layer_args = [
        "--sdpa",
        convrot_flag,
        "--blocks_to_swap",
        "2",
        "--block_swap_h2d_only",
        "--block_swap_granularity",
        "layer",
    ]

    with pytest.raises(ValueError, match="bypasses the ConvRot INT8 forward"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(create_parser().parse_args(layer_args))

    block_args = create_parser().parse_args(layer_args[:-1] + ["block"])
    MiniMaxH3NetworkTrainer().handle_model_specific_args(block_args)

    assert block_args.block_swap_granularity == "block"


def test_h3_attention_auto_dispatch_requires_sdpa():
    args = create_parser().parse_args(["--flash_attn", "--h3_attn_auto_dispatch"])

    with pytest.raises(ValueError, match="requires --sdpa"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_trainer_warns_when_h2d_swap_uses_unpinned_host_memory(caplog):
    args = create_parser().parse_args(["--sdpa", "--blocks_to_swap", "2", "--block_swap_h2d_only"])

    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)

    assert "can be substantially slower" in caplog.text


def test_h3_trainer_rejects_composed_discrete_flow_shift():
    # The common sampler must hand H3 an *unshifted* coordinate; composing
    # --discrete_flow_shift on top would shift video twice and desynchronize audio.
    args = create_parser().parse_args(["--sdpa", "--discrete_flow_shift", "12.0"])

    with pytest.raises(ValueError, match="h3_shift_video"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


@pytest.mark.parametrize(
    "extra_args, message",
    [
        (["--h3_video_loss_weight", "-0.1"], "h3_video_loss_weight"),
        (["--h3_audio_loss_weight", "nan"], "h3_audio_loss_weight"),
        (
            ["--h3_video_loss_weight", "0", "--h3_audio_loss_weight", "0"],
            "at least one",
        ),
    ],
)
def test_h3_trainer_rejects_invalid_modality_loss_weights(extra_args, message):
    args = create_parser().parse_args(["--sdpa", *extra_args])

    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


@pytest.mark.parametrize(
    "extra_args, message",
    [
        (["--h3_observed_modality", "video", "--h3_audio_loss_weight", "0"], "trains audio"),
        (["--h3_observed_modality", "audio", "--h3_video_loss_weight", "0"], "trains video"),
    ],
)
def test_h3_observed_modality_requires_weight_on_generated_side(extra_args, message):
    args = create_parser().parse_args(["--sdpa", *extra_args])

    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_dataset_loss_coverage_rejects_zero_weight_target_modes():
    config = {
        "general": {"h3_target_mode": "video"},
        "datasets": [
            {"video_directory": "/video"},
            {"audio_directory": "/audio", "h3_target_mode": "audio"},
        ],
    }

    with pytest.raises(ValueError, match="dataset 1.*video"):
        h3_train_network._validate_dataset_loss_coverage(config, video_weight=0.0, audio_weight=1.0)
    with pytest.raises(ValueError, match="dataset 2.*audio"):
        h3_train_network._validate_dataset_loss_coverage(config, video_weight=1.0, audio_weight=0.0)


def test_h3_dataset_loss_coverage_accepts_active_and_image_targets():
    config = {
        "general": {"h3_target_mode": "audio"},
        "datasets": [
            {"video_directory": "/av", "h3_target_mode": "av"},
            {"image_directory": "/images"},
        ],
    }

    h3_train_network._validate_dataset_loss_coverage(config, video_weight=1.0, audio_weight=0.0)


@pytest.mark.parametrize("sampling", ["flux_shift", "qwen_shift", "krea2_shift", "ideogram4_shift", "qinglong_flux"])
def test_h3_trainer_rejects_pre_shifted_timestep_sampling(sampling):
    args = create_parser().parse_args(["--sdpa", "--timestep_sampling", sampling])

    with pytest.raises(ValueError, match="resolution-dependent shift"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_timestep_focus_is_exact_uniform_mixture():
    base = (torch.arange(10_000, dtype=torch.float64) + 0.5) / 10_000
    focused = h3_train_network._apply_timestep_focus(base, 0.4, 0.8, 0.25)

    assert focused.min() >= 0.0
    assert focused.max() < 1.0
    assert ((focused >= 0.4) & (focused < 0.8)).double().mean().item() == pytest.approx(0.55, abs=1e-4)


def test_h3_timestep_focus_is_opt_in_and_requires_uniform_full_range():
    parser = create_parser()
    default = parser.parse_args(["--sdpa"])
    assert default.h3_timestep_focus_probability == 0.0

    incompatible = parser.parse_args(["--sdpa", "--timestep_sampling", "logsnr", "--h3_timestep_focus_probability", "0.25"])
    with pytest.raises(ValueError, match="requires --timestep_sampling uniform"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(incompatible)

    clipped = parser.parse_args(["--sdpa", "--h3_timestep_focus_probability", "0.25", "--min_timestep", "100"])
    with pytest.raises(ValueError, match="cannot be combined"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(clipped)


def test_h3_rejects_timestep_buckets_when_sampler_ignores_them():
    args = create_parser().parse_args(["--sdpa", "--timestep_sampling", "sigma", "--num_timestep_buckets", "8"])

    with pytest.raises(ValueError, match="not consumed"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--h3_swiglu_chunk_rows", "-1"], "must be non-negative"),
        (["--h3_swiglu_chunk_rows", "2048", "--compile"], "not supported with --compile"),
    ],
)
def test_h3_rejects_invalid_swiglu_chunking(extra, message):
    args = create_parser().parse_args(["--sdpa", *extra])

    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_common_sampler_can_skip_discarded_noisy_tensor_without_changing_timesteps():
    trainer = MiniMaxH3NetworkTrainer()
    args = create_parser().parse_args(["--sdpa"])
    scheduler = FlowMatchDiscreteScheduler(shift=1.0, reverse=True, solver="euler")
    latents = torch.zeros(1, 2, 1, 2, 2)
    noise = torch.ones_like(latents)

    noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(
        args, noise, latents, [0.375], scheduler, torch.device("cpu"), torch.float32
    )
    skipped, skipped_timesteps = trainer.get_noisy_model_input_and_timesteps(
        args, noise, latents, [0.375], scheduler, torch.device("cpu"), torch.float32, return_noisy=False
    )

    assert skipped is None
    torch.testing.assert_close(skipped_timesteps, timesteps)
    torch.testing.assert_close(noisy, torch.full_like(noisy, 0.375))


@pytest.mark.parametrize("bad_shift", ["0.0", "0.001", "1000.0"])
def test_h3_trainer_rejects_out_of_range_modality_shift(bad_shift):
    args = create_parser().parse_args(["--sdpa", "--h3_shift_video", bad_shift])

    with pytest.raises(ValueError, match="h3_shift_video"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_sigma_sqrt_weighting_is_bounded():
    args = create_parser().parse_args(["--sdpa", "--weighting_scheme", "sigma_sqrt"])
    trainer = MiniMaxH3NetworkTrainer()

    trainer.handle_model_specific_args(args)
    weights = trainer._sample_weight(args, torch.tensor([0.0, 0.1, 0.5, 1.0]))

    torch.testing.assert_close(weights, torch.tensor([10.0, 10.0, 4.0, 1.0]))


def test_h3_trainer_rejects_nonpositive_sigma_sqrt_cap():
    args = create_parser().parse_args(["--sdpa", "--h3_sigma_sqrt_max_weight", "0"])

    with pytest.raises(ValueError, match="must be positive"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_sigma_sqrt_cap_is_recorded_in_checkpoint_metadata():
    args = create_parser().parse_args(["--sdpa", "--h3_sigma_sqrt_max_weight", "8"])
    trainer = MiniMaxH3NetworkTrainer()

    trainer.handle_model_specific_args(args)

    assert trainer.extra_metadata(args)["ss_h3_sigma_sqrt_max_weight"] == "8.0"


def test_h3_trainer_rejects_negative_block_swap_count():
    args = create_parser().parse_args(["--sdpa", "--blocks_to_swap", "-1"])

    with pytest.raises(ValueError, match="non-negative"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_training_parser_defaults_to_native_fl2va_contract():
    parser = create_parser()
    args = parser.parse_args(["--sdpa"])
    assert args.network_module == "networks.lora_minimax_h3"
    assert args.h3_training_mode == "fl2va"
    assert args.mixed_precision == "bf16"
    assert args.timestep_sampling == "uniform"
    assert args.discrete_flow_shift == 1.0
    assert args.h3_shift_video == 12.0
    assert args.h3_shift_audio == 3.0
    assert args.h3_sigma_sqrt_max_weight == 10.0
    assert args.h3_loss_balance == "modality"
    assert args.h3_guidance_distillation_scale is None
    assert args.h3_guidance_loss_form == "normalized"
    assert args.h3_guidance_loss_schedule == "sigma"
    assert args.h3_base_preservation_loss_weight == 0.0
    assert args.fp8_scaled is False
    assert args.int8_convrot_base is False
    assert "--fp8_base" in parser._option_string_actions
    assert "--blocks_to_swap" in parser._option_string_actions
    assert "--block_swap_h2d_only" in parser._option_string_actions
    assert "--block_swap_ring_size" in parser._option_string_actions
    assert "--block_swap_granularity" in parser._option_string_actions
    assert args.block_swap_granularity == "block"
    assert "--fp8_scaled" not in parser._option_string_actions
    assert "--int8" not in parser._option_string_actions
    assert "--allow_prequantized_fp8" not in parser._option_string_actions


def test_h3_training_sampling_evacuates_and_restores_block_swap(monkeypatch):
    events = []

    class FakeTransformer:
        def offload_block_swap_to_cpu(self):
            events.append("offload")

        def move_to_device_except_swap_blocks(self, device):
            events.append(("restore", device.type))

        def switch_block_swap_for_inference(self):
            events.append("inference")

    def fake_denoise(*args, **kwargs):
        del args, kwargs
        events.append("denoise")
        return torch.zeros(1), torch.zeros(1)

    def fake_decode(*args, **kwargs):
        del args, kwargs
        events.append("decode")
        return SimpleNamespace(video=torch.zeros(1), audio=torch.zeros(1), sample_rate=32_000)

    monkeypatch.setattr(h3_train_network, "denoise_fl2va", fake_denoise)
    monkeypatch.setattr(h3_train_network, "decode_latents_sequentially", fake_decode)

    trainer = MiniMaxH3NetworkTrainer()
    trainer.blocks_to_swap = 2
    trainer._generate_sample(
        SimpleNamespace(device=torch.device("cpu")),
        FakeTransformer(),
        SimpleNamespace(video_decoder=object(), audio_decoder=object()),
        {
            "height": 32,
            "width": 32,
            "frame_count": 5,
            "sample_steps": 3,
            "seed": 1,
            H3_TEXT_HIDDEN_KEY: torch.zeros(1, 5120),
            H3_TEXT_TOKEN_TAGS_KEY: torch.ones(1, dtype=torch.long),
        },
    )

    assert events == ["denoise", "offload", "decode", ("restore", "cpu"), "inference"]


def _extension_layout(video_context=0, audio_context=0):
    return build_t2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        num_latent_frames=7,
        latent_height=4,
        latent_width=4,
        num_audio_latents=5,
        patch_size=(1, 2, 2),
        keyframe_anchors=tuple(range(video_context)),
        num_condition_audio_latents=audio_context,
    )


def test_h3_extension_context_rows_anchor_on_their_own_target_frames():
    # Extension observes a leading run of the target, so each condition frame
    # must sit exactly where the target frame it duplicates sits.
    layout = _extension_layout(video_context=3)
    rows_per_frame = 4
    target = layout.video_indices[layout.num_condition_video_rows :]

    for index in range(3):
        condition_time = layout.position_ids[layout.video_indices[index * rows_per_frame], 0]
        target_time = layout.position_ids[target[index * rows_per_frame], 0]
        torch.testing.assert_close(condition_time, target_time)


def test_h3_extension_audio_context_shares_the_opening_target_timeline():
    layout = _extension_layout(audio_context=2)

    assert layout.num_condition_audio_rows == 4
    audio = layout.audio_indices
    condition = layout.position_ids[audio[: layout.num_condition_audio_rows], 0]
    target = layout.position_ids[audio[layout.num_condition_audio_rows :], 0]
    torch.testing.assert_close(condition[:2], target[:2])
    torch.testing.assert_close(condition[2:], target[5:7])


def test_h3_extension_loss_mask_drops_the_observed_video_frames():
    trainer = MiniMaxH3NetworkTrainer()
    target = torch.ones(1, 24, 7, 2, 2)

    mask = trainer._extension_masked(None, target, 3, axis=-3)

    assert mask.shape == target.shape
    assert not bool(mask[:, :, :3].any())
    assert bool(mask[:, :, 3:].all())


def test_h3_extension_loss_mask_intersects_an_existing_mask():
    trainer = MiniMaxH3NetworkTrainer()
    target = torch.ones(1, 24, 7, 2, 2)
    existing = torch.ones(1, 24, 7, 2, 2, dtype=torch.bool)
    existing[:, :, 5:] = False

    mask = trainer._extension_masked(existing, target, 3, axis=-3)

    assert not bool(mask[:, :, :3].any())
    assert bool(mask[:, :, 3:5].all())
    assert not bool(mask[:, :, 5:].any())


def test_h3_extension_loss_mask_is_absent_without_context():
    trainer = MiniMaxH3NetworkTrainer()
    target = torch.ones(1, 24, 7, 2, 2)

    assert trainer._extension_masked(None, target, 0, axis=-3) is None


def test_h3_extension_context_must_be_shorter_than_the_target():
    trainer = MiniMaxH3NetworkTrainer()
    target = torch.ones(1, 24, 7, 2, 2)

    with pytest.raises(ValueError, match="covers the whole"):
        trainer._extension_masked(None, target, 7, axis=-3)


def test_h3_extension_context_reconstructs_the_clean_latents():
    # The packed rows are noised, so the observed context must be rebuilt from
    # the flow identity x0 = x_t + sigma * target. Slicing the noisy rows would
    # hand the model noise as context, which at high sigma carries no signal.
    trainer = MiniMaxH3NetworkTrainer()
    clean = torch.arange(1 * 2 * 4 * 1 * 1, dtype=torch.float32).reshape(1, 2, 4, 1, 1)
    noise = torch.randn(1, 2, 4, 1, 1)
    sigma = torch.tensor([0.8])
    noisy = (1 - sigma) * clean + sigma * noise
    target = clean - noise

    context = trainer._clean_context(noisy, target, sigma, 2, axis=-3)

    torch.testing.assert_close(context, clean[:, :, :2], rtol=1e-4, atol=1e-4)


def test_h3_extension_context_reconstruction_holds_at_extreme_sigma():
    # At sigma near 1 the noisy rows are almost pure noise, which is exactly the
    # case where reusing them as context would be silently useless.
    trainer = MiniMaxH3NetworkTrainer()
    clean = torch.full((1, 2, 3, 1, 1), 5.0)
    noise = torch.randn(1, 2, 3, 1, 1)
    sigma = torch.tensor([0.999])
    noisy = (1 - sigma) * clean + sigma * noise
    target = clean - noise

    context = trainer._clean_context(noisy, target, sigma, 1, axis=-3)

    torch.testing.assert_close(context, clean[:, :, :1], rtol=1e-3, atol=1e-3)
    assert not torch.allclose(context, noisy[:, :, :1], atol=1e-2)


def test_h3_extension_audio_context_slices_the_latent_axis():
    trainer = MiniMaxH3NetworkTrainer()
    clean = torch.arange(1 * 2 * 3 * 5, dtype=torch.float32).reshape(1, 2, 3, 5)
    noise = torch.zeros_like(clean)
    sigma = torch.tensor([0.0])
    target = clean - noise

    context = trainer._clean_context(clean, target, sigma, 2, axis=-1)

    assert context.shape == (1, 2, 3, 2)
    torch.testing.assert_close(context, clean[..., :2])


def test_h3_extension_route_defaults_to_the_released_contract():
    args = create_parser().parse_args(["--dataset_config", "x", "--dit", "y"])

    assert args.h3_extension_route == "condition_rows"


def test_h3_masking_box_generates_an_interior_region():
    g = torch.Generator().manual_seed(0)
    mask = sample_video_mask(mode="box", latent_frames=3, latent_height=8, latent_width=8, generator=g)

    assert mask.shape == (3, 8, 8)
    assert bool(mask.any()) and not bool(mask.all())
    # A box is constant over time and contiguous in space.
    assert bool((mask[0] == mask[1]).all())
    rows = mask[0].any(dim=1).nonzero().flatten()
    assert bool((rows.diff() == 1).all())


def test_h3_masking_border_is_the_complement_of_a_box():
    g = torch.Generator().manual_seed(0)
    border = sample_video_mask(mode="border", latent_frames=2, latent_height=8, latent_width=8, generator=g)
    g = torch.Generator().manual_seed(0)
    box = sample_video_mask(mode="box", latent_frames=2, latent_height=8, latent_width=8, generator=g)

    torch.testing.assert_close(border, ~box)


def test_h3_masking_segment_selects_whole_frames():
    g = torch.Generator().manual_seed(3)
    mask = sample_video_mask(mode="segment", latent_frames=6, latent_height=4, latent_width=4, generator=g)

    per_frame = mask.reshape(6, -1)
    assert bool(((per_frame.all(dim=1)) | (~per_frame.any(dim=1))).all())


def test_h3_masking_rows_flag_a_patch_when_any_latent_inside_it_is_generated():
    # Reducing with any rather than all keeps the boundary inside the generated
    # set, which is where an inpainting seam would otherwise land.
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[0, 0, 0] = True

    rows = video_mask_to_rows(mask, (1, 2, 2))

    assert rows.shape == (4,)
    assert bool(rows[0]) and not bool(rows[1:].any())


def test_h3_masking_audio_rows_repeat_across_channels():
    mask = torch.tensor([True, False, True])

    rows = audio_mask_to_rows(mask, channels=2)

    assert rows.tolist() == [True, False, True, True, False, True]


def test_h3_masking_is_reproducible_from_a_seed():
    a = sample_video_mask(mode="box", latent_frames=2, latent_height=8, latent_width=8, generator=torch.Generator().manual_seed(7))
    b = sample_video_mask(mode="box", latent_frames=2, latent_height=8, latent_width=8, generator=torch.Generator().manual_seed(7))

    torch.testing.assert_close(a, b)


def test_h3_masking_rejects_unsupported_modes_and_fractions():
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="unsupported"):
        sample_video_mask(mode="spiral", latent_frames=2, latent_height=4, latent_width=4, generator=g)
    with pytest.raises(ValueError, match="fractions"):
        sample_video_mask(mode="box", latent_frames=2, latent_height=4, latent_width=4, generator=g, minimum=0.8, maximum=0.2)


class _MaskRecordingBackend(_FakeBackend):
    def __init__(self):
        super().__init__()
        self.observed = []

    def predict_training(
        self, transformer, batch, video_hidden_states, audio_hidden_states, video_timestep, audio_timestep, **kwargs
    ):
        self.observed.append(
            (
                kwargs.get("observed_video_rows"),
                kwargs.get("observed_audio_rows"),
                kwargs.get("clean_video_latents"),
            )
        )
        return super().predict_training(
            transformer,
            batch,
            video_hidden_states,
            audio_hidden_states,
            video_timestep,
            audio_timestep,
            conditioning=kwargs.get("conditioning", "prompt"),
        )


def _run_masked(mode, *, mask_audio=False, guidance_scale=None):
    args = create_parser().parse_args([])
    args.h3_mask_mode = mode
    args.h3_mask_audio = mask_audio
    args.h3_guidance_distillation_scale = guidance_scale
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    backend = _MaskRecordingBackend()
    trainer.backend = backend
    trainer._mask_mode = mode
    trainer._mask_audio = mask_audio
    # A realistic patch grid: at 2x2 patches a box spanning up to 75% of each
    # axis touches every patch, so the any() reduction would mark all rows
    # generated and the mask would carry no information.
    video = torch.zeros(1, 24, 4, 16, 16)
    batch = {
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 8),
        H3_AUDIO_LOSS_MASK_KEY: torch.ones(1, 8, dtype=torch.bool),
        "timesteps": [0.5],
    }
    if guidance_scale is not None:
        batch[H3_EMPTY_TEXT_HIDDEN_KEY] = [torch.zeros(1, 5120)]
        batch[H3_EMPTY_TEXT_TOKEN_TAGS_KEY] = [torch.ones(1, dtype=torch.long)]
    torch.manual_seed(0)
    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        _ScaleTransformer(),
        None,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    return trainer, backend, loss


def test_h3_masked_conditioning_marks_some_rows_observed():
    trainer, backend, loss = _run_masked("box")

    observed_video, _, clean = backend.observed[0]
    assert observed_video is not None and clean is not None
    assert bool(observed_video.any()) and not bool(observed_video.all())
    assert torch.isfinite(loss)


def test_h3_masked_conditioning_is_absent_when_disabled():
    trainer, backend, _ = _run_masked("off")

    observed_video, observed_audio, clean = backend.observed[0]
    assert observed_video is None and observed_audio is None and clean is None
    assert trainer._step_mask is None


def test_h3_masked_conditioning_shares_one_mask_across_every_forward():
    # The empty and trainable branches must agree on what is observed, or the
    # guidance correction compares two different problems.
    _, backend, _ = _run_masked("box", guidance_scale=4.0)

    assert len(backend.observed) == 2
    first, second = backend.observed[0][0], backend.observed[1][0]
    torch.testing.assert_close(first, second)


def test_h3_masked_audio_is_opt_in_alongside_video():
    _, backend, _ = _run_masked("box", mask_audio=True)

    _, observed_audio, _ = backend.observed[0]
    assert observed_audio is not None
    assert bool(observed_audio.any()) and not bool(observed_audio.all())


def test_h3_mask_loss_restricts_scoring_to_the_generated_region():
    trainer = MiniMaxH3NetworkTrainer()
    target = torch.ones(1, 24, 4, 2, 2)
    generated = torch.zeros(4, 2, 2, dtype=torch.bool)
    generated[1:3] = True

    mask = trainer._mask_to_loss(None, target, generated, axis=-3)

    assert mask.shape == target.shape
    assert not bool(mask[:, :, 0].any()) and bool(mask[:, :, 1:3].all())


def test_h3_mask_rejects_combining_with_extension():
    args = create_parser().parse_args([])
    args.h3_mask_mode = "box"
    args.h3_extension_video_frames = 2
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match="only one"):
        trainer.handle_model_specific_args(args)


@pytest.mark.parametrize("mode", ["ref2va", "ref2va_omni"])
def test_h3_masked_conditioning_combines_with_ref2va_training(mode):
    # Masking pins rows inside the target block, which the Ref2VA layout carries
    # unchanged behind its reference prefix, so it is no longer T2VA-only.
    args = create_parser().parse_args(["--sdpa"])
    args.h3_training_mode = mode
    args.h3_mask_mode = "box"
    args.h3_mask_audio = True
    trainer = MiniMaxH3NetworkTrainer()

    trainer.handle_model_specific_args(args)

    assert trainer._mask_mode == "box" and trainer._mask_audio


@pytest.mark.parametrize("mode", ["ref2va", "ref2va_omni"])
def test_h3_extension_under_ref2va_requires_the_per_row_sigma_route(mode):
    args = create_parser().parse_args(["--sdpa"])
    args.h3_training_mode = mode
    args.h3_extension_video_frames = 2
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match="requires --h3_extension_route per_row_sigma"):
        trainer.handle_model_specific_args(args)

    args.h3_extension_route = "per_row_sigma"
    trainer.handle_model_specific_args(args)

    assert trainer._extension_video_frames == 2 and trainer._extension_route == "per_row_sigma"


def test_h3_extension_keeps_the_condition_rows_route_under_fl2va():
    args = create_parser().parse_args(["--sdpa"])
    args.h3_extension_audio_latents = 1
    trainer = MiniMaxH3NetworkTrainer()

    trainer.handle_model_specific_args(args)

    assert trainer._extension_route == "condition_rows"


@pytest.mark.parametrize("mode", ["ref2va", "ref2va_omni"])
def test_h3_keyframe_conditioning_stays_t2va_only(mode):
    # Keyframes duplicate the observed frames as extra condition rows, which
    # only the T2VA packer knows how to place.
    args = create_parser().parse_args([])
    args.h3_training_mode = mode
    args.h3_keyframe_anchors = "first,last"
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match="requires --h3_training_mode fl2va"):
        trainer.handle_model_specific_args(args)


class _PackedRowRecorder:
    """Capture the packed rows and per-row sigma a forward would have received."""

    def __init__(self):
        self.config = SimpleNamespace(in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(video=kwargs["video_hidden_states"], audio=kwargs["audio_hidden_states"])

    @property
    def row_sigma(self):
        call = self.calls[-1]
        return call["timestep"][call["timestep_indices"]]


def _ref2va_conditioning_batch(task="ref2va", *, references=True):
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(3, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor([1, 0, 1] if references else [1, 1, 1])],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS[task])],
    }
    if references:
        # One image reference (one video row) and one audio reference (two
        # channel-major rows), so both modalities carry a conditioning prefix.
        batch.update(
            {
                H3_REFERENCE_KINDS_KEY: [torch.tensor([0, 2])],
                H3_REFERENCE_VIDEO_SHAPES_KEY: [torch.tensor([[1, 2, 2], [0, 0, 0]])],
                H3_REFERENCE_AUDIO_LENGTHS_KEY: [torch.tensor([0, 1])],
                H3_REFERENCE_VIDEO_ROWS_KEY: [torch.full((1, 16), 5.0)],
                H3_REFERENCE_AUDIO_ROWS_KEY: [torch.full((2, 6), 7.0)],
            }
        )
    return batch


def test_h3_masked_conditioning_offsets_past_the_ref2va_reference_rows():
    # The mask is drawn at target-latent granularity, so under Ref2VA it must be
    # applied one reference prefix into the packed rows. Getting the offset wrong
    # would pin the references and leave the matching target rows noisy.
    transformer = _PackedRowRecorder()
    backend = _NativeTrainingBackend(transformer, mode="ref2va")
    video = torch.zeros(1, 4, 2, 4, 4)
    audio = torch.zeros(1, 2, 6, 3)
    observed_video = torch.tensor([True] * 4 + [False] * 4)
    observed_audio = torch.tensor([True, False, False] * 2)

    torch.manual_seed(0)
    prediction = backend.predict_training(
        transformer,
        _ref2va_conditioning_batch(),
        video,
        audio,
        torch.tensor([0.4]),
        torch.tensor([0.7]),
        observed_video_rows=observed_video,
        clean_video_latents=torch.ones_like(video),
        observed_audio_rows=observed_audio,
        clean_audio_latents=torch.ones_like(audio),
    )

    # The prediction covers the target alone: reference rows are conditioning and
    # are dropped before the loss ever sees them.
    assert prediction.video.shape == video.shape and prediction.audio.shape == audio.shape

    call = transformer.calls[-1]
    sigma = transformer.row_sigma
    video_indices, audio_indices = call["video_indices"], call["audio_indices"]
    # One reference video row, then eight target rows; two reference audio rows,
    # then six target rows.
    assert float(sigma[video_indices[0]]) == pytest.approx(0.999)
    assert bool((sigma[video_indices[1:5]] == 0.999).all())
    assert bool((sigma[video_indices[5:9]] == 0.4).all())
    assert bool((sigma[audio_indices[:2]] == 1.0).all())
    torch.testing.assert_close(sigma[audio_indices[2:]], torch.tensor([1.0, 0.7, 0.7, 1.0, 0.7, 0.7]))

    packed_video = call["video_hidden_states"][0]
    packed_audio = call["audio_hidden_states"][0]
    # The reference rows keep their cached content, blended with the released
    # conditioning noise; only the target rows are swapped for clean latents.
    torch.testing.assert_close(packed_video[0], torch.full((16,), 0.999 * 5.0), atol=5e-3, rtol=0)
    torch.testing.assert_close(packed_video[1:5], torch.full((4, 16), 0.999), atol=5e-3, rtol=0)
    assert bool((packed_video[5:9] == 0.0).all())
    assert bool((packed_audio[:2] == 7.0).all())
    torch.testing.assert_close(packed_audio[2:, 0], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))


def test_h3_masked_conditioning_under_ref2va_rejects_a_packed_length_mask():
    transformer = _PackedRowRecorder()
    backend = _NativeTrainingBackend(transformer, mode="ref2va")
    video = torch.zeros(1, 4, 2, 4, 4)

    with pytest.raises(ValueError, match="rows for 8 target rows"):
        backend.predict_training(
            transformer,
            _ref2va_conditioning_batch(),
            video,
            None,
            torch.tensor([0.4]),
            torch.tensor([0.7]),
            observed_video_rows=torch.ones(9, dtype=torch.bool),
            clean_video_latents=torch.ones_like(video),
        )


def test_h3_ref2va_omni_without_references_masks_exactly_as_t2va():
    # A zero-reference omni sample has no conditioning prefix, so the offset is
    # zero and the packed sequence must be identical to the T2VA one.
    video = torch.zeros(1, 4, 2, 4, 4)
    audio = torch.zeros(1, 2, 6, 3)
    observed_video = torch.tensor([False, True] * 4)
    observed_audio = torch.tensor([True, False, False] * 2)
    packed = []
    for mode, task in (("ref2va_omni", "ref2va_omni"), ("fl2va", "t2va")):
        transformer = _PackedRowRecorder()
        backend = _NativeTrainingBackend(transformer, mode=mode)
        torch.manual_seed(0)
        backend.predict_training(
            transformer,
            _ref2va_conditioning_batch(task, references=False),
            video,
            audio,
            torch.tensor([0.4]),
            torch.tensor([0.7]),
            observed_video_rows=observed_video,
            clean_video_latents=torch.ones_like(video),
            observed_audio_rows=observed_audio,
            clean_audio_latents=torch.ones_like(audio),
        )
        packed.append(transformer.calls[-1])

    omni, t2va = packed
    for key in ("video_hidden_states", "audio_hidden_states", "position_ids", "token_tags", "video_indices", "audio_indices"):
        torch.testing.assert_close(omni[key], t2va[key])
    torch.testing.assert_close(
        omni["timestep"][omni["timestep_indices"]],
        t2va["timestep"][t2va["timestep_indices"]],
    )


def test_h3_ref2va_extension_pins_the_leading_target_rows_on_the_per_row_sigma_route():
    transformer = _PackedRowRecorder()
    backend = _NativeTrainingBackend(transformer, mode="ref2va")
    video = torch.zeros(1, 4, 2, 4, 4)
    audio = torch.zeros(1, 2, 6, 3)

    torch.manual_seed(0)
    backend.predict_training(
        transformer,
        _ref2va_conditioning_batch(),
        video,
        audio,
        torch.tensor([0.4]),
        torch.tensor([0.7]),
        extension_video_frames=1,
        extension_video_context=torch.ones(1, 4, 1, 4, 4),
        extension_audio_latents=1,
        extension_audio_context=torch.ones(1, 2, 6, 1),
        extension_route="per_row_sigma",
    )

    call = transformer.calls[-1]
    sigma = transformer.row_sigma
    video_indices, audio_indices = call["video_indices"], call["audio_indices"]
    assert float(sigma[video_indices[0]]) == pytest.approx(0.999)
    assert bool((sigma[video_indices[1:5]] == 0.999).all())
    assert bool((sigma[video_indices[5:9]] == 0.4).all())
    torch.testing.assert_close(sigma[audio_indices[2:]], torch.tensor([1.0, 0.7, 0.7, 1.0, 0.7, 0.7]))

    packed_video = call["video_hidden_states"][0]
    packed_audio = call["audio_hidden_states"][0]
    torch.testing.assert_close(packed_video[0], torch.full((16,), 0.999 * 5.0), atol=5e-3, rtol=0)
    torch.testing.assert_close(packed_video[1:5], torch.full((4, 16), 0.999), atol=5e-3, rtol=0)
    assert bool((packed_video[5:9] == 0.0).all())
    assert bool((packed_audio[:2] == 7.0).all())
    torch.testing.assert_close(packed_audio[2:, 0], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))


def test_h3_ref2va_extension_rejects_the_condition_rows_route():
    transformer = _PackedRowRecorder()
    backend = _NativeTrainingBackend(transformer, mode="ref2va")

    with pytest.raises(ValueError, match="only supported on the per_row_sigma route"):
        backend.predict_training(
            transformer,
            _ref2va_conditioning_batch(),
            torch.zeros(1, 4, 2, 4, 4),
            None,
            torch.tensor([0.4]),
            torch.tensor([0.7]),
            extension_video_frames=1,
            extension_video_context=torch.ones(1, 4, 1, 4, 4),
        )


def test_h3_ref2va_still_rejects_custom_keyframe_anchors():
    transformer = _PackedRowRecorder()
    backend = _NativeTrainingBackend(transformer, mode="ref2va")

    with pytest.raises(ValueError, match="custom keyframe anchors require --task t2va caches"):
        backend.predict_training(
            transformer,
            _ref2va_conditioning_batch(),
            torch.zeros(1, 4, 2, 4, 4),
            None,
            torch.tensor([0.4]),
            torch.tensor([0.7]),
            condition_video_anchors=("first",),
            extension_video_context=torch.ones(1, 4, 1, 4, 4),
        )


def test_h3_random_observed_modality_covers_all_three_tasks():
    # Redrawing the task per step is what keeps one adapter able to do joint
    # generation, A2V and V2A instead of specialising on whichever was fixed.
    args = create_parser().parse_args([])
    args.h3_observed_modality = "random"
    seen = set()
    for seed in range(40):
        trainer = MiniMaxH3NetworkTrainer()
        trainer.dit_dtype = torch.float32
        trainer.backend = _FakeBackend()
        video = torch.zeros(1, 24, 2, 2, 2)
        batch = {
            H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
            H3_AUDIO_LOSS_MASK_KEY: torch.ones(1, 3, dtype=torch.bool),
            "timesteps": [0.5],
        }
        torch.manual_seed(seed)
        _, metrics = trainer.process_batch(
            args,
            _FakeAccelerator(),
            _ScaleTransformer(),
            None,
            batch,
            video,
            torch.ones_like(video),
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )
        # sigma_video is pinned at the conditioning level exactly when video is observed.
        seen.add(round(metrics["h3/sigma_video"], 3))
    assert len(seen) > 1


def test_h3_random_observed_modality_never_selects_a_fully_masked_audio_target():
    args = create_parser().parse_args([])
    args.h3_observed_modality = "random"
    for seed in range(20):
        trainer = MiniMaxH3NetworkTrainer()
        trainer.dit_dtype = torch.float32
        trainer.backend = _FakeBackend()
        video = torch.zeros(1, 24, 2, 2, 2)
        batch = {
            H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
            H3_AUDIO_LOSS_MASK_KEY: torch.zeros(1, 3, dtype=torch.bool),
            "timesteps": [0.5],
        }
        torch.manual_seed(seed)
        loss, metrics = trainer.process_batch(
            args,
            _FakeAccelerator(),
            _ScaleTransformer(),
            None,
            batch,
            video,
            torch.ones_like(video),
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )

        assert float(loss.detach()) > 0
        assert metrics["h3/sigma_video"] > 0.001


def test_h3_random_observed_modality_is_reported_as_joint_in_validation():
    args = create_parser().parse_args([])
    args.h3_observed_modality = "random"

    resolved = None if args.h3_observed_modality == "random" else args.h3_observed_modality

    assert resolved is None


def test_h3_observed_modality_random_is_accepted_by_the_parser():
    args = create_parser().parse_args(["--dataset_config", "x", "--dit", "y", "--h3_observed_modality", "random"])

    assert args.h3_observed_modality == "random"


def _keyframe_trainer(spec="", random_count=0):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._keyframe_anchors = h3_train_network._parse_keyframe_anchors(spec)
    trainer._keyframe_random_count = random_count
    return trainer


def test_h3_keyframe_spec_resolves_named_and_indexed_anchors():
    trainer = _keyframe_trainer("first,11,last")
    video = torch.zeros(1, 24, 22, 2, 2)

    anchors, indices = trainer._resolve_keyframe_anchors(video)
    # "last" survives as itself: the packer needs the released final-pixel-frame
    # coordinate, while the content comes from the final latent window.
    assert anchors == ("first", 11, "last")
    assert indices == (0, 11, 21)


def test_h3_keyframe_anchors_are_sorted_and_deduplicated_against_the_clip():
    trainer = _keyframe_trainer("last,first")
    video = torch.zeros(1, 24, 8, 2, 2)

    assert trainer._resolve_keyframe_anchors(video) == (("first", "last"), (0, 7))

    duplicate = _keyframe_trainer("first,0")
    with pytest.raises(ValueError, match="duplicate"):
        duplicate._resolve_keyframe_anchors(video)


def test_h3_keyframe_anchor_outside_the_clip_is_rejected():
    trainer = _keyframe_trainer("30")
    video = torch.zeros(1, 24, 8, 2, 2)

    with pytest.raises(ValueError, match="outside"):
        trainer._resolve_keyframe_anchors(video)


def test_h3_keyframe_random_draw_is_distinct_and_within_range():
    trainer = _keyframe_trainer(random_count=3)
    video = torch.zeros(1, 24, 10, 2, 2)

    torch.manual_seed(0)
    anchors, _ = trainer._resolve_keyframe_anchors(video)

    assert len(anchors) == 3 and len(set(anchors)) == 3
    assert all(0 <= a < 10 for a in anchors)
    assert list(anchors) == sorted(anchors)


def test_h3_keyframe_random_count_is_clamped_to_the_clip_length():
    trainer = _keyframe_trainer(random_count=99)
    video = torch.zeros(1, 24, 4, 2, 2)

    assert len(trainer._resolve_keyframe_anchors(video)[0]) == 4


def test_h3_keyframe_spec_rejects_nonsense():
    with pytest.raises(ValueError, match="must be"):
        h3_train_network._parse_keyframe_anchors("first,middle")


def test_h3_keyframe_anchors_are_absent_by_default():
    trainer = _keyframe_trainer()

    assert trainer._resolve_keyframe_anchors(torch.zeros(1, 24, 8, 2, 2)) == ((), ())


@pytest.mark.parametrize("conflicting", ["extension", "mask"])
def test_h3_keyframe_conditioning_rejects_competing_observers(conflicting):
    args = create_parser().parse_args([])
    args.h3_keyframe_anchors = "first,last"
    if conflicting == "extension":
        args.h3_extension_video_frames = 2
    else:
        args.h3_mask_mode = "box"
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match="only one"):
        trainer.handle_model_specific_args(args)


def _jitter_inputs(clean, noise):
    from musubi_tuner.minimax_h3.training import H3JointNoisyInputs

    return H3JointNoisyInputs(
        video=clean,
        audio=None,
        video_target=clean - noise,
        audio_target=None,
        video_sigma=torch.tensor([0.5]),
        audio_sigma=torch.tensor([0.5]),
        video_timestep=torch.tensor([0.5]),
        audio_timestep=torch.tensor([0.5]),
    )


def _jitter_trainer(jitter):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._frame_sigma_jitter = jitter
    return trainer


def test_h3_frame_sigma_jitter_is_inert_at_zero():
    trainer = _jitter_trainer(0.0)
    args = create_parser().parse_args([])
    inputs = SimpleNamespace(video=torch.zeros(1, 4, 3, 2, 2))

    result, schedule = trainer._apply_frame_sigma_jitter(
        args, inputs, torch.zeros(1, 4, 3, 2, 2), torch.ones(1, 4, 3, 2, 2), torch.tensor([0.5]), False
    )

    assert result is inputs and schedule is None


def test_h3_frame_sigma_jitter_gives_each_frame_its_own_noise_level():
    trainer = _jitter_trainer(0.2)
    args = create_parser().parse_args([])
    clean = torch.zeros(1, 4, 3, 2, 2)
    noise = torch.ones(1, 4, 3, 2, 2)

    torch.manual_seed(0)
    result, schedule = trainer._apply_frame_sigma_jitter(
        args, _jitter_inputs(clean, noise), clean, noise, torch.tensor([0.5]), False
    )

    # With clean zeros the noised frame equals its own sigma, so the frames must differ.
    per_frame = result.video[0, 0, :, 0, 0]
    assert len(set(per_frame.tolist())) > 1
    # One timestep per packed row, and each frame's rows share a value.
    assert schedule.numel() == 3 * 1
    torch.testing.assert_close(schedule, 1.0 - per_frame.to(schedule.dtype), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("base", [0.0, 1.0])
def test_h3_frame_sigma_jitter_does_not_collapse_probability_at_schedule_endpoints(base):
    trainer = _jitter_trainer(0.2)
    args = create_parser().parse_args([])
    clean = torch.zeros(1, 4, 8, 2, 2)
    noise = torch.ones_like(clean)

    result, _ = trainer._apply_frame_sigma_jitter(args, _jitter_inputs(clean, noise), clean, noise, torch.tensor([base]), False)

    per_frame = result.video[0, 0, :, 0, 0]
    assert bool(((per_frame > 0) & (per_frame < 1)).all())
    assert len(set(per_frame.tolist())) > 1
    torch.testing.assert_close(result.video_sigma, per_frame.mean().reshape(1))
    torch.testing.assert_close(result.video_timestep, 1.0 - result.video_sigma)


def test_h3_frame_sigma_jitter_leaves_the_flow_target_alone():
    # The target x0 - noise does not depend on sigma, so jitter must not change it.
    trainer = _jitter_trainer(0.3)
    args = create_parser().parse_args([])
    clean = torch.randn(1, 4, 3, 2, 2)
    noise = torch.randn(1, 4, 3, 2, 2)
    inputs = _jitter_inputs(clean, noise)

    torch.manual_seed(0)
    result, _ = trainer._apply_frame_sigma_jitter(args, inputs, clean, noise, torch.tensor([0.4]), False)

    torch.testing.assert_close(result.video_target, clean - noise)


def test_h3_frame_sigma_jitter_skips_image_batches():
    # An image batch is one frame, so a per-frame spread has nothing to spread.
    trainer = _jitter_trainer(0.5)
    args = create_parser().parse_args([])
    inputs = SimpleNamespace(video=torch.zeros(1, 4, 1, 2, 2))

    result, schedule = trainer._apply_frame_sigma_jitter(
        args, inputs, torch.zeros(1, 4, 1, 2, 2), torch.ones(1, 4, 1, 2, 2), torch.tensor([0.5]), True
    )

    assert result is inputs and schedule is None


@pytest.mark.parametrize("jitter", [-0.1, 1.5])
def test_h3_frame_sigma_jitter_is_validated(jitter):
    args = create_parser().parse_args([])
    args.h3_frame_sigma_jitter = jitter
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match="h3_frame_sigma_jitter"):
        trainer.handle_model_specific_args(args)


@pytest.mark.parametrize("conflict", ["sigma_sqrt", "cosmap"])
def test_h3_frame_sigma_jitter_rejects_scalar_sigma_objectives(conflict):
    args = create_parser().parse_args([])
    args.h3_frame_sigma_jitter = 0.1
    args.weighting_scheme = conflict

    with pytest.raises(ValueError, match="h3_frame_sigma_jitter"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_frame_sigma_jitter_supports_sigma_scheduled_guidance_per_frame():
    args = create_parser().parse_args([])
    args.h3_guidance_distillation_scale = 3.0
    args.h3_guidance_loss_schedule = "sigma"
    inputs = _jitter_inputs(torch.zeros(1, 4, 3, 1, 1), torch.ones(1, 4, 3, 1, 1))
    inputs = replace(inputs, video_frame_sigma=torch.tensor([0.1, 0.5, 0.9]))
    guided = H3ModelPrediction(torch.ones_like(inputs.video), None)
    empty = H3ModelPrediction(torch.zeros_like(inputs.video), None)

    corrected, _ = MiniMaxH3NetworkTrainer._guidance_loss_inputs(args, guided, empty, inputs)

    expected = 1.0 / (1.0 + 2.0 * inputs.video_frame_sigma)
    torch.testing.assert_close(corrected.video[0, 0, :, 0, 0], expected)


def test_h3_keyframe_last_is_not_collapsed_to_the_final_index():
    # "last" names the final pixel frame and index N-1 the final latent window
    # start; collapsing them would silently move the released anchor.
    trainer = _keyframe_trainer("last")
    video = torch.zeros(1, 24, 8, 2, 2)

    anchors, indices = trainer._resolve_keyframe_anchors(video)

    assert anchors == ("last",)
    assert indices == (7,)


def test_h3_keyframe_last_and_final_index_are_distinct_anchors():
    trainer = _keyframe_trainer("last,7")
    video = torch.zeros(1, 24, 8, 2, 2)

    anchors, indices = trainer._resolve_keyframe_anchors(video)

    assert set(anchors) == {"last", 7}
    assert indices == (7, 7)


@pytest.mark.parametrize("anchors", [("first", "last"), (0, 1), ("first", 1)])
def test_native_h3_backend_accepts_named_and_indexed_condition_anchors(anchors):
    # The released "first"/"last" tokens reach the packer as themselves, so the
    # backend must not range-check them as integers. A unit test of the anchor
    # resolver alone cannot catch this; only the packing path can.
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=2,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer, mode="fl2va")
    video = torch.randn(1, 4, 3, 2, 2)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(3, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor([1, 1, 1])],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
    }

    prediction = backend.predict_training(
        transformer,
        batch,
        video,
        torch.randn(1, 2, 6, 1),
        torch.tensor([0.4]),
        torch.tensor([0.7]),
        condition_video_anchors=anchors,
        extension_video_context=torch.randn(1, 4, len(anchors), 2, 2),
    )

    assert prediction.video.shape == video.shape

    batch[H3_CONDITIONING_TASK_KEY] = [torch.tensor(H3_CONDITIONING_TASK_IDS["i2va"])]
    with pytest.raises(ValueError, match="custom keyframe anchors require --task t2va caches"):
        backend.predict_training(
            transformer,
            batch,
            video,
            torch.randn(1, 2, 6, 1),
            torch.tensor([0.4]),
            torch.tensor([0.7]),
            condition_video_anchors=anchors,
            extension_video_context=torch.randn(1, 4, len(anchors), 2, 2),
        )

    with pytest.raises(ValueError, match="cannot represent observed video rows"):
        backend.predict_training(
            transformer,
            batch,
            video,
            torch.randn(1, 2, 6, 1),
            torch.tensor([0.4]),
            torch.tensor([0.7]),
            observed_video_rows=torch.zeros(3, dtype=torch.bool),
        )


def test_h3_base_sigma_resolves_the_scheduler_branch_in_fp32():
    # The scheduler branch used to read the schedule in the DiT dtype and only
    # widen afterwards, quantizing the base coordinate to BF16's ~256 distinct
    # values in [0, 1] before H3's 12/3 shifts are applied downstream.
    trainer = MiniMaxH3NetworkTrainer()
    args = create_parser().parse_args(["--sdpa", "--timestep_sampling", "sigma"])
    steps = torch.tensor([999.0, 501.0, 3.0])
    scheduler = SimpleNamespace(sigmas=steps / 1000.0, timesteps=steps)

    base_sigma = trainer._base_sigma(args, scheduler, steps, torch.device("cpu"))

    assert base_sigma.dtype is torch.float32
    torch.testing.assert_close(base_sigma, (steps / 1000.0).to(torch.float32))
    # Every value survives exactly; the BF16 round trip collapses two of them.
    assert not torch.equal(base_sigma, base_sigma.to(torch.bfloat16).to(torch.float32))


def _guidance_probability_batch():
    video = torch.zeros(1, 24, 2, 2, 2)
    batch = {
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        H3_AUDIO_LOSS_MASK_KEY: torch.zeros(1, 3, dtype=torch.bool),
        "timesteps": [0.5],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 5120)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
    }
    return video, batch


def _run_guidance_probability(*, guidance_scale, probability, active):
    args = create_parser().parse_args([])
    args.h3_guidance_distillation_scale = guidance_scale
    args.h3_guidance_distillation_probability = probability
    trainer = MiniMaxH3NetworkTrainer()
    trainer.dit_dtype = torch.float32
    backend = _FakeBackend()
    trainer.backend = backend
    trainer._guidance_distillation_active = lambda accelerator, probability: active
    transformer = _ScaleTransformer()
    video, batch = _guidance_probability_batch()

    torch.manual_seed(0)
    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        None,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    return backend, float(loss.detach()), metrics


def test_h3_sparse_guidance_distillation_skips_the_empty_forward():
    backend, loss, metrics = _run_guidance_probability(guidance_scale=4.0, probability=0.5, active=False)
    _, plain_loss, _ = _run_guidance_probability(guidance_scale=None, probability=1.0, active=True)

    assert backend.calls == [("prompt", True)]
    assert metrics["h3/guidance_distillation_active"] == 0.0
    assert loss == pytest.approx(plain_loss)


def test_h3_sparse_guidance_distillation_scales_only_the_guidance_correction():
    _, dense_loss, _ = _run_guidance_probability(guidance_scale=4.0, probability=1.0, active=True)
    _, plain_loss, _ = _run_guidance_probability(guidance_scale=None, probability=1.0, active=True)
    backend, sparse_loss, metrics = _run_guidance_probability(guidance_scale=4.0, probability=0.25, active=True)

    assert backend.calls == [("empty", False), ("prompt", True)]
    assert metrics["h3/guidance_distillation_active"] == 1.0
    assert sparse_loss == pytest.approx(plain_loss + 4 * (dense_loss - plain_loss), rel=1e-5)
    # The running average keeps reporting the dense objective.
    assert metrics["_loss_for_average"] == pytest.approx(dense_loss, rel=1e-5)


@pytest.mark.parametrize("probability", [0.0, -0.1, 1.1, float("nan")])
def test_h3_guidance_distillation_probability_is_validated(probability):
    args = create_parser().parse_args([])
    args.h3_guidance_distillation_scale = 4.0
    args.h3_guidance_distillation_probability = probability

    with pytest.raises(ValueError, match="h3_guidance_distillation_probability"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_guidance_distillation_probability_requires_a_scale():
    args = create_parser().parse_args([])
    args.h3_guidance_distillation_probability = 0.5

    with pytest.raises(ValueError, match="requires --h3_guidance_distillation_scale"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_guidance_distillation_draw_is_independent_of_the_global_cpu_stream(monkeypatch):
    broadcasts = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda value, src: broadcasts.append((value.clone(), src)))

    torch.manual_seed(321)
    MiniMaxH3NetworkTrainer()._guidance_distillation_active(_FakeAccelerator(), 0.5)  # seeds the generator once
    expected_next = torch.rand(())

    trainer = MiniMaxH3NetworkTrainer()
    torch.manual_seed(321)
    for _ in range(6):
        trainer._guidance_distillation_active(_FakeAccelerator(), 0.5)
    actual_next = torch.rand(())

    # Only the one-time seeding touches the global stream; the per-step draws do not.
    assert actual_next == expected_next
    assert len(broadcasts) == 7 and all(source == 0 for _, source in broadcasts)


def test_h3_guidance_distillation_probability_is_recorded_in_metadata():
    args = create_parser().parse_args([])
    args.h3_guidance_distillation_scale = 4.0
    args.h3_guidance_distillation_probability = 0.5

    metadata = MiniMaxH3NetworkTrainer().extra_metadata(args)

    assert metadata["ss_h3_guidance_distillation_probability"] == "0.5"
