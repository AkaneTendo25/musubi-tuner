from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

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
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    H3_VIDEO_GEOMETRY_KEY,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
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
from musubi_tuner.minimax_h3.training import (
    H3ModelPrediction,
    guidance_consistent_prediction,
    joint_velocity_loss,
    map_sigma_between_shifts,
    prepare_joint_noisy_inputs,
    shift_sigma,
    unshift_sigma,
)
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
from musubi_tuner.networks import lora_minimax_h3


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


def test_native_h3_t2va_backend_runs_joint_forward_and_backward():
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
    transformer.enable_gradient_checkpointing()
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
    result = joint_velocity_loss(prediction, inputs)
    result.loss.backward()

    assert prediction.video.shape == video_latents.shape
    assert prediction.audio.shape == audio_latents.shape
    assert torch.isfinite(result.loss)
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None
    assert torch.isfinite(transformer.blocks[0].attn.qkv_proj.weight.grad).all()


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


@pytest.mark.parametrize(("task", "text_tags"), (("i2va", [1, 0, 1]), ("fl2va", [1, 0, 0, 1])))
def test_native_h3_fl2va_backend_runs_keyframe_conditioned_target_only_backward(task, text_tags):
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
    video = torch.randn(1, 4, 1, 2, 2)
    audio = torch.randn(1, 2, 6, 1)
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
    )
    loss = prediction.video.square().mean() + prediction.audio.square().mean()
    loss.backward()

    assert prediction.video.shape == video.shape
    assert prediction.audio.shape == audio.shape
    assert torch.isfinite(loss)
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None


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


def test_native_h3_ref2va_backend_runs_target_only_forward_and_backward():
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
    backend = _NativeTrainingBackend(transformer, mode="ref2va")
    video = torch.randn(1, 4, 1, 2, 2)
    audio = torch.randn(1, 2, 6, 1)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(3, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor([1, 0, 1])],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["ref2va"])],
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
    )
    loss = prediction.video.square().mean() + prediction.audio.square().mean()
    loss.backward()

    assert prediction.video.shape == video.shape
    assert prediction.audio.shape == audio.shape
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


def test_native_h3_ref2va_backend_rejects_text_only_conditioning_cache():
    transformer = SimpleNamespace(config=SimpleNamespace(in_channels=4, audio_in_channels=6, text_dim=8))
    backend = _NativeTrainingBackend(transformer, mode="ref2va")
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(2, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["ref2va"])],
    }

    with pytest.raises(ValueError, match="--task ref2va"):
        backend.predict_training(
            transformer,
            batch,
            torch.randn(1, 4, 1, 2, 2),
            torch.randn(1, 2, 6, 1),
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
    with pytest.raises(ValueError, match="exclude every"):
        joint_velocity_loss(
            prediction,
            inputs,
            video_mask=torch.zeros(1, 1, 1, 2, dtype=torch.bool),
            audio_mask=torch.zeros_like(audio_mask),
        )


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


def test_standard_bucket_manager_loads_h3_joint_cache_without_shared_schema_changes(tmp_path):
    latent_path = tmp_path / "sample_mmh3.safetensors"
    text_path = tmp_path / "sample_mmh3_te.safetensors"
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

    batch = BucketBatchManager({(64, 64): [item]}, batch_size=1)[0]

    assert batch["latents"].shape == (1, 24, 2, 2, 2)
    assert batch[H3_AUDIO_LATENTS_KEY].shape == (1, 2, 32, 3)
    assert batch[H3_AUDIO_LOSS_MASK_KEY].shape == (1, 3)
    assert isinstance(batch[H3_TEXT_HIDDEN_KEY], list)
    assert batch[H3_TEXT_HIDDEN_KEY][0].shape == (2, 5120)
    assert batch[H3_REFERENCE_KINDS_KEY][0].tolist() == [0, 2]
    assert batch[H3_REFERENCE_VIDEO_ROWS_KEY][0].shape == (1, 96)
    assert batch[H3_REFERENCE_AUDIO_ROWS_KEY][0].shape == (2, 32)
    assert batch[H3_KEYFRAME_VIDEO_ROWS_KEY][0].shape == (2, 96)


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(4, 4, bias=False)
        self.ff = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))
        self.adaln_proj = nn.Linear(4, 4, bias=False)
        self.norm_probe = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states):
        return hidden_states + self.attn(hidden_states) + self.ff(hidden_states)


class TinyH3Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([MiniMaxH3TransformerBlock()])

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

    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    transformer(torch.ones(1, 4)).sum().backward()
    adapter_grads = [parameter.grad for parameter in network.parameters()]
    assert any(gradient is not None and torch.isfinite(gradient).all() and bool(gradient.abs().sum()) for gradient in adapter_grads)
    assert all(parameter.grad is None for parameter in transformer.transformer_blocks[0].adaln_proj.parameters())


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


class _ScaleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))


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
    args = SimpleNamespace(h3_training_mode="ref2va", fp8_base=False, int8_convrot_base=False, h3_adaln_rank=None)
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
    }


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--compile", None, "compilation"),
        ("--flash_attn", None, "only --sdpa"),
        ("--sdpa", "--split_attn", "split attention"),
    ],
)
def test_h3_trainer_rejects_release_dependent_common_loading_modes(option, value, message):
    argv = [option] if value is None else [option, value]
    args = create_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


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


@pytest.mark.parametrize("bad_shift", ["0.0", "0.001", "1000.0"])
def test_h3_trainer_rejects_out_of_range_modality_shift(bad_shift):
    args = create_parser().parse_args(["--sdpa", "--h3_shift_video", bad_shift])

    with pytest.raises(ValueError, match="h3_shift_video"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


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
    assert args.h3_loss_balance == "modality"
    assert args.h3_guidance_distillation_scale is None
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
