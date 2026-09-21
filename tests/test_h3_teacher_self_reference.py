from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.cache import (
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_KEYFRAME_VIDEO_ROWS_KEY,
    H3_ONE_FRAME_TARGET_INDEX_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.integration import (
    H3_TEACHER_REFERENCE_AUDIO_KEY,
    H3_TEACHER_REFERENCE_VIDEO_KEY,
    _NativeTrainingBackend,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.packing import pack_audio_latents, patchify_video_latents
from musubi_tuner.minimax_h3.references import H3ReferenceKind
from musubi_tuner.minimax_h3.training import prepare_joint_noisy_inputs
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, _teacher_flow_loss
from musubi_tuner.networks import lora_minimax_h3


def _backend():
    return _NativeTrainingBackend(SimpleNamespace(), mode="ref2va")


def test_self_reference_packs_clean_target_and_detaches_it():
    video = torch.arange(16.0).reshape(1, 4, 1, 2, 2).requires_grad_()
    audio = torch.arange(12.0).reshape(1, 2, 2, 3).requires_grad_()
    backend = _backend()
    batch = {
        H3_TEACHER_REFERENCE_VIDEO_KEY: video,
        H3_TEACHER_REFERENCE_AUDIO_KEY: audio,
    }

    geometry, video_rows, audio_rows = backend._reference_cache(
        batch,
        patch_size=(1, 2, 2),
        video_width=16,
        audio_width=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(geometry) == 1
    assert geometry[0].kind == int(H3ReferenceKind.VIDEO)
    assert (geometry[0].num_latent_frames, geometry[0].latent_height, geometry[0].latent_width) == (1, 2, 2)
    assert geometry[0].num_audio_latents == 3
    torch.testing.assert_close(video_rows, patchify_video_latents(video.detach(), (1, 2, 2))[0])
    torch.testing.assert_close(audio_rows, pack_audio_latents(audio.detach())[0])
    assert not video_rows.requires_grad and not audio_rows.requires_grad
    assert backend._cached_reference_kinds(batch).tolist() == [int(H3ReferenceKind.VIDEO)]


def test_self_reference_requires_video_and_complete_reference():
    backend = _backend()
    video = torch.zeros(1, 4, 1, 2, 2)
    with pytest.raises(ValueError, match="clean video target"):
        backend._reference_cache(
            {H3_TEACHER_REFERENCE_VIDEO_KEY: torch.zeros(1, 4, 0, 2, 2)},
            patch_size=(1, 2, 2),
            video_width=16,
            audio_width=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="complete video/audio reference"):
        backend._reference_cache(
            {H3_TEACHER_REFERENCE_VIDEO_KEY: video},
            patch_size=(1, 2, 2),
            video_width=16,
            audio_width=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            reference_modality="video",
        )


def _plan(backend, batch, video, audio=None):
    model = SimpleNamespace(config=SimpleNamespace(in_channels=4, audio_in_channels=6, text_dim=8, patch_size=(1, 2, 2)))
    return backend._prepare_training_forward(model, batch, video, audio, torch.tensor([0.4]), torch.tensor([0.7]))


def _text_batch(task, tags):
    return {
        H3_TEXT_HIDDEN_KEY: [torch.zeros(len(tags), 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor(tags, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS[task])],
    }


def test_endpoint_teacher_has_condition_rows_but_t2va_student_does_not():
    video = torch.zeros(1, 4, 2, 2, 2)
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="fl2va")
    student = _plan(backend, _text_batch("t2va", [1, 1]), video)
    teacher_batch = _text_batch("fl2va", [1, 0, 0, 1])
    teacher_batch[H3_KEYFRAME_VIDEO_ROWS_KEY] = [torch.ones(2, 16)]
    teacher = _plan(backend, teacher_batch, video)

    assert student.kwargs["video_hidden_states"].shape[1] == 2
    assert teacher.kwargs["video_hidden_states"].shape[1] == 4


def test_self_clip_teacher_packs_video_and_audio_reference_rows():
    video = torch.zeros(1, 4, 2, 2, 2)
    audio = torch.zeros(1, 2, 6, 1)
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="ref2va")
    batch = _text_batch("ref2va", [1, 0, 1])
    batch[H3_TEACHER_REFERENCE_VIDEO_KEY] = torch.ones_like(video)
    batch[H3_TEACHER_REFERENCE_AUDIO_KEY] = torch.ones_like(audio)
    plan = _plan(backend, batch, video, audio)

    assert plan.kwargs["video_hidden_states"].shape[1] == 4
    assert plan.kwargs["audio_hidden_states"].shape[1] == 4


def test_one_frame_subject_reference_teacher_packs_cached_image_reference():
    video = torch.zeros(1, 4, 1, 2, 2)
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="ref2va")
    batch = _text_batch("ref2va", [1, 0, 1])
    batch.update(
        {
            H3_ONE_FRAME_TARGET_INDEX_KEY: [torch.tensor(24, dtype=torch.long)],
            H3_REFERENCE_KINDS_KEY: [torch.tensor([int(H3ReferenceKind.IMAGE)])],
            H3_REFERENCE_VIDEO_SHAPES_KEY: [torch.tensor([[1, 2, 2]])],
            H3_REFERENCE_AUDIO_LENGTHS_KEY: [torch.tensor([0])],
            H3_REFERENCE_VIDEO_ROWS_KEY: [torch.ones(1, 16)],
            H3_REFERENCE_AUDIO_ROWS_KEY: [torch.empty(0, 6)],
        }
    )
    plan = _plan(backend, batch, video)

    assert plan.kwargs["video_hidden_states"].shape[1] == 2


@pytest.mark.parametrize("conditions", ("first,last", "ref", "subject_ref"))
def test_teacher_forward_is_frozen_and_student_lora_receives_gradient(conditions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    transformer = MiniMaxH3Transformer(config).to(device).float().requires_grad_(False)
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, [], transformer)
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    network.to(device)
    backend = _NativeTrainingBackend(transformer, mode="fl2va")
    trainer = MiniMaxH3NetworkTrainer()
    trainer.backend = backend
    frames = 1 if conditions == "subject_ref" else 2
    video = torch.randn(1, 4, frames, 2, 2, device=device)
    audio = torch.randn(1, 2, 6, 1, device=device) if conditions == "ref" else None
    inputs = prepare_joint_noisy_inputs(
        video,
        audio,
        torch.randn_like(video),
        torch.randn_like(audio) if audio is not None else None,
        torch.tensor([0.6], device=device),
    )
    student = _text_batch("t2va", [1, 1])
    student[H3_TEXT_HIDDEN_KEY] = [torch.randn(2, 8, device=device)]
    student["mmh3_teacher_conditions"] = [torch.tensor({"first,last": 1, "ref": 2, "subject_ref": 3}[conditions])]
    teacher_tags = [1, 0, 0, 1] if conditions == "first,last" else [1, 0, 1]
    student["mmh3_teacher_hidden_states"] = [torch.randn(len(teacher_tags), 8, device=device)]
    student["mmh3_teacher_token_tags"] = [torch.tensor(teacher_tags)]
    if conditions == "first,last":
        student[H3_KEYFRAME_VIDEO_ROWS_KEY] = [torch.randn(2, 16, device=device)]
    elif conditions == "subject_ref":
        student.update(
            {
                H3_ONE_FRAME_TARGET_INDEX_KEY: [torch.tensor(24)],
                H3_REFERENCE_KINDS_KEY: [torch.tensor([int(H3ReferenceKind.IMAGE)])],
                H3_REFERENCE_VIDEO_SHAPES_KEY: [torch.tensor([[1, 2, 2]])],
                H3_REFERENCE_AUDIO_LENGTHS_KEY: [torch.tensor([0])],
                H3_REFERENCE_VIDEO_ROWS_KEY: [torch.randn(1, 16, device=device)],
                H3_REFERENCE_AUDIO_ROWS_KEY: [torch.empty(0, 6, device=device)],
            }
        )
    teacher_batch, teacher_mode = trainer._teacher_matching_batch(
        SimpleNamespace(h3_teacher_conditions=conditions), student, inputs, conditioned=True
    )

    network.set_enabled(False)
    backend.mode = teacher_mode
    with torch.no_grad():
        teacher = backend.predict_training(
            transformer,
            teacher_batch,
            inputs.video,
            inputs.audio,
            inputs.video_timestep,
            inputs.audio_timestep,
        )
    network.set_enabled(True)
    backend.mode = "fl2va"
    prediction = backend.predict_training(
        transformer,
        student,
        inputs.video,
        inputs.audio,
        inputs.video_timestep,
        inputs.audio_timestep,
    )
    loss = _teacher_flow_loss(prediction.video, teacher.video, mag_weight=1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert not teacher.video.requires_grad
    assert prediction.video.shape == teacher.video.shape == video.shape
    gradients = [module.lora_up.weight.grad for module in network.unet_loras]
    assert gradients and any(gradient is not None and bool(gradient.abs().sum()) for gradient in gradients)
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is None
