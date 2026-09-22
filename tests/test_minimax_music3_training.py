import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import wave

import torch
import pytest

from musubi_tuner.minimax_music3_train_network import (
    masked_cosine_distance,
    masked_mse_loss,
    sample_music3_flow_times,
    resolve_latest_music3_state,
    setup_music3_parser,
    twinflow_rcgm_target,
)
from musubi_tuner.minimax_music3_train_ar_lora import (
    NextLatPredictor,
    _distillation_cross_entropy,
    _prune_checkpoints,
    _resolve_resume_checkpoint,
    _xm_select_loss,
)
from musubi_tuner.minimax_music3_pipeline import build_commands
from musubi_tuner.minimax_music3_dataset_report import audit_audio_dataset
from musubi_tuner.dataset.audio_dataset import AudioDataset
from musubi_tuner.minimax_music3.model import MiniMaxMusic3DiT


def test_music3_timestep_sampling_parser_defaults_to_sigmoid():
    args = setup_music3_parser(argparse.ArgumentParser()).parse_args([])
    assert args.music3_timestep_sampling == "sigmoid"


def test_music3_timestep_sampling_parser_accepts_new_distributions():
    for value in ("uniform", "cubic"):
        args = setup_music3_parser(argparse.ArgumentParser()).parse_args(
            ["--music3_timestep_sampling", value]
        )
        assert args.music3_timestep_sampling == value


def test_music3_parser_exposes_training_quality_and_checkpoint_controls():
    args = setup_music3_parser(argparse.ArgumentParser()).parse_args(
        [
            "--music3_input_perturbation", "0.1",
            "--music3_conditioning_dropout", "0.15",
            "--music3_gradient_checkpointing_interval", "2",
            "--music3_gradient_checkpointing_segment_stride", "6",
        ]
    )
    assert args.music3_input_perturbation == 0.1
    assert args.music3_conditioning_dropout == 0.15
    assert args.music3_gradient_checkpointing_interval == 2
    assert args.music3_gradient_checkpointing_segment_stride == 6


def test_music3_parser_exposes_self_flow_controls():
    args = setup_music3_parser(argparse.ArgumentParser()).parse_args(
        ["--music3_self_flow", "--music3_self_flow_weight", "0.75", "--music3_self_flow_mask_ratio", "0.4"]
    )
    assert args.music3_self_flow
    assert args.music3_self_flow_weight == 0.75
    assert args.music3_self_flow_mask_ratio == 0.4


def test_mixflow_keeps_model_and_interpolation_times_distinct():
    torch.manual_seed(3)
    model_time, interpolation_time = sample_music3_flow_times(
        32, torch.device("cpu"), "sigmoid", mixflow=True, mixflow_gamma=0.8
    )
    assert torch.all(interpolation_time >= model_time)
    assert torch.all(interpolation_time <= 1)
    assert not torch.equal(interpolation_time, model_time)


def test_music3_reference_and_signed_time_conditioning_forward():
    model = MiniMaxMusic3DiT(
        channels=2,
        condition_input_dim=4,
        condition_layers=2,
        condition_dim=4,
        dim=8,
        layers=1,
        head_dim=4,
        ff_dim=16,
        rotary_dim=2,
    )
    model.enable_flowmap_time_conditioning(gate=0.25, delta_type="r")
    model.enable_time_sign_conditioning()
    x = torch.randn(2, 2, 5)
    context = torch.randn(2, 3, 8)
    output = model(
        x,
        torch.tensor([0.2, 0.7]),
        context,
        reference_timestep=torch.tensor([0.1, 0.4]),
        timestep_sign=torch.tensor([1.0, -1.0]),
    )
    assert output.shape == x.shape


def test_music3_tokenwise_times_and_hidden_state_capture():
    model = MiniMaxMusic3DiT(
        channels=2,
        condition_input_dim=4,
        condition_layers=2,
        condition_dim=4,
        dim=8,
        layers=2,
        head_dim=4,
        ff_dim=16,
        rotary_dim=2,
    )
    x = torch.randn(2, 2, 5)
    context = torch.randn(2, 3, 8)
    prediction, hidden = model(x, torch.rand(2, 5), context, hidden_state_layer=0)
    assert prediction.shape == x.shape
    assert hidden.shape == (2, 5, 8)


def test_masked_cosine_distance_ignores_padding():
    prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    valid = torch.tensor([[[True, False]]])
    assert torch.isclose(masked_cosine_distance(prediction, target, valid), torch.tensor(0.0))


def test_twinflow_recursive_target_is_finite_and_detached():
    base = torch.randn(2, 3, 5, requires_grad=True)
    target = torch.randn_like(base)
    noisy = torch.randn_like(base)
    calls = []

    def teacher(current, previous, following):
        calls.append((previous.clone(), following.clone()))
        return torch.full_like(current, 0.25)

    result = twinflow_rcgm_target(
        base,
        target,
        noisy,
        torch.tensor([0.8, 0.6]),
        torch.tensor([0.2, 0.1]),
        teacher,
        estimate_order=3,
        target_clamp=1.0,
    )
    assert len(calls) == 3
    assert result.shape == base.shape
    assert torch.isfinite(result).all()
    assert not result.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_music3_cuda_forward_backward_with_padding_and_time_conditioning():
    model = MiniMaxMusic3DiT(
        channels=2,
        condition_input_dim=4,
        condition_layers=2,
        condition_dim=4,
        dim=16,
        layers=2,
        head_dim=4,
        ff_dim=32,
        rotary_dim=2,
    ).to(device="cuda", dtype=torch.bfloat16)
    model.enable_flowmap_time_conditioning(gate=0.25, delta_type="t-r")
    model.enable_time_sign_conditioning()
    model.enable_gradient_checkpointing()
    model.train()
    x = torch.randn(2, 2, 7, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    context = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    valid = torch.tensor(
        [[[True, True, True, True, True, True, True]], [[True, True, True, True, False, False, False]]],
        device="cuda",
    )
    output = model(
        x,
        torch.tensor([0.3, 0.8], device="cuda", dtype=torch.bfloat16),
        context,
        valid_mask=valid,
        reference_timestep=torch.tensor([0.1, 0.5], device="cuda", dtype=torch.bfloat16),
        timestep_sign=torch.tensor([1.0, -1.0], device="cuda"),
    )
    loss = masked_mse_loss(output, torch.zeros_like(output), valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_masked_mse_ignores_padded_audio_frames():
    prediction = torch.tensor([[[1.0, 2.0, 100.0]], [[3.0, 100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[True, True, False]], [[True, False, False]]])

    loss = masked_mse_loss(prediction, target, mask)

    assert torch.isclose(loss, torch.tensor((1.0 + 4.0 + 9.0) / 3.0))


def test_ar_latest_checkpoint_resolution_and_retention(tmp_path):
    for step in (10, 20, 30):
        checkpoint = tmp_path / f"step-{step}"
        checkpoint.mkdir()
        (checkpoint / "training_state.pt").touch()

    assert _resolve_resume_checkpoint(tmp_path, "latest") == tmp_path / "step-30"

    _prune_checkpoints(tmp_path, 2)
    assert sorted(path.name for path in tmp_path.glob("step-*")) == ["step-20", "step-30"]


def test_ar_regularization_distillation_prefers_teacher_distribution():
    teacher = torch.tensor([[4.0, 1.0, -2.0]])
    matching = _distillation_cross_entropy(teacher, teacher, top_k=2)
    mismatching = _distillation_cross_entropy(torch.tensor([[-2.0, 1.0, 4.0]]), teacher, top_k=2)
    assert matching < mismatching


def test_xm_selects_lowest_block_balanced_candidate():
    token_losses = torch.tensor([[1.0, 1.0, 9.0, 9.0], [2.0, 2.0, 2.0, 2.0]])
    loss, winner = _xm_select_loss(token_losses, block_size=2)
    assert winner.item() == 1
    assert loss.item() == 2.0


def test_nextlat_predictor_starts_as_zero_residual():
    predictor = NextLatPredictor(hidden_size=4, block_index=2)
    output = predictor(torch.randn(2, 3, 4))
    assert torch.count_nonzero(output) == 0


def test_dit_latest_state_resolution_uses_newest_complete_state(tmp_path):
    older = tmp_path / "music-step00000010-state"
    newer = tmp_path / "music-step00000020-state"
    empty = tmp_path / "music-step00000030-state"
    older.mkdir()
    newer.mkdir()
    empty.mkdir()
    (older / "optimizer.bin").touch()
    (newer / "optimizer.bin").touch()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert resolve_latest_music3_state(tmp_path, "music") == str(newer)


def test_raw_audio_pipeline_builds_cache_then_accelerate_commands():
    args = SimpleNamespace(
        dataset_config=Path("dataset.toml"),
        dav_encoder=Path("dav.pth"),
        ar_model="MiniMaxAI/MiniMax-Music3",
        cache_dtype="float32",
        seed=7,
        skip_existing=True,
        cache_only=False,
        num_processes=2,
        num_cpu_threads_per_process=1,
        dit=Path("dit.safetensors"),
        vae=Path("vae.safetensors"),
        output_dir=Path("output"),
        output_name="adapter",
        skip_dataset_report=False,
        strict_dataset=True,
        dataset_report=Path("report.json"),
    )

    commands = build_commands(args, ["--max_train_steps", "10"])

    assert len(commands) == 4
    assert "musubi_tuner.minimax_music3_dataset_report" in commands[0]
    assert "--strict" in commands[0]
    assert "musubi_tuner.minimax_music3_cache_latents" in commands[1]
    assert "musubi_tuner.minimax_music3_cache_text_encoder_outputs" in commands[2]
    assert "accelerate.commands.launch" in commands[3]
    assert commands[3][-2:] == ["--max_train_steps", "10"]


def _write_silent_wav(path: Path, frames: int = 4410):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(b"\0" * frames * 4)


def test_minimax_dataset_report_finds_missing_metadata_and_duplicate_cache_names(tmp_path):
    first = tmp_path / "a" / "song.wav"
    second = tmp_path / "b" / "song.wav"
    _write_silent_wav(first)
    _write_silent_wav(second)
    first.with_suffix(".txt").write_text("ambient test", encoding="utf-8")
    dataset = AudioDataset(audio_directory=str(tmp_path), cache_directory=str(tmp_path / "cache"))

    report = audit_audio_dataset(dataset)

    assert report["total"] == 2
    assert report["readable"] == 2
    assert len(report["missing_captions"]) == 1
    assert len(report["missing_lyrics"]) == 2
    assert len(report["duplicate_cache_names"]) == 1
    assert report["fatal_issue_count"] == 1
