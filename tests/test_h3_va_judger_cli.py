"""Staged process order and completed-round restart checks without models."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from musubi_tuner.minimax_h3_train_va_judger import _check_cached_judge, _run_staged


@pytest.mark.parametrize("stage", [None, "run"])
def test_public_training_script_defaults_to_cached_training(tmp_path, monkeypatch, stage):
    from musubi_tuner.minimax_h3_train_va_judger import create_parser

    argv = ["minimax_h3_train_va_judger.py"]
    for name in ("model", "text_encoder", "vae", "audio_vae", "prompts", "output_dir", "round_dir"):
        argv += [f"--{name}", str(tmp_path / name)]
    if stage is not None:
        argv += ["--stage", stage]
    observed = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        "musubi_tuner.minimax_h3_train_va_judger.main", lambda values: observed.append(create_parser().parse_args(values).stage)
    )
    runpy.run_path(str(Path(__file__).resolve().parents[1] / argv[0]), run_name="__main__")
    assert observed == [stage or "train"]


def test_round_processes_run_sequentially_and_skip_completed(tmp_path, monkeypatch):
    model = tmp_path / "weights"
    python = tmp_path / "python"
    for path in (model, python):
        path.touch()
    args = SimpleNamespace(
        reward_model=model,
        reward_python=python,
        output_dir=tmp_path / "out",
        resume=None,
        max_steps=2,
        reward_pair_batch_size=2,
        reward_device="cuda",
        reward_attn_implementation="sdpa",
        reward_max_frames=8,
        reward_video_max_pixels=300000,
        reward_max_new_tokens=1024,
        reward_do_sample=False,
        reward_temperature=0.5,
        reward_top_p=0.8,
        reward_top_k=16,
        reward_seed=123,
    )
    calls = []

    def invoke(command, check):
        assert check
        if "--round_manifest" in command:
            stage = "score"
            assert "--reward_source" not in command
            assert command[command.index("--pair_batch_size") + 1] == "2"
            assert command[command.index("--max_frames") + 1] == "8"
            assert command[command.index("--temperature") + 1] == "0.5"
            assert command[command.index("--top_p") + 1] == "0.8"
            assert command[command.index("--top_k") + 1] == "16"
            assert command[command.index("--seed") + 1] == "123"
            assert "--no-do_sample" in command
            directory = Path(command[command.index("--round_manifest") + 1]).parent
            assert (directory / "manifest.json").exists()
            (directory / "rewards.json").write_text("{}")
        else:
            stage = command[command.index("--stage") + 1]
            directory = Path(command[command.index("--round_dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            if stage == "generate":
                (directory / "manifest.json").write_text("{}")
                assert not (directory / "rewards.json").exists()
            else:
                assert (directory / "rewards.json").exists()
                (directory / "trained.json").write_text(
                    json.dumps(
                        {
                            "start_step": 0,
                            "step": 2,
                            "resume_path": str(tmp_path / "final.resume.pt"),
                        }
                    )
                )
        calls.append(stage)

    monkeypatch.setattr("musubi_tuner.minimax_h3_train_va_judger.subprocess.run", invoke)
    _run_staged(args, [])
    assert calls == ["generate", "score", "train"]
    calls.clear()
    _run_staged(args, [])
    assert calls == []


@pytest.mark.parametrize("equals_form", [False, True])
def test_staged_run_replaces_initial_lora_with_resume_after_first_round(tmp_path, monkeypatch, equals_form):
    model = tmp_path / "weights"
    python = tmp_path / "python"
    initial_lora = tmp_path / "initial.safetensors"
    for path in (model, python, initial_lora):
        path.touch()
    args = SimpleNamespace(
        reward_model=model,
        reward_python=python,
        output_dir=tmp_path / "out",
        resume=None,
        max_steps=2,
        reward_pair_batch_size=1,
        reward_device="cuda",
        reward_attn_implementation="sdpa",
        reward_max_frames=12,
        reward_video_max_pixels=602112,
        reward_max_new_tokens=2048,
        reward_do_sample=True,
        reward_temperature=0.75,
        reward_top_p=0.92,
        reward_top_k=32,
        reward_seed=42,
    )
    initial_args = [f"--initial_lora={initial_lora}"] if equals_form else ["--initial_lora", str(initial_lora)]
    generation_commands = []

    def invoke(command, check):
        assert check
        if "--round_manifest" in command:
            directory = Path(command[command.index("--round_manifest") + 1]).parent
            (directory / "rewards.json").write_text("{}")
            return
        stage = command[command.index("--stage") + 1]
        directory = Path(command[command.index("--round_dir") + 1])
        directory.mkdir(parents=True, exist_ok=True)
        if stage == "generate":
            generation_commands.append(command)
            (directory / "manifest.json").write_text("{}")
            return
        start_step = int(directory.name.removeprefix("round_"))
        (directory / "trained.json").write_text(
            json.dumps(
                {
                    "start_step": start_step,
                    "step": start_step + 1,
                    "resume_path": str(tmp_path / f"step-{start_step + 1}.resume.pt"),
                }
            )
        )

    monkeypatch.setattr("musubi_tuner.minimax_h3_train_va_judger.subprocess.run", invoke)
    _run_staged(args, initial_args)

    assert len(generation_commands) == 2
    first, second = generation_commands
    assert any(value == "--initial_lora" or value.startswith("--initial_lora=") for value in first)
    assert not any(value == "--initial_lora" or value.startswith("--initial_lora=") for value in second)
    assert second[second.index("--resume") + 1] == str(tmp_path / "step-1.resume.pt")


@pytest.mark.parametrize("change", [None, "legacy", "version", "hash", "implementation"])
def test_cached_rewards_are_bound_to_builtin_rubric(tmp_path, change):
    from musubi_tuner.minimax_h3_score_va_judger import (
        SCORING_IMPLEMENTATION_VERSION,
        SCORING_PROMPT_SHA256,
        SCORING_PROMPT_VERSION,
    )

    args = SimpleNamespace(
        reward_model=tmp_path / "weights",
        reward_max_new_tokens=2048,
        reward_device="cuda",
        reward_attn_implementation="sdpa",
        reward_max_frames=12,
        reward_video_max_pixels=602112,
        reward_pair_batch_size=1,
        reward_do_sample=True,
        reward_temperature=0.75,
        reward_top_p=0.92,
        reward_top_k=32,
        reward_seed=42,
    )
    judge = {
        "backend": "native",
        "reward_model": str(args.reward_model.resolve()),
        "scoring_prompt_version": SCORING_PROMPT_VERSION,
        "scoring_prompt_sha256": SCORING_PROMPT_SHA256,
        "implementation_version": SCORING_IMPLEMENTATION_VERSION,
        "args": {key.removeprefix("reward_"): value for key, value in vars(args).items() if key != "reward_model"},
    }
    if change == "legacy":
        del judge["scoring_prompt_version"]
        del judge["scoring_prompt_sha256"]
        judge["reward_source_sha256"] = "historical"
    elif change == "version":
        judge["scoring_prompt_version"] = "different"
    elif change == "hash":
        judge["scoring_prompt_sha256"] = "different"
    elif change == "implementation":
        judge["implementation_version"] = "old"
    path = tmp_path / "rewards.json"
    path.write_text(json.dumps({"judge": judge}))
    if change is None:
        _check_cached_judge(path, args)
    else:
        with pytest.raises(ValueError, match="preprocessing" if change == "implementation" else "scoring rubric"):
            _check_cached_judge(path, args)


@pytest.mark.parametrize(
    ("setting", "changed"),
    [
        ("pair_batch_size", 2),
        ("do_sample", False),
        ("temperature", 0.5),
        ("top_p", 0.8),
        ("top_k", 16),
        ("seed", 123),
    ],
)
def test_cached_rewards_are_bound_to_sampling_policy(tmp_path, setting, changed):
    from musubi_tuner.minimax_h3_score_va_judger import (
        SCORING_IMPLEMENTATION_VERSION,
        SCORING_PROMPT_SHA256,
        SCORING_PROMPT_VERSION,
    )

    args = SimpleNamespace(
        reward_model=tmp_path / "weights",
        reward_max_new_tokens=2048,
        reward_device="cuda",
        reward_attn_implementation="sdpa",
        reward_max_frames=12,
        reward_video_max_pixels=602112,
        reward_pair_batch_size=1,
        reward_do_sample=True,
        reward_temperature=0.75,
        reward_top_p=0.92,
        reward_top_k=32,
        reward_seed=42,
    )
    judge_args = {
        "pair_batch_size": 1,
        "max_new_tokens": 2048,
        "device": "cuda",
        "attn_implementation": "sdpa",
        "max_frames": 12,
        "video_max_pixels": 602112,
        "do_sample": True,
        "temperature": 0.75,
        "top_p": 0.92,
        "top_k": 32,
        "seed": 42,
    }
    judge_args[setting] = changed
    path = tmp_path / "rewards.json"
    path.write_text(
        json.dumps(
            {
                "judge": {
                    "backend": "native",
                    "reward_model": str(args.reward_model.resolve()),
                    "scoring_prompt_version": SCORING_PROMPT_VERSION,
                    "scoring_prompt_sha256": SCORING_PROMPT_SHA256,
                    "implementation_version": SCORING_IMPLEMENTATION_VERSION,
                    "args": judge_args,
                }
            }
        )
    )

    with pytest.raises(ValueError, match="different judge settings"):
        _check_cached_judge(path, args)
