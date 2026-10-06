from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from musubi_tuner import minimax_h3_generate_va_judger as collector
from musubi_tuner.minimax_h3_score_va_judger import (
    SCORING_IMPLEMENTATION_VERSION,
    SCORING_PROMPT_SHA256,
    SCORING_PROMPT_VERSION,
)


def _argv(tmp_path):
    paths = {}
    for name in ("model", "text_encoder", "tokenizer", "vae", "audio_vae", "prompts"):
        path = tmp_path / name
        path.write_text("fixture")
        paths[name] = path
    reward_model = tmp_path / "reward-model"
    reward_model.mkdir()
    return [
        "--model",
        str(paths["model"]),
        "--text_encoder",
        str(paths["text_encoder"]),
        "--tokenizer",
        str(paths["tokenizer"]),
        "--vae",
        str(paths["vae"]),
        "--audio_vae",
        str(paths["audio_vae"]),
        "--prompts",
        str(paths["prompts"]),
        "--output_dir",
        str(tmp_path / "output"),
        "--round_dir",
        str(tmp_path / "round"),
        "--reward_model",
        str(reward_model),
        "--reward_python",
        sys.executable,
        "--round_groups",
        "2",
        "--group_size",
        "3",
        "--max_steps",
        "8",
    ]


def _reward_record(manifest, reward_model):
    return {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "judge": {
            "backend": "native",
            "reward_model": str(reward_model.resolve()),
            "scoring_prompt_version": SCORING_PROMPT_VERSION,
            "scoring_prompt_sha256": SCORING_PROMPT_SHA256,
            "implementation_version": SCORING_IMPLEMENTATION_VERSION,
            "args": {
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
            },
        },
    }


@pytest.fixture
def valid_config(monkeypatch):
    validated = []
    monkeypatch.setattr(collector, "config_from_args", lambda _args: SimpleNamespace(validate=lambda: validated.append(True)))
    monkeypatch.setattr(collector, "_validate_manifest", lambda _path, _config: None)
    monkeypatch.setattr(collector, "_validate_cached_reward_structure", lambda *_args: None)
    return validated


def test_collector_runs_generation_then_scoring_as_separate_children(tmp_path, monkeypatch, valid_config):
    argv = _argv(tmp_path)
    round_dir = tmp_path / "round"
    calls = []

    def run(command, check):
        assert check is True
        calls.append(command)
        round_dir.mkdir(exist_ok=True)
        if "minimax_h3_train_va_judger.py" in command[1]:
            assert command[-2:] == ["--stage", "generate"]
            assert "train" not in command
            (round_dir / "manifest.json").write_text('{"round":"fixture"}')
        else:
            manifest = round_dir / "manifest.json"
            reward_model = tmp_path / "reward-model"
            (round_dir / "rewards.json").write_text(json.dumps(_reward_record(manifest, reward_model)))

    monkeypatch.setattr(collector.subprocess, "run", run)
    collector.main(argv)

    assert valid_config == [True]
    assert len(calls) == 2
    assert calls[0][0] == sys.executable
    assert calls[1][0] == sys.executable
    assert "minimax_h3_score_va_judger.py" in calls[1][1]


def test_collector_forwards_sampling_policy_to_scorer(tmp_path, monkeypatch, valid_config):
    argv = [
        *_argv(tmp_path),
        "--no-reward_do_sample",
        "--reward_temperature",
        "0.5",
        "--reward_top_p",
        "0.8",
        "--reward_top_k",
        "16",
        "--reward_seed",
        "123",
    ]
    round_dir = tmp_path / "round"

    def run(command, check):
        assert check is True
        round_dir.mkdir(exist_ok=True)
        if "minimax_h3_train_va_judger.py" in command[1]:
            (round_dir / "manifest.json").write_text('{"round":"fixture"}')
            return
        assert "--no-do_sample" in command
        assert command[command.index("--temperature") + 1] == "0.5"
        assert command[command.index("--top_p") + 1] == "0.8"
        assert command[command.index("--top_k") + 1] == "16"
        assert command[command.index("--seed") + 1] == "123"
        manifest = round_dir / "manifest.json"
        record = _reward_record(manifest, tmp_path / "reward-model")
        record["judge"]["args"].update(
            do_sample=False,
            temperature=0.5,
            top_p=0.8,
            top_k=16,
            seed=123,
        )
        (round_dir / "rewards.json").write_text(json.dumps(record))

    monkeypatch.setattr(collector.subprocess, "run", run)
    collector.main(argv)


def test_collector_reuses_complete_valid_round_without_children(tmp_path, monkeypatch, valid_config):
    argv = _argv(tmp_path)
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    manifest = round_dir / "manifest.json"
    manifest.write_text('{"round":"fixture"}')
    (round_dir / "rewards.json").write_text(json.dumps(_reward_record(manifest, tmp_path / "reward-model")))
    monkeypatch.setattr(collector.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("child launched"))

    collector.main(argv)

    assert valid_config == [True]


def test_collector_resumes_manifest_only_round_at_scoring(tmp_path, monkeypatch, valid_config):
    argv = _argv(tmp_path)
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    manifest = round_dir / "manifest.json"
    manifest.write_text('{"round":"fixture"}')
    calls = []

    def run(command, check):
        assert check is True
        calls.append(command)
        assert "minimax_h3_score_va_judger.py" in command[1]
        (round_dir / "rewards.json").write_text(json.dumps(_reward_record(manifest, tmp_path / "reward-model")))

    monkeypatch.setattr(collector.subprocess, "run", run)
    collector.main(argv)

    assert len(calls) == 1


def test_collector_rejects_cached_rewards_for_changed_manifest(tmp_path, monkeypatch, valid_config):
    argv = _argv(tmp_path)
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    manifest = round_dir / "manifest.json"
    manifest.write_text("before")
    (round_dir / "rewards.json").write_text(json.dumps(_reward_record(manifest, tmp_path / "reward-model")))
    manifest.write_text("after")
    monkeypatch.setattr(collector.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("child launched"))

    with pytest.raises(ValueError, match="exact manifest"):
        collector.main(argv)


def test_collector_checks_reward_model_before_generation(tmp_path, monkeypatch):
    argv = _argv(tmp_path)
    reward_index = argv.index("--reward_model") + 1
    argv[reward_index] = str(tmp_path / "missing-reward")
    monkeypatch.setattr(collector.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("child launched"))

    with pytest.raises(SystemExit):
        collector.main(argv)


def test_collector_rejects_non_generate_stage(tmp_path):
    with pytest.raises(SystemExit):
        collector.main([*_argv(tmp_path), "--stage", "train"])


def test_manifest_validation_rejects_changed_semantics(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("fixture")
    monkeypatch.setattr(collector, "_load_manifest", lambda _path: ({"semantic_fingerprint": "old"}, b"fixture"))
    monkeypatch.setattr(collector, "_semantic_config", lambda _config: {"current": True})
    monkeypatch.setattr(collector, "_config_fingerprint", lambda _semantic: "new")
    monkeypatch.setattr(collector, "_verify", lambda *_args: pytest.fail("artifacts checked after bad semantics"))

    with pytest.raises(ValueError, match="different generation/training semantics"):
        collector._validate_manifest(manifest_path, object())


def test_manifest_validation_checks_all_bound_artifacts(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("fixture")
    manifest = {"semantic_fingerprint": "same"}
    observed = []
    monkeypatch.setattr(collector, "_load_manifest", lambda _path: (manifest, b"fixture"))
    monkeypatch.setattr(collector, "_semantic_config", lambda _config: {})
    monkeypatch.setattr(collector, "_config_fingerprint", lambda _semantic: "same")
    monkeypatch.setattr(collector, "_verify", lambda value, latents: observed.append((value, latents)))
    monkeypatch.setattr(collector, "_validate_prompt_fingerprint", lambda *_args: None)

    collector._validate_manifest(manifest_path, object())

    assert observed == [(manifest, True)]


def _structural_contract():
    candidates = [{"media_sha256": "a" * 64}, {"media_sha256": "b" * 64}]
    manifest = {
        "round_id": "round-000000",
        "behavior_sha256": "behavior",
        "semantic_fingerprint": "semantic",
        "start_step": 0,
        "prompt_cursor": 0,
        "group_count": 1,
        "groups": [{"group_id": 0, "prompt": "prompt", "candidates": candidates}],
    }
    rewards = {
        "version": 1,
        **{
            key: manifest[key]
            for key in ("round_id", "behavior_sha256", "semantic_fingerprint", "start_step", "prompt_cursor", "group_count")
        },
        "groups": [
            {"group_id": 0, "prompt": "prompt", "candidate_hashes": ["a" * 64, "b" * 64], "raw_scores": [[8.0] * 5, [2.0] * 5]}
        ],
    }
    return manifest, rewards


def test_cached_reward_structure_rejects_missing_groups(tmp_path):
    manifest, rewards = _structural_contract()
    del rewards["groups"]
    with pytest.raises(ValueError, match="groups are incomplete"):
        collector._validate_cached_reward_structure(manifest, rewards, tmp_path / "rewards.json")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0, 11.0, True])
def test_cached_reward_structure_rejects_corrupt_scores(tmp_path, bad):
    manifest, rewards = _structural_contract()
    rewards["groups"][0]["raw_scores"][0][0] = bad
    with pytest.raises(ValueError, match="finite values"):
        collector._validate_cached_reward_structure(manifest, rewards, tmp_path / "rewards.json")


def test_manifest_validation_rejects_prompt_file_changed_in_place(tmp_path):
    import torch

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("new prompt\n", encoding="utf-8")
    resume = tmp_path / "behavior.resume.pt"
    torch.save({"prompt_fingerprint": hashlib.sha256(b"old prompt").hexdigest()}, resume)
    with pytest.raises(ValueError, match="different prompt contents"):
        collector._validate_prompt_fingerprint({"resume_path": str(resume)}, SimpleNamespace(prompts=prompts))
