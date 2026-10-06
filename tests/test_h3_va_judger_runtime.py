"""Runtime checks without loading H3 or a reward checkpoint."""

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.va_judger_runtime import (
    _BackwardNativeGenerator,
    VAJudgerRuntime,
    VAJudgerTrainConfig,
    _config_fingerprint,
    _file_sha256,
    _policy_sha256,
    _semantic_config,
    load_prompts,
)


def test_prompt_loader_rejects_missing_prompt(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"caption":"unrelated"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_prompts(path)


def test_prompt_loader_preserves_order(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt":"first"}\n"second"\n', encoding="utf-8")
    assert load_prompts(path) == ["first", "second"]


def test_resume_fingerprint_allows_only_operational_changes(tmp_path):
    config = VAJudgerTrainConfig(
        model=tmp_path / "model",
        text_encoder=tmp_path / "text",
        tokenizer=tmp_path / "tokenizer",
        video_vae=tmp_path / "video",
        audio_vae=tmp_path / "audio",
        prompts=tmp_path / "prompts",
        output_dir=tmp_path / "out",
        reward_endpoint="http://localhost:18080",
    )
    fingerprint = _config_fingerprint(_semantic_config(config))
    extended = replace(
        config,
        max_steps=200,
        output_dir=tmp_path / "new",
        save_every=20,
        device="cuda:1",
        blocks_to_swap=24,
        block_swap_h2d_only=True,
        use_pinned_memory_for_block_swap=True,
    )
    assert fingerprint == _config_fingerprint(_semantic_config(extended))
    for changed in (
        replace(config, beta_mix=0.5),
        replace(config, inference_steps=30),
        replace(config, learning_rate=1e-4),
        replace(config, text_encoder_quantization="none"),
        replace(config, token_refiner=False),
        replace(config, audio_vae=tmp_path / "other_audio"),
    ):
        assert fingerprint != _config_fingerprint(_semantic_config(changed))


def test_swap_config_validation_rejects_invalid_or_orphan_options(tmp_path):
    config = VAJudgerTrainConfig(
        model=tmp_path,
        text_encoder=tmp_path,
        tokenizer=tmp_path,
        video_vae=tmp_path,
        audio_vae=tmp_path,
        prompts=tmp_path,
        output_dir=tmp_path / "out",
        reward_endpoint="http://localhost",
    )
    for changed, message in (
        (replace(config, blocks_to_swap=-1), "blocks_to_swap"),
        (replace(config, blocks_to_swap=49), "blocks_to_swap"),
        (replace(config, block_swap_h2d_only=True), "requires blocks_to_swap"),
        (replace(config, use_pinned_memory_for_block_swap=True), "requires blocks_to_swap"),
    ):
        with pytest.raises(ValueError, match=message):
            changed.validate()


class ToyTransformer:
    def __init__(self):
        self.blocks = [object()]
        self.offloader = SimpleNamespace(finish_truncated_backward=lambda blocks: self.events.append(("finish", blocks)))
        self.events = []

    def switch_block_swap_for_inference(self):
        self.events.append("inference")

    def switch_block_swap_for_training(self):
        self.events.append("training")


def test_swap_lifecycle_uses_forward_only_for_frozen_passes_and_finishes_backward():
    runtime = object.__new__(VAJudgerRuntime)
    runtime.config = SimpleNamespace(blocks_to_swap=8)
    runtime.transformer = ToyTransformer()
    runtime._set_block_swap_training(False)
    runtime._set_block_swap_training(False)
    runtime._set_block_swap_training(True)
    runtime._finish_block_swap_backward()
    assert runtime.transformer.events == ["inference", "inference", "training", ("finish", runtime.transformer.blocks)]


def test_backward_loader_delegates_ordinary_runtime_without_swap(monkeypatch):
    generator = object.__new__(_BackwardNativeGenerator)
    generator.blocks_to_swap = 0
    sentinel = (object(), [])
    monkeypatch.setattr("musubi_tuner.minimax_h3.integration._NativeGenerator._load_transformer", lambda self: sentinel)
    assert generator._load_transformer() is sentinel


def test_old_resume_semantics_can_normalize_new_operational_swap_keys(tmp_path):
    config = VAJudgerTrainConfig(
        model=tmp_path / "model",
        text_encoder=tmp_path / "text",
        tokenizer=tmp_path / "tokenizer",
        video_vae=tmp_path / "video",
        audio_vae=tmp_path / "audio",
        prompts=tmp_path / "prompts",
        output_dir=tmp_path / "out",
        reward_endpoint="http://localhost:18080",
    )
    old_semantic = _semantic_config(config) | {
        "blocks_to_swap": 0,
        "block_swap_h2d_only": False,
        "use_pinned_memory_for_block_swap": False,
    }
    old_fingerprint = _config_fingerprint(old_semantic)
    assert old_fingerprint == _config_fingerprint(old_semantic)
    normalized = {
        key: value
        for key, value in old_semantic.items()
        if key not in {"blocks_to_swap", "block_swap_h2d_only", "use_pinned_memory_for_block_swap"}
    }
    assert _config_fingerprint(normalized) == _config_fingerprint(_semantic_config(replace(config, blocks_to_swap=24)))


class ToyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.3]))
        self.enabled = True

    def is_enabled(self):
        return self.enabled

    def set_enabled(self, enabled):
        self.enabled = enabled


@pytest.mark.parametrize("duration", range(1, 16))
def test_rollout_uses_h3_temporal_grid(duration):
    runtime = object.__new__(VAJudgerRuntime)
    runtime.config = SimpleNamespace(duration=duration)
    count = runtime._frame_count()
    assert count >= duration * 24
    assert count % 17 == 5
    if duration == 5:
        assert count == 124


def test_policy_restores_live_weights_and_enabled_after_error():
    runtime = object.__new__(VAJudgerRuntime)
    runtime.network = ToyAdapter()
    with pytest.raises(RuntimeError, match="intentional"):
        with runtime._policy({"weight": torch.tensor([0.7])}, enabled=False):
            assert runtime.network.weight.item() == pytest.approx(0.7)
            assert not runtime.network.enabled
            raise RuntimeError("intentional")
    assert runtime.network.weight.item() == pytest.approx(0.3)
    assert runtime.network.enabled


def test_policy_hash_supports_scalar_tensors():
    assert _policy_sha256({"alpha": torch.tensor(1.0, dtype=torch.bfloat16)}) == _policy_sha256(
        {"alpha": torch.tensor(1.0, dtype=torch.bfloat16)}
    )


def test_group_backwards_finish_before_policy_weights_are_changed(tmp_path):
    runtime = object.__new__(VAJudgerRuntime)
    runtime.network = ToyAdapter()
    runtime.optimizer = torch.optim.SGD(runtime.network.parameters(), lr=0.01)
    runtime.config = SimpleNamespace(max_grad_norm=1.0, output_dir=tmp_path)
    runtime.step = 0
    runtime.prompt_cursor = 0
    runtime.old_policy = {"weight": runtime.network.weight.detach().clone()}
    runtime.generate_group = lambda prompt: [
        SimpleNamespace(media_path=Path("a.mp4"), seed=42),
        SimpleNamespace(media_path=Path("b.mp4"), seed=43),
    ]
    runtime.reward = SimpleNamespace(score_group=lambda prompt, paths: torch.tensor([[8.0] * 5, [2.0] * 5]))

    def candidate_loss(candidate, advantage):
        # Replaying old weights changes the parameter version counter. A live
        # graph from an earlier candidate must already have been consumed.
        with torch.no_grad(), runtime._policy(runtime.old_policy):
            pass
        return runtime.network.weight.square().sum()

    runtime._candidate_loss = candidate_loss
    before = runtime.network.weight.detach().clone()
    metrics = runtime.train_step("prompt")
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert runtime.step == 1 and runtime.prompt_cursor == 1
    assert not torch.equal(before, runtime.network.weight)
    torch.testing.assert_close(runtime.old_policy["weight"], runtime.network.weight.detach())
    audit = json.loads((tmp_path / "step_000000.reward.json").read_text())
    assert audit["prompt"] == "prompt"
    assert len(audit["raw_scores"]) == 2


def _write_cached_round(tmp_path, runtime, group_count=2):
    from musubi_tuner.minimax_h3_score_va_judger import (
        SCORING_IMPLEMENTATION_VERSION,
        SCORING_PROMPT_SHA256,
        SCORING_PROMPT_VERSION,
    )

    round_dir = tmp_path / "round"
    round_dir.mkdir()
    groups = []
    reward_groups = []
    for group_id in range(group_count):
        prompt = f"prompt-{group_id}"
        entries = []
        hashes = []
        for index in range(2):
            seed = runtime.config.seed + group_id * runtime.config.group_size + index
            media = round_dir / f"{group_id}-{index}.mp4"
            latent = round_dir / f"{group_id}-{index}.latents.pt"
            media.write_bytes(f"media-{group_id}-{index}".encode())
            torch.save({"video": torch.ones(1), "audio": torch.ones(1), "prompt": prompt, "seed": seed}, latent)
            media_hash = _file_sha256(media)
            hashes.append(media_hash)
            entries.append(
                {
                    "path": str(media.resolve()),
                    "latent_path": str(latent.resolve()),
                    "seed": seed,
                    "media_sha256": media_hash,
                    "latent_sha256": _file_sha256(latent),
                }
            )
        groups.append({"group_id": group_id, "prompt": prompt, "candidates": entries})
        reward_groups.append(
            {"group_id": group_id, "prompt": prompt, "candidate_hashes": hashes, "raw_scores": [[8.0] * 5, [2.0] * 5]}
        )
    resume = round_dir / "behavior.resume.pt"
    resume.write_bytes(b"checkpoint")
    manifest = {
        "version": 1,
        "round_id": "round-000000",
        "start_step": 0,
        "prompt_cursor": 0,
        "group_count": group_count,
        "behavior_sha256": _policy_sha256(runtime.old_policy),
        "semantic_fingerprint": runtime.semantic_fingerprint,
        "resume_path": str(resume.resolve()),
        "resume_sha256": _file_sha256(resume),
        "groups": groups,
    }
    manifest_path = round_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    rewards = {
        "version": 1,
        "round_id": manifest["round_id"],
        "behavior_sha256": manifest["behavior_sha256"],
        "semantic_fingerprint": manifest["semantic_fingerprint"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "groups": reward_groups,
        "judge": {
            "backend": "native",
            "reward_model": "toy",
            "scoring_prompt_version": SCORING_PROMPT_VERSION,
            "scoring_prompt_sha256": SCORING_PROMPT_SHA256,
            "implementation_version": SCORING_IMPLEMENTATION_VERSION,
        },
    }
    rewards_path = round_dir / "rewards.json"
    rewards_path.write_text(json.dumps(rewards), encoding="utf-8")
    return manifest_path, rewards_path


def test_cached_round_rejects_legacy_reward_contract_before_update(tmp_path):
    runtime = object.__new__(VAJudgerRuntime)
    runtime.network = ToyAdapter()
    runtime.old_policy = {"weight": runtime.network.weight.detach().clone()}
    runtime.semantic_fingerprint = "semantic"
    runtime.config = SimpleNamespace(group_size=2, seed=42, max_steps=10)
    runtime.prompts = ["prompt-0"]
    runtime.step = runtime.prompt_cursor = 0
    called = []
    runtime.update_group = lambda *args, **kwargs: called.append(True)
    manifest, rewards = _write_cached_round(tmp_path, runtime, group_count=1)
    payload = json.loads(rewards.read_text())
    del payload["judge"]["implementation_version"]
    rewards.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="native scoring contract"):
        runtime.train_round(manifest, rewards)
    assert called == []


def test_cached_round_keeps_behavior_frozen_until_all_groups_finish(tmp_path):
    runtime = object.__new__(VAJudgerRuntime)
    runtime.network = ToyAdapter()
    runtime.old_policy = {"weight": runtime.network.weight.detach().clone()}
    runtime.semantic_fingerprint = "semantic"
    runtime.config = SimpleNamespace(group_size=2, seed=42, max_steps=10)
    runtime.prompts = ["prompt-0", "prompt-1"]
    runtime.step = runtime.prompt_cursor = 0
    observed = []

    def update(candidates, scores, refresh_old=True):
        observed.append((_policy_sha256(runtime.old_policy), refresh_old, candidates[0].prompt))
        with torch.no_grad():
            runtime.network.weight.add_(1)
        runtime.step += 1
        runtime.prompt_cursor += 1
        return {"loss": float(scores.mean())}

    runtime.update_group = update
    manifest, rewards = _write_cached_round(tmp_path, runtime)
    metrics = runtime.train_round(manifest, rewards)
    assert len(metrics) == 2
    assert [item[1] for item in observed] == [False, False]
    assert observed[0][0] == observed[1][0]
    torch.testing.assert_close(runtime.old_policy["weight"], runtime.network.weight.detach())


def test_cached_round_rejects_changed_latent_before_first_update(tmp_path):
    runtime = object.__new__(VAJudgerRuntime)
    runtime.network = ToyAdapter()
    runtime.old_policy = {"weight": runtime.network.weight.detach().clone()}
    runtime.semantic_fingerprint = "semantic"
    runtime.config = SimpleNamespace(group_size=2, seed=42, max_steps=10)
    runtime.prompts = ["prompt-0"]
    runtime.step = runtime.prompt_cursor = 0
    called = []
    runtime.update_group = lambda *args, **kwargs: called.append(True)
    manifest_path, rewards_path = _write_cached_round(tmp_path, runtime, group_count=1)
    manifest = json.loads(manifest_path.read_text())
    Path(manifest["groups"][0]["candidates"][0]["latent_path"]).write_bytes(b"altered")
    with pytest.raises(ValueError, match="integrity"):
        runtime.train_round(manifest_path, rewards_path)
    assert called == []
    assert not (manifest_path.parent / "training.json").exists()


def test_offline_scorer_artifact_is_accepted_by_cached_training(tmp_path):
    from musubi_tuner.minimax_h3_score_va_judger import SCORING_PROMPT, create_parser, score

    runtime = object.__new__(VAJudgerRuntime)
    runtime.network = ToyAdapter()
    runtime.old_policy = {"weight": runtime.network.weight.detach().clone()}
    runtime.semantic_fingerprint = "semantic"
    runtime.config = SimpleNamespace(group_size=2, seed=42, max_steps=10)
    runtime.prompts = ["prompt-0", "prompt-1"]
    runtime.step = runtime.prompt_cursor = 0
    runtime.update_group = lambda candidates, scores, refresh_old: {"loss": float(scores.mean())}
    manifest, rewards = _write_cached_round(tmp_path, runtime)
    rewards.unlink()
    model = tmp_path / "judge-model"
    model.mkdir()

    class FakeNativeJudge:
        def __init__(self, args, prompt):
            assert prompt == SCORING_PROMPT

        def predict(self, pairs):
            return [{"id": pair["id"], "dimension_scores": {key: {"1": 8.0, "2": 2.0} for key in "ABCDE"}} for pair in pairs]

        def close(self):
            pass

    args = create_parser().parse_args(
        [
            "--round_manifest",
            str(manifest),
            "--reward_model",
            str(model),
        ]
    )
    scored = score(args, judge_factory=FakeNativeJudge)
    assert len(runtime.train_round(manifest, scored)) == 2


def test_generation_publishes_only_complete_round_and_retry_uses_fresh_workdir(tmp_path):
    runtime = object.__new__(VAJudgerRuntime)
    runtime.config = SimpleNamespace(max_steps=2)
    runtime.step = runtime.prompt_cursor = 0
    runtime.prompts = ["prompt"]
    runtime.old_policy = {"alpha": torch.tensor(4.0)}
    runtime.semantic_fingerprint = "semantic"
    runtime._save_resume = lambda path: path.write_bytes(b"checkpoint")
    fail = [True]

    def generate(prompt, virtual_step, output_dir):
        media = output_dir / "candidate.mp4"
        media.write_bytes(b"media")
        media.with_suffix(".latents.pt").write_bytes(b"latents")
        if fail[0]:
            fail[0] = False
            raise RuntimeError("interrupted")
        from musubi_tuner.minimax_h3.va_judger_runtime import RolloutCandidate

        return [RolloutCandidate(prompt, 42, torch.ones(1), torch.ones(1), media)]

    runtime.generate_group = generate
    destination = tmp_path / "round"
    with pytest.raises(RuntimeError, match="interrupted"):
        runtime.generate_round(destination, 1)
    assert not destination.exists()
    manifest_path = runtime.generate_round(destination, 1)
    manifest = json.loads(manifest_path.read_text())
    candidate = manifest["groups"][0]["candidates"][0]
    assert Path(candidate["path"]).is_file()
    assert Path(candidate["latent_path"]).is_file()
    assert Path(manifest["resume_path"]).is_file()
    assert runtime.step == runtime.prompt_cursor == 0


@pytest.mark.parametrize("bad_score", [0.0, -1.0, 10.1, float("nan")])
def test_cached_round_rejects_out_of_rubric_scores_before_update(tmp_path, bad_score):
    runtime = object.__new__(VAJudgerRuntime)
    runtime.network = ToyAdapter()
    runtime.old_policy = {"weight": runtime.network.weight.detach().clone()}
    runtime.semantic_fingerprint = "semantic"
    runtime.config = SimpleNamespace(group_size=2, seed=42, max_steps=10)
    runtime.prompts = ["prompt-0"]
    runtime.step = runtime.prompt_cursor = 0
    called = []
    runtime.update_group = lambda *args, **kwargs: called.append(True)
    manifest, rewards = _write_cached_round(tmp_path, runtime, group_count=1)
    payload = json.loads(rewards.read_text())
    payload["groups"][0]["raw_scores"][0][0] = bad_score
    rewards.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete finite Kx5"):
        runtime.train_round(manifest, rewards)
    assert called == []
    assert runtime.step == runtime.prompt_cursor == 0
