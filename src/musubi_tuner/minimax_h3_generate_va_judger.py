from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from musubi_tuner.minimax_h3.va_judger_runtime import _config_fingerprint, _semantic_config, load_prompts
from musubi_tuner.minimax_h3_score_va_judger import _load_manifest, _verify
from musubi_tuner.minimax_h3_train_va_judger import _check_cached_judge, config_from_args, create_parser


def _without_stage(argv: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    skip = False
    for value in argv:
        if skip:
            skip = False
            continue
        if value == "--stage":
            skip = True
            continue
        if value.startswith("--stage="):
            continue
        cleaned.append(value)
    return cleaned


def _validate_cached_rewards(manifest_path: Path, rewards_path: Path, args) -> None:
    try:
        rewards = json.loads(rewards_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cached rewards: {rewards_path}") from exc
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if rewards.get("manifest_sha256") != expected:
        raise ValueError(f"cached rewards do not match the exact manifest: {rewards_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_cached_reward_structure(manifest, rewards, rewards_path)
    _check_cached_judge(rewards_path, args)


def _validate_cached_reward_structure(manifest: dict, rewards: dict, rewards_path: Path) -> None:
    if rewards.get("version") != 1:
        raise ValueError(f"cached rewards use an unsupported contract version: {rewards_path}")
    keys = ("round_id", "behavior_sha256", "semantic_fingerprint", "start_step", "prompt_cursor", "group_count")
    for key in keys:
        if rewards.get(key) != manifest.get(key):
            raise ValueError(f"cached reward {key} does not match the manifest: {rewards_path}")
    reward_groups = rewards.get("groups")
    manifest_groups = manifest["groups"]
    if not isinstance(reward_groups, list) or len(reward_groups) != len(manifest_groups):
        raise ValueError(f"cached reward groups are incomplete: {rewards_path}")
    for group, judged in zip(manifest_groups, reward_groups, strict=True):
        if not isinstance(judged, dict):
            raise ValueError(f"cached reward group is invalid: {rewards_path}")
        if judged.get("group_id") != group["group_id"] or judged.get("prompt") != group["prompt"]:
            raise ValueError(f"cached reward group identity does not match the manifest: {rewards_path}")
        expected_hashes = [candidate["media_sha256"] for candidate in group["candidates"]]
        if judged.get("candidate_hashes") != expected_hashes:
            raise ValueError(f"cached reward candidate hashes do not match the manifest: {rewards_path}")
        scores = judged.get("raw_scores")
        if not isinstance(scores, list) or len(scores) != len(expected_hashes):
            raise ValueError(f"cached reward scores are incomplete: {rewards_path}")
        for row in scores:
            if not isinstance(row, list) or len(row) != 5:
                raise ValueError(f"cached reward scores must be Kx5: {rewards_path}")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 10
                for value in row
            ):
                raise ValueError(f"cached reward scores must be finite values in [1, 10]: {rewards_path}")


def _validate_prompt_fingerprint(manifest: dict, config) -> None:
    import torch

    state = torch.load(Path(manifest["resume_path"]), map_location="cpu", weights_only=False)
    expected = hashlib.sha256("\n".join(load_prompts(config.prompts)).encode("utf-8")).hexdigest()
    if state.get("prompt_fingerprint") != expected:
        raise ValueError("round manifest was generated from different prompt contents")


def _validate_manifest(manifest_path: Path, config) -> None:
    manifest, _ = _load_manifest(manifest_path)
    expected = _config_fingerprint(_semantic_config(config))
    if manifest["semantic_fingerprint"] != expected:
        raise ValueError(f"round manifest uses different generation/training semantics: {manifest_path}")
    _verify(manifest, True)
    _validate_prompt_fingerprint(manifest, config)


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = create_parser()
    parser.description = "Generate and score a MiniMax H3 VA-Judger rollout round without training"
    parser.set_defaults(stage="generate")
    args = parser.parse_args(argv)
    if args.stage != "generate":
        parser.error("this entrypoint only supports --stage generate")
    if args.round_dir is None:
        parser.error("--round_dir is required")
    if args.round_groups < 1:
        parser.error("--round_groups must be positive")
    if args.reward_model is None or not args.reward_model.is_dir():
        parser.error("--reward_model must identify an existing local directory")
    if args.reward_python is None or not args.reward_python.is_file():
        parser.error("--reward_python must identify an existing local executable file")

    args.reward_endpoint = None
    config = config_from_args(args)
    config.validate()
    if args.validate_config:
        return

    round_dir = args.round_dir.resolve()
    manifest = round_dir / "manifest.json"
    rewards = round_dir / "rewards.json"
    trainer_script = Path(__file__).with_name("minimax_h3_train_va_judger.py").resolve()
    scorer_script = Path(__file__).with_name("minimax_h3_score_va_judger.py").resolve()
    if not manifest.exists():
        generation_args = _without_stage(argv)
        subprocess.run(
            [sys.executable, str(trainer_script), *generation_args, "--stage", "generate"],
            check=True,
        )
    if not manifest.is_file():
        raise FileNotFoundError(f"generation did not produce a manifest: {manifest}")
    _validate_manifest(manifest, config)

    if rewards.exists():
        _validate_cached_rewards(manifest, rewards, args)
        return
    scorer_args = [
        str(args.reward_python),
        str(scorer_script),
        "--round_manifest",
        str(manifest),
        "--reward_model",
        str(args.reward_model),
        "--pair_batch_size",
        str(args.reward_pair_batch_size),
    ]
    for name in (
        "max_new_tokens",
        "device",
        "attn_implementation",
        "max_frames",
        "video_max_pixels",
        "temperature",
        "top_p",
        "top_k",
        "seed",
    ):
        scorer_args.extend((f"--{name}", str(getattr(args, f"reward_{name}"))))
    scorer_args.append("--do_sample" if args.reward_do_sample else "--no-do_sample")
    subprocess.run(scorer_args, check=True)
    if not rewards.is_file():
        raise FileNotFoundError(f"scoring did not produce rewards: {rewards}")
    _validate_cached_rewards(manifest, rewards, args)


if __name__ == "__main__":
    main()
