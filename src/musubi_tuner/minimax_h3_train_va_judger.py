from __future__ import annotations

import argparse
import logging
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from musubi_tuner.minimax_h3.assets import default_text_encoder_assets


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMax H3 staged VA-Judger NFT LoRA trainer")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text_encoder", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=default_text_encoder_assets())
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--audio_vae", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="UTF-8 text or JSONL prompt file")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("run", "generate", "train"), default="run")
    parser.add_argument("--round_dir", type=Path, help="immutable rollout/reward round directory")
    parser.add_argument("--round_groups", type=int, default=4, help="prompt groups generated before each training round")
    parser.add_argument("--reward_model", type=Path, help="local VA-Judger checkpoint directory")
    parser.add_argument(
        "--reward_python",
        type=Path,
        default=Path(sys.executable),
        help="scoring Python executable (defaults to the current Python)",
    )
    parser.add_argument("--reward_max_new_tokens", type=int, default=2048)
    parser.add_argument("--reward_device", default="cuda")
    parser.add_argument("--reward_attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--reward_max_frames", type=int, default=12)
    parser.add_argument("--reward_video_max_pixels", type=int, default=602112)
    parser.add_argument("--reward_pair_batch_size", type=int, default=1)
    parser.add_argument("--reward_do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward_temperature", type=float, default=0.75)
    parser.add_argument("--reward_top_p", type=float, default=0.92)
    parser.add_argument("--reward_top_k", type=int, default=32)
    parser.add_argument("--reward_seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--network_dim", type=int, default=32)
    parser.add_argument("--network_alpha", type=float, default=32.0)
    parser.add_argument("--disable_token_refiner", action="store_true")
    parser.add_argument("--beta_mix", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=1e-4)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--video_loss_weight", type=float, default=1.0)
    parser.add_argument("--audio_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--initial_lora", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fp8_base", action="store_true")
    parser.add_argument("--int8_convrot_base", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of H3 transformer blocks to stream from CPU (0-48)")
    parser.add_argument("--block_swap_h2d_only", action="store_true", help="keep swapped blocks on CPU and stream weights H2D")
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true", help="pin streamed block weights in host memory")
    parser.add_argument(
        "--text_encoder_quantization",
        choices=("none", "int8", "nf4", "nvfp4", "nvfp4_awq"),
        default="int8",
    )
    parser.add_argument("--validate_config", action="store_true", help="validate paths/options without loading models")
    parser.add_argument("--log_level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace):
    from musubi_tuner.minimax_h3.va_judger_runtime import VAJudgerTrainConfig

    values = vars(args).copy()
    values.pop("log_level")
    values.pop("validate_config")
    for key in (
        "stage",
        "round_dir",
        "round_groups",
        "reward_model",
        "reward_python",
        "reward_max_new_tokens",
        "reward_device",
        "reward_attn_implementation",
        "reward_max_frames",
        "reward_video_max_pixels",
        "reward_do_sample",
        "reward_temperature",
        "reward_top_p",
        "reward_top_k",
        "reward_seed",
    ):
        values.pop(key)
    values["video_vae"] = values.pop("vae")
    values["token_refiner"] = not values.pop("disable_token_refiner")
    return VAJudgerTrainConfig(**values)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_step(path: Path) -> int:
    import torch

    return int(torch.load(path, map_location="cpu", weights_only=False)["step"])


def _check_cached_judge(path: Path, args: argparse.Namespace) -> None:
    from musubi_tuner.minimax_h3_score_va_judger import (
        SCORING_IMPLEMENTATION_VERSION,
        SCORING_PROMPT_SHA256,
        SCORING_PROMPT_VERSION,
    )

    judge = json.loads(path.read_text(encoding="utf-8")).get("judge", {})
    if judge.get("reward_model") != str(args.reward_model.resolve()):
        raise ValueError(f"cached rewards use a different judge model: {path}")
    if judge.get("scoring_prompt_version") != SCORING_PROMPT_VERSION or judge.get("scoring_prompt_sha256") != SCORING_PROMPT_SHA256:
        raise ValueError(f"cached rewards use a different scoring rubric: {path}")
    if judge.get("backend") != "native":
        raise ValueError(f"cached rewards were not produced by the native backend: {path}")
    if judge.get("implementation_version") != SCORING_IMPLEMENTATION_VERSION:
        raise ValueError(f"cached rewards use different judge preprocessing: {path}")
    expected = {
        key: getattr(args, f"reward_{key}")
        for key in (
            "pair_batch_size",
            "max_new_tokens",
            "device",
            "attn_implementation",
            "max_frames",
            "video_max_pixels",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "seed",
        )
    }
    actual = judge.get("args", {})
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError(f"cached rewards use different judge settings: {path}")


def _without_option(argv: Sequence[str], option: str) -> list[str]:
    cleaned: list[str] = []
    skip_value = False
    prefix = option + "="
    for value in argv:
        if skip_value:
            skip_value = False
            continue
        if value == option:
            skip_value = True
            continue
        if value.startswith(prefix):
            continue
        cleaned.append(value)
    return cleaned


def _run_staged(args: argparse.Namespace, argv: list[str], semantic_fingerprint: str | None = None) -> None:
    for name in ("reward_model", "reward_python"):
        value = getattr(args, name)
        if value is None or not value.exists():
            raise ValueError(f"--{name} must identify an existing local path for staged training")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    score_script = str(Path(__file__).with_name("minimax_h3_score_va_judger.py").resolve())
    step = 0
    resume = args.resume
    if resume is not None:
        step = _resume_step(resume)
    while step < args.max_steps:
        round_dir = root / "rounds" / f"round_{step:06d}"
        manifest = round_dir / "manifest.json"
        rewards = round_dir / "rewards.json"
        trained = round_dir / "trained.json"
        if trained.exists():
            receipt = json.loads(trained.read_text(encoding="utf-8"))
            if receipt["start_step"] != step or int(receipt["step"]) <= step:
                raise ValueError(f"invalid completed round receipt: {trained}")
            if semantic_fingerprint is not None and receipt.get("semantic_fingerprint") != semantic_fingerprint:
                raise ValueError(f"completed round uses different training semantics: {trained}")
            if (
                semantic_fingerprint is not None
                and receipt.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest()
            ):
                raise ValueError(f"completed round manifest changed: {manifest}")
            if semantic_fingerprint is not None:
                _check_cached_judge(rewards, args)
                continuation = Path(receipt["resume_path"])
                if receipt.get("resume_sha256") != _file_sha256(continuation) or _resume_step(continuation) != int(receipt["step"]):
                    raise ValueError(f"completed round resume changed: {continuation}")
            step = int(receipt["step"])
            resume = Path(receipt["resume_path"])
            continue
        generation_args = argv + ["--stage", "generate", "--round_dir", str(round_dir)]
        if resume is not None:
            generation_args = _without_option(generation_args, "--initial_lora")
            generation_args += ["--resume", str(resume)]
        if not manifest.exists():
            subprocess.run([sys.executable, script, *generation_args], check=True)
        if not rewards.exists():
            scorer_options = ["--pair_batch_size", str(args.reward_pair_batch_size)]
            for key in (
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
                scorer_options += [f"--{key}", str(getattr(args, f"reward_{key}"))]
            scorer_options.append("--do_sample" if args.reward_do_sample else "--no-do_sample")
            subprocess.run(
                [
                    str(args.reward_python),
                    score_script,
                    "--round_manifest",
                    str(manifest),
                    "--reward_model",
                    str(args.reward_model),
                    *scorer_options,
                ],
                check=True,
            )
        elif semantic_fingerprint is not None:
            _check_cached_judge(rewards, args)
        subprocess.run([sys.executable, script, *argv, "--stage", "train", "--round_dir", str(round_dir)], check=True)
        receipt = json.loads(trained.read_text(encoding="utf-8"))
        step = int(receipt["step"])
        resume = Path(receipt["resume_path"])


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = create_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    stage = args.stage
    if args.round_groups < 1:
        parser.error("--round_groups must be positive")
    if stage in {"generate", "train"} and args.round_dir is None:
        parser.error(f"--round_dir is required for --stage {stage}")
    args.reward_endpoint = None
    manifest = None
    if stage == "train":
        manifest_path = args.round_dir / "manifest.json"
        if (args.round_dir / "trained.json").exists():
            raise FileExistsError(f"round already trained: {args.round_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        args.resume = Path(manifest["resume_path"])
        args.initial_lora = None
    config = config_from_args(args)
    config.validate()
    if args.validate_config:
        return
    if stage == "run":
        from musubi_tuner.minimax_h3.va_judger_runtime import _config_fingerprint, _semantic_config

        _run_staged(args, argv, _config_fingerprint(_semantic_config(config)))
        return
    from musubi_tuner.minimax_h3.va_judger_runtime import VAJudgerRuntime

    runtime = VAJudgerRuntime(config)
    if stage == "generate":
        runtime.generate_round(args.round_dir, min(args.round_groups, config.max_steps - runtime.step))
    elif stage == "train":
        runtime.train_round(args.round_dir / "manifest.json", args.round_dir / "rewards.json")
        adapter, resume = runtime.save(f"step-{runtime.step:06d}")
        runtime.save("final")
        _write_json(
            args.round_dir / "trained.json",
            {
                "start_step": manifest["start_step"],
                "step": runtime.step,
                "adapter_path": str(adapter.resolve()),
                "resume_path": str(resume.resolve()),
                "resume_sha256": _file_sha256(resume),
                "adapter_sha256": _file_sha256(adapter),
                "semantic_fingerprint": runtime.semantic_fingerprint,
                "manifest_sha256": hashlib.sha256((args.round_dir / "manifest.json").read_bytes()).hexdigest(),
            },
        )


if __name__ == "__main__":
    main()
