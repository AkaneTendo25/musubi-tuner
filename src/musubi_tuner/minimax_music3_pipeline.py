"""Prepare raw audio caches and launch MiniMax Music 3 DiT LoRA training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_commands(args: argparse.Namespace, train_args: list[str]) -> list[list[str]]:
    python = sys.executable
    common = ["--dataset_config", str(args.dataset_config)]
    if args.skip_existing:
        common.append("--skip_existing")
    commands = []
    if not args.skip_dataset_report:
        report_command = [
            python,
            "-m",
            "musubi_tuner.minimax_music3_dataset_report",
            "--dataset_config",
            str(args.dataset_config),
            "--output",
            str(args.dataset_report),
        ]
        if args.strict_dataset:
            report_command.append("--strict")
        commands.append(report_command)
    commands.extend([
        [
            python,
            "-m",
            "musubi_tuner.minimax_music3_cache_latents",
            *common,
            "--dav",
            str(args.dav_encoder),
            "--dtype",
            args.cache_dtype,
        ],
        [
            python,
            "-m",
            "musubi_tuner.minimax_music3_cache_text_encoder_outputs",
            *common,
            "--ar_model",
            args.ar_model,
            "--seed",
            str(args.seed),
        ],
    ])
    if not args.cache_only:
        commands.append(
            [
                python,
                "-m",
                "accelerate.commands.launch",
                "--num_processes",
                str(args.num_processes),
                "--num_cpu_threads_per_process",
                str(args.num_cpu_threads_per_process),
                "-m",
                "musubi_tuner.minimax_music3_train_network",
                "--dataset_config",
                str(args.dataset_config),
                "--dit",
                str(args.dit),
                "--vae",
                str(args.vae),
                "--output_dir",
                str(args.output_dir),
                "--output_name",
                args.output_name,
                *train_args,
            ]
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_config", type=Path, required=True)
    parser.add_argument("--dav_encoder", type=Path, required=True, help="official dav.pth with encoder weights")
    parser.add_argument("--ar_model", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--dit", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    parser.add_argument("--output_name", default="music3_lora")
    parser.add_argument("--cache_dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_cpu_threads_per_process", type=int, default=1)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_dataset_report", action="store_true")
    parser.add_argument("--strict_dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset_report", type=Path)
    args, train_args = parser.parse_known_args()
    if not args.cache_only and (args.dit is None or args.vae is None):
        parser.error("--dit and --vae are required unless --cache_only is used")
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    if args.dataset_report is None:
        args.dataset_report = args.output_dir / f"{args.output_name}_dataset_report.json"

    commands = build_commands(args, train_args)
    for command in commands:
        print(" ".join(map(str, command)), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
