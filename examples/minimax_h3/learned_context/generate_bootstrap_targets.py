from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import toml

from musubi_tuner.minimax_h3_generate_video import create_parser, generator_from_args, request_from_args


def _load_manifest(path: Path) -> tuple[Path, str, list[dict]]:
    payload = toml.load(path)
    bootstrap = payload.get("bootstrap", {})
    output_value = str(bootstrap.get("output_directory", "")).strip()
    concept = str(bootstrap.get("concept", "")).strip()
    scenes = payload.get("scenes", [])
    if not output_value:
        raise ValueError("bootstrap.output_directory is required")
    if not concept:
        raise ValueError("bootstrap.concept is required")
    if not scenes:
        raise ValueError("at least one [[scenes]] entry is required")
    names: set[str] = set()
    normalized = []
    for scene in scenes:
        name = str(scene.get("name", "")).strip()
        caption = str(scene.get("caption", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"invalid scene name {name!r}; use letters, digits, '_' or '-'")
        if name in names:
            raise ValueError(f"duplicate scene name {name!r}")
        if not caption:
            raise ValueError(f"scene {name!r} requires a caption")
        names.add(name)
        normalized.append({"name": name, "caption": caption, "seed": int(scene.get("seed", 42))})
    return Path(output_value), concept, normalized


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--bootstrap_config", type=Path, required=True)
    known, generator_argv = wrapper.parse_known_args()
    output_directory, concept, scenes = _load_manifest(known.bootstrap_config)
    output_directory.mkdir(parents=True, exist_ok=True)

    first = scenes[0]
    first_output = output_directory / f"{first['name']}.mp4"
    parser = create_parser()
    args = parser.parse_args(
        generator_argv
        + [
            "--prompt",
            f"{first['caption']} {concept}",
            "--seed",
            str(first["seed"]),
            "--output",
            str(first_output),
        ]
    )
    missing = [name for name in ("text_encoder", "vae", "audio_vae") if getattr(args, name) is None]
    if missing:
        parser.error("bootstrap generation requires " + ", ".join(f"--{name}" for name in missing))
    first_request = request_from_args(args)
    first_request.validate(check_files=True)
    generator = generator_from_args(args, first_request)

    for index, scene in enumerate(scenes, start=1):
        scene_args = copy.copy(args)
        scene_args.prompt = f"{scene['caption']} {concept}"
        scene_args.seed = scene["seed"]
        scene_args.output = output_directory / f"{scene['name']}.mp4"
        request = request_from_args(scene_args)
        print(f"[{index}/{len(scenes)}] {scene['name']}", flush=True)
        generator.generate(request)
        if not request.output.is_file():
            raise RuntimeError(f"generation did not create {request.output}")
        request.output.with_suffix(".txt").write_text(scene["caption"] + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
