"""Audit MiniMax Music 3 audio datasets before caching or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_MUSIC3
from musubi_tuner.dataset.audio_dataset import AudioDataset
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer


def audit_audio_dataset(dataset: AudioDataset) -> dict:
    import soundfile as sf

    report = {
        "audio_directory": dataset.audio_directory,
        "audio_jsonl_file": dataset.audio_jsonl_file,
        "total": len(dataset.records),
        "readable": 0,
        "missing_files": [],
        "decode_errors": [],
        "missing_captions": [],
        "missing_lyrics": [],
        "too_short": [],
        "clipped_by_max_duration": [],
        "duplicate_cache_names": {},
        "duration_seconds": {"minimum": None, "maximum": None, "total": 0.0},
    }
    cache_names: dict[str, list[str]] = {}
    durations = []
    for record in dataset.records:
        path = Path(record["path"])
        if not path.is_file():
            report["missing_files"].append(str(path))
            continue
        if not str(record.get("caption", "")).strip():
            report["missing_captions"].append(str(path))
        if not str(record.get("lyrics", "")).strip():
            report["missing_lyrics"].append(str(path))
        cache_name = f"{path.stem.replace(' ', '_')}_{dataset.architecture}.safetensors"
        cache_names.setdefault(cache_name, []).append(str(path))
        try:
            info = sf.info(str(path))
            duration = info.frames / info.samplerate
        except Exception as error:
            report["decode_errors"].append({"path": str(path), "error": str(error)})
            continue
        report["readable"] += 1
        durations.append(duration)
        if duration < dataset.min_duration:
            report["too_short"].append(str(path))
        if dataset.max_duration is not None and duration > dataset.max_duration:
            report["clipped_by_max_duration"].append(str(path))
    report["duplicate_cache_names"] = {
        name: paths for name, paths in cache_names.items() if len(paths) > 1
    }
    if durations:
        report["duration_seconds"] = {
            "minimum": min(durations),
            "maximum": max(durations),
            "total": sum(durations),
        }
    report["fatal_issue_count"] = (
        len(report["missing_files"])
        + len(report["decode_errors"])
        + len(report["too_short"])
        + len(report["duplicate_cache_names"])
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, args, architecture=ARCHITECTURE_MINIMAX_MUSIC3
    )
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    reports = [audit_audio_dataset(dataset) for dataset in group.datasets if isinstance(dataset, AudioDataset)]
    payload = {"datasets": reports, "fatal_issue_count": sum(item["fatal_issue_count"] for item in reports)}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.strict and payload["fatal_issue_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
