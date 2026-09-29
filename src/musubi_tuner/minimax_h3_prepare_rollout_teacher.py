"""Prepare a privileged rollout-teacher dataset for MiniMax H3 training."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Callable, Iterable

import av
import toml
from PIL import Image

from musubi_tuner.minimax_h3.references import MAX_REFERENCE_IMAGES, MAX_REFERENCES

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m2ts", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}


def _media_files(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"directory does not exist: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _matching_references(target_stem: str, directory: Path) -> list[Path]:
    matches: list[tuple[int, Path]] = []
    for path in _media_files(directory, IMAGE_SUFFIXES):
        if path.stem == target_stem:
            matches.append((0, path))
            continue
        prefix = target_stem + "_"
        if not path.stem.startswith(prefix):
            continue
        suffix = path.stem.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            raise ValueError(f"reference {path} matches target {target_stem!r} but its final underscore suffix is not numeric")
        matches.append((int(suffix) + 1, path))
    return [path for _, path in sorted(matches, key=lambda item: (item[0], item[1].name))]


def _uniform_indices(frame_count: int, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("teacher frame count must be positive")
    if frame_count < count:
        raise ValueError(f"video has only {frame_count} frames, cannot extract {count} distinct teacher frames")
    # Midpoints of equal-width bins avoid choosing only the endpoints and remain deterministic.
    indices = tuple(min(frame_count - 1, ((2 * index + 1) * frame_count) // (2 * count)) for index in range(count))
    if len(set(indices)) != count:
        raise ValueError(f"could not select {count} distinct frames from a {frame_count}-frame video")
    return indices


def _extract_uniform_frames(video: Path, destinations: list[Path]) -> None:
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        frame_count = int(stream.frames or 0)

    if frame_count <= 0:
        with av.open(str(video)) as container:
            frame_count = sum(1 for _ in container.decode(video=0))

    wanted = dict(zip(_uniform_indices(frame_count, len(destinations)), destinations))
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            destination = wanted.get(index)
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame.to_ndarray(format="rgb24")).save(destination)
    missing = [str(path) for path in destinations if not path.is_file()]
    if missing:
        raise RuntimeError(f"failed to decode selected frames from {video}: {', '.join(missing)}")


def _next_reference_names(target_stem: str, existing: Iterable[Path], count: int) -> list[str]:
    occupied = {path.name.lower() for path in existing}
    result: list[str] = []
    index = 0
    while len(result) < count:
        name = f"{target_stem}_{index}.png"
        if name.lower() not in occupied:
            result.append(name)
            occupied.add(name.lower())
        index += 1
    return result


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"JSONL file does not exist: {path}")
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {path}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} of {path} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"JSONL file has no records: {path}")
    return rows


def _jsonl_has_references(path: Path) -> bool:
    return any(any(key == "control_path" or key.startswith("control_path_") for key in row) for row in _read_jsonl(path))


def _detect_mode(datasets: list[dict]) -> str:
    modes: set[str] = set()
    for dataset in datasets:
        has_directory_references = any(
            dataset.get(key) for key in ("source_image_directory", "source_video_directory", "source_audio_directory")
        )
        jsonl_value = dataset.get("video_jsonl_file")
        has_jsonl_references = bool(jsonl_value) and _jsonl_has_references(Path(jsonl_value).expanduser().resolve())
        modes.add("ref2va" if has_directory_references or has_jsonl_references else "fl2va")
    if len(modes) != 1:
        raise ValueError("--mode auto cannot combine FL2VA and Ref2VA dataset rows; prepare them separately")
    return modes.pop()


def _prepare_fl2va(datasets: list[dict], output_dir: Path) -> None:
    for index, dataset in enumerate(datasets):
        student_cache = dataset.get("cache_directory")
        if not student_cache:
            raise ValueError(f"dataset {index} has no cache_directory to reuse for the FL2VA teacher")
        dataset["latent_cache_directory"] = str(Path(student_cache).resolve())
        dataset["cache_directory"] = str((output_dir / "cache" / f"dataset_{index}").resolve())


def _prepare_ref2va(
    datasets: list[dict],
    output_dir: Path,
    teacher_frames: int,
    extractor: Callable[[Path, list[Path]], None] = _extract_uniform_frames,
) -> None:
    if teacher_frames < 1:
        raise ValueError("--teacher_frames must be positive")
    for index, dataset in enumerate(datasets):
        if dataset.get("video_jsonl_file"):
            _prepare_ref2va_jsonl(dataset, index, output_dir, teacher_frames, extractor)
            continue
        target_value = dataset.get("target_video_directory")
        if not target_value:
            raise ValueError(f"Ref2VA dataset {index} needs target_video_directory or video_jsonl_file")
        target_dir = Path(target_value).expanduser().resolve()
        targets = _media_files(target_dir, VIDEO_SUFFIXES)
        if not targets:
            raise ValueError(f"Ref2VA dataset {index} has no target videos in {target_dir}")

        source_value = dataset.get("source_image_directory")
        source_dir = Path(source_value).expanduser().resolve() if source_value else None
        teacher_refs = (output_dir / "references" / f"dataset_{index}").resolve()
        teacher_refs.mkdir(parents=True, exist_ok=True)

        for target in targets:
            shared = _matching_references(target.stem, source_dir) if source_dir else []
            if len(shared) + teacher_frames > MAX_REFERENCE_IMAGES:
                raise ValueError(
                    f"teacher item {target.stem!r} would have {len(shared) + teacher_frames} image references; "
                    f"MiniMax H3 accepts at most {MAX_REFERENCE_IMAGES}"
                )
            for reference in shared:
                destination = teacher_refs / reference.name
                if destination.exists() and destination.read_bytes() != reference.read_bytes():
                    raise ValueError(f"reference destination collision: {destination}")
                if not destination.exists():
                    shutil.copy2(reference, destination)
            names = _next_reference_names(target.stem, shared, teacher_frames)
            extractor(target, [teacher_refs / name for name in names])
            teacher_count = len(_matching_references(target.stem, teacher_refs))
            if teacher_count <= len(shared):
                raise RuntimeError(f"teacher item {target.stem!r} has {teacher_count} references, student has {len(shared)}")

        dataset["source_image_directory"] = str(teacher_refs)
        modalities = list(dataset.get("source_modalities") or [])
        for field, modality in (
            ("source_video_directory", "video"),
            ("source_audio_directory", "audio"),
        ):
            if dataset.get(field) and modality not in modalities:
                modalities.append(modality)
        if "image" not in modalities:
            modalities.insert(0, "image")
        dataset["source_modalities"] = modalities
        dataset.pop("latent_cache_directory", None)
        dataset["cache_directory"] = str((output_dir / "cache" / f"dataset_{index}").resolve())


def _resolve_row_path(value: str, jsonl_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (jsonl_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _prepare_ref2va_jsonl(
    dataset: dict,
    dataset_index: int,
    output_dir: Path,
    teacher_frames: int,
    extractor: Callable[[Path, list[Path]], None],
) -> None:
    source_jsonl = Path(dataset["video_jsonl_file"]).expanduser().resolve()
    rows = _read_jsonl(source_jsonl)
    reference_dir = (output_dir / "references" / f"dataset_{dataset_index}").resolve()
    reference_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[dict] = []
    for row_index, original in enumerate(rows):
        row = dict(original)
        video_value = row.get("video_path")
        if not video_value:
            raise ValueError(f"row {row_index + 1} of {source_jsonl} has no video_path for target-frame extraction")
        video = _resolve_row_path(video_value, source_jsonl.parent)
        if not video.is_file():
            raise ValueError(f"target video does not exist for row {row_index + 1} of {source_jsonl}: {video}")
        row["video_path"] = str(video)

        if "control_path" in row:
            if any(key.startswith("control_path_") for key in row):
                raise ValueError(f"row {row_index + 1} of {source_jsonl} mixes control_path with control_path_N")
            row["control_path_0"] = row.pop("control_path")
        numbered: list[tuple[int, str]] = []
        for key, value in list(row.items()):
            if not key.startswith("control_path_"):
                continue
            suffix = key.removeprefix("control_path_")
            if not suffix.isdigit():
                raise ValueError(f"row {row_index + 1} of {source_jsonl} has invalid reference key {key!r}")
            numbered.append((int(suffix), key))
            row[key] = str(_resolve_row_path(value, source_jsonl.parent))
        numbered.sort()
        if [number for number, _ in numbered] != list(range(len(numbered))):
            raise ValueError(f"row {row_index + 1} of {source_jsonl} must use contiguous control_path_N keys from zero")
        image_count = sum(Path(row[key]).suffix.lower() in IMAGE_SUFFIXES for _, key in numbered)
        if image_count + teacher_frames > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"row {row_index + 1} of {source_jsonl} would have {image_count + teacher_frames} image references; "
                f"MiniMax H3 accepts at most {MAX_REFERENCE_IMAGES}"
            )
        if len(numbered) + teacher_frames > MAX_REFERENCES:
            raise ValueError(
                f"row {row_index + 1} of {source_jsonl} would have {len(numbered) + teacher_frames} references; "
                f"MiniMax H3 accepts at most {MAX_REFERENCES}"
            )

        destinations = [reference_dir / f"row_{row_index:05d}_{extra}.png" for extra in range(teacher_frames)]
        extractor(video, destinations)
        for extra, destination in enumerate(destinations, start=len(numbered)):
            row[f"control_path_{extra}"] = str(destination)
        prepared.append(row)

    teacher_jsonl = (output_dir / f"teacher_dataset_{dataset_index}.jsonl").resolve()
    teacher_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared), encoding="utf-8")
    dataset["video_jsonl_file"] = str(teacher_jsonl)
    dataset.pop("latent_cache_directory", None)
    dataset["cache_directory"] = str((output_dir / "cache" / f"dataset_{dataset_index}").resolve())


def prepare_teacher(
    student_config: Path,
    output_dir: Path,
    mode: str = "auto",
    teacher_frames: int = 2,
    extractor: Callable[[Path, list[Path]], None] = _extract_uniform_frames,
) -> tuple[Path, str]:
    student_config = student_config.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    config = toml.load(student_config)
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"dataset config has no [[datasets]] rows: {student_config}")
    detected_mode = _detect_mode(datasets)
    resolved_mode = detected_mode if mode == "auto" else mode
    if resolved_mode not in {"fl2va", "ref2va"}:
        raise ValueError("mode must be auto, fl2va, or ref2va")
    if resolved_mode != detected_mode:
        raise ValueError(
            f"--mode {resolved_mode} conflicts with the detected {detected_mode} student dataset; fix the config or use --mode auto"
        )

    destination = output_dir / "teacher.toml"
    references_dir = output_dir / "references"
    if destination.exists() or (references_dir.exists() and any(references_dir.iterdir())):
        raise ValueError(f"output already contains a prepared teacher; choose a new --output_dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if resolved_mode == "fl2va":
        _prepare_fl2va(datasets, output_dir)
    else:
        _prepare_ref2va(datasets, output_dir, teacher_frames, extractor)

    temporary = destination.with_suffix(".toml.tmp")
    temporary.write_text(toml.dumps(config), encoding="utf-8")
    temporary.replace(destination)
    return destination, resolved_mode


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the privileged teacher dataset used by MiniMax H3 rollout/D-OPSD supervision."
    )
    parser.add_argument("--student_config", type=Path, required=True, help="student dataset TOML")
    parser.add_argument("--output_dir", type=Path, required=True, help="new teacher assets, cache directories, and teacher.toml")
    parser.add_argument("--mode", choices=("auto", "fl2va", "ref2va"), default="auto")
    parser.add_argument(
        "--teacher_frames",
        type=int,
        default=2,
        help="Ref2VA only: uniformly spaced target frames added as teacher-only image references (default: 2)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = setup_parser().parse_args()
    teacher_config, mode = prepare_teacher(args.student_config, args.output_dir, args.mode, args.teacher_frames)
    print(f"Prepared {mode.upper()} rollout teacher: {teacher_config}")
    if mode == "fl2va":
        print("Next: cache teacher text with --task fl2va. The student latent cache is reused; do not cache teacher latents.")
    else:
        print("Next: cache teacher latents and text with --task ref2va, using the same sizing options as the student.")


if __name__ == "__main__":
    main()
