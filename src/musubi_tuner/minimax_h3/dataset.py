from __future__ import annotations

import copy
import json
import math
import os
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import DatasetGroup, ItemInfo, VideoDataset
from musubi_tuner.dataset.media_utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, glob_images, glob_videos
from musubi_tuner.minimax_h3.architecture import is_valid_frame_count
from musubi_tuner.minimax_h3.audio_dataset import H3AudioDataset
from musubi_tuner.minimax_h3.image_training import condition_paths, resample_image_targets, sample_fingerprint, validate_image_mode
from musubi_tuner.minimax_h3.media import MediaAsset, MediaModality, slice_media_asset
from musubi_tuner.minimax_h3.references import REFERENCE_FINGERPRINT_KEY, reference_fingerprint

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus")
_CONTROL_PATH_PATTERN = re.compile(r"^control_path_(\d+)$")
_CONTROL_VIDEO_PATH_PATTERN = re.compile(r"^control_video_path_(\d+)$")
_CONTROL_AUDIO_PATH_PATTERN = re.compile(r"^control_audio_path_(\d+)$")
_CONTROL_MODALITY_PATTERN = re.compile(r"^control_modality_(\d+)$")
_CROP_SUFFIX_PATTERN = re.compile(r"^(?P<stem>.+)_(?P<start>\d{5})-(?P<frames>\d+)$")


def _normal_path(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _modality_for_path(path: Path) -> MediaModality:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return MediaModality.IMAGE
    if suffix in VIDEO_EXTENSIONS:
        return MediaModality.VIDEO
    if suffix in AUDIO_EXTENSIONS:
        return MediaModality.AUDIO
    raise ValueError(f"unsupported H3 control media extension: {path}")


def _ordered_control_paths(record: dict[str, Any]) -> tuple[Path, ...]:
    numbered: list[tuple[int, Path]] = []
    for key, value in record.items():
        match = _CONTROL_PATH_PATTERN.fullmatch(key)
        if match and value:
            numbered.append((int(match.group(1)), Path(value)))
    if record.get("control_path"):
        if numbered:
            raise ValueError("use control_path or control_path_N, not both")
        return (Path(record["control_path"]),)
    numbered.sort(key=lambda value: value[0])
    indices = [index for index, _ in numbered]
    if indices != list(range(len(indices))):
        raise ValueError(f"control_path_N indices must be contiguous from zero, got {indices}")
    return tuple(path for _, path in numbered)


def _paired_control_assets(record: dict[str, Any]) -> tuple[MediaAsset, ...]:
    video_paths: dict[int, Path] = {}
    audio_paths: dict[int, Path] = {}
    for key, value in record.items():
        if not value:
            continue
        video_match = _CONTROL_VIDEO_PATH_PATTERN.fullmatch(key)
        audio_match = _CONTROL_AUDIO_PATH_PATTERN.fullmatch(key)
        if video_match:
            video_paths[int(video_match.group(1))] = Path(value)
        elif audio_match:
            audio_paths[int(audio_match.group(1))] = Path(value)
    if not video_paths and not audio_paths:
        return ()
    if set(video_paths) != set(audio_paths):
        raise ValueError("control_video_path_N and control_audio_path_N must use the same indices")
    indices = sorted(video_paths)
    if indices != list(range(len(indices))):
        raise ValueError(f"paired control indices must be contiguous from zero, got {indices}")
    assets = []
    for index in indices:
        video_path = video_paths[index]
        audio_path = audio_paths[index]
        if _modality_for_path(video_path) is not MediaModality.VIDEO:
            raise ValueError(f"control_video_path_{index} must be a video: {video_path}")
        if _modality_for_path(audio_path) is not MediaModality.AUDIO:
            raise ValueError(f"control_audio_path_{index} must be audio: {audio_path}")
        assets.append(
            MediaAsset(
                video_path,
                MediaModality.VIDEO,
                "reference",
                metadata={"audio_path": str(audio_path)},
            )
        )
    return tuple(assets)


def _reference_modes(record: dict[str, Any]) -> dict[int, str]:
    modes: dict[int, str] = {}
    for key, value in record.items():
        match = _CONTROL_MODALITY_PATTERN.fullmatch(key)
        if not match or value is None:
            continue
        mode = str(value).lower()
        if mode not in {"av", "video", "audio"}:
            raise ValueError(f"{key} must be av, video, or audio")
        modes[int(match.group(1))] = mode
    return modes


def _select_reference_modality(asset: MediaAsset, mode: str, *, index: int) -> MediaAsset:
    if mode == "av":
        if asset.modality is not MediaModality.VIDEO:
            raise ValueError(f"control_modality_{index}=av requires a video reference")
        return asset
    if mode == "video":
        if asset.modality not in {MediaModality.IMAGE, MediaModality.VIDEO}:
            raise ValueError(f"control_modality_{index}=video requires an image or video reference")
        metadata = dict(asset.metadata)
        metadata["include_audio"] = False
        metadata.pop("audio_path", None)
        return MediaAsset(asset.path, asset.modality, asset.role, metadata=metadata)
    audio_path = Path(asset.metadata.get("audio_path", asset.path))
    if _modality_for_path(audio_path) not in {MediaModality.VIDEO, MediaModality.AUDIO}:
        raise ValueError(f"control_modality_{index}=audio requires a video or audio reference")
    return MediaAsset(audio_path, MediaModality.AUDIO, asset.role)


def _control_sort_key(path: Path, target_stem: str) -> int:
    if path.stem == target_stem:
        return 0
    suffix = path.stem.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"invalid numbered control suffix: {path.name}")
    return int(suffix) + 1


def _stem_matches(paths: Sequence[Path], stem: str) -> list[Path]:
    return [path for path in paths if path.stem == stem or path.stem.startswith(stem + "_")]


def _fallback_stem(stem: str) -> str | None:
    prefix, separator, suffix = stem.rpartition("_")
    return prefix if separator and suffix.isdigit() and prefix else None


def _references_from_directory(control_directory: str, target_paths: Sequence[str]) -> dict[str, tuple[Path, ...]]:
    """Match ``control_directory`` files to targets without letting one target steal another's controls.

    Controls for target ``X`` are ``X.<ext>`` or ``X_<n>.<ext>`` (Musubi's rule). A target whose own
    stem carries a numeric suffix (``X_0``) and that finds no direct match falls back to the ``X``
    prefix, but only over controls no other target owns directly, and never over controls a second
    fallback target would claim as well.
    """
    root = Path(control_directory)
    if not root.is_dir():
        raise ValueError(f"control_directory does not exist: {root}")
    allowed = {extension.lower() for extension in (*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS, *AUDIO_EXTENSIONS)}
    available = sorted((path for path in root.iterdir() if path.is_file() and path.suffix.lower() in allowed), key=str)
    # Longer target names first so `scene_00.mp4` claims `scene_00_0.png` before `scene.mp4` can.
    ordered_targets = sorted(target_paths, key=lambda path: (-len(Path(path).name), str(path)))

    owned: set[Path] = set()
    matched: dict[str, tuple[str, list[Path]]] = {}
    for target in ordered_targets:
        stem = Path(target).stem
        matches = [path for path in _stem_matches(available, stem) if path not in owned]
        if matches:
            owned.update(matches)
            matched[target] = (stem, matches)

    claims: dict[Path, list[str]] = {}
    for target in ordered_targets:
        if target in matched:
            continue
        stem = _fallback_stem(Path(target).stem)
        matches = [path for path in _stem_matches(available, stem) if path not in owned] if stem else []
        if not matches:
            raise ValueError(f"no matching H3 controls for {target!r} in {root}")
        matched[target] = (stem, matches)
        for path in matches:
            claims.setdefault(path, []).append(target)
    contested = {path: targets for path, targets in claims.items() if len(targets) > 1}
    if contested:
        details = "; ".join(f"{path.name} claimed by {sorted(targets)}" for path, targets in sorted(contested.items(), key=str))
        raise ValueError(f"ambiguous H3 controls in {root}: {details}")

    result: dict[str, tuple[Path, ...]] = {}
    for target in target_paths:
        stem, matches = matched[target]
        matches = sorted(matches, key=lambda path: _control_sort_key(path, stem))
        order = [_control_sort_key(path, stem) for path in matches]
        if len(order) != len(set(order)):
            raise ValueError(f"multiple H3 controls occupy the same index for {target!r}: {matches}")
        result[target] = tuple(matches)
    return result


def _paired_references_from_directories(
    video_directory: str,
    audio_directory: str,
    target_paths: Sequence[str],
) -> dict[str, tuple[MediaAsset, ...]]:
    video_paths = _references_from_directory(video_directory, target_paths)
    audio_paths = _references_from_directory(audio_directory, target_paths)
    result = {}
    for target in target_paths:
        videos = video_paths[target]
        audios = audio_paths[target]
        if len(videos) != 1 or len(audios) != 1:
            raise ValueError(f"paired H3 reference directories require one video and one audio match for {target!r}")
        video_path, audio_path = videos[0], audios[0]
        if _modality_for_path(video_path) is not MediaModality.VIDEO:
            raise ValueError(f"control_video_directory must contain video references, got {video_path}")
        if _modality_for_path(audio_path) is not MediaModality.AUDIO:
            raise ValueError(f"control_audio_directory must contain audio references, got {audio_path}")
        result[target] = (
            MediaAsset(
                video_path,
                MediaModality.VIDEO,
                "reference",
                metadata={"audio_path": str(audio_path)},
            ),
        )
    return result


def _dataset_reference_modes(source: dict[str, Any], general: dict[str, Any], count: int) -> tuple[str | None, ...]:
    mode = _effective(source, general, "control_modality")
    modes = _effective(source, general, "control_modalities")
    if mode is not None and modes is not None:
        raise ValueError("use control_modality or control_modalities, not both")
    if modes is not None:
        if not isinstance(modes, list) or len(modes) != count:
            raise ValueError(f"control_modalities must contain exactly {count} entries")
        values = tuple(str(value).lower() for value in modes)
    elif mode is not None:
        values = (str(mode).lower(),) * count
    else:
        return (None,) * count
    if any(value not in {"av", "video", "audio"} for value in values):
        raise ValueError("control modality values must be av, video, or audio")
    return values


def _dataset_reference_probabilities(source: dict[str, Any], general: dict[str, Any]) -> tuple[float, float, float] | None:
    values = _effective(source, general, "control_modality_probabilities")
    if values is None:
        return None
    if _effective(source, general, "control_modality") is not None or _effective(source, general, "control_modalities") is not None:
        raise ValueError("control_modality_probabilities cannot be combined with control_modality/control_modalities")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("control_modality_probabilities must be [av, video, audio]")
    probabilities = tuple(float(value) for value in values)
    if any(value < 0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
        raise ValueError("control_modality_probabilities must be non-negative and sum to 1")
    return probabilities


def _validate_reference_modality_probabilities(
    target: str,
    references: Sequence[MediaAsset],
    probabilities: tuple[float, float, float],
) -> None:
    """Reject unsatisfiable variants at configuration time.

    ``reference_modality_variant`` keeps image references in every variant, keeps video references
    (audio stripped) in the ``video`` variant, and turns video references into audio rows in the
    ``audio`` variant. Each variant must retain one non-audio reference, so ``video`` needs an image
    or video reference and ``audio`` needs an image reference.
    """
    if not references:
        raise ValueError("control_modality_probabilities requires at least one reference")
    has_image = any(reference.modality is MediaModality.IMAGE for reference in references)
    has_visual = has_image or any(reference.modality is MediaModality.VIDEO for reference in references)
    requirements = (
        ("av", probabilities[0], has_visual, "an image or video reference"),
        ("video", probabilities[1], has_visual, "an image or video reference"),
        ("audio", probabilities[2], has_image, "an image reference"),
    )
    for name, probability, satisfied, requirement in requirements:
        if probability > 0 and not satisfied:
            kinds = sorted({reference.modality.name.lower() for reference in references})
            raise ValueError(
                f"control_modality_probabilities gives the {name} variant probability {probability} "
                f"but {target!r} has only {', '.join(kinds)} reference(s); the {name} variant needs {requirement}"
            )


def _read_media_jsonl(path: str, *, resolve_paths: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                base = Path(path).expanduser().resolve().parent

                def resolve(value: Any) -> str:
                    media_path = Path(value).expanduser()
                    if not media_path.is_absolute():
                        media_path = base / media_path
                    return str(media_path.resolve())

                # Every relative control path resolves against the JSONL's directory, whatever the
                # dataset kind: control_path[_N] follows control_video_path_N/control_audio_path_N.
                for key in tuple(record):
                    if (
                        key == "control_path"
                        or _CONTROL_PATH_PATTERN.fullmatch(key)
                        or _CONTROL_VIDEO_PATH_PATTERN.fullmatch(key)
                        or _CONTROL_AUDIO_PATH_PATTERN.fullmatch(key)
                    ):
                        if record[key]:
                            record[key] = resolve(record[key])
                if resolve_paths:
                    for key in tuple(record):
                        if key in {"image_path", "video_path"} or re.fullmatch(r"image_path_\d+", key):
                            record[key] = resolve(record[key])
                records.append(record)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number} of {path}: {error}") from error
    return records


def _effective(dataset: dict[str, Any], general: dict[str, Any], key: str) -> Any:
    value = dataset.get(key)
    return general.get(key) if value is None else value


@dataclass(frozen=True)
class _ResolvedTarget:
    path: Path
    references: tuple[MediaAsset, ...]


class H3DatasetAdapter:
    """Resolve H3 media while leaving Musubi's shared dataset implementation unchanged.

    Musubi's existing ``control_directory`` and ``control_path[_N]`` fields are
    canonical. The adapter removes them only from the in-memory config passed to
    ``VideoDataset`` so arbitrary H3 references are not mistaken for aligned
    ControlNet video, then attaches them to each ItemInfo at the H3 boundary.
    """

    def __init__(self, user_config: dict[str, Any], args: Any | None = None):
        self.musubi_config = copy.deepcopy(user_config)
        self._targets: dict[str, _ResolvedTarget] = {}
        self._target_groups: list[tuple[str, ...]] = []
        self._target_fps: dict[str, float] = {}
        self._target_modalities: dict[str, MediaModality] = {}
        self._target_modes: dict[str, str] = {}
        self._target_source_paths: dict[str, tuple[Path, ...]] = {}
        self._target_reference_probabilities: dict[str, tuple[float, float, float]] = {}
        self._image_frame_counts: dict[str, int] = {}
        self.image_mode = str(getattr(args, "h3_image_mode", "none"))
        self.image_frame_count = getattr(args, "h3_image_frame_count", None)
        if self.image_mode not in {"none", "first", "first_last"}:
            raise ValueError("MiniMax H3 image mode must be none, first, or first_last")
        self.audio_datasets: list[H3AudioDataset] = []
        self.dataset_kinds: list[str] = []
        general = user_config.get("general", {})
        clean_general = self.musubi_config.get("general", {})
        clean_general.pop("control_directory", None)
        for key in (
            "control_video_directory",
            "control_audio_directory",
            "control_modality",
            "control_modalities",
            "control_modality_probabilities",
        ):
            clean_general.pop(key, None)
        for key in ("h3_target_mode", "h3_image_frame_count", "audio_directory", "audio_jsonl_file"):
            clean_general.pop(key, None)

        source_datasets = user_config.get("datasets", [])
        clean_datasets = self.musubi_config.get("datasets", [])
        for source, clean in zip(source_datasets, clean_datasets):
            target_mode = _effective(source, general, "h3_target_mode") or "av"
            if target_mode not in {"av", "video", "audio"}:
                raise ValueError("h3_target_mode must be av, video, or audio")
            clean.pop("h3_target_mode", None)
            configured_image_frames = _effective(source, general, "h3_image_frame_count")
            clean.pop("h3_image_frame_count", None)
            audio_directory = _effective(source, general, "audio_directory")
            audio_jsonl_file = _effective(source, general, "audio_jsonl_file")
            clean.pop("audio_directory", None)
            clean.pop("audio_jsonl_file", None)
            if audio_directory or audio_jsonl_file:
                if target_mode != "audio":
                    raise ValueError("audio_directory/audio_jsonl_file require h3_target_mode = 'audio'")
                audio_dataset = H3AudioDataset(source, general)
                self.audio_datasets.append(audio_dataset)
                self.dataset_kinds.append("audio")
                paths = tuple(str(path) for path, _ in audio_dataset.records)
                for target in paths:
                    normal = _normal_path(target)
                    self._targets[normal] = _ResolvedTarget(Path(target), ())
                    self._target_modalities[normal] = MediaModality.AUDIO
                    self._target_modes[normal] = "audio"
                self._target_groups.append(tuple(_normal_path(target) for target in paths))
                continue
            if target_mode == "audio":
                raise ValueError("h3_target_mode = 'audio' requires audio_directory or audio_jsonl_file")
            self.dataset_kinds.append("regular")
            control_directory = _effective(source, general, "control_directory")
            clean.pop("control_directory", None)
            control_video_directory = _effective(source, general, "control_video_directory")
            control_audio_directory = _effective(source, general, "control_audio_directory")
            for key in (
                "control_video_directory",
                "control_audio_directory",
                "control_modality",
                "control_modalities",
                "control_modality_probabilities",
            ):
                clean.pop(key, None)
            if bool(control_video_directory) != bool(control_audio_directory):
                raise ValueError("control_video_directory and control_audio_directory must be specified together")
            video_directory = _effective(source, general, "video_directory")
            video_jsonl_file = _effective(source, general, "video_jsonl_file")
            image_directory = _effective(source, general, "image_directory")
            image_jsonl_file = _effective(source, general, "image_jsonl_file")
            caption_extension = _effective(source, general, "caption_extension")
            multiple_target = bool(_effective(source, general, "multiple_target"))

            records: list[dict[str, Any]] | None = None
            if video_directory:
                target_frames = _effective(source, general, "target_frames")
                invalid_frames = [frame_count for frame_count in target_frames or () if not is_valid_frame_count(frame_count)]
                if invalid_frames:
                    raise ValueError(
                        f"MiniMax H3 video target_frames must satisfy frame_count % 17 == 5; invalid values: {invalid_frames}"
                    )
                target_paths = tuple(glob_videos(video_directory))
            elif video_jsonl_file:
                target_frames = _effective(source, general, "target_frames")
                invalid_frames = [frame_count for frame_count in target_frames or () if not is_valid_frame_count(frame_count)]
                if invalid_frames:
                    raise ValueError(
                        f"MiniMax H3 video target_frames must satisfy frame_count % 17 == 5; invalid values: {invalid_frames}"
                    )
                records = _read_media_jsonl(video_jsonl_file)
                target_paths = tuple(record["video_path"] for record in records)
            elif image_directory:
                target_paths = list(glob_images(image_directory, caption_extension=caption_extension))
                all_images = tuple(Path(path) for path in glob_images(image_directory))
                if self.image_mode != "none" and multiple_target and caption_extension:
                    existing = set(target_paths)
                    groups: dict[Path, list[tuple[int, Path]]] = {}
                    for candidate in all_images:
                        prefix, separator, suffix = candidate.stem.rpartition("_")
                        if separator and suffix.isdigit():
                            groups.setdefault(candidate.with_name(prefix), []).append((int(suffix), candidate))
                    for prefix, candidates in groups.items():
                        candidates.sort()
                        if (prefix.parent / (prefix.name + caption_extension)).is_file() and candidates[0][0] in (0, 1):
                            primary = str(candidates[0][1])
                            if primary not in existing:
                                target_paths.append(primary)
                                existing.add(primary)
                target_paths = tuple(sorted(target_paths))
                target_sources = {}
                for target_path in target_paths:
                    primary = Path(target_path)
                    indexed = []
                    match_stem = primary.stem
                    prefix, separator, suffix = primary.stem.rpartition("_")
                    if separator and suffix.isdigit() and (primary.parent / (prefix + (caption_extension or ""))).is_file():
                        match_stem = prefix
                    for candidate in all_images:
                        if not candidate.stem.startswith(match_stem + "_") or candidate == primary:
                            continue
                        suffix = candidate.stem[len(match_stem) + 1 :]
                        if suffix.isdigit():
                            indexed.append((int(suffix), candidate))
                    indexed.sort()
                    target_sources[target_path] = (primary, *(path for _, path in indexed)) if multiple_target else (primary,)
            elif image_jsonl_file:
                records = _read_media_jsonl(image_jsonl_file, resolve_paths=self.image_mode != "none")
                target_paths = tuple(record.get("image_path") or record.get("image_path_0") for record in records)
                if any(not target for target in target_paths):
                    raise ValueError("H3 image JSONL records must contain image_path or image_path_0")
                target_sources = {}
                for target_path, record in zip(target_paths, records):
                    paths = [Path(target_path)]
                    if multiple_target:
                        index = 1
                        while record.get(f"image_path_{index}") is not None:
                            paths.append(Path(record[f"image_path_{index}"]))
                            index += 1
                    target_sources[target_path] = tuple(paths)
            else:
                raise ValueError("MiniMax H3 requires a Musubi image or video dataset")

            if self.image_mode != "none":
                if not (image_directory or image_jsonl_file):
                    raise ValueError("MiniMax H3 conditioned-image mode accepts image datasets only")
                frame_count = self.image_frame_count if self.image_frame_count is not None else (configured_image_frames or 5)
                validate_image_mode(self.image_mode, frame_count)
                clean["h3_image_frame_count"] = int(frame_count)
            else:
                frame_count = None

            if (
                (control_directory or control_video_directory)
                and records
                and any(_ordered_control_paths(record) or _paired_control_assets(record) for record in records)
            ):
                raise ValueError("specify H3 controls in control_directory or video JSONL, not both")
            if control_directory or control_video_directory:
                ordinary_reference_paths = _references_from_directory(control_directory, target_paths) if control_directory else {}
                paired_reference_paths = (
                    _paired_references_from_directories(
                        control_video_directory,
                        control_audio_directory,
                        target_paths,
                    )
                    if control_video_directory
                    else {}
                )
                reference_paths = {
                    target: (*ordinary_reference_paths.get(target, ()), *paired_reference_paths.get(target, ()))
                    for target in target_paths
                }
            elif records is not None:
                reference_paths = {}
                for record in records:
                    controls = _ordered_control_paths(record)
                    paired_controls = _paired_control_assets(record)
                    modes = _reference_modes(record)
                    if controls or paired_controls:
                        target_key = record.get("video_path") or record.get("image_path") or record.get("image_path_0")
                        if target_key is None:
                            raise ValueError("H3 media JSONL records must contain a target path")
                        ordinary_assets = tuple(MediaAsset(path, _modality_for_path(path), "reference") for path in controls)
                        references = (*ordinary_assets, *paired_controls)
                        unknown_modes = sorted(set(modes) - set(range(len(references))))
                        if unknown_modes:
                            raise ValueError(f"control_modality_N has no matching reference indices: {unknown_modes}")
                        reference_paths[target_key] = tuple(
                            _select_reference_modality(reference, modes[index], index=index) if index in modes else reference
                            for index, reference in enumerate(references)
                        )
            else:
                reference_paths = {}

            for target in target_paths:
                modality = _modality_for_path(Path(target))
                if (video_directory or video_jsonl_file) and modality is not MediaModality.VIDEO:
                    raise ValueError(f"MiniMax H3 video datasets must resolve video targets, got {target}")
                if (image_directory or image_jsonl_file) and modality is not MediaModality.IMAGE:
                    raise ValueError(f"MiniMax H3 image datasets must resolve image targets, got {target}")
                raw_references = reference_paths.get(target, ())
                references = tuple(
                    reference
                    if isinstance(reference, MediaAsset)
                    else MediaAsset(reference, _modality_for_path(reference), "reference")
                    for reference in raw_references
                )
                modes = _dataset_reference_modes(source, general, len(references))
                probabilities = _dataset_reference_probabilities(source, general)
                if probabilities is not None:
                    _validate_reference_modality_probabilities(target, references, probabilities)
                references = tuple(
                    _select_reference_modality(reference, mode, index=index) if mode is not None else reference
                    for index, (reference, mode) in enumerate(zip(references, modes))
                )
                resolved = _ResolvedTarget(Path(target), references)
                normal = _normal_path(target)
                existing = self._targets.get(normal)
                if existing is not None:
                    if existing != resolved or self._target_modalities[normal] is not modality:
                        raise ValueError(f"conflicting H3 media metadata for target path across datasets: {target}")
                    if self._target_modes[normal] != target_mode:
                        raise ValueError(f"conflicting h3_target_mode for target path across datasets: {target}")
                    if self._target_reference_probabilities.get(normal) != probabilities:
                        raise ValueError(f"conflicting control_modality_probabilities for target path: {target}")
                else:
                    # A source directory may intentionally be repeated with different
                    # resolutions and cache directories. Those dataset-level settings
                    # do not change the H3 media attached to an item.
                    self._targets[normal] = resolved
                    self._target_modalities[normal] = modality
                    self._target_modes[normal] = target_mode
                    self._target_source_paths[normal] = (
                        target_sources[target] if (image_directory or image_jsonl_file) else (Path(target),)
                    )
                    if frame_count is not None:
                        self._image_frame_counts[normal] = int(frame_count)
                    if probabilities is not None:
                        self._target_reference_probabilities[normal] = probabilities
            self._target_groups.append(tuple(_normal_path(target) for target in target_paths))

        self.requires_audio = any(
            modality in {MediaModality.VIDEO, MediaModality.AUDIO} and self._target_modes[path] != "video"
            for path, modality in self._target_modalities.items()
        ) or any(
            reference.modality is MediaModality.AUDIO or bool(reference.metadata.get("audio_path"))
            for resolved in self._targets.values()
            for reference in resolved.references
        )
        self.requires_video = any(mode != "audio" for mode in self._target_modes.values()) or any(
            reference.modality in {MediaModality.IMAGE, MediaModality.VIDEO}
            for resolved in self._targets.values()
            for reference in resolved.references
        )

    def adapt_dataset_group(self, dataset_group: DatasetGroup) -> None:
        if len(dataset_group.datasets) != len(self._target_groups):
            raise ValueError("H3 dataset adapter and Musubi dataset group have different lengths")
        for dataset, target_group in zip(dataset_group.datasets, self._target_groups):
            if isinstance(dataset, H3AudioDataset):
                continue
            datasource = dataset.datasource
            if hasattr(datasource, "data"):
                for record in datasource.data:
                    for key in tuple(record):
                        if (
                            key == "control_path"
                            or _CONTROL_PATH_PATTERN.fullmatch(key)
                            or _CONTROL_VIDEO_PATH_PATTERN.fullmatch(key)
                            or _CONTROL_AUDIO_PATH_PATTERN.fullmatch(key)
                            or _CONTROL_MODALITY_PATTERN.fullmatch(key)
                        ):
                            record.pop(key, None)
                datasource.has_control = False
            dataset.control_directory = None
            dataset.has_control = False
            image_frame_counts = {self._image_frame_counts[target] for target in target_group if target in self._image_frame_counts}
            if image_frame_counts:
                if len(image_frame_counts) != 1:
                    raise ValueError("one H3 image dataset cannot use multiple image frame counts")
                dataset.h3_image_frame_count = image_frame_counts.pop()
            for target in target_group:
                if self._target_modalities[target] is MediaModality.VIDEO:
                    existing_fps = self._target_fps.get(target)
                    if existing_fps is not None and existing_fps != dataset.target_fps:
                        raise ValueError(f"conflicting target FPS for H3 target path across datasets: {target}")
                    self._target_fps[target] = dataset.target_fps

    def _resolve_target(self, item_key: str) -> tuple[_ResolvedTarget, int | None, int | None]:
        exact = self._targets.get(_normal_path(item_key))
        if exact is not None:
            return exact, None, None

        item = Path(item_key)
        match = _CROP_SUFFIX_PATTERN.fullmatch(item.stem)
        if match:
            candidate = item.with_name(match.group("stem") + item.suffix)
            resolved = self._targets.get(_normal_path(candidate))
            if resolved is not None:
                return resolved, int(match.group("start")), int(match.group("frames"))
        raise KeyError(f"H3 dataset adapter cannot map ItemInfo key to source media: {item_key}")

    def attach(self, item: ItemInfo) -> tuple[MediaAsset, ...]:
        resolved, start_frame, frame_count = self._resolve_target(item.item_key)
        normal = _normal_path(resolved.path)
        modality = self._target_modalities[normal]
        target_fps = self._target_fps.get(normal)
        target_frame_count = frame_count if frame_count is not None else item.frame_count
        image_frame_count = self._image_frame_counts.get(normal)
        target = MediaAsset(
            resolved.path,
            modality,
            "target",
            metadata=(
                {"frame_count": target_frame_count, "fps": target_fps}
                if modality in {MediaModality.VIDEO, MediaModality.AUDIO}
                else {"frame_count": image_frame_count or 1}
            ),
        )
        if modality is MediaModality.VIDEO and start_frame is not None and frame_count is not None:
            target = slice_media_asset(
                target,
                start_seconds=start_frame / target_fps,
                duration_seconds=frame_count / target_fps,
            )
        if image_frame_count is not None:
            controls = condition_paths(self.image_mode, tuple(reference.path for reference in resolved.references))
            if any(reference.modality is not MediaModality.IMAGE for reference in resolved.references):
                raise ValueError("MiniMax H3 conditioned-image controls must all be images")
            item.h3_image_mode = self.image_mode
            item.h3_image_frame_count = image_frame_count
            item.h3_condition_paths = controls
            item.h3_target_paths = self._target_source_paths[normal]
            item.content = resample_image_targets(item.content, image_frame_count)
            item.frame_count = image_frame_count
            item.h3_cache_metadata = {
                "sample_fingerprint": sample_fingerprint(
                    targets=item.h3_target_paths,
                    controls=controls,
                    mode=self.image_mode,
                    frame_count=image_frame_count,
                    original_size=item.original_size,
                    bucket_size=item.bucket_size,
                ),
                "h3_image_mode": self.image_mode,
            }
            assets = (target,)
        else:
            assets = (target, *resolved.references)
        validate_h3_media_assets(item.item_key, assets)
        item.h3_media_assets = assets
        fingerprint = reference_fingerprint(assets)
        if fingerprint is not None:
            item.h3_cache_metadata = {REFERENCE_FINGERPRINT_KEY: fingerprint}
        if normal in self._target_reference_probabilities:
            item.h3_reference_modality_probabilities = self._target_reference_probabilities[normal]
        item.h3_target_mode = "video" if image_frame_count is not None else self._target_modes[normal]
        return assets


def create_h3_dataset_group(
    user_config: dict[str, Any],
    args: Any,
    *,
    training: bool = False,
    num_timestep_buckets: int | None = None,
    shared_epoch: Any = None,
) -> tuple[DatasetGroup, H3DatasetAdapter]:
    adapter = H3DatasetAdapter(user_config, args)
    regular_config = copy.deepcopy(adapter.musubi_config)
    regular_config["datasets"] = [
        dataset for dataset, kind in zip(regular_config.get("datasets", []), adapter.dataset_kinds) if kind == "regular"
    ]
    regular_datasets = []
    if regular_config["datasets"]:
        blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
            regular_config,
            args,
            architecture=ARCHITECTURE_MINIMAX_H3,
        )
        regular_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.dataset_group,
            training=training,
            num_timestep_buckets=num_timestep_buckets,
            shared_epoch=shared_epoch,
        )
        regular_datasets = list(regular_group.datasets)
    regular_iter = iter(regular_datasets)
    audio_iter = iter(adapter.audio_datasets)
    ordered_datasets = [next(audio_iter) if kind == "audio" else next(regular_iter) for kind in adapter.dataset_kinds]
    seed = random.randint(0, 2**31)
    for dataset in adapter.audio_datasets:
        dataset.set_seed(seed, shared_epoch)
        if training:
            dataset.prepare_for_training(num_timestep_buckets=num_timestep_buckets)
    if training:
        load_dino_features = bool(getattr(args, "h3_load_dino_features", False))
        for dataset in ordered_datasets:
            batch_manager = getattr(dataset, "batch_manager", None)
            if batch_manager is not None:
                batch_manager.load_h3_dino_features = load_dino_features and isinstance(dataset, VideoDataset)
                batch_manager.h3_dino_model = getattr(args, "h3_dino_model", None) if load_dino_features else None
    dataset_group = DatasetGroup(ordered_datasets)
    adapter.adapt_dataset_group(dataset_group)
    return dataset_group, adapter


def attach_h3_media(
    batch: Sequence[ItemInfo],
    adapter: H3DatasetAdapter,
) -> None:
    for item in batch:
        adapter.attach(item)


def validate_h3_media_assets(key: str, assets: tuple[MediaAsset, ...], *, check_files: bool = True) -> None:
    targets = tuple(asset for asset in assets if asset.role == "target")
    if len(targets) != 1 or targets[0].modality not in {MediaModality.IMAGE, MediaModality.VIDEO, MediaModality.AUDIO}:
        raise ValueError(f"H3 item {key!r} must contain exactly one target image, video, or audio clip")
    unsupported_roles = sorted({asset.role for asset in assets if asset.role not in {"target", "reference"}})
    if unsupported_roles:
        raise ValueError(f"H3 item {key!r} contains unsupported roles: {unsupported_roles}")
    if check_files:
        media_paths = {asset.path for asset in assets}
        media_paths.update(Path(asset.metadata["audio_path"]) for asset in assets if asset.metadata.get("audio_path"))
        missing = sorted(str(path) for path in media_paths if not path.is_file())
        if missing:
            raise FileNotFoundError(f"media file(s) not found for H3 item {key!r}: {', '.join(missing)}")
