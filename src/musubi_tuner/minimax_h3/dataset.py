from __future__ import annotations

import copy
import json
import logging
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
from musubi_tuner.minimax_h3.cache import QWEN_CONTROL_FINGERPRINT_KEY, QWEN_CONTROL_ROLE, qwen_control_fingerprint
from musubi_tuner.minimax_h3.image_training import condition_paths, resample_image_targets, sample_fingerprint, validate_image_mode
from musubi_tuner.minimax_h3.media import MediaAsset, MediaModality, slice_media_asset
from musubi_tuner.minimax_h3.references import REFERENCE_FINGERPRINT_KEY, reference_fingerprint

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus")
CONDITIONING_MASK_KEY = "conditioning_mask_path"
CONDITIONING_MASK_DIRECTORY_KEY = "conditioning_mask_directory"
QWEN_CONTROL_DIRECTORY_KEY = "qwen_control_directory"
QWEN_CONTROL_KEY = "qwen_control_path"
TARGET_AUDIO_FINGERPRINT_KEY = "target_audio_fingerprint"
SOURCE_DIRECTORY_KEYS = {
    "image": "source_image_directory",
    "video": "source_video_directory",
    "audio": "source_audio_directory",
}
TARGET_DIRECTORY_KEYS = {
    "image": "target_image_directory",
    "video": "target_video_directory",
    "audio": "target_audio_directory",
}
LEGACY_H3_DATASET_KEYS = frozenset(
    {
        "image_directory",
        "video_directory",
        "audio_directory",
        "control_directory",
        "control_video_directory",
        "control_audio_directory",
        "control_modality",
        "control_modalities",
        "control_modality_probabilities",
        "h3_target_mode",
    }
)
logger = logging.getLogger(__name__)
_CONTROL_PATH_PATTERN = re.compile(r"^control_path_(\d+)$")
_QWEN_CONTROL_PATH_PATTERN = re.compile(r"^qwen_control_path_(\d+)$")
_CONTROL_VIDEO_PATH_PATTERN = re.compile(r"^control_video_path_(\d+)$")
_CONTROL_AUDIO_PATH_PATTERN = re.compile(r"^control_audio_path_(\d+)$")
_CONTROL_MODALITY_PATTERN = re.compile(r"^control_modality_(\d+)$")
_CROP_SUFFIX_PATTERN = re.compile(r"^(?P<stem>.+)_(?P<start>\d{5})-(?P<frames>\d+)$")


def _normal_path(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _modality_list(value: Any, key: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty TOML array")
    modalities = tuple(str(item).lower() for item in value)
    if len(set(modalities)) != len(modalities):
        raise ValueError(f"{key} must not contain duplicate modalities")
    unknown = sorted(set(modalities) - {"image", "video", "audio"})
    if unknown:
        raise ValueError(f"{key} contains unsupported modalities: {unknown}")
    return modalities


def _normalize_explicit_modality_config(user_config: dict[str, Any]) -> dict[str, Any]:
    """Validate H3's TOML schema and translate it for Musubi's shared datasets.

    The translated carrier fields exist only in the private copied config passed to
    Musubi's shared image/video implementation. They are not accepted from users.
    """
    sections = [("general", user_config.get("general", {}))]
    sections.extend((f"datasets[{index}]", dataset) for index, dataset in enumerate(user_config.get("datasets", [])))
    for section_name, section in sections:
        found = sorted(key for key in LEGACY_H3_DATASET_KEYS if section.get(key) is not None)
        if found:
            raise ValueError(
                f"{section_name} uses removed MiniMax H3 dataset keys: {', '.join(found)}. "
                "Migrate to target_*_directory + target_modalities and "
                "source_*_directory + source_modalities; use source_modality_probabilities for source dropout."
            )

    normalized = copy.deepcopy(user_config)
    general = normalized.get("general", {})
    for index, dataset in enumerate(normalized.get("datasets", [])):
        source_probabilities = dataset.get("source_modality_probabilities")
        if source_probabilities is not None:
            dataset["control_modality_probabilities"] = source_probabilities
        target_dirs = {kind: dataset.get(key) for kind, key in TARGET_DIRECTORY_KEYS.items() if dataset.get(key)}
        jsonl_kinds = tuple(kind for kind in ("image", "video", "audio") if dataset.get(f"{kind}_jsonl_file"))
        for kind in ("image", "video", "audio"):
            if kind in target_dirs and kind in jsonl_kinds:
                raise ValueError(
                    f"datasets[{index}] cannot combine target_{kind}_directory and {kind}_jsonl_file; "
                    "use one target carrier per dataset section"
                )
        primary_kinds = tuple(kind for kind in ("image", "video") if kind in target_dirs or kind in jsonl_kinds)
        if len(primary_kinds) > 1 or (primary_kinds and "audio" in jsonl_kinds):
            raise ValueError(f"datasets[{index}] must declare exactly one image, video, or audio target carrier")
        if primary_kinds:
            target_kind = primary_kinds[0]
        elif "audio" in target_dirs or "audio" in jsonl_kinds:
            target_kind = "audio"
        else:
            raise ValueError(
                f"datasets[{index}] requires target_image_directory, target_video_directory, target_audio_directory, "
                "or one *_jsonl_file target"
            )
        carrier_path = target_dirs.get(target_kind)
        if carrier_path is not None:
            dataset[f"{target_kind}_directory"] = carrier_path
        requested = _modality_list(dataset.get("target_modalities"), "target_modalities")
        if requested is None:
            raise ValueError(f"datasets[{index}] requires target_modalities")
        allowed = {
            "image": {("image",), ("image", "audio")},
            "video": {("video",), ("video", "audio")},
            "audio": {("audio",)},
        }[target_kind]
        target_modalities = requested
        if target_modalities not in allowed:
            raise ValueError(
                f"datasets[{index}] {target_kind} target carrier does not support target_modalities={list(target_modalities)}"
            )
        if "audio" in target_modalities and "audio" not in target_dirs and target_kind == "image":
            raise ValueError(f"datasets[{index}] image+audio targets require target_audio_directory")
        if "audio" in target_dirs and "audio" not in target_modalities:
            raise ValueError(f"datasets[{index}] target_audio_directory requires audio in target_modalities")
        resolved_mode = (
            "video" if target_modalities in {("image",), ("video",)} else ("audio" if target_modalities == ("audio",) else "av")
        )
        dataset["h3_target_mode"] = resolved_mode

        declared_sources = tuple(kind for kind, key in SOURCE_DIRECTORY_KEYS.items() if dataset.get(key))
        requested_sources = _modality_list(dataset.get("source_modalities"), "source_modalities")
        if declared_sources and requested_sources is None:
            raise ValueError(f"datasets[{index}] requires source_modalities when source_*_directory is configured")
        embedded_av = bool(dataset.get("source_video_audio_embedded"))
        paired_av = bool(dataset.get("source_video_audio_paired"))
        if embedded_av and paired_av:
            raise ValueError(f"datasets[{index}] cannot combine source_video_audio_embedded and source_video_audio_paired")
        if embedded_av and ("video" not in declared_sources or "audio" in declared_sources):
            raise ValueError(
                f"datasets[{index}] source_video_audio_embedded requires source_video_directory and no source_audio_directory"
            )
        expected_source_sets = {frozenset(declared_sources)}
        if embedded_av:
            nonvideo_sources = set(declared_sources) - {"video"}
            expected_source_sets = {
                frozenset((*nonvideo_sources, "audio")),
                frozenset((*nonvideo_sources, "video", "audio")),
            }
        if requested_sources is not None and frozenset(requested_sources) not in expected_source_sets:
            detail = (
                "; with source_video_audio_embedded, the video directory may provide video+audio or audio only"
                if embedded_av
                else ""
            )
            raise ValueError(f"datasets[{index}] source_modalities do not match the configured source directories{detail}")
        if paired_av and not {"video", "audio"}.issubset(declared_sources):
            raise ValueError(f"datasets[{index}] source_video_audio_paired requires video and audio source directories")
    return normalized


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


def _ordered_qwen_control_paths(record: dict[str, Any]) -> tuple[Path, ...]:
    """Parse the EXPERIMENTAL per-record ``qwen_control_path[_N]`` fields."""
    numbered: list[tuple[int, Path]] = []
    for key, value in record.items():
        match = _QWEN_CONTROL_PATH_PATTERN.fullmatch(key)
        if match and value:
            numbered.append((int(match.group(1)), Path(value)))
    if record.get(QWEN_CONTROL_KEY):
        if numbered:
            raise ValueError(f"use {QWEN_CONTROL_KEY} or {QWEN_CONTROL_KEY}_N, not both")
        return (Path(record[QWEN_CONTROL_KEY]),)
    numbered.sort(key=lambda value: value[0])
    indices = [index for index, _ in numbered]
    if indices != list(range(len(indices))):
        raise ValueError(f"{QWEN_CONTROL_KEY}_N indices must be contiguous from zero, got {indices}")
    return tuple(path for _, path in numbered)


def _qwen_control_assets(paths: Sequence[Path]) -> tuple[MediaAsset, ...]:
    assets = []
    for path in paths:
        modality = _modality_for_path(Path(path))
        if modality is MediaModality.AUDIO:
            raise ValueError(f"H3 Qwen control visuals must be images or videos, got audio: {path}")
        assets.append(MediaAsset(Path(path), modality, QWEN_CONTROL_ROLE))
    return tuple(assets)


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


def _references_from_directory(media_directory: str, target_paths: Sequence[str]) -> dict[str, tuple[Path, ...]]:
    """Match media files to targets without letting one target steal another's references.

    Controls for target ``X`` are ``X.<ext>`` or ``X_<n>.<ext>`` (Musubi's rule). A target whose own
    stem carries a numeric suffix (``X_0``) and that finds no direct match falls back to the ``X``
    prefix, but only over controls no other target owns directly, and never over controls a second
    fallback target would claim as well.
    """
    root = Path(media_directory)
    if not root.is_dir():
        raise ValueError(f"H3 media directory does not exist: {root}")
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
            raise ValueError(f"no basename-matched H3 media for {target!r} in {root}")
        matched[target] = (stem, matches)
        for path in matches:
            claims.setdefault(path, []).append(target)
    contested = {path: targets for path, targets in claims.items() if len(targets) > 1}
    if contested:
        details = "; ".join(f"{path.name} claimed by {sorted(targets)}" for path, targets in sorted(contested.items(), key=str))
        raise ValueError(f"ambiguous H3 media matches in {root}: {details}")

    result: dict[str, tuple[Path, ...]] = {}
    for target in target_paths:
        stem, matches = matched[target]
        matches = sorted(matches, key=lambda path: _control_sort_key(path, stem))
        order = [_control_sort_key(path, stem) for path in matches]
        if len(order) != len(set(order)):
            raise ValueError(f"multiple H3 media files occupy the same index for {target!r}: {matches}")
        result[target] = tuple(matches)
    return result


def _masks_from_directory(mask_directory: str, target_paths: Sequence[str]) -> dict[str, Path]:
    """Match ``conditioning_mask_directory`` files to targets by basename.

    One mask per target, matched on the exact stem: unlike controls, a target
    never owns several masks, so the numbered-suffix convention does not apply.
    A target without a mask is simply absent here; whether that is an error
    depends on the configured mask mode, which the adapter checks.
    """
    root = Path(mask_directory)
    if not root.is_dir():
        raise ValueError(f"conditioning_mask_directory does not exist: {root}")
    allowed = {extension.lower() for extension in IMAGE_EXTENSIONS}
    available: dict[str, Path] = {}
    for path in sorted(root.iterdir(), key=str):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if path.stem in available:
            raise ValueError(f"ambiguous H3 conditioning masks for {path.stem!r} in {root}")
        available[path.stem] = path
    return {target: available[Path(target).stem] for target in target_paths if Path(target).stem in available}


def _record_masks(records: Sequence[dict[str, Any]], target_keys: Sequence[str]) -> dict[str, Path]:
    masks: dict[str, Path] = {}
    for record, target_key in zip(records, target_keys):
        value = record.get(CONDITIONING_MASK_KEY)
        if not value:
            continue
        if not target_key:
            raise ValueError("H3 media JSONL records must contain a target path")
        masks[target_key] = Path(value)
    return masks


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
            raise ValueError(f"source_video_directory must contain video references, got {video_path}")
        if _modality_for_path(audio_path) is not MediaModality.AUDIO:
            raise ValueError(f"source_audio_directory must contain audio references, got {audio_path}")
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
        raise ValueError("source_modality_probabilities cannot be combined with fixed per-reference modality overrides")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("source_modality_probabilities must be [av, video, audio]")
    probabilities = tuple(float(value) for value in values)
    if any(value < 0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
        raise ValueError("source_modality_probabilities must be non-negative and sum to 1")
    return probabilities


def _validate_reference_modality_probabilities(
    target: str,
    references: Sequence[MediaAsset],
    probabilities: tuple[float, float, float],
) -> None:
    """Reject unsatisfiable variants at configuration time.

    ``reference_modality_variant`` keeps image references in every variant, keeps video references
    (audio stripped) in the ``video`` variant, and turns audio-bearing video references into audio
    rows in the ``audio`` variant. Each variant must retain at least one reference -- audio-only
    survivors are legal for training -- so ``video`` needs an image or video reference, while
    ``audio`` needs an image, an audio reference, or a video explicitly declared as paired/embedded AV.
    """
    if not references:
        raise ValueError("source_modality_probabilities requires at least one source reference")
    has_image = any(reference.modality is MediaModality.IMAGE for reference in references)
    has_visual = has_image or any(reference.modality is MediaModality.VIDEO for reference in references)
    has_audio = any(
        reference.modality is MediaModality.AUDIO
        or bool(reference.metadata.get("audio_path"))
        or bool(reference.metadata.get("include_audio"))
        for reference in references
    )
    requirements = (
        ("av", probabilities[0], True, "at least one reference"),
        ("video", probabilities[1], has_visual, "an image or video reference"),
        ("audio", probabilities[2], has_image or has_audio, "an image or declared audio-bearing reference"),
    )
    for name, probability, satisfied, requirement in requirements:
        if probability > 0 and not satisfied:
            kinds = sorted({reference.modality.name.lower() for reference in references})
            raise ValueError(
                f"source_modality_probabilities gives the {name} variant probability {probability} "
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
                        or key == CONDITIONING_MASK_KEY
                        or key == QWEN_CONTROL_KEY
                        or _CONTROL_PATH_PATTERN.fullmatch(key)
                        or _CONTROL_VIDEO_PATH_PATTERN.fullmatch(key)
                        or _CONTROL_AUDIO_PATH_PATTERN.fullmatch(key)
                        or _QWEN_CONTROL_PATH_PATTERN.fullmatch(key)
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


def _directory_references(
    control_directory: str | None,
    control_video_directory: str | None,
    control_audio_directory: str | None,
    target_paths: Sequence[str],
) -> dict[str, tuple[Any, ...]]:
    """Resolve ``control_directory`` / paired control directories for every target."""
    ordinary_reference_paths = _references_from_directory(control_directory, target_paths) if control_directory else {}
    paired_reference_paths = (
        _paired_references_from_directories(control_video_directory, control_audio_directory, target_paths)
        if control_video_directory
        else {}
    )
    return {target: (*ordinary_reference_paths.get(target, ()), *paired_reference_paths.get(target, ())) for target in target_paths}


def _explicit_source_references(
    source: dict[str, Any], general: dict[str, Any], target_paths: Sequence[str]
) -> dict[str, tuple[MediaAsset, ...]]:
    """Resolve independently stored image, video, and audio conditioning sides."""
    paired = bool(_effective(source, general, "source_video_audio_paired"))
    embedded = bool(_effective(source, general, "source_video_audio_embedded"))
    requested_sources = set(_effective(source, general, "source_modalities") or ())
    by_modality: dict[str, dict[str, tuple[Path, ...]]] = {}
    for kind, key in SOURCE_DIRECTORY_KEYS.items():
        if paired and kind in {"video", "audio"}:
            continue
        directory = _effective(source, general, key)
        if directory:
            by_modality[kind] = _references_from_directory(directory, target_paths)
    paired_assets = (
        _paired_references_from_directories(
            _effective(source, general, SOURCE_DIRECTORY_KEYS["video"]),
            _effective(source, general, SOURCE_DIRECTORY_KEYS["audio"]),
            target_paths,
        )
        if paired
        else {}
    )
    result = {}
    for target in target_paths:
        assets = []
        for kind in ("image", "video", "audio"):
            for path in by_modality.get(kind, {}).get(target, ()):
                modality = _modality_for_path(path)
                if modality.name.lower() != kind:
                    raise ValueError(f"{SOURCE_DIRECTORY_KEYS[kind]} must contain {kind} references, got {path}")
                if modality is MediaModality.VIDEO and embedded and "video" not in requested_sources:
                    assets.append(MediaAsset(path, MediaModality.AUDIO, "reference"))
                else:
                    metadata = {"include_audio": embedded} if modality is MediaModality.VIDEO else {}
                    assets.append(MediaAsset(path, modality, "reference", metadata=metadata))
        if paired:
            assets.extend(paired_assets[target])
        result[target] = tuple(assets)
    return result


def _resolve_qwen_controls(
    qwen_control_directory: str | None,
    records: Sequence[dict[str, Any]] | None,
    target_paths: Sequence[str],
) -> dict[str, tuple[MediaAsset, ...]]:
    """Resolve the EXPERIMENTAL Qwen-only control visuals from a directory or the JSONL."""
    record_paths: dict[str, tuple[Path, ...]] = {}
    if records is not None:
        for record, target in zip(records, target_paths):
            paths = _ordered_qwen_control_paths(record)
            if not paths:
                continue
            if not target:
                raise ValueError("H3 media JSONL records must contain a target path")
            record_paths[target] = paths
    if qwen_control_directory and record_paths:
        raise ValueError(f"specify H3 Qwen control visuals in {QWEN_CONTROL_DIRECTORY_KEY} or the JSONL, not both")
    if qwen_control_directory:
        record_paths = _references_from_directory(qwen_control_directory, target_paths)
    return {target: _qwen_control_assets(paths) for target, paths in record_paths.items()}


def _record_references(records: Sequence[dict[str, Any]], target_keys: Sequence[str]) -> dict[str, tuple[MediaAsset, ...]]:
    """Resolve per-record ``control_path[_N]`` / paired controls and their modality overrides."""
    reference_paths: dict[str, tuple[MediaAsset, ...]] = {}
    for record, target_key in zip(records, target_keys):
        controls = _ordered_control_paths(record)
        paired_controls = _paired_control_assets(record)
        modes = _reference_modes(record)
        if not (controls or paired_controls):
            continue
        if not target_key:
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
    return reference_paths


def _dataset_references(
    source: dict[str, Any],
    general: dict[str, Any],
    target: str,
    raw_references: Sequence[Any],
) -> tuple[tuple[MediaAsset, ...], tuple[float, float, float] | None]:
    """Apply the dataset-level modality selection to one target's resolved references."""
    references = tuple(
        reference if isinstance(reference, MediaAsset) else MediaAsset(reference, _modality_for_path(reference), "reference")
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
    return references, probabilities


def _effective(dataset: dict[str, Any], general: dict[str, Any], key: str) -> Any:
    value = dataset.get(key)
    return general.get(key) if value is None else value


@dataclass(frozen=True)
class _ResolvedTarget:
    path: Path
    references: tuple[MediaAsset, ...]
    qwen_controls: tuple[MediaAsset, ...] = ()


class H3DatasetAdapter:
    """Resolve H3 media while leaving Musubi's shared dataset implementation unchanged.

    Musubi's existing ``control_directory`` and ``control_path[_N]`` fields are
    canonical. The adapter removes them only from the in-memory config passed to
    ``VideoDataset`` so arbitrary H3 references are not mistaken for aligned
    ControlNet video, then attaches them to each ItemInfo at the H3 boundary.
    """

    def __init__(self, user_config: dict[str, Any], args: Any | None = None):
        user_config = _normalize_explicit_modality_config(user_config)
        self.musubi_config = copy.deepcopy(user_config)
        self._targets: dict[str, _ResolvedTarget] = {}
        self._target_groups: list[tuple[str, ...]] = []
        self._target_fps: dict[str, float] = {}
        self._target_modalities: dict[str, MediaModality] = {}
        self._target_modes: dict[str, str] = {}
        self._target_source_paths: dict[str, tuple[Path, ...]] = {}
        self._target_audio_paths: dict[str, Path] = {}
        self._target_audio_frame_counts: dict[str, int] = {}
        self._target_reference_probabilities: dict[str, tuple[float, float, float]] = {}
        self._image_frame_counts: dict[str, int] = {}
        # Authored conditioning masks: the observed region for --h3_mask_mode
        # dataset. They are applied to the packed rows at train time, so they
        # never enter the latent cache and never change its identity.
        self.conditioning_masks: dict[str, Path] = {}
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
        for key in (
            "h3_target_mode",
            "h3_image_frame_count",
            "audio_directory",
            "audio_jsonl_file",
            CONDITIONING_MASK_DIRECTORY_KEY,
            QWEN_CONTROL_DIRECTORY_KEY,
            "source_modalities",
            "target_modalities",
            *SOURCE_DIRECTORY_KEYS.values(),
            *TARGET_DIRECTORY_KEYS.values(),
            "source_video_audio_paired",
            "source_video_audio_embedded",
            "source_modality_probabilities",
        ):
            clean_general.pop(key, None)

        source_datasets = user_config.get("datasets", [])
        clean_datasets = self.musubi_config.get("datasets", [])
        for source, clean in zip(source_datasets, clean_datasets):
            for key in (
                "source_modalities",
                "target_modalities",
                "source_video_audio_paired",
                "source_video_audio_embedded",
                "source_modality_probabilities",
                *SOURCE_DIRECTORY_KEYS.values(),
                *TARGET_DIRECTORY_KEYS.values(),
            ):
                clean.pop(key, None)
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
            mask_directory = _effective(source, general, CONDITIONING_MASK_DIRECTORY_KEY)
            clean.pop(CONDITIONING_MASK_DIRECTORY_KEY, None)
            qwen_control_directory = _effective(source, general, QWEN_CONTROL_DIRECTORY_KEY)
            clean.pop(QWEN_CONTROL_DIRECTORY_KEY, None)
            if audio_directory or audio_jsonl_file:
                if target_mode != "audio":
                    raise ValueError("audio_directory/audio_jsonl_file require h3_target_mode = 'audio'")
                if mask_directory:
                    raise ValueError(
                        f"{CONDITIONING_MASK_DIRECTORY_KEY} masks video latents, so it cannot be used on an audio dataset"
                    )
                if self.image_mode != "none":
                    # The image-mode check further down is only reached by video/image
                    # datasets; without this one the run fails much later, inside caching.
                    raise ValueError("MiniMax H3 conditioned-image mode accepts image datasets only, not an audio dataset")
                audio_dataset = H3AudioDataset(source, general)
                self.audio_datasets.append(audio_dataset)
                self.dataset_kinds.append("audio")
                paths = tuple(str(path) for path, _ in audio_dataset.records)
                # An audio target carries no synchronized conditioning video, but it may still
                # declare Ref2VA references: an arbitrary conditioning video, a reference voice
                # clip, a visual anchor. The parsing is the one video targets use.
                audio_control_directory = _effective(source, general, "control_directory")
                audio_control_video_directory = _effective(source, general, "control_video_directory")
                audio_control_audio_directory = _effective(source, general, "control_audio_directory")
                explicit_sources = any(_effective(source, general, key) for key in SOURCE_DIRECTORY_KEYS.values())
                if explicit_sources and (audio_control_directory or audio_control_video_directory):
                    raise ValueError("use source_*_directory or legacy control directories, not both")
                if bool(audio_control_video_directory) != bool(audio_control_audio_directory):
                    raise ValueError("control_video_directory and control_audio_directory must be specified together")
                audio_records = _read_media_jsonl(audio_jsonl_file) if audio_jsonl_file else None
                if audio_records is not None and len(audio_records) != len(paths):
                    raise ValueError("H3 audio JSONL control parsing disagrees with the audio dataset records")
                if (
                    explicit_sources
                    and audio_records
                    and any(_ordered_control_paths(record) or _paired_control_assets(record) for record in audio_records)
                ):
                    raise ValueError("specify H3 references in source_*_directory or the audio JSONL, not both")
                if (
                    (audio_control_directory or audio_control_video_directory)
                    and audio_records
                    and any(_ordered_control_paths(record) or _paired_control_assets(record) for record in audio_records)
                ):
                    raise ValueError("specify H3 controls in control_directory or the audio JSONL, not both")
                if explicit_sources:
                    audio_reference_paths = _explicit_source_references(source, general, paths)
                elif audio_control_directory or audio_control_video_directory:
                    audio_reference_paths = _directory_references(
                        audio_control_directory,
                        audio_control_video_directory,
                        audio_control_audio_directory,
                        paths,
                    )
                elif audio_records is not None:
                    audio_reference_paths = _record_references(audio_records, paths)
                else:
                    audio_reference_paths = {}
                audio_qwen_controls = _resolve_qwen_controls(qwen_control_directory, audio_records, paths)
                for target in paths:
                    normal = _normal_path(target)
                    references, probabilities = _dataset_references(
                        source,
                        general,
                        target,
                        audio_reference_paths.get(target, ()),
                    )
                    self._targets[normal] = _ResolvedTarget(Path(target), references, audio_qwen_controls.get(target, ()))
                    self._target_modalities[normal] = MediaModality.AUDIO
                    self._target_modes[normal] = "audio"
                    if probabilities is not None:
                        self._target_reference_probabilities[normal] = probabilities
                self._target_groups.append(tuple(_normal_path(target) for target in paths))
                continue
            if target_mode == "audio":
                raise ValueError("h3_target_mode = 'audio' requires audio_directory or audio_jsonl_file")
            self.dataset_kinds.append("regular")
            control_directory = _effective(source, general, "control_directory")
            clean.pop("control_directory", None)
            control_video_directory = _effective(source, general, "control_video_directory")
            control_audio_directory = _effective(source, general, "control_audio_directory")
            explicit_sources = any(_effective(source, general, key) for key in SOURCE_DIRECTORY_KEYS.values())
            if explicit_sources and (control_directory or control_video_directory):
                raise ValueError("use source_*_directory or legacy control directories, not both")
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
            target_audio_directory = source.get("target_audio_directory")
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

            if image_directory or image_jsonl_file:
                # H3 uses this only to size a separately stored companion-audio
                # target; Musubi's shared image schema has no temporal field.
                clean.pop("target_frames", None)

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
            if (
                explicit_sources
                and records
                and any(_ordered_control_paths(record) or _paired_control_assets(record) for record in records)
            ):
                raise ValueError("specify H3 references in source_*_directory or the media JSONL, not both")
            if explicit_sources:
                reference_paths = _explicit_source_references(source, general, target_paths)
            elif control_directory or control_video_directory:
                reference_paths = _directory_references(
                    control_directory,
                    control_video_directory,
                    control_audio_directory,
                    target_paths,
                )
            elif records is not None:
                reference_paths = _record_references(records, target_paths)
            else:
                reference_paths = {}

            qwen_control_paths = _resolve_qwen_controls(qwen_control_directory, records, target_paths)
            target_audio_paths: dict[str, Path] = {}
            if target_audio_directory:
                matched_audio = _references_from_directory(target_audio_directory, target_paths)
                for target, matches in matched_audio.items():
                    if len(matches) != 1 or _modality_for_path(matches[0]) is not MediaModality.AUDIO:
                        raise ValueError(f"target_audio_directory requires one basename-matched audio target for {target!r}")
                    target_audio_paths[target] = matches[0]

            record_masks = _record_masks(records, target_paths) if records is not None else {}
            if mask_directory and record_masks:
                raise ValueError(f"specify H3 conditioning masks in {CONDITIONING_MASK_DIRECTORY_KEY} or the JSONL, not both")
            mask_paths = _masks_from_directory(mask_directory, target_paths) if mask_directory else record_masks
            for target, mask_path in mask_paths.items():
                if not Path(mask_path).is_file():
                    raise FileNotFoundError(f"H3 conditioning mask not found for {target!r}: {mask_path}")
                normal_target = _normal_path(target)
                existing_mask = self.conditioning_masks.get(normal_target)
                if existing_mask is not None and _normal_path(existing_mask) != _normal_path(mask_path):
                    raise ValueError(f"conflicting H3 conditioning mask for target path across datasets: {target}")
                self.conditioning_masks[normal_target] = Path(mask_path)

            for target in target_paths:
                modality = _modality_for_path(Path(target))
                if (video_directory or video_jsonl_file) and modality is not MediaModality.VIDEO:
                    raise ValueError(f"MiniMax H3 video datasets must resolve video targets, got {target}")
                if (image_directory or image_jsonl_file) and modality is not MediaModality.IMAGE:
                    raise ValueError(f"MiniMax H3 image datasets must resolve image targets, got {target}")
                references, probabilities = _dataset_references(source, general, target, reference_paths.get(target, ()))
                resolved = _ResolvedTarget(Path(target), references, qwen_control_paths.get(target, ()))
                normal = _normal_path(target)
                companion_audio = target_audio_paths.get(target)
                companion_audio_frames = None
                if companion_audio is not None and modality is MediaModality.IMAGE:
                    configured_audio_frames = _effective(source, general, "target_frames")
                    if (
                        not configured_audio_frames
                        or len(configured_audio_frames) != 1
                        or not is_valid_frame_count(int(configured_audio_frames[0]))
                    ):
                        raise ValueError("H3 image+audio targets require exactly one target_frames value on the 17n+5 grid")
                    companion_audio_frames = int(configured_audio_frames[0])
                    if frame_count is not None and companion_audio_frames != frame_count:
                        raise ValueError("H3 conditioned image+audio targets require target_frames to match h3_image_frame_count")
                existing = self._targets.get(normal)
                if existing is not None:
                    if existing != resolved or self._target_modalities[normal] is not modality:
                        raise ValueError(f"conflicting H3 media metadata for target path across datasets: {target}")
                    if self._target_modes[normal] != target_mode:
                        raise ValueError(f"conflicting target_modalities for target path across datasets: {target}")
                    if self._target_reference_probabilities.get(normal) != probabilities:
                        raise ValueError(f"conflicting source_modality_probabilities for target path: {target}")
                    existing_audio = self._target_audio_paths.get(normal)
                    if (existing_audio is None) != (companion_audio is None) or (
                        existing_audio is not None
                        and companion_audio is not None
                        and _normal_path(existing_audio) != _normal_path(companion_audio)
                    ):
                        raise ValueError(f"conflicting target audio for target path across datasets: {target}")
                    if companion_audio_frames is not None and self._target_audio_frame_counts.get(normal) != companion_audio_frames:
                        raise ValueError(f"conflicting target audio frame count for target path across datasets: {target}")
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
                    if companion_audio is not None:
                        self._target_audio_paths[normal] = companion_audio
                    if companion_audio_frames is not None:
                        self._target_audio_frame_counts[normal] = companion_audio_frames
            self._target_groups.append(tuple(_normal_path(target) for target in target_paths))

        self.requires_audio = (
            any(
                modality in {MediaModality.VIDEO, MediaModality.AUDIO} and self._target_modes[path] != "video"
                for path, modality in self._target_modalities.items()
            )
            or bool(self._target_audio_paths)
            or any(
                reference.modality is MediaModality.AUDIO
                or bool(reference.metadata.get("audio_path"))
                or bool(reference.metadata.get("include_audio"))
                for resolved in self._targets.values()
                for reference in resolved.references
            )
        )
        self.requires_video = any(mode != "audio" for mode in self._target_modes.values()) or any(
            reference.modality in {MediaModality.IMAGE, MediaModality.VIDEO}
            for resolved in self._targets.values()
            for reference in resolved.references
        )
        self._validate_conditioning_masks(getattr(args, "h3_mask_mode", None) if args is not None else None)

    def _validate_conditioning_masks(self, mask_mode: str | None) -> None:
        """Refuse a configuration whose masks and mask mode disagree.

        ``mask_mode`` is ``None`` for the caching scripts, which never read a
        conditioning mask: masks are applied to the packed rows at train time,
        so a cache is identical with and without them.
        """
        if mask_mode is None:
            return
        if mask_mode == "dataset":
            missing = sorted(
                str(resolved.path)
                for normal, resolved in self._targets.items()
                if normal not in self.conditioning_masks and self._target_modes[normal] != "audio"
            )
            if missing:
                raise ValueError(
                    "--h3_mask_mode dataset needs a conditioning mask for every item; "
                    f"{CONDITIONING_MASK_DIRECTORY_KEY}/{CONDITIONING_MASK_KEY} covers none for: {', '.join(missing[:5])}"
                )
            return
        if self.conditioning_masks:
            raise ValueError(
                f"the dataset declares conditioning masks ({CONDITIONING_MASK_DIRECTORY_KEY}/{CONDITIONING_MASK_KEY}) "
                f"but --h3_mask_mode {mask_mode} draws its own; pass --h3_mask_mode dataset to use them"
            )

    def conditioning_mask_paths_by_item_key(self) -> dict[str, str]:
        """Map the cached item key -- the target's basename -- to its mask file.

        Training rebuilds items from cache filenames, which keep only the stem,
        so that is the only handle the collator has on the source media. An
        image-mode dataset caches its latents under ``{stem}_00000-{frames:03d}``,
        so both spellings are registered: the bare stem for image datasets that
        do not run image mode, the frame-ranged one for those that do.
        """
        masks: dict[str, str] = {}

        def register(key: str, mask_path: str) -> None:
            # Two targets with the same stem in different folders collapse onto one key
            # here, and the collator would silently apply one target's mask to the other.
            existing = masks.get(key)
            if existing is not None and existing != mask_path:
                raise ValueError(f"ambiguous H3 conditioning masks for {key!r}: {existing} and {mask_path}")
            masks[key] = mask_path

        for normal, mask_path in self.conditioning_masks.items():
            stem = Path(self._targets[normal].path).stem
            register(stem, str(mask_path))
            frame_count = self._image_frame_counts.get(normal)
            if frame_count is not None:
                register(f"{stem}_00000-{int(frame_count):03d}", str(mask_path))
        return masks

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
                            or key == CONDITIONING_MASK_KEY
                            or key == QWEN_CONTROL_KEY
                            or _QWEN_CONTROL_PATH_PATTERN.fullmatch(key)
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
        companion_audio = self._target_audio_paths.get(normal)
        if companion_audio is not None:
            metadata = dict(target.metadata)
            metadata["audio_path"] = str(companion_audio)
            if modality is MediaModality.IMAGE:
                metadata["audio_frame_count"] = self._target_audio_frame_counts[normal]
            target = MediaAsset(
                target.path,
                target.modality,
                target.role,
                stream_index=target.stream_index,
                start_seconds=target.start_seconds,
                duration_seconds=target.duration_seconds,
                metadata=metadata,
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
            assets = (target, *resolved.qwen_controls)
        else:
            assets = (target, *resolved.references, *resolved.qwen_controls)
        validate_h3_media_assets(item.item_key, assets)
        item.h3_media_assets = assets
        # Both fingerprints are optional and independent, so they are merged into
        # whatever the image-mode branch already recorded rather than replacing it.
        metadata = dict(getattr(item, "h3_cache_metadata", {}))
        if companion_audio is not None:
            metadata[TARGET_AUDIO_FINGERPRINT_KEY] = sample_fingerprint(
                targets=(resolved.path, companion_audio),
                controls=(),
                mode="target_audio",
                frame_count=int(target.metadata.get("audio_frame_count", target_frame_count)),
                original_size=item.original_size,
                bucket_size=item.bucket_size,
            )
        fingerprint = reference_fingerprint(assets)
        if fingerprint is not None:
            metadata[REFERENCE_FINGERPRINT_KEY] = fingerprint
        control_fingerprint = qwen_control_fingerprint(assets)
        if control_fingerprint is not None:
            metadata[QWEN_CONTROL_FINGERPRINT_KEY] = control_fingerprint
        if metadata:
            item.h3_cache_metadata = metadata
        if normal in self._target_reference_probabilities:
            item.h3_reference_modality_probabilities = self._target_reference_probabilities[normal]
        item.h3_target_mode = self._target_modes[normal]
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
        # Authored masks are read per batch rather than cached: they are applied
        # to the packed rows, so the latents they mask are the same either way.
        conditioning_masks = adapter.conditioning_mask_paths_by_item_key() if adapter.conditioning_masks else {}
        for dataset in ordered_datasets:
            batch_manager = getattr(dataset, "batch_manager", None)
            if batch_manager is not None:
                batch_manager.load_h3_dino_features = load_dino_features and isinstance(dataset, VideoDataset)
                batch_manager.h3_dino_model = getattr(args, "h3_dino_model", None) if load_dino_features else None
                batch_manager.h3_conditioning_mask_paths = conditioning_masks
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
    unsupported_roles = sorted({asset.role for asset in assets if asset.role not in {"target", "reference", QWEN_CONTROL_ROLE}})
    if unsupported_roles:
        raise ValueError(f"H3 item {key!r} contains unsupported roles: {unsupported_roles}")
    if check_files:
        media_paths = {asset.path for asset in assets}
        media_paths.update(Path(asset.metadata["audio_path"]) for asset in assets if asset.metadata.get("audio_path"))
        missing = sorted(str(path) for path in media_paths if not path.is_file())
        if missing:
            raise FileNotFoundError(f"media file(s) not found for H3 item {key!r}: {', '.join(missing)}")
