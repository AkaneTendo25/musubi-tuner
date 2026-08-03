from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import DatasetGroup, ItemInfo
from musubi_tuner.dataset.media_utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, glob_videos
from musubi_tuner.minimax_h3.architecture import is_valid_frame_count
from musubi_tuner.minimax_h3.media import MediaAsset, MediaModality, slice_media_asset

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus")
_CONTROL_PATH_PATTERN = re.compile(r"^control_path_(\d+)$")
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


def _control_sort_key(path: Path, target_stem: str) -> int:
    if path.stem == target_stem:
        return 0
    suffix = path.stem.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"invalid numbered control suffix: {path.name}")
    return int(suffix) + 1


def _references_from_directory(control_directory: str, target_paths: Sequence[str]) -> dict[str, tuple[Path, ...]]:
    root = Path(control_directory)
    if not root.is_dir():
        raise ValueError(f"control_directory does not exist: {root}")
    allowed = {extension.lower() for extension in (*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS, *AUDIO_EXTENSIONS)}
    available = {path for path in root.iterdir() if path.is_file() and path.suffix.lower() in allowed}
    result: dict[str, tuple[Path, ...]] = {}
    for target in sorted(target_paths, key=lambda path: len(Path(path).name), reverse=True):
        stem = Path(target).stem
        matches = [path for path in available if path.stem == stem or path.stem.startswith(stem + "_")]
        matches.sort(key=lambda path: _control_sort_key(path, stem))
        order = [_control_sort_key(path, stem) for path in matches]
        if len(order) != len(set(order)):
            raise ValueError(f"multiple H3 controls occupy the same index for {target!r}: {matches}")
        if not matches:
            raise ValueError(f"no matching H3 controls for {target!r} in {root}")
        available.difference_update(matches)
        result[target] = tuple(matches)
    return result


def _read_video_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
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

    def __init__(self, user_config: dict[str, Any]):
        self.musubi_config = copy.deepcopy(user_config)
        self._targets: dict[str, _ResolvedTarget] = {}
        self._target_groups: list[tuple[str, ...]] = []
        self._target_fps: dict[str, float] = {}
        general = user_config.get("general", {})
        clean_general = self.musubi_config.get("general", {})
        clean_general.pop("control_directory", None)

        source_datasets = user_config.get("datasets", [])
        clean_datasets = self.musubi_config.get("datasets", [])
        for source, clean in zip(source_datasets, clean_datasets):
            target_frames = _effective(source, general, "target_frames")
            invalid_frames = [frame_count for frame_count in target_frames or () if not is_valid_frame_count(frame_count)]
            if invalid_frames:
                raise ValueError(f"MiniMax H3 target_frames must satisfy frame_count % 17 == 5; invalid values: {invalid_frames}")
            control_directory = _effective(source, general, "control_directory")
            clean.pop("control_directory", None)
            video_directory = _effective(source, general, "video_directory")
            video_jsonl_file = _effective(source, general, "video_jsonl_file")

            records: list[dict[str, Any]] | None = None
            if video_directory:
                target_paths = tuple(glob_videos(video_directory))
            elif video_jsonl_file:
                records = _read_video_jsonl(video_jsonl_file)
                target_paths = tuple(record["video_path"] for record in records)
            else:
                raise ValueError("MiniMax H3 requires a Musubi video dataset")

            if control_directory and records and any(_ordered_control_paths(record) for record in records):
                raise ValueError("specify H3 controls in control_directory or video JSONL, not both")
            if control_directory:
                reference_paths = _references_from_directory(control_directory, target_paths)
            elif records is not None:
                reference_paths = {}
                for record in records:
                    controls = _ordered_control_paths(record)
                    if controls:
                        reference_paths[record["video_path"]] = controls
            else:
                reference_paths = {}

            for target in target_paths:
                references = tuple(
                    MediaAsset(path, _modality_for_path(path), "reference") for path in reference_paths.get(target, ())
                )
                resolved = _ResolvedTarget(Path(target), references)
                normal = _normal_path(target)
                if normal in self._targets:
                    raise ValueError(f"duplicate H3 target path across datasets: {target}")
                self._targets[normal] = resolved
            self._target_groups.append(tuple(_normal_path(target) for target in target_paths))

    def adapt_dataset_group(self, dataset_group: DatasetGroup) -> None:
        if len(dataset_group.datasets) != len(self._target_groups):
            raise ValueError("H3 dataset adapter and Musubi dataset group have different lengths")
        for dataset, target_group in zip(dataset_group.datasets, self._target_groups):
            datasource = dataset.datasource
            if hasattr(datasource, "data"):
                for record in datasource.data:
                    record.pop("control_path", None)
                datasource.has_control = False
            dataset.control_directory = None
            dataset.has_control = False
            for target in target_group:
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
        raise KeyError(f"H3 dataset adapter cannot map ItemInfo key to a source video: {item_key}")

    def attach(self, item: ItemInfo) -> tuple[MediaAsset, ...]:
        resolved, start_frame, frame_count = self._resolve_target(item.item_key)
        target_fps = self._target_fps[_normal_path(resolved.path)]
        target_frame_count = frame_count if frame_count is not None else item.frame_count
        target = MediaAsset(
            resolved.path,
            MediaModality.VIDEO,
            "target",
            metadata={"frame_count": target_frame_count, "fps": target_fps},
        )
        if start_frame is not None and frame_count is not None:
            target = slice_media_asset(
                target,
                start_seconds=start_frame / target_fps,
                duration_seconds=frame_count / target_fps,
            )
        assets = (target, *resolved.references)
        validate_h3_media_assets(item.item_key, assets)
        item.h3_media_assets = assets
        return assets


def create_h3_dataset_group(
    user_config: dict[str, Any],
    args: Any,
    *,
    training: bool = False,
    num_timestep_buckets: int | None = None,
    shared_epoch: Any = None,
) -> tuple[DatasetGroup, H3DatasetAdapter]:
    adapter = H3DatasetAdapter(user_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        adapter.musubi_config,
        args,
        architecture=ARCHITECTURE_MINIMAX_H3,
    )
    dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.dataset_group,
        training=training,
        num_timestep_buckets=num_timestep_buckets,
        shared_epoch=shared_epoch,
    )
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
    if len(targets) != 1 or targets[0].modality is not MediaModality.VIDEO:
        raise ValueError(f"H3 item {key!r} must contain exactly one target video")
    unsupported_roles = sorted({asset.role for asset in assets if asset.role not in {"target", "reference"}})
    if unsupported_roles:
        raise ValueError(f"H3 item {key!r} contains unsupported roles: {unsupported_roles}")
    if check_files:
        missing = sorted({str(asset.path) for asset in assets if not asset.path.is_file()})
        if missing:
            raise FileNotFoundError(f"media file(s) not found for H3 item {key!r}: {', '.join(missing)}")
