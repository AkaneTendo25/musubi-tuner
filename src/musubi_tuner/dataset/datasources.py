from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Optional, TYPE_CHECKING

import torch
from PIL import Image

from musubi_tuner.dataset.audio_utils import (
    AUDIO_SIDECAR_EXTENSIONS,
    AudioSource,
    AudioSpec,
    decode_audio,
    probe_audio,
    resolve_audio_source,
)
from musubi_tuner.dataset.media_utils import glob_images, glob_videos, load_video, VIDEO_EXTENSIONS

if TYPE_CHECKING:
    from musubi_tuner.dataset.bucket import BucketSelector

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItemExtras:
    """Per-item fields beyond the shared dataset schema, for architecture-specific consumers.

    ``fields`` holds the record's keys that the shared schema does not define (for example the
    MiniMax-H3 ``references`` and ``teacher_caption``). Only record-based datasources (JSONL)
    can carry such fields; directory datasources always report an empty mapping. Relative
    paths inside ``fields`` resolve from ``base_directory`` (the JSONL's directory, or the
    dataset directory), and ``label`` names the record's origin for error messages.
    """

    fields: Mapping[str, Any]
    base_directory: str
    label: str


def _extra_fields(data: Mapping[str, Any], shared_keys: tuple[str, ...]) -> dict[str, Any]:
    """The record's keys outside the shared schema; ``key_N`` counts as its ``key`` (control_path_0, image_path_1, ...)."""
    extras = {}
    for key, value in data.items():
        stem, _, suffix = key.rpartition("_")
        if key in shared_keys or (suffix.isdigit() and stem in shared_keys):
            continue
        extras[key] = value
    return extras


class ContentDatasource:
    def __init__(self):
        self.caption_only = False  # set to True to only fetch caption for Text Encoder caching
        self.has_control = False

    def set_caption_only(self, caption_only: bool):
        self.caption_only = caption_only

    def is_indexable(self):
        return False

    def get_caption(self, idx: int) -> tuple[str, str]:
        """
        Returns caption. May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def get_item_extras(self, idx: int) -> ItemExtras:
        """
        Returns the item's fields beyond the shared schema (see ItemExtras). Indices align with
        get_caption and with ItemInfo.datasource_index. May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class ImageDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, list[Image.Image]]:
        """
        Returns image data as a tuple of image path, image, and caption for the given index.
        Key must be unique and valid as a file name.
        May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def get_control_paths(self) -> dict[str, list[str]]:
        """
        Returns {image_path: [control paths in index order]} for cache fingerprinting.
        Control pixels travel through ItemInfo.control_content; this accessor exists only so
        cache scripts can fingerprint the source files behind them.
        """
        return {}


def _create_image_fetcher(datasource: ImageDatasource, index: int):
    def fetch():
        return datasource.get_image_data(index)

    # the datasource record index travels as a fetcher attribute so that ItemInfo can
    # reference the originating record without re-deriving it from item keys
    fetch.datasource_index = index
    return fetch


class ImageDirectoryDatasource(ImageDatasource):
    def __init__(
        self,
        image_directory: str,
        caption_extension: Optional[str] = None,
        control_directory: Optional[str] = None,
        control_count_per_image: Optional[int] = None,
        multiple_target: bool = False,
    ):
        super().__init__()
        self.image_directory = image_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory
        self.control_count_per_image = control_count_per_image
        self.multiple_target = multiple_target
        self.current_idx = 0

        # glob images
        logger.info(f"glob images in {self.image_directory}")
        self.image_paths = glob_images(self.image_directory, caption_extension=self.caption_extension)
        logger.info(f"found {len(self.image_paths)} images")

        # check if multiple-target images exist
        self.target_paths: dict[str, list[str]] = {}  # image_path -> list of target image paths

        if self.multiple_target:
            # sort by length, longer first
            sorted_image_paths = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)

            all_image_paths = set(glob_images(self.image_directory))  # image1.jpg, image1_1.jpg, image1_2.jpg, ...
            multiple_target_candidates = all_image_paths - set(sorted_image_paths)  # those not in the images with captions

            if len(multiple_target_candidates) > 0:
                logger.info("checking for multiple-target images")
                for image_path in sorted_image_paths:
                    image_path_no_ext = os.path.splitext(image_path)[0]

                    # find matching multiple-target images
                    potential_paths = [p for p in multiple_target_candidates if p.startswith(image_path_no_ext + "_")]

                    if potential_paths:
                        # sort by the digits (`_0000`) suffix
                        def sort_key(path):
                            path_no_ext = os.path.splitext(path)[0]
                            digits_suffix = path_no_ext.rsplit("_", 1)[-1]
                            if not digits_suffix.isdigit():
                                raise ValueError(
                                    f"Invalid digits suffix in '{path_no_ext}'. Expected a numeric suffix after '_' "
                                    f"(e.g., '_0', '_1', '_2') for proper sorting of multiple target images."
                                )
                            return int(digits_suffix)

                        potential_paths.sort(key=sort_key)
                        self.target_paths[image_path] = potential_paths

                        # remove to avoid duplicate matching
                        multiple_target_candidates.difference_update(potential_paths)

                # check the number of targets: all multiple-target images should have the same number of targets
                num_targets = 0
                for image_path, paths in self.target_paths.items():
                    if num_targets == 0:
                        num_targets = len(paths)
                    elif num_targets != len(paths):
                        logger.error(
                            f"All multiple-target images must have the same number of targets / 全ての複数ターゲット画像は同じ数のターゲットを持つ必要があります: {image_path}"
                        )
                        raise ValueError(
                            f"All multiple-target images must have the same number of targets / 全ての複数ターゲット画像は同じ数のターゲットを持つ必要があります: {image_path}"
                        )

                if num_targets == 0:
                    logger.error("no multiple-target images found, but multiple_target is set to True")
                    raise ValueError("no multiple-target images found, but multiple_target is set to True")

                logger.info(f"found multiple-target images, max targets per image: {num_targets}")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control images in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}

            # sort image paths for matching control images properly: longer names first
            image_paths_sorted = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)

            # glob control images first
            all_control_image_paths = set(glob_images(self.control_directory))

            for image_path in image_paths_sorted:
                image_basename = os.path.basename(image_path)
                image_basename_no_ext = os.path.splitext(image_basename)[0]

                # find matching control images
                potential_paths = [
                    p
                    for p in all_control_image_paths
                    if os.path.basename(p).startswith(image_basename_no_ext + ".")
                    or os.path.basename(p).startswith(image_basename_no_ext + "_")
                ]

                # remove to avoid duplicate matching
                all_control_image_paths.difference_update(potential_paths)

                if potential_paths:
                    # sort by the digits (`_0000`) suffix, prefer the one without the suffix
                    def sort_key(path):
                        basename = os.path.basename(path)
                        basename_no_ext = os.path.splitext(basename)[0]
                        if image_basename_no_ext == basename_no_ext:  # prefer the one without suffix
                            return 0
                        digits_suffix = basename_no_ext.rsplit("_", 1)[-1]
                        if not digits_suffix.isdigit():
                            raise ValueError(f"Invalid digits suffix in {basename_no_ext}")
                        return int(digits_suffix) + 1

                    potential_paths.sort(key=sort_key)
                    if control_count_per_image is not None and len(potential_paths) < control_count_per_image:
                        logger.error(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )
                        raise ValueError(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )

                    # take the first `control_count_per_image` paths
                    self.control_paths[image_path] = (
                        potential_paths[:control_count_per_image] if control_count_per_image is not None else potential_paths
                    )
            logger.info(
                f"found {len(self.control_paths)} matching control images for {'arbitrary' if control_count_per_image is None else control_count_per_image} images"
            )

            # log the distribution of number of control images
            count_of_num_control_images = {}
            for paths in self.control_paths.values():
                count = len(paths)
                if count not in count_of_num_control_images:
                    count_of_num_control_images[count] = 0
                count_of_num_control_images[count] += 1
            for count, num_images in count_of_num_control_images.items():
                logger.info(f"  {num_images} images have {count} control images")

            missing_controls = len(self.image_paths) - len(self.control_paths)
            if missing_controls > 0:
                missing_control_paths = set(self.image_paths) - set(self.control_paths.keys())
                logger.error(f"Could not find matching control images for {missing_controls} images: {missing_control_paths}")
                raise ValueError(f"Could not find matching control images for {missing_controls} images")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        image_path = self.image_paths[idx]
        image_paths = [image_path]
        if self.multiple_target:
            # load multiple-target images
            image_paths += self.target_paths.get(image_path, [])

        images = []
        for p in image_paths:
            img = Image.open(p)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        _, caption = self.get_caption(idx)

        controls = None
        if self.has_control:
            controls = []
            for control_path in self.control_paths[image_path]:
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls

    def get_caption(self, idx: int) -> tuple[str, str]:
        image_path = self.image_paths[idx]
        caption_path = os.path.splitext(image_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return image_path, caption

    def get_control_paths(self) -> dict[str, list[str]]:
        if not self.has_control:
            return {}
        return {image_path: list(paths) for image_path, paths in self.control_paths.items()}

    def get_item_extras(self, idx: int) -> ItemExtras:
        # a directory item has no place for fields beyond the shared schema
        return ItemExtras(fields={}, base_directory=self.image_directory, label=self.image_paths[idx])

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        """
        Returns a fetcher function that returns image data.
        """
        if self.current_idx >= len(self.image_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:
            fetcher = _create_image_fetcher(self, self.current_idx)

        self.current_idx += 1
        return fetcher


class ImageJsonlDatasource(ImageDatasource):
    # the shared image JSONL schema (numbered variants included); every other key is an item extra
    SHARED_KEYS = ("image_path", "caption", "control_path")

    def __init__(self, image_jsonl_file: str, control_count_per_image: Optional[int] = None, multiple_target: bool = False):
        super().__init__()
        self.image_jsonl_file = image_jsonl_file
        self.control_count_per_image = control_count_per_image
        self.multiple_target = multiple_target
        self.current_idx = 0

        # load jsonl
        logger.info(f"load image jsonl from {self.image_jsonl_file}")
        self.data = []
        with open(self.image_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"failed to load json: {line} @ {self.image_jsonl_file}")
                    raise
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} images")

        # Normalize control paths
        for item in self.data:
            if "control_path" in item:
                item["control_path_0"] = item.pop("control_path")

            # Ensure control paths are named consistently, from control_path_0000 to control_path_0, control_path_1, etc.
            control_path_keys = [key for key in item.keys() if key.startswith("control_path_")]
            control_path_keys.sort(key=lambda x: int(x.split("_")[-1]))
            for i, key in enumerate(control_path_keys):
                if key != f"control_path_{i}":
                    item[f"control_path_{i}"] = item.pop(key)

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path_0" in item for item in self.data)
        if self.has_control:
            if self.control_count_per_image is None:
                logger.info(f"found {len(self.data)} images with arbitrary control images per image in JSONL data")
            else:
                missing_control_images = [
                    item["image_path"]
                    for item in self.data
                    if sum(f"control_path_{i}" not in item for i in range(self.control_count_per_image)) > 0
                ]
                if missing_control_images:
                    logger.error(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                    raise ValueError(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                logger.info(
                    f"found {len(self.data)} images with {self.control_count_per_image} control images per image in JSONL data"
                )

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        image_paths = [image_path]
        if self.multiple_target:
            # load multiple-target images
            while True:
                next_index = len(image_paths)  # start from 1
                next_image_path = data.get("image_path_" + str(next_index), None)
                if next_image_path is None:
                    break
                if not os.path.exists(next_image_path):
                    raise ValueError(f"multiple-target image not found: {next_image_path}")

                image_paths.append(next_image_path)

        images = []
        for path in image_paths:
            img = Image.open(path)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        caption = data["caption"]

        controls = None
        if self.has_control:
            controls = []
            for i in range(self.control_count_per_image or 1000):  # arbitrary large number if control_count_per_image is None
                if f"control_path_{i}" not in data:
                    break
                control_path = data[f"control_path_{i}"]
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        caption = data["caption"]
        return image_path, caption

    def get_control_paths(self) -> dict[str, list[str]]:
        if not self.has_control:
            return {}
        control_paths: dict[str, list[str]] = {}
        for data in self.data:
            image_path = data.get("image_path", data.get("image_path_0"))
            paths = []
            for i in range(self.control_count_per_image or 1000):  # same bound rule as get_image_data
                if f"control_path_{i}" not in data:
                    break
                paths.append(data[f"control_path_{i}"])
            control_paths[image_path] = paths
        return control_paths

    def get_item_extras(self, idx: int) -> ItemExtras:
        return ItemExtras(
            fields=_extra_fields(self.data[idx], ImageJsonlDatasource.SHARED_KEYS),
            base_directory=os.path.dirname(os.path.abspath(self.image_jsonl_file)),
            label=f"{os.path.basename(self.image_jsonl_file)} line {idx + 1}",
        )

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:
            fetcher = _create_image_fetcher(self, self.current_idx)

        self.current_idx += 1
        return fetcher


class VideoDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

        # None means all frames
        self.start_frame = None
        self.end_frame = None

        self.bucket_selector = None

        self.source_fps = None
        self.target_fps = None

        # timestamp-based fps normalization (audio-capable architectures need deterministic fps)
        self.strict_target_fps: Optional[float] = None

        # audio support: set via set_audio_spec by audio-capable architectures
        self.audio_spec: Optional[AudioSpec] = None
        self.audio_sources: Optional[list[Optional[AudioSource]]] = None

    def __len__(self):
        raise NotImplementedError

    def get_video_data_from_path(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> list[Image.Image]:
        # this method can resize the video if bucket_selector is given to reduce the memory usage

        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        if self.strict_target_fps is not None:
            return load_video(
                video_path,
                start_frame,
                end_frame,
                bucket_selector,
                target_fps=self.strict_target_fps,
                fps_resample_mode="timestamps",
            )

        video = load_video(
            video_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return video

    def get_control_data_from_path(
        self,
        control_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> list[Image.Image]:
        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        if self.strict_target_fps is not None:
            return load_video(
                control_path,
                start_frame,
                end_frame,
                bucket_selector,
                target_fps=self.strict_target_fps,
                fps_resample_mode="timestamps",
            )

        control = load_video(
            control_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return control

    def set_start_and_end_frame(self, start_frame: Optional[int], end_frame: Optional[int]):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def set_bucket_selector(self, bucket_selector: BucketSelector):
        self.bucket_selector = bucket_selector

    def set_source_and_target_fps(self, source_fps: Optional[float], target_fps: Optional[float]):
        self.source_fps = source_fps
        self.target_fps = target_fps

    def set_strict_target_fps(self, target_fps: Optional[float]):
        self.strict_target_fps = target_fps

    def set_audio_spec(self, audio_spec: Optional[AudioSpec]):
        """Enables audio for this datasource and eagerly resolves all audio sources (fail-fast)."""
        self.audio_spec = audio_spec
        self.audio_sources = None
        if audio_spec is None:
            return

        audio_sources = []
        missing = []
        for index in range(len(self)):
            video_path, explicit_path = self._audio_resolution_inputs(index)
            source = resolve_audio_source(video_path, explicit_path)
            audio_sources.append(source)
            if source is None:
                missing.append(video_path)
        self.audio_sources = audio_sources

        if missing:
            for video_path in missing[:10]:
                logger.warning(f"Video has no audio source; an unsupervised silence placeholder will be cached: {video_path}")
            logger.info(f"audio sources resolved: {len(audio_sources) - len(missing)} with audio, {len(missing)} without")

    def _audio_resolution_inputs(self, idx: int) -> tuple[str, Optional[str]]:
        """Returns (video_path, explicit_audio_path) for audio source resolution."""
        raise NotImplementedError

    def get_audio_waveform(self, idx: int) -> Optional[torch.Tensor]:
        """Decodes the full waveform [C, L] for the item, or None if it has no audio source."""
        if self.audio_spec is None or self.audio_sources is None:
            raise ValueError("Audio is not enabled for this datasource; call set_audio_spec first")
        source = self.audio_sources[idx]
        if source is None:
            return None
        return decode_audio(source, sample_rate=self.audio_spec.sample_rate, channels=self.audio_spec.channels)

    def _create_video_fetcher(self, index: int):
        if self.audio_spec is not None:

            def fetch():
                video_path, video, caption, control = self.get_video_data(index)
                waveform = self.get_audio_waveform(index)
                return video_path, video, caption, control, waveform

        else:

            def fetch():
                return self.get_video_data(index)

        # the datasource record index travels as a fetcher attribute so that ItemInfo can
        # reference the originating record without re-deriving it from item keys
        fetch.datasource_index = index
        return fetch

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class VideoDirectoryDatasource(VideoDatasource):
    def __init__(self, video_directory: str, caption_extension: Optional[str] = None, control_directory: Optional[str] = None):
        super().__init__()
        self.video_directory = video_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory
        self.current_idx = 0

        # glob videos
        logger.info(f"glob videos in {self.video_directory}")
        self.video_paths = glob_videos(self.video_directory)
        logger.info(f"found {len(self.video_paths)} videos")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control videos in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}
            for video_path in self.video_paths:
                video_basename = os.path.basename(video_path)
                # construct control path from video path
                # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mp4"
                control_path = os.path.join(self.control_directory, video_basename)
                if os.path.exists(control_path):
                    self.control_paths[video_path] = control_path
                else:
                    # use the same base name for control path
                    base_name = os.path.splitext(video_basename)[0]

                    # directory with images. for example: video_path = "vid/video.mp4" -> control_path = "control/video"
                    potential_path = os.path.join(self.control_directory, base_name)  # no extension
                    if os.path.isdir(potential_path):
                        self.control_paths[video_path] = potential_path
                    else:
                        # another extension for control path
                        # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mov"
                        for ext in VIDEO_EXTENSIONS:
                            potential_path = os.path.join(self.control_directory, base_name + ext)
                            if os.path.exists(potential_path):
                                self.control_paths[video_path] = potential_path
                                break

            logger.info(f"found {len(self.control_paths)} matching control videos/images")
            # check if all videos have matching control paths, if not, raise an error
            missing_controls = len(self.video_paths) - len(self.control_paths)
            if missing_controls > 0:
                # logger.warning(f"Could not find matching control videos/images for {missing_controls} videos")
                missing_controls_videos = [video_path for video_path in self.video_paths if video_path not in self.control_paths]
                logger.error(
                    f"Could not find matching control videos/images for {missing_controls} videos: {missing_controls_videos}"
                )
                raise ValueError(f"Could not find matching control videos/images for {missing_controls} videos")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.video_paths)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        video_path = self.video_paths[idx]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        _, caption = self.get_caption(idx)

        control = None
        if self.control_directory is not None and video_path in self.control_paths:
            control_path = self.control_paths[video_path]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        return video_path, video, caption, control

    def get_caption(self, idx: int) -> tuple[str, str]:
        video_path = self.video_paths[idx]
        caption_path = os.path.splitext(video_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return video_path, caption

    def _audio_resolution_inputs(self, idx: int) -> tuple[str, Optional[str]]:
        return self.video_paths[idx], None

    def get_item_extras(self, idx: int) -> ItemExtras:
        # a directory item has no place for fields beyond the shared schema
        return ItemExtras(fields={}, base_directory=self.video_directory, label=self.video_paths[idx])

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.video_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:
            fetcher = self._create_video_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class VideoJsonlDatasource(VideoDatasource):
    PATH_KEYS = ("video_path", "control_path", "audio_path")
    # the shared video JSONL schema; every other key is an item extra
    SHARED_KEYS = ("video_path", "caption", "control_path", "audio_path")

    def __init__(self, video_jsonl_file: str):
        super().__init__()
        self.video_jsonl_file = video_jsonl_file
        self.current_idx = 0

        # load jsonl
        logger.info(f"load video jsonl from {self.video_jsonl_file}")
        self.data = []
        with open(self.video_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} videos")

        # resolve relative paths against the working directory first (the historical behavior),
        # then against the JSONL's own directory. Whichever location matches is rewritten to an
        # absolute path so downstream consumers (e.g. MiniMax-H3 record building) never
        # re-resolve the value against a different base.
        base_directory = os.path.dirname(os.path.abspath(self.video_jsonl_file))
        for data in self.data:
            for key in VideoJsonlDatasource.PATH_KEYS:
                value = data.get(key)
                if not isinstance(value, str) or not value or os.path.isabs(value):
                    continue
                jsonl_candidate = os.path.join(base_directory, value)
                if os.path.exists(value):
                    data[key] = os.path.abspath(value)
                    if os.path.exists(jsonl_candidate) and not os.path.samefile(value, jsonl_candidate):
                        logger.warning(
                            f"{key} {value!r} exists both relative to the working directory and to the JSONL directory; "
                            f"using the working-directory match {data[key]}"
                        )
                elif os.path.exists(jsonl_candidate):
                    data[key] = jsonl_candidate

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path" in item for item in self.data)
        if self.has_control:
            control_count = sum(1 for item in self.data if "control_path" in item)
            if control_count < len(self.data):
                missing_control_videos = [item["video_path"] for item in self.data if "control_path" not in item]
                logger.error(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
                raise ValueError(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
            logger.info(f"found {control_count} control videos/images in JSONL data")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        data = self.data[idx]
        video_path = data["video_path"]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        caption = data["caption"]

        control = None
        if "control_path" in data and data["control_path"]:
            control_path = data["control_path"]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        return video_path, video, caption, control

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        video_path = data["video_path"]
        caption = data["caption"]
        return video_path, caption

    def _audio_resolution_inputs(self, idx: int) -> tuple[str, Optional[str]]:
        data = self.data[idx]
        return data["video_path"], data.get("audio_path")

    def get_item_extras(self, idx: int) -> ItemExtras:
        return ItemExtras(
            fields=_extra_fields(self.data[idx], VideoJsonlDatasource.SHARED_KEYS),
            base_directory=os.path.dirname(os.path.abspath(self.video_jsonl_file)),
            label=f"{os.path.basename(self.video_jsonl_file)} line {idx + 1}",
        )

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:
            fetcher = self._create_video_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


# ---------------------------------------------------------------------------
# audio (music) datasources: one record per audio file or JSONL line
# ---------------------------------------------------------------------------

AUDIO_CAPTION_FORMATS = ("auto", "plain", "tags_lyrics", "json")
ABC_MODES = ("melody", "full")
# optional first line of an ABC sidecar/field naming its flavour (written by the user's own transcription tooling);
# `%%` lines are ABC directives, so the line is harmless to ABC tools. It is stripped from the score.
ABC_MODE_HEADER = "%%yue2_abc_mode"


@dataclass(frozen=True)
class AudioRecord:
    """One audio record: a file (directory datasets) or a JSONL line, possibly an excerpt (start_s/end_s).

    ``lyrics``/``abc`` are None when absent (the text cache substitutes instrumental lyrics); ``abc_mode`` is
    ``melody``/``full`` when ``abc`` is present, else None.
    """

    audio_path: str
    item_key: str
    style: str
    lyrics: Optional[str]
    abc: Optional[str]
    abc_mode: Optional[str]
    song_id: str
    start_s: Optional[float]
    end_s: Optional[float]
    source: AudioSource


@dataclass(frozen=True)
class AudioExtent:
    """Where a decoded record waveform sits in its file, in samples at the decode rate."""

    start_sample: int
    end_sample: int
    file_samples: int


_CAPTION_BLOCKS = ("tags", "lyrics", "abc", "duration")
# non-ASCII letters that case-insensitive matching treats as equal to the ASCII letters of the block names
_BLOCK_NAME_FOLD = str.maketrans({"ſ": "s", "ı": "i", "İ": "i"})
# auto-detection only looks for a [Lyrics] header on a \n-delimited line
_LYRICS_HEADER_LINE = re.compile(r"^\s*\[Lyrics\]\s*$", re.IGNORECASE | re.MULTILINE)
_LEGACY_TAGS = ("<CAPTION>", "<LYRICS>")


def _xml_tag(text: str, name: str) -> Optional[str]:
    """Stripped text between the first ``<name>`` and the nearest ``</name>`` after it, or None."""
    opener = f"<{name}>"
    begin = text.find(opener)
    if begin < 0:
        return None
    begin += len(opener)
    end = text.find(f"</{name}>", begin)
    if end < 0:
        return None
    return text[begin:end].strip()


def _has_legacy_tags(text: str) -> bool:
    return any(tag in text for tag in _LEGACY_TAGS)


def _detect_caption_format(text: str, stripped: str) -> str:
    if stripped.startswith("{"):
        return "json"
    if _has_legacy_tags(text) or _LYRICS_HEADER_LINE.search(text) is not None:
        return "tags_lyrics"
    return "plain"


def _block_header(line: str) -> Optional[str]:
    """Block key when the line is only a bracketed block name such as ``  [lyrics] ``, else None."""
    body = line.strip()
    if len(body) < 3 or not (body.startswith("[") and body.endswith("]")):
        return None
    label = body[1:-1]
    if label.translate(_BLOCK_NAME_FOLD).lower() not in _CAPTION_BLOCKS:
        return None
    return label.lower()


def _scan_caption_blocks(text: str) -> Iterator[tuple[str, Optional[str]]]:
    """Yields ``(block, line)`` per body line and ``(block, None)`` when a header opens a block.

    Lines before any header belong to ``tags``.
    """
    block = "tags"
    for line in text.splitlines():
        opened = _block_header(line)
        if opened is None:
            yield block, line
        else:
            block = opened
            yield block, None


def _parse_tagged_caption(text: str) -> tuple[str, Optional[str], Optional[str]]:
    if _has_legacy_tags(text):
        return _xml_tag(text, "CAPTION") or "", _xml_tag(text, "LYRICS"), _xml_tag(text, "ABC")

    opened = {"tags"}
    bodies: dict[str, list[str]] = {}
    for block, line in _scan_caption_blocks(text):
        if line is None:
            opened.add(block)
        else:
            bodies.setdefault(block, []).append(line)

    def block_text(name: str) -> Optional[str]:
        # a block that was never opened is absent; an opened but empty one is ""
        if name not in opened:
            return None
        return "\n".join(bodies.get(name, ())).strip()

    return block_text("tags"), block_text("lyrics"), block_text("abc")


def _optional_str(value: Any, what: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a string, got {type(value).__name__}")
    return value


def parse_yue2_caption(text: Optional[str], fmt: str = "auto") -> tuple[str, Optional[str], Optional[str]]:
    """Parses a caption into (style, lyrics or None, abc or None).

    ``json``: keys ``style|caption|tags``, ``lyrics``, ``abc``; ``tags_lyrics``: ``[Tags]``/``[Lyrics]``/``[ABC]``
    sections (``[Duration]`` is ignored) or legacy ``<CAPTION>``/``<LYRICS>``/``<ABC>`` tags; ``plain``: the whole
    text is the style; ``auto``: json if it starts with ``{``, tags_lyrics if it has a ``[Lyrics]`` section or legacy
    tags, else plain.
    """
    if fmt not in AUDIO_CAPTION_FORMATS:
        raise ValueError(f"caption_format must be one of {AUDIO_CAPTION_FORMATS}, got {fmt!r}")
    text = text or ""
    stripped = text.strip()
    if fmt == "auto":
        fmt = _detect_caption_format(text, stripped)

    if fmt == "plain":
        return stripped, None, None
    if fmt == "tags_lyrics":
        return _parse_tagged_caption(text)

    data = json.loads(text) if stripped else {}
    if not isinstance(data, dict):
        raise ValueError("JSON caption must be an object")
    style = None
    for key in ("style", "caption", "tags"):
        if data.get(key) is not None:
            style = _optional_str(data[key], f"caption {key}")
            break
    lyrics = _optional_str(data.get("lyrics"), "caption lyrics")
    abc = _optional_str(data.get("abc"), "caption abc")
    return (style or "").strip(), (lyrics.strip() if lyrics is not None else None), abc


def split_abc_mode_header(abc: str) -> tuple[str, Optional[str]]:
    """Strips an optional ``%%yue2_abc_mode <melody|full>`` first line; returns (score, mode or None)."""
    lines = abc.lstrip("﻿").split("\n", 1)
    first = lines[0].strip()
    if not first.startswith(ABC_MODE_HEADER):
        return abc, None
    mode = first[len(ABC_MODE_HEADER) :].strip().strip(":").strip()
    if mode not in ABC_MODES:
        raise ValueError(f"ABC header {first!r}: mode must be one of {ABC_MODES}")
    return (lines[1] if len(lines) > 1 else ""), mode


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def _audio_source_for(path: str) -> AudioSource:
    resolved = Path(path).resolve()
    return AudioSource(path=resolved, embedded=resolved.suffix.lower() not in AUDIO_SIDECAR_EXTENSIONS)


class AudioDatasource(ContentDatasource):
    """Records of an audio dataset. Fetchers return ``(idx, record, waveform, extent)``; ``waveform`` and ``extent``
    are None in caption-only mode (text caching). Every fetcher carries ``fetch.datasource_index``."""

    def __init__(
        self,
        caption_extension: Optional[str] = ".caption.txt",
        lyrics_extension: Optional[str] = ".lyrics.txt",
        abc_extension: Optional[str] = ".abc.txt",
        song_id_extension: Optional[str] = ".song.txt",
        caption_format: str = "auto",
        trigger: Optional[str] = None,
        abc_mode: Optional[str] = None,
    ):
        super().__init__()
        if caption_format not in AUDIO_CAPTION_FORMATS:
            raise ValueError(f"caption_format must be one of {AUDIO_CAPTION_FORMATS}, got {caption_format!r}")
        if abc_mode is not None and abc_mode not in ABC_MODES:
            raise ValueError(f"abc_mode must be one of {ABC_MODES}, got {abc_mode!r}")
        self.caption_extension = caption_extension
        self.lyrics_extension = lyrics_extension
        self.abc_extension = abc_extension
        self.song_id_extension = song_id_extension
        self.caption_format = caption_format
        self.trigger = trigger.strip() if trigger and trigger.strip() else None
        self.abc_mode = abc_mode
        self.records: list[AudioRecord] = []
        self.audio_spec: Optional[AudioSpec] = None
        self.current_idx = 0
        self._missing_caption = 0
        self._missing_lyrics = 0

    def _sidecar(self, audio_path: str, extension: Optional[str]) -> Optional[str]:
        if not extension:
            return None
        path = os.path.splitext(audio_path)[0] + extension
        return _read_text(path) if os.path.isfile(path) else None

    def _build_record(self, audio_path: str, item_key: str, fields: Mapping[str, Any], label: str) -> AudioRecord:
        caption = _optional_str(fields.get("caption"), f"{label}: caption")
        if caption is None and fields.get("style") is None:
            caption = self._sidecar(audio_path, self.caption_extension)
            if caption is None:
                self._missing_caption += 1
        try:
            style, lyrics, abc = parse_yue2_caption(caption, self.caption_format)
        except ValueError as e:
            raise ValueError(f"{label}: {e}") from e
        if fields.get("style") is not None:
            style = _optional_str(fields["style"], f"{label}: style").strip()

        if fields.get("lyrics") is not None:
            lyrics = _optional_str(fields["lyrics"], f"{label}: lyrics").strip()
        if lyrics is None:
            sidecar = self._sidecar(audio_path, self.lyrics_extension)
            lyrics = sidecar.strip() if sidecar is not None else None
        if lyrics is None:
            self._missing_lyrics += 1

        if fields.get("abc") is not None:
            abc = _optional_str(fields["abc"], f"{label}: abc")
        if abc is None:
            abc = self._sidecar(audio_path, self.abc_extension)
        header_mode = None
        if abc is not None:
            try:
                abc, header_mode = split_abc_mode_header(abc)
            except ValueError as e:
                raise ValueError(f"{label}: {e}") from e
            abc = abc.strip() or None

        mode = _optional_str(fields.get("abc_mode"), f"{label}: abc_mode") or header_mode or self.abc_mode
        if mode is not None and mode not in ABC_MODES:
            raise ValueError(f"{label}: abc_mode must be one of {ABC_MODES}, got {mode!r}")
        if abc is None:
            mode = None
        elif mode is None:
            from musubi_tuner.yue2.yue2_protocol import abc_mode_of

            mode = abc_mode_of(abc)

        song_id = fields.get("song_id")
        song_id = str(song_id).strip() if song_id is not None else ""
        if not song_id:
            sidecar = self._sidecar(audio_path, self.song_id_extension)
            song_id = sidecar.strip() if sidecar is not None else ""
        if not song_id:
            song_id = Path(audio_path).stem

        if self.trigger is not None:
            style = f"{self.trigger}, {style}" if style else self.trigger

        start_s = fields.get("start")
        end_s = fields.get("end")
        for name, value in (("start", start_s), ("end", end_s)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{label}: {name} must be a number of seconds")
        start_s = float(start_s) if start_s is not None else None
        end_s = float(end_s) if end_s is not None else None
        if start_s is not None and start_s < 0:
            raise ValueError(f"{label}: start must be nonnegative")
        if end_s is not None and end_s <= (start_s or 0.0):
            raise ValueError(f"{label}: end must be greater than start")

        return AudioRecord(
            audio_path=audio_path,
            item_key=item_key,
            style=style,
            lyrics=lyrics,
            abc=abc,
            abc_mode=mode,
            song_id=song_id,
            start_s=start_s,
            end_s=end_s,
            source=_audio_source_for(audio_path),
        )

    def _log_summary(self) -> None:
        if self._missing_caption:
            logger.warning(f"{self._missing_caption} audio records have no caption/style (empty style)")
        if self._missing_lyrics:
            logger.warning(
                f"{self._missing_lyrics} audio records have no lyrics; they are cached with the instrumental lyrics placeholder"
            )
        num_abc = sum(1 for r in self.records if r.abc is not None)
        logger.info(f"audio records: {len(self.records)}, with ABC: {num_abc}")

    def set_audio_spec(self, audio_spec: Optional[AudioSpec]):
        """Enables decoding and checks every record's audio file up front (fail-fast)."""
        self.audio_spec = audio_spec
        if audio_spec is None:
            return
        bad = []
        for record in self.records:
            path = record.source.path
            try:
                ok = path.is_file() and probe_audio(path)
            except Exception:
                ok = False
            if not ok:
                bad.append(str(path))
        if bad:
            raise ValueError(f"{len(bad)} audio files are missing or have no audio stream: {bad[:10]}")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.records)

    def get_record(self, idx: int) -> AudioRecord:
        return self.records[idx]

    def get_caption(self, idx: int) -> tuple[str, str]:
        record = self.records[idx]
        return record.item_key, record.style

    def decode_with_extent(self, idx: int, *, sample_rate: int, channels: int) -> tuple[torch.Tensor, AudioExtent]:
        """Decodes the record (cropped to its start/end) and reports where it sits in the file."""
        record = self.records[idx]
        waveform = decode_audio(record.source, sample_rate=sample_rate, channels=channels)
        file_samples = waveform.shape[1]
        start = int(round(record.start_s * sample_rate)) if record.start_s is not None else 0
        end = int(round(record.end_s * sample_rate)) if record.end_s is not None else file_samples
        if start >= file_samples:
            raise ValueError(
                f"{record.audio_path}: start {record.start_s}s is beyond the audio end ({file_samples / sample_rate:.2f}s)"
            )
        if end > file_samples:
            if end - file_samples > sample_rate // 10:
                logger.warning(
                    f"{record.audio_path}: end {record.end_s}s is beyond the audio end ({file_samples / sample_rate:.2f}s); clamped"
                )
            end = file_samples
        if start > 0 or end < file_samples:
            waveform = waveform[:, start:end].contiguous()
        return waveform, AudioExtent(start_sample=start, end_sample=end, file_samples=file_samples)

    def decode(self, idx: int, *, sample_rate: int, channels: int) -> torch.Tensor:
        return self.decode_with_extent(idx, sample_rate=sample_rate, channels=channels)[0]

    def _create_audio_fetcher(self, index: int):
        if self.caption_only:

            def fetch():
                return index, self.records[index], None, None

        else:

            def fetch():
                if self.audio_spec is None:
                    raise ValueError("Audio decoding is not enabled for this datasource; call set_audio_spec first")
                waveform, extent = self.decode_with_extent(
                    index, sample_rate=self.audio_spec.sample_rate, channels=self.audio_spec.channels
                )
                return index, self.records[index], waveform, extent

        fetch.datasource_index = index
        return fetch

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.records):
            raise StopIteration
        fetcher = self._create_audio_fetcher(self.current_idx)
        self.current_idx += 1
        return fetcher


class AudioDirectoryDatasource(AudioDatasource):
    def __init__(
        self,
        audio_directory: str,
        caption_extension: Optional[str] = ".caption.txt",
        lyrics_extension: Optional[str] = ".lyrics.txt",
        abc_extension: Optional[str] = ".abc.txt",
        song_id_extension: Optional[str] = ".song.txt",
        caption_format: str = "auto",
        trigger: Optional[str] = None,
        abc_mode: Optional[str] = None,
    ):
        super().__init__(caption_extension, lyrics_extension, abc_extension, song_id_extension, caption_format, trigger, abc_mode)
        self.audio_directory = audio_directory

        logger.info(f"glob audio files in {audio_directory}")
        if not os.path.isdir(audio_directory):
            raise ValueError(f"audio_directory does not exist: {audio_directory}")
        paths = sorted(
            os.path.join(audio_directory, name)
            for name in os.listdir(audio_directory)
            if os.path.splitext(name)[1].lower() in AUDIO_SIDECAR_EXTENSIONS and os.path.isfile(os.path.join(audio_directory, name))
        )
        # sidecars are matched by stem, so stems must be unique (case-insensitively, for Windows file systems)
        seen: dict[str, str] = {}
        for path in paths:
            stem = Path(path).stem
            if stem.casefold() in seen:
                raise ValueError(f"Audio files must have unique stems for sidecars: {seen[stem.casefold()]} and {path}")
            seen[stem.casefold()] = path
        self.records = [self._build_record(path, Path(path).stem, {}, path) for path in paths]
        logger.info(f"found {len(self.records)} audio files")
        self._log_summary()

    def get_item_extras(self, idx: int) -> ItemExtras:
        return ItemExtras(fields={}, base_directory=self.audio_directory, label=self.records[idx].audio_path)


class AudioJsonlDatasource(AudioDatasource):
    PATH_KEYS = ("audio_path",)
    # the shared audio JSONL schema; every other key is an item extra
    SHARED_KEYS = ("audio_path", "caption", "style", "lyrics", "abc", "abc_mode", "song_id", "start", "end")

    def __init__(
        self,
        audio_jsonl_file: str,
        caption_extension: Optional[str] = ".caption.txt",
        lyrics_extension: Optional[str] = ".lyrics.txt",
        abc_extension: Optional[str] = ".abc.txt",
        song_id_extension: Optional[str] = ".song.txt",
        caption_format: str = "auto",
        trigger: Optional[str] = None,
        abc_mode: Optional[str] = None,
    ):
        super().__init__(caption_extension, lyrics_extension, abc_extension, song_id_extension, caption_format, trigger, abc_mode)
        self.audio_jsonl_file = audio_jsonl_file

        logger.info(f"load audio jsonl from {audio_jsonl_file}")
        self.data: list[dict] = []
        with open(audio_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError(f"{audio_jsonl_file}: every line must be a JSON object")
                self.data.append(data)

        # relative paths: the working directory first, then the JSONL's own directory (as VideoJsonlDatasource)
        base_directory = os.path.dirname(os.path.abspath(audio_jsonl_file))
        for index, data in enumerate(self.data):
            value = data.get("audio_path")
            if not isinstance(value, str) or not value:
                raise ValueError(f"{os.path.basename(audio_jsonl_file)} line {index + 1}: audio_path is required")
            if os.path.isabs(value):
                continue
            jsonl_candidate = os.path.join(base_directory, value)
            if os.path.exists(value):
                data["audio_path"] = os.path.abspath(value)
            elif os.path.exists(jsonl_candidate):
                data["audio_path"] = jsonl_candidate

        # item keys are file stems; a repeated stem (excerpts of one file) gets "-r{line index}"
        used: set[str] = set()
        records = []
        for index, data in enumerate(self.data):
            label = f"{os.path.basename(audio_jsonl_file)} line {index + 1}"
            key = Path(data["audio_path"]).stem
            if key.casefold() in used:
                key = f"{key}-r{index:05d}"
                if key.casefold() in used:
                    raise ValueError(f"{label}: item key {key} collides with another record")
            used.add(key.casefold())
            records.append(self._build_record(data["audio_path"], key, data, label))
        self.records = records
        logger.info(f"loaded {len(self.records)} audio records")
        self._log_summary()

    def get_item_extras(self, idx: int) -> ItemExtras:
        return ItemExtras(
            fields=_extra_fields(self.data[idx], AudioJsonlDatasource.SHARED_KEYS),
            base_directory=os.path.dirname(os.path.abspath(self.audio_jsonl_file)),
            label=f"{os.path.basename(self.audio_jsonl_file)} line {idx + 1}",
        )
