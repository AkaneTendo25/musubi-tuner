from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ItemInfo
from musubi_tuner.minimax_h3.architecture import is_valid_frame_count

logger = logging.getLogger(__name__)


class H3AudioDataset(BaseDataset):
    """H3-local audio datasource using video-frame counts as the duration grid."""

    def __init__(self, config: dict[str, Any], general: dict[str, Any]):
        def value(key: str, default: Any = None) -> Any:
            return config.get(key, general.get(key, default))

        frames = value("target_frames")
        if not frames or len(frames) != 1 or not is_valid_frame_count(int(frames[0])):
            raise ValueError("H3 audio datasets require one target_frames value satisfying frame_count % 17 == 5")
        self.target_frames = int(frames[0])
        configured_resolution = value("resolution", (768, 768))
        # The Musubi schema accepts a scalar resolution as a square one, but this dataset
        # is built from the raw config, before that conversion runs.
        if isinstance(configured_resolution, int):
            configured_resolution = (configured_resolution, configured_resolution)
        resolution = tuple(configured_resolution)
        super().__init__(
            resolution=resolution,
            caption_extension=value("caption_extension", ".txt"),
            batch_size=int(value("batch_size", 1)),
            num_repeats=int(value("num_repeats", 1)),
            cache_directory=value("cache_directory"),
            architecture=ARCHITECTURE_MINIMAX_H3,
        )
        audio_directory = value("audio_directory")
        audio_jsonl_file = value("audio_jsonl_file")
        if bool(audio_directory) == bool(audio_jsonl_file):
            raise ValueError("H3 audio datasets require exactly one of audio_directory or audio_jsonl_file")
        if audio_directory:
            root = Path(audio_directory)
            extensions = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
            paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions)
            self.records = [(path, self._sidecar_caption(path)) for path in paths]
            self.cache_directory = self.cache_directory or str(root)
        else:
            source = Path(audio_jsonl_file)
            records = []
            with source.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                        records.append((Path(record["audio_path"]), str(record.get("caption", ""))))
                    except (json.JSONDecodeError, KeyError) as error:
                        raise ValueError(f"invalid H3 audio JSONL record on line {line_number} of {source}") from error
            self.records = records
            self.cache_directory = self.cache_directory or str(source.parent)
        if not self.records:
            raise ValueError("H3 audio dataset contains no supported audio files")
        self.num_train_items = 0
        self.batch_manager = None

    def _sidecar_caption(self, path: Path) -> str:
        caption = path.with_suffix(self.caption_extension)
        return caption.read_text(encoding="utf-8").strip() if caption.is_file() else ""

    def _item(self, path: Path, caption: str) -> ItemInfo:
        item = ItemInfo(str(path), caption, tuple(self.resolution), (*self.resolution, self.target_frames), self.target_frames)
        item.latent_cache_path = self.get_latent_cache_path(item)
        item.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item)
        return item

    @staticmethod
    def _cache_basename(item_key: str) -> str:
        """Name audio caches ``<stem>_audio<hash>`` so they cannot collide with same-stem video caches.

        The hash is the first 8 hex digits of SHA-256 over the absolute source path, which keeps the
        name stable across runs and machines while separating same-stem files from different folders.
        """
        path = Path(item_key).expanduser()
        identity = os.path.normcase(os.path.abspath(str(path))).replace(os.sep, "/")
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
        return f"{path.stem}_audio{digest}"

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        width, height = self.resolution
        return os.path.join(
            self.cache_directory,
            f"{self._cache_basename(item_info.item_key)}_00000-{self.target_frames:03d}"
            f"_{width:04d}x{height:04d}_{self.architecture}.safetensors",
        )

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        return os.path.join(
            self.cache_directory,
            f"{self._cache_basename(item_info.item_key)}_00000-{self.target_frames:03d}_{self.architecture}_te.safetensors",
        )

    def retrieve_latent_cache_batches(self, num_workers: int):
        del num_workers
        for path, caption in self.records:
            yield (*self.resolution, self.target_frames), [self._item(path, caption)]

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        del num_workers
        for path, caption in self.records:
            yield [self._item(path, caption)]

    def prepare_for_training(self, num_timestep_buckets: int | None = None):
        bucket = []
        skipped: list[str] = []
        for path, caption in self.records:
            item = self._item(path, caption)
            if not Path(item.latent_cache_path).is_file() or not Path(item.text_encoder_output_cache_path).is_file():
                skipped.append(item.latent_cache_path)
                continue
            bucket.extend([item] * self.num_repeats)
        if skipped:
            logger.warning(
                f"H3 audio dataset skips {len(skipped)} of {len(self.records)} items without both caches, e.g. {skipped[0]}"
            )
        bucket_key = (*self.resolution, self.target_frames)
        self.batch_manager = BucketBatchManager({bucket_key: bucket}, self.batch_size, num_timestep_buckets)
        self.num_train_items = len(bucket)

    def shuffle_buckets(self):
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        return len(self.batch_manager) if self.batch_manager is not None else len(self.records)

    def __getitem__(self, index):
        super().__getitem__(index)
        return self.batch_manager[index]

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata["target_modality"] = "audio"
        metadata["target_frames"] = self.target_frames
        return metadata
