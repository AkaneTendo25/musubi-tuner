"""Native cached-audio dataset used by MiniMax Music 3 training."""

from __future__ import annotations

import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Optional

import torch

from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ItemInfo


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}


class AudioItemInfo(ItemInfo):
    def __init__(self, item_key: str, caption: str, lyrics: str, samples: int = 0, **kwargs):
        super().__init__(item_key, caption, (samples, 2), frame_count=samples, **kwargs)
        self.lyrics = lyrics


class AudioDataset(BaseDataset):
    def __init__(
        self,
        audio_directory: Optional[str] = None,
        audio_jsonl_file: Optional[str] = None,
        caption_extension: str = ".txt",
        lyrics_extension: str = ".lyrics.txt",
        batch_size: int = 1,
        num_repeats: int = 1,
        cache_directory: Optional[str] = None,
        sample_rate: int = 44100,
        max_duration: Optional[float] = None,
        min_duration: float = 0.04,
        duration_bucket_interval: Optional[float] = None,
        strict_cache_validation: bool = True,
        debug_dataset: bool = False,
        architecture: str = "mm3",
        **_ignored,
    ):
        super().__init__((sample_rate, 2), caption_extension, batch_size, num_repeats, False, False, cache_directory, debug_dataset, architecture)
        if bool(audio_directory) == bool(audio_jsonl_file):
            raise ValueError("Specify exactly one of audio_directory or audio_jsonl_file")
        caption_extension = caption_extension or ".txt"
        lyrics_extension = lyrics_extension or ".lyrics.txt"
        self.caption_extension = caption_extension
        self.audio_directory = audio_directory
        self.audio_jsonl_file = audio_jsonl_file
        self.lyrics_extension = lyrics_extension
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        if duration_bucket_interval is not None and duration_bucket_interval <= 0:
            raise ValueError("duration_bucket_interval must be positive")
        self.duration_bucket_interval = duration_bucket_interval
        self.strict_cache_validation = strict_cache_validation
        self.cache_directory = cache_directory or audio_directory or str(Path(audio_jsonl_file).parent / "cache")
        os.makedirs(self.cache_directory, exist_ok=True)
        self.records = self._load_records()
        self.batch_manager = None
        self.num_train_items = 0

    def _load_records(self) -> list[dict]:
        if self.audio_jsonl_file:
            base = Path(self.audio_jsonl_file).parent
            records = []
            with open(self.audio_jsonl_file, encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    path = Path(record.get("audio_path", record.get("file", "")))
                    if not path.is_absolute():
                        path = base / path
                    records.append({"path": str(path), "caption": record.get("caption", ""), "lyrics": record.get("lyrics", "")})
            return records
        records = []
        for path in sorted(Path(self.audio_directory).rglob("*")):
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            caption_path = path.with_suffix(self.caption_extension)
            lyrics_path = path.with_suffix(self.lyrics_extension)
            records.append(
                {
                    "path": str(path),
                    "caption": caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else "",
                    "lyrics": lyrics_path.read_text(encoding="utf-8").strip() if lyrics_path.exists() else "",
                }
            )
        return records

    def _item(self, record: dict, load_audio: bool) -> AudioItemInfo:
        import soundfile as sf

        info = sf.info(record["path"])
        target_samples = round(info.frames * self.sample_rate / info.samplerate)
        if self.max_duration is not None:
            target_samples = min(target_samples, round(self.max_duration * self.sample_rate))
        if target_samples < round(self.min_duration * self.sample_rate):
            raise ValueError(f"Audio is shorter than min_duration: {record['path']}")
        item = AudioItemInfo(record["path"], record["caption"], record["lyrics"], target_samples)
        item.latent_cache_path = self.get_latent_cache_path(item)
        item.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item)
        if load_audio:
            waveform, source_rate = sf.read(record["path"], dtype="float32", always_2d=True)
            waveform = torch.from_numpy(waveform.T)
            if source_rate != self.sample_rate:
                import torchaudio

                waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            item.content = waveform[:2, :target_samples]
            item.original_size = (item.content.shape[-1], 2)
            item.frame_count = item.content.shape[-1]
        return item

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        stem = Path(item_info.item_key).stem.replace(" ", "_")
        return os.path.join(self.cache_directory, f"{stem}_{self.architecture}.safetensors")

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        stem = Path(item_info.item_key).stem.replace(" ", "_")
        return os.path.join(self.cache_directory, f"{stem}_{self.architecture}_te.safetensors")

    def retrieve_latent_cache_batches(self, num_workers: int):
        buckets = {}
        for record in self.records:
            item = self._item(record, True)
            key = item.content.shape[-1] // 512
            buckets.setdefault(key, []).append(item)
        for key in sorted(buckets):
            items = buckets[key]
            for start in range(0, len(items), self.batch_size):
                yield key, items[start : start + self.batch_size]

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        items = [self._item(record, False) for record in self.records]
        for start in range(0, len(items), self.batch_size):
            yield items[start : start + self.batch_size]

    def prepare_for_training(self, num_timestep_buckets=None):
        if self.strict_cache_validation:
            missing_latent = []
            missing_text = []
            stems = {}
            for record in self.records:
                item = self._item(record, False)
                stem = Path(item.latent_cache_path).name
                stems.setdefault(stem, []).append(record["path"])
                if not os.path.exists(item.latent_cache_path):
                    missing_latent.append(record["path"])
                if not os.path.exists(item.text_encoder_output_cache_path):
                    missing_text.append(record["path"])
            duplicate_stems = {stem: paths for stem, paths in stems.items() if len(paths) > 1}
            if missing_latent or missing_text or duplicate_stems:
                details = []
                if missing_latent:
                    details.append(f"missing latent caches={len(missing_latent)}")
                if missing_text:
                    details.append(f"missing text caches={len(missing_text)}")
                if duplicate_stems:
                    details.append(f"duplicate cache names={len(duplicate_stems)}")
                raise ValueError(
                    "MiniMax Music 3 cache validation failed: " + ", ".join(details) + ". "
                    "Run both cache stages, resolve duplicate audio basenames, or set strict_cache_validation=false."
                )
        buckets = {}
        for latent_path in glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors")):
            stem = Path(latent_path).name[: -len(f"_{self.architecture}.safetensors")]
            te_path = os.path.join(self.cache_directory, f"{stem}_{self.architecture}_te.safetensors")
            if not os.path.exists(te_path):
                continue
            from safetensors import safe_open

            with safe_open(latent_path, framework="pt") as file:
                key = next(key for key in file.keys() if key.startswith("latents_"))
                frames = file.get_slice(key).get_shape()[-1]
            item = AudioItemInfo(stem, "", "", frames * 512, latent_cache_path=latent_path)
            item.text_encoder_output_cache_path = te_path
            if self.duration_bucket_interval is None:
                bucket_frames = frames
            else:
                interval_samples = self.duration_bucket_interval * self.sample_rate
                interval_frames = max(1, round(interval_samples / 512))
                bucket_frames = math.ceil(frames / interval_frames) * interval_frames
            buckets.setdefault((bucket_frames,), []).extend([item] * self.num_repeats)
        self.batch_manager = BucketBatchManager(
            buckets,
            self.batch_size,
            num_timestep_buckets=num_timestep_buckets,
            pad_variable_tensors=self.duration_bucket_interval is not None,
        )
        self.num_train_items = sum(map(len, buckets.values()))

    def shuffle_buckets(self):
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        return 100 if self.batch_manager is None else len(self.batch_manager)

    def __getitem__(self, index):
        super().__getitem__(index)
        return self.batch_manager[index]
