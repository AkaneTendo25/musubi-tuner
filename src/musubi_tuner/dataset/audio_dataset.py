"""Audio (music) dataset: one latent cache per segment of a record, one text cache per record.

A record is an audio file (``audio_directory``) or a JSONL line (``audio_jsonl_file``, optionally an excerpt with
``start``/``end`` seconds). Segments are counted in codec frames of ``audio_spec.samples_per_crop(1)`` samples. Latent
caching hands every segment the **whole record** waveform (``item.audio_content``) plus its frame window
(``item.frame_pos``/``item.frame_count``), so cache scripts encode a record once and slice it; text caching yields one
item per record.

Song coordinates (for AR gating): ``item.song_start_frame`` is the segment start counted from the start of the audio
file, ``item.song_frames`` the frame length of the whole file, and ``item.truncated`` tells that the record ends before
the file does (``max_seconds`` or a JSONL ``end``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import os
import random
import re
import time
from typing import Iterable, Optional, Tuple

import torch
from safetensors import safe_open

from musubi_tuner.dataset.audio_utils import AudioSpec, slice_audio_window
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.datasources import (
    ABC_MODES,
    AUDIO_CAPTION_FORMATS,
    AudioDatasource,
    AudioDirectoryDatasource,
    AudioExtent,
    AudioJsonlDatasource,
    AudioRecord,
)
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ItemInfo

import logging

logger = logging.getLogger(__name__)

AUDIO_SEGMENT_EXTRACTIONS = ("full", "head", "chunk", "slide")
DEFAULT_AUDIO_CAPTION_EXTENSION = ".caption.txt"
# the channel count written as the "height" of a segment's original_size (cache metadata only)
AUDIO_LATENT_CHANNELS = 64
# text cache metadata key naming the record's song identity (read back by prepare_for_training)
SONG_ID_METADATA_KEY = "yue2_song_id"

_SEGMENT_TOKEN = re.compile(r"(\d+)-(\d+)")


def validation_song_split(song_ids: Iterable[str], validation_split: float, seed: int) -> set[str]:
    """Song ids held out for validation.

    Distinct songs are ranked by the hex ``sha256(f"{seed}:{song}")`` (ties by id) and the first
    ``min(n - 1, max(1, round(n * split)))`` are held out; none when split <= 0 or there are fewer than two songs.
    The assignment is persisted behaviour, so the ranking must not change.
    """

    def rank(song) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{song}".encode("utf-8")).hexdigest()
        return digest, song

    ordered = sorted(set(song_ids), key=rank)
    n = len(ordered)
    if not validation_split or validation_split <= 0 or n < 2:
        return set()
    held = min(n - 1, max(1, round(n * validation_split)))
    return set(ordered[:held])


def segment_windows(
    total_frames: int,
    extraction: str,
    segment_frames: Optional[int] = None,
    stride_frames: Optional[int] = None,
    max_segments: Optional[int] = None,
) -> list[tuple[int, int]]:
    """(start_frame, n_frames) windows inside a record of ``total_frames``.

    full: the whole record; head: the first segment; chunk: consecutive segments, the tail dropped; slide: windows every
    ``stride_frames``. Records shorter than one segment give no window. ``max_segments`` keeps that many windows spread
    evenly over the record (first and last included).
    """
    if total_frames < 1:
        return []
    if extraction == "full":
        windows = [(0, total_frames)]
    else:
        if segment_frames is None or segment_frames < 1:
            raise ValueError(f"segment_extraction={extraction} needs a positive segment length")
        if total_frames < segment_frames:
            return []
        if extraction == "head":
            windows = [(0, segment_frames)]
        elif extraction == "chunk":
            windows = [(start, segment_frames) for start in range(0, total_frames - segment_frames + 1, segment_frames)]
        elif extraction == "slide":
            if stride_frames is None or stride_frames < 1:
                raise ValueError("segment_extraction=slide needs a positive stride")
            windows = [(start, segment_frames) for start in range(0, total_frames - segment_frames + 1, stride_frames)]
        else:
            raise ValueError(f"segment_extraction must be one of {AUDIO_SEGMENT_EXTRACTIONS}, got {extraction!r}")
    if max_segments is not None and len(windows) > max_segments:
        if max_segments == 1:
            windows = windows[:1]
        else:
            last = len(windows) - 1
            picks = sorted({round(i * last / (max_segments - 1)) for i in range(max_segments)})
            windows = [windows[i] for i in picks]
    return windows


class AudioDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int] = (960, 544),
        caption_extension: Optional[str] = None,
        batch_size: int = 1,
        num_repeats: int = 1,
        enable_bucket: bool = False,
        bucket_no_upscale: bool = False,
        audio_directory: Optional[str] = None,
        audio_jsonl_file: Optional[str] = None,
        lyrics_extension: Optional[str] = ".lyrics.txt",
        abc_extension: Optional[str] = ".abc.txt",
        song_id_extension: Optional[str] = ".song.txt",
        caption_format: str = "auto",
        trigger: Optional[str] = None,
        segment_extraction: str = "full",
        segment_seconds: Optional[float] = None,
        segment_stride_seconds: Optional[float] = None,
        max_segments: Optional[int] = None,
        min_seconds: float = 5.0,
        max_seconds: float = 360.0,
        validation_split: float = 0.0,
        validation_split_seed: int = 0,
        is_validation: bool = False,
        abc_mode: Optional[str] = None,
        cache_directory: Optional[str] = None,
        debug_dataset: bool = False,
        architecture: str = "no_default",
        audio_spec: Optional[AudioSpec] = None,
    ):
        if caption_extension is None:
            caption_extension = DEFAULT_AUDIO_CAPTION_EXTENSION
        super(AudioDataset, self).__init__(
            resolution,
            caption_extension,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            cache_directory,
            debug_dataset,
            architecture,
        )
        if (audio_directory is None) == (audio_jsonl_file is None):
            raise ValueError("exactly one of audio_directory or audio_jsonl_file must be specified")
        if segment_extraction not in AUDIO_SEGMENT_EXTRACTIONS:
            raise ValueError(f"segment_extraction must be one of {AUDIO_SEGMENT_EXTRACTIONS}, got {segment_extraction!r}")
        if segment_extraction != "full" and (segment_seconds is None or segment_seconds <= 0):
            raise ValueError(f"segment_extraction={segment_extraction} requires a positive segment_seconds")
        if segment_extraction == "slide":
            if segment_stride_seconds is None:
                segment_stride_seconds = segment_seconds
                logger.info(f"segment_stride_seconds is not set; using segment_seconds={segment_seconds}")
            if segment_stride_seconds <= 0:
                raise ValueError("segment_stride_seconds must be positive")
        if max_segments is not None and max_segments < 1:
            raise ValueError("max_segments must be at least 1")
        if min_seconds < 0 or max_seconds <= 0 or max_seconds < min_seconds:
            raise ValueError(f"invalid min_seconds/max_seconds: {min_seconds}/{max_seconds}")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError(f"validation_split must be in [0, 1), got {validation_split}")
        if caption_format not in AUDIO_CAPTION_FORMATS:
            raise ValueError(f"caption_format must be one of {AUDIO_CAPTION_FORMATS}, got {caption_format!r}")
        if abc_mode is not None and abc_mode not in ABC_MODES:
            raise ValueError(f"abc_mode must be one of {ABC_MODES}, got {abc_mode!r}")
        if is_validation and validation_split > 0:
            logger.warning("is_validation=true: the whole dataset is used for validation and validation_split is ignored")

        self.audio_directory = audio_directory
        self.audio_jsonl_file = audio_jsonl_file
        self.lyrics_extension = lyrics_extension
        self.abc_extension = abc_extension
        self.song_id_extension = song_id_extension
        self.caption_format = caption_format
        self.trigger = trigger
        self.segment_extraction = segment_extraction
        self.segment_seconds = segment_seconds
        self.segment_stride_seconds = segment_stride_seconds
        self.max_segments = max_segments
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.validation_split = validation_split
        self.validation_split_seed = validation_split_seed
        self.is_validation = is_validation
        self.abc_mode = abc_mode

        source_args = (caption_extension, lyrics_extension, abc_extension, song_id_extension, caption_format, trigger, abc_mode)
        if audio_directory is not None:
            self.datasource: AudioDatasource = AudioDirectoryDatasource(audio_directory, *source_args)
        else:
            self.datasource = AudioJsonlDatasource(audio_jsonl_file, *source_args)

        self.audio_spec = audio_spec
        if audio_spec is not None:
            self.datasource.set_audio_spec(audio_spec)

        if self.cache_directory is None:
            if audio_directory is None:
                raise ValueError("cache_directory is required for audio_jsonl_file datasets")
            self.cache_directory = audio_directory

        # the cache scripts never call prepare_for_training; DatasetGroup reads num_train_items at construction
        self.batch_manager = None
        self.num_train_items = 0
        self.validation_items: list[ItemInfo] = []
        self.has_control = False
        self.dropped_short = 0
        self.dropped_no_segment = 0

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.audio_directory is not None:
            metadata["audio_directory"] = os.path.basename(self.audio_directory)
        if self.audio_jsonl_file is not None:
            metadata["audio_jsonl_file"] = os.path.basename(self.audio_jsonl_file)
        metadata["segment_extraction"] = self.segment_extraction
        metadata["segment_seconds"] = self.segment_seconds
        metadata["segment_stride_seconds"] = self.segment_stride_seconds
        metadata["max_segments"] = self.max_segments
        metadata["min_seconds"] = self.min_seconds
        metadata["max_seconds"] = self.max_seconds
        metadata["validation_split"] = self.validation_split
        metadata["validation_split_seed"] = self.validation_split_seed
        metadata["is_validation"] = self.is_validation
        return metadata

    # frame grid

    def _frame_samples(self) -> int:
        if self.audio_spec is None:
            raise ValueError("this operation needs the architecture's AudioSpec (pass audio_spec to the dataset group)")
        hop = self.audio_spec.samples_per_crop(1)
        if hop < 1:
            raise ValueError(f"AudioSpec.samples_per_crop(1) must be positive, got {hop}")
        return hop

    def _seconds_to_frames(self, seconds: float) -> int:
        return int(round(seconds * self.audio_spec.sample_rate / self._frame_samples()))

    def segments_for(self, total_frames: int) -> list[tuple[int, int]]:
        segment_frames = self._seconds_to_frames(self.segment_seconds) if self.segment_seconds else None
        stride_frames = self._seconds_to_frames(self.segment_stride_seconds) if self.segment_stride_seconds else None
        return segment_windows(total_frames, self.segment_extraction, segment_frames, stride_frames, self.max_segments)

    # cache paths

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        name = f"{item_info.item_key}_{item_info.frame_pos:06d}-{item_info.frame_count:06d}_{self.architecture}.safetensors"
        return os.path.join(self.cache_directory, name)

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{item_info.item_key}_{self.architecture}_te.safetensors")

    # caching

    def _stamp_record(self, item: ItemInfo, index: int, record: AudioRecord) -> None:
        item.dataset_index = self.dataset_index
        item.datasource_index = index
        item.lyrics = record.lyrics
        item.abc_text = record.abc
        item.abc_mode = record.abc_mode
        item.song_id = record.song_id
        item.audio_source = record.source
        item.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item)

    def _segment_items(self, index: int, record: AudioRecord, waveform: torch.Tensor, extent: AudioExtent) -> list[ItemInfo]:
        hop = self._frame_samples()
        sample_rate = self.audio_spec.sample_rate
        length = waveform.shape[1]
        max_samples = int(round(self.max_seconds * sample_rate))
        total_frames = min(length, max_samples) // hop
        if total_frames < 1 or total_frames * hop < self.min_seconds * sample_rate:
            self.dropped_short += 1
            logger.info(f"skip {record.audio_path}: {length / sample_rate:.2f}s is shorter than min_seconds={self.min_seconds}")
            return []
        windows = self.segments_for(total_frames)
        if not windows:
            self.dropped_no_segment += 1
            logger.info(f"skip {record.audio_path}: {total_frames} frames are shorter than one segment")
            return []

        song_frames = extent.file_samples // hop
        record_offset = extent.start_sample // hop
        # the record stops before the song does: max_seconds truncation or a JSONL end before the file end
        truncated = record_offset + total_frames < song_frames
        audio = slice_audio_window(
            waveform, start_sample=0, sample_count=total_frames * hop, pad_tolerance=hop, context=record.audio_path
        )

        items = []
        for start, frames in windows:
            item = ItemInfo(
                record.item_key, record.style, (frames, AUDIO_LATENT_CHANNELS), (frames,), frame_count=frames, content=None
            )
            item.frame_pos = start
            item.latent_cache_path = self.get_latent_cache_path(item)
            self._stamp_record(item, index, record)
            item.audio_content = audio  # shared by all segments of the record
            item.audio_present = True
            item.record_frames = total_frames
            item.song_frames = song_frames
            item.song_start_frame = record_offset + start
            item.truncated = truncated
            items.append(item)
        return items

    def retrieve_latent_cache_batches(self, num_workers: int):
        """Yields ((n_frames,), [ItemInfo]) batches of at most batch_size segments; a batch never mixes records."""
        self._frame_samples()
        num_workers = max(1, num_workers)
        self.datasource.set_caption_only(False)
        self.dropped_short = 0
        self.dropped_no_segment = 0
        executor = ThreadPoolExecutor(max_workers=num_workers)
        futures = []
        ready: list[tuple[tuple[int], list[ItemInfo]]] = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed = [future for future in futures if future.done()]
                if not completed:
                    time.sleep(0.05)
                    continue
                for future in completed:
                    index, record, waveform, extent = future.result()
                    items = self._segment_items(index, record, waveform, extent)
                    by_frames: dict[int, list[ItemInfo]] = {}
                    for item in items:
                        by_frames.setdefault(item.frame_count, []).append(item)
                    for frames, group in by_frames.items():
                        for i in range(0, len(group), self.batch_size):
                            ready.append(((frames,), group[i : i + self.batch_size]))
                    futures.remove(future)

        try:
            for fetch_op in self.datasource:
                futures.append(executor.submit(fetch_op))
                aggregate_future()
                while ready:
                    yield ready.pop(0)
            aggregate_future(consume_all=True)
            while ready:
                yield ready.pop(0)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if self.dropped_short or self.dropped_no_segment:
            logger.warning(
                f"skipped {self.dropped_short} records shorter than min_seconds and {self.dropped_no_segment} records"
                " shorter than one segment"
            )

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        """Yields lists of at most batch_size ItemInfo, one per record (no audio is decoded)."""
        batch: list[ItemInfo] = []
        for index in range(len(self.datasource)):
            record = self.datasource.get_record(index)
            item = ItemInfo(record.item_key, record.style, (0, 0), (0, 0))
            self._stamp_record(item, index, record)
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # training

    def _cached_song_id(self, te_path: str) -> Optional[str]:
        try:
            with safe_open(te_path, framework="pt", device="cpu") as handle:
                return (handle.metadata() or {}).get(SONG_ID_METADATA_KEY)
        except Exception as e:
            logger.warning(f"cannot read text cache metadata {te_path}: {e}")
            return None

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        latent_cache_files = sorted(glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors")))
        record_song_ids = {record.item_key: record.song_id for record in self.datasource.records}

        entries: list[ItemInfo] = []
        for cache_file in latent_cache_files:
            stem = os.path.basename(cache_file)[: -len(".safetensors")]
            tokens = stem.split("_")
            if len(tokens) < 3 or tokens[-1] != self.architecture:
                continue
            m = _SEGMENT_TOKEN.fullmatch(tokens[-2])
            if m is None:
                logger.warning(f"Not an audio segment cache file, skipped: {cache_file}")
                continue
            frame_pos, frame_count = int(m.group(1)), int(m.group(2))
            item_key = "_".join(tokens[:-2])
            te_path = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
            if not os.path.exists(te_path):
                logger.warning(f"Text encoder output cache file not found: {te_path}")
                continue
            song_id = record_song_ids.get(item_key) or self._cached_song_id(te_path) or item_key

            item = ItemInfo(
                item_key,
                "",
                (frame_count, AUDIO_LATENT_CHANNELS),
                (frame_count,),
                frame_count=frame_count,
                latent_cache_path=cache_file,
            )
            item.text_encoder_output_cache_path = te_path
            item.frame_pos = frame_pos
            item.song_id = song_id
            item.dataset_index = self.dataset_index
            entries.append(item)

        if self.is_validation:
            held = {item.song_id for item in entries}
        else:
            held = validation_song_split((item.song_id for item in entries), self.validation_split, self.validation_split_seed)

        bucketed_item_info: dict[tuple[int], list[ItemInfo]] = {}
        self.validation_items = []
        for item in entries:
            if item.song_id in held:
                self.validation_items.append(item)
                continue
            bucket = bucketed_item_info.setdefault(item.bucket_size, [])
            for _ in range(self.num_repeats):
                bucket.append(item)
        if self.validation_items:
            logger.info(f"validation: {len(held)} songs, {len(self.validation_items)} segments (excluded from training)")

        self.batch_manager = BucketBatchManager(bucketed_item_info, self.batch_size, num_timestep_buckets=num_timestep_buckets)
        self.batch_manager.show_bucket_info()
        self.num_train_items = sum(len(bucket) for bucket in bucketed_item_info.values())

    def validation_song_ids(self) -> set[str]:
        return {item.song_id for item in self.validation_items}

    def shuffle_buckets(self):
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        super().__getitem__(idx)
        return self.batch_manager[idx]
