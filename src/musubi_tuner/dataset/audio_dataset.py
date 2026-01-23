import glob
import json
import random
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

from .audio_utils import glob_audio
from .image_video_dataset import BaseDataset, BucketBatchManager, ContentDatasource, ItemInfo, save_latent_cache_ltx2
from ..utils.model_utils import str_to_dtype
import torch

logger = logging.getLogger(__name__)

class AudioDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

    def get_audio_data(self, idx: int) -> tuple[str, str]:
        """
        Returns audio data as a tuple of audio path and caption.
        Key must be unique and valid as a file name.
        May not be called if is_indexable() returns False.
        """
        raise NotImplementedError


class AudioDirectoryDatasource(AudioDatasource):
    def __init__(
        self,
        audio_directory: str,
        caption_extension: Optional[str] = None,
    ):
        super().__init__()
        self.audio_directory = audio_directory
        self.caption_extension = caption_extension
        self.current_idx = 0

        logger.info(f"glob audio in {self.audio_directory}")
        self.audio_paths = glob_audio(self.audio_directory)
        logger.info(f"found {len(self.audio_paths)} audio files")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.audio_paths)

    def get_audio_data(self, idx: int) -> tuple[str, str]:
        audio_path = self.audio_paths[idx]
        caption_path = os.path.splitext(audio_path)[0] + (self.caption_extension or "")
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return audio_path, caption

    def get_caption(self, idx: int) -> tuple[str, str]:
        return self.get_audio_data(idx)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        if self.current_idx >= len(self.audio_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:

            def create_audio_fetcher(index):
                return lambda: self.get_audio_data(index)

            fetcher = create_audio_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class AudioJsonlDatasource(AudioDatasource):
    def __init__(self, audio_jsonl_file: str):
        super().__init__()
        self.audio_jsonl_file = audio_jsonl_file
        self.current_idx = 0

        logger.info(f"load audio jsonl from {self.audio_jsonl_file}")
        self.data = []
        with open(self.audio_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"failed to load json: {line} @ {self.audio_jsonl_file}")
                    raise
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} audio items")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_audio_data(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        audio_path = data["audio_path"]
        caption = data["caption"]
        return audio_path, caption

    def get_caption(self, idx: int) -> tuple[str, str]:
        return self.get_audio_data(idx)

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

            def create_audio_fetcher(index):
                return lambda: self.get_audio_data(index)

            fetcher = create_audio_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class AudioDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        audio_directory: Optional[str] = None,
        audio_jsonl_file: Optional[str] = None,
        cache_directory: Optional[str] = None,
        reference_cache_directory: Optional[str] = None,
        separate_audio_buckets: bool = False,
        debug_dataset: bool = False,
        architecture: str = "no_default",
    ):
        super(AudioDataset, self).__init__(
            resolution,
            caption_extension,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            cache_directory,
            reference_cache_directory,
            separate_audio_buckets,
            debug_dataset,
            architecture,
        )
        self.audio_directory = audio_directory
        self.audio_jsonl_file = audio_jsonl_file

        if audio_directory is not None:
            self.datasource = AudioDirectoryDatasource(audio_directory, caption_extension)
        elif audio_jsonl_file is not None:
            self.datasource = AudioJsonlDatasource(audio_jsonl_file)
        else:
            raise ValueError("audio_directory or audio_jsonl_file must be specified")

        if self.cache_directory is None:
            self.cache_directory = self.audio_directory

        self.batch_manager = None
        self.num_train_items = 0

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.audio_directory is not None:
            metadata["audio_directory"] = os.path.basename(self.audio_directory)
        if self.audio_jsonl_file is not None:
            metadata["audio_jsonl_file"] = os.path.basename(self.audio_jsonl_file)
        return metadata

    def _dummy_video_cache_path(self, item_key: str) -> str:
        basename = os.path.splitext(os.path.basename(item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_0001x0001_{self.architecture}.safetensors")

    def _strip_dummy_resolution(self, item_key: str) -> str:
        suffix = "_0001x0001"
        return item_key[: -len(suffix)] if item_key.endswith(suffix) else item_key

    def retrieve_latent_cache_batches(self, num_workers: int):
        executor = ThreadPoolExecutor(max_workers=num_workers)
        data: list[ItemInfo] = []
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:
                        time.sleep(0.1)
                        continue
                    break

                for future in completed_futures:
                    audio_path, caption = future.result()
                    bucket_reso = self._append_audio_bucket_key((1, 1), True)
                    item_info = ItemInfo(audio_path, caption, (1, 1), bucket_reso)
                    item_info.latent_cache_path = self._dummy_video_cache_path(audio_path)
                    item_info.audio_latent_cache_path = self.get_audio_latent_cache_path(item_info)
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)
                    item_info.audio_path = audio_path
                    data.append(item_info)
                    futures.remove(future)

        def submit_batch(flush: bool = False):
            nonlocal data
            if len(data) >= self.batch_size or (len(data) > 0 and flush):
                batch = data[0 : self.batch_size]
                if len(data) > self.batch_size:
                    data = data[self.batch_size :]
                else:
                    data = []
                return batch
            return None

        for fetch_op in self.datasource:
            future = executor.submit(fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                batch = submit_batch()
                if batch is None:
                    break
                yield (1, 1), batch

        aggregate_future(consume_all=True)
        while True:
            batch = submit_batch(flush=True)
            if batch is None:
                break
            yield (1, 1), batch
        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        audio_cache_files = glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}_audio.safetensors"))
        bucketed_item_info: dict[tuple[int, int], list[ItemInfo]] = {}

        for audio_cache_file in audio_cache_files:
            base = os.path.basename(audio_cache_file)
            suffix = f"_{self.architecture}_audio.safetensors"
            if not base.endswith(suffix):
                continue
            item_key = self._strip_dummy_resolution(base[: -len(suffix)])
            dummy_cache_file = os.path.join(self.cache_directory, f"{item_key}_0001x0001_{self.architecture}.safetensors")
            if not os.path.exists(dummy_cache_file):
                logger.warning(f"Dummy video cache file not found: {dummy_cache_file}")
                # Create a dummy latent cache on the fly to avoid dropping audio-only items.
                try:
                    dummy_dtype = str_to_dtype(getattr(self, "dummy_video_dtype", "float16"))
                except Exception:
                    dummy_dtype = torch.float16
                dummy_channels = int(getattr(self, "dummy_video_channels", 128))
                item_info_for_dummy = ItemInfo(item_key, "", (1, 1), (1, 1), latent_cache_path=dummy_cache_file)
                latent = torch.zeros((dummy_channels, 1, 1, 1), dtype=dummy_dtype)
                save_latent_cache_ltx2(item_info_for_dummy, latent)
                logger.info(
                    "Created dummy video cache: %s (channels=%s dtype=%s)",
                    dummy_cache_file,
                    dummy_channels,
                    dummy_dtype,
                )
            text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
            if not os.path.exists(text_encoder_output_cache_file):
                logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                continue

            bucket_reso = self._append_audio_bucket_key((1, 1), True)
            item_info = ItemInfo(item_key, "", (1, 1), bucket_reso, latent_cache_path=dummy_cache_file)
            item_info.text_encoder_output_cache_path = text_encoder_output_cache_file
            item_info.audio_latent_cache_path = audio_cache_file

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        self.batch_manager = BucketBatchManager(
            bucketed_item_info,
            self.batch_size,
            num_timestep_buckets=num_timestep_buckets,
            architecture=self.architecture,
        )
        self.batch_manager.show_bucket_info()

        self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

    def shuffle_buckets(self):
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100
        return len(self.batch_manager)

    def __getitem__(self, idx):
        super().__getitem__(idx)
        return self.batch_manager[idx]
