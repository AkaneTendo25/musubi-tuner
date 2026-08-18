from concurrent.futures import ThreadPoolExecutor
import glob
import os
import random
import time
from typing import Any, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

SharedEpoch = Optional["Synchronized[int]"]


import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import remove_dtype_suffix

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _validate_h3_cache_pair(latent_path: str, text_path: str) -> None:
    reference_kinds_prefix = "varlen_mmh3_reference_kinds"
    reference_contract_key = "mmh3_reference_temporal_contract"
    reference_contract_version = 1
    with safe_open(latent_path, framework="pt", device="cpu") as handle:
        latent_fingerprint = (handle.metadata() or {}).get("sample_fingerprint")
        latent_keys = set(handle.keys())
        kinds_keys = [key for key in latent_keys if key.startswith(reference_kinds_prefix)]
        has_video_reference = any(bool((handle.get_tensor(key) == 1).any()) for key in kinds_keys)
        latent_reference_contract = (
            int(handle.get_tensor(reference_contract_key)) if reference_contract_key in latent_keys else None
        )
    with safe_open(text_path, framework="pt", device="cpu") as handle:
        text_fingerprint = (handle.metadata() or {}).get("sample_fingerprint")
        text_keys = set(handle.keys())
        text_reference_contract = int(handle.get_tensor(reference_contract_key)) if reference_contract_key in text_keys else None
    if (latent_fingerprint or text_fingerprint) and latent_fingerprint != text_fingerprint:
        raise ValueError(
            f"MiniMax H3 latent/text caches do not describe the same sample; rebuild both caches: {latent_path} and {text_path}"
        )
    if has_video_reference and (
        latent_reference_contract != reference_contract_version or text_reference_contract != reference_contract_version
    ):
        raise ValueError(
            "MiniMax H3 reference-video preprocessing changed and the latent/text caches are stale or mismatched; "
            f"rebuild both caches: {latent_path} and {text_path}"
        )


from musubi_tuner.dataset.architectures import *  # noqa: F401,F403
from musubi_tuner.dataset.architectures import (  # explicit imports for local use
    ARCHITECTURE_FLUX_2_DEV,
    ARCHITECTURE_FLUX_2_KLEIN_4B,
    ARCHITECTURE_FLUX_2_KLEIN_9B,
    ARCHITECTURE_FLUX_KONTEXT,
    ARCHITECTURE_FRAMEPACK,
    ARCHITECTURE_HIDREAM_O1,
    ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ARCHITECTURE_KANDINSKY5,
    ARCHITECTURE_MINIMAX_H3,
    ARCHITECTURE_QWEN_IMAGE_EDIT,
    ARCHITECTURE_WAN,
)
from musubi_tuner.dataset.media_utils import *  # noqa: F401,F403
from musubi_tuner.dataset.media_utils import resize_image_to_bucket  # explicit import for local use


class ItemInfo:
    def __init__(
        self,
        item_key: str,
        caption: str,
        original_size: tuple[int, int],
        bucket_size: Optional[tuple[Any]] = None,
        frame_count: Optional[int] = None,
        content: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        latent_cache_path: Optional[str] = None,
    ) -> None:
        self.item_key = item_key
        self.caption = caption
        self.original_size = original_size
        self.bucket_size = bucket_size
        self.frame_count = frame_count
        self.content = content
        self.latent_cache_path = latent_cache_path
        self.text_encoder_output_cache_path: Optional[str] = None

        # np.ndarray for video, list[np.ndarray] for image with multiple controls
        self.control_content: Optional[Union[np.ndarray, list[np.ndarray]]] = None
        self.loss_mask_content: Optional[np.ndarray] = None

        # FramePack architecture specific
        self.fp_latent_window_size: Optional[int] = None
        self.fp_1f_clean_indices: Optional[list[int]] = None  # indices of clean latents for 1f
        self.fp_1f_target_index: Optional[int] = None  # target index for 1f clean latents
        self.fp_1f_no_post: Optional[bool] = None  # whether to add zero values as clean latent post

    def __str__(self) -> str:
        return (
            f"ItemInfo(item_key={self.item_key}, caption={self.caption}, "
            + f"original_size={self.original_size}, bucket_size={self.bucket_size}, "
            + f"frame_count={self.frame_count}, latent_cache_path={self.latent_cache_path}, "
            + f"content={[c.shape for c in self.content] if isinstance(self.content, list) else (self.content.shape if self.content is not None else None)}), "
            + f"control_content={[cc.shape for cc in self.control_content] if isinstance(self.control_content, list) else (self.control_content.shape if self.control_content is not None else None)})"
        )


from musubi_tuner.dataset.cache_io import *  # noqa: F401,F403


from musubi_tuner.dataset.bucket import BucketSelector, BucketBatchManager  # noqa: F401


from musubi_tuner.dataset.datasources import (  # noqa: F401
    ContentDatasource,
    ImageDatasource,
    ImageDirectoryDatasource,
    ImageJsonlDatasource,
    VideoDatasource,
    VideoDirectoryDatasource,
    VideoJsonlDatasource,
)


# The following classes have been moved to datasources.py but are kept here
# as a comment reference. They are re-imported above for backward compatibility.
# - ContentDatasource, ImageDatasource, ImageDirectoryDatasource, ImageJsonlDatasource
# - VideoDatasource, VideoDirectoryDatasource, VideoJsonlDatasource


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Tuple[int, int] = (960, 544),
        caption_extension: Optional[str] = None,
        batch_size: int = 1,
        num_repeats: int = 1,
        enable_bucket: bool = False,
        bucket_no_upscale: bool = False,
        cache_directory: Optional[str] = None,
        debug_dataset: bool = False,
        architecture: str = "no_default",
        loss_mask_directory: Optional[str] = None,
        default_loss_mask_path: Optional[str] = None,
        loss_mask_use_alpha: bool = False,
        loss_mask_invert: bool = False,
    ):
        self.resolution = resolution
        self.caption_extension = caption_extension
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale
        self.cache_directory = cache_directory
        self.debug_dataset = debug_dataset
        self.architecture = architecture
        self.loss_mask_directory = loss_mask_directory
        self.default_loss_mask_path = default_loss_mask_path
        self.loss_mask_use_alpha = loss_mask_use_alpha
        self.loss_mask_invert = loss_mask_invert
        self.seed = None
        self.current_epoch = 0
        self.shared_epoch = None

        if not self.enable_bucket:
            self.bucket_no_upscale = False

    def get_metadata(self) -> dict:
        metadata = {
            "resolution": self.resolution,
            "caption_extension": self.caption_extension,
            "batch_size_per_device": self.batch_size,
            "num_repeats": self.num_repeats,
            "enable_bucket": bool(self.enable_bucket),
            "bucket_no_upscale": bool(self.bucket_no_upscale),
        }
        return metadata

    def get_all_latent_cache_files(self):
        return glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

    def get_all_text_encoder_output_cache_files(self):
        return glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}_te.safetensors"))

    def configure_faster_cache_check(self, cache_paths: Sequence[str], text: bool = False) -> set[str]:
        """Filter source fetchers using cache names, without opening cache or source files."""
        datasource = getattr(self, "datasource", None)
        is_full_video = isinstance(datasource, VideoDatasource) and getattr(self, "frame_extraction", None) == "full"
        if not isinstance(datasource, ImageDatasource) and not is_full_video:
            return set()
        if not datasource.is_indexable():
            return set()

        normalized_paths = {os.path.normpath(path) for path in cache_paths}
        suffix = f"_{self.architecture}{'_te' if text else ''}.safetensors"
        source_stems = {os.path.splitext(os.path.basename(datasource.get_item_key(index)))[0] for index in range(len(datasource))}
        cached_source_stems: set[str] = set()
        matched_paths: set[str] = set()
        for path in normalized_paths:
            cache_name = os.path.basename(path)
            if not cache_name.endswith(suffix):
                continue
            cache_stem = cache_name[: -len(suffix)]
            # Dimensions and frame ranges are underscore-delimited suffixes.
            # Prefer the longest source stem when source names contain underscores.
            parts = cache_stem.split("_")
            for end in range(len(parts), 0, -1):
                candidate = "_".join(parts[:end])
                if candidate in source_stems:
                    cached_source_stems.add(candidate)
                    matched_paths.add(path)
                    break

        def has_cache_for_item(item_key: str) -> bool:
            item_stem = os.path.splitext(os.path.basename(item_key))[0]
            return item_stem in cached_source_stems

        datasource.set_item_filter(lambda item_key: not has_cache_for_item(item_key))
        return matched_paths

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        """
        Returns the cache path for the latent tensor.

        item_info: ItemInfo object

        Returns:
            str: cache path

        cache_path is based on the item_key and the resolution.
        """
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_{w:04d}x{h:04d}_{self.architecture}.safetensors")

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_{self.architecture}_te.safetensors")

    def retrieve_latent_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        pass

    def set_seed(self, seed: int, shared_epoch: SharedEpoch):
        self.seed = seed
        self.shared_epoch = shared_epoch

    def set_current_epoch(self, epoch):
        assert self.shared_epoch is not None, "shared_epoch is None"
        assert self.shared_epoch.value == epoch, "shared_epoch does not match"

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def shuffle_buckets(self):
        raise NotImplementedError

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        assert self.shared_epoch is not None, "shared_epoch is None"
        epoch = self.shared_epoch.value
        if epoch > self.current_epoch:
            logger.info(f"epoch is incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
            num_epochs = epoch - self.current_epoch
            for _ in range(num_epochs):
                self.current_epoch += 1
                self.shuffle_buckets()
        elif epoch < self.current_epoch:
            logger.warning(f"epoch is not incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
            self.current_epoch = epoch

    def _default_retrieve_text_encoder_output_cache_batches(self, datasource: ContentDatasource, batch_size: int, num_workers: int):
        datasource.set_caption_only(True)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        data: list[ItemInfo] = []
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    item_key, caption = future.result()
                    item_info = ItemInfo(item_key, caption, (0, 0), (0, 0))
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)
                    data.append(item_info)

                    futures.remove(future)

        def submit_batch(flush: bool = False):
            nonlocal data
            if len(data) >= batch_size or (len(data) > 0 and flush):
                batch = data[0:batch_size]
                if len(data) > batch_size:
                    data = data[batch_size:]
                else:
                    data = []
                return batch
            return None

        for fetch_op in datasource:
            future = executor.submit(fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                batch = submit_batch()
                if batch is None:
                    break
                yield batch

        aggregate_future(consume_all=True)
        while True:
            batch = submit_batch(flush=True)
            if batch is None:
                break
            yield batch

        executor.shutdown()


class ImageDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        image_directory: Optional[str] = None,
        image_jsonl_file: Optional[str] = None,
        control_directory: Optional[str] = None,
        cache_directory: Optional[str] = None,
        multiple_target: bool = False,
        h3_image_frame_count: Optional[int] = None,
        fp_latent_window_size: Optional[int] = 9,
        fp_1f_clean_indices: Optional[list[int]] = None,
        fp_1f_target_index: Optional[int] = None,
        fp_1f_no_post: Optional[bool] = False,
        no_resize_control: Optional[bool] = False,
        control_resolution: Optional[Tuple[int, int]] = None,
        debug_dataset: bool = False,
        architecture: str = "no_default",
        loss_mask_directory: Optional[str] = None,
        default_loss_mask_path: Optional[str] = None,
        loss_mask_use_alpha: bool = False,
        loss_mask_invert: bool = False,
    ):
        super(ImageDataset, self).__init__(
            resolution,
            caption_extension,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            cache_directory,
            debug_dataset,
            architecture,
            loss_mask_directory,
            default_loss_mask_path,
            loss_mask_use_alpha,
            loss_mask_invert,
        )
        self.image_directory = image_directory
        self.image_jsonl_file = image_jsonl_file
        self.control_directory = control_directory
        self.multiple_target = multiple_target
        self.h3_image_frame_count = h3_image_frame_count
        self.fp_latent_window_size = fp_latent_window_size
        self.fp_1f_clean_indices = fp_1f_clean_indices
        self.fp_1f_target_index = fp_1f_target_index
        self.fp_1f_no_post = fp_1f_no_post
        self.no_resize_control = no_resize_control
        self.control_resolution = control_resolution

        control_count_per_image: Optional[int] = 1
        if self.architecture == ARCHITECTURE_FRAMEPACK or self.architecture == ARCHITECTURE_WAN:
            if fp_1f_clean_indices is not None:
                control_count_per_image = len(fp_1f_clean_indices)
            else:
                control_count_per_image = 1
        elif self.architecture == ARCHITECTURE_FLUX_KONTEXT:
            control_count_per_image = 1
        elif (
            self.architecture == ARCHITECTURE_FLUX_2_DEV
            or self.architecture == ARCHITECTURE_FLUX_2_KLEIN_4B
            or self.architecture == ARCHITECTURE_FLUX_2_KLEIN_9B
        ):
            control_count_per_image = None  # can be multiple control images
        elif self.architecture == ARCHITECTURE_QWEN_IMAGE_EDIT:
            control_count_per_image = None  # can be multiple control images
        elif self.architecture == ARCHITECTURE_HIDREAM_O1:
            control_count_per_image = None  # can be multiple control/reference images
        elif self.architecture == ARCHITECTURE_MINIMAX_H3:
            control_count_per_image = None

        mask_kwargs = (
            {
                "loss_mask_directory": loss_mask_directory,
                "default_loss_mask_path": default_loss_mask_path,
                "loss_mask_use_alpha": loss_mask_use_alpha,
                "loss_mask_invert": loss_mask_invert,
            }
            if loss_mask_directory or default_loss_mask_path or loss_mask_use_alpha or loss_mask_invert
            else {}
        )
        if image_directory is not None:
            self.datasource = ImageDirectoryDatasource(
                image_directory,
                caption_extension,
                control_directory,
                control_count_per_image,
                multiple_target,
                **mask_kwargs,
                allow_indexed_caption_alias=(
                    self.architecture == ARCHITECTURE_MINIMAX_H3 and self.h3_image_frame_count is not None
                ),
            )
        elif image_jsonl_file is not None:
            self.datasource = ImageJsonlDatasource(
                image_jsonl_file,
                control_count_per_image,
                multiple_target,
                **mask_kwargs,
                normalize_indexed_paths=(self.architecture == ARCHITECTURE_MINIMAX_H3 and self.h3_image_frame_count is not None),
            )
        else:
            raise ValueError("image_directory or image_jsonl_file must be specified")

        if self.cache_directory is None:
            self.cache_directory = self.image_directory

        self.batch_manager = None
        self.num_train_items = 0
        self.has_control = self.datasource.has_control

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.image_directory is not None:
            metadata["image_directory"] = os.path.basename(self.image_directory)
        if self.image_jsonl_file is not None:
            metadata["image_jsonl_file"] = os.path.basename(self.image_jsonl_file)
        if self.control_directory is not None:
            metadata["control_directory"] = os.path.basename(self.control_directory)
        metadata["has_control"] = self.has_control
        return metadata

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        path = super().get_latent_cache_path(item_info)
        if self.architecture == ARCHITECTURE_MINIMAX_H3 and self.h3_image_frame_count is not None:
            cache_path = os.path.basename(path)
            suffix = f"_{self.architecture}.safetensors"
            prefix = cache_path[: -len(suffix)]
            base, size = prefix.rsplit("_", 1)
            path = os.path.join(os.path.dirname(path), f"{base}_00000-{self.h3_image_frame_count:03d}_{size}{suffix}")
        return path

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        path = super().get_text_encoder_output_cache_path(item_info)
        if self.architecture == ARCHITECTURE_MINIMAX_H3 and self.h3_image_frame_count is not None:
            marker = f"_{self.architecture}_te.safetensors"
            path = path.replace(marker, f"_00000-{self.h3_image_frame_count:03d}{marker}")
        return path

    def get_total_image_count(self):
        return len(self.datasource) if self.datasource.is_indexable() else None

    def retrieve_latent_cache_batches(self, num_workers: int):
        bucket_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale, self.architecture)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        batches: dict[tuple[int, int], list[ItemInfo]] = {}  # (width, height) -> [ItemInfo]
        futures = []

        # aggregate futures and sort by bucket resolution
        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    original_size, item_key, images, caption, controls, loss_mask = future.result()
                    image = images[0]  # use the first image as the main content
                    bucket_height, bucket_width = image.shape[:2]
                    bucket_reso = (bucket_width, bucket_height)

                    item_info = ItemInfo(
                        item_key, caption, original_size, bucket_reso, content=image if len(images) == 1 else images
                    )
                    item_info.latent_cache_path = self.get_latent_cache_path(item_info)
                    item_info.loss_mask_content = loss_mask

                    # for VLM, which require image in addition to text, like Qwen-Image-Edit
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)

                    item_info.fp_latent_window_size = self.fp_latent_window_size
                    item_info.fp_1f_clean_indices = self.fp_1f_clean_indices
                    item_info.fp_1f_target_index = self.fp_1f_target_index
                    item_info.fp_1f_no_post = self.fp_1f_no_post

                    if self.architecture == ARCHITECTURE_FRAMEPACK or self.architecture == ARCHITECTURE_WAN:
                        # we need to split the bucket with latent window size and optional 1f clean indices, zero post
                        bucket_reso = list(bucket_reso) + [self.fp_latent_window_size]
                        if self.fp_1f_clean_indices is not None:
                            bucket_reso.append(len(self.fp_1f_clean_indices))
                            bucket_reso.append(self.fp_1f_no_post)
                        bucket_reso = tuple(bucket_reso)

                    if controls is not None:
                        item_info.control_content = controls
                        # Add every control size to bucket_reso so that different control resolutions AND a
                        # different number of control images go to different batches. Run this whenever control
                        # data is present (not only for no_resize_control / control_resolution): otherwise items
                        # with a different control count share a batch and the collator stacks a ragged batch.
                        # controls is already in index order, so the appended shapes stay index-aligned.
                        bucket_reso = list(bucket_reso)
                        for control in controls:
                            bucket_reso = bucket_reso + list(control.shape[0:2])
                        bucket_reso = tuple(bucket_reso)

                    if bucket_reso not in batches:
                        batches[bucket_reso] = []
                    batches[bucket_reso].append(item_info)

                    futures.remove(future)

        # submit batch if some bucket has enough items
        def submit_batch(flush: bool = False):
            for key in batches:
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][0 : self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size :]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for fetch_op in self.datasource:
            # fetch and resize image in a separate thread
            def fetch_and_resize(
                op: callable,
            ) -> tuple[tuple[int, int], str, list[np.ndarray], str, Optional[list[np.ndarray]], Optional[np.ndarray]]:
                result = op()
                if len(result) == 4:
                    image_key, images, caption, controls = result
                    loss_mask = None
                else:
                    image_key, images, caption, controls, loss_mask = result
                images: list[Image.Image]
                image: Image.Image = images[0]  # use the first image as the main content
                image_size = image.size

                bucket_reso = bucket_selector.get_bucket_resolution(image_size)
                images = [resize_image_to_bucket(img, bucket_reso) for img in images]  # list of np.ndarray
                resized_loss_mask = resize_image_to_bucket(loss_mask, bucket_reso) if loss_mask is not None else None

                resized_controls = None
                if controls is not None:
                    resized_controls = []
                    if self.no_resize_control:
                        for control in controls:
                            # divisible by bucket reso steps
                            width, height = control.size

                            if self.control_resolution is not None:
                                # use control resolution as maximum
                                max_width, max_height = self.control_resolution
                                if width * height > max_width * max_height:
                                    width, height = BucketSelector.calculate_bucket_resolution(
                                        control.size, self.control_resolution, architecture=self.architecture
                                    )
                            else:
                                width = width - (width % bucket_selector.reso_steps)
                                height = height - (height % bucket_selector.reso_steps)

                            resized_control = resize_image_to_bucket(control, (width, height))  # returns np.ndarray
                            resized_controls.append(resized_control)
                    elif self.control_resolution is not None:
                        for control in controls:
                            control_bucket_reso = BucketSelector.calculate_bucket_resolution(
                                control.size, self.control_resolution, architecture=self.architecture
                            )
                            resized_control = resize_image_to_bucket(control, control_bucket_reso)
                            resized_controls.append(resized_control)
                    else:
                        for control in controls:
                            resized_control = resize_image_to_bucket(control, bucket_reso)
                            resized_controls.append(resized_control)

                return image_size, image_key, images, caption, resized_controls, resized_loss_mask

            future = executor.submit(fetch_and_resize, fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        bucket_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale, self.architecture)

        # glob cache files
        latent_cache_files = glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

        # assign cache files to item info
        # (width, height) -> [ItemInfo] or (width, height, other conds...) -> [ItemInfo]
        bucketed_item_info: dict[Union[tuple[int, int], Any], list[ItemInfo]] = {}
        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")

            image_size = tokens[-2]  # 0000x0000
            image_width, image_height = map(int, image_size.split("x"))
            image_size = (image_width, image_height)

            item_key = "_".join(tokens[:-2])
            text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
            if not os.path.exists(text_encoder_output_cache_file):
                logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                continue
            if self.architecture == ARCHITECTURE_MINIMAX_H3:
                _validate_h3_cache_pair(cache_file, text_encoder_output_cache_file)

            bucket_reso = bucket_selector.get_bucket_resolution(image_size)

            if self.architecture == ARCHITECTURE_FRAMEPACK or self.architecture == ARCHITECTURE_WAN:
                # we need to split the bucket with latent window size and optional 1f clean indices, zero post
                bucket_reso = list(bucket_reso) + [self.fp_latent_window_size]
                if self.fp_1f_clean_indices is not None:
                    bucket_reso.append(len(self.fp_1f_clean_indices))
                    bucket_reso.append(self.fp_1f_no_post)
                bucket_reso = tuple(bucket_reso)
            # Split the bucket by control latents so that every item in a batch has the same number of
            # control images AND matching per-control shapes. The collator stacks latents_control_{i}
            # across the batch (see BucketBatchManager.__getitem__), so a count or shape mismatch produces
            # a ragged stack (crash) or silently misaligns controls between samples. This must run whenever
            # control data is present, not only for no_resize_control / control_resolution: even controls
            # resized to the bucket resolution share that resolution but can still differ in *count*.
            # find_keys returns the keys sorted, so the per-control order (and the index<->shape pairing,
            # which remove_dtype_suffix keeps in each element) is deterministic across cache files.
            control_keys = safetensors_utils.find_keys(cache_file, starts_with="latents_control_")
            if control_keys:
                # key: latents_control_{i}_FxHxW_dtype -> "latents_control_{i}_FxHxW" (index + shape)
                control_shapes = [remove_dtype_suffix(key) for key in control_keys]
                bucket_reso = tuple(list(bucket_reso) + control_shapes)

            item_info = ItemInfo(item_key, "", image_size, bucket_reso, latent_cache_path=cache_file)
            item_info.text_encoder_output_cache_path = text_encoder_output_cache_file

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        # prepare batch manager
        self.batch_manager = BucketBatchManager(bucketed_item_info, self.batch_size, num_timestep_buckets=num_timestep_buckets)
        self.batch_manager.show_bucket_info()

        self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        super().__getitem__(idx)
        return self.batch_manager[idx]


def _full_extraction_frame_count(frame_count: int, max_frames: int, base: int, stride: int) -> int | None:
    bounded = min(frame_count, max_frames)
    if bounded < base:
        return None
    return (bounded - base) // stride * stride + base


class VideoDataset(BaseDataset):
    TARGET_FPS_HUNYUAN = 24.0
    TARGET_FPS_WAN = 16.0
    TARGET_FPS_FRAMEPACK = 30.0
    TARGET_FPS_FLUX_KONTEXT = 1.0  # VideoDataset is not used for Flux Kontext, but this is a placeholder
    TARGET_FPS_HUNYUAN_VIDEO_1_5 = 24.0
    TARGET_FPS_MINIMAX_H3 = 24.0

    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        frame_extraction: Optional[str] = "head",
        frame_stride: Optional[int] = 1,
        frame_sample: Optional[int] = 1,
        target_frames: Optional[list[int]] = None,
        max_frames: Optional[int] = None,
        source_fps: Optional[float] = None,
        video_directory: Optional[str] = None,
        video_jsonl_file: Optional[str] = None,
        control_directory: Optional[str] = None,
        cache_directory: Optional[str] = None,
        fp_latent_window_size: Optional[int] = 9,
        debug_dataset: bool = False,
        architecture: str = "no_default",
        loss_mask_directory: Optional[str] = None,
        default_loss_mask_path: Optional[str] = None,
        loss_mask_use_alpha: bool = False,
        loss_mask_invert: bool = False,
    ):
        super(VideoDataset, self).__init__(
            resolution,
            caption_extension,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            cache_directory,
            debug_dataset,
            architecture,
            loss_mask_directory,
            default_loss_mask_path,
            loss_mask_use_alpha,
            loss_mask_invert,
        )
        self.video_directory = video_directory
        self.video_jsonl_file = video_jsonl_file
        self.control_directory = control_directory
        self.frame_extraction = frame_extraction
        self.frame_stride = frame_stride
        self.frame_sample = frame_sample
        self.max_frames = max_frames
        self.source_fps = source_fps
        self.fp_latent_window_size = fp_latent_window_size

        self.vae_frame_stride = 4
        self.vae_frame_base = 1
        if self.architecture == ARCHITECTURE_HUNYUAN_VIDEO:
            self.target_fps = VideoDataset.TARGET_FPS_HUNYUAN
        elif self.architecture == ARCHITECTURE_WAN:
            self.target_fps = VideoDataset.TARGET_FPS_WAN
        elif self.architecture == ARCHITECTURE_FRAMEPACK:
            self.target_fps = VideoDataset.TARGET_FPS_FRAMEPACK
        elif self.architecture == ARCHITECTURE_FLUX_KONTEXT:
            self.target_fps = VideoDataset.TARGET_FPS_FLUX_KONTEXT
        elif self.architecture == ARCHITECTURE_KANDINSKY5:
            self.target_fps = VideoDataset.TARGET_FPS_HUNYUAN
        elif self.architecture == ARCHITECTURE_HUNYUAN_VIDEO_1_5:
            self.target_fps = VideoDataset.TARGET_FPS_HUNYUAN_VIDEO_1_5
        elif self.architecture == ARCHITECTURE_MINIMAX_H3:
            self.target_fps = VideoDataset.TARGET_FPS_MINIMAX_H3
            self.vae_frame_stride = 17
            self.vae_frame_base = 5
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Full extraction derives a valid length from each source video. Its
        # configured/default target_frames value is unused; rounding H3's
        # generic default of 1 with base=5 previously produced the misleading
        # sentinel -12 in logs.
        if self.frame_extraction == "full":
            target_frames = None
        elif target_frames is not None:
            target_frames = list(set(target_frames))
            target_frames.sort()

            rounded_target_frames = [
                (f - self.vae_frame_base) // self.vae_frame_stride * self.vae_frame_stride + self.vae_frame_base
                for f in target_frames
            ]
            rounded_target_frames = list(set(rounded_target_frames))
            rounded_target_frames.sort()

            # if value is changed, warn
            if target_frames != rounded_target_frames:
                logger.warning(f"target_frames are rounded to {rounded_target_frames}")

            target_frames = tuple(rounded_target_frames)

        self.target_frames = target_frames

        mask_kwargs = (
            {
                "loss_mask_directory": loss_mask_directory,
                "default_loss_mask_path": default_loss_mask_path,
                "loss_mask_use_alpha": loss_mask_use_alpha,
                "loss_mask_invert": loss_mask_invert,
            }
            if loss_mask_directory or default_loss_mask_path or loss_mask_use_alpha or loss_mask_invert
            else {}
        )
        if video_directory is not None:
            self.datasource = VideoDirectoryDatasource(video_directory, caption_extension, control_directory, **mask_kwargs)
        elif video_jsonl_file is not None:
            self.datasource = VideoJsonlDatasource(video_jsonl_file, **mask_kwargs)

        if self.frame_extraction == "uniform" and self.frame_sample == 1:
            self.frame_extraction = "head"
            logger.warning("frame_sample is set to 1 for frame_extraction=uniform. frame_extraction is changed to head.")
        if self.frame_extraction == "head":
            # head extraction. we can limit the number of frames to be extracted
            self.datasource.set_start_and_end_frame(0, max(self.target_frames))

        if self.cache_directory is None:
            self.cache_directory = self.video_directory

        self.batch_manager = None
        self.num_train_items = 0
        self.has_control = self.datasource.has_control

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.video_directory is not None:
            metadata["video_directory"] = os.path.basename(self.video_directory)
        if self.video_jsonl_file is not None:
            metadata["video_jsonl_file"] = os.path.basename(self.video_jsonl_file)
        if self.control_directory is not None:
            metadata["control_directory"] = os.path.basename(self.control_directory)
        metadata["frame_extraction"] = self.frame_extraction
        metadata["frame_stride"] = self.frame_stride
        metadata["frame_sample"] = self.frame_sample
        metadata["target_frames"] = self.target_frames
        metadata["max_frames"] = self.max_frames
        metadata["source_fps"] = self.source_fps
        metadata["has_control"] = self.has_control
        return metadata

    def retrieve_latent_cache_batches(self, num_workers: int):
        buckset_selector = BucketSelector(self.resolution, architecture=self.architecture)
        self.datasource.set_bucket_selector(buckset_selector)
        if self.source_fps is not None:
            self.datasource.set_source_and_target_fps(self.source_fps, self.target_fps)
        else:
            self.datasource.set_source_and_target_fps(None, None)  # no conversion

        executor = ThreadPoolExecutor(max_workers=num_workers)

        # key: (width, height, frame_count) and optional latent_window_size, value: [ItemInfo]
        batches: dict[tuple[Any], list[ItemInfo]] = {}
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    original_frame_size, video_key, video, caption, control, loss_mask = future.result()

                    frame_count = len(video)
                    video = np.stack(video, axis=0)
                    height, width = video.shape[1:3]
                    bucket_reso = (width, height)  # already resized

                    # process control images if available
                    control_video = None
                    if control is not None:
                        # set frame count to the same as video
                        if len(control) > frame_count:
                            control = control[:frame_count]
                        elif len(control) < frame_count:
                            # if control is shorter than video, repeat the last frame
                            last_frame = control[-1]
                            control.extend([last_frame] * (frame_count - len(control)))
                        control_video = np.stack(control, axis=0)

                    crop_pos_and_frames = []
                    if self.frame_extraction == "head":
                        for target_frame in self.target_frames:
                            if frame_count >= target_frame:
                                crop_pos_and_frames.append((0, target_frame))
                    elif self.frame_extraction == "chunk":
                        # split by target_frames
                        for target_frame in self.target_frames:
                            for i in range(0, frame_count, target_frame):
                                if i + target_frame <= frame_count:
                                    crop_pos_and_frames.append((i, target_frame))
                    elif self.frame_extraction == "slide":
                        # slide window
                        for target_frame in self.target_frames:
                            if frame_count >= target_frame:
                                for i in range(0, frame_count - target_frame + 1, self.frame_stride):
                                    crop_pos_and_frames.append((i, target_frame))
                    elif self.frame_extraction == "uniform":
                        # select N frames uniformly
                        for target_frame in self.target_frames:
                            if frame_count >= target_frame:
                                frame_indices = np.linspace(0, frame_count - target_frame, self.frame_sample, dtype=int)
                                for i in frame_indices:
                                    crop_pos_and_frames.append((i, target_frame))
                    elif self.frame_extraction == "full":
                        # select all frames
                        target_frame = _full_extraction_frame_count(
                            frame_count,
                            self.max_frames,
                            self.vae_frame_base,
                            self.vae_frame_stride,
                        )
                        if target_frame is not None:
                            crop_pos_and_frames.append((0, target_frame))
                        else:
                            logger.warning(
                                f"video {video_key} has {frame_count} frame(s), fewer than the minimum "
                                f"{self.vae_frame_base} required by {self.architecture}; skipping"
                            )
                    else:
                        raise ValueError(f"frame_extraction {self.frame_extraction} is not supported")

                    for crop_pos, target_frame in crop_pos_and_frames:
                        cropped_video = video[crop_pos : crop_pos + target_frame]
                        body, ext = os.path.splitext(video_key)
                        item_key = f"{body}_{crop_pos:05d}-{target_frame:03d}{ext}"
                        batch_key = (*bucket_reso, target_frame)  # bucket_reso with frame_count

                        if self.architecture == ARCHITECTURE_FRAMEPACK:
                            # add latent window size to bucket resolution
                            batch_key = (*batch_key, self.fp_latent_window_size)

                        # crop control video if available
                        cropped_control = None
                        if control_video is not None:
                            cropped_control = control_video[crop_pos : crop_pos + target_frame]

                        item_info = ItemInfo(
                            item_key, caption, original_frame_size, batch_key, frame_count=target_frame, content=cropped_video
                        )
                        item_info.latent_cache_path = self.get_latent_cache_path(item_info)
                        item_info.control_content = cropped_control  # None is allowed
                        item_info.loss_mask_content = (
                            loss_mask[crop_pos : crop_pos + target_frame] if loss_mask is not None else None
                        )
                        item_info.fp_latent_window_size = self.fp_latent_window_size

                        batch = batches.get(batch_key, [])
                        batch.append(item_info)
                        batches[batch_key] = batch

                    futures.remove(future)

        def submit_batch(flush: bool = False):
            for key in batches:
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][0 : self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size :]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for operator in self.datasource:

            def fetch_and_resize(
                op: callable,
            ) -> tuple[
                tuple[int, int],
                str,
                list[np.ndarray],
                str,
                Optional[list[np.ndarray]],
                Optional[list[np.ndarray]],
            ]:
                result = op()

                if len(result) == 3:  # for backward compatibility TODO remove this in the future
                    video_key, video, caption = result
                    control = None
                    loss_mask = None
                elif len(result) == 4:
                    video_key, video, caption, control = result
                    loss_mask = None
                else:
                    video_key, video, caption, control, loss_mask = result

                video: list[np.ndarray]
                frame_size = (video[0].shape[1], video[0].shape[0])

                # resize if necessary
                bucket_reso = buckset_selector.get_bucket_resolution(frame_size)
                video = [resize_image_to_bucket(frame, bucket_reso) for frame in video]

                # resize control if necessary
                if control is not None:
                    control = [resize_image_to_bucket(frame, bucket_reso) for frame in control]

                if loss_mask is not None:
                    loss_mask = [resize_image_to_bucket(frame, bucket_reso) for frame in loss_mask]

                return frame_size, video_key, video, caption, control, loss_mask

            future = executor.submit(fetch_and_resize, operator)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        bucket_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale, self.architecture)

        # glob cache files
        latent_cache_files = glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

        # assign cache files to item info
        bucketed_item_info: dict[tuple[int, int, int], list[ItemInfo]] = {}  # (width, height, frame_count) -> [ItemInfo]
        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")

            image_size = tokens[-2]  # 0000x0000
            image_width, image_height = map(int, image_size.split("x"))
            image_size = (image_width, image_height)

            frame_pos, frame_count = tokens[-3].split("-")[:2]  # "00000-000", or optional section index "00000-000-00"
            frame_pos, frame_count = int(frame_pos), int(frame_count)

            item_key = "_".join(tokens[:-3])
            text_item_key = f"{item_key}_{tokens[-3]}" if self.architecture == ARCHITECTURE_MINIMAX_H3 else item_key
            text_encoder_output_cache_file = os.path.join(
                self.cache_directory,
                f"{text_item_key}_{self.architecture}_te.safetensors",
            )
            if not os.path.exists(text_encoder_output_cache_file):
                logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                continue
            if self.architecture == ARCHITECTURE_MINIMAX_H3:
                _validate_h3_cache_pair(cache_file, text_encoder_output_cache_file)

            bucket_reso = bucket_selector.get_bucket_resolution(image_size)
            bucket_reso = (*bucket_reso, frame_count)
            item_info = ItemInfo(item_key, "", image_size, bucket_reso, frame_count=frame_count, latent_cache_path=cache_file)
            item_info.text_encoder_output_cache_path = text_encoder_output_cache_file

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        # prepare batch manager
        self.batch_manager = BucketBatchManager(bucketed_item_info, self.batch_size, num_timestep_buckets=num_timestep_buckets)
        self.batch_manager.show_bucket_info()

        self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        super().__getitem__(idx)
        return self.batch_manager[idx]


class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[ImageDataset, VideoDataset]]):
        super().__init__(datasets)
        self.datasets: list[Union[ImageDataset, VideoDataset]] = datasets
        self.num_train_items = 0
        for dataset in self.datasets:
            self.num_train_items += dataset.num_train_items

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)
