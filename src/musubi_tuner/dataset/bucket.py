from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from safetensors.torch import load_file

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_FRAMEPACK,
    ARCHITECTURE_FLUX_2_DEV,
    ARCHITECTURE_FLUX_2_KLEIN_4B,
    ARCHITECTURE_FLUX_2_KLEIN_9B,
    ARCHITECTURE_FLUX_KONTEXT,
    ARCHITECTURE_HIDREAM_O1,
    ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ARCHITECTURE_KANDINSKY5,
    ARCHITECTURE_LTX2,
    ARCHITECTURE_LTX2_FULL,
    ARCHITECTURE_QWEN_IMAGE,
    ARCHITECTURE_QWEN_IMAGE_EDIT,
    ARCHITECTURE_QWEN_IMAGE_LAYERED,
    ARCHITECTURE_WAN,
    ARCHITECTURE_Z_IMAGE,
)
from musubi_tuner.dataset.media_utils import divisible_by
from musubi_tuner.utils.model_utils import remove_dtype_suffix

if TYPE_CHECKING:
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

import logging

logger = logging.getLogger(__name__)


class BucketSelector:
    RESOLUTION_STEPS_HUNYUAN = 16
    RESOLUTION_STEPS_WAN = 16
    RESOLUTION_STEPS_LTX2 = 32
    RESOLUTION_STEPS_FRAMEPACK = 16
    RESOLUTION_STEPS_FLUX_KONTEXT = 16
    RESOLUTION_STEPS_FLUX_2 = 16
    RESOLUTION_STEPS_QWEN_IMAGE = 16
    RESOLUTION_STEPS_QWEN_IMAGE_EDIT = 16
    RESOLUTION_STEPS_KANDINSKY5 = 16
    RESOLUTION_STEPS_HUNYUAN_VIDEO_1_5 = 16
    RESOLUTION_STEPS_Z_IMAGE = 16
    RESOLUTION_STEPS_HIDREAM_O1 = 32

    ARCHITECTURE_STEPS_MAP = {
        ARCHITECTURE_HUNYUAN_VIDEO: RESOLUTION_STEPS_HUNYUAN,
        ARCHITECTURE_WAN: RESOLUTION_STEPS_WAN,
        ARCHITECTURE_LTX2: RESOLUTION_STEPS_LTX2,
        ARCHITECTURE_FRAMEPACK: RESOLUTION_STEPS_FRAMEPACK,
        ARCHITECTURE_FLUX_KONTEXT: RESOLUTION_STEPS_FLUX_KONTEXT,
        ARCHITECTURE_FLUX_2_DEV: RESOLUTION_STEPS_FLUX_2,
        ARCHITECTURE_FLUX_2_KLEIN_4B: RESOLUTION_STEPS_FLUX_2,
        ARCHITECTURE_FLUX_2_KLEIN_9B: RESOLUTION_STEPS_FLUX_2,
        ARCHITECTURE_QWEN_IMAGE: RESOLUTION_STEPS_QWEN_IMAGE,
        ARCHITECTURE_QWEN_IMAGE_EDIT: RESOLUTION_STEPS_QWEN_IMAGE_EDIT,
        ARCHITECTURE_QWEN_IMAGE_LAYERED: RESOLUTION_STEPS_QWEN_IMAGE,  # use same steps as Qwen-Image
        ARCHITECTURE_KANDINSKY5: RESOLUTION_STEPS_KANDINSKY5,
        ARCHITECTURE_HUNYUAN_VIDEO_1_5: RESOLUTION_STEPS_HUNYUAN_VIDEO_1_5,
        ARCHITECTURE_Z_IMAGE: RESOLUTION_STEPS_Z_IMAGE,
        ARCHITECTURE_HIDREAM_O1: RESOLUTION_STEPS_HIDREAM_O1,
    }

    @classmethod
    def resolve_resolution_steps(cls, architecture: str, reference_downscale: int = 1) -> int:
        if architecture not in cls.ARCHITECTURE_STEPS_MAP:
            raise ValueError(f"Invalid architecture: {architecture}")

        reso_steps = cls.ARCHITECTURE_STEPS_MAP[architecture]
        reference_downscale = max(1, int(reference_downscale or 1))
        if architecture == ARCHITECTURE_LTX2 and reference_downscale > 1:
            # LTX2 reference latents are quantized to /32 after spatial downscale.
            # Make target buckets divisible by 32 * downscale to avoid lossy flooring.
            reso_steps *= reference_downscale
        return reso_steps

    def __init__(
        self,
        resolution: Tuple[int, int],
        enable_bucket: bool = True,
        no_upscale: bool = False,
        architecture: str = "no_default",
        reference_downscale: int = 1,
    ):
        self.resolution = resolution
        self.bucket_area = resolution[0] * resolution[1]
        self.architecture = architecture

        self.reso_steps = self.resolve_resolution_steps(architecture, reference_downscale)

        if not enable_bucket:
            # only define one bucket
            if resolution[0] % self.reso_steps != 0 or resolution[1] % self.reso_steps != 0:
                raise ValueError(f"resolution must be divisible by {self.reso_steps} for architecture={architecture}: {resolution}")
            self.bucket_resolutions = [resolution]
            self.no_upscale = False
        else:
            # prepare bucket resolution
            self.no_upscale = no_upscale
            sqrt_size = int(math.sqrt(self.bucket_area))
            min_size = divisible_by(sqrt_size // 2, self.reso_steps)
            self.bucket_resolutions = []
            for w in range(min_size, sqrt_size + self.reso_steps, self.reso_steps):
                h = divisible_by(self.bucket_area // w, self.reso_steps)
                self.bucket_resolutions.append((w, h))
                self.bucket_resolutions.append((h, w))

            self.bucket_resolutions = list(set(self.bucket_resolutions))
            self.bucket_resolutions.sort()

        # calculate aspect ratio to find the nearest resolution
        self.aspect_ratios = np.array([w / h for w, h in self.bucket_resolutions])

    def get_bucket_resolution(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """
        return the bucket resolution for the given image size, (width, height)
        """
        area = image_size[0] * image_size[1]
        if self.no_upscale and area <= self.bucket_area:
            w, h = image_size
            w = divisible_by(w, self.reso_steps)
            h = divisible_by(h, self.reso_steps)
            return w, h

        aspect_ratio = image_size[0] / image_size[1]
        ar_errors = self.aspect_ratios - aspect_ratio
        bucket_id = np.abs(ar_errors).argmin()
        return self.bucket_resolutions[bucket_id]

    @classmethod
    def calculate_bucket_resolution(
        cls,
        image_size: tuple[int, int],
        resolution: tuple[int, int],
        reso_steps: Optional[int] = None,
        architecture: Optional[str] = None,
        reference_downscale: int = 1,
    ) -> tuple[int, int]:
        """
        Get the bucket resolution for the given image size, resolution and resolution steps.
        Return (width, height).
        """
        if reso_steps is None and architecture is None:
            raise ValueError("resolution steps or architecture must be provided")
        if reso_steps is None and architecture is not None:
            reso_steps = cls.resolve_resolution_steps(architecture, reference_downscale)

        max_area = resolution[0] * resolution[1]
        width, height = image_size
        aspect_ratio = width / height
        bucket_width = int(math.sqrt(max_area * aspect_ratio))
        bucket_height = int(math.sqrt(max_area / aspect_ratio))
        bucket_width = divisible_by(bucket_width, reso_steps)
        bucket_height = divisible_by(bucket_height, reso_steps)

        # find appropriate resolutions
        best_resolution = None
        best_aspect_ratio_diff = float("inf")
        for i in range(-2, 3):
            w = bucket_width + i * reso_steps
            h = divisible_by(max_area // w, reso_steps)
            current_aspect_ratio_diff = abs((w / h) - aspect_ratio)
            if current_aspect_ratio_diff < best_aspect_ratio_diff:
                best_aspect_ratio_diff = current_aspect_ratio_diff
                best_resolution = (w, h)

        if best_resolution is not None:
            return best_resolution

        return bucket_width, bucket_height


class BucketBatchManager:
    def __init__(
        self,
        bucketed_item_info: dict[tuple[Any], list[ItemInfo]],
        batch_size: int,
        num_timestep_buckets: Optional[int] = None,
        architecture: Optional[str] = None,
        target_fps: float = 24.0,
        audio_bucket_strategy: str = "pad",
        video_loss_weight: Optional[float] = None,
        audio_loss_weight: Optional[float] = None,
        # Latent-guide config (subset-level — same value across all batches
        # produced by this manager, since items with different guide config
        # land in separate BucketBatchManagers).
        latent_idx_guide_frame_idx: int = 0,
        latent_idx_guide_strength: float = 1.0,
        keyframe_guide_frame_idx: int = -1,
        keyframe_guide_strength: float = 1.0,
        keyframe_guide_extras: Optional[List[Dict[str, Any]]] = None,
    ):
        self.batch_size = batch_size
        self.buckets = bucketed_item_info
        self.bucket_resos = list(self.buckets.keys())
        self.bucket_resos.sort()
        self.num_timestep_buckets = num_timestep_buckets
        self.timestep_pool = None
        self.architecture = architecture
        self.target_fps = target_fps
        self.audio_bucket_strategy = audio_bucket_strategy
        self.video_loss_weight = video_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.latent_idx_guide_frame_idx = int(latent_idx_guide_frame_idx)
        self.latent_idx_guide_strength = float(latent_idx_guide_strength)
        self.keyframe_guide_frame_idx = int(keyframe_guide_frame_idx)
        self.keyframe_guide_strength = float(keyframe_guide_strength)
        self.keyframe_guide_extras: List[Dict[str, Any]] = list(keyframe_guide_extras or [])

        # indices for enumerating batches. each batch is reso + batch_idx. reso is (width, height) or (width, height, frames)
        self.bucket_batch_indices: list[tuple[tuple[Any], int]] = []
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            num_batches = math.ceil(len(bucket) / self.batch_size)
            for i in range(num_batches):
                self.bucket_batch_indices.append((bucket_reso, i))

        # do no shuffle here to avoid multiple datasets have different order
        # self.shuffle()

    def show_bucket_info(self):
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            logger.info(f"bucket: {bucket_reso}, count: {len(bucket)}")

        logger.info(f"total batches: {len(self)}")

    def shuffle(self):
        # shuffle each bucket
        for bucket in self.buckets.values():
            random.shuffle(bucket)

        # shuffle the order of batches
        random.shuffle(self.bucket_batch_indices)

        if self.num_timestep_buckets is not None and self.num_timestep_buckets > 1:
            # prepare timesteps for each timestep buckets

            # 1. Calculate total number of timesteps needed for the entire epoch
            num_batches = len(self.bucket_batch_indices)
            total_timesteps_needed = num_batches * self.batch_size

            # 2. Generate a single large pool of stratified timesteps
            all_timesteps = []
            samples_per_bucket = math.ceil(total_timesteps_needed / self.num_timestep_buckets)

            for i in range(self.num_timestep_buckets):
                min_t = i / self.num_timestep_buckets
                max_t = (i + 1) / self.num_timestep_buckets
                for _ in range(samples_per_bucket):
                    all_timesteps.append(random.uniform(min_t, max_t))

            # 3. Shuffle the entire pool thoroughly
            random.shuffle(all_timesteps)

            # Trim the excess timesteps to match the exact number needed
            all_timesteps = all_timesteps[:total_timesteps_needed]

            # 4. Create the final timestep pool by chunking the shuffled list
            self.timestep_pool = []
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                self.timestep_pool.append(all_timesteps[start_idx:end_idx])
                # print(f"timestep pool {i}: {self.timestep_pool[-1]}")

    def __len__(self):
        return len(self.bucket_batch_indices)

    def __getitem__(self, idx):
        bucket_reso, batch_idx = self.bucket_batch_indices[idx]
        bucket = self.buckets[bucket_reso]
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(bucket))
        batch_count = max(end - start, 0)

        batch_tensor_data = {}
        varlen_keys = set()

        audio_latents_per_item = []
        audio_lengths_per_item = []
        audio_loss_masks_per_item = []
        ref_audio_latents_per_item = []
        ref_audio_lengths_per_item = []
        dino_features_per_item = []
        collect_item_keys = os.getenv("LTX2_COLLECT_BATCH_ITEM_KEYS", "0") == "1" or os.getenv("LTX2_NAN_DIAG", "0") == "1"
        item_keys = []
        latent_cache_paths = []
        audio_cache_paths = []
        text_cache_paths = []
        captions: list[str] = []
        for item_info in bucket[start:end]:
            sd_latent = load_file(item_info.latent_cache_path)
            audio_latent_cache_path = getattr(item_info, "audio_latent_cache_path", None)
            if audio_latent_cache_path is not None and os.path.exists(audio_latent_cache_path):
                sd_audio = load_file(audio_latent_cache_path)
                sd_latent = {**sd_latent, **sd_audio}

            dino_cache_path = getattr(item_info, "dino_feature_cache_path", None)
            if dino_cache_path is not None and os.path.exists(dino_cache_path):
                sd_dino = load_file(dino_cache_path)
                dino_features_per_item.append(sd_dino["dino_features"])  # [T_pixel, N_patches, D_dino]
            else:
                dino_features_per_item.append(None)

            reference_latent_cache_paths = getattr(item_info, "reference_latent_cache_paths", None)
            if not reference_latent_cache_paths:
                reference_latent_cache_path = getattr(item_info, "reference_latent_cache_path", None)
                if reference_latent_cache_path is not None:
                    reference_latent_cache_paths = [reference_latent_cache_path]
            if reference_latent_cache_paths:
                for ref_index, reference_latent_cache_path in enumerate(reference_latent_cache_paths):
                    if not os.path.exists(reference_latent_cache_path):
                        raise FileNotFoundError(f"Reference latent cache file not found: {reference_latent_cache_path}")
                    sd_ref = load_file(reference_latent_cache_path)
                    sd_ref_latents = {}
                    for key, value in sd_ref.items():
                        if key.startswith("latents_"):
                            if ref_index == 0:
                                mapped_key = "ref_" + key
                            else:
                                mapped_key = key.replace("latents_", f"ref_latents_{ref_index}_", 1)
                            sd_ref_latents[mapped_key] = value
                    if not sd_ref_latents:
                        raise ValueError(f"No latent tensors found in reference cache: {reference_latent_cache_path}")
                    sd_latent = {**sd_latent, **sd_ref_latents}

            # Latent guides
            for _guide_attr, _key_prefix in (
                ("latent_idx_guide_cache_path", "latent_idx_guide_latents_"),
                ("keyframe_guide_cache_path", "keyframe_guide_latents_"),
            ):
                _guide_path = getattr(item_info, _guide_attr, None)
                if _guide_path:
                    if not os.path.exists(_guide_path):
                        raise FileNotFoundError(f"Guide latent cache file not found: {_guide_path}")
                    _sd_guide_raw = load_file(_guide_path)
                    _sd_guide = {}
                    for _k, _v in _sd_guide_raw.items():
                        if _k.startswith("latents_"):
                            _sd_guide[_key_prefix + _k[len("latents_") :]] = _v
                    if not _sd_guide:
                        raise ValueError(f"No latent tensors in guide cache: {_guide_path}")
                    sd_latent = {**sd_latent, **_sd_guide}

            # Extra keyframe guides (multi-keyframe dataset). Each extra is loaded
            # with its index in the prefix so the trainer can recover them as a
            # list. Index 0 is reserved for the primary keyframe above.
            _extra_kf_paths = getattr(item_info, "keyframe_guide_extra_cache_paths", None) or []
            for _ix, _kf_path in enumerate(_extra_kf_paths, start=1):
                if not os.path.exists(_kf_path):
                    raise FileNotFoundError(f"Extra keyframe guide cache not found: {_kf_path}")
                _sd_extra_raw = load_file(_kf_path)
                _sd_extra = {}
                _extra_prefix = f"keyframe_guide_extra_{_ix}_latents_"
                for _k, _v in _sd_extra_raw.items():
                    if _k.startswith("latents_"):
                        _sd_extra[_extra_prefix + _k[len("latents_") :]] = _v
                if not _sd_extra:
                    raise ValueError(f"No latent tensors in extra keyframe guide cache: {_kf_path}")
                sd_latent = {**sd_latent, **_sd_extra}

            reference_audio_latent_cache_paths = getattr(item_info, "reference_audio_latent_cache_paths", None)
            if not reference_audio_latent_cache_paths:
                reference_audio_latent_cache_path = getattr(item_info, "reference_audio_latent_cache_path", None)
                if reference_audio_latent_cache_path is not None:
                    reference_audio_latent_cache_paths = [reference_audio_latent_cache_path]
            if reference_audio_latent_cache_paths:
                for ref_index, reference_audio_latent_cache_path in enumerate(reference_audio_latent_cache_paths):
                    if not os.path.exists(reference_audio_latent_cache_path):
                        raise FileNotFoundError(f"Reference audio latent cache file not found: {reference_audio_latent_cache_path}")
                    sd_ref_audio_raw = load_file(reference_audio_latent_cache_path)
                    sd_ref_audio = {}
                    for key, value in sd_ref_audio_raw.items():
                        if key.startswith("audio_latents_"):
                            if ref_index == 0:
                                mapped_key = "ref_" + key
                            else:
                                mapped_key = key.replace("audio_latents_", f"ref_audio_latents_{ref_index}_", 1)
                            sd_ref_audio[mapped_key] = value
                        elif key.startswith("audio_lengths_"):
                            if ref_index == 0:
                                mapped_key = "ref_" + key
                            else:
                                mapped_key = key.replace("audio_lengths_", f"ref_audio_lengths_{ref_index}_", 1)
                            sd_ref_audio[mapped_key] = value
                    if not sd_ref_audio:
                        raise ValueError(
                            f"No audio latent tensors found in reference audio cache: {reference_audio_latent_cache_path}"
                        )
                    sd_latent = {**sd_latent, **sd_ref_audio}

            sd_te = load_file(item_info.text_encoder_output_cache_path)
            sd = {**sd_latent, **sd_te}

            item_audio_latents = None
            item_audio_lengths = None
            item_audio_loss_mask = None
            item_ref_audio_latents: dict[int, torch.Tensor] = {}
            item_ref_audio_lengths: dict[int, torch.Tensor] = {}
            for key, value in sorted(sd.items()):
                if key.startswith("audio_latents_"):
                    item_audio_latents = value
                elif key.startswith("audio_lengths_"):
                    item_audio_lengths = value
                elif key == "audio_loss_mask":
                    item_audio_loss_mask = value
                elif key.startswith("ref_audio_latents_"):
                    ref_suffix = key[len("ref_audio_latents_") :]
                    ref_index = 0
                    if "_" in ref_suffix:
                        maybe_index, _rest = ref_suffix.split("_", 1)
                        if maybe_index.isdigit():
                            ref_index = int(maybe_index)
                    item_ref_audio_latents[ref_index] = value
                elif key.startswith("ref_audio_lengths_"):
                    ref_suffix = key[len("ref_audio_lengths_") :]
                    ref_index = 0
                    if "_" in ref_suffix:
                        maybe_index, _rest = ref_suffix.split("_", 1)
                        if maybe_index.isdigit():
                            ref_index = int(maybe_index)
                    item_ref_audio_lengths[ref_index] = value
            audio_latents_per_item.append(item_audio_latents)
            audio_lengths_per_item.append(item_audio_lengths)
            audio_loss_masks_per_item.append(item_audio_loss_mask)
            ref_audio_latents_per_item.append(
                [item_ref_audio_latents[idx] for idx in sorted(item_ref_audio_latents.keys())] if item_ref_audio_latents else None
            )
            ref_audio_lengths_per_item.append(
                [item_ref_audio_lengths[idx] for idx in sorted(item_ref_audio_lengths.keys())] if item_ref_audio_lengths else None
            )

            if collect_item_keys:
                item_keys.append(item_info.item_key)
                latent_cache_paths.append(item_info.latent_cache_path)
                audio_cache_paths.append(audio_latent_cache_path)
                text_cache_paths.append(item_info.text_encoder_output_cache_path)
            captions.append(item_info.caption)

            # TODO refactor this
            for key in sd.keys():
                if (
                    key.startswith("audio_latents_")
                    or key.startswith("audio_lengths_")
                    or key == "audio_loss_mask"
                    or key.startswith("ref_audio_latents_")
                    or key.startswith("ref_audio_lengths_")
                ):
                    continue
                is_varlen_key = key.startswith("varlen_")  # varlen keys are not stacked
                content_key = key

                if is_varlen_key:
                    content_key = content_key.replace("varlen_", "")

                if content_key.endswith("_mask"):
                    pass
                else:
                    content_key = remove_dtype_suffix(content_key)
                    if (
                        content_key.startswith("latents_")
                        or content_key.startswith("audio_latents_")
                        or content_key.startswith("ref_latents_")
                        or content_key.startswith("latent_idx_guide_latents_")
                        or content_key.startswith("keyframe_guide_latents_")
                        or content_key.startswith("keyframe_guide_extra_")
                    ):
                        content_key = content_key.rsplit("_", 1)[0]  # remove FxHxW

                if content_key not in batch_tensor_data:
                    batch_tensor_data[content_key] = []
                batch_tensor_data[content_key].append(sd[key])

                if is_varlen_key:
                    varlen_keys.add(content_key)

        for key in batch_tensor_data.keys():
            if key not in varlen_keys:
                batch_tensor_data[key] = torch.stack(batch_tensor_data[key])

        if self.architecture in {ARCHITECTURE_LTX2, ARCHITECTURE_LTX2_FULL}:
            present_audio = [x for x in audio_latents_per_item if isinstance(x, torch.Tensor)]
            if present_audio:
                ref = present_audio[0]
                if not isinstance(ref, torch.Tensor) or ref.dim() != 3:
                    raise ValueError(
                        f"Expected cached audio latents to be 3D [C, T, F] before stacking, got: {getattr(ref, 'shape', None)}"
                    )

                channels = int(ref.shape[0])
                mel_bins = int(ref.shape[2])
                dtype = ref.dtype
                device = ref.device

                if self.audio_bucket_strategy == "truncate":
                    # Truncate mode: extract quantized_t from bucket_reso (last int element)
                    # and truncate all audio latents to that length — no padding needed.
                    quantized_t = None
                    for elem in reversed(bucket_reso):
                        if isinstance(elem, int):
                            quantized_t = elem
                            break
                    if quantized_t is None or quantized_t <= 0:
                        quantized_t = int(ref.shape[1])

                    truncated = []
                    truncated_masks = []
                    has_audio_loss_masks = any(isinstance(mask, torch.Tensor) for mask in audio_loss_masks_per_item)
                    for lat in audio_latents_per_item:
                        item_index = len(truncated)
                        if isinstance(lat, torch.Tensor):
                            if lat.dim() != 3:
                                raise ValueError(f"Expected audio latents to be 3D [C, T, F], got {tuple(lat.shape)}")
                            if int(lat.shape[0]) != channels or int(lat.shape[2]) != mel_bins:
                                raise ValueError(
                                    "Audio latents shape mismatch in batch: "
                                    f"expected [C={channels}, *, F={mel_bins}], got {tuple(lat.shape)}"
                                )
                            truncated.append(lat[:, :quantized_t, :].to(device=device, dtype=dtype))
                            if has_audio_loss_masks:
                                mask = audio_loss_masks_per_item[item_index]
                                if isinstance(mask, torch.Tensor):
                                    mask_out = torch.zeros((quantized_t,), device=device, dtype=torch.float32)
                                    use_mask_t = min(int(mask.shape[0]), quantized_t)
                                    if use_mask_t > 0:
                                        mask_out[:use_mask_t] = mask[:use_mask_t].to(device=device, dtype=torch.float32)
                                    truncated_masks.append(mask_out)
                                else:
                                    truncated_masks.append(torch.ones((quantized_t,), device=device, dtype=torch.float32))
                        else:
                            truncated.append(torch.zeros((channels, quantized_t, mel_bins), device=device, dtype=dtype))
                            if has_audio_loss_masks:
                                truncated_masks.append(torch.zeros((quantized_t,), device=device, dtype=torch.float32))

                    batch_tensor_data["audio_latents"] = torch.stack(truncated)
                    batch_tensor_data["audio_lengths"] = torch.full(
                        (len(truncated),), quantized_t, device=device, dtype=torch.int32
                    )
                    if has_audio_loss_masks:
                        batch_tensor_data["audio_loss_mask"] = torch.stack(truncated_masks)
                else:
                    # Pad mode (default): pad shorter clips to max_t and store actual lengths.
                    lengths = []
                    max_t = 0
                    for i, lat in enumerate(audio_latents_per_item):
                        if isinstance(lat, torch.Tensor):
                            t = int(lat.shape[1])
                            length_val = t
                            cached_len = audio_lengths_per_item[i]
                            if isinstance(cached_len, torch.Tensor) and cached_len.numel() == 1:
                                length_val = int(cached_len.view(-1)[0].item())
                            length_val = max(0, min(length_val, t))
                        else:
                            length_val = 0
                            t = 0
                        lengths.append(length_val)
                        max_t = max(max_t, t)

                    if max_t <= 0:
                        max_t = 1

                    padded = []
                    padded_masks = []
                    has_audio_loss_masks = any(isinstance(mask, torch.Tensor) for mask in audio_loss_masks_per_item)
                    for i, lat in enumerate(audio_latents_per_item):
                        if isinstance(lat, torch.Tensor):
                            if lat.dim() != 3:
                                raise ValueError(f"Expected audio latents to be 3D [C, T, F], got {tuple(lat.shape)}")
                            if int(lat.shape[0]) != channels or int(lat.shape[2]) != mel_bins:
                                raise ValueError(
                                    "Audio latents shape mismatch in batch: "
                                    f"expected [C={channels}, *, F={mel_bins}], got {tuple(lat.shape)}"
                                )

                            t = int(lat.shape[1])
                            use_t = min(t, max_t)
                            out = torch.zeros((channels, max_t, mel_bins), device=device, dtype=dtype)
                            if use_t > 0:
                                out[:, :use_t, :] = lat[:, :use_t, :].to(device=device, dtype=dtype)
                            padded.append(out)
                            lengths[i] = int(min(max(0, lengths[i]), max_t))
                            if has_audio_loss_masks:
                                mask_out = torch.zeros((max_t,), device=device, dtype=torch.float32)
                                mask = audio_loss_masks_per_item[i]
                                if isinstance(mask, torch.Tensor):
                                    use_mask_t = min(int(mask.shape[0]), max_t)
                                    if use_mask_t > 0:
                                        mask_out[:use_mask_t] = mask[:use_mask_t].to(device=device, dtype=torch.float32)
                                else:
                                    valid_t = int(min(max(0, lengths[i]), max_t))
                                    if valid_t > 0:
                                        mask_out[:valid_t] = 1.0
                                padded_masks.append(mask_out)
                        else:
                            padded.append(torch.zeros((channels, max_t, mel_bins), device=device, dtype=dtype))
                            if has_audio_loss_masks:
                                padded_masks.append(torch.zeros((max_t,), device=device, dtype=torch.float32))

                    batch_tensor_data["audio_latents"] = torch.stack(padded)
                    batch_tensor_data["audio_lengths"] = torch.tensor(lengths, device=device, dtype=torch.int32)
                    if has_audio_loss_masks:
                        batch_tensor_data["audio_loss_mask"] = torch.stack(padded_masks)

            else:
                # Skip allocating placeholder audio tensors when the batch has no audio.
                pass

            present_ref_audio = [x for x in ref_audio_latents_per_item if isinstance(x, list) and len(x) > 0]
            if present_ref_audio:
                ref = present_ref_audio[0][0]
                if not isinstance(ref, torch.Tensor) or ref.dim() != 3:
                    raise ValueError(
                        "Expected cached reference audio latents to be 3D [C, T, F] before stacking, "
                        f"got: {getattr(ref, 'shape', None)}"
                    )

                ref_channels = int(ref.shape[0])
                ref_mel_bins = int(ref.shape[2])
                ref_dtype = ref.dtype
                ref_device = ref.device
                max_ref_count = max(len(lat_list) if isinstance(lat_list, list) else 0 for lat_list in ref_audio_latents_per_item)

                if self.audio_bucket_strategy == "truncate":
                    quantized_t = None
                    for elem in reversed(bucket_reso):
                        if isinstance(elem, int):
                            quantized_t = elem
                            break
                    if quantized_t is None or quantized_t <= 0:
                        quantized_t = int(ref.shape[1])

                    truncated_ref = []
                    truncated_ref_lengths = []
                    for lat in ref_audio_latents_per_item:
                        item_refs = []
                        item_lengths = []
                        if isinstance(lat, list) and lat:
                            for ref_lat in lat:
                                if ref_lat.dim() != 3:
                                    raise ValueError(
                                        f"Expected reference audio latents to be 3D [C, T, F], got {tuple(ref_lat.shape)}"
                                    )
                                if int(ref_lat.shape[0]) != ref_channels or int(ref_lat.shape[2]) != ref_mel_bins:
                                    raise ValueError(
                                        "Reference audio latents shape mismatch in batch: "
                                        f"expected [C={ref_channels}, *, F={ref_mel_bins}], got {tuple(ref_lat.shape)}"
                                    )
                                item_refs.append(ref_lat[:, :quantized_t, :].to(device=ref_device, dtype=ref_dtype))
                                item_lengths.append(quantized_t)
                        while len(item_refs) < max_ref_count:
                            item_refs.append(
                                torch.zeros((ref_channels, quantized_t, ref_mel_bins), device=ref_device, dtype=ref_dtype)
                            )
                            item_lengths.append(0)
                        truncated_ref.append(torch.stack(item_refs))
                        truncated_ref_lengths.append(torch.tensor(item_lengths, device=ref_device, dtype=torch.int32))

                    stacked_ref = torch.stack(truncated_ref)
                    stacked_ref_lengths = torch.stack(truncated_ref_lengths)
                    batch_tensor_data["ref_audio_latents"] = stacked_ref[:, 0] if max_ref_count == 1 else stacked_ref
                    batch_tensor_data["ref_audio_lengths"] = (
                        stacked_ref_lengths[:, 0] if max_ref_count == 1 else stacked_ref_lengths
                    )
                else:
                    ref_lengths = []
                    ref_max_t = 0
                    for i, lat in enumerate(ref_audio_latents_per_item):
                        item_lengths = []
                        if isinstance(lat, list) and lat:
                            cached_lengths = (
                                ref_audio_lengths_per_item[i] if isinstance(ref_audio_lengths_per_item[i], list) else []
                            )
                            for ref_idx, ref_lat in enumerate(lat):
                                t = int(ref_lat.shape[1])
                                length_val = t
                                cached_len = cached_lengths[ref_idx] if ref_idx < len(cached_lengths) else None
                                if isinstance(cached_len, torch.Tensor) and cached_len.numel() == 1:
                                    length_val = int(cached_len.view(-1)[0].item())
                                length_val = max(0, min(length_val, t))
                                item_lengths.append(length_val)
                                ref_max_t = max(ref_max_t, t)
                        ref_lengths.append(item_lengths)

                    if ref_max_t <= 0:
                        ref_max_t = 1

                    padded_ref = []
                    padded_ref_lengths = []
                    for i, lat in enumerate(ref_audio_latents_per_item):
                        item_refs = []
                        item_lengths = []
                        if isinstance(lat, list) and lat:
                            for ref_idx, ref_lat in enumerate(lat):
                                if ref_lat.dim() != 3:
                                    raise ValueError(
                                        f"Expected reference audio latents to be 3D [C, T, F], got {tuple(ref_lat.shape)}"
                                    )
                                if int(ref_lat.shape[0]) != ref_channels or int(ref_lat.shape[2]) != ref_mel_bins:
                                    raise ValueError(
                                        "Reference audio latents shape mismatch in batch: "
                                        f"expected [C={ref_channels}, *, F={ref_mel_bins}], got {tuple(ref_lat.shape)}"
                                    )

                                t = int(ref_lat.shape[1])
                                use_t = min(t, ref_max_t)
                                out = torch.zeros((ref_channels, ref_max_t, ref_mel_bins), device=ref_device, dtype=ref_dtype)
                                if use_t > 0:
                                    out[:, :use_t, :] = ref_lat[:, :use_t, :].to(device=ref_device, dtype=ref_dtype)
                                item_refs.append(out)
                                base_lengths = ref_lengths[i] if i < len(ref_lengths) else []
                                length_val = base_lengths[ref_idx] if ref_idx < len(base_lengths) else 0
                                item_lengths.append(int(min(max(0, length_val), ref_max_t)))
                        while len(item_refs) < max_ref_count:
                            item_refs.append(
                                torch.zeros((ref_channels, ref_max_t, ref_mel_bins), device=ref_device, dtype=ref_dtype)
                            )
                            item_lengths.append(0)
                        padded_ref.append(torch.stack(item_refs))
                        padded_ref_lengths.append(torch.tensor(item_lengths, device=ref_device, dtype=torch.int32))

                    stacked_ref = torch.stack(padded_ref)
                    stacked_ref_lengths = torch.stack(padded_ref_lengths)
                    batch_tensor_data["ref_audio_latents"] = stacked_ref[:, 0] if max_ref_count == 1 else stacked_ref
                    batch_tensor_data["ref_audio_lengths"] = (
                        stacked_ref_lengths[:, 0] if max_ref_count == 1 else stacked_ref_lengths
                    )

        if self.timestep_pool is not None:
            batch_tensor_data["timesteps"] = self.timestep_pool[idx][: end - start]  # use the pre-generated timesteps
        else:
            batch_tensor_data["timesteps"] = None

        if batch_count > 0:
            if self.video_loss_weight is not None:
                batch_tensor_data["video_loss_weight"] = float(self.video_loss_weight)
            if self.audio_loss_weight is not None:
                batch_tensor_data["audio_loss_weight"] = float(self.audio_loss_weight)

        if self.architecture in {ARCHITECTURE_LTX2, ARCHITECTURE_LTX2_FULL}:
            latents = batch_tensor_data.get("latents")
            if isinstance(latents, torch.Tensor) and latents.dim() == 5:
                bsz, _c, frames, height, width = latents.shape
                virtual_num_frames = batch_tensor_data.pop("ltx2_virtual_num_frames", None)
                virtual_height = batch_tensor_data.pop("ltx2_virtual_height", None)
                virtual_width = batch_tensor_data.pop("ltx2_virtual_width", None)

                num_frames_tensor = torch.full((bsz,), frames, dtype=torch.int32)
                height_tensor = torch.full((bsz,), height, dtype=torch.int32)
                width_tensor = torch.full((bsz,), width, dtype=torch.int32)
                if (
                    isinstance(virtual_num_frames, torch.Tensor)
                    and isinstance(virtual_height, torch.Tensor)
                    and isinstance(virtual_width, torch.Tensor)
                ):
                    if virtual_num_frames.numel() == bsz and virtual_height.numel() == bsz and virtual_width.numel() == bsz:
                        num_frames_tensor = virtual_num_frames.to(dtype=torch.int32).view(-1)
                        height_tensor = virtual_height.to(dtype=torch.int32).view(-1)
                        width_tensor = virtual_width.to(dtype=torch.int32).view(-1)

                batch_tensor_data["latents"] = {
                    "latents": latents,
                    "num_frames": num_frames_tensor,
                    "height": height_tensor,
                    "width": width_tensor,
                    "fps": torch.full((bsz,), self.target_fps, dtype=torch.float32),
                }

            ref_latents = batch_tensor_data.get("ref_latents")
            if isinstance(ref_latents, torch.Tensor) and ref_latents.dim() == 5:
                bsz, _c, frames, height, width = ref_latents.shape
                batch_tensor_data["ref_latents"] = {
                    "latents": ref_latents,
                    "num_frames": torch.full((bsz,), frames, dtype=torch.int32),
                    "height": torch.full((bsz,), height, dtype=torch.int32),
                    "width": torch.full((bsz,), width, dtype=torch.int32),
                    "fps": torch.full((bsz,), self.target_fps, dtype=torch.float32),
                }

            # Wrap latent-guide tensors with frame_idx + strength metadata for the trainer.
            # The bucket invariant guarantees all items in this batch share the same
            # frame_idx + strength (encoded into the bucket key).
            for _gkey, _frame_attr, _strength_attr in (
                ("latent_idx_guide_latents", "latent_idx_guide_frame_idx", "latent_idx_guide_strength"),
                ("keyframe_guide_latents", "keyframe_guide_frame_idx", "keyframe_guide_strength"),
            ):
                _gtensor = batch_tensor_data.get(_gkey)
                if isinstance(_gtensor, torch.Tensor) and _gtensor.dim() == 5:
                    batch_tensor_data[_gkey] = {
                        "latents": _gtensor,
                        "frame_idx": int(getattr(self, _frame_attr)),
                        "strength": float(getattr(self, _strength_attr)),
                    }

            # Multi-keyframe extras: gather all `keyframe_guide_extra_{i}_latents`
            # tensors (already stacked per-batch) into a parallel list of dicts
            # so the trainer can iterate primary + extras uniformly.
            extras_specs = getattr(self, "keyframe_guide_extras", None) or []
            if extras_specs:
                extras_batch: List[Dict[str, Any]] = []
                for _ix, spec in enumerate(extras_specs, start=1):
                    _key = f"keyframe_guide_extra_{_ix}_latents"
                    _t = batch_tensor_data.pop(_key, None)
                    if isinstance(_t, torch.Tensor) and _t.dim() == 5:
                        extras_batch.append(
                            {
                                "latents": _t,
                                "frame_idx": int(spec.get("frame_idx", -1)),
                                "strength": float(spec.get("strength", 1.0)),
                            }
                        )
                if extras_batch:
                    batch_tensor_data["keyframe_guide_extras"] = extras_batch

            audio_latents_tensor = batch_tensor_data.get("audio_latents")
            if isinstance(audio_latents_tensor, torch.Tensor) and audio_latents_tensor.dim() == 4:
                batch_tensor_data["audio_latents"] = {"latents": audio_latents_tensor}

            ref_audio_latents_tensor = batch_tensor_data.get("ref_audio_latents")
            if isinstance(ref_audio_latents_tensor, torch.Tensor) and ref_audio_latents_tensor.dim() == 4:
                batch_tensor_data["ref_audio_latents"] = {"latents": ref_audio_latents_tensor}

            video_prompt_embeds = batch_tensor_data.get("video_prompt_embeds")
            audio_prompt_embeds = batch_tensor_data.get("audio_prompt_embeds")
            prompt_attention_mask = batch_tensor_data.get("prompt_attention_mask")

            conditions: dict[str, torch.Tensor] = {}
            if isinstance(video_prompt_embeds, torch.Tensor):
                conditions["video_prompt_embeds"] = video_prompt_embeds
                if isinstance(audio_prompt_embeds, torch.Tensor):
                    conditions["audio_prompt_embeds"] = audio_prompt_embeds
                if isinstance(prompt_attention_mask, torch.Tensor):
                    conditions["prompt_attention_mask"] = prompt_attention_mask

            # Pre-connector features for --train_connectors training
            video_features = batch_tensor_data.get("video_features")
            if isinstance(video_features, torch.Tensor):
                conditions["video_features"] = video_features
            audio_features = batch_tensor_data.get("audio_features")
            if isinstance(audio_features, torch.Tensor):
                conditions["audio_features"] = audio_features

            if not conditions:
                text = batch_tensor_data.get("text")
                text_mask = batch_tensor_data.get("text_mask")
                if isinstance(text, torch.Tensor):
                    if isinstance(text_mask, torch.Tensor):
                        conditions["prompt_attention_mask"] = text_mask

                    # Legacy cache fallback: keep full prompt_embeds so runtime can
                    # split by model dims (video/audio) when needed.
                    conditions["prompt_embeds"] = text
                    conditions["video_prompt_embeds"] = text

            # DINOv2 features (pre-cached, for CREPA dino mode)
            if any(d is not None for d in dino_features_per_item):
                if all(d is not None for d in dino_features_per_item):
                    # Pad to max T_pixel and stack
                    # Features are [T, N_patches, D] (patch tokens)
                    max_t = max(d.shape[0] for d in dino_features_per_item)
                    padded = []
                    for d in dino_features_per_item:
                        if d.shape[0] < max_t:
                            # Pad along T dim only; N_patches and D are constant
                            pad_shape = [max_t - d.shape[0]] + list(d.shape[1:])
                            pad = torch.zeros(pad_shape, dtype=d.dtype)
                            d = torch.cat([d, pad], dim=0)
                        padded.append(d)
                    conditions["dino_features"] = torch.stack(padded)  # [B, T_pixel, N_patches, D_dino]

            if conditions:
                batch_tensor_data["conditions"] = conditions

        if collect_item_keys:
            batch_tensor_data["item_keys"] = item_keys
            batch_tensor_data["latent_cache_paths"] = latent_cache_paths
            batch_tensor_data["audio_cache_paths"] = audio_cache_paths
            batch_tensor_data["text_cache_paths"] = text_cache_paths
        batch_tensor_data["captions"] = captions

        return batch_tensor_data
