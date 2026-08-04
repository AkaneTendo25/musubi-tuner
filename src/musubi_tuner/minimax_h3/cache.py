from __future__ import annotations

import re
from typing import Any

import torch

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.dataset.cache_io import save_latent_cache_common, save_text_encoder_output_cache_common
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.architecture import AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS, TEXT_DIM, VIDEO_LATENT_CHANNELS
from musubi_tuner.minimax_h3.media import MediaModality
from musubi_tuner.utils.model_utils import dtype_to_str, remove_dtype_suffix

H3_AUDIO_LATENTS_KEY = "latents_audio"
H3_AUDIO_LOSS_MASK_KEY = "audio_loss_mask"
H3_TEXT_HIDDEN_KEY = "mmh3_hidden_states"
H3_TEXT_TOKEN_TAGS_KEY = "mmh3_token_tags"
H3_EMPTY_TEXT_HIDDEN_KEY = "mmh3_empty_hidden_states"
H3_EMPTY_TEXT_TOKEN_TAGS_KEY = "mmh3_empty_token_tags"
H3_CONDITIONING_TASK_KEY = "mmh3_conditioning_task"
H3_CONDITIONING_TASK_IDS = {"t2va": 0, "i2va": 1, "fl2va": 2, "ref2va": 3}
H3_KEYFRAME_VIDEO_ROWS_KEY = "mmh3_keyframe_video_rows"
H3_REFERENCE_KINDS_KEY = "mmh3_reference_kinds"
H3_REFERENCE_VIDEO_SHAPES_KEY = "mmh3_reference_video_shapes"
H3_REFERENCE_AUDIO_LENGTHS_KEY = "mmh3_reference_audio_lengths"
H3_REFERENCE_VIDEO_ROWS_KEY = "mmh3_reference_video_rows"
H3_REFERENCE_AUDIO_ROWS_KEY = "mmh3_reference_audio_rows"


def normalize_batch_tensors(results: Any, expected: int, operation: str) -> tuple[dict[str, torch.Tensor], ...]:
    """Validate a backend result at the architecture boundary.

    Implementations return one flat, cache-ready tensor mapping per ItemInfo. The
    mapping deliberately leaves model-specific tensor names to the H3
    integration while keeping all traversal and persistence inside Musubi.
    """
    values = tuple(results)
    if len(values) != expected:
        raise ValueError(f"H3 {operation} returned {len(values)} results for a batch of {expected}")
    normalized: list[dict[str, torch.Tensor]] = []
    for result in values:
        if not isinstance(result, dict) or not result:
            raise TypeError(f"H3 {operation} must return one non-empty tensor mapping per item")
        if not all(isinstance(key, str) and key and isinstance(value, torch.Tensor) for key, value in result.items()):
            raise TypeError(f"H3 {operation} mappings must contain non-empty string keys and tensors")
        normalized.append(result)
    return tuple(normalized)


def _validated_cache_tensors(
    item_info: ItemInfo,
    tensors: dict[str, torch.Tensor],
    *,
    operation: str,
) -> dict[str, torch.Tensor]:
    if not tensors:
        raise ValueError(f"H3 {operation} returned no tensors for {item_info.item_key}")
    for key, tensor in tensors.items():
        dtype_suffix = f"_{dtype_to_str(tensor.dtype)}"
        if tensor.dim() > 0 and not key.endswith((dtype_suffix, "_mask")):
            raise ValueError(
                f"H3 cache tensor {key!r} must use Musubi's dtype suffix {dtype_suffix!r} (or the established _mask suffix)"
            )
    return {key: value.detach().cpu().contiguous() for key, value in tensors.items()}


def _logical_key(key: str) -> str:
    if key.startswith("varlen_"):
        key = key.removeprefix("varlen_")
    return remove_dtype_suffix(key)


def save_latent_cache_minimax_h3(item_info: ItemInfo, tensors: dict[str, torch.Tensor]) -> None:
    """Save normalized H3 latents in Musubi's cache layout.

    Audio is stereo-major ``[2, 32, T]``. Backends are responsible for
    converting native model layouts such as ``[B, 32, 2, T]`` at this boundary.
    Silent video targets retain audio latents and use an all-false audio loss
    mask. Image targets contain only their one-frame video latent.
    """
    cache_tensors = _validated_cache_tensors(item_info, tensors, operation="latent encoder")
    primary_latents = [key for key in cache_tensors if re.fullmatch(r"latents_\d+x\d+x\d+_.+", key)]
    if len(primary_latents) != 1:
        raise ValueError(f"H3 latent cache for {item_info.item_key} must contain exactly one latents_FxHxW_<dtype> tensor")
    audio_latents = [key for key in cache_tensors if re.fullmatch(r"latents_audio_2x32x\d+_.+", key)]
    is_image = any(
        getattr(asset, "role", None) == "target" and getattr(asset, "modality", None) is MediaModality.IMAGE
        for asset in getattr(item_info, "h3_media_assets", ())
    )
    if len(audio_latents) > 1 or (not audio_latents and not is_image):
        raise ValueError(
            f"H3 latent cache for {item_info.item_key} must contain exactly one "
            "latents_audio_2x32xT_<dtype> tensor unless it is an image item"
        )

    primary_key = primary_latents[0]
    primary_match = re.fullmatch(r"latents_(\d+)x(\d+)x(\d+)_.+", primary_key)
    video = cache_tensors[primary_key]
    expected_video_shape = tuple(int(value) for value in primary_match.groups())
    if video.ndim != 4 or video.shape[0] != VIDEO_LATENT_CHANNELS or tuple(video.shape[-3:]) != expected_video_shape:
        raise ValueError(
            f"H3 {primary_key} must have shape [{VIDEO_LATENT_CHANNELS}, F, H, W] matching its cache key, got {tuple(video.shape)}"
        )

    if audio_latents:
        audio = cache_tensors[audio_latents[0]]
        if audio.ndim != 3 or tuple(audio.shape[:2]) != (AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS):
            raise ValueError(
                f"H3 latents_audio must have shape [{AUDIO_CHANNELS}, {AUDIO_LATENT_CHANNELS}, T], got {tuple(audio.shape)}"
            )
        audio_geometry = int(audio_latents[0].split("_2x32x", 1)[1].split("_", 1)[0])
        if audio.shape[-1] != audio_geometry:
            raise ValueError(f"H3 {audio_latents[0]} has audio length {audio.shape[-1]}, expected {audio_geometry}")
        audio_mask = cache_tensors.get(H3_AUDIO_LOSS_MASK_KEY)
        if audio_mask is None:
            raise ValueError(f"H3 latent cache for {item_info.item_key} must contain {H3_AUDIO_LOSS_MASK_KEY}")
        if audio_mask.dtype is not torch.bool or audio_mask.shape != (audio.shape[-1],):
            raise ValueError(f"H3 {H3_AUDIO_LOSS_MASK_KEY} must be bool with shape [{audio.shape[-1]}]")
    elif H3_AUDIO_LOSS_MASK_KEY in cache_tensors:
        raise ValueError(f"H3 {H3_AUDIO_LOSS_MASK_KEY} requires cached audio latents")

    video_mask = cache_tensors.get("video_loss_mask")
    if video_mask is not None and (video_mask.dtype is not torch.bool or video_mask.shape != video.shape[-3:]):
        raise ValueError(f"H3 video_loss_mask must be bool with shape {tuple(video.shape[-3:])}")
    save_latent_cache_common(item_info, cache_tensors, ARCHITECTURE_MINIMAX_H3_FULL)


def save_text_encoder_output_cache_minimax_h3(
    item_info: ItemInfo,
    tensors: dict[str, torch.Tensor],
) -> None:
    cache_tensors = _validated_cache_tensors(item_info, tensors, operation="conditioning encoder")
    logical_keys = {_logical_key(key) for key in cache_tensors}
    required = {H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY, H3_CONDITIONING_TASK_KEY}
    missing = sorted(required - logical_keys)
    if missing:
        raise ValueError(f"H3 conditioning cache for {item_info.item_key} is missing: {', '.join(missing)}")
    empty_keys = {H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY}
    if logical_keys & empty_keys and not empty_keys <= logical_keys:
        raise ValueError("H3 empty conditioning cache must contain both hidden states and token tags")

    def tensor_for(logical_key: str) -> torch.Tensor:
        matches = [tensor for key, tensor in cache_tensors.items() if _logical_key(key) == logical_key]
        if len(matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain exactly one {logical_key} tensor")
        return matches[0]

    def validate_pair(hidden_key: str, tags_key: str) -> None:
        hidden = tensor_for(hidden_key)
        tags = tensor_for(tags_key)
        if hidden.ndim != 2 or hidden.shape[-1] != TEXT_DIM:
            raise ValueError(f"H3 {hidden_key} must have shape [tokens, {TEXT_DIM}], got {tuple(hidden.shape)}")
        if tags.dtype != torch.long or tags.shape != (hidden.shape[0],):
            raise ValueError(f"H3 {tags_key} must be int64 with shape [{hidden.shape[0]}]")

    validate_pair(H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY)
    task = tensor_for(H3_CONDITIONING_TASK_KEY)
    if task.dtype != torch.long or task.ndim != 0 or int(task) not in H3_CONDITIONING_TASK_IDS.values():
        raise ValueError(f"H3 {H3_CONDITIONING_TASK_KEY} must be a scalar int64 task id")
    if empty_keys <= logical_keys:
        validate_pair(H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY)
    save_text_encoder_output_cache_common(
        item_info,
        cache_tensors,
        ARCHITECTURE_MINIMAX_H3_FULL,
        merge_existing=False,
    )
