from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

import torch

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.dataset.cache_io import save_latent_cache_common, save_text_encoder_output_cache_common
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.architecture import AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS, TEXT_DIM, VIDEO_LATENT_CHANNELS
from musubi_tuner.minimax_h3.image_training import file_identity
from musubi_tuner.minimax_h3.media import MediaAsset, MediaModality
from musubi_tuner.minimax_h3.references import (
    REFERENCE_IMAGE_SHORT_EDGE,
    REFERENCE_IMAGE_SIZE_MODE,
    REFERENCE_VIDEO_FPS,
    REFERENCE_VIDEO_MAX_PIXELS,
    REFERENCE_VIDEO_SHORT_EDGE,
    validate_reference_image_short_edge,
    validate_reference_image_sizing,
    validate_reference_video_fps,
    validate_reference_video_sizing,
)
from musubi_tuner.utils.model_utils import dtype_to_str, remove_dtype_suffix

H3_AUDIO_LATENTS_KEY = "latents_audio"
H3_AUDIO_LOSS_MASK_KEY = "audio_loss_mask"
H3_VIDEO_GEOMETRY_KEY = "mmh3_video_geometry"
H3_TEXT_HIDDEN_KEY = "mmh3_hidden_states"
H3_TEXT_TOKEN_TAGS_KEY = "mmh3_token_tags"
H3_MAX_CAPTION_TOKENS_KEY = "mmh3_max_caption_tokens"
H3_TEXT_VISUAL_MAX_PIXELS_KEY = "mmh3_text_visual_max_pixels"
H3_EMPTY_TEXT_HIDDEN_KEY = "mmh3_empty_hidden_states"
H3_EMPTY_TEXT_TOKEN_TAGS_KEY = "mmh3_empty_token_tags"
H3_CONDITIONING_TASK_KEY = "mmh3_conditioning_task"
H3_REFERENCE_IMAGE_SHORT_EDGE_KEY = "mmh3_reference_image_short_edge"
H3_REFERENCE_IMAGE_SIZE_MODE_KEY = "mmh3_reference_image_size_mode"
H3_REFERENCE_IMAGE_MAX_PIXELS_KEY = "mmh3_reference_image_max_pixels"
H3_REFERENCE_VIDEO_SHORT_EDGE_KEY = "mmh3_reference_video_short_edge"
H3_REFERENCE_VIDEO_MAX_PIXELS_KEY = "mmh3_reference_video_max_pixels"
H3_REFERENCE_VIDEO_FPS_KEY = "mmh3_reference_video_fps"
H3_CONDITIONING_TASK_IDS = {"t2va": 0, "i2va": 1, "fl2va": 2, "ref2va": 3, "ref2va_omni": 4, "l2va": 5}
H3_KEYFRAME_VIDEO_ROWS_KEY = "mmh3_keyframe_video_rows"
H3_REFERENCE_KINDS_KEY = "mmh3_reference_kinds"
H3_REFERENCE_VIDEO_SHAPES_KEY = "mmh3_reference_video_shapes"
H3_REFERENCE_AUDIO_LENGTHS_KEY = "mmh3_reference_audio_lengths"
H3_REFERENCE_VIDEO_ROWS_KEY = "mmh3_reference_video_rows"
H3_REFERENCE_AUDIO_ROWS_KEY = "mmh3_reference_audio_rows"
H3_REFERENCE_ALIGNED_KEY = "mmh3_reference_aligned_to_target"
H3_ALIGNED_GUIDE_COUNT_KEY = "mmh3_aligned_guide_count"
H3_REFERENCE_MODALITY_PROBABILITIES_KEY = "mmh3_reference_modality_probabilities"
H3_REFERENCE_TEMPORAL_CONTRACT_KEY = "mmh3_reference_temporal_contract"
H3_REFERENCE_TEMPORAL_CONTRACT_VERSION = 1
H3_QWEN_CONTROL_VISUALS_KEY = "mmh3_qwen_control_visuals"

# EXPERIMENTAL. Target-video frames presented to the Qwen3-VL conditioner on the
# T2VA route, so custom keyframe anchors regain the conditioner visibility the
# released i2va/fl2va/l2va presentations already have. Nothing here reaches a
# VAE or adds a DiT row: the anchors themselves are still pinned by the trainer.
H3_KEYFRAME_VISUALS_KEY = "mmh3_keyframe_visuals"
# ``last`` cannot be resolved until the item's frame count is known, so it keeps
# its own sentinel inside the cached identity vector.
H3_KEYFRAME_VISUAL_LAST = -1

# EXPERIMENTAL. Control imagery shown only to the Qwen3-VL conditioner. These
# assets never reach a VAE, so they change the text cache and nothing else: the
# role is deliberately distinct from "reference" so the Ref2VA reference channel
# and its DiT rows stay exactly as the released model defines them.
QWEN_CONTROL_ROLE = "qwen_control"
QWEN_CONTROL_FINGERPRINT_KEY = "qwen_control_fingerprint"
MAX_QWEN_CONTROL_IMAGES = 9
MAX_QWEN_CONTROL_VIDEOS = 3


def qwen_control_assets(item: Any) -> tuple[MediaAsset, ...]:
    """The ordered Qwen-only control visuals attached to one dataset item."""
    assets = tuple(asset for asset in getattr(item, "h3_media_assets", ()) if asset.role == QWEN_CONTROL_ROLE)
    limits = ((MediaModality.IMAGE, MAX_QWEN_CONTROL_IMAGES), (MediaModality.VIDEO, MAX_QWEN_CONTROL_VIDEOS))
    for modality, limit in limits:
        count = sum(asset.modality is modality for asset in assets)
        if count > limit:
            raise ValueError(f"MiniMax H3 accepts at most {limit} Qwen control {modality.value}s, got {count}")
    unsupported = sorted(
        {asset.modality.value for asset in assets if asset.modality not in {MediaModality.IMAGE, MediaModality.VIDEO}}
    )
    if unsupported:
        raise ValueError(f"MiniMax H3 Qwen control visuals must be images or videos, got: {', '.join(unsupported)}")
    return assets


def qwen_control_fingerprint(assets: Sequence[MediaAsset]) -> str | None:
    """Identify the Qwen control files so editing or swapping one rebuilds the text cache."""
    controls = [asset for asset in assets if asset.role == QWEN_CONTROL_ROLE]
    if not controls:
        return None
    entries = [
        {"order": order, "modality": asset.modality.value, **file_identity(asset.path)} for order, asset in enumerate(controls)
    ]
    encoded = json.dumps({"format": 1, "qwen_controls": entries}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_keyframe_visuals(spec: str) -> tuple[int, ...]:
    """Parse ``--h3_keyframe_visuals`` into canonical target-frame indices.

    ``first`` collapses to ``0``: unlike a packer anchor, this list names decoded
    *pixel* frames, where the two are literally the same image. ``last`` keeps
    the ``H3_KEYFRAME_VISUAL_LAST`` sentinel because the frame count is only
    known once the item is decoded, and the cached identity must be comparable
    without decoding anything.
    """
    if not spec:
        return ()
    indices: list[int] = []
    for piece in spec.split(","):
        token = piece.strip()
        if token == "first":
            index = 0
        elif token == "last":
            index = H3_KEYFRAME_VISUAL_LAST
        elif token.isdigit():
            index = int(token)
        else:
            raise ValueError(f"H3 keyframe visual {token!r} must be 'first', 'last', or a non-negative frame index")
        if index in indices:
            raise ValueError(f"H3 keyframe visual {token!r} is listed twice")
        indices.append(index)
    return tuple(indices)


def resolve_keyframe_visuals(indices: Sequence[int], frame_count: int) -> tuple[int, ...]:
    """Resolve a parsed keyframe-visual list against one item's decoded frames."""
    resolved: list[int] = []
    for index in indices:
        frame = frame_count - 1 if int(index) == H3_KEYFRAME_VISUAL_LAST else int(index)
        if not 0 <= frame < frame_count:
            raise ValueError(f"H3 keyframe visual {format_keyframe_visuals((index,))} is outside the {frame_count} target frames")
        if frame in resolved:
            raise ValueError(f"H3 keyframe visuals resolved to duplicate target frame {frame}")
        resolved.append(frame)
    return tuple(resolved)


def format_keyframe_visuals(indices: Sequence[int]) -> str:
    """Spell a parsed keyframe-visual list the way the user wrote it."""
    return ",".join("last" if int(index) == H3_KEYFRAME_VISUAL_LAST else str(int(index)) for index in indices)


def qwen_control_dropout_key(key: str) -> str:
    """Name the control-free twin of a cached presentation key.

    EXPERIMENTAL per-step control dropout needs both presentations side by side
    in one cache, exactly as the reference-modality variants do: the trainer
    draws between them per step and never re-encodes anything.
    """
    return f"{key}_no_qwen_control"


def reference_variant_key(key: str, modality: str) -> str:
    if modality not in {"video", "audio"}:
        raise ValueError(f"unsupported H3 reference modality variant: {modality}")
    return f"{key}_reference_{modality}"


def format_reference_video_fps(sample_fps: float) -> str:
    """Spell a subsampling rate for a cache key: 2.0 as ``2``, 2.5 as ``2p5``."""
    return f"{float(sample_fps):g}".replace(".", "p").replace("+", "")


def reference_key_suffix(
    image_short_edge: int,
    image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
    image_max_pixels: int = 0,
    video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
    video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
    video_sample_fps: float = REFERENCE_VIDEO_FPS,
) -> str:
    """Name reference caches whose pixels were scaled to a non-released short edge."""
    validate_reference_image_short_edge(image_short_edge)
    validate_reference_image_sizing(image_size_mode, image_max_pixels)
    validate_reference_video_sizing(video_short_edge, video_max_pixels)
    video_sample_fps = validate_reference_video_fps(video_sample_fps)
    video_suffix = ""
    if video_short_edge != REFERENCE_VIDEO_SHORT_EDGE:
        video_suffix += f"_vse{video_short_edge}"
    if video_max_pixels != REFERENCE_VIDEO_MAX_PIXELS:
        video_suffix += f"_vmp{video_max_pixels}"
    if video_sample_fps != REFERENCE_VIDEO_FPS:
        # Only a non-default rate is named, so caches written before temporal
        # subsampling existed keep their identity.
        video_suffix += f"_vfps{format_reference_video_fps(video_sample_fps)}"
    if image_size_mode == "target_area":
        image_suffix = "_ta" if image_max_pixels == 0 else f"_ta{image_max_pixels}"
    else:
        image_suffix = "" if image_short_edge == REFERENCE_IMAGE_SHORT_EDGE else f"_se{image_short_edge}"
    return image_suffix + video_suffix


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


def logical_cache_key(key: str) -> str:
    if key.startswith("varlen_"):
        key = key.removeprefix("varlen_")
    return remove_dtype_suffix(key)


def save_latent_cache_minimax_h3(item_info: ItemInfo, tensors: dict[str, torch.Tensor]) -> None:
    """Save normalized H3 latents in Musubi's cache layout.

    Audio is stereo-major ``[2, 32, T]``. Backends are responsible for
    converting native model layouts such as ``[B, 32, 2, T]`` at this boundary.
    Modality-only caches omit the unused target tensor entirely.
    """
    cache_tensors = _validated_cache_tensors(item_info, tensors, operation="latent encoder")
    primary_latents = [key for key in cache_tensors if re.fullmatch(r"latents_\d+x\d+x\d+_.+", key)]
    target_mode = getattr(item_info, "h3_target_mode", "av")
    if len(primary_latents) > 1 or (target_mode != "audio" and len(primary_latents) != 1):
        raise ValueError(f"H3 latent cache for {item_info.item_key} must contain exactly one latents_FxHxW_<dtype> tensor")
    audio_latents = [key for key in cache_tensors if re.fullmatch(r"latents_audio_2x32x\d+_.+", key)]
    is_image = any(
        getattr(asset, "role", None) == "target" and getattr(asset, "modality", None) is MediaModality.IMAGE
        for asset in getattr(item_info, "h3_media_assets", ())
    )
    audio_required = target_mode in {"av", "audio"} and not is_image
    if len(audio_latents) > 1 or (audio_required and not audio_latents) or (target_mode == "video" and audio_latents):
        raise ValueError(
            f"H3 latent cache for {item_info.item_key} must contain exactly one "
            "latents_audio_2x32xT_<dtype> tensor unless it is an image item"
        )

    video = None
    if primary_latents:
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

    geometry_matches = [tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_VIDEO_GEOMETRY_KEY]
    geometry = geometry_matches[0] if len(geometry_matches) == 1 else None
    if target_mode == "audio" and (geometry is None or geometry.dtype is not torch.long or geometry.shape != (2,)):
        raise ValueError(f"H3 audio-only cache requires int64 {H3_VIDEO_GEOMETRY_KEY} with shape [2]")
    video_mask = cache_tensors.get("video_loss_mask")
    if video_mask is not None and (video is None or video_mask.dtype is not torch.bool or video_mask.shape != video.shape[-3:]):
        raise ValueError(f"H3 video_loss_mask must be bool with shape {tuple(video.shape[-3:])}")
    save_latent_cache_common(item_info, cache_tensors, ARCHITECTURE_MINIMAX_H3_FULL)


def save_text_encoder_output_cache_minimax_h3(
    item_info: ItemInfo,
    tensors: dict[str, torch.Tensor],
) -> None:
    cache_tensors = _validated_cache_tensors(item_info, tensors, operation="conditioning encoder")
    logical_keys = {logical_cache_key(key) for key in cache_tensors}
    required = {H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY, H3_CONDITIONING_TASK_KEY}
    missing = sorted(required - logical_keys)
    if missing:
        raise ValueError(f"H3 conditioning cache for {item_info.item_key} is missing: {', '.join(missing)}")
    empty_keys = {H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY}
    if logical_keys & empty_keys and not empty_keys <= logical_keys:
        raise ValueError("H3 empty conditioning cache must contain both hidden states and token tags")

    def tensor_for(logical_key: str) -> torch.Tensor:
        matches = [tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == logical_key]
        if len(matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain exactly one {logical_key} tensor")
        return matches[0]

    def validate_pair(hidden_key: str, tags_key: str) -> None:
        hidden = tensor_for(hidden_key)
        tags = tensor_for(tags_key)
        if hidden.ndim != 2 or hidden.shape[-1] != TEXT_DIM:
            raise ValueError(f"H3 {hidden_key} must have shape [tokens, {TEXT_DIM}], got {tuple(hidden.shape)}")
        # A zero-row presentation is structurally well formed and therefore used to
        # cache cleanly, only to fail much later when packing built the sequence.
        if hidden.shape[0] == 0:
            raise ValueError(f"H3 {hidden_key} carries no tokens; conditioning cannot be empty")
        if tags.dtype != torch.long or tags.shape != (hidden.shape[0],):
            raise ValueError(f"H3 {tags_key} must be int64 with shape [{hidden.shape[0]}]")

    validate_pair(H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY)
    task = tensor_for(H3_CONDITIONING_TASK_KEY)
    if task.dtype != torch.long or task.ndim != 0 or int(task) not in H3_CONDITIONING_TASK_IDS.values():
        raise ValueError(f"H3 {H3_CONDITIONING_TASK_KEY} must be a scalar int64 task id")
    caption_cap_matches = [tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_MAX_CAPTION_TOKENS_KEY]
    if caption_cap_matches:
        if len(caption_cap_matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain at most one {H3_MAX_CAPTION_TOKENS_KEY} tensor")
        caption_cap = caption_cap_matches[0]
        if caption_cap.dtype != torch.long or caption_cap.ndim != 0 or int(caption_cap) <= 0:
            raise ValueError(f"H3 {H3_MAX_CAPTION_TOKENS_KEY} must be a positive scalar int64 value")
    visual_cap_matches = [
        tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_TEXT_VISUAL_MAX_PIXELS_KEY
    ]
    if visual_cap_matches:
        if len(visual_cap_matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain at most one {H3_TEXT_VISUAL_MAX_PIXELS_KEY} tensor")
        visual_cap = visual_cap_matches[0]
        if visual_cap.dtype != torch.long or visual_cap.ndim != 0 or int(visual_cap) <= 0:
            raise ValueError(f"H3 {H3_TEXT_VISUAL_MAX_PIXELS_KEY} must be a positive scalar int64 value")
    reference_size_matches = [
        tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_REFERENCE_IMAGE_SHORT_EDGE_KEY
    ]
    if reference_size_matches:
        if len(reference_size_matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain at most one {H3_REFERENCE_IMAGE_SHORT_EDGE_KEY} tensor")
        reference_size = reference_size_matches[0]
        if reference_size.dtype != torch.long or reference_size.ndim != 0:
            raise ValueError(f"H3 {H3_REFERENCE_IMAGE_SHORT_EDGE_KEY} must be a scalar int64 value")
        validate_reference_image_short_edge(int(reference_size))
    size_mode_matches = [
        tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_REFERENCE_IMAGE_SIZE_MODE_KEY
    ]
    max_pixels_matches = [
        tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_REFERENCE_IMAGE_MAX_PIXELS_KEY
    ]
    if bool(size_mode_matches) != bool(max_pixels_matches):
        raise ValueError("H3 reference sizing cache identity must contain both mode and max-pixel tensors")
    if size_mode_matches:
        if len(size_mode_matches) != 1 or len(max_pixels_matches) != 1:
            raise ValueError("H3 conditioning cache must contain at most one reference sizing identity")
        size_mode, max_pixels = size_mode_matches[0], max_pixels_matches[0]
        if size_mode.dtype != torch.long or size_mode.ndim != 0 or int(size_mode) not in (0, 1):
            raise ValueError(f"H3 {H3_REFERENCE_IMAGE_SIZE_MODE_KEY} must be scalar int64 0 or 1")
        if max_pixels.dtype != torch.long or max_pixels.ndim != 0 or int(max_pixels) < 0:
            raise ValueError(f"H3 {H3_REFERENCE_IMAGE_MAX_PIXELS_KEY} must be a non-negative scalar int64")
        validate_reference_image_sizing("short_edge" if int(size_mode) == 0 else "target_area", int(max_pixels))
    video_short_edge_matches = [
        tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_REFERENCE_VIDEO_SHORT_EDGE_KEY
    ]
    video_max_pixels_matches = [
        tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_REFERENCE_VIDEO_MAX_PIXELS_KEY
    ]
    if bool(video_short_edge_matches) != bool(video_max_pixels_matches):
        raise ValueError("H3 reference-video sizing cache identity must contain both short-edge and max-pixel tensors")
    if video_short_edge_matches:
        if len(video_short_edge_matches) != 1 or len(video_max_pixels_matches) != 1:
            raise ValueError("H3 conditioning cache must contain at most one reference-video sizing identity")
        video_short_edge, video_max_pixels = video_short_edge_matches[0], video_max_pixels_matches[0]
        if video_short_edge.dtype != torch.long or video_short_edge.ndim != 0:
            raise ValueError(f"H3 {H3_REFERENCE_VIDEO_SHORT_EDGE_KEY} must be a scalar int64 value")
        if video_max_pixels.dtype != torch.long or video_max_pixels.ndim != 0:
            raise ValueError(f"H3 {H3_REFERENCE_VIDEO_MAX_PIXELS_KEY} must be a scalar int64 value")
        validate_reference_video_sizing(int(video_short_edge), int(video_max_pixels))
    video_fps_matches = [tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_REFERENCE_VIDEO_FPS_KEY]
    if video_fps_matches:
        if len(video_fps_matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain at most one {H3_REFERENCE_VIDEO_FPS_KEY} tensor")
        video_fps = video_fps_matches[0]
        if video_fps.dtype is not torch.float64 or video_fps.ndim != 0:
            raise ValueError(f"H3 {H3_REFERENCE_VIDEO_FPS_KEY} must be a scalar float64 value")
        validate_reference_video_fps(float(video_fps))
    qwen_control_matches = [
        tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_QWEN_CONTROL_VISUALS_KEY
    ]
    if qwen_control_matches:
        if len(qwen_control_matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain at most one {H3_QWEN_CONTROL_VISUALS_KEY} tensor")
        qwen_controls = qwen_control_matches[0]
        if qwen_controls.dtype is not torch.long or qwen_controls.ndim != 0 or int(qwen_controls) <= 0:
            raise ValueError(f"H3 {H3_QWEN_CONTROL_VISUALS_KEY} must be a positive scalar int64 count")
    keyframe_visual_matches = [tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_KEYFRAME_VISUALS_KEY]
    if keyframe_visual_matches:
        if len(keyframe_visual_matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain at most one {H3_KEYFRAME_VISUALS_KEY} tensor")
        keyframe_visuals = keyframe_visual_matches[0]
        if keyframe_visuals.dtype is not torch.long or keyframe_visuals.ndim != 1 or keyframe_visuals.numel() == 0:
            raise ValueError(f"H3 {H3_KEYFRAME_VISUALS_KEY} must be a non-empty int64 vector of frame indices")
        if int(keyframe_visuals.min()) < H3_KEYFRAME_VISUAL_LAST:
            raise ValueError(f"H3 {H3_KEYFRAME_VISUALS_KEY} entries must be frame indices or the 'last' sentinel")
        if int(task) != H3_CONDITIONING_TASK_IDS["t2va"]:
            raise ValueError(f"H3 {H3_KEYFRAME_VISUALS_KEY} is only valid for T2VA conditioning")
    reference_contract_matches = [
        tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_REFERENCE_TEMPORAL_CONTRACT_KEY
    ]
    if reference_contract_matches:
        if len(reference_contract_matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain at most one {H3_REFERENCE_TEMPORAL_CONTRACT_KEY} tensor")
        reference_contract = reference_contract_matches[0]
        if (
            reference_contract.dtype is not torch.long
            or reference_contract.ndim != 0
            or int(reference_contract) != H3_REFERENCE_TEMPORAL_CONTRACT_VERSION
        ):
            raise ValueError(
                f"H3 {H3_REFERENCE_TEMPORAL_CONTRACT_KEY} must be scalar int64 version {H3_REFERENCE_TEMPORAL_CONTRACT_VERSION}"
            )
    if empty_keys <= logical_keys:
        validate_pair(H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY)
    probability_matches = [
        tensor for key, tensor in cache_tensors.items() if logical_cache_key(key) == H3_REFERENCE_MODALITY_PROBABILITIES_KEY
    ]
    if probability_matches:
        if len(probability_matches) != 1:
            raise ValueError(f"H3 conditioning cache must contain exactly one {H3_REFERENCE_MODALITY_PROBABILITIES_KEY}")
        probabilities = probability_matches[0]
        if probabilities.dtype is not torch.float32 or probabilities.shape != (3,):
            raise ValueError(f"H3 {H3_REFERENCE_MODALITY_PROBABILITIES_KEY} must be float32 with shape [3]")
        if bool((probabilities < 0).any()) or not torch.isclose(probabilities.sum(), torch.tensor(1.0)):
            raise ValueError(f"H3 {H3_REFERENCE_MODALITY_PROBABILITIES_KEY} must be non-negative and sum to 1")
        for index, modality in enumerate(("av", "video", "audio")):
            if modality == "av" or float(probabilities[index]) == 0:
                continue
            validate_pair(
                reference_variant_key(H3_TEXT_HIDDEN_KEY, modality),
                reference_variant_key(H3_TEXT_TOKEN_TAGS_KEY, modality),
            )
            if empty_keys <= logical_keys:
                validate_pair(
                    reference_variant_key(H3_EMPTY_TEXT_HIDDEN_KEY, modality),
                    reference_variant_key(H3_EMPTY_TEXT_TOKEN_TAGS_KEY, modality),
                )
    dropout_keys = {qwen_control_dropout_key(H3_TEXT_HIDDEN_KEY), qwen_control_dropout_key(H3_TEXT_TOKEN_TAGS_KEY)}
    if logical_keys & dropout_keys:
        if not dropout_keys <= logical_keys:
            raise ValueError("H3 control-free conditioning cache must contain both hidden states and token tags")
        if not qwen_control_matches:
            raise ValueError(f"H3 control-free conditioning requires cached {H3_QWEN_CONTROL_VISUALS_KEY} visuals")
        validate_pair(qwen_control_dropout_key(H3_TEXT_HIDDEN_KEY), qwen_control_dropout_key(H3_TEXT_TOKEN_TAGS_KEY))
        if empty_keys <= logical_keys:
            validate_pair(
                qwen_control_dropout_key(H3_EMPTY_TEXT_HIDDEN_KEY),
                qwen_control_dropout_key(H3_EMPTY_TEXT_TOKEN_TAGS_KEY),
            )
        if probability_matches:
            for index, modality in enumerate(("av", "video", "audio")):
                if modality == "av" or float(probabilities[index]) == 0:
                    continue
                validate_pair(
                    qwen_control_dropout_key(reference_variant_key(H3_TEXT_HIDDEN_KEY, modality)),
                    qwen_control_dropout_key(reference_variant_key(H3_TEXT_TOKEN_TAGS_KEY, modality)),
                )
                if empty_keys <= logical_keys:
                    validate_pair(
                        qwen_control_dropout_key(reference_variant_key(H3_EMPTY_TEXT_HIDDEN_KEY, modality)),
                        qwen_control_dropout_key(reference_variant_key(H3_EMPTY_TEXT_TOKEN_TAGS_KEY, modality)),
                    )
    save_text_encoder_output_cache_common(
        item_info,
        cache_tensors,
        ARCHITECTURE_MINIMAX_H3_FULL,
        merge_existing=False,
    )
