#!/usr/bin/env python3
"""
Cache video latents for LTX-2 training.

Uses the standard musubi-tuner dataset config so cached files match the trainer.
"""

from __future__ import annotations

import argparse
import math
import os
from contextlib import nullcontext
from typing import List, Optional, Sequence, cast

import logging
import numpy as np
import torch
from safetensors.torch import save_file

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.audio_io_utils import coerce_decoded_audio_to_channels_first
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_LTX2,
    AUDIO_EXTENSIONS,
    BaseDataset,
    IMAGE_EXTENSIONS,
    ItemInfo,
    AudioDataset,
    ImageDataset,
    VIDEO_EXTENSIONS,
    VideoDataset,
    resize_image_to_bucket,
    save_latent_cache_ltx2,
)
from musubi_tuner.ltx_2.model.audio_vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR
from musubi_tuner.ltx_2.env import get_ltx2_env
from musubi_tuner.model_defaults import default_ltx2_checkpoint_path
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LTX2_VIDEO_TEMPORAL_DOWNSAMPLE_FACTOR = 8
LTX2_VIDEO_SPATIAL_DOWNSAMPLE_FACTOR = 32
LTX2_AUDIO_ONLY_PROXY_LATENT_FRAMES = 1
LTX2_AUDIO_ONLY_PROXY_LATENT_HEIGHT = 1
LTX2_AUDIO_ONLY_PROXY_LATENT_WIDTH = 1


def _amp_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "xpu"}:
        try:
            from torch.amp import autocast as torch_autocast  # type: ignore[attr-defined]

            return torch_autocast(device_type=device.type, dtype=dtype)
        except (ImportError, AttributeError):
            from torch.cuda.amp import autocast as torch_autocast

            return torch_autocast(dtype=dtype)
    return nullcontext()


def _load_datasets(args: argparse.Namespace) -> Sequence[BaseDataset]:
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LTX2)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.dataset_group,
        reference_downscale=getattr(args, "reference_downscale", 1),
    )
    datasets = list(dataset_group.datasets)

    if user_config.get("validation_datasets"):
        logger.info("Load validation datasets from dataset config")
        validation_user_config = {
            "general": user_config.get("general", {}),
            "datasets": user_config.get("validation_datasets", []),
        }
        validation_blueprint = blueprint_generator.generate(
            validation_user_config, args, architecture=ARCHITECTURE_LTX2
        )
        validation_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            validation_blueprint.dataset_group,
            reference_downscale=getattr(args, "reference_downscale", 1),
        )
        datasets.extend(validation_dataset_group.datasets)

    return cast(Sequence[BaseDataset], datasets)


def encode_and_save_batch(vae, batch: List[ItemInfo], tiling_config=None) -> None:
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if contents.ndim == 4:
        contents = contents.unsqueeze(1)

    loss_masks = []
    has_loss_masks = any(getattr(item, "loss_mask_content", None) is not None for item in batch)
    if has_loss_masks:
        for item in batch:
            mask = getattr(item, "loss_mask_content", None)
            if mask is None:
                mask = np.ones((contents.shape[1], contents.shape[2], contents.shape[3]), dtype=np.float32)
            mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            if mask_tensor.shape[0] < contents.shape[1]:
                pad = contents.shape[1] - mask_tensor.shape[0]
                mask_tensor = torch.cat([mask_tensor, mask_tensor[-1:].expand(pad, -1, -1)], dim=0)
            elif mask_tensor.shape[0] > contents.shape[1]:
                mask_tensor = mask_tensor[: contents.shape[1]]
            loss_masks.append(mask_tensor.clamp(0.0, 1.0))

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()
    vae_param = next(vae.parameters())
    device = vae_param.device
    vae_dtype = vae_param.dtype
    contents = contents.to(device=device, dtype=vae_dtype)
    contents = contents / 127.5 - 1.0

    frames = contents.shape[2]
    remainder = (frames - 1) % 8
    if remainder != 0:
        pad = 8 - remainder
        last = contents[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
        contents = torch.cat([contents, last], dim=2)
        if has_loss_masks:
            padded_masks = []
            for mask_tensor in loss_masks:
                mask_pad = mask_tensor[-1:].expand(pad, -1, -1)
                padded_masks.append(torch.cat([mask_tensor, mask_pad], dim=0))
            loss_masks = padded_masks

    height, width = contents.shape[-2:]
    if height < 8 or width < 8:
        item = batch[0]
        raise ValueError(f"Image or video size too small: {item.item_key} (and others), size: {item.original_size}")

    with _amp_context(device, vae_dtype), torch.no_grad():
        if tiling_config is not None and hasattr(vae, "tiled_encode"):
            latents = vae.tiled_encode(contents, tiling_config)
        else:
            latents = vae(contents)
        latents = latents.to(device=device, dtype=vae_dtype)

    for idx, item in enumerate(batch):
        extra_tensors = None
        if has_loss_masks:
            latent_mask = torch.nn.functional.interpolate(
                loss_masks[idx].view(1, 1, loss_masks[idx].shape[0], loss_masks[idx].shape[1], loss_masks[idx].shape[2]).to(
                    device=latents.device,
                    dtype=torch.float32,
                ),
                size=tuple(latents[idx].shape[1:]),
                mode="trilinear",
                align_corners=False,
            )[0].clamp(0.0, 1.0)
            extra_tensors = {"video_loss_mask": latent_mask.to(dtype=torch.float32)}
        save_latent_cache_ltx2(item, latents[idx], extra_tensors=extra_tensors)


def _adjust_ltx2_frame_count(frame_count: int) -> int:
    frame_count = max(int(frame_count), 1)
    if frame_count % 8 == 1:
        return frame_count
    return max(((frame_count - 1) // 8) * 8 + 1, 1)


def _estimate_audio_duration_seconds(audio_path: str) -> Optional[float]:
    normalized_audio_path = os.path.normpath(audio_path)
    if not os.path.exists(normalized_audio_path):
        return None

    try:
        import torchaudio

        info = torchaudio.info(normalized_audio_path)
        num_frames = int(getattr(info, "num_frames", 0))
        sample_rate = int(getattr(info, "sample_rate", 0))
        if num_frames > 0 and sample_rate > 0:
            return float(num_frames) / float(sample_rate)
    except Exception:
        pass

    try:
        import av  # type: ignore
    except Exception:
        return None

    try:
        container = av.open(normalized_audio_path)
    except Exception:
        return None

    try:
        audio_stream = None
        for stream in container.streams:
            if stream.type == "audio":
                audio_stream = stream
                break
        if audio_stream is None:
            return None

        duration = getattr(audio_stream, "duration", None)
        time_base = getattr(audio_stream, "time_base", None)
        if duration is not None and time_base is not None:
            duration_s = float(duration * time_base)
            if duration_s > 0:
                return duration_s

        if getattr(container, "duration", None):
            # PyAV container duration is in microseconds.
            duration_s = float(container.duration) / 1_000_000.0
            if duration_s > 0:
                return duration_s
    except Exception:
        return None
    finally:
        try:
            container.close()
        except Exception:
            pass

    return None


def _estimate_video_frame_count_from_audio(audio_path: str, *, target_fps: float) -> Optional[int]:
    duration_s = _estimate_audio_duration_seconds(audio_path)
    if duration_s is None:
        return None
    frame_count = max(int(round(duration_s * max(float(target_fps), 1.0))), 1)
    return _adjust_ltx2_frame_count(frame_count)


def build_audio_only_video_latent(
    *,
    channels: int,
    dtype: torch.dtype,
    target_width: int,
    target_height: int,
    target_fps: float,
    audio_path: str,
    sequence_resolution: int = 64,
    temporal_downsample_factor: int = LTX2_VIDEO_TEMPORAL_DOWNSAMPLE_FACTOR,
    spatial_downsample_factor: int = LTX2_VIDEO_SPATIAL_DOWNSAMPLE_FACTOR,
) -> tuple[torch.Tensor, int, tuple[int, int, int]]:
    frame_count = _estimate_video_frame_count_from_audio(audio_path, target_fps=target_fps)
    if frame_count is None:
        raise ValueError(
            f"Could not determine audio duration for {audio_path}. "
            "Audio-only mode requires duration-aware video latent geometry."
        )
    virtual_latent_frames = max((frame_count - 1) // int(temporal_downsample_factor) + 1, 1)
    # Audio-only training does not optimize video loss, so huge virtual spatial geometry
    # only distorts sequence-length-dependent timestep sampling. Use minimal spatial
    # tokens by default, while keeping an override for advanced users.
    if int(sequence_resolution) > 0:
        virtual_latent_h = max(int(sequence_resolution) // int(spatial_downsample_factor), 1)
        virtual_latent_w = max(int(sequence_resolution) // int(spatial_downsample_factor), 1)
    else:
        virtual_latent_h = max(int(target_height) // int(spatial_downsample_factor), 1)
        virtual_latent_w = max(int(target_width) // int(spatial_downsample_factor), 1)
    latent = torch.zeros(
        (
            int(channels),
            int(LTX2_AUDIO_ONLY_PROXY_LATENT_FRAMES),
            int(LTX2_AUDIO_ONLY_PROXY_LATENT_HEIGHT),
            int(LTX2_AUDIO_ONLY_PROXY_LATENT_WIDTH),
        ),
        dtype=dtype,
    )
    return latent, frame_count, (virtual_latent_frames, virtual_latent_h, virtual_latent_w)


def infer_video_in_channels_from_checkpoint(model_path: str) -> Optional[int]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LTX-2 checkpoint not found: {model_path}")

    with MemoryEfficientSafeOpen(model_path) as handle:
        for key in handle.keys():
            if not key.endswith("patchify_proj.weight"):
                continue
            if key.endswith("audio_patchify_proj.weight"):
                continue
            weight = handle.get_tensor(key)
            if weight.ndim < 2:
                continue
            return int(weight.shape[1])
    return None


def _audio_cache_path(item_info: ItemInfo) -> str:
    base_dir = os.path.dirname(item_info.latent_cache_path)
    base_name = os.path.basename(item_info.latent_cache_path)
    if not base_name.endswith("_ltx2.safetensors"):
        return os.path.join(base_dir, f"{item_info.item_key}_ltx2_audio.safetensors")
    return os.path.join(base_dir, base_name.replace("_ltx2.safetensors", "_ltx2_audio.safetensors"))


def _resolve_audio_path(
    item_info: ItemInfo,
    *,
    source: str,
    audio_dir: Optional[str],
    audio_ext: str,
) -> str:
    if source == "video":
        return getattr(item_info, "source_item_key", None) or item_info.item_key
    if source != "audio_files":
        raise ValueError(f"Unexpected audio source: {source}")

    base = os.path.splitext(os.path.basename(item_info.item_key))[0]
    if audio_dir is not None:
        return os.path.join(audio_dir, base + audio_ext)
    return os.path.join(os.path.dirname(item_info.item_key), base + audio_ext)


def _expected_audio_latent_length_for_item(
    item_info: ItemInfo,
    encoder,
    *,
    fps: float,
) -> Optional[int]:
    frame_count = getattr(item_info, "frame_count", None)
    if not isinstance(frame_count, int) or frame_count <= 0:
        return None
    sample_rate = int(getattr(encoder, "sample_rate", 16000))
    hop_length = int(getattr(encoder, "mel_hop_length", 160))
    latents_per_second = float(sample_rate) / float(hop_length) / float(LATENT_DOWNSAMPLE_FACTOR)
    duration_s = float(frame_count) / max(float(fps), 1.0)
    return max(int(duration_s * latents_per_second), 1)


def _align_audio_latents_to_video(audio_latents: torch.Tensor, expected_length: int) -> torch.Tensor:
    actual_length = int(audio_latents.shape[1])
    if actual_length == expected_length:
        return audio_latents.contiguous()
    if actual_length > expected_length:
        return audio_latents[:, :expected_length, :].contiguous()
    padding_length = expected_length - actual_length
    padding = torch.zeros(
        (audio_latents.shape[0], padding_length, audio_latents.shape[2]),
        device=audio_latents.device,
        dtype=audio_latents.dtype,
    )
    return torch.cat([audio_latents, padding], dim=1).contiguous()


def encode_and_save_audio_cache(
    encoder,
    processor,
    item_info: ItemInfo,
    *,
    audio_path: str,
    dtype: torch.dtype,
    target_fps: float = 25.0,
    audio_only: bool = False,
) -> None:
    try:
        import torchaudio
    except Exception as e:
        raise RuntimeError("torchaudio is required for LTX-2 audio latent caching") from e

    def _load_audio_with_pyav(path: str) -> tuple[torch.Tensor, int]:
        try:
            import av  # type: ignore
        except Exception as e:
            raise RuntimeError("PyAV is required to load audio from video containers when torchaudio can't") from e

        container = av.open(path)
        try:
            audio_stream = None
            for stream in container.streams:
                if stream.type == "audio":
                    audio_stream = stream
                    break
            if audio_stream is None:
                raise RuntimeError(f"No audio stream found in {path}")

            chunks: list[np.ndarray] = []
            sample_rate: Optional[int] = int(getattr(audio_stream, "rate", 0)) or None

            for frame in container.decode(audio=0):
                # PyAV can return packed interleaved 1D data for stereo;
                # normalize everything to [channels, samples].
                frame_channels = None
                try:
                    frame_channels = int(len(frame.layout.channels))  # type: ignore[arg-type]
                except Exception:
                    frame_channels = int(getattr(audio_stream, "channels", 0)) or None

                arr = coerce_decoded_audio_to_channels_first(frame.to_ndarray(), channels=frame_channels)

                if sample_rate is None:
                    sample_rate = int(getattr(frame, "sample_rate", 0)) or None

                chunks.append(arr)

            if not chunks:
                raise RuntimeError(f"No audio frames decoded from {path}")

            audio = np.concatenate(chunks, axis=1)
            if np.issubdtype(audio.dtype, np.integer):
                max_val = float(np.iinfo(audio.dtype).max)
                audio = audio.astype(np.float32) / max_val
            else:
                audio = audio.astype(np.float32)

            if sample_rate is None:
                raise RuntimeError(f"Could not determine sample rate for {path}")

            return torch.from_numpy(audio), int(sample_rate)
        finally:
            try:
                container.close()
            except Exception:
                pass

    normalized_audio_path = os.path.normpath(audio_path)
    if not os.path.exists(normalized_audio_path):
        raise FileNotFoundError(
            f"Audio file not found for {item_info.item_key}: {audio_path} (normalized: {normalized_audio_path})"
        )

    try:
        waveform, sample_rate = torchaudio.load(normalized_audio_path)
    except Exception as e:
        waveform, sample_rate = _load_audio_with_pyav(normalized_audio_path)

    if waveform.dim() != 2:
        raise ValueError(f"Unexpected waveform shape from {audio_path}: {tuple(waveform.shape)}")

    # Audio VAE encoder expects stereo (2ch). Normalize:
    # - mono -> duplicate
    # - >2ch (e.g. 5.1/6ch) -> downmix to mono then duplicate to stereo
    channels = int(waveform.shape[0])
    if channels == 1:
        waveform = waveform.repeat(2, 1)
    elif channels == 2:
        pass
    elif channels > 2:
        mono = waveform.float().mean(dim=0, keepdim=True)
        waveform = mono.repeat(2, 1)

    # If this ItemInfo represents a chunked clip (e.g. *_00000-017.mp4), slice audio to match the
    # chunk interval. We don't rely on container timestamps here; we use proportional slicing by
    # total decoded frames to keep it robust across backends.
    source_total_frames = getattr(item_info, "source_total_frames", None)
    chunk_start_frame = getattr(item_info, "chunk_start_frame", None)
    chunk_num_frames = getattr(item_info, "chunk_num_frames", None)
    if (
        isinstance(source_total_frames, int)
        and isinstance(chunk_start_frame, int)
        and isinstance(chunk_num_frames, int)
        and source_total_frames > 0
        and chunk_num_frames > 0
    ):
        total_samples = int(waveform.shape[-1])
        start = max(0, min(source_total_frames, chunk_start_frame))
        end = max(start, min(source_total_frames, chunk_start_frame + chunk_num_frames))
        start_sample = int(total_samples * (start / source_total_frames))
        end_sample = int(total_samples * (end / source_total_frames))
        if end_sample > start_sample:
            waveform = waveform[:, start_sample:end_sample]

    # Pitch-preserving time stretch when source audio duration doesn't match target video duration.
    # Skip for audio-only mode: frame_count is virtual (derived from the audio itself then adjusted
    # to N%8==1), so time-stretching would circularly compress the audio by 1-8%.
    frame_count = getattr(item_info, "frame_count", None)
    if not audio_only and isinstance(frame_count, int) and frame_count > 0 and waveform.shape[-1] > 0:
        expected_duration = float(frame_count) / max(float(target_fps), 1.0)
        actual_duration = float(waveform.shape[-1]) / float(sample_rate)
        if actual_duration > 0 and abs(actual_duration - expected_duration) / actual_duration > 0.01:
            from musubi_tuner.audio_utils import time_stretch_preserve_pitch

            target_samples = int(expected_duration * sample_rate)
            if target_samples > 0:
                waveform = time_stretch_preserve_pitch(waveform, sample_rate, target_samples)

    waveform = waveform.unsqueeze(0)
    encoder_param = next(encoder.parameters())
    device = encoder_param.device
    encoder_dtype = encoder_param.dtype

    # Compute mel in float32 (torchaudio STFT is most reliable in fp32), then cast to
    # the encoder dtype right before passing it into the encoder.
    waveform = waveform.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mel = processor.waveform_to_mel(waveform, int(sample_rate)).to(device=device, dtype=encoder_dtype)
        latents = encoder(mel)

    latents = latents[0].detach().cpu().contiguous()
    original_steps = int(latents.shape[1])
    align_audio = get_ltx2_env().align_audio_latents_cache
    if align_audio:
        expected_steps = _expected_audio_latent_length_for_item(
            item_info,
            encoder,
            fps=target_fps,
        )
        if expected_steps is not None:
            latents = _align_audio_latents_to_video(latents, expected_steps)
            effective_steps = min(original_steps, expected_steps)
        else:
            effective_steps = original_steps
    else:
        effective_steps = original_steps

    time_steps = int(latents.shape[1])
    mel_bins = int(latents.shape[2])
    channels = int(latents.shape[0])
    audio_loss_mask = None
    loss_mask_intervals = getattr(item_info, "audio_loss_mask_intervals", None)
    if loss_mask_intervals is not None:
        audio_loss_mask = torch.zeros((time_steps,), dtype=torch.float32)
        # Derive latents/sec from the encoder's actual output rather than from
        # sample_rate / mel_hop_length / LATENT_DOWNSAMPLE_FACTOR, which assumes
        # the encoder's internal rate matches the input file's sample_rate. With
        # input audio at any non-default SR, that formula misaligns interval
        # boundaries and silently drops intervals past time_steps.
        waveform_seconds = float(waveform.shape[-1]) / max(float(sample_rate), 1.0)
        if original_steps > 0 and waveform_seconds > 0:
            latents_per_second = float(original_steps) / waveform_seconds
        else:
            latents_per_second = float(sample_rate) / float(getattr(encoder, "mel_hop_length", 160)) / float(LATENT_DOWNSAMPLE_FACTOR)
        for start_s, end_s in loss_mask_intervals:
            start_idx = max(0, min(time_steps, int(math.floor(float(start_s) * latents_per_second))))
            end_idx = max(start_idx, min(time_steps, int(math.ceil(float(end_s) * latents_per_second))))
            if end_idx > start_idx:
                audio_loss_mask[start_idx:end_idx] = 1.0
        if effective_steps < time_steps:
            audio_loss_mask[int(effective_steps) :] = 0.0

    dtype_str = (
        cache_latents.dtype_to_str(dtype)
        if hasattr(cache_latents, "dtype_to_str")
        else ("fp16" if dtype == torch.float16 else "bf16" if dtype == torch.bfloat16 else "fp32")
    )

    audio_lengths = torch.tensor(effective_steps, dtype=torch.int32)
    int_dtype_str = cache_latents.dtype_to_str(audio_lengths.dtype) if hasattr(cache_latents, "dtype_to_str") else "int32"

    audio_cache_path = _audio_cache_path(item_info)
    os.makedirs(os.path.dirname(audio_cache_path), exist_ok=True)
    sd = {
        f"audio_latents_{time_steps}x{mel_bins}x{channels}_{dtype_str}": latents,
        f"audio_lengths_{int_dtype_str}": audio_lengths,
    }
    if audio_loss_mask is not None:
        sd["audio_loss_mask"] = audio_loss_mask

    metadata = {
        "architecture": "ltx2_v1",
        "format_version": "1.0.1",
    }

    save_file(sd, audio_cache_path, metadata=metadata)


def _find_reference_file(reference_directory: str, stem: str) -> Optional[str]:
    """Find a reference file matching the given stem in reference_directory."""
    all_exts = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
    for ext in all_exts:
        candidate = os.path.join(reference_directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    # Try case-insensitive match on lowercase stem
    lower_stem = stem.lower()
    try:
        for fname in os.listdir(reference_directory):
            name_no_ext, ext = os.path.splitext(fname)
            if name_no_ext.lower() == lower_stem and ext.lower() in [e.lower() for e in all_exts]:
                return os.path.join(reference_directory, fname)
    except OSError:
        pass
    return None


def _find_reference_audio_file(reference_audio_directory: str, stem: str, preferred_ext: Optional[str] = None) -> Optional[str]:
    """Find a reference audio file matching stem in reference_audio_directory."""
    exts: list[str] = []
    if preferred_ext:
        exts.append(preferred_ext)
    exts.extend(AUDIO_EXTENSIONS)

    deduped_exts: list[str] = []
    seen = set()
    for ext in exts:
        normalized = ext if ext.startswith(".") else f".{ext}"
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_exts.append(normalized)

    for ext in deduped_exts:
        candidate = os.path.join(reference_audio_directory, stem + ext)
        if os.path.exists(candidate):
            return candidate

    lower_stem = stem.lower()
    valid_exts = {ext.lower() for ext in deduped_exts}
    try:
        for fname in os.listdir(reference_audio_directory):
            name_no_ext, ext = os.path.splitext(fname)
            if name_no_ext.lower() == lower_stem and ext.lower() in valid_exts:
                return os.path.join(reference_audio_directory, fname)
    except OSError:
        pass
    return None


def _normalize_dataset_path_list(
    primary: Optional[str],
    extras: Optional[Sequence[str]],
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in ([primary] if primary is not None else []) + list(extras or []):
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def _load_reference_frames(
    path: str,
    bucket_reso: tuple[int, int],
    num_frames: int,
    downscale_factor: int = 1,
) -> np.ndarray:
    """Load reference frames as [F, H, W, 3] uint8, applying optional spatial downscale."""
    from PIL import Image
    import av

    if downscale_factor > 1:
        ref_w = max((bucket_reso[0] // downscale_factor // 32) * 32, 32)
        ref_h = max((bucket_reso[1] // downscale_factor // 32) * 32, 32)
        ref_reso = (ref_w, ref_h)
    else:
        ref_reso = bucket_reso

    ext = os.path.splitext(path)[1].lower()
    is_video = ext in [e.lower() for e in VIDEO_EXTENSIONS]

    if is_video:
        container = av.open(path)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i >= num_frames:
                break
            pil_frame = frame.to_image()
            arr = resize_image_to_bucket(pil_frame, ref_reso)
            frames.append(arr)
        container.close()
        if not frames:
            raise ValueError(f"No frames decoded from reference video: {path}")
    else:
        image = Image.open(path).convert("RGB")
        arr = resize_image_to_bucket(image, ref_reso)
        frames = [arr] * num_frames

    return np.stack(frames[:num_frames], axis=0)


def encode_and_save_reference_latents(
    vae,
    datasets: Sequence[BaseDataset],
    args: argparse.Namespace,
    device: torch.device,
    tiling_config=None,
) -> None:
    """Encode reference files and save as latent caches for IC-LoRA / v2v training."""
    default_num_frames = max(1, int(getattr(args, "reference_frames", 1) or 1))
    downscale_factor = max(1, getattr(args, "reference_downscale", 1))
    skip_existing = getattr(args, "skip_existing", False)
    num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) - 1)

    vae_param = next(vae.parameters())
    vae_dtype = vae_param.dtype

    for ds_idx, ds in enumerate(datasets):
        if not isinstance(ds, (ImageDataset, VideoDataset)):
            continue

        ref_cache_dirs = _normalize_dataset_path_list(
            getattr(ds, "reference_cache_directory", None),
            getattr(ds, "reference_cache_directories", None),
        )
        if not ref_cache_dirs:
            logger.info(f"[Dataset {ds_idx}] No reference_cache_directory set, skipping reference caching")
            continue

        ref_dirs = _normalize_dataset_path_list(
            getattr(ds, "reference_directory", None) or getattr(ds, "control_directory", None),
            getattr(ds, "reference_directories", None),
        )
        has_reference_dirs = getattr(ds, "reference_directory", None) is not None
        has_reference_dir_list = getattr(ds, "reference_directories", None) is not None
        has_control_dir = getattr(ds, "control_directory", None) is not None
        if has_reference_dirs or has_reference_dir_list:
            ref_dir_source = "reference_directory"
        elif has_control_dir:
            ref_dir_source = "control_directory"
        else:
            ref_dir_source = "control_directory"
        if not ref_dirs:
            logger.info(f"[Dataset {ds_idx}] No reference/control directory set, skipping reference caching")
            continue
        if len(ref_dirs) != len(ref_cache_dirs):
            raise ValueError(
                f"[Dataset {ds_idx}] reference directory count ({len(ref_dirs)}) must match "
                f"reference cache directory count ({len(ref_cache_dirs)})."
            )

        for ref_cache_dir in ref_cache_dirs:
            os.makedirs(ref_cache_dir, exist_ok=True)
        logger.info(
            "[Dataset %s] Caching %s reference latent stream(s) from %s to %s",
            ds_idx,
            len(ref_dirs),
            ref_dir_source,
            ref_cache_dirs,
        )
        dataset_reference_frames = getattr(ds, "reference_frames", None)
        if dataset_reference_frames is None:
            num_frames = default_num_frames
        else:
            num_frames = max(1, int(dataset_reference_frames or 1))
        logger.info(
            "[Dataset %s] Reference frames: %s%s",
            ds_idx,
            num_frames,
            " (dataset override)" if dataset_reference_frames is not None else " (CLI/default)",
        )

        cached_count = 0
        skipped_count = 0
        missing_count = 0

        for _bucket_key, batch in ds.retrieve_latent_cache_batches(num_workers):
            for item_info in batch:
                # For chunked videos, item_key is "video_00000-017.mp4"; use source video name for matching
                source_key = getattr(item_info, "source_item_key", None) or item_info.item_key
                stem = os.path.splitext(os.path.basename(source_key))[0]
                # bucket_size is (width, height, frame_count, ...); extract spatial dims
                bucket_reso = (item_info.bucket_size[0], item_info.bucket_size[1])

                try:
                    ref_cache_paths = getattr(item_info, "reference_latent_cache_paths", None) or ds.get_reference_latent_cache_paths(item_info)
                    for ref_idx, (ref_dir, ref_cache_path) in enumerate(zip(ref_dirs, ref_cache_paths)):
                        if skip_existing and os.path.exists(ref_cache_path):
                            skipped_count += 1
                            continue

                        ref_path = _find_reference_file(ref_dir, stem)
                        if ref_path is None:
                            missing_count += 1
                            if missing_count <= 5:
                                logger.warning(f"No reference file found for '{stem}' in {ref_dir}")
                            elif missing_count == 6:
                                logger.warning("(suppressing further missing-reference warnings)")
                            continue
                        ref_frames = _load_reference_frames(ref_path, bucket_reso, num_frames, downscale_factor)

                        contents = torch.from_numpy(ref_frames).unsqueeze(0)
                        contents = contents.permute(0, 4, 1, 2, 3).contiguous()
                        contents = contents.to(device=device, dtype=vae_dtype)
                        contents = contents / 127.5 - 1.0

                        frames = contents.shape[2]
                        remainder = (frames - 1) % 8
                        if remainder != 0:
                            pad = 8 - remainder
                            last = contents[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
                            contents = torch.cat([contents, last], dim=2)

                        with _amp_context(device, vae_dtype), torch.no_grad():
                            if tiling_config is not None and hasattr(vae, "tiled_encode"):
                                latent = vae.tiled_encode(contents, tiling_config)
                            else:
                                latent = vae(contents)
                            latent = latent.to(device=device, dtype=vae_dtype)

                        ref_latent = latent[0]
                        ref_item_info = ItemInfo(
                            item_info.item_key,
                            item_info.caption,
                            item_info.original_size,
                            item_info.bucket_size,
                        )
                        ref_item_info.latent_cache_path = ref_cache_path
                        ref_item_info.frame_count = num_frames
                        save_latent_cache_ltx2(ref_item_info, ref_latent)
                        cached_count += 1

                except Exception as e:
                    logger.warning(f"Failed to cache reference for '{stem}': {e}")
                    continue

        logger.info(
            f"[Dataset {ds_idx}] Reference caching done: {cached_count} cached, "
            f"{skipped_count} skipped (existing), {missing_count} missing"
        )


def encode_and_save_reference_audio_latents(
    encoder,
    processor,
    datasets: Sequence[BaseDataset],
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
) -> None:
    """Encode reference-audio files and save latent caches for audio_ref_only_ic training."""
    skip_existing = getattr(args, "skip_existing", False)
    num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) - 1)
    preferred_ext = getattr(args, "ltx2_audio_ext", None)

    for ds_idx, ds in enumerate(datasets):
        if not isinstance(ds, VideoDataset):
            continue

        ref_audio_cache_dirs = _normalize_dataset_path_list(
            getattr(ds, "reference_audio_cache_directory", None),
            getattr(ds, "reference_audio_cache_directories", None),
        )
        if not ref_audio_cache_dirs:
            continue

        ref_audio_dirs = _normalize_dataset_path_list(
            getattr(ds, "reference_audio_directory", None),
            getattr(ds, "reference_audio_directories", None),
        )
        if not ref_audio_dirs:
            logger.info(f"[Dataset {ds_idx}] No reference_audio_directory set, skipping reference-audio caching")
            continue
        if len(ref_audio_dirs) != len(ref_audio_cache_dirs):
            raise ValueError(
                f"[Dataset {ds_idx}] reference audio directory count ({len(ref_audio_dirs)}) must match "
                f"reference audio cache directory count ({len(ref_audio_cache_dirs)})."
            )

        for ref_audio_cache_dir in ref_audio_cache_dirs:
            os.makedirs(ref_audio_cache_dir, exist_ok=True)
        logger.info(
            "[Dataset %s] Caching %s reference audio stream(s) to %s",
            ds_idx,
            len(ref_audio_dirs),
            ref_audio_cache_dirs,
        )

        cached_count = 0
        skipped_count = 0
        missing_count = 0
        failed_count = 0
        ds_target_fps = getattr(ds, "target_fps", VideoDataset.TARGET_FPS_LTX2)
        if not isinstance(ds_target_fps, (int, float)) or float(ds_target_fps) <= 0:
            ds_target_fps = float(VideoDataset.TARGET_FPS_LTX2)

        for _bucket_key, batch in ds.retrieve_latent_cache_batches(num_workers):
            for item_info in batch:
                source_key = getattr(item_info, "source_item_key", None) or item_info.item_key
                stem = os.path.splitext(os.path.basename(source_key))[0]
                ref_audio_path = None
                try:
                    ref_audio_cache_paths = getattr(item_info, "reference_audio_latent_cache_paths", None) or ds.get_reference_audio_latent_cache_paths(item_info)
                    for ref_audio_dir, ref_audio_cache_path in zip(ref_audio_dirs, ref_audio_cache_paths):
                        if skip_existing and os.path.exists(ref_audio_cache_path):
                            skipped_count += 1
                            continue

                        ref_audio_path = _find_reference_audio_file(ref_audio_dir, stem, preferred_ext=preferred_ext)
                        if ref_audio_path is None:
                            missing_count += 1
                            if missing_count <= 5:
                                logger.warning(f"No reference audio file found for '{stem}' in {ref_audio_dir}")
                            elif missing_count == 6:
                                logger.warning("(suppressing further missing-reference-audio warnings)")
                            continue

                        cache_audio_suffix = f"_{ds.architecture}_audio.safetensors"
                        cache_latent_suffix = f"_{ds.architecture}.safetensors"
                        if ref_audio_cache_path.endswith(cache_audio_suffix):
                            proxy_latent_cache_path = ref_audio_cache_path[: -len(cache_audio_suffix)] + cache_latent_suffix
                        else:
                            proxy_latent_cache_path = ref_audio_cache_path.replace(".safetensors", cache_latent_suffix)

                        ref_item = ItemInfo(
                            item_key=item_info.item_key,
                            caption=item_info.caption,
                            original_size=item_info.original_size,
                            bucket_size=item_info.bucket_size,
                            frame_count=item_info.frame_count,
                        )
                        ref_item.latent_cache_path = proxy_latent_cache_path
                        ref_item.source_total_frames = getattr(item_info, "source_total_frames", None)
                        ref_item.chunk_start_frame = getattr(item_info, "chunk_start_frame", None)
                        ref_item.chunk_num_frames = getattr(item_info, "chunk_num_frames", None)

                        encode_and_save_audio_cache(
                            encoder,
                            processor,
                            ref_item,
                            audio_path=ref_audio_path,
                            dtype=dtype,
                            target_fps=float(ds_target_fps),
                            audio_only=False,
                        )
                        cached_count += 1
                except Exception as e:
                    failed_count += 1
                    logger.warning(
                        "Skipping reference audio cache for %s (audio_path=%s): %s",
                        item_info.item_key,
                        ref_audio_path,
                        e,
                    )
                    continue

        logger.info(
            f"[Dataset {ds_idx}] Reference-audio caching done: {cached_count} cached, "
            f"{failed_count} failed, {skipped_count} skipped (existing), {missing_count} missing"
        )


def _precache_sample_latents(args: argparse.Namespace, device: torch.device) -> None:
    """Cache I2V / V2V / reference-audio conditioning latents for sample prompts."""
    from musubi_tuner.hv_train_network import load_prompts
    from PIL import Image
    import torchvision.transforms.functional as TF
    import os

    if args.sample_prompts is None:
        raise ValueError("--sample_prompts is required when --precache_sample_latents is set")

    prompts = load_prompts(args.sample_prompts)
    if not prompts:
        raise ValueError(f"No prompts found in {args.sample_prompts}")

    prompts_with_refs = [
        (i, p)
        for i, p in enumerate(prompts)
        if p.get("image_path")
        or p.get("v2v_ref_path")
        or p.get("ref_audio_path")
        or p.get("reference_audio_path")
    ]
    if not prompts_with_refs:
        logger.info("No I2V/V2V/reference-audio entries found in sample prompts - nothing to cache")
        return

    i2v_count = sum(1 for _, p in prompts_with_refs if p.get("image_path"))
    v2v_count = sum(1 for _, p in prompts_with_refs if p.get("v2v_ref_path"))
    ref_audio_count = sum(1 for _, p in prompts_with_refs if p.get("ref_audio_path") or p.get("reference_audio_path"))
    logger.info(
        "Found %d I2V images, %d V2V references, %d reference-audio clips to precache",
        i2v_count,
        v2v_count,
        ref_audio_count,
    )

    need_video_encoder = i2v_count > 0 or v2v_count > 0
    need_audio_encoder = ref_audio_count > 0

    from musubi_tuner.ltx2_defaults import get_ltx2_sampling_preset
    _preset = get_ltx2_sampling_preset(
        getattr(args, "sample_sampling_preset", "defaults"),
        ltx_version=str(getattr(args, "ltx_version", "2.3")),
    )
    _default_w = int(getattr(args, "width", 768) or 768)
    _default_h = int(getattr(args, "height", 512) or 512)
    _default_f = int(getattr(args, "sample_num_frames", 45) or 45)
    if _preset is not None:
        _default_w = _preset.width
        _default_h = _preset.height
        _default_f = _preset.frame_count

    vae_encoder = None
    vae_dtype = torch.bfloat16
    spatial_factor = 32

    if need_video_encoder:
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

        vae_checkpoint = getattr(args, "vae", None) or getattr(args, "ltx2_checkpoint", None)
        if not vae_checkpoint:
            raise ValueError("VAE checkpoint required for I2V/V2V latent precaching (--vae or --ltx2_checkpoint)")

        logger.info("Loading VAE encoder for I2V/V2V precaching")
        vae_encoder = SingleGPUModelBuilder(
            model_path=str(vae_checkpoint),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=vae_dtype)
        vae_encoder.eval()

    audio_encoder = None
    audio_processor = None
    if need_audio_encoder:
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.audio_vae.model_configurator import (
            AudioEncoderConfigurator,
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        from musubi_tuner.ltx_2.model.audio_vae.ops import AudioProcessor

        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required for reference-audio latent precaching")

        audio_dtype = torch.float16 if args.ltx2_audio_dtype is None else str_to_dtype(args.ltx2_audio_dtype)
        logger.info("Loading audio encoder for reference-audio precaching")
        audio_encoder = SingleGPUModelBuilder(
            model_path=str(args.ltx2_checkpoint),
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=audio_dtype)
        audio_encoder.eval()

        audio_processor = AudioProcessor(
            sample_rate=int(getattr(audio_encoder, "sample_rate", 16000)),
            mel_bins=int(getattr(audio_encoder, "mel_bins", 64)),
            mel_hop_length=int(getattr(audio_encoder, "mel_hop_length", 160)),
            n_fft=int(getattr(audio_encoder, "n_fft", 1024)),
        ).to(device=device, dtype=torch.float32)
        audio_processor.eval()

    latent_cache: list[dict] = []

    def _cover_center_crop(pil_img, tw, th):
        cw, ch = pil_img.size
        if ch == th and cw == tw:
            return pil_img
        ar = cw / ch
        tar = tw / th
        if ar > tar:
            rh = th
            rw = max(tw, int(round(th * ar)))
        else:
            rw = tw
            rh = max(th, int(round(tw / ar)))
        pil_img = pil_img.resize((rw, rh), Image.LANCZOS)
        left = max((rw - tw) // 2, 0)
        top = max((rh - th) // 2, 0)
        return pil_img.crop((left, top, left + tw, top + th))

    def _encode_image_to_latent(img_path, target_width, target_height):
        if vae_encoder is None:
            raise RuntimeError("VAE encoder is not initialized")
        image = _cover_center_crop(Image.open(img_path).convert("RGB"), target_width, target_height)
        image_tensor = TF.to_tensor(image).unsqueeze(0)
        image_tensor = (image_tensor * 2.0 - 1.0).to(device=device, dtype=vae_dtype)
        image_tensor = image_tensor.unsqueeze(2)
        with torch.no_grad():
            return vae_encoder(image_tensor)

    def _encode_media_to_latent(media_path, target_width, target_height, max_frames=1):
        if vae_encoder is None:
            raise RuntimeError("VAE encoder is not initialized")
        import av as _av

        ext = os.path.splitext(media_path)[1].lower()
        frames = []
        if ext in {e.lower() for e in VIDEO_EXTENSIONS}:
            container = _av.open(media_path)
            for i, frame in enumerate(container.decode(video=0)):
                if i >= max_frames:
                    break
                frames.append(TF.to_tensor(_cover_center_crop(frame.to_image().convert("RGB"), target_width, target_height)))
            container.close()
            if not frames:
                raise ValueError(f"No frames decoded from video: {media_path}")
        else:
            image = _cover_center_crop(Image.open(media_path).convert("RGB"), target_width, target_height)
            frames.append(TF.to_tensor(image))

        video_tensor = torch.stack(frames, dim=0).unsqueeze(0)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4).contiguous()
        video_tensor = (video_tensor * 2.0 - 1.0).to(device=device, dtype=vae_dtype)

        num_f = video_tensor.shape[2]
        remainder = (num_f - 1) % 8
        if remainder != 0:
            pad = 8 - remainder
            video_tensor = torch.cat([video_tensor, video_tensor[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)

        with torch.no_grad():
            return vae_encoder(video_tensor)

    def _encode_reference_audio_to_latent(audio_path: str) -> torch.Tensor:
        if audio_encoder is None or audio_processor is None:
            raise RuntimeError("Audio encoder is not initialized")
        try:
            import torchaudio
        except Exception as e:
            raise RuntimeError("torchaudio is required for reference-audio precaching") from e

        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.dim() != 2:
            raise ValueError(f"Unexpected waveform shape from {audio_path}: {tuple(waveform.shape)}")
        channels = int(waveform.shape[0])
        if channels == 1:
            waveform = waveform.repeat(2, 1)
        elif channels == 2:
            pass
        elif channels > 2:
            mono = waveform.float().mean(dim=0, keepdim=True)
            waveform = mono.repeat(2, 1)

        waveform = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)
        encoder_dtype = next(audio_encoder.parameters()).dtype
        with torch.no_grad():
            mel = audio_processor.waveform_to_mel(waveform, int(sample_rate)).to(device=device, dtype=encoder_dtype)
            latents = audio_encoder(mel)
        return latents[0].detach().cpu().unsqueeze(0).contiguous()

    for idx, prompt_dict in prompts_with_refs:
        width = prompt_dict.get("width", _default_w)
        height = prompt_dict.get("height", _default_h)
        width = (width // spatial_factor) * spatial_factor
        height = (height // spatial_factor) * spatial_factor

        cache_entry = {
            "prompt_index": idx,
            "width": width,
            "height": height,
            "spatial_factor": spatial_factor,
        }

        image_path = prompt_dict.get("image_path")
        if image_path:
            try:
                if not os.path.exists(image_path):
                    logger.warning(f"I2V image not found, skipping prompt #{idx}: {image_path}")
                else:
                    logger.info(f"Encoding I2V image for prompt #{idx}: {os.path.basename(image_path)}")
                    latent = _encode_image_to_latent(image_path, width, height)
                    cache_entry["image_path"] = image_path
                    cache_entry["conditioning_latent"] = latent.cpu()
                    logger.info(f"Encoded I2V latent for prompt #{idx}: {latent.shape}")
            except Exception as e:
                logger.error(f"Failed to encode I2V image for prompt #{idx} '{image_path}': {e}")

        v2v_ref_path = prompt_dict.get("v2v_ref_path")
        if v2v_ref_path:
            try:
                if not os.path.exists(v2v_ref_path):
                    logger.warning(f"V2V reference not found, skipping prompt #{idx}: {v2v_ref_path}")
                else:
                    ref_downscale = max(1, getattr(args, "reference_downscale", 1))
                    if ref_downscale > 1:
                        ref_w = max((width // ref_downscale // 32) * 32, 32)
                        ref_h = max((height // ref_downscale // 32) * 32, 32)
                    else:
                        ref_w, ref_h = width, height
                    ref_frames = max(1, getattr(args, "reference_frames", 1))
                    latent = _encode_media_to_latent(v2v_ref_path, ref_w, ref_h, max_frames=ref_frames)
                    cache_entry["v2v_ref_path"] = v2v_ref_path
                    cache_entry["v2v_ref_latent"] = latent.cpu()
                    logger.info(f"Encoded V2V ref for prompt #{idx}: {ref_w}x{ref_h} → {latent.shape}")
            except Exception as e:
                logger.error(f"Failed to encode V2V reference for prompt #{idx} '{v2v_ref_path}': {e}")

        ref_audio_path = prompt_dict.get("ref_audio_path") or prompt_dict.get("reference_audio_path")
        if ref_audio_path:
            try:
                if not os.path.exists(ref_audio_path):
                    logger.warning(f"Reference audio not found, skipping prompt #{idx}: {ref_audio_path}")
                else:
                    logger.info(f"Encoding reference audio for prompt #{idx}: {os.path.basename(ref_audio_path)}")
                    latent = _encode_reference_audio_to_latent(ref_audio_path)
                    cache_entry["ref_audio_path"] = ref_audio_path
                    cache_entry["ref_audio_latent"] = latent
                    logger.info(f"Encoded reference audio latent for prompt #{idx}: {latent.shape}")
            except Exception as e:
                logger.error(f"Failed to encode reference audio for prompt #{idx} '{ref_audio_path}': {e}")

        if (
            "conditioning_latent" in cache_entry
            or "v2v_ref_latent" in cache_entry
            or "ref_audio_latent" in cache_entry
        ):
            latent_cache.append(cache_entry)

    if vae_encoder is not None:
        del vae_encoder
    if audio_encoder is not None:
        del audio_encoder
    if audio_processor is not None:
        del audio_processor
    from musubi_tuner.utils.device_utils import clean_memory_on_device
    clean_memory_on_device(device)

    if args.sample_latents_cache:
        cache_path = args.sample_latents_cache
    else:
        try:
            datasets = _load_datasets(args)
            if datasets:
                cache_dir = getattr(datasets[0], "cache_directory", None)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_path = os.path.join(cache_dir, "ltx2_sample_latents_cache.pt")
                else:
                    cache_path = "ltx2_sample_latents_cache.pt"
            else:
                cache_path = "ltx2_sample_latents_cache.pt"
        except Exception as e:
            logger.warning(f"Could not load datasets for cache path resolution: {e}")
            cache_path = "ltx2_sample_latents_cache.pt"

    payload = {
        "version": 1,
        "latent_cache": latent_cache,
    }
    torch.save(payload, cache_path)
    logger.info(
        "Saved %d sample conditioning latents (I2V/V2V/reference-audio) to %s",
        len(latent_cache),
        cache_path,
    )


def main() -> None:
    parser = cache_latents.setup_parser_common()
    parser = ltx2_setup_parser(parser)
    args = parser.parse_args()
    short_map = {"v": "video", "a": "audio", "va": "av"}
    if getattr(args, "ltx_mode", None) in short_map:
        args.ltx_mode = short_map[args.ltx_mode]

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handle I2V sample latent precaching if requested.
    # This is additive: continue with normal dataset latent caching afterward.
    if getattr(args, "precache_sample_latents", False):
        _precache_sample_latents(args, device)
        logger.info("Sample latent precaching (I2V/V2V/reference-audio) complete; continuing with dataset latent caching")

    datasets = _load_datasets(args)
    if args.save_dataset_manifest:
        user_config = config_utils.load_user_config(args.dataset_config)
        manifest = config_utils.create_cache_only_dataset_manifest(
            user_config,
            args,
            architecture=ARCHITECTURE_LTX2,
            source_dataset_config=args.dataset_config,
        )
        manifest_path = config_utils.save_dataset_manifest(manifest, args.save_dataset_manifest)
        logger.info("Saved cache-only dataset manifest: %s", manifest_path)

    if args.debug_mode is not None:
        cache_latents.show_datasets(list(datasets), args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    ltx_mode = getattr(args, "ltx_mode", "video")
    audio_only = ltx_mode == "audio"
    audio_video = ltx_mode == "av"

    # Auto-detect audio-only mode if all datasets are AudioDataset
    audio_datasets = [ds for ds in datasets if isinstance(ds, AudioDataset)]
    non_audio_datasets = [ds for ds in datasets if not isinstance(ds, AudioDataset)]
    if not audio_only and audio_datasets and not non_audio_datasets:
        logger.info("All datasets are audio-only; automatically switching to --ltx2_mode audio")
        audio_only = True


    if not audio_only:
        if args.vae is None:
            if getattr(args, "ltx2_checkpoint", None) is None:
                raise ValueError("--vae is required (or provide --ltx2_checkpoint for integrated checkpoints)")
            logger.info("--vae not provided; using --ltx2_checkpoint as VAE checkpoint")
            args.vae = args.ltx2_checkpoint

        vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

        vae = SingleGPUModelBuilder(
            model_path=str(args.vae),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=vae_dtype)
        vae.eval()
        if args.vae_chunk_size is not None:
            vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
            logger.info("Set chunk_size to %s for CausalConv3d in VAE", args.vae_chunk_size)

        from musubi_tuner.ltx_2.model.video_vae.tiling import TilingConfig, SpatialTilingConfig, TemporalTilingConfig

        spatial_config = None
        temporal_config = None

        if args.vae_spatial_tile_size is not None:
            logger.info("Enabling spatial tiling: size=%s, overlap=%s",
                        args.vae_spatial_tile_size, args.vae_spatial_tile_overlap)
            spatial_config = SpatialTilingConfig(
                tile_size_in_pixels=args.vae_spatial_tile_size,
                tile_overlap_in_pixels=args.vae_spatial_tile_overlap,
            )

        if args.vae_temporal_tile_size is not None:
            logger.info("Enabling temporal tiling: size=%s frames, overlap=%s frames",
                        args.vae_temporal_tile_size, args.vae_temporal_tile_overlap)
            temporal_config = TemporalTilingConfig(
                tile_size_in_frames=args.vae_temporal_tile_size,
                tile_overlap_in_frames=args.vae_temporal_tile_overlap,
            )

        tiling_config = None
        if spatial_config is not None or temporal_config is not None:
            tiling_config = TilingConfig(
                spatial_config=spatial_config,
                temporal_config=temporal_config,
            )

        def encode_fn(batch: List[ItemInfo]) -> None:
            encode_and_save_batch(vae, batch, tiling_config)

        # Only pass non-audio datasets to the video encoder; AudioDataset items have
        # content=None (no visual frames) and are handled separately below.
        cache_latents.encode_datasets(list(non_audio_datasets), encode_fn, args)

        # Cache reference latents for IC-LoRA / v2v training (auto-detected from TOML config)
        # Runs when any dataset has reference_cache_directory and a matching reference source directory.
        encode_and_save_reference_latents(vae, datasets, args, device, tiling_config)

    if audio_only:
        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required when --ltx2_mode audio is used")

        audio_dtype = torch.float16 if args.ltx2_audio_dtype is None else str_to_dtype(args.ltx2_audio_dtype)
        latent_dtype = audio_dtype if args.audio_video_latent_dtype is None else str_to_dtype(args.audio_video_latent_dtype)
        latent_channels = args.audio_video_latent_channels
        target_fps = float(getattr(args, "audio_only_target_fps", VideoDataset.TARGET_FPS_LTX2))
        target_resolution_override = getattr(args, "audio_only_target_resolution", None)
        if latent_channels is None:
            latent_channels = infer_video_in_channels_from_checkpoint(args.ltx2_checkpoint)
            if latent_channels is None:
                raise ValueError(
                    "Unable to infer video input channels from --ltx2_checkpoint; "
                    "set --audio_video_latent_channels explicitly."
                )
        latent_channels = int(latent_channels)
        if target_fps <= 0:
            raise ValueError(f"audio_only_target_fps must be > 0, got {target_fps}")

        # Validate datasets (use variables defined during auto-detection)
        if non_audio_datasets:
            raise ValueError("Audio-only caching only supports audio datasets in the dataset config")
        if not audio_datasets:
            raise ValueError("Audio-only caching requires at least one audio dataset")

        if target_resolution_override is not None:
            target_width = int(target_resolution_override)
            target_height = int(target_resolution_override)
        else:
            candidate_resolutions: list[tuple[int, int]] = []
            for ds in audio_datasets:
                res = getattr(ds, "resolution", None)
                if isinstance(res, (tuple, list)) and len(res) >= 2:
                    width = int(res[0])
                    height = int(res[1])
                    if width > 0 and height > 0:
                        candidate_resolutions.append((width, height))
            unique_resolutions = sorted(set(candidate_resolutions))
            if not unique_resolutions:
                target_width, target_height = config_utils.BaseDatasetParams.resolution
                logger.warning(
                    "Could not infer target resolution from audio datasets; "
                    "falling back to default dataset resolution %sx%s.",
                    target_width,
                    target_height,
                )
            elif len(unique_resolutions) > 1:
                target_width, target_height = max(unique_resolutions, key=lambda r: r[0] * r[1])
                logger.warning(
                    "Multiple audio dataset resolutions detected for audio-only caching %s; "
                    "using largest resolution %sx%s. Set --audio_only_target_resolution to override.",
                    unique_resolutions,
                    target_width,
                    target_height,
                )
            else:
                target_width, target_height = unique_resolutions[0]

        if target_width < 32 or target_height < 32:
            raise ValueError(
                f"Audio-only target resolution must be >= 32x32, got {target_width}x{target_height}."
            )
        audio_only_sequence_resolution = int(getattr(args, "audio_only_sequence_resolution", 64))
        if audio_only_sequence_resolution != 0 and audio_only_sequence_resolution < 32:
            raise ValueError(
                "audio_only_sequence_resolution must be 0 (use dataset/target resolution) "
                f"or >= 32, got {audio_only_sequence_resolution}."
            )

        def encode_audio_only_video_latents(batch: List[ItemInfo]) -> None:
            for item in batch:
                audio_path = getattr(item, "audio_path", None) or item.item_key
                latent, frame_count, virtual_geometry = build_audio_only_video_latent(
                    channels=latent_channels,
                    dtype=latent_dtype,
                    target_width=target_width,
                    target_height=target_height,
                    target_fps=target_fps,
                    audio_path=audio_path,
                    sequence_resolution=audio_only_sequence_resolution,
                )
                virtual_frames, virtual_height, virtual_width = virtual_geometry
                item.frame_count = frame_count
                save_latent_cache_ltx2(
                    item,
                    latent,
                    extra_tensors={
                        "ltx2_virtual_num_frames_int32": torch.tensor(int(virtual_frames), dtype=torch.int32),
                        "ltx2_virtual_height_int32": torch.tensor(int(virtual_height), dtype=torch.int32),
                        "ltx2_virtual_width_int32": torch.tensor(int(virtual_width), dtype=torch.int32),
                    },
                )

        cache_latents.encode_datasets(list(audio_datasets), encode_audio_only_video_latents, args)
    if audio_video or audio_only:
        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required when audio latents are cached")

        audio_dtype = torch.float16 if args.ltx2_audio_dtype is None else str_to_dtype(args.ltx2_audio_dtype)
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.audio_vae.model_configurator import (
            AudioEncoderConfigurator,
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        from musubi_tuner.ltx_2.model.audio_vae.ops import AudioProcessor

        encoder = SingleGPUModelBuilder(
            model_path=str(args.ltx2_checkpoint),
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=audio_dtype)
        encoder.eval()

        processor = AudioProcessor(
            sample_rate=int(getattr(encoder, "sample_rate", 16000)),
            mel_bins=int(getattr(encoder, "mel_bins", 64)),
            mel_hop_length=int(getattr(encoder, "mel_hop_length", 160)),
            n_fft=int(getattr(encoder, "n_fft", 1024)),
        ).to(device=device, dtype=torch.float32)
        processor.eval()

        audio_encoded_count = 0
        audio_failed_count = 0
        audio_skipped_count = 0

        for ds in datasets:
            if not isinstance(ds, (VideoDataset, AudioDataset)):
                continue
            if audio_only:
                ds_target_fps = target_fps
            else:
                ds_target_fps = getattr(ds, "target_fps", VideoDataset.TARGET_FPS_LTX2)
                if not isinstance(ds_target_fps, (int, float)) or float(ds_target_fps) <= 0:
                    ds_target_fps = float(VideoDataset.TARGET_FPS_LTX2)
            num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) - 1)
            for _bucket_key, batch in ds.retrieve_latent_cache_batches(num_workers):
                for item_info in batch:
                    audio_cache_path = _audio_cache_path(item_info)
                    if args.skip_existing and os.path.exists(audio_cache_path):
                        audio_skipped_count += 1
                        continue
                    if isinstance(ds, AudioDataset):
                        audio_path = getattr(item_info, "audio_path", None) or item_info.item_key
                        if audio_only:
                            frame_count = _estimate_video_frame_count_from_audio(audio_path, target_fps=ds_target_fps)
                            if frame_count is None:
                                raise ValueError(
                                    f"Could not determine audio duration for {audio_path}. "
                                    "Audio-only mode requires duration-aware frame_count metadata."
                                )
                            item_info.frame_count = int(frame_count)
                    else:
                        audio_path = _resolve_audio_path(
                            item_info,
                            source=args.ltx2_audio_source,
                            audio_dir=args.ltx2_audio_dir,
                            audio_ext=args.ltx2_audio_ext,
                        )
                    try:
                        encode_and_save_audio_cache(
                            encoder,
                            processor,
                            item_info,
                            audio_path=audio_path,
                            dtype=audio_dtype,
                            target_fps=ds_target_fps,
                            audio_only=audio_only,
                        )
                        audio_encoded_count += 1
                    except Exception as e:
                        audio_failed_count += 1
                        logger.warning(
                            "Skipping audio cache for %s (audio_path=%s): %s",
                            item_info.item_key,
                            audio_path,
                            e,
                        )
                        continue

        logger.info(
            "Audio latent caching complete: %d encoded, %d failed, %d skipped (existing)",
            audio_encoded_count,
            audio_failed_count,
            audio_skipped_count,
        )

        encode_and_save_reference_audio_latents(
            encoder,
            processor,
            datasets,
            args,
            dtype=audio_dtype,
        )


def ltx2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--ltx2_mode", "--ltx_mode",
        dest="ltx_mode",
        type=str,
        default="v",
        choices=["video", "av", "audio", "v", "a", "va"],
        help="Caching modality: 'video' (default) for video-only, 'av' for audio+video, 'audio' for audio-only.",
    )
    parser.add_argument(
        "--ltx2_checkpoint",
        type=str,
        default=default_ltx2_checkpoint_path(),
        help="Path to LTX-2 checkpoint (.safetensors)",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument("--vae_spatial_tile_size", type=int, default=None, help="Spatial tile size in pixels (e.g. 512). Must be >= 64 and divisible by 32.")
    parser.add_argument("--vae_spatial_tile_overlap", type=int, default=64, help="Spatial tile overlap in pixels (default 64). Must be divisible by 32.")
    parser.add_argument("--vae_temporal_tile_size", type=int, default=None, help="Temporal tile size in frames (e.g. 64). Must be >= 16 and divisible by 8.")
    parser.add_argument("--vae_temporal_tile_overlap", type=int, default=24, help="Temporal tile overlap in frames (default 24). Must be divisible by 8.")
    parser.add_argument(
        "--ltx2_audio_source",
        type=str,
        default="video",
        choices=["video", "audio_files"],
        help="Audio source for caching when --ltx2_mode is 'av' or 'audio'.",
    )
    parser.add_argument(
        "--ltx2_audio_dir",
        type=str,
        default=None,
        help="Directory containing audio files when --ltx2_audio_source=audio_files (optional)",
    )
    parser.add_argument(
        "--ltx2_audio_ext",
        type=str,
        default=".wav",
        help="Audio file extension when --ltx2_audio_source=audio_files (default: .wav)",
    )
    parser.add_argument("--ltx2_audio_dtype", type=str, default=None)
    parser.add_argument(
        "--audio_video_latent_channels",
        type=int,
        default=None,
        help="Override video latent channels for audio-only caching (auto-detected by default).",
    )
    parser.add_argument("--audio_video_latent_dtype", type=str, default=None)
    parser.add_argument(
        "--audio_only_target_resolution",
        type=int,
        default=None,
        help=(
            "Optional override (square) for target video resolution used to build "
            "duration-aware video latents in audio-only mode. By default, dataset "
            "resolution is used."
        ),
    )
    parser.add_argument(
        "--audio_only_target_fps",
        type=float,
        default=VideoDataset.TARGET_FPS_LTX2,
        help="Target FPS used to convert audio duration into video frame count in audio-only mode.",
    )
    parser.add_argument(
        "--audio_only_sequence_resolution",
        type=int,
        default=64,
        help=(
            "Virtual pixel resolution used to derive audio-only sequence length for timestep sampling. "
            "Use 0 to fall back to dataset/target resolution behavior."
        ),
    )
    parser.add_argument(
        "--precache_sample_latents",
        action="store_true",
        help="Cache I2V/V2V/reference-audio conditioning latents for sample prompts, then continue normal dataset latent caching.",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="Sample prompts file (for --precache_sample_latents).",
    )
    parser.add_argument(
        "--sample_latents_cache",
        type=str,
        default=None,
        help="Path to save sample conditioning latents cache (default: cache_dir/ltx2_sample_latents_cache.pt).",
    )
    parser.add_argument(
        "--reference_frames",
        type=int,
        default=1,
        help="Number of frames to extract from reference videos (default 1). Images are repeated to fill this count.",
    )
    parser.add_argument(
        "--reference_downscale",
        type=int,
        default=1,
        help="Spatial downscale factor for references (1=same res, 2=half). Must be >= 1.",
    )
    parser.add_argument(
        "--ltx_version",
        type=str,
        default="2.3",
        choices=["2.0", "2.3"],
        help="LTX model version (used to resolve sampling-preset geometry for --precache_sample_latents).",
    )
    parser.add_argument(
        "--sample_sampling_preset",
        "--sampling_preset",
        type=str,
        default="defaults",
        choices=["legacy", "defaults", "ltx20", "ltx23", "ltx23_hq", "distilled_two_stage"],
        help="Sampling preset used to resolve fallback geometry when sample prompts omit --w/--h.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Legacy fallback sample height when sample prompts omit --h and no sample preset is selected.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Legacy fallback sample width when sample prompts omit --w and no sample preset is selected.",
    )
    parser.add_argument(
        "--sample_num_frames",
        type=int,
        default=45,
        help="Legacy fallback sample frame count when sample prompts omit --f and no sample preset is selected.",
    )
    return parser


if __name__ == "__main__":
    main()
