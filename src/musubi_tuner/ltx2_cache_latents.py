#!/usr/bin/env python3
"""
Cache video latents for LTX-2 training.

Uses the standard musubi-tuner dataset config so cached files match the trainer.
"""

from __future__ import annotations

import argparse
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
    BaseDataset,
    ItemInfo,
    AudioDataset,
    VideoDataset,
    save_latent_cache_ltx2,
)
from musubi_tuner.ltx_2.model.audio_vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR
from musubi_tuner.ltx_2.env import get_ltx2_env
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
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
            validation_blueprint.dataset_group
        )
        datasets.extend(validation_dataset_group.datasets)

    return cast(Sequence[BaseDataset], datasets)


def encode_and_save_batch(vae, batch: List[ItemInfo], tiling_config=None) -> None:
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if contents.ndim == 4:
        contents = contents.unsqueeze(1)

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
        save_latent_cache_ltx2(item, latents[idx])


def save_dummy_latent_cache_ltx2(item: ItemInfo, *, channels: int, dtype: torch.dtype) -> None:
    latent = torch.zeros((channels, 1, 1, 1), dtype=dtype)
    save_latent_cache_ltx2(item, latent)


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

    # Pitch-preserving time stretch when source audio duration doesn't match target video duration
    frame_count = getattr(item_info, "frame_count", None)
    if isinstance(frame_count, int) and frame_count > 0 and waveform.shape[-1] > 0:
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

    metadata = {
        "architecture": "ltx2_v1",
        "format_version": "1.0.1",
    }

    save_file(sd, audio_cache_path, metadata=metadata)


def _precache_sample_latents(args: argparse.Namespace, device: torch.device) -> None:
    """Cache I2V conditioning image latents for sample prompts."""
    from musubi_tuner.hv_train_network import load_prompts
    from PIL import Image
    import torchvision.transforms.functional as TF
    import os

    if args.sample_prompts is None:
        raise ValueError("--sample_prompts is required when --precache_sample_latents is set")

    prompts = load_prompts(args.sample_prompts)
    if not prompts:
        raise ValueError(f"No prompts found in {args.sample_prompts}")

    # Filter prompts that have image_path
    prompts_with_images = [(i, p) for i, p in enumerate(prompts) if p.get("image_path")]
    if not prompts_with_images:
        logger.info("No I2V images found in sample prompts - nothing to cache")
        return

    logger.info(f"Found {len(prompts_with_images)} prompts with I2V images")

    # Load VAE encoder
    from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from musubi_tuner.ltx_2.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

    vae_checkpoint = getattr(args, "vae", None) or getattr(args, "ltx2_checkpoint", None)
    if not vae_checkpoint:
        raise ValueError("VAE checkpoint required for I2V latent precaching (--vae or --ltx2_checkpoint)")

    vae_dtype = torch.bfloat16  # Standard for LTX-2
    logger.info("Loading VAE encoder for I2V image precaching")
    vae_encoder = SingleGPUModelBuilder(
        model_path=str(vae_checkpoint),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=vae_dtype)
    vae_encoder.eval()

    # Cache latents
    latent_cache: list[dict] = []
    spatial_factor = 32  # LTX-2 VAE spatial downsample factor

    for idx, prompt_dict in prompts_with_images:
        image_path = prompt_dict["image_path"]
        try:
            if not os.path.exists(image_path):
                logger.warning(f"I2V image not found, skipping prompt #{idx}: {image_path}")
                continue

            # Get dimensions from prompt or use defaults
            width = prompt_dict.get("width", 768)
            height = prompt_dict.get("height", 512)
            width = (width // spatial_factor) * spatial_factor
            height = (height // spatial_factor) * spatial_factor

            # Load and encode image
            logger.info(f"Encoding I2V image for prompt #{idx}: {os.path.basename(image_path)}")
            image = Image.open(image_path).convert("RGB")

            # Match official LTX-2 image-conditioning preprocessing:
            # resize-to-cover while preserving aspect ratio, then center-crop.
            current_width, current_height = image.size
            if current_height != height or current_width != width:
                aspect_ratio = current_width / current_height
                target_aspect_ratio = width / height

                if aspect_ratio > target_aspect_ratio:
                    resize_height = height
                    resize_width = max(width, int(round(height * aspect_ratio)))
                else:
                    resize_width = width
                    resize_height = max(height, int(round(width / aspect_ratio)))

                image = image.resize((resize_width, resize_height), Image.LANCZOS)
                left = max((resize_width - width) // 2, 0)
                top = max((resize_height - height) // 2, 0)
                image = image.crop((left, top, left + width, top + height))

            image_tensor = TF.to_tensor(image).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = (image_tensor * 2.0 - 1.0).to(device=device, dtype=vae_dtype)
            image_tensor = image_tensor.unsqueeze(2)  # [1, 3, 1, H, W]

            with torch.no_grad():
                conditioning_latent = vae_encoder(image_tensor)

            latent_cache.append({
                "prompt_index": idx,
                "image_path": image_path,
                "conditioning_latent": conditioning_latent.cpu(),
            })
            logger.info(f"Encoded I2V latent for prompt #{idx}: {conditioning_latent.shape}")

        except Exception as e:
            logger.error(f"Failed to encode I2V image for prompt #{idx} '{image_path}': {e}")

    # Clean up VAE encoder
    del vae_encoder
    from musubi_tuner.utils.device_utils import clean_memory_on_device
    clean_memory_on_device(device)
    logger.info("VAE encoder cleaned up")

    # Determine cache path
    if args.sample_latents_cache:
        cache_path = args.sample_latents_cache
    else:
        # Default: use first dataset's cache directory
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

    # Save cache
    payload = {
        "version": 1,
        "latent_cache": latent_cache,
    }
    torch.save(payload, cache_path)
    logger.info(f"Saved {len(latent_cache)} I2V conditioning latents to {cache_path}")


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
        logger.info("I2V sample latent precaching complete; continuing with dataset latent caching")

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

        from musubi_tuner.ltx_2.model.video_vae.tiling import TilingConfig, SpatialTilingConfig

        tiling_config = None
        if args.vae_spatial_tile_size is not None:
             logger.info("Enabling spatial tiling: size=%s, overlap=%s", 
                         args.vae_spatial_tile_size, args.vae_spatial_tile_overlap)
             tiling_config = TilingConfig(
                 spatial_config=SpatialTilingConfig(
                     tile_size_in_pixels=args.vae_spatial_tile_size,
                     tile_overlap_in_pixels=args.vae_spatial_tile_overlap
                 )
             )

        def encode_fn(batch: List[ItemInfo]) -> None:
            encode_and_save_batch(vae, batch, tiling_config)

        cache_latents.encode_datasets(list(datasets), encode_fn, args)

    if audio_only:
        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required when --ltx2_mode audio is used")

        audio_dtype = torch.float16 if args.ltx2_audio_dtype is None else str_to_dtype(args.ltx2_audio_dtype)
        dummy_dtype = audio_dtype if args.audio_dummy_video_dtype is None else str_to_dtype(args.audio_dummy_video_dtype)
        dummy_channels = args.audio_dummy_video_channels
        if dummy_channels is None:
            dummy_channels = infer_video_in_channels_from_checkpoint(args.ltx2_checkpoint)
            if dummy_channels is None:
                raise ValueError(
                    "Unable to infer video input channels from --ltx2_checkpoint; "
                    "set --audio_dummy_video_channels explicitly."
                )
        dummy_channels = int(dummy_channels)

        # Validate datasets (use variables defined during auto-detection)
        if non_audio_datasets:
            raise ValueError("Audio-only caching only supports audio datasets in the dataset config")
        if not audio_datasets:
            raise ValueError("Audio-only caching requires at least one audio dataset")

        def encode_dummy(batch: List[ItemInfo]) -> None:
            for item in batch:
                save_dummy_latent_cache_ltx2(item, channels=dummy_channels, dtype=dummy_dtype)

        cache_latents.encode_datasets(list(audio_datasets), encode_dummy, args)
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

        for ds in datasets:
            if not isinstance(ds, (VideoDataset, AudioDataset)):
                continue
            ds_target_fps = getattr(ds, "target_fps", VideoDataset.TARGET_FPS_LTX2)
            num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) - 1)
            for _bucket_key, batch in ds.retrieve_latent_cache_batches(num_workers):
                for item_info in batch:
                    audio_cache_path = _audio_cache_path(item_info)
                    if args.skip_existing and os.path.exists(audio_cache_path):
                        continue
                    if isinstance(ds, AudioDataset):
                        audio_path = getattr(item_info, "audio_path", None) or item_info.item_key
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
                            target_fps=float(ds_target_fps),
                        )
                    except Exception as e:
                        logger.warning(
                            "Skipping audio cache for %s (audio_path=%s): %s",
                            item_info.item_key,
                            audio_path,
                            e,
                        )
                        continue


def ltx2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--ltx2_mode", "--ltx_mode",
        dest="ltx_mode",
        type=str,
        default="v",
        choices=["video", "av", "audio", "v", "a", "va"],
        help="Caching modality: 'video' (default) for video-only, 'av' for audio+video, 'audio' for audio-only.",
    )
    parser.add_argument("--ltx2_checkpoint", type=str, default=None, help="Path to LTX-2 checkpoint (.safetensors)")
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument("--vae_spatial_tile_size", type=int, default=None, help="Spatial tile size in pixels (e.g. 512)")
    parser.add_argument("--vae_spatial_tile_overlap", type=int, default=64, help="Spatial tile overlap in pixels (default 64)")
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
        "--audio_dummy_video_channels",
        type=int,
        default=None,
        help="Override dummy video channels for audio-only caching (auto-detected by default).",
    )
    parser.add_argument("--audio_dummy_video_dtype", type=str, default=None)
    parser.add_argument(
        "--precache_sample_latents",
        action="store_true",
        help="Cache I2V conditioning image latents for sample prompts, then continue normal dataset latent caching.",
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
        help="Path to save I2V conditioning latents cache (default: cache_dir/ltx2_sample_latents_cache.pt).",
    )
    return parser


if __name__ == "__main__":
    main()
