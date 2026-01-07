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
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_LTX2,
    BaseDataset,
    ItemInfo,
    VideoDataset,
    save_latent_cache_ltx2,
)
from musubi_tuner.utils.model_utils import str_to_dtype


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
    return cast(Sequence[BaseDataset], dataset_group.datasets)


def encode_and_save_batch(vae, batch: List[ItemInfo]) -> None:
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
        latents = vae(contents)
        latents = latents.to(device=device, dtype=vae_dtype)

    for idx, item in enumerate(batch):
        save_latent_cache_ltx2(item, latents[idx])


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


def encode_and_save_audio_cache(
    encoder,
    processor,
    item_info: ItemInfo,
    *,
    audio_path: str,
    dtype: torch.dtype,
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
                # Typically returns shape [samples, channels] or [channels, samples]
                arr = frame.to_ndarray()
                if arr.ndim == 1:
                    arr = arr[None, :]
                elif arr.shape[0] > arr.shape[1]:
                    # assume [samples, channels]
                    arr = arr.T

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
    time_steps = latents.shape[1]
    mel_bins = latents.shape[2]
    channels = latents.shape[0]

    dtype_str = (
        cache_latents.dtype_to_str(dtype)
        if hasattr(cache_latents, "dtype_to_str")
        else ("fp16" if dtype == torch.float16 else "bf16" if dtype == torch.bfloat16 else "fp32")
    )

    audio_lengths = torch.tensor(time_steps, dtype=torch.int32)
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


def main() -> None:
    parser = cache_latents.setup_parser_common()
    parser = ltx2_setup_parser(parser)
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = _load_datasets(args)

    if args.debug_mode is not None:
        cache_latents.show_datasets(list(datasets), args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

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

    def encode_fn(batch: List[ItemInfo]) -> None:
        encode_and_save_batch(vae, batch)

    cache_latents.encode_datasets(list(datasets), encode_fn, args)

    audio_video = getattr(args, "ltx_mode", "video") == "av" or getattr(args, "ltx2_audio_video", False)
    if audio_video:
        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required when --ltx_mode av is used")

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
            if not isinstance(ds, VideoDataset):
                continue
            num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) - 1)
            for _bucket_key, batch in ds.retrieve_latent_cache_batches(num_workers):
                for item_info in batch:
                    audio_cache_path = _audio_cache_path(item_info)
                    if args.skip_existing and os.path.exists(audio_cache_path):
                        continue
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
        "--ltx_mode",
        type=str,
        default="video",
        choices=["video", "av", "audio"],
        help="Caching modality. Use 'av' to also cache audio latents.",
    )
    parser.add_argument(
        "--ltx2_audio_video",
        action="store_true",
        help="Enable audio-video caching. Video latents are the same; audio latents are cached separately.",
    )
    parser.add_argument("--ltx2_checkpoint", type=str, default=None, help="Path to LTX-2 checkpoint (.safetensors)")
    parser.add_argument(
        "--ltx2_audio_source",
        type=str,
        default="video",
        choices=["video", "audio_files"],
        help="Audio source for caching when --ltx2_audio_video is set.",
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
    return parser


if __name__ == "__main__":
    main()
