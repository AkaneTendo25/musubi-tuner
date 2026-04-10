import argparse
import logging
import math
import os
import subprocess
from typing import Optional

import numpy as np
import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_MAGIHUMAN,
    ItemInfo,
    save_latent_cache_magihuman,
)
from musubi_tuner.magihuman.model.sa_audio import SAAudioFeatureExtractor
from musubi_tuner.magihuman.model.vae2_2 import get_vae2_2
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_TARGET_FPS = 25.0
AUDIO_CHUNK_DURATION_SECONDS = 29.0
AUDIO_CHUNK_OVERLAP_RATIO = 0.5


def get_magihuman_vae_dtype(vae) -> torch.dtype:
    inner_vae = getattr(vae, "vae", None)
    if inner_vae is not None:
        first_param = next(inner_vae.parameters(), None)
        if first_param is not None:
            return first_param.dtype

    if hasattr(vae, "dtype") and isinstance(vae.dtype, torch.dtype):
        return vae.dtype

    return torch.float32


def magihuman_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--audio_model", type=str, required=True, help="Path to the MagiHuman audio model directory.")
    parser.add_argument(
        "--target_fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="Fallback FPS used to align audio when dataset items do not carry source_fps metadata.",
    )
    parser.add_argument(
        "--allow_missing_audio",
        action="store_true",
        help="If set, missing audio streams are replaced with silence instead of failing.",
    )
    return parser


def resolve_vae_path(args: argparse.Namespace) -> str:
    candidate = args.vae
    if not candidate:
        raise ValueError("VAE path is empty. Pass --vae explicitly.")

    if os.path.isdir(candidate):
        default_file = os.path.join(candidate, "Wan2.2_VAE.pth")
        if os.path.exists(default_file):
            return default_file

    if not os.path.exists(candidate):
        raise FileNotFoundError(f"MagiHuman VAE path does not exist: {candidate}")
    return candidate


def resolve_audio_model_path(args: argparse.Namespace) -> str:
    candidate = args.audio_model
    if not candidate:
        raise ValueError("Audio model path is empty. Pass --audio_model explicitly.")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"MagiHuman audio model path does not exist: {candidate}")
    return candidate


def extract_audio_segment(
    source_path: str,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-ss",
        f"{max(0.0, start_seconds):.6f}",
        "-t",
        f"{max(0.0, duration_seconds):.6f}",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)


def merge_overlapping_features(encoded_chunks: list[torch.Tensor], overlap_ratio: float) -> torch.Tensor:
    if len(encoded_chunks) == 1:
        return encoded_chunks[0]

    batch_size, total_frames, feature_dim = encoded_chunks[0].shape
    overlap_frames = int(total_frames * overlap_ratio)
    step_frames = total_frames - overlap_frames
    final_length = (len(encoded_chunks) - 1) * step_frames + total_frames

    output = torch.zeros(
        batch_size,
        final_length,
        feature_dim,
        device=encoded_chunks[0].device,
        dtype=encoded_chunks[0].dtype,
    )

    for block_idx, current_feat in enumerate(encoded_chunks):
        output_start = block_idx * step_frames
        if block_idx == 0:
            output[:, output_start : output_start + total_frames, :] = current_feat
            continue

        non_overlap_start = output_start + overlap_frames
        non_overlap_end = output_start + total_frames
        output[:, non_overlap_start:non_overlap_end, :] = current_feat[:, overlap_frames:, :]

        for frame_idx in range(overlap_frames):
            output_pos = output_start + frame_idx
            prev_weight = (overlap_frames - frame_idx) / overlap_frames
            curr_weight = frame_idx / overlap_frames
            output[:, output_pos, :] = (
                prev_weight * output[:, output_pos, :] + curr_weight * current_feat[:, frame_idx, :]
            )

    return output


def encode_audio_segment(
    audio_vae,
    audio_np: np.ndarray,
    device: torch.device,
    sample_rate: int,
    downsampling_ratio: int,
) -> torch.Tensor:
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim != 1:
        raise ValueError(f"Expected mono audio array, got shape {audio_np.shape}")

    total_samples = audio_np.shape[0]
    window_size = int(AUDIO_CHUNK_DURATION_SECONDS * sample_rate)
    step_size = int(window_size * (1.0 - AUDIO_CHUNK_OVERLAP_RATIO))

    if total_samples <= window_size:
        audio_tensor = torch.from_numpy(audio_np).to(device=device)
        audio_tensor = audio_tensor.unsqueeze(0).expand(2, -1)
        encoded = audio_vae.vae_model.encode(audio_tensor)
        return encoded.permute(0, 2, 1)[0].detach().cpu()

    encoded_chunks = []
    for offset_start in range(0, total_samples, step_size):
        offset_end = min(offset_start + window_size, total_samples)
        chunk = np.zeros(window_size, dtype=np.float32)
        chunk[: offset_end - offset_start] = audio_np[offset_start:offset_end]

        chunk_tensor = torch.from_numpy(chunk).to(device=device)
        chunk_tensor = chunk_tensor.unsqueeze(0).expand(2, -1)
        encoded_chunk = audio_vae.vae_model.encode(chunk_tensor)

        encoded_chunks.append(encoded_chunk.permute(0, 2, 1))
        if offset_end >= total_samples:
            break

    merged = merge_overlapping_features(encoded_chunks, overlap_ratio=AUDIO_CHUNK_OVERLAP_RATIO).permute(0, 2, 1)
    expected_target_len = math.ceil(total_samples / downsampling_ratio)
    final_target_len = expected_target_len
    return merged[:, :, :final_target_len].permute(0, 2, 1)[0].detach().cpu()


def encode_and_save_batch(vae, audio_vae, batch: list[ItemInfo], args: argparse.Namespace, device: torch.device):
    audio_sample_rate = int(audio_vae.sample_rate)
    audio_downsampling_ratio = int(audio_vae.downsampling_ratio)
    vae_dtype = get_magihuman_vae_dtype(vae)

    contents = torch.stack([torch.from_numpy(item.content) for item in batch])  # B, F, H, W, C
    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(device=device, dtype=vae_dtype)
    contents = contents / 127.5 - 1.0

    with torch.no_grad():
        video_latents = vae.encode(contents).to(torch.float32)
        image_latents = vae.encode(contents[:, :, :1, :, :]).to(torch.float32)

    for index, item in enumerate(batch):
        if not item.source_path or not os.path.exists(item.source_path):
            raise FileNotFoundError(
                f"MagiHuman latent caching requires a real source_path on each item. Missing for {item.item_key}: {item.source_path}"
            )

        source_fps = item.source_fps if item.source_fps is not None else args.target_fps
        start_frame = item.source_start_frame if item.source_start_frame is not None else 0
        frame_count = item.frame_count if item.frame_count is not None else item.content.shape[0]

        start_seconds = start_frame / source_fps
        duration_seconds = max(0.0, frame_count / source_fps)
        if duration_seconds <= 0.0:
            duration_seconds = 1.0 / source_fps

        try:
            audio_np = extract_audio_segment(item.source_path, start_seconds, duration_seconds, sample_rate=audio_sample_rate)
        except Exception:
            if not args.allow_missing_audio:
                raise
            sample_count = max(1, math.ceil(duration_seconds * audio_sample_rate))
            audio_np = np.zeros(sample_count, dtype=np.float32)
            logger.warning("Missing audio for %s, replaced with silence", item.source_path)

        audio_latent = encode_audio_segment(
            audio_vae,
            audio_np,
            device=device,
            sample_rate=audio_sample_rate,
            downsampling_ratio=audio_downsampling_ratio,
        )
        save_latent_cache_magihuman(
            item_info=item,
            video_latent=video_latents[index].detach().cpu(),
            audio_latent=audio_latent,
            image_latent=image_latents[index].detach().cpu(),
        )


def main():
    parser = cache_latents.setup_parser_common()
    parser = magihuman_setup_parser(parser)
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MAGIHUMAN)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=int(args.target_fps)
        )
        return

    vae_path = resolve_vae_path(args)
    audio_model_path = resolve_audio_model_path(args)
    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)

    logger.info(f"Loading MagiHuman video VAE from {vae_path}")
    vae = get_vae2_2(vae_path, device=str(device), weight_dtype=vae_dtype)

    logger.info(f"Loading MagiHuman audio model from {audio_model_path}")
    audio_vae = SAAudioFeatureExtractor(device=str(device), model_path=audio_model_path)

    def encode(batch: list[ItemInfo]):
        encode_and_save_batch(vae, audio_vae, batch, args, device)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
