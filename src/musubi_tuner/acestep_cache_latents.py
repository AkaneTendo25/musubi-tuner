"""
Cache latents for ACE-Step 1.5 architecture.

This script encodes audio files using ACE-Step's VAE (AutoencoderOobleck) and caches
the latent representations for faster training.
"""

import logging
import os
from typing import List, Optional

import torch
import torchaudio
import librosa

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ItemInfo,
    ARCHITECTURE_ACESTEP,
    save_latent_cache_acestep,
)
from musubi_tuner.acestep.acestep_config import (
    ACESTEP_SAMPLE_RATE,
    ACESTEP_MAX_DURATION_SECONDS,
    ACESTEP_LATENT_CHANNELS,
)
from musubi_tuner.acestep import acestep_utils
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_and_preprocess_audio(
    audio_path: str,
    target_sample_rate: int = ACESTEP_SAMPLE_RATE,
    max_duration: float = ACESTEP_MAX_DURATION_SECONDS,
) -> torch.Tensor:
    """Load and preprocess audio file.

    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate (48kHz)
        max_duration: Maximum duration in seconds

    Returns:
        Audio tensor [2, samples] (stereo)
    """
    # Load with librosa (supports MP3, WAV, FLAC, etc.)
    audio_np, sample_rate = librosa.load(audio_path, sr=target_sample_rate, mono=False)

    # Convert to tensor
    waveform = torch.from_numpy(audio_np).float()

    # Handle mono/stereo
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).repeat(2, 1)  # Mono to stereo
    elif waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]  # Take first 2 channels

    # Truncate to max duration
    max_samples = int(max_duration * target_sample_rate)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    return waveform


def encode_and_save_batch(vae, batch: List[ItemInfo], device: torch.device, max_duration: float):
    """Encode a batch of audio files and save their latent representations.

    Args:
        vae: ACE-Step VAE model (AutoencoderOobleck)
        batch: List of ItemInfo containing audio files to encode
        device: Device to use for encoding
        max_duration: Maximum audio duration in seconds
    """
    for item in batch:
        try:
            # Load audio
            audio = load_and_preprocess_audio(item.item_key, max_duration=max_duration)
            audio = audio.unsqueeze(0).to(device)  # [1, 2, samples]

            with torch.no_grad():
                # AutoencoderOobleck encode: [B, 2, T_samples] -> [B, 64, T_latent]
                latent_dist = vae.encode(audio)
                if hasattr(latent_dist, "latent_dist"):
                    latents = latent_dist.latent_dist.sample()
                else:
                    latents = latent_dist.sample() if hasattr(latent_dist, "sample") else latent_dist

                # Transpose to [B, T, 64] format used by decoder
                latents = latents.permute(0, 2, 1).contiguous()  # [1, T, 64]

            # Create attention mask (all valid)
            T = latents.shape[1]
            attention_mask = torch.ones(T, dtype=torch.bool)

            # Create context latents (silence/zeros for unconditional training)
            context_latents = torch.zeros(T, 128)

            # Remove batch dimension
            target_latents = latents.squeeze(0).cpu()  # [T, 64]

            # Update item info with dimensions
            item.original_size = (T, ACESTEP_LATENT_CHANNELS)
            item.frame_count = T

            logger.info(f"Saving cache for {item.item_key}: latents {target_latents.shape}")

            save_latent_cache_acestep(
                item_info=item,
                target_latents=target_latents,
                attention_mask=attention_mask,
                context_latents=context_latents,
            )

        except Exception as e:
            logger.error(f"Error processing {item.item_key}: {e}")
            continue


def main():
    parser = cache_latents.setup_parser_common()
    parser.add_argument(
        "--max_duration",
        type=float,
        default=ACESTEP_MAX_DURATION_SECONDS,
        help=f"Maximum audio duration in seconds (default: {ACESTEP_MAX_DURATION_SECONDS})",
    )

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_ACESTEP)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        logger.info("Debug mode for audio not fully implemented yet")
        return

    assert args.vae is not None, "VAE checkpoint is required (--vae)"

    logger.info(f"Loading ACE-Step VAE from {args.vae}")
    vae = acestep_utils.load_acestep_vae(args.vae, device=str(device))
    logger.info(f"Loaded ACE-Step VAE")

    # Get max_duration from args
    max_duration = args.max_duration

    # Encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch, device, max_duration)

    # Process datasets
    for i, dataset in enumerate(datasets):
        logger.info(f"Processing dataset {i + 1}/{len(datasets)}")

        # Get batches from dataset
        for batch in dataset.retrieve_latent_cache_batches(args.num_workers):
            # Check skip existing
            if args.skip_existing:
                batch = [item for item in batch if not os.path.exists(item.latent_cache_path)]
                if len(batch) == 0:
                    continue

            encode(batch)

    logger.info("Done!")


if __name__ == "__main__":
    main()
