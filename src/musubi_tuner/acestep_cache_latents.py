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

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ItemInfo,
    ARCHITECTURE_ACESTEP,
    save_latent_cache_acestep,
)
from musubi_tuner.acestep.acestep_config import ACESTEP_SAMPLE_RATE, ACESTEP_MAX_DURATION_SECONDS, ACESTEP_LATENT_CHANNELS
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
    # Prefer torchaudio (matches original trainer path). If backend lacks MP3 support,
    # fallback to librosa so MP3 datasets still work.
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.float()

        # Resample to target sample rate if needed.
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
    except Exception as e:
        logger.warning(f"torchaudio failed for {audio_path}: {e}. Falling back to librosa.")
        import librosa

        audio_np, _ = librosa.load(audio_path, sr=target_sample_rate, mono=False)
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


def _encode_single_chunk(vae, audio_chunk: torch.Tensor) -> torch.Tensor:
    """Encode a single audio chunk through VAE.

    Args:
        vae: VAE model
        audio_chunk: Audio tensor [1, 2, samples]

    Returns:
        Latent tensor [1, T, 64]
    """
    with torch.no_grad():
        latent_dist = vae.encode(audio_chunk)
        if hasattr(latent_dist, "latent_dist"):
            latents = latent_dist.latent_dist.sample()
        elif hasattr(latent_dist, "sample"):
            latents = latent_dist.sample()
        else:
            latents = latent_dist
        # [B, 64, T] -> [B, T, 64]
        return latents.permute(0, 2, 1).contiguous()


def encode_audio_chunked(
    vae,
    audio: torch.Tensor,
    device: torch.device,
    chunk_seconds: Optional[float] = None,
    overlap_seconds: float = 2.0,
) -> torch.Tensor:
    """Encode audio with optional chunking and overlap blending for lossless operation.

    Uses overlapping chunks with cosine blending at boundaries to avoid artifacts.
    This ensures the output is identical to non-chunked encoding (within floating point precision).

    Args:
        vae: ACE-Step VAE model (AutoencoderOobleck)
        audio: Audio tensor [1, 2, samples]
        device: Device to use for encoding
        chunk_seconds: If set, encode in chunks of this duration (reduces VRAM)
        overlap_seconds: Overlap duration for blending (default: 2.0 seconds)

    Returns:
        Latents tensor [1, T, 64]
    """
    from musubi_tuner.acestep.acestep_config import ACESTEP_VAE_TEMPORAL_FACTOR

    audio = audio.to(device)

    if chunk_seconds is None or chunk_seconds <= 0:
        # No chunking - encode all at once
        return _encode_single_chunk(vae, audio)

    # Calculate chunk and overlap sizes in samples
    chunk_samples = int(chunk_seconds * ACESTEP_SAMPLE_RATE)
    overlap_samples = int(overlap_seconds * ACESTEP_SAMPLE_RATE)

    # Align to VAE temporal factor
    chunk_samples = (chunk_samples // ACESTEP_VAE_TEMPORAL_FACTOR) * ACESTEP_VAE_TEMPORAL_FACTOR
    overlap_samples = (overlap_samples // ACESTEP_VAE_TEMPORAL_FACTOR) * ACESTEP_VAE_TEMPORAL_FACTOR

    # Ensure overlap is reasonable (at least 1 latent frame, at most half chunk)
    min_overlap = ACESTEP_VAE_TEMPORAL_FACTOR
    max_overlap = chunk_samples // 2
    overlap_samples = max(min_overlap, min(overlap_samples, max_overlap))

    # Calculate overlap in latent space
    overlap_latents = overlap_samples // ACESTEP_VAE_TEMPORAL_FACTOR

    total_samples = audio.shape[-1]
    step_samples = chunk_samples - overlap_samples  # How much to advance each chunk

    # If audio is shorter than one chunk, just encode directly
    if total_samples <= chunk_samples:
        latents = _encode_single_chunk(vae, audio)
        # Move to CPU to free GPU memory
        return latents.cpu().to(device)

    # Build list of chunk start positions
    chunk_starts = []
    pos = 0
    while pos < total_samples:
        chunk_starts.append(pos)
        if pos + chunk_samples >= total_samples:
            break
        pos += step_samples

    logger.info(f"Encoding {len(chunk_starts)} overlapping chunks (chunk={chunk_seconds}s, overlap={overlap_seconds}s)")

    # Encode all chunks
    encoded_chunks = []
    for i, start in enumerate(chunk_starts):
        end = min(start + chunk_samples, total_samples)
        audio_chunk = audio[:, :, start:end]

        # Pad last chunk if needed
        if audio_chunk.shape[-1] < ACESTEP_VAE_TEMPORAL_FACTOR:
            pad_size = ACESTEP_VAE_TEMPORAL_FACTOR - audio_chunk.shape[-1]
            audio_chunk = torch.nn.functional.pad(audio_chunk, (0, pad_size))

        chunk_latents = _encode_single_chunk(vae, audio_chunk)
        encoded_chunks.append(chunk_latents.cpu())

        # Free GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Blend chunks together using cosine crossfade
    # Start with first chunk
    result = encoded_chunks[0]

    for i in range(1, len(encoded_chunks)):
        prev_chunk = result
        curr_chunk = encoded_chunks[i]

        # Calculate actual overlap for this pair
        prev_len = prev_chunk.shape[1]
        curr_len = curr_chunk.shape[1]

        # The overlap region in latent space
        actual_overlap = min(overlap_latents, prev_len, curr_len)

        if actual_overlap <= 0:
            # No overlap, just concatenate
            result = torch.cat([prev_chunk, curr_chunk], dim=1)
        else:
            # Create cosine blend weights for smooth transition
            # prev_weight goes from 1 to 0, curr_weight goes from 0 to 1
            t = torch.linspace(0, 1, actual_overlap, device=prev_chunk.device)
            # Cosine blend for smoother transition
            curr_weight = (1 - torch.cos(t * 3.14159265)) / 2  # 0 -> 1
            prev_weight = 1 - curr_weight  # 1 -> 0

            # Reshape weights for broadcasting [1, overlap, 1]
            prev_weight = prev_weight.view(1, -1, 1)
            curr_weight = curr_weight.view(1, -1, 1)

            # Extract regions
            prev_non_overlap = prev_chunk[:, :-actual_overlap, :]
            prev_overlap = prev_chunk[:, -actual_overlap:, :]
            curr_overlap = curr_chunk[:, :actual_overlap, :]
            curr_non_overlap = curr_chunk[:, actual_overlap:, :]

            # Blend overlap region
            blended_overlap = prev_weight * prev_overlap + curr_weight * curr_overlap

            # Concatenate: prev_non_overlap + blended + curr_non_overlap
            result = torch.cat([prev_non_overlap, blended_overlap, curr_non_overlap], dim=1)

    # Trim to expected length
    expected_length = total_samples // ACESTEP_VAE_TEMPORAL_FACTOR
    if result.shape[1] > expected_length:
        result = result[:, :expected_length, :]

    return result.to(device)


def encode_and_save_batch(
    vae,
    batch: List[ItemInfo],
    device: torch.device,
    max_duration: float,
    chunk_seconds: Optional[float] = None,
    overlap_seconds: float = 2.0,
):
    """Encode a batch of audio files and save their latent representations.

    Args:
        vae: ACE-Step VAE model (AutoencoderOobleck)
        batch: List of ItemInfo containing audio files to encode
        device: Device to use for encoding
        max_duration: Maximum audio duration in seconds
        chunk_seconds: If set, encode in chunks to reduce VRAM usage
        overlap_seconds: Overlap duration for chunk blending (default: 2.0)
    """
    for item in batch:
        try:
            # Load audio
            audio = load_and_preprocess_audio(item.item_key, max_duration=max_duration)
            audio = audio.unsqueeze(0)  # [1, 2, samples]

            # Encode with optional chunking and overlap blending
            latents = encode_audio_chunked(vae, audio, device, chunk_seconds, overlap_seconds)  # [1, T, 64]

            # Create attention mask (all valid)
            T = latents.shape[1]
            attention_mask = torch.ones(T, dtype=torch.bool)

            # Remove batch dimension and ensure on CPU
            target_latents = latents.squeeze(0).cpu()  # [T, 64]
            del latents
            torch.cuda.empty_cache() if device.type == "cuda" else None

            # Update item info with dimensions
            item.original_size = (T, ACESTEP_LATENT_CHANNELS)
            item.frame_count = T

            logger.info(f"Saving cache for {item.item_key}: latents {target_latents.shape}")

            save_latent_cache_acestep(
                item_info=item,
                target_latents=target_latents,
                attention_mask=attention_mask,
                context_latents=None,  # Let the trainer build silence+chunk context from the model.
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
    parser.add_argument(
        "--vae_chunk_seconds",
        type=float,
        default=None,
        help="Encode audio in chunks of N seconds to reduce VRAM usage (e.g., 30 or 60). Default: no chunking.",
    )
    parser.add_argument(
        "--vae_chunk_overlap",
        type=float,
        default=2.0,
        help="Overlap duration in seconds for chunk blending (default: 2.0). Larger values = smoother blending.",
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

    # Get max_duration and chunk_seconds from args
    max_duration = args.max_duration
    chunk_seconds = args.vae_chunk_seconds
    overlap_seconds = args.vae_chunk_overlap

    if chunk_seconds is not None:
        logger.info(f"VAE chunking enabled: {chunk_seconds}s chunks with {overlap_seconds}s overlap blending")

    # Encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch, device, max_duration, chunk_seconds, overlap_seconds)

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
