"""Project statistics calculation endpoints."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stats", tags=["stats"])

# Cache for dataset stats to avoid repeated scanning
_dataset_cache: dict[str, tuple[float, DatasetStats]] = {}  # path -> (mtime, stats)


class DatasetStats(BaseModel):
    """Statistics about the dataset."""
    total_items: int
    video_items: int
    audio_items: int
    avg_resolution: tuple[int, int] | None
    avg_frames: float | None
    max_resolution: tuple[int, int] | None
    max_frames: int | None


class TrainingStats(BaseModel):
    """Calculated training statistics."""
    steps_per_epoch: int | None
    total_epochs: float | None
    effective_batch_size: int
    estimated_time_hours: float | None
    checkpoint_size_mb: float
    total_checkpoints: int
    total_storage_gb: float


class VRAMStats(BaseModel):
    """VRAM usage estimates."""
    peak_training_gb: float
    peak_sampling_gb: float
    model_size_gb: float
    optimizer_size_gb: float
    activations_gb: float
    breakdown: dict[str, float]


class ProjectStats(BaseModel):
    """Complete project statistics."""
    dataset: DatasetStats | None
    training: TrainingStats | None
    vram: VRAMStats | None


def _scan_dataset(dataset_path: str) -> DatasetStats | None:
    """Scan dataset directory and extract statistics (with caching)."""
    try:
        path = Path(dataset_path)
        if not path.exists():
            return None

        # Check cache first
        try:
            mtime = path.stat().st_mtime
            if dataset_path in _dataset_cache:
                cached_mtime, cached_stats = _dataset_cache[dataset_path]
                if cached_mtime == mtime:
                    logger.debug(f"Using cached dataset stats for {dataset_path}")
                    return cached_stats
        except:
            pass

        # Look for dataset config
        toml_path = path / "dataset_config.toml"
        if not toml_path.exists():
            # Try to count files directly
            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            files = [f for f in path.rglob('*') if f.suffix.lower() in video_exts]
            return DatasetStats(
                total_items=len(files),
                video_items=len(files),
                audio_items=0,
                avg_resolution=None,
                avg_frames=None,
                max_resolution=None,
                max_frames=None
            )

        # Parse TOML to get subsets
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # Fallback for older Python
        with open(toml_path, 'rb') as f:
            config = tomllib.load(f)

        total_items = 0
        video_items = 0
        audio_items = 0
        resolutions = []
        frames = []

        for subset in config.get('subsets', []):
            video_dir = subset.get('video_dir', '')
            if not video_dir:
                continue

            subset_path = Path(video_dir) if os.path.isabs(video_dir) else path / video_dir
            if not subset_path.exists():
                continue

            # Count videos in this subset
            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            subset_videos = [f for f in subset_path.rglob('*') if f.suffix.lower() in video_exts]

            num_videos = len(subset_videos)
            total_items += num_videos

            # Check if audio subset
            is_audio = subset.get('is_audio', False) or 'audio' in video_dir.lower()
            if is_audio:
                audio_items += num_videos
            else:
                video_items += num_videos

            # Try to get resolution/frames from metadata if available
            metadata_path = subset_path / '.metadata.json'
            if metadata_path.exists():
                try:
                    with open(metadata_path) as mf:
                        meta = json.load(mf)
                        if 'resolution' in meta:
                            resolutions.append(tuple(meta['resolution']))
                        if 'frames' in meta:
                            frames.append(meta['frames'])
                except:
                    pass

        # Calculate averages
        avg_resolution = None
        max_resolution = None
        if resolutions:
            avg_w = sum(r[0] for r in resolutions) / len(resolutions)
            avg_h = sum(r[1] for r in resolutions) / len(resolutions)
            avg_resolution = (int(avg_w), int(avg_h))
            max_resolution = max(resolutions, key=lambda r: r[0] * r[1])

        avg_frames = sum(frames) / len(frames) if frames else None
        max_frames = max(frames) if frames else None

        stats = DatasetStats(
            total_items=total_items,
            video_items=video_items,
            audio_items=audio_items,
            avg_resolution=avg_resolution,
            avg_frames=avg_frames,
            max_resolution=max_resolution,
            max_frames=max_frames
        )

        # Cache the result
        try:
            mtime = path.stat().st_mtime
            _dataset_cache[dataset_path] = (mtime, stats)
        except:
            pass

        return stats

    except Exception as e:
        logger.warning(f"Failed to scan dataset: {e}")
        return None


def _calculate_training_stats(config: dict, dataset_stats: DatasetStats | None) -> TrainingStats | None:
    """Calculate training statistics from config."""
    try:
        training = config.get('training', {})

        # Batch size
        batch_size = training.get('train_batch_size', 1)
        grad_accum = training.get('gradient_accumulation_steps', 1)
        effective_batch_size = batch_size * grad_accum

        # Steps per epoch
        steps_per_epoch = None
        if dataset_stats and dataset_stats.total_items > 0:
            steps_per_epoch = max(1, dataset_stats.total_items // effective_batch_size)

        # Total epochs
        max_steps = training.get('max_train_steps')
        total_epochs = None
        if max_steps and steps_per_epoch:
            total_epochs = max_steps / steps_per_epoch

        # Estimated time (very rough estimate: 1-3 sec per step depending on config)
        estimated_time_hours = None
        if max_steps:
            # Base time per step
            time_per_step = 2.0  # seconds

            # Adjust based on settings
            if training.get('gradient_checkpointing', True):
                time_per_step *= 1.2  # Slower with grad checkpointing
            if training.get('sample_every_n_steps'):
                # Add sampling overhead
                sample_freq = training['sample_every_n_steps']
                time_per_step += (5.0 / sample_freq)  # ~5 sec per sample

            estimated_time_hours = (max_steps * time_per_step) / 3600

        # Checkpoint size
        network_dim = training.get('network_dim', 16)
        # Rough estimate: LoRA size depends on rank and target modules
        # LTX2 full LoRA is roughly: dim * 2 * hidden_dim * num_layers * 4 bytes
        # For dim=16, roughly 50-100MB
        checkpoint_size_mb = network_dim * 5  # Very rough estimate

        # Total checkpoints
        save_every_n_steps = training.get('save_every_n_steps')
        save_every_n_epochs = training.get('save_every_n_epochs')
        total_checkpoints = 1  # Final checkpoint

        if save_every_n_steps and max_steps:
            total_checkpoints += max_steps // save_every_n_steps
        elif save_every_n_epochs and total_epochs:
            total_checkpoints += int(total_epochs) // save_every_n_epochs

        # Apply keep_last limits
        keep_last_steps = training.get('save_last_n_steps')
        keep_last_epochs = training.get('save_last_n_epochs')
        if keep_last_steps:
            total_checkpoints = min(total_checkpoints, keep_last_steps + 1)
        if keep_last_epochs:
            total_checkpoints = min(total_checkpoints, keep_last_epochs + 1)

        total_storage_gb = (checkpoint_size_mb * total_checkpoints) / 1024

        return TrainingStats(
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            effective_batch_size=effective_batch_size,
            estimated_time_hours=estimated_time_hours,
            checkpoint_size_mb=checkpoint_size_mb,
            total_checkpoints=total_checkpoints,
            total_storage_gb=total_storage_gb
        )

    except Exception as e:
        logger.warning(f"Failed to calculate training stats: {e}")
        return None


def _calculate_vram_stats(config: dict) -> VRAMStats | None:
    """Calculate VRAM usage estimates."""
    try:
        training = config.get('training', {})

        # Model size (LTX2)
        model_size_gb = 10.0  # Base model ~10GB in bf16

        # Apply quantization
        if training.get('fp8_base'):
            model_size_gb *= 0.5

        # Text encoder (Gemma)
        gemma_size = 4.5  # ~4.5GB for Gemma
        if training.get('gemma_load_in_8bit'):
            gemma_size *= 0.5
        elif training.get('gemma_load_in_4bit'):
            gemma_size *= 0.25

        # If precached, text encoder not loaded during training
        if training.get('use_precached_sample_prompts'):
            gemma_size = 0

        model_size_gb += gemma_size

        # LoRA parameters (very small)
        network_dim = training.get('network_dim', 16)
        lora_size_gb = (network_dim * 0.005)  # Tiny

        # Optimizer states (typically 2x model params for Adam)
        optimizer_size_gb = lora_size_gb * 2

        # Activations (depends heavily on config)
        batch_size = training.get('train_batch_size', 1)
        frames = 45  # Typical
        height = training.get('height', 512)
        width = training.get('width', 768)

        hidden_dim = 3840
        num_layers = 48
        blocks_to_swap = training.get('blocks_to_swap', 0)
        active_layers = num_layers - blocks_to_swap

        # Sequence length (rough estimate)
        seq_len = (frames * height * width) // (16 * 16) + 128  # Latent + text tokens

        # Activation memory
        activations_gb = (hidden_dim * active_layers * seq_len * batch_size * 2) / (1024**3)

        # Gradient checkpointing reduces activations dramatically
        if training.get('gradient_checkpointing', True):
            activations_gb *= 0.1  # ~90% reduction
            if training.get('blockwise_checkpointing'):
                activations_gb *= 0.5  # Further reduction

        # Peak training VRAM
        peak_training_gb = model_size_gb + optimizer_size_gb + activations_gb + 2.0  # +2GB overhead

        # Sampling VRAM (includes VAE)
        vae_size = 0.3  # ~300MB
        peak_sampling_gb = model_size_gb + vae_size + (activations_gb * 0.3)  # Less activations

        if training.get('sample_with_offloading'):
            peak_sampling_gb *= 0.6  # Significant reduction with offloading

        breakdown = {
            'model': round(model_size_gb, 2),
            'optimizer': round(optimizer_size_gb, 2),
            'activations': round(activations_gb, 2),
            'overhead': 2.0
        }

        return VRAMStats(
            peak_training_gb=round(peak_training_gb, 2),
            peak_sampling_gb=round(peak_sampling_gb, 2),
            model_size_gb=round(model_size_gb, 2),
            optimizer_size_gb=round(optimizer_size_gb, 2),
            activations_gb=round(activations_gb, 2),
            breakdown=breakdown
        )

    except Exception as e:
        logger.warning(f"Failed to calculate VRAM stats: {e}")
        return None


@router.get("", response_model=ProjectStats)
async def get_project_stats(request: Request):
    """Get comprehensive project statistics."""
    config = request.app.state.project_config
    if not config:
        return ProjectStats(dataset=None, training=None, vram=None)

    config_dict = config.model_dump()

    # Dataset stats
    dataset_stats = None
    dataset_config = config_dict.get('dataset', {})
    datasets = dataset_config.get('datasets', []) if dataset_config else []

    # Scan first dataset entry
    if datasets and len(datasets) > 0:
        first_dataset = datasets[0]
        dataset_dir = first_dataset.get('directory', '')
        if dataset_dir:
            dataset_stats = _scan_dataset(dataset_dir)

    # Training stats (only if we have dataset info)
    training_stats = None
    if dataset_stats:
        training_stats = _calculate_training_stats(config_dict, dataset_stats)

    # VRAM stats (can calculate without dataset)
    vram_stats = _calculate_vram_stats(config_dict)

    return ProjectStats(
        dataset=dataset_stats,
        training=training_stats,
        vram=vram_stats
    )
