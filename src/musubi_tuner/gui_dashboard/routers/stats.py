"""Project statistics calculation endpoints."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

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
    estimated_step_time_sec: float | None = None
    estimated_steps_per_sec: float | None = None
    estimated_time_source: str | None = None
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


def _coerce_int(value, default: int) -> int:
    """Coerce nullable or string-like values to int with a safe fallback."""
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float) -> float:
    """Coerce nullable or string-like values to float with a safe fallback."""
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_training_dataset(config: dict) -> dict:
    """Return the primary dataset used for training/stat estimates."""
    datasets = config.get('dataset', {}).get('datasets', [])
    if not datasets:
        return {}
    return next((d for d in datasets if d.get('type') in ('video', 'image')), datasets[0])


def _estimate_training_step_time_sec(config: dict) -> float | None:
    """Estimate wall-clock seconds per optimizer step from the current config."""
    try:
        training = config.get('training', {})
        dataset = _first_training_dataset(config)

        mode = str(training.get('ltx2_mode', 'video')).lower()
        res_w = max(_coerce_int(dataset.get('resolution_w', 768), 768), 64)
        res_h = max(_coerce_int(dataset.get('resolution_h', 512), 512), 64)
        frames = max(_coerce_int(dataset.get('target_frames', 33), 33), 1)
        batch_size = max(_coerce_int(dataset.get('batch_size', training.get('train_batch_size', 1)), 1), 1)
        grad_accum = max(_coerce_int(training.get('gradient_accumulation_steps', 1), 1), 1)

        pixel_frame_scale = (res_w * res_h * frames) / (768 * 512 * 33)
        step_time = 2.35 * max(pixel_frame_scale, 0.25) * batch_size * grad_accum

        if mode == 'av':
            step_time *= 1.35
        elif mode == 'audio':
            step_time *= 0.55

        if training.get('gradient_checkpointing', True):
            step_time *= 1.20
        if training.get('blockwise_checkpointing'):
            step_time *= 1.08
        if training.get('gradient_checkpointing_cpu_offload'):
            step_time *= 1.15
        if training.get('fp8_w8a8'):
            step_time *= 0.88
        elif training.get('fp8_base'):
            step_time *= 0.95
        if training.get('nf4_base'):
            step_time *= 1.08
        if training.get('self_flow'):
            step_time *= 1.22
        if training.get('blank_preservation'):
            step_time *= 1.12
        if training.get('dop'):
            step_time *= 1.10
        if training.get('audio_dop'):
            step_time *= 1.10
        if training.get('prior_divergence'):
            step_time *= 1.05
        if training.get('crepa'):
            step_time *= 1.08

        sample_every_n_steps = _coerce_int(training.get('sample_every_n_steps', 0), 0)
        if sample_every_n_steps:
            step_time += 5.0 / sample_every_n_steps

        return max(step_time, 0.05)
    except Exception as e:
        logger.debug(f"Failed to estimate training step time: {e}")
        return None


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
        dataset = _first_training_dataset(config)

        # Batch size
        batch_size = max(_coerce_int(dataset.get('batch_size', training.get('train_batch_size', 1)), 1), 1)
        grad_accum = max(_coerce_int(training.get('gradient_accumulation_steps', 1), 1), 1)
        effective_batch_size = batch_size * grad_accum

        # Steps per epoch
        steps_per_epoch = None
        if dataset_stats and dataset_stats.total_items > 0:
            steps_per_epoch = max(1, dataset_stats.total_items // effective_batch_size)

        # Total epochs
        max_steps = _coerce_int(training.get('max_train_steps', 0), 0)
        total_epochs = None
        if max_steps and steps_per_epoch:
            total_epochs = max_steps / steps_per_epoch

        # Estimated time from the same shape/config heuristic used for iteration time.
        estimated_time_hours = None
        estimated_step_time_sec = _estimate_training_step_time_sec(config)
        estimated_steps_per_sec = (1.0 / estimated_step_time_sec) if estimated_step_time_sec else None
        if max_steps:
            estimated_time_hours = (max_steps * (estimated_step_time_sec or 0)) / 3600

        # Checkpoint size
        network_dim = max(_coerce_int(training.get('network_dim', 16), 16), 1)
        # Rough estimate: LoRA size depends on rank and target modules
        # LTX2 full LoRA is roughly: dim * 2 * hidden_dim * num_layers * 4 bytes
        # For dim=16, roughly 50-100MB
        checkpoint_size_mb = network_dim * 5  # Very rough estimate

        # Total checkpoints
        save_every_n_steps = _coerce_int(training.get('save_every_n_steps', 0), 0)
        save_every_n_epochs = _coerce_int(training.get('save_every_n_epochs', 0), 0)
        total_checkpoints = 1  # Final checkpoint

        if save_every_n_steps and max_steps:
            total_checkpoints += max_steps // save_every_n_steps
        elif save_every_n_epochs and total_epochs:
            total_checkpoints += int(total_epochs) // save_every_n_epochs

        # Apply keep_last limits
        keep_last_steps = _coerce_int(training.get('save_last_n_steps', 0), 0)
        keep_last_epochs = _coerce_int(training.get('save_last_n_epochs', 0), 0)
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
            estimated_step_time_sec=round(estimated_step_time_sec, 3) if estimated_step_time_sec else None,
            estimated_steps_per_sec=round(estimated_steps_per_sec, 4) if estimated_steps_per_sec else None,
            estimated_time_source='heuristic' if estimated_step_time_sec else None,
            checkpoint_size_mb=checkpoint_size_mb,
            total_checkpoints=total_checkpoints,
            total_storage_gb=total_storage_gb
        )

    except Exception as e:
        logger.warning(f"Failed to calculate training stats: {e}")
        return None


def _calculate_vram_stats(config: dict) -> VRAMStats | None:
    """Calculate VRAM usage estimates.

    LTX-2 architecture reference:
    - DiT: 48 transformer blocks, inner_dim=4096 (video), 2048 (audio)
    - VAE compression: temporal 8x, spatial 32x32
    - Latent channels: 128, patch_size: 1
    - LTX 2.0: ~19.6B params → BF16 39 GB, FP8 19.5 GB
    - LTX 2.3: ~21.0B params → BF16 42 GB, FP8 21 GB
    """
    try:
        training = config.get('training', {})
        ds = _first_training_dataset(config)

        # ── DiT weights ──
        ltx_version = str(training.get('ltx_version', '2.3'))
        dit_bf16 = 42.0 if ltx_version == '2.3' else 39.0
        is_fp8 = bool(training.get('fp8_base'))
        is_w8a8 = bool(training.get('fp8_w8a8'))
        is_nf4 = bool(training.get('nf4_base'))
        dit_base = (dit_bf16 / 4) if is_nf4 else (dit_bf16 / 2) if is_fp8 else dit_bf16

        total_blocks = 48
        blocks_to_swap = min(max(_coerce_int(training.get('blocks_to_swap', 0), 0), 0), total_blocks - 1)
        swap_savings = blocks_to_swap * (dit_base / total_blocks) * 0.95
        model_size_gb = max(dit_base - swap_savings, 1.0)

        # ── LoRA weights ──
        rank = max(_coerce_int(training.get('network_dim', 16), 16), 1)
        mode = str(training.get('ltx2_mode', 'video')).lower()
        is_av = mode == 'av'
        lora_base_per_rank = (12.75 if is_av else 6.0) / 1024  # GB per rank
        preset_mult = {
            't2v': 1.0, 'v2v': 1.44, 'video_sa': 0.37, 'video_sa_ff': 0.56,
            'video_sa_ca_ff': 0.74, 'audio': 0.37, 'audio_v2a': 0.52, 'audio_ref_only_ic': 0.63,
            'av_ic': 1.44, 'video_ref_only_av': 1.44, 'full': 2.1,
        }.get(training.get('lora_target_preset'), 1.0)
        lora_size_gb = rank * lora_base_per_rank * preset_mult

        # ── Optimizer states ──
        lora_param_count = lora_size_gb * (1024 ** 3) / 2  # bf16 -> count
        opt_type = str(training.get('optimizer_type', 'adamw8bit')).lower()
        is_8bit = '8bit' in opt_type
        is_sf = 'schedulefree' in opt_type or opt_type == 'automagic'
        opt_bytes = 6 if is_8bit else (14 if is_sf else 12)
        optimizer_size_gb = (lora_param_count * opt_bytes) / (1024 ** 3)

        # ── Activations ──
        res_w = max(_coerce_int(ds.get('resolution_w', 768), 768), 64)
        res_h = max(_coerce_int(ds.get('resolution_h', 512), 512), 64)
        frames = max(_coerce_int(ds.get('target_frames', 33), 33), 1)
        batch_size = max(_coerce_int(ds.get('batch_size', 1), 1), 1)

        # Correct VAE compression factors
        latent_f = max(1, (frames - 1) // 8 + 1)
        latent_h = max(1, res_h // 32)
        latent_w = max(1, res_w // 32)
        seq_len = latent_f * latent_h * latent_w

        hidden_dim = 2048 if mode == 'audio' else 4096
        bytes_per_val = 1 if is_w8a8 else 2
        grad_ckpt = training.get('gradient_checkpointing', True)
        blockwise = bool(training.get('blockwise_checkpointing'))

        activ_coeff = 10 if not grad_ckpt else (1 if blockwise else 2)
        effective_layers = total_blocks if not blockwise else 2
        per_layer_bytes = activ_coeff * batch_size * seq_len * hidden_dim * bytes_per_val
        activations_gb = (per_layer_bytes * effective_layers) / (1024 ** 3)
        if is_av:
            activations_gb *= 1.25
        if _coerce_int(training.get('ffn_chunk_size', 0), 0) > 0:
            activations_gb *= 0.90
        if training.get('split_attn_mode') or training.get('split_attn_target'):
            activations_gb *= 0.92
        if training.get('gradient_checkpointing_cpu_offload') and grad_ckpt:
            activations_gb *= 0.35

        # Fixed buffers
        latent_bytes = batch_size * 128 * latent_f * latent_h * latent_w * 2 * 2
        text_bytes = batch_size * 256 * (7680 if is_av else 3840) * 2
        buffer_gb = (latent_bytes + text_bytes) / (1024 ** 3) + 0.5
        if training.get('img_in_txt_in_offloading'):
            buffer_gb = max(0.2, buffer_gb - 0.3)
        activations_gb = max(0.3, activations_gb + buffer_gb)

        # ── Gradients ──
        grads_gb = lora_size_gb

        # ── Gradient accumulation ──
        grad_accum = max(_coerce_int(training.get('gradient_accumulation_steps', 1), 1), 1)
        grad_accum_gb = grads_gb * 0.4 if grad_accum > 1 else 0

        # ── Preservation / DOP ──
        preservation_gb = 0
        if training.get('blank_preservation'):
            preservation_gb += activations_gb * 0.35
        if training.get('dop'):
            preservation_gb += activations_gb * 0.35
        if training.get('audio_dop'):
            preservation_gb += activations_gb * 0.35
        if training.get('prior_divergence'):
            preservation_gb += activations_gb * 0.15

        # ── Self-Flow ──
        self_flow_gb = 0
        if training.get('self_flow'):
            teacher_mode = str(training.get('self_flow_teacher_mode', 'base')).lower()
            if teacher_mode == 'ema':
                self_flow_gb += lora_size_gb
            elif teacher_mode == 'partial_ema':
                self_flow_gb += max(lora_size_gb / total_blocks, 0.01)

            has_audio_projector = (
                mode in {'av', 'audio'}
                and _coerce_float(training.get('self_flow_lambda_audio', 0.0), 0.0) > 0.0
            )
            self_flow_gb += 0.03 if has_audio_projector else 0.02

            feature_factor = 0.03 if training.get('self_flow_offload_teacher_features') else 0.10
            self_flow_gb += activations_gb * feature_factor

        # ── CREPA ──
        crepa_gb = 0
        if training.get('crepa'):
            crepa_gb = 0.08 if str(training.get('crepa_mode', 'backbone')) == 'dino' else 0.15

        peak_training_gb = (model_size_gb + lora_size_gb + optimizer_size_gb + grads_gb +
                            activations_gb + grad_accum_gb + preservation_gb + self_flow_gb + crepa_gb)

        # Sampling VRAM (VAE loaded, lighter activations)
        peak_sampling_gb = model_size_gb + 0.3 + (activations_gb * 0.3)
        if training.get('sample_with_offloading'):
            peak_sampling_gb *= 0.6

        overhead_gb = grad_accum_gb + preservation_gb + self_flow_gb + crepa_gb
        breakdown = {
            'model': round(model_size_gb, 2),
            'lora': round(lora_size_gb, 2),
            'optimizer': round(optimizer_size_gb, 2),
            'gradients': round(grads_gb, 2),
            'activations': round(activations_gb, 2),
            'overhead': round(overhead_gb, 2),
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
