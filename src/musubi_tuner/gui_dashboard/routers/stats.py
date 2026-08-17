"""Project statistics calculation endpoints."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from functools import lru_cache
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
    datasets = config.get("dataset", {}).get("datasets", [])
    if not datasets:
        return {}
    return next((d for d in datasets if d.get("type") in ("video", "image")), datasets[0])


def _training_datasets(config: dict) -> list[dict]:
    """Return every configured training dataset row."""
    datasets = config.get("dataset", {}).get("datasets", [])
    return [dataset for dataset in datasets if isinstance(dataset, dict)] or [{}]


def _h3_temporal_latents(frame_count: int) -> tuple[int, int]:
    """Return H3 video/audio latent lengths for a dashboard frame count.

    Invalid, partially edited UI values are aligned upward for estimation. The
    command validator still rejects them before launch.
    """
    if frame_count <= 1:
        return 1, 0
    frame_count = max(frame_count, 5)
    frame_count += (5 - frame_count % 17) % 17
    return ((frame_count - 5) // 17) * 5 + 2, (10 * frame_count + 3) // 6


def _h3_spatial_rows(width: int, height: int) -> int:
    """Return packed H3 rows for one visual latent frame."""
    return max(height // 32, 1) * max(width // 32, 1)


def _h3_reference_video_rows_per_frame(width: int, height: int, short_edge: int = 768, maximum_pixels: int = 768 * 1344) -> int:
    """Mirror the configured H3 reference-video resize policy."""
    width = max(width, 1)
    height = max(height, 1)
    ratio = width / height
    if ratio >= 1:
        resolved_width, resolved_height = float(short_edge) * ratio, float(short_edge)
    else:
        resolved_width, resolved_height = float(short_edge), float(short_edge) / ratio
    if resolved_width * resolved_height > maximum_pixels:
        scale = (maximum_pixels / (resolved_width * resolved_height)) ** 0.5
        resolved_width *= scale
        resolved_height *= scale
    # references._multiple_size rounds each dimension to the H3 16-pixel
    # canvas; packing then groups each 2x2 latent patch, hence /32 rows.
    resolved_width = max(16, round(resolved_width / 16) * 16)
    resolved_height = max(16, round(resolved_height / 16) * 16)
    return _h3_spatial_rows(resolved_width, resolved_height)


def _h3_dataset_rows(training: dict, caching: dict, dataset: dict) -> int:
    """Approximate the largest packed H3 sequence represented by one row."""
    dataset_type = str(dataset.get("type", "video"))
    target_mode = str(dataset.get("h3_target_mode", "av"))
    if dataset_type == "image":
        target_mode = "video"
    elif dataset_type == "audio":
        target_mode = "audio"

    width = max(_coerce_int(dataset.get("resolution_w", 768), 768), 64)
    height = max(_coerce_int(dataset.get("resolution_h", 512), 512), 64)
    frames = max(
        _coerce_int(
            dataset.get("h3_image_frame_count", 1) if dataset_type == "image" else dataset.get("target_frames", 124),
            1 if dataset_type == "image" else 124,
        ),
        1,
    )
    video_latents, audio_latents = _h3_temporal_latents(frames)
    rows_per_video_frame = _h3_spatial_rows(width, height)
    video_rows = video_latents * rows_per_video_frame if target_mode in {"av", "video"} else 0
    audio_rows = 2 * audio_latents if target_mode in {"av", "audio"} else 0

    condition_rows = 0
    task = str(caching.get("h3_task", "t2va"))
    if task in {"i2va", "l2va"}:
        condition_rows += rows_per_video_frame
    elif task == "fl2va":
        condition_rows += 2 * rows_per_video_frame
    elif task in {"ref2va", "ref2va_omni"}:
        # Exact reference shapes live in the cache, not project JSON. Use the
        # strongest contract the TOML exposes: paired directories are video
        # references, while the generic directory is conservatively one still.
        modality = str(dataset.get("control_modality", "av") or "av")
        if dataset.get("control_video_directory"):
            reference_frames = max(_coerce_int(dataset.get("reference_frames", frames), frames), 1)
            ref_video_latents, ref_audio_latents = _h3_temporal_latents(reference_frames)
            if modality in {"av", "video"}:
                reference_short_edge = max(_coerce_int(training.get("reference_video_short_edge", 768), 768), 16)
                reference_max_pixels = max(_coerce_int(training.get("reference_video_max_pixels", 768 * 1344), 768 * 1344), 256)
                condition_rows += ref_video_latents * _h3_reference_video_rows_per_frame(
                    width, height, reference_short_edge, reference_max_pixels
                )
            if modality in {"av", "audio"} and dataset.get("control_audio_directory"):
                condition_rows += 2 * ref_audio_latents
        elif dataset.get("control_directory"):
            # Directory controls may contain images or videos and their source
            # aspect ratios are not represented in project JSON. A square image
            # at the configured short edge is the least surprising estimate;
            # Video references use their separately configured sizing policy.
            if str(training.get("reference_image_size_mode", "short_edge")) == "target_area":
                max_pixels = max(_coerce_int(training.get("reference_image_max_pixels", 0), 0), 0)
                target_pixels = width * height if not max_pixels else min(width * height, max_pixels)
                side = max(round(target_pixels**0.5 / 16) * 16, 16)
                condition_rows += _h3_spatial_rows(side, side)
            else:
                short_edge = max(_coerce_int(training.get("reference_image_short_edge", 2048), 2048), 16)
                condition_rows += _h3_spatial_rows(short_edge, short_edge)

    keyframe_count = max(_coerce_int(training.get("h3_keyframe_random_count", 0), 0), 0)
    if not keyframe_count:
        keyframe_count = len([value for value in str(training.get("h3_keyframe_anchors", "")).split(",") if value.strip()])
    condition_rows += keyframe_count * rows_per_video_frame
    if str(training.get("h3_extension_route", "condition_rows")) == "condition_rows":
        condition_rows += max(_coerce_int(training.get("h3_extension_video_frames", 0), 0), 0) * rows_per_video_frame
        condition_rows += 2 * max(_coerce_int(training.get("h3_extension_audio_latents", 0), 0), 0)

    # Prompt length varies by caption and visual presentation. A modest text
    # allowance keeps short/image/audio estimates from collapsing to zero.
    return max(video_rows + audio_rows + condition_rows + 256, 256)


def _h3_base_size_gb(training: dict) -> float:
    rank_value = training.get("h3_adaln_rank")
    rank = _coerce_int(rank_value, 0) if rank_value not in (None, "") else 0
    convrot_billions = 20.1
    adaln_billions = 0.077 * rank / 16 if rank > 0 else 13.0
    params_billions = convrot_billions + adaln_billions
    gib_per_billion_bytes = 1e9 / (1024**3)
    if training.get("int8_convrot_base"):
        return 19.53
    if training.get("h3_convrot_int8"):
        return (convrot_billions * 1.04 + adaln_billions * 2) * gib_per_billion_bytes
    if training.get("fp8_base") or training.get("fp8_scaled"):
        return params_billions * 1.04 * gib_per_billion_bytes
    return params_billions * 2 * gib_per_billion_bytes


def _h3_lora_size_gb(training: dict) -> float:
    """BF16 size of the actual H3 attention+MLP LoRA target set."""
    rank = max(_coerce_int(training.get("network_dim", 16), 16), 1)
    # Per block: qkv, attention output, SwiGLU input, and FFN output.
    parameters_per_rank = 50 * (26_880 + 12_544 + 34_048 + 19_712)
    if training.get("h3_lora_token_refiner"):
        parameters_per_rank += 2 * (26_880 + 12_544 + 34_048 + 19_712)
    return parameters_per_rank * rank * 2 / (1024**3)


def _h3_crepa_memory_gb(training: dict, dataset: dict) -> float:
    """Estimate CREPA parameters, optimizer state, and retained activations."""
    if not training.get("crepa"):
        return 0.0
    hidden = 5376
    mode = str(training.get("crepa_mode", "backbone"))
    dino_dims = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }
    output = hidden if mode == "backbone" else dino_dims.get(str(training.get("crepa_dino_model", "dinov2_vitb14")), 768)
    parameter_count = hidden * hidden + hidden + hidden * output + output
    parameter_gb = parameter_count * 4 / (1024**3)
    optimizer = str(training.get("optimizer_type", "adamw8bit")).lower()
    optimizer_bytes = 2 if ("8bit" in optimizer or "4bit" in optimizer or "fp8" in optimizer) else 8
    state_gb = parameter_count * optimizer_bytes / (1024**3)

    dataset_type = str(dataset.get("type", "video"))
    if dataset_type == "image" or str(dataset.get("h3_target_mode", "av")) == "audio":
        activation_gb = 0.0
    else:
        width = max(_coerce_int(dataset.get("resolution_w", 768), 768), 64)
        height = max(_coerce_int(dataset.get("resolution_h", 512), 512), 64)
        frames = max(_coerce_int(dataset.get("target_frames", 124), 124), 1)
        video_latents, _ = _h3_temporal_latents(frames)
        target_rows = video_latents * _h3_spatial_rows(width, height)
        # Student rows are retained in BF16. The FP32 two-layer projector keeps
        # its input/hidden/output tensors for backward; backbone mode projects
        # before spatial pooling, so it has the larger row-dependent footprint.
        projected_rows = target_rows if mode == "backbone" else target_rows
        activation_bytes = target_rows * hidden * 2 + projected_rows * (hidden * 2 + output) * 4
        activation_gb = activation_bytes / (1024**3)
    # FP32 projector parameters and gradients are resident in addition to the
    # main LoRA accounting above.
    return 2 * parameter_gb + state_gb + activation_gb


@lru_cache(maxsize=1)
def _detect_local_gpu_name() -> str | None:
    """Return the selected CUDA GPU name without importing a CUDA runtime."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not names:
            return None
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
        if visible.isdigit() and int(visible) < len(names):
            return names[int(visible)]
        return names[0]
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_time_coefficient(gpu_name: str | None) -> float:
    """Synthetic wall-time multiplier for common CUDA GPU families.

    These coefficients intentionally combine tensor throughput and memory
    bandwidth instead of using peak FLOPS alone. They are approximate and are
    ordered from specific product matches to conservative family fallbacks.
    """
    name = (gpu_name or "").lower()
    coefficients = (
        (("b200",), 0.55),
        (("b100",), 0.65),
        (("h200",), 0.90),
        (("h100",), 1.00),
        (("rtx pro 6000 blackwell", "rtx 6000 blackwell"), 0.85),
        (("geforce rtx 5090", "rtx 5090"), 0.80),
        (("a100-sxm", "a100 sxm"), 1.45),
        (("a100-pcie", "a100 pcie"), 1.65),
        (("a100",), 1.55),
        (("l40s",), 1.85),
        (("rtx 6000 ada",), 1.90),
        (("geforce rtx 4090", "rtx 4090"), 1.75),
        (("l40",), 2.05),
        (("geforce rtx 5080", "rtx 5080"), 1.55),
        (("geforce rtx 4080", "rtx 4080"), 2.30),
        (("a6000",), 2.75),
        (("geforce rtx 3090", "rtx 3090"), 2.85),
        (("geforce rtx 3080", "rtx 3080"), 3.55),
        (("a40",), 3.00),
        (("l4",), 4.10),
    )
    for aliases, coefficient in coefficients:
        if any(alias in name for alias in aliases):
            return coefficient
    if "blackwell" in name:
        return 0.90
    if "hopper" in name:
        return 1.05
    if "ada" in name:
        return 2.20
    if "ampere" in name:
        return 2.80
    return 1.75 if gpu_name else 1.00


def _estimate_training_step_time_sec(config: dict, gpu_name: str | None = None) -> float | None:
    """Estimate wall-clock seconds per optimizer step from the current config."""
    try:
        training = config.get("training", {})
        dataset = _first_training_dataset(config)

        if training.get("model_type") == "minimax_h3":
            caching = config.get("caching", {})
            datasets = _training_datasets(config)
            # Directory cardinality is unavailable without scanning every
            # source. Use the heaviest configured row: it gives users a safe
            # iteration estimate and matches the bucket that defines peak VRAM.
            dataset = max(datasets, key=lambda row: _h3_dataset_rows(training, caching, row))
            return _estimate_h3_training_step_time_sec(training, dataset, caching) * _gpu_time_coefficient(gpu_name)

        mode = str(training.get("ltx2_mode", "video")).lower()
        res_w = max(_coerce_int(dataset.get("resolution_w", 768), 768), 64)
        res_h = max(_coerce_int(dataset.get("resolution_h", 512), 512), 64)
        frames = max(_coerce_int(dataset.get("target_frames", 33), 33), 1)
        batch_size = max(_coerce_int(dataset.get("batch_size", training.get("train_batch_size", 1)), 1), 1)
        grad_accum = max(_coerce_int(training.get("gradient_accumulation_steps", 1), 1), 1)

        pixel_frame_scale = (res_w * res_h * frames) / (768 * 512 * 33)
        step_time = 2.35 * max(pixel_frame_scale, 0.25) * batch_size * grad_accum

        if mode == "av":
            step_time *= 1.35
        elif mode == "audio":
            step_time *= 0.55

        blocks_to_checkpoint = _coerce_int(training.get("blocks_to_checkpoint", -1), -1)
        grad_ckpt = blocks_to_checkpoint != 0 and bool(
            training.get("gradient_checkpointing", True) or training.get("blockwise_checkpointing")
        )
        if grad_ckpt:
            step_time *= 1.20
        if training.get("blockwise_checkpointing"):
            step_time *= 1.08
        if training.get("gradient_checkpointing_cpu_offload"):
            step_time *= 1.15
        if training.get("fp8_w8a8"):
            step_time *= 0.88
        elif training.get("fp8_base"):
            step_time *= 0.95
        if training.get("nf4_base"):
            step_time *= 1.08
        if training.get("self_flow"):
            step_time *= 1.22
        if training.get("blank_preservation") or training.get("dop"):
            step_time *= 1.12
        if training.get("audio_dop"):
            step_time *= 1.10
        if training.get("prior_divergence"):
            step_time *= 1.05
        if training.get("crepa"):
            step_time *= 1.08

        sample_every_n_steps = _coerce_int(training.get("sample_every_n_steps", 0), 0)
        if sample_every_n_steps:
            step_time += 5.0 / sample_every_n_steps

        return max(step_time, 0.05)
    except Exception as e:
        logger.debug(f"Failed to estimate training step time: {e}")
        return None


def _estimate_h3_training_step_time_sec(training: dict, dataset: dict, caching: dict | None = None) -> float:
    """Estimate an H3 LoRA optimizer step from the normalized reference curve.

    The 832x480x124 reference is calibrated from real BF16 LoRA runs. Factors
    describe average steady-state work; checkpoint loading and saving are not
    part of an optimizer step. Extra no-grad teacher forwards affect time but
    are deliberately not treated as equivalent activation-memory multipliers.
    """
    batch_size = max(_coerce_int(dataset.get("batch_size", training.get("train_batch_size", 1)), 1), 1)
    grad_accum = max(_coerce_int(training.get("gradient_accumulation_steps", 1), 1), 1)

    packed_rows = _h3_dataset_rows(training, caching or {}, dataset)
    reference_rows = _h3_dataset_rows(
        {},
        {"h3_task": "t2va"},
        {"type": "video", "h3_target_mode": "av", "resolution_w": 832, "resolution_h": 480, "target_frames": 124},
    )
    work_scale = packed_rows / reference_rows
    # 10.6 s is the fitted unswapped compute intercept. The fully checkpointed,
    # reusable-offload, pinned 48-block reference evaluates to ~22 s/step.
    step_time = 10.6 * max(work_scale, 0.04) ** 0.90 * batch_size * grad_accum

    checkpoint_blocks = _coerce_int(training.get("h3_gradient_checkpointing_blocks", -1), -1)
    uses_checkpointing = bool(training.get("gradient_checkpointing", True)) and checkpoint_blocks != 0
    if uses_checkpointing:
        checkpoint_fraction = 1.0 if checkpoint_blocks < 0 else min(max(checkpoint_blocks, 0), 50) / 50
        step_time *= 1.0 + 0.35 * checkpoint_fraction

    if training.get("gradient_checkpointing_cpu_offload"):
        step_time *= 1.12
        if training.get("h3_reusable_activation_offload"):
            step_time *= 0.92

    blocks_to_swap = min(max(_coerce_int(training.get("blocks_to_swap", 0), 0), 0), 50)
    if blocks_to_swap:
        ring_size = max(_coerce_int(training.get("block_swap_ring_size", 2), 2), 1)
        if training.get("block_swap_h2d_only") and training.get("use_pinned_memory_for_block_swap"):
            # Direction-aware prefetch overlaps almost all transfer through 44
            # blocks. The measured 44->48 tail crossed into a transfer-bound
            # regime: 13.7 -> 22.0 s while saving 4.81 GiB.
            overlap_limit = 44 + min(max(ring_size - 2, 0), 2)
            if str(training.get("block_swap_granularity", "block")) == "layer":
                overlap_limit += 1
            step_time *= 1.0 + 0.15 * max(blocks_to_swap - overlap_limit, 0)
        else:
            # Bidirectional or pageable transfers cannot use the calibrated
            # H2D-only overlap window as effectively.
            transfer_cost = 0.010 if training.get("block_swap_h2d_only") else 0.016
            step_time *= 1.0 + blocks_to_swap * transfer_cost

    if training.get("h3_attn_auto_dispatch"):
        step_time *= 0.94
    if training.get("h3_fused_qk_norm_rope"):
        step_time *= 0.95
    if training.get("h3_convrot_int8"):
        step_time *= 0.91 if training.get("h3_convrot_int8_fwd", "int8") == "int8" else 0.98
        if training.get("h3_convrot_int8_bwd", "bf16") == "int8":
            step_time *= 0.94
    elif training.get("fp8_base") or training.get("fp8_scaled"):
        step_time *= 0.96

    # A frozen-base preservation pass is forward-only. Its Bernoulli
    # probability changes average time, while peak VRAM remains unchanged.
    if _coerce_float(training.get("h3_base_preservation_loss_weight", 0), 0.0) > 0:
        probability = min(max(_coerce_float(training.get("h3_base_preservation_probability", 1), 1.0), 0.0), 1.0)
        # Paired p=0.5 steps measured ~23% compute-only slowdown when active;
        # cold cache I/O can hide part of it, so use the compute-side cost.
        step_time *= 1.0 + 0.23 * probability

    # Guidance distillation performs one additional no-grad conditional forward,
    # the same execution shape as an active preservation teacher pass.
    if training.get("h3_guidance_distillation_scale") is not None or training.get("h3_guidance_scale_range"):
        caption_dropout = min(max(_coerce_float(training.get("h3_caption_dropout_rate", 0), 0.0), 0.0), 1.0)
        guidance_probability = min(max(_coerce_float(training.get("h3_guidance_distillation_probability", 1), 1.0), 0.0), 1.0)
        step_time *= 1.0 + 0.23 * (1.0 - caption_dropout) * guidance_probability

    sample_every_n_steps = _coerce_int(training.get("sample_every_n_steps", 0), 0)
    if sample_every_n_steps:
        # Sampling cost is amortized; this remains deliberately conservative
        # until sampler benchmarks are available for each output shape.
        sample_steps = max(_coerce_int(training.get("sample_steps", 1), 1), 1)
        step_time += (2.0 * sample_steps * max(work_scale, 0.1)) / sample_every_n_steps

    return max(step_time, 0.05)


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
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            files = [f for f in path.rglob("*") if f.suffix.lower() in video_exts]
            return DatasetStats(
                total_items=len(files),
                video_items=len(files),
                audio_items=0,
                avg_resolution=None,
                avg_frames=None,
                max_resolution=None,
                max_frames=None,
            )

        # Parse TOML to get subsets
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # Fallback for older Python
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)

        total_items = 0
        video_items = 0
        audio_items = 0
        resolutions = []
        frames = []

        for subset in config.get("subsets", []):
            video_dir = subset.get("video_dir", "")
            if not video_dir:
                continue

            subset_path = Path(video_dir) if os.path.isabs(video_dir) else path / video_dir
            if not subset_path.exists():
                continue

            # Count videos in this subset
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            subset_videos = [f for f in subset_path.rglob("*") if f.suffix.lower() in video_exts]

            num_videos = len(subset_videos)
            total_items += num_videos

            # Check if audio subset
            is_audio = subset.get("is_audio", False) or "audio" in video_dir.lower()
            if is_audio:
                audio_items += num_videos
            else:
                video_items += num_videos

            # Try to get resolution/frames from metadata if available
            metadata_path = subset_path / ".metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path) as mf:
                        meta = json.load(mf)
                        if "resolution" in meta:
                            resolutions.append(tuple(meta["resolution"]))
                        if "frames" in meta:
                            frames.append(meta["frames"])
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
            max_frames=max_frames,
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
        training = config.get("training", {})
        dataset = _first_training_dataset(config)

        # Batch size
        batch_size = max(_coerce_int(dataset.get("batch_size", training.get("train_batch_size", 1)), 1), 1)
        grad_accum = max(_coerce_int(training.get("gradient_accumulation_steps", 1), 1), 1)
        effective_batch_size = batch_size * grad_accum

        # Steps per epoch
        steps_per_epoch = None
        if dataset_stats and dataset_stats.total_items > 0:
            steps_per_epoch = max(1, dataset_stats.total_items // effective_batch_size)

        # Total epochs
        max_steps = _coerce_int(training.get("max_train_steps", 0), 0)
        total_epochs = None
        if max_steps and steps_per_epoch:
            total_epochs = max_steps / steps_per_epoch

        # Estimated time from the same shape/config heuristic used for iteration time.
        estimated_time_hours = None
        gpu_name = _detect_local_gpu_name() if training.get("model_type") == "minimax_h3" else None
        estimated_step_time_sec = _estimate_training_step_time_sec(config, gpu_name)
        estimated_steps_per_sec = (1.0 / estimated_step_time_sec) if estimated_step_time_sec else None
        if max_steps:
            estimated_time_hours = (max_steps * (estimated_step_time_sec or 0)) / 3600

        # Checkpoint size
        network_dim = max(_coerce_int(training.get("network_dim", 16), 16), 1)
        # Rough estimate: LoRA size depends on rank and target modules
        # LTX2 full LoRA is roughly: dim * 2 * hidden_dim * num_layers * 4 bytes
        # For dim=16, roughly 50-100MB
        checkpoint_size_mb = network_dim * 5  # Very rough estimate

        # Total checkpoints
        save_every_n_steps = _coerce_int(training.get("save_every_n_steps", 0), 0)
        save_every_n_epochs = _coerce_int(training.get("save_every_n_epochs", 0), 0)
        total_checkpoints = 1  # Final checkpoint

        if save_every_n_steps and max_steps:
            total_checkpoints += max_steps // save_every_n_steps
        elif save_every_n_epochs and total_epochs:
            total_checkpoints += int(total_epochs) // save_every_n_epochs

        # Apply keep_last limits
        keep_last_steps = _coerce_int(training.get("save_last_n_steps", 0), 0)
        keep_last_epochs = _coerce_int(training.get("save_last_n_epochs", 0), 0)
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
            estimated_time_source=(
                f"Hardware-adjusted estimate ({gpu_name})"
                if estimated_step_time_sec and training.get("model_type") == "minimax_h3" and gpu_name
                else "Hardware-adjusted estimate"
                if estimated_step_time_sec and training.get("model_type") == "minimax_h3"
                else "heuristic"
                if estimated_step_time_sec
                else None
            ),
            checkpoint_size_mb=checkpoint_size_mb,
            total_checkpoints=total_checkpoints,
            total_storage_gb=total_storage_gb,
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
        training = config.get("training", {})
        if training.get("model_type") == "minimax_h3":
            caching = config.get("caching", {})
            datasets = _training_datasets(config)
            dataset = max(datasets, key=lambda row: _h3_dataset_rows(training, caching, row))
            return _calculate_h3_vram_stats(training, caching, dataset)
        ds = _first_training_dataset(config)

        # ── DiT weights ──
        ltx_version = str(training.get("ltx_version", "2.3"))
        dit_bf16 = 42.0 if ltx_version == "2.3" else 39.0
        is_fp8 = bool(training.get("fp8_base"))
        is_w8a8 = bool(training.get("fp8_w8a8"))
        is_nf4 = bool(training.get("nf4_base"))
        dit_base = (dit_bf16 / 4) if is_nf4 else (dit_bf16 / 2) if is_fp8 else dit_bf16

        total_blocks = 48
        blocks_to_checkpoint = _coerce_int(training.get("blocks_to_checkpoint", -1), -1)
        blockwise = bool(training.get("blockwise_checkpointing")) and blocks_to_checkpoint != 0
        checkpointed_blocks = 0
        if blocks_to_checkpoint != 0:
            checkpointed_blocks = total_blocks if blocks_to_checkpoint < 0 else min(max(blocks_to_checkpoint, 0), total_blocks)
        blocks_to_swap = min(max(_coerce_int(training.get("blocks_to_swap", 0), 0), 0), total_blocks - 1)
        block_size_gb = dit_base / total_blocks
        swap_savings = blocks_to_swap * block_size_gb * 0.95
        resident_blocks_after_swap = max(total_blocks - (swap_savings / max(block_size_gb, 0.0001)), 0)
        blockwise_weight_savings = min(checkpointed_blocks, resident_blocks_after_swap) * block_size_gb * 0.80 if blockwise else 0.0
        model_size_gb = max(dit_base - swap_savings - blockwise_weight_savings, 1.0)

        # ── LoRA weights ──
        rank = max(_coerce_int(training.get("network_dim", 16), 16), 1)
        mode = str(training.get("ltx2_mode", "video")).lower()
        is_av = mode == "av"
        lora_base_per_rank = (12.75 if is_av else 6.0) / 1024  # GB per rank
        preset_mult = {
            "t2v": 1.0,
            "v2v": 1.44,
            "video_sa": 0.37,
            "video_sa_ff": 0.56,
            "video_sa_ca_ff": 0.74,
            "audio": 0.37,
            "audio_v2a": 0.52,
            "audio_ref_ic": 0.63,
            "av_ic": 1.44,
            "video_ref_only_av": 1.44,
            "full": 2.1,
        }.get(training.get("lora_target_preset"), 1.0)
        lora_size_gb = rank * lora_base_per_rank * preset_mult

        # ── Optimizer states ──
        lora_param_count = lora_size_gb * (1024**3) / 2  # bf16 -> count
        opt_type = str(training.get("optimizer_type", "adamw8bit")).lower()
        is_8bit = "8bit" in opt_type
        is_4bit = "4bit" in opt_type
        is_fp8_optim = "fp8" in opt_type
        is_kahan = opt_type.startswith("optimi_") or opt_type.startswith("torchoptimi_") or opt_type.startswith("optimi.")
        is_sf = "schedulefree" in opt_type or opt_type == "automagic"
        if is_4bit:
            opt_bytes = 5
        elif is_8bit or is_fp8_optim:
            opt_bytes = 6
        elif is_kahan:
            opt_bytes = 10
        else:
            opt_bytes = 14 if is_sf else 12
        optimizer_size_gb = (lora_param_count * opt_bytes) / (1024**3)

        # ── Activations ──
        res_w = max(_coerce_int(ds.get("resolution_w", 768), 768), 64)
        res_h = max(_coerce_int(ds.get("resolution_h", 512), 512), 64)
        frames = max(_coerce_int(ds.get("target_frames", 33), 33), 1)
        batch_size = max(_coerce_int(ds.get("batch_size", 1), 1), 1)
        memory_batch_size = 1 if training.get("model_type") == "minimax_h3" else batch_size

        # Correct VAE compression factors
        latent_f = max(1, (frames - 1) // 8 + 1)
        latent_h = max(1, res_h // 32)
        latent_w = max(1, res_w // 32)
        seq_len = latent_f * latent_h * latent_w

        hidden_dim = 2048 if mode == "audio" else 4096
        bytes_per_val = 1 if is_w8a8 else 2
        grad_ckpt = bool(training.get("gradient_checkpointing", True) or blockwise) and blocks_to_checkpoint != 0

        if not grad_ckpt:
            activation_units = total_blocks * 10
        elif blockwise:
            standard_blocks = total_blocks - checkpointed_blocks
            activation_units = max(2, standard_blocks + 2)
        else:
            activation_units = total_blocks * 2
        activations_gb = (memory_batch_size * seq_len * hidden_dim * bytes_per_val * activation_units) / (1024**3)
        if is_av:
            activations_gb *= 1.25
        if _coerce_int(training.get("ffn_chunk_size", 0), 0) > 0:
            activations_gb *= 0.90
        if training.get("split_attn_mode") or training.get("split_attn_target"):
            activations_gb *= 0.92
        if training.get("gradient_checkpointing_cpu_offload") and grad_ckpt:
            activations_gb *= 0.35

        # Fixed buffers
        latent_bytes = memory_batch_size * 128 * latent_f * latent_h * latent_w * 2 * 2
        text_bytes = memory_batch_size * 256 * (7680 if is_av else 3840) * 2
        buffer_gb = (latent_bytes + text_bytes) / (1024**3) + 0.5
        if training.get("img_in_txt_in_offloading"):
            buffer_gb = max(0.2, buffer_gb - 0.3)
        activations_gb = max(0.3, activations_gb + buffer_gb)

        # ── Gradients ──
        grads_gb = lora_size_gb

        # ── Gradient accumulation ──
        grad_accum = max(_coerce_int(training.get("gradient_accumulation_steps", 1), 1), 1)
        grad_accum_gb = grads_gb * 0.4 if grad_accum > 1 else 0

        # ── Preservation / DOP ──
        preservation_gb = 0
        if training.get("blank_preservation") or training.get("dop"):
            preservation_gb += activations_gb * 0.35
        if training.get("audio_dop"):
            preservation_gb += activations_gb * 0.35
        if training.get("prior_divergence"):
            preservation_gb += activations_gb * 0.15

        # ── Self-Flow ──
        self_flow_gb = 0
        if training.get("self_flow"):
            teacher_mode = str(training.get("self_flow_teacher_mode", "ema")).lower()
            if teacher_mode == "ema":
                self_flow_gb += lora_size_gb
            elif teacher_mode == "partial_ema":
                self_flow_gb += max(lora_size_gb / total_blocks, 0.01)

            has_audio_projector = mode in {"av", "audio"} and _coerce_float(training.get("self_flow_lambda_audio", 0.0), 0.0) > 0.0
            self_flow_gb += 0.03 if has_audio_projector else 0.02

            feature_factor = 0.03 if training.get("self_flow_offload_teacher_features") else 0.10
            self_flow_gb += activations_gb * feature_factor

        # ── CREPA ──
        crepa_gb = 0
        if training.get("crepa"):
            crepa_gb = 0.08 if str(training.get("crepa_mode", "backbone")) == "dino" else 0.15

        peak_training_gb = (
            model_size_gb
            + lora_size_gb
            + optimizer_size_gb
            + grads_gb
            + activations_gb
            + grad_accum_gb
            + preservation_gb
            + self_flow_gb
            + crepa_gb
        )

        # Sampling VRAM (VAE loaded, lighter activations)
        peak_sampling_gb = model_size_gb + 0.3 + (activations_gb * 0.3)
        if training.get("sample_with_offloading"):
            peak_sampling_gb *= 0.6

        overhead_gb = grad_accum_gb + preservation_gb + self_flow_gb + crepa_gb
        breakdown = {
            "model": round(model_size_gb, 2),
            "lora": round(lora_size_gb, 2),
            "optimizer": round(optimizer_size_gb, 2),
            "gradients": round(grads_gb, 2),
            "activations": round(activations_gb, 2),
            "overhead": round(overhead_gb, 2),
        }

        return VRAMStats(
            peak_training_gb=round(peak_training_gb, 2),
            peak_sampling_gb=round(peak_sampling_gb, 2),
            model_size_gb=round(model_size_gb, 2),
            optimizer_size_gb=round(optimizer_size_gb, 2),
            activations_gb=round(activations_gb, 2),
            breakdown=breakdown,
        )

    except Exception as e:
        logger.warning(f"Failed to calculate VRAM stats: {e}")
        return None


def _calculate_h3_vram_stats(training: dict, caching: dict, dataset: dict) -> VRAMStats:
    """Estimate H3 LoRA peak residency from its actual packed architecture."""
    dit_base = _h3_base_size_gb(training)
    total_blocks = 50
    max_swapped = 50 if training.get("block_swap_h2d_only") and training.get("block_swap_granularity") == "layer" else 48
    blocks_to_swap = min(max(_coerce_int(training.get("blocks_to_swap", 0), 0), 0), max_swapped)
    swap_savings = blocks_to_swap * (dit_base / total_blocks) * 0.95
    model_size_gb = max(dit_base - swap_savings, 1.0)

    lora_size_gb = _h3_lora_size_gb(training)
    lora_param_count = lora_size_gb * (1024**3) / 2
    opt_type = str(training.get("optimizer_type", "adamw8bit")).lower()
    if "4bit" in opt_type:
        opt_bytes = 5
    elif "8bit" in opt_type or "fp8" in opt_type:
        opt_bytes = 6
    elif opt_type.startswith(("optimi_", "torchoptimi_", "optimi.")):
        opt_bytes = 10
    elif "schedulefree" in opt_type or opt_type == "automagic":
        opt_bytes = 14
    else:
        opt_bytes = 12
    optimizer_size_gb = lora_param_count * opt_bytes / (1024**3)
    grads_gb = lora_size_gb

    sequence_rows = _h3_dataset_rows(training, caching, dataset)
    checkpoint_value = training.get("h3_gradient_checkpointing_blocks")
    checkpoint_blocks = -1 if checkpoint_value in (None, "") else _coerce_int(checkpoint_value, -1)
    uses_checkpointing = bool(training.get("gradient_checkpointing", True)) and checkpoint_blocks != 0
    checkpointed = 0 if not uses_checkpointing else total_blocks if checkpoint_blocks < 0 else min(checkpoint_blocks, total_blocks)
    standard = total_blocks - checkpointed
    activation_units = total_blocks * 10 if not uses_checkpointing else checkpointed * 2 + standard * 10
    activations_gb = sequence_rows * 5376 * 2 * activation_units / (1024**3)
    if training.get("gradient_checkpointing_cpu_offload") and uses_checkpointing:
        activations_gb *= 0.35

    # Latents, packed projections, attention workspaces, and allocator slack.
    fixed_buffers_gb = 2.5
    activations_gb = max(0.3, activations_gb + fixed_buffers_gb)
    grad_accum = max(_coerce_int(training.get("gradient_accumulation_steps", 1), 1), 1)
    grad_accum_gb = grads_gb * 0.4 if grad_accum > 1 else 0.0
    crepa_gb = _h3_crepa_memory_gb(training, dataset)

    # Guidance and preservation are sequential no-grad forwards. They change
    # average time, not the graph high-water mark.
    overhead_gb = grad_accum_gb + crepa_gb
    peak_training_gb = model_size_gb + lora_size_gb + optimizer_size_gb + grads_gb + activations_gb + overhead_gb
    peak_sampling_gb = model_size_gb + 0.3 + activations_gb * 0.3
    return VRAMStats(
        peak_training_gb=round(peak_training_gb, 2),
        peak_sampling_gb=round(peak_sampling_gb, 2),
        model_size_gb=round(model_size_gb, 2),
        optimizer_size_gb=round(optimizer_size_gb, 2),
        activations_gb=round(activations_gb, 2),
        breakdown={
            "model": round(model_size_gb, 2),
            "lora": round(lora_size_gb, 2),
            "optimizer": round(optimizer_size_gb, 2),
            "gradients": round(grads_gb, 2),
            "activations": round(activations_gb, 2),
            "overhead": round(overhead_gb, 2),
        },
    )


@router.get("", response_model=ProjectStats)
async def get_project_stats(request: Request):
    """Get comprehensive project statistics."""
    config = request.app.state.project_config
    if not config:
        return ProjectStats(dataset=None, training=None, vram=None)

    config_dict = config.model_dump()
    try:
        pm = request.app.state.process_manager
        if pm.get_status("full_finetune").get("state") in {"running", "stopping", "finished", "error"}:
            config_dict["training"] = config_dict.get("full_finetune", {})
    except Exception:
        pass

    # Dataset stats
    dataset_stats = None
    dataset_config = config_dict.get("dataset", {})
    datasets = dataset_config.get("datasets", []) if dataset_config else []

    # Scan first dataset entry
    if datasets and len(datasets) > 0:
        first_dataset = datasets[0]
        dataset_dir = first_dataset.get("directory", "")
        if dataset_dir:
            dataset_stats = _scan_dataset(dataset_dir)

    # Training stats (only if we have dataset info)
    training_stats = None
    if dataset_stats:
        training_stats = _calculate_training_stats(config_dict, dataset_stats)

    # VRAM stats (can calculate without dataset)
    vram_stats = _calculate_vram_stats(config_dict)

    return ProjectStats(dataset=dataset_stats, training=training_stats, vram=vram_stats)
