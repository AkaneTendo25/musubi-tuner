"""Kandinsky 6 video super-resolution pipeline.

The SR model implementation is packaged under :mod:`musubi_tuner.kandinsky6.runtime.sr`. This module
owns input preparation,
scale-aware tiling, batching, and stitching, while the SR model owns the
denoising and VAE decode.

Both public paths return the same frame layout as the K6 video pipeline:
``(batch, channels, frames, height, width)`` uint8 frames.  A raw latent is
expected in ``(frames, channels, height, width)`` layout and is unscaled.
Pixel inputs use the same layout and are uint8 (float inputs are accepted when
already in ``[0, 255]``).

Sources of any size are accepted. The latent-upscaler path needs the source
on the VAE stride and no smaller than one tile; the pipeline pads it there
(see :mod:`musubi_tuner.kandinsky6.runtime.sr.pipeline.source_padding`) and crops the result back
to ``source * scale``, so callers see exactly the scale they asked for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger
from torch.nn import functional

from musubi_tuner.kandinsky6.runtime.sr import constants as _sr_constants
from musubi_tuner.kandinsky6.runtime.sr.constants import RESOLUTIONS
from musubi_tuner.kandinsky6.runtime.sr.core.algo.latent_upscaler import (
    latent_upscaler_for_scale,
    run_latent_upscaler,
)
from musubi_tuner.kandinsky6.runtime.sr.core.algo.mux import mux_video_audio
from musubi_tuner.kandinsky6.runtime.sr.core.algo.tiling_utils import (
    TileGrid,
    extract_all_tiles,
    stitch_tiles_hanning,
)
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.vae_io import cast_to_module_dtype
from musubi_tuner.kandinsky6.runtime.sr.core.types import SRPipelineOutput
from musubi_tuner.kandinsky6.runtime.sr.core.utils.offload import NoOpOffload, OffloadHandle
from musubi_tuner.kandinsky6.runtime.sr.pipeline.config import ResolutionScale, TilingScale
from musubi_tuner.kandinsky6.runtime.sr.pipeline.progress import CompositeProgress, ProgressReporter, TqdmProgress
from musubi_tuner.kandinsky6.runtime.sr.pipeline.source_padding import SourcePadding, source_padding
from musubi_tuner.kandinsky6.runtime.sr.pipeline.tile_grid import compute_tile_grid_even
from musubi_tuner.kandinsky6.runtime.sr.pipeline.upscale_utils import pre_upscale_video, resolve_scale_request

VAE_SPATIAL_FACTOR = int(_sr_constants.VAE_SPATIAL_FACTOR)


VIDEO_RANK = 4
VIDEO_BATCH_RANK = 5
MIN_NUM_STEPS = 2


def _release_sr_modules_after_call(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        finally:
            self.offload.release("dit", "vae", "latent_upscaler")

    return wrapped


@dataclass
class RunConfig:
    """Parameters for one tiled SR run."""

    device: str
    num_steps: int = 5
    seed: int = 42
    # Minimum overlap between neighbouring tiles, fraction of the tile size.
    overlap: float = 0.20
    tiles_batch_size: int = 1
    resolution_scale: TilingScale = 4

    def __post_init__(self) -> None:
        if self.num_steps < MIN_NUM_STEPS:
            raise ValueError(f"num_steps must be at least {MIN_NUM_STEPS}")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must satisfy 0 <= overlap < 1")
        if self.tiles_batch_size <= 0:
            raise ValueError("tiles_batch_size must be positive")
        if self.resolution_scale not in (2, 4):
            raise ValueError("resolution_scale must be 2 or 4")


@dataclass
class SRParams:
    """SR sampling parameters extracted from the SR training config."""

    scale_factor: dict[int, list[float]]
    visual_size: list[int]
    scheduler_scale: float = 5.0
    lq_noise_scale: float = 0.7
    lq_noise_type: str = "ddpm"
    lq_channel_noise_scale: float = 0.0
    cap_noise_timestep: bool = False
    fps: int = 24


@dataclass
class SRComponents:
    """The weighted components and config consumed by the SR sampler."""

    dit: torch.nn.Module
    vae: torch.nn.Module
    latent_upscaler: torch.nn.Module | Any | None
    sr_params: SRParams
    cached_text_embeds: dict[str, torch.Tensor] | None = None
    sampler: Callable[..., torch.Tensor] | None = None
    spatial_factor: int | None = None


def _closest_base_resolution(h: int, w: int, visual_size: int) -> tuple[int, int]:
    if visual_size not in RESOLUTIONS:
        raise ValueError(f"Unsupported SR visual_size={visual_size}; known sizes: {sorted(RESOLUTIONS)}")
    ratio = w / h if h else 1.0
    return min(RESOLUTIONS[visual_size], key=lambda hw: abs(hw[1] / hw[0] - ratio))


def _tile_geometry(  # noqa: PLR0913
    h: int,
    w: int,
    visual_size: int,
    scale: int,
    overlap: float,
    spatial_factor: int = VAE_SPATIAL_FACTOR,
) -> tuple[tuple[int, int], tuple[int, int], TileGrid]:
    """Base resolution, tile size and the tile grid (fewest tiles, uniform overlap >= ``overlap``)."""
    base_h, base_w = _closest_base_resolution(h, w, visual_size)
    if base_h % scale or base_w % scale:
        raise ValueError(f"resolution_scale={scale} does not divide the base resolution {base_h}x{base_w} exactly")
    tile_hw = (base_h // scale, base_w // scale)
    grid = compute_tile_grid_even(h, w, tile_hw, overlap, spatial_factor)
    return (base_h, base_w), tile_hw, grid


def latent_path_padding(components: Any, run_config: RunConfig, source_hw: tuple[int, int]) -> SourcePadding:
    """The padding the latent-upscaler path adds to a ``source_hw`` clip: VAE stride and at least one tile.

    Args:
        components: Loaded SR components (base resolutions, VAE factor).
        run_config: The run's tiling settings (the tile is ``base // scale``).
        source_hw: Source size in pixels (after any pre-upscale); for a raw
            latent, its size times the VAE factor.
    """
    visual_size = _visual_size(_component_params(components))
    spatial_factor = _spatial_factor(components)

    def tile_hw_for(height: int, width: int) -> tuple[int, int]:
        return _tile_geometry(height, width, visual_size, run_config.resolution_scale, run_config.overlap, spatial_factor)[1]

    return source_padding(source_hw[0], source_hw[1], spatial_factor, tile_hw_for)


def denoise_progress_total(components: SRComponents, run_config: RunConfig, source_hw: tuple[int, int]) -> tuple[int, int]:
    """Return ``(total_tiles, steps_per_tile)`` of a run: the work a progress reporter counts.

    Args:
        components: Loaded SR components (DiT sampler settings, VAE factor).
        run_config: The run's tiling settings.
        source_hw: Source video size in pixels (after any pre-upscale); with
            ``resolution_scale`` it fixes the tile count.
    """
    total_tiles = _tile_geometry(
        source_hw[0],
        source_hw[1],
        _visual_size(components.sr_params),
        run_config.resolution_scale,
        run_config.overlap,
        _spatial_factor(components),
    )[2].total_tiles
    piflow_params = getattr(components.dit, "piflow_params", None)
    if isinstance(piflow_params, Mapping):
        steps = int(piflow_params["nfe"])
    else:
        steps = int(getattr(piflow_params, "nfe", run_config.num_steps - 1))
    return total_tiles, steps


def _progress_reporter(show_progress: bool, progress: ProgressReporter | None) -> ProgressReporter | None:
    """Combine the console bar and the caller's reporter into one, or ``None`` when neither is wanted."""
    reporters: list[ProgressReporter] = [TqdmProgress()] if show_progress else []
    if progress is not None:
        reporters.append(progress)
    if not reporters:
        return None
    return reporters[0] if len(reporters) == 1 else CompositeProgress(reporters)


def _spatial_factor(components: Any) -> int:
    explicit = getattr(components, "spatial_factor", None)
    if explicit is not None:
        return int(explicit)
    vae = getattr(components, "vae", None)
    config = getattr(vae, "config", None)
    return int(
        getattr(vae, "spatial_factor", None)
        or getattr(config, "spatial_factor", None)
        or getattr(_sr_constants, "VAE_SPATIAL_FACTOR", None)
        or VAE_SPATIAL_FACTOR
    )


def _component_params(components: Any) -> Any:
    params = getattr(components, "sr_params", None)
    if params is None:
        raise ValueError("SR components must provide sr_params")
    return params


def _visual_size(params: Any) -> int:
    visual_size = getattr(params, "visual_size", None)
    if isinstance(visual_size, int):
        return visual_size
    if not visual_size:
        raise ValueError("SR params must provide a non-empty visual_size")
    return int(visual_size[0])


def _scale_factor(params: Any, visual_size: int) -> tuple[float, ...]:
    values = getattr(params, "scale_factor", None)
    if isinstance(values, Mapping):
        try:
            values = values[visual_size]
        except KeyError:
            # YAML keeps numeric resolution keys as integers; JSON converts
            # object keys to strings when the config is stored in a bundle.
            values = values[str(visual_size)]
    return tuple(float(value) for value in values)


def _extract_latent(result: Any) -> torch.Tensor:
    if isinstance(result, tuple):
        result = result[0]
    distribution = getattr(result, "latent_dist", None)
    if distribution is not None:
        result = distribution.sample()
    if not isinstance(result, torch.Tensor):
        raise TypeError("VAE encode must return a Tensor, (Tensor, ...), or latent_dist output")
    return result


def _video_samples(value: torch.Tensor | np.ndarray | list[torch.Tensor]) -> list[torch.Tensor]:
    """Normalize video input to equal-shaped ``[T,C,H,W]`` samples."""
    if isinstance(value, (list, tuple)):
        samples = [_video_sample(item) for item in value]
    else:
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim == VIDEO_RANK:
            samples = [_video_sample(tensor)]
        elif tensor.ndim == VIDEO_BATCH_RANK:
            if tensor.shape[1] in (1, 3, 4):
                samples = [item.permute(1, 0, 2, 3) for item in tensor]
            elif tensor.shape[2] in (1, 3, 4):
                samples = list(tensor)
            elif tensor.shape[-1] in (1, 3, 4):
                samples = [item.permute(0, 3, 1, 2) for item in tensor]
            else:
                raise ValueError("5D video must be [B,C,T,H,W], [B,T,C,H,W], or [B,T,H,W,C]")
        else:
            raise ValueError(f"video must have rank 4 or 5, got shape {tuple(tensor.shape)}")
    _validate_equal_shapes(samples, "video")
    return samples


def _video_sample(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != VIDEO_RANK:
        raise ValueError(f"video sample must have rank 4, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] in (1, 3, 4):
        return tensor
    if tensor.shape[-1] in (1, 3, 4):
        return tensor.permute(0, 3, 1, 2)
    if tensor.shape[0] in (1, 3, 4):
        return tensor.permute(1, 0, 2, 3)
    raise ValueError("4D video must be [T,C,H,W], [T,H,W,C], or [C,T,H,W]")


def _latent_samples(
    value: torch.Tensor | np.ndarray | list[torch.Tensor],
    channels: int,
) -> list[torch.Tensor]:
    """Normalize latent input to equal-shaped ``[T,C,H,W]`` samples."""
    if isinstance(value, (list, tuple)):
        samples = [_latent_sample(item, channels) for item in value]
    else:
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim == VIDEO_RANK:
            samples = [_latent_sample(tensor, channels)]
        elif tensor.ndim == VIDEO_BATCH_RANK:
            if tensor.shape[1] == channels:
                samples = [item.permute(1, 0, 2, 3) for item in tensor]
            elif tensor.shape[2] == channels:
                samples = list(tensor)
            elif tensor.shape[-1] == channels:
                samples = [item.permute(0, 3, 1, 2) for item in tensor]
            else:
                raise ValueError("5D latent must be [B,C,T,H,W], [B,T,C,H,W], or [B,T,H,W,C]")
        else:
            raise ValueError(f"latents must have rank 4 or 5, got shape {tuple(tensor.shape)}")
    _validate_equal_shapes(samples, "latents")
    return samples


def _latent_sample(value: torch.Tensor | np.ndarray, channels: int) -> torch.Tensor:
    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != VIDEO_RANK:
        raise ValueError(f"latent sample must have rank 4, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] == channels:
        return tensor.permute(0, 1, 2, 3)
    if tensor.shape[-1] == channels:
        return tensor.permute(0, 3, 1, 2)
    if tensor.shape[0] == channels:
        return tensor.permute(1, 0, 2, 3)
    raise ValueError(f"4D latent must contain {channels} channels")


def _validate_equal_shapes(values: list[torch.Tensor], name: str) -> None:
    if not values:
        raise ValueError(f"{name} batch must not be empty")
    shape = tuple(values[0].shape)
    if any(tuple(value.shape) != shape for value in values[1:]):
        raise ValueError(f"all {name} batch items must have equal shape")


def _batch_values(value: Any, batch_size: int, name: str) -> list[Any]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(f"{name} must contain one value per input ({batch_size})")
        return list(value)
    if name == "save_path" and batch_size > 1:
        raise ValueError("save_path must be a list when processing multiple SR inputs")
    return [value] * batch_size


def _batch_audio(value: list[np.ndarray] | np.ndarray | None, batch_size: int) -> list[Any]:
    """Broadcast one waveform or split a list of per-sample waveforms."""
    if value is None:
        return [None] * batch_size
    if isinstance(value, list) and len(value) == batch_size:
        return list(value)
    return [value] * batch_size


@torch.no_grad()
def encode_lq_video_to_lr_latent(
    lq_video: torch.Tensor,
    vae: torch.nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    """Encode ``[T, C, H, W]`` pixels into raw ``[T', C, H', W']`` latents."""
    if lq_video.ndim != 4:  # noqa: PLR2004
        raise ValueError(f"lq_video must have rank 4 [T,C,H,W], got {tuple(lq_video.shape)}")
    pixel = lq_video.permute(1, 0, 2, 3).unsqueeze(0).to(device=device)
    pixel = vae.normalize_data(pixel.float()) if hasattr(vae, "normalize_data") else pixel.float() / 127.5 - 1.0
    pixel = cast_to_module_dtype(vae, pixel)
    result = _extract_latent(vae.encode(pixel))
    return result.squeeze(0).permute(1, 0, 2, 3).float()


@torch.no_grad()
def encode_lq_videos_to_lr_latents(
    lq_videos: list[torch.Tensor],
    vae: torch.nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    """Encode an equal-shaped batch of ``[T,C,H,W]`` videos in one VAE call."""
    _validate_equal_shapes(lq_videos, "video")
    pixel = torch.stack(lq_videos).permute(0, 2, 1, 3, 4).to(device=device)
    pixel = vae.normalize_data(pixel.float()) if hasattr(vae, "normalize_data") else pixel.float() / 127.5 - 1.0
    pixel = cast_to_module_dtype(vae, pixel)
    result = _extract_latent(vae.encode(pixel))
    if result.ndim != VIDEO_BATCH_RANK:
        raise ValueError(f"batched VAE encode must return rank 5, got shape {tuple(result.shape)}")
    return result.permute(0, 2, 1, 3, 4).float()


def latent_tile_grid_from_pixel_grid(pixel_grid: TileGrid, spatial_factor: int) -> TileGrid:
    """Convert an aligned pixel grid to the corresponding latent grid."""
    values = (pixel_grid.tile_h, pixel_grid.tile_w, *pixel_grid.tops, *pixel_grid.lefts)
    if any(value % spatial_factor for value in values):
        raise ValueError(
            f"Pixel tile grid (tile {pixel_grid.tile_h}x{pixel_grid.tile_w}, "
            f"tops={pixel_grid.tops}, lefts={pixel_grid.lefts}) not aligned to VAE spatial factor "
            f"{spatial_factor}."
        )
    return TileGrid(
        pixel_grid.tile_h // spatial_factor,
        pixel_grid.tile_w // spatial_factor,
        tuple(top // spatial_factor for top in pixel_grid.tops),
        tuple(left // spatial_factor for left in pixel_grid.lefts),
    )


@torch.no_grad()
def upscale_lr_latent_tile(
    lr_latent_tile: torch.Tensor,
    latent_upscaler: torch.nn.Module,
    vae: torch.nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    """Upscale one raw latent tile into the sampler's channels-last layout."""
    scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))
    tile = lr_latent_tile.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.float32)
    tile = cast_to_module_dtype(latent_upscaler, tile)
    if str(device).startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            upscaled = run_latent_upscaler(latent_upscaler, tile * scaling_factor)
    else:
        upscaled = run_latent_upscaler(latent_upscaler, tile * scaling_factor)
    return upscaled.squeeze(0).permute(1, 2, 3, 0).float()


def _upsample_tiles_to_base(raw_tiles: list[torch.Tensor], base_h: int, base_w: int) -> list[torch.Tensor]:
    return [
        functional.interpolate(tile.float(), size=(base_h, base_w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        for tile in raw_tiles
    ]


def _stitch_batch(
    outputs: list[torch.Tensor],
    grid: TileGrid,
    batch_size: int,
    height: int,
    width: int,
    scale: int,
) -> torch.Tensor:
    """Stitch tile-major outputs into ``[B,C,T,H,W]``."""
    expected = batch_size * grid.total_tiles
    if len(outputs) != expected:
        raise RuntimeError(f"expected {expected} SR tile outputs, got {len(outputs)}")
    return torch.stack(
        [
            stitch_tiles_hanning(
                [outputs[tile * batch_size + sample] for tile in range(grid.total_tiles)],
                grid,
                height,
                width,
                scale=scale,
            )
            .clamp(0, 255)
            .to(torch.uint8)
            for sample in range(batch_size)
        ]
    )


def _run_batched_tiles(
    samples: list[torch.Tensor],
    components: SRComponents,
    run_config: RunConfig,
    *,
    latent_input: bool,
    offload: OffloadHandle,
    progress_callback: Callable[[int], Any] | None,
) -> torch.Tensor:
    """Run equal-shaped samples tile-major so each model call keeps batch ``B``."""
    from musubi_tuner.kandinsky6.runtime.sr.pipeline.stages import _run_tile_batches  # noqa: PLC0415

    params = _component_params(components)
    visual_size = _visual_size(params)
    batch_size = len(samples)
    spatial_factor = _spatial_factor(components)
    if latent_input:
        height = samples[0].shape[-2] * spatial_factor
        width = samples[0].shape[-1] * spatial_factor
        _, _, pixel_grid = _tile_geometry(
            height,
            width,
            visual_size,
            run_config.resolution_scale,
            run_config.overlap,
            spatial_factor,
        )
        grid = latent_tile_grid_from_pixel_grid(pixel_grid, spatial_factor)
        stitch_grid = pixel_grid
        tile_sets = [extract_all_tiles(sample, grid) for sample in samples]
        upscaler = latent_upscaler_for_scale(components.latent_upscaler, run_config.resolution_scale)
        if upscaler is None:
            raise ValueError(f"no latent upscaler for {run_config.resolution_scale}x")

        def prepare_chunk(chunk: list[torch.Tensor]) -> list[torch.Tensor]:
            return [upscale_lr_latent_tile(tile, upscaler, components.vae, run_config.device) for tile in chunk]

    else:
        height, width = samples[0].shape[-2:]
        base, _, grid = _tile_geometry(
            height,
            width,
            visual_size,
            run_config.resolution_scale,
            run_config.overlap,
            spatial_factor,
        )
        tile_sets = [_upsample_tiles_to_base(extract_all_tiles(sample, grid), base[0], base[1]) for sample in samples]
        stitch_grid = grid
        prepare_chunk = None

    tile_inputs = [tile_sets[sample][tile] for tile in range(grid.total_tiles) for sample in range(batch_size)]
    batched_config = replace(run_config, tiles_batch_size=run_config.tiles_batch_size * batch_size)
    outputs = _run_tile_batches(
        tile_inputs,
        components,
        batched_config,
        _scale_factor(params, visual_size),
        stage=None,
        offload=offload,
        prepare_chunk=prepare_chunk,
        progress_callback=progress_callback,
        sample_batch_size=batch_size,
    )
    return _stitch_batch(outputs, stitch_grid, batch_size, height, width, run_config.resolution_scale)


class Kandinsky6SRPipeline:
    """Scale-aware tiled wrapper around the Kandinsky SR model."""

    def __init__(  # noqa: PLR0913
        self,
        dit: torch.nn.Module,
        vae: torch.nn.Module,
        latent_upscaler: torch.nn.Module | Any | None,
        device: str | torch.device,
        sr_params: SRParams | Any,
        num_steps: int = 5,
        resolution_scale: ResolutionScale = 4,
        overlap: float = 0.20,
        tiles_batch_size: int = 1,
        cached_text_embeds: dict[str, torch.Tensor] | None = None,
        sampler: Callable[..., torch.Tensor] | None = None,
        spatial_factor: int | None = None,
        source_vae: torch.nn.Module | None = None,
        kvae_bridge: bool = False,
        mode: Literal["pixel", "latent"] = "pixel",
        vae_backend: Literal["torch", "magi"] = "torch",
        offload: OffloadHandle | None = None,
    ) -> None:
        self.dit = dit
        self.vae = vae
        self.latent_upscaler = latent_upscaler
        self.device = torch.device(device)
        self.sr_params = sr_params
        self.num_steps = num_steps
        self.resolution_scale = resolution_scale
        self.overlap = overlap
        self.tiles_batch_size = tiles_batch_size
        self.cached_text_embeds = cached_text_embeds
        self.sampler = sampler
        self.spatial_factor = spatial_factor
        self.source_vae = source_vae
        self.kvae_bridge = kvae_bridge
        self.mode = mode
        self.vae_backend = vae_backend
        self.offload = offload or NoOpOffload()

    def _components(
        self,
        cached_text_embeds: dict[str, torch.Tensor] | None = None,
    ) -> SRComponents:
        return SRComponents(
            dit=self.dit,
            vae=self.vae,
            latent_upscaler=self.latent_upscaler,
            sr_params=self.sr_params,
            cached_text_embeds=(self.cached_text_embeds if cached_text_embeds is None else cached_text_embeds),
            sampler=self.sampler,
            spatial_factor=self.spatial_factor,
        )

    @_release_sr_modules_after_call
    def __call__(  # noqa: PLR0912, PLR0913
        self,
        video: torch.Tensor | np.ndarray | list[torch.Tensor] | None = None,
        latents: torch.Tensor | np.ndarray | list[torch.Tensor] | None = None,
        *,
        resolution_scale: ResolutionScale | None = None,
        num_steps: int | None = None,
        seed: int = 42,
        overlap: float | None = None,
        tiles_batch_size: int | None = None,
        kvae_bridge: bool | None = None,
        cached_text_embeds: dict[str, torch.Tensor] | None = None,
        save_path: str | Path | list[str | Path] | None = None,
        fps: int | None = None,
        audio: list[np.ndarray] | np.ndarray | None = None,
        source_video: str | Path | list[str | Path] | None = None,
        audio_sample_rate: int = 44100,
        show_progress: bool = False,
        progress: ProgressReporter | None = None,
    ) -> SRPipelineOutput:  # noqa: PLR0912
        """Super-resolve a video/latent and optionally preserve its audio.

        ``show_progress`` prints a console bar; ``progress`` additionally feeds
        a host reporter (see :mod:`musubi_tuner.kandinsky6.runtime.sr.pipeline.progress`) with the
        same per-step updates.

        Stage-level execution is available from musubi_tuner.kandinsky6.runtime.sr.pipeline.stages.
        This end-to-end path intentionally has no stage callback or profiling
        overhead; benchmark integrations call the same tiled functions directly
        with their own stage contexts.
        """
        if (video is None) == (latents is None):
            raise ValueError("pass exactly one of video or latents")
        if audio is not None and source_video is not None:
            raise ValueError("pass either audio or source_video, not both")

        requested_scale = self.resolution_scale if resolution_scale is None else resolution_scale
        tiling_scale, pre_upscale = resolve_scale_request(float(requested_scale))
        bridge = self.kvae_bridge if kvae_bridge is None else kvae_bridge
        components = self._components(cached_text_embeds)
        samples, latent_input = self._source_samples(video, latents, bridge=bridge, pre_upscale=pre_upscale)
        batch_size = len(samples)

        run_config = RunConfig(
            device=str(self.device),
            num_steps=self.num_steps if num_steps is None else num_steps,
            seed=seed,
            overlap=self.overlap if overlap is None else overlap,
            tiles_batch_size=self.tiles_batch_size if tiles_batch_size is None else tiles_batch_size,
            resolution_scale=tiling_scale,
        )

        has_latent_upscaler = latent_upscaler_for_scale(self.latent_upscaler, run_config.resolution_scale) is not None
        samples, padding = _pad_for_latent_path(
            components, run_config, samples, latent_input=latent_input, latent_path=has_latent_upscaler
        )
        if not latent_input and has_latent_upscaler:
            # Pixel input, including the KVAE bridge's decoded source latent,
            # is encoded with the SR VAE before latent upscaling.
            with self.offload.use("vae"):
                if batch_size == 1:
                    samples = [encode_lq_video_to_lr_latent(samples[0], self.vae, self.device)]
                else:
                    encoded = encode_lq_videos_to_lr_latents(samples, self.vae, self.device)
                    samples = list(encoded)
            latent_input = True

        reporter = _progress_reporter(show_progress, progress)
        if reporter is not None:
            factor = _spatial_factor(components) if latent_input else 1
            source_hw = (int(samples[0].shape[-2]) * factor, int(samples[0].shape[-1]) * factor)
            total_tiles, steps_per_tile = denoise_progress_total(components, run_config, source_hw)
            reporter.start(total_tiles * batch_size, steps_per_tile)

        try:
            progress_callback = reporter.update if reporter is not None else None
            if batch_size == 1:
                from musubi_tuner.kandinsky6.runtime.sr.pipeline.stages import (  # noqa: PLC0415
                    run_tiled_sr,
                    run_tiled_sr_from_pixels,
                )

                if latent_input:
                    frames = run_tiled_sr(
                        samples[0],
                        components,
                        run_config,
                        offload=self.offload,
                        progress_callback=progress_callback,
                    ).unsqueeze(0)
                else:
                    frames = run_tiled_sr_from_pixels(
                        samples[0],
                        components,
                        run_config,
                        offload=self.offload,
                        progress_callback=progress_callback,
                    ).unsqueeze(0)
            else:
                frames = _run_batched_tiles(
                    samples,
                    components,
                    run_config,
                    latent_input=latent_input,
                    offload=self.offload,
                    progress_callback=progress_callback,
                )
        finally:
            if reporter is not None:
                reporter.close()
        frames = padding.crop(frames, run_config.resolution_scale)

        path = None
        if save_path is not None:
            paths = [
                str(
                    mux_video_audio(
                        frames[index],
                        sample_audio,
                        output_path,
                        fps=fps or int(getattr(self.sr_params, "fps", 24)),
                        audio_sample_rate=audio_sample_rate,
                        source_video=sample_source,
                    )
                )
                for index, (output_path, sample_source, sample_audio) in enumerate(
                    zip(
                        _batch_values(save_path, batch_size, "save_path"),
                        _batch_values(source_video, batch_size, "source_video"),
                        _batch_audio(audio, batch_size),
                        strict=True,
                    )
                )
            ]
            path = paths[0] if batch_size == 1 else paths
        output_audio = None if audio is None else _batch_audio(audio, batch_size)
        return SRPipelineOutput(frames=frames, audio=output_audio, path=path)

    def _source_samples(
        self,
        video: torch.Tensor | np.ndarray | list[torch.Tensor] | None,
        latents: torch.Tensor | np.ndarray | list[torch.Tensor] | None,
        *,
        bridge: bool,
        pre_upscale: float,
    ) -> tuple[list[torch.Tensor], bool]:
        """Normalise the input to equal-shaped ``[T,C,H,W]`` samples and tell whether they are SR latents."""
        if video is not None:
            samples = _video_samples(video)
            latent_input = False
        else:
            samples = _latent_samples(latents, int(getattr(self.dit, "in_visual_dim", 16)))  # type: ignore[arg-type]
            latent_input = True

        if latent_input:
            if bridge:
                if self.source_vae is None:
                    raise ValueError("KVAE bridge latent SR requires the source VAE; pass pixel video or configure source_vae")
                # The bridge converts the source VAE latent to pixels first;
                # the regular pixel-input branch then encodes it with the
                # SR VAE when a matching latent upscaler is available.
                samples = [_decode_source_latent_video(sample, self.source_vae, self.device) for sample in samples]
                latent_input = False
            elif pre_upscale != 1.0:
                raise ValueError("2.25x latent SR requires the KVAE bridge source VAE; pass pixel video or configure source_vae")
        if not latent_input and pre_upscale != 1.0:
            samples = [pre_upscale_video(sample, pre_upscale, _spatial_factor(self._components())) for sample in samples]
            _validate_equal_shapes(samples, "video")
        return samples, latent_input


def _pad_for_latent_path(
    components: SRComponents,
    run_config: RunConfig,
    samples: list[torch.Tensor],
    *,
    latent_input: bool,
    latent_path: bool,
) -> tuple[list[torch.Tensor], SourcePadding]:
    """Pad the equal-shaped samples the latent-upscaler path is about to consume; the pixel path takes any size."""
    if not (latent_input or latent_path):
        return samples, SourcePadding()
    factor = _spatial_factor(components)
    pixels_per_cell = factor if latent_input else 1
    source_hw = (int(samples[0].shape[-2]) * pixels_per_cell, int(samples[0].shape[-1]) * pixels_per_cell)
    padding = latent_path_padding(components, run_config, source_hw)
    if latent_input:
        samples = [padding.apply_to_latent(sample, factor) for sample in samples]
    else:
        samples = [padding.apply_to_video(sample) for sample in samples]
    if padding.any:
        logger.info(
            "Source {}x{} padded to {}x{} (bottom={}, right={}) for the latent path; the SR result is cropped back to source x{}",
            source_hw[0],
            source_hw[1],
            *padding.padded_hw(source_hw),
            padding.bottom,
            padding.right,
            run_config.resolution_scale,
        )
    return samples, padding


@torch.no_grad()
def _decode_source_latent_video(
    raw_latents: torch.Tensor,
    source_vae: torch.nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    """Decode raw base-VAE latents to ``[T,C,H,W]`` pixels for the KVAE bridge."""
    if raw_latents.ndim != 4:  # noqa: PLR2004
        raise ValueError(f"raw_latents must have rank 4 [T,C,H,W], got {tuple(raw_latents.shape)}")
    latent_5d = raw_latents.permute(1, 0, 2, 3).unsqueeze(0).to(device=device)
    latent_5d = cast_to_module_dtype(source_vae, latent_5d)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        try:
            decoded = source_vae.decode(latent_5d, alternative_fwd=True).sample
        except TypeError as exc:
            if "alternative_fwd" not in str(exc):
                raise
            decoded = source_vae.decode(latent_5d).sample
    return ((decoded.squeeze(0).float().clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).cpu()


__all__ = [
    "Kandinsky6SRPipeline",
    "RunConfig",
    "SRComponents",
    "SRParams",
    "denoise_progress_total",
    "encode_lq_video_to_lr_latent",
    "encode_lq_videos_to_lr_latents",
    "latent_path_padding",
    "latent_tile_grid_from_pixel_grid",
    "pre_upscale_video",
    "resolve_scale_request",
    "upscale_lr_latent_tile",
]
