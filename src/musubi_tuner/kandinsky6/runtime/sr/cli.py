"""Cyclopts CLI for tiled LU + DiT super-resolution.

One subcommand:

* ``from-video`` — read an mp4/mkv, VAE-encode it to an LR latent, then run SR.

The command is exposed as ``kandy-sr`` when the project is installed. The
CUDA device is a regular SR setting (``sr.device`` in the YAML, ``--device``
overrides): SR components, helper tensors, and warmup all use the same one.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import torch
import yaml
from loguru import logger

# Late-bound VAE factors (set from the model config's vae.name inside
# load_sr_components) — read as module attributes, never ``from``-imported.
from musubi_tuner.kandinsky6.runtime.sr import constants
from musubi_tuner.kandinsky6.runtime.sr.constants import TARGET_RESOLUTIONS
from musubi_tuner.kandinsky6.runtime.sr.core.algo.latent_upscaler import (
    LatentUpscalerBank,
    latent_upscaler_for_scale,
    latent_upscaler_scale,
)
from musubi_tuner.kandinsky6.runtime.sr.core.algo.mux import mux_video_audio
from musubi_tuner.kandinsky6.runtime.sr.pipeline.components import scale_factor_for
from musubi_tuner.kandinsky6.runtime.sr.pipeline.config import (
    ResolutionScale,
    SRConfig,
    TargetResizeMode,
    VaeBackend,
    load_sr_config_yaml,
)
from musubi_tuner.kandinsky6.runtime.sr.pipeline.factory import load_sr_pipeline
from musubi_tuner.kandinsky6.runtime.sr.pipeline.output_resize import resize_to_target, resolve_target_hw
from musubi_tuner.kandinsky6.runtime.sr.pipeline.sr_pipeline import RunConfig
from musubi_tuner.kandinsky6.runtime.sr.pipeline.upscale_utils import pre_upscale_video, resolve_scale_request
from musubi_tuner.kandinsky6.runtime.sr.pipeline.video_io import (
    clip_to_aligned_frames,
    read_video_tchw_uint8,
    resample_to_target_fps,
)
from musubi_tuner.kandinsky6.runtime.sr.pipeline.warmup import (
    clip_base_resolution,
    compile_vae_decode,
    log_sampling_mode,
    warmup,
    warmup_target_pass,
)


@dataclass
class CommonOptions:
    """Options shared by both SR input paths.

    ``Parameter(name="*")`` flattens this dataclass into regular top-level
    command options, so both subcommands have the same interface without a
    Click context or duplicated option handling.
    """

    config_path: Annotated[
        Path | None,
        cyclopts.Parameter(
            name="--config",
            show_default=True,
            help=(
                "SR config YAML: either a standalone file with an sr: section or a full K6 "
                "pipeline YAML (sr.enabled is ignored). Defaults to $KANDY_SR_CONFIG when set. "
                "Explicit CLI options take precedence."
            ),
        ),
    ] = None

    checkpoint_path: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--checkpoint-path",
            show_default=True,
            help=(
                "The SR DiT: a Kandinsky6SRPipeline Diffusers bundle (Hugging Face repo id or local dir), "
                "or a native DiT .safetensors / .pt, step dir, or model/ shard dir."
            ),
        ),
    ] = None
    vae_path: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--vae-path",
            show_default="per VAE type",
            help=(
                "Video KVAE: a Diffusers bundle or its vae component ('<repo>/vae'), or a video-kvae sidecar "
                "prefix ({prefix}.yaml + {prefix}.safetensors or {prefix}.ckpt). Unset = the VAE of the "
                "--checkpoint-path bundle; required with a native DiT checkpoint."
            ),
        ),
    ] = None
    vae_backend: Annotated[
        VaeBackend | None,
        cyclopts.Parameter(
            name="--vae-backend",
            show_default=True,
            help=(
                "VAE compilation backend: 'torch' = per-block/leaf torch.compile; 'magi' = MagiCompiler static graphs."
            ),
        ),
    ] = None
    dit_overrides_spec: Annotated[
        tuple[str, ...] | None,
        cyclopts.Parameter(
            name="--dit-override",
            consume_multiple=True,
            show_default=True,
            help=(
                "Post-load DiT attribute override KEY=VALUE (repeatable, YAML-parsed). Passing any "
                "value replaces the defaults; 'none' clears all overrides."
            ),
        ),
    ] = None
    latent_upscaler_config: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--latent-upscaler-config",
            show_default=True,
            help=(
                "Latent upscalers: a Diffusers bundle or its latent_upscaler component "
                "('<repo>/latent_upscaler'), or an LU bank YAML containing architectures, checkpoints, and EMA "
                "settings. Unset = the bank of the --checkpoint-path bundle. Pass 'none' to disable LU and use "
                "the pixel path."
            ),
        ),
    ] = None
    target_resolution: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--target-resolution",
            show_default=True,
            help=(f"Delivery tier {sorted(TARGET_RESOLUTIONS)}, explicit WxH, or 'none' to keep raw SR dimensions."),
        ),
    ] = None
    target_resize_mode: Annotated[
        TargetResizeMode | None,
        cyclopts.Parameter(
            name="--target-resize-mode",
            show_default=True,
            help="Resize mode: 'fit' preserves aspect; 'exact' uses the precise bucket dimensions.",
        ),
    ] = None
    output_dir: Annotated[
        Path,
        cyclopts.Parameter(
            name="--output-dir",
            show_default=True,
            help="Output directory.",
        ),
    ] = Path("outputs/")
    device: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--device",
            show_default=True,
            help="CUDA device used for all SR components and tensors (overrides the YAML's sr.device).",
        ),
    ] = None
    num_steps: Annotated[
        int | None,
        cyclopts.Parameter(
            name="--num-steps",
            show_default=True,
            help=(
                "Denoising grid points per tile (N points = N - 1 denoising steps). Shortcut-trained "
                "checkpoints require N - 1 to be a power of two."
            ),
        ),
    ] = None
    seed: Annotated[int | None, cyclopts.Parameter(name="--seed", show_default=True)] = None
    overlap: Annotated[
        float | None,
        cyclopts.Parameter(
            name="--overlap",
            show_default=True,
            help="Minimum overlap between neighbouring tiles, as a fraction of the tile size, in [0, 1).",
        ),
    ] = None
    tiles_batch_size: Annotated[
        int | None,
        cyclopts.Parameter(name="--tiles-batch-size", show_default=True, help="Tiles per generate call."),
    ] = None
    save_lossless: Annotated[
        bool,
        cyclopts.Parameter(name="--save-lossless", show_default=True, help="Save lossless MKV."),
    ] = False
    warmup: Annotated[
        bool,
        cyclopts.Parameter(
            name="--warmup",
            show_default=True,
            help=(
                "Run one throwaway tile at this clip's tile shape before the SR, so first-tile compile and autotune "
                "costs stay out of the timed run (costs about one tile; for timing only)."
            ),
        ),
    ] = False
    precompile_all_bases: Annotated[
        bool,
        cyclopts.Parameter(
            name="--precompile-all-bases",
            show_default=True,
            help=(
                "Compile the KVAE decode for every trained base resolution at load instead of only the one "
                "this clip tiles into (needs GPU memory for all compiled graphs; magi pins decoder weights per graph)."
            ),
        ),
    ] = False


_DEFAULT_COMMON_OPTIONS = CommonOptions()


def _parse_dit_overrides(specs: tuple[str, ...]) -> dict[str, Any]:
    """Parse repeated ``--dit-override KEY=VALUE`` specs into an attribute dict."""
    if any(spec.lower() == "none" for spec in specs):
        if len(specs) > 1:
            raise ValueError(f"--dit-override none cannot be combined with other overrides, got {specs!r}")
        return {}
    overrides: dict[str, Any] = {}
    for spec in specs:
        key, sep, raw = spec.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--dit-override must be KEY=VALUE, got {spec!r}")
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


def _prepare_common(options: CommonOptions) -> SRConfig:
    """Build the effective :class:`SRConfig`: defaults <- YAML <- CLI options."""
    config_path = options.config_path
    if config_path is None and os.environ.get("KANDY_SR_CONFIG"):
        config_path = Path(os.environ["KANDY_SR_CONFIG"])
        logger.info("Using SR config from KANDY_SR_CONFIG={}", config_path)
    sr = load_sr_config_yaml(config_path) if config_path is not None else SRConfig()

    cli_overrides = {
        "checkpoint_path": options.checkpoint_path,
        "vae_path": options.vae_path,
        "latent_upscaler_config": options.latent_upscaler_config,
        "vae_backend": options.vae_backend,
        "device": options.device,
        "num_steps": options.num_steps,
        "seed": options.seed,
        "overlap": options.overlap,
        "tiles_batch_size": options.tiles_batch_size,
        "target_resolution": options.target_resolution,
        "target_resize_mode": options.target_resize_mode,
    }
    if options.dit_overrides_spec is not None:
        cli_overrides["dit_overrides"] = _parse_dit_overrides(options.dit_overrides_spec)
    sr = sr.model_copy(update={key: value for key, value in cli_overrides.items() if value is not None})

    if not sr.checkpoint_path:
        raise ValueError(
            "checkpoint_path is not set — pass --checkpoint-path, or point --config / "
            "$KANDY_SR_CONFIG at a YAML with sr.checkpoint_path "
            "(example: src/musubi_tuner.kandinsky6.runtime.sr/configs/sr_release_hf.yaml)"
        )
    logger.info("Effective SR config: {}", sr.model_dump(exclude={"enabled", "mode", "kvae_bridge"}))
    options.output_dir.mkdir(parents=True, exist_ok=True)
    return sr


def _parse_resolution_scale(_type: Any, tokens: Sequence[cyclopts.Token]) -> float:
    """Read the ``--resolution-scale`` token as a number.

    cyclopts converts a token for the mixed ``Literal[2, 2.25, 4]`` through its
    int members, which turns ``2.25`` into ``2``; the literal validation still
    runs on the returned value and refuses anything but the supported scales.
    """
    return float(tokens[-1].value)


def _resolve_resolution_scale(value: ResolutionScale | None, sr: SRConfig) -> ResolutionScale:
    """Use the config scale only when the command did not set one explicitly."""
    return value if value is not None else sr.resolution_scale


def _require_input_file(input_path: Path) -> None:
    if not input_path.is_file():
        raise ValueError(f"Input file does not exist or is not a regular file: {input_path}")


def _warn_without_latent_upscaler(latent_upscaler: Any, tiling_scale: float) -> None:
    """Explain the pixel-path fallback when no loaded LU matches ``tiling_scale``."""
    if latent_upscaler_for_scale(latent_upscaler, tiling_scale) is not None:
        return
    available = (
        latent_upscaler.scales
        if isinstance(latent_upscaler, LatentUpscalerBank)
        else (latent_upscaler_scale(latent_upscaler),)
        if latent_upscaler is not None
        else ()
    )
    logger.warning(
        "No latent upscaler for x{} (loaded LU scales: {}) -> bypassing the LU and upsampling "
        "tiles in pixel space (bilinear LQ conditioning instead of the learned latent upscale).",
        tiling_scale,
        available,
    )


def _warm_target_shape(sr: SRConfig, sr_pipe: Any, tiling_scale: int, video: torch.Tensor) -> None:
    """One throwaway tile at the run's tile shape (``--warmup``), on the pre-upscaled source size."""
    run_config = RunConfig(
        device=sr.device,
        num_steps=sr.num_steps,
        seed=sr.seed,
        overlap=sr.overlap,
        tiles_batch_size=sr.tiles_batch_size,
        resolution_scale=tiling_scale,
    )
    run_lu = latent_upscaler_for_scale(sr_pipe.latent_upscaler, tiling_scale)
    source_hw = (video.shape[2], video.shape[3])
    warmup_target_pass(sr_pipe, run_config, source_hw, video.shape[0], run_lu, lq_video=video)


def _run_sr(  # noqa: PLR0913 - CLI knobs stay explicit
    sr: SRConfig,
    resolution_scale: ResolutionScale,
    video: torch.Tensor,
    *,
    precompile_all_bases: bool,
    warm_target_shape: bool,
) -> torch.Tensor:
    """Super-resolve ``video`` through :class:`Kandinsky6SRPipeline` built by the factory."""
    sr_pipe = load_sr_pipeline(sr, sr.device, force=True, resolution_scale=resolution_scale)
    log_sampling_mode(sr_pipe, sr.num_steps)
    # One-time compile work happens here, outside the pipeline: flex/nabla
    # attention kernels and the KVAE decode for the base resolution this clip
    # tiles into (or every trained base with --precompile-all-bases).
    warmup(sr_pipe.dit, scale_factor_for(sr_pipe), sr.device)
    bases = None if precompile_all_bases else [clip_base_resolution(sr_pipe, (video.shape[2], video.shape[3]))]
    compile_vae_decode(sr_pipe, sr.device, bases=bases)

    tiling_scale, pre_upscale = resolve_scale_request(resolution_scale)
    if pre_upscale != 1.0:
        logger.info("Pre-upscale x{} in pixel space, then tiling at x{}", pre_upscale, tiling_scale)
    _warn_without_latent_upscaler(sr_pipe.latent_upscaler, tiling_scale)
    if warm_target_shape:
        warm_video = (
            video if pre_upscale == 1.0 else pre_upscale_video(video, pre_upscale, constants.VAE_SPATIAL_FACTOR)
        )
        _warm_target_shape(sr, sr_pipe, tiling_scale, warm_video)
    return sr_pipe(video=video, seed=sr.seed, show_progress=True).frames[0]


def _save_sr(  # noqa: PLR0913 — output knobs stay explicit
    sr_video: torch.Tensor,
    sr: SRConfig,
    options: CommonOptions,
    stem: str,
    fps: int,
    *,
    source_video: Path | None = None,
) -> None:
    """Resize to the requested delivery tier (if any), then write the video."""
    target_hw = resolve_target_hw(
        sr.target_resolution,
        sr_video.shape[-2:],
        mode=sr.target_resize_mode,
    )
    if target_hw is not None:
        sr_video = resize_to_target(sr_video, target_hw)
        logger.info("Resized to target {}x{} (WxH)", sr_video.shape[3], sr_video.shape[2])
    suffix = ".mkv" if options.save_lossless else ".mp4"
    mux_video_audio(
        sr_video,
        None,
        options.output_dir / f"sr_{stem}{suffix}",
        fps=fps,
        source_video=source_video,
        lossless=options.save_lossless,
    )


app = cyclopts.App(
    name="kandy-sr",
    help="Tiled LU + DiT super-resolution inference CLI",
    version=lambda: version("kandinsky-6-sr"),
)
# Keep the old module-level name available for callers that imported ``cli``.
cli = app


@app.command(name="from-video")
def from_video(
    input_path: Annotated[
        Path,
        cyclopts.Parameter(
            name="--input",
            required=True,
            help="Source LQ video (mp4/mkv); VAE-encoded to an LR latent before SR.",
        ),
    ],
    resolution_scale: Annotated[
        ResolutionScale | None,
        cyclopts.Parameter(
            name="--resolution-scale",
            show_default=True,
            converter=_parse_resolution_scale,
            help="Total upscale factor: 2, 4, or 2.25 (x1.125 pixel pre-upscale + x2 tiling).",
        ),
    ] = None,
    common: Annotated[CommonOptions, cyclopts.Parameter(name="*")] = _DEFAULT_COMMON_OPTIONS,
) -> None:
    """Read an mp4/mkv, resample to the training fps, then run SR.

    ``--resolution-scale 2.25`` pre-upscales the source by 1.125 and then uses
    the regular x2 tiling path. A matching latent upscaler uses the latent path;
    without one the command falls back to pixel-space conditioning.
    """
    _require_input_file(input_path)
    sr_config = _prepare_common(common)
    resolution_scale = _resolve_resolution_scale(resolution_scale, sr_config)
    logger.info("Run resolution_scale={} on {}", resolution_scale, sr_config.device)
    video, src_fps = read_video_tchw_uint8(input_path)
    video, out_fps = resample_to_target_fps(video, src_fps)
    video = clip_to_aligned_frames(video)
    logger.info(
        "Input video: {}x{} (WxH), {} frames @ {:g} fps (source {:g} fps)",
        video.shape[3],
        video.shape[2],
        video.shape[0],
        out_fps,
        src_fps,
    )

    started = time.perf_counter()
    sr_video = _run_sr(
        sr_config,
        resolution_scale,
        video,
        precompile_all_bases=common.precompile_all_bases,
        warm_target_shape=common.warmup,
    )
    logger.info(
        "Output video: {}x{} (WxH), {} frames — SR x{} in {:.1f}s (load + warmup + run)",
        sr_video.shape[3],
        sr_video.shape[2],
        sr_video.shape[1],
        resolution_scale,
        time.perf_counter() - started,
    )
    _save_sr(sr_video, sr_config, common, input_path.stem, fps=out_fps, source_video=input_path)
    logger.info(
        "Done: {} ({} frames @ {}fps, x{})",
        input_path.name,
        video.shape[0],
        out_fps,
        resolution_scale,
    )


def main() -> None:
    """Run the ``kandy-sr`` command."""
    app()


if __name__ == "__main__":
    main()
