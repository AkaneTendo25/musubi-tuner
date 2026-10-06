"""Reusable Kandinsky SR inference stages and tiled orchestration.

The functions in this module deliberately do not depend on a benchmark
runner.  A framework can call ``text_encode`` / ``prepare_latents`` /
``denoise`` / ``vae_decode`` directly, or use the tiled functions for the
scale-aware path.  Native benchmark integrations may add timing contexts to
the tiled calls; the public :class:`Kandinsky6SRPipeline` does not expose
those callbacks.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed import all_gather

from musubi_tuner.kandinsky6.runtime.sr.core.algo.latent_upscaler import (
    latent_upscaler_for_scale,
    latent_upscaler_scale,
)
from musubi_tuner.kandinsky6.runtime.sr.core.algo.tiling_utils import extract_all_tiles, stitch_tiles_hanning
from musubi_tuner.kandinsky6.runtime.sr.core.algo.utils import (
    _build_initial_latent,
    _encode_lq_videos,
    _encode_text,
    decode_latent_to_uint8,
    generate,
)
from musubi_tuner.kandinsky6.runtime.sr.core.utils.offload import NoOpOffload, OffloadHandle
from musubi_tuner.kandinsky6.runtime.sr.pipeline.sr_pipeline import (
    _component_params,
    _scale_factor,
    _spatial_factor,
    _tile_geometry,
    _upsample_tiles_to_base,
    latent_tile_grid_from_pixel_grid,
    upscale_lr_latent_tile,
)

StageCallback = Callable[[str], Any]


def _stage_context(stage: StageCallback | None, name: str) -> Any:
    return stage(name) if stage is not None else nullcontext()


@dataclass(frozen=True)
class SRLatentState:
    """Inputs and initial noisy latent for one SR DiT invocation."""

    lq_latent: torch.Tensor
    image: torch.Tensor
    batch_size: int
    duration: int
    height: int
    width: int


@dataclass(frozen=True)
class SRTextState:
    """Conditional and null text state consumed by SR denoising."""

    text_embeds: dict[str, torch.Tensor]
    text_cu_seqlens: torch.Tensor
    null_text_embeds: dict[str, torch.Tensor]
    null_text_cu_seqlens: torch.Tensor


@torch.no_grad()
def prepare_latents(  # noqa: PLR0913
    *,
    dit: torch.nn.Module,
    vae: torch.nn.Module,
    device: str | int,
    lq_videos: list[torch.Tensor] | None = None,
    lq_latents: torch.Tensor | None = None,
    n_samples: int | None = None,
    seed: int = 42,
    lq_noise_scale: float = 0.0,
    lq_noise_type: str = "linear",
    lq_channel_noise_scale: float = 0.0,
    anchor_latents: torch.Tensor | None = None,
    anchor_masks: torch.Tensor | None = None,
    anchor_free: bool = False,
) -> SRLatentState:
    """Encode LQ input and build the initial latent for one SR tile batch."""
    if lq_latents is not None:
        lq_latent = lq_latents.to(device)
        if n_samples is None:
            raise ValueError("n_samples is required when lq_latents is provided")
        batch_size = n_samples
    elif lq_videos is not None:
        batch_size = len(lq_videos)
        lq_latent = _encode_lq_videos(lq_videos, vae, device)
    else:
        raise ValueError("Either lq_videos or lq_latents must be provided")

    duration = lq_latent.shape[0] // batch_size
    height, width = lq_latent.shape[1], lq_latent.shape[2]
    image = _build_initial_latent(
        dit=dit,
        lq_latent=lq_latent,
        bs=batch_size,
        duration=duration,
        height=height,
        width=width,
        device=device,
        seed=seed,
        lq_noise_scale=lq_noise_scale,
        lq_noise_type=lq_noise_type,  # type: ignore[arg-type]
        lq_channel_noise_scale=lq_channel_noise_scale,
        anchor_latent=anchor_latents.to(device) if anchor_latents is not None else None,
        anchor_mask=anchor_masks.to(device) if anchor_masks is not None else None,
        anchor_free=anchor_free,
    )
    return SRLatentState(lq_latent, image, batch_size, duration, height, width)


@torch.no_grad()
def text_encode(
    *,
    dit: torch.nn.Module,
    batch_size: int,
    device: str | int,
    text_embedder: Any | None = None,
    cached_text_embeds: dict[str, torch.Tensor] | None = None,
) -> SRTextState:
    """Build SR conditional and null text embeddings."""
    text_embeds, text_cu_seqlens, null_text_embeds, null_text_cu_seqlens = _encode_text(
        bs=batch_size,
        device=device,
        text_embedder=text_embedder,
        cached_text_embeds=cached_text_embeds,
        use_text=getattr(dit, "use_text", True),
    )
    return SRTextState(text_embeds, text_cu_seqlens, null_text_embeds, null_text_cu_seqlens)


def _positions(
    *,
    dit: torch.nn.Module,
    latent_state: SRLatentState,
    text_state: SRTextState,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    # The latent is the source of truth after CPU offload hooks have moved a
    # model for its forward pass.  Do not create position tensors from a stale
    # caller-provided device (for example, ``cpu`` after Diffusers offload).
    runtime_device = latent_state.image.device
    visual_cu_seqlens = latent_state.duration * torch.arange(
        latent_state.batch_size + 1,
        dtype=torch.int32,
        device=runtime_device,
    )
    visual_rope_pos = [
        torch.cat(
            [torch.arange(int(end), device=runtime_device) for end in torch.diff(visual_cu_seqlens).cpu()],
        ),
        torch.arange(latent_state.height // dit.patch_size[1], device=runtime_device),  # type: ignore[attr-defined]
        torch.arange(latent_state.width // dit.patch_size[2], device=runtime_device),  # type: ignore[attr-defined]
    ]
    text_cu_seqlens = text_state.text_cu_seqlens.to(runtime_device)
    null_text_cu_seqlens = text_state.null_text_cu_seqlens.to(runtime_device)
    text_rope_pos = torch.cat(
        [torch.arange(int(end), device=runtime_device) for end in torch.diff(text_cu_seqlens).cpu()],
    )
    null_text_rope_pos = torch.cat(
        [torch.arange(int(end), device=runtime_device) for end in torch.diff(null_text_cu_seqlens).cpu()],
    )
    return visual_cu_seqlens, visual_rope_pos, text_rope_pos, null_text_rope_pos


@torch.no_grad()
def denoise(  # noqa: PLR0913
    *,
    dit: torch.nn.Module,
    latent_state: SRLatentState,
    text_state: SRTextState,
    scale_factor: tuple[float, ...],
    num_steps: int = 50,
    guidance_weight: float = 5.0,
    scheduler_scale: float = 5.0,
    tp_mesh: dict[str, Any] | None = None,
    lq_noise_scale: float = 0.0,
    cap_noise_timestep: bool = False,
    prediction_target: str = "velocity",
    channelcat_drop_threshold: float = 0.0,
    rfg_scale: float = 1.0,
    piflow_params: dict[str, Any] | None = None,
    device: str | int | None = None,
    progress_callback: Callable[[int], Any] | None = None,
    scheduler: Any | None = None,
) -> torch.Tensor:
    """Run either the Euler or π-Flow SR denoising stage."""
    piflow_params = piflow_params or _scheduler_piflow_params(scheduler)
    piflow_params = piflow_params or getattr(dit, "piflow_params", None)
    if piflow_params is not None:
        if rfg_scale != 1.0:
            raise ValueError("π-Flow checkpoints do not support Reference-Free Guidance")
        if cap_noise_timestep and dit.instruct_type in ("noise", "hybrid"):
            raise NotImplementedError("piflow sampler does not support cap_noise_timestep for noise/hybrid instruct")

    device = device if device is not None else latent_state.image.device

    visual_cu_seqlens, visual_rope_pos, text_rope_pos, null_text_rope_pos = _positions(
        dit=dit,
        latent_state=latent_state,
        text_state=text_state,
    )

    if piflow_params is not None:
        from musubi_tuner.kandinsky6.runtime.sr.core.algo.piflow_sampler import piflow_generate  # noqa: PLC0415

        out_dim = int(getattr(dit, "base_out_visual_dim", dit.in_visual_dim))
        start_t = 1.0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = piflow_generate(
                latent_state.image,
                dit,
                text_state.text_embeds,
                visual_cu_seqlens,
                text_state.text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                scale_factor,
                **piflow_params,
                out_dim=out_dim,
                start_timestep=start_t,
                device=device,
                progress_callback=progress_callback,
                scheduler=scheduler,
            )
    else:
        start_t = (
            lq_noise_scale if cap_noise_timestep and dit.instruct_type in ("noise", "hybrid", "hybrid_anchor") else
            1.0
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                latent_state.image,
                dit,
                device,
                num_steps,
                text_state.text_embeds,
                text_state.null_text_embeds,
                visual_cu_seqlens,
                text_state.text_cu_seqlens,
                text_state.null_text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                scale_factor,
                guidance_weight,
                scheduler_scale,
                tp_mesh=tp_mesh,
                first_frames=None,
                start_timestep=start_t,
                prediction_target=prediction_target,
                channelcat_drop_threshold=channelcat_drop_threshold,
                rfg_scale=rfg_scale,
                progress_callback=progress_callback,
            )

    if tp_mesh:
        tensor_list = [
            torch.zeros_like(latent_visual, device=latent_visual.device) for _ in range(tp_mesh["tp"].size())
        ]
        all_gather(
            tensor_list,
            latent_visual.contiguous(),
            group=tp_mesh.get_group(mesh_dim="tp"),  # type: ignore[attr-defined]
        )
        latent_visual = torch.cat(tensor_list, dim=1)
    return latent_visual


def _scheduler_piflow_params(scheduler: Any | None) -> dict[str, Any] | None:
    if scheduler is None or not bool(getattr(scheduler, "is_piflow", False)):
        return None
    config = scheduler.config
    nfe = getattr(config, "nfe", None)
    if nfe is None:
        raise ValueError("PiflowScheduler used for SR must define nfe in scheduler_config.json")
    return {
        "nfe": int(nfe),
        "num_policy_substeps": int(config.num_policy_substeps),
        "final_step_size_scale": float(config.final_step_size_scale),
        "shift": float(config.shift),
        "n_grid": int(config.n_grid),
        "eps": float(config.eps),
    }


@torch.no_grad()
def vae_decode(  # noqa: PLR0913
    *,
    vae: torch.nn.Module,
    latent_visual: torch.Tensor,
    batch_size: int,
    duration: int,
    height: int,
    width: int,
    vae_decode_batch: bool = False,
) -> torch.Tensor:
    """ Decode denoised SR latents into ``[batch, 3, T, H, W]`` uint8 frames
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        all_latents = latent_visual.reshape(batch_size, duration, height, width, -1)
        all_latents = (all_latents / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)  # type: ignore[attr-defined]
        if vae_decode_batch:
            return decode_latent_to_uint8(vae, all_latents)
        decoded = [decode_latent_to_uint8(vae, all_latents[i : i + 1]) for i in range(batch_size)]
        return torch.cat(decoded, dim=0)


@torch.no_grad()
def run_stages(  # noqa: PLR0913
    *,
    dit: torch.nn.Module,
    vae: torch.nn.Module,
    scale_factor: tuple[float, ...],
    num_steps: int = 50,
    guidance_weight: float = 5.0,
    scheduler_scale: float = 5.0,
    seed: int = 42,
    device: str | int = "cuda",
    tp_mesh: dict[str, Any] | None = None,
    lq_noise_scale: float = 0.0,
    lq_noise_type: str = "linear",
    lq_channel_noise_scale: float = 0.0,
    text_embedder: Any | None = None,
    cached_text_embeds: dict[str, torch.Tensor] | None = None,
    lq_videos: list[torch.Tensor] | None = None,
    lq_latents: torch.Tensor | None = None,
    n_samples: int | None = None,
    cap_noise_timestep: bool = False,
    vae_decode_batch: bool = False,
    prediction_target: str = "velocity",
    channelcat_drop_threshold: float = 0.0,
    anchor_latents: torch.Tensor | None = None,
    anchor_masks: torch.Tensor | None = None,
    anchor_free: bool = False,
    rfg_scale: float = 1.0,
    piflow_params: dict[str, Any] | None = None,
    stage: StageCallback | None = None,
    progress_callback: Callable[[int], Any] | None = None,
    scheduler: Any | None = None,
) -> torch.Tensor:
    """Run the four reusable SR stages for one tile batch."""
    piflow_params = piflow_params or _scheduler_piflow_params(scheduler)
    piflow_params = piflow_params or getattr(dit, "piflow_params", None)

    with _stage_context(stage, "prepare_latents"):
        latent_state = prepare_latents(
            dit=dit,
            vae=vae,
            device=device,
            lq_videos=lq_videos,
            lq_latents=lq_latents,
            n_samples=n_samples,
            seed=seed,
            lq_noise_scale=lq_noise_scale,
            lq_noise_type=lq_noise_type,
            lq_channel_noise_scale=lq_channel_noise_scale,
            anchor_latents=anchor_latents,
            anchor_masks=anchor_masks,
            anchor_free=anchor_free,
        )

    with _stage_context(stage, "text_encode"):
        text_state = text_encode(
            dit=dit,
            batch_size=latent_state.batch_size,
            device=device,
            text_embedder=text_embedder,
            cached_text_embeds=cached_text_embeds,
        )

    with _stage_context(stage, "denoise"):
        latent_visual = denoise(
            dit=dit,
            latent_state=latent_state,
            text_state=text_state,
            scale_factor=scale_factor,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            device=device,
            tp_mesh=tp_mesh,
            lq_noise_scale=lq_noise_scale,
            cap_noise_timestep=cap_noise_timestep,
            prediction_target=prediction_target,
            channelcat_drop_threshold=channelcat_drop_threshold,
            rfg_scale=rfg_scale,
            piflow_params=piflow_params,
            progress_callback=progress_callback,
            scheduler=scheduler,
        )

    with _stage_context(stage, "vae_decode"):
        return vae_decode(
            vae=vae,
            latent_visual=latent_visual,
            batch_size=latent_state.batch_size,
            duration=latent_state.duration,
            height=latent_state.height,
            width=latent_state.width,
            vae_decode_batch=vae_decode_batch,
        )


def _sampler_accepts_stage(sampler: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(sampler).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == "stage" or parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _generate_kwargs(
    components: Any,
    run_config: Any,
    params: Any,
    scale_factor: tuple[float, ...],
    **inputs: Any,
) -> dict[str, Any]:
    return {
        **inputs,
        "dit": components.dit,
        "vae": components.vae,
        "scale_factor": scale_factor,
        "num_steps": run_config.num_steps,
        "anchor_free": True,
        "guidance_weight": 1.0,
        "scheduler_scale": float(getattr(params, "scheduler_scale", 5.0)),
        "seed": run_config.seed,
        "device": run_config.device,
        "tp_mesh": None,
        "lq_noise_scale": float(getattr(params, "lq_noise_scale", 0.7)),
        "lq_noise_type": getattr(params, "lq_noise_type", "ddpm"),
        "lq_channel_noise_scale": float(getattr(params, "lq_channel_noise_scale", 0.0)),
        "cap_noise_timestep": bool(getattr(params, "cap_noise_timestep", False)),
        "cached_text_embeds": getattr(components, "cached_text_embeds", None),
    }


def _as_tiles(sr_batch: torch.Tensor) -> list[torch.Tensor]:
    if sr_batch.ndim != 5:  # noqa: PLR2004
        raise ValueError(f"SR sampler must return [batch,C,T,H,W], got {tuple(sr_batch.shape)}")
    return [sample.float().cpu() for sample in sr_batch]


def _scale_progress_callback(
    progress_callback: Callable[[int], Any],
    chunk_size: int,
) -> Callable[[int], Any]:
    def update(n: int = 1) -> None:
        progress_callback(n * chunk_size)

    return update


def _loaded_lu_scales(components: Any) -> tuple[int, ...]:
    lu = getattr(components, "latent_upscaler", None)
    if lu is None:
        return ()
    scales = getattr(lu, "scales", None)
    return tuple(int(scale) for scale in scales) if scales is not None else (latent_upscaler_scale(lu),)


def _run_tile_batches(  # noqa: PLR0913
    tile_inputs: list[torch.Tensor],
    sr_components: Any,
    run_config: Any,
    scale_factor: tuple[float, ...],
    *,
    stage: StageCallback | None,
    offload: OffloadHandle,
    prepare_chunk: Callable[[list[torch.Tensor]], list[torch.Tensor]] | None = None,
    progress_callback: Callable[[int], Any] | None = None,
    scheduler: Any | None = None,
    sample_batch_size: int = 1,
) -> list[torch.Tensor]:
    """Run the shared batched sampler loop for latent or pixel tile inputs.

    DiT and SR VAE stay resident for the complete tile loop. Only the optional
    latent-upscaler is streamed per chunk.
    """
    outputs: list[torch.Tensor] = []
    sampler = getattr(sr_components, "sampler", None)
    params = _component_params(sr_components)
    piflow_params = _scheduler_piflow_params(scheduler) or getattr(sr_components.dit, "piflow_params", None)
    progress_steps = int(
        piflow_params["nfe"] if piflow_params is not None else
        run_config.num_steps - 1
    )
    with offload.use("dit", "vae"):
        for start in range(0, len(tile_inputs), run_config.tiles_batch_size):
            raw_chunk = tile_inputs[start : start + run_config.tiles_batch_size]
            chunk = raw_chunk
            if prepare_chunk is not None:
                with _stage_context(stage, "latent_upscaler"), offload.use("latent_upscaler"):
                    chunk = prepare_chunk(raw_chunk)

            input_kwargs: dict[str, Any]
            if prepare_chunk is None:
                input_kwargs = {"lq_videos": chunk}
            else:
                input_kwargs = {
                    "lq_latents": torch.cat(chunk, dim=0),
                    "n_samples": len(raw_chunk),
                }
            kwargs = _generate_kwargs(sr_components, run_config, params, scale_factor, **input_kwargs)
            # ``tile_inputs`` is tile-major for the native batch path. Advance
            # the seed once per tile, not once per flattened sample batch, so
            # batched and independent runs use the same tile seeds.
            kwargs["seed"] = run_config.seed + start // sample_batch_size
            chunk_progress = (
                _scale_progress_callback(progress_callback, len(raw_chunk))
                if progress_callback is not None
                else None
            )

            if sampler is None:
                outputs.extend(
                    _as_tiles(
                        run_stages(
                            **kwargs,
                            piflow_params=piflow_params,
                            stage=stage,
                            progress_callback=chunk_progress,
                            scheduler=scheduler,
                        )
                    )
                )
            else:
                if _sampler_accepts_stage(sampler):
                    kwargs["stage"] = stage
                outputs.extend(_as_tiles(sampler(**kwargs)))
                if chunk_progress is not None:
                    chunk_progress(progress_steps)
    return outputs


@torch.no_grad()
def run_tiled_sr(  # noqa: PLR0913
    lr_latent: torch.Tensor,
    sr_components: Any,
    run_config: Any,
    *,
    stage: StageCallback | None = None,
    offload: OffloadHandle | None = None,
    progress_callback: Callable[[int], Any] | None = None,
    scheduler: Any | None = None,
) -> torch.Tensor:
    """Run latent-upscaler SR over tiles and stitch the decoded result."""
    lu = latent_upscaler_for_scale(sr_components.latent_upscaler, run_config.resolution_scale)
    if lu is None:
        raise ValueError(
            f"The latent-upscaler path has no {run_config.resolution_scale}x model "
            f"(loaded LU scales: {_loaded_lu_scales(sr_components)}). "
            "Use run_tiled_sr_from_pixels() or load a matching LU."
        )
    if lr_latent.ndim != 4:  # noqa: PLR2004
        raise ValueError(f"lr_latent must have rank 4 [T,C,H,W], got {tuple(lr_latent.shape)}")

    params = _component_params(sr_components)
    visual_size = int(params.visual_size[0]) if isinstance(params.visual_size, list) else int(params.visual_size)
    _, _, pixel_grid = _tile_geometry(
        lr_latent.shape[-2] * _spatial_factor(sr_components),
        lr_latent.shape[-1] * _spatial_factor(sr_components),
        visual_size,
        run_config.resolution_scale,
        run_config.overlap,
        _spatial_factor(sr_components),
    )
    latent_grid = latent_tile_grid_from_pixel_grid(pixel_grid, _spatial_factor(sr_components))
    tile_inputs = extract_all_tiles(lr_latent, latent_grid)
    offload = offload or NoOpOffload()
    scale_factor = _scale_factor(params, visual_size)
    outputs = _run_tile_batches(
        tile_inputs,
        sr_components,
        run_config,
        scale_factor,
        stage=stage,
        offload=offload,
        progress_callback=progress_callback,
        scheduler=scheduler,
        prepare_chunk=lambda chunk: [
            upscale_lr_latent_tile(tile, lu, sr_components.vae, run_config.device) for tile in chunk
        ],
    )

    h = lr_latent.shape[-2] * _spatial_factor(sr_components)
    w = lr_latent.shape[-1] * _spatial_factor(sr_components)
    return (
        stitch_tiles_hanning(outputs, pixel_grid, h, w, scale=run_config.resolution_scale).clamp(0, 255).to(torch.uint8)
    )


@torch.no_grad()
def run_tiled_sr_from_pixels(
    lq_video: torch.Tensor,
    sr_components: Any,
    run_config: Any,
    *,
    stage: StageCallback | None = None,
    offload: OffloadHandle | None = None,
    progress_callback: Callable[[int], Any] | None = None,
    scheduler: Any | None = None,
) -> torch.Tensor:
    """Run pixel-input SR over tiles and stitch the decoded result."""
    if lq_video.ndim != 4:  # noqa: PLR2004
        raise ValueError(f"lq_video must have rank 4 [T,C,H,W], got {tuple(lq_video.shape)}")
    params = _component_params(sr_components)
    visual_size = int(params.visual_size[0]) if isinstance(params.visual_size, list) else int(params.visual_size)
    base, _tile_hw, grid = _tile_geometry(
        lq_video.shape[-2],
        lq_video.shape[-1],
        visual_size,
        run_config.resolution_scale,
        run_config.overlap,
        _spatial_factor(sr_components),
    )
    with _stage_context(stage, "prepare_latents"):
        tile_inputs = _upsample_tiles_to_base(extract_all_tiles(lq_video, grid), base[0], base[1])

    offload = offload or NoOpOffload()
    scale_factor = _scale_factor(params, visual_size)
    outputs = _run_tile_batches(
        tile_inputs,
        sr_components,
        run_config,
        scale_factor,
        stage=stage,
        offload=offload,
        progress_callback=progress_callback,
        scheduler=scheduler,
    )
    stitched = stitch_tiles_hanning(
        outputs,
        grid,
        lq_video.shape[-2],
        lq_video.shape[-1],
        scale=run_config.resolution_scale,
    )
    return stitched.clamp(0, 255).to(torch.uint8)


__all__ = [
    "SRLatentState",
    "SRTextState",
    "denoise",
    "prepare_latents",
    "run_stages",
    "run_tiled_sr",
    "run_tiled_sr_from_pixels",
    "text_encode",
    "vae_decode",
]
