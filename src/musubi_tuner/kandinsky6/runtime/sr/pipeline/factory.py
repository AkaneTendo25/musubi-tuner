"""Factory for constructing the packaged super-resolution pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from musubi_tuner.kandinsky6.runtime.sr import constants as sr_constants
from musubi_tuner.kandinsky6.runtime.sr.core.utils.offload import NoOpOffload, OffloadHandle
from musubi_tuner.kandinsky6.runtime.sr.pipeline.components import load_sr_components
from musubi_tuner.kandinsky6.runtime.sr.pipeline.config import SRConfig, load_sr_config_yaml
from musubi_tuner.kandinsky6.runtime.sr.pipeline.sr_pipeline import Kandinsky6SRPipeline


def load_sr_pipeline(  # noqa: PLR0913
    config: SRConfig | str | Path,
    device: str | torch.device,
    *,
    force: bool = False,
    checkpoint_path: str | Path | None = None,
    vae_path: str | Path | None = None,
    latent_upscaler_config: str | Path | None = None,
    resolution_scale: float | None = None,
    source_vae: torch.nn.Module | None = None,
    offload: OffloadHandle | None = None,
) -> Any:
    """Load the SR pipeline described by an :class:`SRConfig`.

    ``config`` may also be a path to a YAML with an ``sr:`` section (a
    standalone SR YAML or a full K6 pipeline YAML). ``None`` is returned when
    ``sr.enabled`` is false unless ``force=True`` (explicit benchmark or
    notebook flags); explicit path overrides win over the config values.
    ``checkpoint_path`` naming a ``Kandinsky6SRPipeline`` Diffusers bundle is
    enough: the VAE and the latent upscalers then come from the same bundle
    unless set separately.

    The factory only builds the pipeline: compile warmups (flex attention,
    ``compile_vae_decode``) are the caller's choice, as in the ``kandy-sr`` CLI.
    The embedding application decides how SR modules are offloaded: pass an
    ``offload`` handle and the DiT / VAE / latent upscaler are registered on it
    under the names the stages use (``"dit"``, ``"vae"``,
    ``"latent_upscaler"``). Without one every module stays resident.
    """
    sr_cfg = load_sr_config_yaml(config) if isinstance(config, (str, Path)) else config
    if not force and not sr_cfg.enabled:
        return None

    resolution_scale = resolution_scale if resolution_scale is not None else sr_cfg.resolution_scale
    checkpoint = str(checkpoint_path or sr_cfg.checkpoint_path or "")
    if not checkpoint:
        raise ValueError("SR is enabled but sr.checkpoint_path is empty; set it in the config or pass --sr-checkpoint")
    # Unset VAE / upscaler references are filled from the checkpoint's Diffusers bundle at load time.
    lu_config = latent_upscaler_config or sr_cfg.latent_upscaler_config
    resolved_vae_path = vae_path or sr_cfg.vae_path
    components = load_sr_components(
        checkpoint_path=checkpoint,
        vae_path=str(resolved_vae_path) if resolved_vae_path else None,
        device=str(device),
        latent_upscaler_config=str(lu_config) if lu_config else None,
        vae_backend=sr_cfg.vae_backend,
        dit_overrides=sr_cfg.dit_overrides,
        instruct_type_override=sr_cfg.instruct_type_override,
        lu_load_scales=sr_cfg.lu_load_scales,
    )

    sr_offload = offload or NoOpOffload()
    sr_offload.register("dit", components.dit)
    sr_offload.register("vae", components.vae)
    if components.latent_upscaler is not None:
        sr_offload.register("latent_upscaler", components.latent_upscaler)

    return Kandinsky6SRPipeline(
        dit=components.dit,
        vae=components.vae,
        latent_upscaler=components.latent_upscaler,
        device=device,
        sr_params=components.sr_params,
        num_steps=sr_cfg.num_steps,
        resolution_scale=resolution_scale,
        overlap=sr_cfg.overlap,
        tiles_batch_size=sr_cfg.tiles_batch_size,
        cached_text_embeds=components.cached_text_embeds,
        spatial_factor=int(sr_constants.VAE_SPATIAL_FACTOR),
        source_vae=source_vae,
        kvae_bridge=sr_cfg.kvae_bridge,
        mode=sr_cfg.mode,
        vae_backend=sr_cfg.vae_backend,
        offload=sr_offload,
    )


__all__ = ["load_sr_pipeline"]
