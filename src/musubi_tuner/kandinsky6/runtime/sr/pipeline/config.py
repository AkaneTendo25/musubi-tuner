"""Configuration for the K6 super-resolution pipeline.

:class:`SRConfig` is the single source of tunable SR settings. Defaults live
on the model; a YAML config (the ``sr:`` section of a k6 pipeline YAML, or a
standalone SR YAML with the same section) overrides them, and explicit CLI
options override the YAML. Model-contract constants (trained fps/frame
budget, resolutions, delivery tiers) are NOT configurable — they live in
:mod:`musubi_tuner.kandinsky6.runtime.sr.constants`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from musubi_tuner.kandinsky6.runtime.sr.pipeline.hub import is_hub_reference

# Total upscale factor (2.25 = x1.125 pixel pre-upscale + x2 tiling).
ResolutionScale = Literal[2, 2.25, 4]
# Tiling factor of the DiT pass, after any pre-upscale.
TilingScale = Literal[2, 4]
TargetResizeMode = Literal["fit", "exact"]
VaeBackend = Literal["torch", "magi"]


class SRConfig(BaseModel):
    """SR settings: the K6-embedded route and the standalone ``kandy-sr`` CLI.

    ``checkpoint_path`` has no baked-in default — set it in the YAML (see
    ``musubi_tuner.kandinsky6.runtime.sr/configs/sr_release_hf.yaml``) or pass the CLI option. When
    it names a ``Kandinsky6SRPipeline`` Diffusers bundle, ``vae_path`` and
    ``latent_upscaler_config`` may stay unset: the VAE and the latent
    upscalers then come from the same bundle. With a native DiT checkpoint
    both are required.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    # --- Model paths (no in-code defaults) ---
    # A Diffusers bundle (Hub repo id or local dir), or a native DiT checkpoint.
    checkpoint_path: str | None = None
    # None = the VAE of the checkpoint's bundle.
    vae_path: str | None = None
    source_vae_path: str | None = None
    # None = the upscalers of the checkpoint's bundle; the literal string
    # "none" disables the LU (pixel path).
    latent_upscaler_config: str | None = None
    # --- Route / backend ---
    mode: Literal["pixel", "latent"] = "pixel"
    kvae_bridge: bool = False
    # KVAE compilation backend: "torch" (torch.compile, no extra deps) or "magi"
    # (MagiCompiler, needs the `magi` extra).
    vae_backend: VaeBackend = "torch"
    # CUDA device the pipeline runs on ("cuda:0", "cuda:1", ...). The CLI
    # --device option overrides it, like every other setting.
    device: str = "cuda:0"
    # --- DiT behavioural overrides ---
    # setattr KEY=VALUE overrides applied to the loaded DiT; None = none.
    dit_overrides: dict[str, Any] | None = None
    # Post-load ``instruct_type`` override ("noise" starts denoising from the
    # LQ latent); None keeps the checkpoint config's value.
    instruct_type_override: str | None = "noise"
    # --- Sampling / tiling ---
    resolution_scale: ResolutionScale = 2.25
    num_steps: int = Field(default=5, ge=2)
    # Minimum overlap between neighbouring tiles as a fraction of the tile
    # size; the grid uses the fewest tiles with uniform overlap >= this.
    overlap: float = Field(default=0.20, ge=0.0, lt=1.0)
    tiles_batch_size: int = Field(default=1, gt=0)
    seed: int = 42
    # --- Latent upscaler loading ---
    # Bank entries (by target_scale, e.g. ["2x"]) loaded eagerly at startup;
    # None = every entry. Unlisted entries lazy-load on first use.
    lu_load_scales: tuple[str, ...] | None = None
    # --- Output ---
    # Delivery tier name, explicit WxH, or None to keep raw SR dimensions.
    target_resolution: str | None = None
    target_resize_mode: TargetResizeMode = "fit"


PATH_FIELDS = ("checkpoint_path", "vae_path", "source_vae_path", "latent_upscaler_config")


def path_from_config(value: str | None, config_path: Path) -> str | None:
    """Resolve a config path relative to the YAML file when necessary.

    Hugging Face references (a repo id, or a component of a Diffusers bundle
    repo such as ``namespace/name/vae``) pass through untouched — they are
    resolved to local snapshot paths at load time (see ``musubi_tuner.kandinsky6.runtime.sr.pipeline.hub``).
    """
    if value is None:
        return None
    if is_hub_reference(value):
        return value
    path = Path(value)
    return str(path if path.is_absolute() else config_path.absolute().parent / path)


def load_sr_config_yaml(config_path: str | Path) -> SRConfig:
    """Load :class:`SRConfig` from a YAML file.

    Accepts either a standalone SR YAML with an ``sr:`` section (see
    ``musubi_tuner.kandinsky6.runtime.sr/configs/sr_release_local.yaml``) or a full K6 pipeline
    YAML — only the ``sr:`` section is read. Relative path fields resolve
    against the YAML's directory.

    Raises:
        ValueError: If the YAML has no ``sr:`` section.
    """
    config_path = Path(config_path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict) or "sr" not in data:
        raise ValueError(f"{config_path} has no 'sr:' section")
    sr = SRConfig.model_validate(data["sr"])
    return sr.model_copy(
        update={
            field: path_from_config(getattr(sr, field), config_path)
            for field in PATH_FIELDS
            if getattr(sr, field) is not None and getattr(sr, field).lower() != "none"
        }
    )


__all__ = [
    "PATH_FIELDS",
    "ResolutionScale",
    "SRConfig",
    "TargetResizeMode",
    "TilingScale",
    "VaeBackend",
    "load_sr_config_yaml",
    "path_from_config",
]
