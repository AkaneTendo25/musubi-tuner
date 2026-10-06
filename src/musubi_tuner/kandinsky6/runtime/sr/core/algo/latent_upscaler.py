"""Latent upscaler loading and execution helpers used by SR inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pydantic
import torch
from loguru import logger
from omegaconf import OmegaConf

from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.config import ModelConfig
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.factory import build_upsampler
from musubi_tuner.kandinsky6.runtime.sr.core.algo.checkpoint import load, validate_local_path
from musubi_tuner.kandinsky6.runtime.sr.core.algo.safetensors_io import is_safetensors_path, load_safetensors_state_dict

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn


SUPPORTED_TARGET_SCALES = ("2x", "4x")


class LatentUpscalerBank(torch.nn.ModuleDict):
    """Per-scale collection of frozen latent upscalers."""

    def for_scale(self, scale: int) -> nn.Module | None:
        """Return the LU trained for ``scale`` or ``None``."""
        return self._modules.get(f"{int(scale)}x")

    @property
    def scales(self) -> tuple[int, ...]:
        """Return the sorted upscale factors available in the bank."""
        return tuple(sorted(int(key.removesuffix("x")) for key in self))


def latent_upscaler_scale(upscaler: nn.Module) -> int:
    """Return the loaded LU's fixed spatial upscale factor."""
    raw = str(getattr(upscaler, "target_scale", "4x"))
    if raw not in SUPPORTED_TARGET_SCALES:
        msg = f"Unexpected latent upscaler target_scale={raw!r}; expected one of {SUPPORTED_TARGET_SCALES}."
        raise ValueError(msg)
    return int(raw.removesuffix("x"))


def latent_upscaler_for_scale(
    latent_upscaler: nn.Module | None,
    scale: int,
) -> nn.Module | None:
    """Return the loaded latent upscaler matching ``scale``."""
    if latent_upscaler is None:
        return None
    for_scale = getattr(latent_upscaler, "for_scale", None)
    if isinstance(latent_upscaler, LatentUpscalerBank) or callable(for_scale):
        module = for_scale(scale)
        if module is None:
            logger.debug(
                "No x{} entry in the latent upscaler bank (scales: {}) — using the pixel/bilinear path",
                scale,
                latent_upscaler.scales,
            )
        return module
    lu_scale = latent_upscaler_scale(latent_upscaler)
    if lu_scale == scale:
        return latent_upscaler
    logger.debug(
        "Latent upscaler is fixed {}x but the batch is x{} — using the pixel/bilinear path for it",
        lu_scale,
        scale,
    )
    return None


def load_latent_upscaler(
    conf: DictConfig,
    device: str | torch.device,
    vae_scaling_factor: float,
) -> nn.Module | None:
    """Build and load the frozen latent upscaler or multi-scale bank."""
    upscaler_conf = getattr(conf, "latent_upscaler", None)
    if upscaler_conf is None or not getattr(upscaler_conf, "enabled", False):
        return None

    models_conf = getattr(upscaler_conf, "models", None)
    if models_conf is None:
        return load_single_latent_upscaler(upscaler_conf, device, vae_scaling_factor)

    bank = LatentUpscalerBank()
    for section in models_conf:
        module = load_single_latent_upscaler(section, device, vae_scaling_factor)
        key = str(module.target_scale)  # type: ignore[union-attr]
        if key in bank:
            msg = f"Duplicate latent_upscaler.models entry for target_scale={key!r}"
            raise ValueError(msg)
        bank[key] = module
    logger.info("Loaded latent upscaler bank with scales {}", bank.scales)
    return bank


def resolve_lu_state_dict(
    checkpoint_path: str, *, use_ema: bool, state_prefix: str | None = None
) -> dict[str, torch.Tensor]:
    """Load the LU state dict from a training ``.pt`` or a release ``.safetensors``.

    A training checkpoint is a pickled dict with ``"ema"`` and ``"model"``
    sub-dicts; ``use_ema`` picks between them. A ``.safetensors`` checkpoint is
    already a flat state dict — the EMA/model choice was baked in at conversion
    time — so ``use_ema`` is ignored. A Diffusers bundle keeps every upscaler
    of the bank in one ``.safetensors`` file; ``state_prefix`` selects one.

    Deserialization stays on CPU: the module is moved to the target device
    after loading, avoiding an unnecessary transient allocation on GPU 0.

    Args:
        checkpoint_path: Local LU checkpoint (``.pt`` or ``.safetensors``).
        use_ema: Select the ``"ema"`` sub-dict of a ``.pt`` checkpoint
            (``"model"`` otherwise); ignored for ``.safetensors``.
        state_prefix: Key prefix of this upscaler inside a shared
            ``.safetensors`` file (e.g. ``"_models.0."`` for the 2x upscaler);
            ``None`` for a file holding a single upscaler.

    Returns:
        Flat ``name -> tensor`` state dict on CPU.
    """
    if is_safetensors_path(checkpoint_path):
        state_dict = load_safetensors_state_dict(checkpoint_path, key_prefix=state_prefix)
        logger.info("Loaded latent upscaler safetensors weights from {} (prefix={!r})", checkpoint_path, state_prefix)
        return state_dict
    ckpt = load(checkpoint_path, map_location="cpu", weights_only=False)
    state_key = "ema" if use_ema else "model"
    logger.info("Loaded latent upscaler weights (key={!r}) from {}", state_key, checkpoint_path)
    return ckpt[state_key]


def load_single_latent_upscaler(
    upscaler_conf: DictConfig,
    device: str | torch.device,
    vae_scaling_factor: float,
) -> nn.Module:
    """Build and load one frozen latent upscaler from a local checkpoint."""
    raw_model = OmegaConf.to_container(upscaler_conf.model, resolve=True)
    model_conf = pydantic.TypeAdapter(ModelConfig).validate_python(raw_model)
    upscaler = build_upsampler(model_conf)

    checkpoint_path: str = upscaler_conf.checkpoint
    validate_local_path(checkpoint_path, "latent upscaler checkpoint path")
    use_ema: bool = getattr(upscaler_conf, "use_ema", True)
    state_prefix: str | None = getattr(upscaler_conf, "state_prefix", None)
    upscaler.load_state_dict(resolve_lu_state_dict(checkpoint_path, use_ema=use_ema, state_prefix=state_prefix))

    target_scale = getattr(upscaler_conf, "target_scale", "4x")
    if target_scale not in SUPPORTED_TARGET_SCALES:
        msg = f"latent_upscaler.target_scale must be one of {SUPPORTED_TARGET_SCALES}, got {target_scale!r}"
        raise ValueError(msg)

    upscaler.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    upscaler.scaling_factor = vae_scaling_factor  # type: ignore[assignment]
    upscaler.target_scale = target_scale  # type: ignore[assignment]
    return upscaler


def run_latent_upscaler(upscaler: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Forward a scaled LQ latent through its configured LU entry."""
    target_scale = getattr(upscaler, "target_scale", None)
    if target_scale is not None:
        entry = f"x{latent_upscaler_scale(upscaler)}"
        try:
            return upscaler(z, entry=entry, return_intermediates=False)
        except TypeError:
            # Flat upscalers do not accept the multi-scale keyword arguments.
            return upscaler(z)
    return upscaler(z)
