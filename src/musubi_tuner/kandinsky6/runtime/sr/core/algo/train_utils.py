"""Inference-only subset of training utilities.

Vendored trim: only the functions the inference pipeline needs
(`get_visual_size`, `degrade_lq_latent`, `get_sparse_params`). The full training
module additionally imports `musubi_tuner.kandinsky6.runtime.sr.train.losses.face`, which pulls heavy
face-detection deps not needed at inference, and applies a training-time linear
``P`` warmup schedule that is irrelevant at inference (the target ``P`` is used
directly).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch

from musubi_tuner.kandinsky6.runtime.sr import constants
from musubi_tuner.kandinsky6.runtime.sr.constants import RESOLUTIONS


def get_visual_size(x: torch.Tensor) -> int:
    """Return the resolution key matching a visual latent tensor's spatial dims.

    Scales the tensor's spatial dims by the VAE spatial factor and looks them
    up in the ``RESOLUTIONS`` registry. Reads ``constants.VAE_SPATIAL_FACTOR``
    late-bound so ``set_vae_factors`` (16 for the KVAE) applies — a
    ``from``-import would freeze the import-time default.

    Args:
        x: Visual latent tensor of shape ``(T, H, W, C)``.

    Returns:
        The matching resolution key.

    Raises:
        ValueError: If tensor dimensions do not match any known resolution.
    """
    actual_size = (x.shape[1] * constants.VAE_SPATIAL_FACTOR, x.shape[2] * constants.VAE_SPATIAL_FACTOR)
    for key, value in RESOLUTIONS.items():
        if actual_size in value:
            return key
    valid_sizes = {size for sizes in RESOLUTIONS.values() for size in sizes}
    msg = (
        f"Visual tensor spatial dimensions {actual_size} do not match any known resolution. "
        f"Tensor shape: {x.shape}. Valid resolutions: {valid_sizes}"
    )
    raise ValueError(msg)


def degrade_lq_latent(
    lq_latent: torch.Tensor,
    noise_scale: float = 0.7,
    noise_type: Literal["linear", "ddpm"] = "linear",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mix random Gaussian noise into an LQ latent for data augmentation.

    Args:
        lq_latent: LQ latent tensor of arbitrary shape.
        noise_scale: Noise fraction ``s`` in ``(0, 1)``. When ``0``, the tensor
            is returned unchanged.
        noise_type: ``"linear"`` for ``(1-s)*lq + s*eps`` or ``"ddpm"`` for
            ``sqrt(1-s²)*lq + s*eps`` (variance-preserving).
        generator: Optional RNG generator for deterministic noise sampling.

    Returns:
        Degraded LQ latent with the same shape and dtype as the input.
    """
    if noise_scale <= 0:
        return lq_latent
    eps = torch.randn(lq_latent.shape, device=lq_latent.device, dtype=lq_latent.dtype, generator=generator)
    if noise_type == "ddpm":
        return (1 - noise_scale**2) ** 0.5 * lq_latent + noise_scale * eps
    return (1 - noise_scale) * lq_latent + noise_scale * eps


def get_sparse_params(
    model: Any,
    visual: torch.Tensor,
    visual_cu_seqlens: torch.Tensor,
) -> dict[str, Any] | None:
    """Build sparse-attention parameters for the given visual tensor and model.

    Supports ``"nabla"`` and ``"nabla_framewise_causal"`` attention types;
    returns ``None`` for dense attention. The target ``P`` is used directly (the
    training-time linear warmup schedule does not apply at inference).

    Args:
        model: Model with ``patch_size`` and ``attention_params`` attributes.
        visual: Visual latent tensor of shape ``(T, H, W, C)``.
        visual_cu_seqlens: Cumulative frame counts for each sample in the batch.

    Returns:
        A dict of sparse attention parameters, or ``None`` for dense attention.

    Raises:
        ValueError: If ``model.patch_size[0]`` is not 1.
    """
    if model.patch_size[0] != 1:
        msg = f"Expected model.patch_size[0] == 1, got {model.patch_size[0]}"
        raise ValueError(msg)
    t, h, w, _ = visual.shape
    t, h, w = (
        t // model.patch_size[0],
        h // model.patch_size[1],
        w // model.patch_size[2],
    )
    visual_size = get_visual_size(visual)
    attention_configs = model.attention_params
    try:
        attention_params = attention_configs[visual_size]
    except KeyError:
        # JSON object keys are always strings, while the native YAML config
        # uses integer resolution keys.
        attention_params = attention_configs[str(visual_size)]

    def config_value(name: str, default: Any = None) -> Any:
        if isinstance(attention_params, Mapping):
            return attention_params.get(name, default)
        return getattr(attention_params, name, default)

    if config_value("type") == "nabla":
        return {
            "attention_type": config_value("type"),
            "to_fractal": True,
            "P": config_value("P"),
            "wT": config_value("wT"),
            "wW": config_value("wW"),
            "wH": config_value("wH"),
            "add_sta": config_value("add_sta"),
            "visual_shape": (t, h, w),
            "visual_seqlens": visual_cu_seqlens,
            "method": config_value("method", "topcdf"),
        }
    if config_value("type") == "nabla_framewise_causal":
        return {
            "attention_type": config_value("type"),
            "to_fractal": True,
            "P": config_value("P"),
            "wT": config_value("wT"),
            "wW": config_value("wW"),
            "wH": config_value("wH"),
            "add_sta": config_value("add_sta"),
            "mf": config_value("mf"),
            "visual_shape": (t, h, w),
            "visual_seqlens": visual_cu_seqlens,
        }
    return None
