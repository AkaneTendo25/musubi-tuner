"""KVAE encode/decode helpers shared by encoding, training, and metrics.

The causal video KVAE carries its own pixel conventions — ``normalize_data``
(x / 128 - 1), ``denormalize_data`` ((x + 1) * 128) — and ``encode`` returns
``(latent, split_list)``. These helpers keep those conventions in one place.
"""

from __future__ import annotations

from pathlib import Path

import torch


def kvae_weights_path(checkpoint_prefix: str) -> str:
    """Return the kvae weights file for a sidecar prefix, preferring safetensors.

    The release format is ``{prefix}.safetensors`` (flat state dict), the
    training format ``{prefix}.ckpt`` (pickled ``{"state_dict": ...}``); both
    load through ``CachedCausalVAE.init_from_ckpt``, which dispatches on the
    suffix.

    Args:
        checkpoint_prefix: Local kvae sidecar prefix (``{prefix}.yaml`` sits
            next to the weights).

    Returns:
        Path to the weights file to load.
    """
    safetensors_path = f"{checkpoint_prefix}.safetensors"
    return safetensors_path if Path(safetensors_path).exists() else f"{checkpoint_prefix}.ckpt"


def module_dtype(module: torch.nn.Module) -> torch.dtype | None:
    """Return the dtype used by the first parameterized layer, if any."""
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return None


def cast_to_module_dtype(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Cast floating-point inputs to the module's parameter dtype."""
    dtype = module_dtype(module)
    if dtype is not None and value.is_floating_point() and value.dtype != dtype:
        return value.to(dtype=dtype)
    return value


def encode_pixels_to_latent(vae: torch.nn.Module, pixels: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
    """Encode ``(B, C, T, H, W)`` pixels in ``[0, 255]`` with the KVAE.

    Normalizes with the KVAE's own convention, then returns the latent
    (``(latent, split_list)[0]`` — the regularizer mode).

    Args:
        vae: Causal video KVAE.
        pixels: ``(B, C, T, H, W)`` tensor in ``[0, 255]``.
        sample: Unused — the KVAE always returns the regularizer mode.

    Returns:
        Latent ``(B, C, T', H', W')`` — not yet scaled by ``scaling_factor``.
    """
    del sample  # the KVAE always returns the regularizer mode
    x = vae.normalize_data(pixels)
    x = cast_to_module_dtype(vae, x)
    result = vae.encode(x)
    return result[0]


def decode_latent_to_uint8(vae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Decode a ``(B, C, T, H, W)`` latent to ``uint8`` pixels in ``[0, 255]``.

    The latent must already be unscaled (divided by ``scaling_factor``);
    denormalization uses the KVAE's ``denormalize_data``.

    Args:
        vae: Causal video KVAE.
        latent: ``(B, C, T, H, W)`` unscaled latent.

    Returns:
        ``(B, C, T_pixel, H, W)`` ``uint8`` pixels in ``[0, 255]``.
    """
    latent = cast_to_module_dtype(vae, latent)
    return denormalize_to_uint8(vae, vae.decode(latent).sample)


def decode_latent_to_float(vae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Decode an unscaled latent to float pixels in ``[0, 1]``.

    Args:
        vae: Causal video KVAE.
        latent: ``(B, C, T, H, W)`` unscaled latent.

    Returns:
        Float pixels with the KVAE denormalization applied.
    """
    latent = cast_to_module_dtype(vae, latent)
    return denormalize_to_float(vae, vae.decode(latent).sample)


def denormalize_to_float(vae: torch.nn.Module, decoded: torch.Tensor) -> torch.Tensor:
    """Convert a decoded tensor to float pixels in ``[0, 1]`` without quantization.

    Args:
        vae: Causal video KVAE.
        decoded: Decode output in the VAE's normalized pixel space.

    Returns:
        Float pixels in ``[0, 1]`` with the same shape.
    """
    return (vae.denormalize_data(decoded) / 255.0).clamp(0.0, 1.0)


def denormalize_to_uint8(vae: torch.nn.Module, decoded: torch.Tensor) -> torch.Tensor:
    """Convert an already-decoded tensor in ``[-1, 1]`` to ``uint8`` ``[0, 255]``.

    Uses the KVAE ``denormalize_data`` convention ((x + 1) * 128). For decode
    outputs that did not pass through :func:`decode_latent_to_uint8` (e.g.
    debug-clip dumps of already-decoded pixels).

    Args:
        vae: Causal video KVAE.
        decoded: Decode output in ``[-1, 1]`` (any shape).

    Returns:
        ``uint8`` pixels in ``[0, 255]`` with the same shape.
    """
    return (denormalize_to_float(vae, decoded) * 255.0).to(torch.uint8)
