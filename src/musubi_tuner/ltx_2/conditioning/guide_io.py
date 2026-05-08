"""Encode image/video paths into LTX-2 latent guides via the existing VAE encoder.

Used by both inference (inject reference at sample time) and training (inject
reference at train time). Emits 5D `[B, C, T, H, W]` guide latents matching the
shape requirements of LatentIndexGuide and KeyframeGuide.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _load_image_as_tensor(path: str, target_h: int, target_w: int) -> torch.Tensor:
    """Load image at path, return [3, T=1, H, W] in [-1, 1] (RGB)."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    # HWC -> CHW -> CTHW (T=1)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(1).contiguous()
    return t  # [3, 1, H, W]


def _load_video_as_tensor(
    path: str, target_h: int, target_w: int, max_frames: int
) -> torch.Tensor:
    """Load up to `max_frames` frames from a video at path, return [3, T, H, W] in [-1, 1]."""
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, width=target_w, height=target_h)
        n = min(len(vr), max_frames)
        frames = vr.get_batch(list(range(n))).asnumpy()  # [T, H, W, 3], uint8
        arr = frames.astype(np.float32) / 127.5 - 1.0
        t = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # [3, T, H, W]
        return t
    except ImportError:
        pass
    # Fallback: open via OpenCV.
    import cv2  # type: ignore
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {path}")
    arr = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
    return t


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def encode_guide_latent(
    path: str,
    vae: torch.nn.Module,
    *,
    target_h: int,
    target_w: int,
    target_frames: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Encode an image or short video at `path` into a 5D guide latent [1, C, T_lat, H_lat, W_lat].

    For images: produces a 1-frame latent (T_lat=1).
    For videos: produces ((T_pixel - 1) // temporal_factor + 1) latent frames.

    The VAE is expected to be the LTX-2 video VAE encoder; pass via
    `SingleGPUModelBuilder(...).build(...)` or the cache-side `vae_encoder`.
    """
    if device is None:
        device = next(vae.parameters()).device
    if dtype is None:
        dtype = next(vae.parameters()).dtype

    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        pixel = _load_image_as_tensor(path, target_h, target_w).unsqueeze(0)  # [1, 3, 1, H, W]
        if target_frames > 1:
            # Repeat the image across target_frames so the encoder sees a short clip.
            pixel = pixel.expand(-1, -1, target_frames, -1, -1).contiguous()
    elif ext in _VIDEO_EXTS:
        pixel = _load_video_as_tensor(path, target_h, target_w, max_frames=target_frames).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported guide file extension: {path!r}")

    # Pad pixel-frame count to (1 + 8N) form expected by LTX-2 video VAE.
    n_frames = int(pixel.shape[2])
    if n_frames > 1 and (n_frames - 1) % 8 != 0:
        pad = 8 - ((n_frames - 1) % 8)
        last = pixel[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
        pixel = torch.cat([pixel, last], dim=2)

    pixel = pixel.to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae(pixel)
    return latent.detach()  # [1, C_lat, T_lat, H_lat, W_lat]


def encode_guide_specs(
    specs,
    vae: torch.nn.Module,
    *,
    target_h_lat: int,
    target_w_lat: int,
    spatial_factor: int = 32,
    target_frames_pixel: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[Tuple[int, torch.Tensor, float]]:
    """Encode a list of `{frame_idx, path, strength}` specs into latents.

    Returns a list of (frame_idx, latent_5d, strength) tuples ready to wrap
    in `LatentIndexGuide` or `KeyframeGuide`.
    """
    pixel_h = target_h_lat * spatial_factor
    pixel_w = target_w_lat * spatial_factor
    out: list[Tuple[int, torch.Tensor, float]] = []
    for spec in specs or []:
        try:
            latent = encode_guide_latent(
                spec["path"], vae,
                target_h=pixel_h, target_w=pixel_w,
                target_frames=target_frames_pixel,
                device=device, dtype=dtype,
            )
            out.append((int(spec["frame_idx"]), latent, float(spec.get("strength", 1.0))))
        except Exception as e:
            logger.warning("Failed to encode guide %r: %s", spec.get("path"), e)
    return out
