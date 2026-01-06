from __future__ import annotations

from typing import Any, Optional

import torch


class VideoVAE(torch.nn.Module):
    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.latents_mean: Optional[torch.Tensor] = None
        self.latents_std: Optional[torch.Tensor] = None
        self.scaling_factor: float = 1.0

        self.spatial_downsample_factor: int = 8
        self.temporal_downsample_factor: int = 4

    def to_device(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.to(self.device)

    def to_dtype(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.to(dtype=dtype)

    def encode(self, videos: torch.Tensor):
        raise NotImplementedError(
            "musubi_tuner.ltxv2.video_vae.VideoVAE is a placeholder. Replace with official LTXV2 video VAE implementation."
        )

    def decode(self, zs):
        raise NotImplementedError(
            "musubi_tuner.ltxv2.video_vae.VideoVAE is a placeholder. Replace with official LTXV2 video VAE implementation."
        )
