from __future__ import annotations

from typing import Any, Optional

import torch


class AudioVAE(torch.nn.Module):
    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device

    def to_device(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.to(self.device)

    def to_dtype(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.to(dtype=dtype)

    def encode(self, audio: dict[str, torch.Tensor]):
        raise NotImplementedError(
            "musubi_tuner.ltxv2.audio_vae.AudioVAE is a placeholder. Replace with official LTXAV audio VAE implementation."
        )

    def decode(self, latents: torch.Tensor, audio_sample_rate: int = 48000):
        raise NotImplementedError(
            "musubi_tuner.ltxv2.audio_vae.AudioVAE is a placeholder. Replace with official LTXAV audio VAE implementation."
        )
