from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class OGMGEState:
    video_coeff: float
    audio_coeff: float
    dominant: str
    discrepancy: float


def compute_ogm_ge_coefficients(
    video_loss: float | torch.Tensor,
    audio_loss: float | torch.Tensor,
    *,
    alpha: float = 0.3,
    eps: float = 1e-8,
) -> OGMGEState:
    """Compute OGM-GE style modality attenuation coefficients from loss disparity.

    Lower denoising loss is treated as the dominant / faster-learning modality and
    gets attenuated, while the weaker modality keeps coefficient 1.0.
    """
    video_value = float(video_loss.detach().item() if isinstance(video_loss, torch.Tensor) else video_loss)
    audio_value = float(audio_loss.detach().item() if isinstance(audio_loss, torch.Tensor) else audio_loss)

    if video_value < audio_value:
        discrepancy = max((audio_value - video_value) / max(audio_value, eps), 0.0)
        video_coeff = 1.0 - torch.tanh(torch.tensor(alpha * discrepancy)).item()
        return OGMGEState(
            video_coeff=max(video_coeff, 0.0),
            audio_coeff=1.0,
            dominant="video",
            discrepancy=discrepancy,
        )

    if audio_value < video_value:
        discrepancy = max((video_value - audio_value) / max(video_value, eps), 0.0)
        audio_coeff = 1.0 - torch.tanh(torch.tensor(alpha * discrepancy)).item()
        return OGMGEState(
            video_coeff=1.0,
            audio_coeff=max(audio_coeff, 0.0),
            dominant="audio",
            discrepancy=discrepancy,
        )

    return OGMGEState(video_coeff=1.0, audio_coeff=1.0, dominant="balanced", discrepancy=0.0)


def maybe_add_ogm_ge_gradient_noise(
    network,
    *,
    video_coeff: float,
    audio_coeff: float,
    noise_std_scale: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> None:
    """Inject GE noise into the attenuated modality gradients.

    This is intentionally conservative: by default noise is off (scale=0.0) and
    only activates when the caller opts in via CLI.
    """
    if noise_std_scale <= 0.0:
        return

    lora_modules = getattr(network, "unet_loras", None)
    if not lora_modules:
        return

    for lora in lora_modules:
        is_audio = "audio_" in lora.lora_name
        coeff = audio_coeff if is_audio else video_coeff
        attenuation = max(0.0, 1.0 - float(coeff))
        if attenuation <= 0.0:
            continue

        for param in lora.parameters():
            grad = param.grad
            if grad is None:
                continue
            grad_std = float(grad.detach().float().std(unbiased=False).item())
            if grad_std <= 0.0:
                grad_std = 1.0
            noise = torch.randn(
                grad.shape,
                device=grad.device,
                dtype=grad.dtype,
                generator=generator,
            )
            grad.add_(noise * (grad_std * attenuation * noise_std_scale))
