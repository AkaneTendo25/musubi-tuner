"""MiniMax Music 3 DAV decoder for ComfyUI ``minimax_music3_dav`` weights.

Architecture adapted from the Apache-2.0 Diffusers MiniMax Music 3 integration
(PR #14456).  Names intentionally retain the ComfyUI checkpoint layout.
"""

from __future__ import annotations

import math

import torch
from accelerate import init_empty_weights
from torch import nn

from musubi_tuner.utils.safetensors_utils import load_safetensors


def _weight_norm(module: nn.Module) -> nn.Module:
    # Released checkpoints store weight-normalization parameters as
    # ``weight_g`` and ``weight_v`` tensors.
    return nn.utils.weight_norm(module)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(x)
        return x + torch.sin(alpha * x).square() / (alpha + 1e-9)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            _weight_norm(nn.Conv1d(dim, dim, 7, dilation=dilation, padding=3 * dilation)),
            Snake1d(dim),
            _weight_norm(nn.Conv1d(dim, dim, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        if residual.shape[-1] != x.shape[-1]:
            trim = (x.shape[-1] - residual.shape[-1]) // 2
            x = x[..., trim : x.shape[-1] - trim]
        return x + residual


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            _weight_norm(
                nn.ConvTranspose1d(input_dim, output_dim, 2 * stride, stride=stride, padding=math.ceil(stride / 2))
            ),
            ResidualUnit(output_dim, 1),
            ResidualUnit(output_dim, 3),
            ResidualUnit(output_dim, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, 1),
            ResidualUnit(dim // 2, 3),
            ResidualUnit(dim // 2, 9),
            Snake1d(dim // 2),
            _weight_norm(nn.Conv1d(dim // 2, dim, 2 * stride, stride=stride, padding=math.ceil(stride / 2))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """Released DAV encoder: 44.1 kHz mono waveform -> 1024 features at /512."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = [_weight_norm(nn.Conv1d(1, 64, 7, padding=3))]
        channels = 64
        for stride in (2, 4, 8, 8):
            channels *= 2
            layers.append(EncoderBlock(channels, stride))
        layers.extend((Snake1d(channels), _weight_norm(nn.Conv1d(channels, 1024, 3, padding=1))))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = [_weight_norm(nn.Conv1d(1024, 1536, 7, padding=3))]
        channels = 1536
        output_dim = channels
        for index, stride in enumerate((8, 8, 4, 2)):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            layers.append(DecoderBlock(input_dim, output_dim, stride))
        layers.extend((Snake1d(output_dim), _weight_norm(nn.Conv1d(output_dim, 1, 7, padding=3)), nn.Tanh()))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MiniMaxMusic3DAV(nn.Module):
    sampling_rate = 44100

    def __init__(self):
        super().__init__()
        self.dec_in_proj = nn.Conv1d(64, 1024, 1)
        self.decoder = Decoder()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch, channels, frames = latent.shape
        if channels != 128:
            raise ValueError(f"DAV expects 128 latent channels, got {channels}")
        folded = latent.reshape(batch * 2, 64, frames)
        waveform = self.decoder(self.dec_in_proj(folded))
        return waveform.reshape(batch, 2, -1)

    forward = decode


class MiniMaxMusic3DAVAutoencoder(MiniMaxMusic3DAV):
    """Full official DAV posterior-mean encoder plus released decoder.

    The ComfyUI DAV intentionally omits these encoder modules. This class is
    only loadable from the official ``dav.pth`` and is used for training-cache
    validation/creation, never as a substitute for ComfyUI DiT weights.
    """

    hop_length = 512

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mean_proj = nn.Conv1d(1024, 64, 1)
        self.logs_proj = nn.Conv1d(1024, 64, 1)

    def encode(self, waveform: torch.Tensor, sample: bool = False, generator: torch.Generator | None = None) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 2:
            raise ValueError(f"DAV expects stereo [B,2,samples], got {tuple(waveform.shape)}")
        batch, _, samples = waveform.shape
        padding = (-samples) % self.hop_length
        if padding:
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        features = self.encoder(waveform.reshape(batch * 2, 1, -1))
        mean = self.mean_proj(features)
        if sample:
            log_scale = self.logs_proj(features).clamp(-30.0, 20.0)
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
            mean = mean + noise * log_scale.exp()
        # The strided encoder rounds up, whereas the DiT duration contract uses
        # floor division. Crop the padded tail so cached targets and AR-derived
        # conditioning have identical temporal geometry.
        target_frames = max(1, samples // self.hop_length)
        return mean[..., :target_frames].reshape(batch, 128, target_frames)


def load_official_dav_autoencoder(
    checkpoint: str, *, device: str | torch.device = "cpu", dtype: torch.dtype | None = None
) -> MiniMaxMusic3DAVAutoencoder:
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    # ``flow.*`` belongs to the posterior regularizer used to train DAV. The
    # production decoder and the DiT consume the folded posterior space.
    state_dict = {key: value for key, value in state_dict.items() if not key.startswith("flow.")}
    if dtype is not None:
        state_dict = {key: value.to(dtype) if value.is_floating_point() else value for key, value in state_dict.items()}
    with init_empty_weights():
        model = MiniMaxMusic3DAVAutoencoder()
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Invalid official DAV checkpoint; missing={missing[:10]}, unexpected={unexpected[:10]}")
    return model.requires_grad_(False).eval()


def load_comfy_dav(
    checkpoint: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    disable_mmap: bool = False,
) -> MiniMaxMusic3DAV:
    state_dict = load_safetensors(checkpoint, device=device, disable_mmap=disable_mmap)
    if dtype is not None:
        state_dict = {key: value.to(dtype) if value.is_floating_point() else value for key, value in state_dict.items()}
    with init_empty_weights():
        model = MiniMaxMusic3DAV()
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Invalid ComfyUI DAV checkpoint; missing={missing[:10]}, unexpected={unexpected[:10]}")
    return model.requires_grad_(False).eval()
