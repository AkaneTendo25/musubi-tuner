"""YuE2 Oobleck VAE (48 kHz stereo <-> 64-channel latents at 25 Hz), FP32 only, posterior mean.

Oobleck and SnakeBeta derived from stable-audio-tools a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45.
Copyright (c) 2023 Stability AI; Copyright (c) 2022 NVIDIA CORPORATION. MIT (see THIRD_PARTY_NOTICES.md and
tests/yue2_ref/licenses).
Vendored from the official ``yue2_infer`` ``modeling_vae.py`` as a plain ``nn.Module`` (no ``PretrainedConfig``), with
legacy ``torch.nn.utils.weight_norm`` so the released ``weight_g``/``weight_v`` keys load directly.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

HOP = 1920
LATENT_DIM = 64


def default_vae_configs() -> tuple[dict, dict]:
    """Encoder / decoder configs of the released m-a-p/YuE2-Vae."""
    encoder = dict(
        in_channels=2, channels=64, c_mults=[1, 2, 4, 8, 16, 32], strides=[2, 2, 4, 4, 5, 6], latent_dim=128, use_snake=True
    )
    decoder = dict(
        out_channels=2,
        channels=64,
        c_mults=[1, 2, 4, 8, 16, 32],
        strides=[2, 2, 4, 4, 5, 6],
        latent_dim=64,
        use_snake=True,
        snake_type="vanilla",
        use_filter=False,
        final_tanh=False,
    )
    return encoder, decoder


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake_beta(x, alpha, beta)


def get_activation(activation: Literal["elu", "snake", "none"], channels=None) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, act_type):
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation(act_type, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation, padding=padding),
            get_activation(act_type, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x) + x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, 1, act_type),
            ResidualUnit(in_channels, in_channels, 3, act_type),
            ResidualUnit(in_channels, in_channels, 9, act_type),
            get_activation(act_type, channels=in_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super().__init__()
        upsample_layer = WNConvTranspose1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
        )
        self.layers = nn.Sequential(
            get_activation(act_type, channels=in_channels),
            upsample_layer,
            ResidualUnit(out_channels, out_channels, 1, act_type),
            ResidualUnit(out_channels, out_channels, 3, act_type),
            ResidualUnit(out_channels, out_channels, 9, act_type),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()
        if antialias_activation:
            raise ValueError("The released encoder does not use antialias_activation")
        self.in_channels = in_channels
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)]
        act_type = "snake" if use_snake else "elu"
        for i in range(self.depth - 1):
            layers.append(
                EncoderBlock(
                    in_channels=c_mults[i] * channels, out_channels=c_mults[i + 1] * channels, stride=strides[i], act_type=act_type
                )
            )
        layers.extend(
            [
                get_activation(act_type, channels=c_mults[-1] * channels),
                WNConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, padding=1),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        use_snake=False,
        snake_type="vanilla",
        antialias_activation=False,
        use_nearest_upsample=False,
        use_filter=False,
        final_tanh=True,
    ):
        super().__init__()
        if antialias_activation or use_nearest_upsample or use_filter:
            raise ValueError("Unsupported option for the released decoder")
        if use_snake and snake_type != "vanilla":
            raise ValueError("The released decoder uses vanilla SnakeBeta")
        self.out_channels = out_channels
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1] * channels, kernel_size=7, padding=3)]
        act_type = "snake" if use_snake else "elu"
        for i in range(self.depth - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    act_type=act_type,
                )
            )
        layers.extend(
            [
                get_activation(act_type, channels=c_mults[0] * channels),
                WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
                nn.Tanh() if final_tanh else nn.Identity(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def _dependency_interval(module, low, high):
    """Inclusive input support of an output interval; no waveform blending."""
    if isinstance(module, (nn.Sequential, OobleckDecoder, OobleckEncoder, DecoderBlock, EncoderBlock)):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in reversed(list(layers)):
            low, high = _dependency_interval(child, low, high)
        return low, high
    if isinstance(module, ResidualUnit):
        a, b = _dependency_interval(module.layers, low, high)
        return min(a, low), max(b, high)
    if isinstance(module, nn.ConvTranspose1d):
        s, p, d, k = module.stride[0], module.padding[0], module.dilation[0], module.kernel_size[0]
        return -(-(low + p - d * (k - 1)) // s), (high + p) // s
    if isinstance(module, nn.Conv1d):
        s, p, d, k = module.stride[0], module.padding[0], module.dilation[0], module.kernel_size[0]
        return low * s - p, high * s - p + d * (k - 1)
    if isinstance(module, (SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return low, high
    raise TypeError(f"No audited support rule for {type(module).__name__}")


def _output_length(module, length):
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in layers:
            length = _output_length(child, length)
        return length
    if isinstance(module, nn.ConvTranspose1d):
        return (
            (length - 1) * module.stride[0]
            - 2 * module.padding[0]
            + module.dilation[0] * (module.kernel_size[0] - 1)
            + module.output_padding[0]
            + 1
        )
    if isinstance(module, nn.Conv1d):
        return (length + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
    if isinstance(module, (ResidualUnit, SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return length
    raise TypeError(f"No audited length rule for {type(module).__name__}")


class YuE2VAE(nn.Module):
    """Attributes: ``source_dtype`` (dtype name of the loaded tensors before the FP32 cast, e.g. ``"float32"``) and
    ``fingerprint`` (identity of the weights; includes ``source_dtype``)."""

    source_dtype: str
    fingerprint: str

    def __init__(self, encoder_config: dict, decoder_config: dict, decoder_only: bool = False):
        super().__init__()
        self.decoder_only = bool(decoder_only)
        self.encoder_config = dict(encoder_config)
        self.decoder_config = dict(decoder_config)
        if not self.decoder_only:
            self.encoder = OobleckEncoder(**self.encoder_config)
        self.decoder = OobleckDecoder(**self.decoder_config)
        self.hop = math.prod(self.decoder_config["strides"])
        self.latent_dim = int(self.decoder_config["latent_dim"])
        self.source_dtype = "float32"
        self.fingerprint = ""
        self.eval().requires_grad_(False)

    def load_state_dict_any(self, sd: dict[str, torch.Tensor]) -> None:
        """Strict load accepting ``weight_g``/``weight_v`` or ``parametrizations.weight.original0/1`` keys."""
        renamed = {}
        for key, value in sd.items():
            key = key.replace(".parametrizations.weight.original0", ".weight_g")
            key = key.replace(".parametrizations.weight.original1", ".weight_v")
            if key in renamed:
                raise ValueError(f"duplicate YuE2 VAE tensor after key normalisation: {key}")
            renamed[key] = value
        bad = [k for k, v in renamed.items() if v.dtype != torch.float32]
        if bad:
            raise ValueError(f"YuE2 VAE tensors must be float32 (cast before loading): {bad[:4]}")
        self.load_state_dict(renamed, strict=True)

    def _check_fp32(self, module: nn.Module) -> torch.device:
        param = next(module.parameters())
        if param.dtype != torch.float32:
            raise ValueError("YuE2 VAE weights must remain float32")
        return param.device

    @torch.no_grad()
    def encode_mean(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio [B, 2, S]`` (``S = T * 1920``) -> posterior mean ``[B, 64, T]``, FP32."""
        if self.decoder_only:
            raise RuntimeError("YuE2 VAE encoder not loaded (decoder_only)")
        if audio.ndim != 3 or audio.shape[1] != 2 or audio.shape[-1] < self.hop or audio.shape[-1] % self.hop != 0:
            raise ValueError(f"expected audio [B, 2, T*{self.hop}], got {tuple(audio.shape)}")
        device = self._check_fp32(self.encoder)
        with torch.autocast(device_type=device.type, enabled=False):
            pre = self.encoder(audio.to(device=device, dtype=torch.float32))
        mean, _ = pre.chunk(2, dim=1)
        return mean

    @torch.no_grad()
    def encode_mean_chunked(self, audio: torch.Tensor, chunk_frames: int = 750, overlap_frames: int = 50) -> torch.Tensor:
        """Whole-record encode of ``audio [2, S]`` in overlapping chunks -> ``[T, 64]``; equals ``encode_mean`` when
        ``overlap_frames`` covers the encoder receptive field."""
        if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[-1] % self.hop != 0 or audio.shape[-1] == 0:
            raise ValueError(f"expected audio [2, T*{self.hop}], got {tuple(audio.shape)}")
        if chunk_frames < 1 or overlap_frames < 0:
            raise ValueError("chunk_frames must be >= 1 and overlap_frames >= 0")
        total = audio.shape[-1] // self.hop
        out = None
        for c in range(0, total, chunk_frames):
            lo, hi = max(0, c - overlap_frames), min(total, c + chunk_frames + overlap_frames)
            lat = self.encode_mean(audio[None, :, lo * self.hop : hi * self.hop])[0].T
            if out is None:
                out = torch.empty(total, lat.shape[1], dtype=torch.float32, device=lat.device)
            n = min(chunk_frames, total - c)
            out[c : c + n] = lat[c - lo : c - lo + n]
        return out

    def _latent(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1] != self.latent_dim or latent.shape[0] < 1 or latent.shape[-1] < 1:
            raise ValueError(f"expected nonempty latents [B, {self.latent_dim}, T], got {tuple(latent.shape)}")
        if not torch.isfinite(latent).all():
            raise ValueError("VAE latents contain non-finite values")
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``latent [B, 64, T]`` -> waveform ``[B, 2, 1920 * T - 64]``, FP32, autocast disabled."""
        latent = self._latent(latent)
        device = self._check_fp32(self.decoder)
        with torch.autocast(device_type=device.type, enabled=False):
            return self.decoder(latent.to(device=device, dtype=torch.float32))

    def natural_output_length(self, frames: int) -> int:
        if int(frames) < 1:
            raise ValueError("frames must be positive")
        return _output_length(self.decoder, int(frames))

    def required_halo(self, core_frames: int) -> int:
        low, high = _dependency_interval(self.decoder, 0, core_frames * self.hop - 1)
        return max(0, -low, high - core_frames + 1)

    @torch.no_grad()
    def decode_tiled(
        self,
        latent: torch.Tensor,
        core_frames: int = 1024,
        halo_frames: int = 16,
        output_device: Optional[torch.device | str] = "cpu",
    ) -> torch.Tensor:
        """Exact tiled decode (cores with ``halo_frames >= required_halo(core_frames)`` context), written to ``output_device``."""
        latent = self._latent(latent)
        if not isinstance(core_frames, int) or core_frames < 1:
            raise ValueError("core_frames must be a positive integer")
        required = self.required_halo(core_frames)
        if not isinstance(halo_frames, int) or halo_frames < required:
            raise ValueError(f"halo_frames must be at least {required} for this decoder")
        output_device = latent.device if output_device is None else output_device
        frames = latent.shape[-1]
        total = self.natural_output_length(frames)
        audio = torch.empty((latent.shape[0], self.decoder.out_channels, total), dtype=torch.float32, device=output_device)
        for start in range(0, frames, core_frames):
            end = min(frames, start + core_frames)
            left, right = max(0, start - halo_frames), min(frames, end + halo_frames)
            tile = self.decode(latent[..., left:right])
            out_start, out_end = start * self.hop, min(end * self.hop, total)
            crop_start = (start - left) * self.hop
            crop = tile[..., crop_start : crop_start + out_end - out_start]
            if crop.shape[-1] != out_end - out_start:
                raise RuntimeError("VAE tile did not cover its requested output core")
            audio[..., out_start:out_end].copy_(crop.to(output_device))
            del tile, crop
        return audio

    def encoder_receptive_frames(self) -> int:
        """Latent frames on each side that influence a latent frame (for choosing ``overlap_frames``)."""
        if self.decoder_only:
            raise RuntimeError("YuE2 VAE encoder not loaded (decoder_only)")
        low, high = _dependency_interval(self.encoder, 0, 0)
        return math.ceil(max(-low, high) / self.hop)
