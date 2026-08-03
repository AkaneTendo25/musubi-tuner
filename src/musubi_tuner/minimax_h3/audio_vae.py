# Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted from Hugging Face Diffusers PR #14355 at revision
# abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _wn_conv1d(*args, **kwargs) -> nn.Module:
    # The published Comfy checkpoint has weight normalization removed and stores
    # each effective convolution kernel directly as ``weight``.
    return nn.Conv1d(*args, **kwargs)


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    r"""Kaiser-windowed sinc low-pass filter of shape `[1, 1, kernel_size]`.

    Kept arithmetically identical to the `alias-free-torch` implementation the checkpoint was trained
    with, because the resulting tensor is stored as a persistent buffer.
    """
    half_size = kernel_size // 2

    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    # Normalize to sum 1 so a constant input does not leak through the resampler.
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class MiniMaxH3AudioSnake1d(nn.Module):
    r"""`x + (alpha + 1e-9)^-1 * sin(alpha * x)^2` over `[batch_size, channels, length]`, with a
    per-channel learnable `alpha` of shape `[1, channels, 1]`. Used throughout the DAC encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)


class MiniMaxH3AudioSnakeBeta(nn.Module):
    r"""`x + (exp(beta) + 1e-9)^-1 * sin(exp(alpha) * x)^2` over `[batch_size, channels, length]`.

    The BigVGAN decoder's activation: separate frequency (`alpha`) and magnitude (`beta`) parameters,
    both stored in log space as `[channels]` vectors.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)


class MiniMaxH3AudioLowPassFilter1d(nn.Module):
    r"""Depthwise Kaiser-sinc low-pass filter with a stride, i.e. the anti-aliased downsampler."""

    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(hidden_states, self.filter.expand(num_channels, -1, -1), stride=self.stride, groups=num_channels)


class MiniMaxH3AudioUpSample1d(nn.Module):
    r"""Anti-aliased `ratio`x upsampler (transposed depthwise Kaiser-sinc convolution)."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, self.filter.expand(num_channels, -1, -1), stride=self.stride, groups=num_channels
        )
        return hidden_states[..., self.pad_left : -self.pad_right]


class MiniMaxH3AudioDownSample1d(nn.Module):
    r"""Anti-aliased `ratio`x downsampler."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = MiniMaxH3AudioLowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=kernel_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lowpass(hidden_states)


class MiniMaxH3AudioActivation1d(nn.Module):
    r"""Upsample -> activation -> downsample: the alias-free activation wrapper used by BigVGAN."""

    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.downsample(hidden_states)


class MiniMaxH3AudioResidualUnit(nn.Module):
    r"""DAC residual unit: `Snake -> dilated Conv1d(k=7) -> Snake -> Conv1d(k=1)`, plus a shortcut
    that is center-cropped when the dilated convolution shrinks the time axis."""

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioSnake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=((7 - 1) * dilation) // 2),
            MiniMaxH3AudioSnake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.block(hidden_states)
        pad = (hidden_states.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            hidden_states = hidden_states[..., pad:-pad]
        return hidden_states + residual


class MiniMaxH3AudioEncoderBlock(nn.Module):
    r"""Three residual units at dilations 1/3/9, then a strided channel-doubling convolution."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=1),
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=3),
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=9),
            MiniMaxH3AudioSnake1d(dim // 2),
            _wn_conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class MiniMaxH3AudioEncoder(nn.Module):
    r"""DAC waveform encoder: `[batch_size, 1, samples] -> [batch_size, latent_dim, samples / 800]`."""

    def __init__(self, d_model: int, strides: tuple[int, ...], d_latent: int):
        super().__init__()
        block: list[nn.Module] = [_wn_conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(MiniMaxH3AudioEncoderBlock(d_model, stride=stride))
        block += [
            MiniMaxH3AudioSnake1d(d_model),
            _wn_conv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class MiniMaxH3AudioGeGluMlp(nn.Module):
    r"""Pre-norm GeGLU MLP used inside the attention projection block."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act(self.w0(hidden_states)) * self.w1(hidden_states)
        return self.w2(hidden_states)


class MiniMaxH3AudioAttnProcessor:
    r"""Processor of [`MiniMaxH3AudioCausalAttention`].

    The causal mask is expressed as `is_causal=True` rather than as a materialized mask. Every
    attention backend honours that flag, with two exceptions: `_native_npu`, whose kernel takes no
    causal argument and would compute *bidirectional* attention, and context parallelism, which
    raises for causal attention.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: MiniMaxH3AudioCausalAttention, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = F.linear(
            input=hidden_states,
            weight=attn.qkv.weight,
            bias=torch.cat((attn.q_bias, attn.zero_k_bias, attn.v_bias)),
        )
        query, key, value = qkv.reshape(batch_size, seq_len, 3, attn.num_heads, attn.head_dim).permute(2, 0, 1, 3, 4).unbind(0)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)
        # The heads are mean-pooled away instead of being concatenated, and the head dimension that
        # remains is adaptively average-pooled down to `out_dim`.
        hidden_states = torch.mean(hidden_states, dim=2)
        hidden_states = F.adaptive_avg_pool1d(hidden_states, attn.out_dim)
        return attn.proj(hidden_states)


class MiniMaxH3AudioCausalAttention(nn.Module):
    r"""Causal self-attention that narrows the feature width from `in_dim` to `out_dim`.

    QKV is a single bias-less `nn.Linear`; query and value biases are separate parameters and the key
    bias is a frozen zero buffer (`zero_k_bias`), exactly as stored in the checkpoint. Heads are
    `in_dim // num_heads` wide; instead of being concatenated they are **mean-pooled away**, and the
    remaining head dimension is adaptively average-pooled down to `out_dim`.
    """

    # The checkpoint stores one fused `qkv` projection, so there is nothing to fuse.
    _supports_qkv_fusion = False

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

        self.processor = MiniMaxH3AudioAttnProcessor()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class MiniMaxH3AudioAttnProjection(nn.Module):
    r"""`pre_block`: residual causal-attention + GeGLU block that rewires `latent_dim` -> `latent_channels`."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MiniMaxH3AudioCausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = MiniMaxH3AudioGeGluMlp(in_features=out_dim, hidden_features=out_dim * mlp_ratio)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(self.norm3(hidden_states)) + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3AudioEncoderModel(nn.Module):
    """Encoder-only FP32 MiniMax H3 audio VAE used for deterministic caches."""

    hop_length = 800

    def __init__(self, latents_mean: tuple[float, ...], latents_std: tuple[float, ...]) -> None:
        super().__init__()
        if len(latents_mean) != 32 or len(latents_std) != 32:
            raise ValueError("MiniMax H3 audio latent statistics must contain 32 channels")
        self.encoder = MiniMaxH3AudioEncoder(d_model=64, strides=(2, 4, 4, 5, 5), d_latent=2048)
        self.pre_block = MiniMaxH3AudioAttnProjection(2048, 32, num_heads=8)
        self.mean_proj = nn.Conv1d(32, 32, 1)
        self.logs_proj = nn.Conv1d(32, 32, 1)
        self.register_buffer("latents_mean", torch.tensor(latents_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("latents_std", torch.tensor(latents_std, dtype=torch.float32), persistent=False)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"audio waveform must have shape [B, 1, samples], got {tuple(waveform.shape)}")
        if waveform.dtype is not torch.float32:
            raise ValueError("MiniMax H3 audio VAE input must be float32")
        right_pad = math.ceil(waveform.shape[-1] / self.hop_length) * self.hop_length - waveform.shape[-1]
        if right_pad:
            waveform = F.pad(waveform, (0, right_pad))
        hidden_states = self.encoder(waveform)
        hidden_states = self.pre_block(hidden_states.transpose(1, 2)).transpose(1, 2)
        latent = self.mean_proj(hidden_states)
        mean = self.latents_mean.view(1, -1, 1)
        std = self.latents_std.view(1, -1, 1)
        return (latent - mean) / std
