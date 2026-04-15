import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from .wan_video_dit import DiTBlock, MLP, sinusoidal_embedding_1d, precompute_freqs_cis


def legacy_precompute_freqs_cis_1d(
    dim: int,
    end: int = 16384,
    theta: float = 10000.0,
    base_tps: float = 4.0,
    target_tps: float = 44100 / 2048,
):
    scale = float(base_tps) / float(target_tps)
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta) ** 1
    positions = torch.arange(end, dtype=torch.float64, device=f_freqs_cis.device) * scale
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, dim - 2 * (dim // 3), 2, dtype=torch.float64, device=f_freqs_cis.device)[
                : ((dim - 2 * (dim // 3)) // 2)
            ]
            / max(dim - 2 * (dim // 3), 1)
        )
    )
    f_freqs_cis = torch.polar(torch.ones((end, freqs.numel()), dtype=torch.float64), torch.outer(positions, freqs))
    no_freqs_cis = torch.ones_like(precompute_freqs_cis(dim // 3, end, theta))
    return f_freqs_cis, no_freqs_cis, no_freqs_cis


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    return precompute_freqs_cis(dim, end, theta).chunk(3, dim=-1)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, ...], eps: float):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(self.patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if t_mod.ndim == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            scale = scale.squeeze(2)
            shift = shift.squeeze(2)
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class WanAudioModel(ModelMixin, ConfigMixin):
    _repeated_blocks = ("DiTBlock",)

    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, ...],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        vae_type: Literal["oobleck", "dac"] = "oobleck",
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(patch_size)
        self.has_image_input = has_image_input
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.vae_type = vae_type

        self.patch_embedding = nn.Conv1d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, self.patch_size, eps)

        head_dim = dim // num_heads
        if vae_type == "dac":
            self.freqs = precompute_freqs_cis_1d(head_dim)
        elif vae_type == "oobleck":
            self.freqs = legacy_precompute_freqs_cis_1d(head_dim)
        else:
            raise ValueError(f"Unsupported vae_type: {vae_type}")

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            control = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, control)]
            x = x[0].unsqueeze(0)
        grid_size = tuple(int(v) for v in x.shape[2:])
        x = rearrange(x, "b c f -> b f c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, ...]) -> torch.Tensor:
        return rearrange(x, "b f (p c) -> b c (f p)", f=grid_size[0], p=self.patch_size[0])

    def build_freqs(self, grid_size: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        f = grid_size[0]
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, -1).expand(f, -1),
                self.freqs[1][:f].view(f, -1).expand(f, -1),
                self.freqs[2][:f].view(f, -1).expand(f, -1),
            ],
            dim=-1,
        )
        return freqs.reshape(f, 1, -1).to(device=device)

