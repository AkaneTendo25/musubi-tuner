"""MiniMax Music 3 diffusion transformer using the ComfyUI checkpoint layout.

The architecture follows the Apache-2.0 MiniMax Music 3 implementation published
for Diffusers. Module names follow the ComfyUI repack so its floating-point
safetensors can be loaded without conversion.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper


MAX_CONDITION_FRAMES = 200
CONDITION_HOP_FRAMES = 100


def latent_length(audio_frames: int) -> int:
    """Convert 25 Hz acoustic frames to 44.1 kHz Flow-VAE latent frames."""
    return max(1, int(audio_frames * 44100 / 24000 * 960 / 512))


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int = 1, out_features: int = 256):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features // 2, in_features))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = 2.0 * math.pi * value @ self.weight.to(value).T
        return torch.cat((features.cos(), features.sin()), dim=-1)


class ComfyLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.gamma.to(x), self.beta.to(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int = 32, theta: float = 10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.to(device=device, dtype=torch.float32))
        frequencies = frequencies.to(dtype)
        cos, sin = frequencies.cos(), frequencies.sin()
        matrix = torch.stack((cos, -sin, sin, cos), dim=-1)
        return matrix.reshape(1, 1, length, frequencies.shape[-1], 2, 2)


def _apply_rope(x: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Apply the interleaved-pair rotary layout used by the Comfy checkpoint."""
    original_dtype = x.dtype
    x = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(rotation_matrix.dtype)
    x = rotation_matrix[..., 0] * x[..., 0] + rotation_matrix[..., 1] * x[..., 1]
    return x.movedim(-1, -2).flatten(-2).to(original_dtype)


class Attention(nn.Module):
    def __init__(self, dim: int = 2048, head_dim: int = 64):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        rotary_dim = rope.shape[-3] * 2
        q = torch.cat((_apply_rope(q[..., :rotary_dim], rope), q[..., rotary_dim:]), dim=-1)
        k = torch.cat((_apply_rope(k[..., :rotary_dim], rope), k[..., rotary_dim:]), dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        return self.to_out(out.transpose(1, 2).reshape(batch, length, dim))


class GLU(nn.Module):
    def __init__(self, dim: int = 2048, inner_dim: int = 8192):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.silu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int = 2048, inner_dim: int = 8192):
        super().__init__()
        self.ff = nn.Sequential(GLU(dim, inner_dim), nn.Identity(), nn.Linear(inner_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 2048, head_dim: int = 64, inner_dim: int = 8192):
        super().__init__()
        self.pre_norm = ComfyLayerNorm(dim)
        self.self_attn = Attention(dim, head_dim)
        self.ff_norm = ComfyLayerNorm(dim)
        self.ff = FeedForward(dim, inner_dim)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.pre_norm(x), rope, attention_mask)
        return x + self.ff(self.ff_norm(x))


class ContinuousTransformer(nn.Module):
    def __init__(self, dim: int, in_dim: int, out_dim: int, layers: int, head_dim: int, ff_dim: int, rotary_dim: int):
        super().__init__()
        self.project_in = nn.Linear(in_dim, dim, bias=False)
        self.project_out = nn.Linear(dim, out_dim, bias=False)
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, head_dim, ff_dim) for _ in range(layers)])


class DiffusionTransformer(nn.Module):
    def __init__(self, channels: int, condition_dim: int, dim: int, layers: int, head_dim: int, ff_dim: int, rotary_dim: int):
        super().__init__()
        full_dim = channels * 2 + condition_dim
        self.transformer = ContinuousTransformer(dim, full_dim, channels, layers, head_dim, ff_dim, rotary_dim)
        self.timestep_features = FourierFeatures(1, 256)
        self.to_timestep_embed = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.to_delta_timestep_embed: Optional[nn.Module] = None
        self.flowmap_delta_type: Optional[str] = None
        self.register_buffer("flowmap_delta_gate", torch.tensor([0.25]), persistent=False)
        self.time_sign_embed: Optional[nn.Embedding] = None
        self.preprocess_conv = nn.Conv1d(full_dim, full_dim, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(channels, channels, 1, bias=False)

    def enable_flowmap_time_conditioning(self, gate: float = 0.25, delta_type: str = "r") -> None:
        if delta_type not in ("r", "t-r"):
            raise ValueError("Music 3 reference-time delta_type must be 'r' or 't-r'")
        if self.to_delta_timestep_embed is None:
            self.to_delta_timestep_embed = copy.deepcopy(self.to_timestep_embed)
            self.to_delta_timestep_embed.requires_grad_(True)
        self.flowmap_delta_type = delta_type
        self.flowmap_delta_gate.fill_(gate)

    def enable_time_sign_conditioning(self, dim: int) -> None:
        if self.time_sign_embed is None:
            reference = self.to_timestep_embed[0].weight
            self.time_sign_embed = nn.Embedding(2, dim, device=reference.device, dtype=reference.dtype)
            nn.init.zeros_(self.time_sign_embed.weight)


class MiniMaxMusic3DiT(nn.Module):
    """Flow-matching DiT compatible with Comfy-Org/MiniMax-Music-3 weights."""

    def __init__(
        self,
        channels: int = 128,
        condition_input_dim: int = 4096,
        condition_layers: int = 8,
        condition_dim: int = 2048,
        dim: int = 2048,
        layers: int = 36,
        head_dim: int = 64,
        ff_dim: int = 8192,
        rotary_dim: int = 32,
    ):
        super().__init__()
        self.latent_conditioners = nn.Sequential(nn.Conv1d(condition_input_dim, condition_dim, 3, padding=1))
        self.diffusion_transformer = DiffusionTransformer(channels, condition_dim, dim, layers, head_dim, ff_dim, rotary_dim)
        self.condition_input_dim = condition_input_dim
        self.condition_layers = condition_layers
        self.cond_layer_logits = nn.Parameter(torch.zeros(condition_layers))
        self.cond_layer_scale = nn.Parameter(torch.ones(1))
        self.gradient_checkpointing = False
        self.gradient_checkpointing_interval = 1
        self.gradient_checkpointing_segment_stride = 1
        self.blocks_to_swap = 0
        self.offloader = None

    def enable_flowmap_time_conditioning(self, gate: float = 0.25, delta_type: str = "r") -> None:
        self.diffusion_transformer.enable_flowmap_time_conditioning(gate, delta_type)

    def enable_time_sign_conditioning(self) -> None:
        self.diffusion_transformer.enable_time_sign_conditioning(
            self.diffusion_transformer.transformer.project_in.out_features
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(p for p in self.parameters() if p.is_floating_point()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_block_swap(self, num_blocks: int, config: BlockSwapConfig):
        blocks = self.diffusion_transformer.transformer.layers
        if num_blocks > len(blocks) - 2:
            raise ValueError(f"Cannot swap more than {len(blocks) - 2} blocks; requested {num_blocks}.")
        self.blocks_to_swap = num_blocks
        self.offloader = create_offloader("minimax_music3", blocks, len(blocks), num_blocks, config)

    def move_to_device_except_swap_blocks(self, device: torch.device):
        transformer = self.diffusion_transformer.transformer
        blocks = transformer.layers
        if self.blocks_to_swap:
            transformer.layers = nn.ModuleList()
        self.to(device)
        transformer.layers = blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(self.diffusion_transformer.transformer.layers)

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def aligned_condition(self, hidden: torch.Tensor) -> torch.Tensor:
        frames = hidden.shape[1]
        hidden = hidden.transpose(1, 2).reshape(hidden.shape[0], self.condition_layers, self.condition_input_dim, frames)
        weights = torch.softmax(self.cond_layer_logits.to(hidden), dim=0)
        hidden = torch.einsum("blht,l->bht", hidden, weights) * self.cond_layer_scale.to(hidden)
        condition = self.latent_conditioners(hidden)
        return F.interpolate(condition, size=latent_length(frames), mode="nearest")

    def _diffusion_forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        reference_timestep: Optional[torch.Tensor] = None,
        timestep_sign: Optional[torch.Tensor] = None,
        hidden_state_layer: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        module = self.diffusion_transformer
        full = torch.cat((x, torch.zeros_like(x), condition), dim=1)
        full = module.preprocess_conv(full) + full
        time_features = module.timestep_features(timestep.unsqueeze(-1)).to(x.dtype)
        time_embedding = module.to_timestep_embed(time_features)
        if reference_timestep is not None:
            if module.to_delta_timestep_embed is None or module.flowmap_delta_type is None:
                raise ValueError("Reference-time conditioning is not enabled")
            reference_timestep = reference_timestep.to(device=timestep.device, dtype=timestep.dtype)
            delta_timestep = reference_timestep if module.flowmap_delta_type == "r" else timestep - reference_timestep
            delta_features = module.timestep_features(delta_timestep.unsqueeze(-1)).to(x.dtype)
            delta_embedding = module.to_delta_timestep_embed(delta_features)
            gate = module.flowmap_delta_gate.to(device=x.device, dtype=time_embedding.dtype)
            time_embedding = (1 - gate) * time_embedding + gate * delta_embedding
        if timestep_sign is not None:
            if module.time_sign_embed is None:
                raise ValueError("Signed-time conditioning is not enabled")
            sign_index = (timestep_sign.to(device=timestep.device) < 0).long()
            time_embedding = time_embedding + module.time_sign_embed(sign_index).to(time_embedding)
        hidden = module.transformer.project_in(full.transpose(1, 2))
        if time_embedding.ndim == 2:
            time_token = time_embedding.unsqueeze(1)
        elif time_embedding.ndim == 3:
            if time_embedding.shape[:2] != hidden.shape[:2]:
                raise ValueError(
                    f"Tokenwise Music 3 times must match latent tokens {tuple(hidden.shape[:2])}, "
                    f"got {tuple(time_embedding.shape[:2])}"
                )
            hidden = hidden + time_embedding.to(hidden)
            time_token = time_embedding.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Music 3 time embedding must be 2D or 3D, got {tuple(time_embedding.shape)}")
        hidden = torch.cat((time_token, hidden), dim=1)
        attention_mask = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=hidden.device, dtype=torch.bool)
            if valid_mask.ndim == 3:
                valid_mask = valid_mask[:, 0]
            time_valid = torch.ones((valid_mask.shape[0], 1), device=hidden.device, dtype=torch.bool)
            key_mask = torch.cat((time_valid, valid_mask), dim=1)
            attention_mask = key_mask[:, None, None, :]
        rope = module.transformer.rotary_pos_emb(hidden.shape[1], hidden.device, hidden.dtype)
        blocks = module.transformer.layers
        captured_hidden = None
        for index, block in enumerate(blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            checkpoint_this_block = (
                self.gradient_checkpointing
                and self.training
                and index % self.gradient_checkpointing_segment_stride < self.gradient_checkpointing_interval
            )
            if checkpoint_this_block:
                forward = block
                if self.activation_cpu_offloading:
                    forward = create_cpu_offloading_wrapper(forward, hidden.device)
                hidden = checkpoint(forward, hidden, rope, attention_mask, use_reentrant=False)
            else:
                hidden = block(hidden, rope, attention_mask)
            if hidden_state_layer == index:
                captured_hidden = hidden[:, 1:]
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(blocks, index)
        out = module.transformer.project_out(hidden[:, 1:]).transpose(1, 2)
        prediction = module.postprocess_conv(out) + out
        if hidden_state_layer is not None:
            if captured_hidden is None:
                raise ValueError(f"Music 3 hidden-state layer {hidden_state_layer} is outside the transformer")
            return prediction, captured_hidden
        return prediction

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        conditioning_scale: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        reference_timestep: Optional[torch.Tensor] = None,
        timestep_sign: Optional[torch.Tensor] = None,
        hidden_state_layer: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        condition = self.aligned_condition(context)
        if conditioning_scale is not None:
            condition = condition * conditioning_scale[:, :1, :1]
        condition = F.pad(condition, (0, max(0, x.shape[-1] - condition.shape[-1])))[..., : x.shape[-1]]
        window, hop = latent_length(MAX_CONDITION_FRAMES), latent_length(CONDITION_HOP_FRAMES)
        if x.shape[-1] <= window:
            return self._diffusion_forward(
                x, timestep, condition, valid_mask, reference_timestep, timestep_sign, hidden_state_layer
            )
        output, count = torch.zeros_like(x), torch.zeros((1, 1, x.shape[-1]), device=x.device, dtype=x.dtype)
        hidden_output = None
        start = 0
        while start < x.shape[-1]:
            end = min(start + window, x.shape[-1])
            window_mask = None if valid_mask is None else valid_mask[..., start:end]
            window_timestep = timestep[..., start:end] if timestep.ndim == 2 else timestep
            window_reference = (
                reference_timestep[..., start:end]
                if reference_timestep is not None and reference_timestep.ndim == 2
                else reference_timestep
            )
            result = self._diffusion_forward(
                x[..., start:end],
                window_timestep,
                condition[..., start:end],
                window_mask,
                window_reference,
                timestep_sign,
                hidden_state_layer,
            )
            if hidden_state_layer is not None:
                prediction, hidden_piece = result
                if hidden_output is None:
                    hidden_output = hidden_piece.new_zeros(
                        (hidden_piece.shape[0], x.shape[-1], hidden_piece.shape[-1])
                    )
                hidden_output[:, start:end] += hidden_piece
            else:
                prediction = result
            output[..., start:end] += prediction
            count[..., start:end] += 1
            if end == x.shape[-1]:
                break
            start += hop
        prediction = output / count
        if hidden_state_layer is not None:
            return prediction, hidden_output / count.transpose(1, 2)
        return prediction
