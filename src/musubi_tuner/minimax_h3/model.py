# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native MiniMax H3 transformer.

The implementation follows the released Diffusers H3 port at revision
``abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`` while retaining the original
checkpoint module names and fused projections. This keeps H3 independent of
Diffusers model APIs and lets Musubi load the Comfy BF16 repack directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from diffusers.models.embeddings import get_timestep_embedding
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader

MINIMAX_H3_MODALITY_COUNT = 3


class MiniMaxH3TokenTag(IntEnum):
    VIDEO = 0
    TEXT = 1
    AUDIO = 2


@dataclass(frozen=True)
class MiniMaxH3TransformerConfig:
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    adaln_t_table_size: int | None = None
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, value: dict) -> MiniMaxH3TransformerConfig:
        fields = cls.__dataclass_fields__
        kwargs = {key: value[key] for key in fields if key in value}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return cls(**kwargs)


@dataclass
class MiniMaxH3TransformerOutput:
    video: torch.Tensor
    audio: torch.Tensor


def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1).contiguous()


class MiniMaxH3RotaryPosEmbed(nn.Module):
    def __init__(self, rope_freq_dim: int, rope_theta: float):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)))
        # The released checkpoint contains this value even though it is fully
        # determined by the config. Keeping it persistent gives strict 535-key loading.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        time, height, width = frequencies.unbind(dim=1)
        frequencies = torch.cat((time, height, width), dim=-1)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies.cos(), frequencies.sin()


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(self, freq_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = get_timestep_embedding(
            timestep,
            self.freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        embedding = embedding.to(self.proj_in.weight.dtype)
        return self.proj_out(F.silu(self.proj_in(embedding)))


class MiniMaxH3AdaLNProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, apply_silu: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.apply_silu = apply_silu

    def forward(self, timestep_embedding: torch.Tensor) -> torch.Tensor:
        # Scaled FP8 stores the frozen weight in E4M3 but dequantizes it to the
        # scale buffer's dtype for the matmul. Casting the activation to the
        # storage dtype would bypass that compute contract and make F.linear see
        # mismatched FP8/BF16 operands.
        compute_dtype = getattr(self.linear, "scale_weight", self.linear.weight).dtype
        activated = F.silu(timestep_embedding) if self.apply_silu else timestep_embedding
        activated = activated.to(compute_dtype)
        return self.linear(activated)


class MiniMaxH3Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, head_dim: int, qk_norm_eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.qkv_proj = nn.Linear(hidden_size, 3 * self.inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        query = self.q_norm(query.unflatten(-1, (self.heads, self.head_dim)))
        key = self.k_norm(key.unflatten(-1, (self.heads, self.head_dim)))
        value = value.unflatten(-1, (self.heads, self.head_dim))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask[None, None, :, :]
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        return self.out_proj(hidden_states)


class MiniMaxH3FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 2 * ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value)


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MiniMaxH3Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MiniMaxH3FeedForward(config.hidden_size, config.ffn_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig):
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxH3TokenRefinerBlock(config) for _ in range(config.num_refiner_layers)])
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)

    def forward(self, hidden_states: torch.Tensor, gradient_checkpointing: bool = False) -> torch.Tensor:
        for block in self.blocks:
            if torch.is_grad_enabled() and gradient_checkpointing:
                hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MiniMaxH3Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MiniMaxH3FeedForward(config.hidden_size, config.ffn_dim)
        self.adaln_proj = MiniMaxH3AdaLNProjection(
            config.time_embed_dim,
            6 * config.hidden_size * MINIMAX_H3_MODALITY_COUNT,
            apply_silu=config.adaln_t_table_size is None,
        )
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        modulation = self.adaln_proj(timestep_embedding).view(-1, 6 * self.hidden_size)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        hidden_dtype = hidden_states.dtype
        shift_attn = shift_attn.to(hidden_dtype)
        scale_attn = scale_attn.to(hidden_dtype)
        gate_attn = gate_attn.to(hidden_dtype)
        shift_mlp = shift_mlp.to(hidden_dtype)
        scale_mlp = scale_mlp.to(hidden_dtype)
        gate_mlp = gate_mlp.to(hidden_dtype)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_attn.index_select(0, adaln_indices))
        norm_hidden_states = norm_hidden_states + shift_attn.index_select(0, adaln_indices)
        attention = self.attn(norm_hidden_states, rotary_emb, attention_mask)
        hidden_states = hidden_states + gate_attn.index_select(0, adaln_indices) * attention

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_mlp.index_select(0, adaln_indices))
        norm_hidden_states = norm_hidden_states + shift_mlp.index_select(0, adaln_indices)
        feed_forward = self.mlp(norm_hidden_states)
        return hidden_states + gate_mlp.index_select(0, adaln_indices) * feed_forward


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig):
        super().__init__()
        video_patch_dim = config.in_channels * torch.Size(config.patch_size).numel()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdaLNProjection(
            config.time_embed_dim,
            2 * config.hidden_size,
            apply_silu=config.adaln_t_table_size is None,
        )
        self.video_out = nn.Linear(config.hidden_size, video_patch_dim)
        self.audio_out = nn.Linear(config.hidden_size, config.audio_in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        timestep_indices: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
    ) -> MiniMaxH3TransformerOutput:
        shift, scale = self.adaln_proj(timestep_embedding).chunk(2, dim=-1)
        shift = shift.to(hidden_states.dtype)
        scale = scale.to(hidden_states.dtype)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1.0 + scale.index_select(0, timestep_indices))
        hidden_states = hidden_states + shift.index_select(0, timestep_indices)
        video_dtype = self.video_out.weight.dtype
        audio_dtype = self.audio_out.weight.dtype
        video = self.video_out(hidden_states.to(video_dtype)).index_select(1, video_indices)
        audio = self.audio_out(hidden_states.to(audio_dtype)).index_select(1, audio_indices)
        return MiniMaxH3TransformerOutput(video=video, audio=audio)


class MiniMaxH3Transformer(nn.Module):
    """Single-stream MiniMax H3 video-audio transformer in the released key layout."""

    def __init__(self, config: MiniMaxH3TransformerConfig | None = None):
        super().__init__()
        self.config = config or MiniMaxH3TransformerConfig()
        config = self.config
        rotary_dim = 2 * 3 * config.rope_freq_dim
        if rotary_dim > config.attention_head_dim:
            raise ValueError(f"H3 rotary width {rotary_dim} exceeds attention head width {config.attention_head_dim}")

        video_patch_dim = config.in_channels * torch.Size(config.patch_size).numel()
        self.video_patch_proj = nn.Linear(video_patch_dim, config.hidden_size)
        self.audio_patch_proj = nn.Linear(config.audio_in_channels, config.hidden_size)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size)
        if config.adaln_t_table_size is None:
            self.time_embedder = MiniMaxH3TimeEmbedder(
                config.freq_dim,
                config.time_embed_hidden_dim,
                config.time_embed_dim,
            )
        else:
            if config.adaln_t_table_size < 2:
                raise ValueError("H3 AdaLN timestep table must contain at least two rows")
            self.time_embedder = None
            self.register_buffer(
                "adaln_t_table",
                torch.zeros(config.adaln_t_table_size, config.time_embed_dim),
                persistent=True,
            )
        self.rope = MiniMaxH3RotaryPosEmbed(config.rope_freq_dim, config.rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(config)
        self.blocks = nn.ModuleList([MiniMaxH3TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(config)
        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        if activation_cpu_offloading:
            raise NotImplementedError("MiniMax H3 does not support activation CPU offloading")
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def enable_block_swap(self, blocks_to_swap: int, config: BlockSwapConfig) -> None:
        num_blocks = len(self.blocks)
        if blocks_to_swap <= 0:
            raise ValueError("MiniMax H3 blocks_to_swap must be positive")
        # Whole-block rings retain two blocks to preserve overlap headroom.
        # Layer streaming owns an independent, smaller ring and can therefore
        # offload the Linear weights from every transformer block.
        layer_streaming = config.h2d_only and config.granularity == "layer"
        max_blocks_to_swap = num_blocks if layer_streaming else num_blocks - 2
        if blocks_to_swap > max_blocks_to_swap:
            raise ValueError(
                f"MiniMax H3 cannot swap more than {max_blocks_to_swap} of {num_blocks} blocks; requested {blocks_to_swap}"
            )
        self.blocks_to_swap = blocks_to_swap
        self.offloader = create_offloader(
            "minimax-h3-block",
            self.blocks,
            num_blocks,
            blocks_to_swap,
            config,
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        if self.blocks_to_swap:
            saved_blocks = self.blocks
            self.blocks = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.blocks = saved_blocks

    def offload_block_swap_to_cpu(self) -> None:
        """Evacuate the transformer, including the active swap ring, between sequential inference stages."""
        if not self.blocks_to_swap:
            self.to("cpu")
            return
        if self.offloader is None:
            raise RuntimeError("MiniMax H3 block swap is enabled without an offloader")
        self.offloader.offload_to_cpu(self.blocks)
        self.move_to_device_except_swap_blocks(torch.device("cpu"))

    def prepare_block_swap_before_forward(self) -> None:
        if not self.blocks_to_swap:
            return
        if self.offloader is None:
            raise RuntimeError("MiniMax H3 block swap is enabled without an offloader")
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def switch_block_swap_for_inference(self) -> None:
        if self.blocks_to_swap:
            if self.offloader is None:
                raise RuntimeError("MiniMax H3 block swap is enabled without an offloader")
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if self.blocks_to_swap:
            if self.offloader is None:
                raise RuntimeError("MiniMax H3 block swap is enabled without an offloader")
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def _checkpointed_block(
        self,
        block: MiniMaxH3TransformerBlock,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        def forward(value: torch.Tensor) -> torch.Tensor:
            return block(value, timestep_embedding, adaln_indices, rotary_emb, attention_mask)

        return checkpoint(forward, hidden_states, use_reentrant=False)

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        if self.time_embedder is not None:
            return self.time_embedder(timestep)
        table = self.adaln_t_table.float()
        position = timestep.to(device=table.device, dtype=torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long()
        upper = (lower + 1).clamp(max=table.shape[0] - 1)
        fraction = (position - lower.float()).unsqueeze(1)
        embedding = table.index_select(0, lower) * (1.0 - fraction) + table.index_select(0, upper) * fraction
        return embedding.to(timestep.device)

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> MiniMaxH3TransformerOutput:
        sequence_length = position_ids.shape[0]
        if position_ids.shape != (sequence_length, 3):
            raise ValueError(f"position_ids must have shape (sequence_length, 3), got {tuple(position_ids.shape)}")
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError("token_tags and timestep_indices must match the packed sequence length")

        rotary_emb = self.rope(position_ids)
        video = self.video_patch_proj(video_hidden_states.to(self.video_patch_proj.weight.dtype))
        audio = self.audio_patch_proj(audio_hidden_states.to(self.audio_patch_proj.weight.dtype))
        text = self.condition_proj(encoder_hidden_states.to(self.condition_proj.weight.dtype))
        text = self.token_refiner(text, self.gradient_checkpointing)

        hidden_states = text.new_zeros((text.shape[0], sequence_length, text.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text)
        hidden_states = hidden_states.index_copy(1, video_indices, video.to(text.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio.to(text.dtype))

        timestep_embedding = self._time_embedding(timestep)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_COUNT + token_tags.clamp(min=0)
        is_padding = token_tags < 0
        attention_mask = None
        if bool(is_padding.any()):
            attention_mask = is_padding[None, :] == is_padding[:, None]

        for block_index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_index)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._checkpointed_block(
                    block,
                    hidden_states,
                    timestep_embedding,
                    adaln_indices,
                    rotary_emb,
                    attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    timestep_embedding,
                    adaln_indices,
                    rotary_emb,
                    attention_mask,
                )
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, block_index)

        return self.final_layer(hidden_states, timestep_embedding, timestep_indices, video_indices, audio_indices)
