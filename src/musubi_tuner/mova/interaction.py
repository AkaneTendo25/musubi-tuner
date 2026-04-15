from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from .wan_video_dit import AttentionModule, RMSNorm


class RotaryEmbedding(nn.Module):
    def __init__(self, base: float, dim: int, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(dtype=x.dtype), (emb.sin() * self.attention_scaling).to(
            dtype=x.dtype
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class PerFrameAttentionPooling(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.probe, std=0.02)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        bsz, _, dim = x.shape
        frames, height, width = grid_size
        spatial = height * width
        x = x.view(bsz, frames, spatial, dim).contiguous().view(bsz * frames, spatial, dim)
        probe = self.probe.expand(bsz * frames, -1, -1)
        pooled = self.attention(probe, x, x, need_weights=False)[0].squeeze(1)
        return self.layernorm(pooled.view(bsz, frames, dim))


class CrossModalInteractionController:
    def __init__(self, visual_layers: int = 30, audio_layers: int = 30):
        self.visual_layers = visual_layers
        self.audio_layers = audio_layers
        self.min_layers = min(visual_layers, audio_layers)

    def get_interaction_layers(self, strategy: str = "shallow_focus") -> Dict[str, List[Tuple[int, int]]]:
        if strategy == "shallow_focus":
            count = min(10, self.min_layers // 3)
            layers = list(range(count))
        elif strategy == "distributed":
            layers = list(range(0, self.min_layers, 3))
        elif strategy == "progressive":
            shallow = list(range(0, min(8, self.min_layers)))
            deep = list(range(8, self.min_layers, 3)) if self.min_layers > 8 else []
            layers = shallow + deep
        elif strategy == "custom":
            layers = [i for i in [0, 2, 4, 6, 8, 12, 16, 20] if i < self.min_layers]
        elif strategy == "full":
            layers = list(range(self.min_layers))
        else:
            raise ValueError(f"Unknown interaction strategy: {strategy}")
        return {"v2a": [(i, i) for i in layers], "a2v": [(i, i) for i in layers]}

    def should_interact(self, layer_idx: int, direction: str, interaction_mapping: Dict[str, List[Tuple[int, int]]]) -> bool:
        if direction not in interaction_mapping:
            return False
        return any(src == layer_idx for src, _ in interaction_mapping[direction])


class ConditionalCrossAttention(nn.Module):
    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(num_heads)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)

        if x_freqs is not None:
            q_view = rearrange(q, "b l (h d) -> b l h d", d=self.head_dim)
            q_view, _ = apply_rotary_pos_emb(q_view, q_view, x_freqs[0].to(q.device, q.dtype), x_freqs[1].to(q.device, q.dtype), unsqueeze_dim=2)
            q = rearrange(q_view, "b l h d -> b l (h d)")
        if y_freqs is not None:
            k_view = rearrange(k, "b l (h d) -> b l h d", d=self.head_dim)
            _, k_view = apply_rotary_pos_emb(k_view, k_view, y_freqs[0].to(k.device, k.dtype), y_freqs[1].to(k.device, k.dtype), unsqueeze_dim=2)
            k = rearrange(k_view, "b l h d -> b l (h d)")

        return self.o(self.attn(q, k, v))


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        output_dim = output_dim or embedding_dim * 2
        self.chunk_dim = chunk_dim
        self.emb = nn.Embedding(num_embeddings, embedding_dim) if num_embeddings is not None else None
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)
        temb = self.linear(self.silu(temb))
        if self.chunk_dim == 2:
            scale, shift = temb.chunk(2, dim=2)
        elif self.chunk_dim == 1:
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)
        return self.norm(x) * (1 + scale) + shift


class ConditionalCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6, pooled_adaln: bool = False):
        super().__init__()
        self.y_norm = nn.LayerNorm(kv_dim, eps=eps)
        self.inner = ConditionalCrossAttention(dim=dim, kv_dim=kv_dim, num_heads=num_heads, eps=eps)
        self.pooled_adaln = pooled_adaln
        if pooled_adaln:
            self.per_frame_pooling = PerFrameAttentionPooling(kv_dim, num_heads=num_heads, eps=eps)
            self.adaln = AdaLayerNorm(kv_dim, output_dim=dim * 2, chunk_dim=2)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if self.pooled_adaln:
            if video_grid_size is None:
                raise ValueError("video_grid_size is required when pooled_adaln=True")
            pooled = self.per_frame_pooling(y, video_grid_size)
            if pooled.shape[1] != x.shape[1]:
                pooled = F.interpolate(pooled.permute(0, 2, 1), size=x.shape[1], mode="linear", align_corners=False).permute(0, 2, 1)
            x = self.adaln(x, temb=pooled)
        return self.inner(x=x, y=self.y_norm(y), x_freqs=x_freqs, y_freqs=y_freqs)


class DualTowerConditionalBridge(ModelMixin, ConfigMixin):
    _repeated_blocks = ("ConditionalCrossAttentionBlock",)

    @register_to_config
    def __init__(
        self,
        visual_layers: int = 30,
        audio_layers: int = 30,
        visual_hidden_dim: int = 3072,
        audio_hidden_dim: int = 1536,
        audio_fps: float = 44100.0 / 2048.0,
        head_dim: int = 128,
        interaction_strategy: str = "shallow_focus",
        apply_cross_rope: bool = False,
        apply_first_frame_bias_in_rope: bool = False,
        trainable_condition_scale: bool = False,
        pooled_adaln: bool = False,
    ):
        super().__init__()
        self.visual_hidden_dim = visual_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.audio_fps = audio_fps
        self.head_dim = head_dim
        self.apply_cross_rope = apply_cross_rope
        self.apply_first_frame_bias_in_rope = apply_first_frame_bias_in_rope
        self.trainable_condition_scale = trainable_condition_scale
        self.pooled_adaln = pooled_adaln
        self.condition_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32)) if trainable_condition_scale else 1.0

        self.controller = CrossModalInteractionController(visual_layers, audio_layers)
        self.interaction_mapping = self.controller.get_interaction_layers(interaction_strategy)
        self.audio_to_video_conditioners = nn.ModuleDict()
        self.video_to_audio_conditioners = nn.ModuleDict()
        self.rotary = RotaryEmbedding(base=10000.0, dim=head_dim)

        for v_layer, _ in self.interaction_mapping["a2v"]:
            self.audio_to_video_conditioners[str(v_layer)] = ConditionalCrossAttentionBlock(
                dim=visual_hidden_dim,
                kv_dim=audio_hidden_dim,
                num_heads=visual_hidden_dim // head_dim,
                pooled_adaln=False,
            )
        for a_layer, _ in self.interaction_mapping["v2a"]:
            self.video_to_audio_conditioners[str(a_layer)] = ConditionalCrossAttentionBlock(
                dim=audio_hidden_dim,
                kv_dim=visual_hidden_dim,
                num_heads=audio_hidden_dim // head_dim,
                pooled_adaln=pooled_adaln,
            )

    def get_conditioner_layers(self, direction: str) -> list[int]:
        if direction == "a2v":
            layers = self.audio_to_video_conditioners.keys()
        elif direction == "v2a":
            layers = self.video_to_audio_conditioners.keys()
        else:
            raise ValueError(f"Invalid direction: {direction}")
        return [int(layer_idx) for layer_idx in layers]

    def get_conditioner_blocks(self, direction: str) -> list[ConditionalCrossAttentionBlock]:
        if direction == "a2v":
            return list(self.audio_to_video_conditioners.values())
        if direction == "v2a":
            return list(self.video_to_audio_conditioners.values())
        raise ValueError(f"Invalid direction: {direction}")

    def get_conditioner_block_index(self, layer_idx: int, direction: str) -> Optional[int]:
        try:
            return self.get_conditioner_layers(direction).index(int(layer_idx))
        except ValueError:
            return None

    @torch.no_grad()
    def build_aligned_freqs(
        self,
        video_fps: float,
        grid_size: Tuple[int, int, int],
        audio_steps: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        frames, height, width = grid_size
        visual_len = frames * height * width
        audio_len = int(audio_steps)
        device = device or next(self.parameters()).device
        dtype = dtype or torch.float32

        audio_pos = torch.arange(audio_len, device=device, dtype=torch.float32).unsqueeze(0)
        if self.apply_first_frame_bias_in_rope:
            effective_fps = float(video_fps) / 4.0
            starts = torch.zeros((frames,), device=device, dtype=torch.float32)
            if frames > 1:
                starts[1:] = (1.0 / float(video_fps)) + torch.arange(frames - 1, device=device, dtype=torch.float32) * (
                    1.0 / effective_fps
                )
            video_pos_per_frame = starts * float(self.audio_fps)
        else:
            scale = float(self.audio_fps) / float(video_fps / 4.0)
            video_pos_per_frame = torch.arange(frames, device=device, dtype=torch.float32) * scale

        video_pos = video_pos_per_frame.repeat_interleave(height * width).unsqueeze(0)
        dummy_v = torch.zeros((1, visual_len, self.head_dim), device=device, dtype=dtype)
        dummy_a = torch.zeros((1, audio_len, self.head_dim), device=device, dtype=dtype)
        return self.rotary(dummy_v, video_pos), self.rotary(dummy_a, audio_pos)

    def should_interact(self, layer_idx: int, direction: str) -> bool:
        return self.controller.should_interact(layer_idx, direction, self.interaction_mapping)

    def apply_conditional_control(
        self,
        layer_idx: int,
        direction: str,
        primary_hidden_states: torch.Tensor,
        condition_hidden_states: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        condition_scale: Optional[float] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if not self.should_interact(layer_idx, direction):
            return primary_hidden_states
        if direction == "a2v":
            conditioner = self.audio_to_video_conditioners[str(layer_idx)]
        elif direction == "v2a":
            conditioner = self.video_to_audio_conditioners[str(layer_idx)]
        else:
            raise ValueError(f"Invalid direction: {direction}")

        conditioned = conditioner(
            x=primary_hidden_states,
            y=condition_hidden_states,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            video_grid_size=video_grid_size,
        )
        scale = condition_scale if condition_scale is not None else self.condition_scale
        return primary_hidden_states + conditioned * scale

    def forward(
        self,
        layer_idx: int,
        visual_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        *,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        condition_scale: Optional[float] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual_source = visual_hidden_states
        audio_source = audio_hidden_states

        visual_hidden_states = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="a2v",
            primary_hidden_states=visual_source,
            condition_hidden_states=audio_source,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            condition_scale=a2v_condition_scale if a2v_condition_scale is not None else condition_scale,
            video_grid_size=video_grid_size,
        )
        audio_hidden_states = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="v2a",
            primary_hidden_states=audio_source,
            condition_hidden_states=visual_source,
            x_freqs=y_freqs,
            y_freqs=x_freqs,
            condition_scale=v2a_condition_scale if v2a_condition_scale is not None else condition_scale,
            video_grid_size=video_grid_size,
        )
        return visual_hidden_states, audio_hidden_states
