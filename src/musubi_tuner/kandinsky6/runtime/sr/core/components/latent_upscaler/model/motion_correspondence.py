# Motion-correspondence attention for latent video features.

from __future__ import annotations

import importlib
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import functional
from einops import rearrange
from pydantic import BaseModel
from torch import Tensor, nn

from musubi_tuner.kandinsky6.runtime.sr.core.components.model.utils import get_freqs
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.attention_merge import merge_attention_branches

if TYPE_CHECKING:
    from types import ModuleType

MotionAttentionBackend = Literal["sdpa", "natten"]
NattenBackend = Literal["cutlass-fna", "hopper-fna", "flex-fna"]


class MotionCorrespondenceSpec(BaseModel):
    """Construction parameters for one motion-correspondence block."""

    channels: int
    num_heads: int
    head_dim: int
    spatial_kernel_size: int
    temporal_offsets: tuple[int, ...]
    backend: MotionAttentionBackend
    natten_backend: NattenBackend = "hopper-fna"
    merge_compile: bool = False


def temporal_neighbor_indices(
    query_index: int,
    num_frames: int,
    temporal_offsets: tuple[int, ...],
) -> tuple[int, ...]:
    """Return valid neighbor-frame indices without shifting or duplication.

    Args:
        query_index: Query frame index.
        num_frames: Number of frames in the clip.
        temporal_offsets: Allowed non-zero offsets from the query frame.

    Returns:
        Valid neighbor indices in the configured offset order.
    """
    return tuple(query_index + offset for offset in temporal_offsets if 0 <= query_index + offset < num_frames)


def spatial_neighborhood_mask(
    height: int,
    width: int,
    kernel_size: int,
    device: torch.device,
) -> Tensor:
    """Build the shifted sliding-window mask used by neighborhood attention.

    Args:
        height: Feature-map height.
        width: Feature-map width.
        kernel_size: Odd spatial neighborhood size.
        device: Device for the returned mask.

    Returns:
        Boolean mask shaped ``(height * width, height * width)``.
    """
    if kernel_size > min(height, width):
        msg = f"spatial kernel {kernel_size} exceeds feature map {height}x{width}"
        raise ValueError(msg)

    query_h, query_w = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    key_h, key_w = query_h.flatten(), query_w.flatten()
    radius = kernel_size // 2
    start_h = (query_h.flatten() - radius).clamp(min=0, max=height - kernel_size)
    start_w = (query_w.flatten() - radius).clamp(min=0, max=width - kernel_size)
    inside_h = (key_h[None, :] >= start_h[:, None]) & (key_h[None, :] < start_h[:, None] + kernel_size)
    inside_w = (key_w[None, :] >= start_w[:, None]) & (key_w[None, :] < start_w[:, None] + kernel_size)
    return inside_h & inside_w


def split_rope_dims(head_dim: int) -> tuple[int, int, int]:
    """Split a head dimension into temporal, height, and width chunks.

    Leftover rotary pairs go to the spatial axes: correspondence is a spatial
    matching task, while temporal positions only span the short offset window.
    """
    if head_dim < 6 or head_dim % 2 != 0:
        msg = f"3D RoPE requires an even head_dim >= 6, got {head_dim}"
        raise ValueError(msg)
    pairs, remainder = divmod(head_dim // 2, 3)
    pair_counts = tuple(pairs + int(axis >= 3 - remainder) for axis in range(3))
    return 2 * pair_counts[0], 2 * pair_counts[1], 2 * pair_counts[2]


class AxialRoPE3D(nn.Module):
    """Axial rotary embeddings for ``(time, height, width)`` coordinates."""

    def __init__(self, head_dim: int, max_period: float = 10000.0) -> None:
        """Initialize per-axis rotary frequencies."""
        super().__init__()
        self.axis_dims = split_rope_dims(head_dim)
        for axis_name, axis_dim in zip(("time", "height", "width"), self.axis_dims, strict=True):
            self.register_buffer(
                f"{axis_name}_frequencies",
                get_freqs(axis_dim // 2, max_period),
                persistent=False,
            )

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """Apply rotary embeddings and return the original working dtype."""
        working_dtype = x.dtype
        chunks = x.to(torch.float32).split(self.axis_dims, dim=-1)
        axes = (1, 2, 3)
        frequencies = (self.time_frequencies, self.height_frequencies, self.width_frequencies)
        rotated = [
            self.rotate_axis(chunk, axis, frequency) for chunk, axis, frequency in zip(chunks, axes, frequencies, strict=True)
        ]
        return torch.cat(rotated, dim=-1).to(working_dtype)

    @staticmethod
    def rotate_axis(x: Tensor, axis: int, frequencies: Tensor) -> Tensor:
        """Rotate one channel chunk according to positions along one axis."""
        positions = torch.arange(x.shape[axis], device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, frequencies.to(device=x.device))
        broadcast_shape = [1] * x.ndim
        broadcast_shape[axis] = positions.numel()
        broadcast_shape[-1] = frequencies.numel()
        cosine = angles.cos().reshape(broadcast_shape)
        sine = angles.sin().reshape(broadcast_shape)
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        first = pairs[..., 0] * cosine - pairs[..., 1] * sine
        second = pairs[..., 0] * sine + pairs[..., 1] * cosine
        return torch.stack((first, second), dim=-1).flatten(-2)


class MotionCorrespondenceBlock(nn.Module):
    """Cross-frame spatial correspondence with an explicit no-match token."""

    def __init__(self, spec: MotionCorrespondenceSpec) -> None:
        """Initialize a motion-correspondence residual block."""
        super().__init__()
        if spec.channels != spec.num_heads * spec.head_dim:
            msg = f"channels must equal num_heads * head_dim, got {spec.channels} != {spec.num_heads} * {spec.head_dim}"
            raise ValueError(msg)
        if spec.spatial_kernel_size < 3 or spec.spatial_kernel_size % 2 == 0:
            msg = "spatial_kernel_size must be odd and at least 3"
            raise ValueError(msg)
        if (
            not spec.temporal_offsets
            or 0 in spec.temporal_offsets
            or tuple(sorted(set(spec.temporal_offsets))) != spec.temporal_offsets
            or set(spec.temporal_offsets) != {-offset for offset in spec.temporal_offsets}
        ):
            msg = "temporal_offsets must be non-zero, unique, sorted, and symmetric"
            raise ValueError(msg)
        self.channels = spec.channels
        self.num_heads = spec.num_heads
        self.head_dim = spec.head_dim
        self.spatial_kernel_size = spec.spatial_kernel_size
        self.temporal_offsets = spec.temporal_offsets
        self.backend = spec.backend
        self.natten_backend = spec.natten_backend
        self.merge_compile = spec.merge_compile

        self.norm = nn.RMSNorm(spec.channels)
        self.to_qkv = nn.Linear(spec.channels, 3 * spec.channels)
        self.q_norm = nn.RMSNorm(spec.head_dim)
        self.k_norm = nn.RMSNorm(spec.head_dim)
        self.rope = AxialRoPE3D(spec.head_dim)
        self.null_key = nn.Parameter(torch.randn(spec.num_heads, spec.head_dim) * 0.02)
        self.null_value = nn.Parameter(torch.zeros(spec.num_heads, spec.head_dim))
        self.to_out = nn.Linear(spec.channels, spec.channels)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply cross-frame correspondence to ``(B,C,T,H,W)`` features."""
        if x.shape[2] == 1:
            return x
        features = rearrange(x, "b c t h w -> b t h w c")
        # fp32 norm inputs keep fused rms_norm kernels under bf16 autocast (same as DiT QK-norm).
        qkv = self.to_qkv(self.norm(features.float()).type_as(features))
        q, k, v = [rearrange(part, "b t h w (nh d) -> b t h w nh d", nh=self.num_heads) for part in qkv.chunk(3, dim=-1)]
        q_unrotated = self.q_norm(q.float()).type_as(q)
        q = self.rope(q_unrotated)
        k = self.rope(self.k_norm(k.float()).type_as(k))
        attention = self.compute_sdpa_attention if self.backend == "sdpa" else self.compute_natten_attention
        context = attention(q, k, v, q_unrotated)
        output = self.to_out(rearrange(context, "b t h w nh d -> b t h w (nh d)"))
        return x + rearrange(output, "b t h w c -> b c t h w")

    def null_match_logits(self, q_unrotated: Tensor) -> Tensor:
        """Score queries against the null token in the shared attention scale.

        Uses pre-RoPE queries: the null token has no position, so pairing it with
        a rotated query would make the no-match threshold depend on where the
        query sits in the clip.
        """
        logits = torch.einsum("...nd,nd->...n", q_unrotated.float(), self.null_key.float())
        return logits * self.head_dim**-0.5

    def compute_sdpa_attention(self, q: Tensor, k: Tensor, v: Tensor, q_unrotated: Tensor) -> Tensor:
        """Compute the reference operation with PyTorch SDPA."""
        batch, frames, height, width, _heads, dim = q.shape
        spatial_mask = spatial_neighborhood_mask(height, width, self.spatial_kernel_size, q.device)
        null_logits = self.null_match_logits(q_unrotated)
        frame_outputs: list[Tensor] = []
        for query_index in range(frames):
            neighbor_indices = temporal_neighbor_indices(query_index, frames, self.temporal_offsets)
            q_frame = rearrange(q[:, query_index], "b h w nh d -> b nh (h w) d")
            k_frames = rearrange(k[:, neighbor_indices], "b t h w nh d -> b nh (t h w) d")
            v_frames = rearrange(v[:, neighbor_indices], "b t h w nh d -> b nh (t h w) d")
            # The null column carries a zero key, so its logit comes solely from the
            # additive bias; SDPA applies the bias after q·k scaling, hence pre-scaled logits.
            null_key = torch.zeros(batch, self.num_heads, 1, dim, dtype=q.dtype, device=q.device)
            null_value = self.null_value.to(q)[None, :, None, :].expand(batch, -1, 1, -1)
            keys = torch.cat((k_frames, null_key), dim=2)
            values = torch.cat((v_frames, null_value), dim=2)
            bias = self.sdpa_attention_bias(spatial_mask, null_logits[:, query_index], len(neighbor_indices))
            attended = functional.scaled_dot_product_attention(q_frame, keys, values, attn_mask=bias.to(q.dtype))
            frame_outputs.append(rearrange(attended, "b nh (h w) d -> b h w nh d", h=height, w=width))
        return torch.stack(frame_outputs, dim=1)

    def sdpa_attention_bias(self, spatial_mask: Tensor, frame_null_logits: Tensor, num_neighbors: int) -> Tensor:
        """Build the additive bias: window gating for real keys, null logits in the last column."""
        window_bias = torch.where(spatial_mask.repeat(1, num_neighbors), 0.0, float("-inf"))
        null_bias = rearrange(frame_null_logits, "b h w nh -> b nh (h w) 1")
        window_bias = window_bias[None, None].expand(*null_bias.shape[:2], -1, -1)
        return torch.cat((window_bias, null_bias), dim=-1)

    def compute_natten_attention(self, q: Tensor, k: Tensor, v: Tensor, q_unrotated: Tensor) -> Tensor:
        """Compute correspondence with fused 2D neighborhood-attention kernels."""
        if q.device.type != "cuda":
            msg = "NATTEN motion attention requires CUDA"
            raise RuntimeError(msg)
        natten = load_natten()
        batch, frames, height, width, _heads, _dim = q.shape
        if self.spatial_kernel_size > min(height, width):
            msg = f"spatial kernel {self.spatial_kernel_size} exceeds feature map {height}x{width}"
            raise ValueError(msg)
        null_logits = self.null_match_logits(q_unrotated)

        grouped_queries: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
        for query_index in range(frames):
            neighbors = temporal_neighbor_indices(query_index, frames, self.temporal_offsets)
            grouped_queries[len(neighbors)].append((query_index, neighbors))

        frame_outputs: list[Tensor | None] = [None] * frames
        for group in grouped_queries.values():
            query_indices = tuple(query_index for query_index, _neighbors in group)
            q_group = rearrange(q[:, query_indices], "b t h w nh d -> (b t) h w nh d").contiguous()
            group_outputs: list[Tensor] = []
            group_lse: list[Tensor] = []
            for neighbor_position in range(len(group[0][1])):
                neighbor_indices = tuple(neighbors[neighbor_position] for _query_index, neighbors in group)
                k_group = rearrange(k[:, neighbor_indices], "b t h w nh d -> (b t) h w nh d").contiguous()
                v_group = rearrange(v[:, neighbor_indices], "b t h w nh d -> (b t) h w nh d").contiguous()
                output, lse = natten.na2d(
                    q_group,
                    k_group,
                    v_group,
                    kernel_size=(self.spatial_kernel_size, self.spatial_kernel_size),
                    backend=self.natten_backend,
                    return_lse=True,
                )
                group_outputs.append(output.flatten(1, 2))
                group_lse.append(lse.flatten(1, 2))
            merged = merge_attention_branches(
                group_outputs,
                group_lse,
                self.null_value.to(q),
                rearrange(null_logits[:, query_indices], "b t h w nh -> (b t) (h w) nh"),
                torch_compile=self.merge_compile,
            )
            merged = rearrange(
                merged,
                "(b t) (h w) nh d -> b t h w nh d",
                b=batch,
                t=len(query_indices),
                h=height,
                w=width,
            )
            for group_index, query_index in enumerate(query_indices):
                frame_outputs[query_index] = merged[:, group_index]
        if any(output is None for output in frame_outputs):
            msg = "NATTEN correspondence did not produce every query frame"
            raise RuntimeError(msg)
        return torch.stack([output for output in frame_outputs if output is not None], dim=1)


def load_natten() -> ModuleType:
    """Import NATTEN lazily and require its fused CUDA extension."""
    try:
        natten = importlib.import_module("natten")
    except ImportError as error:
        msg = "NATTEN backend requested, but the natten package is not installed"
        raise RuntimeError(msg) from error
    if not getattr(natten, "HAS_LIBNATTEN", False):
        msg = "NATTEN backend requested, but fused libnatten kernels are unavailable"
        raise RuntimeError(msg)
    return natten
