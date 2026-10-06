# Diffusion transformer architecture for text-conditioned visual generation.

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from musubi_tuner.kandinsky6.runtime.sr.core.components.model.logger import ModelLogger
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.nn import (
    FeedForward,
    Modulation,
    ModulationLQ,
    MultiheadCrossAttention,
    MultiheadSelfAttention,
    OutLayer,
    RoPE1D,
    RoPE3D,
    TextEmbeddings,
    TimeEmbeddings,
    VisualEmbeddings,
    apply_gate_sum,
    apply_scale_shift_norm,
)
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.utils import fractal_flatten, fractal_unflatten


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with self-attention, feed-forward, and adaptive modulation."""

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int) -> None:
        """Initialize encoder block layers.

        Args:
            model_dim: Hidden dimension of the transformer.
            time_dim: Dimension of time step embeddings.
            ff_dim: Inner dimension of the feed-forward network.
            head_dim: Per-head dimension for self-attention.
        """
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(
        self,
        x: Tensor,
        time_embed: Tensor,
        rope: Tensor,
        cu_seqlens: Tensor,
        time_embed_idx: Tensor,
    ) -> Tensor:
        """Run self-attention and feed-forward with time-conditioned modulation.

        Args:
            x: Input text embeddings.
            time_embed: Time step embeddings for modulation.
            rope: Rotary position embeddings.
            cu_seqlens: Cumulative sequence lengths for packed sequences.
            time_embed_idx: Index mapping tokens to their time embeddings.

        Returns:
            Modulated text embeddings.
        """
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.self_attention(out, rope, cu_seqlens)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.feed_forward(out)
        return apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.text_modulation.reset_parameters()

        self.self_attention_norm.reset_parameters()
        self.self_attention.reset_parameters()

        self.feed_forward_norm.reset_parameters()
        self.feed_forward.reset_parameters()


@ModelLogger.log_transformer_block
class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with self-attention, cross-attention, and feed-forward.

    When ``use_lq_modulation=True``, a parallel ``ModulationLQ`` produces per-token
    scale/shift/gate from LQ encodings, added to the per-sample time modulation.
    Zero-initialized so that at init the block behaves identically to the baseline.
    """

    LQ_SUBLAYER_ORDER = ("self_attention", "cross_attention", "ffn")

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        *,
        use_lq_modulation: bool = False,
        lq_modulation_sublayers: dict[str, bool] | None = None,
        use_text: bool = True,
    ) -> None:
        """Initialize decoder block layers.

        Args:
            model_dim: Hidden dimension of the transformer.
            time_dim: Dimension of time step embeddings.
            ff_dim: Inner dimension of the feed-forward network.
            head_dim: Per-head dimension for attention layers.
            use_lq_modulation: Whether to add per-token LQ modulation.
            lq_modulation_sublayers: Which sub-layers to modulate. Keys:
                ``self_attention``, ``cross_attention``, ``ffn``. Defaults to all True.
            use_text: When False, the block has no text cross-attention: no
                ``cross_attention``/``cross_attention_norm`` modules and the
                modulation produces 6 params (self-attn + FFN) instead of 9.
                Used by the text-free SR variant.
        """
        super().__init__()
        self.use_lq_modulation = use_lq_modulation
        self.use_text = use_text
        self.visual_modulation = Modulation(time_dim, model_dim, 9 if use_text else 6)

        if lq_modulation_sublayers is None:
            lq_modulation_sublayers = dict.fromkeys(self.LQ_SUBLAYER_ORDER, True)
        self.lq_sublayers = lq_modulation_sublayers
        self.lq_num_active = sum(lq_modulation_sublayers[k] for k in self.LQ_SUBLAYER_ORDER)

        if use_lq_modulation and self.lq_num_active > 0:
            self.lq_modulation = ModulationLQ(time_dim, model_dim, 3 * self.lq_num_active)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        if use_text:
            self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
            self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    @torch.compile(dynamic=True)
    def forward(
        self,
        visual_embed: Tensor,
        text_embed: Tensor,
        time_embed: Tensor,
        rope: Tensor,
        visual_cu_seqlens: Tensor,
        text_cu_seqlens: Tensor,
        time_embed_idx: Tensor,
        sparse_params: dict[str, Any] | None,
        lq_tokens: Tensor | None = None,
    ) -> Tensor:
        """Run self-attention, cross-attention, and feed-forward with modulation.

        Args:
            visual_embed: Visual token embeddings.
            text_embed: Encoded text embeddings used as cross-attention context.
            time_embed: Time step embeddings for modulation.
            rope: Rotary position embeddings for visual tokens.
            visual_cu_seqlens: Cumulative sequence lengths for visual tokens.
            text_cu_seqlens: Cumulative sequence lengths for text tokens.
            time_embed_idx: Index mapping visual tokens to their time embeddings.
            sparse_params: Optional sparse attention parameters.
            lq_tokens: Per-token LQ encodings ``(total_tokens, time_dim)`` for
                spatially-varying modulation. ``None`` when LQ modulation is off.

        Returns:
            Updated visual embeddings.
        """
        if self.use_text:
            self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)
        else:
            self_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 2, dim=-1)
            cross_attn_params = None

        has_lq = self.use_lq_modulation and lq_tokens is not None and self.lq_num_active > 0
        lq_sa_shift = lq_sa_scale = lq_sa_gate = None
        lq_ca_shift = lq_ca_scale = lq_ca_gate = None
        lq_ff_shift = lq_ff_scale = lq_ff_gate = None
        if has_lq:
            # lq_modulation output: (total_tokens, 3 * num_active * model_dim)
            lq_chunks = torch.chunk(self.lq_modulation(lq_tokens), self.lq_num_active, dim=-1)
            # Each chunk: (total_tokens, 3 * model_dim) -> (shift, scale, gate) per-token
            idx = 0
            if self.lq_sublayers["self_attention"]:
                lq_sa_shift, lq_sa_scale, lq_sa_gate = torch.chunk(lq_chunks[idx], 3, dim=-1)
                idx += 1
            if self.lq_sublayers["cross_attention"]:
                lq_ca_shift, lq_ca_scale, lq_ca_gate = torch.chunk(lq_chunks[idx], 3, dim=-1)
                idx += 1
            if self.lq_sublayers["ffn"]:
                lq_ff_shift, lq_ff_scale, lq_ff_gate = torch.chunk(lq_chunks[idx], 3, dim=-1)

        # --- Self-attention ---
        # Norm -> modulate by (scale_t + scale_lq + 1, shift_t + shift_lq) -> SelfAttn -> gate by (gate_t + gate_lq)
        shift_t, scale_t, gate_t = torch.chunk(self_attn_params, 3, dim=-1)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            normed = self.self_attention_norm(visual_embed)
            scale = scale_t.index_select(0, time_embed_idx)
            shift = shift_t.index_select(0, time_embed_idx)
            visual_out = normed * (scale + 1.0) + shift
            if lq_sa_scale is not None:
                visual_out = visual_out + lq_sa_scale * normed + lq_sa_shift
        visual_out = visual_out.type_as(visual_embed)
        visual_out = self.self_attention(visual_out, rope, visual_cu_seqlens, sparse_params)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            visual_embed = visual_embed + gate_t.index_select(0, time_embed_idx) * visual_out
            if lq_sa_gate is not None:
                visual_embed = visual_embed + lq_sa_gate * visual_out

        # --- Cross-attention --- (skipped when text-free)
        # Norm -> modulate by (scale_t + scale_lq + 1, shift_t + shift_lq) -> CrossAttn -> gate by (gate_t + gate_lq)
        if self.use_text:
            shift_t, scale_t, gate_t = torch.chunk(cross_attn_params, 3, dim=-1)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                normed = self.cross_attention_norm(visual_embed)
                scale = scale_t.index_select(0, time_embed_idx)
                shift = shift_t.index_select(0, time_embed_idx)
                visual_out = normed * (scale + 1.0) + shift
                if lq_ca_scale is not None:
                    visual_out = visual_out + lq_ca_scale * normed + lq_ca_shift
            visual_out = visual_out.type_as(visual_embed)
            visual_out = self.cross_attention(visual_out, text_embed, visual_cu_seqlens, text_cu_seqlens)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                visual_embed = visual_embed + gate_t.index_select(0, time_embed_idx) * visual_out
                if lq_ca_gate is not None:
                    visual_embed = visual_embed + lq_ca_gate * visual_out

        # --- Feed-forward ---
        # Norm -> modulate by (scale_t + scale_lq + 1, shift_t + shift_lq) -> FF -> gate by (gate_t + gate_lq)
        shift_t, scale_t, gate_t = torch.chunk(ff_params, 3, dim=-1)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            normed = self.feed_forward_norm(visual_embed)
            scale = scale_t.index_select(0, time_embed_idx)
            shift = shift_t.index_select(0, time_embed_idx)
            visual_out = normed * (scale + 1.0) + shift
            if lq_ff_scale is not None:
                visual_out = visual_out + lq_ff_scale * normed + lq_ff_shift
        visual_out = visual_out.type_as(visual_embed)
        visual_out = self.feed_forward(visual_out)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            visual_embed = visual_embed + gate_t.index_select(0, time_embed_idx) * visual_out
            if lq_ff_gate is not None:
                visual_embed = visual_embed + lq_ff_gate * visual_out
        return visual_embed

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.visual_modulation.reset_parameters()
        if self.use_lq_modulation and self.lq_num_active > 0:
            self.lq_modulation.reset_parameters()

        self.self_attention_norm.reset_parameters()
        self.self_attention.reset_parameters()

        if self.use_text:
            self.cross_attention_norm.reset_parameters()
            self.cross_attention.reset_parameters()

        self.feed_forward_norm.reset_parameters()
        self.feed_forward.reset_parameters()


class VideoAdapter(nn.Module):
    """Lighter DiT that conditions a frozen backbone on degraded video input.

    The adapter has the same block design as the backbone but half as many blocks.
    LQ latent tokens c = P(z̃) are processed through adapter blocks whose outputs
    are injected into alternating backbone blocks via learnable γ_ℓ scalars.

    γ_ℓ are zero-initialized so the adapter has no effect at init, enabling
    fine-tuning from a pretrained backbone checkpoint.
    """

    def __init__(
        self,
        in_visual_dim: int,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        patch_size: tuple[int, int, int],
        num_blocks: int,
        gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.gamma = gamma
        patch_vol = math.prod(patch_size)
        self.lq_proj = nn.Linear(patch_vol * in_visual_dim, model_dim)
        self.blocks = nn.ModuleList([TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim) for _ in range(num_blocks)])

    def _patchify(self, x: Tensor, visual_cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
        """Patchify LQ latent with the same logic as VisualEmbeddings._patchify."""
        pt, ph, pw = self.patch_size
        if pt > 1:
            idxs = torch.ones(x.shape[0], dtype=torch.int32, device=visual_cu_seqlens.device)
            idxs[visual_cu_seqlens[:-1]] += pt - 1
            x = torch.repeat_interleave(x, idxs, dim=0)
            visual_cu_seqlens = visual_cu_seqlens + torch.arange(
                visual_cu_seqlens.shape[0], device=visual_cu_seqlens.device, dtype=torch.int32
            )
        T, H, W, C = x.shape
        x = x.view(T // pt, pt, H // ph, ph, W // pw, pw, C).permute(0, 2, 4, 1, 3, 5, 6).flatten(3, 6)
        return x, visual_cu_seqlens // pt

    def forward(
        self,
        lq_visual: Tensor,
        pre_patch_cu_seqlens: Tensor,
        time_embed: Tensor,
        text_embed: Tensor,
        visual_rope: Tensor,
        visual_cu_seqlens: Tensor,
        text_cu_seqlens: Tensor,
        visual_time_embed_idx: Tensor,
        sparse_params: dict[str, Any] | None,
        to_fractal: bool,
    ) -> list[Tensor]:
        """Process LQ tokens through adapter blocks, return γ-scaled features."""
        lq_patches, _ = self._patchify(lq_visual, pre_patch_cu_seqlens)
        adapter_embed = self.lq_proj(lq_patches)
        if to_fractal:
            from musubi_tuner.kandinsky6.runtime.sr.core.components.model.utils import local_patching  # noqa: PLC0415

            visual_shape = adapter_embed.shape[:-1]
            adapter_embed = local_patching(adapter_embed, visual_shape, (1, 8, 8), dim=0)
            adapter_embed = adapter_embed.flatten(0, 1)
        else:
            adapter_embed = adapter_embed.flatten(0, 2)
        features = []
        for block in self.blocks:
            adapter_embed = block(
                adapter_embed,
                text_embed,
                time_embed,
                visual_rope,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_time_embed_idx,
                sparse_params,
            )
            features.append(self.gamma * adapter_embed)
        return features

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.lq_proj.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()


@ModelLogger.log_transformer
class DiffusionTransformer3D(nn.Module):
    """3D diffusion transformer with text-conditioned visual generation.

    Processes visual tokens through encoder (text) and decoder (visual) transformer blocks
    with RoPE, adaptive modulation, and optional fractal attention sparsity.
    """

    def __init__(
        self,
        in_visual_dim: int = 4,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        time_dim: int = 512,
        out_visual_dim: int = 4,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        model_dim: int = 2048,
        ff_dim: int = 5120,
        num_text_blocks: int = 2,
        num_visual_blocks: int = 32,
        axes_dims: tuple[int, int, int] = (16, 24, 24),
        *,
        visual_cond: bool = False,
        instruct_type: str | None = None,
        attention_params: dict[str, Any] | None = None,
        use_motion_score: bool = False,
        use_lq_modulation: bool = False,
        lq_modulation_sublayers: dict[str, bool] | None = None,
        zero_lq_in_main_path: bool = False,
        use_adapter: bool = False,
        adapter_gamma: float = 0.0,
        use_text: bool = True,
        use_lq_noise_cond: bool = False,
    ) -> None:
        """Initialize the 3D diffusion transformer.

        Args:
            in_visual_dim: Number of input visual channels.
            in_text_dim: Dimension of text encoder hidden states.
            in_text_dim2: Dimension of pooled text embeddings.
            time_dim: Dimension of time step embeddings.
            out_visual_dim: Number of output visual channels.
            patch_size: Patch size as (temporal, height, width).
            model_dim: Hidden dimension of the transformer.
            ff_dim: Inner dimension of feed-forward networks.
            num_text_blocks: Number of text encoder blocks.
            num_visual_blocks: Number of visual decoder blocks.
            axes_dims: RoPE dimension split per axis (temporal, height, width).
            visual_cond: Whether to use visual conditioning input.
            instruct_type: Conditioning mode (e.g. ``"channel"``).
            attention_params: Extra attention configuration.
            use_motion_score: Whether to add motion score embeddings to time conditioning.
            use_lq_modulation: Whether to enable per-block spatially-varying LQ modulation.
            lq_modulation_sublayers: Which sub-layers to modulate with LQ. Keys:
                ``self_attention``, ``cross_attention``, ``ffn``. Defaults to all True.
            zero_lq_in_main_path: Zero out LQ+mask channels in ``in_layer`` input so
                LQ reaches the model only via ``ModulationLQ``. Ablation experiment flag.
            use_adapter: Whether to attach a VideoAdapter that conditions the backbone
                on the degraded LQ input. The adapter has ``num_visual_blocks // 2``
                blocks and injects γ-scaled features into alternating backbone blocks.
            adapter_gamma: Fixed scalar γ (not a trainable parameter) controlling
                adapter influence on the backbone. Small values keep the backbone
                dominant; larger values allow stronger structural correction.
                Only used when ``use_adapter=True``.
            use_text: When False, builds a text-free model with no text encoder,
                text RoPE/blocks, pooled-text conditioning, or per-block cross-
                attention. Default True keeps the standard architecture.
            use_lq_noise_cond: When True, adds a second ``TimeEmbeddings`` that
                encodes the LQ-endpoint noise fraction ``s`` (``lq_noise_scale``)
                and sums it into the adaLN conditioning bus (like motion_score).
                Its ``out_layer`` is zero-initialised so a checkpoint without the
                module warm-starts as an exact no-op.
        """
        super().__init__()
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.instruct_type = instruct_type
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.attention_params = attention_params
        self.use_lq_modulation = use_lq_modulation
        if use_lq_modulation and instruct_type not in ("channel", "hybrid"):
            msg = "use_lq_modulation requires instruct_type='channel' or 'hybrid'"
            raise ValueError(msg)

        self.use_adapter = use_adapter

        visual_embed_dim = (
            2 * in_visual_dim + 1 if visual_cond or instruct_type in ("channel", "hybrid", "hybrid_anchor") else in_visual_dim
        )
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        if use_motion_score:
            self.motion_embeddings = TimeEmbeddings(model_dim, time_dim)
        if use_lq_noise_cond:
            self.lq_noise_embeddings = TimeEmbeddings(model_dim, time_dim)
            self.zero_init_lq_noise_out()

        self.use_motion_score = use_motion_score
        self.use_lq_noise_cond = use_lq_noise_cond
        self.use_text = use_text
        if use_adapter and not use_text:
            msg = "use_adapter requires use_text=True: VideoAdapter blocks use text cross-attention"
            raise ValueError(msg)
        if use_text:
            self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
            self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        else:
            # Text-free: under the empty caption the pooled-text contribution
            # `pooled_text_embeddings("")` is a constant added to the time embedding.
            # build_text_free_dit.py bakes it into this bias; dropping it instead corrupts
            # the time conditioning, so the model still needs it (cross-attention does not).
            self.pooled_bias = nn.Parameter(torch.zeros(time_dim))
        self.visual_embeddings = VisualEmbeddings(
            visual_embed_dim,
            model_dim,
            patch_size,
            use_lq_modulation=use_lq_modulation,
            lq_channels=in_visual_dim,
            time_dim=time_dim,
            zero_lq_in_main_path=zero_lq_in_main_path,
        )

        if use_text:
            self.text_rope_embeddings = RoPE1D(head_dim)
            self.text_transformer_blocks = nn.ModuleList(
                [TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim) for _ in range(num_text_blocks)]
            )

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    use_lq_modulation=use_lq_modulation,
                    lq_modulation_sublayers=lq_modulation_sublayers,
                    use_text=use_text,
                )
                for _ in range(num_visual_blocks)
            ]
        )

        if use_adapter:
            self.adapter = VideoAdapter(
                in_visual_dim=in_visual_dim,
                model_dim=model_dim,
                time_dim=time_dim,
                ff_dim=ff_dim,
                head_dim=head_dim,
                patch_size=patch_size,
                num_blocks=num_visual_blocks // 2,
                gamma=adapter_gamma,
            )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

    def forward(
        self,
        x: Tensor,
        text_embed: Tensor,
        pooled_text_embed: Tensor,
        time: Tensor,
        visual_cu_seqlens: Tensor,
        text_cu_seqlens: Tensor,
        visual_rope_pos: tuple[Tensor, Tensor, Tensor],
        text_rope_pos: Tensor,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: dict[str, Any] | None = None,
        motion_score: Tensor | None = None,
        lq_latent: Tensor | None = None,
        lq_noise_level: Tensor | None = None,
    ) -> Tensor:
        """Run the full diffusion transformer forward pass.

        Args:
            x: Noisy visual input tensor.
            text_embed: Raw text encoder hidden states.
            pooled_text_embed: Pooled text embeddings added to time conditioning.
            time: Diffusion time steps.
            visual_cu_seqlens: Cumulative sequence lengths for visual tokens.
            text_cu_seqlens: Cumulative sequence lengths for text tokens.
            visual_rope_pos: 3D positional indices (temporal, height, width) for visual RoPE.
            text_rope_pos: 1D positional indices for text RoPE.
            scale_factor: RoPE frequency scaling per axis.
            sparse_params: Optional sparse/fractal attention parameters.
            motion_score: Optional motion score for temporal conditioning.
            lq_latent: Degraded video latent passed exclusively to the
                adapter when ``use_adapter=True``. Not noised. ``None`` when adapter
                is disabled.
            lq_noise_level: Per-sequence LQ-endpoint noise fraction ``s`` in
                ``[0, 1]`` (shape ``[N_seqs]`` or broadcastable ``[1]``). Only
                used when ``use_lq_noise_cond=True``; embedded like the timestep
                (scaled x1000) and summed into the adaLN conditioning bus.

        Returns:
            Denoised visual output tensor.
        """
        text_embed = self.text_embeddings(text_embed) if self.use_text else None
        time_embed, time_embed_idx = self.time_embeddings(time)
        if motion_score is not None and self.use_motion_score:
            ms_embed, _ = self.motion_embeddings(motion_score)
            time_embed = time_embed + ms_embed
        if lq_noise_level is not None and self.use_lq_noise_cond:
            # Same input scale as the timestep (t*1000 at every call site).
            nl_embed, _ = self.lq_noise_embeddings(1000.0 * lq_noise_level)
            time_embed = time_embed + nl_embed

        if self.use_text:
            time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        else:
            time_embed = time_embed + self.pooled_bias

        # Save cu_seqlens before patchification (adapter needs it for its own _patchify)
        if self.use_adapter:
            pre_patch_cu_seqlens = visual_cu_seqlens

        ve_result = self.visual_embeddings(x, visual_cu_seqlens)
        if self.use_lq_modulation:
            visual_embed, visual_cu_seqlens, lq_tokens = ve_result
        else:
            visual_embed, visual_cu_seqlens = ve_result
            lq_tokens = None

        if self.use_text:
            text_rope = self.text_rope_embeddings(text_rope_pos)
            text_time_embed_idx = time_embed_idx.repeat_interleave(torch.diff(text_cu_seqlens), dim=0)
            for text_transformer_block in self.text_transformer_blocks:
                text_embed = text_transformer_block(text_embed, time_embed, text_rope, text_cu_seqlens, text_time_embed_idx)

        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        # [T, 32, 32, 1792] -> [T*32*32, 1792]
        visual_embed, visual_rope, visual_cu_seqlens = fractal_flatten(
            visual_embed, visual_rope, visual_cu_seqlens, visual_shape, fractal=to_fractal
        )
        if lq_tokens is not None:
            # Flatten lq_tokens the same way as visual_embed (spatial dims only).
            # fractal_flatten already modified visual_cu_seqlens, so we inline the
            # flatten here rather than calling fractal_flatten again.
            if to_fractal:
                from musubi_tuner.kandinsky6.runtime.sr.core.components.model.utils import local_patching  # noqa: PLC0415

                # [T, 32, 32, dim] -> [T * (32/8)*(32/8), 8*8, dim] -> [T*16, 64, dim]
                lq_tokens = local_patching(lq_tokens, visual_shape, (1, 8, 8), dim=0)
                lq_tokens = lq_tokens.flatten(0, 1)
            else:
                # [T, 32, 32, 512] -> [T*32*32, 512]
                lq_tokens = lq_tokens.flatten(0, 2)

        visual_time_embed_idx = time_embed_idx.repeat_interleave(torch.diff(visual_cu_seqlens), dim=0)

        # Run adapter on LQ tokens and collect γ-scaled features for injection
        adapter_features: list[Tensor] | None = None
        if self.use_adapter:
            adapter_features = self.adapter(
                lq_latent,
                pre_patch_cu_seqlens,
                time_embed,
                text_embed,
                visual_rope,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_time_embed_idx,
                sparse_params,
                to_fractal,
            )

        for i, visual_transformer_block in enumerate(self.visual_transformer_blocks):
            # Inject adapter features before alternating backbone blocks (0, 2, 4, ...)
            if adapter_features is not None and i % 2 == 0:
                visual_embed = visual_embed + adapter_features[i // 2]
            visual_embed = visual_transformer_block(
                visual_embed,
                text_embed,
                time_embed,
                visual_rope,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_time_embed_idx,
                sparse_params,
                lq_tokens,
            )
        visual_embed, visual_cu_seqlens = fractal_unflatten(visual_embed, visual_cu_seqlens, visual_shape, fractal=to_fractal)

        visual_time_embed_idx = time_embed_idx.repeat_interleave(torch.diff(visual_cu_seqlens), dim=0)
        return self.out_layer(visual_embed, text_embed, time_embed, visual_cu_seqlens, visual_time_embed_idx)

    def enable_explicit_forward_prefetch(self, factor: int = 1) -> None:
        """Enable FSDP forward prefetching for visual transformer blocks.

        Args:
            factor: Number of blocks to prefetch ahead. Skipped if <= 0 or already enabled.

        Raises:
            RuntimeError: If the model is not wrapped with FSDP.
        """
        if factor <= 0 or getattr(self, "forward_prefetch", False):
            return
        for i, layer in enumerate(self.visual_transformer_blocks):
            if not hasattr(layer, "set_modules_to_forward_prefetch"):
                msg = "wrap DiT with FSDP first to use forward_prefetch"
                raise RuntimeError(msg)

            start, end = i, i + factor
            prefetch_modules = self.visual_transformer_blocks[start:end]
            layer.set_modules_to_forward_prefetch(prefetch_modules)
        self.forward_prefetch = True

    def zero_init_lq_noise_out(self) -> None:
        """Zero the s-embedder output layer so its contribution starts at exactly 0.

        Keeps warm-start from a checkpoint without ``lq_noise_embeddings`` an
        exact no-op (adaLN-zero pattern). Must be re-applied after any
        ``reset_parameters`` (FSDP meta-init materialisation re-randomises it).
        """
        nn.init.zeros_(self.lq_noise_embeddings.out_layer.weight)
        nn.init.zeros_(self.lq_noise_embeddings.out_layer.bias)

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters across every sub-module."""
        self.time_embeddings.reset_parameters()
        if self.use_lq_noise_cond:
            self.lq_noise_embeddings.reset_parameters()
            self.zero_init_lq_noise_out()
        if self.use_text:
            self.text_embeddings.reset_parameters()
            self.pooled_text_embeddings.reset_parameters()
        else:
            self.pooled_bias.data.zero_()
        self.visual_embeddings.reset_parameters()

        if self.use_text:
            self.text_rope_embeddings.reset_parameters()
            for module in self.text_transformer_blocks:
                module.reset_parameters()

        self.visual_rope_embeddings.reset_parameters()
        for module in self.visual_transformer_blocks:
            module.reset_parameters()

        self.out_layer.reset_parameters()

        if self.use_adapter:
            self.adapter.reset_parameters()


def get_dit(conf: dict[str, Any]) -> DiffusionTransformer3D:
    """Create a DiffusionTransformer3D from a config dict.

    Args:
        conf: Keyword arguments forwarded to DiffusionTransformer3D.

    Returns:
        Initialized DiffusionTransformer3D instance.
    """
    return DiffusionTransformer3D(**conf)
