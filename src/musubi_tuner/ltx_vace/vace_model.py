"""
VACE (Video Adaptive Context Encoder) adapted for LTX-2.

A parallel context encoder that generates "hints" injected residually into
DiT blocks. The base DiT stays frozen; only VACE parameters are trained.

Architecture adapted from VACE (arXiv:2503.07598) to match LTX-2's
transformer conventions (Attention, FeedForward, AdaLN, RoPE, RMSNorm).
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from musubi_tuner.ltx_2.model.transformer.attention import Attention, AttentionFunction
from musubi_tuner.ltx_2.model.transformer.adaln import adaln_embedding_coefficient
from musubi_tuner.ltx_2.model.transformer.feed_forward import FeedForward
from musubi_tuner.ltx_2.model.transformer.rope import LTXRopeType
from musubi_tuner.ltx_2.utils import rms_norm

logger = logging.getLogger(__name__)

# Default: inject at every 4th of 48 DiT blocks (12 VACE blocks)
DEFAULT_VACE_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)


class VaceLTXBlock(nn.Module):
    """Single VACE transformer block, mirrors the video path of BasicAVTransformerBlock.

    Each block performs:
      1. AdaLN + self-attention
      2. Text cross-attention
      3. AdaLN + feed-forward

    Additionally maintains skip-connection projections:
      - before_proj (block_id=0 only): fuses initial control features with DiT latent
      - after_proj (all blocks): projects for skip-connection hint output
    """

    def __init__(
        self,
        block_id: int,
        dim: int = 4096,
        num_heads: int = 32,
        d_head: int = 128,
        context_dim: int = 4096,
        ffn_mult: int = 4,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionFunction = AttentionFunction.DEFAULT,
        audio_context_dim: int | None = None,
        timestep_ada_embed_count: int = 6,
    ):
        super().__init__()
        self.block_id = block_id
        self.dim = dim
        self.norm_eps = norm_eps
        self.timestep_ada_embed_count = timestep_ada_embed_count

        # Self-attention
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=d_head,
            context_dim=None,  # self-attention
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )

        # Cross-attention to text
        self.attn2 = Attention(
            query_dim=dim,
            context_dim=context_dim,
            heads=num_heads,
            dim_head=d_head,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )

        # Optional cross-attention to audio hidden states.
        # Output projection is zero-initialized in _init_weights, so this
        # starts as a no-op. Gradients may activate it during training.
        if audio_context_dim is not None:
            self.audio_attn = Attention(
                query_dim=dim,
                context_dim=audio_context_dim,
                heads=num_heads,
                dim_head=d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )

        # Feed-forward
        self.ff = FeedForward(dim, dim_out=dim, mult=ffn_mult)

        # AdaLN: 6 values = (shift, scale, gate) for self-attn + (shift, scale, gate) for FFN
        self.scale_shift_table = nn.Parameter(torch.empty(6, dim))

        # VACE skip-connection projections
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)

    def forward(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        timesteps: torch.Tensor,
        pe: torch.Tensor,
        audio_context: torch.Tensor | None = None,
        audio_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            c: Accumulated VACE state.
               Block 0: (batch, seq_len, dim) -- initial patchified control.
               Block n>0: (n_hints, batch, seq_len, dim) -- stacked skip hints + running state.
            x: DiT video hidden state (batch, seq_len, dim) -- used only in block 0 for fusion.
            context: Text embeddings (batch, num_tokens, context_dim).
            context_mask: Text attention mask or None.
            timesteps: Timestep embeddings for AdaLN (batch, num_tokens, 6*dim) or tuple format.
            pe: RoPE positional embeddings.
            audio_context: Optional audio hidden states (batch, audio_seq_len, audio_dim).
            audio_context_mask: Optional attention mask for audio context.

        Returns:
            Updated c with new skip hint appended.
        """
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        batch_size = c.shape[0]

        # --- Self-attention with AdaLN ---
        shift_msa, scale_msa, gate_msa = self._get_ada_values(
            self.scale_shift_table, batch_size, timesteps, slice(0, 3), c.shape[1]
        )
        norm_c = (
            rms_norm(c, eps=self.norm_eps).to(torch.float32)
            * (1 + scale_msa.to(torch.float32))
            + shift_msa.to(torch.float32)
        ).to(c.dtype)
        c = c + self.attn1(norm_c, pe=pe) * gate_msa

        # --- Cross-attention to text (with RMSNorm on query, matching DiT pattern) ---
        c = c + self.attn2(rms_norm(c, eps=self.norm_eps), context=context, mask=context_mask)

        # --- Optional cross-attention to audio hidden states ---
        if audio_context is not None and hasattr(self, "audio_attn"):
            c = c + self.audio_attn(
                rms_norm(c, eps=self.norm_eps), context=audio_context, mask=audio_context_mask
            )

        # --- FFN with AdaLN ---
        shift_mlp, scale_mlp, gate_mlp = self._get_ada_values(
            self.scale_shift_table, batch_size, timesteps, slice(3, 6), c.shape[1]
        )
        norm_c = (
            rms_norm(c, eps=self.norm_eps).to(torch.float32)
            * (1 + scale_mlp.to(torch.float32))
            + shift_mlp.to(torch.float32)
        ).to(c.dtype)
        c = c + self.ff(norm_c) * gate_mlp

        # --- Skip connection ---
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c

    def _get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep,
        indices: slice,
        num_tokens: int,
    ) -> tuple[torch.Tensor, ...]:
        """Extract AdaLN shift/scale/gate values from timestep embeddings.

        Supports both standard tensor and optimized tuple formats (matching
        BasicAVTransformerBlock.get_ada_values).
        """
        num_ada_params = self.timestep_ada_embed_count

        if isinstance(timestep, tuple) and len(timestep) == 4:
            unique_emb, inverse_indices_1d, orig_batch_size, orig_num_tokens = timestep
            unique_reshaped = unique_emb.reshape(len(unique_emb), num_ada_params, -1)[:, indices, :]
            table_values = scale_shift_table[indices].unsqueeze(0).to(
                device=unique_emb.device, dtype=unique_emb.dtype
            )
            unique_ada = (table_values + unique_reshaped).unbind(dim=1)
            return tuple(
                unique_val[inverse_indices_1d].view(orig_batch_size, orig_num_tokens, -1)
                for unique_val in unique_ada
            )

        timestep_reshaped = timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[
            :, :, indices, :
        ]
        table_values = (
            scale_shift_table[indices]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
        )
        if num_tokens is not None and timestep.shape[1] < num_tokens:
            repeats = num_tokens // timestep.shape[1]
            if repeats > 1:
                if repeats * timestep.shape[1] == num_tokens:
                    timestep_reshaped = torch.repeat_interleave(timestep_reshaped, repeats, dim=1)
                else:
                    timestep_reshaped = torch.repeat_interleave(
                        timestep_reshaped, repeats + 1, dim=1
                    )[:, :num_tokens]
        return (table_values + timestep_reshaped).unbind(dim=2)


class VaceLTXModel(nn.Module):
    """VACE context encoder for LTX-2.

    Runs in parallel with the frozen DiT. Takes control video latents
    (inactive + reactive + mask), patchifies them, processes through
    VACE transformer blocks, and produces per-layer hints that are
    residually added to DiT hidden states.

    Args:
        vace_layers: Tuple of DiT block indices where hints are injected.
        vace_in_dim: Input channels after control signal preparation.
                     For LTX-2: 1280 = 2*128 latent + 1024 mask (32^2 spatial compression).
        dim: Hidden dimension (must match LTX-2 inner_dim).
        num_heads: Number of attention heads.
        d_head: Dimension per head.
        context_dim: Text embedding dimension.
        ffn_mult: FFN expansion factor.
        norm_eps: Epsilon for RMSNorm.
        rope_type: RoPE type (match LTX-2).
        attention_function: Attention backend.
    """

    def __init__(
        self,
        vace_layers: Sequence[int] = DEFAULT_VACE_LAYERS,
        vace_in_dim: int = 1280,
        latent_channels: int = 128,
        dim: int = 4096,
        num_heads: int = 32,
        d_head: int = 128,
        context_dim: int = 4096,
        ffn_mult: int = 4,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionFunction = AttentionFunction.DEFAULT,
        audio_context_dim: int | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self.vace_layers = tuple(vace_layers)
        self.vace_in_dim = vace_in_dim
        self.latent_channels = latent_channels
        self.dim = dim
        self.audio_context_dim = audio_context_dim
        self.vace_layers_mapping = {layer_id: idx for idx, layer_id in enumerate(self.vace_layers)}
        self.timestep_ada_embed_count = adaln_embedding_coefficient(cross_attention_adaln)

        # Project VACE control channels to model dim.
        # Spatial patchification is handled externally by the DiT's patchify_proj or
        # by patchify_vace_context(). This projection handles the channel dimension:
        # vace_in_dim (e.g. 320 = 2*128 latent + 64 mask) -> dim (e.g. 4096).
        self.vace_input_proj = nn.Linear(vace_in_dim, dim)

        # VACE transformer blocks
        self.vace_blocks = nn.ModuleList([
            VaceLTXBlock(
                block_id=i,
                dim=dim,
                num_heads=num_heads,
                d_head=d_head,
                context_dim=context_dim,
                ffn_mult=ffn_mult,
                norm_eps=norm_eps,
                rope_type=rope_type,
                attention_function=attention_function,
                audio_context_dim=audio_context_dim,
                timestep_ada_embed_count=self.timestep_ada_embed_count,
            )
            for i in range(len(self.vace_layers))
        ])

        self.gradient_checkpointing = False

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following LTX-2 conventions."""
        for block in self.vace_blocks:
            nn.init.zeros_(block.scale_shift_table)
            nn.init.zeros_(block.after_proj.weight)
            nn.init.zeros_(block.after_proj.bias)
            if hasattr(block, "before_proj"):
                nn.init.zeros_(block.before_proj.bias)
            # Zero-init audio cross-attention output so it starts as no-op
            if hasattr(block, "audio_attn"):
                nn.init.zeros_(block.audio_attn.to_out[0].weight)
                if block.audio_attn.to_out[0].bias is not None:
                    nn.init.zeros_(block.audio_attn.to_out[0].bias)

    def initialize_input_proj_from_patchify_proj(self, patchify_proj: nn.Linear) -> None:
        """Initialize the Context Embedder from the base video embedder.

        The latent branches (inactive/reactive) reuse the base patchify weights,
        while mask channels start from zero as described in the VACE paper.

        When patch_size > 1, patchify_proj.in_features = latent_channels * prod(patch_size).
        The VACE input proj has in_features = vace_in_dim = (2*latent_channels + mask_channels) * prod(patch_size).
        We copy the patchify weights to the inactive and reactive slices.
        """
        patchified_latent_dim = patchify_proj.weight.shape[1]  # latent_channels * prod(patch_size)

        with torch.no_grad():
            self.vace_input_proj.weight.zero_()

            base_weight = patchify_proj.weight.detach()  # [dim, patchified_latent_dim]
            self.vace_input_proj.weight[:, :patchified_latent_dim].copy_(base_weight)
            self.vace_input_proj.weight[
                :, patchified_latent_dim : 2 * patchified_latent_dim
            ].copy_(base_weight)
            # Remaining columns (mask channels) stay zero-initialized

            if patchify_proj.bias is not None and self.vace_input_proj.bias is not None:
                self.vace_input_proj.bias.copy_(patchify_proj.bias.detach())
            elif self.vace_input_proj.bias is not None:
                self.vace_input_proj.bias.zero_()

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(
        self,
        x: torch.Tensor,
        vace_context: torch.Tensor,
        text_context: torch.Tensor,
        text_context_mask: torch.Tensor | None,
        timesteps: torch.Tensor,
        pe: torch.Tensor,
        audio_context: torch.Tensor | None = None,
        audio_context_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Compute VACE hints for injection into DiT blocks.

        Args:
            x: DiT video hidden state (batch, seq_len, dim) -- from patchify_proj.
            vace_context: Patchified control input (batch, seq_len, vace_in_dim).
            text_context: Text embeddings (batch, num_tokens, context_dim).
            text_context_mask: Text attention mask or None.
            timesteps: Timestep embeddings for AdaLN.
            pe: RoPE positional embeddings.
            audio_context: Optional audio hidden states (batch, audio_seq_len, audio_dim).
            audio_context_mask: Optional attention mask for audio context.

        Returns:
            List of hint tensors, one per VACE layer. Each has shape (batch, seq_len, dim).
        """
        # Project control input to model dimension
        c = self.vace_input_proj(vace_context)

        # Pad to match DiT sequence length if needed
        if c.shape[1] < x.shape[1]:
            pad = x.new_zeros(c.shape[0], x.shape[1] - c.shape[1], c.shape[2])
            c = torch.cat([c, pad], dim=1)
        elif c.shape[1] > x.shape[1]:
            c = c[:, : x.shape[1], :]

        # Process through VACE blocks
        for block in self.vace_blocks:
            if self.gradient_checkpointing and self.training:
                c = checkpoint.checkpoint(
                    block,
                    c, x, text_context, text_context_mask, timesteps, pe,
                    audio_context, audio_context_mask,
                    use_reentrant=False,
                )
            else:
                c = block(
                    c, x, text_context, text_context_mask, timesteps, pe,
                    audio_context=audio_context, audio_context_mask=audio_context_mask,
                )

        # Extract hints (all skip connections, excluding the final running state)
        hints = list(torch.unbind(c))[:-1]
        return hints

    @property
    def num_hints(self) -> int:
        """Number of hint tensors produced (= number of VACE blocks)."""
        return len(self.vace_blocks)

    def get_hint_for_layer(self, hints: list[torch.Tensor], dit_block_idx: int) -> torch.Tensor | None:
        """Get the hint tensor for a specific DiT block index, or None if not mapped."""
        if dit_block_idx not in self.vace_layers_mapping:
            return None
        return hints[self.vace_layers_mapping[dit_block_idx]]


class AudioVaceLTXModel(nn.Module):
    """VACE context encoder for LTX-2 audio modality.

    Mirrors VaceLTXModel but operates on audio latents with audio-specific
    dimensions. Produces per-layer hints injected into DiT audio hidden states.

    Audio control signal: inactive_tokens | reactive_tokens | mask_scalar
    = (B, T, 2 * audio_in_channels + 1), e.g. (B, T, 257) for 128-ch audio.

    Unlike video VACE (which uses 32x32=1024 mask channels due to spatial
    VAE compression), audio uses a single scalar mask per timestep since
    audio VACE operates in token space after patchification.

    Args:
        vace_layers: Tuple of DiT block indices where audio hints are injected.
        vace_in_dim: Input channels (default 257 = 2*128 latent + 1 mask).
        latent_channels: Audio patchified input channels, i.e. audio_patchify_proj.in_features (default 128).
        dim: Audio hidden dimension (must match audio_inner_dim = 2048).
        num_heads: Number of attention heads for audio (default 32).
        d_head: Dimension per head for audio (default 64).
        context_dim: Audio text embedding dimension (default 2048).
        ffn_mult: FFN expansion factor.
        norm_eps: Epsilon for RMSNorm.
        rope_type: RoPE type (match LTX-2).
        attention_function: Attention backend.
    """

    def __init__(
        self,
        vace_layers: Sequence[int] = DEFAULT_VACE_LAYERS,
        vace_in_dim: int = 257,
        latent_channels: int = 128,
        dim: int = 2048,
        num_heads: int = 32,
        d_head: int = 64,
        context_dim: int = 2048,
        ffn_mult: int = 4,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionFunction = AttentionFunction.DEFAULT,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self.vace_layers = tuple(vace_layers)
        self.vace_in_dim = vace_in_dim
        self.latent_channels = latent_channels
        self.dim = dim
        self.vace_layers_mapping = {layer_id: idx for idx, layer_id in enumerate(self.vace_layers)}
        self.timestep_ada_embed_count = adaln_embedding_coefficient(cross_attention_adaln)

        # Project audio VACE control channels to audio model dim.
        # vace_in_dim (257 = 2*128 + 1) -> dim (2048).
        self.vace_input_proj = nn.Linear(vace_in_dim, dim)

        # VACE transformer blocks (reuse VaceLTXBlock with audio dims)
        self.vace_blocks = nn.ModuleList([
            VaceLTXBlock(
                block_id=i,
                dim=dim,
                num_heads=num_heads,
                d_head=d_head,
                context_dim=context_dim,
                ffn_mult=ffn_mult,
                norm_eps=norm_eps,
                rope_type=rope_type,
                attention_function=attention_function,
                timestep_ada_embed_count=self.timestep_ada_embed_count,
            )
            for i in range(len(self.vace_layers))
        ])

        self.gradient_checkpointing = False

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following LTX-2 conventions."""
        for block in self.vace_blocks:
            nn.init.zeros_(block.scale_shift_table)
            nn.init.zeros_(block.after_proj.weight)
            nn.init.zeros_(block.after_proj.bias)
            if hasattr(block, "before_proj"):
                nn.init.zeros_(block.before_proj.bias)

    def initialize_input_proj_from_audio_patchify_proj(self, audio_patchify_proj: nn.Linear) -> None:
        """Initialize the audio VACE input projection from the base audio patchify weights.

        The latent branches (inactive/reactive) reuse the base audio patchify weights,
        while the mask channel starts from zero.
        """
        if audio_patchify_proj.weight.shape[1] != self.latent_channels:
            raise ValueError(
                f"audio_patchify_proj input dim mismatch: got {audio_patchify_proj.weight.shape[1]}, "
                f"expected latent_channels={self.latent_channels}"
            )

        with torch.no_grad():
            self.vace_input_proj.weight.zero_()

            base_weight = audio_patchify_proj.weight.detach()
            # Copy base weights for inactive latent channels
            self.vace_input_proj.weight[:, : self.latent_channels].copy_(base_weight)
            # Copy base weights for reactive latent channels
            self.vace_input_proj.weight[
                :, self.latent_channels : 2 * self.latent_channels
            ].copy_(base_weight)
            # Mask channel(s) remain zero-initialized

            if audio_patchify_proj.bias is not None and self.vace_input_proj.bias is not None:
                self.vace_input_proj.bias.copy_(audio_patchify_proj.bias.detach())
            elif self.vace_input_proj.bias is not None:
                self.vace_input_proj.bias.zero_()

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(
        self,
        x: torch.Tensor,
        vace_context: torch.Tensor,
        text_context: torch.Tensor,
        text_context_mask: torch.Tensor | None,
        timesteps: torch.Tensor,
        pe: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Compute audio VACE hints for injection into DiT audio blocks.

        Args:
            x: DiT audio hidden state (batch, audio_seq_len, dim).
            vace_context: Patchified audio control input (batch, audio_seq_len, vace_in_dim).
            text_context: Audio text embeddings (batch, num_tokens, context_dim).
            text_context_mask: Text attention mask or None.
            timesteps: Timestep embeddings for AdaLN.
            pe: RoPE positional embeddings.

        Returns:
            List of hint tensors, one per VACE layer. Each has shape (batch, audio_seq_len, dim).
        """
        # Project control input to model dimension
        c = self.vace_input_proj(vace_context)

        # Pad to match DiT audio sequence length if needed
        if c.shape[1] < x.shape[1]:
            pad = x.new_zeros(c.shape[0], x.shape[1] - c.shape[1], c.shape[2])
            c = torch.cat([c, pad], dim=1)
        elif c.shape[1] > x.shape[1]:
            c = c[:, : x.shape[1], :]

        # Process through VACE blocks
        for block in self.vace_blocks:
            if self.gradient_checkpointing and self.training:
                c = checkpoint.checkpoint(
                    block,
                    c, x, text_context, text_context_mask, timesteps, pe,
                    use_reentrant=False,
                )
            else:
                c = block(c, x, text_context, text_context_mask, timesteps, pe)

        # Extract hints (all skip connections, excluding the final running state)
        hints = list(torch.unbind(c))[:-1]
        return hints

    @property
    def num_hints(self) -> int:
        """Number of hint tensors produced (= number of VACE blocks)."""
        return len(self.vace_blocks)
