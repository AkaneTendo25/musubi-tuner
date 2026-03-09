from dataclasses import dataclass, replace
import logging
import os

import torch
from musubi_tuner.ltx_2.model.transformer.adaln import AdaLayerNormSingle
from musubi_tuner.ltx_2.model.transformer.modality import Modality
from musubi_tuner.ltx_2.model.transformer.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)

logger = logging.getLogger(__name__)
_LOGGED_PREPROCESSOR_DEVICES = False


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor
    cross_positional_embeddings: torch.Tensor | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool
    prompt_timestep: torch.Tensor | None = None
    self_attention_mask: torch.Tensor | None = None


class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: torch.nn.Module | None,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.prompt_adaln = prompt_adaln

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        hidden_dtype: torch.dtype,
        adaln: AdaLayerNormSingle | None = None,
        use_unique_optimization: bool = True,
    ) -> tuple:
        """Prepare timestep embeddings.
        
        When use_unique_optimization is True (default), computes embeddings only for unique timestep
        values and returns a tuple format for lazy expansion. This drastically reduces VRAM during
        inference when many tokens share the same timestep.
        
        Returns:
            If optimized: tuple of (unique_emb, inverse_indices_1d, batch_size, num_tokens)
            If not optimized: regular tensor [batch_size, num_tokens, dim]
        """
        timestep_scaled = timestep * self.timestep_scale_multiplier
        adaln_module = self.adaln if adaln is None else adaln
        
        # Get original shape for reconstruction
        orig_shape = timestep_scaled.shape
        B = batch_size
        T = orig_shape[1] if len(orig_shape) > 1 else 1

        if use_unique_optimization:
            # Pre-compute embeddings only for unique timestep values
            unique_timesteps, inverse_indices_1d = torch.unique(timestep_scaled.flatten(), return_inverse=True)
            
            # Compute embeddings for unique timesteps only
            unique_emb, unique_embedded = adaln_module(
                unique_timesteps,
                hidden_dtype=hidden_dtype,
            )
            del unique_timesteps
            
            # Store as tuple, expand on-demand in get_ada_values
            timestep_out = (unique_emb, inverse_indices_1d, B, T)
            embedded_timestep_out = (unique_embedded, inverse_indices_1d, B, T)
            return timestep_out, embedded_timestep_out
        else:
            # Standard mode: compute full embeddings
            timestep_emb, embedded_timestep = adaln_module(
                timestep_scaled.flatten(),
                hidden_dtype=hidden_dtype,
            )
            
            # Second dimension is 1 or number of tokens (if timestep_per_token)
            timestep_emb = timestep_emb.view(batch_size, -1, timestep_emb.shape[-1])
            embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
            return timestep_emb, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare context for transformer blocks."""
        batch_size = x.shape[0]
        if self.caption_projection is not None:
            context = self.caption_projection(context)
        expected_hidden = int(x.shape[-1])
        actual_hidden = int(context.shape[-1])
        if actual_hidden != expected_hidden:
            raise ValueError(
                f"Context hidden size mismatch: got {actual_hidden}, expected {expected_hidden}. "
                "Check cached text embeddings and modality selection."
            )
        context = context.reshape(batch_size, -1, expected_hidden)

        # Validate common 2D token masks early to avoid downstream attention errors.
        if attention_mask is not None and attention_mask.dim() == 2:
            if int(attention_mask.shape[0]) != batch_size:
                raise ValueError(
                    "Context mask batch mismatch: "
                    f"got {int(attention_mask.shape[0])}, expected {batch_size}."
                )
            if int(attention_mask.shape[-1]) != int(context.shape[1]):
                raise ValueError(
                    "Context mask length mismatch: "
                    f"got {int(attention_mask.shape[-1])}, expected {int(context.shape[1])}."
                )

        return context, attention_mask

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask
        # If all tokens are valid (no-op mask), return None so SDPA/FlashAttention
        # can use the fast maskless kernel path (~20-25% speedup on cross-attention).
        # Inspired by https://github.com/Nerogar/OneTrainer/pull/1109
        if os.getenv("LTX2_SKIP_NOOP_ATTN_MASK", "0") == "1":
            if attention_mask.dtype == torch.bool and torch.all(attention_mask):
                return None
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.to(torch.int64)

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_self_attention_mask(
        self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        finfo = torch.finfo(x_dtype)
        eps = finfo.tiny
        bias = torch.full_like(attention_mask, finfo.min, dtype=x_dtype)
        positive = attention_mask > 0
        if positive.any():
            bias[positive] = torch.log(attention_mask[positive].clamp(min=eps)).to(x_dtype)
        return bias.unsqueeze(1)

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def _ensure_modules_on_device(self, device: torch.device) -> None:
        if self.patchify_proj.weight.device != device:
            self.patchify_proj.to(device)
        caption_device = "n/a"
        if self.caption_projection is not None:
            caption_param = next(self.caption_projection.parameters(), None)
            if caption_param is not None:
                caption_device = caption_param.device
                if caption_param.device != device:
                    self.caption_projection.to(device)
                    caption_param = next(self.caption_projection.parameters(), None)
                    caption_device = caption_param.device if caption_param is not None else "n/a"
        adaln_param = next(self.adaln.parameters(), None)
        if adaln_param is not None and adaln_param.device != device:
            self.adaln.to(device)
        global _LOGGED_PREPROCESSOR_DEVICES
        if not _LOGGED_PREPROCESSOR_DEVICES:
            adaln_param = next(self.adaln.parameters(), None)
            logger.info(
                "LTX-2 preprocessor devices: patchify_proj=%s caption_projection=%s adaln=%s",
                self.patchify_proj.weight.device,
                caption_device,
                adaln_param.device if adaln_param is not None else "n/a",
            )
            _LOGGED_PREPROCESSOR_DEVICES = True

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,  # noqa: ARG002
    ) -> TransformerArgs:
        self._ensure_modules_on_device(modality.latent.device)
        x = self.patchify_proj(modality.latent)
        timestep, embedded_timestep = self._prepare_timestep(modality.timesteps, x.shape[0], modality.latent.dtype)
        prompt_timestep = None
        if self.prompt_adaln is not None and getattr(modality, "sigma", None) is not None:
            prompt_timestep, _ = self._prepare_timestep(
                modality.sigma,
                x.shape[0],
                modality.latent.dtype,
                adaln=self.prompt_adaln,
                use_unique_optimization=False,
            )
        context, attention_mask = self._prepare_context(modality.context, x, modality.context_mask)
        attention_mask = self._prepare_attention_mask(attention_mask, modality.latent.dtype)
        self_attention_mask = self._prepare_self_attention_mask(modality.attention_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )
        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
            self_attention_mask=self_attention_mask,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: torch.nn.Module | None,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            prompt_adaln=prompt_adaln,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
    ) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        cross_timestep = modality.timesteps
        if cross_modality is not None and getattr(cross_modality, "sigma", None) is not None:
            sigma = cross_modality.sigma
            if sigma.ndim == 1:
                sigma = sigma.view(modality.timesteps.shape[0], 1)
            cross_timestep = sigma.view(modality.timesteps.shape[0], 1).expand_as(modality.timesteps)

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=cross_timestep,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross attention timestep embeddings."""
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep
