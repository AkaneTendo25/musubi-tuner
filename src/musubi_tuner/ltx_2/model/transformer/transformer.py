from dataclasses import dataclass, replace

import torch
from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from musubi_tuner.ltx_2.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import ensure_fp8_modules_on_device
from musubi_tuner.ltx_2.model.transformer.feed_forward import FeedForward
from musubi_tuner.ltx_2.model.transformer.rope import LTXRopeType
from musubi_tuner.ltx_2.model.transformer.transformer_args import TransformerArgs
from musubi_tuner.ltx_2.utils import rms_norm


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int


def _move_non_linear_params(module: torch.nn.Module, device: torch.device) -> None:
    """Move non-linear params/buffers to device; Linear weights are handled by offloader."""
    non_blocking = device.type != "cpu"
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            # Move bias and scale_weight if they exist and are on wrong device
            if submodule.bias is not None and submodule.bias.device != device:
                submodule.bias.data = submodule.bias.data.to(device, non_blocking=non_blocking)
            if hasattr(submodule, "scale_weight") and submodule.scale_weight is not None and submodule.scale_weight.device != device:
                submodule.scale_weight.data = submodule.scale_weight.data.to(device, non_blocking=non_blocking)
            continue
        
        # For non-linear modules, we only move its DIRECT parameters/buffers to avoid recursing into Linear children
        for param in submodule.parameters(recurse=False):
            if param.device != device:
                param.data = param.data.to(device, non_blocking=non_blocking)
        for buf in submodule.buffers(recurse=False):
            if buf.device != device:
                buf.data = buf.data.to(device, non_blocking=non_blocking)


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(6, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(6, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep, indices: slice, num_tokens: int = None
    ) -> tuple[torch.Tensor, ...]:
        """Get adaptive normalization values from scale_shift_table and timestep embeddings.
        
        Supports two modes:
        1. Regular mode: timestep is a tensor [B, num_tokens, dim]
        2. Optimized mode: timestep is tuple (unique_emb, inverse_indices_1d, orig_batch_size, orig_num_tokens)
           This mode computes ada values only on unique embeddings and expands using inverse indices,
           drastically reducing VRAM usage during inference when many tokens share the same timestep.
        
        Optimization source: Kijai's ComfyUI patch
        https://github.com/Comfy-Org/ComfyUI/commit/ac4daffd80cecbc56ee0e31f2b521114fa0f8e08
        """
        num_ada_params = scale_shift_table.shape[0]

        # Check if timestep is in optimized tuple format
        if isinstance(timestep, tuple) and len(timestep) == 4:
            unique_emb, inverse_indices_1d, orig_batch_size, orig_num_tokens = timestep

            # Compute ada values on unique embeddings only  
            unique_reshaped = unique_emb.reshape(len(unique_emb), num_ada_params, -1)[:, indices, :]
            table_values = scale_shift_table[indices].unsqueeze(0).to(device=unique_emb.device, dtype=unique_emb.dtype)
            unique_ada = (table_values + unique_reshaped).unbind(dim=1)

            # Expand each ada value using inverse indices
            ada_values = tuple(
                unique_val[inverse_indices_1d].view(orig_batch_size, orig_num_tokens, -1)
                for unique_val in unique_ada
            )
            return ada_values

        # Standard mode: timestep is a full tensor
        # Reshape and process embeddings
        timestep_reshaped = timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]

        table_values = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)

        # Expand timestep to match target sequence length if needed (for efficiency)
        if num_tokens is not None and timestep.shape[1] < num_tokens:
            repeats = num_tokens // timestep.shape[1]
            if repeats > 1:
                if repeats * timestep.shape[1] == num_tokens:
                    timestep_reshaped = torch.repeat_interleave(timestep_reshaped, repeats, dim=1)
                else:
                    timestep_reshaped = torch.repeat_interleave(timestep_reshaped, repeats + 1, dim=1)[:, :num_tokens]

        ada_values = (table_values + timestep_reshaped).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
        num_tokens: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, slice(None, None),
            num_tokens=num_tokens
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None),
            num_tokens=num_tokens
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        target_device = None
        if video is not None and isinstance(video.x, torch.Tensor):
            target_device = video.x.device
        elif audio is not None and isinstance(audio.x, torch.Tensor):
            target_device = audio.x.device
        
        if target_device is not None:
            _move_non_linear_params(self, target_device)
            ensure_fp8_modules_on_device(self, target_device)
        if video is not None and isinstance(video.x, torch.Tensor):
            batch_size = video.x.shape[0]
        elif audio is not None and isinstance(audio.x, torch.Tensor):
            batch_size = audio.x.shape[0]
        else:
            raise ValueError("Expected video or audio tensor inputs for transformer block")
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and audio.enabled and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and video.enabled and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3), num_tokens=vx.shape[1]
            )
            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                # AdaLN Structural Fix: Force modulation to happen in Float32 to prevent overflow (10^18 issue)
                norm_vx = (rms_norm(vx, eps=self.norm_eps).to(torch.float32) * (1 + vscale_msa.to(torch.float32)) + vshift_msa.to(torch.float32)).to(vx.dtype)
                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                vx = vx + self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa * v_mask

            vx = vx + self.attn2(rms_norm(vx, eps=self.norm_eps), context=video.context, mask=video.context_mask)

            del vshift_msa, vscale_msa, vgate_msa

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3), num_tokens=ax.shape[1]
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                # AdaLN Structural Fix
                norm_ax = (rms_norm(ax, eps=self.norm_eps).to(torch.float32) * (1 + ascale_msa.to(torch.float32)) + ashift_msa.to(torch.float32)).to(ax.dtype)
                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                ax = ax + self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa * a_mask

            ax = ax + self.audio_attn2(rms_norm(ax, eps=self.norm_eps), context=audio.context, mask=audio.context_mask)

            del ashift_msa, ascale_msa, agate_msa

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
                num_tokens=ax.shape[1],
            )

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
                num_tokens=vx.shape[1],
            )

            if run_a2v:
                # AdaLN Structural Fix
                vx_scaled = (vx_norm3.to(torch.float32) * (1 + scale_ca_video_hidden_states_a2v.to(torch.float32)) + shift_ca_video_hidden_states_a2v.to(torch.float32)).to(vx.dtype)
                ax_scaled = (ax_norm3.to(torch.float32) * (1 + scale_ca_audio_hidden_states_a2v.to(torch.float32)) + shift_ca_audio_hidden_states_a2v.to(torch.float32)).to(ax.dtype)
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * a2v_mask
                )

            if run_v2a:
                # AdaLN Structural Fix
                ax_scaled = (ax_norm3.to(torch.float32) * (1 + scale_ca_audio_hidden_states_v2a.to(torch.float32)) + shift_ca_audio_hidden_states_v2a.to(torch.float32)).to(ax.dtype)
                vx_scaled = (vx_norm3.to(torch.float32) * (1 + scale_ca_video_hidden_states_v2a.to(torch.float32)) + shift_ca_video_hidden_states_v2a.to(torch.float32)).to(vx.dtype)
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                    * v2a_mask
                )

            del gate_out_a2v, gate_out_v2a
            del (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
            )

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None), num_tokens=vx.shape[1]
            )
            # AdaLN Structural Fix
            vx_scaled = (rms_norm(vx, eps=self.norm_eps).to(torch.float32) * (1 + vscale_mlp.to(torch.float32)) + vshift_mlp.to(torch.float32)).to(vx.dtype)
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None), num_tokens=ax.shape[1]
            )
            # AdaLN Structural Fix
            ax_scaled = (rms_norm(ax, eps=self.norm_eps).to(torch.float32) * (1 + ascale_mlp.to(torch.float32)) + ashift_mlp.to(torch.float32)).to(ax.dtype)
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None
