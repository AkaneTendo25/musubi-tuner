from dataclasses import dataclass, replace, fields
from typing import Any

import torch
import torch.utils.checkpoint as checkpoint

from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper
from musubi_tuner.modules.custom_offloading_utils import weighs_to_device
from musubi_tuner.modules.block_level_checkpointing import block_checkpoint
from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from musubi_tuner.ltx_2.model.transformer.adaln import AdaLayerNormSingle
from musubi_tuner.ltx_2.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import ensure_fp8_modules_on_device
from musubi_tuner.ltx_2.model.transformer.feed_forward import FeedForward
from musubi_tuner.ltx_2.model.transformer.rope import LTXRopeType
from musubi_tuner.ltx_2.model.transformer.transformer_args import TransformerArgs
from musubi_tuner.ltx_2.utils import rms_norm



def _unpack_transformer_args(args: TransformerArgs | None) -> tuple[list[Any], bool]:
    # Returns (values, is_none). Unpacks all fields.
    if args is None:
        return [], True
    return [getattr(args, f.name) for f in fields(args)], False


def _reconstruct_transformer_args(values: list[Any], is_none: bool) -> TransformerArgs | None:
    if is_none:
        return None
    field_names = [f.name for f in fields(TransformerArgs)]
    kwargs = dict(zip(field_names, values))
    return TransformerArgs(**kwargs)


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int


def _move_non_linear_params(module: torch.nn.Module, device: torch.device, skip_trainable: bool = True) -> None:
    """Move non-linear params/buffers to device; Linear weights are handled by offloader.
    
    Args:
        module: Module to process
        device: Target device
        skip_trainable: If True AND target is CPU, skip parameters with requires_grad=True
    """
    non_blocking = device.type != "cpu"
    # Only skip trainable parameters when moving TO CPU (offloading), not when loading TO GPU
    should_skip_trainable = skip_trainable and device.type == "cpu"
    
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            # Move bias and scale_weight if they exist and are on wrong device
            if submodule.bias is not None and submodule.bias.device != device:
                if not (should_skip_trainable and submodule.bias.requires_grad):
                    submodule.bias.data = submodule.bias.data.to(device, non_blocking=non_blocking)
            if hasattr(submodule, "scale_weight") and submodule.scale_weight is not None and submodule.scale_weight.device != device:
                if not (should_skip_trainable and submodule.scale_weight.requires_grad):
                    submodule.scale_weight.data = submodule.scale_weight.data.to(device, non_blocking=non_blocking)
            continue
        
        # For non-linear modules, we only move its DIRECT parameters/buffers to avoid recursing into Linear children
        for param in submodule.parameters(recurse=False):
            if param.device != device:
                if not (should_skip_trainable and param.requires_grad):
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

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False, weight_cpu_offloading: bool = False) -> None:
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        self.weight_cpu_offloading = weight_cpu_offloading

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

    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if self.training and self.gradient_checkpointing:
            # Define load and offload functions for this block
            def load_weights(b, d):
                weighs_to_device(b, d)
                # Move non-linear params (RMSNorm, etc.) and FP8/LoRA modules
                _move_non_linear_params(b, d)
                ensure_fp8_modules_on_device(b, d)
                # Manually move scale_shift tables as weighs_to_device only targets Linear layers
                # and these are Parameters of the block itself.
                for attr in [
                    "scale_shift_table",
                    "audio_scale_shift_table",
                    "scale_shift_table_a2v_ca_audio",
                    "scale_shift_table_a2v_ca_video",
                ]:
                    p = getattr(b, attr, None)
                    if p is not None:
                        # Skip if already on device to avoid overhead
                        if p.device != d:
                            p.data = p.data.to(d, non_blocking=True)
            
            def offload_weights(b, d):
                cpu_device = torch.device("cpu")
                # When offloading to CPU, we should also move these tables back
                # Reuse the same logic but targeting CPU (d)
                weighs_to_device(b, d)
                for attr in [
                    "scale_shift_table",
                    "audio_scale_shift_table",
                    "scale_shift_table_a2v_ca_audio",
                    "scale_shift_table_a2v_ca_video",
                ]:
                    p = getattr(b, attr, None)
                    if p is not None:
                        if p.device != d:
                            p.data = p.data.to(d, non_blocking=True)
                _move_non_linear_params(b, cpu_device)
                ensure_fp8_modules_on_device(b, cpu_device)

            # Prepare arguments for checkpointing (both standard and block-level need flattened tensors)
            video_vals, video_none = _unpack_transformer_args(video)
            audio_vals, audio_none = _unpack_transformer_args(audio)
            vid_len = len(video_vals)

            def checkpoint_wrapper(*inputs):
                v_vals = list(inputs[:vid_len])
                a_vals = list(inputs[vid_len:])
                v_args = _reconstruct_transformer_args(v_vals, video_none)
                a_args = _reconstruct_transformer_args(a_vals, audio_none)
                return self._forward(v_args, a_args, perturbations)

            flat_inputs = tuple(video_vals + audio_vals)

            if self.weight_cpu_offloading or self.activation_cpu_offloading:
                # Determine offloading hooks based on configuration
                load_fn = load_weights if self.weight_cpu_offloading else None
                offload_fn = offload_weights if self.weight_cpu_offloading else None

                # Use custom block checkpointing
                # With our updated block_checkpoint, this will:
                # 1. Offload all tensor inputs in flat_inputs to CPU
                # 2. Re-load them to GPU during backward
                # 3. Handle weight offloading hooks if provided
                # 4. Return TENSORS (because autograd strips objects)
                
                outputs = block_checkpoint(
                    checkpoint_wrapper,
                    *flat_inputs,
                    block=self,
                    load_fn=load_fn,
                    offload_fn=offload_fn,
                )
                
                # Reconstruct TransformerArgs from returned tensors
                # block_checkpoint/autograd returns a tuple of tensors.
                # output structure corresponds to what checkpoint_wrapper returns.
                # wrapper returns self._forward -> (video_out, audio_out)
                # each is TransformerArgs which has .x tensor.
                # So outputs will correspond to (video_out.x, audio_out.x) 
                
                # Note: BlockCheckpointFunction logic flattens outputs. 
                # If _forward returns (v, a), and v.x is tensor, a.x is tensor/None...
                # We need to ensure we map them back correctly.
                
                # Let's peek at expected returns of _forward: tuple[TransformerArgs|None, TransformerArgs|None]
                # If audio is None, we get (v, None).
                
                # Check if outputs are already TransformerArgs (No-Grad path returns objects directly)
                is_obj = False
                if len(outputs) > 0 and isinstance(outputs[0], TransformerArgs):
                    is_obj = True
                elif len(outputs) > 1 and isinstance(outputs[1], TransformerArgs):
                    is_obj = True
                
                if is_obj:
                    # In no-grad mode, block_checkpoint returns the objects directly
                    return outputs

                # Re-wrapping logic for Tensors (Grad path):
                res_v_x = outputs[0] if len(outputs) > 0 else None
                res_a_x = outputs[1] if len(outputs) > 1 else None
                
                out_video = None
                if video is not None:
                     # Use dataclasses.replace to create new object with updated x
                     if isinstance(res_v_x, torch.Tensor):
                         out_video = replace(video, x=res_v_x)
                     else:
                         out_video = video
                
                out_audio = None
                if audio is not None:
                     if res_a_x is not None and isinstance(res_a_x, torch.Tensor):
                         out_audio = replace(audio, x=res_a_x)
                     else:
                         out_audio = audio
                      
                return out_video, out_audio
            else:
                # Standard gradient checkpointing
                return checkpoint.checkpoint(checkpoint_wrapper, *flat_inputs, use_reentrant=False, determinism_check="none")
        
        return self._forward(video, audio, perturbations)

    def _forward(  # noqa: PLR0915
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

        # Offload weights to CPU at the end of _forward.
        # This runs during forward pass. During backward, the backward hook handles offloading.
        if self.activation_cpu_offloading:
            cpu_device = torch.device("cpu")
            weighs_to_device(self, cpu_device)
            _move_non_linear_params(self, cpu_device)
            ensure_fp8_modules_on_device(self, cpu_device)

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None
