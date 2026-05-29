from dataclasses import dataclass, replace, fields
from typing import Any
import logging
import os

import torch
import torch.utils.checkpoint as checkpoint

from musubi_tuner.ltx_2.model.ltx2_custom_offloading_utils import weighs_to_device
from musubi_tuner.ltx_2.model.transformer.block_level_checkpointing import block_checkpoint
from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from musubi_tuner.ltx_2.model.transformer.adaln import adaln_embedding_coefficient
from musubi_tuner.ltx_2.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import ensure_fp8_modules_on_device
from musubi_tuner.ltx_2.model.transformer.feed_forward import FeedForward
from musubi_tuner.ltx_2.model.transformer.rope import LTXRopeType
from musubi_tuner.ltx_2.model.transformer.transformer_args import TransformerArgs
from musubi_tuner.ltx_2.utils import rms_norm

logger = logging.getLogger(__name__)

# Resolved once at import so torch.compile can treat it as a compile-time
# constant. When False (default; LTX2_ATTN_FP32_RETRY unset), the data-dependent
# `torch.isfinite(out).all()` branch in _run_attn_with_optional_fp32_retry is
# statically pruned by Dynamo instead of forcing a graph break in every block.
# Reading it per-forward via os.getenv prevented constant folding.
_ATTN_RETRY_FP32 = os.getenv("LTX2_ATTN_FP32_RETRY", "0") == "1"

# Thread-local storage for tracking last loaded swapped block during backward
import threading

_swap_backward_state = threading.local()


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
    # NOTE: do not replace fields() with a precomputed constant here. Tested: it
    # eliminates this graph break but causes torch.compile to trace more of the
    # grad-sensitive block, which then recompiles on the slider's per-step
    # grad_mode flips (~64 recompiles, ~3.9s/it vs ~2.0s/it). The break is
    # benign / mildly beneficial for the compiled path.


def _repack_block_checkpoint_outputs(
    video: TransformerArgs | None,
    audio: TransformerArgs | None,
    outputs: tuple[Any, ...],
) -> tuple[TransformerArgs | None, TransformerArgs | None]:
    """Restore block checkpoint tensor outputs to their modality slots."""
    if len(outputs) > 0 and isinstance(outputs[0], TransformerArgs):
        return outputs
    if len(outputs) > 1 and isinstance(outputs[1], TransformerArgs):
        return outputs

    output_idx = 0
    out_video = None
    if video is not None:
        res_v_x = outputs[output_idx] if output_idx < len(outputs) else None
        output_idx += 1
        out_video = replace(video, x=res_v_x) if isinstance(res_v_x, torch.Tensor) else video

    out_audio = None
    if audio is not None:
        res_a_x = outputs[output_idx] if output_idx < len(outputs) else None
        output_idx += 1
        out_audio = replace(audio, x=res_a_x) if isinstance(res_a_x, torch.Tensor) else audio

    return out_video, out_audio


def _run_attn_with_optional_fp32_retry(
    attn_module: torch.nn.Module,
    x_in: torch.Tensor,
    *,
    context: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    pe: torch.Tensor | None = None,
    k_pe: torch.Tensor | None = None,
    force_fp32: bool = False,
    force_pytorch: bool = False,
    attn_retry_fp32: bool = False,
    block_idx: int | None = None,
) -> torch.Tensor:
    """Run attention and fall back to PyTorch SDPA in fp32 if the output is non-finite.

    The fallback intentionally switches away from the current backend. Retrying the
    same FlashAttention kernel in fp32 can still leave the graph in a bad state for
    backward when swap/fp8/checkpointing are active.
    """

    # Fast path: no fp32 retry and no pytorch-backend override. This avoids both
    # the data-dependent branch (torch.isfinite(...).all()) and the attribute
    # mutation below, both of which force a torch.compile graph break in every
    # block. attn_retry_fp32 defaults False (LTX2_ATTN_FP32_RETRY unset), so this
    # is the normal training path. Behavior is identical to the general path when
    # both flags are off.
    if not attn_retry_fp32 and not force_pytorch:
        if force_fp32:
            x = x_in.to(torch.float32)
            ctx = context.to(torch.float32) if isinstance(context, torch.Tensor) else None
            pe_local = pe.to(torch.float32) if isinstance(pe, torch.Tensor) else None
            k_pe_local = k_pe.to(torch.float32) if isinstance(k_pe, torch.Tensor) else None
            return attn_module(x, context=ctx, mask=mask, pe=pe_local, k_pe=k_pe_local).to(dtype=x_in.dtype)
        return attn_module(x_in, context=context, mask=mask, pe=pe, k_pe=k_pe)

    original_fn = getattr(attn_module, "attention_function", None)

    def _call(*, use_fp32: bool) -> torch.Tensor:
        if use_fp32:
            x = x_in.to(torch.float32)
            ctx = context.to(torch.float32) if isinstance(context, torch.Tensor) else None
            pe_local = pe.to(torch.float32) if isinstance(pe, torch.Tensor) else None
            k_pe_local = k_pe.to(torch.float32) if isinstance(k_pe, torch.Tensor) else None
            out_local = attn_module(x, context=ctx, mask=mask, pe=pe_local, k_pe=k_pe_local)
            return out_local.to(dtype=x_in.dtype)

        return attn_module(x_in, context=context, mask=mask, pe=pe, k_pe=k_pe)

    try:
        if force_pytorch and original_fn is not None:
            attn_module.attention_function = AttentionFunction.PYTORCH.to_callable()

        out = _call(use_fp32=force_fp32)
        if not attn_retry_fp32 or torch.isfinite(out).all():
            return out

        logger.warning("LTX-2 attn retry in fp32 for block %s", block_idx)
        if original_fn is not None:
            attn_module.attention_function = AttentionFunction.PYTORCH.to_callable()
        return _call(use_fp32=True)
    finally:
        if original_fn is not None:
            attn_module.attention_function = original_fn


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


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
            if (
                hasattr(submodule, "scale_weight")
                and submodule.scale_weight is not None
                and submodule.scale_weight.device != device
            ):
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
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = torch.nn.Parameter(
                torch.empty(adaln_embedding_coefficient(video.cross_attention_adaln), video.dim)
            )

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = torch.nn.Parameter(
                torch.empty(adaln_embedding_coefficient(audio.cross_attention_adaln), audio.dim)
            )

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
                apply_gated_attention=video.apply_gated_attention,
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
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.norm_eps = norm_eps
        self.cross_attention_adaln = bool(
            (video is not None and video.cross_attention_adaln) or (audio is not None and audio.cross_attention_adaln)
        )
        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.weight_cpu_offloading = False
        self.use_pinned_memory = False

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
                unique_val[inverse_indices_1d].view(orig_batch_size, orig_num_tokens, -1) for unique_val in unique_ada
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
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
            slice(None, None),
            num_tokens=num_tokens,
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None), num_tokens=num_tokens
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep,
        prompt_timestep,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(
                scale_shift_table, x.shape[0], timestep, slice(6, 9), num_tokens=x.shape[1]
            )
            if prompt_scale_shift_table is not None and prompt_timestep is not None:
                return apply_cross_attention_adaln(
                    x,
                    context,
                    attn,
                    shift_q,
                    scale_q,
                    gate,
                    prompt_scale_shift_table,
                    prompt_timestep,
                    context_mask,
                    self.norm_eps,
                )
            attn_input = (
                rms_norm(x, eps=self.norm_eps).to(torch.float32) * (1 + scale_q.to(torch.float32)) + shift_q.to(torch.float32)
            ).to(x.dtype)
            return attn(attn_input, context=context, mask=context_mask) * gate
        return attn(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if self.training and self.gradient_checkpointing:
            # Define load and offload functions for this block (proxies to class methods)
            load_weights = self._load_weights
            offload_weights = self._offload_weights

            # Prepare arguments for checkpointing (both standard and block-level need flattened tensors)
            video_vals, video_none = _unpack_transformer_args(video)
            audio_vals, audio_none = _unpack_transformer_args(audio)
            vid_len = len(video_vals)

            def checkpoint_wrapper(*inputs):
                v_vals = list(inputs[:vid_len])
                a_vals = list(inputs[vid_len:])

                # Determine target device from inputs
                target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                for inp in inputs:
                    if isinstance(inp, torch.Tensor) and inp.device.type == "cuda":
                        target_device = inp.device
                        break

                # For swapped blocks, handle loading during backward recomputation
                # Key insight: During FORWARD, offloader loads block to GPU before checkpoint_wrapper runs
                # During BACKWARD recomputation, block is on CPU (was unloaded after forward)
                # So we detect backward by checking if block is on CPU
                if getattr(self, "swap_weight_offload", False):
                    first_param = next(self.parameters(), None)
                    block_on_cpu = first_param is not None and first_param.device.type == "cpu"

                    if block_on_cpu:
                        # Block is on CPU → we're in backward recomputation
                        # Unload previous block before loading current to prevent VRAM accumulation
                        prev_block = getattr(_swap_backward_state, "last_loaded_block", None)
                        if prev_block is not None and prev_block is not self:
                            prev_block.to("cpu")
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()

                        # Load this block to GPU
                        self.to(target_device)

                        # Track for unloading on next iteration
                        _swap_backward_state.last_loaded_block = self
                        _swap_backward_state.last_loaded_idx = self.idx
                    else:
                        # Block is on GPU → we're in forward pass (offloader loaded it)
                        # Reset backward state to prevent stale data affecting next backward
                        _swap_backward_state.last_loaded_block = None
                        _swap_backward_state.last_loaded_idx = -1

                    ensure_fp8_modules_on_device(self, target_device)

                # For non-swapped (permanent) blocks: unload any pending swapped block from backward
                # This handles transition from swapped blocks (18) to permanent blocks (17→0)
                elif getattr(_swap_backward_state, "last_loaded_block", None) is not None:
                    prev_block = _swap_backward_state.last_loaded_block
                    prev_block.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _swap_backward_state.last_loaded_block = None
                    _swap_backward_state.last_loaded_idx = -1

                # When activation_cpu_offloading is enabled but weight_cpu_offloading is not,
                # inputs may arrive on CPU (from model-level CPU offloading). Move them to GPU
                # before running the forward pass, since LoRA and other trainable weights are on GPU.
                if self.activation_cpu_offloading and not self.weight_cpu_offloading:

                    def _move_to_device(val):
                        """Recursively move tensors to target device, handling tuples."""
                        if isinstance(val, torch.Tensor):
                            return val.to(target_device) if val.device.type == "cpu" else val
                        elif isinstance(val, tuple):
                            return tuple(_move_to_device(v) for v in val)
                        elif isinstance(val, list):
                            return [_move_to_device(v) for v in val]
                        return val

                    v_vals = [_move_to_device(v) for v in v_vals]
                    a_vals = [_move_to_device(a) for a in a_vals]

                v_args = _reconstruct_transformer_args(v_vals, video_none)
                a_args = _reconstruct_transformer_args(a_vals, audio_none)
                return self._forward(v_args, a_args, perturbations)

            flat_inputs = tuple(video_vals + audio_vals)

            # Use block-level checkpointing ONLY when explicit CPU weight offloading is enabled
            # (i.e., --blockwise_checkpointing flag). Block swap is handled separately.
            if self.weight_cpu_offloading:
                # Determine offloading hooks based on configuration
                load_fn = load_weights
                offload_fn = offload_weights

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
                return _repack_block_checkpoint_outputs(video, audio, outputs)
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
        sublayer_diag = os.getenv("LTX2_NAN_SUBLAYER_DIAG", "0") == "1"
        v2a_diag = os.getenv("LTX2_V2A_DIAG", "0") == "1"
        attn_retry_fp32 = _ATTN_RETRY_FP32
        # Clamp FFN outputs to prevent bf16 overflow (max ~65504).
        # Set LTX2_FFN_CLAMP=60000 to enable. Default: disabled (0).
        ffn_clamp = float(os.getenv("LTX2_FFN_CLAMP", "0"))
        force_pytorch_cross_attn = os.getenv("LTX2_FORCE_PYTORCH_CROSS_ATTN", "0") == "1" or getattr(
            self, "_force_pytorch_cross_attn", False
        )
        force_fp32_cross_attn = os.getenv("LTX2_CROSS_ATTN_FP32", "0") == "1" or getattr(self, "_force_fp32_cross_attn", False)
        force_pytorch_audio_ctx = os.getenv("LTX2_FORCE_PYTORCH_AUDIO_CTX_ATTN", "0") == "1" or getattr(
            self, "_force_pytorch_audio_ctx_attn", False
        )
        force_fp32_audio_ctx = os.getenv("LTX2_AUDIO_CTX_ATTN_FP32", "0") == "1" or getattr(
            self, "_force_fp32_audio_ctx_attn", False
        )

        def _check_finite_local(tag: str, tensor: torch.Tensor | None) -> None:
            if not sublayer_diag or tensor is None:
                return
            if not torch.isfinite(tensor).all():
                logger.error("Non-finite detected: %s in block %s", tag, self.idx)
                return

        def _log_stats(tag: str, tensor: torch.Tensor | None) -> None:
            if not v2a_diag or tensor is None:
                return
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                logger.info("V2A_DIAG %s: empty", tag)
                return
            t = tensor.detach()
            try:
                tmin = float(t.min().item())
                tmax = float(t.max().item())
                tmean = float(t.mean().item())
                tstd = float(t.std().item())
            except Exception:
                tmin = tmax = tmean = tstd = float("nan")
            finite = bool(torch.isfinite(t).all().item())
            logger.info(
                "V2A_DIAG %s: shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f finite=%s",
                tag,
                tuple(t.shape),
                tmin,
                tmax,
                tmean,
                tstd,
                finite,
            )

        def _attn_with_retry(
            attn_module: torch.nn.Module,
            x_in: torch.Tensor,
            *,
            context: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
            pe: torch.Tensor | None = None,
            k_pe: torch.Tensor | None = None,
            force_fp32: bool = False,
            force_pytorch: bool = False,
        ) -> torch.Tensor:
            return _run_attn_with_optional_fp32_retry(
                attn_module,
                x_in,
                context=context,
                mask=mask,
                pe=pe,
                k_pe=k_pe,
                force_fp32=force_fp32,
                force_pytorch=force_pytorch,
                attn_retry_fp32=attn_retry_fp32,
                block_idx=self.idx,
            )

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
            _check_finite_local("video_in", vx)
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3), num_tokens=vx.shape[1]
            )
            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                # AdaLN Structural Fix: Force modulation to happen in Float32 to prevent overflow (10^18 issue)
                norm_vx = (
                    rms_norm(vx, eps=self.norm_eps).to(torch.float32) * (1 + vscale_msa.to(torch.float32))
                    + vshift_msa.to(torch.float32)
                ).to(vx.dtype)
                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                attn1_out = _attn_with_retry(
                    self.attn1,
                    norm_vx,
                    pe=video.positional_embeddings,
                    mask=video.self_attention_mask,
                )
                vx = vx + attn1_out * vgate_msa * v_mask
                _check_finite_local("video_after_attn1", vx)

            vx = vx + self._apply_text_cross_attention(
                vx,
                video.context,
                lambda q, context=None, mask=None: _attn_with_retry(
                    self.attn2,
                    q,
                    context=context,
                    mask=mask,
                ),
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
            )
            _check_finite_local("video_after_attn2", vx)

            del vshift_msa, vscale_msa, vgate_msa

        if run_ax:
            _check_finite_local("audio_in", ax)
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3), num_tokens=ax.shape[1]
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                # AdaLN Structural Fix
                norm_ax = (
                    rms_norm(ax, eps=self.norm_eps).to(torch.float32) * (1 + ascale_msa.to(torch.float32))
                    + ashift_msa.to(torch.float32)
                ).to(ax.dtype)
                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                audio_attn1_out = _attn_with_retry(
                    self.audio_attn1,
                    norm_ax,
                    pe=audio.positional_embeddings,
                    mask=audio.self_attention_mask,
                )
                ax = ax + audio_attn1_out * agate_msa * a_mask
                _check_finite_local("audio_after_attn1", ax)

            ax = ax + self._apply_text_cross_attention(
                ax,
                audio.context,
                lambda q, context=None, mask=None: _attn_with_retry(
                    self.audio_attn2,
                    q,
                    context=context,
                    mask=mask,
                    force_fp32=force_fp32_audio_ctx,
                    force_pytorch=force_pytorch_audio_ctx,
                ),
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
            )
            _check_finite_local("audio_after_attn2", ax)

            del ashift_msa, ascale_msa, agate_msa

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            # DCR: per-sample gradient detachment (applied after AdaLN, see below)
            dcr_audio_mask = audio.dcr_detach_mask if audio is not None else None
            dcr_video_mask = video.dcr_detach_mask if video is not None else None

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

            if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx):
                # AdaLN Structural Fix
                vx_scaled = (
                    vx_norm3.to(torch.float32) * (1 + scale_ca_video_hidden_states_a2v.to(torch.float32))
                    + shift_ca_video_hidden_states_a2v.to(torch.float32)
                ).to(vx.dtype)
                ax_scaled = (
                    ax_norm3.to(torch.float32) * (1 + scale_ca_audio_hidden_states_a2v.to(torch.float32))
                    + shift_ca_audio_hidden_states_a2v.to(torch.float32)
                ).to(ax.dtype)
                # DCR: detach audio context AFTER AdaLN so scale/shift params also don't get noisy gradients
                if dcr_audio_mask is not None:
                    ax_scaled = ax_scaled * dcr_audio_mask + ax_scaled.detach() * (1 - dcr_audio_mask)
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                vx = vx + (
                    _attn_with_retry(
                        self.audio_to_video_attn,
                        vx_scaled,
                        context=ax_scaled,
                        mask=video.a2v_cross_attention_mask if video is not None else None,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                        force_fp32=force_fp32_cross_attn,
                        force_pytorch=force_pytorch_cross_attn,
                    )
                    * gate_out_a2v
                    * a2v_mask
                )
                _check_finite_local("video_after_a2v", vx)

            if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx):
                # AdaLN Structural Fix
                ax_scaled = (
                    ax_norm3.to(torch.float32) * (1 + scale_ca_audio_hidden_states_v2a.to(torch.float32))
                    + shift_ca_audio_hidden_states_v2a.to(torch.float32)
                ).to(ax.dtype)
                vx_scaled = (
                    vx_norm3.to(torch.float32) * (1 + scale_ca_video_hidden_states_v2a.to(torch.float32))
                    + shift_ca_video_hidden_states_v2a.to(torch.float32)
                ).to(vx.dtype)
                # DCR: detach video context AFTER AdaLN
                if dcr_video_mask is not None:
                    vx_scaled = vx_scaled * dcr_video_mask + vx_scaled.detach() * (1 - dcr_video_mask)
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                _log_stats("v2a_ax_scaled", ax_scaled)
                _log_stats("v2a_vx_scaled", vx_scaled)
                _log_stats("v2a_gate_out", gate_out_v2a)
                ax = ax + (
                    _attn_with_retry(
                        self.video_to_audio_attn,
                        ax_scaled,
                        context=vx_scaled,
                        mask=audio.v2a_cross_attention_mask if audio is not None else None,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                        force_fp32=force_fp32_cross_attn,
                        force_pytorch=force_pytorch_cross_attn,
                    )
                    * gate_out_v2a
                    * v2a_mask
                )
                _log_stats("v2a_out", ax)
                _check_finite_local("audio_after_v2a", ax)

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
            mlp_slice = slice(3, 6) if self.cross_attention_adaln else slice(3, None)
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, mlp_slice, num_tokens=vx.shape[1]
            )
            # AdaLN Structural Fix
            vx_scaled = (
                rms_norm(vx, eps=self.norm_eps).to(torch.float32) * (1 + vscale_mlp.to(torch.float32))
                + vshift_mlp.to(torch.float32)
            ).to(vx.dtype)
            ff_out = self.ff(vx_scaled) * vgate_mlp
            if ffn_clamp > 0:
                ff_out = ff_out.clamp(-ffn_clamp, ffn_clamp)
            vx = vx + ff_out
            _check_finite_local("video_after_ff", vx)

            del vshift_mlp, vscale_mlp, vgate_mlp

        if run_ax:
            mlp_slice = slice(3, 6) if self.cross_attention_adaln else slice(3, None)
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, mlp_slice, num_tokens=ax.shape[1]
            )
            # AdaLN Structural Fix
            ax_scaled = (
                rms_norm(ax, eps=self.norm_eps).to(torch.float32) * (1 + ascale_mlp.to(torch.float32))
                + ashift_mlp.to(torch.float32)
            ).to(ax.dtype)
            audio_ff_out = self.audio_ff(ax_scaled) * agate_mlp
            if ffn_clamp > 0:
                audio_ff_out = audio_ff_out.clamp(-ffn_clamp, ffn_clamp)
            ax = ax + audio_ff_out
            _check_finite_local("audio_after_ff", ax)

            del ashift_mlp, ascale_mlp, agate_mlp

        # Offload weights to CPU at the end of _forward.
        # This runs during forward pass. During backward, the backward hook handles offloading.
        if self.activation_cpu_offloading:
            cpu_device = torch.device("cpu")
            use_pinned = self.use_pinned_memory and os.getenv("LTX2_SWAP_PINNED", "1") == "1"
            weighs_to_device(self, cpu_device, use_pinned=use_pinned)
            _move_non_linear_params(self, cpu_device)
            ensure_fp8_modules_on_device(self, cpu_device)

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None

    def _load_weights(self, b: torch.nn.Module, d: torch.device) -> None:
        use_pinned = self.use_pinned_memory and os.getenv("LTX2_SWAP_PINNED", "1") == "1"
        weighs_to_device(b, d, use_pinned=use_pinned)
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
            "prompt_scale_shift_table",
            "audio_prompt_scale_shift_table",
        ]:
            p = getattr(b, attr, None)
            if p is not None:
                # Skip if already on device to avoid overhead
                if p.device != d:
                    p.data = p.data.to(d, non_blocking=True)

    def _offload_weights(self, b: torch.nn.Module, d: torch.device) -> None:
        cpu_device = torch.device("cpu")
        # When offloading to CPU, we should also move these tables back
        # Reuse the same logic but targeting CPU (d)
        use_pinned = self.use_pinned_memory and os.getenv("LTX2_SWAP_PINNED", "1") == "1"
        weighs_to_device(b, d, use_pinned=use_pinned)
        for attr in [
            "scale_shift_table",
            "audio_scale_shift_table",
            "scale_shift_table_a2v_ca_audio",
            "scale_shift_table_a2v_ca_video",
            "prompt_scale_shift_table",
            "audio_prompt_scale_shift_table",
        ]:
            p = getattr(b, attr, None)
            if p is not None:
                if p.device != d:
                    if d.type == "cpu":
                        p.data = p.data.to(d, non_blocking=True)
                        if use_pinned:
                            p.data = p.data.pin_memory()
                    else:
                        p.data = p.data.to(d, non_blocking=True)
        _move_non_linear_params(b, cpu_device)
        ensure_fp8_modules_on_device(b, cpu_device)


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    prompt_adaln_fp32 = os.getenv("LTX2_PROMPT_ADALN_FP32", "1") == "1"
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=torch.float32 if prompt_adaln_fp32 else x.dtype)
        + prompt_timestep.to(
            device=x.device,
            dtype=torch.float32 if prompt_adaln_fp32 else x.dtype,
        ).reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    if prompt_adaln_fp32:
        # Keep prompt AdaLN modulation in float32 to match the stability fix
        # used in the self-attention / FF / output AdaLN paths.
        attn_input = (rms_norm(x, eps=norm_eps).to(torch.float32) * (1 + q_scale.to(torch.float32)) + q_shift.to(torch.float32)).to(
            x.dtype
        )
        encoder_hidden_states = (context.to(torch.float32) * (1 + scale_kv) + shift_kv).to(context.dtype)
    else:
        attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale.to(x.dtype)) + q_shift.to(x.dtype)
        encoder_hidden_states = context * (1 + scale_kv.to(context.dtype)) + shift_kv.to(context.dtype)
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate
