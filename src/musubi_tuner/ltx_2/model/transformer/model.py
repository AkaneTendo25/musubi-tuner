from dataclasses import replace
from enum import Enum
from typing import Optional
import os

import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from musubi_tuner.modules.custom_offloading_utils import (
    ModelOffloader as GenericModelOffloader,
    weighs_to_device,
)
from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
from musubi_tuner.ltx_2.model.transformer.adaln import AdaLayerNormSingle
from musubi_tuner.ltx_2.model.transformer.attention import AttentionCallable, AttentionFunction
from musubi_tuner.ltx_2.model.transformer.modality import Modality
from musubi_tuner.ltx_2.model.transformer.rope import LTXRopeType
from musubi_tuner.ltx_2.model.transformer.text_projection import PixArtAlphaTextProjection
from musubi_tuner.ltx_2.model.transformer.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
    _unpack_transformer_args,
    _reconstruct_transformer_args,
)
from musubi_tuner.ltx_2.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from musubi_tuner.ltx_2.utils import to_denoised
from musubi_tuner.ltx_2.utils import create_cpu_offloading_wrapper

logger = logging.getLogger(__name__)

def _align_fp8_scale_weights(block: torch.nn.Module) -> None:
    for mod in block.modules():
        scale = getattr(mod, "scale_weight", None)
        weight = getattr(mod, "weight", None)
        if isinstance(scale, torch.Tensor) and isinstance(weight, torch.Tensor):
            if scale.device != weight.device:
                mod.scale_weight = scale.to(device=weight.device, non_blocking=True)


def _ensure_block_weights_on_device(block: torch.nn.Module, device: torch.device) -> None:
    needs_move = False
    for mod in block.modules():
        if mod.__class__.__name__.endswith("Linear"):
            w = getattr(mod, "weight", None)
            if isinstance(w, torch.Tensor) and w.device != device:
                needs_move = True
                break
    if not needs_move:
        return
    weighs_to_device(block, device)
    _align_fp8_scale_weights(block)


def _move_non_linear_params(module: nn.Module, device: torch.device) -> None:
    """Move non-linear params/buffers to device; Linear weights are handled by offloader."""
    non_blocking = device.type != "cpu"
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
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


def _log_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logger.info("LTX-2 swap mem [%s]: cuda_allocated=%.2fGB cuda_reserved=%.2fGB", tag, allocated, reserved)


def _disable_checkpoint_determinism_check() -> None:
    """Disable checkpoint determinism checks for swap/offload recomputation."""
    if getattr(checkpoint, "_DEFAULT_DETERMINISM_MODE", None) != "none":
        checkpoint._DEFAULT_DETERMINISM_MODE = "none"


def _move_transformer_args(
    args: TransformerArgs | None, device: torch.device
) -> TransformerArgs | None:
    if args is None:
        return None
    if args.x.device == device:
        return args
    # Note: We use synchronous moves (non_blocking=False) for checkpoint recomputation
    # to ensure data is fully available before computation. Async moves can cause issues.
    def _move_tensor(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, (tuple, list)):
            return type(value)(_move_tensor(v) for v in value)
        return value

    return replace(
        args,
        x=_move_tensor(args.x),
        context=_move_tensor(args.context),
        context_mask=_move_tensor(args.context_mask),
        timesteps=_move_tensor(args.timesteps),
        embedded_timestep=_move_tensor(args.embedded_timestep),
        positional_embeddings=_move_tensor(args.positional_embeddings),
        cross_positional_embeddings=_move_tensor(args.cross_positional_embeddings),
        cross_scale_shift_timestep=_move_tensor(args.cross_scale_shift_timestep),
        cross_gate_timestep=_move_tensor(args.cross_gate_timestep),
    )


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        split_attn_target: str | None = None,
        split_attn_mode: str | None = None,
        split_attn_chunk_size: int = 0,
        ffn_chunk_target: str | None = None,
        ffn_chunk_size: int = 0,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        self.offloader = None
        self._ltx2_block_swap = None
        self.num_blocks = 0
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        self.split_attn_target = split_attn_target
        self.split_attn_mode = split_attn_mode
        self.split_attn_chunk_size = split_attn_chunk_size
        self.ffn_chunk_target = ffn_chunk_target
        self.ffn_chunk_size = ffn_chunk_size
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            split_attn_target=self.split_attn_target,
            split_attn_mode=self.split_attn_mode,
            split_attn_chunk_size=self.split_attn_chunk_size,
            ffn_chunk_target=self.ffn_chunk_target,
            ffn_chunk_size=self.ffn_chunk_size,
        )

        self.num_blocks = len(self.transformer_blocks)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)

        self.adaln_single = AdaLayerNormSingle(self.inner_dim)

        # Video caption projection
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.inner_dim,
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
        )

        # Audio caption projection
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.audio_inner_dim,
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        split_attn_target: str | None = None,
        split_attn_mode: str | None = None,
        split_attn_chunk_size: int = 0,
        ffn_chunk_target: str | None = None,
        ffn_chunk_size: int = 0,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                    split_attn_target=split_attn_target,
                    split_attn_mode=split_attn_mode,
                    split_attn_chunk_size=split_attn_chunk_size,
                    ffn_chunk_target=ffn_chunk_target,
                    ffn_chunk_size=ffn_chunk_size,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable
        # Note: If simply toggling enable, we don't change offloading status
        # But for safety/simplicity we can update blocks with current state
        offload = getattr(self, "activation_cpu_offloading", False)
        for block in self.transformer_blocks:
             if enable:
                 block.enable_gradient_checkpointing(offload)
             else:
                 block.gradient_checkpointing = False
                 block.activation_cpu_offloading = False

    # NOTE: enable_gradient_checkpointing is defined later in the file with extended signature

    def disable_gradient_checkpointing(self) -> None:
        self.set_gradient_checkpointing(False)
        self.activation_cpu_offloading = False
        for block in self.transformer_blocks:
            block.gradient_checkpointing = False
            block.activation_cpu_offloading = False

    def enable_block_swap(
        self,
        blocks_to_swap: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
        swap_norms: bool = False,
    ) -> None:
        _log_cuda_memory("before_enable_block_swap")
        self.num_blocks = len(self.transformer_blocks)
        self.blocks_to_swap = int(blocks_to_swap)
        assert self.blocks_to_swap <= self.num_blocks - 1, (
            f"Cannot swap more than {self.num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."
        )

        self._ltx2_block_swap = None
        self.offloader = GenericModelOffloader(
            "ltx2_block",
            self.transformer_blocks,
            self.num_blocks,
            self.blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        logger.info(
            "LTX-2: Block swap enabled. Swapping %s blocks out of %s. Supports backward: %s",
            self.blocks_to_swap,
            self.num_blocks,
            supports_backward,
        )
        _log_cuda_memory("after_enable_block_swap")

    def switch_block_swap_for_inference(self) -> None:
        if self.blocks_to_swap and self.offloader is not None:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if self.blocks_to_swap and self.offloader is not None:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        if self.blocks_to_swap:
            saved_blocks = self.transformer_blocks
            self.transformer_blocks = torch.nn.ModuleList()

        self.to(device)

        if self.blocks_to_swap:
            self.transformer_blocks = saved_blocks

    def prepare_block_swap_before_forward(self) -> None:
        if self.blocks_to_swap is None or self.blocks_to_swap == 0 or self.offloader is None:
            return
        self.offloader.prepare_block_devices_before_forward(self.transformer_blocks)
        for block in self.transformer_blocks:
            _align_fp8_scale_weights(block)
        # Note: Non-linear params are handled by the offloader (kept on GPU)
        if os.getenv("LTX2_SWAP_BLOCK_DEVICE_DIAG", "0") == "1":
            try:
                gpu_blocks = 0
                cpu_blocks = 0
                gpu_weight_blocks = 0
                cpu_weight_blocks = 0
                for block in self.transformer_blocks:
                    dev = None
                    for p in block.parameters():
                        if isinstance(p, torch.Tensor):
                            dev = p.device
                            break
                    if dev is None:
                        continue
                    if dev.type == "cuda":
                        gpu_blocks += 1
                    elif dev.type == "cpu":
                        cpu_blocks += 1
                    # Track Linear weight devices explicitly (more accurate for swap)
                    weight_dev = None
                    for mod in block.modules():
                        if mod.__class__.__name__.endswith("Linear"):
                            w = getattr(mod, "weight", None)
                            if isinstance(w, torch.Tensor):
                                weight_dev = w.device
                                break
                    if weight_dev is not None:
                        if weight_dev.type == "cuda":
                            gpu_weight_blocks += 1
                        elif weight_dev.type == "cpu":
                            cpu_weight_blocks += 1
                logger.info(
                    "LTX-2 swap diag: blocks_to_swap=%s num_blocks=%s gpu_blocks=%s cpu_blocks=%s gpu_weight_blocks=%s cpu_weight_blocks=%s",
                    self.blocks_to_swap,
                    self.num_blocks,
                    gpu_blocks,
                    cpu_blocks,
                    gpu_weight_blocks,
                    cpu_weight_blocks,
                )
            except Exception:
                pass

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks with optional offloading and block swapping."""

        nan_block_diag = os.getenv("LTX2_NAN_BLOCK_DIAG", "0") == "1"

        gpu_device = None
        if video is not None and isinstance(video.x, torch.Tensor):
            gpu_device = video.x.device
        elif audio is not None and isinstance(audio.x, torch.Tensor):
            gpu_device = audio.x.device
            
        cpu_device = torch.device("cpu")
        vram_block_diag = (
            gpu_device is not None
            and gpu_device.type == "cuda"
            and torch.cuda.is_available()
            and os.getenv("LTX2_VRAM_BLOCK_DIAG", "0") == "1"
        )
        vram_block_detail = os.getenv("LTX2_VRAM_BLOCK_DETAIL", "0") == "1"
        vram_block_peaks: list[tuple[int, int]] = []
        vram_total_bytes = None
        if vram_block_diag:
            device_index = gpu_device.index if gpu_device.index is not None else torch.cuda.current_device()
            vram_total_bytes = torch.cuda.get_device_properties(device_index).total_memory
        
        if self.blocks_to_swap and self.offloader is not None:
            self.prepare_block_swap_before_forward()

        # If offloading is enabled for all blocks, move initial inputs to CPU so the first block's checkpoint saves CPU tensors
        if self.activation_cpu_offloading and self.training:
            checkpoint_start = getattr(self, "_checkpoint_start_idx", 0)
            if checkpoint_start == 0:
                video = _move_transformer_args(video, cpu_device)
                audio = _move_transformer_args(audio, cpu_device)

        # Process transformer blocks
        for block_idx, block in enumerate(self.transformer_blocks):
            def _run_block(v_args, a_args):
                if self.blocks_to_swap and self.offloader is not None:
                    self.offloader.wait_for_block(block_idx)
                    _align_fp8_scale_weights(block)
                return block._forward(v_args, a_args, perturbations)

            do_checkpoint = (
                self.training
                and getattr(self, "_enable_gradient_checkpointing", False)
                and block_idx >= getattr(self, "_checkpoint_start_idx", 0)
            )

            if vram_block_diag:
                torch.cuda.synchronize(gpu_device)
                torch.cuda.reset_peak_memory_stats(gpu_device)
                vram_baseline = torch.cuda.memory_allocated(gpu_device)
                if do_checkpoint:
                    video_vals, video_none = _unpack_transformer_args(video)
                    audio_vals, audio_none = _unpack_transformer_args(audio)
                    vid_len = len(video_vals)

                    def _cp_wrapper(*inputs):
                        v_vals = list(inputs[:vid_len])
                        a_vals = list(inputs[vid_len:])
                        v_args = _reconstruct_transformer_args(v_vals, video_none)
                        a_args = _reconstruct_transformer_args(a_vals, audio_none)
                        return _run_block(v_args, a_args)

                    flat_inputs = tuple(video_vals + audio_vals)
                    out_video, out_audio = checkpoint.checkpoint(
                        _cp_wrapper, *flat_inputs, use_reentrant=False, determinism_check="none"
                    )
                    video, audio = out_video, out_audio
                else:
                    video, audio = _run_block(video, audio)
                torch.cuda.synchronize(gpu_device)
                vram_peak = torch.cuda.max_memory_allocated(gpu_device)
                vram_block_peaks.append((block_idx, max(0, int(vram_peak - vram_baseline))))
            else:
                if do_checkpoint:
                    video_vals, video_none = _unpack_transformer_args(video)
                    audio_vals, audio_none = _unpack_transformer_args(audio)
                    vid_len = len(video_vals)

                    def _cp_wrapper(*inputs):
                        v_vals = list(inputs[:vid_len])
                        a_vals = list(inputs[vid_len:])
                        v_args = _reconstruct_transformer_args(v_vals, video_none)
                        a_args = _reconstruct_transformer_args(a_vals, audio_none)
                        return _run_block(v_args, a_args)

                    flat_inputs = tuple(video_vals + audio_vals)
                    out_video, out_audio = checkpoint.checkpoint(
                        _cp_wrapper, *flat_inputs, use_reentrant=False, determinism_check="none"
                    )
                    video, audio = out_video, out_audio
                else:
                    video, audio = _run_block(video, audio)

            if nan_block_diag:
                vx = video.x if video is not None and isinstance(video.x, torch.Tensor) else None
                ax = audio.x if audio is not None and isinstance(audio.x, torch.Tensor) else None

                def _summarize_block_devices(b: torch.nn.Module) -> tuple[int, int, int, int]:
                    params_cpu = params_cuda = buffers_cpu = buffers_cuda = 0
                    for p in b.parameters():
                        if p.device.type == "cpu":
                            params_cpu += 1
                        elif p.device.type == "cuda":
                            params_cuda += 1
                    for buf in b.buffers():
                        if buf.device.type == "cpu":
                            buffers_cpu += 1
                        elif buf.device.type == "cuda":
                            buffers_cuda += 1
                    return params_cpu, params_cuda, buffers_cpu, buffers_cuda

                def _log_block_diag(branch: str):
                    params_cpu, params_cuda, buffers_cpu, buffers_cuda = _summarize_block_devices(block)
                    swap_start = max(0, len(self.transformer_blocks) - int(self.blocks_to_swap or 0))
                    in_swap_range = bool(self.blocks_to_swap) and block_idx >= swap_start
                    logger.error(
                        "NaN/Inf after block %s (%s). swap_range=%s..%s in_swap_range=%s swap_mode=%s offloader=%s "
                        "params_cpu=%s params_cuda=%s buffers_cpu=%s buffers_cuda=%s",
                        block_idx,
                        branch,
                        swap_start,
                        len(self.transformer_blocks) - 1,
                        in_swap_range,
                        getattr(self, "swap_mode", "default"),
                        self.offloader is not None,
                        params_cpu,
                        params_cuda,
                        buffers_cpu,
                        buffers_cuda,
                    )

                if vx is not None and not torch.isfinite(vx).all():
                    _log_block_diag("video")
                    raise RuntimeError(f"Non-finite video activations after block {block_idx}")
                if ax is not None and not torch.isfinite(ax).all():
                    _log_block_diag("audio")
                    raise RuntimeError(f"Non-finite audio activations after block {block_idx}")

            if self.blocks_to_swap and self.offloader is not None:
                # If blockwise checkpointing is handling weight offload for this block,
                # skip swap prefetch to avoid loading extra blocks and spiking VRAM.
                if not getattr(block, "weight_cpu_offloading", False):
                    self.offloader.submit_move_blocks_forward(self.transformer_blocks, block_idx)

        if vram_block_diag and vram_block_detail and vram_block_peaks:
            def _fmt_entry(entry: tuple[int, int]) -> str:
                idx, bytes_used = entry
                mb = bytes_used / (1024 ** 2)
                if vram_total_bytes:
                    pct = (bytes_used / vram_total_bytes) * 100.0
                    return f"{idx}:{mb:.1f}MB({pct:.2f}%)"
                return f"{idx}:{mb:.1f}MB"

            top = sorted(vram_block_peaks, key=lambda x: x[1], reverse=True)
            top_summary = ", ".join(_fmt_entry(e) for e in top[:10])
            full_summary = ", ".join(_fmt_entry(e) for e in vram_block_peaks)
            logger.info("LTX-2 VRAM block peaks (top 10): %s", top_summary)
            logger.info("LTX-2 VRAM block peaks (all): %s", full_summary)

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Handle tuple-format embedded_timestep from unique timestep optimization
        if isinstance(embedded_timestep, tuple) and len(embedded_timestep) == 4:
            unique_embedded, inverse_indices_1d, B, T = embedded_timestep
            embedded_timestep = unique_embedded[inverse_indices_1d].view(B, T, -1)
        
        # AdaLN Structural Fix: Force modulation to happen in Float32 to prevent overflow (10^18 issue)
        # This is strictly required for stability when large activations meet scale factors.
        x_32 = x.to(torch.float32)
        embedded_32 = embedded_timestep.to(device=x.device, dtype=torch.float32)
        
        # Apply scale-shift modulation in float32
        scale_shift_values = scale_shift_table[None, None].to(device=x.device, dtype=torch.float32) + embedded_32[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        # LayerNorm in float32 (using functional to avoid in-place module dtype modification)
        x_32 = torch.nn.functional.layer_norm(x_32, norm_out.normalized_shape, eps=norm_out.eps)
        
        x_32 = x_32 * (1 + scale) + shift
        
        # Back to original dtype for projection
        x = x_32.to(x.dtype)
        x = proj_out(x)
        return x

    def enable_gradient_checkpointing(
        self,
        activation_cpu_offloading: bool = False,
        weight_cpu_offloading: bool = False,
        blocks_to_checkpoint: Optional[int] = None,
    ) -> None:
        """
        Enable gradient checkpointing with optional CPU offloading.
        
        Args:
            activation_cpu_offloading: If True, offload activations to CPU (save memory).
            weight_cpu_offloading: If True, use block-level weight offloading (ultra-low VRAM).
        """
        if blocks_to_checkpoint == 0:
            self._enable_gradient_checkpointing = False
            self.activation_cpu_offloading = False
            self.weight_cpu_offloading = False
        else:
            self._enable_gradient_checkpointing = True
            self.activation_cpu_offloading = activation_cpu_offloading
            self.weight_cpu_offloading = weight_cpu_offloading
        _disable_checkpoint_determinism_check()

        if blocks_to_checkpoint is None or blocks_to_checkpoint == -1:
            checkpoint_start = 0
        else:
            checkpoint_start = max(0, len(self.transformer_blocks) - int(blocks_to_checkpoint))
        self._checkpoint_start_idx = checkpoint_start

        for idx, block in enumerate(self.transformer_blocks):
            if blocks_to_checkpoint == 0:
                block.gradient_checkpointing = False
                block.activation_cpu_offloading = False
                block.weight_cpu_offloading = False
                continue
            
            if idx < checkpoint_start:
                # Use standard checkpointing (no CPU/weight offload) for early blocks.
                block.gradient_checkpointing = True
                block.activation_cpu_offloading = False
                block.weight_cpu_offloading = False
                continue

            if hasattr(block, "enable_gradient_checkpointing"):
                block.enable_gradient_checkpointing(activation_cpu_offloading, weight_cpu_offloading)

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio) if audio is not None else None

        vram_diag = os.getenv("LTX2_VRAM_BLOCK_DIAG", "0") == "1"
        vram_device = None
        if vram_diag and torch.cuda.is_available():
            if video_args is not None and isinstance(video_args.x, torch.Tensor):
                vram_device = video_args.x.device
            elif audio_args is not None and isinstance(audio_args.x, torch.Tensor):
                vram_device = audio_args.x.device
            if vram_device is not None and vram_device.type == "cuda":
                torch.cuda.synchronize(vram_device)
                torch.cuda.reset_peak_memory_stats(vram_device)
                vram_baseline = torch.cuda.memory_allocated(vram_device)
            else:
                vram_diag = False
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )
        if vram_diag and vram_device is not None:
            torch.cuda.synchronize(vram_device)
            vram_peak = torch.cuda.max_memory_allocated(vram_device)
            vram_delta = max(0, int(vram_peak - vram_baseline))
            logger.info(
                "LTX-2 VRAM forward peak: baseline=%.2fMB peak=%.2fMB delta=%.2fMB",
                vram_baseline / (1024 ** 2),
                vram_peak / (1024 ** 2),
                vram_delta / (1024 ** 2),
            )
            # reset peak stats for backward measurement
            torch.cuda.reset_peak_memory_stats(vram_device)

        if self.activation_cpu_offloading and self.training:
            target_device = None
            if self.offloader is not None:
                target_device = self.offloader.device
            elif video_out is not None and isinstance(video_out.x, torch.Tensor):
                target_device = video_out.x.device
            elif audio_out is not None and isinstance(audio_out.x, torch.Tensor):
                target_device = audio_out.x.device
            if target_device is not None and target_device.type != "cpu":
                video_out = _move_transformer_args(video_out, target_device)
                audio_out = _move_transformer_args(audio_out, target_device)

        # Ensure outputs are on the same device as output projections (optional)
        if os.getenv("LTX2_ALIGN_OUTPUT_DEVICE", "0") == "1":
            if video_out is not None and isinstance(video_out.x, torch.Tensor):
                proj_device = self.proj_out.weight.device
                if video_out.x.device != proj_device:
                    video_out = _move_transformer_args(video_out, proj_device)
            if audio_out is not None and isinstance(audio_out.x, torch.Tensor):
                proj_device = self.audio_proj_out.weight.device
                if audio_out.x.device != proj_device:
                    audio_out = _move_transformer_args(audio_out, proj_device)

        # Process output
        vx = (
            self._process_output(self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep)
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        if vram_diag and vram_device is not None:
            def _log_backward_peak(_grad):
                try:
                    torch.cuda.synchronize(vram_device)
                    peak = torch.cuda.max_memory_allocated(vram_device)
                    logger.info(
                        "LTX-2 VRAM backward peak: peak=%.2fMB",
                        peak / (1024 ** 2),
                    )
                except Exception:
                    pass
                return _grad

            if vx is not None and isinstance(vx, torch.Tensor) and vx.requires_grad:
                vx.register_hook(_log_backward_peak)
            elif ax is not None and isinstance(ax, torch.Tensor) and ax.requires_grad:
                ax.register_hook(_log_backward_peak)

        return vx, ax


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
logger = logging.getLogger(__name__)

