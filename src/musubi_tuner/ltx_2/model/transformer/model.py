from dataclasses import replace
from enum import Enum
from typing import Optional
import os

import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from musubi_tuner.ltx_2.model.transformer.offloading_utils import (
    LTX2BlockSwapManager,
    LTX2ModelOffloader,
)
from musubi_tuner.ltx_2.model.ltx2_custom_offloading_utils import _clean_memory_on_device
from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
from musubi_tuner.ltx_2.model.transformer.adaln import AdaLayerNormSingle
from musubi_tuner.ltx_2.model.transformer.attention import AttentionCallable, AttentionFunction
from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import (
    ensure_fp8_modules_on_device,
    move_fp8_scale_weights,
)
from musubi_tuner.ltx_2.model.transformer.modality import Modality
from musubi_tuner.ltx_2.model.transformer.rope import LTXRopeType
from musubi_tuner.ltx_2.model.transformer.text_projection import PixArtAlphaTextProjection
from musubi_tuner.ltx_2.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from musubi_tuner.ltx_2.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from musubi_tuner.ltx_2.utils import to_denoised
from musubi_tuner.ltx_2.utils import create_cpu_offloading_wrapper

logger = logging.getLogger(__name__)


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
        swap_mode = getattr(self, "swap_mode", "default")
        _log_cuda_memory("before_enable_block_swap")
        self.blocks_to_swap = int(blocks_to_swap)
        self.num_blocks = len(self.transformer_blocks)

        if swap_mode in {"aggressive", "aggressive_no_offload"}:
            self._ltx2_block_swap = LTX2BlockSwapManager.build(
                depth=self.num_blocks,
                blocks_to_swap=self.blocks_to_swap,
                swap_device="cpu",
            )
            self.offloader = None
            swap_start = max(0, self.num_blocks - self.blocks_to_swap)
            for idx, block in enumerate(self.transformer_blocks):
                enabled = idx >= swap_start
                setattr(block, "swap_weight_offload", enabled)
                for module in block.modules():
                    if module.__class__.__name__.endswith("Linear"):
                        setattr(module, "swap_weight_offload", enabled)
            _log_cuda_memory("after_enable_block_swap")
            return

        supports_backward_for_offload = supports_backward
        if swap_mode == "aggressive":
            supports_backward_for_offload = False
            if supports_backward:
                logger.warning(
                    "LTX-2 aggressive swap uses forward-only swapping during training; "
                    "enable gradient checkpointing + activation CPU offload to avoid OOM."
                )
        assert self.blocks_to_swap <= self.num_blocks - 1, (
            f"Cannot swap more than {self.num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."
        )

        self.offloader = LTX2ModelOffloader(
            "ltx2_block",
            self.transformer_blocks,
            self.num_blocks,
            self.blocks_to_swap,
            supports_backward_for_offload,
            device,
            use_pinned_memory,
            swap_norms=swap_norms,
        )
        swap_start = max(0, self.num_blocks - self.blocks_to_swap)
        for idx, block in enumerate(self.transformer_blocks):
            enabled = idx >= swap_start
            setattr(block, "swap_weight_offload", enabled)
            for module in block.modules():
                if module.__class__.__name__.endswith("Linear"):
                    setattr(module, "swap_weight_offload", enabled)
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
        import torch  # ensure available
        def _trace_vram(tag):
            if torch.cuda.is_available():
                a = torch.cuda.memory_allocated() / (1024**3)
                r = torch.cuda.memory_reserved() / (1024**3)
                m = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"[VRAM_TRACE_MOVE] {tag}: alloc={a:.2f}GB res={r:.2f}GB max={m:.2f}GB")

        _trace_vram("START move_to_device_except_swap_blocks")
        swap_mode = getattr(self, "swap_mode", "default")
        if self.blocks_to_swap and swap_mode in {"aggressive", "aggressive_no_offload"}:
            target_device = torch.device(device)
            # Move non-block modules/params to the target device.
            saved_blocks = self.transformer_blocks
            self.transformer_blocks = torch.nn.ModuleList()
            _trace_vram("BEFORE self.to(device) [aggressive mode, blocks detached]")
            self.to(target_device)
            _trace_vram("AFTER self.to(device) [aggressive mode]")
            self.transformer_blocks = saved_blocks

            managed_indices = set()
            if self._ltx2_block_swap is not None:
                managed_indices = set(self._ltx2_block_swap.block_indices)
            else:
                managed_indices = set(
                    range(max(0, len(self.transformer_blocks) - self.blocks_to_swap), len(self.transformer_blocks))
                )

            # Move non-managed blocks to GPU; keep managed blocks on CPU.
            cpu_device = torch.device("cpu")
            for idx, block in enumerate(self.transformer_blocks):
                if idx in managed_indices:
                    block.to(cpu_device)
                else:
                    block.to(target_device)
                if idx % 10 == 0:
                    _trace_vram(f"AFTER moving block {idx} (managed={idx in managed_indices})")
            _trace_vram("END move_to_device_except_swap_blocks [aggressive]")
            return

        if self.blocks_to_swap:
            saved_blocks = self.transformer_blocks
            self.transformer_blocks = torch.nn.ModuleList()
            _trace_vram("BEFORE self.to(device) [default mode, blocks detached]")

        self.to(device)
        _trace_vram("AFTER self.to(device) [non-block params moved]")

        if self.blocks_to_swap:
            self.transformer_blocks = saved_blocks
            # Ensure swapped blocks stay on CPU to avoid transient full-model GPU spikes.
            swap_start = max(0, len(self.transformer_blocks) - int(self.blocks_to_swap or 0))
            cpu_device = torch.device("cpu")
            target_device = torch.device(device)
            print(f"[VRAM_TRACE_MOVE] Moving blocks: 0-{swap_start-1} to GPU, {swap_start}-{len(self.transformer_blocks)-1} to CPU")
            for idx, block in enumerate(self.transformer_blocks):
                if idx >= swap_start:
                    block.to(cpu_device)
                else:
                    block.to(target_device)
                if idx % 5 == 0 or idx == swap_start - 1 or idx == swap_start:
                    _trace_vram(f"AFTER moving block {idx} ({'CPU' if idx >= swap_start else 'GPU'})")
        _trace_vram("END move_to_device_except_swap_blocks")

    def prepare_block_swap_before_forward(self) -> None:
        if self.blocks_to_swap is None or self.blocks_to_swap == 0 or self.offloader is None:
            return
        self.offloader.prepare_block_devices_before_forward(self.transformer_blocks)
        # Note: Non-linear params are handled by the offloader (kept on GPU)

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks with optional offloading and block swapping."""

        nan_block_diag = os.getenv("LTX2_NAN_BLOCK_DIAG", "0") == "1"
        strict_swap_sync = os.getenv("LTX2_SWAP_STRICT_SYNC", "0") == "1"
        force_pytorch_attn = os.getenv("LTX2_SWAP_FORCE_PYTORCH_ATTN", "0") == "1"
        force_cross_pytorch = os.getenv("LTX2_FORCE_PYTORCH_CROSS_ATTN", "0") == "1"
        force_cross_fp32 = os.getenv("LTX2_CROSS_ATTN_FP32", "0") == "1"
        cross_attn_swap_only = os.getenv("LTX2_CROSS_ATTN_SWAP_ONLY", "1") == "1"
        force_audio_ctx_pytorch = os.getenv("LTX2_FORCE_PYTORCH_AUDIO_CTX_ATTN", "0") == "1"
        force_audio_ctx_fp32 = os.getenv("LTX2_AUDIO_CTX_ATTN_FP32", "0") == "1"
        audio_ctx_swap_only = os.getenv("LTX2_AUDIO_CTX_ATTN_SWAP_ONLY", "1") == "1"
        fp8_swap_sync = os.getenv("LTX2_SWAP_FP8_SYNC", "1") == "1"
        fp8_swap_sync_strict = os.getenv("LTX2_SWAP_FP8_SYNC_STRICT", "0") == "1"

        swap_manager = self._ltx2_block_swap
        swap_active = False
        compute_device = None
        if swap_manager is not None:
            if video is not None and isinstance(video.x, torch.Tensor):
                compute_device = video.x.device
            elif audio is not None and isinstance(audio.x, torch.Tensor):
                compute_device = audio.x.device
            if compute_device is not None:
                swap_active = swap_manager.activate(
                    self.transformer_blocks,
                    compute_device,
                    self.training and torch.is_grad_enabled(),
                )

        if (swap_active and swap_manager is not None) or (self.blocks_to_swap and self.offloader is not None):
            if video is not None and isinstance(video.x, torch.Tensor):
                _clean_memory_on_device(video.x.device)

        gpu_device = None
        if video is not None and isinstance(video.x, torch.Tensor):
            gpu_device = video.x.device
        elif audio is not None and isinstance(audio.x, torch.Tensor):
            gpu_device = audio.x.device
            
        cpu_device = torch.device("cpu")

        # If offloading is enabled for all blocks, move initial inputs to CPU so the first block's checkpoint saves CPU tensors
        if self.activation_cpu_offloading and self.training:
            checkpoint_start = getattr(self, "_checkpoint_start_idx", 0)
            if checkpoint_start == 0:
                video = _move_transformer_args(video, cpu_device)
                audio = _move_transformer_args(audio, cpu_device)

        use_async_prefetch = os.getenv("LTX2_SWAP_ASYNC_PREFETCH")
        transfer_stream = None
        target_device = gpu_device if gpu_device else torch.device("cuda")
        if use_async_prefetch:
            # Async Stream Setup (Phase 2)
            if not hasattr(self, "_transfer_stream"):
                # Create stream on GPU device if available
                self._transfer_stream = torch.cuda.Stream(device=gpu_device)
            transfer_stream = self._transfer_stream

        # Process transformer blocks
        for block_idx, block in enumerate(self.transformer_blocks):
            swap_start = max(0, len(self.transformer_blocks) - int(self.blocks_to_swap or 0))
            in_swap_range = bool(self.blocks_to_swap) and block_idx >= swap_start
            if force_pytorch_attn and in_swap_range and not getattr(block, "_forced_pytorch_attn", False):
                from musubi_tuner.ltx_2.model.transformer.attention import Attention, AttentionFunction

                for mod in block.modules():
                    if isinstance(mod, Attention):
                        mod.attention_function = AttentionFunction.PYTORCH
                block._forced_pytorch_attn = True
                logger.info("Forced PyTorch attention for swapped block %s", block_idx)

            if force_cross_pytorch or force_cross_fp32:
                enable_cross = (not cross_attn_swap_only) or in_swap_range
                block._force_pytorch_cross_attn = bool(force_cross_pytorch and enable_cross)
                block._force_fp32_cross_attn = bool(force_cross_fp32 and enable_cross)
            if force_audio_ctx_pytorch or force_audio_ctx_fp32:
                enable_audio_ctx = (not audio_ctx_swap_only) or in_swap_range
                block._force_pytorch_audio_ctx_attn = bool(force_audio_ctx_pytorch and enable_audio_ctx)
                block._force_fp32_audio_ctx_attn = bool(force_audio_ctx_fp32 and enable_audio_ctx)

            # Phase 2: Wait for prefetch (if active)
            # If previous block triggered prefetch for this block, wait for it here.
            # Using wait_stream ensures kernels submitted to compute stream (block execution)
            # will wait for transfers submitted to transfer stream to complete.
            if transfer_stream is not None and getattr(block, "weight_cpu_offloading", False):
                torch.cuda.current_stream().wait_stream(transfer_stream)

            if (self.blocks_to_swap or 0) > 0 and self.offloader is not None:
                self.offloader.wait_for_block(block_idx)
                if strict_swap_sync and torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()
            elif (
                (self.blocks_to_swap or 0) > 0
                and self._ltx2_block_swap is not None
                and block_idx in self._ltx2_block_swap.block_indices
            ):
                self._ltx2_block_swap.param_swap(block_idx)
                if strict_swap_sync and torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()

            if fp8_swap_sync and in_swap_range and torch.cuda.is_available():
                # Ensure fp8 weights + scale_weight are on the compute device after swap-in.
                ensure_fp8_modules_on_device(block, gpu_device)
                if fp8_swap_sync_strict:
                    torch.cuda.current_stream().synchronize()

            # Phase 2: Prefetch Next Block
            # Trigger load for N+1 on Transfer Stream while N is about to compute
            if (
                transfer_stream is not None
                and getattr(block, "weight_cpu_offloading", False)
                and block_idx + 1 < len(self.transformer_blocks)
            ):
                next_block = self.transformer_blocks[block_idx + 1]
                # Only prefetch if next block also wants it
                if getattr(next_block, "weight_cpu_offloading", False):
                    with torch.cuda.stream(transfer_stream):
                        # Safe to call because it checks p.device internally
                        # If already loaded, it's a fast no-op.
                        # If on CPU, it triggers H2D copy.
                        next_block._load_weights(next_block, target_device)

            # Execute block (it now handles checkpointing i.e. load/compute/offload)
            # If offloading is on, block expects CPU inputs (for checkpoint savings) and returns CPU outputs
            video, audio = block(video, audio, perturbations)

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

            if swap_active and swap_manager is not None and swap_manager.is_managed_block(block_idx):
                swap_manager.stream_out(block)

            if self.blocks_to_swap and self.offloader is not None:
                # If blockwise checkpointing is handling weight offload for this block,
                # skip swap prefetch to avoid loading extra blocks and spiking VRAM.
                if not getattr(block, "weight_cpu_offloading", False):
                    self.offloader.submit_move_blocks_forward(self.transformer_blocks, block_idx)
                if not self.offloader.forward_only:
                    pass # Keep non-linear params on GPU for backward pass

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
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

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

