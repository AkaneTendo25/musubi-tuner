from dataclasses import replace
from enum import Enum

import logging
import torch
import torch.nn as nn
from musubi_tuner.ltx_2.model.transformer.offloading_utils import (
    LTX2BlockSwapManager,
    LTX2ModelOffloader,
)
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

logger = logging.getLogger(__name__)


def _move_non_linear_params(module: nn.Module, device: torch.device) -> None:
    """Move non-linear params/buffers to device; Linear weights are handled by offloader."""
    non_blocking = device.type != "cpu"
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            if submodule.bias is not None and submodule.bias.device != device:
                submodule.bias = nn.Parameter(submodule.bias.to(device, non_blocking=non_blocking))
            continue
        for _, param in submodule.named_parameters(recurse=False):
            if isinstance(param, torch.Tensor) and param.device != device:
                param.data = param.data.to(device, non_blocking=non_blocking)
        for name, buf in submodule.named_buffers(recurse=False):
            if isinstance(buf, torch.Tensor) and buf.device != device:
                setattr(submodule, name, buf.to(device, non_blocking=non_blocking))


def _log_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logger.info("LTX-2 swap mem [%s]: cuda_allocated=%.2fGB cuda_reserved=%.2fGB", tag, allocated, reserved)


def _move_transformer_args(
    args: TransformerArgs | None, device: torch.device
) -> TransformerArgs | None:
    if args is None:
        return None
    if args.x.device == device:
        return args
    non_blocking = device.type != "cpu"
    def _move_tensor(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=non_blocking)
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

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.set_gradient_checkpointing(True)
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self) -> None:
        self.set_gradient_checkpointing(False)
        self.activation_cpu_offloading = False

    def enable_block_swap(
        self,
        blocks_to_swap: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
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
        swap_mode = getattr(self, "swap_mode", "default")
        if self.blocks_to_swap and swap_mode in {"aggressive", "aggressive_no_offload"}:
            target_device = torch.device(device)
            # Move non-block modules/params to the target device.
            saved_blocks = self.transformer_blocks
            self.transformer_blocks = torch.nn.ModuleList()
            self.to(target_device)
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
            return

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
        cpu_device = torch.device("cpu")
        for block in self.transformer_blocks[self.num_blocks - self.blocks_to_swap :]:
            _move_non_linear_params(block, cpu_device)
            move_fp8_scale_weights(block, cpu_device)

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

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

        # Process transformer blocks
        for block_idx, block in enumerate(self.transformer_blocks):
            if swap_active and swap_manager is not None and swap_manager.is_managed_block(block_idx):
                swap_manager.stream_in(block, compute_device)

            if self.blocks_to_swap and self.offloader is not None:
                self.offloader.wait_for_block(block_idx)

            gpu_device = None
            if self.offloader is not None:
                gpu_device = self.offloader.device
            elif swap_active and compute_device is not None:
                gpu_device = compute_device
            elif video is not None and isinstance(video.x, torch.Tensor):
                gpu_device = video.x.device
            elif audio is not None and isinstance(audio.x, torch.Tensor):
                gpu_device = audio.x.device

            if (
                self.activation_cpu_offloading
                and self.training
                and gpu_device is not None
                and gpu_device.type != "cpu"
            ):
                cpu_device = torch.device("cpu")
                if self._enable_gradient_checkpointing:
                    video = _move_transformer_args(video, cpu_device)
                    audio = _move_transformer_args(audio, cpu_device)

                    def _run_block(video_args, audio_args, perturbations_args):
                        video_args = _move_transformer_args(video_args, gpu_device)
                        audio_args = _move_transformer_args(audio_args, gpu_device)
                        _move_non_linear_params(block, gpu_device)
                        ensure_fp8_modules_on_device(block, gpu_device)
                        return block(
                            video=video_args,
                            audio=audio_args,
                            perturbations=perturbations_args,
                        )

                    video, audio = torch.utils.checkpoint.checkpoint(
                        _run_block,
                        video,
                        audio,
                        perturbations,
                        use_reentrant=False,
                    )
                else:
                    video = _move_transformer_args(video, gpu_device)
                    audio = _move_transformer_args(audio, gpu_device)
                    _move_non_linear_params(block, gpu_device)
                    ensure_fp8_modules_on_device(block, gpu_device)
                    video, audio = block(
                        video=video,
                        audio=audio,
                        perturbations=perturbations,
                    )

                video = _move_transformer_args(video, cpu_device)
                audio = _move_transformer_args(audio, cpu_device)
            else:
                target_device = None
                if video is not None and isinstance(video.x, torch.Tensor):
                    target_device = video.x.device
                elif audio is not None and isinstance(audio.x, torch.Tensor):
                    target_device = audio.x.device
                if target_device is not None:
                    _move_non_linear_params(block, target_device)
                    ensure_fp8_modules_on_device(block, target_device)

                if self._enable_gradient_checkpointing and self.training:
                    video, audio = torch.utils.checkpoint.checkpoint(
                        block,
                        video,
                        audio,
                        perturbations,
                        use_reentrant=False,
                    )
                else:
                    video, audio = block(
                        video=video,
                        audio=audio,
                        perturbations=perturbations,
                    )

            if swap_active and swap_manager is not None and swap_manager.is_managed_block(block_idx):
                swap_manager.stream_out(block)

            if self.blocks_to_swap and self.offloader is not None:
                self.offloader.submit_move_blocks_forward(self.transformer_blocks, block_idx)
                if not self.offloader.forward_only and block_idx < self.blocks_to_swap:
                    _move_non_linear_params(block, torch.device("cpu"))
                    move_fp8_scale_weights(block, torch.device("cpu"))

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

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
