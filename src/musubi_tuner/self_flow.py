"""Self-Flow helper for LTX-2 training.

Implements the Self-Flow training regularizer:
- Dual-timestep noising is performed by the trainer.
- This module handles feature alignment loss with a teacher model.

Teacher modes:
  "base" (default): teacher = frozen pretrained base model (LoRA multipliers zeroed).
      No extra VRAM vs EMA, stronger teacher-student gap, acts as regularizer.
  "ema": teacher = EMA-smoothed copy of LoRA weights (original behaviour).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SelfFlowConfig:
    student_block_idx: int = 16
    teacher_block_idx: int = 32
    student_block_ratio: Optional[float] = None
    teacher_block_ratio: Optional[float] = None
    lambda_self_flow: float = 0.1
    temporal_mode: str = "off"  # "off" | "frame" | "delta" | "hybrid"
    lambda_temporal: float = 0.0
    lambda_delta: float = 0.0
    temporal_tau: float = 1.0
    num_neighbors: int = 2
    temporal_granularity: str = "frame"  # "frame" | "patch"
    patch_spatial_radius: int = 0
    patch_match_mode: str = "hard"  # "hard" | "soft"
    patch_match_temperature: float = 0.1
    delta_num_steps: int = 1
    motion_weighting: str = "none"  # "none" | "teacher_delta"
    motion_weight_strength: float = 0.0
    temporal_schedule: str = "constant"  # "constant" | "linear" | "cosine"
    temporal_warmup_steps: int = 0
    temporal_max_steps: int = 0
    mask_ratio: float = 0.10
    frame_level_mask: bool = False  # mask whole frames instead of individual tokens
    teacher_mode: str = "base"  # "base" | "ema" | "partial_ema"
    teacher_momentum: float = 0.999
    teacher_update_interval: int = 1
    projector_hidden_multiplier: int = 1
    projector_activation: str = "silu"  # "silu" | "gelu"
    loss_type: str = "negative_cosine"  # "negative_cosine" | "one_minus_cosine"
    dual_timestep: bool = True
    tokenwise_timestep: bool = True
    mask_focus_loss: bool = False  # focus rep loss on masked (higher-noise) tokens only
    max_loss: float = 0.0  # cap Self-Flow loss magnitude by rescaling (0 = disabled); caps the summed scalar loss value, not a gradient norm
    student_block_stochastic_range: int = 0  # randomly vary student block ± this many blocks each step
    offload_teacher_features: bool = False
    offload_teacher_params: bool = False
    projector_lr: Optional[float] = None
    lambda_audio: float = 0.0  # audio representation alignment weight (0 = disabled)


def parse_self_flow_args(raw_args: Optional[list[str]]) -> Dict[str, str]:
    """Parse ``key=value`` list into a dict. Returns empty dict for None/[]."""
    if not raw_args:
        return {}
    out: Dict[str, str] = {}
    for item in raw_args:
        if "=" not in item:
            raise ValueError(f"Self-Flow arg must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


class SelfFlowModule:
    """Training helper for Self-Flow feature alignment.

    Hooks are installed on the student model blocks. Teacher features are captured
    by running a second forward pass with EMA LoRA weights.
    """

    def __init__(self, config: SelfFlowConfig, transformer: nn.Module):
        self.config = config
        self.transformer = transformer

        self.projector: Optional[nn.Sequential] = None
        self.audio_projector: Optional[nn.Sequential] = None
        self._hooks: list = []

        self._capture_mode: str = "idle"  # "student" | "teacher" | "idle"
        self._student_features: Optional[torch.Tensor] = None
        self._teacher_features: Optional[torch.Tensor] = None
        self._student_audio_features: Optional[torch.Tensor] = None
        self._teacher_audio_features: Optional[torch.Tensor] = None

        self._shadow_params: Dict[str, torch.Tensor] = {}
        self._step_counter: int = 0
        self._last_cosine: Optional[float] = None
        self._last_audio_cosine: Optional[float] = None
        self._last_frame_cosine: Optional[float] = None
        self._last_delta_cosine: Optional[float] = None
        self._last_ema_drift: Optional[float] = None
        self._current_lambda_self_flow: float = float(config.lambda_self_flow)
        self._current_lambda_audio: float = float(config.lambda_audio)
        self._current_lambda_temporal: float = float(config.lambda_temporal)
        self._current_lambda_delta: float = float(config.lambda_delta)
        self._resolved_student_block_idx: Optional[int] = None
        self._resolved_teacher_block_idx: Optional[int] = None
        self._active_student_block_idx: Optional[int] = None   # may differ each step when stochastic
        self._stochastic_student_indices: list = []            # full list of hookable student blocks

    @property
    def last_cosine(self) -> Optional[float]:
        return self._last_cosine

    @property
    def last_ema_drift(self) -> Optional[float]:
        return self._last_ema_drift

    @property
    def last_audio_cosine(self) -> Optional[float]:
        return self._last_audio_cosine

    @property
    def last_frame_cosine(self) -> Optional[float]:
        return self._last_frame_cosine

    @property
    def last_delta_cosine(self) -> Optional[float]:
        return self._last_delta_cosine

    @property
    def current_lambda_temporal(self) -> float:
        return float(self._current_lambda_temporal)

    @property
    def current_lambda_delta(self) -> float:
        return float(self._current_lambda_delta)

    @property
    def current_lambda_self_flow(self) -> float:
        return float(self._current_lambda_self_flow)

    @property
    def has_active_loss(self) -> bool:
        return (
            self._current_lambda_self_flow > 0.0
            or self._current_lambda_audio > 0.0
            or (str(self.config.temporal_mode).lower() in {"frame", "hybrid"} and self.current_lambda_temporal > 0.0)
            or (str(self.config.temporal_mode).lower() in {"delta", "hybrid"} and self.current_lambda_delta > 0.0)
        )

    def _make_activation(self) -> nn.Module:
        act = str(self.config.projector_activation).lower()
        if act == "gelu":
            return nn.GELU()
        return nn.SiLU()

    def _get_blocks(self) -> tuple[list[nn.Module], int]:
        blocks = getattr(self.transformer, "transformer_blocks", None)
        if blocks is None:
            raise ValueError("Self-Flow requires transformer.transformer_blocks")
        block_list = list(blocks)
        return block_list, len(block_list)

    @staticmethod
    def _matches_block(param_name: str, block_idx: int) -> bool:
        """Return True if a parameter name belongs to the given transformer block index.

        Handles both dot notation (``transformer_blocks.32.``) and kohya-style
        underscore notation (``transformer_blocks_32_``).
        """
        return (
            f"transformer_blocks.{block_idx}." in param_name
            or f"transformer_blocks_{block_idx}_" in param_name
        )

    @staticmethod
    def _resolve_shadow_name(param_name: str, shadow_params: Dict[str, torch.Tensor]) -> Optional[str]:
        if param_name in shadow_params:
            return param_name
        if param_name.startswith("module."):
            stripped = param_name[len("module."):]
            if stripped in shadow_params:
                return stripped
        prefixed = f"module.{param_name}"
        if prefixed in shadow_params:
            return prefixed
        return None

    @staticmethod
    def _resolve_ratio_index(ratio: float, depth: int, *, mode: str) -> int:
        if not (0.0 < float(ratio) < 1.0):
            raise ValueError(f"Self-Flow block ratio must be in (0, 1), got {ratio!r}")
        position = float(ratio) * float(depth)
        if mode == "floor":
            resolved = int(math.floor(position))
        elif mode == "ceil":
            resolved = int(math.ceil(position))
        else:
            raise ValueError(f"Unsupported Self-Flow ratio resolution mode: {mode!r}")
        return max(0, min(depth - 1, resolved))

    def resolve_block_indices(self, depth: int) -> tuple[int, int]:
        student_idx = int(self.config.student_block_idx)
        teacher_idx = int(self.config.teacher_block_idx)
        if self.config.student_block_ratio is not None:
            student_idx = self._resolve_ratio_index(self.config.student_block_ratio, depth, mode="floor")
        if self.config.teacher_block_ratio is not None:
            teacher_idx = self._resolve_ratio_index(self.config.teacher_block_ratio, depth, mode="ceil")
        return student_idx, teacher_idx

    def setup(self, device: torch.device, dtype: torch.dtype) -> None:
        blocks, depth = self._get_blocks()
        student_idx, teacher_idx = self.resolve_block_indices(depth)
        if not (0 <= student_idx < depth):
            raise ValueError(
                f"student_block_idx={student_idx} out of range (model has {depth} blocks)"
            )
        if not (0 <= teacher_idx < depth):
            raise ValueError(
                f"teacher_block_idx={teacher_idx} out of range (model has {depth} blocks)"
            )
        if teacher_idx <= student_idx:
            raise ValueError(
                "teacher_block_idx must be > student_block_idx for Self-Flow"
            )
        _valid_teacher_modes = {"base", "ema", "partial_ema"}
        if str(self.config.teacher_mode).lower() not in _valid_teacher_modes:
            raise ValueError(
                f"Unknown teacher_mode={self.config.teacher_mode!r}. "
                f"Must be one of: {sorted(_valid_teacher_modes)}"
            )
        self._resolved_student_block_idx = student_idx
        self._resolved_teacher_block_idx = teacher_idx
        self._active_student_block_idx = student_idx

        stochastic_range = max(0, int(self.config.student_block_stochastic_range))
        lo = max(0, student_idx - stochastic_range)
        hi = min(teacher_idx - 1, student_idx + stochastic_range)  # must stay below teacher
        self._stochastic_student_indices = list(range(lo, hi + 1))
        if stochastic_range > 0 and len(self._stochastic_student_indices) > 1:
            logger.warning(
                "Self-Flow: student_block_stochastic_range=%d creates %d hookable student blocks [%d..%d], "
                "but a single projector MLP is shared across all depths. "
                "The projector will be a compromise; consider range=0 for best alignment accuracy.",
                stochastic_range,
                len(self._stochastic_student_indices),
                lo,
                hi,
            )
        if self._stochastic_student_indices and max(self._stochastic_student_indices) >= teacher_idx - 1:
            logger.warning(
                "Self-Flow: stochastic student range reaches block %d which is adjacent to teacher block %d. "
                "A 1-block student-teacher gap may produce a trivially satisfied loss. "
                "Consider reducing student_block_stochastic_range.",
                max(self._stochastic_student_indices),
                teacher_idx,
            )

        inner_dim = int(getattr(self.transformer, "inner_dim", 0))
        if inner_dim <= 0:
            raise ValueError("Self-Flow could not resolve transformer.inner_dim")
        hidden_dim = inner_dim * max(1, int(self.config.projector_hidden_multiplier))
        activation = self._make_activation()
        self.projector = nn.Sequential(
            nn.Linear(inner_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, inner_dim),
        ).to(device=device, dtype=dtype)

        # Audio projector: created when lambda_audio > 0 and model has audio_inner_dim
        if float(self.config.lambda_audio) > 0.0:
            audio_inner_dim = int(getattr(self.transformer, "audio_inner_dim", 0))
            if audio_inner_dim <= 0:
                logger.warning(
                    "Self-Flow lambda_audio=%.4f but transformer has no audio_inner_dim; "
                    "audio alignment disabled.",
                    self.config.lambda_audio,
                )
                self.config.lambda_audio = 0.0
                self._current_lambda_audio = 0.0
            else:
                audio_hidden_dim = audio_inner_dim * max(1, int(self.config.projector_hidden_multiplier))
                self.audio_projector = nn.Sequential(
                    nn.Linear(audio_inner_dim, audio_hidden_dim),
                    self._make_activation(),
                    nn.Linear(audio_hidden_dim, audio_inner_dim),
                ).to(device=device, dtype=dtype)
                logger.info("Self-Flow audio projector created: audio_inner_dim=%d", audio_inner_dim)

        self._install_hooks(blocks)
        logger.info(
            "Self-Flow ready: student_block=%d teacher_block=%d student_ratio=%s teacher_ratio=%s "
            "mask_ratio=%.3f frame_level_mask=%s teacher_mode=%s lambda=%.4f "
            "temporal_mode=%s lambda_temporal=%.4f lambda_delta=%.4f "
            "temporal_tau=%.3f num_neighbors=%d temporal_granularity=%s patch_spatial_radius=%d "
            "patch_match_mode=%s patch_match_temperature=%.4f delta_num_steps=%d motion_weighting=%s "
            "motion_weight_strength=%.4f temporal_schedule=%s "
            "temporal_warmup_steps=%d temporal_max_steps=%d "
            "max_loss=%.4f student_block_stochastic_range=%d "
            "momentum=%.4f dual_timestep=%s tokenwise_timestep=%s "
            "offload_teacher_params=%s projector_lr=%s",
            student_idx,
            teacher_idx,
            self.config.student_block_ratio,
            self.config.teacher_block_ratio,
            self.config.mask_ratio,
            str(self.config.frame_level_mask).lower(),
            self.config.teacher_mode,
            self.config.lambda_self_flow,
            self.config.temporal_mode,
            self.config.lambda_temporal,
            self.config.lambda_delta,
            self.config.temporal_tau,
            self.config.num_neighbors,
            self.config.temporal_granularity,
            self.config.patch_spatial_radius,
            self.config.patch_match_mode,
            self.config.patch_match_temperature,
            self.config.delta_num_steps,
            self.config.motion_weighting,
            self.config.motion_weight_strength,
            self.config.temporal_schedule,
            self.config.temporal_warmup_steps,
            self.config.temporal_max_steps,
            self.config.max_loss,
            int(self.config.student_block_stochastic_range),
            self.config.teacher_momentum,
            str(self.config.dual_timestep).lower(),
            str(self.config.tokenwise_timestep).lower(),
            str(self.config.offload_teacher_params).lower(),
            self.config.projector_lr,
        )

    def _install_hooks(self, blocks: list[nn.Module]) -> None:
        def _extract_video_tensor(output: Any) -> Optional[torch.Tensor]:
            if isinstance(output, tuple) and len(output) >= 1:
                video_out = output[0]
                if hasattr(video_out, "x") and torch.is_tensor(video_out.x):
                    return video_out.x
            return None

        def _extract_audio_tensor(output: Any) -> Optional[torch.Tensor]:
            if isinstance(output, tuple) and len(output) >= 2:
                audio_out = output[1]
                if audio_out is not None and hasattr(audio_out, "x") and torch.is_tensor(audio_out.x):
                    return audio_out.x
            return None

        capture_audio = self.audio_projector is not None

        def _make_student_hook(block_idx: int):
            def _hook(_module, _inputs, output):
                if self._capture_mode != "student":
                    return
                if block_idx != self._active_student_block_idx:
                    return
                tensor = _extract_video_tensor(output)
                if tensor is not None:
                    self._student_features = tensor
                if capture_audio:
                    audio_tensor = _extract_audio_tensor(output)
                    if audio_tensor is not None:
                        self._student_audio_features = audio_tensor
            return _hook

        def _teacher_hook(_module, _inputs, output):
            if self._capture_mode != "teacher":
                return
            tensor = _extract_video_tensor(output)
            if tensor is not None:
                self._teacher_features = tensor.detach()
            if capture_audio:
                audio_tensor = _extract_audio_tensor(output)
                if audio_tensor is not None:
                    self._teacher_audio_features = audio_tensor.detach()

        for bidx in self._stochastic_student_indices:
            self._hooks.append(blocks[bidx].register_forward_hook(_make_student_hook(bidx)))
        self._hooks.append(
            blocks[int(self._resolved_teacher_block_idx)].register_forward_hook(_teacher_hook)
        )

    @staticmethod
    def _collect_lora_modules(network: nn.Module) -> list:
        """Return all LoRA modules (identified by lora_down + lora_up + multiplier attrs)."""
        return [
            m for m in network.modules()
            if hasattr(m, "lora_down") and hasattr(m, "lora_up") and hasattr(m, "multiplier")
        ]

    @staticmethod
    def _zero_lora_multipliers(modules: list) -> list:
        """Zero every LoRA module's multiplier; return saved values for restoration."""
        saved = [float(m.multiplier) for m in modules]
        for m in modules:
            m.multiplier = 0.0
        return saved

    @staticmethod
    def _restore_lora_multipliers(modules: list, saved: list) -> None:
        for m, v in zip(modules, saved):
            m.multiplier = v

    def init_teacher(self, network: nn.Module) -> None:
        mode = str(self.config.teacher_mode).lower()
        if mode == "base":
            logger.info("Self-Flow teacher_mode=base: skipping EMA init (using frozen base model as teacher)")
            return
        teacher_block = self._resolved_teacher_block_idx
        self._shadow_params.clear()
        for name, param in network.named_parameters():
            if not param.requires_grad:
                continue
            if mode == "partial_ema" and teacher_block is not None:
                if not self._matches_block(name, teacher_block):
                    continue
            if bool(self.config.offload_teacher_params):
                self._shadow_params[name] = param.detach().to(device="cpu").clone()
            else:
                self._shadow_params[name] = param.detach().clone()
        # Warn if EMA target is the full transformer (full fine-tuning mode).
        is_full_transformer = hasattr(network, "transformer_blocks")
        if mode in ("ema", "partial_ema") and is_full_transformer and self._shadow_params:
            param_count = sum(p.numel() for p in self._shadow_params.values())
            param_mb = param_count * 4 / (1024 * 1024)  # fp32 estimate
            logger.warning(
                "Self-Flow: EMA shadow params cover the full transformer (%.0f MB). "
                "Use teacher_mode=partial_ema to limit shadow params to one block.",
                param_mb,
            )

        if mode == "partial_ema":
            if not self._shadow_params:
                logger.warning(
                    "Self-Flow teacher_mode=partial_ema: no trainable parameters matched block %s. "
                    "Ensure transformer_blocks.{idx} layers are included in the LoRA network. "
                    "EMA teacher will behave identically to student (zero distillation signal). "
                    "Consider using teacher_mode=ema or teacher_mode=base instead.",
                    teacher_block,
                )
            else:
                logger.info(
                    "Self-Flow teacher_mode=partial_ema: EMA for %d tensors in block %s",
                    len(self._shadow_params),
                    teacher_block,
                )
        else:
            logger.info("Self-Flow: initialized EMA teacher with %d tensors", len(self._shadow_params))

    def update_teacher(self, network: nn.Module) -> None:
        if not self._shadow_params:
            return
        self._step_counter += 1
        if self._step_counter % max(1, int(self.config.teacher_update_interval)) != 0:
            return
        momentum = float(self.config.teacher_momentum)
        drift_sum = 0.0
        drift_count = 0
        with torch.no_grad():
            for name, param in network.named_parameters():
                shadow_name = self._resolve_shadow_name(name, self._shadow_params)
                if shadow_name is None:
                    continue
                shadow = self._shadow_params[shadow_name]
                shadow_target_dtype = shadow.dtype
                source = param.detach()
                if source.device != shadow.device or source.dtype != shadow_target_dtype:
                    source = source.to(device=shadow.device, dtype=shadow_target_dtype)
                # Compute drift before EMA update
                drift_sum += (shadow - source).norm().item()
                drift_count += 1
                shadow.mul_(momentum).add_(source, alpha=1.0 - momentum)
        if drift_count > 0:
            self._last_ema_drift = drift_sum / drift_count

    def _swap_in_teacher(self, network: nn.Module) -> Dict[str, torch.Tensor]:
        backups: Dict[str, torch.Tensor] = {}
        if not self._shadow_params:
            return backups
        with torch.no_grad():
            for name, param in network.named_parameters():
                shadow_name = self._resolve_shadow_name(name, self._shadow_params)
                if shadow_name is None:
                    continue
                shadow = self._shadow_params[shadow_name]
                backups[name] = param.detach().clone()
                if shadow.device != param.device or shadow.dtype != param.dtype:
                    param.copy_(shadow.to(device=param.device, dtype=param.dtype))
                else:
                    param.copy_(shadow)
        return backups

    @staticmethod
    def _restore_from_backups(network: nn.Module, backups: Dict[str, torch.Tensor]) -> None:
        if not backups:
            return
        with torch.no_grad():
            for name, param in network.named_parameters():
                backup = backups.get(name)
                if backup is None:
                    continue
                param.copy_(backup.to(device=param.device, dtype=param.dtype))

    def mark_student_forward(self) -> None:
        self._capture_mode = "student"
        self._student_features = None
        if len(self._stochastic_student_indices) > 1:
            self._active_student_block_idx = random.choice(self._stochastic_student_indices)
        else:
            self._active_student_block_idx = self._resolved_student_block_idx

    def cleanup_step(self) -> None:
        self._capture_mode = "idle"
        self._student_features = None
        self._teacher_features = None
        self._student_audio_features = None
        self._teacher_audio_features = None
        self._last_cosine = None
        self._last_audio_cosine = None
        self._last_frame_cosine = None
        self._last_delta_cosine = None
        # Note: _last_ema_drift is NOT cleared here — it's updated in update_teacher
        # which runs after optimizer.step, not during compute_loss

    def on_step(self, global_step: int) -> None:
        scale = self._schedule_scale(global_step)
        self._current_lambda_self_flow = float(self.config.lambda_self_flow) * scale
        self._current_lambda_audio = float(self.config.lambda_audio) * scale
        self._current_lambda_temporal = float(self.config.lambda_temporal) * scale
        self._current_lambda_delta = float(self.config.lambda_delta) * scale

    def _schedule_scale(self, global_step: int) -> float:
        schedule = str(self.config.temporal_schedule).lower()
        if schedule == "constant":
            return 1.0

        warmup_steps = max(0, int(self.config.temporal_warmup_steps))
        if warmup_steps > 0 and global_step < warmup_steps:
            return float(global_step) / float(warmup_steps)

        max_steps = max(0, int(self.config.temporal_max_steps))
        if max_steps <= 0:
            return 1.0

        progress = min(
            max(float(global_step - warmup_steps), 0.0) / max(float(max_steps - warmup_steps), 1.0),
            1.0,
        )
        if schedule == "linear":
            return 1.0 - progress
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        params = []
        if self.projector is not None:
            params.extend(self.projector.parameters())
        if self.audio_projector is not None:
            params.extend(self.audio_projector.parameters())
        return params

    def prepare_teacher_features(
        self,
        *,
        accelerator,
        transformer: nn.Module,
        network: nn.Module,
        teacher_model_input: Any,
        teacher_timesteps: torch.Tensor,
        audio_timestep: Optional[torch.Tensor],
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        frame_rate: int | float,
        transformer_options: Dict[str, Any],
    ) -> None:
        if self.projector is None or not self.has_active_loss:
            return
        self._teacher_features = None
        prev_training = bool(getattr(transformer, "training", False))

        if str(self.config.teacher_mode).lower() == "base":
            # Teacher = frozen pretrained base: zero all LoRA multipliers for this pass.
            lora_mods = self._collect_lora_modules(network)
            if not lora_mods:
                logger.warning(
                    "Self-Flow teacher_mode=base: no LoRA modules found in network "
                    "(expected lora_down/lora_up/multiplier attrs). Teacher forward = student forward; "
                    "self-distillation signal will be zero. Check network type or use teacher_mode=ema."
                )
            saved_mults = self._zero_lora_multipliers(lora_mods)
            try:
                self._capture_mode = "teacher"
                if prev_training:
                    transformer.eval()
                with torch.no_grad(), accelerator.autocast():
                    _ = transformer(
                        teacher_model_input,
                        timestep=teacher_timesteps,
                        audio_timestep=audio_timestep,
                        context=text_embeds,
                        attention_mask=text_mask,
                        frame_rate=frame_rate,
                        transformer_options=transformer_options,
                    )
            finally:
                self._capture_mode = "idle"
                self._restore_lora_multipliers(lora_mods, saved_mults)
                if prev_training:
                    transformer.train()
        else:
            # Teacher = EMA-smoothed LoRA weights ("ema" or "partial_ema").
            # For partial_ema, shadow_params is already scoped to the teacher block only,
            # so _swap_in_teacher naturally only touches those params.
            backups = self._swap_in_teacher(network)
            try:
                self._capture_mode = "teacher"
                if prev_training:
                    transformer.eval()
                with torch.no_grad(), accelerator.autocast():
                    _ = transformer(
                        teacher_model_input,
                        timestep=teacher_timesteps,
                        audio_timestep=audio_timestep,
                        context=text_embeds,
                        attention_mask=text_mask,
                        frame_rate=frame_rate,
                        transformer_options=transformer_options,
                    )
            finally:
                self._capture_mode = "idle"
                self._restore_from_backups(network, backups)
                if prev_training:
                    transformer.train()

        if bool(self.config.offload_teacher_features):
            if (
                self._teacher_features is not None
                and self._teacher_features.device.type != "cpu"
            ):
                self._teacher_features = self._teacher_features.to(device="cpu", non_blocking=False)
            if (
                self._teacher_audio_features is not None
                and self._teacher_audio_features.device.type != "cpu"
            ):
                self._teacher_audio_features = self._teacher_audio_features.to(device="cpu", non_blocking=False)

    def _loss_from_cosine(self, cosine: torch.Tensor) -> torch.Tensor:
        if self.config.loss_type == "one_minus_cosine":
            return 1.0 - cosine
        return -cosine  # negative cosine: gradient pushes cosine toward +1 (same direction as 1-cosine but different magnitude)

    def _reshape_temporal_features(
        self, features: torch.Tensor, num_latent_frames: Optional[int]
    ) -> Optional[torch.Tensor]:
        if num_latent_frames is None:
            return None
        total_tokens = int(features.shape[1])
        num_frames = int(num_latent_frames)
        if num_frames <= 1 or total_tokens < num_frames:
            return None
        usable_tokens = (total_tokens // num_frames) * num_frames
        if usable_tokens <= 0:
            return None
        if usable_tokens != total_tokens:
            features = features[:, :usable_tokens]
        spatial_tokens = usable_tokens // num_frames
        return features.reshape(features.shape[0], num_frames, spatial_tokens, features.shape[-1])

    @staticmethod
    def _reshape_temporal_grid(
        features: torch.Tensor,
        *,
        num_latent_frames: Optional[int],
        latent_height: Optional[int],
        latent_width: Optional[int],
    ) -> Optional[torch.Tensor]:
        if num_latent_frames is None or latent_height is None or latent_width is None:
            return None
        num_frames = int(num_latent_frames)
        height = int(latent_height)
        width = int(latent_width)
        if num_frames <= 1 or height <= 0 or width <= 0:
            return None
        expected_tokens = num_frames * height * width
        total_tokens = int(features.shape[1])
        if total_tokens < expected_tokens:
            return None
        if total_tokens != expected_tokens:
            features = features[:, :expected_tokens]
        return features.reshape(features.shape[0], num_frames, height, width, features.shape[-1])

    @staticmethod
    def _neighbor_weighted_cosine(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        *,
        num_neighbors: int,
        temporal_tau: float,
        motion_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sim = torch.bmm(student_frames, teacher_frames.transpose(1, 2))
        num_frames = sim.shape[1]
        tau = max(float(temporal_tau), 1e-6)
        total = sim.new_zeros(())
        normalizer = sim.new_zeros(())

        for delta in range(0, max(0, int(num_neighbors)) + 1):
            weight = 1.0 if delta == 0 else math.exp(-float(delta) / tau)
            if delta == 0:
                diag = sim.diagonal(dim1=1, dim2=2)
                if motion_weights is None:
                    total = total + diag.sum()
                    normalizer = normalizer + sim.new_tensor(diag.numel() * weight)
                else:
                    cast_weights = motion_weights.to(device=diag.device, dtype=diag.dtype)
                    total = total + weight * (diag * cast_weights).sum()
                    normalizer = normalizer + weight * cast_weights.sum()
                continue

            forward = sim.diagonal(offset=delta, dim1=1, dim2=2)
            backward = sim.diagonal(offset=-delta, dim1=1, dim2=2)
            if motion_weights is None:
                total = total + weight * (forward.sum() + backward.sum())
                normalizer = normalizer + sim.new_tensor(
                    sim.shape[0] * (forward.shape[-1] + backward.shape[-1]) * weight
                )
            else:
                forward_weights = motion_weights[:, :-delta].to(device=forward.device, dtype=forward.dtype)
                backward_weights = motion_weights[:, delta:].to(device=backward.device, dtype=backward.dtype)
                total = total + weight * (forward * forward_weights).sum()
                total = total + weight * (backward * backward_weights).sum()
                normalizer = normalizer + weight * (forward_weights.sum() + backward_weights.sum())

        if normalizer.item() <= 0.0:
            return sim.diagonal(dim1=1, dim2=2).mean()
        return total / normalizer

    @staticmethod
    def _normalize_motion_weights(motion: torch.Tensor, strength: float) -> torch.Tensor:
        if float(strength) <= 0.0:
            return torch.ones_like(motion)
        motion = motion.to(dtype=torch.float32)
        baseline = motion.mean()
        if not torch.isfinite(baseline) or float(baseline.item()) <= 1e-8:
            return torch.ones_like(motion)
        normalized = motion / baseline.clamp_min(1e-8)
        weights = 1.0 + float(strength) * normalized
        return weights.to(dtype=motion.dtype)

    @staticmethod
    def _teacher_delta_motion_weights(
        teacher_frames: torch.Tensor,
        *,
        strength: float,
    ) -> torch.Tensor:
        if float(strength) <= 0.0:
            return torch.ones(teacher_frames.shape[:-1], device=teacher_frames.device, dtype=teacher_frames.dtype)
        if teacher_frames.shape[1] <= 1:
            return torch.ones(teacher_frames.shape[:-1], device=teacher_frames.device, dtype=teacher_frames.dtype)

        forward = (teacher_frames[:, 1:] - teacher_frames[:, :-1]).pow(2).mean(dim=-1)
        motion = torch.zeros(teacher_frames.shape[:-1], device=teacher_frames.device, dtype=teacher_frames.dtype)
        motion[:, :-1] = motion[:, :-1] + forward
        motion[:, 1:] = motion[:, 1:] + forward
        return SelfFlowModule._normalize_motion_weights(motion, strength)

    @staticmethod
    def _neighbor_weighted_local_patch_cosine(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        *,
        num_neighbors: int,
        temporal_tau: float,
        spatial_radius: int,
        patch_match_mode: str,
        patch_match_temperature: float,
        motion_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if spatial_radius <= 0:
            flat_student = student_frames.reshape(
                student_frames.shape[0], student_frames.shape[1], student_frames.shape[2] * student_frames.shape[3], student_frames.shape[4]
            )
            flat_teacher = teacher_frames.reshape(
                teacher_frames.shape[0], teacher_frames.shape[1], teacher_frames.shape[2] * teacher_frames.shape[3], teacher_frames.shape[4]
            )
            sim = torch.einsum("btnd,bsnd->btsn", flat_student, flat_teacher)
            tau = max(float(temporal_tau), 1e-6)
            total = sim.new_zeros(())
            normalizer = sim.new_zeros(())

            def _reduce(values: torch.Tensor, weight: float, weights_slice: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
                local_values = values.mean(dim=-1)
                if weights_slice is None:
                    return weight * local_values.sum(), sim.new_tensor(local_values.numel() * weight)
                weights_view = weights_slice.to(device=local_values.device, dtype=local_values.dtype).reshape_as(local_values)
                return (
                    weight * (local_values * weights_view).sum(),
                    weight * weights_view.sum(),
                )

            for delta in range(0, max(0, int(num_neighbors)) + 1):
                weight = 1.0 if delta == 0 else math.exp(-float(delta) / tau)
                if delta == 0:
                    diag = sim.diagonal(dim1=1, dim2=2).permute(0, 2, 1)
                    part_total, part_norm = _reduce(diag, weight, motion_weights)
                    total = total + part_total
                    normalizer = normalizer + part_norm
                    continue

                forward = sim.diagonal(offset=delta, dim1=1, dim2=2).permute(0, 2, 1)
                backward = sim.diagonal(offset=-delta, dim1=1, dim2=2).permute(0, 2, 1)
                forward_weights = None if motion_weights is None else motion_weights[:, :-delta]
                backward_weights = None if motion_weights is None else motion_weights[:, delta:]
                part_total, part_norm = _reduce(forward, weight, forward_weights)
                total = total + part_total
                normalizer = normalizer + part_norm
                part_total, part_norm = _reduce(backward, weight, backward_weights)
                total = total + part_total
                normalizer = normalizer + part_norm

            if normalizer.item() <= 0.0:
                return sim.diagonal(dim1=1, dim2=2).mean()
            return total / normalizer

        batch_size, num_frames, height, width, channels = teacher_frames.shape
        kernel_size = 2 * int(spatial_radius) + 1
        teacher_bt = teacher_frames.permute(0, 1, 4, 2, 3).reshape(batch_size * num_frames, channels, height, width)
        teacher_neighborhoods = F.unfold(teacher_bt, kernel_size=kernel_size, padding=int(spatial_radius))
        neighborhood_size = kernel_size * kernel_size
        teacher_neighborhoods = teacher_neighborhoods.reshape(
            batch_size, num_frames, channels, neighborhood_size, height, width
        ).permute(0, 1, 4, 5, 3, 2)

        tau = max(float(temporal_tau), 1e-6)
        total = student_frames.new_zeros(())
        normalizer = student_frames.new_zeros(())

        def _accumulate(
            student_slice: torch.Tensor,
            teacher_slice: torch.Tensor,
            weight: float,
            weights_slice: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            similarities = (student_slice.unsqueeze(-2) * teacher_slice).sum(dim=-1)
            if patch_match_mode == "soft":
                temperature = max(float(patch_match_temperature), 1e-6)
                attn = torch.softmax(similarities / temperature, dim=-1)
                matched = (attn * similarities).sum(dim=-1)
            else:
                matched = similarities.max(dim=-1).values
            if weights_slice is None:
                return weight * matched.sum(), student_frames.new_tensor(matched.numel() * weight)
            local_weights = weights_slice.to(device=matched.device, dtype=matched.dtype)
            return weight * (matched * local_weights).sum(), weight * local_weights.sum()

        for delta in range(0, max(0, int(num_neighbors)) + 1):
            weight = 1.0 if delta == 0 else math.exp(-float(delta) / tau)
            if delta == 0:
                delta_total, delta_norm = _accumulate(student_frames, teacher_neighborhoods, weight, motion_weights)
                total = total + delta_total
                normalizer = normalizer + delta_norm
                continue

            forward_weights = None if motion_weights is None else motion_weights[:, :-delta]
            backward_weights = None if motion_weights is None else motion_weights[:, delta:]
            forward_total, forward_norm = _accumulate(
                student_frames[:, :-delta], teacher_neighborhoods[:, delta:], weight, forward_weights
            )
            backward_total, backward_norm = _accumulate(
                student_frames[:, delta:], teacher_neighborhoods[:, :-delta], weight, backward_weights
            )
            total = total + forward_total + backward_total
            normalizer = normalizer + forward_norm + backward_norm

        if normalizer.item() <= 0.0:
            return student_frames.new_zeros(())
        return total / normalizer

    @staticmethod
    def _multi_step_delta_cosine(
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
        *,
        delta_num_steps: int,
        temporal_tau: float,
        motion_weights: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        max_step = min(max(1, int(delta_num_steps)), max(student_frames.shape[1] - 1, 0), max(teacher_frames.shape[1] - 1, 0))
        if max_step <= 0:
            return None

        total = student_frames.new_zeros(())
        normalizer = student_frames.new_zeros(())
        tau = max(float(temporal_tau), 1e-6)

        for step in range(1, max_step + 1):
            weight = math.exp(-float(step - 1) / tau)
            student_delta = F.normalize(student_frames[:, step:] - student_frames[:, :-step], dim=-1)
            teacher_delta = F.normalize(teacher_frames[:, step:] - teacher_frames[:, :-step], dim=-1)
            cosine = F.cosine_similarity(student_delta, teacher_delta, dim=-1)
            step_weights = None if motion_weights is None else motion_weights[:, step:]
            if step_weights is None:
                total = total + weight * cosine.sum()
                normalizer = normalizer + student_frames.new_tensor(cosine.numel() * weight)
            else:
                cast_weights = step_weights.to(device=cosine.device, dtype=cosine.dtype)
                total = total + weight * (cosine * cast_weights).sum()
                normalizer = normalizer + weight * cast_weights.sum()

        if normalizer.item() <= 0.0:
            return None
        return total / normalizer

    def compute_loss_from_cached_features(
        self,
        *,
        num_latent_frames: Optional[int] = None,
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.projector is None or not self.has_active_loss:
            return None
        if self._student_features is None or self._teacher_features is None:
            return None

        student_feat = self._student_features
        teacher_feat = self._teacher_features
        if teacher_feat.device != student_feat.device or teacher_feat.dtype != student_feat.dtype:
            teacher_feat = teacher_feat.to(device=student_feat.device, dtype=student_feat.dtype, non_blocking=True)
        if student_feat.shape[1] != teacher_feat.shape[1]:
            min_tokens = min(student_feat.shape[1], teacher_feat.shape[1])
            student_feat = student_feat[:, :min_tokens]
            teacher_feat = teacher_feat[:, :min_tokens]
            if token_mask is not None and token_mask.shape[1] > min_tokens:
                token_mask = token_mask[:, :min_tokens]

        student_proj = self.projector(student_feat)
        teacher_feat = teacher_feat.detach()

        student_proj_norm = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(teacher_feat, dim=-1)

        # Optionally focus the rep loss on masked (higher-noise) tokens only.
        if self.config.mask_focus_loss and token_mask is not None:
            valid = token_mask.to(device=student_proj_norm.device)  # [B, T]
            feat_tokens = student_proj_norm.shape[1]
            mask_tokens = valid.shape[1]
            if mask_tokens != feat_tokens:
                if mask_tokens > feat_tokens:
                    valid = valid[:, :feat_tokens]
                else:
                    # mask shorter than features — only score tokens that are masked
                    student_proj_norm = student_proj_norm[:, :mask_tokens]
                    teacher_norm = teacher_norm[:, :mask_tokens]
            if valid.any():
                cosine = F.cosine_similarity(student_proj_norm[valid], teacher_norm[valid], dim=-1).mean()
            else:
                cosine = F.cosine_similarity(student_proj_norm, teacher_norm, dim=-1).mean()
        else:
            cosine = F.cosine_similarity(student_proj_norm, teacher_norm, dim=-1).mean()

        self._last_cosine = float(cosine.detach().item())
        loss = cosine.new_zeros(())
        applied_terms = 0
        if self._current_lambda_self_flow > 0.0:
            loss = loss + self._loss_from_cosine(cosine) * self._current_lambda_self_flow
            applied_terms += 1

        # Audio representation alignment loss
        if (
            self._current_lambda_audio > 0.0
            and self.audio_projector is not None
            and self._student_audio_features is not None
            and self._teacher_audio_features is not None
        ):
            audio_student = self._student_audio_features
            audio_teacher = self._teacher_audio_features
            if audio_teacher.device != audio_student.device or audio_teacher.dtype != audio_student.dtype:
                audio_teacher = audio_teacher.to(device=audio_student.device, dtype=audio_student.dtype, non_blocking=True)
            if audio_student.shape[1] != audio_teacher.shape[1]:
                min_t = min(audio_student.shape[1], audio_teacher.shape[1])
                audio_student = audio_student[:, :min_t]
                audio_teacher = audio_teacher[:, :min_t]
            audio_proj = self.audio_projector(audio_student)
            audio_teacher = audio_teacher.detach()
            audio_proj_norm = F.normalize(audio_proj, dim=-1)
            audio_teacher_norm = F.normalize(audio_teacher, dim=-1)
            audio_cosine = F.cosine_similarity(audio_proj_norm, audio_teacher_norm, dim=-1).mean()
            self._last_audio_cosine = float(audio_cosine.detach().item())
            loss = loss + self._loss_from_cosine(audio_cosine) * self._current_lambda_audio
            applied_terms += 1

        temporal_mode = str(self.config.temporal_mode).lower()
        temporal_granularity = str(self.config.temporal_granularity).lower()
        motion_weighting = str(self.config.motion_weighting).lower()
        # Temporal losses operate in the original feature space (not projected) so that
        # frame-to-frame deltas reflect the transformer's actual spatiotemporal representations
        # rather than the learned projection.
        temporal_student = self._reshape_temporal_features(student_feat, num_latent_frames)
        temporal_teacher = self._reshape_temporal_features(teacher_feat, num_latent_frames)
        temporal_student_grid = self._reshape_temporal_grid(
            student_feat,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        temporal_teacher_grid = self._reshape_temporal_grid(
            teacher_feat,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        temporal_motion_weights = None
        temporal_motion_grid_weights = None
        if motion_weighting == "teacher_delta":
            if temporal_teacher is not None:
                temporal_motion_weights = self._teacher_delta_motion_weights(
                    temporal_teacher,
                    strength=self.config.motion_weight_strength,
                )
            if temporal_teacher_grid is not None:
                temporal_motion_grid_weights = self._teacher_delta_motion_weights(
                    temporal_teacher_grid,
                    strength=self.config.motion_weight_strength,
                )

        if (
            temporal_mode in {"frame", "hybrid"}
            and self.current_lambda_temporal > 0.0
            and temporal_student is not None
            and temporal_teacher is not None
        ):
            if temporal_granularity == "patch":
                if temporal_student_grid is not None and temporal_teacher_grid is not None:
                    student_frames = F.normalize(temporal_student_grid, dim=-1)
                    teacher_frames = F.normalize(temporal_teacher_grid, dim=-1)
                    frame_cosine = self._neighbor_weighted_local_patch_cosine(
                        student_frames,
                        teacher_frames,
                        num_neighbors=self.config.num_neighbors,
                        temporal_tau=self.config.temporal_tau,
                        spatial_radius=self.config.patch_spatial_radius,
                        patch_match_mode=self.config.patch_match_mode,
                        patch_match_temperature=self.config.patch_match_temperature,
                        motion_weights=temporal_motion_grid_weights,
                    )
                else:
                    student_frames = F.normalize(temporal_student, dim=-1)
                    teacher_frames = F.normalize(temporal_teacher, dim=-1)
                    flat_motion_weights = None
                    if temporal_motion_weights is not None:
                        flat_motion_weights = temporal_motion_weights.mean(dim=-1).unsqueeze(2)
                    frame_cosine = self._neighbor_weighted_local_patch_cosine(
                        student_frames.reshape(student_frames.shape[0], student_frames.shape[1], 1, student_frames.shape[2], student_frames.shape[3]),
                        teacher_frames.reshape(teacher_frames.shape[0], teacher_frames.shape[1], 1, teacher_frames.shape[2], teacher_frames.shape[3]),
                        num_neighbors=self.config.num_neighbors,
                        temporal_tau=self.config.temporal_tau,
                        spatial_radius=0,
                        patch_match_mode=self.config.patch_match_mode,
                        patch_match_temperature=self.config.patch_match_temperature,
                        motion_weights=flat_motion_weights,
                    )
            else:
                student_frames = F.normalize(temporal_student.mean(dim=2), dim=-1)
                teacher_frames = F.normalize(temporal_teacher.mean(dim=2), dim=-1)
                frame_motion_weights = None
                if temporal_motion_weights is not None:
                    frame_motion_weights = temporal_motion_weights.mean(dim=-1)
                frame_cosine = self._neighbor_weighted_cosine(
                    student_frames,
                    teacher_frames,
                    num_neighbors=self.config.num_neighbors,
                    temporal_tau=self.config.temporal_tau,
                    motion_weights=frame_motion_weights,
                )
            self._last_frame_cosine = float(frame_cosine.detach().item())
            loss = loss + self._loss_from_cosine(frame_cosine) * self.current_lambda_temporal
            applied_terms += 1

        if (
            temporal_mode in {"delta", "hybrid"}
            and self.current_lambda_delta > 0.0
            and temporal_student is not None
            and temporal_teacher is not None
            and temporal_student.shape[1] > 1
            and temporal_teacher.shape[1] > 1
        ):
            if temporal_granularity == "patch":
                delta_cosine = self._multi_step_delta_cosine(
                    temporal_student,
                    temporal_teacher,
                    delta_num_steps=self.config.delta_num_steps,
                    temporal_tau=self.config.temporal_tau,
                    motion_weights=temporal_motion_weights,
                )
            else:
                student_frames = temporal_student.mean(dim=2)
                teacher_frames = temporal_teacher.mean(dim=2)
                frame_motion_weights = None
                if temporal_motion_weights is not None:
                    frame_motion_weights = temporal_motion_weights.mean(dim=-1)
                delta_cosine = self._multi_step_delta_cosine(
                    student_frames,
                    teacher_frames,
                    delta_num_steps=self.config.delta_num_steps,
                    temporal_tau=self.config.temporal_tau,
                    motion_weights=frame_motion_weights,
                )
            if delta_cosine is not None and torch.isfinite(delta_cosine):
                self._last_delta_cosine = float(delta_cosine.detach().item())
                loss = loss + self._loss_from_cosine(delta_cosine) * self.current_lambda_delta
                applied_terms += 1

        if applied_terms == 0:
            return None

        if not torch.isfinite(loss):
            logger.warning("Self-Flow loss is non-finite (%.4g), skipping", loss.item())
            return None

        max_loss = float(self.config.max_loss)
        if max_loss > 0.0:
            loss_abs = float(loss.detach().abs().item())
            if loss_abs > max_loss:
                loss = loss * (max_loss / loss_abs)

        return loss

    def compute_loss(self, **_kwargs) -> Optional[torch.Tensor]:
        # Backward-compatible shim: loss is now computed from already-cached student/teacher features.
        return self.compute_loss_from_cached_features(
            num_latent_frames=_kwargs.get("num_latent_frames"),
            latent_height=_kwargs.get("latent_height"),
            latent_width=_kwargs.get("latent_width"),
            token_mask=_kwargs.get("token_mask"),
        )

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()
        self.cleanup_step()

    def state_dict(self) -> Dict[str, Any]:
        if self.projector is None:
            return {}
        sd = self.projector.state_dict()
        if self.audio_projector is not None:
            for k, v in self.audio_projector.state_dict().items():
                sd[f"audio.{k}"] = v
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if self.projector is not None and sd:
            # Split video and audio projector weights
            video_sd = {k: v for k, v in sd.items() if not k.startswith("audio.")}
            audio_sd = {k[len("audio."):]: v for k, v in sd.items() if k.startswith("audio.")}
            if video_sd:
                self.projector.load_state_dict(video_sd)
                logger.info("Self-Flow: loaded video projector weights (%d tensors)", len(video_sd))
            if audio_sd and self.audio_projector is not None:
                self.audio_projector.load_state_dict(audio_sd)
                logger.info("Self-Flow: loaded audio projector weights (%d tensors)", len(audio_sd))

    def teacher_state_dict(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "__self_flow_step_counter__": torch.tensor([int(self._step_counter)], dtype=torch.int64)
        }
        for name, tensor in self._shadow_params.items():
            out[f"shadow::{name}"] = tensor.detach().clone().to(device="cpu")
        return out

    def load_teacher_state_dict(self, sd: Dict[str, Any]) -> None:
        if not sd:
            return
        if str(self.config.teacher_mode).lower() == "base":
            logger.warning(
                "Self-Flow: teacher_mode=base but EMA teacher state was found in checkpoint. "
                "Ignoring loaded EMA state (base mode uses frozen base weights, not EMA). "
                "This is expected if you switched from ema/partial_ema to base mode."
            )
            return
        step_tensor = sd.get("__self_flow_step_counter__")
        if isinstance(step_tensor, torch.Tensor) and step_tensor.numel() > 0:
            self._step_counter = int(step_tensor.flatten()[0].item())

        restored: Dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if not isinstance(value, torch.Tensor):
                continue
            if not key.startswith("shadow::"):
                continue
            restored[key[len("shadow::") :]] = value.detach().clone()

        if restored:
            self._shadow_params = restored
            logger.info("Self-Flow: loaded EMA teacher state (%d tensors)", len(restored))
