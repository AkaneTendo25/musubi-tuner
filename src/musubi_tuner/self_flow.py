"""Self-Flow helper for LTX-2 training.

Implements the Self-Flow training regularizer:
- Dual-timestep noising is performed by the trainer.
- This module handles feature alignment loss with an EMA teacher over LoRA params.
"""

from __future__ import annotations

import logging
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
    lambda_self_flow: float = 0.1
    mask_ratio: float = 0.10
    teacher_momentum: float = 0.999
    teacher_update_interval: int = 1
    projector_hidden_multiplier: int = 1
    loss_type: str = "negative_cosine"  # "negative_cosine" | "one_minus_cosine"
    dual_timestep: bool = True
    tokenwise_timestep: bool = True
    offload_teacher_features: bool = False


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
        self._hooks: list = []

        self._capture_mode: str = "idle"  # "student" | "teacher" | "idle"
        self._student_features: Optional[torch.Tensor] = None
        self._teacher_features: Optional[torch.Tensor] = None

        self._shadow_params: Dict[str, torch.Tensor] = {}
        self._step_counter: int = 0
        self._last_cosine: Optional[float] = None

    @property
    def last_cosine(self) -> Optional[float]:
        return self._last_cosine

    def _get_blocks(self) -> tuple[list[nn.Module], int]:
        blocks = getattr(self.transformer, "transformer_blocks", None)
        if blocks is None:
            raise ValueError("Self-Flow requires transformer.transformer_blocks")
        block_list = list(blocks)
        return block_list, len(block_list)

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

    def setup(self, device: torch.device, dtype: torch.dtype) -> None:
        blocks, depth = self._get_blocks()
        if not (0 <= self.config.student_block_idx < depth):
            raise ValueError(
                f"student_block_idx={self.config.student_block_idx} out of range (model has {depth} blocks)"
            )
        if not (0 <= self.config.teacher_block_idx < depth):
            raise ValueError(
                f"teacher_block_idx={self.config.teacher_block_idx} out of range (model has {depth} blocks)"
            )
        if self.config.teacher_block_idx <= self.config.student_block_idx:
            raise ValueError(
                "teacher_block_idx must be > student_block_idx for Self-Flow"
            )

        inner_dim = int(getattr(self.transformer, "inner_dim", 0))
        if inner_dim <= 0:
            raise ValueError("Self-Flow could not resolve transformer.inner_dim")
        hidden_dim = inner_dim * max(1, int(self.config.projector_hidden_multiplier))
        self.projector = nn.Sequential(
            nn.Linear(inner_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, inner_dim),
        ).to(device=device, dtype=dtype)

        self._install_hooks(blocks)
        logger.info(
            "Self-Flow ready: student_block=%d teacher_block=%d mask_ratio=%.3f lambda=%.4f "
            "momentum=%.4f dual_timestep=%s tokenwise_timestep=%s",
            self.config.student_block_idx,
            self.config.teacher_block_idx,
            self.config.mask_ratio,
            self.config.lambda_self_flow,
            self.config.teacher_momentum,
            str(self.config.dual_timestep).lower(),
            str(self.config.tokenwise_timestep).lower(),
        )

    def _install_hooks(self, blocks: list[nn.Module]) -> None:
        def _extract_video_tensor(output: Any) -> Optional[torch.Tensor]:
            if isinstance(output, tuple) and len(output) >= 1:
                video_out = output[0]
                if hasattr(video_out, "x") and torch.is_tensor(video_out.x):
                    return video_out.x
            return None

        def _student_hook(_module, _inputs, output):
            if self._capture_mode != "student":
                return
            tensor = _extract_video_tensor(output)
            if tensor is not None:
                self._student_features = tensor

        def _teacher_hook(_module, _inputs, output):
            if self._capture_mode != "teacher":
                return
            tensor = _extract_video_tensor(output)
            if tensor is not None:
                self._teacher_features = tensor.detach()

        self._hooks.append(
            blocks[self.config.student_block_idx].register_forward_hook(_student_hook)
        )
        self._hooks.append(
            blocks[self.config.teacher_block_idx].register_forward_hook(_teacher_hook)
        )

    def init_teacher(self, network: nn.Module) -> None:
        self._shadow_params.clear()
        for name, param in network.named_parameters():
            if not param.requires_grad:
                continue
            self._shadow_params[name] = param.detach().clone()
        logger.info("Self-Flow: initialized EMA teacher with %d tensors", len(self._shadow_params))

    def update_teacher(self, network: nn.Module) -> None:
        if not self._shadow_params:
            return
        self._step_counter += 1
        if self._step_counter % max(1, int(self.config.teacher_update_interval)) != 0:
            return
        momentum = float(self.config.teacher_momentum)
        with torch.no_grad():
            for name, param in network.named_parameters():
                shadow_name = self._resolve_shadow_name(name, self._shadow_params)
                if shadow_name is None:
                    continue
                shadow = self._shadow_params[shadow_name]
                if shadow.device != param.device or shadow.dtype != param.dtype:
                    shadow = shadow.to(device=param.device, dtype=param.dtype)
                    self._shadow_params[shadow_name] = shadow
                shadow.mul_(momentum).add_(param.detach(), alpha=1.0 - momentum)

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
                    shadow = shadow.to(device=param.device, dtype=param.dtype)
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

    def cleanup_step(self) -> None:
        self._capture_mode = "idle"
        self._student_features = None
        self._teacher_features = None

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        if self.projector is None:
            return []
        return list(self.projector.parameters())

    def prepare_teacher_features(
        self,
        *,
        accelerator,
        transformer: nn.Module,
        network: nn.Module,
        teacher_model_input: torch.Tensor,
        teacher_timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        frame_rate: int | float,
        transformer_options: Dict[str, Any],
    ) -> None:
        if self.projector is None or self.config.lambda_self_flow <= 0.0:
            return
        self._teacher_features = None
        backups = self._swap_in_teacher(network)
        prev_training = bool(getattr(transformer, "training", False))
        try:
            self._capture_mode = "teacher"
            if prev_training:
                transformer.eval()
            with torch.no_grad(), accelerator.autocast():
                _ = transformer(
                    teacher_model_input,
                    timestep=teacher_timesteps,
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

        if (
            self._teacher_features is not None
            and bool(self.config.offload_teacher_features)
            and self._teacher_features.device.type != "cpu"
        ):
            self._teacher_features = self._teacher_features.to(device="cpu", non_blocking=False)

    def compute_loss_from_cached_features(self) -> Optional[torch.Tensor]:
        if self.projector is None or self.config.lambda_self_flow <= 0.0:
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

        student_proj = self.projector(student_feat)
        student_proj = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(teacher_feat.detach(), dim=-1)
        cosine = F.cosine_similarity(student_proj, teacher_norm, dim=-1).mean()
        self._last_cosine = float(cosine.detach().item())

        if self.config.loss_type == "one_minus_cosine":
            rep_loss = 1.0 - cosine
        else:
            rep_loss = -cosine
        loss = rep_loss * float(self.config.lambda_self_flow)

        if not torch.isfinite(loss):
            logger.warning("Self-Flow loss is non-finite (%.4g), skipping", loss.item())
            return None
        return loss

    def compute_loss(self, **_kwargs) -> Optional[torch.Tensor]:
        # Backward-compatible shim: loss is now computed from already-cached student/teacher features.
        return self.compute_loss_from_cached_features()

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
        return self.projector.state_dict()

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if self.projector is not None and sd:
            self.projector.load_state_dict(sd)
            logger.info("Self-Flow: loaded projector weights (%d tensors)", len(sd))

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
