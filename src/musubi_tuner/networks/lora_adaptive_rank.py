import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("musubi_tuner.networks.lora")


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if value is None:
        return default
    return bool(value)


def _coerce_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _coerce_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _coerce_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _coerce_lower_str(value: Any, *, allow_disabled: bool = False) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if allow_disabled and normalized in ("", "none", "false", "off"):
        return None
    return normalized


def parse_adaptive_rank_network_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "adaptive_rank": _coerce_bool(kwargs.get("adaptive_rank", False), default=False),
        "adaptive_rank_target": _coerce_int(kwargs.get("adaptive_rank_target", None)),
        "video_adaptive_rank_target": _coerce_int(kwargs.get("video_adaptive_rank_target", None)),
        "audio_adaptive_rank_target": _coerce_int(kwargs.get("audio_adaptive_rank_target", None)),
        "cross_modal_adaptive_rank_target": _coerce_int(kwargs.get("cross_modal_adaptive_rank_target", None)),
        "adaptive_rank_quantile": _coerce_float(kwargs.get("adaptive_rank_quantile", None)),
        "adaptive_rank_weight": _coerce_float(kwargs.get("adaptive_rank_weight", None)),
        "video_adaptive_rank_weight": _coerce_float(kwargs.get("video_adaptive_rank_weight", None)),
        "audio_adaptive_rank_weight": _coerce_float(kwargs.get("audio_adaptive_rank_weight", None)),
        "cross_modal_adaptive_rank_weight": _coerce_float(kwargs.get("cross_modal_adaptive_rank_weight", None)),
        "adaptive_rank_min_rank": _coerce_int(kwargs.get("adaptive_rank_min_rank", None)),
        "adaptive_rank_init_rank": _coerce_int(kwargs.get("adaptive_rank_init_rank", None)),
        "adaptive_rank_budget": _coerce_float(kwargs.get("adaptive_rank_budget", None)),
        "adaptive_rank_budget_ratio": _coerce_float(kwargs.get("adaptive_rank_budget_ratio", None)),
        "adaptive_rank_budget_weight": _coerce_float(kwargs.get("adaptive_rank_budget_weight", None)),
        "adaptive_rank_schedule": _coerce_lower_str(kwargs.get("adaptive_rank_schedule", None), allow_disabled=True),
        "adaptive_rank_schedule_start": _coerce_float(kwargs.get("adaptive_rank_schedule_start", None)),
        "adaptive_rank_schedule_end": _coerce_float(kwargs.get("adaptive_rank_schedule_end", None)),
        "adaptive_rank_estimate": _coerce_bool(kwargs.get("adaptive_rank_estimate", None), default=False),
        "adaptive_rank_estimate_report": _coerce_str(kwargs.get("adaptive_rank_estimate_report", None)),
        "adaptive_rank_estimate_key": _coerce_str(kwargs.get("adaptive_rank_estimate_key", None)),
        "adaptive_rank_estimate_apply": _coerce_lower_str(kwargs.get("adaptive_rank_estimate_apply", None)),
        "adaptive_rank_estimate_reallocate_interval": _coerce_int(
            kwargs.get("adaptive_rank_estimate_reallocate_interval", None)
        ),
        "adaptive_rank_estimate_reallocate_start": _coerce_float(
            kwargs.get("adaptive_rank_estimate_reallocate_start", None)
        ),
        "adaptive_rank_estimate_reallocate_apply": _coerce_lower_str(
            kwargs.get("adaptive_rank_estimate_reallocate_apply", None)
        ),
        "adaptive_rank_finalize_start": _coerce_float(kwargs.get("adaptive_rank_finalize_start", None)),
        "adaptive_rank_finalize_recover_steps": _coerce_int(kwargs.get("adaptive_rank_finalize_recover_steps", None)),
        "adaptive_rank_finalize_recover_warmup_steps": _coerce_int(
            kwargs.get("adaptive_rank_finalize_recover_warmup_steps", None)
        ),
        "adaptive_rank_finalize_recover_lr_scale": _coerce_float(
            kwargs.get("adaptive_rank_finalize_recover_lr_scale", None)
        ),
        "adaptive_rank_finalize_recover_scheduler": _coerce_lower_str(
            kwargs.get("adaptive_rank_finalize_recover_scheduler", None)
        ),
        "adaptive_rank_hard_prune": _coerce_bool(kwargs.get("adaptive_rank_hard_prune", None), default=False),
        "adaptive_rank_hard_prune_start": _coerce_float(kwargs.get("adaptive_rank_hard_prune_start", None)),
        "adaptive_rank_hard_prune_interval": _coerce_int(kwargs.get("adaptive_rank_hard_prune_interval", None)),
        "adaptive_rank_hard_prune_min_delta": _coerce_int(kwargs.get("adaptive_rank_hard_prune_min_delta", None)),
    }


class AdaptiveRankLoRAModuleMixin:
    def _init_adaptive_rank_module_state(self, kwargs: Dict[str, Any]) -> None:
        self.adaptive_rank_max_rank = int(self.lora_dim)
        self.split_dims = kwargs.get("split_dims", None)
        self.module_path = kwargs.get("module_path", None)
        self.adaptive_rank = _coerce_bool(kwargs.get("adaptive_rank", False), default=False)
        self.was_adaptive_rank = bool(self.adaptive_rank)

        adaptive_rank_quantile = kwargs.get("adaptive_rank_quantile", None)
        if adaptive_rank_quantile is None:
            adaptive_rank_quantile = 0.9
        self.adaptive_rank_quantile = float(adaptive_rank_quantile)
        self.adaptive_rank_quantile = min(max(self.adaptive_rank_quantile, 1e-4), 1.0 - 1e-6)

        adaptive_rank_min_rank = kwargs.get("adaptive_rank_min_rank", None)
        if adaptive_rank_min_rank is None:
            adaptive_rank_min_rank = 1
        self.adaptive_rank_min_rank = max(1, int(adaptive_rank_min_rank))

        target_rank = kwargs.get("adaptive_rank_target", None)
        if target_rank is None:
            target_rank = self.lora_dim
        self.adaptive_rank_target = max(self.adaptive_rank_min_rank, min(self.lora_dim, int(target_rank)))

        adaptive_rank_weight = kwargs.get("adaptive_rank_weight", None)
        if adaptive_rank_weight is None:
            adaptive_rank_weight = 1e-4 if self.adaptive_rank else 0.0
        self.adaptive_rank_weight = float(adaptive_rank_weight)

        init_rank = kwargs.get("adaptive_rank_init_rank", None)
        if init_rank is None:
            init_rank = self.lora_dim
        init_rank = max(self.adaptive_rank_min_rank, min(self.lora_dim, int(init_rank)))

        if self.adaptive_rank:
            init_lambda = self._lambda_from_rank(init_rank, self.adaptive_rank_quantile)
            self.rank_lambda_param = torch.nn.Parameter(
                torch.tensor(self._inverse_softplus(init_lambda), dtype=torch.float32)
            )

    @staticmethod
    def _lambda_from_rank(rank: int, quantile: float) -> float:
        rank = max(float(rank), 1e-6)
        quantile = min(max(float(quantile), 1e-4), 1.0 - 1e-6)
        return -math.log(1.0 - quantile) / rank

    @staticmethod
    def _inverse_softplus(x: float) -> float:
        x = max(float(x), 1e-6)
        return math.log(math.expm1(x))

    def _adaptive_lambda(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.rank_lambda_param) + 1e-6

    def get_effective_rank(self) -> int:
        if not self.adaptive_rank:
            return self.lora_dim
        lam = float(self._adaptive_lambda().detach().cpu().item())
        rank = math.ceil(-math.log(1.0 - self.adaptive_rank_quantile) / lam)
        return max(self.adaptive_rank_min_rank, min(self.lora_dim, rank))

    def get_expected_rank_tensor(self) -> torch.Tensor:
        if not self.adaptive_rank:
            return self.alpha.new_tensor(float(self.lora_dim), dtype=torch.float32)
        expected_rank = -math.log(1.0 - self.adaptive_rank_quantile) / self._adaptive_lambda()
        return expected_rank.clamp(min=float(self.adaptive_rank_min_rank), max=float(self.lora_dim))

    def get_rank_regularization_loss(self, target_rank_override: Optional[float] = None) -> Optional[torch.Tensor]:
        if not self.adaptive_rank or self.adaptive_rank_weight <= 0:
            return None
        target_rank = self.adaptive_rank_target if target_rank_override is None else float(target_rank_override)
        target_rank = min(float(self.lora_dim), max(float(self.adaptive_rank_min_rank), float(target_rank)))
        target_lambda = self._lambda_from_rank(target_rank, self.adaptive_rank_quantile)
        return torch.abs(self._adaptive_lambda() - target_lambda) * self.adaptive_rank_weight

    def _adaptive_rank_weights(self, rank_dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.adaptive_rank:
            return torch.ones(rank_dim, dtype=dtype, device=device)

        positions = torch.arange(rank_dim, dtype=dtype, device=device)
        lam = self._adaptive_lambda().to(device=device, dtype=dtype)
        weights = (1.0 - torch.exp(-lam)) * torch.exp(-lam * positions)
        active_rank = self.get_effective_rank()
        if active_rank < rank_dim:
            weights = weights.clone()
            weights[active_rank:] = 0
        return weights

    @staticmethod
    def _reshape_rank_weights(weights: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if like.dim() == 2:
            return weights
        if like.shape[-1] == weights.shape[0]:
            shape = [1] * like.dim()
            shape[-1] = weights.shape[0]
            return weights.view(*shape)
        if like.dim() >= 2 and like.shape[1] == weights.shape[0]:
            shape = [1] * like.dim()
            shape[1] = weights.shape[0]
            return weights.view(*shape)
        raise ValueError(f"Cannot broadcast adaptive rank weights of shape {weights.shape} to tensor {like.shape}")

    def _export_alpha(self, export_rank: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        alpha_value = self.alpha.detach().to(device=device, dtype=torch.float32)
        export_scale = float(export_rank) / float(max(self.lora_dim, 1))
        return alpha_value * export_scale

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict: Dict[str, torch.Tensor] = {}
        export_rank = self.get_effective_rank() if self.adaptive_rank else self.lora_dim

        if self.split_dims is None:
            down_weight = self.lora_down.weight.detach().clone()
            up_weight = self.lora_up.weight.detach().clone()
            if self.adaptive_rank:
                rank_weights = self._adaptive_rank_weights(
                    self.lora_dim, dtype=down_weight.dtype, device=down_weight.device
                )[:export_rank].sqrt()
                down_weight = down_weight[:export_rank]
                up_weight = up_weight[:, :export_rank]
                if down_weight.dim() == 2:
                    down_weight = down_weight * rank_weights[:, None]
                    up_weight = up_weight * rank_weights[None, :]
                else:
                    down_weight = down_weight * rank_weights[:, None, None, None]
                    up_weight = up_weight * rank_weights[None, :, None, None]
                alpha = self._export_alpha(export_rank, dtype=down_weight.dtype, device=down_weight.device)
            else:
                alpha = self.alpha.detach().clone().to(device=down_weight.device, dtype=torch.float32)

            state_dict[f"{self.lora_name}.lora_down.weight"] = down_weight
            state_dict[f"{self.lora_name}.lora_up.weight"] = up_weight
            state_dict[f"{self.lora_name}.alpha"] = alpha
            return state_dict

        rank_weights = None
        if self.adaptive_rank:
            rank_weights = self._adaptive_rank_weights(
                self.lora_dim, dtype=self.lora_down[0].weight.dtype, device=self.lora_down[0].weight.device
            )[:export_rank].sqrt()
            alpha = self._export_alpha(
                export_rank, dtype=self.lora_down[0].weight.dtype, device=self.lora_down[0].weight.device
            )
        else:
            alpha = self.alpha.detach().clone().to(device=self.lora_down[0].weight.device, dtype=torch.float32)

        for i, (lora_down, lora_up) in enumerate(zip(self.lora_down, self.lora_up)):
            down_weight = lora_down.weight.detach().clone()
            up_weight = lora_up.weight.detach().clone()
            if self.adaptive_rank:
                down_weight = down_weight[:export_rank] * rank_weights[:, None]
                up_weight = up_weight[:, :export_rank] * rank_weights[None, :]
            state_dict[f"{self.lora_name}.lora_down.{i}.weight"] = down_weight
            state_dict[f"{self.lora_name}.lora_up.{i}.weight"] = up_weight
        state_dict[f"{self.lora_name}.alpha"] = alpha
        return state_dict

    def hard_prune_to_static(self, min_delta: int = 1, force_static: bool = False) -> Optional[Dict[str, Any]]:
        if not self.adaptive_rank:
            return None

        current_rank = int(self.lora_dim)
        export_rank = int(self.get_effective_rank())
        min_delta = max(1, int(min_delta))
        if not force_static and current_rank - export_rank < min_delta:
            return None

        export_state = self.export_state_dict()
        alpha_tensor = export_state[f"{self.lora_name}.alpha"].detach().clone()

        if self.split_dims is None:
            down_weight = export_state[f"{self.lora_name}.lora_down.weight"].detach().clone()
            up_weight = export_state[f"{self.lora_name}.lora_up.weight"].detach().clone()

            old_down = self.lora_down
            old_up = self.lora_up
            if isinstance(old_down, torch.nn.Conv2d):
                new_down = torch.nn.Conv2d(
                    old_down.in_channels,
                    export_rank,
                    old_down.kernel_size,
                    old_down.stride,
                    old_down.padding,
                    bias=False,
                )
                new_up = torch.nn.Conv2d(
                    export_rank,
                    old_up.out_channels,
                    old_up.kernel_size,
                    old_up.stride,
                    old_up.padding,
                    bias=False,
                )
            else:
                new_down = torch.nn.Linear(old_down.in_features, export_rank, bias=False)
                new_up = torch.nn.Linear(export_rank, old_up.out_features, bias=False)

            new_down = new_down.to(device=old_down.weight.device, dtype=old_down.weight.dtype)
            new_up = new_up.to(device=old_up.weight.device, dtype=old_up.weight.dtype)
            with torch.no_grad():
                new_down.weight.copy_(down_weight.to(device=new_down.weight.device, dtype=new_down.weight.dtype))
                new_up.weight.copy_(up_weight.to(device=new_up.weight.device, dtype=new_up.weight.dtype))
            self.lora_down = new_down
            self.lora_up = new_up
        else:
            new_down_modules = []
            new_up_modules = []
            for i, (old_down, old_up) in enumerate(zip(self.lora_down, self.lora_up)):
                down_weight = export_state[f"{self.lora_name}.lora_down.{i}.weight"].detach().clone()
                up_weight = export_state[f"{self.lora_name}.lora_up.{i}.weight"].detach().clone()
                new_down = torch.nn.Linear(old_down.in_features, export_rank, bias=False).to(
                    device=old_down.weight.device, dtype=old_down.weight.dtype
                )
                new_up = torch.nn.Linear(export_rank, old_up.out_features, bias=False).to(
                    device=old_up.weight.device, dtype=old_up.weight.dtype
                )
                with torch.no_grad():
                    new_down.weight.copy_(down_weight.to(device=new_down.weight.device, dtype=new_down.weight.dtype))
                    new_up.weight.copy_(up_weight.to(device=new_up.weight.device, dtype=new_up.weight.dtype))
                new_down_modules.append(new_down)
                new_up_modules.append(new_up)
            self.lora_down = torch.nn.ModuleList(new_down_modules)
            self.lora_up = torch.nn.ModuleList(new_up_modules)

        self.lora_dim = export_rank
        self.alpha = alpha_tensor.to(dtype=torch.float32)
        self.scale = float(alpha_tensor.detach().cpu().item()) / float(max(export_rank, 1))
        self.adaptive_rank = False
        self.adaptive_rank_target = min(int(getattr(self, "adaptive_rank_target", export_rank)), export_rank)
        self.adaptive_rank_min_rank = min(int(getattr(self, "adaptive_rank_min_rank", 1)), export_rank)
        self.adaptive_rank_weight = 0.0
        if hasattr(self, "rank_lambda_param"):
            del self.rank_lambda_param

        return {
            "lora_name": str(self.lora_name),
            "module_path": str(getattr(self, "module_path", "") or ""),
            "old_rank": current_rank,
            "new_rank": export_rank,
            "shape_changed": bool(current_rank != export_rank),
        }

    def _rebuild_lora_modules_for_rank(self, new_rank: int) -> None:
        new_rank = max(1, int(new_rank))
        if self.split_dims is None:
            old_down = self.lora_down
            old_up = self.lora_up
            if isinstance(old_down, torch.nn.Conv2d):
                new_down = torch.nn.Conv2d(
                    old_down.in_channels,
                    new_rank,
                    old_down.kernel_size,
                    old_down.stride,
                    old_down.padding,
                    bias=False,
                )
                new_up = torch.nn.Conv2d(
                    new_rank,
                    old_up.out_channels,
                    old_up.kernel_size,
                    old_up.stride,
                    old_up.padding,
                    bias=False,
                )
            else:
                new_down = torch.nn.Linear(old_down.in_features, new_rank, bias=False)
                new_up = torch.nn.Linear(new_rank, old_up.out_features, bias=False)

            new_down = new_down.to(device=old_down.weight.device, dtype=old_down.weight.dtype)
            new_up = new_up.to(device=old_up.weight.device, dtype=old_up.weight.dtype)
            torch.nn.init.kaiming_uniform_(new_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(new_up.weight)
            self.lora_down = new_down
            self.lora_up = new_up
        else:
            new_down_modules = []
            new_up_modules = []
            for old_down, old_up in zip(self.lora_down, self.lora_up):
                new_down = torch.nn.Linear(old_down.in_features, new_rank, bias=False).to(
                    device=old_down.weight.device, dtype=old_down.weight.dtype
                )
                new_up = torch.nn.Linear(new_rank, old_up.out_features, bias=False).to(
                    device=old_up.weight.device, dtype=old_up.weight.dtype
                )
                torch.nn.init.kaiming_uniform_(new_down.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(new_up.weight)
                new_down_modules.append(new_down)
                new_up_modules.append(new_up)
            self.lora_down = torch.nn.ModuleList(new_down_modules)
            self.lora_up = torch.nn.ModuleList(new_up_modules)

        self.lora_dim = new_rank
        alpha_value = self.alpha if isinstance(self.alpha, torch.Tensor) else torch.tensor(float(self.alpha), dtype=torch.float32)
        self.scale = float(alpha_value.detach().cpu().item()) / float(max(new_rank, 1))

    def get_adaptive_rank_runtime_state(self) -> Dict[str, Any]:
        is_adaptive = bool(getattr(self, "adaptive_rank", False))
        rank_lambda = None
        if is_adaptive and hasattr(self, "rank_lambda_param"):
            rank_lambda = float(self._adaptive_lambda().detach().cpu().item())
        alpha_value = self.alpha if isinstance(self.alpha, torch.Tensor) else torch.tensor(float(self.alpha), dtype=torch.float32)
        return {
            "lora_name": str(self.lora_name),
            "module_path": str(getattr(self, "module_path", "") or ""),
            "current_rank": int(self.lora_dim),
            "max_rank": int(getattr(self, "adaptive_rank_max_rank", self.lora_dim)),
            "min_rank": int(getattr(self, "adaptive_rank_min_rank", 1)),
            "target_rank": float(getattr(self, "adaptive_rank_target", self.lora_dim)),
            "quantile": float(getattr(self, "adaptive_rank_quantile", 0.9)),
            "weight": float(getattr(self, "adaptive_rank_weight", 0.0)),
            "adaptive_rank_active": is_adaptive,
            "was_adaptive_rank": bool(getattr(self, "was_adaptive_rank", is_adaptive)),
            "rank_lambda": rank_lambda,
            "alpha": float(alpha_value.detach().cpu().item()),
        }

    def load_adaptive_rank_runtime_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("adaptive rank runtime module state must be a dict")

        state_name = state.get("lora_name")
        if state_name is not None and str(state_name) != str(self.lora_name):
            raise ValueError(f"adaptive rank runtime state mismatch: expected {self.lora_name}, got {state_name}")

        current_rank = max(1, int(state.get("current_rank", self.lora_dim)))
        if current_rank != int(self.lora_dim):
            self._rebuild_lora_modules_for_rank(current_rank)

        max_rank = max(current_rank, int(state.get("max_rank", getattr(self, "adaptive_rank_max_rank", current_rank))))
        min_rank = max(1, min(current_rank, int(state.get("min_rank", getattr(self, "adaptive_rank_min_rank", 1)))))
        target_rank = float(state.get("target_rank", current_rank))
        quantile = float(state.get("quantile", getattr(self, "adaptive_rank_quantile", 0.9)))
        quantile = min(max(quantile, 1e-4), 1.0 - 1e-6)
        weight = float(state.get("weight", getattr(self, "adaptive_rank_weight", 0.0)))
        is_adaptive = bool(state.get("adaptive_rank_active", getattr(self, "adaptive_rank", False)))
        was_adaptive = bool(state.get("was_adaptive_rank", is_adaptive))

        self.adaptive_rank_max_rank = max_rank
        self.adaptive_rank_min_rank = min_rank
        self.adaptive_rank_target = max(min_rank, min(current_rank, int(round(target_rank))))
        self.adaptive_rank_quantile = quantile
        self.adaptive_rank_weight = weight
        self.was_adaptive_rank = was_adaptive
        alpha_value = float(
            state.get(
                "alpha",
                (
                    self.alpha.detach().cpu().item()
                    if isinstance(self.alpha, torch.Tensor)
                    else float(self.alpha)
                ),
            )
        )
        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.new_tensor(alpha_value, dtype=torch.float32)
        else:
            self.alpha = torch.tensor(alpha_value, dtype=torch.float32)
        self.scale = alpha_value / float(max(current_rank, 1))

        if is_adaptive:
            lambda_value = state.get("rank_lambda")
            if lambda_value is None:
                lambda_value = self._lambda_from_rank(self.adaptive_rank_target, self.adaptive_rank_quantile)
            lambda_value = max(float(lambda_value), 1e-6)
            inverse_value = self._inverse_softplus(lambda_value)
            if hasattr(self, "rank_lambda_param"):
                with torch.no_grad():
                    self.rank_lambda_param.copy_(torch.tensor(inverse_value, dtype=torch.float32))
            else:
                self.rank_lambda_param = torch.nn.Parameter(torch.tensor(inverse_value, dtype=torch.float32))
            self.adaptive_rank = True
        else:
            self.adaptive_rank = False
            if hasattr(self, "rank_lambda_param"):
                del self.rank_lambda_param


class AdaptiveRankLoRANetworkMixin:
    def _init_adaptive_rank_network_state(
        self,
        *,
        adaptive_rank: bool = False,
        adaptive_rank_target: Optional[int] = None,
        video_adaptive_rank_target: Optional[int] = None,
        audio_adaptive_rank_target: Optional[int] = None,
        cross_modal_adaptive_rank_target: Optional[int] = None,
        adaptive_rank_quantile: Optional[float] = None,
        adaptive_rank_weight: Optional[float] = None,
        video_adaptive_rank_weight: Optional[float] = None,
        audio_adaptive_rank_weight: Optional[float] = None,
        cross_modal_adaptive_rank_weight: Optional[float] = None,
        adaptive_rank_min_rank: Optional[int] = None,
        adaptive_rank_init_rank: Optional[int] = None,
        adaptive_rank_budget: Optional[float] = None,
        adaptive_rank_budget_ratio: Optional[float] = None,
        adaptive_rank_budget_weight: Optional[float] = None,
        adaptive_rank_schedule: Optional[str] = None,
        adaptive_rank_schedule_start: Optional[float] = None,
        adaptive_rank_schedule_end: Optional[float] = None,
        adaptive_rank_estimate: bool = False,
        adaptive_rank_estimate_report: Optional[str] = None,
        adaptive_rank_estimate_key: Optional[str] = None,
        adaptive_rank_estimate_apply: Optional[str] = None,
        adaptive_rank_estimate_reallocate_interval: Optional[int] = None,
        adaptive_rank_estimate_reallocate_start: Optional[float] = None,
        adaptive_rank_estimate_reallocate_apply: Optional[str] = None,
        adaptive_rank_finalize_start: Optional[float] = None,
        adaptive_rank_finalize_recover_steps: Optional[int] = None,
        adaptive_rank_finalize_recover_warmup_steps: Optional[int] = None,
        adaptive_rank_finalize_recover_lr_scale: Optional[float] = None,
        adaptive_rank_finalize_recover_scheduler: Optional[str] = None,
        adaptive_rank_hard_prune: bool = False,
        adaptive_rank_hard_prune_start: Optional[float] = None,
        adaptive_rank_hard_prune_interval: Optional[int] = None,
        adaptive_rank_hard_prune_min_delta: Optional[int] = None,
    ) -> None:
        self.adaptive_rank = adaptive_rank
        self.adaptive_rank_target = adaptive_rank_target
        self.video_adaptive_rank_target = video_adaptive_rank_target
        self.audio_adaptive_rank_target = audio_adaptive_rank_target
        self.cross_modal_adaptive_rank_target = cross_modal_adaptive_rank_target
        self.adaptive_rank_quantile = adaptive_rank_quantile
        self.adaptive_rank_weight = adaptive_rank_weight
        self.video_adaptive_rank_weight = video_adaptive_rank_weight
        self.audio_adaptive_rank_weight = audio_adaptive_rank_weight
        self.cross_modal_adaptive_rank_weight = cross_modal_adaptive_rank_weight
        self.adaptive_rank_min_rank = adaptive_rank_min_rank
        self.adaptive_rank_init_rank = adaptive_rank_init_rank
        self.adaptive_rank_budget = adaptive_rank_budget
        self.adaptive_rank_budget_ratio = adaptive_rank_budget_ratio
        self.adaptive_rank_budget_weight = adaptive_rank_budget_weight
        self.adaptive_rank_schedule = adaptive_rank_schedule
        self.adaptive_rank_schedule_start = adaptive_rank_schedule_start
        self.adaptive_rank_schedule_end = adaptive_rank_schedule_end
        self.adaptive_rank_estimate = adaptive_rank_estimate
        self.adaptive_rank_estimate_report = adaptive_rank_estimate_report
        self.adaptive_rank_estimate_key = adaptive_rank_estimate_key
        self.adaptive_rank_estimate_apply = adaptive_rank_estimate_apply
        self.adaptive_rank_estimate_reallocate_interval = adaptive_rank_estimate_reallocate_interval
        self.adaptive_rank_estimate_reallocate_start = adaptive_rank_estimate_reallocate_start
        self.adaptive_rank_estimate_reallocate_apply = adaptive_rank_estimate_reallocate_apply
        self.adaptive_rank_finalize_start = adaptive_rank_finalize_start
        self.adaptive_rank_finalize_recover_steps = adaptive_rank_finalize_recover_steps
        self.adaptive_rank_finalize_recover_warmup_steps = adaptive_rank_finalize_recover_warmup_steps
        self.adaptive_rank_finalize_recover_lr_scale = adaptive_rank_finalize_recover_lr_scale
        self.adaptive_rank_finalize_recover_scheduler = adaptive_rank_finalize_recover_scheduler
        self.adaptive_rank_hard_prune = bool(adaptive_rank_hard_prune)
        self.adaptive_rank_hard_prune_start = adaptive_rank_hard_prune_start
        self.adaptive_rank_hard_prune_interval = adaptive_rank_hard_prune_interval
        self.adaptive_rank_hard_prune_min_delta = adaptive_rank_hard_prune_min_delta
        self._adaptive_rank_estimate_applied = False
        self._adaptive_rank_estimate_scores: Optional[Dict[str, float]] = None
        self._adaptive_rank_estimate_total_target_budget: Optional[float] = None
        self._adaptive_rank_global_step = 0
        self._adaptive_rank_max_train_steps = None
        self._adaptive_rank_last_estimate_reallocate_step: Optional[int] = None
        self._adaptive_rank_estimate_reallocate_events = 0
        self._adaptive_rank_finalized = False
        self._adaptive_rank_finalize_events = 0
        self._adaptive_rank_finalized_modules = 0
        self._adaptive_rank_last_hard_prune_step: Optional[int] = None
        self._adaptive_rank_hard_prune_events = 0
        self._adaptive_rank_hard_pruned_modules = 0

    def _log_adaptive_rank_configuration(self, modules_dim: Optional[Dict[str, int]]) -> None:
        if modules_dim is not None or not self.adaptive_rank:
            return
        logger.info(
            "adaptive rank: target=%s, quantile=%s, weight=%s, min_rank=%s, init_rank=%s",
            self.adaptive_rank_target if self.adaptive_rank_target is not None else "base-rank",
            self.adaptive_rank_quantile if self.adaptive_rank_quantile is not None else 0.9,
            self.adaptive_rank_weight if self.adaptive_rank_weight is not None else 1e-4,
            self.adaptive_rank_min_rank if self.adaptive_rank_min_rank is not None else 1,
            self.adaptive_rank_init_rank if self.adaptive_rank_init_rank is not None else "base-rank",
        )
        if any(
            value is not None
            for value in (
                self.video_adaptive_rank_target,
                self.audio_adaptive_rank_target,
                self.cross_modal_adaptive_rank_target,
                self.video_adaptive_rank_weight,
                self.audio_adaptive_rank_weight,
                self.cross_modal_adaptive_rank_weight,
            )
        ):
            logger.info(
                "adaptive rank modality overrides: video(target=%s, weight=%s) audio(target=%s, weight=%s) cross-modal(target=%s, weight=%s)",
                self.video_adaptive_rank_target,
                self.video_adaptive_rank_weight,
                self.audio_adaptive_rank_target,
                self.audio_adaptive_rank_weight,
                self.cross_modal_adaptive_rank_target,
                self.cross_modal_adaptive_rank_weight,
            )
        if self.adaptive_rank_budget is not None or self.adaptive_rank_budget_ratio is not None:
            logger.info(
                "adaptive rank shared budget: budget=%s, ratio=%s, weight=%s",
                self.adaptive_rank_budget if self.adaptive_rank_budget is not None else "auto",
                self.adaptive_rank_budget_ratio if self.adaptive_rank_budget_ratio is not None else "off",
                self.adaptive_rank_budget_weight if self.adaptive_rank_budget_weight is not None else 1e-4,
            )
        if self.adaptive_rank_schedule is not None:
            logger.info(
                "adaptive rank schedule: type=%s, start=%s, end=%s",
                self.adaptive_rank_schedule,
                self.adaptive_rank_schedule_start if self.adaptive_rank_schedule_start is not None else 0.0,
                self.adaptive_rank_schedule_end if self.adaptive_rank_schedule_end is not None else 1.0,
            )
        if self.adaptive_rank_estimate or self.adaptive_rank_estimate_report is not None:
            logger.info(
                "adaptive rank estimate: enabled=%s, report=%s, key=%s, apply=%s",
                self.adaptive_rank_estimate,
                self.adaptive_rank_estimate_report if self.adaptive_rank_estimate_report is not None else "auto",
                self.adaptive_rank_estimate_key if self.adaptive_rank_estimate_key is not None else "fisher_mean",
                self.adaptive_rank_estimate_apply if self.adaptive_rank_estimate_apply is not None else "target",
            )
        if self.adaptive_rank_estimate_reallocate_interval is not None:
            logger.info(
                "adaptive rank estimate reallocate: interval=%s, start=%s, apply=%s",
                self.adaptive_rank_estimate_reallocate_interval,
                self.adaptive_rank_estimate_reallocate_start
                if self.adaptive_rank_estimate_reallocate_start is not None
                else 0.0,
                self.adaptive_rank_estimate_reallocate_apply
                if self.adaptive_rank_estimate_reallocate_apply is not None
                else "target",
            )
        if self.adaptive_rank_finalize_start is not None:
            logger.info(
                "adaptive rank finalize: start=%s, recover_steps=%s, recover_warmup=%s, recover_lr_scale=%s, recover_scheduler=%s",
                self.adaptive_rank_finalize_start,
                self.adaptive_rank_finalize_recover_steps
                if self.adaptive_rank_finalize_recover_steps is not None
                else "off",
                self.adaptive_rank_finalize_recover_warmup_steps
                if self.adaptive_rank_finalize_recover_warmup_steps is not None
                else 0,
                self.adaptive_rank_finalize_recover_lr_scale
                if self.adaptive_rank_finalize_recover_lr_scale is not None
                else 1.0,
                self.adaptive_rank_finalize_recover_scheduler
                if self.adaptive_rank_finalize_recover_scheduler is not None
                else "auto",
            )
        if self.adaptive_rank_hard_prune:
            logger.info(
                "adaptive rank hard prune: start=%s, interval=%s, min_delta=%s",
                self.adaptive_rank_hard_prune_start if self.adaptive_rank_hard_prune_start is not None else 0.5,
                self.adaptive_rank_hard_prune_interval if self.adaptive_rank_hard_prune_interval is not None else 100,
                self.adaptive_rank_hard_prune_min_delta if self.adaptive_rank_hard_prune_min_delta is not None else 1,
            )

    @staticmethod
    def _is_audio_module(module_name: str) -> bool:
        return "audio_" in module_name

    @staticmethod
    def _is_cross_modal_module(module_name: str) -> bool:
        return "audio_to_video" in module_name or "video_to_audio" in module_name or "av_ca_" in module_name

    def resolve_module_adaptive_rank_target(self, module_name: str) -> Optional[int]:
        if self._is_cross_modal_module(module_name) and self.cross_modal_adaptive_rank_target is not None:
            return self.cross_modal_adaptive_rank_target
        if self._is_audio_module(module_name):
            if self.audio_adaptive_rank_target is not None:
                return self.audio_adaptive_rank_target
        elif self.video_adaptive_rank_target is not None:
            return self.video_adaptive_rank_target
        return self.adaptive_rank_target

    def resolve_module_adaptive_rank_weight(self, module_name: str) -> Optional[float]:
        if self._is_cross_modal_module(module_name) and self.cross_modal_adaptive_rank_weight is not None:
            return self.cross_modal_adaptive_rank_weight
        if self._is_audio_module(module_name):
            if self.audio_adaptive_rank_weight is not None:
                return self.audio_adaptive_rank_weight
        elif self.video_adaptive_rank_weight is not None:
            return self.video_adaptive_rank_weight
        return self.adaptive_rank_weight

    @staticmethod
    def _adaptive_rank_estimate_lock_path(report_path: str) -> str:
        return f"{report_path}.lock"

    def _maybe_generate_adaptive_rank_estimate_report(self, args: Any) -> None:
        if not self.adaptive_rank_estimate or self.adaptive_rank_estimate_report is None:
            return

        report_path = os.path.abspath(str(self.adaptive_rank_estimate_report))
        report_dir = os.path.dirname(report_path) or "."
        os.makedirs(report_dir, exist_ok=True)
        if os.path.exists(report_path):
            self.adaptive_rank_estimate_report = report_path
            return

        lock_path = self._adaptive_rank_estimate_lock_path(report_path)
        deadline = time.time() + 600.0
        wait_logged = False

        while True:
            if os.path.exists(report_path):
                self.adaptive_rank_estimate_report = report_path
                return

            lock_fd = None
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not wait_logged:
                    logger.info("adaptive rank estimate: waiting for report at %s", report_path)
                    wait_logged = True
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for adaptive rank estimate report: {report_path}")
                time.sleep(1.0)
                continue

            try:
                os.write(lock_fd, str(os.getpid()).encode("utf-8"))
                if os.path.exists(report_path):
                    self.adaptive_rank_estimate_report = report_path
                    return

                logger.info("adaptive rank estimate: generating missing report at %s", report_path)
                from musubi_tuner.ltx2_estimate import generate_estimation_report

                generated_path = generate_estimation_report(args, estimation_output=report_path)
                self.adaptive_rank_estimate_report = os.path.abspath(str(generated_path))
                return
            finally:
                if lock_fd is not None:
                    os.close(lock_fd)
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass

    def prepare_adaptive_rank(self, args: Any) -> None:
        if self.adaptive_rank_estimate_report is None and self.adaptive_rank_estimate:
            output_dir = getattr(args, "output_dir", None)
            if output_dir is None:
                raise ValueError("adaptive_rank_estimate requires args.output_dir to resolve ltx2_estimate.json")
            self.adaptive_rank_estimate_report = os.path.join(str(output_dir), "ltx2_estimate.json")
        if self.adaptive_rank_estimate_report is not None:
            self.adaptive_rank_estimate_report = os.path.abspath(str(self.adaptive_rank_estimate_report))
            if not os.path.exists(self.adaptive_rank_estimate_report):
                self._maybe_generate_adaptive_rank_estimate_report(args)
        if self.adaptive_rank_estimate_report is not None and not self._adaptive_rank_estimate_applied:
            self._apply_adaptive_rank_estimate_overrides()

    def on_step_start(self, global_step: Optional[int] = None, max_train_steps: Optional[int] = None):
        if global_step is not None:
            self._adaptive_rank_global_step = int(global_step)
        if max_train_steps is not None:
            self._adaptive_rank_max_train_steps = max(1, int(max_train_steps))

    def has_adaptive_rank(self) -> bool:
        loras = self.text_encoder_loras + self.unet_loras
        return any(getattr(lora, "adaptive_rank", False) for lora in loras)

    def should_hard_prune_adaptive_rank(self) -> bool:
        if not self.adaptive_rank_hard_prune or not self.has_adaptive_rank():
            return False

        max_train_steps = self._adaptive_rank_max_train_steps
        if max_train_steps is None or max_train_steps <= 0:
            return False

        start = 0.5 if self.adaptive_rank_hard_prune_start is None else float(self.adaptive_rank_hard_prune_start)
        start = min(max(start, 0.0), 1.0)
        progress = float(self._adaptive_rank_global_step) / float(max_train_steps)
        progress = min(max(progress, 0.0), 1.0)
        if progress < start:
            return False

        interval = 100 if self.adaptive_rank_hard_prune_interval is None else int(self.adaptive_rank_hard_prune_interval)
        if interval <= 0:
            interval = 1
        if self._adaptive_rank_last_hard_prune_step is None:
            return True
        return (self._adaptive_rank_global_step - self._adaptive_rank_last_hard_prune_step) >= interval

    def should_reallocate_adaptive_rank_estimate(self) -> bool:
        if self._adaptive_rank_estimate_scores is None or self._adaptive_rank_estimate_total_target_budget is None:
            return False

        interval = self.adaptive_rank_estimate_reallocate_interval
        if interval is None or int(interval) <= 0:
            return False

        max_train_steps = self._adaptive_rank_max_train_steps
        if max_train_steps is None or max_train_steps <= 0:
            return False

        start = 0.0 if self.adaptive_rank_estimate_reallocate_start is None else float(
            self.adaptive_rank_estimate_reallocate_start
        )
        start = min(max(start, 0.0), 1.0)
        progress = float(self._adaptive_rank_global_step) / float(max_train_steps)
        progress = min(max(progress, 0.0), 1.0)
        if progress < start:
            return False

        interval = max(1, int(interval))
        if self._adaptive_rank_last_estimate_reallocate_step is None:
            return True
        return (self._adaptive_rank_global_step - self._adaptive_rank_last_estimate_reallocate_step) >= interval

    def should_finalize_adaptive_rank(self) -> bool:
        if self._adaptive_rank_finalized or self.adaptive_rank_finalize_start is None:
            return False
        if not self.has_adaptive_rank():
            return False

        max_train_steps = self._adaptive_rank_max_train_steps
        if max_train_steps is None or max_train_steps <= 0:
            return False

        start = min(max(float(self.adaptive_rank_finalize_start), 0.0), 1.0)
        progress = float(self._adaptive_rank_global_step) / float(max_train_steps)
        progress = min(max(progress, 0.0), 1.0)
        return progress >= start

    def get_adaptive_rank_finalize_recovery_config(self) -> Optional[Dict[str, Any]]:
        recover_steps = self.adaptive_rank_finalize_recover_steps
        if recover_steps is None or int(recover_steps) <= 0:
            return None

        lr_scale = self.adaptive_rank_finalize_recover_lr_scale
        if lr_scale is not None and float(lr_scale) <= 0:
            raise ValueError("adaptive_rank_finalize_recover_lr_scale must be > 0")

        return {
            "steps": int(recover_steps),
            "warmup_steps": (
                int(self.adaptive_rank_finalize_recover_warmup_steps)
                if self.adaptive_rank_finalize_recover_warmup_steps is not None
                else 0
            ),
            "lr_scale": float(lr_scale) if lr_scale is not None else 1.0,
            "scheduler": self.adaptive_rank_finalize_recover_scheduler,
        }

    def _get_adaptive_rank_schedule_factor(self) -> float:
        schedule = self.adaptive_rank_schedule
        if schedule is None:
            return 1.0

        max_train_steps = self._adaptive_rank_max_train_steps
        if max_train_steps is None or max_train_steps <= 0:
            return 1.0

        start = 0.0 if self.adaptive_rank_schedule_start is None else float(self.adaptive_rank_schedule_start)
        end = 1.0 if self.adaptive_rank_schedule_end is None else float(self.adaptive_rank_schedule_end)
        start = min(max(start, 0.0), 1.0)
        end = min(max(end, 0.0), 1.0)
        if end < start:
            start, end = end, start

        progress = float(self._adaptive_rank_global_step) / float(max_train_steps)
        progress = min(max(progress, 0.0), 1.0)
        if end <= start:
            raw = 1.0 if progress >= end else 0.0
        elif progress <= start:
            raw = 0.0
        elif progress >= end:
            raw = 1.0
        else:
            raw = (progress - start) / (end - start)

        if schedule == "linear":
            return raw
        if schedule == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * raw)
        raise ValueError(f"Unsupported adaptive_rank_schedule: {schedule}")

    def _get_scheduled_target_rank(self, lora: Any) -> float:
        final_target = float(getattr(lora, "adaptive_rank_target", lora.lora_dim))
        final_target = min(float(lora.lora_dim), max(float(getattr(lora, "adaptive_rank_min_rank", 1)), final_target))
        schedule_factor = self._get_adaptive_rank_schedule_factor()
        return float(lora.lora_dim) + (final_target - float(lora.lora_dim)) * schedule_factor

    @staticmethod
    def _load_adaptive_rank_estimate_scores(path: str, key: str) -> Dict[str, float]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        rows = payload.get("module_scores")
        if not isinstance(rows, list):
            rows = payload.get("top_modules")
        if not isinstance(rows, list):
            raise ValueError(f"adaptive_rank_estimate_report does not contain module_scores or top_modules: {path}")

        scores: Dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            module_path = row.get("module_path")
            if not isinstance(module_path, str):
                continue
            score = row.get(key)
            if score is None:
                continue
            score_value = max(0.0, float(score))
            scores[module_path] = max(scores.get(module_path, 0.0), score_value)
        return scores

    @staticmethod
    def _allocate_rank_budget(
        *,
        total_budget: float,
        min_ranks: List[float],
        max_ranks: List[float],
        scores: List[float],
    ) -> List[float]:
        n = len(min_ranks)
        if n == 0:
            return []

        alloc = list(min_ranks)
        budget = min(max(total_budget, sum(min_ranks)), sum(max_ranks))
        remaining = budget - sum(min_ranks)
        if remaining <= 1e-8:
            return alloc

        weights = [max(0.0, float(score)) for score in scores]
        if sum(weights) <= 0:
            weights = [1.0] * n

        active = {i for i in range(n) if max_ranks[i] > min_ranks[i]}
        while remaining > 1e-8 and active:
            weight_sum = sum(weights[i] for i in active)
            if weight_sum <= 0:
                equal_share = remaining / float(len(active))
                for i in list(active):
                    extra = min(max_ranks[i] - alloc[i], equal_share)
                    alloc[i] += extra
                    remaining -= extra
                    if max_ranks[i] - alloc[i] <= 1e-8:
                        active.remove(i)
                continue

            saturated = set()
            spent = 0.0
            for i in list(active):
                share = remaining * (weights[i] / weight_sum)
                extra = min(max_ranks[i] - alloc[i], share)
                alloc[i] += extra
                spent += extra
                if max_ranks[i] - alloc[i] <= 1e-8:
                    saturated.add(i)

            if spent <= 1e-8:
                break
            remaining -= spent
            active -= saturated

        return alloc

    @staticmethod
    def _get_adaptive_rank_capacity(lora: Any) -> float:
        return float(getattr(lora, "adaptive_rank_max_rank", lora.lora_dim))

    def _get_final_total_target_budget(self, adaptive_loras: List[Any]) -> Optional[float]:
        if not adaptive_loras:
            return None
        total_max_rank = sum(self._get_adaptive_rank_capacity(lora) for lora in adaptive_loras)
        total_min_rank = sum(float(getattr(lora, "adaptive_rank_min_rank", 1)) for lora in adaptive_loras)

        target_budget = self.adaptive_rank_budget
        if target_budget is None and self.adaptive_rank_budget_ratio is not None:
            target_budget = total_max_rank * float(self.adaptive_rank_budget_ratio)
        if target_budget is not None:
            return min(total_max_rank, max(total_min_rank, float(target_budget)))

        default_target_budget = sum(
            min(
                self._get_adaptive_rank_capacity(lora),
                max(
                    float(getattr(lora, "adaptive_rank_min_rank", 1)),
                    float(getattr(lora, "adaptive_rank_target", self._get_adaptive_rank_capacity(lora))),
                ),
            )
            for lora in adaptive_loras
        )
        return min(total_max_rank, max(total_min_rank, float(default_target_budget)))

    def _apply_adaptive_rank_estimate_overrides(self) -> None:
        if self._adaptive_rank_estimate_applied:
            return
        if not self.adaptive_rank or self.adaptive_rank_estimate_report is None:
            return

        adaptive_loras = [
            lora for lora in (self.text_encoder_loras + self.unet_loras) if getattr(lora, "adaptive_rank", False)
        ]
        if not adaptive_loras:
            return

        score_key = self.adaptive_rank_estimate_key if self.adaptive_rank_estimate_key is not None else "fisher_mean"
        apply_mode = self.adaptive_rank_estimate_apply if self.adaptive_rank_estimate_apply is not None else "target"
        if apply_mode not in ("target", "init", "both"):
            raise ValueError(f"Unsupported adaptive_rank_estimate_apply: {apply_mode}")

        scores_by_module = self._load_adaptive_rank_estimate_scores(self.adaptive_rank_estimate_report, score_key)
        total_target_budget = self._get_final_total_target_budget(adaptive_loras)
        if total_target_budget is None:
            return

        min_ranks = [float(getattr(lora, "adaptive_rank_min_rank", 1)) for lora in adaptive_loras]
        max_ranks = [self._get_adaptive_rank_capacity(lora) for lora in adaptive_loras]
        scores = [float(scores_by_module.get(str(getattr(lora, "module_path", "") or ""), 0.0)) for lora in adaptive_loras]
        allocated = self._allocate_rank_budget(
            total_budget=total_target_budget,
            min_ranks=min_ranks,
            max_ranks=max_ranks,
            scores=scores,
        )

        for lora, allocated_rank in zip(adaptive_loras, allocated):
            rank_value = int(round(allocated_rank))
            rank_value = max(int(getattr(lora, "adaptive_rank_min_rank", 1)), min(int(lora.lora_dim), rank_value))
            if apply_mode in ("target", "both"):
                lora.adaptive_rank_target = rank_value
            if apply_mode in ("init", "both"):
                init_lambda = lora._lambda_from_rank(rank_value, lora.adaptive_rank_quantile)
                with torch.no_grad():
                    lora.rank_lambda_param.copy_(torch.tensor(lora._inverse_softplus(init_lambda), dtype=torch.float32))
        self._adaptive_rank_estimate_scores = scores_by_module
        self._adaptive_rank_estimate_total_target_budget = float(total_target_budget)
        self._adaptive_rank_estimate_applied = True

    def _get_adaptive_rank_budget_target(self, adaptive_loras: List[Any]) -> Optional[float]:
        if not adaptive_loras:
            return None
        total_max_rank = sum(self._get_adaptive_rank_capacity(lora) for lora in adaptive_loras)
        total_min_rank = sum(float(getattr(lora, "adaptive_rank_min_rank", 1)) for lora in adaptive_loras)

        target_budget = self.adaptive_rank_budget
        if target_budget is None and self.adaptive_rank_budget_ratio is not None:
            target_budget = total_max_rank * float(self.adaptive_rank_budget_ratio)
        if target_budget is None:
            return None

        final_target_budget = min(total_max_rank, max(total_min_rank, float(target_budget)))
        schedule_factor = self._get_adaptive_rank_schedule_factor()
        return total_max_rank + (final_target_budget - total_max_rank) * schedule_factor

    def get_adaptive_rank_budget_loss(self) -> Optional[torch.Tensor]:
        adaptive_loras = [
            lora for lora in (self.text_encoder_loras + self.unet_loras) if getattr(lora, "adaptive_rank", False)
        ]
        target_budget = self._get_adaptive_rank_budget_target(adaptive_loras)
        budget_weight = self.adaptive_rank_budget_weight
        if target_budget is None:
            return None
        if budget_weight is None:
            budget_weight = 1e-4
        if budget_weight <= 0:
            return None

        expected_rank_sum = torch.stack([lora.get_expected_rank_tensor() for lora in adaptive_loras]).sum()
        target_budget_tensor = expected_rank_sum.new_tensor(target_budget)
        budget_error = torch.abs(expected_rank_sum - target_budget_tensor) / target_budget_tensor.clamp_min(1.0)
        return budget_error * float(budget_weight)

    def compute_adaptive_rank_loss(self) -> Optional[torch.Tensor]:
        losses = []
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            scheduled_target_rank = self._get_scheduled_target_rank(lora) if getattr(lora, "adaptive_rank", False) else None
            reg_loss = getattr(lora, "get_rank_regularization_loss", lambda *args, **kwargs: None)(
                target_rank_override=scheduled_target_rank
            )
            if reg_loss is not None:
                losses.append(reg_loss)
        budget_loss = self.get_adaptive_rank_budget_loss()
        if budget_loss is not None:
            losses.append(budget_loss)
        if not losses:
            return None
        return torch.stack(losses).sum()

    def maybe_hard_prune_adaptive_rank(
        self, global_step: Optional[int] = None, max_train_steps: Optional[int] = None, force: bool = False
    ) -> Optional[Dict[str, Any]]:
        self.on_step_start(global_step=global_step, max_train_steps=max_train_steps)
        if not force and not self.should_hard_prune_adaptive_rank():
            return None

        min_delta = 1 if self.adaptive_rank_hard_prune_min_delta is None else int(self.adaptive_rank_hard_prune_min_delta)
        pruned_modules = []
        rank_sum_before = 0
        rank_sum_after = 0

        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            rank_sum_before += int(lora.lora_dim)
            prune_info = getattr(lora, "hard_prune_to_static", lambda **kwargs: None)(min_delta=min_delta)
            rank_sum_after += int(lora.lora_dim)
            if prune_info is not None:
                pruned_modules.append(prune_info)

        if not pruned_modules:
            return None

        self._adaptive_rank_last_hard_prune_step = int(self._adaptive_rank_global_step)
        self._adaptive_rank_hard_prune_events += 1
        self._adaptive_rank_hard_pruned_modules += len(pruned_modules)
        return {
            "step": int(self._adaptive_rank_global_step),
            "events": int(self._adaptive_rank_hard_prune_events),
            "pruned_module_count": len(pruned_modules),
            "rank_sum_before": int(rank_sum_before),
            "rank_sum_after": int(rank_sum_after),
            "pruned_modules": pruned_modules,
        }

    def maybe_finalize_adaptive_rank(
        self, global_step: Optional[int] = None, max_train_steps: Optional[int] = None, force: bool = False
    ) -> Optional[Dict[str, Any]]:
        self.on_step_start(global_step=global_step, max_train_steps=max_train_steps)
        if not force and not self.should_finalize_adaptive_rank():
            return None

        adaptive_loras = [
            lora for lora in (self.text_encoder_loras + self.unet_loras) if getattr(lora, "adaptive_rank", False)
        ]
        if not adaptive_loras:
            self._adaptive_rank_finalized = True
            return None

        finalized_modules = []
        rank_sum_before = 0
        rank_sum_after = 0
        for lora in adaptive_loras:
            rank_sum_before += int(lora.lora_dim)
            finalize_info = getattr(lora, "hard_prune_to_static", lambda **kwargs: None)(min_delta=1, force_static=True)
            rank_sum_after += int(lora.lora_dim)
            if finalize_info is not None:
                finalized_modules.append(finalize_info)

        self._adaptive_rank_finalized = True
        if not finalized_modules:
            return None

        self._adaptive_rank_finalize_events += 1
        self._adaptive_rank_finalized_modules += len(finalized_modules)
        return {
            "step": int(self._adaptive_rank_global_step),
            "events": int(self._adaptive_rank_finalize_events),
            "finalized_module_count": len(finalized_modules),
            "rank_sum_before": int(rank_sum_before),
            "rank_sum_after": int(rank_sum_after),
            "finalized_modules": finalized_modules,
            "recovery_config": self.get_adaptive_rank_finalize_recovery_config(),
        }

    def maybe_reallocate_adaptive_rank_estimate(
        self, global_step: Optional[int] = None, max_train_steps: Optional[int] = None, force: bool = False
    ) -> Optional[Dict[str, Any]]:
        self.on_step_start(global_step=global_step, max_train_steps=max_train_steps)
        if not force and not self.should_reallocate_adaptive_rank_estimate():
            return None
        if self._adaptive_rank_estimate_scores is None or self._adaptive_rank_estimate_total_target_budget is None:
            return None
        apply_mode = (
            self.adaptive_rank_estimate_reallocate_apply
            if self.adaptive_rank_estimate_reallocate_apply is not None
            else "target"
        )
        if apply_mode not in ("target", "init", "both"):
            raise ValueError(f"Unsupported adaptive_rank_estimate_reallocate_apply: {apply_mode}")

        report_loras = [
            lora for lora in (self.text_encoder_loras + self.unet_loras) if getattr(lora, "was_adaptive_rank", False)
        ]
        adaptive_loras = [lora for lora in report_loras if getattr(lora, "adaptive_rank", False)]
        if not adaptive_loras:
            return None

        fixed_budget = sum(float(lora.lora_dim) for lora in report_loras if not getattr(lora, "adaptive_rank", False))
        target_budget = max(0.0, float(self._adaptive_rank_estimate_total_target_budget) - fixed_budget)

        min_ranks = [float(getattr(lora, "adaptive_rank_min_rank", 1)) for lora in adaptive_loras]
        max_ranks = [self._get_adaptive_rank_capacity(lora) for lora in adaptive_loras]
        scores = [
            float(self._adaptive_rank_estimate_scores.get(str(getattr(lora, "module_path", "") or ""), 0.0))
            for lora in adaptive_loras
        ]
        allocated = self._allocate_rank_budget(
            total_budget=target_budget,
            min_ranks=min_ranks,
            max_ranks=max_ranks,
            scores=scores,
        )

        changed_modules = []
        for lora, allocated_rank in zip(adaptive_loras, allocated):
            max_rank = int(round(self._get_adaptive_rank_capacity(lora)))
            new_target = int(round(allocated_rank))
            new_target = max(int(getattr(lora, "adaptive_rank_min_rank", 1)), min(max_rank, new_target))
            old_target = int(getattr(lora, "adaptive_rank_target", new_target))
            changed = False
            init_reset = False
            if apply_mode in ("target", "both") and new_target != old_target:
                lora.adaptive_rank_target = new_target
                changed = True
            if apply_mode in ("init", "both"):
                init_lambda = lora._lambda_from_rank(new_target, lora.adaptive_rank_quantile)
                with torch.no_grad():
                    lora.rank_lambda_param.copy_(torch.tensor(lora._inverse_softplus(init_lambda), dtype=torch.float32))
                init_reset = True
                changed = True
            if changed:
                changed_modules.append(
                    {
                        "lora_name": str(lora.lora_name),
                        "module_path": str(getattr(lora, "module_path", "") or ""),
                        "old_target": old_target,
                        "new_target": new_target,
                        "init_reset": init_reset,
                    }
                )

        self._adaptive_rank_last_estimate_reallocate_step = int(self._adaptive_rank_global_step)
        if not changed_modules:
            return None

        self._adaptive_rank_estimate_reallocate_events += 1
        return {
            "step": int(self._adaptive_rank_global_step),
            "events": int(self._adaptive_rank_estimate_reallocate_events),
            "fixed_budget": float(fixed_budget),
            "target_budget": float(self._adaptive_rank_estimate_total_target_budget),
            "remaining_budget": float(target_budget),
            "apply_mode": apply_mode,
            "changed_module_count": len(changed_modules),
            "changed_modules": changed_modules,
        }

    def get_adaptive_rank_metrics(self) -> Dict[str, float]:
        loras = self.text_encoder_loras + self.unet_loras
        adaptive_loras = [lora for lora in loras if getattr(lora, "adaptive_rank", False)]
        if not adaptive_loras:
            return {}

        effective_ranks = [float(lora.get_effective_rank()) for lora in adaptive_loras]
        expected_ranks = [float(lora.get_expected_rank_tensor().detach().cpu().item()) for lora in adaptive_loras]
        max_ranks = [self._get_adaptive_rank_capacity(lora) for lora in adaptive_loras]
        lambdas = [float(lora._adaptive_lambda().detach().cpu().item()) for lora in adaptive_loras]
        scheduled_targets = [float(self._get_scheduled_target_rank(lora)) for lora in adaptive_loras]
        metrics = {
            "adaptive_rank/modules": float(len(adaptive_loras)),
            "adaptive_rank/mean_effective_rank": sum(effective_ranks) / len(effective_ranks),
            "adaptive_rank/max_effective_rank": max(effective_ranks),
            "adaptive_rank/min_effective_rank": min(effective_ranks),
            "adaptive_rank/mean_rank_ratio": sum(effective_ranks) / max(sum(max_ranks), 1.0),
            "adaptive_rank/mean_expected_rank": sum(expected_ranks) / len(expected_ranks),
            "adaptive_rank/expected_rank_ratio": sum(expected_ranks) / max(sum(max_ranks), 1.0),
            "adaptive_rank/mean_target_rank": sum(scheduled_targets) / len(scheduled_targets),
            "adaptive_rank/mean_lambda": sum(lambdas) / len(lambdas),
            "adaptive_rank/schedule_factor": self._get_adaptive_rank_schedule_factor(),
        }
        if self._adaptive_rank_hard_prune_events > 0:
            metrics["adaptive_rank/hard_prune_events"] = float(self._adaptive_rank_hard_prune_events)
            metrics["adaptive_rank/hard_pruned_modules"] = float(self._adaptive_rank_hard_pruned_modules)
        if self._adaptive_rank_estimate_reallocate_events > 0:
            metrics["adaptive_rank/estimate_reallocate_events"] = float(self._adaptive_rank_estimate_reallocate_events)
        if self._adaptive_rank_finalize_events > 0:
            metrics["adaptive_rank/finalize_events"] = float(self._adaptive_rank_finalize_events)
            metrics["adaptive_rank/finalized_modules"] = float(self._adaptive_rank_finalized_modules)
        target_budget = self._get_adaptive_rank_budget_target(adaptive_loras)
        if target_budget is not None:
            expected_rank_sum = sum(expected_ranks)
            total_max_rank = sum(max_ranks)
            metrics.update(
                {
                    "adaptive_rank/expected_rank_sum": expected_rank_sum,
                    "adaptive_rank/target_budget": target_budget,
                    "adaptive_rank/target_budget_ratio": target_budget / max(total_max_rank, 1.0),
                    "adaptive_rank/budget_error": abs(expected_rank_sum - target_budget) / max(target_budget, 1.0),
                }
            )
        return metrics

    def build_export_state_dict(self) -> Dict[str, torch.Tensor]:
        export_state: Dict[str, torch.Tensor] = {}
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            if hasattr(lora, "export_state_dict"):
                export_state.update(lora.export_state_dict())

        if export_state:
            return export_state
        return self.state_dict()

    def build_adaptive_rank_report(self) -> Optional[Dict[str, Any]]:
        loras = self.text_encoder_loras + self.unet_loras
        report_loras = [lora for lora in loras if getattr(lora, "was_adaptive_rank", False)]
        if not report_loras:
            return None

        modules = []
        for lora in report_loras:
            is_adaptive = bool(getattr(lora, "adaptive_rank", False))
            modules.append(
                {
                    "lora_name": str(lora.lora_name),
                    "module_path": str(getattr(lora, "module_path", "") or ""),
                    "max_rank": int(self._get_adaptive_rank_capacity(lora)),
                    "current_rank": int(lora.lora_dim),
                    "min_rank": int(getattr(lora, "adaptive_rank_min_rank", 1)),
                    "target_rank": float(getattr(lora, "adaptive_rank_target", lora.lora_dim)),
                    "expected_rank": float(
                        lora.get_expected_rank_tensor().detach().cpu().item() if is_adaptive else float(lora.lora_dim)
                    ),
                    "effective_rank": int(lora.get_effective_rank() if is_adaptive else lora.lora_dim),
                    "lambda": float(lora._adaptive_lambda().detach().cpu().item()) if is_adaptive else None,
                    "weight": float(getattr(lora, "adaptive_rank_weight", 0.0)),
                    "adaptive_rank_active": is_adaptive,
                }
            )
        modules.sort(key=lambda row: row["lora_name"])
        summary = self.get_adaptive_rank_metrics()
        if not summary:
            summary = {
                "adaptive_rank/modules": float(len(report_loras)),
            }
            if self._adaptive_rank_hard_prune_events > 0:
                summary["adaptive_rank/hard_prune_events"] = float(self._adaptive_rank_hard_prune_events)
                summary["adaptive_rank/hard_pruned_modules"] = float(self._adaptive_rank_hard_pruned_modules)
            if self._adaptive_rank_finalize_events > 0:
                summary["adaptive_rank/finalize_events"] = float(self._adaptive_rank_finalize_events)
                summary["adaptive_rank/finalized_modules"] = float(self._adaptive_rank_finalized_modules)
        return {
            "summary": summary,
            "modules": modules,
        }

    def build_adaptive_rank_runtime_state(self) -> Optional[Dict[str, Any]]:
        loras = self.text_encoder_loras + self.unet_loras
        runtime_loras = [lora for lora in loras if getattr(lora, "was_adaptive_rank", False)]
        if not runtime_loras:
            return None

        modules = [lora.get_adaptive_rank_runtime_state() for lora in runtime_loras]
        modules.sort(key=lambda row: row["lora_name"])
        return {
            "version": 1,
            "network": {
                "estimate_applied": bool(self._adaptive_rank_estimate_applied),
                "estimate_scores": (
                    {str(key): float(value) for key, value in self._adaptive_rank_estimate_scores.items()}
                    if self._adaptive_rank_estimate_scores is not None
                    else None
                ),
                "estimate_total_target_budget": (
                    float(self._adaptive_rank_estimate_total_target_budget)
                    if self._adaptive_rank_estimate_total_target_budget is not None
                    else None
                ),
                "global_step": int(self._adaptive_rank_global_step),
                "max_train_steps": (
                    int(self._adaptive_rank_max_train_steps) if self._adaptive_rank_max_train_steps is not None else None
                ),
                "last_estimate_reallocate_step": (
                    int(self._adaptive_rank_last_estimate_reallocate_step)
                    if self._adaptive_rank_last_estimate_reallocate_step is not None
                    else None
                ),
                "estimate_reallocate_events": int(self._adaptive_rank_estimate_reallocate_events),
                "finalized": bool(self._adaptive_rank_finalized),
                "finalize_events": int(self._adaptive_rank_finalize_events),
                "finalized_modules": int(self._adaptive_rank_finalized_modules),
                "last_hard_prune_step": (
                    int(self._adaptive_rank_last_hard_prune_step)
                    if self._adaptive_rank_last_hard_prune_step is not None
                    else None
                ),
                "hard_prune_events": int(self._adaptive_rank_hard_prune_events),
                "hard_pruned_modules": int(self._adaptive_rank_hard_pruned_modules),
            },
            "modules": modules,
        }

    def load_adaptive_rank_runtime_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("adaptive rank runtime state must be a dict")

        module_rows = payload.get("modules")
        if not isinstance(module_rows, list):
            raise ValueError("adaptive rank runtime state is missing modules")

        loras = self.text_encoder_loras + self.unet_loras
        loras_by_name = {str(lora.lora_name): lora for lora in loras}
        loaded_names = set()
        for row in module_rows:
            if not isinstance(row, dict):
                continue
            lora_name = row.get("lora_name")
            if not isinstance(lora_name, str):
                continue
            lora = loras_by_name.get(lora_name)
            if lora is None:
                raise ValueError(f"adaptive rank runtime state references unknown module: {lora_name}")
            lora.load_adaptive_rank_runtime_state(row)
            loaded_names.add(lora_name)

        for lora in loras:
            if getattr(lora, "was_adaptive_rank", False) and str(lora.lora_name) not in loaded_names:
                raise ValueError(f"adaptive rank runtime state is missing module: {lora.lora_name}")

        network_state = payload.get("network")
        if not isinstance(network_state, dict):
            network_state = {}

        scores = network_state.get("estimate_scores")
        if isinstance(scores, dict):
            self._adaptive_rank_estimate_scores = {str(key): float(value) for key, value in scores.items()}
        else:
            self._adaptive_rank_estimate_scores = None
        total_target_budget = network_state.get("estimate_total_target_budget")
        self._adaptive_rank_estimate_total_target_budget = (
            None if total_target_budget is None else float(total_target_budget)
        )
        self._adaptive_rank_estimate_applied = bool(network_state.get("estimate_applied", self._adaptive_rank_estimate_applied))
        self._adaptive_rank_global_step = int(network_state.get("global_step", self._adaptive_rank_global_step))
        max_train_steps = network_state.get("max_train_steps")
        self._adaptive_rank_max_train_steps = None if max_train_steps is None else max(1, int(max_train_steps))
        last_reallocate = network_state.get("last_estimate_reallocate_step")
        self._adaptive_rank_last_estimate_reallocate_step = None if last_reallocate is None else int(last_reallocate)
        self._adaptive_rank_estimate_reallocate_events = int(
            network_state.get("estimate_reallocate_events", self._adaptive_rank_estimate_reallocate_events)
        )
        self._adaptive_rank_finalized = bool(network_state.get("finalized", self._adaptive_rank_finalized))
        self._adaptive_rank_finalize_events = int(network_state.get("finalize_events", self._adaptive_rank_finalize_events))
        self._adaptive_rank_finalized_modules = int(
            network_state.get("finalized_modules", self._adaptive_rank_finalized_modules)
        )
        last_hard_prune = network_state.get("last_hard_prune_step")
        self._adaptive_rank_last_hard_prune_step = None if last_hard_prune is None else int(last_hard_prune)
        self._adaptive_rank_hard_prune_events = int(
            network_state.get("hard_prune_events", self._adaptive_rank_hard_prune_events)
        )
        self._adaptive_rank_hard_pruned_modules = int(
            network_state.get("hard_pruned_modules", self._adaptive_rank_hard_pruned_modules)
        )

    @staticmethod
    def _adaptive_rank_runtime_state_path(state_dir: str) -> str:
        return os.path.join(state_dir, "adaptive_rank_runtime.json")

    @staticmethod
    def _adaptive_rank_report_path(file: str) -> str:
        root, _ = os.path.splitext(file)
        return root + ".adaptive_rank.json"
