from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class H3FP8Policy:
    """Architecture-specific FP8 key policy kept outside the vendored source."""

    target_keys: tuple[str, ...]
    exclude_keys: tuple[str, ...] = ()
    quantization_mode: str = "block"
    block_size: int = 64

    def __post_init__(self) -> None:
        if not self.target_keys:
            raise ValueError("H3 FP8 target_keys must match transformer parameter names")
        if self.quantization_mode not in {"tensor", "channel", "block"}:
            raise ValueError(f"unsupported H3 FP8 quantization mode: {self.quantization_mode}")
        if self.block_size <= 0:
            raise ValueError("H3 FP8 block_size must be positive")

    def selected_weight_keys(self, keys: Iterable[str]) -> tuple[str, ...]:
        selected = []
        for key in keys:
            if not key.endswith(".weight"):
                continue
            if not any(pattern in key for pattern in self.target_keys):
                continue
            if any(pattern in key for pattern in self.exclude_keys):
                continue
            selected.append(key)
        return tuple(selected)

    def validate_checkpoint_keys(self, keys: Iterable[str]) -> tuple[str, ...]:
        selected = self.selected_weight_keys(keys)
        if not selected:
            raise ValueError(
                "H3 FP8 policy selected no transformer weights; verify target_keys and exclude_keys against the checkpoint"
            )
        return selected


def optimize_and_patch_h3_fp8(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    policy: H3FP8Policy,
    device: str | torch.device,
    move_to_device: bool,
) -> dict[str, torch.Tensor]:
    """Apply Musubi's established scaled-FP8 implementation with H3 keys."""
    policy.validate_checkpoint_keys(state_dict)
    from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

    optimized = optimize_state_dict_with_fp8(
        state_dict,
        device,
        list(policy.target_keys),
        list(policy.exclude_keys),
        move_to_device=move_to_device,
        quantization_mode=policy.quantization_mode,
        block_size=policy.block_size,
    )
    apply_fp8_monkey_patch(model, optimized, use_scaled_mm=False)
    return optimized
