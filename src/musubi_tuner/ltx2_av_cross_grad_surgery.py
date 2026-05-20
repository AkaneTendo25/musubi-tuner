"""Gradient scaling hooks for LTX-2 audio/video cross-modal attention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


_DEFAULT_A2V_SCHEDULE = "0:0.0,1-10:0.1,40-47:0.3"
_VALID_DIRECTIONS = {"a2v", "v2a"}
_VALID_PROJECTIONS = {"k", "v"}


@dataclass(frozen=True)
class AVCrossGradSurgeryConfig:
    """Per-block gradient scales for cross-modal attention projections."""

    a2v: dict[int, float] = field(default_factory=dict)
    v2a: dict[int, float] = field(default_factory=dict)
    projections: tuple[str, ...] = ("k", "v")

    def scales_for(self, direction: str) -> dict[int, float]:
        if direction == "a2v":
            return self.a2v
        if direction == "v2a":
            return self.v2a
        raise ValueError(f"Unknown AV cross-modal direction: {direction}")

    def format_summary(self) -> str:
        parts: list[str] = [f"projections={','.join(self.projections)}"]
        if self.a2v:
            parts.append(f"a2v={_format_schedule(self.a2v)}")
        if self.v2a:
            parts.append(f"v2a={_format_schedule(self.v2a)}")
        return " ".join(parts)


class AVCrossGradSurgeryHandle:
    """Owns installed hook handles so tests or callers can remove them."""

    def __init__(self, handles: list[Any], installed: list[str]) -> None:
        self._handles = handles
        self.installed = installed

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def parse_av_cross_grad_surgery_args(
    raw_args: Iterable[str] | None,
    *,
    total_layers: int | None = None,
) -> AVCrossGradSurgeryConfig:
    """Parse key=value args for AV cross-modal gradient surgery.

    Supported keys:
      - a2v=<schedule> and v2a=<schedule>, where schedule is block:scale or
        start-end:scale comma-separated entries.
      - a2v_kv=<schedule> and v2a_kv=<schedule> aliases from the OmniNFT paper.
      - projections=k,v to target K/V projections. Only K and V are supported.

    With no schedule keys, the OmniNFT A2V K/V schedule is used.
    """

    raw_items = list(raw_args or [])
    schedules: dict[str, dict[int, float]] = {}
    projections = ("k", "v")

    for item in raw_items:
        if "=" not in str(item):
            raise ValueError(f"AV cross grad surgery args must be key=value, got: {item!r}")
        key, value = str(item).split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key:
            raise ValueError(f"AV cross grad surgery arg has an empty key: {item!r}")
        if key == "projections":
            projections = _parse_projections(value)
            continue
        if key in {"a2v", "v2a", "a2v_kv", "v2a_kv"}:
            direction = key[:3]
            schedules[direction] = _parse_schedule(value, total_layers=total_layers)
            continue
        raise ValueError(f"Unknown AV cross grad surgery arg: {key}")

    if not schedules:
        schedules["a2v"] = _parse_schedule(_DEFAULT_A2V_SCHEDULE, total_layers=total_layers)

    return AVCrossGradSurgeryConfig(
        a2v=schedules.get("a2v", {}),
        v2a=schedules.get("v2a", {}),
        projections=projections,
    )


def install_av_cross_grad_surgery(transformer: torch.nn.Module, config: AVCrossGradSurgeryConfig) -> AVCrossGradSurgeryHandle:
    """Install forward hooks that preserve activations and scale backward gradients."""

    base_model = transformer.model if hasattr(transformer, "model") else transformer
    blocks = getattr(base_model, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("AV cross grad surgery requires a transformer with transformer_blocks")

    handles: list[Any] = []
    installed: list[str] = []
    direction_to_attn = {
        "a2v": "audio_to_video_attn",
        "v2a": "video_to_audio_attn",
    }

    for block_idx, block in enumerate(blocks):
        for direction, attn_name in direction_to_attn.items():
            schedule = config.scales_for(direction)
            if block_idx not in schedule:
                continue
            scale = float(schedule[block_idx])
            attn = getattr(block, attn_name, None)
            if attn is None:
                raise ValueError(f"AV cross grad surgery requested {direction} block {block_idx}, but {attn_name} is missing")
            for projection in config.projections:
                projection_name = f"to_{projection}"
                module = getattr(attn, projection_name, None)
                if module is None:
                    raise ValueError(
                        f"AV cross grad surgery requested {attn_name}.{projection_name} in block {block_idx}, "
                        "but the projection is missing"
                    )
                handles.append(module.register_forward_hook(_make_grad_scale_hook(scale)))
                installed.append(f"{direction}:{block_idx}:{projection_name}:{scale:g}")

    if not handles:
        raise ValueError("AV cross grad surgery matched no cross-modal projection modules")

    return AVCrossGradSurgeryHandle(handles, installed)


def _make_grad_scale_hook(scale: float):
    def hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            return output
        if scale == 1.0:
            return output
        return output.detach() + (output - output.detach()) * scale

    return hook


def _parse_projections(value: str) -> tuple[str, ...]:
    projections = tuple(part.strip().lower().removeprefix("to_") for part in value.split(",") if part.strip())
    if not projections:
        raise ValueError("projections must include at least one projection")
    invalid = [projection for projection in projections if projection not in _VALID_PROJECTIONS]
    if invalid:
        raise ValueError(f"projections supports only k,v; got: {','.join(invalid)}")
    deduped = tuple(dict.fromkeys(projections))
    return deduped


def _parse_schedule(value: str, *, total_layers: int | None) -> dict[int, float]:
    if not value:
        raise ValueError("AV cross grad surgery schedule must not be empty")
    schedule: dict[int, float] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Schedule entry must be block:scale or start-end:scale, got: {part!r}")
        raw_block, raw_scale = part.split(":", 1)
        blocks = _parse_block_selector(raw_block.strip(), total_layers=total_layers)
        scale = _parse_scale(raw_scale.strip())
        for block in blocks:
            schedule[block] = scale
    if not schedule:
        raise ValueError("AV cross grad surgery schedule did not select any blocks")
    return schedule


def _parse_block_selector(raw: str, *, total_layers: int | None) -> list[int]:
    if not raw:
        raise ValueError("Schedule block selector must not be empty")
    if "-" in raw:
        start_raw, end_raw = raw.split("-", 1)
        start = int(start_raw)
        end = int(end_raw)
        if end < start:
            raise ValueError(f"Schedule range end must be >= start, got: {raw}")
        blocks = list(range(start, end + 1))
    else:
        blocks = [int(raw)]
    for block in blocks:
        if block < 0:
            raise ValueError(f"Schedule block indices must be non-negative, got: {block}")
        if total_layers is not None and block >= int(total_layers):
            raise ValueError(f"Schedule block index {block} is outside transformer range 0..{int(total_layers) - 1}")
    return blocks


def _parse_scale(raw: str) -> float:
    scale = float(raw)
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"Gradient scale must be in [0, 1], got: {scale}")
    return scale


def _format_schedule(schedule: dict[int, float]) -> str:
    return ",".join(f"{block}:{scale:g}" for block, scale in sorted(schedule.items()))
