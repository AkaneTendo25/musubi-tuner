from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AudioSupervisionState:
    expected_batches: int = 0
    supervised_batches: int = 0


@dataclass
class AudioSupervisionAlert:
    ratio: float
    min_ratio: float
    expected_batches: int
    supervised_batches: int


def normalize_audio_supervision_mode(raw_mode: Any) -> str:
    if isinstance(raw_mode, bool):
        return "warn" if raw_mode else "off"
    mode = str(raw_mode or "off").strip().lower()
    if mode in {"off", "none", "false", "0", "disable", "disabled"}:
        return "off"
    if mode in {"warn", "warning"}:
        return "warn"
    if mode in {"error", "fail", "raise"}:
        return "error"
    raise ValueError(f"audio_supervision_mode must be one of ['off', 'warn', 'error']. Got: {raw_mode}")


def reset_audio_supervision_state(state: AudioSupervisionState) -> None:
    state.expected_batches = 0
    state.supervised_batches = 0


def update_and_check_audio_supervision(
    state: AudioSupervisionState,
    *,
    mode: str,
    warmup_steps: int,
    check_interval: int,
    min_ratio: float,
    audio_expected_for_batch: bool,
    audio_supervised_for_batch: bool,
) -> Optional[AudioSupervisionAlert]:
    if mode == "off":
        return None

    if audio_expected_for_batch:
        state.expected_batches += 1
        if audio_supervised_for_batch:
            state.supervised_batches += 1

    expected = state.expected_batches
    if expected <= 0:
        return None
    if expected < int(warmup_steps):
        return None
    if int(check_interval) <= 0:
        return None
    if expected % int(check_interval) != 0:
        return None

    supervised = state.supervised_batches
    ratio = float(supervised) / float(expected)
    if ratio >= float(min_ratio):
        return None

    return AudioSupervisionAlert(
        ratio=ratio,
        min_ratio=float(min_ratio),
        expected_batches=expected,
        supervised_batches=supervised,
    )


def format_audio_supervision_alert(alert: AudioSupervisionAlert) -> str:
    return (
        "Audio supervision monitor: low supervised-audio ratio "
        f"({alert.ratio:.3f} < {alert.min_ratio:.3f}, "
        f"supervised={alert.supervised_batches}, expected={alert.expected_batches}). "
        "Audio prediction is likely under-trained because AV batches are missing audio latents."
    )
