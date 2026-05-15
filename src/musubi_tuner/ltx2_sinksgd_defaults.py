"""Shared LTX-2 SinkSGD default helpers.

Keep this module free of torch imports so dashboard command generation can use
the same defaults as the training entry points.
"""

from __future__ import annotations

import shlex
from typing import Iterable


DEFAULT_LTX2_OPTIMIZER_TYPE = "SinkSGD_adv"
DEFAULT_LTX2_LEARNING_RATE = 1.0e-3
DEFAULT_LTX2_DORA_OFT_LEARNING_RATE = 5.0e-4
DEFAULT_LTX2_LR_SCHEDULER = "constant"
DEFAULT_LTX2_LORA_RANK = 4
LEGACY_LTX2_LEARNING_RATE = 2.0e-6
DEFAULT_LTX2_SINKSGD_MOMENTUM = 0.995
DEFAULT_LTX2_SINKSGD_NESTEROV = True
DEFAULT_LTX2_SINKSGD_NESTEROV_COEF = 0.8
DEFAULT_LTX2_SINKSGD_NORMED_MOMENTUM = True
DEFAULT_LTX2_SINKSGD_SINKHORN_ITERATIONS = 3
DEFAULT_LTX2_SINKSGD_ORTHOGONAL_SINKHORN = True
DEFAULT_LTX2_SINKSGD_OPTIMIZER_ARGS = (
    "spectral_normalization=True",
    "scale_lr_with_effective_batch=True",
    f"normed_momentum={DEFAULT_LTX2_SINKSGD_NORMED_MOMENTUM}",
    f"momentum={DEFAULT_LTX2_SINKSGD_MOMENTUM}",
    f"nesterov={DEFAULT_LTX2_SINKSGD_NESTEROV}",
    f"nesterov_coef={DEFAULT_LTX2_SINKSGD_NESTEROV_COEF}",
    f"orthogonal_sinkhorn={DEFAULT_LTX2_SINKSGD_ORTHOGONAL_SINKHORN}",
    f"sinkhorn_iterations={DEFAULT_LTX2_SINKSGD_SINKHORN_ITERATIONS}",
)
SINKSGD_LR_SCALE_ARG_KEYS = (
    "scale_lr_with_effective_batch",
    "scale_lr_with_grad_accum",
    "scale_lr_with_gradient_accumulation",
)

SINKSGD_OPTIMIZER_ALIASES = {
    "sinksgd",
    "sink_sgd",
    "sinksgd_adv",
    "sinksgdadv",
}


def normalize_optimizer_type(value: str | None) -> str:
    return str(value or DEFAULT_LTX2_OPTIMIZER_TYPE).strip().lower()


def is_sinksgd_optimizer(value: str | None) -> bool:
    return normalize_optimizer_type(value) in SINKSGD_OPTIMIZER_ALIASES


def split_key_value_args(raw: str | Iterable[str] | None) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        parts = shlex.split(raw, posix=False)
    else:
        parts = [str(part) for part in raw]
    normalized: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
            normalized.append(part[1:-1])
        else:
            normalized.append(part)
    return normalized


def key_value_arg_key(part: str) -> str | None:
    clean = str(part).strip()
    if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
        clean = clean[1:-1]
    if "=" not in clean:
        return None
    key, _ = clean.split("=", 1)
    return key.strip().lower()


def has_key_value_arg(parts: Iterable[str], key: str) -> bool:
    target = key.strip().lower()
    return any(key_value_arg_key(part) == target for part in parts)


def get_key_value_arg(parts: Iterable[str], key: str) -> str | None:
    target = key.strip().lower()
    for part in parts:
        clean = str(part).strip()
        if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
            clean = clean[1:-1]
        if "=" not in clean:
            continue
        name, value = clean.split("=", 1)
        if name.strip().lower() == target:
            return value.strip()
    return None


def set_key_value_arg(parts: Iterable[str], key: str, value: object) -> list[str]:
    target = key.strip().lower()
    replacement = f"{key}={value}"
    updated: list[str] = []
    replaced = False
    for part in parts:
        if key_value_arg_key(part) == target:
            if not replaced:
                updated.append(replacement)
                replaced = True
            continue
        updated.append(str(part))
    if not replaced:
        updated.append(replacement)
    return updated


def bool_key_value_arg(parts: Iterable[str], key: str, default: bool = False) -> bool:
    value = get_key_value_arg(parts, key)
    if value is None:
        return default
    return value.strip().strip("'\"").lower() in {"1", "true", "yes", "y", "on"}


def uses_sinksgd_spectral_scaling(optimizer_type: str | None, optimizer_args: str | Iterable[str] | None) -> bool:
    if not is_sinksgd_optimizer(optimizer_type):
        return False
    return bool_key_value_arg(split_key_value_args(optimizer_args), "spectral_normalization", default=True)


def resolve_ltx2_optimizer_type(optimizer_type: str | None) -> str:
    value = str(optimizer_type or "").strip()
    return value or DEFAULT_LTX2_OPTIMIZER_TYPE


def resolve_ltx2_optimizer_args(optimizer_type: str | None, optimizer_args: str | Iterable[str] | None) -> list[str]:
    args = split_key_value_args(optimizer_args)
    if not is_sinksgd_optimizer(optimizer_type):
        return args

    has_lr_scale_arg = any(has_key_value_arg(args, key) for key in SINKSGD_LR_SCALE_ARG_KEYS)
    for default_arg in DEFAULT_LTX2_SINKSGD_OPTIMIZER_ARGS:
        key, value = default_arg.split("=", 1)
        if key in SINKSGD_LR_SCALE_ARG_KEYS and has_lr_scale_arg:
            continue
        if not has_key_value_arg(args, key):
            args.append(f"{key}={value}")
    return args


def network_args_enable_dora_oft(network_args: str | Iterable[str] | None, fallback: bool = False) -> bool:
    value = get_key_value_arg(split_key_value_args(network_args), "use_dora_oft")
    if value is None:
        return fallback
    return value.strip().strip("'\"").lower() in {"1", "true", "yes", "y", "on"}


def resolve_ltx2_learning_rate(
    learning_rate: float | None,
    optimizer_type: str | None,
    network_args: str | Iterable[str] | None = None,
    *,
    use_dora_oft: bool = False,
) -> float:
    if learning_rate is not None:
        return float(learning_rate)
    if is_sinksgd_optimizer(optimizer_type):
        if network_args_enable_dora_oft(network_args, fallback=use_dora_oft):
            return DEFAULT_LTX2_DORA_OFT_LEARNING_RATE
        return DEFAULT_LTX2_LEARNING_RATE
    return LEGACY_LTX2_LEARNING_RATE


def effective_lora_rank(network_dim: int | None, default: int = DEFAULT_LTX2_LORA_RANK) -> int:
    if network_dim is None:
        return default
    try:
        rank = int(network_dim)
    except (TypeError, ValueError):
        return default
    return rank if rank > 0 else default


def resolve_ltx2_network_alpha(
    network_alpha: float | None,
    network_dim: int | None,
    optimizer_type: str | None,
    optimizer_args: str | Iterable[str] | None,
) -> float | None:
    if network_alpha is not None:
        return float(network_alpha)
    if uses_sinksgd_spectral_scaling(optimizer_type, optimizer_args):
        return float(effective_lora_rank(network_dim))
    return None


def resolve_ltx2_audio_alpha(
    audio_alpha: float | None,
    audio_dim: int | None,
    optimizer_type: str | None,
    optimizer_args: str | Iterable[str] | None,
) -> float | None:
    if audio_alpha is not None:
        return float(audio_alpha)
    if audio_dim is not None and uses_sinksgd_spectral_scaling(optimizer_type, optimizer_args):
        return float(effective_lora_rank(audio_dim))
    return None
