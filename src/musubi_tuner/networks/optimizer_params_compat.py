import argparse
import inspect
import logging
from typing import Any, Optional

import torch


def _filter_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs, []

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs, []

    filtered: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in kwargs.items():
        param = signature.parameters.get(key)
        if param is None or param.kind == inspect.Parameter.POSITIONAL_ONLY:
            if value is not None:
                skipped.append(key)
            continue
        filtered[key] = value

    return filtered, skipped


def _normalize_optimizer_param_groups(trainable_params: Any) -> tuple[list[Any], int]:
    if trainable_params is None:
        return [], 0

    if isinstance(trainable_params, torch.nn.Parameter):
        groups: list[Any] = [trainable_params]
    elif isinstance(trainable_params, dict):
        groups = [trainable_params]
    elif isinstance(trainable_params, (list, tuple)):
        groups = list(trainable_params)
    else:
        try:
            groups = list(trainable_params)
        except TypeError:
            groups = [trainable_params]

    normalized: list[Any] = []
    total_params = 0

    for group in groups:
        if isinstance(group, dict):
            group_copy = dict(group)
            params_obj = group_copy.get("params", [])
            if isinstance(params_obj, torch.nn.Parameter):
                params = [params_obj]
            else:
                try:
                    params = list(params_obj)
                except TypeError:
                    params = [params_obj]

            params = [param for param in params if isinstance(param, torch.nn.Parameter)]
            if len(params) == 0:
                continue

            group_copy["params"] = params
            normalized.append(group_copy)
            total_params += len(params)
            continue

        if isinstance(group, torch.nn.Parameter):
            normalized.append(group)
            total_params += 1

    return normalized, total_params


def _collect_fallback_trainable_params(network: Any) -> tuple[list[Any], int, Optional[str]]:
    if hasattr(network, "requires_grad_"):
        try:
            network.requires_grad_(True)
        except Exception:
            pass

    for source in ("get_trainable_params", "parameters"):
        getter = getattr(network, source, None)
        if not callable(getter):
            continue
        try:
            candidate_params = getter()
        except TypeError:
            continue

        normalized, count = _normalize_optimizer_param_groups(candidate_params)
        if count > 0:
            return normalized, count, source

    return [], 0, None


def prepare_optimizer_params_compat(
    network: Any,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[list[Any], Optional[list[str]]]:
    prepare_fn = getattr(network, "prepare_optimizer_params", None)
    if not callable(prepare_fn):
        raise AttributeError(f"{network.__class__.__name__} does not implement prepare_optimizer_params")

    requested_kwargs = {
        "unet_lr": args.learning_rate,
        "audio_lr": getattr(args, "audio_lr", None),
        "lr_args": getattr(args, "lr_args", None),
    }
    prepare_kwargs, skipped_kwargs = _filter_supported_kwargs(prepare_fn, requested_kwargs)
    if skipped_kwargs:
        logger.info(
            "Skipping unsupported prepare_optimizer_params kwargs for %s: %s",
            network.__class__.__name__,
            ", ".join(skipped_kwargs),
        )

    prepared = prepare_fn(**prepare_kwargs)
    if isinstance(prepared, tuple):
        trainable_params = prepared[0] if len(prepared) > 0 else None
        lr_descriptions = prepared[1] if len(prepared) > 1 else None
    else:
        trainable_params, lr_descriptions = prepared, None

    normalized_params, param_count = _normalize_optimizer_param_groups(trainable_params)
    if param_count > 0:
        return normalized_params, lr_descriptions

    fallback_params, fallback_count, fallback_source = _collect_fallback_trainable_params(network)
    if fallback_count > 0:
        logger.warning(
            "prepare_optimizer_params for %s returned no params; falling back to network.%s() with %d params.",
            network.__class__.__name__,
            fallback_source,
            fallback_count,
        )
        return fallback_params, lr_descriptions

    raise ValueError(
        "No trainable parameters were found for the network. "
        "Check LoRA/LyCORIS target selection and network configuration."
    )
