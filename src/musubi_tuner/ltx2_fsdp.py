"""Opt-in FSDP1 / ZeRO param + optimizer-state sharding for LTX-2 full fine-tuning.

Off by default; enable with --ltx2_fsdp (or LTX2_FSDP=1). Builds an Accelerate
FullyShardedDataParallelPlugin (fsdp_version=1) that shards the transformer's
parameters, gradients, and optimizer state across the data-parallel ranks.

Multi-GPU only. Mutually exclusive with model-parallel, remote-stage, block swap,
blockwise / grad-checkpoint weight offload, fp8-gemm, fp8 / int8 weights, qgalore,
fused backward, compile, full-FT text-encoder, EMA, Self-Flow, and non-Adam/AdamW/
Adafactor optimizers; each raises at startup.
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)

ENV_ENABLED = "LTX2_FSDP"

# Optimizers compatible with FSDP's flat-parameter remap.
_FSDP_SAFE_OPTIMIZERS = {"adam", "adamw", "adafactor"}


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def add_ltx2_fsdp_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if _parser_has_option(parser, "--ltx2_fsdp"):
        return parser

    parser.add_argument(
        "--ltx2_fsdp",
        action="store_true",
        help="opt-in: shard transformer params/grads/optimizer-state across data-parallel ranks via "
        "Accelerate FSDP1 (ZeRO-3 style). Multi-GPU full fine-tune only. Off by default.",
    )
    parser.add_argument(
        "--ltx2_fsdp_reshard_after_forward",
        type=str,
        default="true",
        help="FSDP reshard_after_forward: 'true' (ZeRO-3, reshard params after fwd), 'false' (ZeRO-2, keep "
        "gathered), or an int prefetch degree. Default 'true'.",
    )
    parser.add_argument(
        "--ltx2_fsdp_cpu_offload",
        action="store_true",
        help="with --ltx2_fsdp, offload sharded params + optimizer state to CPU (ZeRO-3 + offload).",
    )
    parser.add_argument(
        "--ltx2_fsdp_transformer_cls",
        type=str,
        default="BasicAVTransformerBlock",
        help="transformer block class name to auto-wrap for FSDP (default BasicAVTransformerBlock).",
    )
    parser.add_argument(
        "--ltx2_fsdp_state_dict_type",
        type=str,
        default="FULL_STATE_DICT",
        choices=["FULL_STATE_DICT", "SHARDED_STATE_DICT"],
        help="FSDP state_dict_type for checkpointing. FULL_STATE_DICT gathers an unsharded checkpoint "
        "(default); SHARDED_STATE_DICT keeps per-rank shards.",
    )
    parser.add_argument(
        "--ltx2_fsdp_cpu_ram_efficient_loading",
        action="store_true",
        help="with --ltx2_fsdp, load the checkpoint on rank 0 only and broadcast (lower peak host RAM).",
    )
    parser.add_argument(
        "--ltx2_fsdp_activation_checkpointing",
        action="store_true",
        help="with --ltx2_fsdp, use FSDP-native activation checkpointing (sets --gradient_checkpointing off "
        "to avoid double-wrapping).",
    )
    return parser


def is_ltx2_fsdp_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "ltx2_fsdp", False)) or os.getenv(ENV_ENABLED) == "1"


def _resolve_reshard_after_forward(value: str):
    # FSDP1 sharding strategy: true=ZeRO-3 (FULL_SHARD), false=ZeRO-2 (SHARD_GRAD_OP).
    v = str(value).strip().lower()
    if v in {"true", "1", "full_shard"}:
        return "FULL_SHARD"
    if v in {"false", "0", "shard_grad_op"}:
        return "SHARD_GRAD_OP"
    if v == "no_shard":
        return "NO_SHARD"
    return "FULL_SHARD"


def build_ltx2_fsdp_plugin(args: argparse.Namespace):
    """Build the Accelerate FullyShardedDataParallelPlugin (fsdp_version=1), or None when disabled."""
    if not is_ltx2_fsdp_enabled(args):
        return None

    try:
        import inspect

        from accelerate import FullyShardedDataParallelPlugin
    except ImportError as exc:  # pragma: no cover - accelerate is always present in training
        raise RuntimeError("--ltx2_fsdp requires the 'accelerate' package with FSDP support.") from exc

    supported = set(inspect.signature(FullyShardedDataParallelPlugin.__init__).parameters)
    if "fsdp_version" not in supported:
        raise RuntimeError("--ltx2_fsdp needs a newer accelerate that exposes fsdp_version; upgrade accelerate.")

    desired = {
        "fsdp_version": 1,
        "reshard_after_forward": _resolve_reshard_after_forward(getattr(args, "ltx2_fsdp_reshard_after_forward", "true")),
        "cpu_offload": bool(getattr(args, "ltx2_fsdp_cpu_offload", False)),
        "transformer_cls_names_to_wrap": [str(getattr(args, "ltx2_fsdp_transformer_cls", "BasicAVTransformerBlock"))],
        "auto_wrap_policy": "transformer_based_wrap",
        # Keep original Parameters as views into the flat param so the pre-built optimizer
        # updates the live sharded params.
        "use_orig_params": True,
        "state_dict_type": str(getattr(args, "ltx2_fsdp_state_dict_type", "FULL_STATE_DICT")),
        "cpu_ram_efficient_loading": bool(getattr(args, "ltx2_fsdp_cpu_ram_efficient_loading", False)),
        "activation_checkpointing": bool(getattr(args, "ltx2_fsdp_activation_checkpointing", False)),
    }
    kwargs = {k: v for k, v in desired.items() if k in supported}
    dropped = sorted(set(desired) - set(kwargs))
    if dropped:
        logger.warning("LTX-2 FSDP: installed accelerate ignores unsupported plugin kwargs: %s", ", ".join(dropped))

    logger.info("LTX-2 FSDP1 plugin: %s", {k: kwargs[k] for k in sorted(kwargs)})
    return FullyShardedDataParallelPlugin(**kwargs)


def gather_fsdp_full_state_dict(accelerator, module):
    """Return an unsharded (rank-0) full state dict. Collective: call on all ranks."""
    return accelerator.get_state_dict(module)


def validate_ltx2_fsdp_setup(args: argparse.Namespace, accelerator) -> None:
    """Raise if FSDP is enabled with an incompatible feature. No-op when disabled."""
    if not is_ltx2_fsdp_enabled(args):
        return

    if int(getattr(accelerator, "num_processes", 1) or 1) < 2:
        raise RuntimeError("--ltx2_fsdp requires multi-GPU (accelerate --num_processes >= 2).")

    # (flag attr, label) -> raise if truthy
    incompatible = [
        ("ltx2_model_parallel", "--ltx2_model_parallel"),
        ("ltx2_remote_stage", "--ltx2_remote_stage"),
        ("blockwise_checkpointing", "--blockwise_checkpointing"),
        ("gradient_checkpointing_cpu_offload", "--gradient_checkpointing_cpu_offload"),
        ("fp8_gemm", "--fp8_gemm"),
        ("fp8_base", "--fp8_base"),
        ("int8_weights", "--int8_weights"),
        ("qgalore_full_ft", "--qgalore_full_ft"),
        ("fused_backward_pass", "--fused_backward_pass"),
        ("compile", "--compile"),
        ("full_ft_train_text_encoder", "--full_ft_train_text_encoder"),
        ("use_ema", "--use_ema"),
    ]
    for attr, label in incompatible:
        if bool(getattr(args, attr, False)):
            raise RuntimeError(f"--ltx2_fsdp is mutually exclusive with {label}.")

    if int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
        raise RuntimeError("--ltx2_fsdp is mutually exclusive with --blocks_to_swap.")

    optimizer_type = str(getattr(args, "optimizer_type", "") or "").lower()
    if optimizer_type and optimizer_type not in _FSDP_SAFE_OPTIMIZERS:
        raise RuntimeError(f"--ltx2_fsdp supports only {sorted(_FSDP_SAFE_OPTIMIZERS)} optimizers (got {optimizer_type!r}).")

    if getattr(getattr(args, "_trainer", None), "_self_flow", False):
        raise RuntimeError("--ltx2_fsdp is mutually exclusive with Self-Flow.")
