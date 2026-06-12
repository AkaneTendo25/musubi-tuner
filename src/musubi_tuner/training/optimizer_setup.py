"""Optimizer construction helpers for NetworkTrainer."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import importlib
import logging
import math
from typing import Any, Optional

import torch
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from musubi_tuner.modules.group_lr_scheduler import GroupWarmupScheduler, parse_group_lr_warmup_args
from musubi_tuner.modules.lr_schedulers import RexLR
from musubi_tuner.networks.optimizer_params_compat import prepare_optimizer_params_compat

logger = logging.getLogger(__name__)

DEFAULT_PRODIGY_PLUS_OPTIMIZER_TYPE = "ProdigyPlusScheduleFree"
DEFAULT_PRODIGY_PLUS_OPTIMIZER_ARGS = (
    "betas=(0.9,0.99)",
    "beta3=None",
    "weight_decay=0.0",
    "weight_decay_by_lr=True",
    "use_bias_correction=False",
    "d0=1e-6",
    "d_coef=1.0",
    "prodigy_steps=0",
    "use_speed=False",
    "eps=1e-8",
    "split_groups=True",
    "split_groups_mean=False",
    "factored=True",
    "factored_fp32=True",
    "use_stableadamw=True",
    "use_cautious=False",
    "use_grams=False",
    "use_adopt=False",
    "d_limiter=True",
    "stochastic_rounding=True",
    "use_schedulefree=True",
    "schedulefree_c=0.0",
    "use_orthograd=False",
)
PRODIGY_PLUS_OPTIMIZER_ALIASES = {
    "pplus",
    "prodigyplus",
    "prodigyplusschedulefree",
}


def _optimizer_alias_key(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return "".join(ch for ch in normalized if ch not in {"-", "_", " "})


def is_prodigy_plus_optimizer(value: str | None) -> bool:
    return _optimizer_alias_key(value) in PRODIGY_PLUS_OPTIMIZER_ALIASES


def _literal_or_raw(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def get_optimizer(self, args, trainable_params: list[torch.nn.Parameter]) -> tuple[str, str, torch.optim.Optimizer]:
    optimizer_type = args.optimizer_type.lower()

    optimizer_kwargs = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            if "=" not in arg:
                raise ValueError(f"Invalid --optimizer_args entry (expected key=value): {arg}")
            key, value = arg.split("=", 1)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            optimizer_kwargs[key] = value

    lr = args.learning_rate
    optimizer = None
    optimizer_class = None

    if is_prodigy_plus_optimizer(args.optimizer_type):
        try:
            from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
        except ImportError as exc:
            raise ImportError(
                "Prodigy Plus requires the 'prodigy-plus-schedule-free' package. "
                "Install it with `pip install prodigy-plus-schedule-free==2.0.1`."
            ) from exc

        args.optimizer_type = DEFAULT_PRODIGY_PLUS_OPTIMIZER_TYPE
        for default_arg in DEFAULT_PRODIGY_PLUS_OPTIMIZER_ARGS:
            key, value = default_arg.split("=", 1)
            optimizer_kwargs.setdefault(key, _literal_or_raw(value))
        logger.info("use Prodigy Plus Schedule-Free optimizer | lr=%s | %s", lr, optimizer_kwargs)
        optimizer_class = ProdigyPlusScheduleFree
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type in {"sinksgd", "sink_sgd", "sinksgd_adv", "sinksgdadv"}:
        from musubi_tuner.optimizers.sink_sgd import SinkSGD

        def pop_bool_arg(*names: str) -> bool:
            for name in names:
                if name not in optimizer_kwargs:
                    continue
                value = optimizer_kwargs.pop(name)
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
                return bool(value)
            return False

        if pop_bool_arg("scale_lr_with_grad_accum", "scale_lr_with_gradient_accumulation"):
            grad_accum = max(float(getattr(args, "gradient_accumulation_steps", 1) or 1), 1.0)
            lr *= math.sqrt(grad_accum)
            logger.info("SinkSGD: scaled learning rate by sqrt(gradient_accumulation_steps=%g) -> lr=%s", grad_accum, lr)

        if pop_bool_arg("scale_lr_with_effective_batch"):
            train_batch_size = max(float(getattr(args, "train_batch_size", 1) or 1), 1.0)
            grad_accum = max(float(getattr(args, "gradient_accumulation_steps", 1) or 1), 1.0)
            effective_batch = train_batch_size * grad_accum
            lr *= math.sqrt(effective_batch)
            logger.info("SinkSGD: scaled learning rate by sqrt(effective_batch=%g) -> lr=%s", effective_batch, lr)

        logger.info("use SinkSGD optimizer | lr=%s | %s", lr, optimizer_kwargs)
        optimizer_class = SinkSGD
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type in {"came", "camesimple", "came_simple", "came8bit", "came_8bit"}:
        from musubi_tuner.optimizers.came_8bit import CAME, CAME8bit

        if "stochastic_rounding" not in optimizer_kwargs and (
            bool(getattr(args, "full_bf16", False))
            or bool(getattr(args, "fused_backward_pass", False))
            or getattr(args, "mixed_precision", None) == "bf16"
        ):
            optimizer_kwargs["stochastic_rounding"] = True
            logger.info("CAME: defaulting stochastic_rounding=True for BF16/fused training")
        optimizer_class = CAME if optimizer_type in {"came", "camesimple", "came_simple"} else CAME8bit
        logger.info(
            "use %s optimizer (stochastic_rounding, cautious, step_parameter) | %s", optimizer_class.__name__, optimizer_kwargs
        )
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.startswith("torchao_") or optimizer_type.startswith("ao_") or optimizer_type.startswith("torchao."):
        from musubi_tuner.optimizers.backends import resolve_torchao_optimizer_class

        if "bf16_stochastic_round" not in optimizer_kwargs and (
            bool(getattr(args, "full_bf16", False))
            or bool(getattr(args, "fused_backward_pass", False))
            or getattr(args, "mixed_precision", None) == "bf16"
        ):
            optimizer_kwargs["bf16_stochastic_round"] = True
            logger.info("torchao optimizer: defaulting bf16_stochastic_round=True for BF16/fused training")
        optimizer_class = resolve_torchao_optimizer_class(args.optimizer_type)
        logger.info("use torchao optimizer %s | %s", optimizer_class.__name__, optimizer_kwargs)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type in {"qgalore", "q_galore", "qgaloreadamw8bit", "q_galore_adamw8bit", "q-galore-adamw8bit"}:
        from musubi_tuner.optimizers.q_galore import QGaLoreAdamW8bit

        optimizer_class = QGaLoreAdamW8bit
        logger.info("use Q-GaLore AdamW8bit optimizer | %s", optimizer_kwargs)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.startswith("apollo") or optimizer_type.startswith("qapollo") or optimizer_type.startswith("q_apollo"):
        from musubi_tuner.optimizers.backends import is_qapollo_optimizer_type, resolve_apollo_optimizer_class

        if is_qapollo_optimizer_type(args.optimizer_type):
            optimizer_kwargs.setdefault("optim_bits", 8)
        optimizer_class = resolve_apollo_optimizer_class(args.optimizer_type)
        logger.info("use APOLLO optimizer %s | %s", optimizer_class.__name__, optimizer_kwargs)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.startswith("optimi_") or optimizer_type.startswith("torchoptimi_") or optimizer_type.startswith("optimi."):
        from musubi_tuner.optimizers.backends import resolve_optimi_optimizer_class

        if bool(getattr(args, "fused_backward_pass", False)):
            gradient_release = optimizer_kwargs.get("gradient_release", True)
            if isinstance(gradient_release, str):
                gradient_release = gradient_release.lower() in {"1", "true", "yes", "on"}
                optimizer_kwargs["gradient_release"] = gradient_release
            if gradient_release is not True:
                raise ValueError("optimi optimizers require gradient_release=True for --fused_backward_pass")
            optimizer_kwargs.setdefault("gradient_release", True)
            logger.info("optimi optimizer: defaulting gradient_release=True for fused backward")
        optimizer_class = resolve_optimi_optimizer_class(args.optimizer_type)
        logger.info("use torch-optimi optimizer %s | %s", optimizer_class.__name__, optimizer_kwargs)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type in ("lion", "lion8bit", "lion8bitint8"):
        from musubi_tuner.optimizers.lion import Lion as _Lion, Lion8bit as _Lion8bit, Lion8bitInt8 as _Lion8bitInt8

        _cls = {"lion": _Lion, "lion8bit": _Lion8bit, "lion8bitint8": _Lion8bitInt8}[optimizer_type]
        logger.info("use %s optimizer | %s", _cls.__name__, optimizer_kwargs)
        optimizer_class = _cls
        optimizer = _cls(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "smmf":
        from musubi_tuner.optimizers.smmf import SMMF as _SMMF

        logger.info("use SMMF optimizer | %s", optimizer_kwargs)
        optimizer_class = _SMMF
        optimizer = _SMMF(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("8bit".lower()):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes / bitsandbytes does not appear to be installed")

        if optimizer_type == "AdamW8bit".lower():
            logger.info("use 8-bit AdamW optimizer | %s", optimizer_kwargs)
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        elif optimizer_type == "PagedAdamW8bit".lower():
            logger.info("use 8-bit PagedAdamW optimizer | %s", optimizer_kwargs)
            optimizer_class = getattr(bnb.optim, "PagedAdamW8bit", None)
            if optimizer_class is None:
                raise ValueError("bitsandbytes.optim.PagedAdamW8bit is not available in this bitsandbytes build")
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        elif optimizer_type == "PagedAdam8bit".lower():
            logger.info("use 8-bit PagedAdam optimizer | %s", optimizer_kwargs)
            optimizer_class = getattr(bnb.optim, "PagedAdam8bit", None)
            if optimizer_class is None:
                raise ValueError("bitsandbytes.optim.PagedAdam8bit is not available in this bitsandbytes build")
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            logger.info("set relative_step to True because warmup_init is True")
            optimizer_kwargs["relative_step"] = True
        logger.info("use Adafactor optimizer | %s", optimizer_kwargs)

        if optimizer_kwargs["relative_step"]:
            logger.info("relative_step is true")
            if lr != 0.0:
                logger.warning("learning rate is used as initial_lr")
            args.learning_rate = None

            if args.lr_scheduler != "adafactor":
                logger.info("use adafactor_scheduler")
            args.lr_scheduler = f"adafactor:{lr}"

            lr = None
        else:
            if args.max_grad_norm != 0.0:
                logger.warning("because max_grad_norm is set, clip_grad_norm is enabled. consider setting it to 0")
            if args.lr_scheduler != "constant_with_warmup":
                logger.warning("constant_with_warmup may be a good scheduler choice")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                logger.warning("clip_threshold=1.0 may be a good setting")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        logger.info("use AdamW optimizer | %s", optimizer_kwargs)
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "automagic":
        from musubi_tuner.optimizers.automagic import Automagic

        logger.info("use Automagic optimizer | lr=%s | %s", lr, optimizer_kwargs)
        optimizer_class = Automagic
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type in ("badam", "blockadam", "block_optimizer", "blockoptimizer"):
        base_type = str(optimizer_kwargs.get("base_optimizer_type") or "AdamW")
        if base_type.lower() in ("badam", "blockadam", "block_optimizer", "blockoptimizer"):
            raise ValueError("base_optimizer_type must name the wrapped optimizer, not BAdam")
        logger.info("BAdam: building base optimizer %r (wrap deferred to trainer) | wrapper_kwargs=%s", base_type, optimizer_kwargs)
        saved_optimizer_type = args.optimizer_type
        saved_optimizer_args = args.optimizer_args
        args.optimizer_type = base_type
        args.optimizer_args = list(getattr(args, "base_optimizer_args", None) or [])
        try:
            return self.get_optimizer(args, trainable_params)
        finally:
            args.optimizer_type = saved_optimizer_type
            args.optimizer_args = saved_optimizer_args

    if optimizer is None:
        case_sensitive_optimizer_type = args.optimizer_type
        logger.info("use %s | %s", case_sensitive_optimizer_type, optimizer_kwargs)

        if "." not in case_sensitive_optimizer_type:
            optimizer_module = torch.optim
        else:
            values = case_sensitive_optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            case_sensitive_optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    if hasattr(optimizer, "train") and callable(optimizer.train):
        train_fn = optimizer.train
        eval_fn = optimizer.eval
    else:
        train_fn = lambda: None
        eval_fn = lambda: None

    return optimizer_name, optimizer_args, optimizer, train_fn, eval_fn


def is_schedulefree_optimizer(self, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> bool:
    return args.optimizer_type.lower().endswith("schedulefree".lower()) or args.optimizer_type.lower() == "automagic"


def get_dummy_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
    class DummyScheduler:
        def __init__(self, optimizer: torch.optim.Optimizer):
            self.optimizer = optimizer

        def step(self):
            pass

        def get_last_lr(self):
            return [group["lr"] for group in self.optimizer.param_groups]

    return DummyScheduler(optimizer)


def enable_lycoris_fp8_forward_compat(self, args: argparse.Namespace, network: Any) -> None:
    network_module_name = str(getattr(args, "network_module", "") or "")
    uses_lycoris_module = "lycoris" in network_module_name.lower()
    uses_fp8_base = bool(getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False))
    if not uses_lycoris_module or not uses_fp8_base:
        return
    if bool(getattr(network, "_lycoris_fp8_forward_compat_applied", False)):
        return

    if args.mixed_precision == "fp16":
        compat_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        compat_dtype = torch.bfloat16
    else:
        compat_dtype = torch.float32

    converted = 0
    checked = 0
    for lora in getattr(network, "loras", []):
        org_modules = getattr(lora, "org_module", None)
        if not isinstance(org_modules, (list, tuple)) or len(org_modules) == 0:
            continue
        module = org_modules[0]
        if module is None or not hasattr(module, "weight"):
            continue
        if not isinstance(module.weight, torch.nn.Parameter):
            continue

        checked += 1
        weight_data = module.weight.data
        if not isinstance(weight_data, torch.Tensor) or weight_data.dtype.itemsize != 1:
            continue

        module.weight.data = weight_data.to(dtype=compat_dtype)
        if hasattr(module, "bias") and isinstance(module.bias, torch.nn.Parameter) and module.bias is not None:
            module.bias.data = module.bias.data.to(dtype=compat_dtype)
        converted += 1

    setattr(network, "_lycoris_fp8_forward_compat_applied", True)
    if converted > 0:
        logger.warning(
            "LyCORIS FP8 forward compat enabled: upcasted %d/%d adapted base layers to %s to avoid FP8 op limitations.",
            converted,
            checked,
            compat_dtype,
        )
    else:
        logger.info(
            "LyCORIS FP8 forward compat checked %d adapted layers; no FP8 base layers required upcast.",
            checked,
        )


def maybe_wrap_group_warmup_scheduler(
    self,
    lr_scheduler,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: Optional[int],
    warmup_overrides: dict[str, int],
):
    if not warmup_overrides:
        return lr_scheduler
    logger.info("Per-group LR warmup overrides enabled: %s", warmup_overrides)
    return GroupWarmupScheduler(
        lr_scheduler,
        optimizer,
        default_warmup_steps=int(num_warmup_steps or 0),
        warmup_overrides=warmup_overrides,
    )


def prepare_network_optimizer_params(self, args: argparse.Namespace, network: Any):
    network_module_name = str(getattr(args, "network_module", "") or "")
    uses_lycoris_module = "lycoris" in network_module_name.lower()
    if uses_lycoris_module:
        return prepare_optimizer_params_compat(network, args, logger)
    return network.prepare_optimizer_params(
        unet_lr=args.learning_rate,
        audio_lr=getattr(args, "audio_lr", None),
        lr_args=getattr(args, "lr_args", None),
    )


def copy_optimizer_state_subset(state: dict, keep_param_ids: set[int]) -> dict:
    if isinstance(state, defaultdict):
        copied = defaultdict(dict)
    else:
        copied = type(state)()
    for param, value in state.items():
        if id(param) in keep_param_ids:
            copied[param] = value
    return copied


def refresh_prodigy_plus_late_param_group_state(
    optimizer: torch.optim.Optimizer,
    *,
    split_groups: Optional[bool] = None,
) -> None:
    """Initialize Prodigy Plus bookkeeping for param groups added after construction."""
    inner_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    param_groups = getattr(inner_optimizer, "param_groups", None)
    defaults = getattr(inner_optimizer, "defaults", None)
    if not param_groups or not isinstance(defaults, dict):
        return

    optimizer_module = type(inner_optimizer).__module__
    is_prodigy_plus = optimizer_module.startswith("prodigyplus") or any(
        "running_d_numerator" in group or "running_d_denom" in group for group in param_groups
    )
    if not is_prodigy_plus:
        return

    # Match Prodigy Plus semantics for the final pre-step group layout. If
    # setup adds groups before the first step, honor the requested default;
    # once training has started, preserve the active split-groups mode.
    if split_groups is None:
        default_split_groups = bool(defaults.get("split_groups", False))
        optimizer_started = bool(getattr(inner_optimizer, "state", None)) or any(group.get("k", 1) != 1 for group in param_groups)
        if default_split_groups and len(param_groups) > 1 and not optimizer_started:
            split_groups = True
        else:
            for group in param_groups:
                if "split_groups" in group:
                    split_groups = bool(group["split_groups"])
                    break
            else:
                split_groups = default_split_groups
                if split_groups and len(param_groups) == 1:
                    split_groups = False

    def first_param_device(group: dict[str, Any]) -> torch.device:
        for param in group.get("params", []):
            if isinstance(param, torch.Tensor):
                return param.device
        first_group = param_groups[0]
        tensor = first_group.get("running_d_numerator")
        if tensor is None:
            tensor = first_group.get("running_d_denom")
        if isinstance(tensor, torch.Tensor):
            return tensor.device
        return torch.device("cpu")

    groups_requiring_running_state = param_groups if split_groups else param_groups[:1]
    for group in param_groups:
        group["split_groups"] = split_groups
    for group in groups_requiring_running_state:
        device = first_param_device(group)
        if "running_d_numerator" not in group:
            group["running_d_numerator"] = torch.tensor(0.0, dtype=torch.float32, device=device)
        if "running_d_denom" not in group:
            group["running_d_denom"] = torch.tensor(0.0, dtype=torch.float32, device=device)


def get_prodigy_plus_split_groups(optimizer: torch.optim.Optimizer) -> Optional[bool]:
    inner_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    for group in getattr(inner_optimizer, "param_groups", []):
        if "split_groups" in group:
            return bool(group["split_groups"])
    return None


def refresh_optimizer_after_adaptive_rank_prune(
    self,
    args: argparse.Namespace,
    accelerator,
    network: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    *,
    old_network_param_ids: set[int],
    global_step: int,
    recovery_config: Optional[dict[str, Any]] = None,
):
    unwrapped_network = accelerator.unwrap_model(network)
    new_network_param_groups, lr_descriptions = self._prepare_network_optimizer_params(args, unwrapped_network)

    inner_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    prodigy_plus_split_groups = self._get_prodigy_plus_split_groups(inner_optimizer)

    preserved_extra_groups = []
    preserved_extra_param_ids: set[int] = set()
    for group in inner_optimizer.param_groups:
        extra_params = [param for param in group["params"] if id(param) not in old_network_param_ids]
        if not extra_params:
            continue
        group_copy = {k: v for k, v in group.items() if k != "params"}
        group_copy["params"] = extra_params
        preserved_extra_groups.append(group_copy)
        preserved_extra_param_ids.update(id(param) for param in extra_params)

    new_network_param_ids: set[int] = set()
    normalized_network_groups = []
    for group in new_network_param_groups:
        group_copy = {k: v for k, v in group.items() if k != "params"}
        params = list(group["params"])
        if recovery_config is not None and "lr" in group_copy:
            group_copy["lr"] = float(group_copy["lr"]) * float(recovery_config.get("lr_scale", 1.0))
        group_copy["params"] = params
        normalized_network_groups.append(group_copy)
        new_network_param_ids.update(id(param) for param in params)

    keep_param_ids = new_network_param_ids | preserved_extra_param_ids
    inner_optimizer.state = self._copy_optimizer_state_subset(inner_optimizer.state, keep_param_ids)
    inner_optimizer.param_groups = []

    for group in normalized_network_groups:
        inner_optimizer.add_param_group(group)
    for group in preserved_extra_groups:
        inner_optimizer.add_param_group(group)
    self._refresh_prodigy_plus_late_param_group_state(inner_optimizer, split_groups=prodigy_plus_split_groups)

    for group in inner_optimizer.param_groups:
        if "initial_lr" in group:
            group["lr"] = group["initial_lr"]

    scheduler_args = args
    if recovery_config is not None:
        scheduler_args = argparse.Namespace(**vars(args))
        scheduler_args.max_train_steps = max(1, int(recovery_config["steps"]))
        scheduler_args.lr_warmup_steps = int(recovery_config.get("warmup_steps", 0))
        recover_scheduler = recovery_config.get("scheduler")
        if recover_scheduler is None:
            recover_scheduler = "constant_with_warmup" if scheduler_args.lr_warmup_steps > 0 else "constant"
        scheduler_args.lr_scheduler = recover_scheduler
    new_inner_scheduler = self.get_lr_scheduler(scheduler_args, inner_optimizer, accelerator.num_processes)
    if recovery_config is None and global_step > 0 and not self.is_schedulefree_optimizer(inner_optimizer, args):
        for _ in range(global_step):
            new_inner_scheduler.step()

    if hasattr(lr_scheduler, "scheduler"):
        lr_scheduler.scheduler = new_inner_scheduler
        refreshed_scheduler = lr_scheduler
    else:
        refreshed_scheduler = new_inner_scheduler

    return refreshed_scheduler, lr_descriptions


def refresh_optimizer_param_groups_after_adaptive_rank_resume(
    self,
    args: argparse.Namespace,
    accelerator,
    network: Any,
    optimizer: torch.optim.Optimizer,
    *,
    old_network_param_ids: set[int],
) -> list[str]:
    unwrapped_network = accelerator.unwrap_model(network)
    new_network_param_groups, lr_descriptions = self._prepare_network_optimizer_params(args, unwrapped_network)

    inner_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    prodigy_plus_split_groups = self._get_prodigy_plus_split_groups(inner_optimizer)

    preserved_extra_groups = []
    preserved_extra_param_ids: set[int] = set()
    for group in inner_optimizer.param_groups:
        extra_params = [param for param in group["params"] if id(param) not in old_network_param_ids]
        if not extra_params:
            continue
        group_copy = {k: v for k, v in group.items() if k != "params"}
        group_copy["params"] = extra_params
        preserved_extra_groups.append(group_copy)
        preserved_extra_param_ids.update(id(param) for param in extra_params)

    normalized_network_groups = []
    new_network_param_ids: set[int] = set()
    for group in new_network_param_groups:
        group_copy = {k: v for k, v in group.items() if k != "params"}
        params = list(group["params"])
        group_copy["params"] = params
        normalized_network_groups.append(group_copy)
        new_network_param_ids.update(id(param) for param in params)

    keep_param_ids = new_network_param_ids | preserved_extra_param_ids
    inner_optimizer.state = self._copy_optimizer_state_subset(inner_optimizer.state, keep_param_ids)
    inner_optimizer.param_groups = []

    for group in normalized_network_groups:
        inner_optimizer.add_param_group(group)
    for group in preserved_extra_groups:
        inner_optimizer.add_param_group(group)
    self._refresh_prodigy_plus_late_param_group_state(inner_optimizer, split_groups=prodigy_plus_split_groups)

    return lr_descriptions


def register_optimizer_resume_safe_globals(args: argparse.Namespace) -> None:
    optimizer_type = getattr(args, "optimizer_type", None)
    try:
        from musubi_tuner.optimizers.backends import is_apollo_optimizer_type, register_apollo_resume_safe_globals

        if is_apollo_optimizer_type(optimizer_type):
            register_apollo_resume_safe_globals()
    except Exception as exc:
        logger.debug("could not register optimizer resume safe globals for %s: %s", optimizer_type, exc)
    try:
        from musubi_tuner.optimizers.q_galore import is_qgalore_optimizer_type, register_qgalore_resume_safe_globals

        if is_qgalore_optimizer_type(optimizer_type):
            register_qgalore_resume_safe_globals()
    except Exception as exc:
        logger.debug("could not register Q-GaLore resume safe globals for %s: %s", optimizer_type, exc)


def get_lr_scheduler(self, args, optimizer: torch.optim.Optimizer, num_processes: int):
    """
    Unified API to get any scheduler from its name.
    """
    # if schedulefree optimizer, return dummy scheduler
    if self.is_schedulefree_optimizer(optimizer, args):
        return self.get_dummy_scheduler(optimizer)

    name = args.lr_scheduler
    num_training_steps = args.max_train_steps * num_processes  # * args.gradient_accumulation_steps
    num_warmup_steps: Optional[int] = (
        int(args.lr_warmup_steps * num_training_steps) if isinstance(args.lr_warmup_steps, float) else args.lr_warmup_steps
    )
    num_decay_steps: Optional[int] = (
        int(args.lr_decay_steps * num_training_steps) if isinstance(args.lr_decay_steps, float) else args.lr_decay_steps
    )
    num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
    num_cycles = args.lr_scheduler_num_cycles
    power = args.lr_scheduler_power
    timescale = args.lr_scheduler_timescale
    min_lr_ratio = args.lr_scheduler_min_lr_ratio
    group_lr_warmup_overrides = parse_group_lr_warmup_args(getattr(args, "lr_group_warmup_args", None))

    lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
    if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
        for arg in args.lr_scheduler_args:
            if "=" not in arg:
                raise ValueError(f"Invalid --lr_scheduler_args entry (expected key=value): {arg}")
            key, value = arg.split("=", 1)
            value = ast.literal_eval(value)
            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
        return return_vals

    # using any lr_scheduler from other library
    if args.lr_scheduler_type:
        lr_scheduler_type = args.lr_scheduler_type
        logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
        if "." not in lr_scheduler_type:  # default to use torch.optim
            lr_scheduler_module = torch.optim.lr_scheduler
        else:
            values = lr_scheduler_type.split(".")
            lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
            lr_scheduler_type = values[-1]
        lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        return self._maybe_wrap_group_warmup_scheduler(lr_scheduler, optimizer, num_warmup_steps, group_lr_warmup_overrides)

    if name.startswith("adafactor"):
        assert type(optimizer) == transformers.optimization.Adafactor, (
            "adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
        )
        initial_lr = float(name.split(":")[1])
        # logger.info(f"adafactor scheduler init lr {initial_lr}")
        return self._maybe_wrap_group_warmup_scheduler(
            wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr)),
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    if name.lower() == "rex":
        return self._maybe_wrap_group_warmup_scheduler(
            RexLR(
                optimizer,
                max_lr=args.learning_rate,
                min_lr=(  # Will start and end with min_lr, use non-zero min_lr by default
                    args.learning_rate * min_lr_ratio if min_lr_ratio is not None else args.learning_rate * 0.01
                ),
                num_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                **lr_scheduler_kwargs,
            ),
            optimizer,
            num_warmup_steps=num_warmup_steps,
            warmup_overrides=group_lr_warmup_overrides,
        )

    if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
        name = DiffusersSchedulerType(name)
        schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
        return self._maybe_wrap_group_warmup_scheduler(
            schedule_func(optimizer, **lr_scheduler_kwargs),  # step_rules and last_epoch are given as kwargs
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return self._maybe_wrap_group_warmup_scheduler(
            wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs)),
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return self._maybe_wrap_group_warmup_scheduler(
            schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs),
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    if name == SchedulerType.INVERSE_SQRT:
        return self._maybe_wrap_group_warmup_scheduler(
            schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs),
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return self._maybe_wrap_group_warmup_scheduler(
            schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                **lr_scheduler_kwargs,
            ),
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    if name == SchedulerType.POLYNOMIAL:
        return self._maybe_wrap_group_warmup_scheduler(
            schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=power,
                **lr_scheduler_kwargs,
            ),
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    if name == SchedulerType.COSINE_WITH_MIN_LR:
        return self._maybe_wrap_group_warmup_scheduler(
            schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles / 2,
                min_lr_rate=min_lr_ratio,
                **lr_scheduler_kwargs,
            ),
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    # these schedulers do not require `num_decay_steps`
    if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
        return self._maybe_wrap_group_warmup_scheduler(
            schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **lr_scheduler_kwargs,
            ),
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    # All other schedulers require `num_decay_steps`
    if num_decay_steps is None:
        raise ValueError(f"{name} requires `num_decay_steps`, please provide that argument.")
    if name == SchedulerType.WARMUP_STABLE_DECAY:
        return self._maybe_wrap_group_warmup_scheduler(
            schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_stable_steps=num_stable_steps,
                num_decay_steps=num_decay_steps,
                num_cycles=num_cycles / 2,
                min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
                **lr_scheduler_kwargs,
            ),
            optimizer,
            num_warmup_steps,
            group_lr_warmup_overrides,
        )

    return self._maybe_wrap_group_warmup_scheduler(
        schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps,
            **lr_scheduler_kwargs,
        ),
        optimizer,
        num_warmup_steps,
        group_lr_warmup_overrides,
    )
