from __future__ import annotations

import importlib
import types
from typing import Any

import torch


TORCHAO_OPTIMIZER_ALIASES: dict[str, str] = {
    "torchao_adam8bit": "Adam8bit",
    "torchao_adam4bit": "Adam4bit",
    "torchao_adamfp8": "AdamFp8",
    "torchao_adam_fp8": "AdamFp8",
    "torchao_adamw": "_AdamW",
    "torchao_adamw_sr": "_AdamW",
    "torchao_adamw8bit": "AdamW8bit",
    "torchao_adamw4bit": "AdamW4bit",
    "torchao_adamwfp8": "AdamWFp8",
    "torchao_adamw_fp8": "AdamWFp8",
    "ao_adam8bit": "Adam8bit",
    "ao_adam4bit": "Adam4bit",
    "ao_adamfp8": "AdamFp8",
    "ao_adam_fp8": "AdamFp8",
    "ao_adamw": "_AdamW",
    "ao_adamw_sr": "_AdamW",
    "ao_adamw8bit": "AdamW8bit",
    "ao_adamw4bit": "AdamW4bit",
    "ao_adamwfp8": "AdamWFp8",
    "ao_adamw_fp8": "AdamWFp8",
}


OPTIMI_OPTIMIZER_ALIASES: dict[str, str] = {
    "optimi_adam": "Adam",
    "optimi_adamw": "AdamW",
    "optimi_stableadamw": "StableAdamW",
    "optimi_stable_adamw": "StableAdamW",
    "optimi_lion": "Lion",
    "optimi_adan": "Adan",
    "optimi_radam": "RAdam",
    "optimi_ranger": "Ranger",
    "optimi_sgd": "SGD",
    "torchoptimi_adam": "Adam",
    "torchoptimi_adamw": "AdamW",
    "torchoptimi_stableadamw": "StableAdamW",
    "torchoptimi_stable_adamw": "StableAdamW",
    "torchoptimi_lion": "Lion",
    "torchoptimi_adan": "Adan",
    "torchoptimi_radam": "RAdam",
    "torchoptimi_ranger": "Ranger",
    "torchoptimi_sgd": "SGD",
}


def _load_class(module_names: tuple[str, ...], class_name: str, package_name: str):
    import_errors: list[str] = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue
        optimizer_class = getattr(module, class_name, None)
        if optimizer_class is not None:
            return optimizer_class

    detail = "; ".join(import_errors) if import_errors else f"{class_name} was not found"
    raise ImportError(
        f"{package_name} optimizer '{class_name}' is unavailable. Install {package_name} "
        f"or choose a different optimizer. Details: {detail}"
    )


def is_torchao_optimizer_type(optimizer_type: str) -> bool:
    opt = optimizer_type.lower()
    if opt in TORCHAO_OPTIMIZER_ALIASES:
        return True
    return opt.startswith("torchao.optim.") or opt.startswith("torchao.prototype.low_bit_optim.")


def resolve_torchao_optimizer_class(optimizer_type: str):
    opt = optimizer_type.lower()
    if opt in TORCHAO_OPTIMIZER_ALIASES:
        class_name = TORCHAO_OPTIMIZER_ALIASES[opt]
        return _load_class(
            ("torchao.optim", "torchao.prototype.low_bit_optim"),
            class_name,
            "torchao",
        )

    values = optimizer_type.split(".")
    module_name = ".".join(values[:-1])
    class_name = values[-1]
    return _load_class((module_name,), class_name, "torchao")


def is_optimi_optimizer_type(optimizer_type: str) -> bool:
    opt = optimizer_type.lower()
    if opt in OPTIMI_OPTIMIZER_ALIASES:
        return True
    return opt.startswith("optimi.")


def resolve_optimi_optimizer_class(optimizer_type: str):
    opt = optimizer_type.lower()
    if opt in OPTIMI_OPTIMIZER_ALIASES:
        class_name = OPTIMI_OPTIMIZER_ALIASES[opt]
    else:
        class_name = optimizer_type.split(".")[-1]
    return _load_class(("optimi",), class_name, "torch-optimi")


def is_torchao_optimizer_instance(optimizer: Any) -> bool:
    return optimizer.__class__.__module__.startswith("torchao.")


def is_optimi_optimizer_instance(optimizer: Any) -> bool:
    return optimizer.__class__.__module__.startswith("optimi.")


def patch_optimi_fused_step_param(optimizer: Any) -> bool:
    if callable(getattr(optimizer, "step_param", None)):
        return True
    if not is_optimi_optimizer_instance(optimizer):
        return False

    def step_param(self, p: torch.nn.Parameter, group: dict[str, Any]) -> None:
        if p.grad is None:
            return
        self.step(param=p)
        self.zero_grad(param=p)

    setattr(optimizer, "step_param", types.MethodType(step_param, optimizer))
    return True


def _find_torchao_single_param_adam(optimizer: Any):
    module_names = [optimizer.__class__.__module__, "torchao.optim.adam", "torchao.prototype.low_bit_optim"]
    seen: set[str] = set()
    for module_name in module_names:
        if module_name in seen:
            continue
        seen.add(module_name)
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        single_param_adam = getattr(module, "single_param_adam", None)
        if callable(single_param_adam):
            return single_param_adam
    return None


def patch_torchao_fused_step_param(optimizer: Any) -> bool:
    if callable(getattr(optimizer, "step_param", None)):
        return True
    if not is_torchao_optimizer_instance(optimizer):
        return False

    single_param_adam = _find_torchao_single_param_adam(optimizer)
    if single_param_adam is None or not callable(getattr(optimizer, "_new_buffer", None)):
        return False

    def step_param(self, p: torch.nn.Parameter, group: dict[str, Any]) -> None:
        if p.grad is None:
            return
        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Sparse gradient is not supported by torchao fused stepping")

        state = self.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = self._new_buffer(p, True)
            state["exp_avg_sq"] = self._new_buffer(p, False)
            if group["amsgrad"]:
                state["max_exp_avg_sq"] = self._new_buffer(p, False)

        state["step"] += 1
        if not isinstance(group["lr"], torch.Tensor):
            raise RuntimeError(
                "torchao optimizer lr must be a Tensor. If a scheduler changed it to a "
                "float, update it with optimizer.param_groups[i]['lr'].fill_(new_lr)."
            )

        single_param_adam(
            p.detach(),
            grad,
            state["step"],
            state["exp_avg"],
            state["exp_avg_sq"],
            state.get("max_exp_avg_sq", None),
            group["lr"],
            group["betas"][0],
            group["betas"][1],
            group["weight_decay"],
            group["eps"],
            self.is_adamw,
            self.bf16_stochastic_round and p.dtype is torch.bfloat16,
        )

    setattr(optimizer, "step_param", types.MethodType(step_param, optimizer))
    return True
