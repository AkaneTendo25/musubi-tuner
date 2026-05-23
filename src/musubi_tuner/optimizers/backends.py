from __future__ import annotations

import importlib
import math
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


APOLLO_OPTIMIZER_ALIASES: dict[str, str] = {
    "apollo": "APOLLOAdamW",
    "apollo_adamw": "APOLLOAdamW",
    "apolloadamw": "APOLLOAdamW",
    "apollo_adam_w": "APOLLOAdamW",
    "qapollo": "QAPOLLOAdamW",
    "q_apollo": "QAPOLLOAdamW",
    "qapollo_adamw": "QAPOLLOAdamW",
    "qapolloadamw": "QAPOLLOAdamW",
    "q_apollo_adamw": "QAPOLLOAdamW",
}


QAPOLLO_OPTIMIZER_ALIASES = {key for key, value in APOLLO_OPTIMIZER_ALIASES.items() if value == "QAPOLLOAdamW"}


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


def is_apollo_optimizer_type(optimizer_type: str | None) -> bool:
    if not optimizer_type:
        return False
    opt = optimizer_type.lower()
    if opt in APOLLO_OPTIMIZER_ALIASES:
        return True
    return opt.startswith("apollo_torch.")


def is_qapollo_optimizer_type(optimizer_type: str | None) -> bool:
    if not optimizer_type:
        return False
    opt = optimizer_type.lower()
    if opt in QAPOLLO_OPTIMIZER_ALIASES:
        return True
    return opt == "apollo_torch.qapolloadamw" or opt.startswith("apollo_torch.q_apollo.")


def resolve_apollo_optimizer_class(optimizer_type: str):
    opt = optimizer_type.lower()
    if opt in APOLLO_OPTIMIZER_ALIASES:
        class_name = APOLLO_OPTIMIZER_ALIASES[opt]
        return _load_class(("apollo_torch",), class_name, "apollo-torch")

    values = optimizer_type.split(".")
    module_name = ".".join(values[:-1])
    class_name = values[-1]
    return _load_class((module_name,), class_name, "apollo-torch")


def is_torchao_optimizer_instance(optimizer: Any) -> bool:
    return optimizer.__class__.__module__.startswith("torchao.")


def is_optimi_optimizer_instance(optimizer: Any) -> bool:
    return optimizer.__class__.__module__.startswith("optimi.")


def is_apollo_optimizer_instance(optimizer: Any) -> bool:
    return optimizer.__class__.__module__ in {"apollo_torch.apollo", "apollo_torch.q_apollo"}


def is_qapollo_optimizer_instance(optimizer: Any) -> bool:
    return optimizer.__class__.__module__ == "apollo_torch.q_apollo"


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


def apollo_group_kwargs_from_args(args: Any) -> dict[str, Any]:
    return {
        "rank": int(getattr(args, "apollo_rank", 256)),
        "update_proj_gap": int(getattr(args, "apollo_update_proj_gap", 200)),
        "scale": float(getattr(args, "apollo_scale", 1.0)),
        "proj": str(getattr(args, "apollo_proj", "random")),
        "proj_type": str(getattr(args, "apollo_proj_type", "std")),
        "scale_type": str(getattr(args, "apollo_scale_type", "channel")),
    }


def _find_param_location(optimizer: Any, p: torch.nn.Parameter, group: dict[str, Any] | None) -> tuple[int, int] | None:
    if group is not None:
        for gindex, candidate_group in enumerate(optimizer.param_groups):
            if candidate_group is not group:
                continue
            for pindex, candidate in enumerate(candidate_group.get("params", [])):
                if candidate is p:
                    return gindex, pindex
    for gindex, candidate_group in enumerate(optimizer.param_groups):
        for pindex, candidate in enumerate(candidate_group.get("params", [])):
            if candidate is p:
                return gindex, pindex
    return None


def _apollo_scaling_factor(group: dict[str, Any], norm_grad: torch.Tensor, grad: torch.Tensor, norm_dim: int) -> torch.Tensor:
    scale_type = str(group.get("scale_type", "channel"))
    if scale_type == "channel":
        scaling_factor = torch.norm(norm_grad, dim=norm_dim) / (torch.norm(grad, dim=norm_dim) + 1e-8)
        if norm_dim == 1:
            scaling_factor = scaling_factor.unsqueeze(1)
        return scaling_factor
    if scale_type == "tensor":
        return torch.norm(norm_grad) / (torch.norm(grad) + 1e-8)
    raise ValueError(f"Unsupported APOLLO scale_type={scale_type!r}; expected 'channel' or 'tensor'")


def _apollo_apply_norm_growth_limiter(
    optimizer: Any,
    state: dict[str, Any],
    group: dict[str, Any],
    scaled_grad: torch.Tensor,
) -> torch.Tensor:
    if bool(getattr(optimizer, "disable_nl", False)):
        return scaled_grad

    scaled_grad_norm = torch.norm(scaled_grad)
    if "scaled_grad" in state:
        limiter = max(scaled_grad_norm / (state["scaled_grad"] + 1e-8), 1.01) / 1.01
        scaled_grad = scaled_grad / limiter
        state["scaled_grad"] = scaled_grad_norm / limiter
    else:
        state["scaled_grad"] = scaled_grad_norm
    return scaled_grad


def _patch_apollo_adamw_fused_step_param(optimizer: Any) -> bool:
    def step_param(self, p: torch.nn.Parameter, group: dict[str, Any]) -> None:
        with torch.no_grad():
            if p.grad is None:
                return
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Sparse gradient is not supported by APOLLO fused stepping")

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0

            if "rank" in group:
                if grad.dim() != 2:
                    raise RuntimeError(f"APOLLO low-rank stepping expects 2-D gradients, got shape {tuple(grad.shape)}")
                norm_dim = 0 if grad.shape[0] < grad.shape[1] else 1
                if "projector" not in state:
                    state["projector"] = self._initialize_projector(group, state)
                low_rank_grad = state["projector"].project(grad, state["step"])
            else:
                low_rank_grad = grad
                norm_dim = 0

            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(low_rank_grad)
                state["exp_avg_sq"] = torch.zeros_like(low_rank_grad)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]
            state["step"] += 1

            exp_avg.mul_(beta1).add_(low_rank_grad, alpha=(1.0 - beta1))
            exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

            step_size = group["lr"]
            if group["correct_bias"]:
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            update = exp_avg / denom
            if "rank" in group:
                scaling_factor = _apollo_scaling_factor(group, update, low_rank_grad, norm_dim)
                update = grad * scaling_factor
                if bool(getattr(self, "scale_front", False)):
                    update = update * math.sqrt(float(group["scale"]))
                update = _apollo_apply_norm_growth_limiter(self, state, group, update)
                if not bool(getattr(self, "scale_front", False)):
                    update = update * math.sqrt(float(group["scale"]))

            p.add_(update, alpha=-step_size)
            if group["weight_decay"] > 0.0:
                p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

    setattr(optimizer, "step_param", types.MethodType(step_param, optimizer))
    return True


def _patch_qapollo_adamw_fused_step_param(optimizer: Any) -> bool:
    def step_param(self, p: torch.nn.Parameter, group: dict[str, Any]) -> None:
        with torch.no_grad():
            uses_float_grad = getattr(p, "float_grad", None) is not None
            if not uses_float_grad and p.grad is None:
                return

            location = _find_param_location(self, p, group)
            if location is None:
                raise RuntimeError("QAPOLLO step_param received a parameter that is not in the optimizer")
            gindex, pindex = location

            if not self.initialized:
                self.check_overrides()
                self.to_gpu()
                self.initialized = True

            if uses_float_grad:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size()
                    if world_size > 1:
                        grad_list = [torch.zeros_like(p.float_grad) for _ in range(world_size)]
                        torch.distributed.all_gather(grad_list, p.float_grad)
                        p.float_grad.copy_(sum(grad_list) / float(world_size))

                p.data = self._dequantize(p.data, p.float_grad.dtype, p.group_size, p.scales, p.zeros).clone().to(p.device)

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0

            saved_data: torch.Tensor | None = None
            full_rank_grad: torch.Tensor | None = None
            low_rank_grad: torch.Tensor | None = None
            if "rank" in group:
                full_rank_grad = p.float_grad if uses_float_grad else p.grad
                if full_rank_grad is None:
                    return
                if full_rank_grad.dim() != 2:
                    raise RuntimeError(f"QAPOLLO low-rank stepping expects 2-D gradients, got shape {tuple(full_rank_grad.shape)}")
                if "projector" not in state:
                    state["projector"] = self._initialize_projector(group, state)
                low_rank_grad = state["projector"].project(full_rank_grad, state["step"])
                saved_data = p.data.clone()
                p.data = torch.zeros_like(low_rank_grad, dtype=p.data.dtype, device=p.data.device)
                if uses_float_grad:
                    p.float_grad = low_rank_grad
                else:
                    p.grad = low_rank_grad

            if "state1" not in state:
                self.init_state(group, p, gindex, pindex)

            self.prefetch_state(p)
            self.update_step(group, p, gindex, pindex, flag_use_float_grad=uses_float_grad)

            if "rank" in group:
                if saved_data is None or full_rank_grad is None or low_rank_grad is None:
                    raise RuntimeError("QAPOLLO low-rank state is incomplete")
                norm_grad = p.data.clone()
                norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                scaling_factor = _apollo_scaling_factor(group, norm_grad, low_rank_grad, norm_dim)
                scaled_grad = full_rank_grad * scaling_factor
                if bool(getattr(self, "scale_front", False)):
                    scaled_grad = scaled_grad * math.sqrt(float(group["scale"]))
                scaled_grad = _apollo_apply_norm_growth_limiter(self, state, group, scaled_grad)
                if not bool(getattr(self, "scale_front", False)):
                    scaled_grad = scaled_grad * math.sqrt(float(group["scale"]))

                p.data = saved_data.add_(scaled_grad, alpha=-group["lr"])
                if group.get("weight_decay", 0) > 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

            if uses_float_grad:
                float_weight = p.data.clone()
                if bool(getattr(p, "stochastic_round", False)):
                    p.data, p.scales, p.zeros = self._quantize_stochastic_round(float_weight, q_group_size=p.group_size)
                else:
                    p.data, p.scales, p.zeros = self._quantize(float_weight, q_group_size=p.group_size)
                p.float_grad = None
                owner_ref = getattr(p, "_qgalore_owner", None)
                owner = owner_ref() if callable(owner_ref) else None
                if owner is not None:
                    owner._buffers["scales"] = p.scales
                    owner._buffers["zeros"] = p.zeros
                    owner._refresh_weight_attrs()

    setattr(optimizer, "step_param", types.MethodType(step_param, optimizer))
    return True


def patch_apollo_fused_step_param(optimizer: Any) -> bool:
    if callable(getattr(optimizer, "step_param", None)):
        return True
    if not is_apollo_optimizer_instance(optimizer):
        return False
    if is_qapollo_optimizer_instance(optimizer):
        return _patch_qapollo_adamw_fused_step_param(optimizer)
    return _patch_apollo_adamw_fused_step_param(optimizer)


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
