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


def register_apollo_resume_safe_globals() -> None:
    """Allow PyTorch 2.6+ weights-only loading of APOLLO optimizer state.

    APOLLO stores its projector object in the optimizer checkpoint. PyTorch 2.6
    changed torch.load's default to weights_only=True, so the class must be
    allowlisted before Accelerate loads optimizer.bin.
    """
    try:
        from apollo_torch.random_projector import GradientProjector
    except ImportError:
        return

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([GradientProjector])
    patch_apollo_projector_device_transfer()


def _move_apollo_projector_value(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device) if value.device != device else value
    if isinstance(value, list):
        return [_move_apollo_projector_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_apollo_projector_value(item, device) for item in value)
    return value


def _move_apollo_projector_to_device(projector: Any, device: torch.device) -> None:
    if not hasattr(projector, "ortho_matrix"):
        return
    projector.ortho_matrix = _move_apollo_projector_value(projector.ortho_matrix, device)


def _apollo_random_projector_project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
    """``project_back`` for the random ``GradientProjector``.

    apollo-torch's ``GradientProjector`` (``--apollo_proj random``) ships ``project``
    but no ``project_back``, so the Fira update rule (which needs the full-rank
    residual ``G - P P^T G``) cannot run on it. The projection is a plain matmul by
    ``ortho_matrix``, so the inverse mapping mirrors ``GaLoreProjector.project_back``
    exactly (same ``ortho_matrix`` / ``proj_type`` / ``scale`` conventions and the
    same branch on the low-rank shape).
    """
    proj_type = getattr(self, "proj_type", "std")
    ortho = self.ortho_matrix
    if proj_type == "std":
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, ortho)
        else:
            full_rank_grad = torch.matmul(ortho, low_rank_grad)
    elif proj_type == "reverse_std":
        if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(ortho, low_rank_grad)
        else:
            full_rank_grad = torch.matmul(low_rank_grad, ortho)
    elif proj_type == "right":
        full_rank_grad = torch.matmul(low_rank_grad, ortho)
    elif proj_type == "left":
        full_rank_grad = torch.matmul(ortho, low_rank_grad)
    else:
        raise NotImplementedError(f"random projector project_back is not supported for proj_type={proj_type!r}")
    return full_rank_grad * float(getattr(self, "scale", 1.0))


def patch_apollo_projector_device_transfer() -> None:
    """Make APOLLO projector checkpoints device-safe after optimizer resume."""
    try:
        from apollo_torch.random_projector import GradientProjector
        from apollo_torch.svd_projector import GaLoreProjector
    except ImportError:
        return

    # The random projector lacks project_back; supply one so Fira works with
    # --apollo_proj random (no SVD init cost). Added before the device-safety wrap
    # below so it gets the same device-transfer treatment as the native methods.
    if not hasattr(GradientProjector, "project_back"):
        GradientProjector.project_back = _apollo_random_projector_project_back

    for projector_class in (GradientProjector, GaLoreProjector):
        if not getattr(projector_class, "_musubi_device_safe_project", False):
            original_project = projector_class.project

            def project(self, full_rank_grad, iter, _original_project=original_project):
                _move_apollo_projector_to_device(self, full_rank_grad.device)
                return _original_project(self, full_rank_grad, iter)

            projector_class.project = project
            projector_class._musubi_device_safe_project = True

        if hasattr(projector_class, "project_back") and not getattr(projector_class, "_musubi_device_safe_project_back", False):
            original_project_back = projector_class.project_back

            def project_back(self, low_rank_grad, _original_project_back=original_project_back):
                _move_apollo_projector_to_device(self, low_rank_grad.device)
                return _original_project_back(self, low_rank_grad)

            projector_class.project_back = project_back
            projector_class._musubi_device_safe_project_back = True


def resolve_apollo_optimizer_class(optimizer_type: str):
    patch_apollo_projector_device_transfer()
    opt = optimizer_type.lower()
    if opt in APOLLO_OPTIMIZER_ALIASES:
        class_name = APOLLO_OPTIMIZER_ALIASES[opt]
        optimizer_class = _load_class(("apollo_torch",), class_name, "apollo-torch")
        if class_name == "QAPOLLOAdamW":
            return _patch_qapollo_adamw_optim_bits(optimizer_class)
        return optimizer_class

    values = optimizer_type.split(".")
    module_name = ".".join(values[:-1])
    class_name = values[-1]
    optimizer_class = _load_class((module_name,), class_name, "apollo-torch")
    if is_qapollo_optimizer_type(optimizer_type):
        return _patch_qapollo_adamw_optim_bits(optimizer_class)
    return optimizer_class


def is_torchao_optimizer_instance(optimizer: Any) -> bool:
    return optimizer.__class__.__module__.startswith("torchao.")


def is_optimi_optimizer_instance(optimizer: Any) -> bool:
    return optimizer.__class__.__module__.startswith("optimi.")


def is_apollo_optimizer_instance(optimizer: Any) -> bool:
    return bool(getattr(optimizer, "_musubi_apollo_optimizer", False)) or optimizer.__class__.__module__ in {
        "apollo_torch.apollo",
        "apollo_torch.q_apollo",
    }


def is_qapollo_optimizer_instance(optimizer: Any) -> bool:
    return bool(getattr(optimizer, "_musubi_qapollo_optimizer", False)) or optimizer.__class__.__module__ == "apollo_torch.q_apollo"


def _patch_qapollo_adamw_optim_bits(optimizer_class: Any):
    """Return a QAPOLLO AdamW class that honors optim_bits.

    apollo-torch 1.0.3 exposes an optim_bits argument on q_apollo.AdamW, but its
    constructor passes a hard-coded 32 to bitsandbytes Optimizer2State. Keep the
    upstream update logic and only replace the constructor so QAPOLLO can use
    quantized optimizer state when requested.
    """
    if getattr(optimizer_class, "_musubi_qapollo_honors_optim_bits", False):
        return optimizer_class

    try:
        from bitsandbytes.optim.optimizer import Optimizer2State
    except ImportError as exc:  # pragma: no cover - resolved by apollo-torch users
        raise ImportError("QAPOLLOAdamW requires bitsandbytes") from exc

    class MusubiQAPOLLOAdamW(optimizer_class):
        _musubi_apollo_optimizer = True
        _musubi_qapollo_optimizer = True
        _musubi_qapollo_honors_optim_bits = True

        def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False,
            optim_bits=8,
            args=None,
            min_8bit_size=4096,
            percentile_clipping=100,
            block_wise=True,
            is_paged=False,
            scale_front: bool = False,
            no_deprecation_warning: bool = True,
        ):
            del amsgrad, no_deprecation_warning  # kept for upstream signature compatibility
            optim_bits = int(optim_bits)
            if optim_bits not in {8, 32}:
                raise ValueError(f"QAPOLLOAdamW supports optim_bits=8 or 32, got {optim_bits}")

            Optimizer2State.__init__(
                self,
                "adam",
                params,
                lr,
                betas,
                eps,
                weight_decay,
                optim_bits,
                args,
                min_8bit_size,
                percentile_clipping,
                block_wise,
                is_paged=is_paged,
            )
            self.scale_front = scale_front
            self._musubi_optim_bits = optim_bits
            self.init_seeds()

        @torch.no_grad()
        def update_step(self, group, p, gindex, pindex, flag_use_float_grad=False):
            from bitsandbytes import functional as bnb_F

            p.data = p.data.contiguous()
            if flag_use_float_grad:
                p.float_grad = p.float_grad.contiguous()
                grad = p.float_grad
            else:
                p.grad = p.grad.contiguous()
                grad = p.grad

            state = self.state[p]
            config = self.get_config(gindex, pindex, group)

            lr = group["lr"]
            if "rank" in group:
                lr = 1.0

            state["step"] += 1
            step = state["step"]

            if config["percentile_clipping"] < 100:
                _current_gnorm, _clip_value, gnorm_scale = bnb_F.percentile_clipping(
                    grad,
                    state["gnorm_vec"],
                    step,
                    config["percentile_clipping"],
                )
            else:
                gnorm_scale = 1.0

            if state["state1"].dtype == torch.float:
                bnb_F.optimizer_update_32bit(
                    self.optimizer_name,
                    g=grad,
                    p=p,
                    state1=state["state1"],
                    beta1=config["betas"][0],
                    eps=config["eps"],
                    step=step,
                    lr=lr,
                    state2=state["state2"],
                    beta2=config["betas"][1],
                    gnorm_scale=gnorm_scale,
                    unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                    max_unorm=config["max_unorm"],
                    skip_zeros=config["skip_zeros"],
                    weight_decay=0.0,
                )
                return

            if state["state1"].dtype == torch.uint8 and not config["block_wise"]:
                bnb_F.optimizer_update_8bit(
                    self.optimizer_name,
                    grad,
                    p,
                    state["state1"],
                    state["state2"],
                    config["betas"][0],
                    config["betas"][1],
                    config["eps"],
                    step,
                    lr,
                    state["qmap1"],
                    state["qmap2"],
                    state["max1"],
                    state["max2"],
                    state["new_max1"],
                    state["new_max2"],
                    0.0,
                    gnorm_scale=gnorm_scale,
                    unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                    max_unorm=config["max_unorm"],
                )
                state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
                state["max2"], state["new_max2"] = state["new_max2"], state["max2"]
                return

            if state["state1"].dtype == torch.uint8 and config["block_wise"]:
                bnb_F.optimizer_update_8bit_blockwise(
                    self.optimizer_name,
                    grad,
                    p,
                    state["state1"],
                    state["state2"],
                    config["betas"][0],
                    config["betas"][1],
                    config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                    config.get("alpha", 0.0),
                    config["eps"],
                    step,
                    lr,
                    state["qmap1"],
                    state["qmap2"],
                    state["absmax1"],
                    state["absmax2"],
                    0.0,
                    gnorm_scale=gnorm_scale,
                    skip_zeros=config["skip_zeros"],
                )
                return

            raise RuntimeError(f"Unsupported QAPOLLO state dtype: {state['state1'].dtype}")

    MusubiQAPOLLOAdamW.__name__ = getattr(optimizer_class, "__name__", "QAPOLLOAdamW")
    MusubiQAPOLLOAdamW.__qualname__ = getattr(optimizer_class, "__qualname__", MusubiQAPOLLOAdamW.__name__)
    MusubiQAPOLLOAdamW.__module__ = getattr(optimizer_class, "__module__", __name__)
    return MusubiQAPOLLOAdamW


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
        "update_rule": str(getattr(args, "apollo_update_rule", "apollo")),
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


def _fira_split_update(
    projector: Any,
    norm_grad: torch.Tensor,
    low_rank_grad: torch.Tensor,
    full_rank_grad: torch.Tensor,
    scaling_factor: torch.Tensor,
    *,
    negate_proj_adam: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fira update decomposition for an APOLLO/QAPOLLO low-rank group.

    Returns ``(proj_adam, scaled_residual)`` where:
      * ``proj_adam`` is the exact in-subspace Adam direction projected back to full
        rank, ``P @ Adam(P^T G)``;
      * ``scaled_residual`` is the gradient residual outside the subspace,
        ``G - P P^T G``, scaled channel-wise by ``scaling_factor``.

    The caller is expected to apply the norm-growth limiter to ``scaled_residual``,
    sum the two terms, apply ``sqrt(scale)``, and finally apply the learning rate.

    ``negate_proj_adam`` accounts for the sign convention of the in-subspace Adam
    direction:
      * non-quantized APOLLO computes ``norm_grad = exp_avg / denom`` (= +Adam dir),
        so ``negate_proj_adam=False``;
      * QAPOLLO runs the bitsandbytes step with lr=1.0 into a zeroed ``p.data``, so
        ``norm_grad = p.data = -Adam dir`` and ``negate_proj_adam=True`` restores the
        descent sign.

    ``scaling_factor`` already broadcasts onto ``full_rank_grad`` in the existing
    APOLLO path; the residual shares ``full_rank_grad``'s shape, so the same
    broadcast applies unchanged.
    """
    project_back = getattr(projector, "project_back", None)
    if not callable(project_back):
        raise RuntimeError(
            "--apollo_update_rule fira requires a projector exposing project_back(); the active "
            f"projector {type(projector).__name__!r} does not. Use --apollo_proj random or svd."
        )
    proj_adam = project_back(norm_grad)
    if negate_proj_adam:
        proj_adam = -proj_adam
    residual = full_rank_grad - project_back(low_rank_grad)
    return proj_adam, scaling_factor * residual


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
                if str(group.get("update_rule", "apollo")) == "fira":
                    proj_adam, scaled_residual = _fira_split_update(
                        state["projector"], update, low_rank_grad, grad, scaling_factor, negate_proj_adam=False
                    )
                    scaled_residual = _apollo_apply_norm_growth_limiter(self, state, group, scaled_residual)
                    update = (proj_adam + scaled_residual) * math.sqrt(float(group["scale"]))
                else:
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
    def _qapollo_group_size(p: torch.nn.Parameter) -> int:
        group_size = int(getattr(p, "group_size", -1))
        if group_size == 0 and p.dim() == 2:
            return int(p.shape[-1])
        return group_size

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
                q_group_size = _qapollo_group_size(p)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size()
                    if world_size > 1:
                        grad_list = [torch.zeros_like(p.float_grad) for _ in range(world_size)]
                        torch.distributed.all_gather(grad_list, p.float_grad)
                        p.float_grad.copy_(sum(grad_list) / float(world_size))

                p.data = self._dequantize(p.data, p.float_grad.dtype, q_group_size, p.scales, p.zeros).clone().to(p.device)

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
                if str(group.get("update_rule", "apollo")) == "fira":
                    # QAPOLLO ran the bnb step with lr=1.0 into a zeroed p.data, so
                    # norm_grad = -Adam direction; negate_proj_adam restores the descent sign.
                    proj_adam, scaled_residual = _fira_split_update(
                        state["projector"], norm_grad, low_rank_grad, full_rank_grad, scaling_factor, negate_proj_adam=True
                    )
                    scaled_residual = _apollo_apply_norm_growth_limiter(self, state, group, scaled_residual)
                    scaled_grad = (proj_adam + scaled_residual) * math.sqrt(float(group["scale"]))
                else:
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
                q_group_size = _qapollo_group_size(p)
                if bool(getattr(p, "stochastic_round", False)):
                    p.data, p.scales, p.zeros = self._quantize_stochastic_round(float_weight, q_group_size=q_group_size)
                else:
                    p.data, p.scales, p.zeros = self._quantize(float_weight, q_group_size=q_group_size)
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
