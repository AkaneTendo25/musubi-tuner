from __future__ import annotations

import logging
import re
import weakref
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from musubi_tuner.utils.safetensors_utils import LazyTensorForSave

try:
    from bitsandbytes.optim.optimizer import Optimizer2State
    import bitsandbytes.functional as bnb_F
except ImportError:  # pragma: no cover - exercised on systems without bitsandbytes
    Optimizer2State = None
    bnb_F = None


logger = logging.getLogger(__name__)


QGALORE_OPTIMIZER_ALIASES = {
    "qgalore",
    "q_galore",
    "qgaloreadamw8bit",
    "q_galore_adamw8bit",
    "q-galore-adamw8bit",
}


def is_qgalore_optimizer_type(optimizer_type: str | None) -> bool:
    if not optimizer_type:
        return False
    return optimizer_type.lower() in QGALORE_OPTIMIZER_ALIASES


def is_qgalore_optimizer_instance(optimizer: Any) -> bool:
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    return isinstance(base_optimizer, QGaLoreAdamW8bit)


def is_qgalore_parameter(parameter: Any) -> bool:
    return bool(getattr(parameter, "_qgalore_weight", False))


def register_qgalore_resume_safe_globals() -> None:
    """Allow PyTorch 2.6+ weights-only loading of Q-GaLore optimizer state."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([QGaLoreProjector])


def _quantize_affine_uint(
    weight: torch.Tensor,
    *,
    group_size: int = -1,
    n_bit: int = 8,
    stochastic_round: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_shape = weight.shape
    if group_size is not None and group_size > 0:
        if weight.numel() % group_size != 0:
            raise ValueError(f"Tensor with {weight.numel()} elements is not divisible by group_size={group_size}")
        weight = weight.reshape(-1, group_size)
    else:
        weight = weight.reshape(1, -1) if weight.dim() == 1 else weight

    if weight.dim() != 2:
        raise ValueError(f"Expected a 2-D tensor for affine quantization, got {weight.dim()}-D")

    max_int = 2**n_bit - 1
    min_int = 0
    compute = weight.float()
    max_val = compute.amax(dim=1, keepdim=True)
    min_val = compute.amin(dim=1, keepdim=True)
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    q = compute / scales
    if stochastic_round:
        down = torch.floor(q)
        prob = q - down
        q = torch.where(torch.rand_like(prob) < prob, down + 1.0, down)
    else:
        q = torch.round(q)
    q = torch.clamp(q + zeros, min_int, max_int).reshape(original_shape).to(torch.uint8)
    return q, scales.to(torch.float32), zeros.to(torch.float32)


def _dequantize_affine_uint(
    weight: torch.Tensor,
    *,
    dtype: torch.dtype,
    group_size: int,
    scales: torch.Tensor,
    zeros: torch.Tensor,
) -> torch.Tensor:
    original_shape = weight.shape
    if group_size is not None and group_size > 0:
        dequant = weight.to(scales.dtype).reshape(-1, group_size)
    else:
        dequant = weight.to(scales.dtype).reshape(scales.shape[0], -1)
    dequant = (dequant - zeros) * scales
    return dequant.reshape(original_shape).to(dtype)


class QGaLoreProjector:
    def __init__(
        self,
        rank: int,
        *,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
        quant: bool = True,
        group_size: int = 256,
        n_bit: int = 4,
        cos_threshold: float = 0.4,
        gamma_proj: float = 2.0,
        queue_size: int = 5,
        svd_method: str = "full",
        svd_oversampling: int = 32,
        svd_niter: int = 1,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"Q-GaLore rank must be > 0, got {rank}")
        if update_proj_gap <= 0:
            raise ValueError(f"Q-GaLore update_proj_gap must be > 0, got {update_proj_gap}")
        if proj_type != "std":
            raise ValueError("Only Q-GaLore proj_type='std' is currently supported")

        self.rank = int(rank)
        self.update_proj_gap = int(update_proj_gap)
        self.scale = float(scale)
        self.proj_type = proj_type
        self.quant = bool(quant)
        self.quant_group_size = int(group_size)
        self.quant_n_bit = int(n_bit)
        self.cos_threshold = float(cos_threshold)
        self.gamma_proj = float(gamma_proj)
        self.queue_size = int(queue_size)
        self.svd_method = str(svd_method).lower()
        if self.svd_method not in {"full", "lowrank"}:
            raise ValueError(f"Unsupported Q-GaLore svd_method={svd_method!r}; expected 'full' or 'lowrank'")
        self.svd_oversampling = int(svd_oversampling)
        self.svd_niter = int(svd_niter)

        self.ortho_matrix: torch.Tensor | None = None
        self.ortho_matrix_scales: torch.Tensor | None = None
        self.ortho_matrix_zeros: torch.Tensor | None = None
        self.ortho_matrix_shape: torch.Size | None = None
        self.ortho_matrix_group_size: int = int(group_size)
        self.ortho_matrix_dtype: torch.dtype | None = None
        self.past_ortho_vector: torch.Tensor | None = None
        self.queue: list[float] = []
        self.svd_count = 0

    def project(self, full_rank_grad: torch.Tensor, step: int) -> torch.Tensor:
        if full_rank_grad.dim() != 2:
            raise RuntimeError(f"Q-GaLore expects 2-D gradients, got shape {tuple(full_rank_grad.shape)}")

        self._move_state_to_device(full_rank_grad.device)
        use_right = full_rank_grad.shape[0] >= full_rank_grad.shape[1]
        if self.ortho_matrix is None or step % self.update_proj_gap == 0:
            float_ortho_matrix = self._get_orthogonal_matrix(full_rank_grad, "right" if use_right else "left")
            self._maybe_extend_gap(float_ortho_matrix, use_right=use_right)
            self._store_ortho_matrix(float_ortho_matrix)
            self.svd_count += 1

        ortho = self._load_ortho_matrix()
        if use_right:
            return torch.matmul(full_rank_grad, ortho.t())
        return torch.matmul(ortho.t(), full_rank_grad)

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        self._move_state_to_device(low_rank_grad.device)
        ortho = self._load_ortho_matrix()
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, ortho)
        else:
            full_rank_grad = torch.matmul(ortho, low_rank_grad)
        return full_rank_grad * self.scale

    def _move_state_to_device(self, device: torch.device) -> None:
        if self.ortho_matrix is not None and self.ortho_matrix.device != device:
            self.ortho_matrix = self.ortho_matrix.to(device=device)
        if self.ortho_matrix_scales is not None and self.ortho_matrix_scales.device != device:
            self.ortho_matrix_scales = self.ortho_matrix_scales.to(device=device)
        if self.ortho_matrix_zeros is not None and self.ortho_matrix_zeros.device != device:
            self.ortho_matrix_zeros = self.ortho_matrix_zeros.to(device=device)
        if self.past_ortho_vector is not None and self.past_ortho_vector.device != device:
            self.past_ortho_vector = self.past_ortho_vector.to(device=device)

    def _maybe_extend_gap(self, ortho_matrix: torch.Tensor, *, use_right: bool) -> None:
        if self.past_ortho_vector is not None and self.queue_size > 0:
            current = ortho_matrix[:1, :].flatten() if use_right else ortho_matrix[:, :1].flatten()
            self.queue.append(F.cosine_similarity(self.past_ortho_vector, current.detach().clone(), dim=0).item())
            if len(self.queue) > self.queue_size:
                self.queue.pop(0)
            if len(self.queue) == self.queue_size and sum(self.queue) / self.queue_size >= self.cos_threshold:
                self.update_proj_gap = int(max(self.update_proj_gap + 1, self.update_proj_gap * self.gamma_proj))
        self.past_ortho_vector = (ortho_matrix[:1, :] if use_right else ortho_matrix[:, :1]).detach().clone().flatten()

    def _store_ortho_matrix(self, ortho_matrix: torch.Tensor) -> None:
        if not self.quant:
            self.ortho_matrix = ortho_matrix
            self.ortho_matrix_scales = None
            self.ortho_matrix_zeros = None
            self.ortho_matrix_shape = None
            self.ortho_matrix_group_size = self.quant_group_size
            self.ortho_matrix_dtype = ortho_matrix.dtype
            return

        group_size = self.quant_group_size
        if group_size > 0 and ortho_matrix.numel() % group_size != 0:
            group_size = -1
        q, scales, zeros = _quantize_affine_uint(
            ortho_matrix,
            group_size=group_size,
            n_bit=self.quant_n_bit,
            stochastic_round=False,
        )
        self.ortho_matrix = q
        self.ortho_matrix_scales = scales
        self.ortho_matrix_zeros = zeros
        self.ortho_matrix_shape = ortho_matrix.shape
        self.ortho_matrix_group_size = group_size
        self.ortho_matrix_dtype = ortho_matrix.dtype

    def _load_ortho_matrix(self) -> torch.Tensor:
        if self.ortho_matrix is None:
            raise RuntimeError("Q-GaLore projection matrix has not been initialized")
        if not self.quant:
            return self.ortho_matrix
        if self.ortho_matrix_scales is None or self.ortho_matrix_zeros is None or self.ortho_matrix_shape is None:
            raise RuntimeError("Q-GaLore quantized projection state is incomplete")
        return _dequantize_affine_uint(
            self.ortho_matrix,
            dtype=self.ortho_matrix_dtype or self.ortho_matrix_scales.dtype,
            group_size=self.ortho_matrix_group_size,
            scales=self.ortho_matrix_scales,
            zeros=self.ortho_matrix_zeros,
        ).reshape(self.ortho_matrix_shape)

    def _get_orthogonal_matrix(self, grad: torch.Tensor, side: str) -> torch.Tensor:
        matrix = grad.float() if grad.dtype != torch.float32 else grad
        rank = min(self.rank, min(matrix.shape))
        if self.svd_method == "lowrank" and rank < min(matrix.shape):
            q = min(rank + max(0, self.svd_oversampling), min(matrix.shape))
            u, _, v = torch.svd_lowrank(matrix, q=q, niter=max(0, self.svd_niter))
            if side == "right":
                return v[:, :rank].t().contiguous().to(device=grad.device, dtype=grad.dtype)
            return u[:, :rank].contiguous().to(device=grad.device, dtype=grad.dtype)

        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        if side == "right":
            return vh[:rank, :].to(device=grad.device, dtype=grad.dtype)
        return u[:, :rank].to(device=grad.device, dtype=grad.dtype)


class QGaLoreLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Parameter, bias: Parameter | None) -> Tensor:
        float_weight = _dequantize_affine_uint(
            weight,
            dtype=x.dtype,
            group_size=int(weight.group_size),
            scales=weight.scales,
            zeros=weight.zeros,
        )
        ctx.has_bias = bias is not None
        if bias is None:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(x, weight, bias)
        return F.linear(x, float_weight, bias)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
            bias = None
        float_weight = _dequantize_affine_uint(
            weight,
            dtype=grad_output.dtype,
            group_size=int(weight.group_size),
            scales=weight.scales,
            zeros=weight.zeros,
        )
        grad_input = grad_output @ float_weight

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.reshape(-1, bias.shape[0]).sum(0)

        out_features, in_features = weight.shape
        grad_weight = grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features)
        if getattr(weight, "float_grad", None) is None:
            weight.float_grad = grad_weight
        else:
            weight.float_grad = weight.float_grad + grad_weight

        backward_hook = getattr(weight, "backward_hook", None)
        if callable(backward_hook):
            backward_hook(weight)

        return grad_input, None, grad_bias


class QGaLoreLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        *,
        weight_bits: int = 8,
        group_size: int = 256,
        stochastic_round: bool = True,
    ) -> None:
        super().__init__()
        if weight_bits != 8:
            raise NotImplementedError("Q-GaLore weight quantization currently supports only 8 bits")
        if source.weight.dim() != 2:
            raise ValueError(f"Q-GaLore can only wrap 2-D Linear weights, got shape {tuple(source.weight.shape)}")
        if group_size > 0 and source.weight.numel() % group_size != 0:
            raise ValueError(
                f"Q-GaLore Linear weight with {source.weight.numel()} elements is not divisible by group_size={group_size}"
            )

        q_weight, scales, zeros = _quantize_affine_uint(
            source.weight.detach(),
            group_size=group_size,
            n_bit=weight_bits,
            stochastic_round=False,
        )
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.compute_dtype = source.weight.dtype
        self.group_size = int(group_size)
        self.weight_bits = int(weight_bits)
        self.stochastic_round = bool(stochastic_round)

        self.weight = Parameter(q_weight.to(device=source.weight.device), requires_grad=False)
        self.register_buffer("scales", scales.to(device=source.weight.device))
        self.register_buffer("zeros", zeros.to(device=source.weight.device))
        if source.bias is None:
            self.bias = None
        else:
            self.bias = Parameter(source.bias.detach().clone(), requires_grad=source.bias.requires_grad)
        self._refresh_weight_attrs()

    def _refresh_weight_attrs(self) -> None:
        self.weight._qgalore_weight = True
        self.weight._qgalore_owner = weakref.ref(self)
        self.weight.scales = self.scales
        self.weight.zeros = self.zeros
        self.weight.group_size = self.group_size
        self.weight.stochastic_round = self.stochastic_round
        self.weight.float_grad = getattr(self.weight, "float_grad", None)

    def _apply(self, fn):
        super()._apply(fn)
        self._refresh_weight_attrs()
        return self

    def forward(self, input: Tensor) -> Tensor:
        self._refresh_weight_attrs()
        return QGaLoreLinearFunction.apply(input, self.weight, self.bias)

    @torch.no_grad()
    def dequantized_weight(
        self,
        dtype: torch.dtype | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        self._refresh_weight_attrs()
        weight = self.weight
        scales = self.scales
        zeros = self.zeros
        if device is not None:
            weight = weight.to(device=device)
            scales = scales.to(device=device)
            zeros = zeros.to(device=device)
        return _dequantize_affine_uint(
            weight,
            dtype=dtype or self.compute_dtype,
            group_size=self.group_size,
            scales=scales,
            zeros=zeros,
        )


@dataclass
class QGaLoreReplacementSummary:
    replaced: int = 0
    skipped: int = 0
    skipped_not_target: int = 0
    skipped_small: int = 0
    skipped_group_size: int = 0
    replaced_numel: int = 0
    replaced_names: list[str] | None = None


def _parse_target_tokens(targets: str | Iterable[str]) -> set[str]:
    if isinstance(targets, str):
        raw_tokens = re.split(r"[,+]", targets)
    else:
        raw_tokens = list(targets)
    tokens = {str(token).strip().lower() for token in raw_tokens if str(token).strip()}
    return tokens or {"video"}


def _matches_ltx2_target(module_name: str, tokens: set[str]) -> bool:
    if "all" in tokens:
        return True
    in_block = re.search(r"(?:^|\.)transformer_blocks\.\d+\.", module_name) is not None
    if "blocks" in tokens and in_block:
        return True
    if "non_block" in tokens and not in_block:
        return True
    if "video" in tokens and re.search(r"\.transformer_blocks\.\d+\.(?:attn1|attn2|ff)\.", "." + module_name):
        return True
    if "audio" in tokens and re.search(
        r"\.transformer_blocks\.\d+\.(?:audio_attn1|audio_attn2|audio_ff|audio_to_video_attn|video_to_audio_attn)\.",
        "." + module_name,
    ):
        return True
    if "ff" in tokens and re.search(r"\.transformer_blocks\.\d+\.(?:ff|audio_ff)\.", "." + module_name):
        return True
    if "attn" in tokens and re.search(
        r"\.transformer_blocks\.\d+\.(?:attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.",
        "." + module_name,
    ):
        return True
    return False


def replace_ltx2_linear_with_qgalore(
    model: nn.Module,
    *,
    targets: str | Iterable[str] = "video",
    weight_bits: int = 8,
    weight_group_size: int = 256,
    stochastic_round: bool = True,
    min_weight_numel: int = 16384,
    max_modules: int | None = None,
) -> QGaLoreReplacementSummary:
    tokens = _parse_target_tokens(targets)
    summary = QGaLoreReplacementSummary(replaced_names=[])

    modules = list(model.named_modules())
    module_by_name = dict(modules)
    for module_name, module in modules:
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_ltx2_target(module_name, tokens):
            summary.skipped_not_target += 1
            continue
        if module.weight.numel() < int(min_weight_numel):
            summary.skipped_small += 1
            continue
        if weight_group_size > 0 and module.weight.numel() % int(weight_group_size) != 0:
            summary.skipped_group_size += 1
            continue
        if max_modules is not None and summary.replaced >= max_modules:
            summary.skipped += 1
            continue

        parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        parent = module_by_name[parent_name] if parent_name else model
        parent._modules[child_name] = QGaLoreLinear(
            module,
            weight_bits=weight_bits,
            group_size=int(weight_group_size),
            stochastic_round=stochastic_round,
        )
        summary.replaced += 1
        summary.replaced_numel += int(module.weight.numel())
        summary.replaced_names.append(module_name)

    return summary


def qgalore_group_kwargs_from_args(args: Any) -> dict[str, Any]:
    return {
        "rank": int(getattr(args, "qgalore_rank", 256)),
        "update_proj_gap": int(getattr(args, "qgalore_update_proj_gap", 200)),
        "scale": float(getattr(args, "qgalore_scale", 0.25)),
        "proj_type": str(getattr(args, "qgalore_proj_type", "std")),
        "quant": bool(getattr(args, "qgalore_proj_quant", True)),
        "quant_n_bit": int(getattr(args, "qgalore_proj_bits", 4)),
        "quant_group_size": int(getattr(args, "qgalore_proj_group_size", 256)),
        "cos_threshold": float(getattr(args, "qgalore_cos_threshold", 0.4)),
        "gamma_proj": float(getattr(args, "qgalore_gamma_proj", 2.0)),
        "queue_size": int(getattr(args, "qgalore_queue_size", 5)),
        "svd_method": str(getattr(args, "qgalore_svd_method", "full")),
        "svd_oversampling": int(getattr(args, "qgalore_svd_oversampling", 32)),
        "svd_niter": int(getattr(args, "qgalore_svd_niter", 1)),
    }


def dequantize_qgalore_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
    lazy: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    for module_name, module in model.named_modules():
        if not isinstance(module, QGaLoreLinear):
            continue
        weight_key = f"{module_name}.weight"
        target_dtype = dtype or module.compute_dtype
        if lazy:
            state_dict[weight_key] = LazyTensorForSave(
                shape=tuple(module.weight.shape),
                dtype=target_dtype,
                materialize_fn=lambda module=module, target_dtype=target_dtype: module.dequantized_weight(
                    dtype=target_dtype,
                    device=device,
                ),
            )
        else:
            state_dict[weight_key] = module.dequantized_weight(dtype=target_dtype)
        state_dict.pop(f"{module_name}.scales", None)
        state_dict.pop(f"{module_name}.zeros", None)
    return state_dict


_BaseOptimizer = torch.optim.Optimizer if Optimizer2State is None else Optimizer2State


class QGaLoreAdamW8bit(_BaseOptimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        optim_bits: int = 32,
        args=None,
        min_8bit_size: int = 4096,
        percentile_clipping: int = 100,
        block_wise: bool = True,
        is_paged: bool = False,
        synchronize_each_param: bool = False,
    ) -> None:
        if Optimizer2State is None:
            raise ImportError("QGaLoreAdamW8bit requires bitsandbytes")
        self.synchronize_each_param = bool(synchronize_each_param)
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            is_paged=is_paged,
        )

    @torch.no_grad()
    def step(self, closure=None, exchange_step: int = 0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True

        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                self._step_param_at(group, p, gindex, pindex)

        if self.is_paged:
            torch.cuda.synchronize()
        return loss

    @torch.no_grad()
    def step_param(self, p: torch.nn.Parameter, group: dict[str, Any]) -> None:
        location = self._find_param_location(p, group)
        if location is None:
            raise RuntimeError("Q-GaLore step_param received a parameter that is not in the optimizer")
        gindex, pindex = location
        self._step_param_at(group, p, gindex, pindex)

    def _find_param_location(self, p: torch.nn.Parameter, group: dict[str, Any] | None) -> tuple[int, int] | None:
        if group is not None:
            for gindex, candidate_group in enumerate(self.param_groups):
                if candidate_group is not group:
                    continue
                for pindex, candidate in enumerate(candidate_group.get("params", [])):
                    if candidate is p:
                        return gindex, pindex
        for gindex, candidate_group in enumerate(self.param_groups):
            for pindex, candidate in enumerate(candidate_group.get("params", [])):
                if candidate is p:
                    return gindex, pindex
        return None

    @torch.no_grad()
    def _step_param_at(self, group: dict[str, Any], p: torch.nn.Parameter, gindex: int, pindex: int) -> None:
        uses_float_grad = getattr(p, "float_grad", None) is not None
        if not uses_float_grad and p.grad is None:
            return

        if uses_float_grad:
            self._average_float_grad_if_distributed(p)
            float_weight = _dequantize_affine_uint(
                p.data,
                dtype=p.float_grad.dtype,
                group_size=int(p.group_size),
                scales=p.scales,
                zeros=p.zeros,
            )
            p.data = float_weight.clone().to(p.device)

        state = self.state[p]
        if "step" not in state:
            state["step"] = 0

        if "rank" in group:
            self._project_qgalore_gradient(group, p, state, uses_float_grad=uses_float_grad)

        if "state1" not in state:
            self.init_state(group, p, gindex, pindex)

        self.prefetch_state(p)
        self.update_step(group, p, gindex, pindex, uses_float_grad=uses_float_grad)
        if self.synchronize_each_param and torch.cuda.is_available():
            torch.cuda.synchronize()

        if "rank" in group:
            p.data = p.saved_data.add_(state["projector"].project_back(p.data))
            del p.saved_data
            if "weight_decay_saved" in group:
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay_saved"])
                group["weight_decay"] = group["weight_decay_saved"]
                del group["weight_decay_saved"]

        if uses_float_grad:
            self._requantize_qgalore_weight(p)

    @torch.no_grad()
    def _project_qgalore_gradient(
        self,
        group: dict[str, Any],
        p: torch.nn.Parameter,
        state: dict[str, Any],
        *,
        uses_float_grad: bool,
    ) -> None:
        if "projector" not in state:
            state["projector"] = QGaLoreProjector(
                int(group["rank"]),
                update_proj_gap=int(group["update_proj_gap"]),
                scale=float(group["scale"]),
                proj_type=str(group["proj_type"]),
                quant=bool(group["quant"]),
                group_size=int(group["quant_group_size"]),
                n_bit=int(group["quant_n_bit"]),
                cos_threshold=float(group["cos_threshold"]),
                gamma_proj=float(group["gamma_proj"]),
                queue_size=int(group["queue_size"]),
                svd_method=str(group.get("svd_method", "full")),
                svd_oversampling=int(group.get("svd_oversampling", 32)),
                svd_niter=int(group.get("svd_niter", 1)),
            )

        if group.get("weight_decay", 0) > 0:
            group["weight_decay_saved"] = group["weight_decay"]
            group["weight_decay"] = 0

        grad = p.float_grad if uses_float_grad else p.grad
        projected_grad = state["projector"].project(grad, state["step"])
        p.saved_data = p.data.clone()
        p.data = torch.zeros_like(projected_grad, dtype=p.data.dtype, device=p.data.device)
        if uses_float_grad:
            p.float_grad = projected_grad
        else:
            p.grad = projected_grad

    @torch.no_grad()
    def update_step(
        self, group: dict[str, Any], p: torch.nn.Parameter, gindex: int, pindex: int, uses_float_grad: bool = False
    ) -> None:
        if bnb_F is None:
            raise ImportError("QGaLoreAdamW8bit requires bitsandbytes")
        state = self.state[p]
        grad = p.float_grad if uses_float_grad else p.grad
        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]

        if config["percentile_clipping"] < 100:
            _, _, gnorm_scale = bnb_F.percentile_clipping(
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
                grad,
                p,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                config["lr"],
                state2=state["state2"],
                beta2=config["betas"][1],
                weight_decay=config["weight_decay"],
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )
        elif state["state1"].dtype == torch.uint8 and not config["block_wise"]:
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
                config["lr"],
                state["qmap1"],
                state["qmap2"],
                state["max1"],
                state["max2"],
                state["new_max1"],
                state["new_max2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
            )
            state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
            state["max2"], state["new_max2"] = state["new_max2"], state["max2"]
        elif state["state1"].dtype == torch.uint8 and config["block_wise"]:
            bnb_F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                0.0,
                0.0,
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if getattr(p, "float_grad", None) is not None:
                    if set_to_none:
                        p.float_grad = None
                    else:
                        p.float_grad.detach_()
                        p.float_grad.zero_()
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    @staticmethod
    def _average_float_grad_if_distributed(p: torch.nn.Parameter) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return
        grad_list = [torch.zeros_like(p.float_grad) for _ in range(world_size)]
        torch.distributed.all_gather(grad_list, p.float_grad)
        p.float_grad.copy_(sum(grad_list) / float(world_size))

    @torch.no_grad()
    def _requantize_qgalore_weight(self, p: torch.nn.Parameter) -> None:
        q_weight, scales, zeros = _quantize_affine_uint(
            p.data,
            group_size=int(p.group_size),
            n_bit=8,
            stochastic_round=bool(getattr(p, "stochastic_round", True)),
        )
        p.data = q_weight.to(p.device)
        p.scales = scales.to(p.device)
        p.zeros = zeros.to(p.device)
        p.float_grad = None

        owner_ref = getattr(p, "_qgalore_owner", None)
        owner = owner_ref() if callable(owner_ref) else None
        if owner is not None:
            owner._buffers["scales"] = p.scales
            owner._buffers["zeros"] = p.zeros
            owner._refresh_weight_attrs()
