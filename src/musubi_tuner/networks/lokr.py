# LoKr (Low-rank Kronecker Product) network module
# Linear layers only (no Conv2d/Tucker decomposition)
# Reference: https://arxiv.org/abs/2309.14859
#
# Based on the LyCORIS project by KohakuBlueleaf
# https://github.com/KohakuBlueleaf/LyCORIS

import ast
import copy
import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lora as lora_module
from .network_arch import detect_arch_config

logger = logging.getLogger(__name__)


def factorization(dimension: int, factor: int = -1) -> tuple:
    """Return a tuple of two values whose product equals dimension,
    optimized for balanced factors.

    In LoKr, the first value is for the weight scale (smaller),
    and the second value is for the weight (larger).

    Examples:
        factor=-1: 128 -> (8, 16), 512 -> (16, 32), 1024 -> (32, 32)
        factor=4:  128 -> (4, 32), 512 -> (4, 128)
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1, w2, scale):
    """Compute Kronecker product of w1 and w2, scaled by scale."""
    if w1.dim() != w2.dim():
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    if scale != 1:
        rebuild = rebuild * scale
    return rebuild


def _materialize_lokr_weight_from_state_dict(
    sd: Dict[str, torch.Tensor], scale: float, device: torch.device, non_blocking: bool = False
) -> torch.Tensor:
    """Materialize a dense LoKr delta weight from a saved state dict."""
    if "lokr_w1" in sd:
        w1 = sd["lokr_w1"].to(device, dtype=torch.float, non_blocking=non_blocking)
    else:
        w1a = sd["lokr_w1_a"].to(device, dtype=torch.float, non_blocking=non_blocking)
        w1b = sd["lokr_w1_b"].to(device, dtype=torch.float, non_blocking=non_blocking)
        w1 = w1a @ w1b

    if "lokr_w2" in sd:
        w2 = sd["lokr_w2"].to(device, dtype=torch.float, non_blocking=non_blocking)
    else:
        w2a = sd["lokr_w2_a"].to(device, dtype=torch.float, non_blocking=non_blocking)
        w2b = sd["lokr_w2_b"].to(device, dtype=torch.float, non_blocking=non_blocking)
        w2 = w2a @ w2b

    return make_kron(w1, w2, scale)


def _get_dokr_weight_norm(base_weight: torch.Tensor, delta_weight: torch.Tensor) -> torch.Tensor:
    """Return per-output norms for DokR's merged weight."""
    return torch.linalg.vector_norm(base_weight + delta_weight, dim=1)


def _magnitude_ratio_to_absolute(ratio: torch.Tensor, initial_norm: torch.Tensor) -> torch.Tensor:
    return ratio * initial_norm.to(device=ratio.device, dtype=ratio.dtype)


def _convert_absolute_magnitudes_to_ratios(state_dict: Dict[str, torch.Tensor]) -> None:
    for key in list(state_dict.keys()):
        if not key.endswith(".lora_magnitude_vector.weight"):
            continue
        module_name = key[: -len(".lora_magnitude_vector.weight")]
        initial_norm_key = f"{module_name}.initial_norm"
        if initial_norm_key not in state_dict:
            continue
        magnitude = state_dict[key]
        initial_norm = state_dict[initial_norm_key].to(dtype=magnitude.dtype)
        eps = 1e-12 if magnitude.dtype in (torch.float32, torch.float64) else 1e-6
        state_dict[key] = magnitude / initial_norm.clamp_min(eps)


def _solve_oft_block_size(n_elements: int) -> int:
    """Recover OFT block size from the flattened upper-triangle element count."""
    return int(round((1.0 + math.sqrt(1.0 + 8.0 * float(n_elements))) / 2.0))


def _metadata_tensor_to_int(value: Optional[torch.Tensor], default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, torch.Tensor):
        return int(round(float(value.detach().cpu().item())))
    return int(round(float(value)))


def _metadata_tensor_to_bool(value: Optional[torch.Tensor], default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, torch.Tensor):
        return bool(int(round(float(value.detach().cpu().item()))))
    return bool(value)


def _metadata_tensor_to_float(value: Optional[torch.Tensor], default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _resolve_oft_config_from_state_dict(
    module_state: Dict[str, torch.Tensor],
    in_features: int,
) -> tuple[int, bool, bool, float]:
    """Resolve OFT metadata from a saved state dict, with shape-based fallbacks."""
    n_elements = int(module_state["oft_R.weight"].shape[1])
    requested_block_size = _metadata_tensor_to_int(
        module_state.get("oft_block_size_metadata"),
        _solve_oft_block_size(n_elements),
    )
    block_size = lora_module.OFTModule._adjust_oft_block_size(int(in_features), requested_block_size)
    block_share = _metadata_tensor_to_bool(
        module_state.get("oft_block_share_metadata"),
        default=bool(int(module_state["oft_R.weight"].shape[0]) == 1),
    )
    coft = _metadata_tensor_to_bool(module_state.get("oft_coft_metadata"), default=False)
    coft_eps = _metadata_tensor_to_float(module_state.get("coft_eps_metadata"), 6e-5)
    return block_size, block_share, coft, coft_eps


def _materialized_lokr_w1_input_dim(module_state: Dict[str, torch.Tensor]) -> int:
    if "lokr_w1" in module_state:
        return int(module_state["lokr_w1"].shape[1])
    return int(module_state["lokr_w1_b"].shape[1])


def _lokr_w2_input_dim(module_state: Dict[str, torch.Tensor]) -> int:
    if "lokr_w2" in module_state:
        return int(module_state["lokr_w2"].shape[1])
    return int(module_state["lokr_w2_b"].shape[1])


def _rotate_weight_with_oft_state_dict(
    weight: torch.Tensor,
    module_state: Dict[str, torch.Tensor],
    multiplier: float,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Apply the saved OFT rotation to a dense weight tensor."""
    in_features = int(weight.shape[1])
    block_size, block_share, coft, coft_eps = _resolve_oft_config_from_state_dict(module_state, in_features)
    oft_weight = module_state["oft_R.weight"].to(device=weight.device, dtype=weight.dtype, non_blocking=non_blocking)
    rotation_module = lora_module.OFTRotationModule(
        r=int(oft_weight.shape[0]),
        n_elements=int(oft_weight.shape[1]),
        block_size=block_size,
        in_features=in_features,
        coft=coft,
        coft_eps=coft_eps,
        block_share=block_share,
        scaled_oft=True,
        use_cayley_neumann=True,
        num_cayley_neumann_terms=5,
        dropout_probability=0.0,
    ).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        rotation_module.weight.copy_(oft_weight)
    rotation = rotation_module.rotation_matrix(multiplier=multiplier).to(device=weight.device, dtype=weight.dtype)
    rank = in_features // block_size
    if block_share:
        rotation = rotation.repeat(rank, 1, 1)
    rotation = rotation.transpose(-1, -2)
    reshaped_weight = weight.reshape(weight.shape[0], rank, block_size)
    return torch.einsum("ork,rkc->orc", reshaped_weight, rotation).reshape_as(weight)


def _apply_dokr_oft_weight_merge(
    rotated_weight: torch.Tensor,
    magnitude: torch.Tensor,
    weight_norm: torch.Tensor,
) -> torch.Tensor:
    if weight_norm.is_floating_point():
        eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
        weight_norm = weight_norm.clamp_min(eps)
    factor = magnitude.to(device=weight_norm.device, dtype=weight_norm.dtype) / weight_norm
    return lora_module._reshape_dora_factor_for_weight(factor, rotated_weight) * rotated_weight


class LoKrModule(torch.nn.Module):
    """LoKr module for training. Replaces forward method of the original Linear."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        factor=-1,
        decompose_both=False,
        decompose_w1_rank=None,
        **kwargs,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("LoKr Conv2d is not supported in this implementation")
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        factor = int(factor)
        self.decompose_both = lora_module._parse_bool_network_arg(decompose_both)
        self.use_w2 = False

        # Factorize dimensions
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)

        # w1 is usually a full matrix (the "scale" factor, small), but can
        # also be decomposed for higher-order Kronecker structure.
        if self.decompose_both:
            w1_rank = max(1, min(int(decompose_w1_rank or lora_dim), out_l, in_m))
            self.lokr_w1_a = nn.Parameter(torch.empty(out_l, w1_rank))
            self.lokr_w1_b = nn.Parameter(torch.empty(w1_rank, in_m))
        else:
            self.lokr_w1 = nn.Parameter(torch.empty(out_l, in_m))

        # w2: low-rank decomposition if rank is small enough, otherwise full matrix
        if lora_dim < max(out_k, in_n) / 2:
            self.lokr_w2_a = nn.Parameter(torch.empty(out_k, lora_dim))
            self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, in_n))
        else:
            self.use_w2 = True
            self.lokr_w2 = nn.Parameter(torch.empty(out_k, in_n))
            if lora_dim >= max(out_k, in_n) / 2:
                logger.warning(
                    f"LoKr: lora_dim {lora_dim} is large for dim={max(in_dim, out_dim)} "
                    f"and factor={factor}, using full matrix mode."
                )

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        # if both w1 and w2 are full matrices, use scale = 1
        if self.use_w2:
            alpha = lora_dim
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Initialization
        if self.decompose_both:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.constant_(self.lokr_w1_b, 1)
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        if self.use_w2:
            torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.constant_(self.lokr_w2_b, 0)
        # Ensures ΔW = kron(w1, 0) = 0 at init

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def get_diff_weight(self):
        """Return materialized weight delta."""
        if self.decompose_both:
            w1 = self.lokr_w1_a @ self.lokr_w1_b
        else:
            w1 = self.lokr_w1
        if self.use_w2:
            w2 = self.lokr_w2
        else:
            w2 = self.lokr_w2_a @ self.lokr_w2_b
        return make_kron(w1, w2, self.scale)

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict: Dict[str, torch.Tensor] = {
            f"{self.lora_name}.alpha": self.alpha.detach().clone().to(dtype=torch.float32),
        }
        if self.decompose_both:
            state_dict[f"{self.lora_name}.lokr_w1_a"] = self.lokr_w1_a.detach().clone()
            state_dict[f"{self.lora_name}.lokr_w1_b"] = self.lokr_w1_b.detach().clone()
        else:
            state_dict[f"{self.lora_name}.lokr_w1"] = self.lokr_w1.detach().clone()
        if self.use_w2:
            state_dict[f"{self.lora_name}.lokr_w2"] = self.lokr_w2.detach().clone()
        else:
            state_dict[f"{self.lora_name}.lokr_w2_a"] = self.lokr_w2_a.detach().clone()
            state_dict[f"{self.lora_name}.lokr_w2_b"] = self.lokr_w2_b.detach().clone()
        return state_dict

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        diff_weight = self.get_diff_weight()

        # rank dropout
        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(diff_weight.size(0), device=diff_weight.device) > self.rank_dropout).to(diff_weight.dtype)
            drop = drop.view(-1, 1)
            diff_weight = diff_weight * drop
            scale = 1.0 / (1.0 - self.rank_dropout)
        else:
            scale = 1.0

        return org_forwarded + F.linear(x, diff_weight) * self.multiplier * scale


class LoKrInfModule(LoKrModule):
    """LoKr module for inference. Supports merge_to and get_weight."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference; pass factor from kwargs if present
        factor = kwargs.pop("factor", -1)
        decompose_both = kwargs.pop("decompose_both", False)
        decompose_w1_rank = kwargs.pop("decompose_w1_rank", None)
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            factor=factor,
            decompose_both=decompose_both,
            decompose_w1_rank=decompose_w1_rank,
        )

        self.org_module_ref = [org_module]
        self.enabled = True
        self.network: lora_module.LoRANetwork = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device, non_blocking=False):
        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(device, dtype=torch.float, non_blocking=non_blocking)

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        diff_weight = _materialize_lokr_weight_from_state_dict(sd, self.scale, device, non_blocking=non_blocking)

        # merge
        weight = weight + self.multiplier * diff_weight

        org_sd["weight"] = weight.to(org_device, dtype=dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        if self.decompose_both:
            w1 = (self.lokr_w1_a @ self.lokr_w1_b).to(torch.float)
        else:
            w1 = self.lokr_w1.to(torch.float)
        if self.use_w2:
            w2 = self.lokr_w2.to(torch.float)
        else:
            w2 = (self.lokr_w2_a @ self.lokr_w2_b).to(torch.float)

        weight = make_kron(w1, w2, self.scale) * multiplier
        return weight

    def default_forward(self, x):
        diff_weight = self.get_diff_weight()
        return self.org_forward(x) + F.linear(x, diff_weight) * self.multiplier

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class DoKrModule(LoKrModule):
    """DoRA magnitude scaling applied to a LoKr directional update."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        factor=-1,
        **kwargs,
    ):
        decompose_both = kwargs.pop("decompose_both", False)
        decompose_w1_rank = kwargs.pop("decompose_w1_rank", None)
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            factor=factor,
            decompose_both=decompose_both,
            decompose_w1_rank=decompose_w1_rank,
            **kwargs,
        )

        self.org_module_ref = [org_module]
        reference_param = self.lokr_w1_a if self.decompose_both else self.lokr_w1
        initial_norm = self._get_base_weight_norm().to(device=reference_param.device, dtype=reference_param.dtype)
        self.register_buffer("initial_norm", initial_norm.detach().clone())
        self.lora_magnitude_vector = lora_module.DoRAMagnitudeModule(torch.ones_like(initial_norm))

    def _get_base_module(self) -> torch.nn.Module:
        if hasattr(self, "org_module"):
            return self.org_module
        return self.org_module_ref[0]

    def _get_base_weight_norm(self) -> torch.Tensor:
        with torch.no_grad():
            base_weight = lora_module._get_effective_module_weight(self._get_base_module(), dtype=torch.float, detach=True)
            return torch.linalg.vector_norm(base_weight, dim=1)

    def _get_delta_weight(
        self,
        multiplier: Optional[float] = None,
        diff_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        multiplier = self.multiplier if multiplier is None else multiplier
        if diff_weight is None:
            diff_weight = self.get_diff_weight()
        return diff_weight.to(torch.float) * float(multiplier)

    def _get_weight_norm(
        self,
        multiplier: float,
        diff_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            base_weight = lora_module._get_effective_module_weight(self._get_base_module(), dtype=torch.float, detach=True)
            delta_weight = self._get_delta_weight(multiplier=multiplier, diff_weight=diff_weight).to(base_weight.device)
            return _get_dokr_weight_norm(base_weight, delta_weight)

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        magnitude = _magnitude_ratio_to_absolute(self.lora_magnitude_vector.weight, self.initial_norm)
        state_dict = super().export_state_dict()
        state_dict[f"{self.lora_name}.lora_magnitude_vector.weight"] = magnitude.detach().clone()
        state_dict[f"{self.lora_name}.initial_norm"] = self.initial_norm.detach().clone()
        return state_dict

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return org_forwarded

        diff_weight = self.get_diff_weight()
        effective_diff_weight = diff_weight
        scale = 1.0

        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(diff_weight.size(0), device=diff_weight.device) > self.rank_dropout).to(diff_weight.dtype)
            effective_diff_weight = diff_weight * drop.view(-1, 1)
            scale = 1.0 / (1.0 - self.rank_dropout)

        lora_output = F.linear(x, effective_diff_weight)
        effective_multiplier = self.multiplier * scale

        magnitude = _magnitude_ratio_to_absolute(self.lora_magnitude_vector.weight, self.initial_norm)
        weight_norm = self._get_weight_norm(effective_multiplier, diff_weight=effective_diff_weight)
        if weight_norm.device != magnitude.device:
            weight_norm = weight_norm.to(device=magnitude.device)
        if weight_norm.is_floating_point():
            eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
            weight_norm = weight_norm.clamp_min(eps)

        org_module = self.org_module_ref[0]
        base_without_bias = lora_module._remove_module_bias(org_forwarded, org_module.bias, False)
        mag_norm_scale = lora_module._reshape_dora_factor_for_output(
            magnitude / weight_norm,
            lora_output,
            False,
        )

        compose_dtype = torch.promote_types(torch.promote_types(base_without_bias.dtype, lora_output.dtype), mag_norm_scale.dtype)
        factor = mag_norm_scale.to(compose_dtype)
        delta = (factor - 1.0) * base_without_bias.to(compose_dtype) + factor * (
            lora_output.to(compose_dtype) * effective_multiplier
        )

        return org_forwarded + delta.to(org_forwarded.dtype)


class DoKrInfModule(DoKrModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        factor = kwargs.pop("factor", -1)
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, factor=factor, **kwargs)

        self.enabled = True
        self.network: lora_module.LoRANetwork = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device, non_blocking=False):
        org_module = self.org_module_ref[0]
        org_sd = org_module.state_dict()
        org_dtype = org_sd["weight"].dtype
        org_device = org_sd["weight"].device

        if device is None:
            device = org_device
        if dtype is None:
            dtype = org_dtype

        base_weight = lora_module._get_effective_module_weight(
            org_module,
            dtype=torch.float,
            device=device,
            detach=True,
            non_blocking=non_blocking,
        )
        delta_weight = _materialize_lokr_weight_from_state_dict(sd, self.scale, device, non_blocking=non_blocking)
        delta_weight = delta_weight * self.multiplier
        weight_norm = _get_dokr_weight_norm(base_weight, delta_weight)
        magnitude = sd["lora_magnitude_vector.weight"].to(device, dtype=weight_norm.dtype, non_blocking=non_blocking)
        merged_weight = lora_module._apply_dora_weight_merge(base_weight, delta_weight, magnitude, weight_norm)

        org_sd["weight"] = merged_weight.to(org_device, dtype=dtype)
        org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        multiplier = self.multiplier if multiplier is None else multiplier
        org_module = self.org_module_ref[0]
        base_weight = lora_module._get_effective_module_weight(org_module, dtype=torch.float)
        delta_weight = self._get_delta_weight(multiplier=multiplier).to(base_weight.device)
        weight_norm = self._get_weight_norm(multiplier).to(base_weight.device)
        magnitude = _magnitude_ratio_to_absolute(self.lora_magnitude_vector.weight, self.initial_norm).to(
            device=base_weight.device, dtype=weight_norm.dtype
        )
        merged_weight = lora_module._apply_dora_weight_merge(base_weight, delta_weight, magnitude, weight_norm)
        return merged_weight - base_weight

    def default_forward(self, x):
        return DoKrModule.forward(self, x)

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class DoKrOFTModule(DoKrModule):
    """DokR combined with Musubi's scaled OFT input/filter rotation."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        factor=-1,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            factor=factor,
            **kwargs,
        )
        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("DoKr-OFT Conv2d is not supported in this implementation")

        self.in_features = org_module.in_features
        requested_block_size = int(kwargs.get("oft_block_size", self.lora_dim))
        if requested_block_size <= 0:
            raise ValueError("oft_block_size must be positive")
        self.adjustment_info: Optional[tuple[int, int]] = None
        if self.in_features % requested_block_size != 0 or requested_block_size > self.in_features:
            old_block_size = requested_block_size
            requested_block_size = lora_module.OFTModule._adjust_oft_block_size(self.in_features, requested_block_size)
            self.adjustment_info = (old_block_size, requested_block_size)
            logger.warning(
                "Adjusted OFT block size for %s from %s to %s to divide input features %s",
                self.lora_name,
                old_block_size,
                requested_block_size,
                self.in_features,
            )
        self.oft_block_size = requested_block_size
        self.oft_block_share = lora_module._parse_bool_network_arg(kwargs.get("oft_block_share", kwargs.get("block_share", False)))
        self.oft_coft = lora_module._parse_bool_network_arg(kwargs.get("oft_coft", kwargs.get("coft", False)))
        self.coft_eps = float(kwargs.get("coft_eps", 6e-5))
        self.scaled_oft = True
        self.oft_dropout = float(kwargs.get("oft_dropout", kwargs.get("dropout_probability", 0.0)) or 0.0)
        self.rank = self.in_features // self.oft_block_size
        n_elements = self.oft_block_size * (self.oft_block_size - 1) // 2
        self.oft_R = lora_module.OFTRotationModule(
            r=self.rank if not self.oft_block_share else 1,
            n_elements=n_elements,
            block_size=self.oft_block_size,
            in_features=self.in_features,
            coft=self.oft_coft,
            coft_eps=self.coft_eps,
            block_share=self.oft_block_share,
            scaled_oft=True,
            use_cayley_neumann=True,
            num_cayley_neumann_terms=5,
            dropout_probability=self.oft_dropout,
        )
        torch.nn.init.zeros_(self.oft_R.weight)
        metadata_device = self.oft_R.weight.device
        self.register_buffer("oft_block_size_metadata", torch.tensor(float(self.oft_block_size), device=metadata_device))
        self.register_buffer(
            "oft_block_share_metadata",
            torch.tensor(1.0 if self.oft_block_share else 0.0, device=metadata_device),
        )
        self.register_buffer("oft_coft_metadata", torch.tensor(1.0 if self.oft_coft else 0.0, device=metadata_device))
        self.register_buffer("coft_eps_metadata", torch.tensor(float(self.coft_eps), device=metadata_device))

    def _rotated_input(self, x: torch.Tensor, multiplier: Optional[float] = None) -> torch.Tensor:
        multiplier = self.multiplier if multiplier is None else multiplier
        return self.oft_R(x, multiplier=multiplier)

    def _rotated_weight(
        self,
        multiplier: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        multiplier = self.multiplier if multiplier is None else multiplier
        base_weight = lora_module._get_effective_module_weight(self._get_base_module())
        compute_dtype = dtype or lora_module._get_dora_compute_dtype(base_weight)
        weight = base_weight.to(dtype=compute_dtype)
        rotation = self.oft_R.rotation_matrix(multiplier=multiplier).to(device=weight.device, dtype=compute_dtype)
        if self.oft_block_share:
            rotation = rotation.repeat(self.rank, 1, 1)
        rotation = rotation.transpose(-1, -2)
        reshaped_weight = weight.reshape(weight.shape[0], self.rank, self.oft_block_size)
        rotated_weight = torch.einsum("ork,rkc->orc", reshaped_weight, rotation)
        return rotated_weight.reshape_as(weight)

    def _rotated_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.org_forward(self._rotated_input(x))

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        magnitude = _magnitude_ratio_to_absolute(self.lora_magnitude_vector.weight, self.initial_norm)
        state_dict = LoKrModule.export_state_dict(self)
        state_dict[f"{self.lora_name}.lora_magnitude_vector.weight"] = magnitude.detach().clone()
        state_dict[f"{self.lora_name}.initial_norm"] = self.initial_norm.detach().clone()
        state_dict[f"{self.lora_name}.oft_R.weight"] = self.oft_R.weight.detach().clone()
        state_dict[f"{self.lora_name}.oft_block_size_metadata"] = self.oft_block_size_metadata.detach().clone()
        state_dict[f"{self.lora_name}.oft_block_share_metadata"] = self.oft_block_share_metadata.detach().clone()
        state_dict[f"{self.lora_name}.oft_coft_metadata"] = self.oft_coft_metadata.detach().clone()
        state_dict[f"{self.lora_name}.coft_eps_metadata"] = self.coft_eps_metadata.detach().clone()
        return state_dict

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return org_forwarded

        rotated_x = self._rotated_input(x)
        rotated_output = self.org_forward(rotated_x)
        org_module = self.org_module_ref[0]
        base_without_bias = lora_module._remove_module_bias(rotated_output, org_module.bias, False)

        diff_weight = self.get_diff_weight()
        effective_diff_weight = diff_weight
        scale = 1.0

        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(diff_weight.size(0), device=diff_weight.device) > self.rank_dropout).to(diff_weight.dtype)
            effective_diff_weight = diff_weight * drop.view(-1, 1)
            scale = 1.0 / (1.0 - self.rank_dropout)

        lora_output = F.linear(rotated_x, effective_diff_weight)
        effective_multiplier = self.multiplier * scale

        magnitude = _magnitude_ratio_to_absolute(self.lora_magnitude_vector.weight, self.initial_norm)
        with torch.no_grad():
            base_weight = lora_module._get_effective_module_weight(org_module, dtype=torch.float, detach=True)
            delta_weight = self._get_delta_weight(multiplier=effective_multiplier, diff_weight=effective_diff_weight).to(
                base_weight.device
            )
            weight_norm = _get_dokr_weight_norm(base_weight, delta_weight)
        if weight_norm.device != magnitude.device:
            weight_norm = weight_norm.to(device=magnitude.device)
        if weight_norm.is_floating_point():
            eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
            weight_norm = weight_norm.clamp_min(eps)

        mag_norm_scale = lora_module._reshape_dora_factor_for_output(
            magnitude / weight_norm,
            lora_output,
            False,
        )
        compose_dtype = torch.promote_types(torch.promote_types(base_without_bias.dtype, lora_output.dtype), mag_norm_scale.dtype)
        factor = mag_norm_scale.to(compose_dtype)
        result = factor * (base_without_bias.to(compose_dtype) + lora_output.to(compose_dtype) * effective_multiplier)
        if org_module.bias is not None:
            bias = org_module.bias.to(device=result.device, dtype=result.dtype)
            result = result + bias.view(*([1] * (result.dim() - 1)), -1)
        return result.to(org_forwarded.dtype)

    def merge_to(self, sd, dtype, device, non_blocking=False):
        org_module = self.org_module_ref[0]
        org_sd = org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device

        if device is None:
            device = org_device
        if dtype is None:
            dtype = org_dtype

        base_weight = lora_module._get_effective_module_weight(
            org_module,
            dtype=torch.float,
            device=device,
            detach=True,
            non_blocking=non_blocking,
        )
        delta_weight = _materialize_lokr_weight_from_state_dict(sd, self.scale, device, non_blocking=non_blocking)
        delta_weight = delta_weight * self.multiplier
        merged_unrotated = base_weight + delta_weight
        rotated_weight = _rotate_weight_with_oft_state_dict(
            merged_unrotated,
            sd,
            multiplier=self.multiplier,
            non_blocking=non_blocking,
        )
        weight_norm = torch.linalg.vector_norm(merged_unrotated, dim=1)
        magnitude = sd["lora_magnitude_vector.weight"].to(device, dtype=weight_norm.dtype, non_blocking=non_blocking)
        merged_weight = _apply_dokr_oft_weight_merge(rotated_weight, magnitude, weight_norm)

        org_sd["weight"] = merged_weight.to(org_device, dtype=dtype)
        org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        multiplier = self.multiplier if multiplier is None else multiplier
        org_module = self.org_module_ref[0]
        base_weight = lora_module._get_effective_module_weight(org_module, dtype=torch.float)
        delta_weight = self._get_delta_weight(multiplier=multiplier).to(base_weight.device)
        merged_unrotated = base_weight + delta_weight
        rotation = self.oft_R.rotation_matrix(multiplier=multiplier).to(device=base_weight.device, dtype=base_weight.dtype)
        if self.oft_block_share:
            rotation = rotation.repeat(self.rank, 1, 1)
        rotation = rotation.transpose(-1, -2)
        reshaped_weight = merged_unrotated.reshape(merged_unrotated.shape[0], self.rank, self.oft_block_size)
        rotated_weight = torch.einsum("ork,rkc->orc", reshaped_weight, rotation).reshape_as(merged_unrotated)
        weight_norm = torch.linalg.vector_norm(merged_unrotated, dim=1)
        magnitude = _magnitude_ratio_to_absolute(self.lora_magnitude_vector.weight, self.initial_norm).to(
            device=base_weight.device, dtype=weight_norm.dtype
        )
        merged_weight = _apply_dokr_oft_weight_merge(rotated_weight, magnitude, weight_norm)
        return merged_weight - base_weight


class DoKrOFTInfModule(DoKrOFTModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        factor = kwargs.pop("factor", -1)
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, factor=factor, **kwargs)
        self.enabled = True
        self.network: lora_module.LoRANetwork = None

    def set_network(self, network):
        self.network = network

    def default_forward(self, x):
        return DoKrOFTModule.forward(self, x)

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class LoKrNetwork(lora_module.LoRANetwork):
    def prepare_weights_state_dict_for_load(self, state_dict):
        state_dict = super().prepare_weights_state_dict_for_load(state_dict)
        if any(key.endswith(".lora_magnitude_vector.weight") for key in state_dict.keys()):
            state_dict = copy.copy(state_dict)
            _convert_absolute_magnitudes_to_ratios(state_dict)

        return state_dict


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """Create a LoKr network with auto-detected architecture."""
    target_replace_modules, default_excludes = detect_arch_config(unet)

    # merge user exclude_patterns with defaults
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
    exclude_patterns.extend(default_excludes)
    kwargs["exclude_patterns"] = exclude_patterns

    # extract factor from kwargs
    factor = kwargs.pop("factor", -1)
    factor = int(factor)
    use_dora = lora_module._parse_bool_network_arg(kwargs.get("use_dora", False))
    use_dora_oft = lora_module._parse_bool_network_arg(kwargs.get("use_dora_oft", False))
    decompose_both = lora_module._parse_bool_network_arg(kwargs.get("decompose_both", False))
    if use_dora and use_dora_oft:
        raise ValueError("use_dora and use_dora_oft cannot both be enabled")

    module_class = LoKrModule
    if use_dora_oft:
        if lora_module._parse_bool_network_arg(kwargs.get("adaptive_rank", False)):
            raise ValueError("adaptive_rank is not supported with use_dora_oft")
        module_class = DoKrOFTModule
    elif use_dora:
        module_class = DoKrModule

    module_kwargs = {"factor": factor, "decompose_both": decompose_both}
    if use_dora_oft:
        module_kwargs["scaled_oft"] = True
        for key in ("oft_block_size", "oft_coft", "coft_eps", "oft_block_share", "oft_dropout"):
            if kwargs.get(key, None) is not None:
                module_kwargs[key] = kwargs.get(key)

    forwarded_kwargs = dict(kwargs)
    forwarded_kwargs.pop("use_dora", None)
    forwarded_kwargs.pop("use_dora_oft", None)
    forwarded_kwargs.pop("decompose_both", None)

    network = lora_module.create_network(
        target_replace_modules,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        module_class=module_class,
        module_kwargs=module_kwargs,
        network_class=LoKrNetwork,
        **forwarded_kwargs,
    )
    return network


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora_module.LoRANetwork:
    """Create LoKr network from saved weights (internal)."""
    modules_dim = {}
    modules_alpha = {}
    has_dokr_weights = lora_module._parse_bool_network_arg(kwargs.get("use_dora", False))
    has_dokr_oft_weights = lora_module._parse_bool_network_arg(kwargs.get("use_dora_oft", False))
    decompose_both = lora_module._parse_bool_network_arg(kwargs.get("decompose_both", False))
    per_module_kwargs: Dict[str, Dict[str, object]] = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_magnitude_vector" in key:
            has_dokr_weights = True
        elif "oft_R.weight" in key:
            has_dokr_oft_weights = True
        elif "lokr_w1_a" in key:
            decompose_both = True
            module_kwargs_for_name = per_module_kwargs.setdefault(lora_name, {})
            module_kwargs_for_name["decompose_w1_rank"] = int(value.shape[1])
        elif "lokr_w2_a" in key:
            # low-rank mode: dim = w2_a.shape[1]
            dim = value.shape[1]
            modules_dim[lora_name] = dim
        elif "lokr_w2" in key and "lokr_w2_a" not in key and "lokr_w2_b" not in key:
            # full matrix mode: set dim large enough to trigger full-matrix path
            if lora_name not in modules_dim:
                modules_dim[lora_name] = max(value.shape)

    if has_dokr_oft_weights:
        for lora_name in {key.split(".")[0] for key in weights_sd.keys() if "." in key and key.startswith("lora_")}:
            module_state = {key.split(".", 1)[1]: tensor for key, tensor in weights_sd.items() if key.startswith(lora_name + ".")}
            if "oft_R.weight" not in module_state:
                continue
            in_features = _materialized_lokr_w1_input_dim(module_state) * _lokr_w2_input_dim(module_state)
            block_size, block_share, coft, coft_eps = _resolve_oft_config_from_state_dict(module_state, in_features)
            module_kwargs_for_name = per_module_kwargs.setdefault(lora_name, {})
            module_kwargs_for_name["oft_block_size"] = block_size
            module_kwargs_for_name["oft_block_share"] = block_share
            module_kwargs_for_name["oft_coft"] = coft
            module_kwargs_for_name["coft_eps"] = coft_eps

    # extract factor for LoKr (user must specify via --network_args factor=N if different from default)
    factor = int(kwargs.pop("factor", -1))

    if has_dokr_oft_weights:
        module_class = DoKrOFTInfModule if for_inference else DoKrOFTModule
    elif has_dokr_weights:
        module_class = DoKrInfModule if for_inference else DoKrModule
    else:
        module_class = LoKrInfModule if for_inference else LoKrModule
    module_kwargs = {"factor": factor, "decompose_both": decompose_both}
    if per_module_kwargs:
        module_kwargs["per_module_kwargs"] = per_module_kwargs
    if has_dokr_oft_weights:
        module_kwargs["scaled_oft"] = True

    network = LoKrNetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs=module_kwargs,
    )
    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora_module.LoRANetwork:
    """Create LoKr network from saved weights with auto-detected architecture."""
    target_replace_modules, _ = detect_arch_config(unet)
    return create_network_from_weights(target_replace_modules, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs)


def merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Merge LoKr weights directly into a model weight tensor.

    No Module/Network creation needed. Consumed keys are removed from lora_weight_keys.
    Returns model_weight unchanged if no matching LoKr keys found.
    """
    w1_key = lora_name + ".lokr_w1"
    w1a_key = lora_name + ".lokr_w1_a"
    w1b_key = lora_name + ".lokr_w1_b"
    w2_key = lora_name + ".lokr_w2"
    w2a_key = lora_name + ".lokr_w2_a"
    w2b_key = lora_name + ".lokr_w2_b"
    alpha_key = lora_name + ".alpha"
    magnitude_key = lora_name + ".lora_magnitude_vector.weight"

    if w1_key not in lora_weight_keys and w1a_key not in lora_weight_keys:
        return model_weight

    if w1a_key in lora_weight_keys:
        w1a = lora_sd[w1a_key].to(calc_device)
        w1b = lora_sd[w1b_key].to(calc_device)
        w1 = None
        w1_consumed_keys = [w1a_key, w1b_key]
    else:
        w1 = lora_sd[w1_key].to(calc_device)
        w1a = None
        w1b = None
        w1_consumed_keys = [w1_key]

    # determine low-rank vs full matrix mode
    if w2a_key in lora_weight_keys:
        # low-rank: w2 = w2_a @ w2_b
        w2a = lora_sd[w2a_key].to(calc_device)
        w2b = lora_sd[w2b_key].to(calc_device)
        dim = w2a.shape[1]
        consumed_keys = w1_consumed_keys + [w2a_key, w2b_key, alpha_key]
    elif w2_key in lora_weight_keys:
        # full matrix mode
        w2a = None
        w2b = None
        dim = None  # will use scale=1.0
        consumed_keys = w1_consumed_keys + [w2_key, alpha_key]
    else:
        return model_weight

    alpha = lora_sd.get(alpha_key, None)
    if alpha is not None and isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    # compute scale
    if w2a is not None:
        # low-rank mode
        if alpha is None:
            alpha = dim
        scale = alpha / dim
    else:
        # full matrix mode: scale = 1.0
        scale = 1.0

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:  # fp8
        model_weight = model_weight.to(torch.float16)
        if w1a is not None:
            w1a, w1b = w1a.to(torch.float16), w1b.to(torch.float16)
        else:
            w1 = w1.to(torch.float16)
        if w2a is not None:
            w2a, w2b = w2a.to(torch.float16), w2b.to(torch.float16)

    if w1a is not None:
        w1 = w1a @ w1b

    # compute w2
    if w2a is not None:
        w2 = w2a @ w2b
    else:
        w2 = lora_sd[w2_key].to(calc_device)
        if original_dtype.itemsize == 1:
            w2 = w2.to(torch.float16)

    # ΔW = kron(w1, w2) * scale
    diff_weight = make_kron(w1, w2, scale)
    if lora_name + ".oft_R.weight" in lora_weight_keys:
        module_state = {
            key[len(lora_name) + 1 :]: lora_sd[key] for key in list(lora_weight_keys) if key.startswith(lora_name + ".")
        }
        if magnitude_key not in lora_weight_keys:
            raise ValueError(f"DoKr-OFT weights for {lora_name} are missing lora_magnitude_vector.weight")
        delta_weight = multiplier * diff_weight
        if original_dtype.itemsize == 1:
            delta_weight = delta_weight.to(torch.float16)
        merged_unrotated = model_weight + delta_weight
        rotated_weight = _rotate_weight_with_oft_state_dict(
            merged_unrotated,
            module_state,
            multiplier=multiplier,
        )
        magnitude = lora_sd[magnitude_key].to(calc_device)
        if original_dtype.itemsize == 1:
            magnitude = magnitude.to(torch.float16)
        weight_norm = torch.linalg.vector_norm(merged_unrotated, dim=1)
        model_weight = _apply_dokr_oft_weight_merge(rotated_weight, magnitude, weight_norm)
        consumed_keys.extend(
            [
                magnitude_key,
                lora_name + ".oft_R.weight",
                lora_name + ".oft_block_size_metadata",
                lora_name + ".oft_block_share_metadata",
                lora_name + ".oft_coft_metadata",
                lora_name + ".coft_eps_metadata",
            ]
        )
    elif magnitude_key in lora_weight_keys:
        magnitude = lora_sd[magnitude_key].to(calc_device)
        delta_weight = multiplier * diff_weight
        if original_dtype.itemsize == 1:
            magnitude = magnitude.to(torch.float16)
            delta_weight = delta_weight.to(torch.float16)
        weight_norm = _get_dokr_weight_norm(model_weight, delta_weight)
        model_weight = lora_module._apply_dora_weight_merge(model_weight, delta_weight, magnitude, weight_norm)
        consumed_keys.append(magnitude_key)
    else:
        model_weight = model_weight + multiplier * diff_weight

    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(original_dtype)

    # remove consumed keys
    for key in consumed_keys:
        lora_weight_keys.discard(key)

    return model_weight
