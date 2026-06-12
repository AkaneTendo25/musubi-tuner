import logging
import math
from typing import Dict, List, Optional

import torch

from musubi_tuner.networks.lora_shared import (
    _get_dora_compute_dtype,
    _get_effective_module_weight,
    _parse_bool_network_arg,
    _parse_optional_float_network_arg,
    _remove_module_bias,
    _reshape_dora_factor_for_output,
    _reshape_dora_factor_for_weight,
)

logger = logging.getLogger(__name__)


class MultiplicativeDropoutLayer(torch.nn.Module):
    """Drop whole OFT rotation blocks by replacing them with identity blocks."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p or 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("OFT multiplicative dropout expects square rotation blocks")
        if x.shape[0] == 1:
            return x

        keep_prob = 1.0 - self.p
        mask = torch.empty(x.shape[0], 1, 1, device=x.device, dtype=x.dtype).bernoulli_(p=keep_prob)
        eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype).expand_as(x)
        return mask * x + (1.0 - mask) * eye


class OFTRotationModule(torch.nn.Module):
    def __init__(
        self,
        r: int,
        n_elements: int,
        block_size: int,
        in_features: int,
        coft: bool = False,
        coft_eps: float = 6e-5,
        block_share: bool = False,
        scaled_oft: bool = True,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
        dropout_probability: float = 0.0,
    ):
        super().__init__()
        self.r = int(r)
        self.n_elements = int(n_elements)
        self.block_size = int(block_size)
        self.in_features = int(in_features)
        self.coft = bool(coft)
        self.coft_eps = float(coft_eps)
        self.block_share = bool(block_share)
        self.use_scaled_oft = bool(scaled_oft)
        self.use_cayley_neumann = bool(use_cayley_neumann)
        self.num_cayley_neumann_terms = int(num_cayley_neumann_terms)
        self.weight = torch.nn.Parameter(torch.empty(self.r, self.n_elements))
        self.weight._is_oft = True
        rows, cols = torch.triu_indices(self.block_size, self.block_size, 1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)

    def _pytorch_skew_symmetric(self, vec: torch.Tensor, block_size: int) -> torch.Tensor:
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)
        batch_idx = torch.arange(batch_size, device=vec.device)[:, None]
        matrix = matrix.index_put((batch_idx, self.rows, self.cols), vec)
        return matrix - matrix.transpose(-2, -1)

    def _pytorch_skew_symmetric_inv(self, matrix: torch.Tensor, block_size: int) -> torch.Tensor:
        return matrix[:, self.rows, self.cols]

    def _cayley_batch(
        self,
        q: torch.Tensor,
        block_size: int,
        use_cayley_neumann: bool = True,
        num_neumann_terms: int = 5,
    ) -> torch.Tensor:
        b, _ = q.shape
        previous_dtype = q.dtype
        q_skew = self._pytorch_skew_symmetric(q, block_size)

        if use_cayley_neumann:
            r = torch.eye(block_size, device=q.device, dtype=q.dtype).repeat(b, 1, 1)
            if num_neumann_terms > 1:
                r.add_(q_skew, alpha=2.0)
                if num_neumann_terms > 2:
                    q_squared = torch.bmm(q_skew, q_skew)
                    r.add_(q_squared, alpha=2.0)
                    q_power = q_squared
                    for _ in range(3, num_neumann_terms - 1):
                        q_power = torch.bmm(q_power, q_skew)
                        r.add_(q_power, alpha=2.0)
                    q_power = torch.bmm(q_power, q_skew)
                    r.add_(q_power)
        else:
            identity = torch.eye(q_skew.shape[-1], device=q_skew.device, dtype=q_skew.dtype).unsqueeze(0).expand_as(q_skew)
            r = torch.linalg.solve(identity + q_skew, identity - q_skew, left=False)

        return r.to(previous_dtype)

    def _project_batch(self, q: torch.Tensor, coft_eps: float = 1e-4) -> torch.Tensor:
        oft_r = self._pytorch_skew_symmetric(q, self.block_size)
        coft_eps = float(coft_eps) / torch.sqrt(torch.tensor(oft_r.shape[0], device=oft_r.device, dtype=oft_r.dtype))
        origin = torch.zeros((oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype).unsqueeze(0).expand_as(oft_r)
        diff = oft_r - origin
        norm_diff = torch.norm(diff, dim=(1, 2), keepdim=True)
        mask = norm_diff <= coft_eps
        out = torch.where(mask, oft_r, origin + coft_eps * (diff / norm_diff.clamp_min(1e-12)))
        return self._pytorch_skew_symmetric_inv(out, self.block_size)

    def rotation_matrix(self, weight: Optional[torch.Tensor] = None, multiplier: float = 1.0) -> torch.Tensor:
        if self.coft and weight is None:
            with torch.no_grad():
                self.weight.copy_(self._project_batch(self.weight, coft_eps=self.coft_eps))

        effective_weight = self.weight if weight is None else weight
        effective_weight = effective_weight * float(multiplier)
        scaling_factor = 2 * math.sqrt(self.block_size - 1) if self.use_scaled_oft else 1.0
        effective_weight = effective_weight / scaling_factor
        rotation = self._cayley_batch(
            effective_weight,
            self.block_size,
            self.use_cayley_neumann,
            self.num_cayley_neumann_terms,
        )
        return self.dropout(rotation)

    def forward(self, x: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
        required_dtype = x.dtype
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        orig_shape = x.shape
        rotation = self.rotation_matrix(multiplier=multiplier)
        rank = self.in_features // self.block_size if self.block_share else self.r
        batch_dims = x.shape[:-1]
        x_reshaped = x.reshape(*batch_dims, rank, self.block_size)

        if self.block_share:
            rotation = rotation.repeat(rank, 1, 1)
        x_rotated = torch.einsum("...rk,rkc->...rc", x_reshaped, rotation)
        return x_rotated.reshape(*orig_shape).to(required_dtype)


class OFTModule(torch.nn.Module):
    """
    Orthogonal Finetuning module using the same monkey-patch interface as LoRAModule.
    The LoRA dim field is reused as the default OFT block size for CLI compatibility.
    """

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
        split_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        if split_dims is not None:
            raise ValueError("DoRA-OFT does not support split_dims modules")

        self.lora_name = lora_name
        self.lora_dim = int(lora_dim)
        self.oft_block_size = int(kwargs.get("oft_block_size", self.lora_dim))
        if self.oft_block_size <= 0:
            raise ValueError("oft_block_size must be positive")
        self.scale = 1.0
        self.multiplier = multiplier
        self.org_module = org_module
        self.org_module_ref = [org_module]
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.split_dims = None
        self.module_path = kwargs.get("module_path", None)
        self.oft_coft = _parse_bool_network_arg(kwargs.get("oft_coft", kwargs.get("coft", False)))
        self.coft_eps = float(kwargs.get("coft_eps", 6e-5))
        self.oft_block_share = _parse_bool_network_arg(kwargs.get("oft_block_share", kwargs.get("block_share", False)))
        self.scaled_oft = _parse_bool_network_arg(kwargs.get("scaled_oft", True))
        self.oft_dropout = float(kwargs.get("oft_dropout", kwargs.get("dropout_probability", 0.0)) or 0.0)
        self.enabled = True
        self.dropout_enabled = True

        self.is_conv2d = org_module.__class__.__name__ == "Conv2d"
        if self.is_conv2d:
            if org_module.dilation[0] > 1 or org_module.dilation[1] > 1:
                raise ValueError("DoRA-OFT does not support Conv2d dilation > 1")
            self.in_features = org_module.in_channels * org_module.kernel_size[0] * org_module.kernel_size[1]
        elif org_module.__class__.__name__ == "Linear":
            self.in_features = org_module.in_features
        else:
            raise NotImplementedError("DoRA-OFT only supports Linear and Conv2d modules")

        self.adjustment_info: Optional[tuple[int, int]] = None
        block_size = self.oft_block_size
        if self.in_features % block_size != 0 or block_size > self.in_features:
            old_block_size = block_size
            block_size = self._adjust_oft_block_size(self.in_features, block_size)
            self.adjustment_info = (old_block_size, block_size)
            logger.warning(
                "Adjusted OFT block size for %s from %s to %s to divide input features %s",
                self.lora_name,
                old_block_size,
                block_size,
                self.in_features,
            )
        self.oft_block_size = block_size
        self.rank = self.in_features // self.oft_block_size
        self.lora_dim = self.oft_block_size
        self.register_buffer("alpha", torch.tensor(float(self.oft_block_size)))

        n_elements = self.oft_block_size * (self.oft_block_size - 1) // 2
        self.oft_R = OFTRotationModule(
            r=self.rank if not self.oft_block_share else 1,
            n_elements=n_elements,
            block_size=self.oft_block_size,
            in_features=self.in_features,
            coft=self.oft_coft,
            coft_eps=self.coft_eps,
            block_share=self.oft_block_share,
            scaled_oft=self.scaled_oft,
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
        self.register_buffer("scaled_oft_metadata", torch.tensor(1.0 if self.scaled_oft else 0.0, device=metadata_device))

    @staticmethod
    def _adjust_oft_block_size(in_features: int, requested: int) -> int:
        if requested >= in_features:
            return in_features

        higher = requested
        while higher <= in_features and in_features % higher != 0:
            higher += 1

        lower = requested
        while lower > 1 and in_features % lower != 0:
            lower -= 1

        if higher > in_features:
            return lower
        return lower if (requested - lower) <= (higher - requested) else higher

    def _get_base_module(self) -> torch.nn.Module:
        if hasattr(self, "org_module"):
            return self.org_module
        return self.org_module_ref[0]

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _module_forward_with_rotated_weight(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        org_module = self._get_base_module()
        if self.is_conv2d:
            return torch.nn.functional.conv2d(
                x,
                weight,
                org_module.bias,
                org_module.stride,
                org_module.padding,
                org_module.dilation,
                org_module.groups,
            )
        return torch.nn.functional.linear(x, weight, org_module.bias)

    def _rotated_weight(self, multiplier: Optional[float] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        multiplier = self.multiplier if multiplier is None else multiplier
        org_module = self._get_base_module()
        base_weight = _get_effective_module_weight(org_module)
        compute_dtype = dtype or _get_dora_compute_dtype(base_weight)
        weight = base_weight.to(dtype=compute_dtype)
        rotation = self.oft_R.rotation_matrix(multiplier=multiplier).to(device=weight.device, dtype=compute_dtype)
        if self.oft_block_share:
            rotation = rotation.repeat(self.rank, 1, 1)
        if not self.is_conv2d:
            rotation = rotation.transpose(-1, -2)
        weight_reshaped = weight.reshape(weight.shape[0], self.rank, self.oft_block_size)
        rotated_weight = torch.einsum("ork,rkc->orc", weight_reshaped, rotation)
        return rotated_weight.reshape_as(weight)

    def _oft_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_conv2d:
            return self._module_forward_with_rotated_weight(x, self._rotated_weight(dtype=x.dtype))

        rotated_x = self.oft_R(x, multiplier=self.multiplier)
        return self.org_forward(rotated_x)

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training and self.dropout_enabled:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return org_forwarded

        return self._oft_forward(x)

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict: Dict[str, torch.Tensor] = {}
        state_dict[f"{self.lora_name}.oft_R.weight"] = self.oft_R.weight.detach().clone()
        if self.scaled_oft:
            state_dict[f"{self.lora_name}.oft_R.scaled_oft"] = torch.tensor(1.0, device=self.oft_R.weight.device)
        state_dict[f"{self.lora_name}.alpha"] = torch.tensor(float(self.oft_block_size), device=self.oft_R.weight.device)
        state_dict[f"{self.lora_name}.oft_block_size_metadata"] = self.oft_block_size_metadata.detach().clone()
        state_dict[f"{self.lora_name}.oft_block_share_metadata"] = self.oft_block_share_metadata.detach().clone()
        state_dict[f"{self.lora_name}.oft_coft_metadata"] = self.oft_coft_metadata.detach().clone()
        state_dict[f"{self.lora_name}.coft_eps_metadata"] = self.coft_eps_metadata.detach().clone()
        state_dict[f"{self.lora_name}.scaled_oft_metadata"] = self.scaled_oft_metadata.detach().clone()
        return state_dict

    def get_weight(self, multiplier=None):
        base_weight = _get_effective_module_weight(self._get_base_module(), dtype=torch.float)
        rotated_weight = self._rotated_weight(multiplier=multiplier, dtype=torch.float).to(base_weight.device)
        return rotated_weight - base_weight

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

        original_weight = self.oft_R.weight.detach().clone()
        loaded_weight = sd["oft_R.weight"].to(device=original_weight.device, dtype=original_weight.dtype, non_blocking=non_blocking)
        try:
            with torch.no_grad():
                self.oft_R.weight.copy_(loaded_weight)
            merged_weight = self._rotated_weight(multiplier=self.multiplier, dtype=torch.float).to(device=device)
        finally:
            with torch.no_grad():
                self.oft_R.weight.copy_(original_weight)

        org_sd["weight"] = merged_weight.to(org_device, dtype=dtype)
        org_module.load_state_dict(org_sd)

    def default_forward(self, x):
        return self._oft_forward(x)

    def set_network(self, network):
        self.network = network


class DoRAOFTModule(OFTModule):
    """
    DoRA applied to scaled OFT. OFT preserves the original row/filter norm, so
    DoRA only needs a learnable magnitude over the norm-preserving rotated base.
    """

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
        split_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        kwargs["scaled_oft"] = True
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            split_dims=split_dims,
            **kwargs,
        )

        initial_norm = self._get_initial_norm(org_module)
        initial_norm = initial_norm.to(device=self.oft_R.weight.device, dtype=self.oft_R.weight.dtype)
        self.freeze_dora_scale = _parse_bool_network_arg(kwargs.get("freeze_dora_scale", False))
        self.dora_scale_min_ratio = _parse_optional_float_network_arg(kwargs.get("dora_scale_min_ratio"), None)
        self.dora_scale_max_ratio = _parse_optional_float_network_arg(kwargs.get("dora_scale_max_ratio"), None)
        if (
            self.dora_scale_min_ratio is not None
            and self.dora_scale_max_ratio is not None
            and self.dora_scale_min_ratio > self.dora_scale_max_ratio
        ):
            raise ValueError("dora_scale_min_ratio must be <= dora_scale_max_ratio")
        self.register_buffer("initial_norm", initial_norm.detach().clone())
        self.dora_scale = torch.nn.Parameter(torch.ones_like(initial_norm), requires_grad=not self.freeze_dora_scale)
        self.dora_scale._is_dora_scale = True
        self.scaled_oft = True

    def _get_initial_norm(self, org_module: torch.nn.Module) -> torch.Tensor:
        with torch.no_grad():
            weight = _get_effective_module_weight(org_module, dtype=torch.float, detach=True)
            if self.is_conv2d:
                return torch.norm(weight.reshape(weight.shape[0], -1), dim=1).view(weight.shape[0], 1, 1, 1)
            return torch.norm(weight, dim=1, keepdim=True)

    def _dora_factor(self, output: torch.Tensor) -> torch.Tensor:
        factor = self.dora_scale.to(dtype=torch.promote_types(self.dora_scale.dtype, output.dtype))
        return _reshape_dora_factor_for_output(factor, output, self.is_conv2d)

    def clamp_dora_scale(self) -> int:
        if self.dora_scale_min_ratio is None and self.dora_scale_max_ratio is None:
            return 0
        with torch.no_grad():
            min_scale = self.dora_scale_min_ratio
            max_scale = self.dora_scale_max_ratio
            original = self.dora_scale.detach().clone()
            self.dora_scale.clamp_(min=min_scale, max=max_scale)
            return int(torch.count_nonzero(self.dora_scale != original).item())

    def _export_dora_scale(self) -> torch.Tensor:
        initial_norm = self.initial_norm.to(device=self.dora_scale.device, dtype=self.dora_scale.dtype)
        return (self.dora_scale * initial_norm).detach().clone()

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training and self.dropout_enabled:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return org_forwarded

        oft_output = self._oft_forward(x)
        org_module = self.org_module_ref[0]
        oft_without_bias = _remove_module_bias(oft_output, org_module.bias, self.is_conv2d)
        factor = self._dora_factor(oft_without_bias)
        compose_dtype = torch.promote_types(oft_without_bias.dtype, factor.dtype)
        result = oft_without_bias.to(compose_dtype) * factor.to(compose_dtype)
        if org_module.bias is not None:
            bias = org_module.bias.to(device=result.device, dtype=result.dtype)
            if self.is_conv2d:
                result = result + bias.view(1, -1, *([1] * (result.dim() - 2)))
            else:
                result = result + bias.view(*([1] * (result.dim() - 1)), -1)
        return result.to(org_forwarded.dtype)

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = super().export_state_dict()
        state_dict[f"{self.lora_name}.dora_scale"] = self._export_dora_scale()
        state_dict[f"{self.lora_name}.initial_norm"] = self.initial_norm.detach().clone()
        return state_dict

    def get_weight(self, multiplier=None):
        base_weight = _get_effective_module_weight(self._get_base_module(), dtype=torch.float)
        rotated_weight = self._rotated_weight(multiplier=multiplier, dtype=torch.float).to(base_weight.device)
        scale = self.dora_scale.to(device=rotated_weight.device, dtype=rotated_weight.dtype)
        merged_weight = _reshape_dora_factor_for_weight(scale, rotated_weight) * rotated_weight
        return merged_weight - base_weight

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

        original_oft_weight = self.oft_R.weight.detach().clone()
        original_dora_scale = self.dora_scale.detach().clone()
        original_initial_norm = self.initial_norm.detach().clone()
        loaded_oft_weight = sd["oft_R.weight"].to(
            device=original_oft_weight.device,
            dtype=original_oft_weight.dtype,
            non_blocking=non_blocking,
        )
        loaded_dora_scale = sd["dora_scale"].to(
            device=original_dora_scale.device,
            dtype=original_dora_scale.dtype,
            non_blocking=non_blocking,
        )
        loaded_initial_norm = sd.get("initial_norm")
        try:
            with torch.no_grad():
                self.oft_R.weight.copy_(loaded_oft_weight)
                if loaded_initial_norm is not None:
                    self.initial_norm.copy_(
                        loaded_initial_norm.to(
                            device=original_initial_norm.device,
                            dtype=original_initial_norm.dtype,
                            non_blocking=non_blocking,
                        )
                    )
                initial_norm = self.initial_norm.to(device=original_dora_scale.device, dtype=original_dora_scale.dtype)
                eps = 1e-12 if initial_norm.dtype in (torch.float32, torch.float64) else 1e-6
                self.dora_scale.copy_(loaded_dora_scale / initial_norm.clamp_min(eps))
            base_weight = _get_effective_module_weight(
                org_module,
                dtype=torch.float,
                device=device,
                detach=True,
                non_blocking=non_blocking,
            )
            merged_delta = self.get_weight(multiplier=self.multiplier).to(device=device, dtype=torch.float)
            merged_weight = base_weight + merged_delta
        finally:
            with torch.no_grad():
                self.oft_R.weight.copy_(original_oft_weight)
                self.dora_scale.copy_(original_dora_scale)
                self.initial_norm.copy_(original_initial_norm)

        org_sd["weight"] = merged_weight.to(org_device, dtype=dtype)
        org_module.load_state_dict(org_sd)


class DoRAOFTInfModule(DoRAOFTModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, **kwargs)
        self.enabled = True
        self.network = None

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)

    def default_forward(self, x):
        return DoRAOFTModule.forward(self, x)
