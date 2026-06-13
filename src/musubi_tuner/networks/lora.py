# LoRA network module: currently conv2d is not fully supported
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import ast
import copy
import contextlib
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Type, Union
from transformers import CLIPTextModel
import torch
import torch.nn as nn

import logging

from musubi_tuner.networks.lora_adaptive_rank import (
    AdaptiveRankLoRAModuleMixin,
    AdaptiveRankLoRANetworkMixin,
    parse_adaptive_rank_network_kwargs,
)
from musubi_tuner.networks.lora_oft import DoRAOFTInfModule, DoRAOFTModule, OFTModule, OFTRotationModule
from musubi_tuner.networks.lora_shared import (
    _apply_dora_weight_merge,
    _convert_absolute_dora_oft_scales_to_ratios,
    _get_dora_weight_norm,
    _get_effective_module_weight,
    _get_lora_weight_from_tensors,
    _get_split_lora_weight,
    _metadata_tensor_to_bool,
    _metadata_tensor_to_float,
    _metadata_tensor_to_int,
    _parse_bool_network_arg,
    _parse_optional_float_network_arg,
    _remove_module_bias,
    _reshape_dora_factor_for_output,
    _solve_oft_block_size,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HUNYUAN_TARGET_REPLACE_MODULES = ["MMDoubleStreamBlock", "MMSingleStreamBlock"]


class DoRAMagnitudeModule(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.weight._is_dora_scale = True


class LoRAModule(AdaptiveRankLoRAModuleMixin, torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
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
        """
        if alpha == 0 or None, alpha is rank (no scaling).

        split_dims is used to mimic the split qkv of multi-head attention.
        """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ in ("Conv2d", "Conv3d"):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self._init_adaptive_rank_module_state(kwargs)
        self.split_dims = split_dims

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
                self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            elif org_module.__class__.__name__ == "Conv3d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = torch.nn.Conv3d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
                self.lora_up = torch.nn.Conv3d(self.lora_dim, out_dim, (1, 1, 1), (1, 1, 1), bias=False)
            else:
                self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
                self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)

            # LoftQ override: if pre-computed (lora_A, lora_B) are provided, use them
            loftq_init_data = kwargs.get("loftq_init_data", None)
            if loftq_init_data is not None:
                lora_A, lora_B = loftq_init_data
                self.lora_down.weight.data.copy_(lora_A)
                self.lora_up.weight.data.copy_(lora_B)
        else:
            # conv2d not supported
            assert sum(split_dims) == out_dim, "sum of split_dims must be equal to out_dim"
            assert org_module.__class__.__name__ == "Linear", "split_dims is only supported for Linear"
            # print(f"split_dims: {split_dims}")
            self.lora_down = torch.nn.ModuleList(
                [torch.nn.Linear(in_dim, self.lora_dim, bias=False) for _ in range(len(split_dims))]
            )
            self.lora_up = torch.nn.ModuleList([torch.nn.Linear(self.lora_dim, split_dim, bias=False) for split_dim in split_dims])
            for lora_down in self.lora_down:
                torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            for lora_up in self.lora_up:
                torch.nn.init.zeros_(lora_up.weight)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        alpha = float(alpha)
        self.use_rslora = _parse_bool_network_arg(kwargs.get("use_rslora", kwargs.get("rslora", False)))
        if self.use_rslora:
            self.scale = alpha / math.sqrt(self.lora_dim)
            # Store the effective alpha so regular LoRA loaders apply the same alpha/r scale.
            saved_alpha = alpha * math.sqrt(self.lora_dim)
        else:
            self.scale = alpha / self.lora_dim
            saved_alpha = alpha
        self.register_buffer("alpha", torch.tensor(saved_alpha))  # for save/load
        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        # RL hooks (default-on so supervised / slider training is unchanged):
        #   enabled=False         -> bypass the adapter entirely (NFT `ref` forward). DoRA-safe:
        #                            returns org_forward, skipping the magnitude path (multiplier=0
        #                            would NOT, because magnitude still rescales the base).
        #   dropout_enabled=False -> deterministic forward even under .train() (NFT old/default/ref
        #                            must be bit-comparable policy evaluations).
        self.enabled = True
        self.dropout_enabled = True

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training and self.dropout_enabled:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.split_dims is None:
            lx = self.lora_down(x)

            # normal dropout
            if self.dropout is not None and self.training and self.dropout_enabled:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout
            if self.rank_dropout is not None and self.training and self.dropout_enabled:
                mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)  # for Text Encoder
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
                elif len(lx.size()) == 5:
                    mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # for Conv3d
                lx = lx * mask

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            if self.adaptive_rank:
                rank_weights = self._adaptive_rank_weights(self.lora_dim, dtype=lx.dtype, device=lx.device)
                lx = lx * self._reshape_rank_weights(rank_weights, lx)

            lx = self.lora_up(lx)

            return org_forwarded + lx * self.multiplier * scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]

            # normal dropout
            if self.dropout is not None and self.training and self.dropout_enabled:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
            if self.rank_dropout is not None and self.training and self.dropout_enabled:
                masks = [torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout for lx in lxs]
                for i in range(len(lxs)):
                    if len(lx.size()) == 3:
                        masks[i] = masks[i].unsqueeze(1)
                    elif len(lx.size()) == 4:
                        masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                    lxs[i] = lxs[i] * masks[i]

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            if self.adaptive_rank:
                rank_weights = self._adaptive_rank_weights(self.lora_dim, dtype=lxs[0].dtype, device=lxs[0].device)
                lxs = [lx * self._reshape_rank_weights(rank_weights, lx) for lx in lxs]

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]  # for reference
        self.enabled = True
        self.network: LoRANetwork = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device, non_blocking=False):
        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(device, dtype=torch.float, non_blocking=non_blocking)  # for calculation

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        if self.split_dims is None:
            # get up/down weight
            down_weight = sd["lora_down.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
            up_weight = sd["lora_up.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)

            # merge weight
            if len(weight.size()) == 2:
                # linear
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
            elif len(weight.size()) == 4:
                if down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + self.multiplier
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * self.scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + self.multiplier * conved * self.scale
            elif len(weight.size()) == 5:
                if down_weight.size()[2:5] == (1, 1, 1):
                    # conv3d 1x1x1
                    weight = (
                        weight
                        + self.multiplier
                        * (up_weight.squeeze(4).squeeze(3).squeeze(2) @ down_weight.squeeze(4).squeeze(3).squeeze(2))
                        .unsqueeze(2)
                        .unsqueeze(3)
                        .unsqueeze(4)
                        * self.scale
                    )
                else:
                    conved = torch.nn.functional.conv3d(down_weight.permute(1, 0, 2, 3, 4), up_weight).permute(1, 0, 2, 3, 4)
                    weight = weight + self.multiplier * conved * self.scale
            else:
                raise ValueError(f"Unsupported LoRA target weight shape: {weight.size()}")

            # set weight to org_module
            org_sd["weight"] = weight.to(org_device, dtype=dtype)  # back to CPU without non_blocking
            self.org_module.load_state_dict(org_sd)
        else:
            # split_dims
            total_dims = sum(self.split_dims)
            for i in range(len(self.split_dims)):
                # get up/down weight
                down_weight = sd[f"lora_down.{i}.weight"].to(device, torch.float, non_blocking=non_blocking)  # (rank, in_dim)
                up_weight = sd[f"lora_up.{i}.weight"].to(device, torch.float, non_blocking=non_blocking)  # (split dim, rank)

                # pad up_weight -> (total_dims, rank)
                padded_up_weight = torch.zeros((total_dims, up_weight.size(0)), device=device, dtype=torch.float)
                padded_up_weight[sum(self.split_dims[:i]) : sum(self.split_dims[: i + 1])] = up_weight

                # merge weight
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale

            # set weight to org_module
            org_sd["weight"] = weight.to(org_device, dtype)  # back to CPU without non_blocking
            self.org_module.load_state_dict(org_sd)

    # return weight for merge
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif len(down_weight.size()) == 4:
            if down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    self.multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * self.scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = self.multiplier * conved * self.scale
        elif len(down_weight.size()) == 5:
            if down_weight.size()[2:5] == (1, 1, 1):
                # conv3d 1x1x1
                weight = (
                    self.multiplier
                    * (up_weight.squeeze(4).squeeze(3).squeeze(2) @ down_weight.squeeze(4).squeeze(3).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .unsqueeze(4)
                    * self.scale
                )
            else:
                conved = torch.nn.functional.conv3d(down_weight.permute(1, 0, 2, 3, 4), up_weight).permute(1, 0, 2, 3, 4)
                weight = self.multiplier * conved * self.scale
        else:
            raise ValueError(f"Unsupported LoRA weight shape: {down_weight.size()}")

        return weight

    def default_forward(self, x):
        # logger.info(f"default_forward {self.lora_name} {x.size()}")
        if self.split_dims is None:
            lx = self.lora_down(x)
            lx = self.lora_up(lx)
            return self.org_forward(x) + lx * self.multiplier * self.scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]
            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
            return self.org_forward(x) + torch.cat(lxs, dim=-1) * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class DoRAModule(LoRAModule):
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

        self.org_module_ref = [org_module]
        magnitude = self._get_weight_norm(self.multiplier * self.scale)
        reference_param = self.lora_down[0].weight if self.split_dims is not None else self.lora_down.weight
        magnitude = magnitude.to(device=reference_param.device, dtype=reference_param.dtype)
        self.lora_magnitude_vector = DoRAMagnitudeModule(magnitude)

    def _get_base_module(self) -> torch.nn.Module:
        if hasattr(self, "org_module"):
            return self.org_module
        return self.org_module_ref[0]

    def _get_is_conv2d(self) -> bool:
        return self._get_base_module().__class__.__name__ == "Conv2d"

    def _apply_adaptive_rank_to_down_weight(self, down_weight: torch.Tensor) -> torch.Tensor:
        if not self.adaptive_rank:
            return down_weight

        rank_weights = self._adaptive_rank_weights(self.lora_dim, dtype=down_weight.dtype, device=down_weight.device)
        if down_weight.dim() == 2:
            return down_weight * rank_weights[:, None]
        return down_weight * rank_weights[:, None, None, None]

    def _get_weight_norm(self, scaling: float) -> torch.Tensor:
        org_module = self._get_base_module()
        with torch.no_grad():
            base_weight = _get_effective_module_weight(org_module, detach=True)
            if self.split_dims is None:
                return _get_dora_weight_norm(
                    base_weight,
                    self._apply_adaptive_rank_to_down_weight(self.lora_down.weight),
                    self.lora_up.weight,
                    scaling,
                )
            return _get_dora_weight_norm(
                base_weight,
                [self._apply_adaptive_rank_to_down_weight(module.weight) for module in self.lora_down],
                [module.weight for module in self.lora_up],
                scaling,
                split_dims=self.split_dims,
            )

    def _get_delta_weight(self, multiplier: Optional[float] = None) -> torch.Tensor:
        multiplier = self.multiplier if multiplier is None else multiplier
        scaling = multiplier * self.scale

        if self.split_dims is None:
            down_weight = self._apply_adaptive_rank_to_down_weight(self.lora_down.weight.to(torch.float))
            return _get_lora_weight_from_tensors(down_weight, self.lora_up.weight.to(torch.float)) * scaling
        return (
            _get_split_lora_weight(
                self.split_dims,
                [self._apply_adaptive_rank_to_down_weight(module.weight.to(torch.float)) for module in self.lora_down],
                [module.weight.to(torch.float) for module in self.lora_up],
            )
            * scaling
        )

    def _get_lora_output(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.split_dims is None:
            lx = self.lora_down(x)

            if self.dropout is not None and self.training and self.dropout_enabled:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            if self.rank_dropout is not None and self.training and self.dropout_enabled:
                mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)
                lx = lx * mask
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
            else:
                scale = self.scale

            if self.adaptive_rank:
                rank_weights = self._adaptive_rank_weights(self.lora_dim, dtype=lx.dtype, device=lx.device)
                lx = lx * self._reshape_rank_weights(rank_weights, lx)

            return self.lora_up(lx), scale

        lxs = [lora_down(x) for lora_down in self.lora_down]

        if self.dropout is not None and self.training and self.dropout_enabled:
            lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

        if self.rank_dropout is not None and self.training and self.dropout_enabled:
            masks = [torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout for lx in lxs]
            for i in range(len(lxs)):
                if len(lxs[i].size()) == 3:
                    masks[i] = masks[i].unsqueeze(1)
                elif len(lxs[i].size()) == 4:
                    masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                lxs[i] = lxs[i] * masks[i]
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        if self.adaptive_rank:
            rank_weights = self._adaptive_rank_weights(self.lora_dim, dtype=lxs[0].dtype, device=lxs[0].device)
            lxs = [lx * self._reshape_rank_weights(rank_weights, lx) for lx in lxs]

        lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
        return torch.cat(lxs, dim=-1), scale

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = super().export_state_dict()
        state_dict[f"{self.lora_name}.lora_magnitude_vector.weight"] = self.lora_magnitude_vector.weight.detach().clone()
        return state_dict

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training and self.dropout_enabled:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return org_forwarded

        lora_output, scale = self._get_lora_output(x)
        effective_scale = self.multiplier * scale

        magnitude = self.lora_magnitude_vector.weight
        weight_norm = self._get_weight_norm(effective_scale)
        if weight_norm.device != magnitude.device:
            weight_norm = weight_norm.to(device=magnitude.device)
        if weight_norm.is_floating_point():
            eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
            weight_norm = weight_norm.clamp_min(eps)

        org_module = self.org_module_ref[0]
        base_without_bias = _remove_module_bias(org_forwarded, org_module.bias, self._get_is_conv2d())
        mag_norm_scale = _reshape_dora_factor_for_output(
            magnitude / weight_norm,
            lora_output,
            self._get_is_conv2d(),
        )

        compose_dtype = torch.promote_types(torch.promote_types(base_without_bias.dtype, lora_output.dtype), mag_norm_scale.dtype)
        factor = mag_norm_scale.to(compose_dtype)
        delta = (factor - 1.0) * base_without_bias.to(compose_dtype) + factor * (lora_output.to(compose_dtype) * effective_scale)

        return org_forwarded + delta.to(org_forwarded.dtype)


class DoRAInfModule(DoRAModule):
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
        self.network: LoRANetwork = None

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

        base_weight = _get_effective_module_weight(
            org_module,
            dtype=torch.float,
            device=device,
            detach=True,
            non_blocking=non_blocking,
        )
        if self.split_dims is None:
            down_weight = sd["lora_down.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
            up_weight = sd["lora_up.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
            delta_weight = _get_lora_weight_from_tensors(down_weight, up_weight) * (self.multiplier * self.scale)
            weight_norm = _get_dora_weight_norm(base_weight, down_weight, up_weight, self.multiplier * self.scale)
        else:
            down_weight = [
                sd[f"lora_down.{i}.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
                for i in range(len(self.split_dims))
            ]
            up_weight = [
                sd[f"lora_up.{i}.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
                for i in range(len(self.split_dims))
            ]
            delta_weight = _get_split_lora_weight(self.split_dims, down_weight, up_weight) * (self.multiplier * self.scale)
            weight_norm = _get_dora_weight_norm(
                base_weight,
                down_weight,
                up_weight,
                self.multiplier * self.scale,
                split_dims=self.split_dims,
            )

        magnitude = sd["lora_magnitude_vector.weight"].to(device, dtype=weight_norm.dtype, non_blocking=non_blocking)
        merged_weight = _apply_dora_weight_merge(base_weight, delta_weight, magnitude, weight_norm)
        org_sd["weight"] = merged_weight.to(org_device, dtype=dtype)
        org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        multiplier = self.multiplier if multiplier is None else multiplier
        org_module = self.org_module_ref[0]
        base_weight = _get_effective_module_weight(org_module, dtype=torch.float)
        delta_weight = self._get_delta_weight(multiplier=multiplier).to(base_weight.device)
        weight_norm = self._get_weight_norm(multiplier * self.scale).to(base_weight.device)
        magnitude = self.lora_magnitude_vector.weight.to(device=base_weight.device, dtype=weight_norm.dtype)
        merged_weight = _apply_dora_weight_merge(base_weight, delta_weight, magnitude, weight_norm)
        return merged_weight - base_weight

    def default_forward(self, x):
        return DoRAModule.forward(self, x)

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


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
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # exclude if 'img_mod', 'txt_mod' or 'modulation' in the name
    exclude_patterns.append(r".*(img_mod|txt_mod|modulation).*")

    kwargs["exclude_patterns"] = exclude_patterns

    return create_network(
        HUNYUAN_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    module_class: Type[object] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
    network_class: Type["LoRANetwork"] = None,
    **kwargs,
):
    """architecture independent network creation"""
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    use_dora = _parse_bool_network_arg(kwargs.get("use_dora", False))
    use_dora_oft = _parse_bool_network_arg(kwargs.get("use_dora_oft", False))
    use_oft = _parse_bool_network_arg(kwargs.get("use_oft", False)) or use_dora_oft
    use_rslora = _parse_bool_network_arg(kwargs.get("use_rslora", kwargs.get("rslora", False)))
    if use_dora and use_dora_oft:
        raise ValueError("use_dora and use_dora_oft cannot both be enabled")
    if use_dora_oft and _parse_bool_network_arg(kwargs.get("adaptive_rank", False)):
        raise ValueError("adaptive_rank is not supported with use_dora_oft")
    if use_rslora and use_oft:
        raise ValueError("use_rslora is not supported with use_oft or use_dora_oft")

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # per-modality dim/alpha overrides
    audio_dim = kwargs.get("audio_dim", None)
    if audio_dim is not None:
        audio_dim = int(audio_dim)
    audio_alpha = kwargs.get("audio_alpha", None)
    if audio_alpha is not None:
        audio_alpha = float(audio_alpha)
    cross_modal_dim = kwargs.get("cross_modal_dim", None)
    if cross_modal_dim is not None:
        cross_modal_dim = int(cross_modal_dim)
    cross_modal_alpha = kwargs.get("cross_modal_alpha", None)
    if cross_modal_alpha is not None:
        cross_modal_alpha = float(cross_modal_alpha)
    adaptive_rank_kwargs = parse_adaptive_rank_network_kwargs(kwargs)

    # per-modality dropout overrides
    audio_dropout = kwargs.get("audio_dropout", None)
    if audio_dropout is not None:
        audio_dropout = float(audio_dropout)
    video_dropout = kwargs.get("video_dropout", None)
    if video_dropout is not None:
        video_dropout = float(video_dropout)
    cross_modal_dropout = kwargs.get("cross_modal_dropout", None)
    if cross_modal_dropout is not None:
        cross_modal_dropout = float(cross_modal_dropout)

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # verbose
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    # regular expression for module selection: exclude and include
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)

    if module_class is None:
        if use_dora_oft:
            module_class = DoRAOFTModule
        elif use_oft:
            module_class = OFTModule
        else:
            module_class = DoRAModule if use_dora else LoRAModule

    effective_module_kwargs = dict(module_kwargs or {})
    if use_rslora:
        effective_module_kwargs["use_rslora"] = True
    if use_oft:
        if use_dora_oft:
            effective_module_kwargs["scaled_oft"] = True
        for key in (
            "oft_block_size",
            "oft_coft",
            "coft_eps",
            "oft_block_share",
            "oft_dropout",
            "scaled_oft",
            "freeze_dora_scale",
            "dora_scale_lr_ratio",
            "dora_scale_min_ratio",
            "dora_scale_max_ratio",
        ):
            if kwargs.get(key, None) is not None:
                effective_module_kwargs[key] = kwargs.get(key)

    if network_class is None:
        network_class = LoRANetwork

    # too many arguments ( ^ω^)･･･
    network = network_class(
        target_replace_modules,
        prefix,
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        module_class=module_class,
        module_kwargs=effective_module_kwargs,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
        audio_dim=audio_dim,
        audio_alpha=audio_alpha,
        audio_dropout=audio_dropout,
        video_dropout=video_dropout,
        cross_modal_dim=cross_modal_dim,
        cross_modal_alpha=cross_modal_alpha,
        cross_modal_dropout=cross_modal_dropout,
        adaptive_rank_config=adaptive_rank_kwargs,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    # loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    # loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    # loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    # loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None:  # or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)  # , loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


class LoRANetwork(AdaptiveRankLoRANetworkMixin, torch.nn.Module):
    # only supports U-Net (DiT), Text Encoders are not supported

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Union[List[CLIPTextModel], CLIPTextModel],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        module_kwargs: Optional[Dict[str, Any]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        audio_dim: Optional[int] = None,
        audio_alpha: Optional[float] = None,
        audio_dropout: Optional[float] = None,
        video_dropout: Optional[float] = None,
        cross_modal_dim: Optional[int] = None,
        cross_modal_alpha: Optional[float] = None,
        cross_modal_dropout: Optional[float] = None,
        adaptive_rank_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.target_replace_modules = target_replace_modules
        self.prefix = prefix
        raw_module_kwargs = dict(module_kwargs or {})
        per_module_kwargs = raw_module_kwargs.pop("per_module_kwargs", {})
        self.module_kwargs = raw_module_kwargs
        self.per_module_kwargs = per_module_kwargs if isinstance(per_module_kwargs, dict) else {}
        self.use_rslora = _parse_bool_network_arg(raw_module_kwargs.get("use_rslora", raw_module_kwargs.get("rslora", False)))
        self.audio_dim = audio_dim
        self.audio_alpha = audio_alpha
        self.audio_dropout = audio_dropout
        self.video_dropout = video_dropout
        self.cross_modal_dim = cross_modal_dim
        self.cross_modal_alpha = cross_modal_alpha
        self.cross_modal_dropout = cross_modal_dropout
        normalized_adaptive_rank_config = parse_adaptive_rank_network_kwargs(adaptive_rank_config or {})
        self._init_adaptive_rank_network_state(**normalized_adaptive_rank_config)

        self.loraplus_lr_ratio = None
        self.dora_scale_lr_ratio = _parse_optional_float_network_arg(raw_module_kwargs.pop("dora_scale_lr_ratio", None), None)
        if self.dora_scale_lr_ratio is not None and self.dora_scale_lr_ratio < 0:
            raise ValueError("dora_scale_lr_ratio must be non-negative")
        # self.loraplus_unet_lr_ratio = None
        # self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            if self.audio_dim is not None:
                logger.info(
                    f"audio modules: dim (rank): {self.audio_dim}, alpha: {self.audio_alpha if self.audio_alpha is not None else alpha}"
                )
            if self.cross_modal_dim is not None:
                logger.info(
                    f"cross-modal modules: dim (rank): {self.cross_modal_dim}, alpha: {self.cross_modal_alpha if self.cross_modal_alpha is not None else alpha}"
                )
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            if self.use_rslora:
                logger.info("rsLoRA scaling enabled: alpha / sqrt(rank)")
            if self.audio_dropout is not None or self.video_dropout is not None or self.cross_modal_dropout is not None:
                logger.info(
                    f"per-modality dropout overrides: video={self.video_dropout}, audio={self.audio_dropout}, cross-modal={self.cross_modal_dropout}"
                )
            if self.dora_scale_lr_ratio is not None:
                logger.info(f"DoRA scale LR ratio: {self.dora_scale_lr_ratio}")
            self._log_adaptive_rank_configuration(modules_dim)
            # if self.conv_lora_dim is not None:
            #     logger.info(
            #         f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}"
            #     )
        # if train_t5xxl:
        #     logger.info(f"train T5XXL as well")

        # compile regular expression if specified
        exclude_re_patterns = []
        if exclude_patterns is not None:
            for pattern in exclude_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid exclude pattern '{pattern}': {e}")
                    continue
                exclude_re_patterns.append(re_pattern)

        include_re_patterns = []
        has_include_filter = include_patterns is not None
        if include_patterns is not None:
            for pattern in include_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid include pattern '{pattern}': {e}")
                    continue
                include_re_patterns.append(re_pattern)

        # create module instances
        def create_modules(
            is_unet: bool,
            pfx: str,
            root_module: torch.nn.Module,
            target_replace_mods: Optional[List[str]] = None,
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
        ) -> List[LoRAModule]:
            loras = []
            skipped = []

            def is_audio_module(module_name: str) -> bool:
                return self._is_audio_module(module_name)

            def is_cross_modal_module(module_name: str) -> bool:
                return self._is_cross_modal_module(module_name)

            def resolve_module_dropout(module_name: str) -> Optional[float]:
                if is_cross_modal_module(module_name) and self.cross_modal_dropout is not None:
                    return self.cross_modal_dropout
                if is_audio_module(module_name):
                    if self.audio_dropout is not None:
                        return self.audio_dropout
                elif self.video_dropout is not None:
                    return self.video_dropout
                return self.dropout

            def resolve_adaptive_rank_target(module_name: str) -> Optional[int]:
                return self.resolve_module_adaptive_rank_target(module_name)

            def resolve_adaptive_rank_weight(module_name: str) -> Optional[float]:
                return self.resolve_module_adaptive_rank_weight(module_name)

            for name, module in root_module.named_modules():
                if target_replace_mods is None or module.__class__.__name__ in target_replace_mods:
                    if target_replace_mods is None:  # dirty hack for all modules
                        module = root_module  # search all modules

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv3d = child_module.__class__.__name__ == "Conv3d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
                        is_conv3d_1x1 = is_conv3d and child_module.kernel_size == (1, 1, 1)

                        if is_linear or is_conv2d or is_conv3d:
                            original_name = (name + "." if name else "") + child_name
                            lora_name = f"{pfx}.{original_name}".replace(".", "_")

                            # exclude/include filter
                            excluded = False
                            for pattern in exclude_re_patterns:
                                if pattern.fullmatch(original_name):
                                    excluded = True
                                    break
                            included = False
                            for pattern in include_re_patterns:
                                if pattern.fullmatch(original_name):
                                    included = True
                                    break
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue
                            if has_include_filter and not included:
                                if verbose:
                                    logger.info(f"not included: {original_name}")
                                continue

                            # filter by name (not used in the current implementation)
                            if filter is not None and filter not in lora_name:
                                continue

                            dim = None
                            alpha = None
                            module_dropout_value = resolve_module_dropout(original_name)

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            else:
                                # 通常、すべて対象とする
                                if is_linear or is_conv2d_1x1 or is_conv3d_1x1:
                                    dim = default_dim if default_dim is not None else self.lora_dim
                                    alpha = self.alpha
                                    # per-modality override: audio modules get audio_dim/audio_alpha
                                    if self.audio_dim is not None and is_audio_module(original_name):
                                        dim = self.audio_dim
                                        if self.audio_alpha is not None:
                                            alpha = self.audio_alpha
                                    if is_cross_modal_module(original_name):
                                        if self.cross_modal_dim is not None:
                                            dim = self.cross_modal_dim
                                        if self.cross_modal_alpha is not None:
                                            alpha = self.cross_modal_alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or is_conv3d_1x1 or (self.conv_lora_dim is not None):
                                    skipped.append(lora_name)
                                continue

                            # Build per-module kwargs, injecting per-module OFT metadata and LoftQ data if available
                            per_module_kwargs = dict(self.module_kwargs)
                            if self.per_module_kwargs and lora_name in self.per_module_kwargs:
                                per_module_kwargs.update(self.per_module_kwargs[lora_name])
                            if (
                                modules_dim is not None
                                and lora_name in modules_dim
                                and "oft_block_size" not in per_module_kwargs
                                and (issubclass(module_class, OFTModule) or issubclass(module_class, DoRAOFTModule))
                            ):
                                per_module_kwargs["oft_block_size"] = int(modules_dim[lora_name])
                            loftq_data = per_module_kwargs.pop("loftq_data", None)
                            if loftq_data is not None and lora_name in loftq_data:
                                per_module_kwargs["loftq_init_data"] = loftq_data[lora_name]
                            adaptive_rank_target_value = resolve_adaptive_rank_target(original_name)
                            adaptive_rank_weight_value = resolve_adaptive_rank_weight(original_name)

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=module_dropout_value,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                module_path=original_name,
                                adaptive_rank=self.adaptive_rank,
                                adaptive_rank_target=adaptive_rank_target_value,
                                adaptive_rank_quantile=self.adaptive_rank_quantile,
                                adaptive_rank_weight=adaptive_rank_weight_value,
                                adaptive_rank_min_rank=self.adaptive_rank_min_rank,
                                adaptive_rank_init_rank=self.adaptive_rank_init_rank,
                                **per_module_kwargs,
                            )
                            loras.append(lora)

                if target_replace_mods is None:
                    break  # all modules are searched
            return loras, skipped

        # # create LoRA for text encoder
        # # it is redundant to create LoRA modules even if they are not used

        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        # skipped_te = []
        # for i, text_encoder in enumerate(text_encoders):
        #     index = i
        #     if not train_t5xxl and index > 0:  # 0: CLIP, 1: T5XXL, so we skip T5XXL if train_t5xxl is False
        #         break
        #     logger.info(f"create LoRA for Text Encoder {index+1}:")
        #     text_encoder_loras, skipped = create_modules(False, index, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        #     logger.info(f"create LoRA for Text Encoder {index+1}: {len(text_encoder_loras)} modules.")
        #     self.text_encoder_loras.extend(text_encoder_loras)
        #     skipped_te += skipped

        # create LoRA for U-Net
        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(True, prefix, unet, target_replace_modules)

        logger.info(f"create LoRA for U-Net/DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(
                f"because dim (rank) is 0, {len(skipped)} LoRA modules are skipped / dim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                logger.info(f"\t{name}")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        if self.adaptive_rank_estimate_report is not None:
            self._apply_adaptive_rank_estimate_overrides()

    def prepare_network(self, args):
        """
        called after the network is created
        """
        self.prepare_adaptive_rank(args)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def set_dropout_enabled(self, is_enabled):
        """Toggle LoRA dropout independently of ``.train()`` mode.

        RL forwards (NFT old/default/ref) must be deterministic to be comparable, so the
        driver calls ``set_dropout_enabled(False)`` while leaving the module in train mode.
        Default is True (no change to supervised training).
        """
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.dropout_enabled = is_enabled

    def trainable_lora_params(self):
        """Canonical, stably-ordered list of every trainable adapter param.

        Includes lora_down/lora_up, DoRA magnitude vectors, and adaptive-rank lambdas
        (everything with ``requires_grad``) — so the NFT ``old`` EMA snapshot covers all
        trainable tensors, not a hardcoded {down, up, magnitude} subset.
        """
        return [p for _, p in self.named_parameters() if p.requires_grad]

    @contextlib.contextmanager
    def swapped_weights(self, ema_tensors):
        """Temporarily replace trainable params' ``.data`` with ``ema_tensors`` (aligned to
        ``trainable_lora_params()`` order), restoring the originals on exit **even if the body
        raises**. Used to run the NFT ``old`` (EMA) forward without a second full model copy.
        """
        params = self.trainable_lora_params()
        ema_tensors = list(ema_tensors)
        if len(params) != len(ema_tensors):
            raise ValueError(f"swapped_weights: got {len(ema_tensors)} ema tensors for {len(params)} trainable params")
        saved = [p.data for p in params]
        try:
            for param, ema in zip(params, ema_tensors):
                param.data = ema
            yield
        finally:
            for param, original in zip(params, saved):
                param.data = original

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_weights_state_dict(weights_sd, False)
        return info

    def prepare_weights_state_dict_for_load(self, state_dict):
        """Return a loadable copy of an exported adapter state dict.

        Training checkpoints use raw module state_dict values. Exported adapter
        files use absolute DoRA-OFT scales, so this conversion is intentionally
        kept out of load_state_dict() to preserve normal PyTorch resume semantics.
        """
        if any(key.endswith(".dora_scale") or key.endswith(".oft_R.scaled_oft") for key in state_dict.keys()):
            state_dict = copy.copy(state_dict)
            for key in list(state_dict.keys()):
                if key.endswith(".oft_R.scaled_oft"):
                    state_dict.pop(key, None)
            _convert_absolute_dora_oft_scales_to_ratios(state_dict)

        return state_dict

    def load_weights_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = self.prepare_weights_state_dict_for_load(state_dict)
        if assign:
            return self.load_state_dict(state_dict, strict=strict, assign=assign)
        return self.load_state_dict(state_dict, strict)

    def apply_to(
        self,
        text_encoders: Optional[nn.Module],
        unet: Optional[nn.Module],
        apply_text_encoder: bool = True,
        apply_unet: bool = True,
    ):
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        if len(self.text_encoder_loras) == 0 and len(self.unet_loras) == 0:
            logger.error(
                "No LoRA modules. Please check `--network_module` and `--network_args`"
                " / LoRAモジュールがありません。`--network_module`と`--network_args`を確認してください"
            )
            raise RuntimeError("No LoRA modules found")

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    # マージできるかどうかを返す
    def is_mergeable(self):
        return True

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None, non_blocking=False):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:  # 2 workers is enough
            futures = []
            for lora in self.text_encoder_loras + self.unet_loras:
                sd_for_lora = {}
                for key in weights_sd.keys():
                    if key.startswith(lora.lora_name):
                        sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
                if len(sd_for_lora) == 0:
                    logger.info(f"no weight for {lora.lora_name}")
                    continue

                # lora.merge_to(sd_for_lora, dtype, device)
                futures.append(executor.submit(lora.merge_to, sd_for_lora, dtype, device, non_blocking))

        for future in futures:
            future.result()

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio):  # , loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_lr_ratio}")
        # logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params(self, unet_lr: float = 1e-4, audio_lr=None, lr_args=None, **kwargs):
        self.requires_grad_(True)
        self._apply_trainable_param_overrides()

        # Parse lr_args from CLI format ["pattern=lr", ...] → dict
        lr_patterns = {}
        if lr_args:
            for entry in lr_args:
                if "=" not in entry:
                    raise ValueError(f"Invalid --lr_args entry (expected pattern=lr): {entry}")
                pattern, lr_str = entry.split("=", 1)
                lr_patterns[pattern] = float(lr_str)

        # If no custom LR config, use original fast path
        if not lr_patterns and audio_lr is None:
            return self._prepare_optimizer_params_simple(unet_lr)

        # Group LoRA modules by resolved LR
        lr_to_params = {}  # lr_value → {"lora": {name: param}, "plus": {name: param}}
        lr_to_desc = {}  # lr_value → description string

        for lora in self.unet_loras:
            resolved_lr = unet_lr  # default
            desc = "video"

            # Check lr_args patterns first (highest priority)
            matched_pattern = False
            for pattern, pattern_lr in lr_patterns.items():
                if re.search(pattern, lora.lora_name):
                    resolved_lr = pattern_lr
                    desc = pattern
                    matched_pattern = True
                    break

            # If no pattern matched, check audio_lr
            if not matched_pattern and audio_lr is not None:
                if "audio_" in lora.lora_name:
                    resolved_lr = audio_lr
                    desc = "audio"

            for name, param in lora.named_parameters():
                if not param.requires_grad:
                    continue
                param_lr = resolved_lr
                param_desc = desc
                param_group_key = "lora"
                if name == "dora_scale" and self.dora_scale_lr_ratio is not None:
                    param_lr = resolved_lr * self.dora_scale_lr_ratio
                    param_desc = f"{desc}_dora_scale"
                    if param_lr == 0:
                        continue
                group = lr_to_params.setdefault(param_lr, {"lora": {}, "plus": {}})
                lr_to_desc.setdefault(param_lr, param_desc)
                key = f"{lora.lora_name}.{name}"
                if self.loraplus_lr_ratio is not None and "lora_up" in name:
                    param_group_key = "plus"
                group[param_group_key][key] = param

        # Build final param groups
        all_params = []
        lr_descriptions = []
        for lr_val in sorted(lr_to_params.keys()):
            groups = lr_to_params[lr_val]
            desc = lr_to_desc[lr_val]
            for key in ("lora", "plus"):
                if not groups[key]:
                    continue
                suffix = " plus" if key == "plus" else ""
                param_data = {"params": list(groups[key].values()), "lr": lr_val}
                if key == "plus" and self.loraplus_lr_ratio:
                    param_data["lr"] = lr_val * self.loraplus_lr_ratio
                param_data["group_name"] = f"unet_{desc}{suffix}".replace(" ", "_")
                all_params.append(param_data)
                lr_descriptions.append(f"unet_{desc}{suffix}")

        # Log group breakdown
        logger.info(f"LR groups: {len(all_params)} groups created")
        for param_data, desc in zip(all_params, lr_descriptions):
            logger.info(f"  {desc}: lr={param_data['lr']}, {len(param_data['params'])} params")

        return all_params, lr_descriptions

    def _prepare_optimizer_params_simple(self, unet_lr: float = 1e-4):
        """Original single-group optimizer param assembly (no per-module LR)."""
        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            dora_scale_params = {}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name == "dora_scale" and self.dora_scale_lr_ratio is not None:
                        dora_scale_params[f"{lora.lora_name}.{name}"] = param
                        continue
                    if loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            if loraplus_ratio is not None and len(param_groups["plus"]) == 0:
                logger.warning("LoRA+ is not effective for this network type (no 'lora_up' parameters found)")

            params = []
            descriptions = []
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}

                if len(param_data["params"]) == 0:
                    continue

                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr

                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue

                param_data["group_name"] = "unet_plus" if key == "plus" else "unet"
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")

            if dora_scale_params and lr is not None and self.dora_scale_lr_ratio is not None:
                dora_scale_lr = lr * self.dora_scale_lr_ratio
                if dora_scale_lr != 0:
                    params.append(
                        {
                            "params": dora_scale_params.values(),
                            "lr": dora_scale_lr,
                            "group_name": "unet_dora_scale",
                        }
                    )
                    descriptions.append("dora_scale")

            return params, descriptions

        if self.unet_loras:
            params, descriptions = assemble_params(self.unet_loras, unet_lr, self.loraplus_lr_ratio)
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, unet):
        self.requires_grad_(True)
        self._apply_trainable_param_overrides()

    def _apply_trainable_param_overrides(self):
        for lora in self.unet_loras:
            if getattr(lora, "freeze_dora_scale", False) and hasattr(lora, "dora_scale"):
                lora.dora_scale.requires_grad_(False)

    def on_epoch_start(self, unet):
        self.train()

    def get_trainable_params(self):
        return (param for param in self.parameters() if param.requires_grad)

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.build_export_state_dict()
        adaptive_rank_report = self.build_adaptive_rank_report()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from musubi_tuner.utils import model_utils

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

        if adaptive_rank_report is not None:
            with open(self._adaptive_rank_report_path(file), "w", encoding="utf-8") as f:
                json.dump(adaptive_rank_report, f, indent=2, sort_keys=True)

    def backup_weights(self):
        # 重みのバックアップを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        # 重みのリストアを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        # 事前計算を行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()

        if any("lora_magnitude_vector" in key or "dora_scale" in key for key in state_dict.keys()):
            logger.warning("max_norm_regularization is not supported for DoRA networks")
            return 0, 0.0, 0.0

        # guard: only supported for LoRA (lora_down/lora_up parameterization)
        if not any("lora_down" in k and "weight" in k for k in state_dict.keys()):
            logger.warning("max_norm_regularization is only supported for LoRA")
            return 0, 0.0, 0.0

        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)

    def apply_dora_scale_regularization(self) -> int:
        keys_clamped = 0
        for lora in self.unet_loras:
            if hasattr(lora, "clamp_dora_scale"):
                keys_clamped += lora.clamp_dora_scale()
        return keys_clamped


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
    return create_network_from_weights(
        HUNYUAN_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )


def merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Merge native LoRA/DoRA/OFT/DoRA-OFT weights into a model weight tensor."""
    oft_key = lora_name + ".oft_R.weight"
    dora_oft_scale_key = lora_name + ".dora_scale"
    initial_norm_key = lora_name + ".initial_norm"
    alpha_key = lora_name + ".alpha"

    if oft_key in lora_weight_keys:
        original_dtype = model_weight.dtype
        work_dtype = torch.float16 if original_dtype.itemsize == 1 else model_weight.dtype
        model_weight = model_weight.to(calc_device, dtype=work_dtype)

        oft_weight = lora_sd[oft_key].to(device=calc_device, dtype=work_dtype)
        module_state = {
            key[len(lora_name) + 1 :]: lora_sd[key] for key in list(lora_weight_keys) if key.startswith(lora_name + ".")
        }
        n_elements = int(oft_weight.shape[1])
        default_block_size = _solve_oft_block_size(n_elements)
        alpha = lora_sd.get(alpha_key)
        if alpha is not None:
            default_block_size = int(round(float(alpha.detach().cpu().item() if isinstance(alpha, torch.Tensor) else alpha)))
        flat_shape = model_weight.reshape(model_weight.shape[0], -1)
        in_features = int(flat_shape.shape[1])
        block_size = _metadata_tensor_to_int(module_state.get("oft_block_size_metadata"), default_block_size)
        block_size = OFTModule._adjust_oft_block_size(in_features, block_size)
        block_share = _metadata_tensor_to_bool(
            module_state.get("oft_block_share_metadata"),
            default=bool(int(oft_weight.shape[0]) == 1),
        )
        coft = _metadata_tensor_to_bool(module_state.get("oft_coft_metadata"), default=False)
        coft_eps = _metadata_tensor_to_float(module_state.get("coft_eps_metadata"), 6e-5)
        scaled_oft = _metadata_tensor_to_bool(
            module_state.get("scaled_oft_metadata"),
            default=(lora_name + ".oft_R.scaled_oft") in lora_weight_keys,
        )
        rotation_module = OFTRotationModule(
            r=int(oft_weight.shape[0]),
            n_elements=n_elements,
            block_size=block_size,
            in_features=in_features,
            coft=coft,
            coft_eps=coft_eps,
            block_share=block_share,
            scaled_oft=scaled_oft,
            use_cayley_neumann=True,
            num_cayley_neumann_terms=5,
            dropout_probability=0.0,
        ).to(device=calc_device, dtype=work_dtype)
        with torch.no_grad():
            rotation_module.weight.copy_(oft_weight)
        rotation = rotation_module.rotation_matrix(multiplier=multiplier).to(device=calc_device, dtype=work_dtype)
        rank = in_features // block_size
        if block_share:
            rotation = rotation.repeat(rank, 1, 1)
        if model_weight.dim() != 4:
            rotation = rotation.transpose(-1, -2)

        flat_weight = model_weight.reshape(model_weight.shape[0], in_features)
        reshaped_weight = flat_weight.reshape(flat_weight.shape[0], rank, block_size)
        rotated_weight = torch.einsum("ork,rkc->orc", reshaped_weight, rotation).reshape_as(flat_weight)
        rotated_weight = rotated_weight.reshape_as(model_weight)

        if dora_oft_scale_key in lora_weight_keys:
            dora_scale = lora_sd[dora_oft_scale_key].to(device=calc_device, dtype=work_dtype)
            if initial_norm_key in lora_weight_keys:
                norm = lora_sd[initial_norm_key].to(device=calc_device, dtype=work_dtype)
            elif model_weight.dim() == 4:
                norm = torch.norm(model_weight.reshape(model_weight.shape[0], -1), dim=1).reshape(model_weight.shape[0], 1, 1, 1)
            else:
                norm = torch.norm(model_weight.reshape(model_weight.shape[0], -1), dim=1, keepdim=True)
            eps = 1e-12 if work_dtype in (torch.float32, torch.float64) else 1e-6
            factor_shape = (model_weight.shape[0], *([1] * (model_weight.dim() - 1)))
            factor = dora_scale.reshape(factor_shape) / norm.clamp_min(eps).reshape(factor_shape)
            model_weight = factor * rotated_weight
        else:
            model_weight = rotated_weight

        consumed_keys = [
            oft_key,
            alpha_key,
            lora_name + ".oft_R.scaled_oft",
            lora_name + ".oft_block_size_metadata",
            lora_name + ".oft_block_share_metadata",
            lora_name + ".oft_coft_metadata",
            lora_name + ".coft_eps_metadata",
            lora_name + ".scaled_oft_metadata",
            dora_oft_scale_key,
            initial_norm_key,
        ]
        for key in consumed_keys:
            lora_weight_keys.discard(key)
        return model_weight.to(original_dtype) if original_dtype.itemsize == 1 else model_weight

    down_key = lora_name + ".lora_down.weight"
    up_key = lora_name + ".lora_up.weight"
    magnitude_key = lora_name + ".lora_magnitude_vector.weight"
    if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
        return model_weight

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(torch.float16)

    down_weight = lora_sd[down_key].to(calc_device)
    up_weight = lora_sd[up_key].to(calc_device)
    if original_dtype.itemsize == 1:
        down_weight = down_weight.to(torch.float16)
        up_weight = up_weight.to(torch.float16)

    alpha = lora_sd.get(alpha_key, None)
    if alpha is not None and isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    dim = down_weight.shape[0]
    scale = (alpha if alpha is not None else dim) / dim
    delta_weight = _get_lora_weight_from_tensors(down_weight, up_weight) * (multiplier * scale)

    if magnitude_key in lora_weight_keys:
        weight_norm = _get_dora_weight_norm(model_weight, down_weight, up_weight, multiplier * scale)
        magnitude = lora_sd[magnitude_key].to(calc_device, dtype=weight_norm.dtype)
        model_weight = _apply_dora_weight_merge(model_weight, delta_weight, magnitude, weight_norm)
        lora_weight_keys.discard(magnitude_key)
    else:
        model_weight = model_weight + delta_weight

    for key in (down_key, up_key, alpha_key):
        lora_weight_keys.discard(key)

    return model_weight.to(original_dtype) if original_dtype.itemsize == 1 else model_weight


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    module_class: Optional[Type[object]] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> LoRANetwork:
    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    oft_elements = {}
    has_dora_weights = _parse_bool_network_arg(kwargs.get("use_dora", False))
    has_dora_oft_weights = _parse_bool_network_arg(kwargs.get("use_dora_oft", False))
    has_oft_weights = False
    per_module_kwargs: Dict[str, Dict[str, object]] = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.shape[0]
            modules_dim[lora_name] = dim
            # logger.info(lora_name, value.size(), dim)
        elif "oft_R.weight" in key:
            has_oft_weights = True
            oft_elements[lora_name] = int(value.shape[1])
        elif "lora_magnitude_vector" in key:
            has_dora_weights = True
        elif "dora_scale" in key:
            has_dora_oft_weights = True

    for lora_name, n_elements in oft_elements.items():
        module_state = {key.split(".", 1)[1]: tensor for key, tensor in weights_sd.items() if key.startswith(lora_name + ".")}
        alpha_value = modules_alpha.get(lora_name)
        default_block_size = (
            int(round(float(alpha_value.detach().cpu().item() if isinstance(alpha_value, torch.Tensor) else alpha_value)))
            if alpha_value is not None
            else _solve_oft_block_size(n_elements)
        )
        block_size = _metadata_tensor_to_int(module_state.get("oft_block_size_metadata"), default_block_size)
        modules_dim[lora_name] = max(1, block_size)
        module_kwargs_for_name = per_module_kwargs.setdefault(lora_name, {})
        module_kwargs_for_name["oft_block_size"] = modules_dim[lora_name]
        module_kwargs_for_name["oft_block_share"] = _metadata_tensor_to_bool(
            module_state.get("oft_block_share_metadata"),
            default=bool(int(module_state["oft_R.weight"].shape[0]) == 1),
        )
        module_kwargs_for_name["oft_coft"] = _metadata_tensor_to_bool(module_state.get("oft_coft_metadata"), default=False)
        module_kwargs_for_name["coft_eps"] = _metadata_tensor_to_float(module_state.get("coft_eps_metadata"), 6e-5)
        module_kwargs_for_name["scaled_oft"] = _metadata_tensor_to_bool(
            module_state.get("scaled_oft_metadata"),
            default=f"{lora_name}.oft_R.scaled_oft" in weights_sd,
        )

    if module_class is None:
        if has_dora_oft_weights:
            module_class = DoRAOFTInfModule if for_inference else DoRAOFTModule
        elif has_oft_weights:
            module_class = OFTModule
        elif has_dora_weights:
            module_class = DoRAInfModule if for_inference else DoRAModule
        else:
            module_class = LoRAInfModule if for_inference else LoRAModule

    effective_module_kwargs = dict(module_kwargs or {})
    if has_oft_weights:
        effective_module_kwargs["scaled_oft"] = True
        effective_module_kwargs["per_module_kwargs"] = per_module_kwargs

    network = LoRANetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs=effective_module_kwargs,
    )
    return network
