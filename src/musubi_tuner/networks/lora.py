# LoRA network module: currently conv2d is not fully supported
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import ast
import math
import os
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Type, Union
from transformers import CLIPTextModel
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HUNYUAN_TARGET_REPLACE_MODULES = ["MMDoubleStreamBlock", "MMSingleStreamBlock"]

#: ``.lora_down.weight`` or the split_dims form ``.lora_down.<i>.weight`` at the end of a state-dict key.
_DOWN_WEIGHT_KEY = re.compile(r"\.lora_down(?:\.\d+)?\.weight$")

#: ``nora`` network argument (Normalized Low-Rank Adaptation, Kang et al. 2026, arXiv:2608.31036).
#: ``off`` is plain LoRA, ``forward`` re-normalises ``lora_down`` on every forward with gradients
#: flowing through the normalisation, ``init`` normalises once after initialisation.
NORA_MODES = ("off", "forward", "init")
#: ``init`` network argument: ``default`` is Kaiming-uniform ``lora_down`` / zero ``lora_up``;
#: ``bimi`` tiles ``r x r`` identity blocks along ``lora_down`` (columns orthonormal within each block).
LORA_INITS = ("default", "bimi")


def normalize_lora_down(weight: torch.Tensor) -> torch.Tensor:
    """Unit-norm columns of a ``lora_down`` weight along the rank axis (dim 0).

    ``weight`` is ``(rank, in)`` for Linear and ``(rank, in, *kernel)`` for Conv; in both
    cases every input position holds one rank-length vector that is scaled to unit norm, so
    the conv case is the per-column normalisation over the flattened input dims. Uses the
    ``torch.nn.functional.normalize`` epsilon (1e-12), i.e. an all-zero column stays zero.
    """
    return torch.nn.functional.normalize(weight, p=2, dim=0)


def bimi_init_(weight: torch.Tensor) -> None:
    """Block-identity initialisation: ``A[:, k*r:(k+1)*r] = I_r`` over the flattened input dims.

    The trailing block is a truncated identity when the flattened input size is not a
    multiple of the rank; every column still holds exactly one 1, so the columns are
    unit-norm and, within a block, mutually orthogonal.
    """
    rank = weight.shape[0]
    flat = weight.detach().reshape(rank, -1)
    repeats = -(-flat.shape[1] // rank)
    tiled = torch.eye(rank, dtype=weight.dtype, device=weight.device).repeat(1, repeats)[:, : flat.shape[1]]
    with torch.no_grad():
        weight.copy_(tiled.reshape(weight.shape))


def _profile_scope(name: str):
    """A ``torch.profiler.record_function`` label while a profiler collects, else a null context."""
    if getattr(torch.autograd.profiler, "_is_profiler_enabled", False):
        return torch.profiler.record_function(name)
    return nullcontext()


class LoRAModule(torch.nn.Module):
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
        nora: str = "off",
        init: str = "default",
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).

        split_dims is used to mimic the split qkv of multi-head attention.

        nora / init: see ``NORA_MODES`` and ``LORA_INITS``. With ``nora="forward"`` the
        parameter ``lora_down.weight`` is the raw direction; the effective weight the
        forward and the saved adapter use is ``normalize_lora_down`` of it.
        """
        super().__init__()
        self.lora_name = lora_name
        if nora not in NORA_MODES:
            raise ValueError(f"nora must be one of {NORA_MODES}, got {nora!r}")
        if init not in LORA_INITS:
            raise ValueError(f"init must be one of {LORA_INITS}, got {init!r}")
        self.nora = nora

        if org_module.__class__.__name__ in ("Conv2d", "Conv3d"):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
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

        if init == "bimi":
            for down in self._down_modules():
                bimi_init_(down.weight)
        if nora == "init":
            with torch.no_grad():
                for down in self._down_modules():
                    down.weight.copy_(normalize_lora_down(down.weight))

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # for save/load

        # same as microsoft's
        self.multiplier = multiplier
        self.enabled = True
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def _down_modules(self):
        return [self.lora_down] if self.split_dims is None else list(self.lora_down)

    def effective_down_weight(self, down: torch.nn.Module) -> torch.Tensor:
        """The ``lora_down`` weight the delta is computed with: normalised under ``nora=forward``."""
        if self.nora == "forward":
            return normalize_lora_down(down.weight)
        return down.weight

    def _down_forward(self, down: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.nora != "forward":
            return down(x)  # the module call itself, so ``nora=off`` stays bit-identical
        weight = normalize_lora_down(down.weight)
        if isinstance(down, torch.nn.Linear):
            return torch.nn.functional.linear(x, weight, down.bias)
        return down._conv_forward(x, weight, down.bias)

    def export_state_dict(self, destination=None, prefix: str = ""):
        """State dict for the adapter file: a standard LoRA that merges and infers without NoRA.

        Under ``nora=forward`` the saved ``lora_down`` is the normalised weight, so
        ``lora_up @ lora_down_saved`` is the training-time delta up to the save dtype's
        rounding (bf16 moves column norms by about 1e-3, fp32 is exact). ``state_dict``
        itself is left raw, so a resumed training state keeps the parameters the optimizer
        moments were accumulated on.
        """
        sd = self.state_dict(destination=destination, prefix=prefix)
        if self.nora == "forward":
            if self.split_dims is None:
                sd[prefix + "lora_down.weight"] = normalize_lora_down(self.lora_down.weight).detach()
            else:
                for i, down in enumerate(self.lora_down):
                    sd[prefix + f"lora_down.{i}.weight"] = normalize_lora_down(down.weight).detach()
        return sd

    def down_column_norm_deviation(self) -> float:
        """Largest ``| ||column|| - 1 |`` over the raw ``lora_down`` columns; 0 for a NoRA-saved adapter."""
        deviation = 0.0
        for down in self._down_modules():
            norms = down.weight.detach().float().flatten(1).norm(dim=0)
            deviation = max(deviation, float((norms - 1.0).abs().max()))
        return deviation

    def _autocast_enabled_for(self, x):
        if not x.is_floating_point():
            return False
        try:
            return torch.is_autocast_enabled(x.device.type)
        except TypeError:
            return torch.is_autocast_enabled()

    def _lora_input(self, x):
        if self.split_dims is None:
            target_dtype = self.lora_down.weight.dtype
        else:
            target_dtype = self.lora_down[0].weight.dtype
        if x.is_floating_point() and x.dtype != target_dtype and not self._autocast_enabled_for(x):
            return x.to(target_dtype)
        return x

    def _match_org_dtype(self, value, org_forwarded):
        """Round ``value`` to the base output dtype.

        Used to bring the LoRA-augmented output back to ``org_forwarded``'s dtype in the
        autocast-free mixed-dtype regime (e.g. fp32 LoRA on a bf16 base). Callers pass the
        full ``org_forwarded + delta`` sum here so the (possibly higher-precision) delta is
        kept through the addition and the result is rounded only once. A no-op when dtypes
        already match (autocast-on / all-fp32 regimes)."""
        if value.is_floating_point() and org_forwarded.is_floating_point() and value.dtype != org_forwarded.dtype:
            return value.to(org_forwarded.dtype)
        return value

    def _fuse_delta(self, org_forwarded, delta, scale):
        """``org_forwarded + delta * self.multiplier * scale`` without the redundant copies.

        The naive expression materializes three full-size tensors on top of ``delta``
        (``delta*multiplier``, ``*scale``, and the sum) before the block's activation is
        finally produced. At the H3 LoRA sites the full size is ``[S, 2*ffn_dim] = [S, 28672]``
        for ``mlp.fc1`` and ``[S, 3*heads*head_dim] = [S, 21504]`` for ``attn.qkv_proj``, i.e.
        hundreds of MiB per copy for a long packed sequence, so the redundancy dominates the
        allocator traffic of a non-checkpointed block.

        Here the scalar factors are applied one out-of-place multiply (which yields a tensor
        this module exclusively owns) followed by in-place ones, and the base output is then
        accumulated in place. Every element goes through exactly the same elementwise kernels
        in the same order as before, so the result is bit-identical; only the destination
        buffer changes.

        The in-place accumulate is skipped in two cases: when it would have to downcast
        (bf16 delta on an fp32 base output, where ``add_`` is illegal), and when no scalar
        factor produced an owned tensor -- ``lora_up``'s output on a 3-D input is an autograd
        *view* of the underlying matmul result, and writing into a view rebases the graph onto
        ``CopySlices``, which costs a full-size temporary in backward instead of forward.
        """
        owned = False
        for factor in (self.multiplier, scale):
            if factor == 1.0:
                continue  # x * 1.0 is bit-identical to x
            delta = delta.mul_(factor) if owned else delta * factor
            owned = True
        # promote_types (not result_type) keeps this a dtype-only check, which torch.compile
        # can fold as a constant instead of graph-breaking on a torch.* op returning a dtype.
        if owned and delta.is_floating_point() and torch.promote_types(delta.dtype, org_forwarded.dtype) == delta.dtype:
            return delta.add_(org_forwarded)
        return org_forwarded + delta

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        with _profile_scope("h3.lora"):
            return self._lora_forward(x)

    def _lora_forward(self, x):
        base_module = getattr(self.org_forward, "__self__", None)
        if (
            getattr(base_module, "_convrot_lora_fused", False)
            and self.split_dims is None
            and self.dropout is None
            and self.rank_dropout is None
            and self.module_dropout is None
            and (self._autocast_enabled_for(x) or (x.dtype == self.lora_down.weight.dtype == self.lora_up.weight.dtype))
        ):
            from musubi_tuner.modules.convrot_int8_utils import convrot_int8_lora_forward

            return convrot_int8_lora_forward(
                base_module,
                x,
                self.effective_down_weight(self.lora_down),
                self.lora_up.weight,
                self.multiplier * self.scale,
            )
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lora_input = self._lora_input(x)
        if self.split_dims is None:
            lx = self._down_forward(self.lora_down, lora_input)

            # normal dropout
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout
            if self.rank_dropout is not None and self.training:
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

            lx = self.lora_up(lx)

            # Add in the (possibly higher-precision) delta dtype, then round the sum once.
            return self._match_org_dtype(self._fuse_delta(org_forwarded, lx, scale), org_forwarded)
        else:
            lxs = [self._down_forward(lora_down, lora_input) for lora_down in self.lora_down]

            # normal dropout
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
            if self.rank_dropout is not None and self.training:
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

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

            # Add in the (possibly higher-precision) delta dtype, then round the sum once.
            return self._match_org_dtype(self._fuse_delta(org_forwarded, torch.cat(lxs, dim=-1), scale), org_forwarded)


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
        lora_input = self._lora_input(x)
        if self.split_dims is None:
            lx = self.lora_down(lora_input)
            lx = self.lora_up(lx)
            org_forwarded = self.org_forward(x)
            # Add in the (possibly higher-precision) delta dtype, then round the sum once.
            return self._match_org_dtype(self._fuse_delta(org_forwarded, lx, self.scale), org_forwarded)
        else:
            lxs = [lora_down(lora_input) for lora_down in self.lora_down]
            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
            org_forwarded = self.org_forward(x)
            # Add in the (possibly higher-precision) delta dtype, then round the sum once.
            return self._match_org_dtype(self._fuse_delta(org_forwarded, torch.cat(lxs, dim=-1), self.scale), org_forwarded)

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
    **kwargs,
):
    """architecture independent network creation"""
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # TODO generic rank/dim setting with regular expression

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
        module_class = LoRAModule

    # NoRA column normalisation and bimi initialisation. Only forwarded to the module when
    # set, so module classes without these keyword arguments keep working with defaults.
    nora = str(kwargs.get("nora", "off")).lower()
    init = str(kwargs.get("init", "default")).lower()
    if nora not in NORA_MODES:
        raise ValueError(f"network_args nora must be one of {NORA_MODES}, got {nora!r}")
    if init not in LORA_INITS:
        raise ValueError(f"network_args init must be one of {LORA_INITS}, got {init!r}")
    if nora != "off" or init != "default":
        module_kwargs = dict(module_kwargs or {})
        module_kwargs.update(nora=nora, init=init)
        logger.info(f"NoRA: nora={nora}, init={init}")
    if nora != "off" and float(network_alpha) != float(network_dim):
        logger.warning(
            f"NoRA is intended for network_alpha == network_dim (scaling 1); got alpha={network_alpha}, dim={network_dim}. "
            "With unit-norm lora_down columns the alpha/dim factor only rescales lora_up's learning signal."
        )

    # too many arguments ( ^ω^)･･･
    network = LoRANetwork(
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
        module_kwargs=module_kwargs,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
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


class LoRANetwork(torch.nn.Module):
    # only supports U-Net (DiT), Text Encoders are not supported

    def __init__(
        self,
        target_replace_modules: Optional[List[str]],
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
        self.module_kwargs = module_kwargs or {}

        self.loraplus_lr_ratio = None
        # self.loraplus_unet_lr_ratio = None
        # self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
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

                            # filter by name (not used in the current implementation)
                            if filter is not None and filter not in lora_name:
                                continue

                            dim = None
                            alpha = None

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
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or is_conv3d_1x1 or (self.conv_lora_dim is not None):
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                **self.module_kwargs,
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

    def prepare_network(self, args):
        """
        called after the network is created
        """
        pass

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        self._check_loaded_nora_columns(file)
        return info

    def _check_loaded_nora_columns(self, file):
        """Warn when a NoRA run continues from an adapter whose ``lora_down`` columns are not unit-norm.

        Adapters saved by a NoRA run carry normalised columns, so re-normalising under
        ``nora=forward`` reproduces the saved delta exactly (up to the save dtype), and
        ``nora=init`` continues with them unchanged. A plain-LoRA adapter has arbitrary
        column norms: ``forward`` mode would rescale its delta on the first step, ``init``
        mode would train from non-unit directions.
        """
        worst = 0.0
        for lora in self.text_encoder_loras + self.unet_loras:
            if getattr(lora, "nora", "off") != "off":
                worst = max(worst, lora.down_column_norm_deviation())
        if worst > 1e-2:
            logger.warning(
                f"{file}: lora_down columns deviate from unit norm by up to {worst:.3g}; the adapter was not saved by a "
                "NoRA run. nora=forward re-normalises them (its delta changes), nora=init keeps them as loaded."
            )

    def export_state_dict(self):
        """The adapter-file state dict: ``state_dict()`` with NoRA modules' normalised ``lora_down``."""
        sd = self.state_dict()
        for lora in self.text_encoder_loras + self.unet_loras:
            if getattr(lora, "nora", "off") == "forward":
                sd.update(lora.export_state_dict(prefix=lora.lora_name + "."))
        return sd

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

    def prepare_optimizer_params(self, unet_lr: float = 1e-4, **kwargs):
        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
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

                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")

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

    def on_epoch_start(self, unet):
        self.train()

    def on_step_start(self):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def snapshot_weights(self, dtype):
        """Return a CPU copy of the LoRA state dict, detached from live training tensors.

        This is the only part of a save that must run on the training thread: it
        fixes the checkpoint to the current step. Hashing and writing the snapshot
        can then happen anywhere, see ``utils.async_save.write_state_dict_file``.
        """
        from musubi_tuner.utils import async_save

        # Duck-typed consumers borrow this method without export_state_dict; plain state then.
        export = getattr(self, "export_state_dict", None)
        return async_save.snapshot_state_dict(export() if export is not None else self.state_dict(), dtype)

    def save_weights(self, file, dtype, metadata):
        from musubi_tuner.utils import async_save

        async_save.write_state_dict_file(self.snapshot_weights(dtype), file, metadata)

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

        # guard: only supported for LoRA (lora_down/lora_up parameterization)
        if not any("lora_down" in k and "weight" in k for k in state_dict.keys()):
            logger.warning("max_norm_regularization is only supported for LoRA")
            return 0, 0.0, 0.0

        # Under nora=forward the delta uses the normalised lora_down, and scaling the raw
        # parameter would be undone by the normalisation: measure with the effective weight
        # and put the whole correction on lora_up.
        nora_forward = {
            lora.lora_name for lora in self.text_encoder_loras + self.unet_loras if getattr(lora, "nora", "off") == "forward"
        }

        lora_names = []
        for key in state_dict.keys():
            # ``<name>.lora_down.weight`` or, for split_dims, ``<name>.lora_down.<i>.weight``;
            # alpha is one buffer per module either way.
            match = _DOWN_WEIGHT_KEY.search(key)
            if match is not None:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                lora_names.append(key[: match.start()])
                alphakeys.append(key[: match.start()] + ".alpha")

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim
            normalised = lora_names[i] in nora_forward
            if normalised:
                down = normalize_lora_down(down)

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
                if normalised:
                    state_dict[upkeys[i]] *= ratio
                else:
                    state_dict[upkeys[i]] *= sqrt_ratio
                    state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)


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


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(
    target_replace_modules: Optional[List[str]],
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

    if module_class is None:
        module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
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
