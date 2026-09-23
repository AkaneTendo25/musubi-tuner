# LoRA network for YuE2 (m-a-p/YuE2-3B): AR and NAR block linears, optional NAR I/O modules

import ast
from dataclasses import dataclass, fields
import logging
import os
import re
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import musubi_tuner.networks.lora as lora
from musubi_tuner.yue2 import yue2_lora_formats as formats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

YUE2_TARGET_REPLACE_MODULES = ["YuE2Block"]
BLOCK_TARGETS = {"attn": ("self_attn.qkv_proj", "self_attn.o_proj"), "mlp": ("mlp.gate_up_proj", "mlp.down_proj")}
IO_MODULES = {"vae2llm": "nar.vae2llm", "llm2vae": "nar.llm2vae"}
TIME_MODULES = ("nar.time_embedder.mlp.0", "nar.time_embedder.mlp.2")
BRANCHES = ("ar", "nar")
LAYOUTS = ("split", "fused")
TRAIN_IO_MODES = ("none", "lora", "full")

# network args understood by the generic musubi code path (everything else must be a YuE2NetworkOptions field)
_GENERIC_KWARGS = {"exclude_patterns", "include_patterns", "rank_dropout", "module_dropout", "loraplus_lr_ratio", "verbose"}
_OPTION_ALIASES = {"lora_layout": "layout"}


def _str_list(value) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    text = str(value).strip()
    if text.startswith("["):
        return tuple(str(v).strip() for v in ast.literal_eval(text))
    return tuple(v.strip() for v in re.split(r"[,|+]", text) if v.strip())


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no", "none", ""):
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _opt_float(value) -> Optional[float]:
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none")):
        return None
    return float(value)


def _opt_int(value) -> Optional[int]:
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none")):
        return None
    return int(value)


def _pattern_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return list(value)


@dataclass
class YuE2NetworkOptions:
    branches: tuple[str, ...] = ("nar",)
    layout: str = "split"  # split | fused | auto (from weights)
    targets: tuple[str, ...] = ("attn", "mlp")
    ar_dim: Optional[int] = None
    ar_alpha: Optional[float] = None
    nar_dim: Optional[int] = None
    nar_alpha: Optional[float] = None
    train_io: str = "none"  # none | lora | full
    io_modules: tuple[str, ...] = ("vae2llm", "llm2vae")
    include_time_embedder: bool = False
    io_dim: Optional[int] = None
    io_alpha: Optional[float] = None
    ar_lr_ratio: float = 1.0
    io_lr: Optional[float] = None

    def __post_init__(self):
        self.branches = tuple(self.branches)
        self.targets = tuple(self.targets)
        self.io_modules = tuple(self.io_modules)
        if not self.branches or set(self.branches) - set(BRANCHES):
            raise ValueError(f"YuE2 LoRA branches must be a non-empty subset of {BRANCHES}, got {self.branches}")
        if self.layout not in LAYOUTS + ("auto",):
            raise ValueError(f"lora_layout must be one of {LAYOUTS}, got {self.layout!r}")
        if not self.targets or set(self.targets) - set(BLOCK_TARGETS):
            raise ValueError(f"YuE2 LoRA targets must be a non-empty subset of {tuple(BLOCK_TARGETS)}, got {self.targets}")
        if self.train_io not in TRAIN_IO_MODES:
            raise ValueError(f"train_io must be one of {TRAIN_IO_MODES}, got {self.train_io!r}")
        if set(self.io_modules) - set(IO_MODULES):
            raise ValueError(f"io_modules must be a subset of {tuple(IO_MODULES)}, got {self.io_modules}")
        if self.train_io != "none" and "nar" not in self.branches:
            raise ValueError("train_io targets NAR modules and needs the nar branch")
        if self.include_time_embedder and self.train_io == "none":
            raise ValueError("include_time_embedder needs train_io=lora or train_io=full")
        if self.ar_lr_ratio < 0:
            raise ValueError("ar_lr_ratio must be >= 0")

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "YuE2NetworkOptions":
        """Parse (and pop) the YuE2 network args; values may be strings as they come from ``--network_args``."""
        parsers: Dict[str, Callable] = {
            "branches": _str_list,
            "layout": str,
            "targets": _str_list,
            "ar_dim": _opt_int,
            "ar_alpha": _opt_float,
            "nar_dim": _opt_int,
            "nar_alpha": _opt_float,
            "train_io": str,
            "io_modules": _str_list,
            "include_time_embedder": _bool,
            "io_dim": _opt_int,
            "io_alpha": _opt_float,
            "ar_lr_ratio": float,
            "io_lr": _opt_float,
        }
        values = {}
        for key in list(kwargs):
            name = _OPTION_ALIASES.get(key, key)
            if name in parsers:
                if name in values:
                    raise ValueError(f"YuE2 network arg {name} given twice")
                values[name] = parsers[name](kwargs.pop(key))
        unknown = sorted(k for k in kwargs if k not in _GENERIC_KWARGS)
        if unknown:
            known = sorted(set(_OPTION_ALIASES) | {f.name for f in fields(cls)} | _GENERIC_KWARGS)
            raise ValueError(f"unknown YuE2 network args {unknown}; known: {known}")
        return cls(**values)

    def io_paths(self) -> list[str]:
        if self.train_io == "none":
            return []
        paths = [IO_MODULES[m] for m in self.io_modules]
        if self.include_time_embedder:
            paths += list(TIME_MODULES)
        return paths


def build_include_pattern(opts: YuE2NetworkOptions, user_excludes: list[str]) -> str:
    """One include pattern for the default block targets; user excludes become a negative lookahead."""
    targets = "|".join(re.escape(t) for key in opts.targets for t in BLOCK_TARGETS[key])
    branches = "|".join(opts.branches)
    core = rf"(?:{branches})\.blocks\.\d+\.(?:{targets})"
    return rf"(?!(?:{'|'.join(user_excludes)})$){core}" if user_excludes else core


def _branch_of_name(lora_name: str) -> str:
    if lora_name.startswith("lora_unet_ar_blocks_"):
        return "ar"
    if lora_name.startswith("lora_unet_nar_blocks_"):
        return "nar"
    return "io"


class YuE2LoRAModule(lora.LoRAModule):
    """``LoRAModule`` with an exact split merge and a single-matmul split forward.

    The base module is never registered (``org_module_ref``), so the network state dict never holds base weights.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
    ):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, dropout, rank_dropout, module_dropout, split_dims)
        self.org_module_ref = [org_module]
        del self.org_module

    def apply_to(self):
        org_module = self.org_module_ref[0]
        self.org_forward = org_module.forward
        org_module.forward = self.forward

    def lora_pairs(self) -> list[tuple[nn.Linear, nn.Linear]]:
        if self.split_dims is None:
            return [(self.lora_down, self.lora_up)]
        return list(zip(self.lora_down, self.lora_up))

    def forward(self, x):
        if self.split_dims is None:
            return super().forward(x)
        if not self.enabled:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lora_input = self._lora_input(x)
        down_weight = torch.cat([down.weight for down in self.lora_down], dim=0)
        lx = F.linear(lora_input, down_weight)

        if self.dropout is not None and self.training:
            lx = F.dropout(lx, p=self.dropout)

        scale = self.scale
        if self.rank_dropout is not None and self.training:
            # one mask per sample over the concatenated ranks of all pieces
            mask = torch.rand((lx.size(0), lx.size(-1)), device=lx.device) > self.rank_dropout
            mask = mask.view(lx.size(0), *([1] * (lx.dim() - 2)), lx.size(-1))
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))

        lxs = lx.split(self.lora_dim, dim=-1)
        out = torch.cat([up(piece) for up, piece in zip(self.lora_up, lxs)], dim=-1)
        return self._match_org_dtype(org_forwarded + out * self.multiplier * scale, org_forwarded)

    def delta_weight(self, multiplier=None, device=None) -> torch.Tensor:
        """fp32 ``multiplier * scale * cat_rows(up_i @ down_i)``."""
        multiplier = self.multiplier if multiplier is None else multiplier
        rows = [up.weight.to(device, torch.float) @ down.weight.to(device, torch.float) for down, up in self.lora_pairs()]
        return torch.cat(rows, dim=0) * (self.scale * multiplier)

    def get_weight(self, multiplier=None):
        return self.delta_weight(multiplier)

    def merge_to(self, sd, dtype, device, non_blocking=False):
        """``W += m * scale * ΔW`` from ``sd`` (fused ``lora_down.weight`` or split ``lora_down.{i}.weight`` keys)."""
        org_module = self.org_module_ref[0]
        weight = org_module.weight
        if not weight.is_floating_point() or weight.element_size() == 1:
            raise ValueError(f"{self.lora_name}: cannot merge into a quantized base weight ({weight.dtype})")
        org_dtype, org_device = weight.dtype, weight.device
        device = org_device if device is None else device
        dtype = org_dtype if dtype is None else dtype

        if "lora_down.weight" in sd:
            pairs = [(sd["lora_down.weight"], sd["lora_up.weight"])]
        else:
            n = len([k for k in sd if k.startswith("lora_down.")])
            pairs = [(sd[f"lora_down.{i}.weight"], sd[f"lora_up.{i}.weight"]) for i in range(n)]
        if not pairs:
            raise ValueError(f"{self.lora_name}: no LoRA weights to merge")
        rank = pairs[0][0].shape[0]
        scale = float(sd["alpha"]) / rank if "alpha" in sd else self.scale
        rows = [
            up.to(device, torch.float, non_blocking=non_blocking) @ down.to(device, torch.float, non_blocking=non_blocking)
            for down, up in pairs
        ]
        delta = torch.cat(rows, dim=0)
        if delta.shape != weight.shape:
            raise ValueError(f"{self.lora_name}: LoRA delta {tuple(delta.shape)} does not match weight {tuple(weight.shape)}")
        merged = weight.detach().to(device, torch.float) + self.multiplier * scale * delta
        org_module.weight.data = merged.to(org_device, dtype)


class YuE2LoRAInfModule(YuE2LoRAModule):
    """Inference variant (no dropout), API of ``lora.LoRAInfModule``."""

    def __init__(self, lora_name, org_module: nn.Module, multiplier=1.0, lora_dim=4, alpha=1, split_dims=None, **kwargs):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, split_dims=split_dims)
        self.network = None

    def set_network(self, network):
        self.network = network

    def default_forward(self, x):
        return self.forward(x)


class YuE2DiffModule(nn.Module):
    """Full-rank delta (``diff`` [out, in], ``diff_b`` [out]) on a NAR I/O Linear, trained and saved as network params.

    Holds the base only as ``org_module_ref`` (never registered), so ``state_dict()`` is exactly ``diff``/``diff_b`` and
    ``requires_grad_`` never reaches a (possibly int8) base weight.
    """

    def __init__(self, lora_name, org_module: nn.Linear, multiplier=1.0, with_bias: bool = True, **kwargs):
        super().__init__()
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.enabled = True
        self.lora_dim = 0
        self.split_dims = None
        self.org_module_ref = [org_module]
        if with_bias and getattr(org_module, "bias", None) is None:
            raise ValueError(f"{lora_name}: diff_b given for a Linear without bias")
        self.diff = nn.Parameter(torch.zeros(org_module.out_features, org_module.in_features, dtype=torch.float32))
        if with_bias:
            self.diff_b = nn.Parameter(torch.zeros(org_module.out_features, dtype=torch.float32))
        else:
            self.register_parameter("diff_b", None)

    def apply_to(self):
        org_module = self.org_module_ref[0]
        self.org_forward = org_module.forward
        org_module.forward = self.forward

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        org_forwarded = self.org_forward(x)
        inp = x
        if x.is_floating_point() and x.dtype != self.diff.dtype:
            try:
                autocast = torch.is_autocast_enabled(x.device.type)
            except TypeError:
                autocast = torch.is_autocast_enabled()
            if not autocast:
                inp = x.to(self.diff.dtype)
        bias = None if self.diff_b is None else self.diff_b * self.multiplier
        delta = F.linear(inp, self.diff * self.multiplier, bias)
        out = org_forwarded + delta
        return out.to(org_forwarded.dtype) if out.dtype != org_forwarded.dtype else out

    def delta_weight(self, multiplier=None, device=None) -> torch.Tensor:
        multiplier = self.multiplier if multiplier is None else multiplier
        return self.diff.to(device, torch.float) * multiplier

    def get_weight(self, multiplier=None):
        return self.delta_weight(multiplier)

    def merge_to(self, sd, dtype, device, non_blocking=False):
        org_module = self.org_module_ref[0]
        weight = org_module.weight
        if not weight.is_floating_point() or weight.element_size() == 1:
            raise ValueError(f"{self.lora_name}: cannot merge into a quantized base weight ({weight.dtype})")
        org_dtype, org_device = weight.dtype, weight.device
        device = org_device if device is None else device
        dtype = org_dtype if dtype is None else dtype
        diff = sd["diff"].to(device, torch.float)
        org_module.weight.data = (weight.detach().to(device, torch.float) + self.multiplier * diff).to(org_device, dtype)
        if "diff_b" in sd:
            bias = org_module.bias
            merged = bias.detach().to(device, torch.float) + self.multiplier * sd["diff_b"].to(device, torch.float)
            org_module.bias.data = merged.to(org_device, dtype)


def _split_dims(sub: str, org_module: nn.Module, config) -> Optional[List[int]]:
    if sub not in formats.SPLIT_ROLES:
        return None
    dims = formats.split_dims_for(sub, config)
    if sum(dims) != org_module.out_features:
        raise ValueError(f"{sub}: split dims {dims} do not match out_features {org_module.out_features}")
    return dims


def _module_factory(opts: YuE2NetworkOptions, split_map: Optional[Dict[str, List[int]]], config, for_inference: bool) -> Callable:
    """Module class for ``LoRANetwork.create_modules``: per-branch rank/alpha (training) and split dims."""
    module_class = YuE2LoRAInfModule if for_inference else YuE2LoRAModule

    def factory(lora_name, org_module, multiplier, dim, alpha, dropout=None, rank_dropout=None, module_dropout=None, **kwargs):
        path = formats.path_of(lora_name)
        sub = path.split(".", 3)[3]
        if split_map is not None:
            split_dims = split_map.get(lora_name)
        else:
            branch = _branch_of_name(lora_name)
            dim = (opts.ar_dim if branch == "ar" else opts.nar_dim) or dim
            branch_alpha = opts.ar_alpha if branch == "ar" else opts.nar_alpha
            alpha = branch_alpha if branch_alpha is not None else alpha
            split_dims = _split_dims(sub, org_module, config) if opts.layout == "split" else None
        return module_class(
            lora_name,
            org_module,
            multiplier,
            dim,
            alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            split_dims=split_dims,
        )

    return factory


class YuE2LoRANetwork(lora.LoRANetwork):
    def __init__(
        self,
        unet: nn.Module,
        opts: YuE2NetworkOptions,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: bool = False,
        *,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, float]] = None,
        split_map: Optional[Dict[str, List[int]]] = None,
        io_spec: Optional[Dict[str, dict]] = None,
        for_inference: bool = False,
    ):
        """``io_spec`` (from weights): ``{lora_name: {"kind": "lora"|"diff", "with_bias": bool, "dim", "alpha"}}``;
        None builds the I/O modules from ``opts.train_io``."""
        config = getattr(unet, "config", None)
        factory = _module_factory(opts, split_map if modules_dim is not None else None, config, for_inference)
        super().__init__(
            YUE2_TARGET_REPLACE_MODULES,
            "lora_unet",
            None,
            unet,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            module_class=factory,
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
            verbose=verbose,
        )
        self.opts = opts
        self.for_inference = for_inference
        self._config = config
        self.unet_loras += self._create_io_modules(unet, io_spec)
        names = set()
        for module in self.unet_loras:
            assert module.lora_name not in names, f"duplicated lora name: {module.lora_name}"
            names.add(module.lora_name)
        self.layout = self._detect_layout()

    def _create_io_modules(self, unet, io_spec) -> list:
        modules = []
        if io_spec is None:
            kind = "diff" if self.opts.train_io == "full" else "lora"
            dim = self.opts.io_dim or self.opts.nar_dim or self.lora_dim
            alpha = self.opts.io_alpha
            if alpha is None:
                alpha = self.opts.nar_alpha if self.opts.nar_alpha is not None else self.alpha
            io_spec = {
                formats.lora_name_of(p): {"kind": kind, "with_bias": True, "dim": dim, "alpha": alpha} for p in self.opts.io_paths()
            }
        for lora_name, spec in io_spec.items():
            org_module = unet.get_submodule(formats.path_of(lora_name))
            if spec["kind"] == "diff":
                with_bias = bool(spec.get("with_bias", True)) and org_module.bias is not None
                modules.append(YuE2DiffModule(lora_name, org_module, self.multiplier, with_bias=with_bias))
                continue
            dim, alpha = spec["dim"], spec["alpha"]
            module_class = YuE2LoRAInfModule if self.for_inference else YuE2LoRAModule
            kwargs = {}
            if not self.for_inference:
                kwargs = dict(dropout=self.dropout, rank_dropout=self.rank_dropout, module_dropout=self.module_dropout)
            modules.append(module_class(lora_name, org_module, self.multiplier, dim, alpha, **kwargs))
        if modules:
            logger.info(f"create LoRA for YuE2 I/O modules: {len(modules)} ({', '.join(m.lora_name for m in modules)})")
        return modules

    def _detect_layout(self) -> str:
        kinds = {
            "split" if m.split_dims is not None else "fused"
            for m in self.unet_loras
            if isinstance(m, YuE2LoRAModule) and formats.path_of(m.lora_name).split(".", 3)[-1] in formats.SPLIT_ROLES
        }
        return kinds.pop() if len(kinds) == 1 else ("mixed" if kinds else "none")

    @staticmethod
    def branch_of(lora_name: str) -> str:
        """``ar`` | ``nar`` | ``io``"""
        return _branch_of_name(lora_name)

    def modules_of_branch(self, branch: str) -> list:
        return [m for m in self.unet_loras if _branch_of_name(m.lora_name) == branch]

    def set_multiplier_by_branch(self, ar: float = 1.0, nar: float = 1.0, io: Optional[float] = None):
        factors = {"ar": ar, "nar": nar, "io": nar if io is None else io}
        for module in self.unet_loras:
            module.multiplier = self.multiplier * factors[_branch_of_name(module.lora_name)]

    def prepare_optimizer_params(self, unet_lr: float = 1e-4, **kwargs):
        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []
        frozen = []
        io_lr = self.opts.io_lr if self.opts.io_lr is not None else unet_lr
        ar_lr = unet_lr * self.opts.ar_lr_ratio if unet_lr is not None else None
        for branch, lr in (("nar", unet_lr), ("ar", ar_lr), ("io", io_lr)):
            modules = self.modules_of_branch(branch)
            if not modules:
                continue
            groups = {"lora": [], "plus": []}
            for module in modules:
                for name, param in module.named_parameters():
                    if self.loraplus_lr_ratio is not None and "lora_up" in name:
                        groups["plus"].append(param)
                    else:
                        groups["lora"].append(param)
            if self.loraplus_lr_ratio is not None and not groups["plus"]:
                logger.info(f"LoRA+ has no lora_up parameters in the {branch} group")
            for key, params in groups.items():
                if not params:
                    continue
                group_lr = lr * self.loraplus_lr_ratio if (key == "plus" and lr is not None) else lr
                if not group_lr:
                    logger.info(f"YuE2 LoRA: learning rate of the {branch} {key} group is {group_lr}; its parameters are frozen")
                    for p in params:
                        p.requires_grad_(False)
                    frozen.extend(params)
                    continue
                all_params.append({"params": params, "lr": group_lr})
                lr_descriptions.append(branch + (" plus" if key == "plus" else ""))
        self._frozen_params = frozen
        return all_params, lr_descriptions

    def prepare_grad_etc(self, unet):
        # the base re-enables every parameter; zero-lr groups must stay out of backward and grad clipping
        super().prepare_grad_etc(unet)
        for p in getattr(self, "_frozen_params", []):
            p.requires_grad_(False)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def enable_gradient_checkpointing(self):
        pass

    def adapt_state_dict(self, native: "formats.NativeLoRA") -> Dict[str, torch.Tensor]:
        """A ``NativeLoRA`` in this network's own layout, ranks and alphas (scales baked into ``up``, exact).

        Fused -> split is exact; split -> fused needs the fused rank to equal the sum of the piece ranks; a diff can
        only load into a diff module. Modules the network does not have are dropped with a warning.
        """
        sd = {}
        own = {m.lora_name: m for m in self.unet_loras}
        dropped = []
        for path, md in native.modules.items():
            name = formats.lora_name_of(path)
            module = own.get(name)
            if module is None:
                dropped.append(name)
                continue
            if md.kind == "full":
                raise ValueError(f"{path} holds full replacement weights; convert with base_io (read_base_io) first")
            if isinstance(module, YuE2DiffModule):
                sd[name + ".diff"] = md.weight_delta()
                if module.diff_b is not None:
                    b = md.bias_delta()
                    sd[name + ".diff_b"] = b if b is not None else torch.zeros_like(module.diff_b, dtype=torch.float32)
                elif md.bias_delta() is not None and md.bias_delta().abs().max() > 0:
                    raise ValueError(f"{path}: the file has a bias delta but the network module has none")
                continue
            if md.kind != "lora":
                raise ValueError(f"{path}: a diff cannot be loaded into a rank-{module.lora_dim} LoRA module (use train_io=full)")
            if module.split_dims is not None:
                if not md.is_split:
                    md = formats.fused_module_to_split(md, module.split_dims, rank=module.lora_dim)
                if len(md.downs) != len(module.split_dims) or md.rank != module.lora_dim:
                    raise ValueError(
                        f"{path}: {len(md.downs)} pieces of rank {md.rank} do not fit the network's "
                        f"{len(module.split_dims)} pieces of rank {module.lora_dim}"
                    )
            else:
                if md.is_split:
                    md = formats.split_module_to_fused(md)
                if md.rank != module.lora_dim:
                    raise ValueError(
                        f"{path}: rank {md.rank} does not match the network rank {module.lora_dim} "
                        "(a split file into a fused network needs rank = sum of the piece ranks)"
                    )
            factor = md.scale / module.scale
            for i, ((down, up), (d_mod, u_mod)) in enumerate(zip(zip(md.downs, md.ups), module.lora_pairs())):
                suffix = "" if module.split_dims is None else f".{i}"
                sd[f"{name}.lora_down{suffix}.weight"] = down.float()
                sd[f"{name}.lora_up{suffix}.weight"] = up.float() * factor
            sd[name + ".alpha"] = module.alpha.detach().clone()
        if dropped:
            logger.warning(f"YuE2 LoRA: {len(dropped)} modules of the weights are not in this network: {dropped[:8]}")
        return sd

    def base_io(self) -> Dict[str, torch.Tensor]:
        """fp32 base weights of this network's I/O modules (native names), for Mothersuperior full I/O weights."""
        out = {}
        for module in self.unet_loras:
            if _branch_of_name(module.lora_name) != "io":
                continue
            org_module = module.org_module_ref[0]
            path = formats.path_of(module.lora_name)
            for name in ("weight", "bias"):
                t = getattr(org_module, name, None)
                if t is not None and t.is_floating_point() and t.element_size() > 1:
                    out[f"{path}.{name}"] = t.detach().float().cpu()
        return out

    def load_weights(self, file, base_io=None):
        """Load any known format into this network (layout, rank and alpha adapted exactly, see ``adapt_state_dict``).
        Full I/O weights become diffs against ``base_io`` (default: the base weights of the network's I/O modules)."""
        weights_sd, metadata = formats.load_lora_file(file)
        if base_io is None:
            base_io = self.base_io()
        native = formats.to_native(weights_sd, metadata, base_io=base_io, config=getattr(self, "_config", None))
        sd = self.adapt_state_dict(native)
        info = self.load_state_dict(sd, False)
        if info.missing_keys:
            logger.warning(
                f"YuE2 LoRA: {len(info.missing_keys)} network keys not in {os.path.basename(file)}: {info.missing_keys[:8]}"
            )
        return info

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None, non_blocking=False):
        if not any(k.startswith(formats.NATIVE_PREFIX) for k in weights_sd):
            weights_sd = convert_lora_state_dict(weights_sd)
        for module in self.text_encoder_loras + self.unet_loras:
            prefix = module.lora_name + "."
            sd_for_lora = {k[len(prefix) :]: v for k, v in weights_sd.items() if k.startswith(prefix)}
            if not sd_for_lora:
                logger.info(f"no weight for {module.lora_name}")
                continue
            module.merge_to(sd_for_lora, dtype, device, non_blocking)
        logger.info("weights are merged")

    def apply_max_norm_regularization(self, max_norm_value, device):
        """Split-aware max-norm: per module ΔW = scale * cat_rows(up_i @ down_i); every pair of a module is
        rescaled by the same sqrt(ratio); diff modules are normed and rescaled on ``diff``."""
        keys_scaled = 0
        norms = []
        with torch.no_grad():
            for module in self.unet_loras:
                if isinstance(module, YuE2DiffModule):
                    updown = module.diff.to(device, torch.float)
                else:
                    updown = module.delta_weight(multiplier=1.0, device=device)
                raw_norm = updown.norm().item()  # before any in-place rescale (``diff.to`` may alias the parameter)
                norm = updown.norm().clamp(min=max_norm_value / 2)
                desired = torch.clamp(norm, max=max_norm_value)
                ratio = (desired.cpu() / norm.cpu()).item()
                if ratio != 1:
                    keys_scaled += 1
                    if isinstance(module, YuE2DiffModule):
                        module.diff.mul_(ratio)
                    else:
                        sqrt_ratio = ratio**0.5
                        for down, up in module.lora_pairs():
                            down.weight.mul_(sqrt_ratio)
                            up.weight.mul_(sqrt_ratio)
                norms.append(raw_norm * ratio)
        if not norms:
            return 0, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)


def convert_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    *,
    metadata: Optional[Dict[str, str]] = None,
    target: str = "native",
    config=None,
    base_io: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Any known YuE2 LoRA format -> musubi native keys.

    ``target="native"`` keeps the layout (split stays split) and is what ``create_arch_network_from_weights`` and
    ``attach_lora_weights`` consume; ``target="native_fused"`` is the merge-input dict for ``load_yue2_model``
    (block-diagonal fused block modules, I/O modules as diffs). Mothersuperior full I/O weights need ``base_io``
    (``yue2_checkpoint.read_base_io(dit)``). Pass ``metadata`` for ``yue2-lora-v1`` files, whose scale lives there.
    """
    native = formats.to_native(weights_sd, metadata, base_io=base_io, config=config)
    if target == "native":
        return formats.native_state_dict(native)
    if target == "native_fused":
        return formats.native_fused_state_dict(native)
    raise ValueError(f"target must be native or native_fused, got {target!r}")


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
    kwargs = dict(kwargs)
    opts = YuE2NetworkOptions.from_kwargs(kwargs)
    user_excludes = _pattern_list(kwargs.get("exclude_patterns"))
    user_includes = _pattern_list(kwargs.get("include_patterns"))
    rank_dropout = _opt_float(kwargs.get("rank_dropout"))
    module_dropout = _opt_float(kwargs.get("module_dropout"))
    verbose = str(kwargs.get("verbose", "False")) == "True"
    network = YuE2LoRANetwork(
        unet,
        opts,
        multiplier,
        network_dim if network_dim is not None else 4,
        network_alpha if network_alpha is not None else 1.0,
        neuron_dropout,
        rank_dropout,
        module_dropout,
        exclude_patterns=[".*"],
        include_patterns=[build_include_pattern(opts, user_excludes)] + user_includes,
        verbose=verbose,
    )
    loraplus_lr_ratio = _opt_float(kwargs.get("loraplus_lr_ratio"))
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)
    if not network.unet_loras:
        raise RuntimeError("YuE2 LoRA found zero target modules; check branches/targets and the include/exclude patterns")
    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> YuE2LoRANetwork:
    """Network whose modules, layout, ranks and alphas come from the weights (any known format).

    ``weights_sd`` in native keys is used as is; other formats are converted (pass ``metadata=`` / ``base_io=`` in
    kwargs when needed). Keys that match no module of ``unet`` raise.
    """
    if not any(k.startswith(formats.NATIVE_PREFIX) for k in weights_sd):
        weights_sd = convert_lora_state_dict(weights_sd, metadata=kwargs.get("metadata"), base_io=kwargs.get("base_io"))
    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, torch.Tensor] = {}
    split_pieces: Dict[str, Dict[int, int]] = {}
    io_spec: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        name, _, part = key.partition(".")
        if part == "alpha":
            modules_alpha[name] = value
        elif part == "lora_down.weight":
            modules_dim[name] = value.shape[0]
        elif part.startswith("lora_up.") and part != "lora_up.weight":
            split_pieces.setdefault(name, {})[int(part.split(".")[1])] = value.shape[0]
        elif part.startswith("lora_down."):
            modules_dim[name] = value.shape[0]
        elif part in ("diff", "diff_b"):
            spec = io_spec.setdefault(name, {"kind": "diff", "with_bias": False})
            spec["with_bias"] = spec["with_bias"] or part == "diff_b"
        elif part != "lora_up.weight":
            raise ValueError(f"unexpected YuE2 LoRA key: {key}")
    split_map = {name: [pieces[i] for i in range(len(pieces))] for name, pieces in split_pieces.items()}
    for name in list(modules_dim):
        if name not in modules_alpha:
            modules_alpha[name] = torch.tensor(float(modules_dim[name]))
        if _branch_of_name(name) == "io":
            io_spec[name] = {"kind": "lora", "dim": modules_dim[name], "alpha": modules_alpha[name]}
    for name in io_spec:
        if _branch_of_name(name) != "io":
            raise ValueError(f"diff weights are only supported on the NAR I/O modules: {name}")
    block_dim = {k: v for k, v in modules_dim.items() if _branch_of_name(k) != "io"}
    network = YuE2LoRANetwork(
        unet,
        YuE2NetworkOptions(branches=BRANCHES, layout="auto"),
        multiplier,
        exclude_patterns=None,
        include_patterns=None,
        modules_dim=block_dim,
        modules_alpha=modules_alpha,
        split_map=split_map,
        io_spec=io_spec,
        for_inference=for_inference,
    )
    created = {m.lora_name for m in network.unet_loras}
    missing = sorted(set(modules_dim) - created)
    if missing:
        raise ValueError(f"YuE2 LoRA weights have {len(missing)} modules that are not in the model: {missing[:8]}")
    return network
