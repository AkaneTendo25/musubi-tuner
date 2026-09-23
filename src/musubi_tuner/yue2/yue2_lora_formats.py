"""YuE2 LoRA formats: detection, import to one canonical in-memory form, and export.

Pure tensor/key conversions, no model needed. Canonical form (``NativeLoRA``): one ``ModuleDelta`` per native module
path (``nar.blocks.3.self_attn.qkv_proj``, ``nar.vae2llm``, ...). A LoRA delta holds one ``(down, up)`` pair per
piece; fused modules have one piece, split ``qkv_proj`` / ``gate_up_proj`` modules have one piece per role (q, k, v /
gate, up) stacked by output rows. All pieces of a module share one rank and one ``alpha`` (scale = alpha / rank), the
musubi ``LoRAModule(split_dims=...)`` convention.

Known formats (``detect_format``):

* ``native``: musubi keys ``lora_unet_{ar,nar}_blocks_N_<module>`` (``lora_down.weight`` or ``lora_down.{i}.weight``),
  ``lora_unet_nar_{vae2llm,...}`` (LoRA, or ``.diff``/``.diff_b``).
* ``comfy``: ComfyUI native (``diffusion_model.model.layers.N.*`` NAR, ``text_encoders.model.layers.N.*`` AR, fused
  q|k|v and gate|up, ``lora_down/lora_up`` or ``lora_A/lora_B``, optional ``.alpha``, I/O ``.diff/.diff_b``); written by
  Starnodes, ai-toolkit (``lora_A/B``, no alpha) and Mothersuperior ``*_comfyui`` (block-diagonal, no alpha).
* ``aitk``: ai-toolkit in-memory keys ``transformer.{ar,nar}.model.layers.N.*`` (same layout as ``comfy``).
* ``hf``: Hugging Face layer names ``model.layers.N.(nar_)?(self_attn|mlp).<proj>`` with separate projections, top level
  I/O names; ``yue2-lora-v1`` (Starnodes / Studio, scale from metadata ``alpha``), ``yue2-artist-ar-v1`` (Studio).
* ``fl``: the same names with metadata ``format=fl-yue2-lora-v1`` (one file per branch, I/O as ``.diff/.diff_b``).
* ``ms_safetensors``: Mothersuperior ``layers.N.(nar_)?(self_attn|mlp).<proj>.lora_{A,B}`` plus full ``vae2llm``/
  ``llm2vae`` weights; ``ms_pt``: the ``.pt`` dict ``{"lora": [A0, B0, ...], "rank", "io"?}`` (layer-major, per layer
  q, k, v, o, gate, up, down).

Scale rules: a per-key ``.alpha`` wins; ``yue2-lora-v1`` falls back to metadata ``alpha``; otherwise alpha = rank
(scale 1, as the ComfyUI, fl-yue2-lora-v1 and Mothersuperior loaders apply them). Q/K/V and gate/up pieces are
always grouped by role name, never by key order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import re
from typing import Mapping, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

NATIVE_PREFIX = "lora_unet_"
BLOCK_SUBMODULES = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
IO_SUBMODULES = ("vae2llm", "llm2vae", "time_embedder.mlp.0", "time_embedder.mlp.2")
SPLIT_ROLES = {"self_attn.qkv_proj": ("q", "k", "v"), "mlp.gate_up_proj": ("gate", "up")}

# (block, projection) in HF / Comfy naming -> (native submodule, role)
_PROJ_MAP = {
    ("self_attn", "q_proj"): ("self_attn.qkv_proj", "q"),
    ("self_attn", "k_proj"): ("self_attn.qkv_proj", "k"),
    ("self_attn", "v_proj"): ("self_attn.qkv_proj", "v"),
    ("self_attn", "qkv_proj"): ("self_attn.qkv_proj", None),
    ("self_attn", "o_proj"): ("self_attn.o_proj", None),
    ("mlp", "gate_proj"): ("mlp.gate_up_proj", "gate"),
    ("mlp", "up_proj"): ("mlp.gate_up_proj", "up"),
    ("mlp", "gate_up_proj"): ("mlp.gate_up_proj", None),
    ("mlp", "down_proj"): ("mlp.down_proj", None),
}
_ROLE_PROJ = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "gate": "gate_proj", "up": "up_proj"}
# per-layer order of the Mothersuperior ``.pt`` ``lora`` list
MS_PT_ORDER = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)

# key suffix -> part name, longest first
_SUFFIXES = (
    ("lora_down.weight", "down"),
    ("lora_up.weight", "up"),
    ("lora_A.weight", "down"),
    ("lora_B.weight", "up"),
    ("lora_A", "down"),
    ("lora_B", "up"),
    ("alpha", "alpha"),
    ("diff_b", "diff_b"),
    ("diff", "diff"),
    ("weight", "full_w"),
    ("bias", "full_b"),
)
_NATIVE_SPLIT_RE = re.compile(r"^(lora_down|lora_up)\.(\d+)\.weight$")
_NATIVE_BLOCK_RE = re.compile(
    r"^lora_unet_(ar|nar)_blocks_(\d+)_(self_attn_qkv_proj|self_attn_o_proj|mlp_gate_up_proj|mlp_down_proj)$"
)
_NATIVE_IO_RE = re.compile(r"^lora_unet_nar_(vae2llm|llm2vae|time_embedder_mlp_0|time_embedder_mlp_2)$")
_LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.(nar_)?(self_attn|mlp)\.(\w+)$")
_COMFY_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$")

FORMATS = ("native", "comfy", "aitk", "hf", "fl", "ms_safetensors", "ms_pt")


@dataclass
class ModuleDelta:
    """Delta of one native module. ``kind`` is ``lora`` (``downs``/``ups``/``alpha``), ``diff`` (``diff``/``diff_b``)
    or ``full`` (complete replacement ``full_w``/``full_b``, Mothersuperior I/O; needs the base to become a diff)."""

    kind: str
    downs: list[torch.Tensor] = field(default_factory=list)
    ups: list[torch.Tensor] = field(default_factory=list)
    alpha: Optional[float] = None
    diff: Optional[torch.Tensor] = None
    diff_b: Optional[torch.Tensor] = None
    full_w: Optional[torch.Tensor] = None
    full_b: Optional[torch.Tensor] = None

    @property
    def rank(self) -> int:
        return int(self.downs[0].shape[0])

    @property
    def scale(self) -> float:
        return float(self.alpha if self.alpha is not None else self.rank) / self.rank

    @property
    def is_split(self) -> bool:
        return self.kind == "lora" and len(self.downs) > 1

    def weight_delta(self) -> torch.Tensor:
        """fp32 ``[out, in]`` weight delta (LoRA pieces stacked by rows)."""
        if self.kind == "lora":
            s = self.scale
            return torch.cat([u.float() @ d.float() for d, u in zip(self.downs, self.ups)], dim=0) * s
        if self.kind == "diff":
            return self.diff.float()
        raise ValueError("a 'full' module has no delta without the base weights (pass base_io)")

    def bias_delta(self) -> Optional[torch.Tensor]:
        if self.kind == "diff" and self.diff_b is not None:
            return self.diff_b.float()
        return None


@dataclass
class NativeLoRA:
    modules: dict[str, ModuleDelta] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    source_format: str = "native"


# ----------------------------------------------------------------------------------------------------------------------
# names


def module_branch(path: str) -> str:
    """``ar`` / ``nar`` for block modules, ``io`` for the NAR I/O modules."""
    if path.startswith("ar.blocks."):
        return "ar"
    if path.startswith("nar.blocks."):
        return "nar"
    return "io"


def lora_name_of(path: str) -> str:
    return NATIVE_PREFIX + path.replace(".", "_")


def path_of(lora_name: str) -> str:
    """Native LoRA module name -> module path (``lora_unet_nar_time_embedder_mlp_0`` -> ``nar.time_embedder.mlp.0``)."""
    m = _NATIVE_BLOCK_RE.match(lora_name)
    if m is not None:
        sub = m.group(3).replace("self_attn_", "self_attn.").replace("mlp_", "mlp.")
        return f"{m.group(1)}.blocks.{m.group(2)}.{sub}"
    m = _NATIVE_IO_RE.match(lora_name)
    if m is not None:
        return "nar." + {"time_embedder_mlp_0": "time_embedder.mlp.0", "time_embedder_mlp_2": "time_embedder.mlp.2"}.get(
            m.group(1), m.group(1)
        )
    raise ValueError(f"not a YuE2 LoRA module name: {lora_name}")


def _block_parts(path: str) -> tuple[str, int, str]:
    branch, _, rest = path.partition(".blocks.")
    index, _, sub = rest.partition(".")
    return branch, int(index), sub


def split_dims_for(sub: str, config=None) -> list[int]:
    """Row split of a fused submodule: ``[q_dim, kv_dim, kv_dim]`` / ``[I, I]`` (YuE2-3B dims when ``config`` is None)."""
    if config is None:
        from musubi_tuner.yue2.yue2_model import YuE2Config

        config = YuE2Config()
    if sub == "self_attn.qkv_proj":
        q_dim = config.num_heads * config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim
        return [q_dim, kv_dim, kv_dim]
    if sub == "mlp.gate_up_proj":
        return [config.intermediate_size, config.intermediate_size]
    raise ValueError(f"{sub} is not a fused YuE2 submodule")


def _fused_split_dims(sub: str, config, out_features: int, in_features: int) -> list[int]:
    """``split_dims_for`` checked against the fused shape; without a config a non-3B shape falls back to the YuE2
    geometry (q_dim = hidden = in_features, gate = up)."""
    dims = split_dims_for(sub, config)
    if sum(dims) == out_features or config is not None:
        return dims
    if sub == "self_attn.qkv_proj" and (out_features - in_features) > 0 and (out_features - in_features) % 2 == 0:
        kv = (out_features - in_features) // 2
        return [in_features, kv, kv]
    if sub == "mlp.gate_up_proj" and out_features % 2 == 0:
        return [out_features // 2, out_features // 2]
    return dims


def _sub_of(path: str) -> Optional[str]:
    if module_branch(path) == "io":
        return None
    return _block_parts(path)[2]


# ----------------------------------------------------------------------------------------------------------------------
# file I/O


def load_lora_file(path: str) -> tuple[dict, dict[str, str]]:
    """Read a LoRA file: ``(state_dict, metadata)`` for safetensors, ``(checkpoint_dict, {})`` for ``.pt``."""
    if os.path.splitext(path)[1].lower() == ".safetensors":
        from safetensors import safe_open

        sd = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return sd, metadata
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # pickled objects beyond tensors/containers are not accepted
        raise ValueError(f"cannot load {path} with torch.load(weights_only=True): {e}") from e
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected a dict checkpoint, got {type(ckpt).__name__}")
    return ckpt, {}


def save_lora_file(path: str, sd: Mapping[str, torch.Tensor], metadata: Optional[Mapping[str, str]] = None) -> None:
    from safetensors.torch import save_file

    tensors, storages = {}, set()
    for k, v in sd.items():
        v = v.detach().contiguous().cpu()
        ptr = v.untyped_storage().data_ptr()
        if ptr in storages:  # e.g. the shared down of a fused module split for yue2-lora-v1; save_file refuses aliases
            v = v.clone()
        storages.add(v.untyped_storage().data_ptr())
        tensors[k] = v
    save_file(tensors, path, {k: str(v) for k, v in (metadata or {}).items()})


# ----------------------------------------------------------------------------------------------------------------------
# detection


def detect_format(sd: Mapping, metadata: Optional[Mapping[str, str]] = None) -> str:
    """First match wins: native, comfy, aitk, hf/fl (by metadata ``format``), ms_safetensors, ms_pt."""
    metadata = metadata or {}
    if isinstance(sd.get("lora"), (list, tuple)):
        return "ms_pt"
    keys = [k for k in sd.keys() if isinstance(k, str)]
    if not keys:
        raise ValueError("empty LoRA state dict")
    if any(k.startswith(NATIVE_PREFIX) for k in keys):
        return "native"
    if any(k.startswith(("diffusion_model.", "text_encoders.")) for k in keys):
        return "comfy"
    if any(k.startswith(("transformer.ar.", "transformer.nar.")) for k in keys):
        return "aitk"
    if any(k.startswith("layers.") and k.endswith(("lora_A", "lora_B")) for k in keys):
        return "ms_safetensors"
    top_level_io = any(k.split(".", 1)[0] in ("vae2llm", "llm2vae", "time_embedder") for k in keys)
    if top_level_io or any(k.startswith("model.layers.") for k in keys):
        return "fl" if metadata.get("format") == "fl-yue2-lora-v1" else "hf"
    raise ValueError(f"unknown YuE2 LoRA format; first keys: {keys[:5]}")


# ----------------------------------------------------------------------------------------------------------------------
# import


def _split_suffix(key: str) -> tuple[str, str, Optional[int]]:
    """``(module_part, part, piece_index)``; ``piece_index`` only for native ``lora_down.{i}.weight``."""
    for sep in (".lora_down.", ".lora_up."):
        pos = key.find(sep)
        if pos > 0:
            m = _NATIVE_SPLIT_RE.match(key[pos + 1 :])
            if m is not None:
                return key[:pos], "down" if m.group(1) == "lora_down" else "up", int(m.group(2))
    for suffix, part in _SUFFIXES:
        if key.endswith("." + suffix):
            return key[: -len(suffix) - 1], part, None
    raise ValueError(f"unrecognised LoRA key suffix: {key}")


def _map_module(fmt: str, module: str) -> tuple[str, Optional[str]]:
    """Format module name -> ``(native path, role)``."""
    if fmt == "native":
        return path_of(module), None
    if fmt in ("comfy", "aitk"):
        prefixes = (
            (("diffusion_model.", "nar"), ("text_encoders.", "ar"))
            if fmt == "comfy"
            else (("transformer.nar.", "nar"), ("transformer.ar.", "ar"))
        )
        for prefix, branch in prefixes:
            if module.startswith(prefix):
                rest = module[len(prefix) :]
                break
        else:
            raise ValueError(f"unrecognised {fmt} LoRA module: {module}")
        m = _COMFY_LAYER_RE.match(rest)
        if m is not None:
            mapped = _PROJ_MAP.get((m.group(2), m.group(3)))
            if mapped is None:
                raise ValueError(f"unsupported {fmt} LoRA target: {module}")
            return f"{branch}.blocks.{int(m.group(1))}.{mapped[0]}", mapped[1]
        if branch == "nar" and rest in IO_SUBMODULES:
            return "nar." + rest, None
        raise ValueError(f"unsupported {fmt} LoRA target: {module}")
    # hf / fl / ms
    m = _LAYER_RE.match(module)
    if m is not None:
        if fmt.startswith("ms") != (not module.startswith("model.")):
            raise ValueError(f"unsupported {fmt} LoRA target: {module}")
        mapped = _PROJ_MAP.get((m.group(3), m.group(4)))
        if mapped is None:
            raise ValueError(f"unsupported {fmt} LoRA target: {module}")
        branch = "nar" if m.group(2) else "ar"
        return f"{branch}.blocks.{int(m.group(1))}.{mapped[0]}", mapped[1]
    if module in IO_SUBMODULES:
        return "nar." + module, None
    raise ValueError(f"unsupported {fmt} LoRA target: {module}")


def _ms_pt_to_sd(ckpt: Mapping) -> dict[str, torch.Tensor]:
    """Mothersuperior ``.pt`` -> ``ms_safetensors`` keys (layer-major q, k, v, o, gate, up, down; ``[A, B]`` each)."""
    values = list(ckpt["lora"])
    nar = "io" in ckpt
    per_layer = len(MS_PT_ORDER) * 2
    if len(values) % per_layer != 0:
        raise ValueError(f"Mothersuperior .pt: {len(values)} LoRA tensors is not a multiple of {per_layer}")
    prefix = "nar_" if nar else ""
    sd = {}
    for i in range(0, len(values), 2):
        layer, j = divmod(i // 2, len(MS_PT_ORDER))
        block, proj = MS_PT_ORDER[j]
        name = f"layers.{layer}.{prefix}{block}.{proj}"
        sd[name + ".lora_A"] = values[i]
        sd[name + ".lora_B"] = values[i + 1]
    for module, params in (ckpt.get("io") or {}).items():
        for pname, value in params.items():
            sd[f"{module}.{pname}"] = value
    return sd


def to_native(
    sd: Mapping,
    metadata: Optional[Mapping[str, str]] = None,
    *,
    layout_hint: Optional[str] = None,
    base_io: Optional[Mapping[str, torch.Tensor]] = None,
    config=None,
) -> NativeLoRA:
    """Any known format -> ``NativeLoRA``.

    ``layout_hint`` forces a format name (``FORMATS``) instead of detection. ``base_io`` (native names
    ``nar.vae2llm.weight`` ... as returned by ``yue2_checkpoint.read_base_io``) turns ``full`` I/O weights into diffs.
    ``config`` gives the q/k/v and gate/up row split for zero-filling missing pieces (YuE2-3B when None).
    """
    metadata = dict(metadata or {})
    fmt = layout_hint or detect_format(sd, metadata)
    if fmt not in FORMATS:
        raise ValueError(f"unknown YuE2 LoRA format {fmt!r}; choose one of {FORMATS}")
    if fmt == "ms_pt":
        sd = _ms_pt_to_sd(sd)
    default_alpha = None
    if fmt == "hf" and metadata.get("format") == "yue2-lora-v1" and metadata.get("alpha"):
        default_alpha = float(metadata["alpha"])

    # path -> role -> part -> tensor
    raw: dict[str, dict[Optional[str], dict[str, torch.Tensor]]] = {}
    native_pieces: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for key, value in sd.items():
        if not isinstance(value, torch.Tensor):
            continue
        module, part, piece = _split_suffix(key)
        path, role = _map_module(fmt, module)
        if part in ("full_w", "full_b") and module_branch(path) != "io":
            raise ValueError(f"full weights are only supported on the NAR I/O modules: {key}")
        if piece is not None:
            native_pieces.setdefault(path, {}).setdefault(piece, {})[part] = value
            raw.setdefault(path, {}).setdefault(None, {})
            continue
        slot = raw.setdefault(path, {}).setdefault(role, {})
        if part in slot:
            raise ValueError(f"duplicate LoRA tensor for {path} ({role or 'fused'}.{part}): {key}")
        slot[part] = value

    native = NativeLoRA(metadata=metadata, source_format=fmt)
    for path in sorted(raw, key=_path_sort_key):
        roles = raw[path]
        if path in native_pieces:
            md = _native_split_module(path, native_pieces[path], roles.get(None, {}))
        elif None in roles and len(roles) > 1:
            raise ValueError(f"{path}: both fused and per-projection LoRA tensors")
        elif None in roles:
            md = _single_module(path, roles[None], default_alpha)
        else:
            md = _grouped_module(path, roles, default_alpha, config)
        if md.kind == "full" and base_io is not None and path + ".weight" in base_io:  # else stays full (raises on export)
            md = _full_to_diff(path, md, base_io)
        native.modules[path] = md
    if not native.modules:
        raise ValueError("no LoRA modules found")
    return native


def _path_sort_key(path: str):
    if module_branch(path) == "io":
        return (2, 0, IO_SUBMODULES.index(path[4:]) if path[4:] in IO_SUBMODULES else 99)
    branch, index, sub = _block_parts(path)
    return (0 if branch == "ar" else 1, index, BLOCK_SUBMODULES.index(sub))


def _alpha_value(t: Optional[torch.Tensor]) -> Optional[float]:
    return None if t is None else float(t.detach().float().reshape(-1)[0].item())


def _single_module(path: str, parts: Mapping[str, torch.Tensor], default_alpha: Optional[float]) -> ModuleDelta:
    has_lora = "down" in parts or "up" in parts
    has_diff = "diff" in parts or "diff_b" in parts
    has_full = "full_w" in parts or "full_b" in parts
    if has_lora + has_diff + has_full != 1:
        raise ValueError(f"{path}: a module must be exactly one of LoRA, diff or full weights ({sorted(parts)})")
    if has_lora:
        if "down" not in parts or "up" not in parts:
            raise ValueError(f"{path}: incomplete LoRA pair ({sorted(parts)})")
        down, up = parts["down"], parts["up"]
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError(f"{path}: LoRA shapes down {tuple(down.shape)} / up {tuple(up.shape)} do not chain")
        alpha = _alpha_value(parts.get("alpha"))
        if alpha is None:
            alpha = default_alpha if default_alpha is not None else float(down.shape[0])
        return ModuleDelta("lora", [down], [up], alpha)
    if "alpha" in parts:
        raise ValueError(f"{path}: alpha without a LoRA pair")
    if has_diff:
        if "diff" not in parts:
            raise ValueError(f"{path}: diff_b without diff")
        return ModuleDelta("diff", diff=parts["diff"], diff_b=parts.get("diff_b"))
    if "full_w" not in parts:
        raise ValueError(f"{path}: full bias without full weight")
    return ModuleDelta("full", full_w=parts["full_w"], full_b=parts.get("full_b"))


def _native_split_module(path, pieces: dict[int, dict[str, torch.Tensor]], shared: Mapping[str, torch.Tensor]) -> ModuleDelta:
    extra = set(shared) - {"alpha"}
    if extra:
        raise ValueError(f"{path}: split and fused LoRA tensors mixed ({sorted(extra)})")
    if sorted(pieces) != list(range(len(pieces))):
        raise ValueError(f"{path}: split pieces {sorted(pieces)} are not contiguous from 0")
    downs, ups = [], []
    for i in range(len(pieces)):
        p = pieces[i]
        if "down" not in p or "up" not in p:
            raise ValueError(f"{path}: split piece {i} is missing lora_down or lora_up")
        downs.append(p["down"])
        ups.append(p["up"])
    ranks = {d.shape[0] for d in downs}
    if len(ranks) != 1:
        raise ValueError(f"{path}: split pieces have different ranks {sorted(ranks)}")
    alpha = _alpha_value(shared.get("alpha"))
    return ModuleDelta("lora", downs, ups, alpha if alpha is not None else float(downs[0].shape[0]))


def _grouped_module(path: str, roles: dict, default_alpha: Optional[float], config) -> ModuleDelta:
    sub = _sub_of(path)
    order = SPLIT_ROLES.get(sub)
    if order is None or set(roles) - set(order):
        raise ValueError(f"{path}: unexpected projection roles {sorted(roles)}")
    singles = {role: _single_module(f"{path}[{role}]", roles[role], default_alpha) for role in roles}
    if any(md.kind != "lora" for md in singles.values()):
        raise ValueError(f"{path}: per-projection entries must be LoRA pairs")
    in_dims = {md.downs[0].shape[1] for md in singles.values()}
    if len(in_dims) != 1:
        raise ValueError(f"{path}: projections disagree on in_features {sorted(in_dims)}")
    in_dim = in_dims.pop()
    missing = [role for role in order if role not in singles]
    if missing:
        dims = dict(zip(order, split_dims_for(sub, config)))
        rank = next(iter(singles.values())).rank
        ref = next(iter(singles.values())).downs[0]
        for role in missing:
            singles[role] = ModuleDelta("lora", [ref.new_zeros(rank, in_dim)], [ref.new_zeros(dims[role], rank)], float(rank))
    pieces = [singles[role] for role in order]
    ranks = {md.rank for md in pieces}
    alphas = {md.alpha for md in pieces}
    if len(ranks) == 1 and len(alphas) == 1:
        return ModuleDelta("lora", [md.downs[0] for md in pieces], [md.ups[0] for md in pieces], pieces[0].alpha)
    if len(ranks) == 1:
        rank = ranks.pop()
        return ModuleDelta("lora", [md.downs[0] for md in pieces], [_bake(md.ups[0], md.scale) for md in pieces], float(rank))
    # unequal ranks cannot share one musubi split module: store the exact block-diagonal fusion
    return _block_diag(pieces)


def _bake(up: torch.Tensor, scale: float) -> torch.Tensor:
    return up if scale == 1.0 else up.float() * scale


def _block_diag(pieces: Sequence[ModuleDelta]) -> ModuleDelta:
    """Exact fusion of per-row-block LoRA pieces: ``down = cat(downs)``, ``up = blockdiag(up_i * scale_i)``, alpha = Σr."""
    downs = [d.float() for md in pieces for d in md.downs]
    ups = [u.float() * md.scale for md in pieces for u in md.ups]
    total_rank = sum(d.shape[0] for d in downs)
    up = torch.zeros(sum(u.shape[0] for u in ups), total_rank, dtype=torch.float32)
    row = col = 0
    for u in ups:
        up[row : row + u.shape[0], col : col + u.shape[1]] = u
        row += u.shape[0]
        col += u.shape[1]
    return ModuleDelta("lora", [torch.cat(downs, dim=0)], [up], float(total_rank))


def _full_to_diff(path: str, md: ModuleDelta, base_io: Mapping[str, torch.Tensor]) -> ModuleDelta:
    wkey, bkey = path + ".weight", path + ".bias"
    if wkey not in base_io:
        raise ValueError(f"base_io has no {wkey} to turn the full {path} weights into a diff")
    base_w = base_io[wkey].float()
    if tuple(base_w.shape) != tuple(md.full_w.shape):
        raise ValueError(f"{path}: full weight {tuple(md.full_w.shape)} does not match the base {tuple(base_w.shape)}")
    diff_b = None
    if md.full_b is not None:
        if bkey not in base_io:
            raise ValueError(f"base_io has no {bkey}")
        diff_b = md.full_b.float() - base_io[bkey].float()
    return ModuleDelta("diff", diff=md.full_w.float() - base_w, diff_b=diff_b)


# ----------------------------------------------------------------------------------------------------------------------
# layout conversions


def split_module_to_fused(md: ModuleDelta) -> ModuleDelta:
    if not md.is_split:
        return md
    alpha = md.alpha if md.alpha is not None else float(md.rank)
    return _block_diag([ModuleDelta("lora", [d], [u], alpha) for d, u in zip(md.downs, md.ups)])


def fused_module_to_split(md: ModuleDelta, split_dims: Sequence[int], rank: Optional[int] = None) -> ModuleDelta:
    """Exact fused -> split. Pieces either share the fused ``down`` (piece rank = fused rank R, ``up`` sliced by rows)
    or, when ``up`` is block-diagonal with equal blocks, are its blocks (piece rank R / n). ``rank`` picks the piece
    rank (raises when neither form has it); None prefers the block decomposition."""
    if md.kind != "lora" or md.is_split:
        return md
    down, up = md.downs[0], md.ups[0]
    n = len(split_dims)
    if sum(split_dims) != up.shape[0]:
        raise ValueError(f"split dims {list(split_dims)} do not sum to the fused out_features {up.shape[0]}")
    fused_rank = down.shape[0]
    bounds = [0]
    for d in split_dims:
        bounds.append(bounds[-1] + d)
    block = None
    if fused_rank % n == 0 and rank != fused_rank:
        r = fused_rank // n
        off_block = up.clone()
        for i in range(n):
            off_block[bounds[i] : bounds[i + 1], i * r : (i + 1) * r] = 0
        if not off_block.any():
            downs = [down[i * r : (i + 1) * r] for i in range(n)]
            ups = [up[bounds[i] : bounds[i + 1], i * r : (i + 1) * r] for i in range(n)]
            block = ModuleDelta("lora", downs, ups, md.scale * r)
    if block is not None and (rank is None or rank == block.rank):
        return block
    if rank is not None and rank != fused_rank:
        raise ValueError(
            f"a fused rank-{fused_rank} LoRA cannot be split exactly into {n} pieces of rank {rank}"
            + ("" if block is None else f" (block-diagonal pieces have rank {block.rank})")
        )
    return ModuleDelta("lora", [down] * n, [up[bounds[i] : bounds[i + 1]] for i in range(n)], md.alpha)


def split_to_fused(native: NativeLoRA) -> NativeLoRA:
    """Block-diagonal fusion of every split module (for merge-at-load and Comfy export); alpha = fused rank."""
    modules = {p: split_module_to_fused(md) for p, md in native.modules.items()}
    return NativeLoRA(modules, dict(native.metadata), native.source_format)


def fused_to_split(native: NativeLoRA, config=None) -> NativeLoRA:
    out = {}
    for path, md in native.modules.items():
        sub = _sub_of(path)
        if md.kind == "lora" and not md.is_split and sub in SPLIT_ROLES:
            md = fused_module_to_split(md, _fused_split_dims(sub, config, md.ups[0].shape[0], md.downs[0].shape[1]))
        out[path] = md
    return NativeLoRA(out, dict(native.metadata), native.source_format)


def io_lora_to_diff(native: NativeLoRA) -> NativeLoRA:
    """I/O LoRA modules -> their product diff (the merge-input contract has diffs only on I/O modules)."""
    out = {}
    for path, md in native.modules.items():
        if module_branch(path) == "io" and md.kind == "lora":
            md = ModuleDelta("diff", diff=md.weight_delta())
        out[path] = md
    return NativeLoRA(out, dict(native.metadata), native.source_format)


def concat_rank(a: NativeLoRA, b: NativeLoRA) -> NativeLoRA:
    """Sum of two adapters as one: LoRA pairs are rank-concatenated with scales baked (alpha = total rank), diffs add.
    A module that is a LoRA in one and a diff in the other becomes the diff of the summed deltas."""
    out = {}
    for path in sorted(set(a.modules) | set(b.modules), key=_path_sort_key):
        ma, mb = a.modules.get(path), b.modules.get(path)
        if ma is None or mb is None:
            out[path] = ma if mb is None else mb
            continue
        if "full" in (ma.kind, mb.kind):
            raise ValueError(f"{path}: full weights must be converted to a diff (base_io) before concatenation")
        if ma.kind == mb.kind == "lora":
            if len(ma.downs) != len(mb.downs):
                ma, mb = split_module_to_fused(ma), split_module_to_fused(mb)
            downs = [torch.cat([da.float(), db.float()], dim=0) for da, db in zip(ma.downs, mb.downs)]
            ups = [torch.cat([ua.float() * ma.scale, ub.float() * mb.scale], dim=1) for ua, ub in zip(ma.ups, mb.ups)]
            out[path] = ModuleDelta("lora", downs, ups, float(downs[0].shape[0]))
            continue
        diff = ma.weight_delta() + mb.weight_delta()
        ba, bb = ma.bias_delta(), mb.bias_delta()
        diff_b = None if ba is None and bb is None else (ba if bb is None else bb if ba is None else ba + bb)
        out[path] = ModuleDelta("diff", diff=diff, diff_b=diff_b)
    return NativeLoRA(out, dict(a.metadata), a.source_format)


def filter_branch(native: NativeLoRA, branch: str) -> NativeLoRA:
    """``both`` keeps everything; ``ar`` keeps AR blocks; ``nar`` keeps NAR blocks and I/O."""
    if branch == "both":
        return native
    if branch not in ("ar", "nar"):
        raise ValueError(f"branch must be both, ar or nar, got {branch!r}")
    keep = {p: md for p, md in native.modules.items() if (module_branch(p) == "ar") == (branch == "ar")}
    return NativeLoRA(keep, dict(native.metadata), native.source_format)


def scale_branches(native: NativeLoRA, ar: float = 1.0, nar: float = 1.0, io: Optional[float] = None) -> NativeLoRA:
    """Multiply the delta of every AR / NAR / I/O module (``io`` defaults to ``nar``)."""
    factors = {"ar": ar, "nar": nar, "io": nar if io is None else io}
    out = {}
    for path, md in native.modules.items():
        f = factors[module_branch(path)]
        if f != 1.0:
            if md.kind == "lora":
                md = ModuleDelta("lora", list(md.downs), [u.float() * f for u in md.ups], md.alpha)
            elif md.kind == "diff":
                md = ModuleDelta("diff", diff=md.diff.float() * f, diff_b=None if md.diff_b is None else md.diff_b.float() * f)
            else:
                raise ValueError(f"{path}: cannot scale full weights (convert with base_io first)")
        out[path] = md
    return NativeLoRA(out, dict(native.metadata), native.source_format)


def weight_deltas(native: NativeLoRA) -> dict[str, tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """``{path: (weight_delta fp32, bias_delta fp32 or None)}`` for comparisons and verification."""
    return {p: (md.weight_delta(), md.bias_delta()) for p, md in native.modules.items()}


# ----------------------------------------------------------------------------------------------------------------------
# export


def _cast(t: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    t = t.detach()
    return (t.to(dtype) if dtype is not None else t).contiguous()


def native_state_dict(native: NativeLoRA, *, dtype: Optional[torch.dtype] = None) -> dict[str, torch.Tensor]:
    """``NativeLoRA`` -> musubi native keys, layout kept; every LoRA module carries ``.alpha``."""
    sd = {}
    for path, md in native.modules.items():
        name = lora_name_of(path)
        if md.kind == "full":
            raise ValueError(
                f"{path} holds full replacement weights; pass base_io (read_base_io of the base checkpoint) "
                "to convert them to a diff"
            )
        if md.kind == "diff":
            sd[name + ".diff"] = _cast(md.diff, dtype)
            if md.diff_b is not None:
                sd[name + ".diff_b"] = _cast(md.diff_b, dtype)
            continue
        if md.is_split:
            for i, (d, u) in enumerate(zip(md.downs, md.ups)):
                sd[f"{name}.lora_down.{i}.weight"] = _cast(d, dtype)
                sd[f"{name}.lora_up.{i}.weight"] = _cast(u, dtype)
        else:
            sd[name + ".lora_down.weight"] = _cast(md.downs[0], dtype)
            sd[name + ".lora_up.weight"] = _cast(md.ups[0], dtype)
        sd[name + ".alpha"] = torch.tensor(float(md.alpha if md.alpha is not None else md.rank))
    return sd


def native_fused_state_dict(native: NativeLoRA) -> dict[str, torch.Tensor]:
    """The merge-input dict: block modules fused (block-diagonal), I/O modules as diffs, no ``full`` entries."""
    return native_state_dict(io_lora_to_diff(split_to_fused(native)))


def _comfy_module_name(path: str) -> str:
    branch = module_branch(path)
    if branch == "io":
        return "diffusion_model." + path[len("nar.") :]
    b, index, sub = _block_parts(path)
    return f"{'text_encoders' if b == 'ar' else 'diffusion_model'}.model.layers.{index}.{sub}"


def native_to_comfy(native: NativeLoRA, *, dtype: Optional[torch.dtype] = torch.float32) -> tuple[dict, dict[str, str]]:
    """ComfyUI native LoRA (loads with ``LoraLoaderModelOnly`` for NAR, CLIP ``LoraLoader`` for AR).

    Split modules become block-diagonal fused pairs with the scale baked into ``up`` and ``.alpha`` = fused rank (so
    Comfy's alpha/rank is exactly 1); fused modules keep their ``.alpha``; I/O diffs as ``.diff``/``.diff_b``.
    """
    sd = {}
    branches = set()
    split_any = False
    for path, md in native.modules.items():
        name = _comfy_module_name(path)
        branches.add("ar" if module_branch(path) == "ar" else "nar")
        if md.kind == "full":
            raise ValueError(f"{path}: full weights need base_io before export")
        if md.kind == "diff":
            sd[name + ".diff"] = _cast(md.diff, dtype)
            if md.diff_b is not None:
                sd[name + ".diff_b"] = _cast(md.diff_b, dtype)
            continue
        if md.is_split:
            split_any = True
            md = split_module_to_fused(md)
        sd[name + ".lora_down.weight"] = _cast(md.downs[0], dtype)
        sd[name + ".lora_up.weight"] = _cast(md.ups[0], dtype)
        sd[name + ".alpha"] = torch.tensor(float(md.alpha if md.alpha is not None else md.rank))
    metadata = {
        "format": "comfyui-native-lora",
        "source_format": "musubi-yue2",
        "yue2_lora_branch": "+".join(sorted(branches)),
    }
    if split_any:
        metadata["layout"] = (
            "qkv_proj and gate_up_proj are block-diagonal fusions of separate q/k/v and gate/up LoRAs "
            "(rank 3r / 2r, scale baked into lora_up, alpha = fused rank); up@down == stacked deltas exactly"
        )
    return sd, metadata


def _hf_block_name(path: str, piece_role: Optional[str]) -> str:
    branch, index, sub = _block_parts(path)
    block, proj = sub.split(".")
    if piece_role is not None:
        proj = _ROLE_PROJ[piece_role]
    return f"model.layers.{index}.{'nar_' if branch == 'nar' else ''}{block}.{proj}"


def _hf_pieces(path: str, md: ModuleDelta, config) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Per HF projection ``(name, down, up*scale)``; fused q|k|v and gate|up are split exactly."""
    sub = _sub_of(path)
    if sub in SPLIT_ROLES:
        if not md.is_split:
            md = fused_module_to_split(md, _fused_split_dims(sub, config, md.ups[0].shape[0], md.downs[0].shape[1]))
        return [(_hf_block_name(path, role), d, u.float() * md.scale) for role, d, u in zip(SPLIT_ROLES[sub], md.downs, md.ups)]
    return [(_hf_block_name(path, None), md.downs[0], md.ups[0].float() * md.scale)]


def _uniform_rank_metadata(sd: Mapping[str, torch.Tensor]) -> dict[str, str]:
    ranks = {v.shape[0] for k, v in sd.items() if k.endswith(".lora_down.weight")}
    if len(ranks) == 1:
        r = str(ranks.pop())
        return {"rank": r, "alpha": r}
    return {}


def native_to_hf(
    native: NativeLoRA, branch: str = "nar", *, config=None, dtype: Optional[torch.dtype] = torch.float32
) -> tuple[dict, dict[str, str]]:
    """``yue2-lora-v1``: separate projections, scale baked into ``up`` (every key scale 1).

    ``branch="nar"`` writes NAR blocks and I/O LoRAs (an I/O diff is written as an exact full-rank factorization and
    must have no bias delta); ``branch="ar"`` writes the AR blocks under ``model.layers.N.self_attn.*`` names (the
    ``fl-yue2-lora-v1`` AR layout).
    """
    if branch not in ("ar", "nar"):
        raise ValueError("native_to_hf: branch must be 'ar' or 'nar'")
    sd = {}
    for path, md in native.modules.items():
        mb = module_branch(path)
        if (mb == "ar") != (branch == "ar"):
            if branch == "nar" and mb == "ar":
                raise ValueError("yue2-lora-v1 is NAR-only; export the AR modules with branch='ar' (or use comfy/fl)")
            continue
        if md.kind == "full":
            raise ValueError(f"{path}: full weights need base_io before export")
        if mb == "io":
            name = path[len("nar.") :]
            if md.kind == "diff":
                if md.diff_b is not None and md.diff_b.float().abs().max() > 0:
                    raise ValueError(f"{path}: yue2-lora-v1 cannot carry a bias delta; export to comfy or fl instead")
                diff = md.diff.float()
                out_f, in_f = diff.shape
                if in_f <= out_f:
                    down, up = torch.eye(in_f), diff
                else:
                    down, up = diff, torch.eye(out_f)
            else:
                down, up = md.downs[0], md.ups[0].float() * md.scale
            sd[name + ".lora_down.weight"] = _cast(down, dtype)
            sd[name + ".lora_up.weight"] = _cast(up, dtype)
            continue
        for name, down, up in _hf_pieces(path, md, config):
            sd[name + ".lora_down.weight"] = _cast(down, dtype)
            sd[name + ".lora_up.weight"] = _cast(up, dtype)
    if not sd:
        raise ValueError(f"no {branch} modules to export")
    metadata = {"format": "yue2-lora-v1", "branch": branch, "source_format": "musubi-yue2", **_uniform_rank_metadata(sd)}
    return sd, metadata


def native_to_fl(
    native: NativeLoRA, *, config=None, dtype: Optional[torch.dtype] = torch.float32
) -> dict[str, tuple[dict, dict[str, str]]]:
    """``fl-yue2-lora-v1``: one ``(state_dict, metadata)`` per present branch; scale baked (no alpha), I/O as
    ``vae2llm/llm2vae .diff/.diff_b`` (an I/O LoRA is written as its product). The format has no time-embedder target."""
    per_branch: dict[str, dict] = {}
    for path, md in native.modules.items():
        mb = module_branch(path)
        if md.kind == "full":
            raise ValueError(f"{path}: full weights need base_io before export")
        if mb == "io":
            name = path[len("nar.") :]
            if name not in ("vae2llm", "llm2vae"):
                raise ValueError(f"{path}: fl-yue2-lora-v1 has no {name} target")
            sd = per_branch.setdefault("nar", {})
            sd[name + ".diff"] = _cast(md.weight_delta(), dtype)
            if md.bias_delta() is not None:
                sd[name + ".diff_b"] = _cast(md.bias_delta(), dtype)
            continue
        sd = per_branch.setdefault(mb, {})
        for name, down, up in _hf_pieces(path, md, config):
            sd[name + ".lora_down.weight"] = _cast(down, dtype)
            sd[name + ".lora_up.weight"] = _cast(up, dtype)
    out = {}
    for b, sd in per_branch.items():
        meta = {"format": "fl-yue2-lora-v1", "branch": b, "source_format": "musubi-yue2"}
        rank_meta = _uniform_rank_metadata(sd)
        if rank_meta:
            meta["rank"] = rank_meta["rank"]
        out[b] = (sd, meta)
    return out
