"""Our ``yue2-lora-v1`` (and NAR ComfyUI) exports, read and merged by a minimal standalone consumer, on CPU.

``read_lora_v1`` / ``module_linears`` / ``merge_patches`` below are a small reader written in this repository for
this test from our own description of the format (keys, shapes, metadata ``alpha``, allowed targets); they do not
import ``yue2_lora_formats`` and do not reuse its parsing. The reader targets the official ``YuE2ForCausalLM``
Linears (``tests/yue2_ref``) rather than our model, and the merged weight changes are compared with the deltas of
the musubi LoRA we exported. It is a stand-in for an external consumer, not a copy of any particular one.
"""

from dataclasses import dataclass
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_yue2_lora_formats import CFG, make_native  # noqa: E402

from musubi_tuner.yue2 import yue2_lora_formats as fm  # noqa: E402

# region standalone reader

ACCEPTED_FORMATS = {None, "yue2-lora-v1", "comfyui-native-lora"}
IO_LINEARS = {"vae2llm", "llm2vae", "time_embedder.mlp.0", "time_embedder.mlp.2"}
V1_PROJECTIONS = {
    "nar_self_attn": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "nar_mlp": {"gate_proj", "up_proj", "down_proj"},
}
# ComfyUI NAR names: block -> {projection: official projections it stacks by rows}
COMFY_PROJECTIONS = {
    "self_attn": {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "o_proj": ("o_proj",)},
    "mlp": {"gate_up_proj": ("gate_proj", "up_proj"), "down_proj": ("down_proj",)},
}
COMFY_PREFIX = "diffusion_model."


@dataclass(frozen=True)
class Patch:
    module: str
    linears: tuple  # official Linear names, their rows stacked in this order in ``up``
    down: torch.Tensor  # fp32 [rank, in]
    up: torch.Tensor  # fp32 [sum(out), rank]
    scale: float


def finite_number(value, what: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{what}: expected a number, got a bool")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: {value!r} is not a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{what}: {value!r} is not finite")
    return number


def module_linears(module: str) -> tuple:
    """File module name -> the official Linear names it patches (NAR and I/O only)."""
    comfy = module.startswith(COMFY_PREFIX)
    name = module[len(COMFY_PREFIX) :] if comfy else module
    if name in IO_LINEARS:
        return (name,)
    parts = name.split(".")
    if len(parts) != 5 or parts[0] != "model" or parts[1] != "layers":
        raise ValueError(f"not a NAR layer or I/O module: {module!r}")
    layer, block, proj = parts[2:]
    if not (layer.isascii() and layer.isdigit()):
        raise ValueError(f"layer index {layer!r} in {module!r} is not a decimal number")
    if comfy:
        stacked = COMFY_PROJECTIONS.get(block, {}).get(proj)
        if stacked is None:
            raise ValueError(f"no NAR projection {block}.{proj} in ComfyUI naming: {module!r}")
        return tuple(f"model.layers.{layer}.nar_{block}.{p}" for p in stacked)
    if proj not in V1_PROJECTIONS.get(block, set()):
        raise ValueError(f"no NAR projection {block}.{proj} in yue2-lora-v1 naming: {module!r}")
    return (name,)


def _metadata_of(path) -> dict:
    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    return dict(header.get("__metadata__") or {})


def _check_matrix(t: torch.Tensor, label: str) -> torch.Tensor:
    if t.ndim != 2 or 0 in t.shape:
        raise ValueError(f"{label}: expected a non-empty matrix, got shape {tuple(t.shape)}")
    if not t.is_floating_point():
        raise ValueError(f"{label}: expected a floating point tensor, got {t.dtype}")
    t = t.float()
    if not torch.isfinite(t).all():
        raise ValueError(f"{label}: contains inf or nan")
    return t


def read_lora_v1(path):
    """``(metadata, [Patch, ...])`` sorted by module name; every tensor in the file must be used."""
    from safetensors import safe_open

    metadata = _metadata_of(path)
    fmt = metadata.get("format")
    if fmt not in ACCEPTED_FORMATS:
        raise ValueError(f"format {fmt!r} is not one this reader understands")

    parts: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if ".lora_" in key:
                module, tail = key.split(".lora_", 1)
                role = {"down.weight": "down", "up.weight": "up"}.get(tail)
            elif key.endswith(".alpha"):
                module, role = key[: -len(".alpha")], "alpha"
            else:
                role = None
            if role is None or not module:
                raise ValueError(f"unexpected tensor {key!r}")
            parts.setdefault(module, {})[role] = f.get_tensor(key)
    if not parts:
        raise ValueError("file holds no LoRA tensors")

    patches = []
    for module in sorted(parts):
        entry = parts[module]
        if "down" not in entry or "up" not in entry:
            raise ValueError(f"{module}: needs both lora_down and lora_up, found {sorted(entry)}")
        down = _check_matrix(entry["down"], f"{module} lora_down")
        up = _check_matrix(entry["up"], f"{module} lora_up")
        rank = down.shape[0]
        if up.shape[1] != rank:
            raise ValueError(f"{module}: lora_up has {up.shape[1]} columns, lora_down has rank {rank}")
        if "alpha" in entry:
            alpha_t = entry["alpha"]
            if alpha_t.numel() != 1 or alpha_t.dtype == torch.bool:
                raise ValueError(f"{module}: alpha must be one number, got {alpha_t.dtype} {tuple(alpha_t.shape)}")
            alpha = finite_number(alpha_t.reshape(()).item(), f"{module} alpha")
        elif fmt == "yue2-lora-v1" and "alpha" in metadata:
            alpha = finite_number(metadata["alpha"], "metadata alpha")
        else:
            alpha = float(rank)
        patches.append(Patch(module, module_linears(module), down, up, alpha / rank))
    return metadata, patches


def merge_patches(model: torch.nn.Module, patches, multiplier=1.0) -> None:
    """``W += multiplier * scale * up_rows @ down`` per Linear; all checks run before any weight is written."""
    factor = finite_number(multiplier, "multiplier")
    plan, seen = [], set()
    for patch in patches:
        start = 0
        for name in patch.linears:
            try:
                linear = model.get_submodule(name)
            except AttributeError:
                raise ValueError(f"{patch.module}: the model has no {name}") from None
            if not isinstance(linear, torch.nn.Linear):
                raise ValueError(f"{patch.module}: {name} is a {type(linear).__name__}, not a Linear")
            if name in seen:
                raise ValueError(f"{patch.module}: {name} is patched twice")
            seen.add(name)
            if patch.down.shape[1] != linear.in_features:
                raise ValueError(f"{patch.module}: lora_down takes {patch.down.shape[1]} inputs, {name} has {linear.in_features}")
            stop = start + linear.out_features
            if stop > patch.up.shape[0]:
                raise ValueError(f"{patch.module}: lora_up has {patch.up.shape[0]} rows, fewer than the target Linears need")
            plan.append((linear.weight, patch.up[start:stop], patch.down, patch.scale * factor))
            start = stop
        if start != patch.up.shape[0]:
            raise ValueError(f"{patch.module}: lora_up has {patch.up.shape[0]} rows, the target Linears have {start}")
    with torch.no_grad():
        for weight, up_rows, down, k in plan:
            weight.copy_((weight.float() + k * (up_rows @ down)).to(weight.dtype))


# endregion


def _ref_weight_deltas(native: fm.NativeLoRA) -> dict[str, torch.Tensor]:
    """Our native deltas under the official ``YuE2ForCausalLM`` parameter names (fused rows split q|k|v, gate|up)."""
    out = {}
    for path, (dw, _) in fm.weight_deltas(native).items():
        if fm.module_branch(path) == "io":
            out[path[len("nar.") :] + ".weight"] = dw
            continue
        _, index, sub = fm._block_parts(path)
        block, proj = sub.split(".")
        prefix = f"model.layers.{index}.nar_{block}."
        if sub in fm.SPLIT_ROLES:
            dims = fm.split_dims_for(sub, CFG)
            for role, piece in zip(fm.SPLIT_ROLES[sub], dw.split(dims, dim=0)):
                out[prefix + fm._ROLE_PROJ[role] + ".weight"] = piece
        else:
            out[prefix + proj + ".weight"] = dw
    return out


def _merged_deltas(path) -> dict[str, torch.Tensor]:
    pytest.importorskip("transformers")
    from yue2_fakes import make_ref_model

    model = make_ref_model(CFG)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    _, patches = read_lora_v1(path)
    merge_patches(model, patches)
    return {k: v.detach() - before[k] for k, v in model.named_parameters() if not torch.equal(v, before[k])}


def _assert_deltas(native, got):
    expected = _ref_weight_deltas(native)
    assert set(got) == set(expected), (sorted(set(got) ^ set(expected)))[:5]
    for key, delta in expected.items():
        err = ((got[key] - delta).norm() / delta.norm()).item()
        assert err < 1e-4, (key, err)


@pytest.mark.parametrize(
    "kind",
    [
        dict(split=True, io="lora", time_embedder=True, rank=4, alpha=8.0),  # uniform rank: metadata alpha == rank
        dict(split=False, io="diff", io_bias=False, rank=5, alpha=2.0),  # factorized I/O diffs: no metadata alpha
    ],
)
def test_hf_nar_export_reads_and_merges_in_lora_v1_reader(tmp_path, kind):
    native = fm.filter_branch(make_native(branches=("ar", "nar"), **kind), "nar")
    sd, meta = fm.native_to_hf(native)
    path = tmp_path / "lora.hf-nar.safetensors"
    fm.save_lora_file(str(path), sd, meta)
    got_meta, patches = read_lora_v1(path)
    assert got_meta["format"] == "yue2-lora-v1" and got_meta["branch"] == "nar"
    assert all(p.scale == 1.0 for p in patches)  # the exporter bakes the scale into lora_up
    _assert_deltas(native, _merged_deltas(path))


def test_comfy_nar_export_reads_and_merges_in_lora_v1_reader(tmp_path):
    native = fm.filter_branch(make_native(branches=("nar",), io="lora", rank=4, alpha=2.0), "nar")
    sd, meta = fm.native_to_comfy(native)
    path = tmp_path / "lora.comfy.safetensors"
    fm.save_lora_file(str(path), sd, meta)
    _assert_deltas(native, _merged_deltas(path))


def test_lora_v1_reader_applies_metadata_alpha_and_key_alpha(tmp_path):
    from safetensors.torch import save_file

    o = "model.layers.0.nar_self_attn.o_proj"
    g = torch.Generator().manual_seed(0)
    down, up = torch.randn(3, CFG.q_dim, generator=g), torch.randn(CFG.hidden_size, 3, generator=g)
    pair = {f"{o}.lora_down.weight": down, f"{o}.lora_up.weight": up}
    cases = [
        ({"format": "yue2-lora-v1", "alpha": "6"}, {}, 2.0),
        ({"format": "yue2-lora-v1", "alpha": "6"}, {f"{o}.alpha": torch.tensor(1.5)}, 0.5),
        ({"format": "comfyui-native-lora", "alpha": "6"}, {}, 1.0),
    ]
    for i, (meta, extra, scale) in enumerate(cases):
        path = tmp_path / f"case{i}.safetensors"
        save_file({**pair, **extra}, str(path), metadata=meta)
        (patch,) = read_lora_v1(path)[1]
        assert patch.linears == (o,) and patch.scale == scale, (meta, extra, patch.scale)


def test_lora_v1_reader_rejects_ar_keys_of_a_joint_comfy_export(tmp_path):
    sd, meta = fm.native_to_comfy(make_native(branches=("ar", "nar"), layers=1))
    path = tmp_path / "joint.comfy.safetensors"
    fm.save_lora_file(str(path), sd, meta)
    with pytest.raises(ValueError, match="not a NAR layer or I/O module"):
        read_lora_v1(path)


def test_lora_v1_reader_rejects_ar_hf_export(tmp_path):
    sd, meta = fm.native_to_hf(fm.filter_branch(make_native(branches=("ar", "nar"), layers=1), "ar"), branch="ar")
    path = tmp_path / "ar.hf.safetensors"
    fm.save_lora_file(str(path), sd, meta)
    with pytest.raises(ValueError, match="no NAR projection"):
        read_lora_v1(path)


def test_comfy_key_report_flags_stray_keys():
    from gpu.yue2_comfy_export_check import comfy_key_report, expected_patches, report_ok

    native = make_native(branches=("nar",), io="diff", layers=1)
    comfy_sd, _ = fm.native_to_comfy(native)
    shapes = {}
    for name, (dw, db) in expected_patches(native).items():
        target = "model." + name if name.startswith("diffusion_model.") else name
        shapes[target + ".weight"] = tuple(dw.shape)
        if db is not None:
            shapes[target + ".bias"] = tuple(db.shape)
    clean = comfy_key_report(comfy_sd, shapes)
    assert clean["unmapped"] == [] and clean["shape_errors"] == [] and clean["stray_keys"] == []
    assert report_ok({"header": clean, "comfy": "skipped"})
    comfy_sd["diffusion_model.model.layers.0.self_attn.qkv_proj.lora_mid.weight"] = torch.zeros(1)
    stray = comfy_key_report(comfy_sd, shapes)
    assert stray["stray_keys"] == ["diffusion_model.model.layers.0.self_attn.qkv_proj.lora_mid.weight"]
    assert not report_ok({"header": stray, "comfy": "skipped"})
