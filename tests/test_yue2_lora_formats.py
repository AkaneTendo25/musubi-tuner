import os
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from musubi_tuner.networks import convert_yue2_lora  # noqa: E402
from musubi_tuner.yue2 import yue2_lora_formats as fm  # noqa: E402
from musubi_tuner.yue2.yue2_checkpoint import check_merge_input  # noqa: E402
from musubi_tuner.yue2.yue2_model import YuE2Config  # noqa: E402

CFG = YuE2Config.tiny()
H = CFG.hidden_size
Q_DIM, KV_DIM, INTER = CFG.q_dim, CFG.kv_dim, CFG.intermediate_size
BLOCK_SHAPES = {
    "self_attn.qkv_proj": (Q_DIM + 2 * KV_DIM, H),
    "self_attn.o_proj": (H, Q_DIM),
    "mlp.gate_up_proj": (2 * INTER, H),
    "mlp.down_proj": (H, INTER),
}
IO_SHAPES = {
    "nar.vae2llm": (H, CFG.latent_dim),
    "nar.llm2vae": (CFG.latent_dim, H),
    "nar.time_embedder.mlp.0": (H, CFG.time_freq_dim),
    "nar.time_embedder.mlp.2": (H, H),
}


def make_native(
    seed=0,
    branches=("ar", "nar"),
    split=True,
    rank=4,
    alpha=2.0,
    io=None,
    io_bias=True,
    time_embedder=False,
    layers=None,
) -> fm.NativeLoRA:
    """Random canonical LoRA on the tiny config (non-zero ups, so every delta is non-trivial)."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g) * 0.1

    modules = {}
    for branch in branches:
        for i in range(CFG.num_layers if layers is None else layers):
            for sub, (out_f, in_f) in BLOCK_SHAPES.items():
                if split and sub in fm.SPLIT_ROLES:
                    dims = fm.split_dims_for(sub, CFG)
                    md = fm.ModuleDelta("lora", [rnd(rank, in_f) for _ in dims], [rnd(d, rank) for d in dims], alpha)
                else:
                    md = fm.ModuleDelta("lora", [rnd(rank, in_f)], [rnd(out_f, rank)], alpha)
                modules[f"{branch}.blocks.{i}.{sub}"] = md
    if io is not None:
        paths = ["nar.vae2llm", "nar.llm2vae"] + (["nar.time_embedder.mlp.0", "nar.time_embedder.mlp.2"] if time_embedder else [])
        for path in paths:
            out_f, in_f = IO_SHAPES[path]
            if io == "lora":
                modules[path] = fm.ModuleDelta("lora", [rnd(rank, in_f)], [rnd(out_f, rank)], alpha)
            else:
                modules[path] = fm.ModuleDelta("diff", diff=rnd(out_f, in_f), diff_b=rnd(out_f) if io_bias else None)
    return fm.NativeLoRA(modules)


def assert_same_deltas(expected: fm.NativeLoRA, got: fm.NativeLoRA, atol=1e-6):
    e, g = fm.weight_deltas(expected), fm.weight_deltas(got)
    assert set(e) == set(g), (sorted(set(e) - set(g))[:5], sorted(set(g) - set(e))[:5])
    for path, (w, b) in e.items():
        gw, gb = g[path]
        assert gw.shape == w.shape, path
        assert torch.allclose(gw, w, atol=atol, rtol=0), (path, (gw - w).abs().max().item())
        if b is None:
            assert gb is None or gb.abs().max() == 0, path
        else:
            assert gb is not None and torch.allclose(gb, b, atol=atol, rtol=0), path


# per-layer order of the Mothersuperior ``.pt`` ``lora`` list, as its checkpoint writer lays it out (q, k, v, o,
# gate, up, down); deliberately a literal and not fm.MS_PT_ORDER, so a wrong production constant fails the tests
MS_RELEASE_ORDER = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)


def ms_layer_keys(layer, nar=True):
    prefix = "nar_" if nar else ""
    return [f"layers.{layer}.{prefix}{block}.{proj}" for block, proj in MS_RELEASE_ORDER]


def ms_pt_from_sd(sd, layers, rank, order=MS_RELEASE_ORDER):
    """The Mothersuperior ``.pt`` dict built from its safetensors keys, in ``order``."""
    values = []
    for layer in range(layers):
        for block, proj in order:
            key = f"layers.{layer}.nar_{block}.{proj}"
            values += [sd[key + ".lora_A"], sd[key + ".lora_B"]]
    return {
        "lora": values,
        "rank": rank,
        "io": {m: {p: sd[f"{m}.{p}"] for p in ("weight", "bias")} for m in ("vae2llm", "llm2vae")},
    }


def swapped_order(i, j):
    order = list(MS_RELEASE_ORDER)
    order[i], order[j] = order[j], order[i]
    return order


# ----------------------------------------------------------------------------------------------------------------------
# detection


def test_detect_every_format():
    native = make_native(io="diff")
    sd = fm.native_state_dict(native)
    assert fm.detect_format(sd) == "native"
    comfy, meta = fm.native_to_comfy(native)
    assert fm.detect_format(comfy, meta) == "comfy"
    aitk = {
        k.replace("diffusion_model.", "transformer.nar.").replace("text_encoders.", "transformer.ar."): v for k, v in comfy.items()
    }
    assert fm.detect_format(aitk) == "aitk"
    hf, meta = fm.native_to_hf(fm.filter_branch(make_native(io="lora"), "nar"))
    assert fm.detect_format(hf, meta) == "hf"
    fl = fm.native_to_fl(native)
    assert fm.detect_format(*fl["nar"]) == "fl" and fm.detect_format(*fl["ar"]) == "fl"
    ms = {k + ".lora_A": torch.zeros(4, 8) for k in ms_layer_keys(0)}
    assert fm.detect_format(ms) == "ms_safetensors"
    assert fm.detect_format({"lora": [torch.zeros(1)], "rank": 4}) == "ms_pt"
    with pytest.raises(ValueError):
        fm.detect_format({"something.else": torch.zeros(1)})


def test_native_state_dict_roundtrip_and_alpha_always_present():
    for split in (True, False):
        native = make_native(split=split, io="diff")
        sd = fm.native_state_dict(native)
        lora_names = {k.split(".")[0] for k in sd if ".lora_down" in k}
        assert all(name + ".alpha" in sd for name in lora_names)
        if split:
            assert "lora_unet_nar_blocks_0_self_attn_qkv_proj.lora_down.2.weight" in sd
            assert "lora_unet_nar_blocks_0_mlp_gate_up_proj.lora_up.1.weight" in sd
            assert "lora_unet_nar_blocks_0_self_attn_o_proj.lora_down.weight" in sd
        assert "lora_unet_nar_vae2llm.diff" in sd and "lora_unet_nar_llm2vae.diff_b" in sd
        back = fm.to_native(sd)
        assert_same_deltas(native, back, atol=0)
        assert fm.native_state_dict(back).keys() == sd.keys()


def test_native_missing_alpha_means_rank():
    native = make_native(split=True, alpha=None, layers=1, branches=("nar",))
    sd = {k: v for k, v in fm.native_state_dict(native).items() if not k.endswith(".alpha")}
    back = fm.to_native(sd)
    for md in back.modules.values():
        assert md.alpha == md.rank and md.scale == 1.0


# ----------------------------------------------------------------------------------------------------------------------
# comfy / aitk


def test_comfy_roundtrip_block_diagonal():
    native = make_native(io="diff", rank=4, alpha=2.0)
    comfy, meta = fm.native_to_comfy(native)
    assert meta["format"] == "comfyui-native-lora" and meta["yue2_lora_branch"] == "ar+nar"
    assert comfy["diffusion_model.model.layers.0.self_attn.qkv_proj.lora_down.weight"].shape == (12, H)
    assert comfy["diffusion_model.model.layers.0.self_attn.qkv_proj.lora_up.weight"].shape == (Q_DIM + 2 * KV_DIM, 12)
    assert comfy["diffusion_model.model.layers.0.self_attn.qkv_proj.alpha"].item() == 12.0
    assert comfy["text_encoders.model.layers.2.mlp.gate_up_proj.alpha"].item() == 8.0
    assert comfy["text_encoders.model.layers.2.mlp.down_proj.alpha"].item() == 2.0  # fused modules keep their alpha
    assert "diffusion_model.vae2llm.diff" in comfy and "diffusion_model.llm2vae.diff_b" in comfy
    back = fm.to_native(comfy, meta)
    assert back.source_format == "comfy"
    assert_same_deltas(native, back)
    # block-diagonal fused modules decompose back to the original split pieces exactly
    split = fm.fused_to_split(back, CFG)
    md = split.modules["nar.blocks.1.self_attn.qkv_proj"]
    assert md.is_split and md.rank == 4 and len(md.downs) == 3
    assert_same_deltas(native, split)


def test_split_to_fused_alpha_is_fused_rank():
    native = make_native(rank=3, alpha=1.5, branches=("nar",))
    fused = fm.split_to_fused(native)
    qkv = fused.modules["nar.blocks.0.self_attn.qkv_proj"]
    gu = fused.modules["nar.blocks.0.mlp.gate_up_proj"]
    assert not qkv.is_split and qkv.rank == 9 and qkv.alpha == 9.0
    assert gu.rank == 6 and gu.alpha == 6.0
    assert fused.modules["nar.blocks.0.self_attn.o_proj"].alpha == 1.5
    assert_same_deltas(native, fused)


def test_fused_to_split_shared_down_is_exact():
    native = make_native(split=False, rank=4, alpha=4.0, branches=("nar",), layers=1)
    split = fm.fused_to_split(native, CFG)
    md = split.modules["nar.blocks.0.self_attn.qkv_proj"]
    assert md.is_split and md.rank == 4 and all(d is md.downs[0] for d in md.downs)
    assert_same_deltas(native, split)
    with pytest.raises(ValueError):
        fm.fused_module_to_split(
            native.modules["nar.blocks.0.self_attn.qkv_proj"], fm.split_dims_for("self_attn.qkv_proj", CFG), rank=3
        )


def test_aitk_peft_keys_without_alpha_scale_one():
    native = make_native(split=False, rank=4, alpha=4.0, layers=1)  # scale 1 already
    comfy, _ = fm.native_to_comfy(native)
    aitk = {}
    for k, v in comfy.items():
        if k.endswith(".alpha"):
            continue
        k = k.replace("lora_down.weight", "lora_A.weight").replace("lora_up.weight", "lora_B.weight")
        aitk[k.replace("diffusion_model.", "transformer.nar.").replace("text_encoders.", "transformer.ar.")] = v
    back = fm.to_native(aitk)
    assert back.source_format == "aitk"
    assert_same_deltas(native, back)
    peft_comfy = {
        k.replace("transformer.nar.", "diffusion_model.").replace("transformer.ar.", "text_encoders."): v for k, v in aitk.items()
    }
    assert_same_deltas(native, fm.to_native(peft_comfy))


# ----------------------------------------------------------------------------------------------------------------------
# hf (yue2-lora-v1) / fl


def test_hf_roundtrip_nar_and_ar():
    native = make_native(io="lora", time_embedder=True, rank=4, alpha=8.0)
    nar = fm.filter_branch(native, "nar")
    sd, meta = fm.native_to_hf(nar)
    assert meta["format"] == "yue2-lora-v1" and meta["rank"] == "4" and meta["alpha"] == "4"
    assert "model.layers.0.nar_self_attn.k_proj.lora_up.weight" in sd
    assert sd["model.layers.0.nar_self_attn.k_proj.lora_up.weight"].shape == (KV_DIM, 4)
    assert "time_embedder.mlp.2.lora_down.weight" in sd and not any(k.endswith(".alpha") for k in sd)
    assert_same_deltas(nar, fm.to_native(sd, meta))
    with pytest.raises(ValueError, match="NAR-only"):
        fm.native_to_hf(native)
    ar_sd, ar_meta = fm.native_to_hf(native, branch="ar")
    assert "model.layers.1.self_attn.q_proj.lora_down.weight" in ar_sd
    assert_same_deltas(fm.filter_branch(native, "ar"), fm.to_native(ar_sd, ar_meta))


def test_hf_fused_module_is_split_exactly_and_diff_factorized():
    native = make_native(split=False, rank=5, alpha=2.0, branches=("nar",), io="diff", io_bias=False)
    sd, meta = fm.native_to_hf(native)
    assert "rank" not in meta  # ranks differ between blocks and the factorized I/O diffs: every key has scale 1
    assert_same_deltas(native, fm.to_native(sd, meta), atol=1e-5)
    with_bias = make_native(branches=("nar",), io="diff", io_bias=True, layers=1)
    with pytest.raises(ValueError, match="bias"):
        fm.native_to_hf(with_bias)


def test_hf_yue2_lora_v1_metadata_alpha():
    g = torch.Generator().manual_seed(1)
    sd = {}
    for proj, out_f in (("q_proj", Q_DIM), ("k_proj", KV_DIM), ("v_proj", KV_DIM), ("o_proj", H)):
        sd[f"model.layers.0.nar_self_attn.{proj}.lora_down.weight"] = torch.randn(4, H, generator=g)
        sd[f"model.layers.0.nar_self_attn.{proj}.lora_up.weight"] = torch.randn(out_f, 4, generator=g)
    o_expected = (
        sd["model.layers.0.nar_self_attn.o_proj.lora_up.weight"] @ sd["model.layers.0.nar_self_attn.o_proj.lora_down.weight"]
    )
    native = fm.to_native(sd, {"format": "yue2-lora-v1", "alpha": "8", "rank": "4"})
    assert torch.allclose(native.modules["nar.blocks.0.self_attn.o_proj"].weight_delta(), o_expected * 2, atol=1e-5)
    native = fm.to_native(sd, {"format": "yue2-lora-v1"})
    assert torch.allclose(native.modules["nar.blocks.0.self_attn.o_proj"].weight_delta(), o_expected, atol=1e-5)
    # fl-yue2-lora-v1 files have no scale in the metadata; a per-key alpha always wins
    sd["model.layers.0.nar_self_attn.o_proj.alpha"] = torch.tensor(2.0)
    native = fm.to_native(sd, {"format": "yue2-lora-v1", "alpha": "8"})
    assert torch.allclose(native.modules["nar.blocks.0.self_attn.o_proj"].weight_delta(), o_expected * 0.5, atol=1e-5)


@pytest.mark.parametrize("order", [("k", "q", "v"), ("v", "k", "q")])
def test_hf_qkv_grouped_by_role_not_key_order(order):
    g = torch.Generator().manual_seed(2)
    parts = {
        r: (torch.randn(4, H, generator=g), torch.randn(d, 4, generator=g)) for r, d in (("q", Q_DIM), ("k", KV_DIM), ("v", KV_DIM))
    }
    sd = {}
    for role in order:
        down, up = parts[role]
        sd[f"model.layers.0.nar_self_attn.{role}_proj.lora_down.weight"] = down
        sd[f"model.layers.0.nar_self_attn.{role}_proj.lora_up.weight"] = up
    delta = fm.to_native(sd).modules["nar.blocks.0.self_attn.qkv_proj"].weight_delta()
    expected = torch.cat([parts[r][1] @ parts[r][0] for r in ("q", "k", "v")], dim=0)
    assert torch.allclose(delta, expected, atol=1e-5)


def test_hf_unequal_ranks_and_missing_roles():
    g = torch.Generator().manual_seed(3)
    sd = {
        "model.layers.0.nar_self_attn.q_proj.lora_down.weight": torch.randn(2, H, generator=g),
        "model.layers.0.nar_self_attn.q_proj.lora_up.weight": torch.randn(Q_DIM, 2, generator=g),
        "model.layers.0.nar_self_attn.v_proj.lora_down.weight": torch.randn(6, H, generator=g),
        "model.layers.0.nar_self_attn.v_proj.lora_up.weight": torch.randn(KV_DIM, 6, generator=g),
        "model.layers.0.nar_self_attn.v_proj.alpha": torch.tensor(3.0),
        "model.layers.0.nar_mlp.up_proj.lora_down.weight": torch.randn(4, H, generator=g),
        "model.layers.0.nar_mlp.up_proj.lora_up.weight": torch.randn(INTER, 4, generator=g),
    }
    native = fm.to_native(sd, config=CFG)
    qkv = native.modules["nar.blocks.0.self_attn.qkv_proj"]
    expected = torch.cat(
        [
            sd["model.layers.0.nar_self_attn.q_proj.lora_up.weight"] @ sd["model.layers.0.nar_self_attn.q_proj.lora_down.weight"],
            torch.zeros(KV_DIM, H),
            0.5
            * sd["model.layers.0.nar_self_attn.v_proj.lora_up.weight"]
            @ sd["model.layers.0.nar_self_attn.v_proj.lora_down.weight"],
        ]
    )
    assert not qkv.is_split and qkv.rank == 2 + 2 + 6  # unequal ranks -> exact block-diagonal fusion
    assert torch.allclose(qkv.weight_delta(), expected, atol=1e-5)
    gu = native.modules["nar.blocks.0.mlp.gate_up_proj"]
    assert gu.is_split and torch.equal(gu.weight_delta()[:INTER], torch.zeros(INTER, H))


def test_fl_roundtrip_per_branch():
    native = make_native(io="diff", io_bias=True, rank=4, alpha=2.0)
    out = fm.native_to_fl(native)
    assert set(out) == {"ar", "nar"}
    nar_sd, nar_meta = out["nar"]
    assert nar_meta == {"format": "fl-yue2-lora-v1", "branch": "nar", "source_format": "musubi-yue2", "rank": "4"}
    assert (
        "vae2llm.diff" in nar_sd
        and "llm2vae.diff_b" in nar_sd
        and not any(k.startswith("model.layers.0.self_attn") for k in nar_sd)
    )
    assert not any(k.endswith(".alpha") for k in nar_sd)
    merged = fm.to_native(nar_sd, nar_meta)
    merged.modules.update(fm.to_native(*out["ar"]).modules)
    assert_same_deltas(native, merged)
    # an I/O LoRA is written as its product diff; fl-yue2-lora-v1 has no time-embedder target
    io_lora = make_native(io="lora", branches=("nar",), layers=1)
    assert_same_deltas(io_lora, fm.to_native(*fm.native_to_fl(io_lora)["nar"]))
    with pytest.raises(ValueError, match="time_embedder"):
        fm.native_to_fl(make_native(io="lora", time_embedder=True, branches=("nar",), layers=1))


# ----------------------------------------------------------------------------------------------------------------------
# Mothersuperior


def _ms_checkpoint(seed=4, rank=4, layers=2, nar=True):
    g = torch.Generator().manual_seed(seed)
    dims = {
        "q_proj": (Q_DIM, H),
        "k_proj": (KV_DIM, H),
        "v_proj": (KV_DIM, H),
        "o_proj": (H, Q_DIM),
        "gate_proj": (INTER, H),
        "up_proj": (INTER, H),
        "down_proj": (H, INTER),
    }
    values = []
    for _ in range(layers):
        for _, proj in MS_RELEASE_ORDER:
            out_f, in_f = dims[proj]
            values += [torch.randn(rank, in_f, generator=g), torch.randn(out_f, rank, generator=g)]
    ckpt = {"lora": values, "rank": rank}
    if nar:
        ckpt["io"] = {
            "vae2llm": {"weight": torch.randn(H, CFG.latent_dim, generator=g), "bias": torch.randn(H, generator=g)},
            "llm2vae": {"weight": torch.randn(CFG.latent_dim, H, generator=g), "bias": torch.randn(CFG.latent_dim, generator=g)},
        }
    return ckpt


def _base_io(seed=5):
    g = torch.Generator().manual_seed(seed)
    return {
        "nar.vae2llm.weight": torch.randn(H, CFG.latent_dim, generator=g),
        "nar.vae2llm.bias": torch.randn(H, generator=g),
        "nar.llm2vae.weight": torch.randn(CFG.latent_dim, H, generator=g),
        "nar.llm2vae.bias": torch.randn(CFG.latent_dim, generator=g),
    }


def test_ms_pt_list_order_q_k_v_o_gate_up_down():
    ckpt = _ms_checkpoint()
    native = fm.to_native(ckpt)
    assert native.source_format == "ms_pt"
    lora = ckpt["lora"]

    def prod(layer, j):
        a, b = lora[(layer * 7 + j) * 2], lora[(layer * 7 + j) * 2 + 1]
        return b @ a

    for layer in range(2):
        qkv = native.modules[f"nar.blocks.{layer}.self_attn.qkv_proj"].weight_delta()
        assert torch.allclose(qkv, torch.cat([prod(layer, 0), prod(layer, 1), prod(layer, 2)]), atol=1e-4)
        assert torch.allclose(native.modules[f"nar.blocks.{layer}.self_attn.o_proj"].weight_delta(), prod(layer, 3), atol=1e-4)
        gu = native.modules[f"nar.blocks.{layer}.mlp.gate_up_proj"].weight_delta()
        assert torch.allclose(gu, torch.cat([prod(layer, 4), prod(layer, 5)]), atol=1e-4)
        assert torch.allclose(native.modules[f"nar.blocks.{layer}.mlp.down_proj"].weight_delta(), prod(layer, 6), atol=1e-4)
    assert native.modules["nar.vae2llm"].kind == "full"
    with pytest.raises(ValueError, match="base_io"):
        fm.native_state_dict(native)
    ar = fm.to_native(_ms_checkpoint(nar=False))
    assert all(p.startswith("ar.blocks.") for p in ar.modules)


def test_ms_full_io_to_diff_with_base_io():
    ckpt = _ms_checkpoint()
    base = _base_io()
    native = fm.to_native(ckpt, base_io=base)
    md = native.modules["nar.llm2vae"]
    assert md.kind == "diff"
    assert torch.allclose(md.diff, ckpt["io"]["llm2vae"]["weight"] - base["nar.llm2vae.weight"])
    assert torch.allclose(md.diff_b, ckpt["io"]["llm2vae"]["bias"] - base["nar.llm2vae.bias"])
    sd = fm.native_fused_state_dict(native)
    assert check_merge_input(sd)["lora_unet_nar_vae2llm"] == "diff"


def test_ms_safetensors_equals_ms_pt():
    ckpt = _ms_checkpoint()
    sd = fm._ms_pt_to_sd(ckpt)
    assert "layers.1.nar_mlp.down_proj.lora_B" in sd and "vae2llm.bias" in sd
    base = _base_io()
    assert fm.detect_format(sd) == "ms_safetensors"
    assert_same_deltas(fm.to_native(ckpt, base_io=base), fm.to_native(sd, base_io=base), atol=0)


def test_ms_pt_order_is_release_order():
    # safetensors keys and the .pt list are built from the literal release order only, never from fm
    values = _ms_checkpoint()["lora"]
    sd = {}
    for layer in range(2):
        for j, key in enumerate(ms_layer_keys(layer)):
            sd[key + ".lora_A"], sd[key + ".lora_B"] = values[(layer * 7 + j) * 2], values[(layer * 7 + j) * 2 + 1]
    for name, value in _base_io(seed=6).items():
        sd[name[len("nar.") :]] = value
    base = _base_io()
    expected = fm.to_native(sd, base_io=base)
    assert expected.source_format == "ms_safetensors"
    got = fm.to_native(ms_pt_from_sd(sd, layers=2, rank=4), base_io=base)
    assert got.source_format == "ms_pt"
    assert_same_deltas(expected, got, atol=0)
    # the check is sensitive to the order: a .pt list in any other order gives other deltas
    for i, j, path in ((1, 2, "nar.blocks.0.self_attn.qkv_proj"), (4, 5, "nar.blocks.1.mlp.gate_up_proj")):
        wrong = fm.to_native(ms_pt_from_sd(sd, layers=2, rank=4, order=swapped_order(i, j)), base_io=base)
        assert not torch.allclose(wrong.modules[path].weight_delta(), expected.modules[path].weight_delta()), (i, j)


def test_ms_pt_file_roundtrip(tmp_path):
    ckpt = _ms_checkpoint()
    path = str(tmp_path / "nar_lora.pt")
    torch.save(ckpt, path)
    sd, meta = fm.load_lora_file(path)
    assert meta == {}
    assert_same_deltas(fm.to_native(ckpt, base_io=_base_io()), fm.to_native(sd, base_io=_base_io()), atol=0)


# ----------------------------------------------------------------------------------------------------------------------
# concat, merge input, scaling


def test_concat_rank_equals_sum_of_deltas():
    a = make_native(seed=1, rank=4, alpha=2.0, io="diff")
    b = make_native(seed=2, rank=3, alpha=6.0, split=False, io="lora")
    c = fm.concat_rank(a, b)
    ea, eb, ec = fm.weight_deltas(a), fm.weight_deltas(b), fm.weight_deltas(c)
    for path in ea:
        assert torch.allclose(ec[path][0], ea[path][0] + eb[path][0], atol=1e-5), path
    assert c.modules["nar.vae2llm"].kind == "diff"
    assert torch.allclose(c.modules["nar.vae2llm"].diff_b, a.modules["nar.vae2llm"].diff_b)
    qkv = c.modules["nar.blocks.0.self_attn.qkv_proj"]
    assert not qkv.is_split and qkv.rank == 3 * 4 + 3 and qkv.scale == 1.0  # split + fused -> fused
    both_split = fm.concat_rank(a, make_native(seed=3, rank=2, alpha=1.0))
    qkv = both_split.modules["ar.blocks.2.self_attn.qkv_proj"]
    assert qkv.is_split and qkv.rank == 6
    # only-in-one modules are copied
    only_nar = fm.concat_rank(fm.filter_branch(a, "ar"), fm.filter_branch(b, "nar"))
    assert set(only_nar.modules) == set(a.modules) | set(b.modules)


def test_native_fused_is_the_merge_input_contract():
    native = make_native(io="lora", time_embedder=True)
    sd = fm.native_fused_state_dict(native)
    kinds = check_merge_input(sd)
    assert kinds["lora_unet_nar_time_embedder_mlp_0"] == "diff"
    assert kinds["lora_unet_ar_blocks_0_self_attn_qkv_proj"] == "lora"
    assert not any(".lora_down.0." in k for k in sd)
    assert_same_deltas(native, fm.to_native(sd))


def test_scale_branches():
    native = make_native(io="diff")
    scaled = fm.scale_branches(native, ar=0.5, nar=2.0, io=0.0)
    e, s = fm.weight_deltas(native), fm.weight_deltas(scaled)
    assert torch.allclose(s["ar.blocks.0.mlp.down_proj"][0], 0.5 * e["ar.blocks.0.mlp.down_proj"][0])
    assert torch.allclose(s["nar.blocks.0.self_attn.qkv_proj"][0], 2.0 * e["nar.blocks.0.self_attn.qkv_proj"][0])
    assert s["nar.vae2llm"][0].abs().max() == 0


def test_path_and_name_mapping():
    for path in ("ar.blocks.27.self_attn.qkv_proj", "nar.blocks.3.mlp.gate_up_proj", "nar.vae2llm", "nar.time_embedder.mlp.2"):
        assert fm.path_of(fm.lora_name_of(path)) == path
    with pytest.raises(ValueError):
        fm.path_of("lora_unet_ar_vae2llm")


# ----------------------------------------------------------------------------------------------------------------------
# converter CLI


@pytest.mark.parametrize("target", ["native", "comfy", "hf", "fl"])
def test_convert_cli_verify(tmp_path, target):
    native = make_native(io="diff" if target != "hf" else "lora", rank=4, alpha=2.0)
    src = str(tmp_path / "in.safetensors")
    fm.save_lora_file(src, fm.native_state_dict(native), {"ss_network_module": "networks.lora_yue2"})
    out = str(tmp_path / "out.safetensors")
    argv = ["--input", src, "--output", out, "--to", target, "--verify"]
    if target == "hf":
        argv += ["--branch", "nar"]
    convert_yue2_lora.main(argv)
    if target == "fl":
        assert os.path.exists(str(tmp_path / "out-ar.safetensors")) and os.path.exists(str(tmp_path / "out-nar.safetensors"))
    else:
        assert os.path.exists(out)


def test_convert_cli_concat_ms_base_model_and_bf16(tmp_path, monkeypatch):
    ckpt = _ms_checkpoint()
    pt = str(tmp_path / "ms.pt")
    torch.save(ckpt, pt)
    base = _base_io()
    import musubi_tuner.yue2.yue2_checkpoint as ckpt_mod

    monkeypatch.setattr(ckpt_mod, "read_base_io", lambda path: base)
    other = make_native(seed=9, branches=("nar",), layers=2)
    companion = str(tmp_path / "companion.safetensors")
    fm.save_lora_file(companion, fm.native_state_dict(other))
    out = str(tmp_path / "comfy.safetensors")
    with pytest.raises(ValueError, match="base_model"):
        convert_yue2_lora.main(["--input", pt, "--output", out, "--to", "comfy"])
    convert_yue2_lora.main(
        ["--input", pt, "--output", out, "--to", "comfy", "--base_model", "x", "--concat", companion, "--verify"]
    )
    sd, meta = fm.load_lora_file(out)
    expected = fm.concat_rank(fm.to_native(ckpt, base_io=base), other)
    assert_same_deltas(expected, fm.to_native(sd, meta), atol=1e-5)
    convert_yue2_lora.main(["--input", pt, "--output", out, "--to", "comfy", "--base_model", "x", "--dtype", "bf16", "--verify"])
    sd, _ = fm.load_lora_file(out)
    assert sd["diffusion_model.model.layers.0.self_attn.qkv_proj.lora_up.weight"].dtype == torch.bfloat16


# ----------------------------------------------------------------------------------------------------------------------
# real community adapters (header shapes against the Comfy bf16 checkpoint); runs where the weights are


WEIGHTS = os.environ.get("YUE2_WEIGHTS_DIR")
MS_DIR = os.path.join(WEIGHTS or "", "Mothersuperior__yue2-mothersuperior-realaudio-tokenizer-v4")
COMFY_BF16 = os.path.join(WEIGHTS or "", "Comfy-Org__YuE2", "checkpoints", "yue2_3b_bf16.safetensors")
real_weights = pytest.mark.skipif(
    not WEIGHTS or not os.path.isdir(MS_DIR) or not os.path.isfile(COMFY_BF16),
    reason="YUE2_WEIGHTS_DIR with the community adapters not set",
)


def _comfy_checkpoint_key(path: str) -> str:
    if path.startswith("ar.blocks."):
        _, _, rest = path.partition("ar.blocks.")
        return f"text_encoders.model.layers.{rest}.weight"
    if path.startswith("nar.blocks."):
        _, _, rest = path.partition("nar.blocks.")
        return f"model.diffusion_model.model.layers.{rest}.weight"
    return "model.diffusion_model." + path[len("nar.") :] + ".weight"


def _real_base_io():
    from musubi_tuner.yue2 import yue2_checkpoint

    try:
        return yue2_checkpoint.read_base_io(COMFY_BF16)
    except NotImplementedError:
        from safetensors import safe_open

        out = {}
        with safe_open(COMFY_BF16, framework="pt") as f:
            for m in ("vae2llm", "llm2vae"):
                for p in ("weight", "bias"):
                    out[f"nar.{m}.{p}"] = f.get_tensor(f"model.diffusion_model.{m}.{p}").float()
        return out


def _check_against_header(native: fm.NativeLoRA, header_shapes: dict):
    for path, md in native.modules.items():
        key = _comfy_checkpoint_key(path)
        assert key in header_shapes, key
        w = md.weight_delta()
        assert tuple(w.shape) == tuple(header_shapes[key]), (path, tuple(w.shape), header_shapes[key])
        assert torch.isfinite(w).all(), path
        if md.bias_delta() is not None:
            assert tuple(md.bias_delta().shape) == (header_shapes[key][0],)


@real_weights
def test_real_community_adapters_load_into_native():
    from safetensors import safe_open

    with safe_open(COMFY_BF16, framework="pt") as f:
        header_shapes = {k: f.get_slice(k).get_shape() for k in f.keys() if k.endswith(".weight")}
    full_cfg = YuE2Config()
    base_io = _real_base_io()

    comfy_sd, comfy_meta = fm.load_lora_file(os.path.join(MS_DIR, "nar_lora_joint_v9_comfyui.safetensors"))
    comfy = fm.to_native(comfy_sd, comfy_meta, config=full_cfg)
    assert comfy.source_format == "comfy" and len(comfy.modules) == 28 * 4 + 2
    _check_against_header(comfy, header_shapes)
    # the Mothersuperior Comfy file is block-diagonal (rank 3x32 / 2x32): it splits back into rank-32 pieces exactly
    split = fm.fused_to_split(comfy, full_cfg)
    qkv = split.modules["nar.blocks.0.self_attn.qkv_proj"]
    assert qkv.is_split and qkv.rank == 32 and len(qkv.downs) == 3
    assert split.modules["nar.blocks.5.mlp.gate_up_proj"].rank == 32

    ms_sd, ms_meta = fm.load_lora_file(os.path.join(MS_DIR, "nar_lora_joint_v9.safetensors"))
    ms = fm.to_native(ms_sd, ms_meta, base_io=base_io, config=full_cfg)
    assert ms.source_format == "ms_safetensors" and len(ms.modules) == 28 * 4 + 2
    assert ms.modules["nar.blocks.0.self_attn.qkv_proj"].is_split and ms.modules["nar.vae2llm"].kind == "diff"
    _check_against_header(ms, header_shapes)
    # the Comfy repack and the full-weight file describe the same adapter (bf16 LoRA factors in the Comfy file)
    ec, em = fm.weight_deltas(comfy), fm.weight_deltas(ms)
    worst = 0.0
    for path, (w, b) in em.items():
        rel = (ec[path][0] - w).abs().max().item() / max(w.abs().max().item(), 1e-12)
        worst = max(worst, rel)
        if b is not None:
            assert (ec[path][1] - b).abs().max().item() <= 2e-2 * max(b.abs().max().item(), 1e-6), path
    assert worst < 2e-2, worst

    # merge input and export paths accept the real adapters
    check_merge_input(fm.native_fused_state_dict(ms))
    sd = fm.native_state_dict(ms)
    assert sd["lora_unet_nar_blocks_0_self_attn_qkv_proj.lora_down.0.weight"].shape == (32, 2048)
    assert sd["lora_unet_nar_blocks_0_self_attn_qkv_proj.alpha"].item() == 32.0
    out, _ = fm.native_to_comfy(ms)
    assert out["diffusion_model.model.layers.0.self_attn.qkv_proj.lora_up.weight"].shape == (4096, 96)

    # v4: the .pt layout (list order q, k, v, o, gate, up, down + io) against its safetensors release
    v4_sd, v4_meta = fm.load_lora_file(os.path.join(MS_DIR, "nar_lora_joint_v4.safetensors"))
    v4 = fm.to_native(v4_sd, v4_meta, base_io=base_io, config=full_cfg)
    _check_against_header(v4, header_shapes)
    pt_path = os.path.join(MS_DIR, "nar_lora_joint_v4.pt")
    if os.path.isfile(pt_path):
        ckpt, _ = fm.load_lora_file(pt_path)
    else:  # rebuild the .pt dict from the safetensors release in the literal release order (not fm.MS_PT_ORDER)
        ckpt = ms_pt_from_sd(v4_sd, layers=28, rank=32)
    v4_pt = fm.to_native(ckpt, base_io=base_io, config=full_cfg)
    assert v4_pt.source_format == "ms_pt"
    assert_same_deltas(v4, v4_pt, atol=1e-5)
    wrong = fm.to_native(ms_pt_from_sd(v4_sd, layers=28, rank=32, order=swapped_order(1, 2)), base_io=base_io, config=full_cfg)
    path = "nar.blocks.0.self_attn.qkv_proj"
    assert not torch.allclose(wrong.modules[path].weight_delta(), v4.modules[path].weight_delta(), atol=1e-5)


def _ms_adapter_files():
    try:
        names = os.listdir(MS_DIR)
    except OSError:
        return []
    return sorted(n for n in names if re.fullmatch(r"nar_lora_joint_v\d+(_comfyui)?(\.bf16)?\.safetensors", n))


@pytest.fixture(scope="module")
def real_reference():
    from safetensors import safe_open

    with safe_open(COMFY_BF16, framework="pt") as f:
        header_shapes = {k: f.get_slice(k).get_shape() for k in f.keys() if k.endswith(".weight")}
    return header_shapes, _real_base_io(), {}


@real_weights
@pytest.mark.parametrize("name", _ms_adapter_files() or ["no adapter files found"])
def test_real_community_adapter_file(real_reference, name):
    """Every Mothersuperior release file loads, fits the checkpoint and equals its full-precision ``.safetensors``."""
    header_shapes, base_io, ref_cache = real_reference
    full_cfg = YuE2Config()
    sd, meta = fm.load_lora_file(os.path.join(MS_DIR, name))
    native = fm.to_native(sd, meta, base_io=base_io, config=full_cfg)
    assert native.source_format == ("comfy" if "_comfyui" in name else "ms_safetensors")
    assert len(native.modules) == 28 * 4 + 2
    _check_against_header(native, header_shapes)
    check_merge_input(fm.native_fused_state_dict(native))

    version = re.match(r"nar_lora_joint_(v\d+)", name).group(1)
    got = fm.weight_deltas(native)
    if name == f"nar_lora_joint_{version}.safetensors":
        ref_cache[version] = got
    if version not in ref_cache:
        ref_sd, ref_meta = fm.load_lora_file(os.path.join(MS_DIR, f"nar_lora_joint_{version}.safetensors"))
        ref_cache[version] = fm.weight_deltas(fm.to_native(ref_sd, ref_meta, base_io=base_io, config=full_cfg))
    ref = ref_cache[version]
    assert set(got) == set(ref)
    worst_lora = worst_io = 0.0
    for path, (w, b) in ref.items():
        if path.startswith("nar.blocks."):
            worst_lora = max(worst_lora, (got[path][0] - w).abs().max().item() / max(w.abs().max().item(), 1e-12))
        else:  # full I/O weights: a bf16 file rounds the full weight, so compare against the base weight's scale
            scale = base_io[path + ".weight"].abs().max().item()
            worst_io = max(worst_io, (got[path][0] - w).abs().max().item() / scale)
            if b is not None:
                bscale = base_io[path + ".bias"].abs().max().item()
                worst_io = max(worst_io, (got[path][1] - b).abs().max().item() / max(bscale, 1e-6))
    print(f"{name}: rel max weight delta vs {version}.safetensors: blocks {worst_lora:.3e}, io {worst_io:.3e}")
    assert worst_lora < 2e-2 and worst_io < 2e-2, (worst_lora, worst_io)
