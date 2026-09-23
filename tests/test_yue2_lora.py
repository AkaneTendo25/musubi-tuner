import copy
import functools
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_yue2_lora_formats import CFG, make_native, ms_layer_keys  # noqa: E402

from musubi_tuner.networks import lora_yue2  # noqa: E402
from musubi_tuner.utils.lora_utils import attach_lora_weights  # noqa: E402
from musubi_tuner.yue2 import yue2_lora_formats as fm  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------------
# a small module tree with the YuE2Model module names (fast CPU stand-in for the real model)


class _Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class _Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.qkv_proj = nn.Linear(c.hidden_size, c.q_dim + 2 * c.kv_dim, bias=False)
        self.o_proj = nn.Linear(c.q_dim, c.hidden_size, bias=False)
        self.q_norm = _Norm(c.head_dim)
        self.k_norm = _Norm(c.head_dim)


class _MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_up_proj = nn.Linear(c.hidden_size, 2 * c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)


class YuE2Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = _Norm(c.hidden_size)
        self.self_attn = _Attention(c)
        self.post_attention_layernorm = _Norm(c.hidden_size)
        self.mlp = _MLP(c)


class _TimeEmbedder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c.time_freq_dim, c.hidden_size), nn.SiLU(), nn.Linear(c.hidden_size, c.hidden_size))


class _AR(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.blocks = nn.ModuleList([YuE2Block(c) for _ in range(c.num_layers)])
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)


class _NAR(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blocks = nn.ModuleList([YuE2Block(c) for _ in range(c.num_layers)])
        self.vae2llm = nn.Linear(c.latent_dim, c.hidden_size)
        self.llm2vae = nn.Linear(c.hidden_size, c.latent_dim)
        self.time_embedder = _TimeEmbedder(c)


class FakeYuE2Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.ar = _AR(c)
        self.nar = _NAR(c)
        self.norm = _Norm(c.hidden_size)


def _build_model():
    try:
        from musubi_tuner.yue2.yue2_model import YuE2Model

        model = YuE2Model(CFG)
    except NotImplementedError:
        model = FakeYuE2Model(CFG)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
    return model.float().eval()


@pytest.fixture(scope="module")
def base_model():
    return _build_model()


@pytest.fixture
def model(base_model):
    return copy.deepcopy(base_model)


def _net(model, dim=4, alpha=2.0, **kwargs):
    kwargs = {k: str(v) for k, v in kwargs.items()}
    return lora_yue2.create_arch_network(1.0, dim, alpha, None, None, model, **kwargs)


def _applied(model, **kwargs):
    """Network applied to ``model`` with random weights (LoRA modules are registered on the network in ``apply_to``)."""
    net = _net(model, **kwargs)
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    _randomize(net)
    return net


def _randomize(network, seed=1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in network.named_parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)


def _linear_targets(model):
    return {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}


def _check_outputs(model, base_model, deltas, atol=1e-5):
    """Every Linear of ``model`` equals the ``base_model`` Linear with ``deltas[path] = (dW, db)`` added."""
    gen = torch.Generator().manual_seed(7)
    base_linears = _linear_targets(base_model)
    for path, module in _linear_targets(model).items():
        if path == "ar.lm_head":
            continue
        base = base_linears[path]
        x = torch.randn(2, 3, module.in_features, generator=gen)
        w, b = base.weight, base.bias
        if path in deltas:
            dw, db = deltas[path]
            w = w + dw
            if db is not None:
                b = b + db
        expected = F.linear(x, w, b)
        got = module(x)
        assert torch.allclose(got, expected, atol=atol, rtol=1e-5), (path, (got - expected).abs().max().item())


def _network_deltas(network):
    out = {}
    for m in network.unet_loras:
        db = None
        if isinstance(m, lora_yue2.YuE2DiffModule) and m.diff_b is not None:
            db = m.diff_b.detach().float() * m.multiplier
        out[fm.path_of(m.lora_name)] = (m.delta_weight().detach(), db)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# targets


def _names(network):
    return sorted(m.lora_name for m in network.unet_loras)


@pytest.mark.parametrize("branches,count_ar,count_nar", [("nar", 0, 12), ("ar", 12, 0), ("ar,nar", 12, 12)])
def test_default_targets_per_branch(model, branches, count_ar, count_nar):
    net = _net(model, branches=branches)
    names = _names(net)
    assert sum(n.startswith("lora_unet_ar_blocks_") for n in names) == count_ar == CFG.num_layers * 4 * (count_ar > 0)
    assert sum(n.startswith("lora_unet_nar_blocks_") for n in names) == count_nar
    assert len(names) == count_ar + count_nar
    assert net.layout == "split"
    qkv = next(m for m in net.unet_loras if m.lora_name.endswith("self_attn_qkv_proj"))
    assert qkv.split_dims == [CFG.q_dim, CFG.kv_dim, CFG.kv_dim]
    gu = next(m for m in net.unet_loras if m.lora_name.endswith("mlp_gate_up_proj"))
    assert gu.split_dims == [CFG.intermediate_size, CFG.intermediate_size]
    fused = _net(copy.deepcopy(model), branches=branches, lora_layout="fused")
    assert fused.layout == "fused" and all(m.split_dims is None for m in fused.unet_loras)


def test_c4_include_exclude_semantics(model):
    net = _net(model, exclude_patterns=[r".*\.mlp\..*"])
    assert len(net.unet_loras) == CFG.num_layers * 2 and all("self_attn" in n for n in _names(net))
    net = _net(copy.deepcopy(model), include_patterns=[r"ar\.blocks\.0\.self_attn\.o_proj"])
    assert "lora_unet_ar_blocks_0_self_attn_o_proj" in _names(net) and len(net.unet_loras) == CFG.num_layers * 4 + 1
    net = _net(copy.deepcopy(model), include_patterns=[r"nar\.vae2llm", r"nar\.time_embedder\.mlp\.0"])
    assert len(net.unet_loras) == CFG.num_layers * 4  # I/O only through options
    net = _net(copy.deepcopy(model), targets="attn")
    assert len(net.unet_loras) == CFG.num_layers * 2
    with pytest.raises(RuntimeError, match="zero target"):
        _net(copy.deepcopy(model), exclude_patterns=[".*"])


def test_io_modules_only_via_options(model):
    net = _net(model, train_io="lora")
    assert {"lora_unet_nar_vae2llm", "lora_unet_nar_llm2vae"} <= set(_names(net))
    net = _net(copy.deepcopy(model), train_io="full", include_time_embedder=True, io_modules="llm2vae")
    io = [m for m in net.unet_loras if net.branch_of(m.lora_name) == "io"]
    assert sorted(m.lora_name for m in io) == [
        "lora_unet_nar_llm2vae",
        "lora_unet_nar_time_embedder_mlp_0",
        "lora_unet_nar_time_embedder_mlp_2",
    ]
    assert all(isinstance(m, lora_yue2.YuE2DiffModule) for m in io)
    with pytest.raises(ValueError):
        _net(copy.deepcopy(model), branches="ar", train_io="lora")
    with pytest.raises(ValueError, match="unknown YuE2 network args"):
        _net(copy.deepcopy(model), nar_dimm=4)


def test_options_parsing():
    kwargs = {
        "branches": "ar,nar",
        "lora_layout": "fused",
        "ar_dim": "8",
        "nar_alpha": "1.5",
        "include_time_embedder": "True",
        "train_io": "full",
        "io_lr": "none",
        "ar_lr_ratio": "0.5",
        "verbose": "True",
    }
    opts = lora_yue2.YuE2NetworkOptions.from_kwargs(kwargs)
    assert opts.branches == ("ar", "nar") and opts.layout == "fused" and opts.ar_dim == 8 and opts.nar_alpha == 1.5
    assert opts.include_time_embedder and opts.io_lr is None and opts.ar_lr_ratio == 0.5
    assert kwargs == {"verbose": "True"}  # own keys are popped
    assert (
        lora_yue2.build_include_pattern(opts, [])
        == r"(?:ar|nar)\.blocks\.\d+\.(?:self_attn\.qkv_proj|self_attn\.o_proj|mlp\.gate_up_proj|mlp\.down_proj)"
    )


def test_per_branch_rank_and_alpha(model):
    net = _net(model, branches="ar,nar", ar_dim=8, ar_alpha=4, nar_dim=2, nar_alpha=1, train_io="lora", io_dim=3)
    for m in net.unet_loras:
        branch = net.branch_of(m.lora_name)
        expected = {"ar": (8, 0.5), "nar": (2, 0.5), "io": (3, 1 / 3)}[branch]
        assert (m.lora_dim, pytest.approx(m.scale)) == (expected[0], expected[1]), m.lora_name


# ----------------------------------------------------------------------------------------------------------------------
# forward, merge


@pytest.mark.parametrize("layout", ["split", "fused"])
def test_forward_equals_manual_delta(model, base_model, layout):
    net = _net(model, branches="ar,nar", lora_layout=layout, train_io="lora", include_time_embedder=True)
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    _randomize(net)
    _check_outputs(model, base_model, _network_deltas(net))
    net.set_enabled(False)
    _check_outputs(model, base_model, {})
    net.set_enabled(True)
    net.set_multiplier_by_branch(ar=0.0, nar=1.0, io=0.0)
    deltas = {p: v for p, v in _network_deltas(net).items() if p.startswith("nar.blocks.")}
    _check_outputs(model, base_model, deltas)


def test_split_training_forward_with_dropouts(model):
    net = _net(model, rank_dropout=0.5, module_dropout=0.0)
    net.dropout = None
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    _randomize(net)
    net.train()
    x = torch.randn(2, 5, CFG.hidden_size, requires_grad=True)
    y = model.nar.blocks[0].self_attn.qkv_proj(x)
    assert y.shape == (2, 5, CFG.q_dim + 2 * CFG.kv_dim)
    y.sum().backward()
    qkv = next(m for m in net.unet_loras if m.lora_name == "lora_unet_nar_blocks_0_self_attn_qkv_proj")
    assert all(d.weight.grad is not None for d in qkv.lora_down)


@pytest.mark.parametrize("layout", ["split", "fused"])
def test_merge_to_equals_runtime_forward(model, base_model, layout):
    net = _net(model, branches="ar,nar", lora_layout=layout, train_io="full")
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    _randomize(net)
    sd = net.state_dict()
    merged = copy.deepcopy(base_model)
    inf = lora_yue2.create_arch_network_from_weights(1.0, sd, unet=merged, for_inference=True)
    assert inf.layout == layout
    inf.merge_to(None, merged, sd, torch.float32, "cpu")
    deltas = _network_deltas(net)
    _check_outputs(merged, base_model, deltas)
    _check_outputs(model, base_model, deltas)


def test_merge_to_split_places_rows(model, base_model):
    """Generic ``LoRAInfModule`` split merge adds ``up_i @ down_i`` to the whole weight (shape error); ours places rows."""
    net = _net(model, branches="nar")
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    _randomize(net)
    sd = net.state_dict()
    merged = copy.deepcopy(base_model)
    inf = lora_yue2.create_arch_network_from_weights(1.0, sd, unet=merged, for_inference=True)
    inf.merge_to(None, merged, sd, None, None)
    w0 = base_model.nar.blocks[1].self_attn.qkv_proj.weight
    w1 = merged.nar.blocks[1].self_attn.qkv_proj.weight
    delta = (w1 - w0)[CFG.q_dim : CFG.q_dim + CFG.kv_dim]
    k_scale = 2.0 / 4
    expected = (
        k_scale
        * sd["lora_unet_nar_blocks_1_self_attn_qkv_proj.lora_up.1.weight"]
        @ sd["lora_unet_nar_blocks_1_self_attn_qkv_proj.lora_down.1.weight"]
    )
    assert torch.allclose(delta, expected, atol=1e-6)


# ----------------------------------------------------------------------------------------------------------------------
# diff modules


def test_diff_module_state_dict_has_no_base_weights(model, base_model):
    net = _net(model, train_io="full", include_time_embedder=True)
    for m in net.unet_loras:  # no module registers its base Linear, before or after apply_to
        assert all("weight" not in k or "lora_" in k for k in m.state_dict()), m.lora_name
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    sd = net.state_dict()
    assert all(k.endswith(("weight", ".alpha", ".diff", ".diff_b")) for k in sd)
    assert "lora_unet_nar_vae2llm.diff" in sd and "lora_unet_nar_vae2llm.diff_b" in sd
    assert not any(k.endswith((".weight", ".bias")) and ".lora_" not in k for k in sd)
    assert not any("org_module" in k for k in sd)
    diff = next(m for m in net.unet_loras if m.lora_name == "lora_unet_nar_llm2vae")
    assert sorted(dict(diff.named_parameters())) == ["diff", "diff_b"] and not hasattr(diff, "alpha")
    _randomize(net)
    _check_outputs(model, base_model, _network_deltas(net))
    net.set_enabled(False)
    _check_outputs(model, base_model, {})


def test_int8_llm2vae_prepare_optimizer_params(model):
    model.nar.llm2vae.weight = nn.Parameter(torch.zeros(CFG.latent_dim, CFG.hidden_size, dtype=torch.int8), requires_grad=False)
    net = _net(model, train_io="full")
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    params, descriptions = net.prepare_optimizer_params(unet_lr=1e-4)
    assert descriptions == ["nar", "io"]
    assert model.nar.llm2vae.weight.dtype == torch.int8 and not model.nar.llm2vae.weight.requires_grad
    diff = next(m for m in net.unet_loras if m.lora_name == "lora_unet_nar_llm2vae")
    with pytest.raises(ValueError, match="quantized"):
        diff.merge_to({"diff": torch.zeros(CFG.latent_dim, CFG.hidden_size)}, None, None)


# ----------------------------------------------------------------------------------------------------------------------
# optimizer groups


def test_prepare_optimizer_params_groups(model):
    net = _net(model, branches="ar,nar", ar_lr_ratio=0.5, io_lr=3e-5, train_io="lora")
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    params, descriptions = net.prepare_optimizer_params(unet_lr=1e-3)
    assert descriptions == ["nar", "ar", "io"]
    assert [g["lr"] for g in params] == [1e-3, 5e-4, 3e-5]
    counts = [sum(p.numel() for p in g["params"]) for g in params]
    assert counts[0] == sum(p.numel() for m in net.modules_of_branch("nar") for p in m.parameters())
    assert counts[1] == sum(p.numel() for m in net.modules_of_branch("ar") for p in m.parameters())

    net = _net(copy.deepcopy(model), branches="ar,nar", train_io="full", loraplus_lr_ratio=4)
    params, descriptions = net.prepare_optimizer_params(unet_lr=1e-3)
    assert descriptions == ["nar", "nar plus", "ar", "ar plus", "io"]
    assert [g["lr"] for g in params] == [1e-3, 4e-3, 1e-3, 4e-3, 1e-3]

    net = _net(copy.deepcopy(model), branches="ar,nar", ar_lr_ratio=0)
    params, descriptions = net.prepare_optimizer_params(unet_lr=1e-3)
    assert descriptions == ["nar"]
    assert not any(p.requires_grad for m in net.modules_of_branch("ar") for p in m.parameters())


@pytest.mark.parametrize(
    "kwargs,frozen_branch,descriptions",
    [(dict(branches="ar,nar", ar_lr_ratio=0), "ar", ["nar"]), (dict(train_io="lora", io_lr=0), "io", ["nar"])],
)
def test_zero_lr_group_stays_frozen_in_training(model, kwargs, frozen_branch, descriptions):
    # the order of the base trainer: prepare_optimizer_params, prepare_grad_etc, then steps clipping get_trainable_params
    model.requires_grad_(False)
    net = _applied(model, **kwargs)
    params, got = net.prepare_optimizer_params(unet_lr=1e-3)
    assert got == descriptions
    net.prepare_grad_etc(model)
    frozen = [p for m in net.modules_of_branch(frozen_branch) for p in m.parameters()]
    assert frozen and not any(p.requires_grad for p in frozen)
    trainable = list(net.get_trainable_params())
    assert {id(p) for p in trainable} == {id(p) for g in params for p in g["params"]}

    optimizer = torch.optim.SGD(params)
    gen = torch.Generator().manual_seed(3)
    for _ in range(3):
        loss = sum(m(torch.randn(2, m.org_module_ref[0].in_features, generator=gen)).square().mean() for m in net.unet_loras)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        # a plain SGD update: optimizer.step() initialises CUDA when it is visible, and this is a CPU test
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group["params"]:
                    p.sub_(p.grad, alpha=group["lr"])
        optimizer.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in frozen)
        assert all(p.grad is None for p in trainable)


# ----------------------------------------------------------------------------------------------------------------------
# load / save


def test_load_weights_comfy_fused_into_split_is_exact(model, base_model, tmp_path):
    src = _applied(copy.deepcopy(base_model), branches="ar,nar", train_io="full")
    _randomize(src)
    native = fm.to_native(src.state_dict())
    comfy, meta = fm.native_to_comfy(native)
    path = str(tmp_path / "comfy.safetensors")
    fm.save_lora_file(path, comfy, meta)
    dst = _net(model, branches="ar,nar", train_io="full", dim=4, alpha=1.0)  # different alpha: scales are baked
    dst.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    info = dst.load_weights(path)
    assert not info.missing_keys and not info.unexpected_keys
    _check_outputs(model, base_model, fm.weight_deltas(native))


def test_load_weights_aitk_fused_into_split_uses_shared_down(model, base_model, tmp_path):
    fused = _applied(copy.deepcopy(base_model), branches="nar", lora_layout="fused", dim=4, alpha=4.0)
    _randomize(fused)
    native = fm.to_native(fused.state_dict())
    comfy, _ = fm.native_to_comfy(native)
    aitk = {
        k.replace("lora_down.weight", "lora_A.weight").replace("lora_up.weight", "lora_B.weight"): v
        for k, v in comfy.items()
        if not k.endswith(".alpha")
    }
    path = str(tmp_path / "aitk.safetensors")
    fm.save_lora_file(path, aitk)
    dst = _net(model, branches="nar", dim=4, alpha=4.0)
    dst.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    dst.load_weights(path)
    _check_outputs(model, base_model, fm.weight_deltas(native))


def test_load_weights_split_into_fused_rank_change_raises(model, tmp_path):
    src = _applied(copy.deepcopy(model), branches="nar", dim=4)
    path = str(tmp_path / "split.safetensors")
    src.save_weights(path, torch.float32, {})
    dst = _net(model, branches="nar", lora_layout="fused", dim=4)
    with pytest.raises(ValueError, match="rank"):
        dst.load_weights(path)
    ok = _net(copy.deepcopy(model), branches="nar", lora_layout="fused", dim=4, targets="attn")
    with pytest.raises(ValueError, match="rank 12"):
        ok.load_weights(path)


def test_load_weights_ms_full_io_uses_network_base(model, base_model, tmp_path):
    name, sd, meta, expected, _ = _formats(base_model)[-1]
    assert name == "ms_safetensors"
    path = str(tmp_path / "ms.safetensors")
    fm.save_lora_file(path, sd, meta)
    net = _net(model, branches="nar", train_io="full", dim=4, alpha=4.0)
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    net.load_weights(path)
    _check_outputs(model, base_model, fm.weight_deltas(expected), atol=2e-5)


def test_save_and_create_from_weights_roundtrip(model, base_model, tmp_path):
    net = _net(model, branches="ar,nar", ar_dim=6, train_io="full", include_time_embedder=True)
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    _randomize(net)
    path = str(tmp_path / "native.safetensors")
    net.save_weights(path, torch.float32, {"ss_network_module": "networks.lora_yue2"})
    sd = load_file(path)
    fresh = copy.deepcopy(base_model)
    loaded = lora_yue2.create_arch_network_from_weights(1.0, sd, unet=fresh)
    loaded.apply_to(None, fresh, apply_text_encoder=False, apply_unet=True)
    loaded.load_state_dict(sd, strict=True)
    assert sorted(m.lora_name for m in loaded.unet_loras) == _names(net)
    _check_outputs(fresh, base_model, _network_deltas(net))
    with pytest.raises(ValueError, match="not in the model"):
        bad = dict(sd)
        bad["lora_unet_ar_blocks_9_mlp_down_proj.lora_down.weight"] = torch.zeros(4, CFG.intermediate_size)
        bad["lora_unet_ar_blocks_9_mlp_down_proj.lora_up.weight"] = torch.zeros(CFG.hidden_size, 4)
        lora_yue2.create_arch_network_from_weights(1.0, bad, unet=copy.deepcopy(base_model))


# ----------------------------------------------------------------------------------------------------------------------
# attach_lora_weights (strict) from every source format


def _ms_sd(native_split_nar):
    """Mothersuperior safetensors keys from a split, scale-1 NAR LoRA (+ full I/O = base + diff)."""
    sd = {}
    for layer in range(CFG.num_layers):
        pieces = []
        for sub in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"):
            md = native_split_nar.modules[f"nar.blocks.{layer}.{sub}"]
            pieces += list(zip(md.downs, md.ups))
        order = [0, 1, 2, 3, 4, 5, 6]  # q, k, v, o, gate, up, down
        for key, idx in zip(ms_layer_keys(layer), order):
            sd[key + ".lora_A"], sd[key + ".lora_B"] = pieces[idx]
    return sd


def _formats(base_model):
    native = make_native(seed=11, rank=4, alpha=4.0, io="diff")  # scale 1: valid for every format
    comfy, comfy_meta = fm.native_to_comfy(native)
    nar = fm.filter_branch(make_native(seed=12, rank=4, alpha=4.0, io="lora", time_embedder=True), "nar")
    hf, hf_meta = fm.native_to_hf(nar)
    fl = fm.native_to_fl(native)
    aitk = {k.replace("lora_down.weight", "lora_A.weight").replace("lora_up.weight", "lora_B.weight"): v for k, v in comfy.items()}
    ms_native = fm.filter_branch(make_native(seed=13, rank=4, alpha=4.0), "nar")
    ms = _ms_sd(ms_native)
    base_io = {}
    g = torch.Generator().manual_seed(14)
    for m in ("vae2llm", "llm2vae"):
        lin = getattr(base_model.nar, m)
        dw, db = torch.randn(lin.weight.shape, generator=g) * 0.1, torch.randn(lin.bias.shape, generator=g) * 0.1
        ms[f"{m}.weight"], ms[f"{m}.bias"] = lin.weight.detach() + dw, lin.bias.detach() + db
        base_io[f"nar.{m}.weight"], base_io[f"nar.{m}.bias"] = lin.weight.detach().clone(), lin.bias.detach().clone()
        ms_native.modules[f"nar.{m}"] = fm.ModuleDelta("diff", diff=dw, diff_b=db)
    return [
        ("native_split", fm.native_state_dict(native), {}, native, None),
        ("native_fused", fm.native_state_dict(fm.split_to_fused(native)), {}, native, None),
        ("comfy", comfy, comfy_meta, native, None),
        ("aitk", aitk, {}, native, None),
        ("hf", hf, hf_meta, nar, None),
        ("fl_nar", *fl["nar"], fm.filter_branch(native, "nar"), None),
        ("fl_ar", *fl["ar"], fm.filter_branch(native, "ar"), None),
        ("ms_safetensors", ms, {}, ms_native, base_io),
    ]


def test_attach_lora_weights_every_format(base_model, tmp_path):
    for name, sd, meta, expected, base_io in _formats(base_model):
        path = str(tmp_path / f"{name}.safetensors")
        fm.save_lora_file(path, sd, meta)
        model = copy.deepcopy(base_model)
        converter = functools.partial(lora_yue2.convert_lora_state_dict, metadata=meta, config=CFG, base_io=base_io)
        networks = attach_lora_weights(lora_yue2, model, [path], [1.0], None, None, torch.device("cpu"), converter=converter)
        assert len(networks) == 1, name
        _check_outputs(model, base_model, fm.weight_deltas(expected), atol=2e-5)


def test_network_merge_to_accepts_foreign_format(base_model):
    native = make_native(seed=21, rank=4, alpha=2.0, io="diff")
    comfy, _ = fm.native_to_comfy(native)
    merged = copy.deepcopy(base_model)
    net = lora_yue2.create_arch_network_from_weights(1.0, comfy, unet=merged, for_inference=True)
    assert net.layout == "fused"  # the Comfy file is fused (block-diagonal)
    net.merge_to(None, merged, comfy, None, "cpu")
    _check_outputs(merged, base_model, fm.weight_deltas(native), atol=2e-5)


def test_real_model_forward_attached_vs_merged(base_model):
    """End to end on the real YuE2Model: the network changes the AR/NAR outputs, ``set_enabled(False)`` restores the base,
    and merging the saved weights gives the attached result."""
    if type(base_model).__name__ != "YuE2Model":
        pytest.skip("yue2_model.YuE2Model is not implemented yet")
    gen = torch.Generator().manual_seed(3)
    ids = torch.randint(0, CFG.vocab_size, (1, 12), generator=gen)
    x_t = torch.randn(1, 6, CFG.latent_dim, generator=gen)

    def run(m):
        with torch.no_grad():
            out = m.ar_forward(m.embed(ids), return_kv=True)
            v = m.nar_forward(x_t, torch.tensor([0.4]), out.kv, rope_offset=12)
        return out.hidden, v

    model = copy.deepcopy(base_model)
    base = run(model)
    net = _applied(model, branches="ar,nar", train_io="full", include_time_embedder=True)
    adapted = run(model)
    assert not torch.allclose(adapted[0], base[0]) and not torch.allclose(adapted[1], base[1])
    net.set_enabled(False)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(run(model), base))
    net.set_enabled(True)
    merged = copy.deepcopy(base_model)
    sd = net.state_dict()
    lora_yue2.create_arch_network_from_weights(1.0, sd, unet=merged, for_inference=True).merge_to(None, merged, sd, None, None)
    assert all(torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(run(merged), adapted))


# ----------------------------------------------------------------------------------------------------------------------
# max-norm regularization


def test_apply_max_norm_regularization_split_fused_diff(model):
    for layout in ("split", "fused"):
        m = copy.deepcopy(model)
        net = _net(m, branches="nar", lora_layout=layout, train_io="full")
        net.apply_to(None, m, apply_text_encoder=False, apply_unet=True)
        _randomize(net)
        before = {
            x.lora_name: (x.diff.detach().clone() if isinstance(x, lora_yue2.YuE2DiffModule) else x.delta_weight().detach())
            for x in net.unet_loras
        }
        max_norm = 0.5 * min(v.norm().item() for v in before.values())
        keys_scaled, mean_norm, max_seen = net.apply_max_norm_regularization(max_norm, torch.device("cpu"))
        assert keys_scaled == len(net.unet_loras)
        assert mean_norm == pytest.approx(max_norm, rel=1e-5) and max_seen == pytest.approx(max_norm, rel=1e-5)
        for x in net.unet_loras:
            after = x.diff.detach() if isinstance(x, lora_yue2.YuE2DiffModule) else x.delta_weight().detach()
            ratio = max_norm / before[x.lora_name].norm().item()
            assert torch.allclose(after, before[x.lora_name] * ratio, atol=1e-6), x.lora_name
            assert after.norm().item() == pytest.approx(max_norm, rel=1e-4)
        # already within the bound: nothing changes
        assert net.apply_max_norm_regularization(max_norm * 4, torch.device("cpu"))[0] == 0
