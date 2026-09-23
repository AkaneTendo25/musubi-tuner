import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import (  # noqa: E402
    make_ref_model,
    make_tiny_tokenizer_json,
    native_to_comfy,
    quantize_comfy,
    save_ref_hf,
    tiny_vae_configs,
    tiny_vae_state_dict,
)

from musubi_tuner.modules.convrot_int8_kernels import dequantize_int8_convrot_weight, quantize_int8_convrot_weight  # noqa: E402
from musubi_tuner.yue2 import yue2_checkpoint as ck  # noqa: E402
from musubi_tuner.yue2.yue2_checkpoint import CheckpointLayout, detect_layout, load_yue2_model, read_base_io  # noqa: E402
from musubi_tuner.yue2.yue2_model import YuE2Config, YuE2Int8Embedding  # noqa: E402

TINY = YuE2Config.tiny()


def _load(path, dtype=torch.float32, **kw):
    return load_yue2_model(path, device="cpu", loading_device="cpu", dtype=dtype, config=TINY, **kw)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("yue2ck")
    ref = make_ref_model(TINY, seed=1)
    hf32 = save_ref_hf(root / "hf32.safetensors", ref)
    hf16 = save_ref_hf(root / "hf16.safetensors", ref, dtype=torch.bfloat16)
    native = {k: v.to(torch.bfloat16) for k, v in _load(hf32).state_dict().items()}
    tok_json = make_tiny_tokenizer_json()
    comfy = native_to_comfy(native, vae_sd=tiny_vae_state_dict(), tokenizer_json=tok_json)
    comfy_path = str(root / "comfy_bf16.safetensors")
    save_file(comfy, comfy_path, metadata={"yue2_format": "1"})
    int8_path = str(root / "comfy_int8.safetensors")
    save_file(quantize_comfy(comfy), int8_path, metadata={"quantization": "convrot_int8", "vae_dtype": "fp16"})
    native_path = str(root / "native.safetensors")
    save_file(native, native_path)
    return dict(ref=ref, hf32=hf32, hf16=hf16, comfy=comfy_path, int8=int8_path, native=native_path, tok_json=tok_json, root=root)


def test_detect_layout(files, tmp_path):
    assert detect_layout(files["hf32"]) == (CheckpointLayout.HF, False)
    assert detect_layout(files["comfy"]) == (CheckpointLayout.COMFY, False)
    assert detect_layout(files["int8"]) == (CheckpointLayout.COMFY, True)
    assert detect_layout(files["native"]) == (CheckpointLayout.NATIVE, False)
    junk = str(tmp_path / "junk.safetensors")
    save_file({"foo.weight": torch.zeros(2)}, junk)
    with pytest.raises(ValueError, match="not a YuE2 checkpoint"):
        detect_layout(junk)


def test_hf_load_fuses_by_role(files):
    ref_sd = files["ref"].state_dict()
    model = _load(files["hf16"], dtype=torch.bfloat16)
    sd = model.state_dict()
    for i in range(TINY.num_layers):
        for src, dst in (("self_attn", "ar"), ("nar_self_attn", "nar")):
            p = f"model.layers.{i}.{src}."
            expected = torch.cat([ref_sd[p + "q_proj.weight"], ref_sd[p + "k_proj.weight"], ref_sd[p + "v_proj.weight"]])
            assert torch.equal(sd[f"{dst}.blocks.{i}.self_attn.qkv_proj.weight"], expected.to(torch.bfloat16))
        for src, dst in (("mlp", "ar"), ("nar_mlp", "nar")):
            p = f"model.layers.{i}.{src}."
            expected = torch.cat([ref_sd[p + "gate_proj.weight"], ref_sd[p + "up_proj.weight"]])
            assert torch.equal(sd[f"{dst}.blocks.{i}.mlp.gate_up_proj.weight"], expected.to(torch.bfloat16))
        assert torch.equal(
            sd[f"nar.blocks.{i}.post_attention_layernorm.weight"],
            ref_sd[f"model.layers.{i}.nar_pre_mlp_layernorm.weight"].to(torch.bfloat16),
        )
    assert torch.equal(sd["ar.lm_head.weight"], ref_sd["lm_head.weight"].to(torch.bfloat16))
    assert torch.equal(sd["nar.latent_pos_embed.pe"], ref_sd["latent_pos_embed.pe"].to(torch.bfloat16))
    assert torch.equal(sd["norm.weight"], ref_sd["model.norm.weight"].to(torch.bfloat16))


def test_comfy_bf16_equals_hf(files):
    hf = _load(files["hf32"], dtype=torch.bfloat16).state_dict()
    comfy = _load(files["comfy"], dtype=torch.bfloat16)
    assert comfy.checkpoint_layout == "comfy"
    sd = comfy.state_dict()
    assert sd.keys() == hf.keys()
    for k in hf:
        assert torch.equal(sd[k], hf[k]), k


def test_native_round_trip(files):
    native = _load(files["native"], dtype=torch.bfloat16)
    hf = _load(files["hf32"], dtype=torch.bfloat16).state_dict()
    assert native.checkpoint_layout == "native"
    for k, v in native.state_dict().items():
        assert torch.equal(v, hf[k]), k


@pytest.mark.parametrize(
    "mode,needed,lm_int8",
    [("auto", True, False), ("auto", False, True), ("keep", True, True), ("dequant", False, False)],
)
def test_comfy_int8_prequantized(files, mode, needed, lm_int8):
    src = _load(files["hf32"], dtype=torch.bfloat16)
    model = _load(files["int8"], dtype=torch.bfloat16, prequant_lm_head=mode, lm_head_needed=needed)
    assert model.base_quant == "prequant_int8" and model.is_convrot_int8
    src_sd = src.state_dict()
    for name, module in model.named_modules():
        if name.startswith(("ar.blocks.", "nar.blocks.")) and isinstance(module, nn.Linear):
            assert module.weight.dtype == torch.int8 and hasattr(module, "scale_weight"), name
            w = dequantize_int8_convrot_weight(module.weight, module.scale_weight, 256)
            ref = src_sd[name + ".weight"].float()
            assert ((w - ref).norm() / ref.norm()).item() < 1e-2, name
    for name in ("nar.llm2vae", "nar.time_embedder.mlp.0", "nar.time_embedder.mlp.2", "nar.vae2llm"):
        module = model.get_submodule(name)
        assert module.weight.dtype == torch.bfloat16 and not hasattr(module, "scale_weight"), name
        assert "forward" not in module.__dict__, name
    head = model.ar.lm_head
    assert (head.weight.dtype == torch.int8) == lm_int8 and hasattr(head, "scale_weight") == lm_int8
    assert isinstance(model.ar.embed_tokens, YuE2Int8Embedding)
    ids = torch.randint(0, TINY.vocab_size, (1, 24), generator=torch.Generator().manual_seed(0))
    emb = model.embed(ids).float()
    ref_emb = src.embed(ids).float()
    assert ((emb - ref_emb).norm() / ref_emb.norm()).item() < 1e-2
    with torch.no_grad():
        a = model.lm_logits(model.ar_forward(model.embed(ids)).hidden).float()
        b = src.lm_logits(src.ar_forward(src.embed(ids)).hidden).float()
    assert F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item() > 0.99


def test_int8_quantize_dequantize_round_trip():
    torch.manual_seed(0)
    w = torch.randn(64, 512)
    q, s = quantize_int8_convrot_weight(w, 256)
    back = dequantize_int8_convrot_weight(q, s, 256)
    assert ((back - w).norm() / w.norm()).item() < 1e-2


def _lora(rows, cols, rank, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rank, cols, generator=g) * 0.1, torch.randn(rows, rank, generator=g) * 0.1


def _merge_input():
    c = TINY
    qkv = c.q_dim + 2 * c.kv_dim
    sd = {}
    for name, (rows, cols) in {
        "lora_unet_ar_blocks_0_self_attn_qkv_proj": (qkv, c.hidden_size),
        "lora_unet_nar_blocks_1_mlp_gate_up_proj": (2 * c.intermediate_size, c.hidden_size),
        "lora_unet_nar_blocks_2_mlp_down_proj": (c.hidden_size, c.intermediate_size),
    }.items():
        down, up = _lora(rows, cols, 4, len(sd))
        sd[name + ".lora_down.weight"], sd[name + ".lora_up.weight"], sd[name + ".alpha"] = down, up, torch.tensor(2.0)
    return sd


def test_merge_at_load_equals_post_hoc_merge(files):
    base = _load(files["hf32"]).state_dict()
    lora = _merge_input()
    merged = _load(files["hf32"], lora_weights=[lora], lora_multipliers=[0.7]).state_dict()
    targets = {
        "ar.blocks.0.self_attn.qkv_proj.weight": "lora_unet_ar_blocks_0_self_attn_qkv_proj",
        "nar.blocks.1.mlp.gate_up_proj.weight": "lora_unet_nar_blocks_1_mlp_gate_up_proj",
        "nar.blocks.2.mlp.down_proj.weight": "lora_unet_nar_blocks_2_mlp_down_proj",
    }
    for key in base:
        if key in targets:
            n = targets[key]
            delta = 0.7 * (2.0 / 4) * lora[n + ".lora_up.weight"] @ lora[n + ".lora_down.weight"]
            assert (merged[key] - (base[key] + delta)).abs().max().item() < 1e-6, key
        else:
            assert torch.equal(merged[key], base[key]), key


def test_merge_at_load_then_convrot(files):
    merged = _load(files["hf32"], lora_weights=[_merge_input()], lora_multipliers=[0.7]).state_dict()
    model = _load(files["hf32"], convrot_int8=True, lora_weights=[_merge_input()], lora_multipliers=[0.7])
    m = model.get_submodule("ar.blocks.0.self_attn.qkv_proj")
    w = dequantize_int8_convrot_weight(m.weight, m.scale_weight, 256)
    ref = merged["ar.blocks.0.self_attn.qkv_proj.weight"]
    assert ((w - ref).norm() / ref.norm()).item() < 1e-2


def _quantized_modules(model):
    return {n for n, m in model.named_modules() if isinstance(m, nn.Linear) and hasattr(m, "scale_weight")}


def _block_linears(model):
    return {n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith(("ar.blocks.", "nar.blocks."))}


@pytest.mark.parametrize("kind", ["convrot", "fp8"])
def test_quantization_scope_on_bf16_source(files, kind):
    kw = {"convrot_int8": True} if kind == "convrot" else {"fp8_scaled": True}
    model = _load(files["hf16"], dtype=torch.bfloat16, **kw)
    assert _quantized_modules(model) == _block_linears(model)
    assert model.base_quant == ("convrot_int8" if kind == "convrot" else "fp8_scaled")
    for name in ("ar.lm_head", "nar.vae2llm", "nar.llm2vae", "nar.time_embedder.mlp.0"):
        assert model.get_submodule(name).weight.dtype == torch.bfloat16
    assert model.ar.embed_tokens.weight.dtype == torch.bfloat16
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            assert p.dtype == torch.bfloat16
    with_head = _load(files["hf16"], dtype=torch.bfloat16, quantize_lm_head=True, **kw)
    assert _quantized_modules(with_head) == _block_linears(model) | {"ar.lm_head"}
    ids = torch.randint(0, TINY.vocab_size, (1, 10))
    ref = _load(files["hf16"], dtype=torch.bfloat16)
    with torch.no_grad():
        a = model.lm_logits(model.ar_forward(model.embed(ids)).hidden).float()
        b = ref.lm_logits(ref.ar_forward(ref.embed(ids)).hidden).float()
    assert F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item() > 0.99


def test_io_diff_and_ms_full_application(files):
    base = _load(files["hf32"])
    g = torch.Generator().manual_seed(3)
    diff = torch.randn(64, TINY.hidden_size, generator=g) * 0.01
    diff_b = torch.randn(64, generator=g) * 0.01
    model = _load(
        files["hf32"],
        lora_weights=[{"lora_unet_nar_llm2vae.diff": diff, "lora_unet_nar_llm2vae.diff_b": diff_b}],
        lora_multipliers=[0.5],
    )
    assert torch.allclose(model.nar.llm2vae.weight, base.nar.llm2vae.weight + 0.5 * diff, atol=1e-7)
    assert torch.allclose(model.nar.llm2vae.bias, base.nar.llm2vae.bias + 0.5 * diff_b, atol=1e-7)
    assert torch.equal(model.nar.vae2llm.weight, base.nar.vae2llm.weight)

    # "full" I/O weights reach the loader as full - base (the LoRA converter builds them with read_base_io)
    full = torch.randn(TINY.hidden_size, 64, generator=g)
    base_io = read_base_io(files["hf32"])
    full_diff = {"lora_unet_nar_vae2llm.diff": full - base_io["nar.vae2llm.weight"]}
    model = _load(files["hf32"], lora_weights=[full_diff])
    assert (model.nar.vae2llm.weight - full).abs().max().item() < 1e-6


def test_merge_input_rejected_for_prequant_and_fp8(files):
    with pytest.raises(ValueError, match="pre-quantized"):
        _load(files["int8"], dtype=torch.bfloat16, lora_weights=[_merge_input()])
    with pytest.raises(ValueError, match="fp8"):
        _load(files["int8"], dtype=torch.bfloat16, fp8_scaled=True)
    with pytest.raises(ValueError, match="native fused dict"):
        _load(files["hf32"], lora_weights=[{"lora_unet_ar_blocks_0_self_attn_qkv_proj.lora_down.0.weight": torch.zeros(1)}])


def test_read_base_io(files):
    model = _load(files["hf32"])
    expected = {
        f"nar.{n}.{s}": model.get_submodule(f"nar.{n}").get_parameter(s).float()
        for n in ck.YUE2_IO_NAMES
        for s in ("weight", "bias")
    }
    hf = read_base_io(files["hf32"])
    assert hf.keys() == expected.keys()
    for k in expected:
        assert torch.equal(hf[k], expected[k]), k
    comfy = read_base_io(files["comfy"])
    for k in expected:
        assert torch.equal(comfy[k], expected[k].to(torch.bfloat16).float()), k
    q = read_base_io(files["int8"])
    for k in expected:
        assert q[k].dtype == torch.float32
        rel = ((q[k] - expected[k]).norm() / expected[k].norm()).item()
        assert rel < (1e-2 if k.endswith("weight") else 1e-2), k


def test_vae_and_tokenizer_extraction(files):
    ref_sd = tiny_vae_state_dict()
    vae = ck.load_yue2_vae(None, files["comfy"], vae_configs=tiny_vae_configs())
    assert vae.source_dtype == "float32"
    for k, v in vae.state_dict().items():
        assert torch.equal(v, ref_sd[k]), k
    with pytest.raises(ValueError, match="not float32"):
        ck.load_yue2_vae(None, files["int8"], vae_configs=tiny_vae_configs(), allow_fp16_source=False)
    vae16 = ck.load_yue2_vae(None, files["int8"], vae_configs=tiny_vae_configs())
    assert vae16.source_dtype == "float16" and vae16.fingerprint != vae.fingerprint
    again = ck.load_yue2_vae(None, files["comfy"], vae_configs=tiny_vae_configs())
    assert again.fingerprint == vae.fingerprint
    dec = ck.load_yue2_vae(None, files["comfy"], vae_configs=tiny_vae_configs(), decoder_only=True)
    assert not hasattr(dec, "encoder")

    assert ck.read_tokenizer_json(files["comfy"]) == files["tok_json"]
    tok = ck.load_yue2_tokenizer(None, files["comfy"])
    assert tok.backend == "json" and tok.encode("la la la")
    json_path = files["root"] / "tok.json"
    json_path.write_bytes(files["tok_json"])
    assert ck.load_yue2_tokenizer(str(json_path), None).fingerprint == tok.fingerprint
    with pytest.raises(ValueError, match="tokenizer not found"):
        ck.load_yue2_tokenizer(None, files["hf32"])


def test_load_vae_from_directory_and_prefixed_file(files, tmp_path):
    import json

    enc, dec = tiny_vae_configs()
    sd = tiny_vae_state_dict()
    d = tmp_path / "vae"
    d.mkdir()
    save_file(sd, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"encoder_config": enc, "decoder_config": dec}))
    vae = ck.load_yue2_vae(str(d), None)
    prefixed = tmp_path / "prefixed.safetensors"
    save_file({"vae." + k: v for k, v in sd.items()} | {"other": torch.zeros(1)}, str(prefixed))
    vae2 = ck.load_yue2_vae(str(prefixed), None, vae_configs=(enc, dec))
    for k, v in vae.state_dict().items():
        assert torch.equal(v, sd[k]) and torch.equal(vae2.state_dict()[k], v), k
