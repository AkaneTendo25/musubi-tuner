"""YuE2 real-weight checks on CUDA: loaders, tokenizer, AR/NAR/VAE parity with the official reference, ComfyUI export keys.

Skipped unless ``YUE2_WEIGHTS_DIR`` is set and CUDA is available (see ``conftest.py``). The official reference is the
``yue2_infer`` wheel next to the HF checkpoint, imported by zipimport. Numbers are printed as ``[tag] {...}`` lines, so
``pytest -s`` output doubles as the report.
"""

import gc
import json
import math
from pathlib import Path

import pytest
import torch

from musubi_tuner.modules.convrot_int8_kernels import dequantize_int8_convrot_weight
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
from musubi_tuner.yue2 import yue2_protocol as P
from musubi_tuner.yue2.yue2_checkpoint import load_yue2_model, load_yue2_tokenizer, load_yue2_vae, read_base_io
from musubi_tuner.yue2.yue2_model import yue2_t_embed_input
from musubi_tuner.yue2.yue2_sampling import synthesize_latents

DEV = torch.device("cuda")
STYLE = "pop rock, female vocal, energetic, 120 bpm"
FALLBACK_LYRICS = "[verse]\nWalking down the empty street\nCounting every heartbeat\n[chorus]\nHold on, hold on tonight\n"


def report(tag: str, value) -> None:
    print(f"[{tag}] {json.dumps(value, default=str)}", flush=True)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def free(*objs) -> None:
    del objs
    gc.collect()
    torch.cuda.empty_cache()


def load(path, **kw):
    return load_yue2_model(str(path), device=DEV, loading_device=DEV, **kw)


@pytest.fixture(scope="module")
def tokenizer(yue2_weights):
    yue2_weights.require(yue2_weights.comfy_bf16)
    return load_yue2_tokenizer(None, str(yue2_weights.comfy_bf16))


@pytest.fixture(scope="module")
def prefix(tokenizer):
    import os

    lyrics = FALLBACK_LYRICS
    data = os.environ.get("YUE2_DATA_DIR")
    files = sorted((Path(data) / "jamendolyrics" / "lyrics").glob("*.txt")) if data else []
    if files:
        lyrics = "\n".join(files[0].read_text(encoding="utf-8", errors="replace").splitlines()[:12])
    style, lyrics = P.normalize_prompt_fields(STYLE, lyrics)
    return P.build_prefix(P.text_ids(tokenizer, style, lyrics, "off"), "off", None)


@pytest.fixture(scope="module")
def hf_model(yue2_weights):
    yue2_weights.require(yue2_weights.hf)
    model = load(yue2_weights.hf)
    yield model
    free(model)


@pytest.fixture(scope="module")
def ref_model(yue2_weights, reference_wheel):
    from yue2.modeling_yue2 import YuE2Config as RefConfig, YuE2ForCausalLM

    raw = json.loads(yue2_weights.hf_config.read_text())
    cfg = RefConfig(**{k: v for k, v in raw.items() if k in RefConfig._inference_fields})
    with torch.device("meta"):
        model = YuE2ForCausalLM(cfg)
    sd = {}
    with MemoryEfficientSafeOpen(str(yue2_weights.hf)) as f:
        for key in f.keys():
            sd[key] = f.get_tensor(key, device=DEV)
    model.load_state_dict(sd, strict=True, assign=True)
    model = model.to(DEV).eval()
    yield model
    free(model, sd)


# region loaders


def test_hf_equals_comfy_bf16(yue2_weights, hf_model):
    yue2_weights.require(yue2_weights.comfy_bf16)
    comfy = load(yue2_weights.comfy_bf16)
    sd_hf, sd_cb = hf_model.state_dict(), comfy.state_dict()
    assert sd_hf.keys() == sd_cb.keys()
    mismatches = [k for k in sd_hf if not torch.equal(sd_hf[k], sd_cb[k])]
    with MemoryEfficientSafeOpen(str(yue2_weights.comfy_bf16)) as f:
        norm_te = f.get_tensor("text_encoders.model.norm.weight")
        norm_dm = f.get_tensor("model.diffusion_model.model.norm.weight")
    pe = sd_hf["nar.latent_pos_embed.pe"].float()
    n, h = pe.shape
    pos = torch.arange(n, dtype=torch.float32, device=pe.device)[:, None]
    div = torch.exp(torch.arange(0, h, 2, dtype=torch.float32, device=pe.device) * (-math.log(10000.0) / h))
    sinus = torch.zeros_like(pe)
    sinus[:, 0::2], sinus[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    report(
        "hf_vs_comfy_bf16",
        {"tensors": len(sd_hf), "bit_mismatches": len(mismatches), "pe_vs_sinusoid_maxabs": (pe - sinus).abs().max().item()},
    )
    free(comfy, sd_cb)
    assert not mismatches, mismatches[:5]
    assert torch.equal(norm_te, norm_dm)


def test_convrot_from_hf_equals_from_comfy_bf16(yue2_weights):
    yue2_weights.require(yue2_weights.hf, yue2_weights.comfy_bf16)
    a = load(yue2_weights.hf, convrot_int8=True)
    sd_a = {k: v.cpu() for k, v in a.state_dict().items()}
    free(a)
    b = load(yue2_weights.comfy_bf16, convrot_int8=True)
    sd_b = b.state_dict()
    mismatches = [k for k in sd_a if not torch.equal(sd_a[k], sd_b[k].cpu())]
    n_int8 = sum(1 for v in sd_a.values() if v.dtype == torch.int8)
    report("convrot_hf_vs_comfy_bf16", {"tensors": len(sd_a), "int8_tensors": n_int8, "mismatches": len(mismatches)})
    free(b, sd_b)
    assert n_int8 > 0
    assert not mismatches, mismatches[:5]


def test_prequant_int8_dequant_error(yue2_weights, hf_model):
    yue2_weights.require(yue2_weights.comfy_int8)
    q = load(yue2_weights.comfy_int8, prequant_lm_head="dequant")
    sd_hf = hf_model.state_dict()
    per_tensor = {}
    for name, mod in q.named_modules():
        if hasattr(mod, "scale_weight") and mod.weight.dtype == torch.int8:
            w = dequantize_int8_convrot_weight(mod.weight, mod.scale_weight, mod._convrot_groupsize)
            per_tensor[name] = rel(w, sd_hf[name + ".weight"])
    rows = torch.arange(0, P.EOD, 997, device=DEV)[None]
    with torch.no_grad():
        embed_rel = rel(q.embed(rows), hf_model.embed(rows))
    heads = {m: rel(q.get_submodule(m).weight, sd_hf[m + ".weight"]) for m in ("ar.lm_head", "nar.llm2vae")}
    io_rel = rel(read_base_io(str(yue2_weights.comfy_int8))["nar.llm2vae.weight"].to(DEV), sd_hf["nar.llm2vae.weight"])
    worst = max(per_tensor.items(), key=lambda kv: kv[1])
    report(
        "prequant_int8",
        {"quantized_linears": len(per_tensor), "worst": worst, "embed_rows": embed_rel, "heads": heads, "read_base_io": io_rel},
    )
    free(q)
    # int8 block rounding: the worst block Linear measured 1.15% (target 1%, see docs/yue2.md)
    assert len(per_tensor) == 2 * 28 * 4
    assert worst[1] < 2e-2
    assert embed_rel < 2e-2 and io_rel < 2e-2 and all(v < 2e-2 for v in heads.values())


# endregion

# region tokenizer

TOKENIZER_STRINGS = [
    "你好世界 音乐",
    "さくら 桜の花が咲く",
    'X:1\nM:4/4\nL:1/8\nK:C\n"C"CDEF|"G"GABc|',
    "a <|endoftext|> b",
    "<abc>x</abc>",
    "<extra_0> y",
    "école école",
    "[verse]\nWalking down the street\n",
    "rock, electric guitar, male vocal, 120 bpm",
]


def _tokenizer_corpus():
    import os

    strings = list(TOKENIZER_STRINGS)
    data = os.environ.get("YUE2_DATA_DIR")
    if data:
        for path in sorted((Path(data) / "jamendolyrics" / "lyrics").glob("*.txt"))[:40]:
            strings.extend(path.read_text(encoding="utf-8", errors="replace").splitlines()[:5])
    return strings


def test_tokenizer_ids_below_eod(tokenizer):
    strings = _tokenizer_corpus()
    ids = [tokenizer.encode(s) for s in strings]
    report("tokenizer_json", {"strings": len(strings), "tokens": sum(map(len, ids)), "backend": tokenizer.backend})
    assert all(0 <= i < P.EOD for seq in ids for i in seq)
    assert tokenizer.encode("<|endoftext|>") != [P.EOD]


def test_json_equals_tiktoken(yue2_weights, tokenizer):
    pytest.importorskip("tiktoken")
    yue2_weights.require(yue2_weights.tiktoken)
    tt = load_yue2_tokenizer(str(yue2_weights.tiktoken), None)
    strings = _tokenizer_corpus()
    diff = [s for s in strings if tokenizer.encode(s) != tt.encode(s)]
    report("tokenizer_json_vs_tiktoken", {"strings": len(strings), "different": len(diff)})
    assert not diff, diff[:3]


# endregion

# region AR parity


def test_ar_prefill_and_greedy_parity(hf_model, ref_model, prefix, reference_wheel):
    from yue2.modeling_yue2 import StaticKVCache

    ids = torch.tensor([prefix], device=DEV)
    with torch.no_grad():
        lr = ref_model(input_ids=ids, use_cache=False, return_dict=True).logits.float()
        lo = hf_model.lm_logits(hf_model.ar_forward(hf_model.embed(ids)).hidden).float()
    top1 = (lr.argmax(-1) == lo.argmax(-1)).float().mean().item()

    steps = 50
    with torch.no_grad():
        cfg = ref_model.config
        cache = StaticKVCache(
            cfg.num_hidden_layers, 1, cfg.num_key_value_heads, len(prefix) + steps + 1, cfg.head_dim, torch.bfloat16, DEV
        )
        out = ref_model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
        ref_tokens = []
        for _ in range(steps):
            nxt = out.logits[:, -1].argmax(-1)
            ref_tokens.append(int(nxt))
            out = ref_model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True, return_dict=True)
        oc = hf_model.new_kv_cache(len(prefix) + steps + 1)
        logits = hf_model.ar_prefill_into_cache(ids, oc)
        our_tokens = []
        for _ in range(steps):
            nxt = logits.argmax(-1)
            our_tokens.append(int(nxt))
            logits = hf_model.ar_decode_step(nxt[:, None], oc)
    equal = sum(a == b for a, b in zip(ref_tokens, our_tokens))
    report(
        "ar_parity",
        {
            "prefix_len": len(prefix),
            "top1_agree": top1,
            "maxabs": (lr - lo).abs().max().item(),
            "rel": rel(lo, lr),
            "greedy_equal": equal,
        },
    )
    assert top1 >= 0.995
    assert equal >= 45


# endregion

# region NAR parity


def test_nar_velocity_and_ode_parity(yue2_weights, hf_model, ref_model, prefix, reference_wheel):
    from yue2 import nar as wnar

    yue2_weights.require(yue2_weights.vae_dir)
    g = torch.Generator().manual_seed(0)
    codes = [int(c) for c in torch.randint(0, P.CODEC_SIZE, (1500,), generator=g)]
    result = {}
    for secs in (10, 60):
        n = secs * P.FRAME_RATE
        chunk = wnar.song_chunks(prefix, codes[:n], seed=123)[0]
        engine = wnar.CachedNAR(ref_model, chunk)
        per_t = {}
        with torch.no_grad():
            kv = hf_model.ar_forward(
                hf_model.embed(torch.tensor([chunk.ar_tokens], device=DEV)), return_kv=True, return_hidden=False
            ).kv
            for t in (0.98, 0.8, 0.5, 0.2, 0.03):
                state = (0.7 * chunk.noise).to(DEV, torch.bfloat16)
                raw = torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
                vr = engine.velocity(state, raw).float()
                t_in = yue2_t_embed_input(torch.tensor([t], dtype=torch.float64)).to(DEV)
                vo = hf_model.nar_forward(state[None], t_in, kv, len(chunk.ar_tokens))[0].float()
                per_t[t] = rel(vo, vr)
        engine.close()
        result[f"{secs}s_velocity_rel"] = per_t
        assert max(per_t.values()) < 1e-2, per_t

    n = 10 * P.FRAME_RATE
    lat_ref = wnar.synthesize(ref_model, prefix, codes[:n], seed=123)
    lat_our = synthesize_latents(hf_model, prefix, codes[:n], seed=123)  # the shipped sampler, incl. its noise
    assert lat_our.shape == lat_ref.shape, (lat_our.shape, lat_ref.shape)
    result["10s_ode_latent_rel"] = rel(lat_our, lat_ref)
    result["10s_ode_bit_equal"] = torch.equal(lat_our, lat_ref.float().cpu())
    vae = load_yue2_vae(str(yue2_weights.vae_dir), None, device=DEV, decoder_only=True, allow_fp16_source=False)
    with torch.no_grad():
        audio_our = vae.decode_tiled(lat_our.T[None].to(DEV), output_device="cpu")
        audio_ref = vae.decode_tiled(lat_ref.T[None].to(DEV), output_device="cpu")
    noise = (audio_our - audio_ref).pow(2).sum().item()
    snr = math.inf if noise == 0 else 10 * math.log10(audio_ref.pow(2).sum().item() / noise)
    result["10s_decoded_snr_db"] = snr
    report("nar_parity", result)
    free(vae)
    assert result["10s_ode_latent_rel"] < 2e-2
    assert snr > 30


# endregion

# region VAE


def _decode_mp3(path: Path) -> torch.Tensor:
    import av

    chunks = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=P.SAMPLE_RATE)
        for frame in container.decode(audio=0):
            for x in resampler.resample(frame):
                chunks.append(torch.from_numpy(x.to_ndarray()))
        for x in resampler.resample(None):
            chunks.append(torch.from_numpy(x.to_ndarray()))
    return torch.cat(chunks, dim=1).float()


def test_vae_parity(yue2_weights, yue2_data_dir, reference_wheel):
    from yue2.modeling_vae import YuE2VAE as RefVAE

    yue2_weights.require(yue2_weights.vae_dir)
    mp3 = sorted((yue2_data_dir / "mp3").glob("*.mp3"))[0]
    wav = _decode_mp3(mp3)
    hop = P.SAMPLE_RATE // P.FRAME_RATE
    start = P.SAMPLE_RATE * 30
    audio30 = wav[:, start : start + 750 * hop].contiguous()
    audio60 = wav[:, start : start + 1500 * hop].contiguous()
    vae = load_yue2_vae(str(yue2_weights.vae_dir), None, device=DEV, allow_fp16_source=False)
    ref = RefVAE.from_pretrained(str(yue2_weights.vae_dir), device="cuda")
    with torch.no_grad():
        lo = vae.encode_mean(audio30[None].to(DEV))
        lr = ref.encode(audio30[None].to(DEV))
        do = vae.decode_tiled(lo, output_device="cpu")
        dr = ref.decode_tiled(lr, output_device="cpu")
        full60 = vae.encode_mean(audio60[None].to(DEV))[0].T
        chunked60 = vae.encode_mean_chunked(audio60.to(DEV), 750, 50)
    res = {
        "file": mp3.name,
        "encode_maxabs": (lo - lr).abs().max().item(),
        "decode_tiled_maxabs": (do - dr).abs().max().item(),
        "chunked_750_50_vs_full_rel": rel(chunked60, full60),
        "latent_shape": list(lo.shape),
    }
    report("vae_parity", res)
    free(vae, ref)
    assert res["encode_maxabs"] < 1e-4
    assert res["decode_tiled_maxabs"] < 1e-5
    assert res["chunked_750_50_vs_full_rel"] < 1e-3


# endregion

# region ComfyUI export keys


def test_comfy_export_keys_match_checkpoint(yue2_weights):
    from gpu.yue2_comfy_export_check import checkpoint_shapes, comfy_key_report, random_joint_lora
    from musubi_tuner.yue2.yue2_lora_formats import native_to_comfy, to_native

    yue2_weights.require(yue2_weights.comfy_bf16)
    shapes = checkpoint_shapes(str(yue2_weights.comfy_bf16))
    sd, metadata = random_joint_lora(train_io="full")
    comfy_sd, _ = native_to_comfy(to_native(sd, metadata))
    res = comfy_key_report(comfy_sd, shapes)
    report("comfy_export_keys", res)
    assert res["unmapped"] == [] and res["shape_errors"] == [] and res["stray_keys"] == []
    assert res["modules"]["ar"] == 28 * 4 and res["modules"]["nar"] == 28 * 4
    assert res["modules"]["io"] == 2


# endregion
