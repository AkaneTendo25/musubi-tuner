"""yue2_generate_music.py: CLI parsing, prompt files (new line options), end to end with monkeypatched loaders, LoRA
merge/attach routing."""

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_yue2_sampling import FakeLM  # noqa: E402
from yue2_fakes import FakeTokenizer, load_fake_yue2_vae  # noqa: E402

from musubi_tuner import yue2_generate_music as G  # noqa: E402
from musubi_tuner.training.sampling_prompts import line_to_prompt_dict, load_prompts  # noqa: E402
from musubi_tuner.yue2 import yue2_lora_formats as formats  # noqa: E402
from musubi_tuner.yue2.yue2_checkpoint import CheckpointLayout  # noqa: E402
from musubi_tuner.yue2.yue2_model import YuE2Config, YuE2Model  # noqa: E402
from musubi_tuner.yue2.yue2_protocol import CODEC_SIZE  # noqa: E402

TINY = YuE2Config.tiny()


class GenFakeModel(FakeLM):
    """FakeLM with the loader-side API the generator touches; runs under the generator's inference_mode."""

    base_quant = "bf16"
    checkpoint_layout = "hf"
    blocks_to_swap = 0

    def __init__(self, **kw):
        super().__init__(end_after=kw.pop("end_after", None), strict=False)
        self.load_kwargs = kw

    def set_attention(self, *a, **k):
        self.attention = (a, k)

    def eval(self):
        return self

    def requires_grad_(self, flag=True):
        return self

    def to(self, *a, **k):
        return self


@pytest.fixture
def patched(monkeypatch):
    state = {}

    def fake_load(path, **kw):
        state["kwargs"] = kw
        state["model"] = GenFakeModel(end_after=8)
        return state["model"]

    monkeypatch.setattr(G, "load_yue2_model", fake_load)
    monkeypatch.setattr(G, "load_yue2_vae", load_fake_yue2_vae)
    monkeypatch.setattr(G, "load_yue2_tokenizer", lambda tokenizer, dit: FakeTokenizer())
    monkeypatch.setattr(G, "detect_layout", lambda path: (CheckpointLayout.HF, state.get("prequant", False)))
    return state


BASE = ["--dit", "fake.safetensors", "--device", "cpu", "--ode_steps", "2", "--vae_core_frames", "8", "--vae_halo_frames", "2"]


def test_cli_defaults_and_validation():
    args = G.parse_args(["--dit", "x", "--save_path", "o", "--style", "pop", "--device", "cpu"])
    assert (args.cot, args.seed, args.ode_steps, args.ode_state_dtype, args.max_tokens, args.min_tokens) == (
        "full",
        831001,
        32,
        "bf16",
        9000,
        200,
    )
    assert (args.abc_temperature, args.abc_top_k, args.abc_max_tokens, args.cfg_scale) == (0.7, 30, 4096, None)
    assert (args.blocks_to_swap, args.ar_blocks_to_swap, args.nar_blocks_to_swap) == (0, 0, 0)
    assert args.nar_context == "codes" and args.output_format == "flac" and args.t_embed_dtype == "bf16"
    args = G.parse_args(["--dit", "x", "--save_path", "o", "--blocks_to_swap", "10", "--nar_blocks_to_swap", "4"])
    assert (args.ar_blocks_to_swap, args.nar_blocks_to_swap, args.blocks_to_swap) == (10, 4, 10)
    with pytest.raises(ValueError):
        G.parse_args(["--dit", "x", "--save_path", "o", "--convrot_int8", "--fp8_scaled"])
    with pytest.raises(ValueError):
        G.parse_args(["--dit", "x", "--save_path", "o", "--codes_file", "a", "--reconstruct_cache", "b"])


def test_prompt_line_options():
    d = line_to_prompt_dict("dream pop, airy --lyf words.txt --cot OFF --abcf score.abc --sec 30 --mode render --d 42 --s 16")
    assert d == {
        "prompt": "dream pop, airy",
        "lyrics_file": "words.txt",
        "cot": "off",
        "abc_file": "score.abc",
        "seconds": 30.0,
        "mode": "render",
        "seed": 42,
        "sample_steps": 16,
    }
    d = line_to_prompt_dict("x --recon cache/song_000000-000750_yue2.safetensors --l 1.2 --o name")
    assert (
        d["reconstruct_cache"] == "cache/song_000000-000750_yue2.safetensors"
        and d["cfg_scale"] == 1.2
        and d["output_name"] == "name"
    )
    # existing options keep their meaning
    assert line_to_prompt_dict("p --w 64 --f 5")["width"] == 64


def test_render_end_to_end_then_reconstruct_from_artifacts(patched, tmp_path):
    out = tmp_path / "out"
    args = BASE + ["--save_path", str(out), "--style", "pop", "--lyrics", "la la", "--cot", "off", "--min_tokens", "0"]
    paths = G.main(args + ["--save_artifacts", "--output_name", "song"])[0]
    assert Path(paths["audio"]).name == "song.flac" and Path(paths["audio"]).stat().st_size > 0
    art = Path(paths["artifacts"])
    codes = load_file(str(art / "codes.safetensors"))
    latents = load_file(str(art / "latents.safetensors"))["latents"]
    settings = json.loads((art / "settings.json").read_text(encoding="utf-8"))
    assert codes["codes"].numel() == 8 and latents.shape == (8, 64)
    assert settings["cot"] == "off" and settings["cfg_scale"] == 1.01 and settings["frames"] == 8
    assert patched["kwargs"]["lora_weights"] is None and patched["kwargs"]["lm_head_needed"] is True
    assert int(codes["codes"].max()) < CODEC_SIZE

    # reconstruct from the saved codes + exact prefix reproduces the NAR latents
    paths = G.main(args + ["--codes_file", str(art / "codes.safetensors"), "--save_artifacts", "--output_name", "rec"])[0]
    rec = load_file(str(Path(paths["artifacts"]) / "latents.safetensors"))["latents"]
    assert torch.equal(rec, latents)
    assert patched["kwargs"]["lm_head_needed"] is False
    assert patched["model"].prefills == []  # no AR


def test_from_file_with_relative_paths_and_reconstruct_cache(patched, tmp_path):
    (tmp_path / "words.txt").write_text("verse one\nverse two\n", encoding="utf-8")
    cache = tmp_path / "song_000000-000012_yue2.safetensors"
    save_file({"latents_12x64_float32": torch.randn(12, 64), "codes_int64": torch.arange(12)}, str(cache))
    te = {f"varlen_yue2_text_{c}_int64": torch.tensor([151643, 65, 66]) for c in ("off", "melody", "full")}
    te.update({f"varlen_yue2_neg_{c}_int64": torch.tensor([151643]) for c in ("off", "melody", "full")})
    te.update({"varlen_yue2_abc_int64": torch.zeros(0, dtype=torch.int64), "yue2_has_abc_int64": torch.tensor(0)})
    te["yue2_abc_mode_int64"] = torch.tensor(0)
    save_file(te, str(tmp_path / "song_yue2_te.safetensors"))
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        f"# comment\nsynth pop --lyf words.txt --cot off --sec 0.3 --o a\nignored --recon {cache.name} --cot full --o b\n",
        encoding="utf-8",
    )
    assert load_prompts(str(prompts))[0]["lyrics_file"] == "words.txt"
    outputs = G.main(BASE + ["--save_path", str(tmp_path / "o"), "--from_file", str(prompts), "--min_tokens", "0"])
    assert [Path(p["audio"]).name for p in outputs] == ["a.flac", "b.flac"]
    assert Path(outputs[1]["ground_truth"]).name == "b_gt.flac"
    # line 1: lyrics read from the file next to the prompt file
    model = patched["model"]
    assert model.prefills and bytes(t for t in model.prefills[0] if t < 256).decode().count("verse two") == 1


def test_reconstruct_cache_uses_cached_text_ids(patched, tmp_path):
    cache = tmp_path / "s_000000-000010_yue2.safetensors"
    save_file({"latents_10x64_float32": torch.randn(10, 64), "codes_int64": torch.arange(10)}, str(cache))
    te = {f"varlen_yue2_text_{c}_int64": torch.tensor([151643, 1, 2, 3]) for c in ("off", "melody", "full")}
    te.update({f"varlen_yue2_neg_{c}_int64": torch.tensor([151643]) for c in ("off", "melody", "full")})
    te.update(
        {
            "varlen_yue2_abc_int64": torch.tensor([9, 9]),
            "yue2_has_abc_int64": torch.tensor(1),
            "yue2_abc_mode_int64": torch.tensor(1),
        }
    )
    save_file(te, str(tmp_path / "s_yue2_te.safetensors"))
    args = G.parse_args(BASE + ["--save_path", str(tmp_path), "--reconstruct_cache", str(cache), "--cot", "melody"])
    request, extras = G.build_request(args)
    assert request.mode == "reconstruct" and request.cot == "melody" and request.abc_ids == [9, 9]
    assert request.text_ids == [151643, 1, 2, 3] and request.codes == list(range(10)) and extras["gt_latents"].shape == (10, 64)


def _write_native_lora(path, config=TINY, rank=4):
    g = torch.Generator().manual_seed(0)
    sd = {}
    for name, (out_f, in_f) in {
        "lora_unet_nar_blocks_0_self_attn_o_proj": (config.hidden_size, config.q_dim),
        "lora_unet_ar_blocks_1_mlp_down_proj": (config.hidden_size, config.intermediate_size),
    }.items():
        sd[name + ".lora_down.weight"] = torch.randn(rank, in_f, generator=g) * 0.1
        sd[name + ".lora_up.weight"] = torch.randn(out_f, rank, generator=g) * 0.1
        sd[name + ".alpha"] = torch.tensor(float(rank))
    save_file(sd, str(path))
    return sd


def test_lora_merge_route_scales_branches(patched, tmp_path):
    lora = tmp_path / "l.safetensors"
    sd = _write_native_lora(lora)
    G.main(
        BASE
        + ["--save_path", str(tmp_path / "o"), "--style", "p", "--cot", "off", "--seconds", "0.2", "--min_tokens", "0"]
        + ["--lora_weight", str(lora), "--lora_multiplier", "0.5", "--lora_ar_multiplier", "0", "--lora_nar_multiplier", "2"]
    )
    kw = patched["kwargs"]
    assert kw["lora_multipliers"] == [0.5]
    merged = kw["lora_weights"][0]
    nar, ar = "lora_unet_nar_blocks_0_self_attn_o_proj", "lora_unet_ar_blocks_1_mlp_down_proj"
    assert torch.allclose(merged[nar + ".lora_up.weight"], sd[nar + ".lora_up.weight"] * 2)
    assert merged[ar + ".lora_up.weight"].abs().max() == 0


def test_lora_attach_route_for_prequant(patched, tmp_path, monkeypatch):
    lora = tmp_path / "l.safetensors"
    _write_native_lora(lora)
    patched["prequant"] = True
    seen = {}
    monkeypatch.setattr(G, "attach_loras", lambda model, loras, ar, nar, device: seen.update(n=len(loras), ar=ar, nar=nar) or [])
    G.main(
        BASE
        + ["--save_path", str(tmp_path / "o"), "--style", "p", "--cot", "off", "--seconds", "0.2", "--min_tokens", "0"]
        + ["--lora_weight", str(lora), "--lora_ar_multiplier", "0.25"]
    )
    assert patched["kwargs"]["lora_weights"] is None and seen == {"n": 1, "ar": 0.25, "nar": 1.0}


def test_attach_loras_on_tiny_model(tmp_path):
    torch.manual_seed(0)
    model = YuE2Model(TINY).float().eval()
    lora = tmp_path / "l.safetensors"
    _write_native_lora(lora)
    sd, meta = formats.load_lora_file(str(lora))
    loras = [(str(lora), formats.to_native(sd, meta), 1.0)]
    x = torch.randn(1, 5, TINY.q_dim)
    o_proj = model.nar.blocks[0].self_attn.o_proj
    with torch.no_grad():
        base = o_proj(x)
    networks = G.attach_loras(model, loras, ar_multiplier=0.5, nar_multiplier=1.0, device="cpu")
    mults = {m.lora_name: m.multiplier for m in networks[0].unet_loras}
    assert mults == {"lora_unet_nar_blocks_0_self_attn_o_proj": 1.0, "lora_unet_ar_blocks_1_mlp_down_proj": 0.5}
    with torch.no_grad():
        delta = o_proj(x) - base
        want = (
            x
            @ sd["lora_unet_nar_blocks_0_self_attn_o_proj.lora_down.weight"].T
            @ sd["lora_unet_nar_blocks_0_self_attn_o_proj.lora_up.weight"].T
        )
    assert torch.allclose(delta, want, atol=1e-5)
