import argparse
import base64
import dataclasses
import hashlib
import inspect
import logging
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import FakeTokenizer  # noqa: E402
from yue2_ref import protocol as ref  # noqa: E402

from musubi_tuner.training.parser_common import setup_parser_common  # noqa: E402
from musubi_tuner.yue2 import yue2_args, yue2_checkpoint, yue2_model  # noqa: E402
from musubi_tuner.yue2 import yue2_protocol as p  # noqa: E402

STYLE = "dream pop, female vocal, 92 bpm"
LYRICS = "[verse]\nhello <|endoftext|> world\n[chorus]\nla la"
ABC_IDS = [5, 17, 300, 151642]


# region protocol vs the vendored reference


def test_constants_match_reference():
    for name in (
        "EOD",
        "ABC_START",
        "ABC_END",
        "MUSIC_START",
        "MUSIC_END",
        "CODEC_OFFSET",
        "CODEC_SIZE",
        "LATENT_START",
        "LATENT_END",
        "LATENT_PAD",
        "VOCAB_SIZE",
        "CONTEXT",
        "PROTOCOL_VERSION",
    ):
        assert getattr(p, name) == getattr(ref, name), name
    assert p.INSTRUCTIONS == ref.INSTRUCTIONS
    assert p.COT_MODES == tuple(ref.INSTRUCTIONS)
    assert p.SAMPLE_RATE == p.HOP * p.FRAME_RATE
    assert p.MERT_SAMPLE_RATE == p.MERT_HOP * p.FRAME_RATE
    assert p.CODEC_OFFSET + p.CODEC_SIZE == p.LATENT_START


@pytest.mark.parametrize("cot", p.COT_MODES)
def test_prompt_text_matches_reference(cot):
    assert p.prompt_text(STYLE, LYRICS, cot) == ref.SongRequest(STYLE, LYRICS, cot=cot).text()


@pytest.mark.parametrize("cot", p.COT_MODES)
@pytest.mark.parametrize("abc", [None, ABC_IDS])
def test_prefix_matches_reference(cot, abc):
    tok = FakeTokenizer()
    request = ref.SongRequest(STYLE, LYRICS, cot=cot)
    expected = ref.token_prefixes(request, tok, abc_ids=abc)
    got = p.build_prefix(p.text_ids(tok, STYLE, LYRICS, cot), cot, abc)
    assert got == expected
    assert got[0] == p.EOD
    if cot == "off":
        assert got[-3:] == [p.ABC_START, p.ABC_END, p.MUSIC_START]
    elif abc is None:
        assert got[-1] == p.ABC_START
    else:
        assert got[-len(abc) - 3 :] == [p.ABC_START, *abc, p.ABC_END, p.MUSIC_START]


@pytest.mark.parametrize("cot", p.COT_MODES)
def test_negative_prefix_matches_reference(cot):
    tok = FakeTokenizer()
    request = ref.SongRequest(STYLE, LYRICS, cot=cot)
    neg = p.negative_text_ids(tok, cot)
    assert p.build_negative_prefix(neg, cot, ABC_IDS) == ref.negative_prefix(request, tok, abc_ids=ABC_IDS)
    if cot == "off":
        assert p.build_negative_prefix(neg, cot, None) == ref.negative_prefix(request, tok) == neg + [p.MUSIC_START]
    else:
        with pytest.raises(ValueError):
            p.build_negative_prefix(neg, cot, None)
        with pytest.raises(ValueError):
            ref.negative_prefix(request, tok)


@pytest.mark.parametrize(
    "bad", [[p.EOD], [-1], [p.ABC_START], [True], [1.0], [p.CODEC_OFFSET]], ids=["eod", "neg", "abc", "bool", "float", "codec"]
)
def test_abc_ids_validated(bad):
    text = [p.EOD, 1, 2]
    for cot in ("melody", "full"):
        with pytest.raises(ValueError):
            p.build_prefix(text, cot, bad)
        with pytest.raises(ValueError):
            p.build_negative_prefix(text, cot, bad)
    # off ignores ABC entirely (the AR never sees the score)
    assert p.build_prefix(text, "off", bad) == text + [p.ABC_START, p.ABC_END, p.MUSIC_START]


def test_abc_ids_accept_tensors_and_numpy():
    np = pytest.importorskip("numpy")
    text = [p.EOD, 7]
    expected = p.build_prefix(text, "full", ABC_IDS)
    assert p.build_prefix(torch.tensor(text), "full", torch.tensor(ABC_IDS)) == expected
    assert p.build_prefix(np.array(text), "full", np.array(ABC_IDS, dtype=np.int64)) == expected


def test_unknown_cot_rejected():
    with pytest.raises(ValueError):
        p.prompt_text(STYLE, LYRICS, "auto")
    with pytest.raises(ValueError):
        p.build_prefix([p.EOD], "chords", None)
    with pytest.raises(ValueError):
        p.default_cfg("auto")


def _expected_chunk_ranges(frames, prefix_len, context=p.CONTEXT):
    # independent oracle: consecutive windows of the largest size s with prefix_len + 2 * s + 3 <= context
    size = (context - prefix_len - 3) // 2
    assert frames >= 1 and size >= 1
    ranges, start = [], 0
    while start < frames:
        ranges.append((start, min(start + size, frames)))
        start += size
    return ranges


@pytest.mark.parametrize("frames", [1, 2, 749, 750, 12285, 12286, 30000])
@pytest.mark.parametrize("prefix_len", [3, 100, 1000, 16000, 24570])
def test_chunk_ranges_match_reference(frames, prefix_len):
    got = p.chunk_ranges(frames, prefix_len)
    assert got == ref.chunk_ranges(frames, prefix_len) == _expected_chunk_ranges(frames, prefix_len)
    assert got[0][0] == 0 and got[-1][1] == frames
    assert all(a < b for a, b in got) and all(got[i][1] == got[i + 1][0] for i in range(len(got) - 1))
    size = p.max_nar_window(prefix_len)
    assert size == (p.CONTEXT - prefix_len - 3) // 2
    assert all(b - a <= size for a, b in got)
    assert prefix_len + 2 * size + 3 <= p.CONTEXT


def test_chunk_ranges_errors():
    for frames, prefix_len in ((0, 10), (10, p.CONTEXT - 4), (10, p.CONTEXT)):
        with pytest.raises(ValueError):
            p.chunk_ranges(frames, prefix_len)
        with pytest.raises(ValueError):
            ref.chunk_ranges(frames, prefix_len)
    with pytest.raises(ValueError):
        p.max_nar_window(p.CONTEXT - 4)
    assert p.max_nar_window(p.CONTEXT - 5) == 1


def test_codec_id_conversion():
    codes = [0, 1, p.CODEC_SIZE - 1]
    ids = p.codec_to_ids(codes)
    assert ids == [p.CODEC_OFFSET, p.CODEC_OFFSET + 1, p.CODEC_OFFSET + p.CODEC_SIZE - 1]
    assert p.ids_to_codec(ids) == codes
    t = torch.tensor(codes, dtype=torch.long)
    assert torch.equal(p.codec_to_ids(t), torch.tensor(ids))
    assert torch.equal(p.ids_to_codec(p.codec_to_ids(t)), t)
    assert p.codec_to_ids(torch.zeros(0, dtype=torch.long)).numel() == 0
    for bad in ([-1], [p.CODEC_SIZE]):
        with pytest.raises(ValueError):
            p.codec_to_ids(bad)
        with pytest.raises(ValueError):
            p.codec_to_ids(torch.tensor(bad))
    for bad in ([p.CODEC_OFFSET - 1], [p.CODEC_OFFSET + p.CODEC_SIZE], [p.MUSIC_END]):
        with pytest.raises(ValueError):
            p.ids_to_codec(bad)
        with pytest.raises(ValueError):
            p.ids_to_codec(torch.tensor(bad))


def test_sampling_defaults_match_reference():
    cfg = ref.GenerationConfig()
    assert dataclasses.asdict(p.SEMANTIC_DEFAULTS) == dataclasses.asdict(cfg.semantic)
    assert dataclasses.asdict(p.ABC_DEFAULTS) == dataclasses.asdict(cfg.abc)
    assert dataclasses.asdict(p.SamplingParams()) == dataclasses.asdict(ref.Sampling())
    assert [f.name for f in dataclasses.fields(p.SamplingParams)] == [f.name for f in dataclasses.fields(ref.Sampling)]
    assert p.ODE_STEPS == cfg.ode_steps
    for bad in (
        dict(top_k=0),
        dict(top_k=1.0),
        dict(top_p=0.0),
        dict(temperature=6.0),
        dict(temperature=float("nan")),
        dict(penalty_window=101),
        dict(repetition_penalty=0.0),
        dict(min_tokens=10, max_tokens=5),
        dict(max_tokens=0, min_tokens=0),
    ):
        with pytest.raises(ValueError):
            p.SamplingParams(**bad)
        with pytest.raises(ValueError):
            ref.Sampling(**bad)


@pytest.mark.parametrize("cot", p.COT_MODES)
def test_default_cfg_matches_reference(cot):
    assert p.default_cfg(cot) == ref.SongRequest(STYLE, LYRICS, cot=cot).guidance


# endregion

# region prompt fields and budgets


def test_normalize_prompt_fields():
    style, lyrics = p.normalize_prompt_fields("  rock,\n\tguitar   solo ", "\n[verse]\r\nline one\r\n\n")
    assert style == "rock, guitar solo"
    assert lyrics == "[verse]\nline one"
    assert p.normalize_prompt_fields("a", None) == ("a", "[instrumental]")
    assert p.normalize_prompt_fields("a", " \n ") == ("a", "[instrumental]")
    assert p.normalize_prompt_fields("a", "", instrumental_lyrics="[inst]") == ("a", "[inst]")
    assert p.normalize_prompt_fields(None, "x") == ("", "x")
    long_style, _ = p.normalize_prompt_fields("word " * 1000, "x")
    assert len(long_style) == p.STYLE_MAX_CHARS
    # whitespace collapsed, then truncated
    raw = "a  b\n" * 700
    assert p.normalize_prompt_fields(raw, "x")[0] == " ".join(raw.split())[:1500]


def test_abc_mode_of():
    melody = "X:1\nT:song\nM:4/4\nL:1/8\nK:C\n|: CDEF GABc :|\n"
    full = 'X:1\nK:G\n|"G" GABc "D7" d2 d2 |"Em" e4 |\n'
    annotated = 'X:1\nK:C\n"^Chorus" CDEF | "_slow" GABc |\n'
    assert p.abc_mode_of(melody) == "melody"
    assert p.abc_mode_of(full) == "full"
    assert p.abc_mode_of(annotated) == "melody"
    assert p.abc_mode_of(annotated + '"Am" A4 |\n') == "full"
    assert p.abc_mode_of("") == "melody"


def test_generation_budget():
    assert p.generation_budget(500, 40, 9000) == 9000
    # a 16k-token prefix caps the budget instead of raising (the official sampler raises)
    assert p.generation_budget(16000, 40, 9000) == p.CONTEXT - 16000
    assert p.generation_budget(100, 17000, 9000) == p.CONTEXT - 17000
    assert p.generation_budget(p.CONTEXT - 5, 10, 9000) == 5
    with pytest.raises(ValueError):
        p.generation_budget(p.CONTEXT - 4, 10, 9000)
    with pytest.raises(ValueError):
        p.generation_budget(10, p.CONTEXT, 9000)
    with pytest.raises(ValueError):
        p.generation_budget(10, 10, 0)


def test_frames_and_audio_spec():
    assert p.frames_for_samples(1920 * 25 + 1919) == 25
    assert p.yue2_samples_per_crop(25) == 48000
    with pytest.raises(ValueError):
        p.yue2_samples_per_crop(0)
    fn = pickle.loads(pickle.dumps(p.yue2_samples_per_crop))
    assert fn is p.yue2_samples_per_crop
    spec = p.YUE2_AUDIO_SPEC
    assert (spec.sample_rate, spec.channels, spec.codec_pad_tolerance) == (48000, 2, p.HOP)
    assert spec.samples_per_crop(3) == 3 * p.HOP
    assert p.YUE2_AUDIO_SPEC is spec
    clone = pickle.loads(pickle.dumps(spec))
    assert clone == spec
    from musubi_tuner.yue2.yue2_protocol import YUE2_AUDIO_SPEC

    assert YUE2_AUDIO_SPEC is spec
    with pytest.raises(AttributeError):
        p.NOT_A_NAME  # noqa: B018


def test_protocol_import_is_light():
    code = "import sys; import musubi_tuner.yue2.yue2_protocol as p; p.chunk_ranges(10, 10); assert 'torch' not in sys.modules, 'torch imported'"
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# endregion

# region shared arguments

SHARED_FLAGS = (
    "--tokenizer",
    "--vae",
    "--sdpa_gqa",
    "--attn_query_tile",
    "--ar_blocks_to_swap",
    "--nar_blocks_to_swap",
    "--convrot_int8",
    "--convrot_int8_bwd",
    "--fp8_scaled",
    "--quantize_lm_head",
    "--prequant_lm_head",
    "--t_embed_dtype",
    "--allow_fp16_vae",
    "--instrumental_lyrics",
)


def _train_parser():
    parser = setup_parser_common()
    yue2_args.setup_parser_yue2_model(parser, training=True)
    return parser


def _gen_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    yue2_args.setup_parser_yue2_model(parser, training=False)
    return parser


def test_shared_parser_defaults():
    for parser, ode_flag, ode_dest in (
        (_train_parser(), "--sample_ode_state_dtype", "sample_ode_state_dtype"),
        (_gen_parser(), "--ode_state_dtype", "ode_state_dtype"),
    ):
        for flag in SHARED_FLAGS + (ode_flag,):
            assert flag in parser._option_string_actions, flag
        args = parser.parse_args([])
        assert args.tokenizer is None and args.vae is None
        assert args.sdpa_gqa == "repeat"
        assert args.attn_query_tile == 256
        assert args.ar_blocks_to_swap is None and args.nar_blocks_to_swap is None
        assert args.convrot_int8 is False and args.convrot_int8_bwd == "bf16"
        assert args.fp8_scaled is False and args.quantize_lm_head is False
        assert args.prequant_lm_head == "auto"
        assert args.t_embed_dtype == "bf16"
        assert getattr(args, ode_dest) == "bf16"
        assert args.allow_fp16_vae is False
        assert args.instrumental_lyrics == "[instrumental]"
    assert "--ode_state_dtype" not in _train_parser()._option_string_actions
    assert "--sample_ode_state_dtype" not in _gen_parser()._option_string_actions
    with pytest.raises(SystemExit):
        _gen_parser().parse_args(["--sdpa_gqa", "grouped"])
    with pytest.raises(SystemExit):
        _gen_parser().parse_args(["--prequant_lm_head", "int8"])


@pytest.mark.parametrize(
    "argv, expected, blocks",
    [
        ([], (0, 0), 0),
        (["--blocks_to_swap", "14"], (14, 14), 14),
        (["--nar_blocks_to_swap", "14"], (0, 14), 14),
        (["--ar_blocks_to_swap", "20"], (20, 0), 20),
        (["--blocks_to_swap", "14", "--ar_blocks_to_swap", "0", "--nar_blocks_to_swap", "0"], (0, 0), 0),
        (["--blocks_to_swap", "10", "--ar_blocks_to_swap", "4"], (4, 10), 10),
    ],
)
def test_normalize_swap_counts(argv, expected, blocks):
    args = _train_parser().parse_args(argv)
    assert yue2_args.normalize_swap_counts(args) == expected
    assert args.blocks_to_swap == blocks
    assert (args.ar_blocks_to_swap, args.nar_blocks_to_swap) == expected
    # idempotent
    assert yue2_args.normalize_swap_counts(args) == expected
    assert args.blocks_to_swap == blocks


def test_normalize_swap_counts_rejects_negative():
    args = _gen_parser().parse_args(["--ar_blocks_to_swap", "-1"])
    with pytest.raises(ValueError):
        yue2_args.normalize_swap_counts(args)


@pytest.mark.parametrize(
    "argv",
    [
        ["--fp8_base"],
        ["--fp8_scaled"],
        ["--convrot_int8", "--fp8_base", "--fp8_scaled"],
        ["--quantize_lm_head"],
        ["--mixed_precision", "fp16"],
        ["--sage_attn"],
        ["--blocks_to_swap", "27"],
        ["--nar_blocks_to_swap", "27"],
        ["--attn_query_tile", "0"],
    ],
)
def test_validate_training_rejects(argv):
    args = _train_parser().parse_args(["--mixed_precision", "bf16", *argv] if "--mixed_precision" not in argv else argv)
    with pytest.raises(ValueError):
        yue2_args.validate_yue2_model_args(args, training=True)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--fp8_base", "--fp8_scaled"],
        ["--convrot_int8"],
        ["--convrot_int8", "--quantize_lm_head", "--convrot_int8_bwd", "int8"],
        ["--convrot_int8_bwd", "int8"],  # a pre-quantized checkpoint is an int8 base too
        ["--blocks_to_swap", "26", "--ar_blocks_to_swap", "0"],
        ["--split_attn", "--attn_query_tile", "64", "--sdpa_gqa", "native"],
    ],
)
def test_validate_training_accepts(argv):
    args = _train_parser().parse_args(["--mixed_precision", "bf16", *argv])
    yue2_args.validate_yue2_model_args(args, training=True)


def test_validate_mixed_precision_warns(caplog):
    for value in ("no", None):
        argv = [] if value is None else ["--mixed_precision", value]
        args = _train_parser().parse_args(argv)
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            yue2_args.validate_yue2_model_args(args, training=True)
        assert "bf16" in caplog.text


def test_validate_generator_rules():
    yue2_args.validate_yue2_model_args(_gen_parser().parse_args(["--fp8_scaled"]), training=False)
    for argv in (["--convrot_int8", "--fp8_scaled"], ["--quantize_lm_head"], ["--blocks_to_swap", "30"]):
        with pytest.raises(ValueError):
            yue2_args.validate_yue2_model_args(_gen_parser().parse_args(argv), training=False)


# endregion

# region contracts (merge input, module API, vendored reference)


def _lora(name, out_f, in_f, r=4, alpha=4.0):
    return {
        f"{name}.lora_down.weight": torch.randn(r, in_f),
        f"{name}.lora_up.weight": torch.randn(out_f, r),
        f"{name}.alpha": torch.tensor(alpha),
    }


def test_merge_input_contract_accepts_native_fused():
    sd = {}
    sd.update(_lora("lora_unet_ar_blocks_0_self_attn_qkv_proj", 4096, 2048))
    sd.update(_lora("lora_unet_nar_blocks_27_mlp_gate_up_proj", 12288, 2048, r=8))
    sd.update(_lora("lora_unet_nar_blocks_3_mlp_down_proj", 2048, 6144))
    sd.update(_lora("lora_unet_ar_blocks_12_self_attn_o_proj", 2048, 2048))
    sd["lora_unet_nar_llm2vae.diff"] = torch.randn(64, 2048)
    sd["lora_unet_nar_llm2vae.diff_b"] = torch.randn(64)
    sd["lora_unet_nar_vae2llm.diff"] = torch.randn(2048, 64)
    sd["lora_unet_nar_time_embedder_mlp_0.diff"] = torch.randn(2048, 256)
    kinds = yue2_checkpoint.check_merge_input(sd)
    assert kinds["lora_unet_ar_blocks_0_self_attn_qkv_proj"] == "lora"
    assert kinds["lora_unet_nar_llm2vae"] == "diff"
    assert len(kinds) == 7
    assert yue2_checkpoint.check_merge_input({}) == {}


@pytest.mark.parametrize(
    "extra",
    [
        {"lora_unet_ar_blocks_0_self_attn_qkv_proj.lora_down.0.weight": torch.zeros(4, 2048)},  # split layout
        {"lora_unet_ar_blocks_0_self_attn_q_proj.lora_down.weight": torch.zeros(4, 2048)},  # HF names
        {"diffusion_model.model.layers.0.self_attn.qkv_proj.lora_down.weight": torch.zeros(4, 2048)},  # ComfyUI LoRA names
        {"lora_unet_nar_llm2vae.lora_down.weight": torch.zeros(4, 2048)},  # I/O LoRA must arrive as a diff
        {"lora_unet_nar_llm2vae.weight": torch.zeros(64, 2048)},  # "full" I/O weights must arrive as a diff
        {"lora_unet_nar_latent_pos_embed.diff": torch.zeros(4, 4)},
        {"lora_unet_nar_vae2llm.diff_b": torch.zeros(2048)},  # diff_b without diff
    ],
)
def test_merge_input_contract_rejects(extra):
    sd = _lora("lora_unet_nar_blocks_0_self_attn_o_proj", 2048, 2048)
    sd.update(extra)
    with pytest.raises(ValueError):
        yue2_checkpoint.check_merge_input(sd)


def test_merge_input_contract_rejects_incomplete_modules():
    base = _lora("lora_unet_nar_blocks_0_self_attn_o_proj", 2048, 2048)
    for drop in ("alpha", "lora_up.weight"):
        sd = {k: v for k, v in base.items() if not k.endswith(drop)}
        with pytest.raises(ValueError):
            yue2_checkpoint.check_merge_input(sd)
    sd = dict(base)
    sd["lora_unet_nar_blocks_0_self_attn_o_proj.lora_up.weight"] = torch.zeros(2048, 3)
    with pytest.raises(ValueError):
        yue2_checkpoint.check_merge_input(sd)
    with pytest.raises(ValueError):
        yue2_checkpoint.check_merge_input(
            {"lora_unet_nar_llm2vae.diff": torch.zeros(64, 2048), "lora_unet_nar_llm2vae.diff_b": torch.zeros(3)}
        )


def test_merge_input_module_names():
    f = yue2_checkpoint.merge_input_module_name
    assert f("lora_unet_ar_blocks_3_self_attn_qkv_proj") == "ar.blocks.3.self_attn.qkv_proj"
    assert f("lora_unet_nar_blocks_27_mlp_gate_up_proj") == "nar.blocks.27.mlp.gate_up_proj"
    assert f("lora_unet_ar_blocks_0_mlp_down_proj") == "ar.blocks.0.mlp.down_proj"
    assert f("lora_unet_nar_blocks_1_self_attn_o_proj") == "nar.blocks.1.self_attn.o_proj"
    assert f("lora_unet_nar_llm2vae") == "nar.llm2vae"
    assert f("lora_unet_nar_time_embedder_mlp_2") == "nar.time_embedder.mlp.2"
    with pytest.raises(ValueError):
        f("lora_unet_ar_llm2vae")
    assert yue2_checkpoint.YUE2_QUANT_TARGET_KEYS == ["ar.blocks.", "nar.blocks."]
    assert yue2_checkpoint.YUE2_QUANT_EXCLUDE_KEYS == ["norm"]
    assert {e.value for e in yue2_checkpoint.CheckpointLayout} == {"hf", "comfy", "native"}


def test_config_contract():
    cfg = yue2_model.YuE2Config()
    assert (cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (2048, 28, 16, 8, 128)
    assert (cfg.q_dim, cfg.kv_dim, cfg.intermediate_size, cfg.vocab_size) == (2048, 1024, 6144, p.VOCAB_SIZE)
    assert cfg.max_position_embeddings == p.CONTEXT and cfg.latent_dim == p.LATENT_DIM
    assert cfg.num_layers - 2 == yue2_args.YUE2_MAX_BLOCKS_TO_SWAP
    tiny = yue2_model.YuE2Config.tiny()
    assert (tiny.hidden_size, tiny.num_layers, tiny.num_heads, tiny.num_kv_heads, tiny.head_dim) == (256, 3, 4, 2, 64)
    assert (tiny.intermediate_size, tiny.vocab_size, tiny.max_latent_frames) == (512, 1024, 128)
    assert yue2_model.YuE2Config.tiny(num_layers=5).num_layers == 5
    # the HF config.json of m-a-p/YuE2-3B
    hf = dict(
        hidden_size=2048,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=6144,
        vocab_size=184704,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        max_position_embeddings=24576,
        latent_dim=64,
        max_latent_frames=24576,
        timestep_shift=1.0,
    )
    assert cfg.num_layers == hf["num_hidden_layers"] and cfg.rope_theta == hf["rope_theta"]
    assert cfg.rms_norm_eps == hf["rms_norm_eps"] and cfg.max_latent_frames == hf["max_latent_frames"]


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_module_api_contract():
    m = yue2_model.YuE2Model
    for name in (
        "set_attention",
        "embed",
        "ar_forward",
        "lm_logits",
        "nar_forward",
        "new_kv_cache",
        "ar_prefill_into_cache",
        "ar_decode_step",
        "enable_gradient_checkpointing",
        "disable_gradient_checkpointing",
        "set_block_swap_plan",
        "begin_train_step",
        "enable_block_swap",
        "move_to_device_except_swap_blocks",
        "prepare_block_swap_before_forward",
        "switch_block_swap_for_inference",
        "switch_block_swap_for_training",
        "set_ar_resident",
    ):
        assert callable(getattr(m, name)), name
    assert _params(yue2_model.YuE2Block.forward) == ["self", "x", "cos", "sin", "prefix_k", "prefix_v", "causal", "return_kv"]
    assert _params(m.nar_forward) == ["self", "x_t", "t_embed", "kv", "rope_offset"]
    assert _params(m.set_block_swap_plan) == ["self", "ar_blocks", "nar_blocks", "ar_backward", "nar_backward", "branches"]
    assert _params(yue2_model.yue2_t_embed_input) == ["t", "mode"]
    load_params = inspect.signature(yue2_checkpoint.load_yue2_model).parameters
    for name in ("device", "loading_device", "fp8_scaled", "convrot_int8", "prequant_lm_head", "lora_weights", "lora_multipliers"):
        assert name in load_params, name
    assert "allow_fp16_source" in inspect.signature(yue2_checkpoint.load_yue2_vae).parameters


# official wheel RECORD hashes (yue2_infer-0.1.5.dist-info/RECORD)
_REFERENCE_RECORD = {
    "modeling_vae.py": "tleP9GHmvp_Qecpd3xKJOUDzszOP80Nfzq6JxGZIb1o",
    "modeling_yue2.py": "YtK9dYyP_xoNZFhKaZNDB2_KmniA212OGSY7wwrnnXw",
    "protocol.py": "4A0yvyK6rEbH1Nr67s1PPVpHU9ZRs9yPQWS583dJgK0",
    "storage.py": "pbzBgbFkErRbxWDlJkugswNme1_YGVeZGsAmbIf-Kl0",
    "tokenization_yue2.py": "_Ig4f5bl2yAgoMP1N0uCcmIGAm2DFZtouD5BNrt0Gsw",
}


@pytest.mark.parametrize("name", sorted(_REFERENCE_RECORD))
def test_vendored_reference_is_unmodified(name):
    # compare with LF line endings so a CRLF checkout (git autocrlf) still verifies
    data = (ROOT / "tests" / "yue2_ref" / name).read_bytes().replace(b"\r\n", b"\n")
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    assert digest == _REFERENCE_RECORD[name]


def test_vendored_reference_imports():
    pytest.importorskip("transformers")
    from yue2_ref import modeling_vae, modeling_yue2

    assert hasattr(modeling_yue2, "YuE2ForCausalLM") and hasattr(modeling_yue2.YuE2ForCausalLM, "nar_velocity")
    assert hasattr(modeling_vae, "YuE2VAE")
    for lic in ("LICENSE", "THIRD_PARTY_NOTICES.md", "stable-audio-tools-MIT.txt", "SnakeBeta-NVIDIA-MIT.txt"):
        assert (ROOT / "tests" / "yue2_ref" / "licenses" / lic).is_file()


# endregion
