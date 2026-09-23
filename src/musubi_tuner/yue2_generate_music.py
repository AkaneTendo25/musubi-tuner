"""YuE2 music generation: style + lyrics -> (ABC plan) -> semantic codes -> NAR flow latents -> 48 kHz stereo audio.

Follows the official ``yue2_infer`` (same prompt protocol, sampler, seeds and 32-step midpoint ODE) on the musubi
model: bf16, ComfyUI pre-quantized int8, on-the-fly ``--convrot_int8`` / ``--fp8_scaled``, two-list block swap and
LoRA.

LoRA: bf16 sources merge LoRA(s) into the weights at load (before quantization); a pre-quantized int8 checkpoint, or
``--lora_attach``, attaches them as runtime branches. Every known YuE2 LoRA format is accepted (native, ComfyUI,
ai-toolkit, yue2-lora-v1, fl-yue2-lora-v1, Mothersuperior safetensors / .pt). ``--lora_ar_multiplier`` /
``--lora_nar_multiplier`` scale the AR / NAR (incl. I/O) modules.

Modes:
  * render (default): the AR writes the ABC score (cot melody/full without ``--abc``) and the semantic codes
  * reconstruct: ``--codes_file`` (codes .safetensors/.npy/.json/.txt) or ``--reconstruct_cache`` (a YuE2 latent
    cache with codes; its text cache supplies the prompt when ``--style`` is not given) -> NAR -> VAE

Input: a single prompt (``--style``/``--lyrics``), ``--from_file`` (.txt lines, .json or .toml as for training sample
prompts) or ``--interactive``. Line options: ``--lyf <lyrics file> --cot off --abcf <abc file> --sec 30 --d 42 --s 32
--l 1.01 --mode render --recon <latent cache> --o <output name>``.
"""

import argparse
import copy
import dataclasses
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import torch

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_yue2
from musubi_tuner.training.sampling_prompts import line_to_prompt_dict, load_prompts
from musubi_tuner.yue2 import yue2_lora_formats as formats
from musubi_tuner.yue2.yue2_args import normalize_swap_counts, setup_parser_yue2_model, validate_yue2_model_args
from musubi_tuner.yue2.yue2_audio_io import write_audio
from musubi_tuner.yue2.yue2_checkpoint import detect_layout, load_yue2_model, load_yue2_tokenizer, load_yue2_vae, read_base_io
from musubi_tuner.yue2.yue2_protocol import ABC_DEFAULTS, COT_MODES, PROTOCOL_VERSION, SEMANTIC_DEFAULTS, SamplingParams
from musubi_tuner.yue2.yue2_sampling import (
    NAR_CONTEXTS,
    TEXT_ONLY_ROPE,
    YuE2SongRequest,
    YuE2SongResult,
    decode_latents,
    load_reconstruct_cache,
    reconstruct,
    render_song,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# prompt-file keys (training sample prompt names) -> generator args
PROMPT_KEYS = {
    "prompt": "style",
    "style": "style",
    "lyrics": "lyrics",
    "lyrics_file": "lyrics_file",
    "cot": "cot",
    "abc": "abc",
    "abc_file": "abc_file",
    "seconds": "seconds",
    "seed": "seed",
    "sample_steps": "ode_steps",
    "cfg_scale": "cfg_scale",
    "guidance_scale": "cfg_scale",
    "mode": "mode",
    "reconstruct_cache": "reconstruct_cache",
    "codes_file": "codes_file",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "repetition_penalty": "repetition_penalty",
    "penalty_window": "penalty_window",
    "min_tokens": "min_tokens",
    "max_tokens": "max_tokens",
    "abc_temperature": "abc_temperature",
    "abc_top_p": "abc_top_p",
    "abc_top_k": "abc_top_k",
    "output_name": "output_name",
}
PATH_KEYS = ("lyrics_file", "abc_file", "reconstruct_cache", "codes_file")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate music with YuE2")
    parser.add_argument("--dit", type=str, required=True, help="YuE2 checkpoint: HF model.safetensors or ComfyUI bf16/int8 file")

    g = parser.add_argument_group("prompt")
    g.add_argument("--style", "--prompt", dest="style", type=str, default=None, help="style tags (the [Tags] section)")
    g.add_argument("--lyrics", type=str, default=None, help="lyrics text (empty -> --instrumental_lyrics)")
    g.add_argument("--lyrics_file", type=str, default=None, help="read lyrics from a text file")
    g.add_argument("--cot", type=str, default="full", choices=COT_MODES, help="ABC chain-of-thought mode (default full)")
    g.add_argument("--abc", type=str, default=None, help="ABC score to use instead of planning one (cot melody/full)")
    g.add_argument("--abc_file", type=str, default=None, help="read the ABC score from a file")
    g.add_argument("--seed", type=int, default=831001, help="seed of both AR phases and the NAR noise (default 831001)")
    g.add_argument("--seconds", type=float, default=None, help="limit the song to this many seconds (25 codes per second)")

    g = parser.add_argument_group("semantic sampling")
    s = SEMANTIC_DEFAULTS
    g.add_argument("--max_tokens", type=int, default=s.max_tokens, help=f"semantic token budget (default {s.max_tokens})")
    g.add_argument(
        "--min_tokens", type=int, default=s.min_tokens, help=f"no MUSIC_END before this many codes (default {s.min_tokens})"
    )
    g.add_argument("--temperature", type=float, default=s.temperature)
    g.add_argument("--top_p", type=float, default=s.top_p)
    g.add_argument("--top_k", type=int, default=s.top_k)
    g.add_argument("--repetition_penalty", type=float, default=s.repetition_penalty)
    g.add_argument("--penalty_window", type=int, default=s.penalty_window)
    g.add_argument("--cfg_scale", type=float, default=None, help="CFG scale (default: protocol, 1.01 for cot=off, else 1.0)")

    g = parser.add_argument_group("ABC sampling")
    a = ABC_DEFAULTS
    g.add_argument("--abc_temperature", type=float, default=a.temperature)
    g.add_argument("--abc_top_p", type=float, default=a.top_p)
    g.add_argument("--abc_top_k", type=int, default=a.top_k)
    g.add_argument("--abc_repetition_penalty", type=float, default=a.repetition_penalty)
    g.add_argument("--abc_penalty_window", type=int, default=a.penalty_window)
    g.add_argument("--abc_min_tokens", type=int, default=a.min_tokens)
    g.add_argument("--abc_max_tokens", type=int, default=a.max_tokens)

    g = parser.add_argument_group("NAR / VAE")
    g.add_argument("--ode_steps", type=int, default=32, help="midpoint ODE steps (default 32)")
    g.add_argument("--nar_context", type=str, default="codes", choices=NAR_CONTEXTS, help="NAR context (default codes)")
    g.add_argument(
        "--nar_text_only_rope", type=str, default="full", choices=TEXT_ONLY_ROPE, help="NAR RoPE layout for --nar_context text_only"
    )
    g.add_argument("--codes_file", type=str, default=None, help="reconstruct from these codes (skips the AR)")
    g.add_argument("--reconstruct_cache", type=str, default=None, help="reconstruct from a YuE2 latent cache (codes_int64)")
    g.add_argument("--vae_core_frames", type=int, default=1024, help="VAE tile core frames (default 1024)")
    g.add_argument("--vae_halo_frames", type=int, default=16, help="VAE tile halo frames (default 16)")
    g.add_argument("--vae_cpu", action="store_true", help="decode on CPU instead of the device")

    g = parser.add_argument_group("LoRA")
    g.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA file(s), any known YuE2 format")
    g.add_argument("--lora_multiplier", type=float, nargs="*", default=None, help="multiplier per LoRA (default 1.0)")
    g.add_argument("--lora_ar_multiplier", type=float, default=1.0, help="extra factor on the AR modules of every LoRA")
    g.add_argument("--lora_nar_multiplier", type=float, default=1.0, help="extra factor on the NAR (and I/O) modules of every LoRA")
    g.add_argument(
        "--lora_attach", action="store_true", help="attach LoRA(s) at runtime instead of merging (implied for int8 prequant)"
    )

    g = parser.add_argument_group("memory / speed")
    g.add_argument("--blocks_to_swap", type=int, default=0, help="blocks per stack to swap to CPU (max 26)")
    g.add_argument("--use_pinned_memory_for_block_swap", action="store_true", help="pinned CPU memory for block swap")
    g.add_argument("--block_swap_h2d_only", action="store_true", help="H2D-only block swap (CPU master, ring buffer)")
    g.add_argument("--block_swap_ring_size", type=int, default=2, help="ring buffers for --block_swap_h2d_only")
    g.add_argument(
        "--sample_ar_resident",
        action="store_true",
        help="with block swap: keep the AR stack fully on the device during AR decoding (NAR blocks to CPU)",
    )
    g.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["torch", "sdpa", "flash", "flash3", "xformers", "sageattn"],
        help="attention backend (torch = SDPA)",
    )
    g.add_argument("--split_attn", action="store_true", help="query-tiled attention (--attn_query_tile)")
    g.add_argument("--device", type=str, default=None, help="device (default cuda if available)")

    g = parser.add_argument_group("output / input modes")
    g.add_argument("--save_path", type=str, required=True, help="output directory")
    g.add_argument("--output_format", type=str, default="flac", choices=["flac", "wav"])
    g.add_argument("--output_name", type=str, default=None, help="output file stem (default: timestamp_seed)")
    g.add_argument("--save_artifacts", action="store_true", help="also save abc.txt, codes/latents .safetensors and settings.json")
    g.add_argument("--from_file", type=str, default=None, help="prompts from a .txt (one per line), .json or .toml file")
    g.add_argument("--interactive", action="store_true", help="read prompt lines from the console")

    setup_parser_yue2_model(parser, training=False)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    args = setup_parser().parse_args(argv)
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive")
    if args.codes_file and args.reconstruct_cache:
        raise ValueError("Pass --codes_file or --reconstruct_cache, not both")
    if args.lora_multiplier and args.lora_weight and len(args.lora_multiplier) > len(args.lora_weight):
        raise ValueError("more --lora_multiplier values than --lora_weight files")
    validate_yue2_model_args(args, training=False)
    normalize_swap_counts(args)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.mode = None
    return args


# region models


def _lora_natives(args) -> list[tuple[str, "formats.NativeLoRA", float]]:
    out, base_io = [], None
    for i, path in enumerate(args.lora_weight or []):
        multiplier = args.lora_multiplier[i] if args.lora_multiplier and len(args.lora_multiplier) > i else 1.0
        sd, metadata = formats.load_lora_file(path)
        fmt = formats.detect_format(sd, metadata)
        if fmt.startswith("ms") and base_io is None:
            base_io = read_base_io(args.dit)  # Mothersuperior full I/O weights become diffs against the base
        native = formats.to_native(sd, metadata, base_io=base_io)
        logger.info(f"LoRA {os.path.basename(path)}: format {fmt}, {len(native.modules)} modules, multiplier {multiplier}")
        out.append((path, native, multiplier))
    return out


def attach_loras(model, loras, ar_multiplier: float, nar_multiplier: float, device) -> list:
    """Attach LoRA(s) as runtime branches (base weights untouched); returns the networks, keep them alive."""
    networks = []
    for path, native, multiplier in loras:
        sd = formats.native_state_dict(native)
        network = lora_yue2.create_arch_network_from_weights(multiplier, sd, unet=model, for_inference=True)
        if not network.unet_loras:
            raise ValueError(f"LoRA {path} contains no modules that match the model")
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        network.load_state_dict(sd, strict=True)
        network.set_multiplier_by_branch(ar=ar_multiplier, nar=nar_multiplier)
        network.eval().requires_grad_(False).to(device)
        networks.append(network)
    return networks


def load_models(args):
    """Model (+ merged or attached LoRA, block swap), decoder-only VAE and tokenizer."""
    device = torch.device(args.device)
    _, prequant = detect_layout(args.dit)
    swap = args.blocks_to_swap > 0
    loras = _lora_natives(args)
    attach = bool(loras) and (args.lora_attach or prequant)
    merge_inputs = merge_mults = None
    if loras and not attach:
        merge_inputs = [
            formats.native_fused_state_dict(formats.scale_branches(n, ar=args.lora_ar_multiplier, nar=args.lora_nar_multiplier))
            for _, n, _ in loras
        ]
        merge_mults = [m for _, _, m in loras]

    t0 = time.perf_counter()
    model = load_yue2_model(
        args.dit,
        device=device,
        loading_device="cpu" if swap else device,
        dtype=torch.bfloat16,
        attn_mode=args.attn_mode,
        split_attn=args.split_attn,
        fp8_scaled=args.fp8_scaled,
        convrot_int8=args.convrot_int8,
        convrot_int8_bwd=args.convrot_int8_bwd,
        quantize_lm_head=args.quantize_lm_head,
        prequant_lm_head=args.prequant_lm_head,
        lm_head_needed=not (args.codes_file or args.reconstruct_cache),
        lora_weights=merge_inputs,
        lora_multipliers=merge_mults,
    )
    model.set_attention(args.attn_mode, args.split_attn, args.attn_query_tile, args.sdpa_gqa)
    model.eval().requires_grad_(False)
    if swap:
        model.set_block_swap_plan(args.ar_blocks_to_swap, args.nar_blocks_to_swap, ar_backward=False, nar_backward=False)
        model.enable_block_swap(args.blocks_to_swap, BlockSwapConfig.from_args(args, device, supports_backward=False))
        model.move_to_device_except_swap_blocks(device)
        model.switch_block_swap_for_inference()
    else:
        model.to(device)
    networks = attach_loras(model, loras, args.lora_ar_multiplier, args.lora_nar_multiplier, device) if attach else []
    logger.info(
        f"YuE2 model ready in {time.perf_counter() - t0:.1f}s (base {model.base_quant}, layout {model.checkpoint_layout},"
        f" LoRA {'attached' if attach else 'merged' if loras else 'none'})"
    )

    vae = load_yue2_vae(args.vae, args.dit if args.vae is None else None, device="cpu", decoder_only=True, allow_fp16_source=True)
    if not args.vae_cpu:
        vae.to(device)
    tokenizer = load_yue2_tokenizer(args.tokenizer, args.dit)
    return model, vae, tokenizer, networks


# endregion

# region requests


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_codes_file(path: str) -> tuple[list[int], Optional[list[int]]]:
    """Codes from ``.safetensors`` (``codes``/``codes_int64``, optional ``prefix``), ``.npy``, ``.json`` or ``.txt``."""
    ext = os.path.splitext(path)[1].lower()
    prefix = None
    if ext == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(path)
        key = "codes" if "codes" in sd else "codes_int64"
        if key not in sd:
            raise ValueError(f"{path} has no codes tensor (codes / codes_int64)")
        codes = sd[key].flatten().tolist()
        if "prefix" in sd:
            prefix = [int(t) for t in sd["prefix"].tolist()]
    elif ext == ".npy":
        import numpy as np

        codes = np.load(path, allow_pickle=False).reshape(-1).tolist()
    elif ext == ".json":
        codes = json.loads(_read_text(path))
    else:
        codes = [int(x) for x in _read_text(path).replace(",", " ").split()]
    return [int(c) for c in codes], prefix


def build_request(args) -> tuple[YuE2SongRequest, dict]:
    """``YuE2SongRequest`` from resolved args; also returns extras (ground-truth latents, exact prefix, sources)."""
    extras: dict[str, Any] = {}
    lyrics = args.lyrics
    if args.lyrics_file:
        lyrics = _read_text(args.lyrics_file)
    abc = args.abc
    if args.abc_file:
        abc = _read_text(args.abc_file)
    if abc is not None and args.cot == "off":
        logger.warning("--abc is ignored with --cot off")
        abc = None

    kwargs: dict[str, Any] = {}
    codes = None
    mode = args.mode or ("reconstruct" if (args.codes_file or args.reconstruct_cache) else "render")
    if mode == "reconstruct":
        if args.codes_file:
            codes, prefix = load_codes_file(args.codes_file)
            extras["prefix"] = prefix
            extras["source"] = args.codes_file
        elif args.reconstruct_cache:
            src = load_reconstruct_cache(args.reconstruct_cache)
            codes = src.codes
            extras["gt_latents"] = src.latents
            extras["source"] = args.reconstruct_cache
            if args.style is None and args.lyrics is None and args.lyrics_file is None:
                if not src.text_ids:
                    raise ValueError(f"no text cache next to {args.reconstruct_cache}; pass --style/--lyrics")
                cot = args.cot if (args.cot == "off" or src.abc_ids is not None) else "off"
                kwargs.update(text_ids=src.text_ids[cot], neg_ids=src.neg_ids.get(cot), cot=cot)
                if cot != "off":
                    kwargs["abc_ids"] = src.abc_ids
        else:
            raise ValueError("reconstruct mode needs --codes_file or --reconstruct_cache (--recon)")

    semantic = SamplingParams(
        args.temperature,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
        args.penalty_window,
        min(args.min_tokens, args.max_tokens),
        args.max_tokens,
    )
    abc_params = SamplingParams(
        args.abc_temperature,
        args.abc_top_p,
        args.abc_top_k,
        args.abc_repetition_penalty,
        args.abc_penalty_window,
        min(args.abc_min_tokens, args.abc_max_tokens),
        args.abc_max_tokens,
    )
    request = YuE2SongRequest(
        style=args.style or "",
        lyrics=lyrics or "",
        cot=kwargs.pop("cot", args.cot),
        abc=abc,
        seed=args.seed,
        seconds=args.seconds,
        cfg_scale=args.cfg_scale,
        semantic=semantic,
        abc_params=abc_params,
        ode_steps=args.ode_steps,
        state_dtype=args.ode_state_dtype,
        mode=mode,
        codes=codes,
        nar_context=args.nar_context,
        text_only_rope=args.nar_text_only_rope,
        t_embed_mode=args.t_embed_dtype,
        instrumental_lyrics=args.instrumental_lyrics,
        vae_core_frames=args.vae_core_frames,
        vae_halo_frames=args.vae_halo_frames,
        ar_resident=args.sample_ar_resident,
        **kwargs,
    )
    return request, extras


def apply_prompt(args, prompt: dict, base_dir: Optional[str] = None) -> argparse.Namespace:
    """Copy of ``args`` with the prompt-file / line overrides applied (relative paths resolved against ``base_dir``)."""
    out = copy.deepcopy(args)
    for key, value in prompt.items():
        if key in ("enum",):
            continue
        name = PROMPT_KEYS.get(key)
        if name is None:
            logger.warning(f"YuE2 prompt key ignored: {key}")
            continue
        if name in PATH_KEYS and value and base_dir and not os.path.isabs(value):
            value = os.path.join(base_dir, value)
        setattr(out, name, value)
    if out.mode is not None and out.mode not in ("render", "reconstruct"):
        raise ValueError(f"mode must be render or reconstruct, got {out.mode!r}")
    return out


# endregion

# region run


def _time_flag() -> str:
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]


def save_outputs(args, result: YuE2SongResult, request: YuE2SongRequest, extras: dict, vae=None) -> dict:
    os.makedirs(args.save_path, exist_ok=True)
    stem = args.output_name or f"{_time_flag()}_{request.seed}"
    base = os.path.join(args.save_path, stem)
    tags = {
        "yue2_protocol": PROTOCOL_VERSION,
        "yue2_mode": request.mode,
        "yue2_cot": result.cot,
        "yue2_seed": request.seed,
        "yue2_cfg_scale": result.cfg_scale,
        "comment": request.style,
    }
    paths = {"audio": write_audio(f"{base}.{args.output_format}", result.waveform, result.sample_rate, args.output_format, tags)}
    gt = extras.get("gt_latents")
    if gt is not None and vae is not None:
        wave = decode_latents(vae, gt[: len(result.codes)], request.vae_core_frames, request.vae_halo_frames)
        paths["ground_truth"] = write_audio(f"{base}_gt.{args.output_format}", wave, result.sample_rate, args.output_format)
    if args.save_artifacts:
        from safetensors.torch import save_file

        directory = base + "_artifacts"
        os.makedirs(directory, exist_ok=True)
        if result.abc_text is not None:
            with open(os.path.join(directory, "abc.txt"), "w", encoding="utf-8") as f:
                f.write(result.abc_text)
        save_file(
            {"codes": torch.tensor(result.codes, dtype=torch.int64), "prefix": torch.tensor(result.prefix, dtype=torch.int64)},
            os.path.join(directory, "codes.safetensors"),
        )
        save_file({"latents": result.latents.float().contiguous()}, os.path.join(directory, "latents.safetensors"))
        settings = {
            "request": {k: v for k, v in dataclass_dict(request).items() if k not in ("codes", "text_ids", "neg_ids", "abc_ids")},
            "dit": args.dit,
            "lora_weight": args.lora_weight,
            "lora_multiplier": args.lora_multiplier,
            "lora_ar_multiplier": args.lora_ar_multiplier,
            "lora_nar_multiplier": args.lora_nar_multiplier,
            "cot": result.cot,
            "cfg_scale": result.cfg_scale,
            "truncated": result.truncated,
            "budget": result.budget,
            "timings": result.timings,
            "frames": len(result.codes),
            "seconds": result.seconds,
            "source": extras.get("source"),
            "protocol": PROTOCOL_VERSION,
        }
        with open(os.path.join(directory, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False, default=str)
        paths["artifacts"] = directory
    for kind, path in paths.items():
        logger.info(f"saved {kind}: {path}")
    return paths


def dataclass_dict(obj) -> dict:
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def generate(args, model, vae, tokenizer) -> tuple[YuE2SongResult, dict]:
    request, extras = build_request(args)
    if getattr(model, "blocks_to_swap", 0):
        model.prepare_block_swap_before_forward()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    if extras.get("prefix") is not None and request.mode == "reconstruct":
        # exact prefix saved by --save_artifacts: reproduce the NAR input without re-tokenizing
        result = reconstruct(
            model,
            vae,
            extras["prefix"],
            request.codes if request.seconds is None else request.codes[: max(1, round(request.seconds * 25))],
            request.seed,
            request.ode_steps,
            request.state_dtype,
            request.nar_context,
            request.text_only_rope,
            request.t_embed_mode,
            request.vae_core_frames,
            request.vae_halo_frames,
        )
        result.cot = request.cot
    else:
        result = render_song(model, vae, tokenizer, request)
    if dev.type == "cuda":
        result.timings["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 3)
    sem = result.timings.get("semantic", {})
    logger.info(
        f"YuE2 {request.mode}: {len(result.codes)} frames ({result.seconds:.1f}s audio)"
        + (f", {sem.get('output_tps', 0):.1f} tok/s" if sem else "")
        + f", NAR {result.timings.get('nar_seconds', 0):.1f}s, VAE {result.timings.get('vae_seconds', 0):.1f}s"
        + (f", truncated {result.truncated}" if any(result.truncated.values()) else "")
    )
    return result, save_outputs(args, result, request, extras, vae)


def _prompt_file_entries(path: str) -> list[dict]:
    return load_prompts(path)


def main(argv=None):
    args = parse_args(argv)
    model, vae, tokenizer, networks = load_models(args)
    outputs = []
    # generation only: models are built outside; inference_mode is used only in this entry point (the sampler also
    # runs inside training, where inference tensors are not allowed)
    with torch.inference_mode():
        if args.from_file:
            base_dir = os.path.dirname(os.path.abspath(args.from_file))
            for i, prompt in enumerate(_prompt_file_entries(args.from_file)):
                prompt_args = apply_prompt(args, prompt, base_dir)
                logger.info(f"[{i}] style: {prompt_args.style!r}")
                outputs.append(generate(prompt_args, model, vae, tokenizer)[1])
        elif args.interactive:
            print("Interactive mode. Enter prompt lines (Ctrl+D or Ctrl+Z (Windows) to exit):")
            while True:
                try:
                    line = input("> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                try:
                    outputs.append(generate(apply_prompt(args, line_to_prompt_dict(line), os.getcwd()), model, vae, tokenizer)[1])
                except KeyboardInterrupt:
                    print("\nInterrupted")
        else:
            if args.style is None and args.lyrics is None and args.lyrics_file is None and not args.reconstruct_cache:
                raise ValueError("pass --style/--lyrics (or --from_file / --interactive)")
            outputs.append(generate(args, model, vae, tokenizer)[1])
    del networks
    logger.info("Done!")
    return outputs


if __name__ == "__main__":
    main()
