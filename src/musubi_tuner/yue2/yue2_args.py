"""Model/runtime arguments shared by the YuE2 entry points (trainer, generator, cache scripts),
so one option is spelled, defaulted and validated once."""

from __future__ import annotations

import argparse
import logging

from musubi_tuner.yue2.yue2_protocol import DEFAULT_INSTRUMENTAL_LYRICS

logger = logging.getLogger(__name__)

YUE2_NUM_LAYERS = 28
# ModelOffloader needs at least two resident blocks per list
YUE2_MAX_BLOCKS_TO_SWAP = YUE2_NUM_LAYERS - 2

SDPA_GQA_MODES = ("repeat", "native")
PREQUANT_LM_HEAD_MODES = ("auto", "keep", "dequant")
T_EMBED_DTYPES = ("bf16", "fp32")
ODE_STATE_DTYPES = ("bf16", "fp32")


def setup_parser_yue2_model(parser: argparse.ArgumentParser, training: bool) -> argparse.ArgumentParser:
    """Add the shared YuE2 model/runtime flags.

    ``--vae`` is added only when the parser does not define it yet (the training and cache parsers already do).
    The ODE state dtype is ``--sample_ode_state_dtype`` for training (sampling during training) and
    ``--ode_state_dtype`` for the generator.
    """
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="YuE2 text tokenizer: a tokenizer .json, qwen.tiktoken, or a directory holding one"
        " (default: embedded in a ComfyUI --dit, else qwen.tiktoken next to a Hugging Face model.safetensors)",
    )
    if "--vae" not in parser._option_string_actions:
        parser.add_argument(
            "--vae",
            type=str,
            default=None,
            help="YuE2 VAE: m-a-p/YuE2-Vae directory or safetensors file (default: vae.* of a ComfyUI --dit)",
        )
    parser.add_argument(
        "--sdpa_gqa",
        type=str,
        default="repeat",
        choices=SDPA_GQA_MODES,
        help="grouped-query attention with torch SDPA: repeat (expand K/V heads, default; avoids the math fallback)"
        " or native (enable_gqa)",
    )
    parser.add_argument(
        "--attn_query_tile",
        type=int,
        default=256,
        help="query tile length for --split_attn (exact causal masks per tile)",
    )
    parser.add_argument(
        "--ar_blocks_to_swap",
        type=int,
        default=None,
        help=f"AR blocks to swap to CPU (default: --blocks_to_swap, max {YUE2_MAX_BLOCKS_TO_SWAP})",
    )
    parser.add_argument(
        "--nar_blocks_to_swap",
        type=int,
        default=None,
        help=f"NAR blocks to swap to CPU (default: --blocks_to_swap, max {YUE2_MAX_BLOCKS_TO_SWAP})",
    )
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use dynamic scaled fp8 for the AR/NAR block Linears" + (" (requires --fp8_base)" if training else ""),
    )
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help="quantize the bf16 AR/NAR block Linears to ConvRot int8 at load time (Hadamard rotation + int8;"
        " cannot be combined with fp8). ComfyUI pre-quantized int8 checkpoints are detected automatically",
    )
    parser.add_argument(
        "--convrot_int8_bwd",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="backward mode for a ConvRot int8 base. bf16 (default): transient dequantized matmul, most accurate."
        " int8: reuse the fused int8 GEMM for grad_x (faster, requires triton and CUDA)",
    )
    parser.add_argument(
        "--quantize_lm_head",
        action="store_true",
        help="also quantize ar.lm_head with --convrot_int8/--fp8_scaled (default: keep it bf16 for exact CE)",
    )
    parser.add_argument(
        "--prequant_lm_head",
        type=str,
        default="auto",
        choices=PREQUANT_LM_HEAD_MODES,
        help="ar.lm_head of a pre-quantized int8 checkpoint: auto (dequantize to bf16 only when AR logits are needed),"
        " keep (int8) or dequant",
    )
    parser.add_argument(
        "--t_embed_dtype",
        type=str,
        default="bf16",
        choices=T_EMBED_DTYPES,
        help="time embedder input: bf16 (reference expression sigmoid(bf16(logit(t))), default) or fp32 (plain t)",
    )
    parser.add_argument(
        "--sample_ode_state_dtype" if training else "--ode_state_dtype",
        type=str,
        default="bf16",
        choices=ODE_STATE_DTYPES,
        help="dtype of the NAR ODE state during sampling (bf16 matches the official sampler)",
    )
    parser.add_argument(
        "--allow_fp16_vae",
        action="store_true",
        help="accept a non-fp32 VAE source (the int8 all-in-one stores vae.* as fp16) for latent caching;"
        " encoded targets shift slightly. Decode-only use only warns",
    )
    parser.add_argument(
        "--instrumental_lyrics",
        type=str,
        default=DEFAULT_INSTRUMENTAL_LYRICS,
        help=f"lyrics used for songs without lyrics (default: {DEFAULT_INSTRUMENTAL_LYRICS}); must match the text cache",
    )
    return parser


def normalize_swap_counts(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve per-list swap counts and set ``args.blocks_to_swap = max(ar_n, nar_n)``.

    ``--ar_blocks_to_swap`` / ``--nar_blocks_to_swap`` default to ``--blocks_to_swap``. The resolved counts are
    written back to ``args`` so the call is idempotent; the base trainer reads only ``args.blocks_to_swap``.
    """
    base = getattr(args, "blocks_to_swap", None) or 0
    ar_n = getattr(args, "ar_blocks_to_swap", None)
    nar_n = getattr(args, "nar_blocks_to_swap", None)
    ar_n = base if ar_n is None else ar_n
    nar_n = base if nar_n is None else nar_n
    if ar_n < 0 or nar_n < 0:
        raise ValueError(f"block swap counts must be non-negative, got AR {ar_n}, NAR {nar_n}")
    args.ar_blocks_to_swap = ar_n
    args.nar_blocks_to_swap = nar_n
    args.blocks_to_swap = max(ar_n, nar_n)
    return ar_n, nar_n


def validate_yue2_model_args(args: argparse.Namespace, training: bool) -> None:
    """Reject unsupported combinations of the shared flags (and, for training, the precision/attention regime).

    ``--convrot_int8_bwd int8`` without ``--convrot_int8`` is not rejected here: a pre-quantized checkpoint is an
    int8 base too, which is known only after the header is read.
    """
    fp8_base = bool(getattr(args, "fp8_base", False))
    fp8_scaled = bool(getattr(args, "fp8_scaled", False))
    convrot = bool(getattr(args, "convrot_int8", False))
    if training:
        if fp8_base and not fp8_scaled:
            raise ValueError("YuE2 fp8 supports only scaled fp8: pass --fp8_scaled together with --fp8_base")
        if fp8_scaled and not fp8_base:
            raise ValueError("--fp8_scaled requires --fp8_base for training")
    if convrot and (fp8_base or fp8_scaled):
        raise ValueError("--convrot_int8 cannot be combined with --fp8_base/--fp8_scaled: choose one quantization")
    if getattr(args, "quantize_lm_head", False) and not (convrot or fp8_scaled):
        raise ValueError("--quantize_lm_head requires --convrot_int8 or --fp8_scaled")

    query_tile = getattr(args, "attn_query_tile", 256)
    if query_tile is None or query_tile < 1:
        raise ValueError(f"--attn_query_tile must be >= 1, got {query_tile}")

    for flag in ("blocks_to_swap", "ar_blocks_to_swap", "nar_blocks_to_swap"):
        n = getattr(args, flag, None)
        if n is not None and not 0 <= n <= YUE2_MAX_BLOCKS_TO_SWAP:
            raise ValueError(f"--{flag} must be in [0, {YUE2_MAX_BLOCKS_TO_SWAP}] for YuE2, got {n}")

    if training:
        mixed_precision = getattr(args, "mixed_precision", None)
        if mixed_precision == "fp16":
            raise ValueError("YuE2 training requires --mixed_precision bf16 (fp16 is not supported)")
        if mixed_precision != "bf16":
            logger.warning(f"YuE2 is validated with --mixed_precision bf16; got {mixed_precision!r}. The model still runs in bf16.")
        if getattr(args, "sage_attn", False):
            raise ValueError("--sage_attn is not supported for YuE2 training (no backward); use --sdpa or --flash_attn")
