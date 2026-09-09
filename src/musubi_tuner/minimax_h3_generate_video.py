from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from pathlib import Path

from musubi_tuner.minimax_h3.assets import default_text_encoder_assets
from musubi_tuner.minimax_h3.backend import create_generator
from musubi_tuner.minimax_h3.one_frame import parse_one_frame_options
from musubi_tuner.minimax_h3.references import (
    REFERENCE_IMAGE_SHORT_EDGE,
    REFERENCE_IMAGE_SIZE_MODES,
    REFERENCE_VIDEO_MAX_PIXELS,
    REFERENCE_VIDEO_SHORT_EDGE,
)
from musubi_tuner.minimax_h3.request import SUPPORTED_RATIOS, H3GenerationRequest, H3Guide, make_references
from musubi_tuner.minimax_h3.weights import inspect_checkpoint


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMax H3 local inference entrypoint")
    parser.add_argument("--model", type=Path, required=True, help="Local H3 checkpoint directory, index, or safetensors file")
    parser.add_argument("--text_encoder", type=Path, help="Qwen3-VL H3 BF16 checkpoint or component directory")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=default_text_encoder_assets(),
        help="H3 tokenizer/processor directory; defaults to the metadata bundled with Musubi",
    )
    parser.add_argument("--vae", type=Path, help="MiniMax H3 video VAE checkpoint or component directory")
    parser.add_argument("--audio_vae", type=Path, help="MiniMax H3 audio VAE checkpoint or component directory")
    parser.add_argument("--prompt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from_file", type=Path, help="generate each prompt line into the output directory")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--frame_count", type=int, help="explicit pixel-frame count; 1 enables one-frame generation")
    parser.add_argument("--ratio", choices=SUPPORTED_RATIOS, default="16:9")
    parser.add_argument("--height", type=int, help="explicit non-native canvas height; requires --width")
    parser.add_argument("--width", type=int, help="explicit non-native canvas width; requires --height")
    parser.add_argument("--steps", type=int, default=20, help="sigma grid points, including terminal zero")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first_frame")
    parser.add_argument("--last_frame")
    parser.add_argument(
        "--condition_image",
        action="append",
        default=[],
        help="ordered one-frame FL2VA control; repeatable",
    )
    parser.add_argument(
        "--one_frame",
        nargs="?",
        const="",
        default=None,
        metavar="target_index=N,control_index=A;B",
        help="generate one latent frame with optional 24 fps timeline placement",
    )
    parser.add_argument(
        "--h3_image_mode",
        choices=("none", "first", "first_last"),
        default="none",
        help="generate a still image with one first-frame control or separate first/last controls",
    )
    parser.add_argument("--h3_image_frame_count", type=int, default=5)
    parser.add_argument("--h3_select_frame", type=int, default=0)
    parser.add_argument("--h3_text_visual_max_pixels", type=int, default=0)
    parser.add_argument(
        "--keyframe",
        action="append",
        default=[],
        metavar="INDEX:PATH",
        help=(
            "condition latent frame INDEX on an image, repeatable; negative indices count from the end. "
            "May be combined with Ref2VA reference media"
        ),
    )
    parser.add_argument("--reference_image", action="append", default=[])
    parser.add_argument("--reference_video", action="append", default=[])
    parser.add_argument("--reference_audio", action="append", default=[])
    parser.add_argument(
        "--guide_image",
        action="append",
        default=[],
        metavar="PIXEL_FRAME:PATH",
        help="anchor an image at an arbitrary pixel frame; repeatable, negative frames count from the end",
    )
    parser.add_argument(
        "--guide_video",
        action="append",
        default=[],
        metavar="PIXEL_FRAME:PATH",
        help="anchor a short 24-fps guide video at a pixel frame; repeatable, negative frames count from the end",
    )
    parser.add_argument(
        "--guide_audio",
        action="append",
        default=[],
        metavar="PIXEL_FRAME:PATH",
        help="anchor guide audio at a pixel frame; pair with --guide_video at the same frame for an AV guide",
    )
    parser.add_argument(
        "--reference_image_short_edge",
        type=int,
        default=REFERENCE_IMAGE_SHORT_EDGE,
        help=("scale every reference image so its short edge reaches this many pixels; match the value the LoRA was trained with"),
    )
    parser.add_argument(
        "--reference_image_size_mode",
        choices=REFERENCE_IMAGE_SIZE_MODES,
        default="short_edge",
        help="reference-image sizing strategy; target_area matches the output canvas area",
    )
    parser.add_argument(
        "--reference_image_max_pixels",
        type=int,
        default=0,
        help="optional target-area reference pixel cap; 0 uses the output canvas area",
    )
    parser.add_argument("--reference_video_short_edge", type=int, default=REFERENCE_VIDEO_SHORT_EDGE)
    parser.add_argument("--reference_video_max_pixels", type=int, default=REFERENCE_VIDEO_MAX_PIXELS)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--fp8_base", action="store_true", help="use weight-only scaled FP8 transformer blocks")
    parser.add_argument(
        "--int8_convrot_base",
        action="store_true",
        help="load the pruned Comfy INT8 ConvRot transformer",
    )
    parser.add_argument(
        "--text_encoder_quantization",
        choices=("none", "int8", "nf4", "nvfp4", "nvfp4_awq"),
        default="none",
        help="quantize the Qwen3-VL conditioner while encoding the prompt",
    )
    parser.add_argument(
        "--h3_text_encoder_blocks_to_stream",
        type=int,
        default=0,
        help="stream this many of the 50 frozen Qwen3-VL layers from CPU while encoding prompts (CUDA only)",
    )
    parser.add_argument(
        "--h3_nvfp4_scaled_mm",
        "--nvfp4_scaled_mm",
        action="store_true",
        help="use Blackwell W4A4 scaled_mm for a native NVFP4/AWQ text encoder (PyTorch 2.10+)",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--block_swap_h2d_only", action="store_true")
    parser.add_argument("--block_swap_ring_size", type=int, default=2)
    parser.add_argument("--block_swap_granularity", choices=("block", "layer"), default="block")
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    parser.add_argument("--lora_weight", type=Path, action="append", default=[])
    parser.add_argument("--lora_multiplier", type=float, action="append", default=[])
    parser.add_argument(
        "--h3_learned_context",
        type=Path,
        action="append",
        default=[],
        help="ComfyUI-compatible H3 learned context; repeat to prepend multiple contexts in command-line order",
    )
    parser.add_argument(
        "--h3_learned_context_multiplier",
        type=float,
        action="append",
        default=[],
        help=(
            "scale the corresponding learned context; omitted values default to 1, zero disables that context, "
            "and negative values are experimental rather than a guaranteed inverse"
        ),
    )
    parser.add_argument(
        "--h3_learned_context_composition",
        choices=("prepend", "replace"),
        default="prepend",
        help="prepend the context to Qwen output, or replace the prompt conditioning with the context",
    )
    parser.add_argument(
        "--first_pass_scale",
        type=float,
        default=0.0,
        help=(
            "run a first pass at this fraction of the canvas, then refine at full size. Off by default. "
            "Speed adapters trade motion for steps, so the first pass runs without them and settles the "
            "motion, and the adapter is applied to the refinement that only sharpens"
        ),
    )
    parser.add_argument(
        "--first_pass_steps",
        type=int,
        default=0,
        help="steps for the first pass; defaults to --steps",
    )
    parser.add_argument(
        "--second_pass_strength",
        type=float,
        default=0.5,
        help=(
            "share of the schedule the refinement walks, the usual image-to-image meaning: 1.0 repeats the "
            "whole schedule, 0.5 runs its later half"
        ),
    )
    parser.add_argument(
        "--latent_upscaler",
        type=Path,
        default=None,
        help=(
            "trained latent upscaler weights. H3 is trained at one megapixel and adds no detail above it, "
            "so a larger canvas needs a second model; doing it in latent space keeps speech and lip "
            "movement intact, which a round trip out to pixels does not"
        ),
    )
    parser.add_argument(
        "--latent_upscale_scale",
        type=float,
        default=1.0,
        help="how much to enlarge the finished latent with --latent_upscaler; 1.0 leaves it alone",
    )
    parser.add_argument(
        "--first_pass_lora",
        action="store_true",
        help="also apply the adapters to the first pass, which the two-pass split exists to avoid",
    )
    parser.add_argument(
        "--h3_null_guidance_scale",
        type=float,
        default=0.0,
        help=(
            "classifier-free guidance at sampling time against the BASE checkpoint's empty-prompt branch: each step "
            "runs the prompted forward with the adapters and a null-prompt forward with the adapters off, and combines "
            "them as null + scale * (prompted - null). Restores amplification an adapter trained with a plain data loss "
            "lost (its field settles near 1/w of the base's; a scale near w brings it back). Two forwards per step. "
            "Needs --lora_weight. 0 (default) samples the prompted branch only"
        ),
    )
    parser.add_argument("--compile", action="store_true", help="regionally compile H3 transformer blocks")
    parser.add_argument("--compile_backend", default="inductor")
    parser.add_argument(
        "--compile_mode",
        default="max-autotune-no-cudagraphs",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
    )
    parser.add_argument("--compile_dynamic", choices=("true", "false", "auto"))
    parser.add_argument("--compile_fullgraph", action="store_true")
    parser.add_argument("--compile_cache_size_limit", type=int)
    parser.add_argument("--compile_auto_cache_size_limit", action="store_true")
    parser.add_argument("--compile_fallback_to_eager", action="store_true")
    parser.add_argument("--inductor_config", nargs="*", default=[])
    parser.add_argument("--h3_fused_qk_norm_rope", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Validate inputs and print checkpoint tensor metadata without loading weights or running inference",
    )
    return parser


def _parse_keyframe(entry: str) -> tuple[int, str]:
    index, separator, path = entry.partition(":")
    try:
        parsed_index = int(index.strip())
    except ValueError:
        parsed_index = None
    if not separator or not path or parsed_index is None:
        raise ValueError(f"--keyframe expects INDEX:PATH with an integer index, got {entry!r}")
    return parsed_index, path


def _parse_guides(image_entries: list[str], video_entries: list[str], audio_entries: list[str]) -> tuple[H3Guide, ...]:
    by_frame: dict[int, dict[str, Path]] = {}
    for kind, entries in (("image", image_entries), ("video", video_entries), ("audio", audio_entries)):
        for entry in entries:
            frame, path = _parse_keyframe(entry)
            if kind in by_frame.setdefault(frame, {}):
                raise ValueError(f"--guide_{kind} lists pixel frame {frame} twice")
            by_frame[frame][kind] = Path(path)
    return tuple(H3Guide(frame, **media) for frame, media in by_frame.items())


def request_from_args(args: argparse.Namespace) -> H3GenerationRequest:
    first_frame = args.first_frame
    last_frame = args.last_frame
    one_frame = args.frame_count == 1
    if args.one_frame is not None and not one_frame:
        raise ValueError("--one_frame placement options require --frame_count 1")
    if one_frame and args.h3_image_mode != "none":
        raise ValueError("--frame_count 1 cannot be combined with legacy --h3_image_mode")
    if args.h3_image_mode == "first":
        if not first_frame:
            raise ValueError("--h3_image_mode first requires --first_frame")
        if last_frame and last_frame != first_frame:
            raise ValueError("--h3_image_mode first does not accept a different --last_frame")
        last_frame = first_frame
    elif args.h3_image_mode == "first_last" and (not first_frame or not last_frame):
        raise ValueError("--h3_image_mode first_last requires --first_frame and --last_frame")
    if args.condition_image and (first_frame or last_frame):
        raise ValueError("--condition_image cannot be combined with --first_frame/--last_frame")
    target_index, control_indices = parse_one_frame_options(args.one_frame)
    if args.condition_image and not one_frame:
        raise ValueError("--condition_image requires --frame_count 1")
    references = make_references(
        first_frame=first_frame,
        last_frame=last_frame,
        keyframes=[_parse_keyframe(entry) for entry in args.keyframe],
        images=args.reference_image,
        videos=args.reference_video,
        audio=args.reference_audio,
    )
    return H3GenerationRequest(
        args.prompt,
        args.output,
        args.duration,
        args.ratio,
        args.seed,
        references,
        args.frame_count if args.frame_count is not None else (args.h3_image_frame_count if args.h3_image_mode != "none" else None),
        args.h3_select_frame,
        _parse_guides(args.guide_image, args.guide_video, args.guide_audio),
        tuple(Path(path) for path in args.condition_image),
        target_index if one_frame else None,
        control_indices if one_frame else None,
    )


def parse_prompt_line(line: str) -> dict[str, object]:
    """Parse upstream-compatible per-line ``--ci`` and ``--of`` controls."""
    parts = ["", *line[2:].split(" --")] if line.startswith("--") else line.split(" --")
    result: dict[str, object] = {}
    if parts[0].strip():
        result["prompt"] = parts[0].strip().replace("\\n", "\n")
    condition_images: list[str] = []
    references: list[str] = []
    names = {
        "i": "first_frame",
        "ei": "last_frame",
        "of": "one_frame",
        "o": "output",
        "d": "seed",
        "f": "frame_count",
        "w": "width",
        "h": "height",
        "s": "steps",
        "duration": "duration",
        "ratio": "ratio",
    }
    integer_options = {"d", "f", "w", "h", "s", "duration"}
    for part in parts[1:]:
        option, separator, value = part.strip().partition(" ")
        value = value.strip()
        if not separator or not value:
            raise ValueError(f"MiniMax H3 prompt-line --{option} requires a value")
        if option == "ci":
            condition_images.append(value)
        elif option == "ref":
            references.append(value)
        elif option in names:
            result[names[option]] = int(value) if option in integer_options else value
        else:
            raise ValueError(f"MiniMax H3 prompt line has unknown option --{option}")
    if condition_images:
        result["condition_image"] = condition_images
    if references:
        result["reference_image"] = references
    return result


def apply_prompt_overrides(args: argparse.Namespace, overrides: dict[str, object], index: int) -> argparse.Namespace:
    item = copy.deepcopy(args)
    for name, value in overrides.items():
        if name == "output":
            output = Path(value)
            value = output if output.is_absolute() else args.output / output
        setattr(item, name, value)
    if "output" not in overrides:
        suffix = ".png" if item.frame_count == 1 else ".mp4"
        item.output = args.output / f"{index:04d}{suffix}"
    return item


def generator_from_args(args: argparse.Namespace, request: H3GenerationRequest):
    """Create one reusable H3 generator from validated CLI arguments."""
    return create_generator(
        model=args.model,
        text_encoder=args.text_encoder,
        tokenizer=args.tokenizer,
        video_vae=args.vae,
        audio_vae=args.audio_vae,
        device=args.device,
        dtype=args.dtype,
        request=request,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        fp8_scaled=args.fp8_base,
        int8_convrot=args.int8_convrot_base,
        text_encoder_quantization=args.text_encoder_quantization,
        text_encoder_blocks_to_stream=args.h3_text_encoder_blocks_to_stream,
        text_encoder_nvfp4_scaled_mm=args.h3_nvfp4_scaled_mm,
        blocks_to_swap=args.blocks_to_swap,
        block_swap_h2d_only=args.block_swap_h2d_only,
        block_swap_ring_size=args.block_swap_ring_size,
        block_swap_granularity=args.block_swap_granularity,
        use_pinned_memory_for_block_swap=args.use_pinned_memory_for_block_swap,
        lora_weights=tuple(args.lora_weight),
        lora_multipliers=tuple(args.lora_multiplier),
        learned_contexts=tuple(args.h3_learned_context),
        learned_context_multipliers=tuple(args.h3_learned_context_multiplier),
        learned_context_composition=args.h3_learned_context_composition,
        first_pass_scale=args.first_pass_scale,
        first_pass_steps=args.first_pass_steps,
        second_pass_strength=args.second_pass_strength,
        first_pass_lora=args.first_pass_lora,
        null_guidance_scale=args.h3_null_guidance_scale,
        latent_upscaler=args.latent_upscaler,
        latent_upscale_scale=args.latent_upscale_scale,
        compile_model=args.compile,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        compile_dynamic=args.compile_dynamic,
        compile_fullgraph=args.compile_fullgraph,
        compile_cache_size_limit=args.compile_cache_size_limit,
        compile_auto_cache_size_limit=args.compile_auto_cache_size_limit,
        compile_fallback_to_eager=args.compile_fallback_to_eager,
        inductor_config=tuple(args.inductor_config),
        fused_qk_norm_rope=args.h3_fused_qk_norm_rope,
        reference_image_short_edge=args.reference_image_short_edge,
        reference_image_size_mode=args.reference_image_size_mode,
        reference_image_max_pixels=args.reference_image_max_pixels,
        reference_video_short_edge=args.reference_video_short_edge,
        reference_video_max_pixels=args.reference_video_max_pixels,
        text_visual_max_pixels=args.h3_text_visual_max_pixels,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.fp8_base and args.int8_convrot_base:
            raise ValueError("--int8_convrot_base cannot be combined with --fp8_base")
        if args.from_file:
            if args.prompt:
                raise ValueError("--from_file cannot be combined with --prompt")
            args.output.mkdir(parents=True, exist_ok=True)
            lines = args.from_file.read_text(encoding="utf-8").splitlines()
            item_args = [
                apply_prompt_overrides(args, parse_prompt_line(line.strip()), index)
                for index, line in enumerate(lines)
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if not item_args:
                raise ValueError("--from_file contains no prompt lines")
            requests = [request_from_args(item) for item in item_args]
            for request in requests:
                request.validate(check_files=True)
                request.output.parent.mkdir(parents=True, exist_ok=True)
            requirements = [("--text_encoder", args.text_encoder), ("--vae", args.vae)]
            if any(request.frame_count_override != 1 for request in requests):
                requirements.append(("--audio_vae", args.audio_vae))
            missing = [name for name, value in requirements if value is None]
            if missing:
                raise ValueError("native H3 generation requires " + ", ".join(missing))
            for item, request in zip(item_args, requests):
                generator = generator_from_args(item, request)
                generator.generate(request)
                if not request.output.is_file():
                    raise RuntimeError(f"H3 implementation returned without creating {request.output}")
            return
        if not args.prompt:
            raise ValueError("--prompt is required unless --from_file is used")
        request = request_from_args(args)
        request.validate(check_files=True)
        image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
        if args.output.suffix.lower() in image_suffixes and args.h3_image_mode == "none" and args.frame_count != 1:
            raise ValueError("image-file output requires --frame_count 1 or --h3_image_mode first/first_last")
        if args.h3_text_visual_max_pixels < 0:
            raise ValueError("--h3_text_visual_max_pixels must be non-negative")
        inventory = inspect_checkpoint(args.model)
        if args.inspect:
            print(json.dumps({"mode": request.mode, "checkpoint": inventory.to_dict()}, indent=2))
            return
        image_output = (args.h3_image_mode != "none" or args.frame_count == 1) and args.output.suffix.lower() in image_suffixes
        required = {
            "--text_encoder": args.text_encoder,
            "--vae": args.vae,
        }
        if not image_output:
            required["--audio_vae"] = args.audio_vae
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("native H3 generation requires " + ", ".join(missing))
        request.output.parent.mkdir(parents=True, exist_ok=True)
        generator = generator_from_args(args, request)
        generator.generate(request)
        if not request.output.is_file():
            raise RuntimeError(f"H3 implementation returned without creating {request.output}")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
