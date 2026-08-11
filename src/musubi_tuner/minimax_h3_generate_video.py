from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from musubi_tuner.minimax_h3.assets import default_text_encoder_assets
from musubi_tuner.minimax_h3.backend import create_generator
from musubi_tuner.minimax_h3.references import REFERENCE_IMAGE_SHORT_EDGE
from musubi_tuner.minimax_h3.request import SUPPORTED_RATIOS, H3GenerationRequest, make_references
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
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--ratio", choices=SUPPORTED_RATIOS, default="16:9")
    parser.add_argument("--height", type=int, help="explicit non-native canvas height; requires --width")
    parser.add_argument("--width", type=int, help="explicit non-native canvas width; requires --height")
    parser.add_argument("--steps", type=int, default=20, help="sigma grid points, including terminal zero")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first_frame")
    parser.add_argument("--last_frame")
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
            "condition latent frame INDEX on an image, repeatable. The transformer places the conditioning row at "
            "that temporal position, which it honours for interior positions as well as for the first and the last"
        ),
    )
    parser.add_argument("--reference_image", action="append", default=[])
    parser.add_argument("--reference_video", action="append", default=[])
    parser.add_argument("--reference_audio", action="append", default=[])
    parser.add_argument(
        "--reference_image_short_edge",
        type=int,
        default=REFERENCE_IMAGE_SHORT_EDGE,
        help=("scale every reference image so its short edge reaches this many pixels; match the value the LoRA was trained with"),
    )
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
        choices=("none", "int8", "nf4", "nvfp4_awq"),
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
    if not separator or not path or not index.strip().isdigit():
        raise ValueError(f"--keyframe expects INDEX:PATH with a non-negative index, got {entry!r}")
    return int(index), path


def request_from_args(args: argparse.Namespace) -> H3GenerationRequest:
    first_frame = args.first_frame
    last_frame = args.last_frame
    if args.h3_image_mode == "first":
        if not first_frame:
            raise ValueError("--h3_image_mode first requires --first_frame")
        if last_frame and last_frame != first_frame:
            raise ValueError("--h3_image_mode first does not accept a different --last_frame")
        last_frame = first_frame
    elif args.h3_image_mode == "first_last" and (not first_frame or not last_frame):
        raise ValueError("--h3_image_mode first_last requires --first_frame and --last_frame")
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
        args.h3_image_frame_count if args.h3_image_mode != "none" else None,
        args.h3_select_frame,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.fp8_base and args.int8_convrot_base:
            raise ValueError("--int8_convrot_base cannot be combined with --fp8_base")
        request = request_from_args(args)
        request.validate(check_files=True)
        image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
        if args.output.suffix.lower() in image_suffixes and args.h3_image_mode == "none":
            raise ValueError("image-file output requires --h3_image_mode first or first_last")
        if args.h3_text_visual_max_pixels < 0:
            raise ValueError("--h3_text_visual_max_pixels must be non-negative")
        inventory = inspect_checkpoint(args.model)
        if args.inspect:
            print(json.dumps({"mode": request.mode, "checkpoint": inventory.to_dict()}, indent=2))
            return
        image_output = args.h3_image_mode != "none" and args.output.suffix.lower() in image_suffixes
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
        generator = create_generator(
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
            text_visual_max_pixels=args.h3_text_visual_max_pixels,
        )
        generator.generate(request)
        if not request.output.is_file():
            raise RuntimeError(f"H3 implementation returned without creating {request.output}")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
