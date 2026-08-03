from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from musubi_tuner.minimax_h3.backend import create_generator
from musubi_tuner.minimax_h3.request import SUPPORTED_RATIOS, H3GenerationRequest, make_references
from musubi_tuner.minimax_h3.weights import inspect_checkpoint


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMax H3 local inference entrypoint")
    parser.add_argument("--model", type=Path, required=True, help="Local H3 checkpoint directory, index, or safetensors file")
    parser.add_argument("--text_encoder", type=Path, help="Qwen3-VL H3 BF16 checkpoint or component directory")
    parser.add_argument("--tokenizer", type=Path, help="Official FL2VA text_encoder tokenizer/processor metadata directory")
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
    parser.add_argument("--reference_image", action="append", default=[])
    parser.add_argument("--reference_video", action="append", default=[])
    parser.add_argument("--reference_audio", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--fp8_base", action="store_true", help="use weight-only scaled FP8 transformer blocks")
    parser.add_argument(
        "--text_encoder_quantization",
        choices=("none", "int8", "nf4"),
        default="none",
        help="quantize the Qwen3-VL conditioner while encoding the prompt",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--block_swap_h2d_only", action="store_true")
    parser.add_argument("--block_swap_ring_size", type=int, default=2)
    parser.add_argument("--block_swap_granularity", choices=("block", "layer"), default="block")
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    parser.add_argument("--lora_weight", type=Path, action="append", default=[])
    parser.add_argument("--lora_multiplier", type=float, action="append", default=[])
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Validate inputs and print checkpoint tensor metadata without loading weights or running inference",
    )
    return parser


def request_from_args(args: argparse.Namespace) -> H3GenerationRequest:
    references = make_references(
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        images=args.reference_image,
        videos=args.reference_video,
        audio=args.reference_audio,
    )
    return H3GenerationRequest(args.prompt, args.output, args.duration, args.ratio, args.seed, references)


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        request = request_from_args(args)
        request.validate(check_files=True)
        inventory = inspect_checkpoint(args.model)
        if args.inspect:
            print(json.dumps({"mode": request.mode, "checkpoint": inventory.to_dict()}, indent=2))
            return
        required = {
            "--text_encoder": args.text_encoder,
            "--tokenizer": args.tokenizer,
            "--vae": args.vae,
            "--audio_vae": args.audio_vae,
        }
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
            text_encoder_quantization=args.text_encoder_quantization,
            blocks_to_swap=args.blocks_to_swap,
            block_swap_h2d_only=args.block_swap_h2d_only,
            block_swap_ring_size=args.block_swap_ring_size,
            block_swap_granularity=args.block_swap_granularity,
            use_pinned_memory_for_block_swap=args.use_pinned_memory_for_block_swap,
            lora_weights=tuple(args.lora_weight),
            lora_multipliers=tuple(args.lora_multiplier),
        )
        generator.generate(request)
        if not request.output.is_file():
            raise RuntimeError(f"H3 implementation returned without creating {request.output}")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
