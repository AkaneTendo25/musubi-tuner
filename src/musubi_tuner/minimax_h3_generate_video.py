from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from musubi_tuner.minimax_h3.backend import create_backend
from musubi_tuner.minimax_h3.load_options import H3LoadOptions, add_h3_load_arguments
from musubi_tuner.minimax_h3.request import H3GenerationRequest, SUPPORTED_RATIOS, make_references
from musubi_tuner.minimax_h3.weights import inspect_checkpoint


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMax H3 local inference entrypoint")
    parser.add_argument("--model", type=Path, required=True, help="Local H3 checkpoint directory, index, or safetensors file")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--ratio", choices=SUPPORTED_RATIOS, default="16:9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first_frame")
    parser.add_argument("--last_frame")
    parser.add_argument("--reference_image", action="append", default=[])
    parser.add_argument("--reference_video", action="append", default=[])
    parser.add_argument("--reference_audio", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    add_h3_load_arguments(parser)
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
        request.output.parent.mkdir(parents=True, exist_ok=True)
        load_options = H3LoadOptions.from_namespace(args, dtype=args.dtype)
        backend = create_backend(model=args.model, device=args.device, load_options=load_options)
        backend.generate(request)
        if not request.output.is_file():
            raise RuntimeError(f"H3 implementation returned without creating {request.output}")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
