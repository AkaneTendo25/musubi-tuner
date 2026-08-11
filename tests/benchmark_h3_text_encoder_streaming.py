from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from musubi_tuner.minimax_h3.integration import create_conditioning_encoder


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_encoder", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--quantization", choices=("none", "nvfp4_awq"), default="none")
    parser.add_argument("--blocks_to_stream", type=int, default=0)
    parser.add_argument("--prompt", default="A woman drinks coffee beside a sunny window.")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    encoder = create_conditioning_encoder(
        text_encoder=args.text_encoder,
        tokenizer=args.tokenizer,
        task="t2va",
        device="cuda",
        dtype="bfloat16",
        quantization=args.quantization,
        blocks_to_stream=args.blocks_to_stream,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    load_peak = torch.cuda.max_memory_allocated(device)
    resident = torch.cuda.memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = encoder.encode_prompt(args.prompt)
    torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - started
    encode_peak = torch.cuda.max_memory_allocated(device)
    encoder.close()

    print(
        json.dumps(
            {
                "quantization": args.quantization,
                "blocks_to_stream": args.blocks_to_stream,
                "load_seconds": load_seconds,
                "encode_seconds": encode_seconds,
                "resident_gib": resident / 2**30,
                "load_peak_gib": load_peak / 2**30,
                "encode_peak_gib": encode_peak / 2**30,
                "outputs": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _digest(value)}
                    for key, value in sorted(output.items())
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
