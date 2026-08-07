"""Measure H3 checkpoint load and block-swap preparation memory.

Run in separate processes for comparable peak RSS values, for example:

  /usr/bin/time -v python tests/benchmark_h3_loading.py MODEL --blocks-to-swap 48 --h2d-only
  /usr/bin/time -v python tests/benchmark_h3_loading.py MODEL --blocks-to-swap 48 --h2d-only --legacy-staging
  python tests/benchmark_h3_loading.py MODEL --blocks-to-swap 48 --h2d-only \
    --adaln-rank 16 --convrot-int8 --convrot-int8-fwd bf16
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import torch

from musubi_tuner.minimax_h3.model_loader import load_transformer
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig


def _peak_rss_gib() -> float:
    # Linux reports KiB; macOS reports bytes. Workbox validation uses Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--mode", choices=("fl2va", "ref2va", "ref2va_omni"), default="fl2va")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks-to-swap", type=int, default=48)
    parser.add_argument("--h2d-only", action="store_true")
    parser.add_argument("--granularity", choices=("block", "layer"), default="block")
    parser.add_argument("--ring-size", type=int, default=2)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--int8-convrot", action="store_true")
    parser.add_argument("--adaln-rank", type=int)
    parser.add_argument("--convrot-int8", action="store_true")
    parser.add_argument("--convrot-int8-fwd", choices=("int8", "bf16"), default="int8")
    parser.add_argument("--legacy-staging", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this benchmark requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = load_transformer(
        args.model,
        mode=args.mode,
        loading_device="cpu",
        fp8_scaled=args.fp8,
        quantization_device=device,
        int8_convrot=args.int8_convrot,
        adaln_rank=args.adaln_rank,
        convrot_int8=args.convrot_int8,
        convrot_int8_fwd=args.convrot_int8_fwd,
        target_device=device,
        blocks_to_swap=args.blocks_to_swap,
        block_swap_h2d_only=args.h2d_only,
        low_ram_load=not args.legacy_staging,
    )
    loaded = time.perf_counter()
    model.requires_grad_(False).eval()
    config = BlockSwapConfig(
        device=device,
        supports_backward=False,
        h2d_only=args.h2d_only,
        ring_size=args.ring_size,
        granularity=args.granularity,
        use_pinned_memory=False,
    )
    model.enable_block_swap(args.blocks_to_swap, config)
    model.move_to_device_except_swap_blocks(device)
    model.switch_block_swap_for_inference()
    torch.cuda.synchronize(device)
    prepared = time.perf_counter()

    print(
        {
            "low_ram_load": not args.legacy_staging,
            "load_seconds": loaded - started,
            "prepare_seconds": prepared - loaded,
            "total_seconds": prepared - started,
            "peak_rss_gib": _peak_rss_gib(),
            "cuda_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
    )


if __name__ == "__main__":
    main()
