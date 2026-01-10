#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from types import SimpleNamespace
from typing import List, Optional

import torch
from accelerate import Accelerator
from safetensors.torch import load_file

from musubi_tuner.hv_generate_video import setup_parser_compile
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer
from musubi_tuner.networks import lora_ltx2
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LTX-2 inference script")

    parser.add_argument(
        "--ltx2_checkpoint",
        type=str,
        required=True,
        help="Path to LTX-2 checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--gemma_root",
        type=str,
        default=None,
        help="Local directory containing Gemma weights/tokenizer",
    )
    parser.add_argument(
        "--gemma_load_in_8bit",
        action="store_true",
        help="Load Gemma LLM in 8-bit (bitsandbytes). CUDA only.",
    )
    parser.add_argument(
        "--gemma_load_in_4bit",
        action="store_true",
        help="Load Gemma LLM in 4-bit (bitsandbytes). CUDA only.",
    )
    parser.add_argument(
        "--gemma_bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="bitsandbytes 4-bit quant type (nf4 or fp4)",
    )
    parser.add_argument(
        "--gemma_bnb_4bit_disable_double_quant",
        action="store_true",
        help="Disable bitsandbytes double quant for 4-bit loading.",
    )

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path(s)")
    parser.add_argument(
        "--lora_multiplier",
        type=float,
        nargs="*",
        default=None,
        help="LoRA multiplier(s), align with lora_weight order",
    )
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")

    # Prompt and output
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt for generation")
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--output_name", type=str, default="ltx2_sample", help="Base output name")

    # Sampling controls
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--width", type=int, default=768, help="Output width")
    parser.add_argument("--frame_count", type=int, default=45, help="Number of frames")
    parser.add_argument("--frame_rate", type=float, default=25.0, help="Frames per second")
    parser.add_argument("--sample_steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--cfg_scale", type=float, default=None, help="CFG scale (requires negative prompt)")
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=5.0,
        help="Discrete flow shift for FlowMatch scheduler",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation")

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--vae_dtype", type=str, default=None, help="VAE dtype (default: bfloat16)")
    parser.add_argument(
        "--ltx_mode",
        type=str,
        default="video",
        choices=["video", "av", "audio"],
        help="Inference mode (use 'av' for audio+video sampling).",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "flash2", "flash3", "torch", "xformers", "sdpa"],
        help="Attention backend",
    )
    parser.add_argument("--device", type=str, default=None, help="Force device to cpu or cuda")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of blocks to swap")
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="Use pinned memory for block swapping",
    )
    parser.add_argument("--fp8_base", action="store_true", help="Use FP8 for DiT")
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled FP8 for DiT")
    parser.add_argument(
        "--offloading",
        action="store_true",
        help="Offload DiT to CPU between prompts (saves VRAM for multi-prompt runs)",
    )

    setup_parser_compile(parser)

    args = parser.parse_args()
    if args.prompt is None and args.from_file is None:
        raise ValueError("Either --prompt or --from_file must be specified")
    if args.gemma_root is None:
        raise ValueError("--gemma_root is required for LTX-2 prompt encoding")
    return args


def _prepare_prompt_list(
    trainer: LTX2NetworkTrainer,
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> List[dict]:
    if args.from_file:
        args.sample_num_frames = args.frame_count
        args.sample_prompts = args.from_file
        prompts = trainer.process_sample_prompts(args, accelerator, args.from_file) or []
        return prompts

    prompt = args.prompt or ""
    sample = {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt or "",
        "height": args.height,
        "width": args.width,
        "frame_count": args.frame_count,
        "frame_rate": args.frame_rate,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "discrete_flow_shift": args.discrete_flow_shift,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "enum": 0,
    }
    return [sample]


def _configure_attention_flags(args: argparse.Namespace) -> None:
    args.sdpa = False
    args.flash_attn = False
    args.flash3 = False
    args.xformers = False
    attn_mode = (args.attn_mode or "torch").lower()
    if attn_mode in {"sdpa", "torch"}:
        args.sdpa = attn_mode == "sdpa"
    elif attn_mode in {"flash", "flash2"}:
        args.flash_attn = True
    elif attn_mode in {"flash3"}:
        args.flash3 = True
    elif attn_mode in {"xformers"}:
        args.xformers = True


def _merge_lora_weights(
    transformer: torch.nn.Module,
    weights: list[str],
    multipliers: Optional[list[float]],
    include_patterns: Optional[list[str]],
    exclude_patterns: Optional[list[str]],
) -> None:
    for idx, path in enumerate(weights):
        multiplier = multipliers[idx] if multipliers and len(multipliers) > idx else 1.0
        lora_sd = load_file(path)
        net = lora_ltx2.create_arch_network_from_weights(
            multiplier,
            lora_sd,
            unet=transformer,
            for_inference=True,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        net.merge_to(None, transformer, lora_sd, device=next(transformer.parameters()).device, non_blocking=True)


def main() -> None:
    args = parse_args()
    args.dit = args.ltx2_checkpoint
    args.vae = args.ltx2_checkpoint
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    _configure_attention_flags(args)

    use_cpu = args.device == "cpu"
    mixed_precision = args.mixed_precision if args.mixed_precision != "no" else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision, cpu=use_cpu)
    device = accelerator.device

    trainer = LTX2NetworkTrainer()
    trainer.blocks_to_swap = int(args.blocks_to_swap or 0)
    trainer.handle_model_specific_args(args)

    loading_device = "cpu" if trainer.blocks_to_swap > 0 else device
    transformer = trainer.load_transformer(
        accelerator=SimpleNamespace(device=device),
        args=args,
        dit_path=args.ltx2_checkpoint,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=loading_device,
        dit_weight_dtype=None,
    )

    if args.lora_weight:
        _merge_lora_weights(
            transformer,
            args.lora_weight,
            args.lora_multiplier,
            args.include_patterns,
            args.exclude_patterns,
        )
        clean_memory_on_device(device)

    if trainer.blocks_to_swap > 0:
        logger.info(
            "Enable swap %s blocks to CPU from device: %s",
            trainer.blocks_to_swap,
            device,
        )
        transformer.enable_block_swap(
            trainer.blocks_to_swap,
            device,
            supports_backward=False,
            use_pinned_memory=bool(args.use_pinned_memory_for_block_swap),
        )
        if hasattr(transformer, "move_to_device_except_swap_blocks"):
            transformer.move_to_device_except_swap_blocks(device)
        if hasattr(transformer, "switch_block_swap_for_inference"):
            transformer.switch_block_swap_for_inference()

    if args.compile:
        transformer = trainer.compile_transformer(args, transformer)

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = _prepare_prompt_list(trainer, args, accelerator)

    transformer_offloaded = bool(args.offloading) and device.type == "cuda"
    if transformer_offloaded:
        transformer.to("cpu")
        clean_memory_on_device(device)

    for idx, sample_parameter in enumerate(prompts):
        if transformer_offloaded:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(device)
            else:
                transformer.to(device)
            clean_memory_on_device(device)

        trainer.sample_image_inference(
            accelerator=accelerator,
            args=args,
            transformer=transformer,
            dit_dtype=trainer.dit_dtype or torch.float32,
            vae=None,
            save_dir=args.output_dir,
            sample_parameter=sample_parameter,
            epoch=None,
            steps=idx,
        )

        if transformer_offloaded:
            transformer.to("cpu")
            clean_memory_on_device(device)

    logger.info("LTX-2 generation complete. Outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
