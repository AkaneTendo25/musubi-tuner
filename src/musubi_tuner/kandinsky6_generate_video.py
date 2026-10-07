from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from musubi_tuner.kandinsky6.defaults import DEFAULT_NEGATIVE_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_BUNDLED_CONFIG_NAMES = frozenset({"lite", "lite-distill", "lite-pretrain", "pro", "pro-distill", "pro-pretrain"})


def resolve_config(value: str | Path) -> Path:
    """Resolve a checkpoint preset name while preserving explicit YAML paths."""
    value = Path(value)
    if str(value) not in _BUNDLED_CONFIG_NAMES:
        return value
    path = Path(__file__).resolve().parent / "kandinsky6" / "runtime" / "configs" / "checkpoints" / f"{value}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled Kandinsky6 config is missing: {path}")
    return path


def _load_pipeline_factory():
    try:
        from musubi_tuner.kandinsky6.runtime.pipeline.factory import get_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Kandinsky6 generation requires the optional Kandinsky6 dependencies; install `musubi-tuner[kandinsky6]`."
        ) from exc
    return get_pipeline


def apply_lora_weights(
    dit: torch.nn.Module,
    paths: list[str],
    multipliers: list[float] | None = None,
    *,
    merge: bool = True,
) -> None:
    """Apply Musubi Kandinsky6 LoRAs, optionally keeping them live for an INT8 base."""
    if not paths:
        return
    from musubi_tuner.networks import lora_kandinsky6

    multipliers = multipliers or []
    for index, path in enumerate(paths):
        multiplier = multipliers[index] if index < len(multipliers) else 1.0
        state = load_file(path)
        network = lora_kandinsky6.create_arch_network_from_weights(multiplier, state, unet=dit, for_inference=True)
        device = next(dit.parameters()).device
        if merge:
            network.merge_to(None, dit, state, device=device, non_blocking=True)
            logger.info("Merged LoRA %s (multiplier %.4g)", path, multiplier)
            continue

        network.apply_to(None, dit, apply_text_encoder=False, apply_unet=True)
        incompatible = network.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"LoRA state mismatch for {path}: missing={incompatible.missing_keys[:5]}, "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )
        network.to(device)
        # Register the live network on the DiT so module/block offload moves its
        # floating adapter parameters along with the frozen quantized base.
        dit.add_module(f"_inference_lora_{index}", network)
        logger.info("Attached live LoRA %s (multiplier %.4g)", path, multiplier)


def _convrot_dit_loader(
    *,
    enabled: bool,
    compute_device: str | torch.device,
    lora_paths: list[str],
    lora_multipliers: list[float] | None,
):
    """Return a pipeline DiT callback backed by the authentic ConvRot loader."""
    if not enabled:
        return None

    from musubi_tuner.kandinsky6 import load_dit_convrot_int8
    from musubi_tuner.kandinsky6.attention import SelfAttentionEngine as ModelSelfAttentionEngine
    from musubi_tuner.kandinsky6.runtime.runtime.kernels import SelfAttentionEngine

    def create_convrot_dit(cfg, device, attention_engine):
        dit = load_dit_convrot_int8(
            cfg.paths.dit,
            device=device,
            quant_device=compute_device,
        )
        # The standalone model loader shares the DiT layout with the bundled
        # runtime but has a deliberately small attention backend surface.
        # Bind the runtime engine selected by the pipeline onto every slot.
        for module in dit.modules():
            if isinstance(getattr(module, "attn", None), ModelSelfAttentionEngine):
                module.attn = SelfAttentionEngine("sdpa" if getattr(module, "force_sdpa", False) else attention_engine)
        apply_lora_weights(dit, lora_paths, lora_multipliers, merge=False)
        return dit

    return create_convrot_dit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synchronized Kandinsky6 video/audio")
    parser.add_argument(
        "--config",
        required=True,
        help="bundled preset name (lite/pro and variants) or an explicit Kandinsky6 YAML path",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", help="Optional TI2VA/I2VA first-frame image")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--time_length", type=int, help="Duration in whole seconds")
    parser.add_argument("--latent_frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--scheduler_scale", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--attention_engine")
    parser.add_argument("--cache_mode", choices=("none", "magcache", "navicache"))
    parser.add_argument("--offload", choices=("none", "module", "block"))
    parser.add_argument("--visual_cond_scheme", choices=("pretrain", "i2v", "tail_cond_first_frame"))
    parser.add_argument("--lora_weight", nargs="*", default=[])
    parser.add_argument("--lora_multiplier", type=float, nargs="*")
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help="stream-quantize the frozen DiT base to ConvRot INT8; LoRAs remain live floating adapters",
    )
    parser.add_argument("--upstream", help="deprecated compatibility option; ignored because the runtime is bundled")
    parser.add_argument("--sr_upstream", help="deprecated compatibility option; ignored because the runtime is bundled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args.config)
    get_pipeline = _load_pipeline_factory()
    if args.lora_weight or args.convrot_int8:
        from musubi_tuner.kandinsky6.runtime.pipeline.config import load_config

        config = load_config(config_path)
        if config.paths.dit_export is not None:
            raise ValueError("LoRA/ConvRot INT8 inference requires an eager DiT; paths.dit_export cannot reload weights")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dit_loader = _convrot_dit_loader(
        enabled=args.convrot_int8,
        compute_device=args.device,
        lora_paths=args.lora_weight,
        lora_multipliers=args.lora_multiplier,
    )
    pipe = get_pipeline(
        config_path,
        device=args.device,
        attention_engine=args.attention_engine,
        cache_mode=args.cache_mode,
        offload_strategy=args.offload,
        dit_loader=dit_loader,
    )
    if not args.convrot_int8:
        base_dit = pipe._base_dit if hasattr(pipe, "_base_dit") else getattr(pipe.dit, "module", pipe.dit)
        apply_lora_weights(base_dit, args.lora_weight, args.lora_multiplier)
    result = pipe(
        text=args.prompt,
        negative_text=args.negative_prompt,
        height=args.height,
        width=args.width,
        time_length=args.time_length,
        latent_frames=args.latent_frames,
        num_steps=args.steps,
        guidance_weight=args.guidance,
        scheduler_scale=args.scheduler_scale,
        seed=args.seed,
        image=args.image,
        visual_cond_scheme=args.visual_cond_scheme,
        sample_audio=args.audio,
        save_path=output,
        show_progress=True,
    )
    logger.info("Saved %d frames%s to %s", result.frames.shape[2], " with audio" if result.audio is not None else "", result.path)


if __name__ == "__main__":
    main()
