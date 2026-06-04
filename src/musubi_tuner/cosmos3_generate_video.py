import argparse
import copy
import json
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.cosmos3 import cosmos3_utils
from musubi_tuner.hv_generate_video import save_images_grid, save_videos_grid
from musubi_tuner.networks import lora_cosmos3
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos3-Nano inference")
    parser.add_argument("--dit", type=str, required=True, help="Cosmos3 model path/repo root")
    parser.add_argument("--vae", type=str, default=None, help="Cosmos3/Wan2.2 VAE path, defaults to --dit")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer path/repo, defaults to --dit")
    parser.add_argument("--sound_tokenizer", type=str, default=None, help="sound tokenizer path/repo, defaults to --dit")
    parser.add_argument("--transformer_subfolder", type=str, default="transformer")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--tokenizer_subfolder", type=str, default="text_tokenizer")
    parser.add_argument("--sound_tokenizer_subfolder", type=str, default="sound_tokenizer")

    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--prompt_json", type=str, default=None, help="upsampled JSON prompt file for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for CFG; defaults to the official Cosmos3 negative prompt JSON",
    )
    parser.add_argument(
        "--negative_prompt_json",
        type=str,
        default=None,
        help="upsampled JSON negative prompt file for generation",
    )
    parser.add_argument("--from_file", type=str, default=None, help="sample prompt file in Musubi training-sample format")
    parser.add_argument("--save_path", type=str, required=True, help="output file or directory")
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size: height width")
    parser.add_argument("--video_length", type=int, default=81, help="number of output frames")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--infer_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--flow_shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image_path", type=str, default=None, help="first-frame image condition for I2V sampling")
    parser.add_argument("--audio", action="store_true", help="generate Cosmos3 sound/audio together with video")
    parser.add_argument("--sound_latent_fps", type=float, default=25.0)
    parser.add_argument("--sound_latent_length", type=int, default=None, help="override generated sound latent length")
    parser.add_argument("--system_prompt", action="store_true", help="enable the Cosmos3 system prompt wrapper")
    parser.add_argument("--no_system_prompt", action="store_true", help="disable the Cosmos3 system prompt wrapper")
    parser.add_argument("--no_resolution_template", action="store_true", help="disable the Cosmos3 resolution template")
    parser.add_argument("--no_duration_template", action="store_true", help="disable the Cosmos3 duration template")
    parser.add_argument(
        "--negative_metadata_mode",
        type=str,
        default=cosmos3_utils.DEFAULT_NEGATIVE_METADATA_MODE,
        choices=["same", "inverse", "none"],
        help="metadata template mode for the negative prompt",
    )

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vae_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--sound_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn_mode", type=str, default="torch", choices=["torch", "flash", "flash3", "xformers", "sageattn"])
    parser.add_argument("--fp8_base", action="store_true", help="load base transformer weights as raw fp8")
    parser.add_argument("--fp8_scaled", action="store_true", help="apply Musubi scaled fp8 optimization to the transformer")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of MoT blocks to swap to CPU")
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    parser.add_argument("--offload_dit_during_sampling", action="store_true", help="offload DiT while VAE/AVAE sampling decode runs")

    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path(s) to merge")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier(s)")
    parser.add_argument(
        "--output_type",
        type=str,
        default="video",
        choices=["video", "images", "latent", "both", "latent_images"],
        help="what to save",
    )
    parser.add_argument("--vae_scale_factor_temporal", type=int, default=4)

    args = parser.parse_args()
    if args.prompt is None and args.prompt_json is None and args.from_file is None:
        raise ValueError("Specify --prompt, --prompt_json, or --from_file.")
    if args.fp8_scaled and not args.fp8_base:
        raise ValueError("--fp8_scaled requires --fp8_base, matching Musubi training behavior.")
    if args.lora_weight and args.lora_multiplier and len(args.lora_multiplier) not in {1, len(args.lora_weight)}:
        raise ValueError("--lora_multiplier must have length 1 or match --lora_weight length.")
    if args.lora_weight and args.fp8_base and not args.fp8_scaled:
        raise ValueError("Merging LoRA into raw fp8 weights is not supported; use --fp8_base --fp8_scaled or bf16/fp16.")
    return args


def build_prompt_list(args: argparse.Namespace) -> list[dict[str, Any]]:
    negative_prompt_default = args.negative_prompt
    if args.negative_prompt_json is not None:
        negative_prompt_default = json.dumps(json.loads(Path(args.negative_prompt_json).read_text(encoding="utf-8")))
    elif negative_prompt_default is None:
        negative_prompt_default = cosmos3_utils.load_default_negative_prompt()

    default_use_system_prompt = bool(args.system_prompt and not args.no_system_prompt)
    default_add_resolution_template = not args.no_resolution_template
    default_add_duration_template = not args.no_duration_template

    if args.from_file is not None:
        prompts = load_prompts(args.from_file)
    elif args.prompt_json is not None:
        prompt_text = json.dumps(json.loads(Path(args.prompt_json).read_text(encoding="utf-8")))
        prompts = [
            {
                "prompt": prompt_text,
                "negative_prompt": negative_prompt_default,
                "width": args.video_size[1],
                "height": args.video_size[0],
                "frame_count": args.video_length,
                "sample_steps": args.infer_steps,
                "guidance_scale": args.guidance_scale,
                "discrete_flow_shift": args.flow_shift,
                "seed": args.seed,
                "image_path": args.image_path,
                "fps": args.fps,
                "use_system_prompt": default_use_system_prompt,
                "add_resolution_template": default_add_resolution_template,
                "add_duration_template": default_add_duration_template,
                "negative_metadata_mode": args.negative_metadata_mode,
                "enum": 0,
            }
        ]
    else:
        prompts = [
            {
                "prompt": args.prompt,
                "negative_prompt": negative_prompt_default,
                "width": args.video_size[1],
                "height": args.video_size[0],
                "frame_count": args.video_length,
                "sample_steps": args.infer_steps,
                "guidance_scale": args.guidance_scale,
                "discrete_flow_shift": args.flow_shift,
                "seed": args.seed,
                "image_path": args.image_path,
                "fps": args.fps,
                "use_system_prompt": default_use_system_prompt,
                "add_resolution_template": default_add_resolution_template,
                "add_duration_template": default_add_duration_template,
                "negative_metadata_mode": args.negative_metadata_mode,
                "enum": 0,
            }
        ]

    for i, prompt in enumerate(prompts):
        prompt.setdefault("negative_prompt", negative_prompt_default)
        prompt.setdefault("width", args.video_size[1])
        prompt.setdefault("height", args.video_size[0])
        prompt.setdefault("frame_count", args.video_length)
        prompt.setdefault("sample_steps", args.infer_steps)
        prompt.setdefault("guidance_scale", args.guidance_scale)
        prompt.setdefault("discrete_flow_shift", args.flow_shift)
        prompt.setdefault("seed", args.seed)
        prompt.setdefault("image_path", args.image_path)
        prompt.setdefault("fps", args.fps)
        prompt.setdefault("use_system_prompt", default_use_system_prompt)
        prompt.setdefault("add_resolution_template", default_add_resolution_template)
        prompt.setdefault("add_duration_template", default_add_duration_template)
        prompt.setdefault("negative_metadata_mode", args.negative_metadata_mode)
        prompt.setdefault("enum", i)
    return prompts


def output_stem(save_path: str, prompt_index: int, output_type: str) -> tuple[Path, str]:
    path = Path(save_path)
    if output_type in {"video", "both"} and path.suffix.lower() == ".mp4" and prompt_index == 0:
        return path.parent, path.stem
    if output_type == "latent" and path.suffix.lower() == ".safetensors" and prompt_index == 0:
        return path.parent, path.stem
    if path.suffix:
        parent = path.parent
        stem = path.stem
    else:
        parent = path
        stem = "cosmos3"
    suffix = f"_{prompt_index:02d}" if prompt_index > 0 else ""
    return parent, f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}{suffix}"


def save_outputs(
    latents: torch.Tensor,
    video: Optional[torch.Tensor],
    args: argparse.Namespace,
    prompt: dict[str, Any],
    sound_latents: Optional[torch.Tensor] = None,
    audio: Optional[torch.Tensor] = None,
) -> None:
    parent, stem = output_stem(args.save_path, int(prompt.get("enum", 0)), args.output_type)
    parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "prompt": str(prompt.get("prompt", "")),
        "negative_prompt": str(prompt.get("negative_prompt", "")),
        "seed": str(prompt.get("seed", "")),
        "infer_steps": str(prompt.get("sample_steps", "")),
        "guidance_scale": str(prompt.get("guidance_scale", "")),
        "flow_shift": str(prompt.get("discrete_flow_shift", "")),
        "fps": str(prompt.get("fps", "")),
    }

    if args.output_type in {"latent", "both", "latent_images"}:
        latent_path = parent / f"{stem}.safetensors"
        tensors = {"latents": latents.detach().cpu().contiguous()}
        if sound_latents is not None:
            tensors["sound_latents"] = sound_latents.detach().cpu().contiguous()
        save_file(tensors, str(latent_path), metadata=metadata)
        logger.info(f"Saved latent to {latent_path}")

    if audio is not None:
        audio_path = parent / f"{stem}.wav"
        save_audio(audio, audio_path, sample_rate=48000)
        logger.info(f"Saved audio to {audio_path}")

    if video is None:
        return

    if args.output_type in {"video", "both"}:
        video_path = parent / f"{stem}.mp4"
        save_videos_grid(video.detach().cpu(), str(video_path), fps=int(float(prompt.get("fps", args.fps))))
        logger.info(f"Saved video to {video_path}")

    if args.output_type in {"images", "latent_images"}:
        image_paths = save_images_grid(video.detach().cpu(), str(parent), stem, n_rows=video.shape[0], create_subdir=True)
        logger.info(f"Saved {len(image_paths)} image(s) under {parent / stem}")


def save_audio(audio: torch.Tensor, path: Path, sample_rate: int) -> None:
    audio_np = audio.detach().cpu().float().clamp(-1.0, 1.0)
    if audio_np.dim() == 3:
        audio_np = audio_np[0]
    audio_np = audio_np.numpy()
    audio_np = np.transpose(audio_np, (1, 0))
    pcm = (audio_np * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def merge_lora_weights(transformer: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    if not args.lora_weight:
        return
    multipliers = args.lora_multiplier or [1.0]
    for index, lora_path in enumerate(args.lora_weight):
        multiplier = multipliers[index] if len(multipliers) > 1 else multipliers[0]
        logger.info(f"Merging LoRA {lora_path} with multiplier {multiplier}")
        weights_sd = load_file(lora_path)
        network = lora_cosmos3.create_arch_network_from_weights(multiplier, weights_sd, unet=transformer, for_inference=True)
        network.merge_to(None, transformer, weights_sd, device=device, non_blocking=device.type != "cpu")


def should_offload_dit_during_sampling(args: argparse.Namespace, device: torch.device) -> bool:
    return device.type == "cuda" and (
        bool(args.offload_dit_during_sampling) or getattr(args, "blocks_to_swap", 0) > 0
    )


def move_transformer_to_sampling_device(transformer: torch.nn.Module, device: torch.device) -> None:
    if getattr(transformer, "blocks_to_swap", 0) and hasattr(transformer, "move_to_device_except_swap_blocks"):
        transformer.move_to_device_except_swap_blocks(device)
    else:
        transformer.to(device)
    if hasattr(transformer, "prepare_block_swap_before_forward"):
        transformer.prepare_block_swap_before_forward()


def move_sound_tokenizer(sound_tokenizer, device: torch.device | str) -> None:
    if sound_tokenizer is None:
        return
    sound_tokenizer.model.to(device)
    sound_tokenizer.device = torch.device(device) if isinstance(device, str) else device


def load_models(args: argparse.Namespace, device: torch.device, prompts: list[dict[str, Any]]):
    dtype = model_utils.str_to_dtype(args.dtype)
    vae_dtype = model_utils.str_to_dtype(args.vae_dtype)
    loading_device = "cpu" if args.blocks_to_swap > 0 else device
    transformer_dtype = dtype if args.fp8_scaled else (torch.float8_e4m3fn if args.fp8_base else dtype)
    offload_dit = should_offload_dit_during_sampling(args, device)

    transformer = cosmos3_utils.load_transformer(args.dit, args.transformer_subfolder, transformer_dtype, loading_device)
    cosmos3_utils.set_attention_backend(transformer, args.attn_mode)
    merge_lora_weights(transformer, args, device)
    if args.fp8_scaled:
        keep_fp8_weights_on_cpu = args.blocks_to_swap > 0
        if keep_fp8_weights_on_cpu:
            logger.info("Cosmos3 generation: keeping FP8 weights on CPU because block swap is enabled.")
        cosmos3_utils.apply_scaled_fp8(
            transformer,
            device,
            move_to_device=device.type == "cuda" and not keep_fp8_weights_on_cpu,
        )
    transformer.eval()
    transformer.requires_grad_(False)

    if args.blocks_to_swap > 0:
        transformer.enable_block_swap(
            args.blocks_to_swap,
            device,
            supports_backward=False,
            use_pinned_memory=args.use_pinned_memory_for_block_swap,
        )
        transformer.move_to_device_except_swap_blocks(device)
        transformer.prepare_block_swap_before_forward()
        transformer.switch_block_swap_for_inference()
    else:
        transformer.to(device)

    if offload_dit:
        logger.info("Cosmos3 generation: offloading DiT between denoising and VAE/AVAE decode")
        transformer.to("cpu")
        clean_memory_on_device(device)

    tokenizer_path = args.tokenizer if args.tokenizer is not None else args.dit
    tokenizer = cosmos3_utils.load_tokenizer(tokenizer_path, args.tokenizer_subfolder)

    needs_vae = args.output_type != "latent" or any(prompt.get("image_path") is not None for prompt in prompts)
    vae = None
    if needs_vae:
        vae_source = args.vae if args.vae is not None else args.dit
        vae_device = "cpu" if offload_dit else device
        vae = cosmos3_utils.load_vae(vae_source, args.vae_subfolder, dtype=vae_dtype, device=vae_device)
        vae.requires_grad_(False)
        vae.eval()
    sound_tokenizer = None
    if args.audio:
        sound_source = args.sound_tokenizer if args.sound_tokenizer is not None else args.dit
        sound_device = "cpu" if offload_dit else device
        sound_tokenizer = cosmos3_utils.load_sound_tokenizer(
            sound_source,
            args.sound_tokenizer_subfolder,
            dtype=model_utils.str_to_dtype(args.sound_dtype),
            device=sound_device,
        )
    scheduler = cosmos3_utils.load_scheduler(args.dit, flow_shift=args.flow_shift)
    return transformer, tokenizer, vae, sound_tokenizer, scheduler, dtype


def sample_one(
    args: argparse.Namespace,
    prompt: dict[str, Any],
    transformer,
    tokenizer,
    scheduler,
    vae,
    sound_tokenizer,
    dtype: torch.dtype,
    device: torch.device,
):
    width = (int(prompt["width"]) // 16) * 16
    height = (int(prompt["height"]) // 16) * 16
    frame_count = max(1, int(prompt["frame_count"]))
    sample_steps = int(prompt["sample_steps"])
    fps = float(prompt.get("fps", args.fps))
    guidance_scale = float(prompt["guidance_scale"]) if prompt.get("guidance_scale") is not None else None
    seed = prompt.get("seed", None)
    image_path = prompt.get("image_path", None)
    negative_prompt = prompt.get("negative_prompt", "")
    use_system_prompt = bool(prompt.get("use_system_prompt", args.system_prompt and not args.no_system_prompt))
    add_resolution_template = bool(prompt.get("add_resolution_template", not args.no_resolution_template))
    add_duration_template = bool(prompt.get("add_duration_template", not args.no_duration_template))
    negative_metadata_mode = str(prompt.get("negative_metadata_mode", args.negative_metadata_mode))

    if seed is not None:
        seed = int(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(torch.seed())

    logger.info(f"Prompt: {prompt.get('prompt', '')}")
    logger.info(f"Size: {height}x{width}, frames: {frame_count}, steps: {sample_steps}, seed: {seed}")

    offload_dit = should_offload_dit_during_sampling(args, device)
    image_condition_latent = None
    if image_path is not None:
        if vae is None:
            raise ValueError("I2V sampling requires VAE. Do not use output_type=latent without VAE for --image_path.")
        if offload_dit:
            vae.to(device)
            clean_memory_on_device(device)
        image_condition_latent = cosmos3_utils.encode_image_to_condition_latent(vae, image_path, width, height, device, dtype)
        if offload_dit:
            vae.to("cpu")
            clean_memory_on_device(device)

    sound_latent_length = None
    if args.audio:
        sound_latent_length = args.sound_latent_length
        if sound_latent_length is None:
            sound_latent_length = max(1, int(np.ceil(frame_count / fps * float(args.sound_latent_fps))))

    if offload_dit:
        move_transformer_to_sampling_device(transformer, device)
        clean_memory_on_device(device)

    with torch.no_grad():
        generated = cosmos3_utils.generate_latents(
            transformer,
            tokenizer,
            scheduler,
            prompt=str(prompt.get("prompt", "")),
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            sample_steps=sample_steps,
            flow_shift=float(prompt.get("discrete_flow_shift", args.flow_shift)),
            guidance_scale=guidance_scale,
            generator=generator,
            device=device,
            dtype=dtype,
            vae_scale_factor_temporal=args.vae_scale_factor_temporal,
            image_condition_latent=image_condition_latent,
            use_system_prompt=use_system_prompt,
            add_resolution_template=add_resolution_template,
            add_duration_template=add_duration_template,
            negative_metadata_mode=negative_metadata_mode,
            sound_latent_length=sound_latent_length,
            sound_fps=float(args.sound_latent_fps),
            progress=True,
        )
    if offload_dit:
        transformer.to("cpu")
        clean_memory_on_device(device)

    if args.audio:
        latents, sound_latents = generated
    else:
        latents = generated
        sound_latents = None

    video = None
    if args.output_type != "latent":
        if vae is None:
            raise ValueError("Decoding requires VAE.")
        with torch.no_grad():
            if offload_dit:
                vae.to(device)
                clean_memory_on_device(device)
            video = cosmos3_utils.decode_latents_to_video(vae, latents.detach().to(vae.device, dtype=torch.float32)).cpu()
            if offload_dit:
                vae.to("cpu")
                clean_memory_on_device(device)
    audio = None
    if sound_latents is not None:
        if sound_tokenizer is None:
            raise ValueError("--audio requires a loaded sound tokenizer.")
        with torch.no_grad():
            if offload_dit:
                move_sound_tokenizer(sound_tokenizer, device)
                clean_memory_on_device(device)
            audio = cosmos3_utils.decode_sound_latents_to_audio(
                sound_tokenizer,
                sound_latents.detach().to(sound_tokenizer.device, dtype=sound_tokenizer.dtype),
            ).cpu()
            if offload_dit:
                move_sound_tokenizer(sound_tokenizer, "cpu")
                clean_memory_on_device(device)
    return latents, video, sound_latents, audio


def main():
    args = parse_args()
    device = get_device(args.device)
    prompts = build_prompt_list(args)
    transformer, tokenizer, vae, sound_tokenizer, scheduler, dtype = load_models(args, device, prompts)

    try:
        for prompt in prompts:
            prompt_args = copy.deepcopy(prompt)
            latents, video, sound_latents, audio = sample_one(
                args, prompt_args, transformer, tokenizer, scheduler, vae, sound_tokenizer, dtype, device
            )
            save_outputs(latents, video, args, prompt_args, sound_latents=sound_latents, audio=audio)
            clean_memory_on_device(device)
    finally:
        if vae is not None:
            vae.to("cpu")
        if sound_tokenizer is not None:
            move_sound_tokenizer(sound_tokenizer, "cpu")
        transformer.to("cpu")
        clean_memory_on_device(device)


if __name__ == "__main__":
    main()
