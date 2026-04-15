import argparse
import copy
import logging
import os
from contextlib import nullcontext
from typing import Optional

import torch

from musubi_tuner.hv_generate_video import get_time_flag
from musubi_tuner.hv_train_network import load_prompts
from musubi_tuner.mova.generation import save_audio_waveform, save_video_with_audio
from musubi_tuner.mova.text_encoder import encode_hidden_states, load_text_encoder, load_tokenizer
from musubi_tuner.mova_train_network import MovaNetworkTrainer
from musubi_tuner.networks import lora_mova
from musubi_tuner.utils import model_utils
from musubi_tuner.wan_generate_video import merge_lora_weights


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InferenceAccelerator:
    def __init__(self, device: torch.device):
        self.device = device

    def autocast(self):
        return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenMOSS MOVA inference script")
    parser.add_argument("--dit", type=str, default=None, help="MOVA checkpoint root path")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="alias for --dit")
    parser.add_argument("--vae", type=str, required=True, help="video/audio VAE root path")
    parser.add_argument("--text_encoder", type=str, required=True, help="text encoder root path or HF repo")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer root path or HF repo, defaults to text_encoder")
    parser.add_argument("--visual_subfolder", type=str, default="video_dit", help="primary visual tower subfolder")
    parser.add_argument("--alternate_visual_subfolder", type=str, default=None, help="secondary visual tower subfolder")
    parser.add_argument("--audio_subfolder", type=str, default="audio_dit", help="audio tower subfolder")
    parser.add_argument("--bridge_subfolder", type=str, default="dual_tower_bridge", help="bridge subfolder")
    parser.add_argument("--video_vae_subfolder", type=str, default=None, help="optional video VAE subfolder")
    parser.add_argument("--audio_vae", type=str, default=None, help="audio VAE root path or HF repo, defaults to --vae")
    parser.add_argument("--audio_vae_subfolder", type=str, default="audio_vae", help="audio VAE subfolder")
    parser.add_argument("--audio_vae_type", type=str, default="dac", choices=["dac", "oobleck"], help="audio VAE type")
    parser.add_argument(
        "--audio_vae_model_spec",
        type=str,
        default=None,
        help="module spec for official DAC class, e.g. path/to/dac_vae.py:DAC",
    )
    parser.add_argument("--text_encoder_subfolder", type=str, default=None, help="optional text encoder subfolder")
    parser.add_argument("--tokenizer_subfolder", type=str, default=None, help="optional tokenizer subfolder")
    parser.add_argument("--text_encoder_max_length", type=int, default=512, help="maximum prompt length")
    parser.add_argument("--text_encoder_dtype", type=str, default="bfloat16", help="text encoder dtype")
    parser.add_argument("--dit_dtype", type=str, default="bfloat16", help="DiT dtype")
    parser.add_argument("--vae_dtype", type=str, default="bfloat16", help="VAE dtype")
    parser.add_argument("--trust_remote_code", action="store_true", help="allow trust_remote_code for tokenizer/text encoder")
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="negative prompt for generation")
    parser.add_argument("--from_file", type=str, default=None, help="read prompts from a file using musubi sample-prompt syntax")
    parser.add_argument("--image_path", type=str, default=None, help="optional first-frame conditioning image")
    parser.add_argument("--video_size", type=int, nargs=2, default=[544, 960], help="video size as height width")
    parser.add_argument("--video_length", type=int, default=81, help="video frame count")
    parser.add_argument("--fps", type=float, default=16.0, help="output video fps")
    parser.add_argument("--infer_steps", type=int, default=20, help="number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="guidance scale")
    parser.add_argument("--cfg_scale", type=float, default=None, help="classifier-free guidance scale override")
    parser.add_argument("--discrete_flow_shift", type=float, default=14.5, help="flow shift for MOVA sampling")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--save_path", type=str, required=True, help="output directory or .mp4 file path")
    parser.add_argument("--save_wav", action="store_true", help="also save decoded audio as .wav")
    parser.add_argument("--device", type=str, default=None, help="device to use, defaults to cuda if available")
    parser.add_argument("--condition_scale", type=float, default=None, help="override cross-modal condition scale")
    parser.add_argument("--offload_inactive_dit", action="store_true", help="offload inactive alternate visual tower to CPU")
    parser.add_argument("--alternate_split_timestep", type=float, default=0.9, help="normalized split for dual visual towers")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap to CPU")
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="use pinned memory for block swap",
    )
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for MOVA modules")
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    return parser.parse_args()


def prepare_prompt_records(args: argparse.Namespace) -> list[dict]:
    if args.prompt is None and args.from_file is None:
        raise ValueError("Either --prompt or --from_file must be specified")
    if args.prompt is not None and args.from_file is not None:
        raise ValueError("Cannot use both --prompt and --from_file")

    if args.from_file is not None:
        records = load_prompts(args.from_file)
    else:
        records = [{"prompt": args.prompt, "enum": 0}]

    for index, record in enumerate(records):
        record.setdefault("enum", index)
        if args.negative_prompt is not None and "negative_prompt" not in record:
            record["negative_prompt"] = args.negative_prompt
        if args.image_path is not None and "image_path" not in record:
            record["image_path"] = args.image_path
        record.setdefault("width", int(args.video_size[1]))
        record.setdefault("height", int(args.video_size[0]))
        record.setdefault("frame_count", int(args.video_length))
        record.setdefault("sample_steps", int(args.infer_steps))
        record.setdefault("guidance_scale", float(args.guidance_scale))
        record.setdefault("cfg_scale", args.cfg_scale)
        record.setdefault("video_fps", float(args.fps))
        if args.seed is not None and "seed" not in record:
            record["seed"] = args.seed
    return records


def encode_prompts(args: argparse.Namespace, records: list[dict], device: torch.device) -> list[dict]:
    tokenizer_path = args.tokenizer if args.tokenizer is not None else args.text_encoder
    tokenizer = load_tokenizer(
        tokenizer_path,
        subfolder=args.tokenizer_subfolder,
        trust_remote_code=args.trust_remote_code,
    )

    text_encoder_dtype = model_utils.str_to_dtype(args.text_encoder_dtype)
    text_encoder = load_text_encoder(
        args.text_encoder,
        text_encoder_dtype,
        subfolder=args.text_encoder_subfolder,
        trust_remote_code=args.trust_remote_code,
    )
    text_encoder.eval().requires_grad_(False).to(device=device, dtype=text_encoder_dtype)

    prompt_cache: dict[str, torch.Tensor] = {}

    def encode_prompt(prompt: str) -> torch.Tensor:
        if prompt not in prompt_cache:
            encoded = tokenizer(
                [prompt],
                padding=True,
                truncation=True,
                max_length=args.text_encoder_max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device=device) for k, v in encoded.items()}
            with torch.no_grad():
                hidden_states = encode_hidden_states(text_encoder, encoded)[0]
            valid_length = int(encoded["attention_mask"][0].sum().item())
            prompt_cache[prompt] = hidden_states[:valid_length].detach().cpu()
        return prompt_cache[prompt]

    encoded_records = []
    for record in records:
        record = copy.deepcopy(record)
        record["t5_embeds"] = encode_prompt(record.get("prompt", ""))
        negative_prompt = record.get("negative_prompt")
        if negative_prompt is not None:
            record["negative_t5_embeds"] = encode_prompt(negative_prompt)
        encoded_records.append(record)

    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return encoded_records


def build_output_paths(save_path: str, prompt_idx: int, seed: Optional[int], multi_prompt: bool) -> tuple[str, str]:
    time_flag = get_time_flag()
    seed_part = "seedless" if seed is None else str(seed)
    stem = f"{time_flag}_{prompt_idx:02d}_{seed_part}"

    if save_path.lower().endswith(".mp4"):
        base, _ = os.path.splitext(save_path)
        if multi_prompt:
            video_path = f"{base}_{prompt_idx:02d}.mp4"
        else:
            video_path = save_path
    else:
        os.makedirs(save_path, exist_ok=True)
        video_path = os.path.join(save_path, stem + ".mp4")

    wav_path = os.path.splitext(video_path)[0] + ".wav"
    return video_path, wav_path


def load_model_bundle(args: argparse.Namespace, trainer: MovaNetworkTrainer, device: torch.device):
    accelerator = InferenceAccelerator(device)
    requested_fp8 = bool(args.fp8_scaled)
    loading_device = "cpu" if args.blocks_to_swap > 0 else str(device)

    load_args = copy.copy(args)
    load_args.fp8_scaled = False if args.lora_weight else requested_fp8
    load_args.offload_inactive_dit = False

    dit_weight_dtype = model_utils.str_to_dtype(args.dit_dtype)
    transformer = trainer.load_transformer(
        accelerator,
        load_args,
        args.dit,
        "torch",
        False,
        loading_device,
        dit_weight_dtype,
    )
    transformer.eval().requires_grad_(False)

    if args.lora_weight:
        merge_lora_weights(
            lora_mova,
            transformer,
            args.lora_weight,
            args.lora_multiplier,
            args.include_patterns,
            args.exclude_patterns,
            device,
        )

    if requested_fp8 and args.lora_weight:
        logger.info("Optimizing MOVA modules to FP8 after LoRA merge")
        transformer.fp8_optimization(device, move_to_device=loading_device == "cpu")

    if args.blocks_to_swap > 0:
        transformer.enable_block_swap(
            args.blocks_to_swap,
            device,
            supports_backward=False,
            use_pinned_memory=args.use_pinned_memory_for_block_swap,
        )
        transformer.move_to_device_except_swap_blocks(device)
        transformer.prepare_block_swap_before_forward()
    else:
        transformer.to(device)
        if args.offload_inactive_dit and hasattr(transformer, "video_dit_2") and device.type != "cpu":
            transformer.video_dit_2.to("cpu")

    return accelerator, transformer


def main():
    args = parse_args()

    if args.pretrained_model_path is not None and args.dit is None:
        args.dit = args.pretrained_model_path
    if args.dit is None:
        raise ValueError("--dit or --pretrained_model_path is required")
    if args.offload_inactive_dit and args.blocks_to_swap:
        raise ValueError("--offload_inactive_dit cannot be combined with --blocks_to_swap")

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    args.device = device
    args.video_fps_default = args.fps
    args.network_args = []

    trainer = MovaNetworkTrainer()
    trainer.handle_model_specific_args(args)

    prompt_records = prepare_prompt_records(args)
    encoded_records = encode_prompts(args, prompt_records, device)

    accelerator, transformer = load_model_bundle(args, trainer, device)

    vae_dtype = model_utils.str_to_dtype(args.vae_dtype)
    vae = trainer.load_vae(args, vae_dtype, args.vae)
    vae.eval().requires_grad_(False)

    dit_dtype = model_utils.str_to_dtype(args.dit_dtype)
    multi_prompt = len(encoded_records) > 1

    for record in encoded_records:
        width = int(record.get("width", args.video_size[1]))
        height = int(record.get("height", args.video_size[0]))
        frame_count = int(record.get("frame_count", args.video_length))
        sample_steps = int(record.get("sample_steps", args.infer_steps))
        guidance_scale = float(record.get("guidance_scale", args.guidance_scale))
        cfg_scale = record.get("cfg_scale", args.cfg_scale)
        seed = record.get("seed", args.seed)
        flow_shift = float(record.get("discrete_flow_shift", args.discrete_flow_shift))

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        else:
            generator = torch.Generator(device=device).manual_seed(torch.seed())

        logger.info(
            f"Generating prompt {record.get('enum', 0)}: {record.get('prompt', '')[:80]} "
            f"(size={width}x{height}, frames={frame_count}, steps={sample_steps}, fps={record.get('video_fps', args.fps)})"
        )

        result = trainer.do_inference(
            accelerator,
            args,
            record,
            vae,
            dit_dtype,
            transformer,
            flow_shift,
            sample_steps,
            width,
            height,
            frame_count,
            generator,
            "negative_t5_embeds" in record,
            guidance_scale,
            cfg_scale,
        )

        prompt_idx = int(record.get("enum", 0))
        video_path, wav_path = build_output_paths(args.save_path, prompt_idx, seed, multi_prompt)
        save_video_with_audio(
            result["video"],
            video_path,
            fps=float(result.get("video_fps", args.fps)),
            audio=result.get("audio"),
            audio_sample_rate=result.get("audio_sample_rate"),
            rescale=False,
        )
        logger.info(f"Saved video: {video_path}")

        if args.save_wav and result.get("audio") is not None and result.get("audio_sample_rate") is not None:
            save_audio_waveform(result["audio"], int(result["audio_sample_rate"]), wav_path)
            logger.info(f"Saved audio: {wav_path}")


if __name__ == "__main__":
    main()
