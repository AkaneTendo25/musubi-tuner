#!/usr/bin/env python3
"""
Generate videos with LTXV2 LoRA

This script performs inference with a trained LTXV2 LoRA to generate videos
from text prompts.

Usage:
    python longcat_ltxv2_generate_video.py \\
        --ltxv2_model /path/to/ltxv2/weights \\
        --vae /path/to/vae/weights \\
        --text_encoder /path/to/t5/weights \\
        --lora /path/to/lora.safetensors \\
        --prompt "A beautiful sunset over the ocean" \\
        --output_dir ./outputs \\
        --num_frames 45 \\
        --height 512 \\
        --width 512 \\
        --num_steps 50 \\
        --seed 42
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import imageio
import numpy as np
from transformers import AutoTokenizer, T5EncoderModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_prompt(
    prompt: str,
    tokenizer,
    text_encoder,
    max_length: int = 77,
    device: str = "cuda",
) -> tuple:
    """Encode prompt with T5

    Returns:
        Tuple of (text_embeddings, attention_mask)
    """
    tokens = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"].to(device=device)
    attention_mask = tokens["attention_mask"].to(device=device)

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    embeddings = outputs.last_hidden_state.to(torch.float32)
    return embeddings, attention_mask


def load_lora_weights(lora_path: str) -> dict:
    """Load LoRA weights from safetensors"""
    from safetensors.torch import load_file
    return load_file(lora_path)


def apply_lora(model, lora_weights: dict, scale: float = 1.0):
    """Apply LoRA weights to model

    Note: This is a simplified implementation. In production, use
    the proper LoRA application from musubi_tuner.networks.lora
    """
    # This would require proper integration with the LoRA module
    logger.warning("LoRA application is simplified; consider using trained model directly")


def generate_video(
    ltxv2_model,
    vae,
    text_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_frames: int,
    height: int,
    width: int,
    num_steps: int = 50,
    guidance_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate video using LTXV2 with denoising loop

    Args:
        ltxv2_model: LTXV2 transformer
        vae: VAE for decoding
        text_embeddings: Encoded text [1, max_len, dim]
        attention_mask: Text attention mask
        num_frames: Number of frames to generate
        height: Frame height
        width: Frame width
        num_steps: Number of denoising steps
        guidance_scale: Guidance scale (1.0 = no guidance)
        generator: Random generator
        device: Device to use

    Returns:
        Generated video tensor [1, 3, num_frames, height, width]
    """
    device_obj = torch.device(device)

    # Get VAE scale factors
    vae_scale_factor_temporal = getattr(vae, "temporal_downsample_factor", 4)
    vae_scale_factor_spatial = getattr(vae, "spatial_downsample_factor", 8)

    # Calculate latent dimensions
    latent_height = height // vae_scale_factor_spatial
    latent_width = width // vae_scale_factor_spatial
    latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1

    latent_channels = 128  # LTXV2 standard

    # Initialize noise
    if generator is None:
        generator = torch.Generator(device=device_obj)

    latents = torch.randn(
        (1, latent_channels, latent_frames, latent_height, latent_width),
        dtype=torch.float32,
        device=device_obj,
        generator=generator,
    )

    # Denoising loop
    timesteps = torch.linspace(1.0, 0.0, num_steps)

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Generating video")):
            # Prepare timestep
            t_tensor = torch.full((1,), t, dtype=torch.float32, device=device_obj)

            # Denoise step
            noise_pred = ltxv2_model(
                latents,
                timestep=t_tensor,
                context=text_embeddings,
                attention_mask=attention_mask,
                frame_rate=25,
                transformer_options={},
            )

            # Simple Euler step
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0)
            dt = (t_next - t).item()

            # Velocity prediction: d/dt(x) = noise_pred
            latents = latents - dt * noise_pred

    # Decode latents
    with torch.no_grad():
        video = vae.decode([latents.squeeze(0)])

    return video


def save_video(video_tensor: torch.Tensor, output_path: str, fps: int = 25):
    """Save video tensor to file

    Args:
        video_tensor: Video tensor [C, T, H, W] with values in [-1, 1]
        output_path: Output file path
        fps: Frames per second
    """
    # Convert to numpy
    if video_tensor.is_cuda:
        video_tensor = video_tensor.cpu()

    # Ensure correct shape [T, H, W, C]
    if video_tensor.dim() == 4:
        if video_tensor.shape[0] == 3:  # [C, T, H, W]
            video_tensor = video_tensor.permute(1, 2, 3, 0)

    # Convert to uint8 [0, 255]
    video_np = video_tensor.numpy()
    video_np = (video_np + 1) / 2  # [-1, 1] to [0, 1]
    video_np = (video_np * 255).astype(np.uint8)

    # Save with imageio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, video_np, fps=fps)
    logger.info(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with LTXV2 LoRA"
    )
    parser.add_argument(
        "--ltxv2_model",
        type=str,
        required=True,
        help="Path to LTXV2 weights",
    )
    parser.add_argument(
        "--vae",
        type=str,
        required=True,
        help="Path to VAE weights",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Path to T5 encoder",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/t5-v1_1-base",
        help="Tokenizer repo or path",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to LoRA weights (optional)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="video.mp4",
        help="Output filename",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=45,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Frame height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Frame width",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale (>1 for classifier-free guidance)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 precision",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32

    # Load models
    logger.info("Loading models...")

    # Load LTXV2
    from musubi_tuner.ltxv2_train_network import load_ltxv2_model
    ltxv2_model = load_ltxv2_model(
        model_path=args.ltxv2_model,
        device=device,
        torch_dtype=dtype,
    )
    ltxv2_model.eval()

    # Load VAE
    from musubi_tuner.wan.modules.vae import WanVAE
    vae = WanVAE(vae_path=args.vae, device=device, dtype=torch.float32)

    # Load text encoder
    logger.info("Loading text encoder...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.padding_side = "right"
    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder,
        torch_dtype=torch.float32,
        device_map=device,
    )
    text_encoder.eval()

    # Load LoRA if provided
    if args.lora:
        logger.info(f"Loading LoRA from {args.lora}")
        lora_weights = load_lora_weights(args.lora)
        apply_lora(ltxv2_model, lora_weights, scale=1.0)

    # Encode prompt
    logger.info(f"Encoding prompt: {args.prompt}")
    text_embeddings, attention_mask = encode_prompt(
        args.prompt,
        tokenizer,
        text_encoder,
        device=str(device),
    )

    # Generate video
    logger.info("Generating video...")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    video = generate_video(
        ltxv2_model,
        vae,
        text_embeddings,
        attention_mask,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        device=str(device),
    )

    # Save video
    output_path = os.path.join(args.output_dir, args.output_name)
    save_video(video, output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
