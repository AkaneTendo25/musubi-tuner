from __future__ import annotations

import torch

from .hunyuan_vae import AutoencoderKLHunyuanVideo


def build_vae(vae_path: str, device: str | torch.device = "cuda"):
    """Load HunyuanVideo VAE (K5 custom tiling impl) from a local directory."""
    vae = AutoencoderKLHunyuanVideo.from_pretrained(vae_path, torch_dtype=torch.float16).to(device).eval()
    return vae
