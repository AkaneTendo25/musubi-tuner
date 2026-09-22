"""Flow-matching sampling for MiniMax Music 3."""

from __future__ import annotations

import torch

from .model import MiniMaxMusic3DiT, latent_length

CHUNK_FRAMES = 200
CHUNK_HOP_FRAMES = 100
OVERLAP_LATENTS = 172
CROP_LEFT_LATENTS = 86
CROP_RIGHT_LATENTS = 258


@torch.no_grad()
def sample_latent_chunks(model: MiniMaxMusic3DiT, context: torch.Tensor, *, num_steps: int = 30,
                         generator: torch.Generator | None = None, dtype: torch.dtype = torch.float16,
                         guidance_scale: float = 1.7) -> list[torch.Tensor]:
    if context.ndim == 2:
        context = context.unsqueeze(0)
    device = generator.device if generator is not None else model.device
    context = context.to(device=device, dtype=dtype)
    frame_count = context.shape[1]
    starts = [0] if frame_count <= CHUNK_FRAMES else list(range(0, frame_count - CHUNK_HOP_FRAMES, CHUNK_HOP_FRAMES))
    sigmas = torch.cat((
        torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=torch.float32),
        torch.zeros(1, device=device, dtype=torch.float32),
    ))
    chunks = []
    previous_latent = previous_condition = None

    for start in starts:
        condition = model.aligned_condition(context[:, start:start + CHUNK_FRAMES])
        length = condition.shape[-1]
        overlap = 0 if previous_latent is None else min(OVERLAP_LATENTS, length, previous_latent.shape[-1])
        if overlap:
            condition[..., :overlap] = previous_condition[..., :overlap]
        latents = torch.randn((context.shape[0], 128, length), device=device, dtype=dtype, generator=generator)
        noise_prompt = latents[..., :overlap].clone() if overlap else None

        for current, following in zip(sigmas[:-1], sigmas[1:]):
            data_time = 1.0 - current
            if overlap:
                latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * data_time) * noise_prompt + data_time * previous_latent[..., :overlap]
            timestep = data_time.expand(context.shape[0])
            conditioned = model._diffusion_forward(latents, timestep, condition)
            unconditioned = model._diffusion_forward(latents, timestep, torch.zeros_like(condition))
            velocity = unconditioned + guidance_scale * (conditioned - unconditioned)
            latents = latents + velocity.to(dtype) * (current - following)

        if overlap:
            latents[..., :overlap] = previous_latent[..., :overlap]
        carry_start = max(0, length - 2 * OVERLAP_LATENTS)
        carry_end = max(carry_start, length - OVERLAP_LATENTS)
        previous_latent = latents[..., carry_start:carry_end]
        previous_condition = condition[..., carry_start:carry_end]
        chunks.append(latents)
    return chunks


@torch.no_grad()
def decode_latent_chunks(dav, chunks: list[torch.Tensor]) -> torch.Tensor:
    """Decode and stitch the overlapping latent windows used by Music 3."""
    waveforms = []
    hop_length = 512
    for index, latents in enumerate(chunks):
        waveform = dav.decode(latents.float()).float()
        left = 0 if index == 0 else CROP_LEFT_LATENTS * hop_length
        right = 0 if index == len(chunks) - 1 else CROP_RIGHT_LATENTS * hop_length
        waveforms.append(waveform[..., left: waveform.shape[-1] - right if right else None])
    return torch.cat(waveforms, dim=-1).clamp(-1.0, 1.0)


@torch.no_grad()
def sample_latents(model: MiniMaxMusic3DiT, context: torch.Tensor, *, num_steps: int = 30,
                   generator: torch.Generator | None = None, dtype: torch.dtype = torch.float16,
                   guidance_scale: float = 1.7) -> torch.Tensor:
    if context.ndim == 2:
        context = context.unsqueeze(0)
    device = model.device
    context = context.to(device=device, dtype=dtype)
    shape = (context.shape[0], 128, latent_length(context.shape[1]))
    if generator is not None and generator.device.type == "cpu":
        latents = torch.randn(shape, device="cpu", dtype=torch.float32, generator=generator).to(device=device)
    else:
        latents = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    stride = 1000.0 / num_steps
    sigmas = torch.tensor(
        [(1000 - int(index * stride)) / 1000 for index in range(num_steps)] + [0.0],
        device=device, dtype=torch.float32,
    )
    for current, following in zip(sigmas[:-1], sigmas[1:]):
        timestep = (1.0 - current).expand(context.shape[0])
        model_input = latents.to(dtype)
        conditioned = model(model_input, timestep, context)
        unconditioned = model(model_input, timestep, torch.zeros_like(context))
        velocity = unconditioned + guidance_scale * (conditioned - unconditioned)
        latents = latents + velocity.float() * (current - following)
    return latents
