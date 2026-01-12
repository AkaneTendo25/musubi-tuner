#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import wave

import torch

from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
from musubi_tuner.ltx_2.model.audio_vae.model_configurator import (
    AudioDecoderConfigurator,
    VocoderConfigurator,
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
)
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _save_audio_wav(path: str, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.detach().cpu().float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    if audio.shape[0] > 2:
        audio = audio[:2, :]
    audio_int16 = (audio.clamp(-1, 1) * 32767.0).to(torch.int16)
    interleaved = audio_int16.t().contiguous().numpy().tobytes()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(audio_int16.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(interleaved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode LTX-2 audio latents into a wav file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LTX-2 checkpoint (.safetensors)")
    parser.add_argument("--input", type=str, required=True, help="Path to audio latents (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Output wav path")
    parser.add_argument("--device", type=str, default="cpu", help="Device for decoding (default: cpu)")
    parser.add_argument("--dtype", type=str, default="fp32", help="Decoder dtype (default: fp32)")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = str_to_dtype(args.dtype)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        logger.warning("CPU decoding does not support %s reliably; using float32", dtype)
        dtype = torch.float32

    data = torch.load(args.input, map_location="cpu")
    audio_latents = data["latents"]

    logger.info("Loading LTX-2 audio decoder/vocoder for preview on %s", device)
    audio_decoder = SingleGPUModelBuilder(
        model_path=str(args.checkpoint),
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype)
    vocoder = SingleGPUModelBuilder(
        model_path=str(args.checkpoint),
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype)

    audio_decoder.eval()
    vocoder.eval()

    audio_latents = audio_latents.to(device=device, dtype=dtype)
    with torch.no_grad():
        decoded_audio = audio_decoder(audio_latents)
        audio_waveform = vocoder(decoded_audio).squeeze(0)

    sample_rate = int(getattr(vocoder, "output_sample_rate", 24000))
    _save_audio_wav(args.output, audio_waveform, sample_rate)
    logger.info("Saved audio preview to %s", args.output)


if __name__ == "__main__":
    main()
