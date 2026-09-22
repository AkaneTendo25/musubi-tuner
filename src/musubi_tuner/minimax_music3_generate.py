"""Generate audio directly with MiniMax Music 3 checkpoint files."""

import argparse
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file

from musubi_tuner.minimax_music3.sampling import decode_latent_chunks, sample_latent_chunks
from musubi_tuner.minimax_music3.ar import generate_conditioning, load_ar, teacher_force_conditioning
from musubi_tuner.minimax_music3.utils import load_comfy_dit
from musubi_tuner.minimax_music3.vocoder import load_comfy_dav
from musubi_tuner.networks import lora_minimax_music3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True, help="Floating-point Comfy-Org DiT checkpoint")
    parser.add_argument("--dav", required=True, help="Comfy-Org DAV checkpoint")
    parser.add_argument("--conditioning", help="Cached *_mm3_te.safetensors; omit to generate from text")
    parser.add_argument("--rvq_codes", help="Cached *_mm3_rvq.safetensors used as autoregressive conditioning")
    caption_group = parser.add_mutually_exclusive_group()
    caption_group.add_argument("--caption", help="Music description used by the autoregressive stage")
    caption_group.add_argument("--caption_file", help="UTF-8 file with the music description")
    lyrics_group = parser.add_mutually_exclusive_group()
    lyrics_group.add_argument("--lyrics", help="Lyrics with structural tags such as [verse] and [chorus]")
    lyrics_group.add_argument("--lyrics_file", help="UTF-8 lyrics file")
    parser.add_argument("--duration", type=float, default=60.0, help="Maximum duration in seconds")
    parser.add_argument("--ar_model", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--ar_lora", help="AR LoRA directory for the autoregressive language model")
    parser.add_argument("--lora")
    parser.add_argument("--lora_strength", type=float, default=1.0)
    parser.add_argument("--music3_auxiliary", help="signed-time companion checkpoint")
    parser.add_argument("--music3_flowmap", action="store_true")
    parser.add_argument("--music3_flowmap_gate", type=float, default=0.25)
    parser.add_argument("--music3_flowmap_delta_type", choices=("r", "t-r"), default="r")
    parser.add_argument("--music3_signed_time", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="music3.wav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dit_dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--convrot_int8", action="store_true")
    args = parser.parse_args()
    caption = args.caption if args.caption is not None else (
        Path(args.caption_file).read_text(encoding="utf-8").strip() if args.caption_file else None
    )

    device = torch.device(args.device)
    ar_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dit_dtype = ({"fp16": torch.float16, "bf16": torch.bfloat16}[args.dit_dtype]
                 if device.type == "cuda" else torch.float32)
    if args.conditioning and args.rvq_codes:
        parser.error("conditioning and rvq_codes are mutually exclusive")
    if args.conditioning:
        tensors = load_file(args.conditioning)
        key = next((key for key in tensors if key.startswith("varlen_music3_hidden_")), None)
        if key is None:
            raise ValueError("Conditioning cache does not contain Music 3 hidden states")
        context = tensors[key]
    elif args.rvq_codes:
        if not caption or not (args.lyrics or args.lyrics_file):
            parser.error("rvq_codes requires both --caption and --lyrics/--lyrics_file")
        lyrics = args.lyrics if args.lyrics is not None else Path(args.lyrics_file).read_text(encoding="utf-8")
        codes = load_file(args.rvq_codes)["rvq_codes"].long()
        language_model, decoder, tokenizer = load_ar(args.ar_model, device=device, dtype=ar_dtype)
        context = teacher_force_conditioning(language_model, decoder, tokenizer, caption, lyrics, codes)
        del language_model, decoder, tokenizer
        torch.cuda.empty_cache()
    else:
        if not caption or not (args.lyrics or args.lyrics_file):
            parser.error("either --conditioning or both --caption and --lyrics/--lyrics_file are required")
        lyrics = args.lyrics if args.lyrics is not None else Path(args.lyrics_file).read_text(encoding="utf-8")
        language_model, decoder, tokenizer = load_ar(
            args.ar_model, device=device, dtype=ar_dtype, lora_path=args.ar_lora
        )
        context = generate_conditioning(
            language_model, decoder, tokenizer, caption, lyrics,
            max(1, round(args.duration * 25)), seed=args.seed,
        )
        del language_model, decoder, tokenizer
        torch.cuda.empty_cache()
    model = load_comfy_dit(args.dit, device=device, dtype=dit_dtype, convrot_int8=args.convrot_int8, calc_device=device)
    weights = load_file(args.lora) if args.lora else None
    lora_uses_flowmap = bool(weights) and any("delta_timestep" in key for key in weights)
    if args.music3_flowmap or lora_uses_flowmap:
        model.enable_flowmap_time_conditioning(args.music3_flowmap_gate, args.music3_flowmap_delta_type)
    if args.music3_signed_time:
        model.enable_time_sign_conditioning()
    if args.lora:
        network = lora_minimax_music3.create_arch_network_from_weights(args.lora_strength, weights, unet=model, for_inference=True)
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        network.load_state_dict(weights, strict=True)
        network.to(device=device, dtype=dit_dtype).eval()
    auxiliary_path = Path(args.music3_auxiliary) if args.music3_auxiliary else None
    if auxiliary_path is None and args.lora:
        candidate = Path(args.lora).with_name(f"{Path(args.lora).stem}_music3_aux.safetensors")
        if candidate.exists():
            auxiliary_path = candidate
    if auxiliary_path is not None:
        auxiliary = load_file(str(auxiliary_path), device="cpu")
        if "time_sign_embed.weight" in auxiliary:
            if model.diffusion_transformer.time_sign_embed is None:
                model.enable_time_sign_conditioning()
            model.diffusion_transformer.time_sign_embed.weight.data.copy_(
                auxiliary["time_sign_embed.weight"].to(model.diffusion_transformer.time_sign_embed.weight)
            )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent_chunks = sample_latent_chunks(
        model, context, num_steps=args.steps, generator=generator, dtype=dit_dtype,
        guidance_scale=args.guidance_scale,
    )
    dav = load_comfy_dav(args.dav, device=device, dtype=torch.float32).to(device)
    audio = decode_latent_chunks(dav, latent_chunks)
    sf.write(args.output, audio[0].T.cpu().numpy(), dav.sampling_rate)
    print(args.output)


if __name__ == "__main__":
    main()
