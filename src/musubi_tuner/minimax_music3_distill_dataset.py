"""Generate synthetic DAV-latent and RVQ-code pairs for Music 3 distillation."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from musubi_tuner.minimax_music3.ar import AUDIO_OFFSET, SEMANTIC_VOCAB, derive_seed, generate_conditioning, load_ar
from musubi_tuner.minimax_music3.rvq_distill import distill_item_id, save_distill_cache
from musubi_tuner.minimax_music3.sampling import sample_latents
from musubi_tuner.minimax_music3.utils import load_comfy_dit
from musubi_tuner.minimax_music3.vocoder import load_official_dav_autoencoder


def _records(path: Path, max_items: int | None) -> list[dict[str, str]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            caption = str(record.get("caption", "")).strip()
            lyrics = str(record.get("lyrics", "[instrumental]")).strip()
            if not caption:
                raise ValueError(f"Missing caption at {path}:{line_number}")
            records.append({"caption": caption, "lyrics": lyrics})
            if max_items is not None and len(records) >= max_items:
                break
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def _synthetic_records(count: int, seed: int) -> list[dict[str, str]]:
    genres = (
        "drum and bass", "neurofunk", "liquid funk", "breakbeat", "Detroit techno",
        "deep house", "ambient", "trip hop", "synthwave", "industrial", "dub techno",
        "neo-soul", "jazz fusion", "Latin jazz", "funk", "reggae", "afrobeat",
        "acoustic folk", "indie rock", "post-punk", "progressive metal", "orchestral",
        "chamber music", "minimal piano", "cinematic electronica", "experimental pop",
    )
    instrumentation = (
        "analog synthesizers and punchy drums", "distorted bass and chopped breaks",
        "upright bass, brushed drums, and piano", "string quartet and concert piano",
        "electric guitar, bass, and live drums", "modular synths and granular textures",
        "acoustic guitar and hand percussion", "brass section and syncopated percussion",
        "deep sub bass and sparse drum machine", "lush pads, arpeggiators, and gated drums",
        "woodwinds, strings, and orchestral percussion", "Rhodes piano and tight funk drums",
    )
    moods = (
        "dark and tense", "warm and nostalgic", "euphoric and energetic", "restrained and intimate",
        "melancholic but hopeful", "aggressive and mechanical", "dreamlike and spacious",
        "playful and syncopated", "solemn and cinematic", "hypnotic and minimal",
    )
    structures = (
        "a gradual intro, two contrasting sections, and a resolved ending",
        "an immediate hook, a breakdown, and a larger final section",
        "a sparse opening that develops into dense counterpoint",
        "alternating rhythmic and melodic sections with clear transitions",
        "a slow build toward a single climax followed by a short outro",
    )
    combinations = list(itertools.product(genres, instrumentation, moods, structures))
    if count > len(combinations):
        raise ValueError(f"synthetic_items cannot exceed {len(combinations)}")
    random.Random(seed).shuffle(combinations)
    return [
        {
            "caption": f"Instrumental {genre}, {mood}, featuring {instruments}; {structure}.",
            "lyrics": "[instrumental]",
        }
        for genre, instruments, mood, structure in combinations[:count]
    ]


@torch.inference_mode()
def _export_codebooks(language_model, depth_decoder, path: Path, code_dim: int, seed: int) -> None:
    if path.exists():
        tensors = load_file(str(path))
        if tensors["semantic_codebook"].shape[-1] == code_dim:
            return
    device = language_model.device
    source_dim = language_model.model.embed_tokens.weight.shape[1]
    generator = torch.Generator(device=device).manual_seed(derive_seed(seed, "rvq-codebook-projection"))
    projection = torch.randn(source_dim, code_dim, device=device, dtype=torch.float32, generator=generator)
    projection.mul_(source_dim**-0.5)

    def project(source: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
        chunks = []
        for chunk in source.split(chunk_size):
            projected = F.normalize(chunk.float(), dim=-1) @ projection
            chunks.append(F.normalize(projected, dim=-1).to(torch.float16).cpu())
        return torch.cat(chunks)

    semantic_source = language_model.model.embed_tokens.weight[AUDIO_OFFSET : AUDIO_OFFSET + SEMANTIC_VOCAB]
    depth_source = depth_decoder.audio_embeddings.weight.reshape(7, 1024, source_dim)
    save_file(
        {
            "semantic_codebook": project(semantic_source),
            "depth_codebooks": torch.stack([project(codebook) for codebook in depth_source]),
        },
        str(path),
        {"format": "minimax_music3_projected_rvq_codebooks", "code_dim": str(code_dim)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompts", type=Path, help="JSONL with caption and optional lyrics fields")
    source.add_argument("--synthetic_items", type=int, help="Generate a deterministic instrumental prompt corpus")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--dav", required=True, help="Official dav.pth with encoder weights")
    parser.add_argument("--ar_model", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--code_dim", type=int, default=256)
    parser.add_argument("--max_items", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if args.duration <= 0 or args.variants < 1 or args.steps < 1:
        parser.error("duration, variants, and steps must be positive")
    if args.synthetic_items is not None and args.synthetic_items < 1:
        parser.error("synthetic_items must be positive")
    records = (
        _records(args.prompts, args.max_items)
        if args.prompts is not None
        else _synthetic_records(args.synthetic_items, args.seed)
    )
    if args.max_items is not None:
        records = records[: args.max_items]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    teacher_paths: list[tuple[Path, Path, dict[str, str], int]] = []

    language_model, depth_decoder, tokenizer = load_ar(args.ar_model, device=device, dtype=dtype)
    _export_codebooks(
        language_model,
        depth_decoder,
        args.output_dir / "rvq_codebooks.safetensors",
        args.code_dim,
        args.seed,
    )
    for record in records:
        item_id = distill_item_id(record["caption"], record["lyrics"])
        output_path = args.output_dir / f"{item_id}_mm3_rvq_distill.safetensors"
        if args.skip_existing and output_path.exists():
            continue
        teacher_path = args.output_dir / f".{item_id}_teacher.safetensors"
        ar_seed = derive_seed(args.seed, item_id, "ar")
        if args.skip_existing and teacher_path.exists():
            teacher_paths.append((teacher_path, output_path, record, ar_seed))
            continue
        conditioning, codes = generate_conditioning(
            language_model,
            depth_decoder,
            tokenizer,
            record["caption"],
            record["lyrics"],
            max(1, round(args.duration * 25)),
            seed=ar_seed,
            return_codes=True,
        )
        save_file(
            {"conditioning": conditioning.to(torch.bfloat16), "rvq_codes": codes.to(torch.int32)},
            str(teacher_path),
        )
        teacher_paths.append((teacher_path, output_path, record, ar_seed))
        if len(teacher_paths) % 10 == 0 or len(teacher_paths) == len(records):
            print(f"teacher {len(teacher_paths)}/{len(records)}", flush=True)

    del language_model, depth_decoder, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not teacher_paths:
        return
    dit = load_comfy_dit(args.dit, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    dav = load_official_dav_autoencoder(args.dav, device=device, dtype=torch.float32).to(device)
    for item_index, (teacher_path, output_path, record, ar_seed) in enumerate(teacher_paths, 1):
        teacher = load_file(str(teacher_path))
        diffusion_seeds = [derive_seed(args.seed, output_path.stem, "diffusion", str(index)) for index in range(args.variants)]
        latents = []
        for diffusion_seed in diffusion_seeds:
            generated = sample_latents(
                dit,
                teacher["conditioning"],
                num_steps=args.steps,
                generator=torch.Generator("cpu").manual_seed(diffusion_seed),
                dtype=dit.dtype,
                guidance_scale=args.guidance_scale,
            )
            waveform = dav.decode(generated.float()).clamp_(-1.0, 1.0)
            latents.append(dav.encode(waveform).cpu())
        save_distill_cache(
            output_path,
            torch.cat(latents),
            teacher["rvq_codes"],
            caption=record["caption"],
            lyrics=record["lyrics"],
            ar_seed=ar_seed,
            diffusion_seeds=diffusion_seeds,
        )
        teacher_path.unlink()
        if item_index % 10 == 0 or item_index == len(teacher_paths):
            print(f"latent {item_index}/{len(teacher_paths)}", flush=True)


if __name__ == "__main__":
    main()
