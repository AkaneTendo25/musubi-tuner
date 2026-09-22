"""Evaluate a distilled Music 3 RVQ encoder through teacher-forced AR conditioning."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from musubi_tuner.minimax_music3.ar import decode_codes_with_prior, load_ar, teacher_force_conditioning
from musubi_tuner.minimax_music3.rvq_distill import load_distill_cache
from musubi_tuner.minimax_music3.rvq_encoder import MiniMaxMusic3RVQEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--feature_dir", type=Path)
    parser.add_argument("--input_key", choices=("dav_latents", "dav_mel_features"), default="dav_latents")
    parser.add_argument("--ar_model", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--max_items", type=int, default=4)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic_prior_weight", type=float, default=0.0)
    parser.add_argument("--depth_prior_weights", type=float, nargs="+", default=[0.0])
    args = parser.parse_args()

    paths = sorted(args.cache_dir.glob("*_mm3_rvq_distill.safetensors"))
    if len(paths) < 2:
        parser.error("At least two sequence caches are required")
    random.Random(args.seed).shuffle(paths)
    holdout_count = max(1, min(len(paths) - 1, round(len(paths) * args.holdout)))
    paths = paths[:holdout_count][: args.max_items]
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    encoder = MiniMaxMusic3RVQEncoder.load(args.encoder, device).eval()
    language_model, depth_decoder, tokenizer = load_ar(args.ar_model, device=device, dtype=dtype)

    totals = {weight: torch.zeros(12, dtype=torch.float64) for weight in args.depth_prior_weights}
    for item_index, path in enumerate(paths):
        tensors, metadata = load_distill_cache(path)
        latents = tensors["dav_latents"][:1].to(device=device, dtype=dtype)
        inputs = latents
        if args.input_key == "dav_mel_features":
            if args.feature_dir is None:
                parser.error("feature_dir is required for DAV+mel evaluation")
            mel = load_file(str(args.feature_dir / path.name))["mel_features"][:1].to(device=device, dtype=dtype)
            mel = F.interpolate(mel, size=latents.shape[-1], mode="linear", align_corners=False)
            inputs = torch.cat((latents, mel), dim=1)
        target_codes = tensors["rvq_codes"].long()
        if args.max_frames is not None:
            target_codes = target_codes[: args.max_frames]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            semantic, depth = encoder(inputs, target_codes.shape[0])
        target = teacher_force_conditioning(
            language_model, depth_decoder, tokenizer, metadata["caption"], metadata["lyrics"], target_codes
        ).float()
        generator = torch.Generator().manual_seed(args.seed + item_index)
        shuffled_codes = target_codes[torch.randperm(target_codes.shape[0], generator=generator)]
        shuffled = teacher_force_conditioning(
            language_model, depth_decoder, tokenizer, metadata["caption"], metadata["lyrics"], shuffled_codes
        ).float()
        baseline = F.cosine_similarity(target, shuffled, dim=-1).mean().item()
        for weight in args.depth_prior_weights:
            predicted_codes = decode_codes_with_prior(
                language_model,
                depth_decoder,
                tokenizer,
                metadata["caption"],
                metadata["lyrics"],
                semantic[0].transpose(0, 1).cpu(),
                [logits[0].transpose(0, 1).cpu() for logits in depth],
                args.semantic_prior_weight,
                weight,
            )
            predicted = teacher_force_conditioning(
                language_model, depth_decoder, tokenizer, metadata["caption"], metadata["lyrics"], predicted_codes
            ).float()
            overall = F.cosine_similarity(target, predicted, dim=-1).mean().item()
            semantic_cosine = F.cosine_similarity(target[:, :4096], predicted[:, :4096], dim=-1).mean().item()
            depth_cosine = F.cosine_similarity(target[:, 4096:], predicted[:, 4096:], dim=-1).mean().item()
            stream_accuracy = (target_codes == predicted_codes).float().mean(0)
            totals[weight] += torch.cat(
                (
                    torch.tensor([overall, baseline, semantic_cosine, depth_cosine], dtype=torch.float64),
                    stream_accuracy.double(),
                )
            )
            print(
                f"{path.name} depth_prior={weight:g} conditioning_cosine={overall:.6f} "
                f"semantic_cosine={semantic_cosine:.6f} depth_cosine={depth_cosine:.6f}"
            )
    for weight, values in totals.items():
        values /= max(1, len(paths))
        accuracies = " ".join(f"stream_{index}_accuracy={values[4 + index]:.6f}" for index in range(8))
        print(
            f"mean depth_prior={weight:g} conditioning_cosine={values[0]:.6f} shuffled_cosine={values[1]:.6f} "
            f"semantic_cosine={values[2]:.6f} depth_cosine={values[3]:.6f} {accuracies} items={len(paths)}"
        )


if __name__ == "__main__":
    main()
