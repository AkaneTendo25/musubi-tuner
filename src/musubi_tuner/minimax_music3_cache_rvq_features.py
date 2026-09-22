"""Cache acoustic features for MiniMax Music 3 RVQ encoder distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import save_file
import torch
import torchaudio

from musubi_tuner.minimax_music3.rvq_distill import load_distill_cache
from musubi_tuner.minimax_music3.vocoder import load_official_dav_autoencoder


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dav", type=Path, required=True)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--ssl_model", help="Optional Hugging Face audio SSL model")
    parser.add_argument("--ssl_revision")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_items", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.cache_dir.glob("*_mm3_rvq_distill.safetensors"))
    if args.max_items is not None:
        paths = paths[: args.max_items]
    if not paths:
        parser.error(f"No distillation caches found in {args.cache_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dav = load_official_dav_autoencoder(args.dav, device=device, dtype=torch.float32).to(device)
    sample_rate = dav.sampling_rate
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        win_length=2048,
        hop_length=round(sample_rate / 25),
        n_mels=args.n_mels,
        power=1.0,
    ).to(device)
    ssl_model = ssl_extractor = None
    ssl_sample_rate = None
    if args.ssl_model:
        from transformers import AutoFeatureExtractor, AutoModel

        ssl_extractor = AutoFeatureExtractor.from_pretrained(
            args.ssl_model, revision=args.ssl_revision, trust_remote_code=args.trust_remote_code
        )
        ssl_model = AutoModel.from_pretrained(
            args.ssl_model,
            revision=args.ssl_revision,
            trust_remote_code=args.trust_remote_code,
            dtype=torch.float32,
        ).to(device=device, dtype=torch.float32).eval()
        ssl_sample_rate = ssl_extractor.sampling_rate

    for index, path in enumerate(paths, 1):
        output = args.output_dir / path.name
        if args.skip_existing and output.exists():
            continue
        tensors, _ = load_distill_cache(path)
        features = []
        ssl_features = []
        for latents in tensors["dav_latents"]:
            waveform = dav.decode(latents.unsqueeze(0).to(device, torch.float32))[0]
            value = mel(waveform).clamp_min_(1e-5).log_()
            mean = value.mean(dim=-1, keepdim=True)
            scale = value.std(dim=-1, keepdim=True).clamp_min_(1e-4)
            features.append(((value - mean) / scale).flatten(0, 1).cpu().to(torch.float16))
            if ssl_model is not None:
                mono = waveform.mean(0)
                if sample_rate != ssl_sample_rate:
                    mono = torchaudio.functional.resample(mono, sample_rate, ssl_sample_rate)
                values = ssl_extractor(
                    mono.cpu().numpy(), sampling_rate=ssl_sample_rate, return_tensors="pt"
                ).input_values.to(device=device, dtype=next(ssl_model.parameters()).dtype)
                ssl_features.append(ssl_model(values).last_hidden_state[0].transpose(0, 1).cpu().to(torch.float16))
        output_tensors = {"mel_features": torch.stack(features).contiguous()}
        if ssl_features:
            output_tensors["ssl_features"] = torch.stack(ssl_features).contiguous()
        save_file(
            output_tensors,
            str(output),
            metadata={
                "format": "minimax_music3_rvq_features",
                "sample_rate": str(sample_rate),
                "ssl_model": args.ssl_model or "",
            },
        )
        if index % 25 == 0 or index == len(paths):
            print(f"features {index}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
