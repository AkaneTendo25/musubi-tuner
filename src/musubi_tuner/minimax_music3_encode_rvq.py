"""Encode audio into distilled MiniMax Music 3 RVQ code streams."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import save_file
from safetensors import safe_open

from musubi_tuner.minimax_music3.rvq_encoder import MiniMaxMusic3RVQEncoder
from musubi_tuner.minimax_music3.stitched_rvq_encoder import StitchedRVQEncoder
from musubi_tuner.minimax_music3.vocoder import load_official_dav_autoencoder
from musubi_tuner.minimax_music3.ar import decode_codes_with_prior, load_ar


def _audio(
    path: Path, sample_rate: int, start_time: float, max_duration: float | None, segment_strategy: str
) -> tuple[torch.Tensor, float]:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T)
    if source_rate != sample_rate:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    waveform = waveform[:2]
    if segment_strategy == "loudest":
        if max_duration is None:
            raise ValueError("loudest segment selection requires max_duration")
        seconds = waveform.shape[-1] // sample_rate
        window_seconds = max(1, round(max_duration))
        if seconds > window_seconds:
            energy = waveform[:, : seconds * sample_rate].square().mean(0).reshape(seconds, sample_rate).mean(1)
            window_energy = F.avg_pool1d(energy[None, None], window_seconds, stride=1)[0, 0]
            start_time = float(window_energy.argmax().item())
    waveform = waveform[:, round(start_time * sample_rate) :]
    if max_duration is not None:
        waveform = waveform[:, : round(max_duration * sample_rate)]
    return waveform, start_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--dav", required=True, help="Official dav.pth with encoder weights")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_duration", type=float)
    parser.add_argument("--start_time", type=float, default=0.0)
    parser.add_argument("--segment_strategy", choices=("fixed", "loudest"), default="fixed")
    parser.add_argument("--ar_prior_weight", type=float, default=0.0)
    parser.add_argument("--depth_prior_weight", type=float)
    parser.add_argument("--ar_model", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--caption_dir", type=Path)
    parser.add_argument("--ssl_model")
    parser.add_argument("--ssl_revision")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start_time < 0:
        parser.error("start_time must be non-negative")

    device = torch.device(args.device)
    with safe_open(args.encoder, framework="pt", device="cpu") as stream:
        encoder_format = (stream.metadata() or {}).get("format")
    stitched_encoder = encoder_format == StitchedRVQEncoder.format
    encoder = (
        StitchedRVQEncoder.load(args.encoder, device=device)
        if stitched_encoder
        else MiniMaxMusic3RVQEncoder.load(args.encoder, device=device)
    ).eval()
    if stitched_encoder and (args.ssl_model or args.ar_prior_weight or args.depth_prior_weight):
        parser.error("stitched encoders do not support acoustic side inputs or AR priors")
    dav = load_official_dav_autoencoder(args.dav, device=device, dtype=torch.float32).to(device)
    ssl_model = ssl_extractor = None
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
    if args.ar_prior_weight < 0 or (args.depth_prior_weight is not None and args.depth_prior_weight < 0):
        parser.error("prior weights must be non-negative")
    use_prior = args.ar_prior_weight > 0 or (args.depth_prior_weight is not None and args.depth_prior_weight > 0)
    ar = load_ar(args.ar_model, device=device) if use_prior else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in args.audio:
        waveform, selected_start = _audio(
            path, dav.sampling_rate, args.start_time, args.max_duration, args.segment_strategy
        )
        waveform = waveform.unsqueeze(0).to(device)
        with torch.inference_mode():
            latents = dav.encode(waveform)
            inputs = latents
            if not stitched_encoder and encoder.config.acoustic_channels:
                acoustic_parts = []
                ssl_channels = ssl_model.config.hidden_size if ssl_model is not None else 0
                mel_channels = encoder.config.acoustic_channels - ssl_channels
                if mel_channels:
                    if mel_channels < 0 or mel_channels % waveform.shape[1]:
                        raise ValueError("Encoder acoustic channels do not match the requested SSL model")
                    n_mels = mel_channels // waveform.shape[1]
                    mel = torchaudio.transforms.MelSpectrogram(
                        sample_rate=dav.sampling_rate,
                        n_fft=2048,
                        win_length=2048,
                        hop_length=round(dav.sampling_rate / 25),
                        n_mels=n_mels,
                        power=1.0,
                    ).to(device)
                    mel_features = mel(waveform[0]).clamp_min_(1e-5).log_()
                    mel_features = (mel_features - mel_features.mean(dim=-1, keepdim=True)) / mel_features.std(
                        dim=-1, keepdim=True
                    ).clamp_min_(1e-4)
                    acoustic_parts.append(mel_features.flatten(0, 1))
                if ssl_model is not None:
                    mono = waveform[0].mean(0)
                    if dav.sampling_rate != ssl_extractor.sampling_rate:
                        mono = torchaudio.functional.resample(mono, dav.sampling_rate, ssl_extractor.sampling_rate)
                    values = ssl_extractor(
                        mono.cpu().numpy(), sampling_rate=ssl_extractor.sampling_rate, return_tensors="pt"
                    ).input_values.to(device=device, dtype=next(ssl_model.parameters()).dtype)
                    ssl_features = ssl_model(values).last_hidden_state[0].transpose(0, 1)
                    acoustic_parts.append(ssl_features)
                acoustic_length = max(features.shape[-1] for features in acoustic_parts)
                acoustic = torch.cat(
                    [
                        F.interpolate(features[None].float(), size=acoustic_length, mode="linear", align_corners=False)[0]
                        for features in acoustic_parts
                    ],
                    dim=0,
                )
                acoustic = F.interpolate(
                    acoustic.unsqueeze(0), size=latents.shape[-1], mode="linear", align_corners=False
                ).to(latents)
                inputs = torch.cat((latents, acoustic), dim=1)
            frame_count = max(1, round(waveform.shape[-1] * 25 / dav.sampling_rate))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if stitched_encoder:
                    codes, confidence = encoder.encode(inputs, frame_count)
                else:
                    semantic, depth = encoder(inputs, frame_count)
            if stitched_encoder:
                codes, confidence = codes.cpu(), confidence.cpu()
            elif ar is None:
                codes = torch.stack([semantic.argmax(1), *(logits.argmax(1) for logits in depth)], dim=-1)[0].cpu()
            else:
                sidecar_dir = args.caption_dir or path.parent
                caption_path = sidecar_dir / f"{path.stem}.txt"
                lyrics_path = sidecar_dir / f"{path.stem}.lyrics.txt"
                caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else path.stem
                lyrics = lyrics_path.read_text(encoding="utf-8").strip() if lyrics_path.exists() else "[instrumental]"
                codes = decode_codes_with_prior(
                    *ar,
                    caption,
                    lyrics,
                    semantic[0].transpose(0, 1).cpu(),
                    [logits[0].transpose(0, 1).cpu() for logits in depth],
                    args.ar_prior_weight,
                    args.depth_prior_weight,
                )
            if not stitched_encoder:
                confidence = torch.stack(
                    [semantic.float().softmax(1).amax(1), *(logits.float().softmax(1).amax(1) for logits in depth)], dim=-1
                )[0].cpu()
        output = args.output_dir / f"{path.stem}_mm3_rvq.safetensors"
        save_file(
            {"rvq_codes": codes.to(torch.int32), "confidence": confidence.to(torch.float16)},
            str(output),
            metadata={
                "source": path.name,
                "sample_rate": str(dav.sampling_rate),
                "start_time": str(selected_start),
            },
        )
        print(f"{output} start_time={selected_start:g}")


if __name__ == "__main__":
    main()
