from fractions import Fraction
import os
from typing import Optional

import av
import numpy as np
import torch
import torchaudio
from einops import rearrange
import torchvision


def normalize_audio_length(audio: torch.Tensor, sample_rate: int, duration_sec: Optional[float]) -> torch.Tensor:
    if duration_sec is None:
        return audio

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    expected_samples = max(1, int(round(float(duration_sec) * float(sample_rate))))
    if audio.shape[-1] < expected_samples:
        audio = torch.nn.functional.pad(audio, (0, expected_samples - audio.shape[-1]))
    elif audio.shape[-1] > expected_samples:
        audio = audio[..., :expected_samples]
    return audio


def save_audio_waveform(audio: torch.Tensor, sample_rate: int, path: str) -> str:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    audio = audio.detach().cpu().to(dtype=torch.float32).clamp(-1.0, 1.0)
    torchaudio.save(path, audio, sample_rate)
    return path


def _video_frames_from_tensor(videos: torch.Tensor, rescale: bool, n_rows: int) -> list[np.ndarray]:
    if videos.ndim == 4:
        videos = videos.unsqueeze(0)

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for frame in videos:
        frame = torchvision.utils.make_grid(frame, nrow=n_rows)
        frame = frame.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            frame = (frame + 1.0) / 2.0
        frame = torch.clamp(frame, 0.0, 1.0)
        outputs.append((frame.cpu().numpy() * 255).astype(np.uint8))
    return outputs


def _encode_audio_stream(container, stream, audio: torch.Tensor, sample_rate: int):
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    audio = audio.detach().cpu().to(dtype=torch.float32).clamp(-1.0, 1.0)
    if audio.shape[0] > 2:
        audio = audio[:2]

    time_base = Fraction(1, sample_rate)
    frame_samples = 1024
    for start in range(0, audio.shape[1], frame_samples):
        chunk = audio[:, start : start + frame_samples]
        if chunk.shape[1] < frame_samples:
            chunk = torch.nn.functional.pad(chunk, (0, frame_samples - chunk.shape[1]))
        frame = av.AudioFrame.from_ndarray(chunk.numpy(), format="fltp", layout=stream.layout.name)
        frame.sample_rate = sample_rate
        frame.time_base = time_base
        frame.pts = start
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)


def save_video_with_audio(
    videos: torch.Tensor,
    path: str,
    *,
    fps: float,
    audio: Optional[torch.Tensor] = None,
    audio_sample_rate: Optional[int] = None,
    rescale: bool = False,
    n_rows: int = 1,
) -> str:
    outputs = _video_frames_from_tensor(videos, rescale=rescale, n_rows=n_rows)
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    height, width, _ = outputs[0].shape
    container = av.open(path, mode="w")
    frame_rate = fps if isinstance(fps, int) else Fraction(str(fps)).limit_denominator(1000)
    video_stream = container.add_stream("libx264", rate=frame_rate)
    video_stream.width = width
    video_stream.height = height
    video_stream.pix_fmt = "yuv420p"
    video_stream.bit_rate = 4000000

    audio_stream = None
    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        layout = "mono" if audio.shape[0] == 1 else "stereo"
        audio_stream = container.add_stream("aac", rate=audio_sample_rate)
        audio_stream.layout = layout
        audio_stream.bit_rate = 192000 if audio.shape[0] == 2 else 128000

    for frame_array in outputs:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in video_stream.encode(frame):
            container.mux(packet)

    for packet in video_stream.encode():
        container.mux(packet)

    if audio is not None:
        _encode_audio_stream(container, audio_stream, audio, audio_sample_rate)

    container.close()
    return path


class MovaVAEBundle:
    def __init__(self, video_vae, audio_vae):
        self.video_vae = video_vae
        self.audio_vae = audio_vae

    @property
    def dtype(self) -> torch.dtype:
        return self.video_vae.dtype

    @property
    def device(self) -> torch.device:
        return self.video_vae.device

    def to(self, device=None, dtype=None):
        self.video_vae.to(device=device, dtype=dtype)
        if self.audio_vae is not None:
            self.audio_vae.to(device=device, dtype=dtype)
        return self

    def eval(self):
        self.video_vae.eval()
        if self.audio_vae is not None:
            self.audio_vae.eval()
        return self

    def requires_grad_(self, requires_grad: bool = False):
        self.video_vae.requires_grad_(requires_grad)
        if self.audio_vae is not None:
            self.audio_vae.requires_grad_(requires_grad)
        return self

    def decode_video(self, latents: torch.Tensor, frame_count: int) -> torch.Tensor:
        device = self.video_vae.device
        dtype = self.video_vae.dtype
        with torch.amp.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
            decoded = self.video_vae.decode([latents[0].to(device=device, dtype=dtype)])[0]
        decoded = (decoded[:, :frame_count] / 2 + 0.5).clamp(0, 1)
        return decoded.unsqueeze(0).cpu().float()

    def decode_audio(self, latents: torch.Tensor, duration_sec: Optional[float] = None) -> Optional[torch.Tensor]:
        if self.audio_vae is None:
            return None
        waveform = self.audio_vae.decode(latents).cpu().float()
        waveform = normalize_audio_length(waveform[0], self.audio_vae.sample_rate, duration_sec)
        return waveform
