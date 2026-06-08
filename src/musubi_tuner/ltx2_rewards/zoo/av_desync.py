"""Synchformer av_desync reward — audio-video synchronization (route=sync).

Wraps the Synchformer model (github.com/v-iashin/Synchformer, vendored in
``..vendor.synchformer``) to measure how well the audio track lines up with the
video. The model predicts, over a 21-bin offset grid in [-2, 2] seconds, the most
likely temporal offset between the two streams; the reward is the mean absolute
offset ``d`` (lower = better sync) transformed to higher-is-better via
``1.0 / (1.0 + d)``.

Inputs: a decoded **video file path** plus a
**sidecar .wav** (``xxx.mp4`` -> ``xxx.wav``). The reward declares
``needs={"video_file", "audio_file"}``; if ``audio_file`` is absent it falls back
to the ``xxx.mp4 -> xxx.wav`` sidecar convention. Both streams are loaded with
torio/torchaudio:

  - video -> 224x224 @ 25 fps, RGB, normalized to [-1, 1], truncated/padded to 8 s;
  - audio -> 16 kHz mono, log-MelSpectrogram (n_fft=1024, win=400, hop=160,
    n_mels=128), AudioSet mean/std normalized.

VRAM: the Synchformer model loads only in ``setup`` and is freed in ``teardown``
(RewardStack sequencing), so it never co-resides with the DiT / other rewards.

NOTE: this reward needs BOTH a video file and its audio. The current RL rollout
generates video-only, so ``av_desync`` is only usable once AV generation is wired;
until then it is vendored + standalone-tested. Pass the checkpoint via
``--reward_args checkpoint_path=<synchformer_state_dict.pth>``.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Tuple

from ..registry import BaseReward, register_reward

logger = logging.getLogger(__name__)

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg", ".m4v")

# AudioSet log-mel normalization constants (from Synchformer).
_MEL_MEAN = -4.2677393
_MEL_STD = 4.5689974


def _smart_pad_right(x, pad_len: int, dim: int = 0):
    """Right-pad ``x`` with zeros by ``pad_len`` along ``dim`` (the only mode the
    av_desync loader uses; subset of Synchformer ``utils.smart_pad``)."""
    import torch.nn.functional as F

    if pad_len <= 0:
        return x
    if dim < 0:
        dim += x.ndim
    assert dim < x.ndim, "invalid padding dimension"
    pad_dim = [0, 0] * (x.ndim - dim - 1)
    pad_dim += [0, pad_len]
    return F.pad(x, pad_dim, mode="constant", value=0)


def _pad_or_truncate(audio, max_spec_t: int, pad_value: float = 0.0):
    """Pad/truncate the last (time) dim of a spectrogram to ``max_spec_t``
    (from Synchformer ``utils.pad_or_truncate``)."""
    import torch

    difference = max_spec_t - audio.shape[-1]
    if difference > 0:
        audio = torch.nn.functional.pad(audio, (0, difference), "constant", pad_value)
    elif difference < 0:
        audio = audio[..., :max_spec_t]
    return audio


def _find_sidecar_audio(video_path: str) -> Optional[str]:
    """``xxx.mp4`` -> ``xxx.wav`` if it exists (sidecar convention)."""
    if not isinstance(video_path, str):
        return None
    base, ext = os.path.splitext(video_path)
    if ext.lower() not in VIDEO_EXT:
        return None
    wav_path = base + ".wav"
    return wav_path if os.path.exists(wav_path) else None


@register_reward("av_desync")
class AVDesyncReward(BaseReward):
    """Synchformer audio-video desync reward (higher = better sync; route=sync)."""

    kind = "blackbox"
    route = "sync"
    needs = frozenset({"video_file", "audio_file"})

    def __init__(self) -> None:
        self._synchformer = None
        self._sync_mel = None
        self._sync_grid = None
        self._device = "cuda"
        self._max_length_s = 8.0

    # ----- model lifecycle --------------------------------------------------

    def setup(self, device, *, checkpoint_path: str = None, max_length_s=8.0, **_ignored) -> None:
        import torch
        import torchaudio

        from ..vendor.synchformer import Synchformer, make_class_grid

        if not checkpoint_path:
            raise ValueError("av_desync reward requires --reward_args checkpoint_path=<synchformer_state_dict.pth>")
        self._device = "cuda" if device is None else str(device)
        self._max_length_s = float(max_length_s)

        synchformer = Synchformer().to(self._device).eval()
        sd = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        synchformer.load_state_dict(sd)
        self._synchformer = synchformer

        sync_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            win_length=400,
            hop_length=160,
            n_fft=1024,
            n_mels=128,
            wkwargs={"device": self._device},
        )
        # MelScale's filterbank buffer is created on CPU; move it onto the device.
        mel_scale_fb = sync_mel.mel_scale.fb.to(self._device)
        sync_mel.mel_scale.register_buffer("fb", mel_scale_fb)
        self._sync_mel = sync_mel

        self._sync_grid = make_class_grid(-2, 2, 21)
        logger.info("av_desync: Synchformer loaded on %s (max_length_s=%.1f)", self._device, self._max_length_s)

    def teardown(self) -> None:
        if self._synchformer is not None:
            try:
                import torch

                del self._synchformer
                del self._sync_mel
                self._synchformer = None
                self._sync_mel = None
                self._sync_grid = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort VRAM release
                self._synchformer = None
                self._sync_mel = None

    # ----- data loading -----------------------------------------------------

    @staticmethod
    def _decode_video_frames(video_path: str, frame_rate: float, num_frames: int):
        """Decode up to ``num_frames`` of ``video_path`` resampled to ``frame_rate``
        fps as a ``(T, C, H, W)`` uint8 RGB tensor (the layout torio's
        StreamingMediaDecoder ``add_basic_video_stream(format='rgb24')`` produces,
        which is fed straight into the torchvision v2 transform).

        Primary path is torio's ``StreamingMediaDecoder`` (byte-for-byte the upstream
        loader). torchaudio>=2.10 dropped torio, so we fall back to
        ``torchvision.io.read_video`` + uniform index resampling to ``frame_rate``,
        which yields the same preprocessing input.
        """
        try:
            from torio.io import StreamingMediaDecoder

            reader = StreamingMediaDecoder(video_path)
            reader.add_basic_video_stream(
                frames_per_chunk=num_frames,
                frame_rate=frame_rate,
                format="rgb24",
            )
            reader.fill_buffer()
            return reader.pop_chunks()[0]  # (T, C, H, W) uint8
        except ImportError:
            pass

        import torch
        from torchvision.io import read_video

        # read_video -> (T, H, W, C) uint8 at the source fps
        frames, _audio, info = read_video(video_path, pts_unit="sec", output_format="THWC")
        src_fps = float(info.get("video_fps", frame_rate) or frame_rate)
        src_t = frames.shape[0]
        if src_t == 0:
            return frames.permute(0, 3, 1, 2).contiguous()
        # resample source frames to `frame_rate` by nearest-time index selection
        duration_s = src_t / src_fps
        want = min(num_frames, max(1, int(round(duration_s * frame_rate))))
        idx = torch.arange(want, dtype=torch.float64) * (src_fps / frame_rate)
        idx = idx.round().clamp(max=src_t - 1).long()
        return frames[idx].permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

    @staticmethod
    def _load_audio_waveform(audio_path: str):
        """Load ``audio_path`` as a ``(channels, samples)`` float32 tensor + sample rate.

        Primary path is ``torchaudio.load``. torchaudio>=2.10 routes
        ``load`` through torchcodec, which may be absent; we then fall back to
        ``soundfile`` and finally to the stdlib ``wave`` module (the av_desync sidecar
        is always a ``.wav``, so wave covers the real contract dependency-free).
        """
        import torch

        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform.to(torch.float32), int(sample_rate)
        except Exception:  # noqa: BLE001 - torchcodec missing / unsupported backend
            pass

        try:
            import numpy as np
            import soundfile as sf

            data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)  # (samples, ch)
            waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # (ch, samples)
            return waveform.to(torch.float32), int(sample_rate)
        except Exception:  # noqa: BLE001 - soundfile absent
            pass

        # stdlib wave fallback (PCM .wav only)
        import wave

        import numpy as np

        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        if sampwidth not in dtype_map:
            raise ValueError(f"av_desync: unsupported wav sample width {sampwidth} bytes")
        raw = np.frombuffer(frames, dtype=dtype_map[sampwidth]).astype(np.float32)
        max_val = float(np.iinfo(dtype_map[sampwidth]).max)
        raw = raw / max_val  # normalize to [-1, 1] like torchaudio
        raw = raw.reshape(-1, n_channels).T  # (ch, samples)
        return torch.from_numpy(np.ascontiguousarray(raw)), int(sample_rate)

    def _load_video_audio_as_tensors(
        self,
        video_path: str,
        audio_path: str,
        size: int = 224,
        video_fps: float = 25.0,
        audio_sr: int = 16000,
        max_length_s: float = 8.0,
    ):
        import torch
        import torchaudio
        from torchvision.transforms import v2 as transforms_v2

        expected_video_length = int(video_fps * max_length_s)
        expected_audio_length = int(audio_sr * max_length_s)

        video_transform = transforms_v2.Compose(
            [
                transforms_v2.Resize(size, interpolation=transforms_v2.InterpolationMode.BICUBIC),
                transforms_v2.CenterCrop(size),
                transforms_v2.ToImage(),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Load Video (T, H, W, 3) uint8 RGB at video_fps
        video = self._decode_video_frames(video_path, video_fps, expected_video_length)

        video = video[:expected_video_length]
        video = _smart_pad_right(video, expected_video_length - video.shape[0], dim=0)
        video = video_transform(video)  # (T, C, H, W)

        # Load Audio (channels, samples) float32
        waveform, sample_rate = self._load_audio_waveform(audio_path)
        waveform = waveform.mean(dim=0)  # mono

        if sample_rate != audio_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, audio_sr)

        audio = waveform[:expected_audio_length]
        audio = _smart_pad_right(audio, expected_audio_length - audio.shape[0], dim=0)

        # Add batch dim
        video = video.unsqueeze(0).to(self._device)  # (1, T, C, H, W)
        audio = audio.unsqueeze(0).to(self._device)  # (1, Ta)
        return video, audio

    # ----- scoring ----------------------------------------------------------

    def _single_desync_score(self, video_path: str, audio_path: str, max_length_s: float = 8.0) -> float:
        """Mean absolute audio-video offset (seconds); the desync ``d``, pre-transform."""
        import numpy as np
        import torch
        from einops import rearrange

        video, audio = self._load_video_audio_as_tensors(
            video_path=video_path,
            audio_path=audio_path,
            size=224,
            video_fps=25.0,
            audio_sr=16000,
            max_length_s=max_length_s,
        )

        # Step1: video feats
        b, t, c, h, w = video.shape
        assert b == 1 and c == 3 and h == 224 and w == 224

        v_seg, v_step = 16, 8
        nvs = (t - v_seg) // v_step + 1
        if nvs <= 0:
            return 2.0

        v_segments = [video[:, i * v_step : i * v_step + v_seg] for i in range(nvs)]
        vx = torch.stack(v_segments, dim=1)  # (1, S, T, C, H, W)
        vx = rearrange(vx, "b s t c h w -> (b s) 1 t c h w")
        vx = self._synchformer.extract_vfeats(vx)
        vx = rearrange(vx, "(b s) 1 t d -> b s t d", b=b)

        # Step2: audio feats
        _, ta = audio.shape
        a_seg, a_step = 10240, 5120
        nas = (ta - a_seg) // a_step + 1
        if nas <= 0:
            return 2.0

        a_segments = [audio[:, i * a_step : i * a_step + a_seg] for i in range(nas)]
        ax = torch.stack(a_segments, dim=1)

        ax = torch.log(self._sync_mel(ax) + 1e-6)
        ax = _pad_or_truncate(ax, 66)
        # (ax - (-4.2677393)) / (2 * 4.5689974); _MEL_MEAN == -4.2677393.
        ax = (ax - _MEL_MEAN) / (2 * _MEL_STD)
        ax = self._synchformer.extract_afeats(ax.unsqueeze(2))

        # Step3: compare
        frame_num = min(vx.shape[1], ax.shape[1])
        vx, ax = vx[:, :frame_num], ax[:, :frame_num]

        seg_size = 14
        seg_num = math.ceil(frame_num / seg_size)
        sync_scores = []

        for si in range(seg_num):
            fstart, fend = si * seg_size, min((si + 1) * seg_size, frame_num)
            vx_seg, ax_seg = vx[:, fstart:fend], ax[:, fstart:fend]
            flen = fend - fstart
            delta = seg_size - flen

            if delta > 0:
                if si == 0:
                    rep = math.ceil(delta / flen)
                    vpad = vx_seg.repeat(1, rep, *([1] * (vx_seg.dim() - 2)))[:, :delta]
                    apad = ax_seg.repeat(1, rep, *([1] * (ax_seg.dim() - 2)))[:, :delta]
                    vx_seg = torch.cat((vx_seg, vpad), dim=1)
                    ax_seg = torch.cat((ax_seg, apad), dim=1)
                else:
                    vx_seg = vx[:, -seg_size:]
                    ax_seg = ax[:, -seg_size:]

            logits = self._synchformer.compare_v_a(vx_seg, ax_seg)  # (1, 21)
            top_id = int(torch.argmax(logits, dim=-1).item())
            sync_scores.append(abs(self._sync_grid[top_id].item()))

        return float(np.mean(sync_scores)) if len(sync_scores) > 0 else 2.0

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        if self._synchformer is None:
            raise RuntimeError("av_desync reward: setup() must run before score()")
        import numpy as np
        import torch

        scores: List[float] = []
        with torch.no_grad():
            for sample in samples:
                video_path = sample.get("video_file")
                audio_path = sample.get("audio_file")
                if audio_path is None and video_path is not None:
                    audio_path = _find_sidecar_audio(video_path)
                try:
                    if not isinstance(video_path, str):
                        raise ValueError(f"av_desync: 'video_file' must be a path str, got {type(video_path)}")
                    if not isinstance(audio_path, str):
                        raise ValueError(f"av_desync: paired wav not found for video: {video_path}")
                    d = self._single_desync_score(video_path, audio_path, max_length_s=self._max_length_s)
                    # desync d smaller is better -> reward larger is better
                    s = 1.0 / (1.0 + d)
                    if np.isnan(s) or np.isinf(s):
                        s = 0.0
                except Exception as exc:  # noqa: BLE001 - failure -> 0.0
                    logger.warning("av_desync: failed for video=%s: %r", video_path, exc)
                    s = 0.0
                scores.append(float(s))
        return scores, {"reward": "av_desync"}
