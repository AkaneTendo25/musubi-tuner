"""Audio -> YuE2 semantic codec ids. YuE2 ships no audio tokenizer; this uses the published MERT + head tokenizer.

MERT-v2-FullSong layer-20 features at 25 Hz (24 kHz mono, 30 s chunks, bf16 autocast), rounded to fp16, per-track
instance-normalised, then a ``SemanticHead`` (the Mothersuperior tokenizer heads: 8-layer transformer
classifier over 512-frame windows, half-window stride, a quarter window trimmed at inner edges). Feature extraction
and windowing reproduce the pipeline the published heads were trained with.

Alignment: ``tokenize_frames(mono24, frames)`` returns one code per 48 kHz latent frame. It keeps the heads' training
feature grid: MERT frames of the whole record resized to ``round(S / 960)`` rows, then the first ``frames`` rows.
That resize is usually a one-frame stretch, and the heads were trained on features made this way (the minted corpus
audio is ``960*T - 32`` samples at 24 kHz). Cropping the audio to ``frames * 960`` instead would remove the stretch,
but it halves top-1 accuracy against ground-truth codes (0.07-0.14 against 0.18-0.23 on songs YuE2 generated).
The one departure: the training pipeline drops a tail shorter than one second and stretches the whole song by up to
24 frames. Here that tail is encoded from the last second of audio, so the stretch stays at one frame at most.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.machinery
import json
import logging
import math
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from musubi_tuner.yue2.yue2_protocol import CODEC_SIZE, MERT_HOP, MERT_SAMPLE_RATE

logger = logging.getLogger(__name__)

MERT_REPO = "m-a-p/MERT-v2-FullSong"
# hidden_states[20], the layer the heads were trained on (MERT2 lists block outputs only, so this is block 21 of 24)
MERT_LAYER = 20
MERT_DIM = 1024
MERT_CHUNK_SECONDS = 30
HEAD_WINDOW = 512
# bump when the feature/head algorithm changes (part of the cache fingerprint)
SEMANTIC_VERSION = "1"


# region torchaudio stand-in

# MERT2's remote code imports Spectrogram / MelScale / AmplitudeToDB from torchaudio, which is not a musubi
# dependency. When torchaudio is missing, these pure-torch equivalents (same buffers and forward as torchaudio) are
# importable while MERT loads. The checkpoint carries `spectrogram.window` and `mel_scale.fb`, so only the forward
# expressions matter.


class _Spectrogram(nn.Module):
    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn=torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ):
        super().__init__()
        if pad != 0 or normalized or not onesided:
            raise NotImplementedError("the torchaudio stand-in supports pad=0, normalized=False, onesided=True only")
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window)
        self.power = power
        self.center = center
        self.pad_mode = pad_mode

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        shape = waveform.size()
        spec = torch.stft(
            waveform.reshape(-1, shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.reshape(shape[:-1] + spec.shape[-2:])
        if self.power is None:
            return spec
        if self.power == 1.0:
            return spec.abs()
        return spec.abs().pow(self.power)


def _hz_to_mel_htk(freq: float) -> float:
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _melscale_fbanks(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int) -> torch.Tensor:
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_pts = torch.linspace(_hz_to_mel_htk(f_min), _hz_to_mel_htk(f_max), n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return torch.max(torch.zeros(1), torch.min(down_slopes, up_slopes))


class _MelScale(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ):
        super().__init__()
        if norm is not None or mel_scale != "htk":
            raise NotImplementedError("the torchaudio stand-in supports norm=None, mel_scale='htk' only")
        f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.n_mels, self.sample_rate, self.f_min, self.f_max = n_mels, sample_rate, f_min, f_max
        self.register_buffer("fb", _melscale_fbanks(n_stft, f_min, f_max, n_mels, sample_rate))

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        return torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)


class _AmplitudeToDB(nn.Module):
    def __init__(self, stype: str = "power", top_db: Optional[float] = None):
        super().__init__()
        self.stype = stype
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db = x_db - self.multiplier * self.db_multiplier
        if self.top_db is not None:
            shape = x_db.size()
            packed_channels = shape[-3] if x_db.dim() > 2 else 1
            x_db = x_db.reshape(-1, packed_channels, shape[-2], shape[-1])
            x_db = torch.max(x_db, (x_db.amax(dim=(-3, -2, -1)) - self.top_db).view(-1, 1, 1, 1))
            x_db = x_db.reshape(shape)
        return x_db


@contextlib.contextmanager
def torchaudio_transforms_available():
    """Makes ``torchaudio.transforms`` importable for the duration of the block (real torchaudio when installed,
    else the stand-in, removed again afterwards)."""
    try:
        import torchaudio.transforms  # noqa: F401

        installed = True
    except (ImportError, OSError) as e:  # OSError: a wheel built for another torch fails to load its library
        if not isinstance(e, ImportError):
            logger.warning(f"torchaudio is installed but cannot be imported ({e}); using the built-in stand-in")
        installed = False
    if installed:
        yield False
        return
    saved = {name: sys.modules.get(name) for name in ("torchaudio", "torchaudio.transforms")}
    package = types.ModuleType("torchaudio")
    package.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None, is_package=True)
    package.__path__ = []
    package.__version__ = "0.0.0+yue2-stand-in"
    transforms = types.ModuleType("torchaudio.transforms")
    transforms.__spec__ = importlib.machinery.ModuleSpec("torchaudio.transforms", None)
    transforms.Spectrogram = _Spectrogram
    transforms.MelScale = _MelScale
    transforms.AmplitudeToDB = _AmplitudeToDB
    package.transforms = transforms
    sys.modules["torchaudio"] = package
    sys.modules["torchaudio.transforms"] = transforms
    logger.info("torchaudio is not installed; using the built-in Spectrogram/MelScale/AmplitudeToDB for MERT")
    try:
        yield True
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


# endregion

# region head


@dataclass(frozen=True)
class HeadConfig:
    """Shape of a semantic head. ``window`` is the length of the learned position table."""

    in_dim: int = MERT_DIM
    width: int = 512
    depth: int = 8
    num_heads: int = 8
    vocab: int = CODEC_SIZE
    window: int = HEAD_WINDOW


class SemanticHead(nn.Module):
    """Frame classifier over a window of MERT features. Submodule names (``inp``, ``pos``, ``enc``, ``norm``, ``head``)
    are the key names of the published head files."""

    def __init__(self, config: Optional[HeadConfig] = None):
        super().__init__()
        cfg = config if config is not None else HeadConfig()
        self.config = cfg
        self.inp = nn.Linear(cfg.in_dim, cfg.width)
        self.pos = nn.Parameter(0.02 * torch.randn(1, cfg.window, cfg.width))
        layer = nn.TransformerEncoderLayer(
            cfg.width,
            cfg.num_heads,
            dim_feedforward=cfg.width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # pre-norm layers never take the nested-tensor path; turning it off avoids the warning
        self.enc = nn.TransformerEncoder(layer, cfg.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab)

    @property
    def window(self) -> int:
        return int(self.pos.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        hidden = self.enc(self.inp(x) + self.pos[:, :length])
        return self.head(self.norm(hidden))


def _head_cfg_from_metadata(meta: dict[str, str]) -> dict:
    cfg = {}
    if meta.get("cfg"):
        try:
            cfg.update(json.loads(meta["cfg"]))
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"cannot parse the head cfg metadata: {meta['cfg']!r}")
    m = re.search(r"heads=(\d+)", meta.get("architecture", ""))
    if m and "heads" not in cfg and "H" not in cfg:
        cfg["heads"] = int(m.group(1))
    if "instnorm" not in cfg and "input" in meta:
        cfg["instnorm"] = "instnorm=false" not in meta["input"]
    return cfg


def _config_from_state_dict(sd: dict[str, torch.Tensor], num_heads: int) -> HeadConfig:
    layer_ids = {int(k.split(".")[2]) for k in sd if k.startswith("enc.layers.")}
    width, in_dim = sd["inp.weight"].shape
    return HeadConfig(
        in_dim=int(in_dim),
        width=int(width),
        depth=len(layer_ids),
        num_heads=num_heads,
        vocab=int(sd["head.weight"].shape[0]),
        window=int(sd["pos"].shape[1]),
    )


def load_head(path: Union[str, Path]) -> tuple[SemanticHead, dict]:
    """A semantic head from a ``.pt`` ({"model", "cfg"}) or a safetensors release (plain state dict + metadata).

    Dimensions come from the tensors; the head count from cfg (``heads``/``H``) or the metadata, default 8.
    ``cfg["instnorm"]`` defaults to True (every published head is trained on instance-normalised features)."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        from safetensors.torch import load_file

        sd = load_file(path)
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        cfg = _head_cfg_from_metadata(meta)
    else:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ck, dict) or "model" not in ck:
            raise ValueError(f"{path}: expected a dict with 'model' (and optional 'cfg')")
        sd = ck["model"]
        cfg = dict(ck.get("cfg") or {})

    missing = [k for k in ("inp.weight", "pos", "head.weight") if k not in sd]
    if missing:
        raise ValueError(f"{path} is not a semantic tokenizer head (missing {missing})")
    head = SemanticHead(_config_from_state_dict(sd, int(cfg.get("heads", cfg.get("H", 8)))))
    head.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    if head.head.out_features != CODEC_SIZE:
        raise ValueError(f"{path}: head vocabulary {head.head.out_features} != YuE2 codec size {CODEC_SIZE}")
    cfg.setdefault("instnorm", True)
    cfg["instnorm"] = bool(cfg["instnorm"])
    return head.eval().requires_grad_(False), cfg


def instance_norm(features: torch.Tensor) -> torch.Tensor:
    """Per-channel standardisation over time: ``(x - mean) / (unbiased std + 1e-5)``, float32."""
    x = features.float()
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-5)


class HeadPass(NamedTuple):
    """One head forward: reads rows ``[start, start + window)``, writes codes for frames ``[first, stop)``."""

    start: int
    first: int
    stop: int


def window_plan(total: int, window: int) -> list[HeadPass]:
    """Head passes over ``total`` frames, in order; a later pass overwrites the frames it shares with an earlier one.

    Windows advance by half their length while they fit; if the last one stops short of the end, one more window is
    placed flush with the end. Each pass gives up a quarter window at every edge that is not an edge of the sequence."""
    if total <= 0:
        return []
    hop, margin = max(1, window // 2), window // 4
    if total <= window:
        starts = [0]
    else:
        starts = [k * hop for k in range((total - window) // hop + 1)]
        if starts[-1] + window < total:
            starts.append(total - window)
    passes = []
    for start in starts:
        at_end = start + window >= total
        first = start + margin if start > 0 else start
        stop = total if at_end else start + window - margin
        passes.append(HeadPass(start, first, stop))
    return passes


@torch.no_grad()
def classify_features(
    head: SemanticHead,
    features: torch.Tensor,
    instnorm: bool = True,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    """``features [T, din]`` -> codes ``[T]`` (int64 on the head's device). Every pass is its own batch-1 forward over a
    full window, zero-padded past the end of the input."""
    x = instance_norm(features) if instnorm else features.float()
    device = head.head.weight.device
    window = head.window
    total, channels = x.shape
    enabled = autocast_dtype is not None and device.type == "cuda"
    amp_dtype = autocast_dtype if autocast_dtype is not None else torch.bfloat16
    codes = torch.zeros(total, dtype=torch.long, device=device)
    for p in window_plan(total, window):
        chunk = x[p.start : p.start + window]
        valid = chunk.shape[0]
        if valid < window:
            chunk = torch.cat([chunk, chunk.new_zeros(window - valid, channels)], dim=0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
            logits = head(chunk.to(device).unsqueeze(0))
        best = logits[0, :valid].float().argmax(dim=-1)
        codes[p.first : p.stop] = best[p.first - p.start : p.stop - p.start]
    return codes


# name of the rotary position class in MERT2's remote modeling module
_MERT_ROTARY_CLASS = "RotaryEmbedding"


def _instance_state(module: nn.Module) -> dict:
    """Attributes a module's own ``__init__`` set on top of what ``nn.Module.__init__`` sets (buffers excluded)."""
    base = vars(nn.Module())
    return {name: value for name, value in vars(module).items() if name not in base}


def _reinit_mert_rotary(model: nn.Module, rotary_cls: Optional[type] = None, config=None) -> int:
    """Gives every rotary module of a loaded MERT2 the state its constructor produces.

    MERT2 derives the rotary ``inv_freq`` in ``__init__`` and keeps it as a non-persistent buffer, so the checkpoint
    cannot restore it, and a meta-device load (transformers 5) leaves it holding garbage. Rather than re-deriving the
    formula here, a fresh instance of MERT2's own rotary class is built from ``config`` (default ``model.config``);
    MERT2's constructor computes it on CPU in float32. Its buffers are copied into each loaded instance on that
    instance's device, and every attribute the constructor sets (the lazily built cos/sin tables and their
    length/device keys included) is reset to the fresh value.

    ``rotary_cls`` defaults to the ``RotaryEmbedding`` class of the module that defines ``type(model)``; instances
    are matched by ``isinstance``. Returns the number of rotary modules reinitialised (0 when the model's module
    defines no such class)."""
    if rotary_cls is None:
        defining = sys.modules.get(type(model).__module__)
        rotary_cls = getattr(defining, _MERT_ROTARY_CLASS, None)
        if not isinstance(rotary_cls, type):
            logger.warning(f"{type(model).__qualname__}: no {_MERT_ROTARY_CLASS} class in {type(model).__module__}")
            return 0
    targets = [module for module in model.modules() if isinstance(module, rotary_cls)]
    if not targets:
        return 0
    fresh = rotary_cls(model.config if config is None else config)
    fresh_buffers = dict(fresh.named_buffers(recurse=False))
    fresh_state = _instance_state(fresh)
    for module in targets:
        for name, value in fresh_buffers.items():
            old = getattr(module, name)
            device = old.device if isinstance(old, torch.Tensor) else value.device
            setattr(module, name, value.to(device=device, copy=True))
        for name, value in fresh_state.items():
            setattr(module, name, value)
    return len(targets)


def mert_identity(mert_model: str, mert_revision: Optional[str] = None) -> str:
    """``{repo or directory name}@{revision}`` for cache fingerprints. A local directory reports the commit recorded by
    ``huggingface_hub`` (``.cache/huggingface/download/config.json.metadata``) when present, so moving it does not
    invalidate caches."""
    p = Path(mert_model)
    if p.is_dir():
        name = p.name
        revision = mert_revision
        meta = p / ".cache" / "huggingface" / "download" / "config.json.metadata"
        if revision is None and meta.is_file():
            first = meta.read_text(encoding="utf-8").splitlines()[:1]
            revision = first[0].strip() if first else None
        return f"{name}@{revision or 'local'}"
    return f"{mert_model}@{mert_revision or 'main'}"


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def semantic_fingerprint(
    head_path: Union[str, Path], mert_model: str = MERT_REPO, mert_revision: Optional[str] = None
) -> dict[str, str]:
    """Cache metadata identifying the codes a head + MERT produce (computable without loading either)."""
    return {
        "yue2_semantic_head": file_sha256(head_path),
        "yue2_semantic_mert": mert_identity(mert_model, mert_revision),
        "yue2_semantic_version": SEMANTIC_VERSION,
    }


def _resample_time(timeline: torch.Tensor, rows: int) -> torch.Tensor:
    """``[N, C]`` -> ``[rows, C]`` by per-channel linear interpolation over time (half-pixel centres), returned as the
    transpose of the ``[C, rows]`` interpolation output. Always interpolates, also when ``rows == N``."""
    channels_first = timeline.t().unsqueeze(0)
    return F.interpolate(channels_first, size=rows, mode="linear", align_corners=False)[0].t()


def split_mert_chunks(mono: torch.Tensor, sample_rate: int = MERT_SAMPLE_RATE, chunk_seconds: int = MERT_CHUNK_SECONDS):
    """Plans MERT calls for ``mono [S]``: ``(groups, keep_tail_frames)``.

    Full chunks form one batch. A tail of at least one second is its own batch; a shorter tail is replaced by the last
    second of the input and only its final ``keep_tail_frames`` frames are kept."""
    total = mono.shape[0]
    if total < sample_rate:
        raise ValueError(f"MERT needs at least one second of audio, got {total / sample_rate:.2f}s")
    chunk = sample_rate * chunk_seconds
    n_full = total // chunk
    tail = total - n_full * chunk

    groups = []
    if n_full > 0:
        groups.append([mono[i * chunk : (i + 1) * chunk] for i in range(n_full)])
    keep_tail_frames = None
    if tail >= sample_rate:
        groups.append([mono[n_full * chunk :]])
    elif tail > 0:
        groups.append([mono[total - sample_rate :]])
        keep_tail_frames = tail // MERT_HOP
    return groups, keep_tail_frames


class SemanticTokenizer:
    """MERT-v2-FullSong + semantic head. ``tokenize_frames(mono24, frames)`` -> ``LongTensor[frames]``.

    ``mert`` / ``processor`` may be passed in (tests); otherwise they are loaded with ``trust_remote_code``.
    ``dtype`` is the autocast dtype of MERT and the head on CUDA (the heads were trained on bf16 features; CPU runs in fp32)."""

    def __init__(
        self,
        head_path: Union[str, Path],
        mert_model: str = MERT_REPO,
        mert_revision: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = torch.bfloat16,
        *,
        mert: Optional[nn.Module] = None,
        processor=None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.head_path = str(head_path)
        self.head, self.head_cfg = load_head(head_path)
        self.head.to(self.device)
        self.instnorm = self.head_cfg["instnorm"]
        self.head_fingerprint = file_sha256(head_path)
        self.mert_id = mert_identity(mert_model, mert_revision)

        if mert is None:
            from transformers import AutoFeatureExtractor, AutoModel

            logger.info(f"Loading MERT from {mert_model}")
            with torchaudio_transforms_available():
                processor = AutoFeatureExtractor.from_pretrained(mert_model, revision=mert_revision, trust_remote_code=True)
                mert = AutoModel.from_pretrained(mert_model, revision=mert_revision, trust_remote_code=True)
            mert = mert.to(device=self.device, dtype=torch.float32)
            _reinit_mert_rotary(mert)
        self.mert = mert.eval().requires_grad_(False) if isinstance(mert, nn.Module) else mert
        self.processor = processor
        sampling_rate = getattr(processor, "sampling_rate", MERT_SAMPLE_RATE) if processor is not None else MERT_SAMPLE_RATE
        if sampling_rate != MERT_SAMPLE_RATE:
            raise ValueError(f"MERT expects {sampling_rate} Hz input; YuE2 codes are defined at {MERT_SAMPLE_RATE} Hz")

    @property
    def fingerprint(self) -> dict[str, str]:
        """Cache metadata identifying the codes."""
        return {
            "yue2_semantic_head": self.head_fingerprint,
            "yue2_semantic_mert": self.mert_id,
            "yue2_semantic_version": SEMANTIC_VERSION,
        }

    def _autocast(self):
        enabled = self.dtype is not None and self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, dtype=self.dtype or torch.bfloat16, enabled=enabled)

    def _mert_inputs(self, group: list[torch.Tensor]) -> dict:
        if self.processor is None:
            return {"input_values": torch.stack([c.float() for c in group]).to(self.device)}
        inputs = self.processor([c.float().cpu().numpy() for c in group], sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def layer_features(self, mono24: torch.Tensor, size: Optional[int] = None) -> torch.Tensor:
        """``mono24 [S]`` -> layer-20 MERT features ``[size, 1024]`` (float32 holding fp16 values), linearly resampled
        from the MERT frame count to ``size`` rows (default ``round(S / 960)``).

        The result is a transposed view (time is the inner dimension in memory), even when no stretch is needed: the head
        GEMMs under bf16 autocast depend on that layout, and the published codes were produced with it."""
        if mono24.ndim != 1:
            raise ValueError(f"expected mono audio [S], got {tuple(mono24.shape)}")
        if not bool(torch.isfinite(mono24).all()):
            raise ValueError("audio contains non-finite samples")
        groups, keep_tail_frames = split_mert_chunks(mono24)
        last = len(groups) - 1

        pieces = []
        for index, group in enumerate(groups):
            with self._autocast():
                outputs = self.mert(**self._mert_inputs(group), output_hidden_states=True)
            layer = outputs.hidden_states[MERT_LAYER]
            frames = layer.reshape(-1, layer.shape[-1])
            if index == last and keep_tail_frames is not None:
                frames = frames[frames.shape[0] - keep_tail_frames :]
            pieces.append(frames)
        timeline = torch.cat(pieces, dim=0).float()

        rows = int(round(mono24.shape[0] / MERT_SAMPLE_RATE * 25)) if size is None else size
        resampled = _resample_time(timeline, rows)
        non_finite = int((~torch.isfinite(resampled)).sum())
        if non_finite:
            logger.warning(f"MERT features contain {non_finite} non-finite values")
        return resampled.half().float()

    @torch.no_grad()
    def tokenize_features(self, features: torch.Tensor) -> torch.Tensor:
        return classify_features(self.head, features, instnorm=self.instnorm, autocast_dtype=self.dtype).cpu()

    @torch.no_grad()
    def tokenize_frames(self, mono24: torch.Tensor, frames: int) -> torch.Tensor:
        """``mono24 [S]`` (the whole record) -> the codes of its first ``frames`` 48 kHz latent frames (int64, CPU).

        Reference convention (``prep_real.py`` + ``ar_prep.py``): features of all samples on the ``round(S / 960)`` grid,
        the first ``frames`` rows instance-normalised and classified. The heads were trained on exactly these features."""
        if frames < 1:
            raise ValueError("frames must be positive")
        mono24 = mono24.reshape(-1)
        deficit = frames * MERT_HOP - mono24.shape[0]
        if deficit > MERT_HOP:
            raise ValueError(f"24 kHz audio is {deficit} samples shorter than {frames} frames")
        size = max(int(round(mono24.shape[0] / MERT_SAMPLE_RATE * 25)), frames)
        features = self.layer_features(mono24, size)[:frames]
        codes = self.tokenize_features(features)
        unique = int(codes.unique().numel())
        if unique < 8:
            logger.warning(f"degenerate semantic token stream: {unique} unique codes in {frames} frames")
        return codes

    def unload(self) -> None:
        self.mert = None
        self.processor = None
        self.head = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# endregion
