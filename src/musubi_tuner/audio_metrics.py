"""Audio quality metrics for AV training.

Standalone module.  Injected via ~6 lines in the training loop.

Per-step (latent-space): Latent FD, temporal coherence, AV latent sync.
Periodic (mel-space): spectral convergence, MCD, log-spectral distance.
Sampling (embedding-space): CLAP similarity, AV onset alignment.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AudioMetricsConfig:
    # Per-step (latent-space)
    latent_fd: bool = True
    latent_fd_window: int = 500
    latent_fd_compute_every: int = 50
    temporal_coherence: bool = True
    av_latent_sync: bool = True

    # Periodic (mel-space, every N steps)
    mel_metrics: bool = False
    mel_compute_every: int = 100
    spectral_convergence: bool = True
    mcd: bool = True
    mcd_coefficients: int = 13
    log_spectral_distance: bool = True

    # Sampling-time (embedding-space)
    clap_similarity: bool = False
    clap_model: str = "laion/clap-htsat-unfused"
    av_onset_alignment: bool = False


def parse_audio_metrics_args(raw_args: list[str] | None) -> dict[str, str]:
    """Parse key=value CLI args (same pattern as parse_preservation_args)."""
    if not raw_args:
        return {}
    out: dict[str, str] = {}
    for arg in raw_args:
        if "=" in arg:
            k, v = arg.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _config_from_kwargs(kw: dict[str, str]) -> AudioMetricsConfig:
    """Build config from parsed key=value dict with type coercion."""
    bool_keys = {
        "latent_fd", "temporal_coherence", "av_latent_sync",
        "mel_metrics", "spectral_convergence", "mcd", "log_spectral_distance",
        "clap_similarity", "av_onset_alignment",
    }
    int_keys = {
        "latent_fd_window", "latent_fd_compute_every",
        "mel_compute_every", "mcd_coefficients",
    }
    str_keys = {"clap_model"}

    typed: dict[str, Any] = {}
    for k, v in kw.items():
        if k in bool_keys:
            typed[k] = v.lower() in ("true", "1", "yes")
        elif k in int_keys:
            typed[k] = int(v)
        elif k in str_keys:
            typed[k] = v
        else:
            logger.warning("Unknown audio_metrics arg: %s=%s", k, v)
    return AudioMetricsConfig(**typed)


# ---------------------------------------------------------------------------
# Running statistics for Fréchet Distance
# ---------------------------------------------------------------------------

class RunningStats:
    """Online mean and covariance accumulator (Welford's algorithm).

    Keeps a circular buffer of the last ``window`` batch contributions.
    """

    def __init__(self, dim: int, window: int = 500) -> None:
        self.dim = dim
        self.window = window
        self._count = 0
        self._mean = torch.zeros(dim, dtype=torch.float64)
        self._M2 = torch.zeros(dim, dim, dtype=torch.float64)  # sum of outer products of deviations

    def update(self, batch: torch.Tensor) -> None:
        """Update with a batch of vectors ``[N, dim]``."""
        batch = batch.detach().double().cpu()
        for x in batch:
            self._count += 1
            delta = x - self._mean
            self._mean += delta / self._count
            delta2 = x - self._mean
            self._M2 += delta.unsqueeze(1) * delta2.unsqueeze(0)

        # Soft window: decay old stats when count exceeds window
        if self._count > self.window:
            decay = self.window / self._count
            self._M2 *= decay
            self._count = self.window

    def get_stats(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return (mean, covariance) or None if too few samples."""
        if self._count < 2:
            return None
        cov = self._M2 / (self._count - 1)
        return self._mean.clone(), cov.clone()

    def reset(self) -> None:
        self._count = 0
        self._mean.zero_()
        self._M2.zero_()


# ---------------------------------------------------------------------------
# Pure-PyTorch Fréchet Distance
# ---------------------------------------------------------------------------

def _frechet_distance(
    mu1: torch.Tensor, sigma1: torch.Tensor,
    mu2: torch.Tensor, sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet distance between two multivariate Gaussians.

    Uses eigendecomposition instead of scipy sqrtm for portability.
    """
    diff = mu1 - mu2

    # Regularize covariance matrices
    sigma1 = sigma1 + eps * torch.eye(sigma1.shape[0], dtype=sigma1.dtype)
    sigma2 = sigma2 + eps * torch.eye(sigma2.shape[0], dtype=sigma2.dtype)

    # Product of covariance matrices
    product = sigma1 @ sigma2

    # Matrix square root via eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(product)
    eigvals = eigvals.clamp(min=0)  # numerical safety
    sqrt_product = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T

    trace_term = sigma1.trace() + sigma2.trace() - 2 * sqrt_product.trace()
    fd = float((diff @ diff) + trace_term)
    return max(fd, 0.0)


# ---------------------------------------------------------------------------
# Tier 1: Latent-space metrics
# ---------------------------------------------------------------------------

def _temporal_coherence(latent: torch.Tensor) -> float | None:
    """Cosine similarity between adjacent frames in ``[B, C, T, F]`` latent.

    Returns mean cosine similarity (higher = more temporally coherent).
    """
    if latent.dim() != 4 or latent.shape[2] < 2:
        return None
    # Flatten spatial dims: [B, C, T, F] -> [B, T, C*F]
    B, C, T, Freq = latent.shape
    flat = latent.permute(0, 2, 1, 3).reshape(B, T, C * Freq)
    # Cosine sim between adjacent frames
    cos = F.cosine_similarity(flat[:, :-1, :], flat[:, 1:, :], dim=-1)  # [B, T-1]
    return float(cos.mean())


def _av_latent_sync(
    audio: torch.Tensor, video: torch.Tensor,
) -> float | None:
    """Pearson correlation between audio and video per-frame energy curves.

    audio: [B, C_a, T_a, F_a],  video: [B, C_v, T_v, H, W] or [B, C_v, T_v, S].
    Resamples the longer to match the shorter via linear interpolation.
    """
    if audio.dim() < 3 or video.dim() < 3:
        return None

    # Per-frame energy (L2 norm over non-batch, non-temporal dims)
    # Audio: [B, C, T, F] -> energy per frame [B, T]
    a_energy = audio.flatten(start_dim=1, end_dim=1)  # keep batch
    # Actually, compute norm over C and F per frame
    B = audio.shape[0]
    T_a = audio.shape[2]
    a_flat = audio.permute(0, 2, 1, 3).reshape(B, T_a, -1)  # [B, T_a, C*F]
    a_energy = a_flat.norm(dim=-1)  # [B, T_a]

    # Video: handle both [B, C, T, H, W] and [B, C, T, S] shapes
    if video.dim() == 5:
        T_v = video.shape[2]
        v_flat = video.permute(0, 2, 1, 3, 4).reshape(B, T_v, -1)
    elif video.dim() == 4:
        T_v = video.shape[2]
        v_flat = video.permute(0, 2, 1, 3).reshape(B, T_v, -1)
    else:
        return None
    v_energy = v_flat.norm(dim=-1)  # [B, T_v]

    # Resample to match lengths
    T_min = min(T_a, T_v)
    if T_min < 2:
        return None
    if T_a != T_min:
        a_energy = F.interpolate(a_energy.unsqueeze(1), size=T_min, mode="linear", align_corners=False).squeeze(1)
    if T_v != T_min:
        v_energy = F.interpolate(v_energy.unsqueeze(1), size=T_min, mode="linear", align_corners=False).squeeze(1)

    # Pearson correlation per sample, then average
    a_centered = a_energy - a_energy.mean(dim=-1, keepdim=True)
    v_centered = v_energy - v_energy.mean(dim=-1, keepdim=True)
    num = (a_centered * v_centered).sum(dim=-1)
    denom = a_centered.norm(dim=-1) * v_centered.norm(dim=-1)
    corr = num / denom.clamp(min=1e-8)  # [B]
    return float(corr.mean())


# ---------------------------------------------------------------------------
# Periodic: Mel-space metrics
# ---------------------------------------------------------------------------

def _spectral_convergence(mel_pred: torch.Tensor, mel_target: torch.Tensor) -> float:
    """||S_pred - S_target||_F / ||S_target||_F."""
    diff_norm = torch.linalg.norm(mel_pred - mel_target)
    target_norm = torch.linalg.norm(mel_target)
    if target_norm < 1e-8:
        return 0.0
    return float(diff_norm / target_norm)


def _mel_cepstral_distortion(
    mel_pred: torch.Tensor, mel_target: torch.Tensor,
    n_coefficients: int = 13,
) -> float | None:
    """MCD: DCT of log-mel, first N coefficients, Euclidean distance per frame.

    mel_pred, mel_target: [T, F] (single sample, already log-mel).
    Returns MCD in dB.  <4 dB is near-perceptual equivalence for speech.
    """
    if mel_pred.dim() != 2 or mel_pred.shape[0] < 1:
        return None

    # Ensure log-mel (if not already)
    # The AudioDecoder output may already be in log scale — caller must verify.
    # We apply a safety clamp.
    pred_log = mel_pred.float()
    target_log = mel_target.float()

    # Type-II DCT via torch.fft (real-to-real cosine transform)
    # DCT-II: X[k] = 2 * sum_n x[n] * cos(pi*(2n+1)*k / (2N))
    N = pred_log.shape[-1]
    n_coeff = min(n_coefficients, N)

    pred_dct = torch.fft.rfft(pred_log, dim=-1).real[:, :n_coeff]
    target_dct = torch.fft.rfft(target_log, dim=-1).real[:, :n_coeff]

    # Euclidean distance per frame, then average
    frame_dist = (pred_dct - target_dct).norm(dim=-1)  # [T]
    # Scale factor: (10 * sqrt(2) / ln(10)) ≈ 6.1416 for dB conversion
    mcd = float(frame_dist.mean()) * (10.0 * math.sqrt(2) / math.log(10))
    return mcd


def _log_spectral_distance(mel_pred: torch.Tensor, mel_target: torch.Tensor) -> float | None:
    """Per-frame log-spectral distance in dB.

    mel_pred, mel_target: [T, F].
    """
    if mel_pred.dim() != 2 or mel_pred.shape[0] < 1:
        return None

    pred = mel_pred.float().clamp(min=1e-8)
    target = mel_target.float().clamp(min=1e-8)

    # Convert to power if in log domain — assume inputs are log-mel
    # LSD = sqrt(mean((10*log10(S_pred) - 10*log10(S_target))^2))
    # If already in log domain: LSD = sqrt(mean((pred - target)^2)) * 10/ln(10)
    diff_sq = (pred - target) ** 2
    lsd_per_frame = diff_sq.mean(dim=-1).sqrt()  # [T]
    # Scale: the inputs are in natural log, convert difference to dB
    lsd_db = float(lsd_per_frame.mean()) * (10.0 / math.log(10))
    return lsd_db


# ---------------------------------------------------------------------------
# Sampling-time: Embedding-space metrics (lazy-loaded)
# ---------------------------------------------------------------------------

class _CLAPComputer:
    """CLAP audio-text cosine similarity."""

    def __init__(self, model_name: str = "laion/clap-htsat-unfused") -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._scores: list[float] = []

    def _load_model(self, device: torch.device) -> None:
        if self._model is not None:
            return
        try:
            from transformers import ClapModel, ClapProcessor
            self._model = ClapModel.from_pretrained(self.model_name).to(device).eval()
            self._processor = ClapProcessor.from_pretrained(self.model_name)
        except ImportError as e:
            raise ImportError(
                "CLAP similarity requires transformers with CLAP support. "
                "Install with: pip install transformers>=4.30.0"
            ) from e

    @torch.no_grad()
    def accumulate(
        self, waveform: torch.Tensor, text_prompt: str,
        device: torch.device, sample_rate: int = 24000,
    ) -> None:
        """Compute and accumulate CLAP similarity for one sample."""
        self._load_model(device)
        self._model.to(device)

        # Resample to 48kHz (CLAP expectation)
        audio_np = waveform.cpu().float().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np[0]  # mono

        inputs = self._processor(
            audios=[audio_np],
            sampling_rate=sample_rate,
            text=[text_prompt],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        audio_emb = self._model.get_audio_features(**{
            k: v for k, v in inputs.items() if k.startswith("input_features") or k == "is_longer"
        })
        text_emb = self._model.get_text_features(**{
            k: v for k, v in inputs.items() if k.startswith("input_ids") or k == "attention_mask"
        })

        cos = F.cosine_similarity(audio_emb, text_emb, dim=-1)
        self._scores.append(float(cos.mean()))

    def compute(self) -> float | None:
        if not self._scores:
            return None
        return sum(self._scores) / len(self._scores)

    def reset(self) -> None:
        self._scores.clear()

    def offload(self) -> None:
        if self._model is not None:
            self._model.cpu()


def _av_onset_alignment(
    audio_waveform: torch.Tensor,
    video_latent: torch.Tensor,
) -> float | None:
    """Correlation between audio spectral flux and video motion.

    audio_waveform: [samples] or [1, samples].
    video_latent: [C, T, H, W] or [C, T, S] (single sample).
    """
    if audio_waveform.dim() == 2:
        audio_waveform = audio_waveform[0]
    if audio_waveform.numel() < 2:
        return None

    # Audio spectral flux: energy derivative (simple frame-level energy)
    frame_size = max(1, audio_waveform.numel() // 100)
    n_frames = audio_waveform.numel() // frame_size
    if n_frames < 3:
        return None
    audio_frames = audio_waveform[:n_frames * frame_size].reshape(n_frames, frame_size)
    audio_energy = (audio_frames ** 2).mean(dim=-1)  # [n_frames]
    audio_flux = torch.diff(audio_energy)  # [n_frames - 1]

    # Video motion: L2 norm of frame differences
    if video_latent.dim() == 4:
        T_v = video_latent.shape[1]
        v_flat = video_latent.permute(1, 0, 2, 3).reshape(T_v, -1)
    elif video_latent.dim() == 3:
        T_v = video_latent.shape[1]
        v_flat = video_latent.permute(1, 0, 2).reshape(T_v, -1)
    else:
        return None
    if T_v < 3:
        return None
    video_motion = torch.diff(v_flat, dim=0).norm(dim=-1)  # [T_v - 1]

    # Resample to match
    T_min = min(len(audio_flux), len(video_motion))
    if T_min < 2:
        return None
    af = F.interpolate(audio_flux.float().unsqueeze(0).unsqueeze(0), size=T_min, mode="linear", align_corners=False).squeeze()
    vm = F.interpolate(video_motion.float().unsqueeze(0).unsqueeze(0), size=T_min, mode="linear", align_corners=False).squeeze()

    # Pearson correlation
    af_c = af - af.mean()
    vm_c = vm - vm.mean()
    num = (af_c * vm_c).sum()
    denom = af_c.norm() * vm_c.norm()
    if denom < 1e-8:
        return None
    return float(num / denom)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class AudioMetricsModule:
    """Standalone audio metrics computer.

    Initialized once, called per-step and at sampling time.
    Returns plain dicts that get merged into the existing logging flow.
    """

    def __init__(self, config: AudioMetricsConfig) -> None:
        self.config = config
        self._step = 0

        # Per-step: running stats for Latent FD
        self._pred_stats: RunningStats | None = None
        self._target_stats: RunningStats | None = None
        self._latent_dim: int | None = None

        # Sampling-time: lazy-loaded computers
        self._clap: _CLAPComputer | None = None
        if config.clap_similarity:
            self._clap = _CLAPComputer(config.clap_model)

    def on_step(self, global_step: int) -> None:
        self._step = global_step

    def should_compute_mel(self, global_step: int) -> bool:
        return (
            self.config.mel_metrics
            and global_step > 0
            and global_step % self.config.mel_compute_every == 0
        )

    # ----- Per-step: latent-space -----

    @torch.no_grad()
    def compute_latent_metrics(
        self,
        audio_pred: torch.Tensor,
        audio_target: torch.Tensor,
        video_pred: torch.Tensor | None = None,
        video_target: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute per-step latent-space metrics.

        audio_pred, audio_target: [B, C, T, F] detached float tensors.
        video_pred/target: optional, for AV sync.
        """
        metrics: dict[str, float] = {}

        # Temporal coherence
        if self.config.temporal_coherence:
            tc = _temporal_coherence(audio_pred)
            if tc is not None:
                metrics["audio_metrics/temporal_coherence"] = tc

        # AV latent sync
        if self.config.av_latent_sync and video_pred is not None:
            vp = video_pred.detach().float() if video_pred.is_floating_point() else video_pred.float()
            sync = _av_latent_sync(audio_pred, vp)
            if sync is not None:
                metrics["audio_metrics/av_latent_sync"] = sync

        # Latent FD (accumulate every step, compute periodically)
        if self.config.latent_fd:
            self._update_latent_fd(audio_pred, audio_target)
            if self._step > 0 and self._step % self.config.latent_fd_compute_every == 0:
                fd = self._compute_latent_fd()
                if fd is not None:
                    metrics["audio_metrics/latent_fd"] = fd

        return metrics

    def _update_latent_fd(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate running stats for latent FD."""
        B = pred.shape[0]
        flat_pred = pred.reshape(B, -1)
        flat_target = target.reshape(B, -1)
        dim = flat_pred.shape[1]

        # Cap dimensionality to avoid huge covariance matrices
        max_dim = 1024
        if dim > max_dim:
            # Random projection to reduce dimensionality
            if self._latent_dim != dim:
                self._proj = torch.randn(dim, max_dim, dtype=torch.float32) / math.sqrt(max_dim)
                self._latent_dim = dim
            flat_pred = flat_pred.float().cpu() @ self._proj
            flat_target = flat_target.float().cpu() @ self._proj
            dim = max_dim

        if self._pred_stats is None or self._pred_stats.dim != dim:
            self._pred_stats = RunningStats(dim, self.config.latent_fd_window)
            self._target_stats = RunningStats(dim, self.config.latent_fd_window)

        self._pred_stats.update(flat_pred)
        self._target_stats.update(flat_target)

    def _compute_latent_fd(self) -> float | None:
        if self._pred_stats is None or self._target_stats is None:
            return None
        pred = self._pred_stats.get_stats()
        target = self._target_stats.get_stats()
        if pred is None or target is None:
            return None
        try:
            return _frechet_distance(pred[0], pred[1], target[0], target[1])
        except Exception:
            return None

    # ----- Periodic: mel-space -----

    @torch.no_grad()
    def compute_mel_metrics(
        self,
        audio_pred_latent: torch.Tensor,
        audio_target_latent: torch.Tensor,
        audio_decoder: torch.nn.Module,
    ) -> dict[str, float]:
        """Compute periodic mel-space metrics.

        Decodes 1 sample through audio_decoder (no vocoder needed).
        audio_pred_latent, audio_target_latent: [B, C, T, F] raw latents.
        audio_decoder: AudioDecoder module (outputs mel-like intermediate).
        """
        metrics: dict[str, float] = {}

        # Decode only the first sample to save compute
        device = audio_pred_latent.device
        dtype = audio_pred_latent.dtype
        pred_lat = audio_pred_latent[0:1].to(device=device, dtype=dtype)
        target_lat = audio_target_latent[0:1].to(device=device, dtype=dtype)

        try:
            audio_decoder.to(device)
            mel_pred = audio_decoder(pred_lat).squeeze(0)  # [C_mel, T_mel] or similar
            mel_target = audio_decoder(target_lat).squeeze(0)
            audio_decoder.cpu()
        except Exception as e:
            logger.warning("Mel-space metrics decode failed: %s", e)
            return metrics

        # Flatten to [T, F] if needed
        if mel_pred.dim() == 3:
            mel_pred = mel_pred.squeeze(0)
        if mel_target.dim() == 3:
            mel_target = mel_target.squeeze(0)
        if mel_pred.dim() == 2 and mel_pred.shape[0] > mel_pred.shape[1]:
            # Likely [T, F] already
            pass
        elif mel_pred.dim() == 2:
            # Likely [F, T] — transpose
            mel_pred = mel_pred.T
            mel_target = mel_target.T

        mel_pred = mel_pred.float()
        mel_target = mel_target.float()

        if self.config.spectral_convergence:
            sc = _spectral_convergence(mel_pred, mel_target)
            metrics["audio_metrics/spectral_convergence"] = sc

        if self.config.mcd:
            mcd = _mel_cepstral_distortion(mel_pred, mel_target, self.config.mcd_coefficients)
            if mcd is not None:
                metrics["audio_metrics/mcd_db"] = mcd

        if self.config.log_spectral_distance:
            lsd = _log_spectral_distance(mel_pred, mel_target)
            if lsd is not None:
                metrics["audio_metrics/log_spectral_distance_db"] = lsd

        return metrics

    # ----- Sampling-time: embedding-space -----

    @torch.no_grad()
    def on_sample(
        self,
        waveform: torch.Tensor | None = None,
        text_prompt: str | None = None,
        video_latent: torch.Tensor | None = None,
        device: torch.device | None = None,
        sample_rate: int = 24000,
    ) -> dict[str, float]:
        """Compute sampling-time metrics for one generated sample.

        Called from sample_images() after audio decode.
        waveform: [channels, samples] or [samples].
        video_latent: [C, T, H, W] or [C, T, S] (single sample, optional).
        """
        if device is None:
            device = torch.device("cpu")
        metrics: dict[str, float] = {}

        if waveform is not None and self._clap is not None and text_prompt:
            try:
                self._clap.accumulate(waveform, text_prompt, device, sample_rate)
                clap_score = self._clap.compute()
                if clap_score is not None:
                    metrics["sample_audio/clap_similarity"] = clap_score
                self._clap.offload()
            except Exception as e:
                logger.warning("CLAP scoring failed: %s", e)

        if (
            self.config.av_onset_alignment
            and waveform is not None
            and video_latent is not None
        ):
            onset = _av_onset_alignment(waveform, video_latent)
            if onset is not None:
                metrics["sample_audio/av_onset_alignment"] = onset

        return metrics

    def compute_validation_summary(self) -> dict[str, float]:
        """Return empty dict — kept for API compat. Sampling metrics are returned inline."""
        return {}

    def reset(self) -> None:
        """Full reset of all running state."""
        if self._pred_stats is not None:
            self._pred_stats.reset()
        if self._target_stats is not None:
            self._target_stats.reset()
        if self._clap is not None:
            self._clap.reset()
