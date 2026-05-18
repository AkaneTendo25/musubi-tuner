"""Cross-frame representation alignment regularization for video DiT fine-tuning.

Training-time regularization that aligns DiT hidden states across video frames
by encouraging temporal consistency in a learned feature space.

Two modes:
- **backbone**: teacher signal from a deeper transformer block within the same model.
  Inspired by SimpleTuner's LayerSync (https://github.com/bghira/SimpleTuner).
- **dino**: teacher signal from pre-cached DINOv2 per-frame patch tokens (zero VRAM
  at training time — features are loaded from disk).
  Based on CREPA – Cross-frame Representation Alignment (arxiv 2506.09229).

Only the small projector MLP is trained; all other modules stay frozen.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# DINOv2 model name → token dimension
DINO_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CREPAConfig:
    mode: str = "backbone"               # "backbone" | "dino"
    student_block_idx: int = 16           # block whose hidden states are aligned
    teacher_block_idx: int = 32           # backbone teacher block (backbone mode)
    dino_model: str = "dinov2_vitb14"     # DINOv2 model name (dino mode, future)
    lambda_crepa: float = 0.5             # starting loss weight
    lambda_end: float = 0.1               # ending loss weight for linear/cosine schedules
    tau: float = 1.0                      # temporal neighbor decay factor
    num_neighbors: int = 2                # K frames on each side
    schedule: str = "cosine"              # "constant" | "linear" | "cosine"
    warmup_steps: int = 100
    max_steps: int = 0                    # needed for cosine/linear schedules; 0 = trainer max steps
    normalize: bool = True                # L2-normalize features before similarity
    cutoff_step: int = 0                  # hard-disable CREPA at/after this global step (0 = disabled)
    similarity_threshold: Optional[float] = 0.85  # EMA alignment score cutoff (None = disabled)
    similarity_ema_decay: float = 0.99    # EMA smoothing for similarity_threshold
    threshold_mode: str = "permanent"     # "permanent" | "recoverable"


# ---------------------------------------------------------------------------
# Projector MLP
# ---------------------------------------------------------------------------

class CREPAProjector(nn.Module):
    """Small 2-layer MLP: student_dim → teacher_dim."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class CREPAModule:
    """Orchestrates hook installation, feature capture, and loss computation."""

    def __init__(self, config: CREPAConfig, transformer: nn.Module):
        self.config = config
        self.transformer = transformer
        self._normalize_config()

        self.projector: Optional[CREPAProjector] = None
        self._student_features: Optional[torch.Tensor] = None
        self._teacher_features: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._current_lambda: float = config.lambda_crepa
        self._effective_lambda: float = config.lambda_crepa
        self._current_step: int = 0
        self._similarity_ema: Optional[float] = None
        self._cutoff_triggered: bool = False
        self._cutoff_active: bool = False
        self._last_alignment_score: Optional[float] = None
        self._last_self_similarity: Optional[float] = None
        # Track the number of temporal tokens per frame for reshape
        self._num_temporal_frames: Optional[int] = None

    def _normalize_config(self) -> None:
        cfg = self.config
        cfg.mode = str(cfg.mode).lower()
        cfg.schedule = str(cfg.schedule).lower()
        cfg.lambda_crepa = float(cfg.lambda_crepa)
        cfg.lambda_end = float(cfg.lambda_end)
        if cfg.lambda_crepa < 0.0 or cfg.lambda_end < 0.0:
            raise ValueError(f"CREPA lambda values must be >= 0, got: {cfg.lambda_crepa}, {cfg.lambda_end}")
        cfg.threshold_mode = str(cfg.threshold_mode or "permanent").lower()
        if cfg.threshold_mode not in {"permanent", "recoverable"}:
            raise ValueError(f"CREPA threshold_mode must be permanent|recoverable, got: {cfg.threshold_mode}")
        cfg.cutoff_step = int(cfg.cutoff_step or 0)
        if cfg.cutoff_step < 0:
            raise ValueError(f"CREPA cutoff_step must be >= 0, got: {cfg.cutoff_step}")
        if cfg.similarity_threshold is not None:
            cfg.similarity_threshold = float(cfg.similarity_threshold)
            if not math.isfinite(cfg.similarity_threshold):
                raise ValueError(f"CREPA similarity_threshold must be finite or None, got: {cfg.similarity_threshold}")
            if not 0.0 <= cfg.similarity_threshold <= 0.99:
                raise ValueError(f"CREPA similarity_threshold must be in [0, 0.99] or None, got: {cfg.similarity_threshold}")
        cfg.similarity_ema_decay = float(cfg.similarity_ema_decay)
        if not (0.0 <= cfg.similarity_ema_decay < 1.0):
            raise ValueError(f"CREPA similarity_ema_decay must be in [0, 1), got: {cfg.similarity_ema_decay}")

    # ----- setup ----------------------------------------------------------

    def setup(self, device: torch.device, dtype: torch.dtype) -> None:
        """Create projector, install hooks.  Call once after model is ready."""
        cfg = self.config

        # Determine dimensions from transformer blocks
        blocks = self.transformer.transformer_blocks
        num_blocks = len(blocks)

        if cfg.student_block_idx >= num_blocks:
            raise ValueError(
                f"student_block_idx={cfg.student_block_idx} out of range (model has {num_blocks} blocks)"
            )
        if cfg.mode == "backbone" and cfg.teacher_block_idx >= num_blocks:
            raise ValueError(
                f"teacher_block_idx={cfg.teacher_block_idx} out of range (model has {num_blocks} blocks)"
            )

        # inner_dim is the hidden dimension of video hidden states (TransformerArgs.x)
        # For LTX-2: inner_dim = num_attention_heads * attention_head_dim = 32 * 128 = 4096
        inner_dim = self.transformer.inner_dim

        if cfg.mode == "backbone":
            # Both student and teacher are from the same model → same dim
            self.projector = CREPAProjector(inner_dim, inner_dim).to(device=device, dtype=dtype)
            logger.info(
                "CREPA backbone mode: student_block=%d, teacher_block=%d, dim=%d, projector params=%s",
                cfg.student_block_idx,
                cfg.teacher_block_idx,
                inner_dim,
                f"{sum(p.numel() for p in self.projector.parameters()):,}",
            )
        elif cfg.mode == "dino":
            dino_dim = DINO_DIMS.get(cfg.dino_model)
            if dino_dim is None:
                raise ValueError(
                    f"Unknown DINOv2 model '{cfg.dino_model}'. "
                    f"Supported: {', '.join(DINO_DIMS.keys())}"
                )
            # Project student DiT features → DINOv2 feature space
            self.projector = CREPAProjector(inner_dim, dino_dim).to(device=device, dtype=dtype)
            logger.info(
                "CREPA dino mode: student_block=%d, dino_model=%s (dim=%d), projector params=%s",
                cfg.student_block_idx,
                cfg.dino_model,
                dino_dim,
                f"{sum(p.numel() for p in self.projector.parameters()):,}",
            )
        else:
            raise NotImplementedError(f"CREPA mode '{cfg.mode}' not implemented")

        self._install_hooks()

    # ----- hooks ----------------------------------------------------------

    def _install_hooks(self) -> None:
        blocks = self.transformer.transformer_blocks
        cfg = self.config

        def _make_student_hook():
            def hook(_module, _input, output):
                # output is (video: TransformerArgs|None, audio: TransformerArgs|None)
                video_out = output[0]
                if video_out is not None:
                    # .x has shape [B, T*H*W, D]
                    self._student_features = video_out.x
            return hook

        def _make_teacher_hook():
            def hook(_module, _input, output):
                video_out = output[0]
                if video_out is not None:
                    self._teacher_features = video_out.x.detach()
            return hook

        h1 = blocks[cfg.student_block_idx].register_forward_hook(_make_student_hook())
        self._hooks.append(h1)

        if cfg.mode == "backbone":
            h2 = blocks[cfg.teacher_block_idx].register_forward_hook(_make_teacher_hook())
            self._hooks.append(h2)

        logger.info("CREPA: installed %d forward hooks", len(self._hooks))

    # ----- trainable params -----------------------------------------------

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        if self.projector is None:
            return []
        return list(self.projector.parameters())

    # ----- schedule -------------------------------------------------------

    def on_step(self, global_step: int) -> None:
        cfg = self.config
        self._current_step = int(global_step)
        if cfg.schedule == "constant":
            self._current_lambda = cfg.lambda_crepa
            return

        if cfg.warmup_steps > 0 and global_step < cfg.warmup_steps:
            self._current_lambda = cfg.lambda_crepa * (global_step / cfg.warmup_steps)
            return

        if cfg.max_steps <= 0:
            self._current_lambda = cfg.lambda_crepa
            return

        progress = min((global_step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1), 1.0)

        if cfg.schedule == "linear":
            self._current_lambda = cfg.lambda_end + (cfg.lambda_crepa - cfg.lambda_end) * (1.0 - progress)
        elif cfg.schedule == "cosine":
            self._current_lambda = cfg.lambda_end + (cfg.lambda_crepa - cfg.lambda_end) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        else:
            self._current_lambda = cfg.lambda_crepa

    # ----- cutoff --------------------------------------------------------

    @property
    def last_alignment_score(self) -> Optional[float]:
        return self._last_alignment_score

    @property
    def last_self_similarity(self) -> Optional[float]:
        return self._last_self_similarity

    @property
    def similarity_ema(self) -> Optional[float]:
        return self._similarity_ema

    @property
    def effective_lambda(self) -> float:
        return float(self._effective_lambda)

    @property
    def cutoff_active(self) -> bool:
        return bool(self._cutoff_active or (self._cutoff_triggered and self.config.threshold_mode == "permanent"))

    def get_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "crepa/weight": float(self._effective_lambda),
            "crepa/cutoff": 1.0 if self.cutoff_active else 0.0,
        }
        if self._last_alignment_score is not None:
            metrics["crepa/alignment_score"] = float(self._last_alignment_score)
        if self._last_self_similarity is not None:
            metrics["crepa/similarity_self"] = float(self._last_self_similarity)
        if self._similarity_ema is not None:
            metrics["crepa/alignment_score_ema"] = float(self._similarity_ema)
        return metrics

    def _update_similarity_ema(self, similarity: Optional[float]) -> None:
        if similarity is None:
            return
        if self._similarity_ema is None:
            self._similarity_ema = similarity
            return
        decay = self.config.similarity_ema_decay
        self._similarity_ema = decay * self._similarity_ema + (1.0 - decay) * similarity

    def _check_cutoff(self) -> bool:
        cfg = self.config
        if cfg.cutoff_step > 0 and self._current_step >= cfg.cutoff_step:
            return True
        if cfg.similarity_threshold is not None and self._similarity_ema is not None:
            return self._similarity_ema >= cfg.similarity_threshold
        return False

    def _current_effective_lambda(self, similarity: Optional[float]) -> float:
        self._update_similarity_ema(similarity)

        if self._cutoff_triggered and self.config.threshold_mode == "permanent":
            self._cutoff_active = True
            self._effective_lambda = 0.0
            return 0.0

        cutoff_active = self._check_cutoff()
        self._cutoff_active = cutoff_active
        if cutoff_active:
            if self.config.threshold_mode == "permanent":
                self._cutoff_triggered = True
            self._effective_lambda = 0.0
            return 0.0

        if self.config.threshold_mode == "recoverable":
            self._cutoff_triggered = False

        self._effective_lambda = float(self._current_lambda)
        return self._current_lambda

    # ----- loss -----------------------------------------------------------

    def compute_loss(
        self,
        num_latent_frames: int,
        dino_features: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Compute CREPA loss from captured features.

        Args:
            num_latent_frames: number of temporal frames in the latent space (T).
            dino_features: pre-cached DINOv2 patch tokens ``[B, T_pixel, N_patches, D_dino]``
                (only used in dino mode).

        Returns:
            Scalar loss tensor, or None if features were not captured.
        """
        cfg = self.config
        self._last_alignment_score = None
        self._last_self_similarity = None
        self._cutoff_active = bool(self._cutoff_triggered and cfg.threshold_mode == "permanent")
        self._effective_lambda = 0.0 if self._cutoff_active else float(self._current_lambda)

        if cfg.mode == "dino":
            return self._compute_loss_dino(num_latent_frames, dino_features)
        else:
            return self._compute_loss_backbone(num_latent_frames)

    def _compute_loss_backbone(self, num_latent_frames: int) -> Optional[torch.Tensor]:
        if self._student_features is None or self._teacher_features is None:
            return None
        if self._current_lambda == 0.0:
            self._effective_lambda = 0.0
            self._cutoff_active = bool(self._cutoff_triggered and self.config.threshold_mode == "permanent")
            return None
        if self._cutoff_triggered and self.config.threshold_mode == "permanent":
            self._effective_lambda = 0.0
            self._cutoff_active = True
            return None

        cfg = self.config
        student_feat = self._student_features   # [B, T*H*W, D]
        teacher_feat = self._teacher_features   # [B, T*H*W, D]

        B, THW, D_s = student_feat.shape
        T = num_latent_frames
        if T <= 0 or THW % T != 0:
            logger.warning("CREPA: cannot reshape features (THW=%d, T=%d), skipping", THW, T)
            return None
        HW = THW // T

        B_t, THW_t, D_t = teacher_feat.shape
        HW_t = THW_t // T

        # Project student features
        projected = self.projector(student_feat)  # [B, T*H*W, D_t]

        # Reshape to frame-level and average pool spatial dims → [B, T, D]
        proj_frames = projected.reshape(B, T, HW, -1).mean(dim=2)
        teach_frames = teacher_feat.reshape(B_t, T, HW_t, D_t).mean(dim=2)

        return self._similarity_loss(proj_frames, teach_frames, T)

    def _compute_loss_dino(
        self,
        num_latent_frames: int,
        dino_features: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self._student_features is None:
            return None
        if dino_features is None:
            return None
        if self._current_lambda == 0.0:
            self._effective_lambda = 0.0
            self._cutoff_active = bool(self._cutoff_triggered and self.config.threshold_mode == "permanent")
            return None
        if self._cutoff_triggered and self.config.threshold_mode == "permanent":
            self._effective_lambda = 0.0
            self._cutoff_active = True
            return None

        student_feat = self._student_features  # [B, T_latent*H*W, D_s]
        B, THW, D_s = student_feat.shape
        T = num_latent_frames
        if T <= 0 or THW % T != 0:
            logger.warning("CREPA dino: cannot reshape features (THW=%d, T=%d), skipping", THW, T)
            return None
        HW = THW // T

        # Project student → DINOv2 space: [B, T*H*W, D_dino]
        projected = self.projector(student_feat)
        # Reshape to [B, T, HW, D_dino] — keep spatial tokens (no mean-pool)
        proj_frames = projected.reshape(B, T, HW, -1)

        # dino_features: [B, T_pixel, N_patches, D_dino]
        dino_features = dino_features.to(device=proj_frames.device, dtype=proj_frames.dtype)
        T_pixel = dino_features.shape[1]

        # Temporal alignment: subsample T_pixel → T_latent
        if T_pixel != T:
            # Select evenly-spaced frame indices
            indices = torch.linspace(0, T_pixel - 1, T, device=dino_features.device).long()
            teach_frames = dino_features[:, indices]  # [B, T, N_patches, D]
        else:
            teach_frames = dino_features  # [B, T, N_patches, D]

        # Spatial alignment: interpolate token counts if HW != N_patches
        N_teach = teach_frames.shape[2]
        if HW != N_teach:
            # Interpolate student spatial tokens to match teacher count
            # [B, T, HW, D] → [B*T, D, HW] → interpolate → [B*T, D, N_teach] → [B, T, N_teach, D]
            D_dino = proj_frames.shape[-1]
            proj_flat = proj_frames.reshape(B * T, HW, D_dino).permute(0, 2, 1)  # [B*T, D, HW]
            proj_flat = F.interpolate(proj_flat, size=N_teach, mode="linear", align_corners=False)
            proj_frames = proj_flat.permute(0, 2, 1).reshape(B, T, N_teach, D_dino)  # [B, T, N_teach, D]

        # proj_frames: [B, T, N, D], teach_frames: [B, T, N, D]
        return self._similarity_loss(proj_frames, teach_frames, T)

    def _similarity_loss(
        self,
        proj_frames: torch.Tensor,
        teach_frames: torch.Tensor,
        T: int,
    ) -> Optional[torch.Tensor]:
        """Shared cosine-similarity + neighbor weighting loss.

        Supports both 3D ``[B, T, D]`` (backbone mode) and 4D ``[B, T, N, D]``
        (dino patch mode). For 4D input, computes per-patch cosine similarity
        and averages over the patch dimension.
        """
        cfg = self.config
        B = proj_frames.shape[0]
        is_4d = proj_frames.ndim == 4

        if cfg.normalize:
            proj_frames = F.normalize(proj_frames, dim=-1)
            teach_frames = F.normalize(teach_frames, dim=-1)

        if is_4d:
            # [B, T, N, D] — per-patch cosine similarity, mean over patches
            # sim[b, t1, t2] = mean_over_n( sum_d(proj[b,t1,n,d] * teach[b,t2,n,d]) )
            # Use einsum: [B, T1, N, D] x [B, T2, N, D] → [B, T1, T2, N] → mean over N
            sim = torch.einsum("btnd,bsnd->btsn", proj_frames, teach_frames).mean(dim=-1)  # [B, T, T]
        else:
            # [B, T, D] — standard cosine similarity matrix
            sim = torch.bmm(proj_frames, teach_frames.transpose(1, 2))  # [B, T, T]

        K = cfg.num_neighbors
        tau = max(cfg.tau, 1e-8)

        loss = torch.zeros(B, device=sim.device, dtype=sim.dtype)
        alignment_sum = torch.zeros(B, T, device=sim.device, dtype=sim.dtype)
        alignment_weight_sum = torch.zeros(B, T, device=sim.device, dtype=sim.dtype)
        for f in range(T):
            self_sim = sim[:, f, f]
            loss = loss - self_sim
            alignment_sum[:, f] = alignment_sum[:, f] + self_sim
            alignment_weight_sum[:, f] = alignment_weight_sum[:, f] + 1.0
            for delta in range(1, K + 1):
                weight = math.exp(-delta / tau)
                if f - delta >= 0:
                    neighbor_sim = sim[:, f, f - delta]
                    loss = loss - weight * neighbor_sim
                    alignment_sum[:, f] = alignment_sum[:, f] + weight * neighbor_sim
                    alignment_weight_sum[:, f] = alignment_weight_sum[:, f] + weight
                if f + delta < T:
                    neighbor_sim = sim[:, f, f + delta]
                    loss = loss - weight * neighbor_sim
                    alignment_sum[:, f] = alignment_sum[:, f] + weight * neighbor_sim
                    alignment_weight_sum[:, f] = alignment_weight_sum[:, f] + weight

        # Normalize by number of terms per frame
        num_terms = T
        for f in range(T):
            for delta in range(1, K + 1):
                if f - delta >= 0:
                    num_terms += 1
                if f + delta < T:
                    num_terms += 1
        loss = loss.mean() / max(num_terms / T, 1.0)

        alignment = alignment_sum / alignment_weight_sum.clamp_min(torch.finfo(alignment_weight_sum.dtype).eps)
        alignment_score = float(alignment.mean().detach().item())
        self_similarity = float(sim.diagonal(dim1=1, dim2=2).mean().detach().item())
        self._last_alignment_score = alignment_score
        self._last_self_similarity = self_similarity

        effective_lambda = self._current_effective_lambda(alignment_score)
        if effective_lambda == 0.0:
            return None

        crepa_loss = loss * effective_lambda

        if not torch.isfinite(crepa_loss):
            logger.warning("CREPA loss is non-finite (%.4g), skipping", crepa_loss.item())
            return None

        return crepa_loss

    # ----- cleanup --------------------------------------------------------

    def cleanup_step(self) -> None:
        """Clear captured features for next step."""
        self._student_features = None
        self._teacher_features = None

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("CREPA: removed all hooks")

    # ----- checkpoint -----------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        if self.projector is None:
            return {}
        return self.projector.state_dict()

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if self.projector is not None and sd:
            self.projector.load_state_dict(sd)
            logger.info("CREPA: loaded projector weights (%d tensors)", len(sd))

    def training_state_dict(self) -> Dict[str, torch.Tensor]:
        ema_valid = self._similarity_ema is not None
        ema_value = 0.0 if self._similarity_ema is None else float(self._similarity_ema)
        return {
            "similarity_ema": torch.tensor([ema_value], dtype=torch.float64),
            "similarity_ema_valid": torch.tensor([1 if ema_valid else 0], dtype=torch.int64),
            "cutoff_triggered": torch.tensor([1 if self._cutoff_triggered else 0], dtype=torch.int64),
        }

    def load_training_state_dict(self, sd: Dict[str, Any]) -> None:
        if not sd:
            return
        ema_valid = sd.get("similarity_ema_valid")
        ema_value = sd.get("similarity_ema")
        if isinstance(ema_valid, torch.Tensor) and ema_valid.numel() > 0 and int(ema_valid.flatten()[0].item()) != 0:
            if isinstance(ema_value, torch.Tensor) and ema_value.numel() > 0:
                self._similarity_ema = float(ema_value.flatten()[0].item())
        else:
            self._similarity_ema = None

        cutoff_triggered = sd.get("cutoff_triggered")
        if isinstance(cutoff_triggered, torch.Tensor) and cutoff_triggered.numel() > 0:
            self._cutoff_triggered = int(cutoff_triggered.flatten()[0].item()) != 0
            self._cutoff_active = bool(self._cutoff_triggered and self.config.threshold_mode == "permanent")
            self._effective_lambda = 0.0 if self._cutoff_active else float(self._current_lambda)

        logger.info(
            "CREPA: loaded cutoff state (similarity_ema=%s, cutoff_triggered=%s)",
            "None" if self._similarity_ema is None else f"{self._similarity_ema:.6f}",
            self._cutoff_triggered,
        )


# ---------------------------------------------------------------------------
# CLI arg parsing helper
# ---------------------------------------------------------------------------

def parse_crepa_args(raw_args: Optional[list[str]]) -> Dict[str, str]:
    """Parse ``key=value`` list into a dict.  Returns empty dict for None/[]."""
    if not raw_args:
        return {}
    out: Dict[str, str] = {}
    for item in raw_args:
        if "=" not in item:
            raise ValueError(f"CREPA arg must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out
