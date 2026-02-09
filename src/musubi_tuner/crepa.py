"""CREPA – Cross-frame Representation Alignment (arxiv 2506.09229).

Training-time regularization that aligns DiT hidden states across video frames
by encouraging temporal consistency in a learned feature space.

Two modes:
- **backbone**: teacher signal from a deeper transformer block within the same model.
- **dino**: teacher signal from pre-cached DINOv2 per-frame patch tokens (zero VRAM
  at training time — features are loaded from disk).

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
    lambda_crepa: float = 0.1             # loss weight
    tau: float = 1.0                      # temporal neighbor decay factor
    num_neighbors: int = 2                # K frames on each side
    schedule: str = "constant"            # "constant" | "linear" | "cosine"
    warmup_steps: int = 0
    max_steps: int = 0                    # needed for cosine/linear schedules
    normalize: bool = True                # L2-normalize features before similarity


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

        self.projector: Optional[CREPAProjector] = None
        self._student_features: Optional[torch.Tensor] = None
        self._teacher_features: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._current_lambda: float = config.lambda_crepa
        # Track the number of temporal tokens per frame for reshape
        self._num_temporal_frames: Optional[int] = None

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
            self._current_lambda = cfg.lambda_crepa * (1.0 - progress)
        elif cfg.schedule == "cosine":
            self._current_lambda = cfg.lambda_crepa * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            self._current_lambda = cfg.lambda_crepa

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

        if cfg.mode == "dino":
            return self._compute_loss_dino(num_latent_frames, dino_features)
        else:
            return self._compute_loss_backbone(num_latent_frames)

    def _compute_loss_backbone(self, num_latent_frames: int) -> Optional[torch.Tensor]:
        if self._student_features is None or self._teacher_features is None:
            return None
        if self._current_lambda == 0.0:
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
        tau = cfg.tau

        loss = torch.zeros(B, device=sim.device, dtype=sim.dtype)
        for f in range(T):
            loss = loss - sim[:, f, f]
            for delta in range(1, K + 1):
                weight = math.exp(-delta / tau)
                if f - delta >= 0:
                    loss = loss - weight * sim[:, f, f - delta]
                if f + delta < T:
                    loss = loss - weight * sim[:, f, f + delta]

        # Normalize by number of terms per frame
        num_terms = T
        for f in range(T):
            for delta in range(1, K + 1):
                if f - delta >= 0:
                    num_terms += 1
                if f + delta < T:
                    num_terms += 1
        loss = loss.mean() / max(num_terms / T, 1.0)

        crepa_loss = loss * self._current_lambda

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
