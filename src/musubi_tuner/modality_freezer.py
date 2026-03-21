"""G2D-style sequential modality prioritization for joint audio-video training.

Monitors per-modality loss EMA and freezes the dominant modality's LoRA
parameters when the loss ratio exceeds a threshold.  This lets the
under-performing modality train without gradient interference from the
modality that has already converged.

Reference: G2D, "Sequential Modality Prioritization" (2025).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModalityFreezer:
    """Tracks per-modality loss EMA and freezes/unfreezes LoRA modules."""

    def __init__(
        self,
        *,
        check_interval: int = 500,
        ratio_threshold: float = 0.5,
        warmup_steps: int = 100,
        ema_decay: float = 0.99,
    ):
        self.check_interval = max(1, check_interval)
        self.ratio_threshold = ratio_threshold
        self.warmup_steps = max(0, warmup_steps)
        self.ema_decay = ema_decay

        self._video_loss_ema: float = 0.0
        self._audio_loss_ema: float = 0.0
        self._ema_initialized: bool = False
        self._state: str = "both"  # "both" | "video_frozen" | "audio_frozen"
        self._steps_with_losses: int = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def video_loss_ema(self) -> float:
        return self._video_loss_ema

    @property
    def audio_loss_ema(self) -> float:
        return self._audio_loss_ema

    def update_losses(self, video_loss: Optional[float], audio_loss: Optional[float]) -> None:
        """Update per-modality loss EMAs. Call every step."""
        if video_loss is None or audio_loss is None:
            return
        if not self._ema_initialized:
            self._video_loss_ema = float(video_loss)
            self._audio_loss_ema = float(audio_loss)
            self._ema_initialized = True
        else:
            d = self.ema_decay
            self._video_loss_ema = d * self._video_loss_ema + (1.0 - d) * float(video_loss)
            self._audio_loss_ema = d * self._audio_loss_ema + (1.0 - d) * float(audio_loss)
        self._steps_with_losses += 1

    def maybe_update_freeze(self, global_step: int, network) -> Optional[str]:
        """Check and update freeze state. Returns the new state if changed, else None.

        Should be called every step; only acts at check_interval boundaries.
        """
        if global_step < self.warmup_steps:
            return None
        if global_step % self.check_interval != 0:
            return None
        if not self._ema_initialized:
            return None

        video_ema = max(self._video_loss_ema, 1e-12)
        audio_ema = max(self._audio_loss_ema, 1e-12)
        ratio = audio_ema / video_ema

        old_state = self._state

        if ratio < self.ratio_threshold:
            # Audio loss << video loss → audio has overfit, freeze audio
            new_state = "audio_frozen"
        elif ratio > 1.0 / self.ratio_threshold:
            # Video loss << audio loss → video has overfit, freeze video
            new_state = "video_frozen"
        else:
            new_state = "both"

        if new_state == old_state:
            return None

        self._state = new_state
        self._apply_freeze(network, new_state)
        logger.info(
            "ModalityFreezer: %s → %s (audio_ema=%.4f, video_ema=%.4f, ratio=%.4f, threshold=%.4f)",
            old_state, new_state, audio_ema, video_ema, ratio, self.ratio_threshold,
        )
        return new_state

    @staticmethod
    def _apply_freeze(network, state: str) -> None:
        """Set requires_grad on LoRA modules based on freeze state."""
        lora_modules = getattr(network, "unet_loras", None)
        if not lora_modules:
            return
        for lora in lora_modules:
            is_audio = "audio_" in lora.lora_name
            if state == "audio_frozen":
                requires_grad = not is_audio
            elif state == "video_frozen":
                requires_grad = is_audio
            else:  # "both"
                requires_grad = True
            for param in lora.parameters():
                param.requires_grad_(requires_grad)
