"""Small NetworkTrainer batch-state accessors."""

from __future__ import annotations

from typing import Any, Optional


def set_current_batch_latents_info(self, latents_info: Optional[dict[str, Any]]) -> None:
    self._current_batch_latents_info = latents_info


def get_current_batch_latents_info(self) -> Optional[dict[str, Any]]:
    return self._current_batch_latents_info
