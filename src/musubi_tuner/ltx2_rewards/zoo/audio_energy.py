"""Mean audio loudness reward (no model), route=audio.

RMS energy of the decoded waveform, squashed to [0,1) via ``rms / (rms + half_point)``;
higher = louder. Model-free and dense — the audio analog of ``saturation``: a demo reward
whose only purpose is making audio-branch learning cheaply observable.
``half_point`` (reward_arg, default 0.1) is the RMS that scores 0.5.

Example: ``--ltx2_mode av --reward_fn "audio_energy:1.0"``.
"""

from __future__ import annotations

from typing import List, Tuple

from ..registry import BaseReward, register_reward


def _rms_energy(waveform) -> float:
    """RMS of a waveform tensor (any shape, samples in [-1, 1]); 0.0 for empty."""
    import torch

    t = waveform.detach().to("cpu", torch.float32)
    if t.numel() == 0:
        return 0.0
    return float(t.pow(2).mean().sqrt())


@register_reward("audio_energy")
class AudioEnergyReward(BaseReward):
    """Mean RMS loudness reward — model-free, demonstrative (audio analog of ``saturation``)."""

    kind = "blackbox"
    route = "audio"
    needs = frozenset({"audio_waveform"})

    def __init__(self) -> None:
        self._half_point = 0.1

    def setup(self, device, *, half_point=0.1, **_ignored) -> None:
        self._half_point = float(half_point)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            waveform = sample.get("audio_waveform")
            if waveform is None:
                scores.append(0.0)
                continue
            rms = _rms_energy(waveform)
            scores.append(rms / (rms + self._half_point))
        return scores, {"reward": "audio_energy", "half_point": self._half_point}

    def teardown(self) -> None:
        pass
