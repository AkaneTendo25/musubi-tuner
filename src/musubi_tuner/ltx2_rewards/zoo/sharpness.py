"""Example / template reward: per-frame sharpness (no model, no checkpoint).

This is the canonical pattern for adding **your own** reward target. Copy this file,
rename the class + ``@register_reward`` name, set ``route``/``kind``/``needs``, and
implement ``score()``. Register it by importing it in ``zoo/__init__.py``, then run
with ``--reward_fn <name>:<weight>`` (Phase A) — GRPO advantages, the rollout cache,
the NFT loss, VRAM-sequenced setup/teardown, and video/audio routing all happen
automatically.

The reward contract (see ``registry.BaseReward``):
  - ``kind``  : "blackbox" (scored, not differentiated) or "differentiable" (ReFL).
  - ``route`` : "video" | "audio" | "sync" (sync adds the advantage to BOTH branches).
  - ``needs`` : the per-sample keys ``generate_fn`` must provide; declaring "video"
                or "audio_waveform" makes the generator decode that media for scoring.
  - ``setup(device, **reward_args)`` : load any model here (called once, then torn down
                before the next reward / the training step — never co-resident in VRAM).
  - ``score(samples) -> ([float per sample], info)`` : MUST be higher-is-better
                (apply any inversion such as ``1/(1+distance)`` here).
  - ``teardown()`` : free the model.

This particular reward needs no model: it measures mean frame sharpness (variance of a
Laplacian-style high-pass), a cheap proxy for "crisp, in-focus, low-blur" frames. It is a
real usable signal as well as the template.
"""

from __future__ import annotations

from typing import List, Tuple

from ..registry import BaseReward, register_reward


def _frame_sharpness(video, num_frames: int = 5) -> float:
    """Mean high-pass energy over uniformly sampled frames of a decoded video tensor.

    ``video`` is ``[C,T,H,W]`` or ``[1,C,T,H,W]`` in [0,1] (the do_inference decoded layout).
    Higher = sharper / less blurry.
    """
    import torch
    import torch.nn.functional as F

    t = video
    if t.dim() == 5:
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"sharpness: expected decoded video [C,T,H,W], got {tuple(video.shape)}")
    t = t.detach().to("cpu", torch.float32)
    c, frames, _h, _w = t.shape
    gray = t.mean(dim=0, keepdim=True)  # [1,T,H,W] luminance proxy
    # 3x3 Laplacian high-pass per frame
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    idx = [round(i * (frames - 1) / max(1, num_frames - 1)) for i in range(min(num_frames, frames))]
    energies = []
    for f in idx:
        frame = gray[:, f].unsqueeze(0)  # [1,1,H,W]
        hp = F.conv2d(frame, kernel, padding=1)
        energies.append(float(hp.pow(2).mean()))
    return sum(energies) / len(energies) if energies else 0.0


@register_reward("sharpness")
class SharpnessReward(BaseReward):
    """Mean per-frame sharpness (Laplacian-energy) reward — example/template, no model."""

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video"})

    def __init__(self) -> None:
        self._num_frames = 5

    def setup(self, device, *, num_frames=5, **_ignored) -> None:
        # No model to load. A real reward would construct its model here (on `device`),
        # reading any paths/options from reward_args kwargs.
        self._num_frames = int(num_frames)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            if video is None:
                scores.append(0.0)
                continue
            scores.append(_frame_sharpness(video, self._num_frames))
        return scores, {"reward": "sharpness", "num_frames": self._num_frames}

    def teardown(self) -> None:
        # No model to free. A real reward would `del self._model; torch.cuda.empty_cache()` here.
        pass
