"""HPSv3 video reward — wraps the HPSv3 VLM preference model (MizzenAI/HPSv3).

Extra deps (not installed with this package): ``pip install qwen-vl-utils``

HPSv3 is a Qwen2-VL-7B reward model scoring image-prompt alignment / human
preference; its ``logits`` output is ``(mu, sigma)`` per image and ``mu`` (higher =
better) is the reward. For a video we sample a few frames uniformly, score each
frame, and aggregate the top fraction.

VRAM: the 7B reward model loads only during offline Phase A scoring and is torn
down before the next reward / the training step (``RewardStack`` sequencing), so it
never co-resides with the DiT or the NFT update.

The ``hpsv3`` package (vendored from github.com/MizzenAI/HPSv3) and its
checkpoints are not bundled here — they are imported lazily in ``setup`` so the
registry, CPU tests, and the placeholder path never need them. Pass the checkpoint
and config via ``--reward_args checkpoint_path=... config_path=...``.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

from ..registry import BaseReward, register_reward

logger = logging.getLogger(__name__)


def _aggregate_top_frac(scores: List[float], frac: float) -> float:
    """Mean of the top ``frac`` of frame scores (default frac=0.3)."""
    if not scores:
        return 0.0
    ordered = sorted(scores, reverse=True)
    k = max(1, math.ceil(len(ordered) * frac))
    return sum(ordered[:k]) / k


def _uniform_frame_indices(num_frames_total: int, want: int) -> List[int]:
    """Uniformly spaced frame indices over ``[0, num_frames_total-1]`` (incl. ends)."""
    if num_frames_total <= 0:
        return []
    want = min(want, num_frames_total)
    if want <= 1:
        return [0]
    return [round(i * (num_frames_total - 1) / (want - 1)) for i in range(want)]


def _video_tensor_to_pil_frames(video, want: int):
    """Sample ``want`` frames from a decoded video tensor -> list of PIL.Image (RGB).

    Accepts ``[C,T,H,W]`` or ``[1,C,T,H,W]`` (the ``do_inference`` decoded layout,
    float in [0,1]); also tolerates [0,255] tensors. Channel dim must be 3 (RGB).
    """
    import numpy as np
    from PIL import Image

    t = video
    if t.dim() == 5:  # [1,C,T,H,W]
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"hpsv3: expected decoded video [C,T,H,W] (or [1,C,T,H,W]), got shape {tuple(video.shape)}")
    c, frames, _h, _w = t.shape
    if c != 3:
        raise ValueError(f"hpsv3: expected 3 (RGB) channels in decoded video, got {c}")
    t = t.detach().to("cpu").float()
    scale = 255.0 if float(t.max()) <= 1.5 else 1.0
    pil_frames = []
    for idx in _uniform_frame_indices(frames, want):
        frame = t[:, idx]  # [C,H,W]
        arr = (frame * scale).clamp(0, 255).round().to("cpu").numpy().astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))  # [H,W,C]
        pil_frames.append(Image.fromarray(arr, mode="RGB"))
    return pil_frames


@register_reward("hpsv3")
class HPSv3Reward(BaseReward):
    """HPSv3 human-preference reward over decoded video frames (route=video)."""

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video", "prompt"})

    def __init__(self) -> None:
        self._inferencer = None
        self._num_frames = 5
        self._top_frac = 0.3
        self._score_cap = 15.0

    def setup(
        self,
        device,
        *,
        checkpoint_path: str = None,
        config_path: str = None,
        num_frames=5,
        top_frac=0.3,
        score_cap=15.0,
        **_ignored,
    ) -> None:
        self._num_frames = int(num_frames)
        self._top_frac = float(top_frac)
        self._score_cap = float(score_cap)
        try:
            # Prefer the in-repo inference-only vendored copy (no trl/datasets/peft/...).
            from ..vendor.hpsv3 import HPSv3RewardInferencer
        except ImportError:
            try:
                # Fall back to the external 'hpsv3' package if it is installed.
                from hpsv3 import HPSv3RewardInferencer
            except ImportError as exc:  # pragma: no cover - exercised only on the GPU box
                raise ImportError(
                    "hpsv3 reward requires the vendored copy "
                    "(musubi_tuner.ltx2_rewards.vendor.hpsv3) or the external 'hpsv3' package "
                    "(github.com/MizzenAI/HPSv3). Then pass "
                    "--reward_args checkpoint_path=<HPSv3.safetensors>."
                ) from exc
        dev = "cuda" if device is None else str(device)
        self._inferencer = HPSv3RewardInferencer(config_path=config_path, checkpoint_path=checkpoint_path, device=dev)
        logger.info("hpsv3: reward model loaded on %s (frames=%d top_frac=%.2f)", dev, self._num_frames, self._top_frac)

    def _score_frames(self, pil_frames, prompt: str) -> List[float]:
        """Score a list of frames against one prompt; returns per-frame mu (capped).

        Runs under ``torch.no_grad()`` — the inferencer's own ``reward()`` uses
        ``@torch.inference_mode()`` but we call ``.model()`` directly, so without this the 7B
        Qwen2-VL forward would build an autograd graph and retain every activation (the scoring
        peak was ~52 GB; no_grad brings it down to roughly the model footprint).
        """
        if not pil_frames:
            return []
        import torch

        prompts = [prompt] * len(pil_frames)
        with torch.no_grad():
            batch = self._inferencer.prepare_batch(pil_frames, prompts)
            logits = self._inferencer.model(return_dict=True, **batch)["logits"]
        return [min(float(logits[i][0].item()), self._score_cap) for i in range(len(pil_frames))]

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        if self._inferencer is None:
            raise RuntimeError("hpsv3 reward: setup() must run before score()")
        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            prompt = sample.get("prompt", "")
            if video is None:
                logger.warning("hpsv3: sample has no decoded 'video'; scoring 0.0")
                scores.append(0.0)
                continue
            pil_frames = _video_tensor_to_pil_frames(video, self._num_frames)
            frame_scores = self._score_frames(pil_frames, prompt)
            scores.append(_aggregate_top_frac(frame_scores, self._top_frac))
        return scores, {"reward": "hpsv3", "num_frames": self._num_frames}

    def teardown(self) -> None:
        if self._inferencer is not None:
            try:
                import torch

                del self._inferencer
                self._inferencer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort VRAM release
                self._inferencer = None
