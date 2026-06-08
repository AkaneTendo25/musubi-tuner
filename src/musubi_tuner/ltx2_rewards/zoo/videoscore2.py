"""VideoScore2 video reward -- wraps the TIGER-Lab/VideoScore2 generative VLM judge.

Extra deps (not installed with this package): ``pip install qwen-vl-utils``

VideoScore2 (arxiv 2509.22799) is a Qwen2.5-VL-7B model that thinks before it scores:
given a video + the generation prompt it produces a chain-of-thought rationale ending in a
"visual quality: N; text-to-video alignment: N, physical/common-sense consistency: N"
block. The vendored inferencer reads a soft score off the digit logits for each of the
three dimensions -- Visual Quality (VQ), Text-to-video Alignment (TA), and
Physical/common-sense Consistency (PC) -- each a float in roughly [0, 5].

This plugin's reward is the mean of a user-selected subset of those dimensions
(--reward_args dims=vq,ta,pc by default -> (VQ + TA + PC) / 3). Select a single head
for a targeted signal, e.g. dims=pc for the
PHYSICS / common-sense consistency head, dims=ta for prompt fidelity, or dims=vq for visual
quality. Higher = better, no inversion. A dimension that the model fails to emit (None soft
score) contributes 0.0 as a fallback.

VRAM: the 7B judge loads only during offline Phase A scoring and is torn down before the
next reward / the training step (RewardStack sequencing), so it never co-resides with the
DiT or the NFT update.

The vendored copy (musubi_tuner.ltx2_rewards.vendor.videoscore2) and its checkpoint are not
bundled with the registry; they are imported lazily in setup so the registry, CPU tests,
and the placeholder path never need them. Pass the checkpoint dir via
--reward_args checkpoint_path=<.../VideoScore2> dims=pc.
"""

from __future__ import annotations

import logging
from typing import List

from ..registry import BaseReward, register_reward

logger = logging.getLogger(__name__)

# Canonical dim key -> the score-dict key the vendored scorer emits.
_DIM_TO_KEY = dict(VQ="visual_quality", TA="text_to_video_alignment", PC="physical_consistency")
_VALID_DIMS = tuple(_DIM_TO_KEY)


def _parse_dims(dims) -> List[str]:
    """Parse a dims spec ("vq,ta,pc" / ["vq","pc"] / "PC") into canonical axes (VQ/TA/PC)."""
    if isinstance(dims, str):
        items = [d.strip() for d in dims.split(",")]
    else:
        items = [str(d).strip() for d in dims]
    out: List[str] = []
    for d in items:
        if not d:
            continue
        key = d.upper()
        if key not in _DIM_TO_KEY:
            raise ValueError(f"videoscore2: unknown dim {d!r}; valid dims are {_VALID_DIMS} (case-insensitive)")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("videoscore2: 'dims' resolved to empty; expected e.g. 'vq,ta,pc' or 'pc'")
    return out


def _video_tensor_to_uint8_thwc(video):
    """Decoded video [C,T,H,W] or [1,C,T,H,W] (float in [0,1], tolerant of [0,255]) ->
    uint8 [T,H,W,C] on CPU, the layout torchvision.io.write_video expects.
    """
    import torch

    t = video
    if t.dim() == 5:  # [1,C,T,H,W]
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"videoscore2: expected decoded video [C,T,H,W] (or [1,C,T,H,W]), got shape {tuple(video.shape)}")
    c, _frames, _h, _w = t.shape
    if c != 3:
        raise ValueError(f"videoscore2: expected 3 (RGB) channels in decoded video, got {c}")
    t = t.detach().to("cpu").float()
    scale = 255.0 if float(t.max()) <= 1.5 else 1.0
    t = (t * scale).clamp(0, 255).round().to(torch.uint8)  # [C,T,H,W]
    return t.permute(1, 2, 3, 0).contiguous()  # [T,H,W,C]


@register_reward("videoscore2")
class VideoScore2Reward(BaseReward):
    """VideoScore2 generative-VLM reward over a decoded video (route=video, higher=better)."""

    kind = "blackbox"
    route = "video"
    needs = frozenset(["video", "prompt"])

    def __init__(self):
        self._inferencer = None
        self._dims = ["VQ", "TA", "PC"]
        self._fps = 2.0
        self._max_new_tokens = 1024
        self._temperature = 0.7
        self._do_sample = True
        self._seed = None
        self._write_fps = 16

    def setup(
        self,
        device,
        *,
        checkpoint_path=None,
        dims="vq,ta,pc",
        fps=2.0,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        seed=None,
        write_fps=16,
        **_ignored,
    ):
        self._dims = _parse_dims(dims)
        self._fps = float(fps)
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._do_sample = str(do_sample).lower() not in ("false", "0", "no") if isinstance(do_sample, str) else bool(do_sample)
        self._seed = None if seed is None else int(seed)
        self._write_fps = int(write_fps)

        if checkpoint_path is None:
            raise ValueError(
                "videoscore2 reward requires --reward_args checkpoint_path=<.../VideoScore2> "
                "(the dir holding config.json + the model-*.safetensors shards)."
            )

        try:
            from ..vendor.videoscore2 import VideoScore2Inferencer
        except ImportError as exc:  # pragma: no cover - exercised only on the GPU box
            raise ImportError(
                "videoscore2 reward requires the vendored copy "
                "(musubi_tuner.ltx2_rewards.vendor.videoscore2) and its torch/transformers deps."
            ) from exc

        import torch

        dev = "cuda" if device is None else str(device)
        self._inferencer = VideoScore2Inferencer(model_name_or_path=checkpoint_path, device=dev, dtype=torch.bfloat16)
        logger.info("videoscore2: judge loaded on %s (dims=%s do_sample=%s)", dev, self._dims, self._do_sample)

    def _select(self, reward):
        """Mean of the selected axes from the scorer's score dict; a None (unparsed) axis
        contributes 0.0."""
        vals = []
        for d in self._dims:
            v = reward.get(_DIM_TO_KEY[d])
            vals.append(float(v) if v is not None else 0.0)
        return sum(vals) / len(vals)

    def score(self, samples):
        if self._inferencer is None:
            raise RuntimeError("videoscore2 reward: setup() must run before score()")
        import os
        import tempfile

        import torch
        from torchvision.io import write_video

        scores = []
        for sample in samples:
            video = sample.get("video")
            prompt = sample.get("prompt", "")
            if video is None:
                logger.warning("videoscore2: sample has no decoded 'video'; scoring 0.0")
                scores.append(0.0)
                continue
            # Write the decoded clip to a temp mp4 so the vendored scorer reads it back
            # (qwen_vl_utils -> smart_resize -> processor), matching the file-based pipeline.
            frames = _video_tensor_to_uint8_thwc(video)  # [T,H,W,C] uint8 cpu
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                write_video(tmp_path, frames, fps=self._write_fps, video_codec="h264")
                with torch.no_grad():
                    reward = self._inferencer.score(
                        tmp_path,
                        prompt,
                        fps=self._fps,
                        max_new_tokens=self._max_new_tokens,
                        temperature=self._temperature,
                        do_sample=self._do_sample,
                        seed=self._seed,
                    )
                scores.append(self._select(reward))
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        info = dict(reward="videoscore2", dims=list(self._dims), do_sample=self._do_sample)
        return scores, info

    def teardown(self):
        if self._inferencer is not None:
            try:
                import torch

                del self._inferencer
                self._inferencer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort VRAM release
                self._inferencer = None
