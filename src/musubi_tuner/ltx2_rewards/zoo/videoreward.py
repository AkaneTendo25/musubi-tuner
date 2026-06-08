"""VideoReward (VideoAlign) video reward -- wraps the KwaiVGI/VideoAlign VLM reward.

Extra deps (not installed with this package): ``pip install qwen-vl-utils peft``
(``peft`` is not needed when scoring with a pre-merged checkpoint).

VideoReward (arxiv 2501.13918) is a Qwen2-VL reward model that scores a generated
video on three axes via three special reward tokens: Visual Quality (VQ), Motion
Quality (MQ), and Text Alignment (TA). The vendored inferencer normalizes each axis
with the checkpoint's ``inference_config`` mean/std and exposes them as a per-video
dict {VQ, MQ, TA, Overall}.

This plugin's reward is the mean of a user-selected subset of those axes
(``--reward_args dims=vq,ta`` by default -> ``(VQ + TA) / 2``, matching the common
VideoAlign preference signal; pass e.g. ``dims=mq`` for motion coherence, or
``dims=vq,mq,ta`` for the overall mean). Higher = better, no inversion.

VRAM: the Qwen2-VL reward model loads only during offline Phase A scoring and is torn
down before the next reward / the training step (``RewardStack`` sequencing), so it
never co-resides with the DiT or the NFT update.

The vendored copy (``musubi_tuner.ltx2_rewards.vendor.videoreward``) and its checkpoint
are not bundled with the registry; they are imported lazily in ``setup`` so the
registry, CPU tests, and the placeholder path never need them. Pass the checkpoint dir
via ``--reward_args checkpoint_path=<.../VideoReward-merged> dims=vq,ta``.
``checkpoint_path`` may be the parent dir holding ``model_config.json`` + ``checkpoint-*``
subdirs, or a ``checkpoint-*`` dir directly.

The inference path is peft-free and loads a MERGED checkpoint. The released checkpoint
ships an unmerged LoRA state dict; fold it once with the offline merge helper
(``python -m musubi_tuner.ltx2_rewards.vendor.videoreward.merge_checkpoint --src ... --dst ...``)
and point ``checkpoint_path`` at the merged dir. An unmerged checkpoint raises a clear
error in ``setup`` pointing back to the merge step.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from ..registry import BaseReward, register_reward

logger = logging.getLogger(__name__)

_VALID_DIMS = ("VQ", "MQ", "TA")


def _parse_dims(dims) -> List[str]:
    """Parse a ``dims`` spec ("vq,ta" / ["vq","ta"] / "VQ") into a list of canonical axes."""
    if isinstance(dims, str):
        items = [d.strip() for d in dims.split(",")]
    else:
        items = [str(d).strip() for d in dims]
    out: List[str] = []
    for d in items:
        if not d:
            continue
        key = d.upper()
        if key not in _VALID_DIMS:
            raise ValueError(f"videoreward: unknown dim {d!r}; valid dims are {_VALID_DIMS} (case-insensitive)")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("videoreward: 'dims' resolved to empty; expected e.g. 'vq,ta' or 'mq'")
    return out


def _video_tensor_to_uint8_thwc(video):
    """Decoded video ``[C,T,H,W]`` or ``[1,C,T,H,W]`` (float in [0,1], tolerant of [0,255])
    -> uint8 ``[T,H,W,C]`` on CPU, the layout ``torchvision.io.write_video`` expects.
    """
    import torch

    t = video
    if t.dim() == 5:  # [1,C,T,H,W]
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"videoreward: expected decoded video [C,T,H,W] (or [1,C,T,H,W]), got shape {tuple(video.shape)}")
    c, _frames, _h, _w = t.shape
    if c != 3:
        raise ValueError(f"videoreward: expected 3 (RGB) channels in decoded video, got {c}")
    t = t.detach().to("cpu").float()
    scale = 255.0 if float(t.max()) <= 1.5 else 1.0
    t = (t * scale).clamp(0, 255).round().to(torch.uint8)  # [C,T,H,W]
    return t.permute(1, 2, 3, 0).contiguous()  # [T,H,W,C]


@register_reward("videoreward")
class VideoRewardReward(BaseReward):
    """VideoAlign VLM reward (VQ/MQ/TA) over a decoded video (route=video, higher=better)."""

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video", "prompt"})

    def __init__(self) -> None:
        self._inferencer = None
        self._dims = ["VQ", "TA"]
        self._use_norm = True
        self._fps = None
        self._num_frames = None
        self._write_fps = 16

    def setup(
        self,
        device,
        *,
        checkpoint_path: str = None,
        dims="vq,ta",
        use_norm=True,
        fps=None,
        num_frames=None,
        write_fps=16,
        **_ignored,
    ) -> None:
        self._dims = _parse_dims(dims)
        self._use_norm = str(use_norm).lower() not in ("false", "0", "no") if isinstance(use_norm, str) else bool(use_norm)
        self._fps = None if fps is None else float(fps)
        self._num_frames = None if num_frames is None else int(num_frames)
        self._write_fps = int(write_fps)

        if checkpoint_path is None:
            raise ValueError(
                "videoreward reward requires --reward_args checkpoint_path=<.../VideoReward> "
                "(the dir holding model_config.json + checkpoint-* subdirs, or a checkpoint-* dir)."
            )

        try:
            from ..vendor.videoreward import VideoVLMRewardInference
        except ImportError as exc:  # pragma: no cover - exercised only on the GPU box
            raise ImportError(
                "videoreward reward requires the vendored copy "
                "(musubi_tuner.ltx2_rewards.vendor.videoreward) and its torch/transformers deps."
            ) from exc

        import torch

        dev = "cuda" if device is None else str(device)
        self._inferencer = VideoVLMRewardInference(load_from_pretrained=checkpoint_path, device=dev, dtype=torch.bfloat16)
        logger.info("videoreward: reward model loaded on %s (dims=%s use_norm=%s)", dev, self._dims, self._use_norm)

    def _select(self, reward: dict) -> float:
        """Mean of the selected axes from a {VQ, MQ, TA, Overall} reward dict."""
        return sum(float(reward[d]) for d in self._dims) / len(self._dims)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        if self._inferencer is None:
            raise RuntimeError("videoreward reward: setup() must run before score()")
        import os
        import tempfile

        import torch
        from torchvision.io import write_video

        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            prompt = sample.get("prompt", "")
            if video is None:
                logger.warning("videoreward: sample has no decoded 'video'; scoring 0.0")
                scores.append(0.0)
                continue

            # Feed the model exactly as it expects: write the decoded clip to a temp mp4 and
            # let the vendored inferencer read it back (torchvision -> smart_resize -> processor),
            # so the in-process score matches the original file-based pipeline.
            frames = _video_tensor_to_uint8_thwc(video)  # [T,H,W,C] uint8 cpu
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                write_video(tmp_path, frames, fps=self._write_fps, video_codec="h264")
                with torch.no_grad():
                    rewards = self._inferencer.reward(
                        [tmp_path],
                        [prompt],
                        fps=self._fps,
                        num_frames=self._num_frames,
                        use_norm=self._use_norm,
                    )
                scores.append(self._select(rewards[0]))
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return scores, {"reward": "videoreward", "dims": list(self._dims), "use_norm": self._use_norm}

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
