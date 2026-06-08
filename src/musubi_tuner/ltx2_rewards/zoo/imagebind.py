"""ImageBind multimodal-similarity reward — text/video/audio alignment (route=sync).

Extra deps (not installed with this package): ``pip install ftfy regex`` (CLIP tokenizer);
optionally ``pip install pytorchvideo fvcore iopath`` for the reference-exact video decode
(without them a torchvision fallback is used).

Wraps Meta's ImageBind "huge" model (facebookresearch/ImageBind, vendored in-repo at
``musubi_tuner.ltx2_rewards.vendor.imagebind``) to score how well a generated clip's
*video*, *audio*, and *text prompt* agree with each other. For each sample it embeds
the three modalities and combines three cosine similarities::

    sim_tv = cos(text_v, video)   # text <-> video alignment
    sim_ta = cos(text_a, audio)   # text <-> audio alignment
    sim_av = cos(audio,  video)   # audio <-> video sync
    score  = (sim_tv + sim_ta + sim_av) / 3   # higher = better

Because it spans both branches, ``route="sync"`` (the advantage is added to both the
video and audio branches). The reward NEEDS decoded media on disk: ``video_file`` (mp4)
+ ``audio_file`` (wav) + ``prompt``. Our current pipeline is video-only, so this reward
becomes usable once AV generation/decoding is wired; until then it is exercised by the
CPU unit test (fake model) and the standalone GPU parity check.

VRAM: the model loads only during offline Phase-A scoring and is torn down before the
next reward / training step (``RewardStack`` sequencing), so it never co-resides with
the DiT. Pass the checkpoint via ``--reward_args checkpoint_path=<imagebind_huge.pth>``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ..registry import BaseReward, register_reward

logger = logging.getLogger(__name__)


@register_reward("imagebind")
class ImageBindReward(BaseReward):
    """ImageBind (text+video+audio) cosine-similarity reward — (tv+ta+av)/3, route=sync."""

    kind = "blackbox"
    route = "sync"
    needs = frozenset({"video_file", "audio_file", "prompt"})

    def __init__(self) -> None:
        self._model = None
        self._device = "cuda"
        # Lazily imported handles (populated in setup()).
        self._data = None
        self._ModalityType = None
        self._cos = None

    def setup(self, device, *, checkpoint_path: Optional[str] = None, **_ignored) -> None:
        """Lazy-load ``imagebind_huge`` from ``checkpoint_path`` onto ``device``.

        Prefers the in-repo vendored copy; falls back to an externally installed
        ``imagebind`` package only if the vendored import fails.
        """
        import torch

        try:
            # In-repo inference-only vendored copy (no fvcore/iopath/matplotlib).
            from ..vendor.imagebind import data as imagebind_data
            from ..vendor.imagebind.models import imagebind_model
            from ..vendor.imagebind.models.imagebind_model import ModalityType
        except ImportError:
            try:  # pragma: no cover - exercised only where the external pkg is installed
                from imagebind import data as imagebind_data
                from imagebind.models import imagebind_model
                from imagebind.models.imagebind_model import ModalityType
            except ImportError as exc:  # pragma: no cover - GPU box only
                raise ImportError(
                    "imagebind reward requires the vendored copy "
                    "(musubi_tuner.ltx2_rewards.vendor.imagebind) or an external "
                    "'imagebind' package (github.com/facebookresearch/ImageBind). Then pass "
                    "--reward_args checkpoint_path=<imagebind_huge.pth>."
                ) from exc

        self._device = "cuda" if device is None else str(device)
        self._data = imagebind_data
        self._ModalityType = ModalityType
        # Higher-is-better cosine similarity (eps=1e-6).
        self._cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        model = imagebind_model.imagebind_huge(pretrained=True, ckpt_path=checkpoint_path)
        model.eval().to(self._device)
        self._model = model
        logger.info("imagebind: reward model loaded on %s (ckpt=%s)", self._device, checkpoint_path)

    def _score_one(self, video_file: str, audio_file: str, prompt_v: str, prompt_a: str) -> Tuple[float, float, float]:
        """Embed the three modalities for one clip and return (sim_tv, sim_ta, sim_av)."""
        import torch

        MT = self._ModalityType
        inputs = {
            MT.VISION: self._data.load_and_transform_video_data([video_file], self._device),
            MT.AUDIO: self._data.load_and_transform_audio_data([audio_file], self._device),
            # Two text rows: one for the video text, one for the audio text.
            MT.TEXT: self._data.load_and_transform_text([prompt_v, prompt_a], self._device),
        }
        with torch.no_grad():
            emb = self._model(inputs)

        text_v, text_a = emb[MT.TEXT][0:1], emb[MT.TEXT][1:2]
        video_e = emb[MT.VISION]  # [1, D]
        audio_e = emb[MT.AUDIO]  # [1, D]

        sim_tv = float(self._cos(text_v, video_e).item())
        sim_ta = float(self._cos(text_a, audio_e).item())
        sim_av = float(self._cos(audio_e, video_e).item())
        return sim_tv, sim_ta, sim_av

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        if self._model is None:
            raise RuntimeError("imagebind reward: setup() must run before score()")

        scores: List[float] = []
        sim_tv_all: List[float] = []
        sim_ta_all: List[float] = []
        sim_av_all: List[float] = []

        for sample in samples:
            video_file = sample.get("video_file")
            audio_file = sample.get("audio_file")
            prompt = sample.get("prompt", "") or ""
            # Optional per-sample overrides for the video / audio text (default: shared prompt).
            prompt_v = sample.get("prompt_v", prompt)
            prompt_a = sample.get("prompt_a", prompt)

            if not isinstance(video_file, str) or not isinstance(audio_file, str):
                logger.warning(
                    "imagebind: sample missing 'video_file'/'audio_file' (video=%r audio=%r); scoring 0.0",
                    video_file,
                    audio_file,
                )
                scores.append(0.0)
                sim_tv_all.append(0.0)
                sim_ta_all.append(0.0)
                sim_av_all.append(0.0)
                continue

            try:
                sim_tv, sim_ta, sim_av = self._score_one(video_file, audio_file, prompt_v, prompt_a)
            except Exception as exc:  # pragma: no cover - per-sample robustness
                logger.warning("imagebind: failed for video=%s audio=%s: %r", video_file, audio_file, exc)
                scores.append(0.0)
                sim_tv_all.append(0.0)
                sim_ta_all.append(0.0)
                sim_av_all.append(0.0)
                continue

            scores.append((sim_tv + sim_ta + sim_av) / 3.0)
            sim_tv_all.append(sim_tv)
            sim_ta_all.append(sim_ta)
            sim_av_all.append(sim_av)

        info = {
            "reward": "imagebind",
            "sim_tv": sim_tv_all,
            "sim_ta": sim_ta_all,
            "sim_av": sim_av_all,
        }
        return scores, info

    def teardown(self) -> None:
        if self._model is not None:
            try:
                import torch

                del self._model
                self._model = None
                self._data = None
                self._ModalityType = None
                self._cos = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort VRAM release
                self._model = None
