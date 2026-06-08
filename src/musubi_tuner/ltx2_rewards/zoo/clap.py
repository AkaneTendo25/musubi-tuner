from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ..registry import BaseReward, register_reward

logger = logging.getLogger(__name__)

_CLAP_SAMPLE_RATE = 48000
_DEFAULT_CHECKPOINT = "laion/clap-htsat-unfused"


def _waveform_to_48k_mono(waveform, source_sr: int, max_length_s: Optional[float]):
    """Resample a decoded waveform to a 48 kHz mono 1-D numpy array."""
    import torch
    import torchaudio

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    wav = waveform.detach().to("cpu").float()

    # Collapse leading singleton (batch) dims only — squeeze(0) on a non-1 dim is a no-op,
    # so loop on the size, not on squeeze, to avoid spinning forever on a bad shape.
    while wav.dim() > 2 and wav.shape[0] == 1:
        wav = wav.squeeze(0)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() != 2:
        raise ValueError(f"clap: expected decoded audio [C,T] or [T], got shape {tuple(waveform.shape)}")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if source_sr != _CLAP_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, source_sr, _CLAP_SAMPLE_RATE)

    if max_length_s is not None:
        max_len = int(float(max_length_s) * _CLAP_SAMPLE_RATE)
        wav = wav[:, :max_len]

    return wav.squeeze(0).numpy()


@register_reward("clap")
class ClapReward(BaseReward):
    """CLAP audio-text cosine-similarity reward over decoded audio (route=audio)."""

    kind = "blackbox"
    route = "audio"
    needs = frozenset({"audio_waveform", "prompt"})

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._sample_rate = 24000
        self._max_length_s: Optional[float] = None

    def setup(
        self,
        device,
        *,
        checkpoint_path: str = None,
        sample_rate=24000,
        max_length_s=None,
        **_ignored,
    ) -> None:
        import torch
        from transformers import AutoProcessor, ClapModel

        self._sample_rate = int(sample_rate)
        self._max_length_s = None if max_length_s is None else float(max_length_s)
        ckpt = checkpoint_path or _DEFAULT_CHECKPOINT
        self._device = "cuda" if device is None else str(device)

        self._model = ClapModel.from_pretrained(ckpt).eval().to(device=self._device)
        self._processor = AutoProcessor.from_pretrained(ckpt)
        self._cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        logger.info(
            "clap: reward model loaded on %s (ckpt=%s, in_sr=%d, max_length_s=%s)",
            self._device,
            ckpt,
            self._sample_rate,
            self._max_length_s,
        )

    def _score_one(self, waveform, prompt: str) -> float:
        """Cosine(text, audio) remapped to [0,1] for one (waveform, prompt) pair."""
        import numpy as np
        import torch

        audio_arr = _waveform_to_48k_mono(waveform, self._sample_rate, self._max_length_s)

        inputs = self._processor(
            text=prompt,
            audios=audio_arr,
            return_tensors="pt",
            padding=True,
            truncation=True,
            sampling_rate=_CLAP_SAMPLE_RATE,
        )
        inputs = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        s = self._cos(outputs.text_embeds, outputs.audio_embeds).mean().item()
        if np.isnan(s):
            s = 0.0

        s = (float(s) + 1.0) / 2.0
        return max(0.0, min(1.0, s))

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        if self._model is None:
            raise RuntimeError("clap reward: setup() must run before score()")
        import torch

        scores: List[float] = []
        with torch.no_grad():
            for sample in samples:
                waveform = sample.get("audio_waveform")
                prompt = sample.get("prompt", "")
                if waveform is None:
                    logger.warning("clap: sample has no decoded audio_waveform; scoring 0.0")
                    scores.append(0.0)
                    continue
                if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                    logger.warning("clap: sample has empty prompt; scoring 0.0")
                    scores.append(0.0)
                    continue
                try:
                    scores.append(self._score_one(waveform, prompt))
                except Exception as exc:
                    logger.warning("clap: scoring failed (%r); scoring 0.0", exc)
                    scores.append(0.0)
        return scores, {"reward": "clap", "sample_rate": self._sample_rate}

    def teardown(self) -> None:
        if self._model is not None:
            try:
                import torch

                del self._model
                self._model = None
                self._processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                self._model = None
                self._processor = None
