"""Audiobox-Aesthetics audio reward — Meta/FAIR no-reference audio quality (route=audio).

Wraps ``audiobox_aesthetics`` (Meta's "Unified automatic quality assessment for
speech, music, and sound"). Its predictor emits four aesthetic axes per clip —
CE (content enjoyment), CU (content usefulness), PC (production complexity),
PQ (production quality) — and we collapse them to the scalar reward

    score = (CE + CU + PQ - PC) / 40      # higher = better

i.e. reward enjoyable, useful, high-quality audio while *penalising* production
complexity (PC enters with a minus sign). The /40 keeps the scalar in a
small range (each axis is ~[1,10]).

VENDORING DECISION — VENDORED (inference-only), with a pip fallback.
    The inference path (``infer.py`` + ``model/aes.py`` + ``model/utils.py`` anchored on
    ``model/wavlm.py``, a ~59 KB WavLM transformer encoder) is fully self-contained: its
    only deps are torch / torchaudio / numpy. So we vendor it verbatim into
    ``musubi_tuner.ltx2_rewards.vendor.audiobox`` (training / CLI / submitit / rich /
    tqdm / huggingface_hub-download surface stripped) and import it first, falling back
    to the external ``audiobox_aesthetics`` PyPI package on ImportError. This mirrors the
    vendored-first + external fallback in ``zoo/hpsv3.py`` and removes the hard runtime
    dependency on the pip package (which also pulled ``submitit`` and ``rich``).

    Scores are bit-identical to the pip package: the WavLM/aes/utils architecture and
    scoring path are copied verbatim, the same local Lightning-style ``checkpoint.pt``
    (state_dict + model_cfg + target_transform) is loaded via ``load_state_dict``, and
    the only structural change (dropping the unused ``PyTorchModelHubMixin`` base from
    ``AesMultiOutput``) does not touch any weight or the forward pass.

    Optional fallback install: ``pip install audiobox_aesthetics==0.0.4``.

VRAM: the predictor (WavLM ~95M params) loads only during offline audio scoring and is
torn down before the next reward / the training step (``RewardStack`` sequencing), so it
never co-resides with the DiT or the NFT update.

Inputs (``needs={"audio_waveform"}``): each sample provides ``audio_waveform``, which may be
  - a 16 kHz (or any-rate, resampled internally) mono/stereo waveform tensor/ndarray
    shaped ``[C, T]`` or ``[T]`` (preferred — in-process, no temp files), optionally with a
    per-sample ``sample_rate`` (default 16000); or
  - a path string to a ``.wav`` (or any torchaudio-decodable) file.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from ..registry import BaseReward, register_reward

logger = logging.getLogger(__name__)

# Default on-box checkpoint (workbox OmniNFT-Reward-Series mirror); override via
# --reward_args checkpoint_path=... . The vendored predictor requires a local
# checkpoint; if it is None *and* the external pip package is in use, that package
# falls back to HF ``facebook/audiobox-aesthetics`` (needs network / HF cache).
_DEFAULT_CHECKPOINT = "/mnt/data2/sgornostaev/models/OmniNFT-Reward-Series/audiobox-aesthetics/checkpoint.pt"

# Axis keys emitted by the predictor (audiobox_aesthetics.infer.AXES_NAME).
_AXES = ("CE", "CU", "PC", "PQ")


def _aesthetics_to_reward(out: dict) -> float:
    """Collapse the 4 aesthetic axes to the scalar: (CE + CU + PQ - PC) / 40.

    Missing axes default to 0.0 (``out.get(axis, 0.0)``). Higher = better.
    """
    ce = float(out.get("CE", 0.0))
    cu = float(out.get("CU", 0.0))
    pc = float(out.get("PC", 0.0))
    pq = float(out.get("PQ", 0.0))
    return (ce + cu + pq - pc) / 40.0


def _to_predictor_item(audio, sample_rate: int) -> dict:
    """Build one ``predictor.forward`` batch item from a sample's ``audio_waveform``.

    Accepts a path string, or a waveform tensor/ndarray (``[C,T]`` or ``[T]``). The
    audiobox predictor's ``audio_resample_mono`` reads ``item["path"]`` (str -> torchaudio
    load) or treats ``item["path"]`` as a waveform with ``item["sample_rate"]``, then
    resamples to 16 kHz and averages to mono — so we hand it the right shape and let it
    do the resample/mono-mixdown (no extra ffmpeg step needed for in-process waveforms).
    """
    if isinstance(audio, str):
        return {"path": audio, "sample_rate": sample_rate}

    import torch

    if hasattr(audio, "detach"):  # torch.Tensor
        wav = audio.detach().to("cpu").float()
    else:  # numpy / list
        wav = torch.as_tensor(audio, dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # [T] -> [1, T]
    elif wav.ndim > 2:
        raise ValueError(f"audiobox: expected waveform [C,T] or [T], got shape {tuple(wav.shape)}")
    return {"path": wav, "sample_rate": int(sample_rate)}


@register_reward("audiobox")
class AudioboxReward(BaseReward):
    """Audiobox-Aesthetics no-reference audio quality reward (route=audio)."""

    kind = "blackbox"
    route = "audio"
    needs = frozenset({"audio_waveform"})

    def __init__(self) -> None:
        self._predictor = None
        self._sample_rate = 16000

    def setup(
        self,
        device,
        *,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        sample_rate=16000,
        **_ignored,
    ) -> None:
        self._sample_rate = int(sample_rate)
        try:
            # Prefer the in-repo inference-only vendored copy (no submitit / rich / tqdm /
            # huggingface_hub). Deps: torch / torchaudio / numpy only.
            from ..vendor.audiobox import initialize_predictor
        except ImportError:
            try:
                # Fall back to the external 'audiobox_aesthetics' package if installed.
                from audiobox_aesthetics.infer import initialize_predictor
            except ImportError as exc:  # pragma: no cover - exercised only on the GPU box
                raise ImportError(
                    "audiobox reward requires the vendored copy "
                    "(musubi_tuner.ltx2_rewards.vendor.audiobox) or the external "
                    "'audiobox_aesthetics' package (pip install audiobox_aesthetics==0.0.4). "
                    "Then pass --reward_args checkpoint_path=<checkpoint.pt> (or leave it to "
                    "use the default on-box checkpoint)."
                ) from exc

        # checkpoint_path=None -> package loads facebook/audiobox-aesthetics from HF.
        ckpt = checkpoint_path or None
        self._predictor = initialize_predictor(ckpt=ckpt)
        # initialize_predictor() picks cuda/mps/cpu itself; honour an explicit device.
        if device is not None:
            import torch

            dev = torch.device(str(device))
            self._predictor.model = self._predictor.model.to(dev)
            self._predictor.device = dev
        logger.info(
            "audiobox: predictor loaded on %s (ckpt=%s sr=%d)",
            getattr(self._predictor, "device", device),
            ckpt or "HF:facebook/audiobox-aesthetics",
            self._sample_rate,
        )

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        if self._predictor is None:
            raise RuntimeError("audiobox reward: setup() must run before score()")
        import torch

        scores: List[float] = []
        with torch.no_grad():
            for sample in samples:
                audio = sample.get("audio_waveform")
                if audio is None:
                    logger.warning("audiobox: sample has no 'audio_waveform'; scoring 0.0")
                    scores.append(0.0)
                    continue
                sr = int(sample.get("sample_rate", self._sample_rate))
                item = _to_predictor_item(audio, sr)
                outputs = self._predictor.forward([item])  # list[dict], len == 1
                out = outputs[0] if outputs else {}
                scores.append(_aesthetics_to_reward(out or {}))
        return scores, {"reward": "audiobox", "axes": list(_AXES)}

    def teardown(self) -> None:
        if self._predictor is not None:
            try:
                import torch

                del self._predictor
                self._predictor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort VRAM release
                self._predictor = None
