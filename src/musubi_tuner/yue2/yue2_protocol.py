"""Checkpoint-native YuE2 token protocol and generation defaults (single source of truth).

Mirrors the official ``yue2_infer`` 0.1.5 ``protocol.py`` (token ids, instruction texts, prefix layout, chunking).
Pure Python: no torch import, so dataset workers and argument parsing stay light. Tokenizers are duck-typed (any object with ``encode(str) -> list[int]``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import re
from typing import Any, Optional, Sequence

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START, LATENT_END, LATENT_PAD = 184621, 184622, 184623
VOCAB_SIZE, CONTEXT = 184704, 24576
PROTOCOL_VERSION = "yue2-native-v1"

SAMPLE_RATE = 48000
HOP = 1920  # waveform samples per latent/codec frame at 48 kHz
FRAME_RATE = 25
LATENT_DIM = 64
MERT_SAMPLE_RATE = 24000
MERT_HOP = 960  # MERT samples per frame at 24 kHz

COT_MODES = ("off", "melody", "full")
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}

STYLE_MAX_CHARS = 1500
DEFAULT_INSTRUMENTAL_LYRICS = "[instrumental]"


def _check_cot(cot: str) -> None:
    if cot not in INSTRUCTIONS:
        raise ValueError(f"cot must be one of {COT_MODES}, got {cot!r}")


def _as_int_list(ids: Sequence[Any], what: str) -> list[int]:
    out = []
    for token in ids:
        if isinstance(token, bool):
            raise ValueError(f"{what} must be integer token ids, got a bool")
        try:
            out.append(operator.index(token))
        except TypeError as e:
            raise ValueError(f"{what} must be integer token ids, got {type(token).__name__}") from e
    return out


def _ordinary_ids(ids: Sequence[Any], what: str) -> list[int]:
    out = _as_int_list(ids, what)
    if any(not 0 <= token < EOD for token in out):
        raise ValueError(f"{what} must remain inside the ordinary text vocabulary [0, {EOD})")
    return out


def prompt_text(style: str, lyrics: str, cot: str) -> str:
    _check_cot(cot)
    return f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"


def text_ids(tokenizer, style: str, lyrics: str, cot: str) -> list[int]:
    """``[EOD] + encode(prompt_text)``; the positive text part of a prefix."""
    return [EOD] + _as_int_list(tokenizer.encode(prompt_text(style, lyrics, cot)), "text ids")


def negative_text_ids(tokenizer, cot: str) -> list[int]:
    """``[EOD] + encode(instruction)``; the text part of the CFG negative prefix."""
    _check_cot(cot)
    return [EOD] + _as_int_list(tokenizer.encode(INSTRUCTIONS[cot]), "text ids")


def build_prefix(text: Sequence[int], cot: str, abc_ids: Optional[Sequence[int]]) -> list[int]:
    """Positive AR prefix from ``text_ids`` output.

    off: ``text + [ABC_START, ABC_END, MUSIC_START]`` (``abc_ids`` is ignored);
    melody/full with ABC: ``text + [ABC_START] + abc + [ABC_END, MUSIC_START]``;
    melody/full with ``abc_ids=None``: ``text + [ABC_START]`` (the AR writes the score itself).
    """
    _check_cot(cot)
    text = _as_int_list(text, "text ids")
    if cot == "off":
        return text + [ABC_START, ABC_END, MUSIC_START]
    if abc_ids is None:
        return text + [ABC_START]
    return text + [ABC_START] + _ordinary_ids(abc_ids, "ABC ids") + [ABC_END, MUSIC_START]


def build_negative_prefix(neg_text: Sequence[int], cot: str, abc_ids: Optional[Sequence[int]]) -> list[int]:
    """CFG negative prefix from ``negative_text_ids`` output.

    off: ``neg + [MUSIC_START]``; melody/full: ``neg + [ABC_START] + abc + [ABC_END, MUSIC_START]``, where ``abc``
    must be the exact positive-branch ABC ids (symbolic CFG of the official sampler).
    """
    _check_cot(cot)
    neg_text = _as_int_list(neg_text, "negative text ids")
    if cot == "off":
        return neg_text + [MUSIC_START]
    if abc_ids is None:
        raise ValueError("Symbolic CFG must retain the exact positive-branch ABC ids")
    return neg_text + [ABC_START] + _ordinary_ids(abc_ids, "ABC ids") + [ABC_END, MUSIC_START]


def max_nar_window(prefix_len: int, context: int = CONTEXT) -> int:
    """Largest NAR chunk (frames) that fits ``prefix + 2 * frames + 3`` positions into ``context``."""
    size = min((context - prefix_len - 3) // 2, CONTEXT)
    if size < 1:
        raise ValueError(f"YuE2 prefix of {prefix_len} tokens leaves no acoustic context (context {context})")
    return size


def chunk_ranges(frames: int, prefix_len: int, context: int = CONTEXT) -> list[tuple[int, int]]:
    """Half-open NAR chunk ranges over ``frames``, as the official ``protocol.chunk_ranges``."""
    size = min((context - prefix_len - 3) // 2, CONTEXT)
    if frames < 1 or size < 1:
        raise ValueError("YuE2 needs music tokens and enough context for at least one acoustic frame")
    return [(start, min(start + size, frames)) for start in range(0, frames, size)]


def _is_sequence(value) -> bool:
    return isinstance(value, (list, tuple, range))


def codec_to_ids(codes):
    """Codec indices ``[0, CODEC_SIZE)`` to AR token ids. Lists give lists; tensors/arrays give the same type."""
    if _is_sequence(codes):
        values = _as_int_list(codes, "codec indices")
        if any(not 0 <= c < CODEC_SIZE for c in values):
            raise ValueError(f"codec indices must be in [0, {CODEC_SIZE})")
        return [c + CODEC_OFFSET for c in values]
    count = codes.numel() if hasattr(codes, "numel") else codes.size
    if count and (int(codes.min()) < 0 or int(codes.max()) >= CODEC_SIZE):
        raise ValueError(f"codec indices must be in [0, {CODEC_SIZE})")
    return codes + CODEC_OFFSET


def ids_to_codec(ids):
    """Inverse of ``codec_to_ids``; raises on ids outside the codec range."""
    if _is_sequence(ids):
        values = _as_int_list(ids, "codec token ids")
        if any(not CODEC_OFFSET <= t < CODEC_OFFSET + CODEC_SIZE for t in values):
            raise ValueError(f"codec token ids must be in [{CODEC_OFFSET}, {CODEC_OFFSET + CODEC_SIZE})")
        return [t - CODEC_OFFSET for t in values]
    count = ids.numel() if hasattr(ids, "numel") else ids.size
    if count and (int(ids.min()) < CODEC_OFFSET or int(ids.max()) >= CODEC_OFFSET + CODEC_SIZE):
        raise ValueError(f"codec token ids must be in [{CODEC_OFFSET}, {CODEC_OFFSET + CODEC_SIZE})")
    return ids - CODEC_OFFSET


@dataclass(frozen=True)
class SamplingParams:
    """AR sampling parameters; field order and validation follow the official ``protocol.Sampling``."""

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    penalty_window: int = 50
    min_tokens: int = 200
    max_tokens: int = 9000

    def __post_init__(self):
        if any(type(x) is not int for x in (self.top_k, self.penalty_window, self.min_tokens, self.max_tokens)):
            raise ValueError("Sampling counts must be integers")
        if not all(math.isfinite(x) for x in (self.temperature, self.top_p, self.repetition_penalty)):
            raise ValueError("Sampling numbers must be finite")
        if not 0 <= self.temperature <= 5 or not 0 < self.top_p <= 1 or self.top_k < 1:
            raise ValueError("Invalid sampling temperature/top_p/top_k")
        if self.repetition_penalty <= 0 or not 1 <= self.penalty_window <= 100:
            raise ValueError("Invalid repetition penalty/window")
        if not 0 <= self.min_tokens <= self.max_tokens or self.max_tokens < 1:
            raise ValueError("Require 0 <= min_tokens <= max_tokens")


SEMANTIC_DEFAULTS = SamplingParams(1.0, 0.95, 100, 1.2, 50, 200, 9000)
ABC_DEFAULTS = SamplingParams(0.7, 0.9, 30, 1.005, 100, 32, 4096)
ODE_STEPS = 32


def default_cfg(cot: str) -> float:
    """Protocol CFG scale: 1.01 for ``off``, 1.0 (no CFG) otherwise."""
    _check_cot(cot)
    return 1.01 if cot == "off" else 1.0


def normalize_prompt_fields(
    style: Optional[str], lyrics: Optional[str], instrumental_lyrics: str = DEFAULT_INSTRUMENTAL_LYRICS
) -> tuple[str, str]:
    """Whitespace-normalised style truncated to 1500 chars; stripped lyrics, empty -> ``instrumental_lyrics``.

    Matches the preprocessing the default Mothersuperior semantic head and adapters were trained with; the official
    convention is unknown. Line endings of lyrics are normalised to ``\\n``.
    """
    style = " ".join((style or "").split())[:STYLE_MAX_CHARS]
    lyrics = (lyrics or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not lyrics:
        lyrics = instrumental_lyrics
    return style, lyrics


_ABC_QUOTED = re.compile(r'"([^"\n]*)"')


def abc_mode_of(abc_text: str) -> str:
    """ABC flavour heuristic: ``full`` when quoted chord symbols are present, else ``melody``.

    Quoted strings starting with ``^ _ < > @`` are ABC annotations, not chord symbols.
    """
    for quoted in _ABC_QUOTED.findall(abc_text or ""):
        if quoted.strip() and quoted[0] not in "^_<>@":
            return "full"
    return "melody"


def generation_budget(prefix_len: int, negative_len: int, requested: int, context: int = CONTEXT) -> int:
    """AR token budget: ``requested``, limited by the context left after the longer CFG prefix.

    Raises when the positive prefix leaves no room for even a one-frame NAR chunk (see ``max_nar_window``)."""
    if requested < 1:
        raise ValueError(f"requested token count must be positive, got {requested}")
    room = context - max(prefix_len, negative_len)
    if room < 1 or (context - prefix_len - 3) // 2 < 1:
        raise ValueError("YuE2 prompt leaves no room for music; shorten the style, lyrics, or ABC")
    return min(requested, room)


def frames_for_samples(samples: int) -> int:
    return samples // HOP


def yue2_samples_per_crop(frames: int) -> int:
    # module-level (not a lambda) so the audio spec stays picklable for spawned DataLoader workers
    if frames <= 0:
        raise ValueError(f"YuE2 frame count must be positive, got {frames}")
    return frames * HOP


def __getattr__(name: str):
    # YUE2_AUDIO_SPEC is built lazily: AudioSpec lives in dataset.audio_utils, which imports torch and av
    if name == "YUE2_AUDIO_SPEC":
        from musubi_tuner.dataset.audio_utils import AudioSpec

        spec = AudioSpec(sample_rate=SAMPLE_RATE, channels=2, samples_per_crop=yue2_samples_per_crop, codec_pad_tolerance=HOP)
        globals()[name] = spec
        return spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
