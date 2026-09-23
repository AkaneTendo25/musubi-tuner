"""YuE2 text/ABC BPE tokenizer (ordinary encoding only).

Two backends: ``tokenizers`` JSON (embedded in ComfyUI checkpoints or a ``.json`` file) and ``tiktoken`` (Hugging Face
``qwen.tiktoken``, lazy import). Both are equivalent to the official ``YuE2TextTokenizer.encode`` =
``encode_ordinary(NFC(text))``: special-token strings in the text (``<|endoftext|>``, ``<abc>``, ``<extra_N>``) are
byte-encoded, no special tokens are added, and every id is ``< EOD``.
"""

from __future__ import annotations

import base64
import hashlib
import unicodedata
from pathlib import Path

from musubi_tuner.yue2.yue2_protocol import EOD

TIKTOKEN_ORDINARY_RANKS = 151643
TIKTOKEN_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def _tiktoken_specials() -> list[str]:
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
    specials += [f"<extra_{i}>" for i in range(200)]
    specials[204:206] = ["<abc>", "</abc>"]
    return specials


class YuE2TextTokenizer:
    """Attributes: ``backend`` (``"json"`` or ``"tiktoken"``), ``fingerprint`` (``sha256:`` of the vocab source bytes)."""

    backend: str
    fingerprint: str

    def __init__(self, backend: str, impl, fingerprint: str):
        self.backend = backend
        self._impl = impl
        self.fingerprint = fingerprint

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "YuE2TextTokenizer":
        """JSON backend over ``tokenizers.Tokenizer.from_str``.

        Ordinary encoding runs the normalizer, the pre-tokenizer and the BPE model directly, so no added token (special
        or not) is ever matched in the text; this is ``encode_ordinary`` by construction.
        """
        from tokenizers import Tokenizer

        tok = Tokenizer.from_str(data.decode("utf-8"))
        if tok.pre_tokenizer is None:
            raise ValueError("YuE2 tokenizer JSON has no pre-tokenizer")
        return cls("json", tok, "sha256:" + hashlib.sha256(data).hexdigest())

    @classmethod
    def from_tiktoken(cls, path: str) -> "YuE2TextTokenizer":
        """tiktoken backend over ``qwen.tiktoken`` (151,643 ordinary ranks; the official specials table verbatim)."""
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError("the qwen.tiktoken YuE2 tokenizer needs `pip install tiktoken` (or pass a tokenizer .json)") from e
        data = Path(path).read_bytes()
        ranks = {base64.b64decode(t): int(r) for t, r in (line.split() for line in data.splitlines() if line)}
        if len(ranks) != TIKTOKEN_ORDINARY_RANKS:
            raise ValueError(
                f"expected checkpoint-native qwen.tiktoken ({TIKTOKEN_ORDINARY_RANKS} ordinary tokens), got {len(ranks)}"
            )
        specials = _tiktoken_specials()
        enc = tiktoken.Encoding(
            "YuE2",
            pat_str=TIKTOKEN_PATTERN,
            mergeable_ranks=ranks,
            special_tokens={s: i + len(ranks) for i, s in enumerate(specials)},
        )
        return cls("tiktoken", enc, "sha256:" + hashlib.sha256(data).hexdigest())

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        if self.backend == "tiktoken":
            ids = self._impl.encode_ordinary(text)
        else:
            tok = self._impl
            if tok.normalizer is not None:
                text = tok.normalizer.normalize_str(text)
            ids = [t.id for piece, _ in tok.pre_tokenizer.pre_tokenize_str(text) for t in tok.model.tokenize(piece)]
        bad = [i for i in ids if not 0 <= i < EOD]
        if bad:
            raise ValueError(f"YuE2 tokenizer produced ids outside the ordinary vocabulary: {bad[:8]}")
        return ids

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids]
        if self.backend == "tiktoken":
            return self._impl.decode([i for i in ids if 0 <= i < self._impl.n_vocab], errors="replace")
        size = self._impl.get_vocab_size(with_added_tokens=True)
        return self._impl.decode([i for i in ids if 0 <= i < size], skip_special_tokens=False)
