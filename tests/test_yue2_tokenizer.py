import base64
import json
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("tokenizers")

from yue2_fakes import make_tiny_tokenizer_json  # noqa: E402

from musubi_tuner.yue2.yue2_protocol import EOD  # noqa: E402
from musubi_tuner.yue2.yue2_tokenizer import YuE2TextTokenizer  # noqa: E402

TEXTS = [
    "hello darkness my old friend",
    "[verse]\nla la <|endoftext|> la\n",
    "<abc>X:1</abc> and <extra_0> literal",
    "école café Ａ",  # decomposed accent -> NFC
    "你好世界 さくら",
    "",
]


@pytest.fixture(scope="module")
def data():
    return make_tiny_tokenizer_json()


def _plain_tokenizer(data):
    """The same BPE without any added tokens: its regular encode is ordinary encoding by construction."""
    from tokenizers import Tokenizer

    spec = json.loads(data)
    spec["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(spec))


def test_json_backend_is_ordinary_encoding(data):
    from tokenizers import Tokenizer

    tok = YuE2TextTokenizer.from_json_bytes(data)
    plain = _plain_tokenizer(data)
    hf = Tokenizer.from_str(data.decode())
    added = {t.content: i for i, t in hf.get_added_tokens_decoder().items()}
    for text in TEXTS:
        ids = tok.encode(text)
        nfc = unicodedata.normalize("NFC", text)
        assert ids == plain.encode(nfc, add_special_tokens=False).ids, text
        assert all(0 <= i < EOD for i in ids)
        assert not set(ids) & set(added.values()), text
        assert tok.decode(ids) == nfc
    # the plain HF call does match added tokens, which is what ordinary encoding avoids
    assert added["<|endoftext|>"] in hf.encode("a <|endoftext|> b", add_special_tokens=False).ids
    assert tok.backend == "json" and tok.fingerprint.startswith("sha256:")
    assert tok.encode("é") == tok.encode("é")


def test_json_matches_tiktoken_reference(data, tmp_path):
    pytest.importorskip("tiktoken")
    from yue2_ref.tokenization_yue2 import YuE2TextTokenizer as RefTokenizer

    # a qwen.tiktoken-style ranks file needs 151643 ordinary ranks; build one from the byte-level BPE vocab plus filler
    spec = json.loads(data)
    byte_decoder = _byte_decoder()
    ranks = {}
    for token, rank in sorted(spec["model"]["vocab"].items(), key=lambda kv: kv[1]):
        ranks[bytes(byte_decoder[c] for c in token)] = rank
    filler = 0
    while len(ranks) < 151643:
        key = b"\xff\xfe" + filler.to_bytes(4, "big")
        filler += 1
        ranks.setdefault(key, len(ranks))
    path = tmp_path / "qwen.tiktoken"
    path.write_bytes(b"\n".join(base64.b64encode(k) + b" " + str(r).encode() for k, r in ranks.items()))
    ours = YuE2TextTokenizer.from_tiktoken(str(path))
    ref = RefTokenizer(str(path))
    for text in TEXTS:
        assert ours.encode(text) == ref.encode(text), text


def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
