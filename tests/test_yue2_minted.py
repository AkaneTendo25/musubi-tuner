import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import FakeTokenizer  # noqa: E402

from musubi_tuner.dataset.audio_dataset import validation_song_split  # noqa: E402
from musubi_tuner.dataset.yue2_minted import YuE2MintedPack, read_minted_records  # noqa: E402
from musubi_tuner.yue2 import yue2_protocol as protocol  # noqa: E402


def _kit(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    data = []
    for i in range(5):
        data.append(
            {
                "codec": rng.integers(0, protocol.CODEC_SIZE, size=50 + i, dtype=np.int32),
                "src": "minted_val" if i == 4 else "minted",
                "style": f"style {i}",
                "lyrics": "" if i == 0 else f"line {i}",
            }
        )
    path = tmp_path / "kit.pt"
    torch.save(data, path)
    return path


def test_pt_kit_pack_reads_splits_samples_and_tokenizes(tmp_path: Path):
    pack = YuE2MintedPack(str(_kit(tmp_path)))
    assert len(pack) == 5
    assert [r.codes.numel() for r in pack.records] == [50, 51, 52, 53, 54]
    assert all(r.codes.dtype == torch.int64 for r in pack.records)

    with pytest.raises(ValueError, match="tokenize"):
        pack.sample(random.Random(0))
    tok = FakeTokenizer()
    pack.tokenize(tok)
    item = pack.sample(random.Random(3))
    again = pack.sample(random.Random(3))
    assert torch.equal(item.codes, again.codes) and item.song_id == again.song_id
    assert set(item.text_ids) == set(protocol.COT_MODES)
    index = [r.song_id for r in pack.records].index(item.song_id)
    style, lyrics = protocol.normalize_prompt_fields(pack.records[index].style, pack.records[index].lyrics)
    assert item.text_ids["off"] == protocol.text_ids(tok, style, lyrics, "off")
    assert pack.item(0).text_ids["full"] == protocol.text_ids(tok, "style 0", protocol.DEFAULT_INSTRUMENTAL_LYRICS, "full")

    train, val = pack.split(0.5, seed=0)  # the kit names its split: minted_val wins over the fraction
    assert len(train) == 4 and len(val) == 1 and val.records[0].style == "style 4"
    assert val.text_ids is not None and val.sample(random.Random(0)).text_ids["off"][0] == protocol.EOD

    # draws over many seeds reach every record
    seen = {pack.sample(random.Random(s)).song_id for s in range(200)}
    assert len(seen) == 5


def test_manifest_and_jsonl_sources(tmp_path: Path):
    codes = np.arange(40, dtype=np.int64)
    np.save(tmp_path / "a.npy", codes)
    manifest = {
        "songs": [
            {"tokens": "a.npy", "true_tokens": True, "split": "train", "style": "s", "lyrics": "l", "song": "A"},
            {
                "tokens": str(tmp_path / "a.npy"),
                "true_tokens": True,
                "split": "validation",
                "style": "t",
                "lyrics": "m",
                "name": "B",
            },
        ]
    }
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    records = read_minted_records(str(tmp_path / "m.json"))
    assert [r.song_id for r in records] == ["A", "B"] and [r.is_validation for r in records] == [False, True]
    assert torch.equal(records[0].codes, torch.from_numpy(codes))
    manifest["songs"][0]["true_tokens"] = False
    (tmp_path / "bad.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="true_tokens is not set"):
        read_minted_records(str(tmp_path / "bad.json"))

    save_file({"codes": torch.arange(30, dtype=torch.int32)}, str(tmp_path / "b.safetensors"))
    lines = [
        {"codes": [1, 2, 3], "style": "x", "lyrics": "y", "song_id": "s1"},
        {"codec": "b.safetensors", "caption": "z", "song_id": "s2"},
        {"tokens": "a.npy", "style": "w", "lyrics": None},
    ]
    (tmp_path / "p.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    pack = YuE2MintedPack(str(tmp_path / "p.jsonl"))
    assert [r.codes.numel() for r in pack.records] == [3, 30, 40]
    assert [r.style for r in pack.records] == ["x", "z", "w"] and pack.records[2].lyrics == ""
    assert pack.records[2].song_id == "minted-000003"


def test_split_by_song_is_deterministic(tmp_path: Path):
    lines = [{"codes": [i, i + 1], "style": "x", "song_id": f"song{i % 6}"} for i in range(12)]
    (tmp_path / "p.jsonl").write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    pack = YuE2MintedPack(str(tmp_path / "p.jsonl"))
    train, val = pack.split(0.34, seed=5)
    held = validation_song_split([f"song{i}" for i in range(6)], 0.34, 5)
    assert {r.song_id for r in val.records} == held and len(held) == 2
    assert not {r.song_id for r in train.records} & held and len(train) + len(val) == 12
    train2, val2 = pack.split(0.34, seed=5)
    assert [r.codes.tolist() for r in val2.records] == [r.codes.tolist() for r in val.records]
    assert pack.split(0.0, seed=5)[1].records == []


def test_invalid_codes_and_tokenizer_pin(tmp_path: Path):
    for bad in ([protocol.CODEC_SIZE], [-1], [], [[1, 2]], [0.5]):
        (tmp_path / "p.jsonl").write_text(json.dumps({"codes": bad, "style": "x"}), encoding="utf-8")
        with pytest.raises(ValueError):
            read_minted_records(str(tmp_path / "p.jsonl"))
    with pytest.raises(ValueError, match="unsupported"):
        read_minted_records(str(tmp_path / "x.npz"))

    (tmp_path / "p.jsonl").write_text(json.dumps({"codes": [1, 2], "style": "x"}), encoding="utf-8")
    tok = FakeTokenizer()
    tok.fingerprint = "tok-a"
    pack = YuE2MintedPack(str(tmp_path / "p.jsonl"), tokenizer_fp="tok-b")
    with pytest.raises(ValueError, match="tokenizer"):
        pack.tokenize(tok)
    pack = YuE2MintedPack(str(tmp_path / "p.jsonl"), tokenizer_fp="tok-a")
    pack.tokenize(tok)
    assert pack.tokenizer_fp == "tok-a"

    fp = pack.fingerprint
    assert fp == YuE2MintedPack(str(tmp_path / "p.jsonl")).fingerprint
    (tmp_path / "p.jsonl").write_text(json.dumps({"codes": [1, 3], "style": "x"}), encoding="utf-8")
    assert YuE2MintedPack(str(tmp_path / "p.jsonl")).fingerprint != fp
