from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import FakeTokenizer  # noqa: E402

from musubi_tuner.dataset.bucket import BucketBatchManager  # noqa: E402
from musubi_tuner.dataset.cache_io import (  # noqa: E402
    YUE2_ABC_KEY,
    YUE2_ABC_MODE_KEY,
    YUE2_CODES_KEY,
    YUE2_HAS_ABC_KEY,
    YUE2_LATENT_CACHE_VERSION,
    YUE2_NEG_KEY_FMT,
    YUE2_SEG_KEY,
    YUE2_TEXT_CACHE_VERSION,
    YUE2_TEXT_KEY_FMT,
    save_latent_cache_yue2,
    save_text_encoder_output_cache_yue2,
    yue2_latent_key,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo  # noqa: E402
from musubi_tuner.yue2 import yue2_protocol as protocol  # noqa: E402
from musubi_tuner.yue2.cache_plan import item_record, latent_metadata_matches, plan_yue2_datasets  # noqa: E402


def _segment(tmp_path: Path, key: str = "song", frames: int = 40, start: int = 0, song_frames: int = 100, truncated=False):
    item = ItemInfo(key, "style", (frames, 64), (frames,), frame_count=frames)
    item.frame_pos = start
    item.song_start_frame = start
    item.song_frames = song_frames
    item.truncated = truncated
    item.latent_cache_path = str(tmp_path / f"{key}_{start:06d}-{frames:06d}_yue2.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / f"{key}_yue2_te.safetensors")
    return item


def _texts(tokenizer, style="pop", lyrics="la la"):
    texts = {c: torch.tensor(protocol.text_ids(tokenizer, style, lyrics, c)) for c in protocol.COT_MODES}
    negs = {c: torch.tensor(protocol.negative_text_ids(tokenizer, c)) for c in protocol.COT_MODES}
    return texts, negs


def _keys_and_meta(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, f.metadata()


def test_latent_cache_keys_segment_and_metadata(tmp_path: Path):
    item = _segment(tmp_path, frames=40, start=20, song_frames=100, truncated=True)
    latents = torch.randn(40, 64)
    codes = torch.randint(0, protocol.CODEC_SIZE, (40,), dtype=torch.int32)
    save_latent_cache_yue2(item, latents, codes, {"yue2_vae": "fp", "yue2_song_id": "s1"})
    sd, meta = _keys_and_meta(item.latent_cache_path)
    assert set(sd) == {"latents_40x64_float32", YUE2_CODES_KEY, YUE2_SEG_KEY}
    assert torch.equal(sd["latents_40x64_float32"], latents)
    assert sd[YUE2_CODES_KEY].dtype == torch.int64 and torch.equal(sd[YUE2_CODES_KEY], codes.long())
    assert sd[YUE2_SEG_KEY].tolist() == [20, 100, 1] and sd[YUE2_SEG_KEY].dtype == torch.int64
    assert meta["architecture"] == "yue2" and meta["yue2_cache_version"] == YUE2_LATENT_CACHE_VERSION
    assert (meta["yue2_start_frame"], meta["yue2_frames"], meta["yue2_song_frames"], meta["yue2_truncated"]) == (
        "20",
        "40",
        "100",
        "1",
    )
    assert meta["yue2_has_codes"] == "1" and meta["frame_count"] == "40"
    assert meta["yue2_vae"] == "fp" and meta["yue2_song_id"] == "s1"  # caller metadata is merged
    assert yue2_latent_key(40, torch.bfloat16) == "latents_40x64_bfloat16"

    # bf16, no codes
    save_latent_cache_yue2(item, latents, None, None, dtype=torch.bfloat16)
    sd, meta = _keys_and_meta(item.latent_cache_path)
    assert set(sd) == {"latents_40x64_bfloat16", YUE2_SEG_KEY} and meta["yue2_has_codes"] == "0"
    assert sd["latents_40x64_bfloat16"].dtype == torch.bfloat16


def test_latent_cache_rejects_inconsistent_inputs(tmp_path: Path):
    item = _segment(tmp_path, frames=40)
    with pytest.raises(ValueError, match="frames"):
        save_latent_cache_yue2(item, torch.zeros(39, 64), None)
    with pytest.raises(ValueError, match=r"\[T,64\]"):
        save_latent_cache_yue2(item, torch.zeros(40, 32), None)
    with pytest.raises(ValueError, match="codes"):
        save_latent_cache_yue2(item, torch.zeros(40, 64), torch.zeros(39, dtype=torch.int64))
    with pytest.raises(ValueError, match="codes"):
        save_latent_cache_yue2(item, torch.zeros(40, 64), torch.zeros(40))
    with pytest.raises(ValueError, match="codes must be in"):
        save_latent_cache_yue2(item, torch.zeros(40, 64), torch.full((40,), protocol.CODEC_SIZE))
    with pytest.raises(ValueError, match="contradicts"):
        save_latent_cache_yue2(item, torch.zeros(40, 64), None, {"yue2_frames": "41"})
    with pytest.raises(ValueError, match="floating"):
        save_latent_cache_yue2(item, torch.zeros(40, 64), None, dtype=torch.int64)
    outside = _segment(tmp_path, frames=40, start=70, song_frames=100)
    with pytest.raises(ValueError, match="outside the song"):
        save_latent_cache_yue2(outside, torch.zeros(40, 64), None)
    bare = ItemInfo("x", "", (40, 64), (40,), frame_count=40)
    bare.latent_cache_path = str(tmp_path / "x.safetensors")
    with pytest.raises(ValueError, match="provenance"):
        save_latent_cache_yue2(bare, torch.zeros(40, 64), None)
    assert not Path(item.latent_cache_path).exists()


def test_text_cache_keys_empty_abc_and_mode(tmp_path: Path):
    tok = FakeTokenizer()
    item = _segment(tmp_path)
    texts, negs = _texts(tok)
    save_text_encoder_output_cache_yue2(item, texts, negs, None, 0, {"yue2_tokenizer": "fake", "yue2_song_id": "s"})
    sd, meta = _keys_and_meta(item.text_encoder_output_cache_path)
    expected = {YUE2_TEXT_KEY_FMT.format(cot=c) for c in protocol.COT_MODES} | {
        YUE2_NEG_KEY_FMT.format(cot=c) for c in protocol.COT_MODES
    }
    expected |= {YUE2_ABC_KEY, YUE2_HAS_ABC_KEY, YUE2_ABC_MODE_KEY}
    assert set(sd) == expected
    assert "varlen_yue2_text_off_int64" in sd and "yue2_abc_mode_int64" in sd
    assert sd[YUE2_ABC_KEY].shape == (0,) and sd[YUE2_ABC_KEY].dtype == torch.int64
    assert sd[YUE2_HAS_ABC_KEY].item() == 0 and sd[YUE2_ABC_MODE_KEY].item() == 0 and sd[YUE2_ABC_MODE_KEY].shape == ()
    assert sd["varlen_yue2_text_full_int64"].tolist() == texts["full"].tolist()
    assert meta["yue2_text_cache_version"] == YUE2_TEXT_CACHE_VERSION and meta["yue2_tokenizer"] == "fake"
    assert meta["caption1"] == "style" and meta["architecture"] == "yue2"

    abc = torch.tensor(tok.encode("X:1\nK:C\nCDE|"))
    save_text_encoder_output_cache_yue2(item, texts, negs, abc, "full")
    sd, meta = _keys_and_meta(item.text_encoder_output_cache_path)
    assert sd[YUE2_HAS_ABC_KEY].item() == 1 and sd[YUE2_ABC_MODE_KEY].item() == 2
    assert torch.equal(sd[YUE2_ABC_KEY], abc) and "yue2_tokenizer" not in meta  # rewritten, not merged

    with pytest.raises(ValueError, match="does not match"):
        save_text_encoder_output_cache_yue2(item, texts, negs, abc, 0)
    with pytest.raises(ValueError, match="does not match"):
        save_text_encoder_output_cache_yue2(item, texts, negs, None, "melody")
    with pytest.raises(ValueError, match="cot modes"):
        save_text_encoder_output_cache_yue2(item, {"off": texts["off"]}, negs, None, 0)
    with pytest.raises(ValueError, match="ABC ids"):
        save_text_encoder_output_cache_yue2(item, texts, negs, torch.tensor([protocol.EOD]), 1)
    with pytest.raises(ValueError, match="abc_mode"):
        save_text_encoder_output_cache_yue2(item, texts, negs, None, "chords")


def test_batch_manager_collates_the_yue2_batch_contract(tmp_path: Path):
    tok = FakeTokenizer()
    items = []
    for i, (abc, mode) in enumerate([(None, None), ("X:1\nABC|", "melody")]):
        item = _segment(tmp_path, key=f"s{i}", frames=30, start=0, song_frames=30)
        save_latent_cache_yue2(item, torch.full((30, 64), float(i)), torch.arange(30))
        texts, negs = _texts(tok, style=f"style {i}", lyrics="x" * (i + 1))
        save_text_encoder_output_cache_yue2(item, texts, negs, torch.tensor(tok.encode(abc)) if abc else None, mode)
        items.append(item)

    batch = BucketBatchManager({(30,): items}, batch_size=2)[0]
    assert set(batch) >= {"latents", "codes", "yue2_seg", "yue2_text_off", "yue2_text_melody", "yue2_text_full"}
    assert set(batch) >= {"yue2_neg_off", "yue2_abc", "yue2_has_abc", "yue2_abc_mode", "timesteps"}
    assert batch["latents"].shape == (2, 30, 64) and batch["codes"].shape == (2, 30)
    assert batch["yue2_seg"].tolist() == [[0, 30, 0], [0, 30, 0]]
    assert batch["yue2_has_abc"].tolist() == [0, 1] and batch["yue2_abc_mode"].tolist() == [0, 1]
    assert [t.shape[0] for t in batch["yue2_abc"]] == [0, len(tok.encode("X:1\nABC|"))]
    lengths = [t.shape[0] for t in batch["yue2_text_off"]]
    assert lengths[1] == lengths[0] + 1  # varlen rows stay a list

    bf16 = _segment(tmp_path, key="bf", frames=30, song_frames=30)
    save_latent_cache_yue2(bf16, torch.ones(30, 64), None, dtype=torch.bfloat16)
    texts, negs = _texts(tok)
    save_text_encoder_output_cache_yue2(bf16, texts, negs, None, 0)
    batch = BucketBatchManager({(30,): [bf16]}, batch_size=1)[0]  # the dtype suffix never reaches the batch key
    assert batch["latents"].dtype == torch.bfloat16 and "codes" not in batch


def test_cache_plan_and_metadata_check(tmp_path: Path):
    import av

    from musubi_tuner.dataset.audio_dataset import AudioDataset
    from musubi_tuner.dataset.image_video_dataset import DatasetGroup

    d = tmp_path / "s"
    d.mkdir()
    for name in ("a", "b"):
        with av.open(str(d / f"{name}.wav"), mode="w") as container:
            stream = container.add_stream("pcm_s16le", rate=48000)
            stream.layout = "stereo"
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 2 * 48000), dtype=np.int16), format="s16", layout="stereo")
            frame.sample_rate = 48000
            for packet in list(stream.encode(frame)) + list(stream.encode()):
                container.mux(packet)
        (d / f"{name}.caption.txt").write_text(f"style {name}", encoding="utf-8")
    dataset = AudioDataset(audio_directory=str(d), cache_directory=str(tmp_path / "c"), architecture="yue2")
    DatasetGroup([dataset])
    plans = plan_yue2_datasets([dataset])
    assert plans[0].dataset_index == 0 and [r.item_key for r in plans[0].records] == ["a", "b"]
    items = [b[0] for b in dataset.retrieve_text_encoder_output_cache_batches(num_workers=1)]
    assert item_record(plans, items[1]).style == "style b"
    items[1].datasource_index = 0
    with pytest.raises(ValueError, match="does not match"):
        item_record(plans, items[1])
    with pytest.raises(ValueError, match="only audio datasets"):
        plan_yue2_datasets([object()])

    item = _segment(tmp_path)
    save_latent_cache_yue2(item, torch.zeros(40, 64), None, {"yue2_vae": "v1"})
    assert latent_metadata_matches(item.latent_cache_path, {"yue2_vae": "v1", "yue2_frames": "40"})
    assert not latent_metadata_matches(item.latent_cache_path, {"yue2_vae": "v2"})
    assert not latent_metadata_matches(tmp_path / "missing.safetensors", {})
