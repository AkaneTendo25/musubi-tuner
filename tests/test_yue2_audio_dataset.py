import argparse
import csv
import json
import logging
import os
from pathlib import Path
import re
import sys

import av
import numpy as np
import pytest
import toml
import torch
from voluptuous import MultipleInvalid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import FakeTokenizer  # noqa: E402

from musubi_tuner.dataset.audio_dataset import (  # noqa: E402
    AudioDataset,
    segment_windows,
    validation_song_split,
)
from musubi_tuner.dataset.audio_utils import AudioSource, decode_audio  # noqa: E402
from musubi_tuner.dataset.cache_io import save_latent_cache_yue2, save_text_encoder_output_cache_yue2  # noqa: E402
from musubi_tuner.dataset.config_utils import (  # noqa: E402
    AudioDatasetParams,
    BaseDatasetParams,
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from musubi_tuner.dataset.datasources import parse_yue2_caption, split_abc_mode_header  # noqa: E402
from musubi_tuner.yue2 import yue2_protocol as protocol  # noqa: E402
from musubi_tuner.yue2.yue2_protocol import YUE2_AUDIO_SPEC  # noqa: E402

HOP = 1920
SR = 48000
ARCH = "yue2"


def _tone(seconds: float, sample_rate: int, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    return np.stack([0.5 * np.sin(2 * np.pi * freq * t), 0.25 * np.sin(2 * np.pi * 1.5 * freq * t)]).astype(np.float32)


def _write_audio(path: Path, seconds: float, sample_rate: int = SR, freq: float = 440.0) -> Path:
    samples = (np.clip(_tone(seconds, sample_rate, freq), -1, 1) * 32767.0).astype(np.int16)
    codec = "flac" if path.suffix == ".flac" else "pcm_s16le"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=sample_rate)
        stream.layout = "stereo"
        for start in range(0, samples.shape[1], 4096):
            chunk = samples[:, start : start + 4096]
            interleaved = np.empty((1, chunk.shape[1] * 2), dtype=np.int16)
            interleaved[0, 0::2] = chunk[0]
            interleaved[0, 1::2] = chunk[1]
            frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout="stereo")
            frame.sample_rate = sample_rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _decoded_frames(path: Path) -> int:
    return decode_audio(AudioSource(path.resolve(), False), sample_rate=SR, channels=2).shape[1] // HOP


def _blueprint(tmp_path: Path, datasets: list[dict], general: dict | None = None, namespace=None):
    config_path = tmp_path / "dataset.toml"
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump({"general": general or {}, "datasets": datasets}, f)
    user_config = load_user_config(str(config_path))
    namespace = namespace if namespace is not None else argparse.Namespace(debug_dataset=False)
    return BlueprintGenerator(ConfigSanitizer()).generate(user_config, namespace, architecture=ARCH)


def _group(tmp_path: Path, datasets: list[dict], general: dict | None = None, training: bool = False, audio_spec=YUE2_AUDIO_SPEC):
    blueprint = _blueprint(tmp_path, datasets, general)
    return generate_dataset_group_by_blueprint(blueprint.dataset_group, training=training, audio_spec=audio_spec)


def _items(dataset: AudioDataset) -> list:
    batches = list(dataset.retrieve_latent_cache_batches(num_workers=2))
    for key, batch in batches:
        assert key == (batch[0].frame_count,)
        assert len({item.datasource_index for item in batch}) == 1  # a batch never mixes records
        assert len(batch) <= dataset.batch_size
    return sorted((item for _, batch in batches for item in batch), key=lambda i: (i.datasource_index, i.frame_pos))


def _write_fake_caches(dataset: AudioDataset, tokenizer=None) -> None:
    """Writes latent/text caches the way the cache scripts do (zero-cost fake encoders)."""
    tokenizer = tokenizer or FakeTokenizer()
    for item in _items(dataset):
        latents = torch.arange(item.frame_count * 64, dtype=torch.float32).reshape(item.frame_count, 64) / 1000.0
        codes = torch.arange(item.frame_count, dtype=torch.int64) % protocol.CODEC_SIZE
        save_latent_cache_yue2(item, latents, codes, {"yue2_song_id": item.song_id})
    for batch in dataset.retrieve_text_encoder_output_cache_batches(num_workers=1):
        for item in batch:
            style, lyrics = protocol.normalize_prompt_fields(item.caption, item.lyrics)
            texts = {c: torch.tensor(protocol.text_ids(tokenizer, style, lyrics, c)) for c in protocol.COT_MODES}
            negs = {c: torch.tensor(protocol.negative_text_ids(tokenizer, c)) for c in protocol.COT_MODES}
            abc = torch.tensor(tokenizer.encode(item.abc_text)) if item.abc_text else None
            save_text_encoder_output_cache_yue2(item, texts, negs, abc, item.abc_mode, {"yue2_song_id": item.song_id})


@pytest.fixture()
def song_dir(tmp_path: Path) -> Path:
    d = tmp_path / "songs"
    d.mkdir()
    _write_audio(d / "alpha.wav", 7.0)
    _write(d / "alpha.caption.txt", "dream pop, airy vocals\n")
    _write(d / "alpha.lyrics.txt", "[Verse 1]\nhello there\n")
    _write_audio(d / "beta.flac", 6.0, sample_rate=44100, freq=330.0)
    _write(d / "beta.caption.txt", '{"style": "lo-fi hip hop", "lyrics": "[Chorus]\\nla la"}')
    _write(d / "beta.abc.txt", 'X:1\nK:C\n"Am"A2 B2|')
    return d


# --- config / blueprint -------------------------------------------------------------------------------------------


def test_toml_blueprint_builds_audio_dataset_for_caching(tmp_path: Path, song_dir: Path, caplog):
    caplog.set_level(logging.INFO)
    blueprint = _blueprint(tmp_path, [{"audio_directory": str(song_dir), "cache_directory": str(tmp_path / "cache")}])
    ds_blueprint = blueprint.dataset_group.datasets[0]
    assert ds_blueprint.is_audio_dataset and not ds_blueprint.is_image_dataset
    assert isinstance(ds_blueprint.params, AudioDatasetParams)

    group = generate_dataset_group_by_blueprint(blueprint.dataset_group, training=False, audio_spec=YUE2_AUDIO_SPEC)
    dataset = group.datasets[0]
    assert isinstance(dataset, AudioDataset)
    assert dataset.caption_extension == ".caption.txt"
    assert dataset.num_train_items == 0 and group.num_train_items == 0
    assert dataset.batch_manager is None and len(group) == 100  # the VideoDataset convention before prepare_for_training
    assert dataset.dataset_index == 0
    log = caplog.text
    assert "is_audio_dataset: True" in log and "segment_extraction: full" in log and "max_seconds: 360.0" in log
    assert "video_directory" not in log


def test_unknown_or_foreign_keys_are_rejected(tmp_path: Path, song_dir: Path):
    with pytest.raises(MultipleInvalid):
        _blueprint(tmp_path, [{"audio_directory": str(song_dir), "target_frames": [1]}])
    with pytest.raises(MultipleInvalid):
        _blueprint(tmp_path, [{"audio_directory": str(song_dir), "no_such_key": 1}])
    with pytest.raises(MultipleInvalid):
        _blueprint(tmp_path, [{"audio_directory": str(song_dir), "segment_extraction": "uniform"}])
    with pytest.raises(MultipleInvalid):  # audio-only keys are not accepted by image datasets
        _blueprint(tmp_path, [{"image_directory": str(song_dir), "trigger": "x"}])
    # [general] may carry the audio ascendable keys; image/video datasets ignore them
    blueprint = _blueprint(
        tmp_path, [{"audio_directory": str(song_dir)}], general={"min_seconds": 2, "trigger": "sv", "lyrics_extension": ".lyr"}
    )
    params = blueprint.dataset_group.datasets[0].params
    assert params.min_seconds == 2 and params.trigger == "sv" and params.lyrics_extension == ".lyr"


def _collect_parsers() -> dict[str, argparse.ArgumentParser]:
    from musubi_tuner import cache_latents, cache_text_encoder_outputs
    from musubi_tuner.training.parser_common import setup_parser_common
    from musubi_tuner.yue2.yue2_args import setup_parser_yue2_model

    parsers = {
        "train": setup_parser_yue2_model(setup_parser_common(), training=True),
        "cache_latents": setup_parser_yue2_model(cache_latents.setup_parser_common(), training=False),
        "cache_text": setup_parser_yue2_model(cache_text_encoder_outputs.setup_parser_common(), training=False),
    }
    # the YuE2 entry points of the other work packages, when they exist
    import importlib

    for module_name in (
        "musubi_tuner.yue2_train_network",
        "musubi_tuner.yue2_cache_latents",
        "musubi_tuner.yue2_cache_text_encoder_outputs",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in dir(module):
            fn = getattr(module, name)
            if callable(fn) and re.fullmatch(r"(yue2_)?setup_parser(_yue2)?", name):
                for base in parsers.values():
                    try:
                        fn(base)
                    except Exception:
                        pass
    return parsers


AUDIO_ONLY_FIELDS = sorted(set(AudioDatasetParams.__dataclass_fields__) - set(BaseDatasetParams.__dataclass_fields__))


def test_audio_dataset_keys_do_not_collide_with_cli_flags(tmp_path: Path, song_dir: Path):
    # BlueprintGenerator falls back to the whole argparse namespace, so a CLI flag named like an audio
    # dataset key would silently become a dataset parameter
    parsers = _collect_parsers()
    for name, parser in parsers.items():
        dests = {action.dest for action in parser._actions}
        assert not dests & set(AUDIO_ONLY_FIELDS), f"{name} parser defines dataset keys {sorted(dests & set(AUDIO_ONLY_FIELDS))}"

        namespace, _ = parser.parse_known_args(["--dataset_config", str(tmp_path / "dataset.toml")])
        params = _blueprint(tmp_path, [{"audio_directory": str(song_dir)}], namespace=namespace).dataset_group.datasets[0].params
        defaults = AudioDatasetParams()
        for field in AUDIO_ONLY_FIELDS:
            if field in ("audio_directory",):
                continue
            assert getattr(params, field) == getattr(defaults, field), f"{name}: {field} picked up a CLI value"

    # source scan: no YuE2 entry point may add a flag with a dataset key name, whatever its parser API
    sources = list((ROOT / "src" / "musubi_tuner").glob("yue2*.py")) + list((ROOT / "src" / "musubi_tuner" / "yue2").glob("*.py"))
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for field in AUDIO_ONLY_FIELDS:
            assert f'"--{field}"' not in text, f"{source.name} defines --{field}"


# --- captions, records ---------------------------------------------------------------------------------------------


def test_parse_yue2_caption_formats():
    assert parse_yue2_caption("  indie rock  ", "auto") == ("indie rock", None, None)
    assert parse_yue2_caption('{"tags": "jazz", "lyrics": " la ", "abc": "X:1"}', "auto") == ("jazz", "la", "X:1")
    assert parse_yue2_caption('{"caption": "jazz"}', "json") == ("jazz", None, None)
    text = "[Tags]\nsynthwave, male vocals\n[Lyrics]\n[Verse 1]\nline one\n[Duration]\n30\n"
    assert parse_yue2_caption(text, "auto") == ("synthwave, male vocals", "[Verse 1]\nline one", None)
    assert parse_yue2_caption("style only\n[ABC]\nX:1\nK:C", "tags_lyrics") == ("style only", None, "X:1\nK:C")
    assert parse_yue2_caption("<CAPTION>pop</CAPTION><LYRICS>hey</LYRICS>", "auto") == ("pop", "hey", None)
    # plain keeps section-looking text as style; json refuses non-objects
    assert parse_yue2_caption("[Lyrics]\nabc", "plain") == ("[Lyrics]\nabc", None, None)
    with pytest.raises(ValueError):
        parse_yue2_caption("[1, 2]", "json")
    with pytest.raises(ValueError):
        parse_yue2_caption("x", "yaml")
    assert parse_yue2_caption("[Tags]\npop\n[Lyrics]\n", "auto") == ("pop", "", None)  # explicit empty = instrumental


def test_directory_records_sidecars_trigger_and_abc_modes(tmp_path: Path, song_dir: Path):
    _write(song_dir / "alpha.song.txt", "song-A\n")
    _write_audio(song_dir / "gamma.wav", 6.0)
    _write(song_dir / "gamma.caption.txt", "[Tags]\nfolk\n[Lyrics]\nfrom caption\n")
    _write(song_dir / "gamma.lyrics.txt", "from sidecar")
    _write(song_dir / "gamma.abc.txt", "%%yue2_abc_mode full\nX:1\nK:G\nGABc|")
    _write_audio(song_dir / "delta.wav", 6.0)
    _write(song_dir / "delta.abc.txt", "X:1\nK:D\nDEF|")

    dataset = AudioDataset(audio_directory=str(song_dir), trigger="sv_artist", cache_directory=str(tmp_path / "c"))
    records = {r.item_key: r for r in dataset.datasource.records}
    assert list(records) == ["alpha", "beta", "delta", "gamma"]  # sorted by name
    assert records["alpha"].style == "sv_artist, dream pop, airy vocals"
    assert records["alpha"].lyrics == "[Verse 1]\nhello there" and records["alpha"].song_id == "song-A"
    assert records["alpha"].abc is None and records["alpha"].abc_mode is None
    assert records["beta"].style == "sv_artist, lo-fi hip hop" and records["beta"].lyrics == "[Chorus]\nla la"
    assert records["beta"].abc_mode == "full" and records["beta"].song_id == "beta"  # quoted chord -> full
    assert records["gamma"].lyrics == "from caption"  # caption sections win over sidecars
    assert records["gamma"].abc == "X:1\nK:G\nGABc|" and records["gamma"].abc_mode == "full"  # header, stripped
    assert records["delta"].style == "sv_artist" and records["delta"].lyrics is None  # missing sidecars
    assert records["delta"].abc_mode == "melody"  # heuristic: no chord symbols

    # dataset-wide abc_mode applies where neither the record nor a header names one
    dataset = AudioDataset(audio_directory=str(song_dir), abc_mode="full", cache_directory=str(tmp_path / "c"))
    assert {r.item_key: r.abc_mode for r in dataset.datasource.records} == {
        "alpha": None,
        "beta": "full",
        "delta": "full",
        "gamma": "full",
    }
    assert split_abc_mode_header("%%yue2_abc_mode melody\nX:1") == ("X:1", "melody")
    with pytest.raises(ValueError):
        split_abc_mode_header("%%yue2_abc_mode chords\nX:1")

    _write_audio(song_dir / "ALPHA.flac", 6.0)
    with pytest.raises(ValueError, match="unique stems"):
        AudioDataset(audio_directory=str(song_dir), cache_directory=str(tmp_path / "c"))


def test_jsonl_records_paths_duplicate_stems_and_abc_mode(tmp_path: Path, song_dir: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jsonl = song_dir / "list.jsonl"
    lines = [
        {"audio_path": "alpha.wav", "caption": "pop", "lyrics": "a", "start": 0.0, "end": 4.0},
        {"audio_path": "alpha.wav", "style": "pop", "start": 2.0, "end": 6.0, "abc": "X:1\nABC|", "abc_mode": "full"},
        {"audio_path": str(song_dir / "beta.flac"), "caption": "[Tags]\nrock\n[Lyrics]\nyo", "song_id": "b", "extra": 1},
    ]
    _write(jsonl, "\n".join(json.dumps(line) for line in lines) + "\n\n")
    dataset = AudioDataset(audio_jsonl_file=str(jsonl), cache_directory=str(tmp_path / "c"))
    records = dataset.datasource.records
    assert [r.item_key for r in records] == ["alpha", "alpha-r00001", "beta"]
    assert records[0].audio_path == str(song_dir / "alpha.wav")  # resolved against the JSONL directory
    assert records[0].song_id == records[1].song_id == "alpha"
    assert records[1].abc_mode == "full" and records[1].lyrics == "[Verse 1]\nhello there"  # lyrics sidecar
    assert records[2].style == "rock" and records[2].lyrics == "yo" and records[2].song_id == "b"
    assert records[2].abc == 'X:1\nK:C\n"Am"A2 B2|'  # abc sidecar next to the audio file
    assert dataset.datasource.get_item_extras(2).fields == {"extra": 1}

    with pytest.raises(ValueError, match="cache_directory"):
        AudioDataset(audio_jsonl_file=str(jsonl))
    _write(tmp_path / "bad.jsonl", '{"audio_path": "x.wav", "start": 5, "end": 4}\n')
    with pytest.raises(ValueError, match="end must be greater"):
        AudioDataset(audio_jsonl_file=str(tmp_path / "bad.jsonl"), cache_directory=str(tmp_path / "c"))


# --- segments ------------------------------------------------------------------------------------------------------


def test_segment_windows():
    assert segment_windows(300, "full") == [(0, 300)]
    assert segment_windows(300, "head", 100) == [(0, 100)]
    assert segment_windows(80, "head", 100) == []
    assert segment_windows(275, "chunk", 100) == [(0, 100), (100, 100)]  # tail dropped
    assert segment_windows(300, "slide", 100, 50) == [(s, 100) for s in (0, 50, 100, 150, 200)]
    assert segment_windows(300, "slide", 100, 50, max_segments=3) == [(0, 100), (100, 100), (200, 100)]
    assert segment_windows(300, "chunk", 100, max_segments=1) == [(0, 100)]
    assert segment_windows(0, "full") == []
    with pytest.raises(ValueError):
        segment_windows(300, "chunk", None)


def test_full_mode_items_cover_the_whole_record(tmp_path: Path, song_dir: Path):
    group = _group(tmp_path, [{"audio_directory": str(song_dir), "cache_directory": str(tmp_path / "cache")}])
    dataset = group.datasets[0]
    items = _items(dataset)
    assert [i.item_key for i in items] == ["alpha", "beta"]
    for item, path in zip(items, [song_dir / "alpha.wav", song_dir / "beta.flac"]):
        frames = _decoded_frames(path)
        assert item.frame_count == frames and item.frame_pos == 0 and item.bucket_size == (frames,)
        assert item.audio_content.shape == (2, frames * HOP) and item.audio_content.dtype == torch.float32
        assert item.audio_present and item.content is None
        assert item.song_frames == frames and item.song_start_frame == 0 and item.truncated is False
        assert item.record_frames == frames
        assert os.path.basename(item.latent_cache_path) == f"{item.item_key}_000000-{frames:06d}_{ARCH}.safetensors"
        assert os.path.basename(item.text_encoder_output_cache_path) == f"{item.item_key}_{ARCH}_te.safetensors"
        assert item.dataset_index == 0 and item.audio_source.path == path.resolve()
    assert items[0].frame_count == 175  # 7 s at 25 Hz
    assert 148 <= items[1].frame_count <= 150  # 6 s at 44.1 kHz, resampled
    assert items[0].lyrics == "[Verse 1]\nhello there" and items[0].caption == "dream pop, airy vocals"
    assert items[1].abc_text is not None and items[1].abc_mode == "full"
    # the 440 Hz tone survives decoding at 48 kHz
    left = items[0].audio_content[0, : SR // 2].numpy()
    ref = 0.5 * np.sin(2 * np.pi * 440.0 * np.arange(SR // 2) / SR)
    assert np.abs(left - ref).max() < 1e-3


@pytest.mark.parametrize(
    "mode,extra,expected",
    [
        ("head", {"segment_seconds": 4.0}, [(0, 100)]),
        ("chunk", {"segment_seconds": 4.0}, [(0, 100), (100, 100), (200, 100)]),
        ("slide", {"segment_seconds": 4.0, "segment_stride_seconds": 2.0}, [(s, 100) for s in (0, 50, 100, 150, 200)]),
        ("slide", {"segment_seconds": 4.0, "segment_stride_seconds": 2.0, "max_segments": 3}, [(0, 100), (100, 100), (200, 100)]),
    ],
)
def test_segment_modes_share_the_whole_record_waveform(tmp_path: Path, mode, extra, expected):
    d = tmp_path / "s"
    d.mkdir()
    _write_audio(d / "long.wav", 11.0)  # 275 frames: the chunk tail is dropped
    _write_audio(d / "longer.wav", 12.0)  # 300 frames
    group = _group(
        tmp_path,
        [
            {
                "audio_directory": str(d),
                "cache_directory": str(tmp_path / "cache"),
                "segment_extraction": mode,
                "batch_size": 2,
                **extra,
            }
        ],
    )
    items = _items(group.datasets[0])
    by_key = {}
    for item in items:
        by_key.setdefault(item.item_key, []).append(item)
    assert [(i.frame_pos, i.frame_count) for i in by_key["longer"]] == expected
    long_expected = [w for w in segment_windows(275, mode, 100, 50 if mode == "slide" else None, extra.get("max_segments"))]
    assert [(i.frame_pos, i.frame_count) for i in by_key["long"]] == long_expected
    for key, total in (("long", 275), ("longer", 300)):
        whole = by_key[key][0].audio_content
        assert whole.shape == (2, total * HOP)
        for item in by_key[key]:
            assert item.audio_content is whole  # decoded once per record
            window = item.audio_content[:, item.frame_pos * HOP : (item.frame_pos + item.frame_count) * HOP]
            assert window.shape == (2, item.frame_count * HOP)
            assert item.song_start_frame == item.frame_pos and item.song_frames == total and item.truncated is False
            assert item.latent_cache_path.endswith(f"_{item.frame_pos:06d}-{item.frame_count:06d}_{ARCH}.safetensors")


def test_min_seconds_drop_and_max_seconds_truncation(tmp_path: Path):
    d = tmp_path / "s"
    d.mkdir()
    _write_audio(d / "short.wav", 3.0)
    _write_audio(d / "long.wav", 20.0)
    group = _group(tmp_path, [{"audio_directory": str(d), "cache_directory": str(tmp_path / "c"), "max_seconds": 10}])
    dataset = group.datasets[0]
    items = _items(dataset)
    assert [i.item_key for i in items] == ["long"] and dataset.dropped_short == 1
    item = items[0]
    assert item.frame_count == 250 and item.audio_content.shape == (2, 250 * HOP)
    assert item.song_frames == 500 and item.truncated is True and item.song_start_frame == 0

    # head longer than the record: no segment
    group = _group(
        tmp_path,
        [
            {
                "audio_directory": str(d),
                "cache_directory": str(tmp_path / "c2"),
                "segment_extraction": "head",
                "segment_seconds": 30,
                "min_seconds": 1,
            }
        ],
    )
    dataset = group.datasets[0]
    assert _items(dataset) == [] and dataset.dropped_no_segment == 2

    with pytest.raises(ValueError, match="segment_seconds"):
        AudioDataset(audio_directory=str(d), segment_extraction="chunk", audio_spec=YUE2_AUDIO_SPEC)
    with pytest.raises(ValueError, match="min_seconds"):
        AudioDataset(audio_directory=str(d), min_seconds=10, max_seconds=5)


def test_jsonl_excerpts_are_in_song_coordinates(tmp_path: Path):
    d = tmp_path / "s"
    d.mkdir()
    _write_audio(d / "song.wav", 12.0)  # 300 frames
    jsonl = _write(
        d / "list.jsonl",
        "\n".join(
            [
                '{"audio_path": "song.wav", "caption": "a", "start": 0, "end": 4}',
                '{"audio_path": "song.wav", "caption": "a", "start": 4, "end": 10}',
                '{"audio_path": "song.wav", "caption": "a", "start": 6}',
                '{"audio_path": "song.wav", "caption": "a", "end": 30}',
            ]
        ),
    )
    group = _group(tmp_path, [{"audio_jsonl_file": str(jsonl), "cache_directory": str(tmp_path / "c"), "min_seconds": 1}])
    items = _items(group.datasets[0])
    got = [(i.item_key, i.frame_pos, i.frame_count, i.song_start_frame, i.song_frames, i.truncated) for i in items]
    assert got == [
        ("song", 0, 100, 0, 300, True),  # song start, ends before the song does
        ("song-r00001", 0, 150, 100, 300, True),  # mid-song excerpt: song start frame 100
        ("song-r00002", 0, 150, 150, 300, False),  # runs to the song end
        ("song-r00003", 0, 300, 0, 300, False),  # end beyond the file is clamped
    ]
    whole = decode_audio(AudioSource((d / "song.wav").resolve(), False), sample_rate=SR, channels=2)
    assert torch.equal(items[1].audio_content, whole[:, 4 * SR : 4 * SR + 150 * HOP])


# --- text items, training, validation ------------------------------------------------------------------------------


def test_text_items_are_per_record_without_audio(tmp_path: Path, song_dir: Path):
    dataset = AudioDataset(
        audio_directory=str(song_dir), batch_size=1, cache_directory=str(tmp_path / "c"), architecture=ARCH
    )  # no audio spec
    batches = list(dataset.retrieve_text_encoder_output_cache_batches(num_workers=4))
    assert [len(b) for b in batches] == [1, 1]
    items = [b[0] for b in batches]
    assert [i.item_key for i in items] == ["alpha", "beta"]
    assert items[0].caption == "dream pop, airy vocals" and items[0].lyrics == "[Verse 1]\nhello there"
    assert items[1].abc_mode == "full" and items[0].audio_content is None
    assert [i.datasource_index for i in items] == [0, 1]
    assert items[0].text_encoder_output_cache_path == str(tmp_path / "c" / "alpha_yue2_te.safetensors")
    with pytest.raises(ValueError, match="AudioSpec"):
        list(dataset.retrieve_latent_cache_batches(num_workers=1))


def test_prepare_for_training_batch_contract(tmp_path: Path):
    d = tmp_path / "s"
    d.mkdir()
    for name in ("a", "b", "c"):
        _write_audio(d / f"{name}.wav", 4.0)
        _write(d / f"{name}.caption.txt", f"style {name}")
    _write(d / "b.abc.txt", "X:1\nK:C\nCDE|")
    datasets = [{"audio_directory": str(d), "cache_directory": str(tmp_path / "cache"), "min_seconds": 1, "batch_size": 2}]
    _write_fake_caches(_group(tmp_path, datasets).datasets[0])

    group = _group(tmp_path, datasets, training=True)
    dataset = group.datasets[0]
    assert dataset.num_train_items == 3 and group.num_train_items == 3 and dataset.validation_items == []
    manager = dataset.batch_manager
    assert list(manager.buckets) == [(100,)] and len(manager) == 2
    batch = manager[0]
    expected = {"latents", "codes", "yue2_seg", "yue2_abc", "yue2_has_abc", "yue2_abc_mode", "timesteps"}
    expected |= {f"yue2_{kind}_{cot}" for kind in ("text", "neg") for cot in protocol.COT_MODES}
    assert set(batch) == expected
    assert batch["latents"].shape == (2, 100, 64) and batch["latents"].dtype == torch.float32
    assert batch["codes"].shape == (2, 100) and batch["codes"].dtype == torch.int64
    assert batch["yue2_seg"].tolist() == [[0, 100, 0], [0, 100, 0]]
    assert batch["yue2_has_abc"].shape == (2,) and batch["yue2_abc_mode"].shape == (2,)
    assert isinstance(batch["yue2_text_off"], list) and batch["yue2_text_off"][0][0] == protocol.EOD
    assert isinstance(batch["yue2_abc"], list) and batch["timesteps"] is None
    all_abc = [manager[i]["yue2_abc_mode"].tolist() for i in range(len(manager))]
    assert sorted(sum(all_abc, [])) == [0, 0, 1]  # b has a melody score
    noise = torch.randn_like(batch["latents"])  # the trainer loop's first use of the batch
    assert noise.shape == batch["latents"].shape


def test_validation_split_by_song_is_deterministic_and_excluded(tmp_path: Path):
    d = tmp_path / "s"
    d.mkdir()
    for i in range(6):
        _write_audio(d / f"t{i}.wav", 2.0, freq=200.0 + 50 * i)
    # two excerpts of t0 belong to one song and must land on the same side
    lines = [f'{{"audio_path": "t{i}.wav", "caption": "x"}}' for i in range(6)]
    lines.append('{"audio_path": "t0.wav", "caption": "x", "start": 0.5}')
    jsonl = _write(d / "list.jsonl", "\n".join(lines))
    base = {"audio_jsonl_file": str(jsonl), "cache_directory": str(tmp_path / "cache"), "min_seconds": 1}
    _write_fake_caches(_group(tmp_path, [base]).datasets[0])

    split = {**base, "validation_split": 0.34, "validation_split_seed": 7}
    first = _group(tmp_path, [split], training=True).datasets[0]
    second = _group(tmp_path, [split], training=True).datasets[0]
    held = first.validation_song_ids()
    assert held == second.validation_song_ids() == validation_song_split([f"t{i}" for i in range(6)], 0.34, 7)
    assert len(held) == 2  # round(6 * 0.34)
    assert {i.item_key for i in first.validation_items} == {i.item_key for i in second.validation_items}
    trained = {item.item_key for bucket in first.batch_manager.buckets.values() for item in bucket}
    validated = {item.item_key for item in first.validation_items}
    assert not trained & validated and len(trained | validated) == 7
    assert ("t0" in held) == ("t0-r00006" in validated) == ("t0" in validated)
    assert first.num_train_items == len(trained)

    everything = _group(tmp_path, [{**base, "is_validation": True}], training=True).datasets[0]
    assert everything.num_train_items == 0 and len(everything.validation_items) == 7 and len(everything.batch_manager) == 0

    assert validation_song_split(["a"], 0.5, 0) == set()
    assert validation_song_split(["a", "b"], 0.0, 0) == set()
    assert len(validation_song_split([str(i) for i in range(10)], 0.95, 0)) == 9  # at least one song trains
    assert validation_song_split([str(i) for i in range(10)], 0.3, 1) != validation_song_split([str(i) for i in range(10)], 0.3, 2)


def test_song_id_from_text_cache_metadata_when_record_is_gone(tmp_path: Path):
    d = tmp_path / "s"
    d.mkdir()
    for i in range(4):
        _write_audio(d / f"t{i}.wav", 2.0)
        _write(d / f"t{i}.song.txt", "shared" if i < 2 else f"own{i}")
    base = {"audio_directory": str(d), "cache_directory": str(tmp_path / "cache"), "min_seconds": 1}
    _write_fake_caches(_group(tmp_path, [base]).datasets[0])
    (d / "t0.wav").unlink()
    dataset = _group(tmp_path, [{**base, "validation_split": 0.5}], training=True).datasets[0]
    songs = {item.item_key: item.song_id for bucket in dataset.batch_manager.buckets.values() for item in bucket}
    songs.update({item.item_key: item.song_id for item in dataset.validation_items})
    assert songs["t0"] == songs["t1"] == "shared"


# --- real data (YUE2_DATA_DIR) ------------------------------------------------------------------------------------


def _jamendo_dir():
    data_dir = os.environ.get("YUE2_DATA_DIR")
    path = Path(data_dir) / "jamendolyrics" if data_dir else None
    if path is None or not (path / "JamendoLyrics.csv").is_file():
        pytest.skip("YUE2_DATA_DIR/jamendolyrics is not available")
    return path


def test_jamendolyrics_smoke(tmp_path: Path):
    root = _jamendo_dir()
    with open(root / "JamendoLyrics.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:3]
    records = []
    for row in rows:
        stem = Path(row["Filepath"]).stem
        lyrics = (root / "lyrics" / f"{stem}.txt").read_text(encoding="utf-8")
        records.append(
            {"audio_path": str(root / "mp3" / row["Filepath"]), "caption": f"{row['Genre']}, {row['Language']}", "lyrics": lyrics}
        )
    first = dict(records[0], start=30.0, end=60.0)
    jsonl = _write(tmp_path / "jamendo.jsonl", "\n".join(json.dumps(r) for r in records + [first]))

    full = _group(tmp_path, [{"audio_jsonl_file": str(jsonl), "cache_directory": str(tmp_path / "c_full")}]).datasets[0]
    items = _items(full)
    assert [i.item_key for i in items][:3] == [Path(r["audio_path"]).stem for r in records]
    song_frames = {}
    for item in items:
        decoded = item.audio_content
        assert decoded.shape == (2, item.frame_count * HOP)
        assert torch.isfinite(decoded).all() and decoded.abs().max() < 2.0
        assert decoded.pow(2).mean().sqrt() > 1e-3  # real music, not silence
        assert item.lyrics and item.caption
        if item.item_key.endswith("-r00003"):
            assert item.frame_count == 750 and item.song_start_frame == 750 and item.truncated is True
        else:
            seconds = item.song_frames / 25
            assert 60 < seconds < 600, seconds
            assert item.frame_count == min(item.song_frames, 360 * 25) and item.song_start_frame == 0
            song_frames[item.item_key] = item.song_frames
        print(f"{item.item_key}: {item.frame_count} frames ({item.frame_count / 25:.1f}s), song {item.song_frames}")
    assert items[-1].song_frames == song_frames[items[0].item_key]

    chunked = _group(
        tmp_path,
        [
            {
                "audio_jsonl_file": str(jsonl),
                "cache_directory": str(tmp_path / "c_chunk"),
                "segment_extraction": "chunk",
                "segment_seconds": 30,
                "batch_size": 4,
            }
        ],
    ).datasets[0]
    segments = {}
    for item in _items(chunked):
        segments.setdefault(item.item_key, []).append(item)
    for key, frames in song_frames.items():
        got = segments[key]
        assert len(got) == min(frames, 360 * 25) // 750
        assert [s.frame_pos for s in got] == [750 * i for i in range(len(got))]
        assert all(s.audio_content is got[0].audio_content for s in got)
