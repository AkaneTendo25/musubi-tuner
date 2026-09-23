import os
from pathlib import Path
import sys

import av
import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import FakeTokenizer, FakeYuE2VAE, load_fake_yue2_vae  # noqa: E402

from musubi_tuner import yue2_cache_latents, yue2_cache_text_encoder_outputs  # noqa: E402
from musubi_tuner.dataset import config_utils  # noqa: E402
from musubi_tuner.dataset.audio_utils import AudioSource, decode_audio  # noqa: E402
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer  # noqa: E402
from musubi_tuner.yue2 import yue2_protocol as protocol  # noqa: E402
from musubi_tuner.yue2.yue2_protocol import COT_MODES, HOP, MERT_HOP, MERT_SAMPLE_RATE, YUE2_AUDIO_SPEC  # noqa: E402

SR = 48000
ABC = 'X:1\nK:C\n"C"cdef|"G"gabc|'


def _write_audio(path: Path, seconds: float, sample_rate: int = SR, freq: float = 440.0, channels: int = 2) -> Path:
    n = int(round(seconds * sample_rate))
    t = np.arange(n) / sample_rate
    env = 0.5 + 0.4 * np.sin(2 * np.pi * 0.7 * t)  # a varying envelope so frames differ
    left = env * np.sin(2 * np.pi * freq * t)
    right = 0.5 * env * np.sin(2 * np.pi * 1.5 * freq * t)
    samples = np.stack([left, right] if channels == 2 else [left])
    samples = (np.clip(samples, -1, 1) * 32767.0).astype(np.int16)
    layout = "stereo" if channels == 2 else "mono"
    codec = "flac" if path.suffix == ".flac" else "pcm_s16le"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=sample_rate)
        stream.layout = layout
        for start in range(0, n, 4096):
            chunk = samples[:, start : start + 4096]
            interleaved = np.ascontiguousarray(chunk.T).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout=layout)
            frame.sample_rate = sample_rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def songs(tmp_path: Path) -> Path:
    d = tmp_path / "songs"
    d.mkdir()
    _write_audio(d / "alpha.wav", 12.0)  # 300 frames: 3 chunks of 100
    (d / "alpha.caption.txt").write_text("pop, bright", encoding="utf-8")
    (d / "alpha.lyrics.txt").write_text("[verse]\nla la la\n", encoding="utf-8")
    (d / "alpha.abc.txt").write_text(ABC, encoding="utf-8")
    _write_audio(d / "beta.flac", 9.0, sample_rate=44100, freq=330.0)  # 225 frames: 2 chunks, tail dropped
    (d / "beta.caption.txt").write_text("ambient  drone", encoding="utf-8")
    _write_audio(d / "gamma.wav", 8.4, freq=550.0, channels=1)  # mono source
    (d / "gamma.caption.txt").write_text("folk", encoding="utf-8")
    (d / "gamma.lyrics.txt").write_text("hey", encoding="utf-8")
    return d


def _config(tmp_path: Path, songs: Path, **extra) -> Path:
    cache = tmp_path / "cache"
    lines = [
        "[general]",
        "batch_size = 1",
        "[[datasets]]",
        f'audio_directory = "{songs.as_posix()}"',
        f'cache_directory = "{cache.as_posix()}"',
        'segment_extraction = "chunk"',
        "segment_seconds = 4.0",
        "min_seconds = 2.0",
    ]
    lines += [f"{k} = {v!r}" if isinstance(v, str) else f"{k} = {v}" for k, v in extra.items()]
    path = tmp_path / "dataset.toml"
    path.write_text("\n".join(lines).replace("'", '"') + "\n", encoding="utf-8")
    return path


def _fake_codes(mono24: torch.Tensor, frames: int) -> torch.Tensor:
    x = mono24[: frames * MERT_HOP]
    if x.shape[0] < frames * MERT_HOP:
        x = torch.nn.functional.pad(x, (0, frames * MERT_HOP - x.shape[0]))
    energy = x.reshape(frames, MERT_HOP).abs().mean(1).double()
    z = (energy - energy.mean()) / (energy.std() + 1e-5)  # normalised over the whole record, like the real head input
    return ((z * 1000).round().long() + 16384).clamp(0, protocol.CODEC_SIZE - 1)


class FakeSemantic:
    instances: list["FakeSemantic"] = []

    def __init__(self, head_path, mert_model=None, mert_revision=None, device="cpu", dtype=None, **kwargs):
        self.calls: list[tuple[int, int]] = []
        self.unloaded = False
        FakeSemantic.instances.append(self)

    def tokenize_frames(self, mono24: torch.Tensor, frames: int) -> torch.Tensor:
        self.calls.append((int(mono24.shape[0]), frames))
        return _fake_codes(mono24, frames)

    def unload(self):
        self.unloaded = True


class Harness:
    def __init__(self, monkeypatch, tmp_path: Path, songs: Path):
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.songs = songs
        self.cache = tmp_path / "cache"
        self.config = _config(tmp_path, songs)
        self.head = tmp_path / "head.safetensors"
        self.head.write_bytes(b"head-v1")
        self.vaes: list[FakeYuE2VAE] = []
        self.source_dtype = "float32"
        FakeSemantic.instances = []

        def loader(vae_path, dit_path, *, device="cpu", decoder_only=False, allow_fp16_source=True):
            vae = load_fake_yue2_vae(
                vae_path, dit_path, device=device, allow_fp16_source=allow_fp16_source, source_dtype=self.source_dtype
            )
            self.vaes.append(vae)
            return vae

        monkeypatch.setattr(yue2_cache_latents, "load_yue2_vae", loader)
        monkeypatch.setattr(yue2_cache_latents, "SemanticTokenizer", FakeSemantic)
        monkeypatch.setattr(yue2_cache_text_encoder_outputs, "load_yue2_tokenizer", lambda tokenizer, dit: FakeTokenizer())

        # which cache files each run writes (file mtimes are too coarse for back-to-back runs)
        self.written: list[str] = []
        writers = (
            (yue2_cache_latents, "save_latent_cache_yue2"),
            (yue2_cache_text_encoder_outputs, "save_text_encoder_output_cache_yue2"),
        )
        for module, name in writers:
            original = getattr(module, name)

            def spy(item, *args, _original=original, _te=name.startswith("save_text"), **kwargs):
                self.written.append(os.path.basename(item.text_encoder_output_cache_path if _te else item.latent_cache_path))
                return _original(item, *args, **kwargs)

            monkeypatch.setattr(module, name, spy)

    def latents(self, *extra: str, codes: bool = True):
        argv = ["yue2_cache_latents.py", "--dataset_config", str(self.config), "--vae", "fake", "--device", "cpu"]
        argv += ["--num_workers", "1"]
        if codes:
            argv += ["--semantic_head", str(self.head)]
        self.monkeypatch.setattr(sys, "argv", argv + list(extra))
        FakeSemantic.instances = []
        self.vaes = []
        self.written = []
        yue2_cache_latents.main()

    def text(self, *extra: str):
        argv = ["yue2_cache_text_encoder_outputs.py", "--dataset_config", str(self.config), "--num_workers", "1"]
        self.monkeypatch.setattr(sys, "argv", argv + list(extra))
        self.written = []
        yue2_cache_text_encoder_outputs.main()

    def vae_calls(self) -> int:
        return sum(len(v.encode_calls) for v in self.vaes)

    def semantic_calls(self) -> list:
        return [c for s in FakeSemantic.instances for c in s.calls]

    def files(self) -> list[str]:
        return sorted(p.name for p in self.cache.glob("*_yue2.safetensors"))


def _meta(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata())


def _record48(path: Path, frames: int) -> torch.Tensor:
    wave = decode_audio(AudioSource(path.resolve(), False), sample_rate=SR, channels=2)
    wave = wave[:, : frames * HOP]
    return torch.nn.functional.pad(wave, (0, frames * HOP - wave.shape[1]))


def _decoded_frames(path: Path) -> int:
    return decode_audio(AudioSource(path.resolve(), False), sample_rate=SR, channels=2).shape[1] // HOP


def _record24(path: Path, channels: int) -> torch.Tensor:
    return decode_audio(AudioSource(path.resolve(), False), sample_rate=MERT_SAMPLE_RATE, channels=channels).mean(0)


EXPECTED_FILES = [
    "alpha_000000-000100_yue2.safetensors",
    "alpha_000100-000100_yue2.safetensors",
    "alpha_000200-000100_yue2.safetensors",
    "beta_000000-000100_yue2.safetensors",
    "beta_000100-000100_yue2.safetensors",
    "gamma_000000-000100_yue2.safetensors",
    "gamma_000100-000100_yue2.safetensors",
]


def test_latent_cache_segments_are_slices_of_whole_record_encodes(monkeypatch, tmp_path: Path, songs: Path):
    h = Harness(monkeypatch, tmp_path, songs)
    h.latents()
    assert h.files() == EXPECTED_FILES and sorted(h.written) == EXPECTED_FILES

    fake = FakeYuE2VAE()
    sources = {"alpha": (songs / "alpha.wav", 2), "beta": (songs / "beta.flac", 2), "gamma": (songs / "gamma.wav", 1)}
    record_frames = {k: _decoded_frames(p) for k, (p, _) in sources.items()}
    assert record_frames["alpha"] == 300 and record_frames["gamma"] == 210 and record_frames["beta"] in (224, 225)
    for name in EXPECTED_FILES:
        key, seg, _ = name.split("_")
        start, n = (int(v) for v in seg.split("-"))
        sd = load_file(str(h.cache / name))
        assert set(sd) == {"latents_100x64_float32", "codes_int64", "yue2_seg_int64"}
        path, channels = sources[key]
        audio = _record48(path, record_frames[key])
        full = fake.encode_mean_chunked(audio, 750, 50)
        assert torch.equal(sd["latents_100x64_float32"], full[start : start + n])
        codes = _fake_codes(_record24(path, channels), record_frames[key])
        assert torch.equal(sd["codes_int64"], codes[start : start + n])
        assert sd["yue2_seg_int64"].tolist() == [start, record_frames[key], 0]
        meta = _meta(h.cache / name)
        assert meta["yue2_has_codes"] == "1" and meta["yue2_semantic_head"].startswith("sha256:")
        assert meta["yue2_vae_source_dtype"] == "float32" and meta["yue2_song_id"] == key
        assert meta["yue2_record"] == f"0+{record_frames[key]}"
        if start == 100 and key == "alpha":
            # a segment-only encode would see zero padding at its edges (fake VAE: +-1 frame receptive field)
            alone = fake.encode_mean_chunked(audio[:, start * HOP : (start + n) * HOP], 750, 50)
            assert not torch.allclose(alone, full[start : start + n])
            assert not torch.equal(_fake_codes(_record24(path, channels)[start * MERT_HOP :], n), codes[start : start + n])

    # one whole-record encode and one tokenization per record
    assert sorted(frames for _, frames in h.semantic_calls()) == sorted(record_frames.values())
    # each record is encoded once (a single 750-frame chunk here)
    assert sorted(h.vaes[0].encode_calls) == sorted(n * HOP for n in record_frames.values())
    assert FakeSemantic.instances[0].unloaded


def test_skip_existing_rebuilds_on_metadata_change_and_removes_stale(monkeypatch, tmp_path: Path, songs: Path):
    h = Harness(monkeypatch, tmp_path, songs)
    h.latents()
    ghost = h.cache / "ghost_000000-000100_yue2.safetensors"
    ghost.write_bytes(b"stale")
    stamps = [p for p in h.files() if p != ghost.name]

    h.latents("--skip_existing")
    assert h.vae_calls() == 0 and h.semantic_calls() == []
    assert not ghost.exists(), "a cache file not in the dataset is removed"
    assert h.written == []

    # another head: codes are recomputed, latents are reused from the files
    old_head = {_meta(h.cache / p)["yue2_semantic_head"] for p in stamps}
    latents_before = {p: load_file(str(h.cache / p))["latents_100x64_float32"] for p in stamps}
    h.head.write_bytes(b"head-v2")
    h.latents("--skip_existing")
    assert h.vae_calls() == 0 and len(h.semantic_calls()) == 3
    new_head = {_meta(h.cache / p)["yue2_semantic_head"] for p in stamps}
    assert len(old_head) == len(new_head) == 1 and old_head != new_head
    assert all(torch.equal(load_file(str(h.cache / p))["latents_100x64_float32"], z) for p, z in latents_before.items())

    # an fp16 VAE source is refused, then accepted with --allow_fp16_vae and rebuilds everything
    h.source_dtype = "float16"
    with pytest.raises(ValueError, match="float16"):
        h.latents("--skip_existing")
    h.latents("--skip_existing", "--allow_fp16_vae")
    assert h.vae_calls() == 3
    assert {_meta(h.cache / p)["yue2_vae_source_dtype"] for p in stamps} == {"float16"}

    # other VAE chunking also rebuilds
    h.latents("--skip_existing", "--allow_fp16_vae", "--vae_chunk_frames", "100", "--vae_overlap_frames", "5")
    assert h.vae_calls() > 3
    h.latents("--skip_existing", "--allow_fp16_vae", "--vae_chunk_frames", "100", "--vae_overlap_frames", "5")
    assert h.vae_calls() == 0


def test_sequential_models_and_no_codes(monkeypatch, tmp_path: Path, songs: Path):
    h = Harness(monkeypatch, tmp_path, songs)
    h.latents()
    single = {p: load_file(str(h.cache / p)) for p in h.files()}
    single_meta = {p: _meta(h.cache / p) for p in h.files()}
    for p in h.files():
        (h.cache / p).unlink()

    h.latents("--sequential_models")
    assert h.files() == EXPECTED_FILES
    for p in h.files():
        sd = load_file(str(h.cache / p))
        assert set(sd) == set(single[p]) and all(torch.equal(sd[k], single[p][k]) for k in sd)
        assert _meta(h.cache / p) == single_meta[p]

    h.latents("--no_codes", codes=False)
    for p in h.files():
        sd = load_file(str(h.cache / p))
        assert "codes_int64" not in sd and _meta(h.cache / p)["yue2_has_codes"] == "0"
        assert torch.equal(sd["latents_100x64_float32"], single[p]["latents_100x64_float32"])

    with pytest.raises(ValueError, match="--semantic_head"):
        h.latents(codes=False)


def test_bfloat16_latents_and_debug_listing(monkeypatch, tmp_path: Path, songs: Path, capsys):
    h = Harness(monkeypatch, tmp_path, songs)
    h.latents("--debug_mode", "console")
    assert h.files() == [] and h.vaes == []
    out = capsys.readouterr().out
    assert "alpha: segment 100+100 of 300 frames" in out
    h.latents("--latent_dtype", "bfloat16")
    sd = load_file(str(h.cache / EXPECTED_FILES[0]))
    assert sd["latents_100x64_bfloat16"].dtype == torch.bfloat16


def test_text_cache_contents_and_skip_existing(monkeypatch, tmp_path: Path, songs: Path):
    h = Harness(monkeypatch, tmp_path, songs)
    h.text()
    tok = FakeTokenizer()
    te = {p.name: p for p in h.cache.glob("*_yue2_te.safetensors")}
    assert sorted(te) == ["alpha_yue2_te.safetensors", "beta_yue2_te.safetensors", "gamma_yue2_te.safetensors"]

    sd = load_file(str(te["alpha_yue2_te.safetensors"]))
    keys = {f"varlen_yue2_text_{c}_int64" for c in COT_MODES} | {f"varlen_yue2_neg_{c}_int64" for c in COT_MODES}
    assert set(sd) == keys | {"varlen_yue2_abc_int64", "yue2_has_abc_int64", "yue2_abc_mode_int64"}
    for cot in COT_MODES:
        assert sd[f"varlen_yue2_text_{cot}_int64"].tolist() == protocol.text_ids(tok, "pop, bright", "[verse]\nla la la", cot)
        assert sd[f"varlen_yue2_neg_{cot}_int64"].tolist() == protocol.negative_text_ids(tok, cot)
    assert sd["varlen_yue2_abc_int64"].tolist() == tok.encode(ABC)
    assert int(sd["yue2_has_abc_int64"]) == 1 and int(sd["yue2_abc_mode_int64"]) == 2  # quoted chords -> full
    meta = _meta(te["alpha_yue2_te.safetensors"])
    assert meta["yue2_song_id"] == "alpha" and meta["yue2_tokenizer"] == tok.fingerprint
    assert meta["yue2_instrumental_lyrics"] == "[instrumental]"

    beta = load_file(str(te["beta_yue2_te.safetensors"]))
    # no lyrics sidecar: the instrumental placeholder; style whitespace normalised
    assert beta["varlen_yue2_text_off_int64"].tolist() == protocol.text_ids(tok, "ambient drone", "[instrumental]", "off")
    assert beta["varlen_yue2_abc_int64"].numel() == 0 and int(beta["yue2_has_abc_int64"]) == 0
    assert int(beta["yue2_abc_mode_int64"]) == 0

    ghost = h.cache / "ghost_yue2_te.safetensors"
    ghost.write_bytes(b"stale")
    h.text("--skip_existing")
    assert h.written == []
    assert not ghost.exists()

    # a new placeholder rewrites every record (the trainer checks the placeholder against each cache)
    h.text("--skip_existing", "--instrumental_lyrics", "[inst]")
    changed = set(h.written)
    assert changed == set(te)
    assert {_meta(path)["yue2_instrumental_lyrics"] for path in te.values()} == {"[inst]"}
    beta = load_file(str(te["beta_yue2_te.safetensors"]))
    assert beta["varlen_yue2_text_off_int64"].tolist() == protocol.text_ids(tok, "ambient drone", "[inst]", "off")

    # changed lyrics rewrite that record only
    (songs / "gamma.lyrics.txt").write_text("hey hey", encoding="utf-8")
    h.text("--skip_existing", "--instrumental_lyrics", "[inst]")
    assert h.written == ["gamma_yue2_te.safetensors"]


def test_caches_feed_the_training_batch(monkeypatch, tmp_path: Path, songs: Path):
    h = Harness(monkeypatch, tmp_path, songs)
    h.latents()
    h.text()
    import argparse

    user_config = config_utils.load_user_config(str(h.config))
    namespace = argparse.Namespace(debug_dataset=False)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, namespace, architecture="yue2")
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True, audio_spec=YUE2_AUDIO_SPEC)
    dataset = group.datasets[0]
    assert dataset.num_train_items == len(EXPECTED_FILES)
    batch = dataset.batch_manager[0]
    assert batch["latents"].shape == (1, 100, 64) and batch["codes"].shape == (1, 100)
    assert {"yue2_seg", "yue2_text_off", "yue2_neg_full", "yue2_abc", "yue2_has_abc", "yue2_abc_mode"} <= set(batch)
