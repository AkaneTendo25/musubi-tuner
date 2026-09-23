import os
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.yue2 import yue2_semantic as sem  # noqa: E402
from musubi_tuner.yue2.yue2_protocol import CODEC_SIZE, MERT_HOP, MERT_SAMPLE_RATE  # noqa: E402

WIN = 16
DIN = 8


def _tiny_head(seed: int = 0, window: int = WIN) -> sem.SemanticHead:
    torch.manual_seed(seed)
    head = sem.SemanticHead(sem.HeadConfig(in_dim=DIN, width=16, depth=2, num_heads=2, vocab=CODEC_SIZE, window=window))
    with torch.no_grad():
        head.pos.normal_(0, 0.5)
        head.head.weight.normal_(0, 1.0)
    return head.eval()


def _save_head_safetensors(head: sem.SemanticHead, path: Path, metadata=None, dtype=torch.float32) -> Path:
    sd = {k: v.detach().to(dtype).contiguous() for k, v in head.state_dict().items()}
    meta = {
        "input": "MERT-v2-FullSong layer 20 features @25 Hz, per-track instance-normalised (instnorm=true)",
        "architecture": "2-layer TransformerEncoder d=16 heads=2 ffn=64 norm_first gelu",
    }
    meta.update(metadata or {})
    save_file(sd, str(path), metadata=meta)
    return path


def _reference_tokens(head: sem.SemanticHead, feats: torch.Tensor) -> torch.Tensor:
    """Naive oracle: run every window of the plan, then give each frame the prediction of the last window whose trimmed
    range holds it."""
    w, total, din = head.window, feats.shape[0], feats.shape[1]
    f = feats.float()
    x = (f - f.mean(0)) / (f.std(0) + 1e-5)

    starts = [0]
    while starts[-1] + w // 2 + w <= total:
        starts.append(starts[-1] + w // 2)
    if starts[-1] + w < total:
        starts.append(total - w)

    windows = []
    with torch.no_grad():
        for s in starts:
            real = x[s : s + w]
            n = real.shape[0]
            batch = torch.cat([real, torch.zeros(w - n, din)], dim=0).reshape(1, w, din)
            pred = head(batch)[0, :n].argmax(-1)
            # drop a quarter window at each edge unless that edge is the sequence edge
            lo = s if s == 0 else s + w // 4
            hi = s + n if s + n == total else s + n - w // 4
            windows.append((s, lo, hi, pred))

    out = torch.empty(total, dtype=torch.long)
    for t in range(total):
        for s, lo, hi, pred in reversed(windows):
            if lo <= t < hi:
                out[t] = pred[t - s]
                break
        else:
            raise AssertionError(f"frame {t} has no owning window")
    return out


@pytest.mark.parametrize("total", [1, 3, 15, 16, 17, 23, 24, 25, 31, 32, 33, 40, 47, 48, 49, 100, 513])
def test_window_plan_covers_every_frame_away_from_inner_edges(total):
    plan = sem.window_plan(total, WIN)
    owner = [None] * total
    for s0, lo, hi in plan:
        assert s0 <= lo <= hi <= min(total, s0 + WIN)
        for t in range(lo, hi):
            owner[t] = (s0, lo, hi)
    assert all(o is not None for o in owner), "every frame gets a code"
    for t, (s0, lo, hi) in enumerate(owner):
        # the window that finally assigns a frame never sees it within a quarter window of an inner edge
        if s0 > 0:
            assert t - s0 >= WIN // 4
        if s0 + WIN < total:
            assert s0 + WIN - t > WIN // 4


@pytest.mark.parametrize("total", [5, 16, 24, 37, 64, 101])
def test_tokens_match_the_reference_windowing(total):
    head = _tiny_head()
    feats = torch.randn(total, DIN) * 3 + 1
    ours = sem.classify_features(head, feats, instnorm=True)
    ref = _reference_tokens(head, feats)
    assert torch.equal(ours, ref)
    assert ours.dtype == torch.int64 and ours.shape == (total,)
    assert int(ours.min()) >= 0 and int(ours.max()) < CODEC_SIZE


@pytest.mark.parametrize("total", [0, 2, 6, 7, 13, 29])
def test_tokens_match_the_reference_windowing_odd_window(total):
    head = _tiny_head(seed=1, window=6)
    feats = torch.randn(total, DIN) * 2 - 1
    ours = sem.classify_features(head, feats, instnorm=True)
    assert ours.dtype == torch.int64 and ours.shape == (total,)
    if total > 1:
        assert torch.equal(ours, _reference_tokens(head, feats))


def test_window_plan_reference_plans():
    assert sem.window_plan(0, 512) == []
    assert sem.window_plan(100, 512) == [(0, 0, 100)]
    assert sem.window_plan(1512, 512) == [
        (0, 0, 384),
        (256, 384, 640),
        (512, 640, 896),
        (768, 896, 1152),
        (1000, 1128, 1512),
    ]
    assert sem.window_plan(13, 6) == [(0, 0, 5), (3, 4, 8), (6, 7, 11), (7, 8, 13)]


def test_instance_norm_matches_float64():
    feats = torch.randn(57, DIN) * 5 + 3
    x = feats.double()
    ref = (x - x.mean(0)) / (x.var(0, unbiased=True).sqrt() + 1e-5)
    out = sem.instance_norm(feats)
    assert out.dtype == torch.float32 and out.shape == feats.shape
    assert torch.allclose(out.double(), ref, atol=1e-5)
    assert sem.instance_norm(feats.half()).dtype == torch.float32


def _plan(groups):
    # chunk views -> [start, length] relative to the original storage
    return [[[int(c.storage_offset()), int(c.shape[0])] for c in g] for g in groups]


@pytest.mark.parametrize(
    "samples, plan, keep",
    [
        (24000, [[[0, 24000]]], None),
        (720000, [[[0, 720000]]], None),
        (720001, [[[0, 720000]], [[696001, 24000]]], 0),
        (743999, [[[0, 720000]], [[719999, 24000]]], 24),
        (744000, [[[0, 720000]], [[720000, 24000]]], None),
        (1451520, [[[0, 720000], [720000, 720000]], [[1427520, 24000]]], 12),
        (1080000, [[[0, 720000]], [[720000, 360000]]], None),
    ],
)
def test_split_mert_chunks_plans(samples, plan, keep):
    groups, keep_tail = sem.split_mert_chunks(torch.zeros(samples))
    assert _plan(groups) == plan
    assert keep_tail == keep


def test_split_mert_chunks_rejects_short_audio():
    with pytest.raises(ValueError, match="at least one second"):
        sem.split_mert_chunks(torch.zeros(23999))


class _RotaryEmbedding(torch.nn.Module):
    """Shaped like MERT2's rotary class: derives a non-persistent ``inv_freq`` and lazy tables from a config."""

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.base = config.rotary_embedding_base
        steps = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        self.register_buffer("inv_freq", 1.0 / (self.base**steps), persistent=False)
        self._sequence_length, self._cache_device, self._cos, self._sin = 0, None, None, None


class _Lookalike(torch.nn.Module):
    """Has every attribute of the rotary class but is not one; must be left alone."""

    def __init__(self):
        super().__init__()
        self.head_dim, self.base = 8, 10
        self.register_buffer("inv_freq", torch.full((4,), 3.0), persistent=False)
        self._cos = torch.ones(2)


class _Mert(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList([torch.nn.Sequential(_RotaryEmbedding(config)), _Lookalike()])
        self.embed_positions = _RotaryEmbedding(config)


def _mert_config(hidden=256, heads=4, base=10000):
    return types.SimpleNamespace(hidden_size=hidden, num_attention_heads=heads, rotary_embedding_base=base)


def _spoil(rotary):
    rotary.inv_freq = torch.full_like(rotary.inv_freq, 7e34)
    rotary._cos, rotary._sin, rotary._sequence_length, rotary._cache_device = torch.ones(2), torch.ones(2), 99, "cpu"


def test_reinit_mert_rotary_restores_constructor_state(monkeypatch):
    remote = types.ModuleType("fake_modeling_mert2")
    remote.RotaryEmbedding = _RotaryEmbedding
    monkeypatch.setitem(sys.modules, remote.__name__, remote)
    monkeypatch.setattr(_Mert, "__module__", remote.__name__)

    config = _mert_config()
    model = _Mert(config)
    rotaries = [model.layers[0][0], model.embed_positions]
    for rot in rotaries:
        _spoil(rot)
    assert sem._reinit_mert_rotary(model) == 2

    reference = _RotaryEmbedding(config)
    for rot in rotaries:
        assert torch.equal(rot.inv_freq, reference.inv_freq) and rot.inv_freq.dtype == torch.float32
        assert "inv_freq" in dict(rot.named_buffers()) and "inv_freq" not in rot.state_dict()
        assert rot._cos is None and rot._sin is None and rot._sequence_length == 0 and rot._cache_device is None
    assert rotaries[0].inv_freq.data_ptr() != rotaries[1].inv_freq.data_ptr()
    lookalike = model.layers[1]
    assert torch.equal(lookalike.inv_freq, torch.full((4,), 3.0)) and torch.equal(lookalike._cos, torch.ones(2))


def test_reinit_mert_rotary_uses_given_class_and_config():
    model = _Mert(_mert_config())
    _spoil(model.embed_positions)
    other = _mert_config(hidden=64, heads=2, base=500)
    assert sem._reinit_mert_rotary(model, rotary_cls=_RotaryEmbedding, config=other) == 2
    assert torch.equal(model.embed_positions.inv_freq, _RotaryEmbedding(other).inv_freq)
    assert model.embed_positions.head_dim == 32 and model.embed_positions.base == 500


def test_reinit_mert_rotary_without_rotary_class_changes_nothing():
    model = _Mert(_mert_config())  # defined in this test module, which has no RotaryEmbedding
    _spoil(model.embed_positions)
    assert sem._reinit_mert_rotary(model) == 0
    assert torch.equal(model.embed_positions.inv_freq, torch.full((32,), 7e34))


def test_short_input_equals_a_direct_padded_pass():
    head = _tiny_head()
    feats = torch.randn(11, DIN)
    x = sem.instance_norm(feats)
    direct = head(F.pad(x, (0, 0, 0, WIN - 11))[None])[0, :11].argmax(-1)
    assert torch.equal(sem.classify_features(head, feats), direct)


def test_instance_norm_makes_codes_affine_invariant():
    head = _tiny_head(seed=3)
    feats = torch.randn(40, DIN, dtype=torch.float64).float()
    scale = torch.rand(DIN) * 4 + 0.5
    shift = torch.randn(DIN) * 10
    a = sem.classify_features(head, feats, instnorm=True)
    b = sem.classify_features(head, feats * scale + shift, instnorm=True)
    assert (a == b).float().mean() > 0.97  # argmax near-ties may flip under float rounding
    c = sem.classify_features(head, feats * scale + shift, instnorm=False)
    assert not torch.equal(a, c)
    x = sem.instance_norm(feats * scale + shift)
    assert torch.allclose(x.mean(0), torch.zeros(DIN), atol=1e-5)
    assert torch.allclose(x.std(0), torch.ones(DIN), atol=1e-3)


def test_load_head_pt_and_safetensors(tmp_path: Path):
    head = _tiny_head(seed=1)
    st = _save_head_safetensors(head, tmp_path / "head.safetensors")
    pt = tmp_path / "head.pt"
    torch.save({"model": head.state_dict(), "cfg": {"NAME": "x", "instnorm": True, "H": 2}}, pt)

    for path in (st, pt):
        loaded, cfg = sem.load_head(path)
        assert cfg["instnorm"] is True
        assert loaded.window == WIN and loaded.inp.in_features == DIN and len(loaded.enc.layers) == 2
        assert loaded.enc.layers[0].self_attn.num_heads == 2
        for (k, a), (_, b) in zip(sorted(head.state_dict().items()), sorted(loaded.state_dict().items())):
            assert torch.equal(a, b), k
        assert not loaded.training

    feats = torch.randn(30, DIN)
    assert torch.equal(sem.classify_features(sem.load_head(st)[0], feats), sem.classify_features(head, feats))

    # bf16 release: loaded as float32
    bf = _save_head_safetensors(head, tmp_path / "head.bf16.safetensors", dtype=torch.bfloat16)
    loaded, _ = sem.load_head(bf)
    assert loaded.inp.weight.dtype == torch.float32
    assert torch.equal(loaded.inp.weight, head.inp.weight.to(torch.bfloat16).float())

    # metadata cfg (tokenizer_head_v5_30k style) and an explicit instnorm=false
    v5 = _save_head_safetensors(head, tmp_path / "v5.safetensors", {"cfg": '{"H": 2, "instnorm": true}', "architecture": ""})
    assert sem.load_head(v5)[0].enc.layers[0].self_attn.num_heads == 2
    off = _save_head_safetensors(head, tmp_path / "off.safetensors", {"input": "raw features (instnorm=false)"})
    assert sem.load_head(off)[1]["instnorm"] is False

    save_file({"foo": torch.zeros(1)}, str(tmp_path / "bad.safetensors"))
    with pytest.raises(ValueError, match="not a semantic tokenizer head"):
        sem.load_head(tmp_path / "bad.safetensors")
    small = sem.SemanticHead(sem.HeadConfig(in_dim=DIN, width=16, depth=1, num_heads=2, vocab=100, window=WIN))
    save_file({k: v.contiguous() for k, v in small.state_dict().items()}, str(tmp_path / "small.safetensors"))
    with pytest.raises(ValueError, match="codec size"):
        sem.load_head(tmp_path / "small.safetensors")


class _Out:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class MockMERT:
    """Deterministic frame-local stand-in for MERT2: 960 samples -> one 1024-d frame, 24 'layers'."""

    def __init__(self, dim: int = DIN):
        g = torch.Generator().manual_seed(0)
        self.w = torch.randn(4, dim, generator=g)
        self.calls: list[tuple[int, int]] = []

    def frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        stats = torch.stack([frames.mean(-1), frames.std(-1), frames.abs().max(-1).values, frames[..., ::7].mean(-1)], -1)
        return stats @ self.w

    def __call__(self, input_values, attention_mask=None, output_hidden_states=False):
        b, length = input_values.shape
        self.calls.append((b, length))
        frames = input_values[:, : length // MERT_HOP * MERT_HOP].reshape(b, length // MERT_HOP, MERT_HOP)
        h = self.frame_features(frames.float())
        return _Out(tuple(h + i for i in range(-20, 4)))  # hidden_states[20] == h


def _semantic(tmp_path: Path, mert=None) -> sem.SemanticTokenizer:
    head = _save_head_safetensors(_tiny_head(seed=2), tmp_path / "head.safetensors")
    return sem.SemanticTokenizer(head, "mock/mert", device="cpu", dtype=None, mert=mert or MockMERT())


def _audio(seconds_or_samples, seed: int = 0) -> torch.Tensor:
    n = seconds_or_samples if isinstance(seconds_or_samples, int) else int(seconds_or_samples * MERT_SAMPLE_RATE)
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / MERT_SAMPLE_RATE
    return 0.3 * torch.sin(2 * torch.pi * 220 * t * (1 + 0.1 * torch.sin(t))) + 0.05 * torch.randn(n, generator=g)


def test_layer_features_chunking_and_exact_frames(tmp_path: Path):
    mert = MockMERT()
    st = _semantic(tmp_path, mert)
    frames = 1750  # 70 s: two full 30 s chunks batched together, a 10 s tail alone
    audio = _audio(frames * MERT_HOP)
    feats = st.layer_features(audio, frames)
    assert mert.calls == [(2, 30 * MERT_SAMPLE_RATE), (1, 10 * MERT_SAMPLE_RATE)]
    direct = mert.frame_features(audio.reshape(frames, MERT_HOP)).half().float()
    assert feats.shape == (frames, DIN)
    assert torch.allclose(feats, direct, rtol=2e-3, atol=1e-3)


def test_short_tail_is_encoded_from_the_last_second_not_stretched(tmp_path: Path):
    mert = MockMERT()
    st = _semantic(tmp_path, mert)
    frames = 1512  # 60.48 s: the 0.48 s tail is shorter than one second
    audio = _audio(frames * MERT_HOP, seed=1)
    feats = st.layer_features(audio, frames)
    assert mert.calls == [(2, 30 * MERT_SAMPLE_RATE), (1, MERT_SAMPLE_RATE)]
    direct = mert.frame_features(audio.reshape(frames, MERT_HOP)).half().float()
    assert torch.allclose(feats, direct, rtol=2e-3, atol=1e-3), "every frame keeps its own audio (no stretch)"
    # dropping the tail would stretch 1500 frames over 1512: frame 1400 would then see other audio
    stretched = F.interpolate(direct[:1500].T[None], size=frames, mode="linear", align_corners=False)[0].T
    assert not torch.allclose(stretched[1400], direct[1400])


class RecordingMERT:
    """Returns random layer-20 frames (one per 960 samples) and keeps them, so a test can rebuild the raw timeline."""

    def __init__(self, dim: int = DIN):
        self.gen = torch.Generator().manual_seed(7)
        self.dim = dim
        self.outputs: list[torch.Tensor] = []

    def __call__(self, input_values, attention_mask=None, output_hidden_states=False):
        b, length = input_values.shape
        h = torch.randn(b, length // MERT_HOP, self.dim, generator=self.gen)
        self.outputs.append(h)
        return _Out(tuple(h if i == 20 else torch.zeros_like(h) for i in range(24)))


@pytest.mark.parametrize("frames, size", [(1750, 1750), (100, 100), (1750, 1760), (1512, 1512)])
def test_layer_features_always_come_from_the_interpolation(tmp_path: Path, frames: int, size: int):
    mert = RecordingMERT()
    st = _semantic(tmp_path, mert)
    feats = st.layer_features(_audio(frames * MERT_HOP, seed=3), size)
    raw = torch.cat([h.reshape(-1, DIN) for h in mert.outputs], dim=0)
    if frames == 1512:
        raw = torch.cat([raw[:1500], raw[-12:]], dim=0)
    assert raw.shape[0] == frames
    expected = F.interpolate(raw.t().unsqueeze(0), size=size, mode="linear", align_corners=False)[0].t().half().float()
    assert torch.equal(feats, expected)
    assert feats.stride() == expected.stride() == (1, size)
    assert not feats.is_contiguous() and feats.t().is_contiguous()


def _reference_codes(st: sem.SemanticTokenizer, mert: MockMERT, audio: torch.Tensor, frames: int) -> torch.Tensor:
    # the heads' training feature grid: all MERT frames resized to round(len/960), the first `frames` classified
    length = audio.shape[0]
    h = mert.frame_features(audio[: length // MERT_HOP * MERT_HOP].reshape(-1, MERT_HOP))
    size = max(round(length / MERT_HOP), frames)
    feats = F.interpolate(h.T[None], size=size, mode="linear", align_corners=False)[0].T[:frames].half().float()
    return sem.classify_features(st.head, feats)


@pytest.mark.parametrize("extra", [500, 0, -300, -959])
def test_tokenize_frames_follows_the_reference_feature_grid(tmp_path: Path, extra: int):
    mert = MockMERT()
    st = _semantic(tmp_path, mert)
    frames = 100
    audio = _audio(frames * MERT_HOP + extra, seed=2)
    codes = st.tokenize_frames(audio, frames)
    assert codes.shape == (frames,) and codes.dtype == torch.int64 and codes.device.type == "cpu"
    ref = _reference_codes(st, mert, audio, frames)
    assert (codes == ref).float().mean() >= 0.98  # argmax near-ties may flip under batched-matmul rounding


def test_tokenize_frames_errors(tmp_path: Path):
    st = _semantic(tmp_path)
    frames = 100
    audio = _audio(frames * MERT_HOP, seed=2)
    with pytest.raises(ValueError, match="shorter"):
        st.tokenize_frames(audio[: (frames - 2) * MERT_HOP], frames)
    with pytest.raises(ValueError, match="at least one second"):
        st.tokenize_frames(audio[: 20 * MERT_HOP], 20)
    with pytest.raises(ValueError, match="non-finite"):
        bad = audio.clone()
        bad[5] = float("nan")
        st.layer_features(bad, frames)


def test_fingerprints(tmp_path: Path):
    head = _save_head_safetensors(_tiny_head(), tmp_path / "head.safetensors")
    fp = sem.semantic_fingerprint(head, "m-a-p/MERT-v2-FullSong", None)
    assert fp["yue2_semantic_head"].startswith("sha256:") and fp["yue2_semantic_mert"] == "m-a-p/MERT-v2-FullSong@main"
    assert fp["yue2_semantic_version"] == sem.SEMANTIC_VERSION
    local = tmp_path / "m-a-p__MERT-v2-FullSong"
    (local / ".cache" / "huggingface" / "download").mkdir(parents=True)
    assert sem.mert_identity(str(local)) == "m-a-p__MERT-v2-FullSong@local"
    (local / ".cache" / "huggingface" / "download" / "config.json.metadata").write_text("abc123\netag\n1.0\n")
    assert sem.mert_identity(str(local)) == "m-a-p__MERT-v2-FullSong@abc123"
    assert sem.mert_identity(str(local), "v2") == "m-a-p__MERT-v2-FullSong@v2"
    st = sem.SemanticTokenizer(head, "m-a-p/MERT-v2-FullSong", device="cpu", dtype=None, mert=MockMERT())
    assert st.fingerprint == fp


def _torchaudio_importable() -> bool:
    try:
        import torchaudio.transforms  # noqa: F401

        return True
    except (ImportError, OSError):  # OSError: a wheel built for another torch
        return False


def test_torchaudio_stand_in():
    if _torchaudio_importable():
        pytest.skip("torchaudio is installed; the stand-in is not used")
    before = (sys.modules.get("torchaudio"), sys.modules.get("torchaudio.transforms"))
    with sem.torchaudio_transforms_available() as stand_in:
        assert stand_in is True
        from torchaudio.transforms import AmplitudeToDB, MelScale, Spectrogram

        spec = Spectrogram(n_fft=2048, win_length=2048, hop_length=240, power=2.0)
        mel = MelScale(n_mels=128, sample_rate=24000, n_stft=1025)
        db = AmplitudeToDB(stype="power", top_db=None)
    assert (sys.modules.get("torchaudio"), sys.modules.get("torchaudio.transforms")) == before

    assert set(spec.state_dict()) == {"window"} and set(mel.state_dict()) == {"fb"}
    assert torch.equal(spec.window, torch.hann_window(2048))
    t = torch.arange(24000) / 24000
    x = torch.sin(2 * torch.pi * 1000 * t)[None]
    s = spec(x)
    assert s.shape == (1, 1025, 24000 // 240 + 1)
    assert int(s[0, :, 50].argmax()) == round(1000 / (24000 / 2048))
    assert mel.fb.shape == (1025, 128) and float(mel.fb.min()) >= 0
    m = mel(s)
    assert m.shape == (1, 128, s.shape[-1])
    assert torch.allclose(m, (s.transpose(-1, -2) @ mel.fb).transpose(-1, -2))
    v = torch.tensor([1e-12, 1e-3, 1.0, 100.0])
    assert torch.allclose(db(v), torch.tensor([-100.0, -30.0, 0.0, 20.0]))


def test_torchaudio_stand_in_matches_torchaudio():
    if not _torchaudio_importable():
        pytest.skip("torchaudio is not importable")
    import torchaudio.transforms as ta

    x = torch.randn(2, 24000)
    ours, theirs = (
        sem._Spectrogram(n_fft=2048, win_length=2048, hop_length=240),
        ta.Spectrogram(n_fft=2048, win_length=2048, hop_length=240),
    )
    assert torch.allclose(ours(x), theirs(x), rtol=1e-5, atol=1e-5)
    om, tm = sem._MelScale(n_mels=128, sample_rate=24000, n_stft=1025), ta.MelScale(n_mels=128, sample_rate=24000, n_stft=1025)
    assert torch.allclose(om.fb, tm.fb, atol=1e-6)
    v = theirs(x)
    assert torch.allclose(sem._AmplitudeToDB()(v), ta.AmplitudeToDB(stype="power", top_db=None)(v))


# --- real weights (YUE2_WEIGHTS_DIR) ----------------------------------------------------------------------------------

WEIGHTS = os.environ.get("YUE2_WEIGHTS_DIR")
DATA = os.environ.get("YUE2_DATA_DIR")
HEAD_REPO_DIR = "Mothersuperior__yue2-mothersuperior-realaudio-tokenizer-v4"


@pytest.mark.skipif(not (WEIGHTS and DATA and torch.cuda.is_available()), reason="needs YUE2_WEIGHTS_DIR, YUE2_DATA_DIR and CUDA")
def test_real_semantic_tokenizer():
    from musubi_tuner.dataset.audio_utils import AudioSource, decode_audio

    mp3 = Path(DATA) / "jamendolyrics" / "mp3" / "10._Disparan_-_criatura.mp3"
    wave = decode_audio(AudioSource(mp3, False), sample_rate=MERT_SAMPLE_RATE, channels=2).mean(0)
    frames = 1000  # 40 s from t=30 s
    mono = wave[30 * MERT_SAMPLE_RATE : 30 * MERT_SAMPLE_RATE + frames * MERT_HOP]
    mert = str(Path(WEIGHTS) / "m-a-p__MERT-v2-FullSong")
    st = sem.SemanticTokenizer(Path(WEIGHTS) / HEAD_REPO_DIR / "tokenizer_head_joint_v4.safetensors", mert, device="cuda")
    # transformers 5 leaves MERT2's non-persistent inv_freq as garbage (seen: 7e34 off); the loader reinitialises it
    # from a fresh instance of MERT2's own rotary class
    rotary_cls = getattr(sys.modules[type(st.mert).__module__], "RotaryEmbedding")
    rotary = [m for m in st.mert.modules() if hasattr(m, "inv_freq") and hasattr(m, "base")]
    assert rotary and all(isinstance(m, rotary_cls) for m in rotary)
    fresh = rotary_cls(st.mert.config)
    for m in rotary:
        assert torch.equal(m.inv_freq.cpu(), fresh.inv_freq)
        expected = 1.0 / (m.base ** (torch.arange(0, m.head_dim, 2, dtype=torch.float32) / m.head_dim))
        assert torch.allclose(m.inv_freq.cpu(), expected)
    feats = st.layer_features(mono, frames)
    assert feats.shape == (frames, 1024) and torch.isfinite(feats).all()
    codes = st.tokenize_features(feats)
    assert codes.shape == (frames,) and int(codes.max()) < CODEC_SIZE
    assert codes.unique().numel() > 100
    assert torch.equal(codes, st.tokenize_frames(mono, frames))  # deterministic
    bf16_head, _ = sem.load_head(Path(WEIGHTS) / HEAD_REPO_DIR / "tokenizer_head_joint_v4.bf16.safetensors")
    bf16_codes = sem.classify_features(bf16_head.cuda(), feats).cpu()
    agree = (bf16_codes == codes).float().mean().item()
    print(f"v4 fp32 vs bf16 head agreement: {agree:.4f}, unique {codes.unique().numel()}/{frames}")
    assert agree > 0.9  # the head model card reports 98.6% top-1 agreement
