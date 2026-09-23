"""YuE2 sampling / inference: AR sampler vs the official ``yue2_infer`` 0.1.5 ``sampling.py``, midpoint ODE vs
its ``nar.py`` on a tiny model, song rendering with fakes, audio I/O. No weights; the real-weight comparison is a
separate GPU script."""

import math
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import FakeTokenizer, FakeYuE2VAE, make_ref_model, save_ref_hf  # noqa: E402

from musubi_tuner.yue2 import yue2_sampling as S  # noqa: E402
from musubi_tuner.yue2.yue2_audio_io import read_audio, write_audio  # noqa: E402
from musubi_tuner.yue2.yue2_checkpoint import load_yue2_model  # noqa: E402
from musubi_tuner.yue2.yue2_model import YuE2AROutput, YuE2Config, YuE2Model, yue2_t_embed_input  # noqa: E402
from musubi_tuner.yue2.yue2_protocol import (  # noqa: E402
    ABC_END,
    ABC_START,
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    EOD,
    MUSIC_END,
    MUSIC_START,
    VOCAB_SIZE,
    SamplingParams,
    build_prefix,
    negative_text_ids,
    text_ids,
)

TINY = YuE2Config.tiny()

# region direct ports of the official yue2_infer 0.1.5 (sampling.py, nar.py), verbatim semantics; see THIRD_PARTY_NOTICES.md


def _upstream_window_penalty(logits, recent_ids, penalty):
    if penalty == 1.0 or len(recent_ids) == 0:
        return logits
    recent = torch.as_tensor(recent_ids, dtype=torch.long, device=logits.device).reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
    alpha = penalty**freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _upstream_distribution(logits, sampling, history, step, phase, legacy_off=False):
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores = scores + allowed
    if step < sampling.min_tokens:
        scores[..., end] = -torch.inf
    scores = _upstream_window_penalty(scores, history[-sampling.penalty_window :], sampling.repetition_penalty)
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / sampling.temperature
    threshold = scores.topk(min(sampling.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if sampling.top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
        removed[..., : 3 if legacy_off else 1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


@torch.no_grad()
def _upstream_generate(model, prefix, sampling, seed, phase, negative=None, cfg_scale=1.0, legacy_off=False):
    """Upstream eager ``generate_tokens`` loop; ``model(ids, cache)`` returns last-position logits."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def prefill(ids):
        cache = model.new_kv_cache(len(ids) + sampling.max_tokens)
        return model.ar_prefill_into_cache(torch.tensor([ids]), cache), cache

    conditional, positive_cache = prefill(prefix)
    unconditional = negative_cache = None
    if cfg_scale != 1.0:
        unconditional, negative_cache = prefill(negative)
    history, eos = [], False
    end = ABC_END if phase == "abc" else MUSIC_END
    for step in range(sampling.max_tokens):
        logits = conditional if cfg_scale == 1.0 else unconditional + cfg_scale * (conditional - unconditional)
        scores = _upstream_distribution(logits, sampling, history, step, phase, legacy_off)
        if sampling.temperature == 0:
            next_id = scores.argmax(-1, keepdim=True)
        else:
            next_id = torch.multinomial(scores.softmax(-1), 1, generator=generator)
        token = int(next_id.item())
        if token == end:
            eos = True
            break
        history.append(token)
        if step + 1 < sampling.max_tokens:
            conditional = model.ar_decode_step(next_id, positive_cache)
            if negative_cache is not None:
                unconditional = model.ar_decode_step(next_id, negative_cache)
    return history, not eos


def _upstream_song_chunks(prefix, codec, seed, context=CONTEXT):
    size = min((context - len(prefix) - 3) // 2, CONTEXT)
    ranges = [(a, min(a + size, len(codec))) for a in range(0, len(codec), size)]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)
    return [(prefix + [value + CODEC_OFFSET for value in codec[a:b]] + [MUSIC_END], noise[a:b]) for a, b in ranges]


def _upstream_attention(q, k, v, causal=False):
    block = len(q) if q.device.type == "cuda" else 256
    query, key, value = q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)
    grouped = query.shape[1] != key.shape[1]
    outputs = []
    for start in range(0, len(q), block):
        end = min(start + block, len(q))
        used_key = key[..., :end, :] if causal else key
        used_value = value[..., :end, :] if causal else value
        mask = None
        if causal and start:
            mask = torch.arange(end, device=q.device)[None, :] <= torch.arange(start, end, device=q.device)[:, None]
        outputs.append(
            F.scaled_dot_product_attention(
                query[..., start:end, :], used_key, used_value, attn_mask=mask, is_causal=causal and start == 0, enable_gqa=grouped
            )
        )
    return torch.cat(outputs, dim=-2)[0].transpose(0, 1)


class _UpstreamCachedNAR:
    def __init__(self, model, ar_tokens, noise):
        self.model, self.ar_tokens, self.noise = model, ar_tokens, noise
        weight = next(model.vae2llm.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        self.ar_length, self.nar_length = len(ar_tokens), len(noise) + 2
        positions = torch.arange(self.ar_length, self.ar_length + self.nar_length, device=self.device)[None]
        self.cos, self.sin = model.model.rotary_emb(positions)
        local = torch.arange(self.nar_length, device=self.device).clamp(max=model.config.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.cache = []
        self._prefill()

    @torch.no_grad()
    def _prefill(self):
        backbone = self.model.model
        ids = torch.tensor([self.ar_tokens], dtype=torch.long, device=self.device)
        cos, sin = backbone.rotary_emb(torch.arange(self.ar_length, device=self.device)[None])
        x = backbone.embed_tokens(ids)
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            self.cache.append((k[0], v[0]))
            h = _upstream_attention(q[0], k[0], v[0], causal=True)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))

    @torch.no_grad()
    def velocity(self, state, raw_t):
        model = self.model
        x_nar = F.pad(state, (0, 0, 1, 1))
        shifted = model._shift_t_value(raw_t, self.device, self.dtype)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(shifted.expand(self.nar_length))[None]
        x = x + self.pos_emb
        for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache):
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
            k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
            h = _upstream_attention(q[0], k, v)
            x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        return model.llm2vae(model.model.norm(x))[0, 1:-1]

    @torch.no_grad()
    def solve(self, steps=32):
        state = self.noise.to(device=self.device, dtype=self.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            first = self.velocity(state, raw)
            mid = state - first * (dt / 2)
            raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            state = state - self.velocity(mid, raw_mid) * dt
        return state.float().cpu()


# endregion

# region fakes


class _FakeCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.ids = []
        self.prefix_len = None


class FakeLM:
    """AR/NAR stand-in with the ``YuE2Model`` inference API. Logits ``[1, VOCAB_SIZE]`` (bf16) are a deterministic
    function of the cached sequence; ``end_after`` puts a large logit on both end tokens after that many generated
    tokens. Every call asserts that inference mode is off."""

    def __init__(self, end_after=None, dtype=torch.bfloat16, strict=True):
        self.device = torch.device("cpu")
        self.compute_dtype = dtype
        self.end_after = end_after
        self.prefills = []
        self.decodes = 0
        self.checks = 0
        self.nar_calls = []
        self.ar_contexts = []
        self.strict = strict

    def _check(self):
        if self.strict:
            assert not torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        self.checks += 1

    def new_kv_cache(self, capacity, batch=1):
        self._check()
        return _FakeCache(capacity)

    def _logits(self, cache):
        seq = cache.ids
        key = (sum((i + 1) * t for i, t in enumerate(seq[-8:])) + 7919 * len(seq)) % (2**31 - 1)
        logits = torch.randn(1, VOCAB_SIZE, generator=torch.Generator().manual_seed(key)) * 3
        if self.end_after is not None and len(seq) - cache.prefix_len >= self.end_after:
            logits[0, MUSIC_END] = 60.0
            logits[0, ABC_END] = 60.0
        return logits.to(self.compute_dtype)

    def ar_prefill_into_cache(self, ids, cache):
        self._check()
        cache.ids = [int(t) for t in ids[0].tolist()]
        cache.prefix_len = len(cache.ids)
        self.prefills.append(list(cache.ids))
        assert len(cache.ids) <= cache.capacity
        return self._logits(cache)

    def ar_decode_step(self, ids, cache):
        self._check()
        assert ids.shape == (1, 1)
        cache.ids.append(int(ids.item()))
        assert len(cache.ids) <= cache.capacity
        self.decodes += 1
        return self._logits(cache)

    def embed(self, ids):
        self._check()
        self.ar_contexts.append([int(t) for t in ids[0].tolist()])
        return torch.zeros(1, ids.shape[1], 4, dtype=self.compute_dtype)

    def ar_forward(self, embeds, *, return_kv=False, kv_visible=None, return_hidden=True, valid_len=None):
        self._check()
        n = embeds.shape[1] if kv_visible is None else min(kv_visible, embeds.shape[1])
        return YuE2AROutput(hidden=None, kv=[("kv", n)])

    def nar_forward(self, x_t, t_embed, kv, rope_offset):
        self._check()
        self.nar_calls.append((float(t_embed.reshape(-1)[0]), kv[0][1], rope_offset, x_t.shape[1]))
        return (0.5 * x_t.float() + float(t_embed.reshape(-1)[0])).to(self.compute_dtype)


class FieldModel(FakeLM):
    """NAR velocity ``v = a + b * x`` (independent of t) for exact midpoint checks."""

    def __init__(self, a=0.0, b=0.0):
        super().__init__(dtype=torch.float32)
        self.a, self.b = a, b
        self.ts = []

    def nar_forward(self, x_t, t_embed, kv, rope_offset):
        self._check()
        self.ts.append(float(t_embed.reshape(-1)[0]))
        return self.a + self.b * x_t


# endregion

# region distribution / window penalty


def _random_logits(seed, dtype=torch.float32):
    return (torch.randn(1, VOCAB_SIZE, generator=torch.Generator().manual_seed(seed)) * 4).to(dtype)


@pytest.mark.parametrize("phase", ["abc", "semantic"])
@pytest.mark.parametrize("legacy_off", [False, True])
@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(1.0, 0.95, 100, 1.2, 50, 200, 9000),
        SamplingParams(0.7, 0.9, 30, 1.005, 100, 32, 4096),
        SamplingParams(1.3, 1.0, 5, 0.8, 3, 0, 10),
        SamplingParams(0.0, 0.5, 7, 1.5, 10, 0, 10),
    ],
)
def test_distribution_matches_upstream(phase, legacy_off, params):
    dtype = torch.bfloat16 if legacy_off else torch.float32
    logits = _random_logits(3, dtype)
    lo, hi = (0, EOD) if phase == "abc" else (CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE)
    g = torch.Generator().manual_seed(1)
    history = torch.randint(lo, hi, (120,), generator=g).tolist()
    history += history[:10]  # repeats
    for step in (0, 5, 31, 199, 250):
        got = S.distribution(logits, params, history, step, phase, legacy_off)
        want = _upstream_distribution(logits, params, history, step, phase, legacy_off)
        assert got.dtype == want.dtype
        assert torch.equal(got, want)


def test_distribution_masks_and_min_tokens():
    params = SamplingParams(1.0, 1.0, VOCAB_SIZE, 1.0, 50, 10, 100)
    logits = _random_logits(0)
    sem = S.distribution(logits, params, [], 0, "semantic")
    finite = torch.isfinite(sem[0]).nonzero().flatten()
    assert finite.min() >= CODEC_OFFSET and finite.max() < CODEC_OFFSET + CODEC_SIZE  # end masked before min_tokens
    assert torch.isfinite(S.distribution(logits, params, [], 10, "semantic")[0, MUSIC_END])
    abc = S.distribution(logits, params, [], 10, "abc")
    finite = set(torch.isfinite(abc[0]).nonzero().flatten().tolist())
    assert finite == set(range(EOD)) | {ABC_END}
    assert not torch.isfinite(S.distribution(logits, params, [], 9, "abc")[0, ABC_END])


def test_window_penalty_sign_and_window():
    logits = torch.tensor([[2.0, -2.0, 1.0, -1.0]])
    out = S.window_penalty(logits, [0, 1, 0], 2.0)
    assert torch.allclose(out, torch.tensor([[2.0 / 4, -2.0 * 2, 1.0, -1.0]]))
    assert S.window_penalty(logits, [0], 1.0) is logits
    params = SamplingParams(1.0, 1.0, VOCAB_SIZE, 2.0, 2, 0, 10)
    big = torch.zeros(1, VOCAB_SIZE)
    big[0, CODEC_OFFSET : CODEC_OFFSET + 3] = torch.tensor([1.0, 1.0, 1.0])
    out = S.distribution(big, params, [CODEC_OFFSET, CODEC_OFFSET + 1, CODEC_OFFSET + 2], 0, "semantic")
    # only the last two history ids are penalised
    assert out[0, CODEC_OFFSET] == 1.0 and out[0, CODEC_OFFSET + 1] == 0.5 and out[0, CODEC_OFFSET + 2] == 0.5


# endregion

# region generate_tokens


@pytest.mark.parametrize("cfg", [1.0, 1.5])
@pytest.mark.parametrize("legacy_off", [False, True])
def test_generate_tokens_matches_upstream_loop(cfg, legacy_off):
    params = SamplingParams(1.0, 0.95, 100, 1.2, 50, 5, 40)
    prefix = [EOD, 10, 20, 30, ABC_START, ABC_END, MUSIC_START]
    negative = [EOD, 10, MUSIC_START]
    ours_model, ref_model = FakeLM(end_after=25), FakeLM(end_after=25)
    timing = {}
    ids, truncated = S.generate_tokens(
        ours_model,
        prefix,
        params,
        123,
        "semantic",
        negative=negative if cfg != 1 else None,
        cfg_scale=cfg,
        legacy_off=legacy_off,
        timing=timing,
    )
    want, want_trunc = _upstream_generate(
        ref_model, prefix, params, 123, "semantic", negative if cfg != 1 else None, cfg, legacy_off
    )
    assert ids == want and truncated == want_trunc
    assert all(CODEC_OFFSET <= t < CODEC_OFFSET + CODEC_SIZE for t in ids)
    assert not truncated and len(ids) == 25  # stopped at MUSIC_END, which is not returned
    assert timing["cfg_branches"] == (1 if cfg == 1 else 2) and timing["content_tokens"] == 25
    if cfg != 1:
        assert ours_model.prefills == [prefix, negative]  # both caches used
        assert ours_model.decodes == 2 * 25
    else:
        assert ours_model.prefills == [prefix]


def test_generate_tokens_seed_budget_and_abc_range():
    params = SamplingParams(1.0, 0.95, 50, 1.0, 50, 0, 30)
    prefix = [EOD, 1, 2, ABC_START]
    a, ta = S.generate_tokens(FakeLM(), prefix, params, 5, "abc")
    b, _ = S.generate_tokens(FakeLM(), prefix, params, 5, "abc")
    c, _ = S.generate_tokens(FakeLM(), prefix, params, 6, "abc")
    assert a == b and a != c
    assert ta and len(a) == 30  # budget exhausted -> truncated
    assert all(0 <= t < EOD for t in a)
    with pytest.raises(ValueError):
        S.generate_tokens(FakeLM(), list(range(CONTEXT - 10)), params, 0, "semantic")
    with pytest.raises(ValueError):
        S.generate_tokens(FakeLM(), prefix, params, 0, "semantic", cfg_scale=1.5)


# endregion

# region ODE


def test_midpoint_solver_integrates_constant_and_linear_fields_exactly():
    noise = torch.randn(9, 64, generator=torch.Generator().manual_seed(0))
    model = FieldModel(a=0.75)
    out = S.solve_chunk(model, [1, 2, 3], noise, steps=32, state_dtype="fp32")
    assert torch.allclose(out, noise - 0.75, atol=1e-6)
    steps = 16
    dt = 1.0 / steps
    out = S.solve_chunk(FieldModel(b=1.0), [1, 2, 3], noise, steps=steps, state_dtype="fp32")
    assert torch.allclose(out, noise * (1 - dt + dt * dt / 2) ** steps, rtol=1e-5, atol=1e-6)
    # time grid: t_s = 1 - s/steps and the midpoints, through the shared bf16 t-embed helper
    expected = []
    for s in range(steps):
        t = 1 - s * dt
        expected += [t, t - dt / 2]
    want = [float(yue2_t_embed_input(torch.tensor([t], dtype=torch.float64))) for t in expected]
    model = FieldModel()
    S.solve_chunk(model, [1], noise, steps=steps, state_dtype="fp32")
    assert model.ts == want


def test_song_chunks_match_upstream_and_noise_drawn_once():
    prefix = [EOD] + list(range(1, 10))
    codes = torch.randint(0, CODEC_SIZE, (50,), generator=torch.Generator().manual_seed(2)).tolist()
    context = len(prefix) + 3 + 2 * 20  # chunk size 20 -> 3 chunks
    ours = S.song_chunks(prefix, codes, 77, context)
    want = _upstream_song_chunks(prefix, codes, 77, context)
    assert len(ours) == 3
    for (a_ids, a_noise), (b_ids, b_noise) in zip(ours, want):
        assert a_ids == b_ids and torch.equal(a_noise, b_noise)
    assert torch.equal(torch.cat([n for _, n in ours]), S.song_noise(50, 77))
    with pytest.raises(ValueError):
        S.song_chunks(prefix, [CODEC_SIZE], 0)


@pytest.fixture(scope="module")
def ref_and_ours(tmp_path_factory):
    pytest.importorskip("transformers")
    ref = make_ref_model(TINY, seed=0)
    path = save_ref_hf(tmp_path_factory.mktemp("yue2s") / "hf.safetensors", ref)
    return ref, path


def test_ode_midpoint_matches_upstream_nar_fp32(ref_and_ours):
    ref, path = ref_and_ours
    ours = load_yue2_model(path, device="cpu", loading_device="cpu", dtype=torch.float32, config=TINY)
    g = torch.Generator().manual_seed(4)
    ctx = torch.randint(0, TINY.vocab_size, (37,), generator=g).tolist()
    noise = torch.randn(12, 64, generator=g)
    want = _UpstreamCachedNAR(ref, ctx, noise).solve(steps=8)
    # fp32 upstream feeds sigmoid(logit(t)) in fp32 to the time embedder, i.e. t up to fp32 rounding
    got = S.solve_chunk(ours, ctx, noise, steps=8, state_dtype="fp32", t_embed_mode="fp32")
    assert (got - want).abs().max().item() < 1e-4


def test_ode_midpoint_matches_upstream_nar_bf16(ref_and_ours):
    ref, path = ref_and_ours
    ours = load_yue2_model(path, device="cpu", loading_device="cpu", dtype=torch.bfloat16, config=TINY)
    ref_bf16 = make_ref_model(TINY, seed=0).to(torch.bfloat16)
    g = torch.Generator().manual_seed(5)
    ctx = torch.randint(0, TINY.vocab_size, (29,), generator=g).tolist()
    noise = torch.randn(10, 64, generator=g)
    want = _UpstreamCachedNAR(ref_bf16, ctx, noise).solve(steps=8)
    got = S.solve_chunk(ours, ctx, noise, steps=8)  # bf16 state and bf16 t-embed, as upstream
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 2e-2, rel


@pytest.fixture(scope="module")
def full_vocab_model():
    # tiny dimensions but the real vocabulary, so protocol ids (codec tokens, MUSIC_END) embed
    config = YuE2Config.tiny(
        hidden_size=64, num_heads=2, num_kv_heads=1, head_dim=32, intermediate_size=128, num_layers=2, vocab_size=VOCAB_SIZE
    )
    torch.manual_seed(0)
    model = YuE2Model(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape) / p.shape[1] ** 0.5)
            elif name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape))
            else:
                p.copy_(0.1 * torch.randn(p.shape))
    return model.float().eval()


def test_synthesize_latents_tiny_model_finite_and_chunk_invariant(full_vocab_model):
    model = full_vocab_model
    prefix = [EOD, 5, 6, 7, ABC_START, ABC_END, MUSIC_START]
    codes = torch.randint(0, CODEC_SIZE, (30,), generator=torch.Generator().manual_seed(0)).tolist()
    lat = S.synthesize_latents(model, prefix, codes, seed=9, steps=4, state_dtype="fp32")
    assert lat.shape == (30, 64) and lat.dtype == torch.float32 and torch.isfinite(lat).all()
    # one chunk (T below the chunk size): identical to solving that chunk directly
    ctx, noise = S.song_chunks(prefix, codes, 9)[0]
    assert ctx == prefix + [c + CODEC_OFFSET for c in codes] + [MUSIC_END]
    direct = S.solve_chunk(model, ctx, noise, steps=4, state_dtype="fp32")
    assert torch.equal(lat, direct)
    # several chunks: each chunk is solved on its own context with its noise slice
    context = len(prefix) + 3 + 2 * 12
    lat3 = S.synthesize_latents(model, prefix, codes, seed=9, steps=4, state_dtype="fp32", context=context)
    parts = [S.solve_chunk(model, c, n, steps=4, state_dtype="fp32") for c, n in S.song_chunks(prefix, codes, 9, context)]
    assert torch.equal(lat3, torch.cat(parts))
    assert not torch.equal(lat3, lat)


def test_text_only_context_and_rope_layouts():
    prefix = [EOD, 1, 2, ABC_START, ABC_END, MUSIC_START]
    codes = list(range(7))
    for rope, offset in (("full", len(prefix) + 7 + 1), ("compact", len(prefix))):
        model = FakeLM()
        S.synthesize_latents(model, prefix, codes, 1, steps=2, nar_context="text_only", text_only_rope=rope)
        assert model.ar_contexts == [prefix]
        assert {c[1] for c in model.nar_calls} == {len(prefix)} and {c[2] for c in model.nar_calls} == {offset}
    model = FakeLM()
    S.synthesize_latents(model, prefix, codes, 1, steps=2)
    assert model.ar_contexts == [prefix + [c + CODEC_OFFSET for c in codes] + [MUSIC_END]]
    assert {c[2] for c in model.nar_calls} == {len(prefix) + 8}


# endregion

# region songs


def _request(**kw):
    base = dict(style="indie rock", lyrics="la la", seed=11, ode_steps=2, vae_core_frames=8, vae_halo_frames=2)
    base.update(kw)
    return S.YuE2SongRequest(**base)


def test_render_song_cot_off_uses_cfg_negative_and_decodes():
    model, tok, vae = FakeLM(end_after=12), FakeTokenizer(), FakeYuE2VAE()
    res = S.render_song(model, vae, tok, _request(cot="off", semantic=SamplingParams(min_tokens=3, max_tokens=50)))
    text = text_ids(tok, "indie rock", "la la", "off")
    assert res.prefix == build_prefix(text, "off", None) and res.cfg_scale == 1.01
    assert model.prefills[0] == res.prefix and model.prefills[1] == negative_text_ids(tok, "off") + [MUSIC_START]
    assert len(res.codes) == 12 and all(0 <= c < CODEC_SIZE for c in res.codes)
    assert res.latents.shape == (12, 64) and res.waveform.shape == (2, 12 * 1920 - 64)
    assert res.waveform.abs().max() <= 1.0 and not res.truncated["semantic"]
    assert model.checks > 0  # every model call ran outside inference mode, under no_grad
    assert {"abc", "semantic", "nar_seconds", "vae_seconds", "e2e_seconds"} - set(res.timings) == {"abc"}


def test_render_song_plans_abc_then_semantic():
    model, tok = FakeLM(end_after=6), FakeTokenizer()
    res = S.render_song(model, FakeYuE2VAE(), tok, _request(cot="full", abc_params=SamplingParams(0.7, 0.9, 30, 1.005, 100, 2, 40)))
    text = text_ids(tok, "indie rock", "la la", "full")
    assert model.prefills[0] == build_prefix(text, "full", None)  # ABC phase: text + [ABC_START]
    assert len(res.abc_ids) == 6 and all(0 <= t < EOD for t in res.abc_ids)
    assert res.prefix == build_prefix(text, "full", res.abc_ids)
    assert res.prefix[-2:] == [ABC_END, MUSIC_START] and res.cfg_scale == 1.0 and res.negative is None
    assert len(model.prefills) == 2 and res.abc_text == tok.decode(res.abc_ids)


def test_render_song_caps_budget_for_long_prefix_and_seconds():
    tok = FakeTokenizer()
    lyrics = "x" * 16000
    model = FakeLM(end_after=3)
    res = S.render_song(model, FakeYuE2VAE(), tok, _request(cot="off", lyrics=lyrics, semantic=SamplingParams(min_tokens=0)))
    budget = CONTEXT - len(res.prefix)
    assert len(res.prefix) > 16000 and budget < 9000
    assert res.budget["semantic"] == {"requested": 9000, "used": budget}
    model = FakeLM()
    res = S.render_song(model, FakeYuE2VAE(), tok, _request(cot="off", seconds=0.4, semantic=SamplingParams(min_tokens=0)))
    assert len(res.codes) == 10 and res.truncated["semantic"] and "semantic" not in res.budget


def test_reconstruct_mode_and_no_inference_mode_anywhere():
    model, tok = FakeLM(), FakeTokenizer()
    codes = list(range(100, 125))
    res = S.render_song(model, FakeYuE2VAE(), tok, _request(cot="full", mode="reconstruct", codes=codes))
    # no ABC for cot=full -> the NAR context falls back to cot=off
    assert res.cot == "off" and res.prefix == build_prefix(text_ids(tok, "indie rock", "la la", "off"), "off", None)
    assert model.prefills == [] and res.codes == codes and res.latents.shape == (25, 64)
    res = S.render_song(model, FakeYuE2VAE(), tok, _request(cot="melody", mode="reconstruct", codes=codes, abc="X:1\nK:C\nCDEF|"))
    assert res.prefix[-2:] == [ABC_END, MUSIC_START] and ABC_START in res.prefix
    with pytest.raises(ValueError):
        S.YuE2SongRequest(mode="reconstruct")
    # a caller that enables inference mode is not the module's doing; the module itself never does
    src = (ROOT / "src" / "musubi_tuner" / "yue2" / "yue2_sampling.py").read_text(encoding="utf-8")
    assert "inference_mode()" not in src and "@torch.inference_mode" not in src


def test_grad_ctx_is_honoured():
    calls = []

    def ctx():
        calls.append(1)
        return torch.no_grad()

    S.reconstruct(FakeLM(), FakeYuE2VAE(), [EOD, MUSIC_START], [1, 2, 3], 0, steps=1, grad_ctx=ctx)
    assert calls
    with pytest.raises(AssertionError):  # a grad-enabled context reaches the model (FakeLM asserts no_grad)
        S.reconstruct(FakeLM(), FakeYuE2VAE(), [EOD, MUSIC_START], [1, 2, 3], 0, steps=1, grad_ctx=nullcontext)


def test_load_reconstruct_cache(tmp_path):
    from safetensors.torch import save_file

    lat = tmp_path / "song_000000-000010_yue2.safetensors"
    save_file({"latents_10x64_float32": torch.randn(10, 64), "codes_int64": torch.arange(10)}, str(lat))
    src = S.load_reconstruct_cache(str(lat))
    assert src.codes == list(range(10)) and src.latents.shape == (10, 64) and src.text_cache_path is None
    te = {f"varlen_yue2_text_{c}_int64": torch.tensor([EOD, i + 1]) for i, c in enumerate(("off", "melody", "full"))}
    te.update({f"varlen_yue2_neg_{c}_int64": torch.tensor([EOD]) for c in ("off", "melody", "full")})
    te.update(
        {
            "varlen_yue2_abc_int64": torch.tensor([7, 8]),
            "yue2_has_abc_int64": torch.tensor(1),
            "yue2_abc_mode_int64": torch.tensor(2),
        }
    )
    save_file(te, str(tmp_path / "song_yue2_te.safetensors"))
    src = S.load_reconstruct_cache(str(lat))
    assert src.text_ids["full"] == [EOD, 3] and src.abc_ids == [7, 8] and src.abc_mode == "full"
    assert S.text_cache_path_for("/a/my_song_000100-000750_yue2.safetensors").replace("\\", "/") == "/a/my_song_yue2_te.safetensors"


# endregion

# region audio io


@pytest.mark.parametrize("fmt", ["flac", "wav"])
def test_write_audio_round_trip(tmp_path, fmt):
    t = torch.arange(48000 * 2) / 48000
    wave = torch.stack([0.5 * torch.sin(2 * math.pi * 440 * t), 0.3 * torch.sin(2 * math.pi * 220 * t + 1)]).float()
    path = write_audio(str(tmp_path / f"a.{fmt}"), wave, metadata={"comment": "yue2 test", "yue2_seed": 5})
    back = read_audio(path, 48000, 2)
    assert back.shape == wave.shape
    assert (back - wave).abs().max().item() < 1e-4
    if fmt == "flac":
        import av

        with av.open(path) as c:
            meta = {k.lower(): v for k, v in c.metadata.items()}
            assert meta.get("yue2_seed") == "5"
            assert c.streams.audio[0].codec_context.name == "flac"


def test_write_audio_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        write_audio(str(tmp_path / "a.mp3"), torch.zeros(2, 10))
    with pytest.raises(ValueError):
        write_audio(str(tmp_path / "a.flac"), torch.zeros(3, 10))


# endregion
