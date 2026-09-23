import argparse
import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import FakeOffloaderFactory, make_ref_model, save_ref_hf  # noqa: E402

from musubi_tuner.modules.convrot_int8_kernels import dequantize_int8_convrot_weight, quantize_int8_convrot_weight  # noqa: E402
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig  # noqa: E402
from musubi_tuner.yue2 import yue2_args, yue2_model  # noqa: E402
from musubi_tuner.yue2.yue2_attention import _tiled_sdpa, yue2_attention  # noqa: E402
from musubi_tuner.yue2.yue2_checkpoint import load_yue2_model  # noqa: E402
from musubi_tuner.yue2.yue2_model import YuE2Config, YuE2Int8Embedding, YuE2Model, yue2_t_embed_input  # noqa: E402

TINY = YuE2Config.tiny()
CUDA = torch.cuda.is_available()


@pytest.fixture(scope="module")
def ref_and_path(tmp_path_factory):
    pytest.importorskip("transformers")
    ref = make_ref_model(TINY, seed=0)
    path = save_ref_hf(tmp_path_factory.mktemp("yue2") / "hf.safetensors", ref)
    return ref, path


@pytest.fixture(scope="module")
def model_fp32(ref_and_path):
    _, path = ref_and_path
    return load_yue2_model(path, device="cpu", loading_device="cpu", dtype=torch.float32, config=TINY)


def _random_model(config=TINY, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    model = YuE2Model(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape) / p.shape[1] ** 0.5)
            elif name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape))
            else:
                p.copy_(0.1 * torch.randn(p.shape))
    return model.to(dtype).eval()


def _nar_inputs(prefix_len, frames, *, cond_len=None, seed=0, vocab=TINY.vocab_size):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randint(0, vocab, (1, prefix_len), generator=g)
    n = frames + 2
    tokens = torch.cat([ctx, torch.zeros(1, n, dtype=torch.long)], dim=1)
    ar_mask = torch.zeros(1, prefix_len + n, dtype=torch.bool)
    ar_mask[:, :prefix_len] = True
    nar_mask = ~ar_mask
    content = nar_mask.clone()
    content[:, prefix_len] = False
    content[:, -1] = False
    x_t = torch.randn(frames, 64, generator=g)
    return ctx, tokens, ar_mask, nar_mask, content, x_t


def _ours_velocity(model, ctx, x_t, t_embed, kv_visible=None):
    with torch.no_grad():
        out = model.ar_forward(model.embed(ctx), return_kv=True, return_hidden=False, kv_visible=kv_visible)
        return model.nar_forward(x_t[None], t_embed, out.kv, rope_offset=ctx.shape[1])[0]


# region parity with the vendored reference (torch.where MoT form)


def test_ar_logits_match_reference(ref_and_path, model_fp32):
    ref, _ = ref_and_path
    ids = torch.randint(0, TINY.vocab_size, (2, 37), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = ref(input_ids=ids, use_cache=False, return_dict=True).logits
        got = model_fp32.lm_logits(model_fp32.ar_forward(model_fp32.embed(ids)).hidden)
    assert (got - expected).abs().max().item() < 1e-4


@pytest.mark.parametrize("frames", [1, 7, 30])
def test_nar_velocity_matches_reference(ref_and_path, model_fp32, frames):
    ref, _ = ref_and_path
    ctx, tokens, ar_mask, nar_mask, content, x_t = _nar_inputs(23, frames, seed=frames)
    raw = 0.37
    expected = ref.nar_velocity(tokens, ar_mask, nar_mask, content, x_t, raw)
    t_embed = torch.sigmoid(torch.tensor([raw], dtype=torch.float32))
    got = _ours_velocity(model_fp32, ctx, x_t, t_embed)
    assert got.shape == (frames, 64)
    assert (got - expected).abs().max().item() < 1e-4


def test_nar_text_only_matches_reference_cond_end(ref_and_path, model_fp32):
    ref, _ = ref_and_path
    prefix_len, frames = 11, 9
    ctx, tokens, ar_mask, nar_mask, content, x_t = _nar_inputs(prefix_len + 14, frames, seed=5)
    raw = -1.3
    expected = ref.nar_velocity(tokens, ar_mask, nar_mask, content, x_t, raw, nar_cond_end=prefix_len)
    t_embed = torch.sigmoid(torch.tensor([raw], dtype=torch.float32))
    got = _ours_velocity(model_fp32, ctx, x_t, t_embed, kv_visible=prefix_len)
    assert (got - expected).abs().max().item() < 1e-4
    full = _ours_velocity(model_fp32, ctx, x_t, t_embed)
    assert (full - got).abs().max().item() > 1e-3


def test_bf16_parity_with_reference(ref_and_path, tmp_path):
    ref, path = ref_and_path
    model = load_yue2_model(path, device="cpu", loading_device="cpu", dtype=torch.bfloat16, config=TINY)
    ref_bf16 = copy.deepcopy(ref).to(torch.bfloat16)
    ctx, tokens, ar_mask, nar_mask, content, x_t = _nar_inputs(40, 12, seed=3)
    t = 0.73
    raw = float(torch.logit(torch.tensor(t, dtype=torch.float64)))
    expected = ref_bf16.nar_velocity(tokens, ar_mask, nar_mask, content, x_t, raw).float()
    got = _ours_velocity(model, ctx, x_t, yue2_t_embed_input(torch.tensor([t], dtype=torch.float64))).float()
    rel = ((got - expected).norm() / expected.norm()).item()
    cos = F.cosine_similarity(got.flatten(), expected.flatten(), dim=0).item()
    assert rel < 2e-2 and cos > 0.999, (rel, cos)
    ids = ctx[:, :30]
    with torch.no_grad():
        exp_logits = ref_bf16(input_ids=ids, use_cache=False, return_dict=True).logits.float()
        got_logits = model.lm_logits(model.ar_forward(model.embed(ids)).hidden).float()
    cos = F.cosine_similarity(got_logits.flatten(), exp_logits.flatten(), dim=0).item()
    assert cos > 0.999


def test_t_embed_input_bit_exact_vs_reference():
    pytest.importorskip("transformers")
    from yue2_ref.modeling_yue2 import YuE2ForCausalLM

    grid = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
    grid += [0.999, 1.0, 1.0 - 1 / 64, 1 / 64, 0.5 + 1 / 128]
    t = torch.tensor(grid, dtype=torch.float64)
    ours = yue2_t_embed_input(t)
    fake_self = SimpleNamespace(config=SimpleNamespace(timestep_shift=1.0))
    for i, value in enumerate(grid):
        raw = torch.logit(torch.tensor(value, dtype=torch.float64)).clamp(-20, 20).item()
        ref = YuE2ForCausalLM._shift_t_value(fake_self, raw, "cpu", torch.bfloat16)
        assert ours[i].item() == ref.float().item(), value
    assert ours.dtype == torch.float32
    assert torch.equal(yue2_t_embed_input(t, "fp32"), t.float())
    # differs from plain bf16(t) near the ends (the logit, not t, is rounded to bf16)
    near = torch.tensor([0.003, 0.007, 0.021, 0.033, 0.047, 0.953, 0.967, 0.983, 0.997], dtype=torch.float64)
    assert (yue2_t_embed_input(near) != near.to(torch.bfloat16).float()).any()


# endregion

# region forward structure


def test_nar_forward_shape_and_rope_offset():
    model = _random_model()
    ctx = torch.randint(0, TINY.vocab_size, (2, 9))
    x_t = torch.randn(2, 5, 64)
    t = torch.tensor([0.3, 0.8])
    with torch.no_grad():
        kv = model.ar_forward(model.embed(ctx), return_kv=True, return_hidden=False).kv
        a = model.nar_forward(x_t, t, kv, rope_offset=9)
        b = model.nar_forward(x_t, t, kv, rope_offset=20)
        c = model.nar_forward(x_t, t, kv, rope_offset=9)
    assert a.shape == (2, 5, 64)
    assert torch.equal(a, c)
    assert (a - b).abs().max().item() > 1e-4
    assert len(kv) == TINY.num_layers and kv[0][0].shape == (2, 9, TINY.num_kv_heads, TINY.head_dim)


def test_kv_detach_no_ar_grad():
    model = _random_model().train()
    model.requires_grad_(True)
    ctx = torch.randint(0, TINY.vocab_size, (1, 8))
    with torch.no_grad():
        kv = model.ar_forward(model.embed(ctx), return_kv=True, return_hidden=False).kv
    out = model.nar_forward(torch.randn(1, 4, 64), torch.tensor([0.5]), kv, rope_offset=8)
    out.square().mean().backward()
    assert all(p.grad is None for p in model.ar.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.nar.blocks.parameters())
    with pytest.raises(RuntimeError, match="no-grad context prefills"):
        model.ar_forward(model.embed(ctx), return_kv=True)


def _gc_run(model, ctx, x_t, t):
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        kv = model.ar_forward(model.embed(ctx), return_kv=True, return_hidden=False).kv
    hidden = model.ar_forward(model.embed(ctx)).hidden
    v = model.nar_forward(x_t, t, kv, rope_offset=ctx.shape[1])
    loss = hidden.square().mean() + v.square().mean()
    loss.backward()
    grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    return hidden.detach().cpu(), v.detach().cpu(), grads


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_gradient_checkpointing_equivalence(device, cpu_offload, monkeypatch):
    model = _random_model().to(device).train()
    model.requires_grad_(True)
    model._execution_device = torch.device(device)
    ctx = torch.randint(0, TINY.vocab_size, (1, 12), generator=torch.Generator().manual_seed(0)).to(device)
    x_t = torch.randn(1, 6, 64, generator=torch.Generator().manual_seed(1)).to(device)
    t = torch.tensor([0.4], device=device)
    h0, v0, g0 = _gc_run(model, ctx, x_t, t)

    calls = []
    real = torch.utils.checkpoint.checkpoint

    def recording(fn, *args, **kwargs):
        calls.append((args[0].device.type, args[-1]))  # hidden device, return_kv flag
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recording)
    model.enable_gradient_checkpointing(activation_cpu_offloading=cpu_offload)
    h1, v1, g1 = _gc_run(model, ctx, x_t, t)
    model.disable_gradient_checkpointing()

    assert len(calls) == 2 * TINY.num_layers  # grad AR pass + NAR pass; the no-grad prefill is not checkpointed
    assert all(flag is False for _, flag in calls)
    if cpu_offload and device == "cuda":
        for stack in (calls[: TINY.num_layers], calls[TINY.num_layers :]):
            assert stack[0][0] == "cuda"
            assert all(dev == "cpu" for dev, _ in stack[1:])
    tol = 1e-5 if device == "cpu" else 1e-4
    assert (h0 - h1).abs().max().item() < tol
    assert (v0 - v1).abs().max().item() < tol
    assert g0.keys() == g1.keys()
    for name in g0:
        assert (g0[name] - g1[name]).abs().max().item() < tol * 10, name


def test_ar_right_padding_valid_len():
    model = _random_model()
    ids = torch.randint(0, TINY.vocab_size, (1, 21))
    padded = torch.cat([ids, torch.full((1, 11), 7)], dim=1)
    with torch.no_grad():
        a = model.ar_forward(model.embed(ids), return_kv=True)
        b = model.ar_forward(model.embed(padded), return_kv=True, valid_len=21)
        c = model.ar_forward(model.embed(padded), return_kv=True, valid_len=21, kv_visible=5)
    # different GEMM shapes may round differently in fp32; real rows are otherwise unaffected by the padding
    assert (a.hidden - b.hidden[:, :21]).abs().max().item() < 1e-5
    for (ka, va), (kb, vb), (kc, _) in zip(a.kv, b.kv, c.kv):
        assert kb.shape == ka.shape and kc.shape[1] == 5
        assert (ka - kb).abs().max().item() < 1e-5 and (va - vb).abs().max().item() < 1e-5


def test_kv_cache_decode_matches_full_forward():
    model = _random_model()
    ids = torch.randint(0, TINY.vocab_size, (1, 16), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        full = model.lm_logits(model.ar_forward(model.embed(ids)).hidden)[0]
        cache = model.new_kv_cache(32)
        steps = [model.ar_prefill_into_cache(ids[:, :5], cache)]
        steps.append(model.ar_prefill_into_cache(ids[:, 5:9], cache))  # chunked prefill (bottom-right causal)
        for i in range(9, 16):
            steps.append(model.ar_decode_step(ids[:, i : i + 1], cache))
    assert cache.length == 16
    positions = [4, 8] + list(range(9, 16))
    for pos, logits in zip(positions, steps):
        assert (logits[0] - full[pos]).abs().max().item() < 1e-5, pos
    with pytest.raises(ValueError):
        model.ar_decode_step(ids[:, :2], cache)


def test_no_meta_no_nonpersistent_buffers_and_bf16(ref_and_path):
    _, path = ref_and_path
    model = load_yue2_model(path, device="cpu", loading_device="cpu", dtype=torch.bfloat16, config=TINY)
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    assert all(t.device.type != "meta" for _, t in tensors)
    assert all(not m._non_persistent_buffers_set for m in model.modules())
    assert {n for n, _ in model.named_buffers()} <= set(model.state_dict())
    assert all(t.dtype == torch.bfloat16 for _, t in tensors if t.is_floating_point())
    assert model.compute_dtype == torch.bfloat16
    assert model.checkpoint_layout == "hf" and model.base_quant == "bf16"


def test_no_grad_sampling_then_grad_step_convrot(ref_and_path):
    _, path = ref_and_path
    model = load_yue2_model(path, device="cpu", loading_device="cpu", dtype=torch.float32, config=TINY, convrot_int8=True)
    assert model.base_quant == "convrot_int8"
    ids = torch.randint(0, TINY.vocab_size, (1, 6))
    with torch.no_grad():
        cache = model.new_kv_cache(16)
        model.ar_prefill_into_cache(ids, cache)
        model.ar_decode_step(ids[:, :1], cache)
        kv = model.ar_forward(model.embed(ids), return_kv=True, return_hidden=False).kv
        model.nar_forward(torch.randn(1, 3, 64), torch.tensor([0.5]), kv, rope_offset=6)
    model.train()
    x_t = torch.randn(1, 3, 64, requires_grad=True)
    out = model.nar_forward(x_t, torch.tensor([0.5]), kv, rope_offset=6)
    out.square().mean().backward()
    assert x_t.grad is not None and torch.isfinite(x_t.grad).all() and x_t.grad.abs().sum() > 0


def test_int8_embedding_matches_dequantized_table():
    torch.manual_seed(0)
    w = torch.randn(300, 512)
    q, s = quantize_int8_convrot_weight(w, 256)
    emb = YuE2Int8Embedding(300, 512, 256, dtype=torch.float32)
    emb.weight.copy_(q)
    emb.scale.copy_(s)
    ids = torch.tensor([[0, 5, 299, 5], [17, 1, 2, 3]])
    table = dequantize_int8_convrot_weight(q, s, 256)
    assert (emb(ids) - table[ids]).abs().max().item() < 1e-6
    assert (table - w).abs().max().item() / w.abs().max().item() < 2e-2


# endregion

# region attention backends


def _manual_attention(q, k, v, causal):
    b, lq, h, d = q.shape
    lk, kvh = k.shape[1], k.shape[2]
    k = k.repeat_interleave(h // kvh, dim=2)
    v = v.repeat_interleave(h // kvh, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(d)
    if causal:
        off = lk - lq
        mask = torch.arange(lk)[None, :] <= (torch.arange(lq)[:, None] + off)
        scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v)
    return out.reshape(b, lq, h * d)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("lq,lk", [(13, 13), (5, 13), (1, 9)])
def test_attention_modes_agree(causal, lq, lk):
    if not causal and lq != lk:
        lk = lk + 3
    g = torch.Generator().manual_seed(lq * 100 + lk)
    q = torch.randn(2, lq, 4, 16, generator=g)
    k = torch.randn(2, lk, 2, 16, generator=g)
    v = torch.randn(2, lk, 2, 16, generator=g)
    ref = _manual_attention(q, k, v, causal)
    for kwargs in (
        dict(sdpa_gqa="repeat"),
        dict(sdpa_gqa="native"),
        dict(split_attn=True, query_tile=3),
        dict(split_attn=True, query_tile=3, sdpa_gqa="native"),
        dict(split_attn=True, query_tile=64),
    ):
        out = yue2_attention(q, k, v, causal=causal, attn_mode="torch", **kwargs)
        assert out.shape == (2, lq, 64)
        assert (out - ref).abs().max().item() < 1e-5, kwargs
    assert torch.allclose(yue2_attention(q, k, v, causal=causal, attn_mode="sdpa"), ref, atol=1e-5)


def test_tiled_sdpa_causal_masks():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, 10, 8, generator=g) for _ in range(3))
    full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    for tile in (1, 3, 4, 10, 16):
        assert (_tiled_sdpa(q, k, v, causal=True, query_tile=tile, enable_gqa=False) - full).abs().max().item() < 1e-6
    # queries 6..9 over keys 0..9 (bottom-right)
    part = _tiled_sdpa(q[..., 6:, :], k, v, causal=True, query_tile=3, enable_gqa=False, q_offset=6)
    assert (part - full[..., 6:, :]).abs().max().item() < 1e-6


def test_attention_rejects_bad_causal_shapes():
    q = torch.randn(1, 5, 4, 8)
    k = torch.randn(1, 3, 2, 8)
    with pytest.raises(ValueError):
        yue2_attention(q, k, k, causal=True, attn_mode="torch")
    with pytest.raises(RuntimeError):
        yue2_attention(q, torch.randn(1, 5, 2, 8), torch.randn(1, 5, 2, 8), causal=True, attn_mode="sageattn")


def test_set_attention_reaches_every_block():
    model = _random_model()
    model.set_attention("sdpa", split_attn=True, query_tile=7, sdpa_gqa="native")
    for block in list(model.ar.blocks) + list(model.nar.blocks):
        assert block.self_attn.attn_kwargs == {"attn_mode": "torch", "split_attn": True, "query_tile": 7, "sdpa_gqa": "native"}
    ids = torch.randint(0, TINY.vocab_size, (1, 20))
    with torch.no_grad():
        tiled = model.ar_forward(model.embed(ids)).hidden
        model.set_attention("torch")
        plain = model.ar_forward(model.embed(ids)).hidden
    assert (tiled - plain).abs().max().item() < 1e-5


# endregion

# region block swap control flow (layout-simulating fake offloader)

SWAP_CFG = YuE2Config.tiny(num_layers=6)


def _swap_model(monkeypatch, ar_n=2, nar_n=3, ar_backward=True, nar_backward=True, branches=("ar", "nar"), device="cpu"):
    factory = FakeOffloaderFactory()
    monkeypatch.setattr(yue2_model, "create_offloader", factory)
    model = _random_model(SWAP_CFG)
    model.set_block_swap_plan(ar_n, nar_n, ar_backward, nar_backward, branches)
    model.enable_block_swap(max(ar_n, nar_n), BlockSwapConfig(device=torch.device(device), supports_backward=True))
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()
    return model, factory


def _trainable_norms(model):
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        if name.endswith("norm.weight") and "blocks" in name:
            p.requires_grad_(True)


def _train_step(model, ctx, x_t, train_ar=True):
    model.begin_train_step()
    with torch.no_grad():
        kv = model.ar_forward(model.embed(ctx), return_kv=True, return_hidden=False).kv
    loss = 0
    if train_ar:
        loss = loss + model.ar_forward(model.embed(ctx)).hidden.square().mean()
    loss = loss + model.nar_forward(x_t, torch.tensor([0.5]), kv, rope_offset=ctx.shape[1]).square().mean()
    loss.backward()
    return loss.detach()


def test_block_swap_forward_only_switching(monkeypatch):
    model, factory = _swap_model(monkeypatch)
    _trainable_norms(model)
    model.train()
    ctx, x_t = torch.randint(0, 100, (1, 7)), torch.randn(1, 4, 64)
    for _ in range(2):
        _train_step(model, ctx, x_t)
    ar, nar = factory.created["yue2-ar"], factory.created["yue2-nar"]
    assert ar.passes() == [True, False, True, False]
    assert nar.passes() == [False, False]
    assert ar.resident == ar.initial_layout and nar.resident == nar.initial_layout


def test_block_swap_matches_no_swap_grads(monkeypatch):
    base = _random_model(SWAP_CFG)
    _trainable_norms(base)
    base.train()
    ctx, x_t = torch.randint(0, 100, (1, 7)), torch.randn(1, 4, 64)
    ref_loss = _train_step(base, ctx, x_t)
    ref_grads = {n: p.grad.clone() for n, p in base.named_parameters() if p.grad is not None}
    model, _ = _swap_model(monkeypatch, ar_n=4, nar_n=1)
    model.load_state_dict(base.state_dict())
    _trainable_norms(model)
    model.train()
    loss = _train_step(model, ctx, x_t)
    assert torch.equal(loss, ref_loss)
    for n, g in ref_grads.items():
        assert torch.equal(dict(model.named_parameters())[n].grad, g), n


def test_forward_only_ar_offloader_never_switched(monkeypatch):
    model, factory = _swap_model(monkeypatch, ar_backward=False)
    _trainable_norms(model)
    model.train()
    _train_step(model, torch.randint(0, 100, (1, 7)), torch.randn(1, 4, 64), train_ar=False)
    ar = factory.created["yue2-ar"]
    assert not ar.supports_backward
    assert not any(e[0] == "forward_only" for e in ar.events)
    assert ar.passes() == [True]


def test_grad_pass_without_grad_output_raises(monkeypatch):
    model, _ = _swap_model(monkeypatch)
    model.requires_grad_(False)
    model.begin_train_step()
    with pytest.raises(RuntimeError, match="needs no grad"):
        model.ar_forward(model.embed(torch.randint(0, 100, (1, 5))))


def test_second_grad_pass_and_late_no_grad_pass_raise(monkeypatch):
    model, _ = _swap_model(monkeypatch)
    _trainable_norms(model)
    model.train()
    ids = torch.randint(0, 100, (1, 5))
    model.begin_train_step()
    model.ar_forward(model.embed(ids))
    with pytest.raises(RuntimeError, match="second grad forward"):
        model.ar_forward(model.embed(ids))
    model.prepare_block_swap_before_forward()  # the abandoned grad pass left blocks swapped out
    model.begin_train_step()
    model.ar_forward(model.embed(ids))
    with pytest.raises(RuntimeError, match="no-grad pass after a grad pass"):
        with torch.no_grad():
            model.ar_forward(model.embed(ids))


def test_switch_for_inference_and_training(monkeypatch):
    model, factory = _swap_model(monkeypatch)
    model.switch_block_swap_for_inference()
    assert model._swap_inference and all(o.forward_only for o in factory.created.values())
    with torch.enable_grad():
        model.ar_forward(model.embed(torch.randint(0, 100, (1, 4))))  # counts as a no-grad pass
    model.switch_block_swap_for_training()
    assert not model._swap_inference and not any(o.forward_only for o in factory.created.values())


def _devices(module):
    return {t.device.type for t in list(module.parameters()) + list(module.buffers())}


def test_move_to_device_except_swap_blocks(monkeypatch):
    model, _ = _swap_model(monkeypatch, ar_n=2, nar_n=0, device="meta")
    assert _devices(model.ar.blocks) == {"cpu"}  # swapped list untouched (the offloader places it)
    assert _devices(model.nar.blocks) == {"meta"}  # n=0 list moved normally
    assert _devices(model.nar.vae2llm) == {"meta"} and _devices(model.norm) == {"meta"}
    assert "yue2-nar" not in yue2_model.create_offloader.created


@pytest.mark.parametrize("branches,cpu_parts", [(("nar",), ("ar.lm_head", "ar.embed_tokens")), (("ar",), ("nar",))])
def test_branch_residency(monkeypatch, branches, cpu_parts):
    model, _ = _swap_model(monkeypatch, ar_n=0, nar_n=0, branches=branches, device="meta")
    assert model.cpu_resident_modules == cpu_parts
    for path in cpu_parts:
        assert _devices(model.get_submodule(path)) == {"cpu"}
    assert _devices(model.norm) == {"meta"}
    others = model.nar if branches == ("nar",) else model.ar.blocks
    assert _devices(others) == {"meta"}


@pytest.mark.skipif(not CUDA, reason="CUDA required for residency round trips")
@pytest.mark.parametrize("branches", [("nar",), ("ar",)])
def test_residency_round_trip_cuda(monkeypatch, branches):
    model, _ = _swap_model(monkeypatch, ar_n=2, nar_n=2, branches=branches, device="cuda")
    parts = model.cpu_resident_modules
    model.switch_block_swap_for_inference()
    for path in parts:
        module = model.get_submodule(path)
        if path == "nar":
            module = module.vae2llm
        assert _devices(module) == {"cuda"}
    model.switch_block_swap_for_training()
    for path in parts:
        assert _devices(model.get_submodule(path)) == {"cpu"}


def test_normalize_swap_counts():
    args = argparse.Namespace(blocks_to_swap=None, ar_blocks_to_swap=None, nar_blocks_to_swap=14)
    assert yue2_args.normalize_swap_counts(args) == (0, 14) and args.blocks_to_swap == 14
    args = argparse.Namespace(blocks_to_swap=14, ar_blocks_to_swap=0, nar_blocks_to_swap=0)
    assert yue2_args.normalize_swap_counts(args) == (0, 0) and args.blocks_to_swap == 0
    args = argparse.Namespace(blocks_to_swap=10, ar_blocks_to_swap=None, nar_blocks_to_swap=None)
    assert yue2_args.normalize_swap_counts(args) == (10, 10)


def test_enable_block_swap_rejects_too_many(monkeypatch):
    monkeypatch.setattr(yue2_model, "create_offloader", FakeOffloaderFactory())
    model = _random_model(SWAP_CFG)
    model.set_block_swap_plan(5, 0, True)
    with pytest.raises(ValueError, match="cannot swap more"):
        model.enable_block_swap(5, BlockSwapConfig(device=torch.device("cpu"), supports_backward=True))


# endregion
