"""Pure YuE2 training helpers (planning, RNG, windows, AR targets, CE/KL, timesteps, LoRA eval)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.hv_train_network import setup_parser_common  # noqa: E402
from musubi_tuner.yue2 import yue2_protocol as P  # noqa: E402
from musubi_tuner.yue2.yue2_training import (  # noqa: E402
    ar_targets,
    chunked_ce_kl,
    flow_loss,
    item_rng,
    lora_eval,
    optimizer_eval,
    pad_right,
    plan_item,
    sample_t,
    select_window,
    validation_windows,
)

TEXTS = {cot: [P.EOD] + [10 + i for i in range(5 + k)] for k, cot in enumerate(P.COT_MODES)}
NEGS = {cot: [P.EOD] + [200 + i for i in range(2 + k)] for k, cot in enumerate(P.COT_MODES)}
ABC = [60, 61, 62, 63]


def _plan(**kw):
    frames = kw.pop("total_frames", 100)
    base = dict(
        texts=TEXTS,
        negatives=NEGS,
        abc=[],
        abc_mode=None,
        codes=list(range(frames)),
        total_frames=frames,
        seg=(0, frames, 0),
        cot="auto",
        abc_dropout=0.0,
        caption_dropout=0.0,
        nar_caption_dropout=0.0,
        window_frames=0,
        nar_context="codes",
        codec_dropout=0.0,
        text_only_rope="full",
        ar_max_tokens=0,
        ar_ce_targets="codec",
        train_ar=True,
        rng=random.Random(0),
        training=True,
    )
    base.update(kw)
    return plan_item(**base)


def test_cot_auto_follows_the_record_and_falls_back_to_off():
    p = _plan()
    assert p.cot_used == "off" and not p.score_used
    assert p.prefix == P.build_prefix(TEXTS["off"], "off", None) == p.nar_prefix

    p = _plan(abc=ABC, abc_mode="full")
    assert p.cot_used == "full" and p.score_used
    assert p.prefix == P.build_prefix(TEXTS["full"], "full", ABC)

    p = _plan(abc=ABC, abc_mode="melody")
    assert p.cot_used == "melody" and p.prefix == P.build_prefix(TEXTS["melody"], "melody", ABC)

    assert _plan(abc=ABC, abc_mode="full", cot="off").cot_used == "off"
    assert _plan(abc=ABC, abc_mode="melody", cot="melody").cot_used == "melody"
    # explicit flavour without a score: per-item fallback to off
    assert _plan(cot="full").cot_used == "off"
    # abc dropout 1 -> off
    assert _plan(abc=ABC, abc_mode="full", abc_dropout=1.0).cot_used == "off"


def test_abc_dropout_rate_over_many_draws():
    used = 0
    n = 10000
    for i in range(n):
        p = _plan(abc=ABC, abc_mode="full", abc_dropout=0.5, total_frames=8, rng=item_rng(1, 0, 0, 0, i))
        used += p.score_used
    assert abs(used / n - 0.5) < 0.02


def test_validation_uses_the_score_and_never_drops():
    p = _plan(abc=ABC, abc_mode="full", abc_dropout=1.0, caption_dropout=1.0, codec_dropout=1.0, training=False)
    assert p.score_used and not p.dropped_caption and p.kv_visible is None


def test_caption_dropout_hits_the_ar_prefix_only():
    p = _plan(abc=ABC, abc_mode="full", caption_dropout=1.0)
    assert p.dropped_caption
    assert p.prefix == P.build_negative_prefix(NEGS["full"], "full", ABC)
    assert p.nar_prefix == P.build_prefix(TEXTS["full"], "full", ABC)
    assert p.ctx_ids[: len(p.nar_prefix)] == p.nar_prefix
    assert p.ar_ids[: len(p.prefix)] == p.prefix

    p = _plan(nar_caption_dropout=1.0)
    assert p.nar_prefix == P.build_negative_prefix(NEGS["off"], "off", None) and p.prefix != p.nar_prefix


def test_codec_abc_targets_and_fallback():
    p = _plan(abc=ABC, abc_mode="full", ar_ce_targets="codec_abc")
    assert p.ar_target_start == p.prefix.index(P.ABC_START) + 1
    assert p.ar_ids[p.ar_target_start : p.ar_target_start + len(ABC)] == ABC
    # off negative prefix has no ABC_START: codec targets, no error
    p = _plan(caption_dropout=1.0, ar_ce_targets="codec_abc")
    assert P.ABC_START not in p.prefix and p.ar_target_start == len(p.prefix)
    p = _plan(abc=ABC, abc_mode="full", ar_ce_targets="codec")
    assert p.ar_target_start == len(p.prefix)


def test_codes_context_codec_dropout_and_text_only_rope():
    p = _plan(total_frames=50, window_frames=20, rng=random.Random(3))
    s, w = p.window_start, p.window_frames
    assert w == 20 and 0 <= s <= 30
    assert p.ctx_ids == p.nar_prefix + P.codec_to_ids(list(range(s, s + w))) + [P.MUSIC_END]
    assert p.rope_offset == len(p.ctx_ids) and p.kv_visible is None

    p = _plan(total_frames=50, window_frames=20, codec_dropout=1.0)
    assert p.kv_visible == len(p.nar_prefix) and p.rope_offset == len(p.nar_prefix) + 20 + 1

    p = _plan(total_frames=50, window_frames=20, nar_context="text_only", codes=None, train_ar=False)
    assert p.ctx_ids == p.nar_prefix and p.rope_offset == len(p.nar_prefix) + 21
    p = _plan(total_frames=50, window_frames=20, nar_context="text_only", text_only_rope="compact", train_ar=False)
    assert p.rope_offset == len(p.nar_prefix)

    with pytest.raises(ValueError, match="codes"):
        _plan(codes=None, train_ar=False)


def test_window_cap_and_centre():
    assert select_window(100, 0, 10, None, training=False) == (0, 100)
    assert select_window(100, 40, 10, None, training=False) == (30, 40)
    assert validation_windows(101, 10, 40) == (30, 40)
    cap = P.max_nar_window(5000)
    start, frames = select_window(20000, 15000, 5000, random.Random(0), training=True)
    assert frames == cap and 0 <= start <= 20000 - cap
    starts = {select_window(100, 10, 10, random.Random(i), training=True)[0] for i in range(50)}
    assert len(starts) > 5 and max(starts) <= 90


@pytest.mark.parametrize(
    "seg, total, max_tokens, expect_ar, expect_end",
    [
        ((0, 100, 0), 100, 0, True, True),  # full song
        ((0, 250, 0), 100, 0, True, False),  # head segment
        ((100, 250, 0), 100, 0, False, False),  # chunk / slide past the start
        ((0, 100, 1), 100, 0, True, False),  # truncated by max_seconds / excerpt end
        ((300, 900, 0), 100, 0, False, False),  # JSONL excerpt in the middle of the file
        ((0, 100, 0), 100, 40, True, False),  # --ar_max_tokens
    ],
)
def test_ar_gating_and_music_end(seg, total, max_tokens, expect_ar, expect_end):
    p = _plan(total_frames=total, seg=seg, ar_max_tokens=max_tokens)
    if not expect_ar:
        assert p.ar_ids is None and p.ar_skipped_reason == "mid_song_segment"
        return
    count = total if max_tokens <= 0 else min(total, max_tokens)
    assert p.ar_ids[: len(p.prefix)] == p.prefix
    assert p.ar_ids[len(p.prefix) : len(p.prefix) + count] == P.codec_to_ids(list(range(count)))
    assert p.ar_complete == expect_end
    assert (p.ar_ids[-1] == P.MUSIC_END) == expect_end
    assert len(p.ar_ids) == len(p.prefix) + count + int(expect_end)


def test_ar_needs_codes():
    with pytest.raises(ValueError, match="codes"):
        _plan(codes=None, nar_context="text_only")


def test_item_rng_is_derived_and_deterministic():
    a = [item_rng(42, 3, 1, 0, 2).random() for _ in range(2)]
    b = [item_rng(42, 3, 1, 0, 2).random() for _ in range(2)]
    assert a == b
    others = {item_rng(42, s, m, r, i).random() for s, m, r, i in [(3, 1, 0, 3), (3, 2, 0, 2), (4, 1, 0, 2), (3, 1, 1, 2)]}
    assert a[0] not in others and len(others) == 4


def test_pad_right_and_ar_targets():
    ids, n = pad_right([1, 2, 3], 8, P.EOD)
    assert ids.shape == (1, 8) and n == 3 and ids[0, 3:].tolist() == [P.EOD] * 5
    ids, n = pad_right([1, 2, 3], 0)
    assert ids.shape == (1, 3)
    pos, labels = ar_targets([5, 6, 7, 8, 9], 2, 5)
    assert pos == slice(1, 4) and labels.tolist() == [7, 8, 9]
    with pytest.raises(ValueError):
        ar_targets([5, 6], 0, 2)


def test_chunked_ce_kl_equals_the_unchunked_computation():
    torch.manual_seed(0)
    head = torch.nn.Linear(16, 50, bias=False)
    hidden = torch.randn(37, 16, requires_grad=True)
    labels = torch.randint(0, 50, (37,))
    base = 3.0 * torch.randn(37, 16)  # a sharper base distribution: KL(base||adapted) != KL(adapted||base)

    ce, kl = chunked_ce_kl(head, hidden, labels, base, chunk=8)
    (ce + kl).backward()
    grad_chunked = hidden.grad.clone()

    hidden.grad = None
    logits = head(hidden).float()
    ce_ref = F.cross_entropy(logits, labels)
    # KL(base || adapted) written out: sum_v p_base * (log p_base - log p_adapted), mean over rows
    log_p_base = head(base).float().log_softmax(-1)
    kl_ref = (log_p_base.exp() * (log_p_base - logits.log_softmax(-1))).sum(-1).mean()
    kl_reverse = (logits.softmax(-1) * (logits.log_softmax(-1) - log_p_base)).sum(-1).mean()
    assert abs(kl_ref.item() - kl_reverse.item()) > 1e-2  # the direction is observable on these inputs
    (ce_ref + kl_ref).backward()
    assert torch.allclose(ce, ce_ref, atol=1e-5) and torch.allclose(kl, kl_ref, atol=1e-5)
    assert torch.allclose(grad_chunked, hidden.grad, atol=1e-5)

    ce0, kl0 = chunked_ce_kl(head, hidden.detach(), labels, hidden.detach(), chunk=5)
    assert kl0.abs().item() < 1e-6
    ce1, none = chunked_ce_kl(head, hidden.detach(), labels, None)
    assert none is None and torch.allclose(ce1, ce_ref.detach(), atol=1e-5)


def test_flow_loss_is_mse_against_eps_minus_z():
    v, eps, z = torch.randn(3, 4, 64).unbind(0)
    assert torch.allclose(flow_loss(v, eps, z), ((v - (eps - z)) ** 2).mean())


def _args(**kw):
    from musubi_tuner.yue2_train_network import yue2_setup_parser

    args = yue2_setup_parser(setup_parser_common()).parse_args([])
    for k, v in kw.items():
        setattr(args, k, v)
    return args


@pytest.mark.parametrize("method", ["uniform", "sigmoid", "shift", "logsnr", "beta"])
def test_sample_t_shapes_and_ranges(method):
    from musubi_tuner.yue2_train_network import YuE2NetworkTrainer

    trainer = YuE2NetworkTrainer()
    args = _args(timestep_sampling=method)
    torch.manual_seed(0)
    t = sample_t(trainer, args, 4096, None, "cpu")
    assert t.shape == (4096,) and t.dtype == torch.float32
    assert t.min() >= 0 and t.max() <= 1
    if method == "beta":
        assert t.min() >= 0.02 - 1e-6 and t.max() <= 0.98 + 1e-6 and abs(t.mean().item() - 0.5) < 0.02
    args = _args(timestep_sampling=method, min_timestep=200, max_timestep=400)
    t = sample_t(trainer, args, 512, None, "cpu")
    assert t.min() >= 0.2 - 1e-6 and t.max() <= 0.4 + 1e-6


def test_sample_t_uses_bucketed_pools_and_first_timestep_chance():
    from musubi_tuner.yue2_train_network import YuE2NetworkTrainer

    trainer = YuE2NetworkTrainer()
    args = _args(timestep_sampling="uniform")
    assert sample_t(trainer, args, 3, [0.1, 0.5, 0.9], "cpu").tolist() == pytest.approx([0.1, 0.5, 0.9])
    args = _args(timestep_sampling="sigmoid", first_timestep_chance=1.0)
    assert sample_t(trainer, args, 5, None, "cpu").tolist() == [1.0] * 5
    args = _args(timestep_sampling="sigmoid", first_timestep_chance=1.0, max_timestep=900)
    assert sample_t(trainer, args, 2, None, "cpu").tolist() == pytest.approx([0.9, 0.9])
    torch.manual_seed(0)
    args = _args(timestep_sampling="uniform", first_timestep_chance=0.25)
    t = sample_t(trainer, args, 20000, None, "cpu")
    assert abs((t == 1.0).float().mean().item() - 0.25) < 0.02


def test_lora_eval_makes_context_prefills_deterministic_under_dropout():
    from musubi_tuner.networks import lora_yue2
    from musubi_tuner.yue2.yue2_model import YuE2Config, YuE2Model

    torch.manual_seed(0)
    cfg = YuE2Config.tiny(num_layers=2)
    model = YuE2Model(cfg).eval()
    net = lora_yue2.create_arch_network(
        1.0, 4, 4.0, None, None, model, neuron_dropout=0.3, branches="ar,nar", rank_dropout="0.5", module_dropout="0.5"
    )
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    for m in net.modules():  # make the LoRA delta non-zero
        if hasattr(m, "lora_up"):
            for up in m.lora_up if isinstance(m.lora_up, torch.nn.ModuleList) else [m.lora_up]:
                torch.nn.init.normal_(up.weight, std=0.5)
    net.train()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))

    def prefill():
        with torch.no_grad():
            return model.ar_forward(model.embed(ids), return_kv=True, return_hidden=False).kv

    noisy = [prefill() for _ in range(2)]
    assert any(not torch.equal(a[0], b[0]) for a, b in zip(*noisy))  # dropout acts in no-grad passes
    with lora_eval(net):
        kv1, kv2 = prefill(), prefill()
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(kv1, kv2))
    assert all(m.training for m in net.modules())


def test_optimizer_eval_switches_schedule_free_optimizers():
    calls = []

    class SF:
        def eval(self):
            calls.append("eval")

        def train(self):
            calls.append("train")

    with optimizer_eval(SF()):
        calls.append("body")
    with optimizer_eval(object()):
        pass
    with optimizer_eval(None):
        pass
    assert calls == ["eval", "body", "train"]
