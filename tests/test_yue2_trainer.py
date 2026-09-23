"""YuE2 trainer: argument handling, process_batch in every mode, phase order under block swap, validation, metadata,
sampling hooks and end-to-end CPU runs of the real training loop on a tiny model with synthetic caches."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import FakeOffloaderFactory, FakeTokenizer, FakeYuE2VAE  # noqa: E402

import musubi_tuner.yue2 as yue2_pkg  # noqa: E402
from musubi_tuner import yue2_train_network as T  # noqa: E402
from musubi_tuner.dataset.cache_io import save_latent_cache_yue2, save_text_encoder_output_cache_yue2  # noqa: E402
from musubi_tuner.dataset.image_video_dataset import ItemInfo  # noqa: E402
from musubi_tuner.dataset.yue2_minted import MintedRecord, YuE2MintedPack  # noqa: E402
from musubi_tuner.hv_train_network import setup_parser_common  # noqa: E402
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig  # noqa: E402
from musubi_tuner.networks import lora_yue2  # noqa: E402
from musubi_tuner.yue2 import yue2_lora_formats as lora_formats  # noqa: E402
from musubi_tuner.yue2 import yue2_model  # noqa: E402
from musubi_tuner.yue2 import yue2_protocol as P  # noqa: E402
from musubi_tuner.yue2.yue2_model import YuE2Config, YuE2Model  # noqa: E402

# tiny dims with the full protocol vocabulary (prefix/codec ids are embedded); 3 layers so one block can swap
CFG = YuE2Config.tiny(
    hidden_size=64,
    num_heads=2,
    num_kv_heads=1,
    head_dim=32,
    intermediate_size=128,
    num_layers=3,
    vocab_size=P.VOCAB_SIZE,
    max_latent_frames=256,
)
TOK = FakeTokenizer()
ABC_TEXT = 'X:1\nM:4/4\nK:C\n"C"CDEF|"G"GABc|'


class _Accelerator:
    device = torch.device("cpu")
    process_index = 0
    is_main_process = True
    is_local_main_process = True
    trackers: list = []

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def unwrap_model(model):
        return model

    def log(self, *args, **kwargs):
        pass

    def print(self, *args, **kwargs):
        pass


def _parser():
    return T.yue2_setup_parser(setup_parser_common())


def _args(**kw):
    args = _parser().parse_args([])
    args.mixed_precision = "bf16"
    args.seed = 42
    args.output_dir = None
    args.output_name = "yue2test"
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def _model(seed: int = 0) -> YuE2Model:
    torch.manual_seed(seed)
    model = YuE2Model(CFG)
    model.requires_grad_(False)
    return model


def _network(model, args, randomize_up: bool = False, **extra):
    kwargs = dict(a.split("=", 1) for a in (args.network_args or []))
    kwargs.update(extra)
    net = lora_yue2.create_arch_network(1.0, 4, 4.0, None, None, model, **kwargs)
    net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    net.prepare_optimizer_params(unet_lr=1e-3)
    net.requires_grad_(True)
    if randomize_up:
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for name, p in net.named_parameters():
                if "lora_up" in name:
                    p.copy_(0.1 * torch.randn(p.shape, generator=g))
    return net


def _setup(branches="nar", model_seed=0, **kw):
    args = _args(train_branches=branches, **kw)
    trainer = T.YuE2NetworkTrainer()
    trainer.model_config = CFG
    trainer.handle_model_specific_args(args)
    model = _model(model_seed)
    return trainer, args, model


def _texts(style="pop, female vocal", lyrics="la la la"):
    texts = {c: torch.tensor(P.text_ids(TOK, style, lyrics, c)) for c in P.COT_MODES}
    negs = {c: torch.tensor(P.negative_text_ids(TOK, c)) for c in P.COT_MODES}
    return texts, negs


def _batch(bsz=1, frames=24, abc=False, seg=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    texts, negs = _texts()
    abc_ids = torch.tensor(TOK.encode(ABC_TEXT)) if abc else torch.zeros(0, dtype=torch.int64)
    return {
        "latents": torch.randn(bsz, frames, 64, generator=g),
        "codes": torch.randint(0, P.CODEC_SIZE, (bsz, frames), generator=g),
        "yue2_seg": torch.tensor([list(seg or (0, frames, 0))] * bsz),
        **{f"yue2_text_{c}": [texts[c]] * bsz for c in P.COT_MODES},
        **{f"yue2_neg_{c}": [negs[c]] * bsz for c in P.COT_MODES},
        "yue2_abc": [abc_ids] * bsz,
        "yue2_has_abc": torch.tensor([int(abc)] * bsz),
        "yue2_abc_mode": torch.tensor([2 if abc else 0] * bsz),
        "timesteps": None,
    }


def _step(trainer, args, model, net, batch, global_step=0, seed=0):
    torch.manual_seed(seed)
    latents = batch["latents"]
    noise = torch.randn_like(latents)
    return trainer.process_batch(
        args, _Accelerator(), model, net, batch, latents, noise, None, torch.bfloat16, torch.float32, None, global_step
    )


def _grads(net, branch):
    return [p.grad for m in net.modules_of_branch(branch) for p in m.parameters()]


# region arguments


def test_parser_defaults():
    args = _parser().parse_args([])
    assert args.timestep_sampling == "sigmoid" and args.weighting_scheme == "none"
    assert args.network_module == "networks.lora_yue2" and args.compile_cache_size_limit == 32
    assert args.train_branches == "nar" and args.nar_context == "codes" and args.nar_window_frames == 1500
    assert args.cot == "auto" and args.abc_dropout == 0.5 and args.ar_kl_weight == 0.2 and args.ar_ce_chunk == 512
    assert args.validation_noise_seed == 1234 and args.sample_mode == "reconstruct" and args.sample_seconds == 30.0
    assert args.sample_ode_state_dtype == "bf16" and args.t_embed_dtype == "bf16" and args.ar_pad_multiple is None
    assert "beta" in _parser()._option_string_actions["--timestep_sampling"].choices


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mixed_precision": "fp16"}, "bf16"),
        ({"sage_attn": True}, "sage"),
        ({"dim_from_weights": "x.safetensors"}, "dim_from_weights"),
        ({"network_module": "networks.lora"}, "lora_yue2"),
        ({"timestep_sampling": "qinglong_flux"}, "timestep_sampling"),
        ({"weighting_scheme": "logit_normal"}, "weighting_scheme"),
        ({"train_branches": "nar,foo"}, "train_branches"),
        ({"fp8_base": True}, "fp8"),
        ({"convrot_int8": True, "fp8_base": True, "fp8_scaled": True}, "convrot"),
        ({"abc_dropout": 1.5}, "abc_dropout"),
        ({"ar_replay_fraction": 0.5, "train_branches": "nar"}, "ar_replay_fraction"),
        ({"nar_codec_dropout": 0.2, "nar_context": "text_only"}, "codec_dropout"),
        ({"timestep_sampling": "beta", "show_timesteps": "console"}, "show_timesteps"),
        ({"network_args": ["branches=ar"]}, "conflicts"),
        ({"blocks_to_swap": 27}, "blocks_to_swap"),
        ({"train_io": "full", "yue2_export_formats": "comfy,hf"}, "bias delta"),
        ({"train_io": "lora", "network_args": ["include_time_embedder=true"], "yue2_export_formats": "fl"}, "time_embedder"),
    ],
)
def test_handle_model_specific_args_rejections(overrides, message):
    args = _args(**overrides)
    with pytest.raises(ValueError, match=message):
        T.YuE2NetworkTrainer().handle_model_specific_args(args)


def test_network_arg_injection_and_swap_normalisation():
    args = _args(train_branches="ar,nar", train_io="full", ar_lr_ratio=0.5, io_lr=2e-4, nar_blocks_to_swap=14)
    trainer = T.YuE2NetworkTrainer()
    trainer.handle_model_specific_args(args)
    assert trainer.branches == ("ar", "nar")
    assert set(args.network_args) == {"branches=ar,nar", "train_io=full", "ar_lr_ratio=0.5", "io_lr=0.0002"}
    assert args.blocks_to_swap == 14 and trainer._swap_counts == (0, 14)
    assert args.ar_kl_weight == 0.2 and args.ar_pad_multiple == 0

    # same values given explicitly are fine; the order of branches does not matter
    args = _args(train_branches="nar,ar", network_args=["branches=ar,nar", "ar_lr_ratio=1"], compile=True)
    T.YuE2NetworkTrainer().handle_model_specific_args(args)
    assert args.network_args.count("branches=ar,nar") == 1 and args.ar_pad_multiple == 256

    args = _args(blocks_to_swap=14, ar_blocks_to_swap=0, nar_blocks_to_swap=0)
    T.YuE2NetworkTrainer().handle_model_specific_args(args)
    assert args.blocks_to_swap == 0

    args = _args(train_branches="nar", ar_kl_weight=0.5)
    T.YuE2NetworkTrainer().handle_model_specific_args(args)
    assert args.ar_kl_weight == 0.0


def test_noising_is_rank_generic_for_training_latents_and_the_show_timesteps_probe():
    trainer, args, _ = _setup(timestep_sampling="uniform")
    latents = torch.zeros(4, 10, 64)
    noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(
        args, torch.ones_like(latents), latents, None, None, "cpu", torch.float32
    )
    t = (timesteps - 1.0) / 1000.0
    assert noisy.shape == latents.shape and torch.allclose(noisy[:, 3, 5], t)
    probe = torch.zeros(8, 1, 1, 16, 16, dtype=torch.float16)
    noisy, _ = trainer.get_noisy_model_input_and_timesteps(args, torch.ones_like(probe), probe, None, None, "cpu", torch.float16)
    assert noisy.shape == probe.shape and noisy[:, 0, 0, 0, 0].min() >= 0


# endregion

# region process_batch


@pytest.mark.parametrize("branches", ["nar", "ar", "ar,nar"])
def test_process_batch_is_finite_in_every_mode(branches):
    trainer, args, model = _setup(branches, nar_window_frames=16)
    net = _network(model, args)
    loss, logs = _step(trainer, args, model, net, _batch(abc=True))
    assert torch.isfinite(loss)
    loss.backward()
    branches = branches.split(",")
    if "nar" in branches:
        assert "loss/flow" in logs and any(g is not None and g.abs().sum() > 0 for g in _grads(net, "nar"))
    if "ar" in branches:
        assert "loss/ar_ce" in logs and "loss/ar_kl" in logs
        assert abs(logs["loss/ar_kl"]) < 1e-5  # zero-init LoRA: adapted == base
        assert any(g is not None and g.abs().sum() > 0 for g in _grads(net, "ar"))
    else:
        assert not net.modules_of_branch("ar")  # NAR-only: no AR LoRA exists
    assert logs["yue2/window"] <= 16


def test_ar_right_padding_does_not_change_the_loss():
    losses = []
    for pad in (0, 8):
        trainer, args, model = _setup("ar,nar", ar_pad_multiple=pad, nar_window_frames=16)
        net = _network(model, args, randomize_up=True)
        loss, logs = _step(trainer, args, model, net, _batch(abc=True))
        losses.append((loss.item(), logs["loss/ar_ce"], logs["loss/ar_kl"], logs["loss/flow"]))
    assert losses[0] == pytest.approx(losses[1], rel=1e-5, abs=1e-6)


def test_gradient_isolation_between_branches():
    trainer, args, model = _setup("ar,nar", ar_ce_weight=0.0, ar_kl_weight=0.0, nar_window_frames=16)
    net = _network(model, args, randomize_up=True)
    loss, _ = _step(trainer, args, model, net, _batch())
    loss.backward()
    assert all(g is None or g.abs().sum() == 0 for g in _grads(net, "ar"))
    assert any(g is not None and g.abs().sum() > 0 for g in _grads(net, "nar"))

    trainer, args, model = _setup("ar,nar", flow_loss_weight=0.0, nar_window_frames=16)
    net = _network(model, args, randomize_up=True)
    loss, _ = _step(trainer, args, model, net, _batch())
    loss.backward()
    assert all(g is None or g.abs().sum() == 0 for g in _grads(net, "nar"))
    assert any(g is not None and g.abs().sum() > 0 for g in _grads(net, "ar"))


def test_mid_song_items_contribute_no_ar_loss():
    trainer, args, model = _setup("ar")
    net = _network(model, args)
    loss, logs = _step(trainer, args, model, net, _batch(seg=(50, 200, 0)))
    assert "loss/ar_ce" not in logs and logs["yue2/ar_skipped"] == 1.0
    loss.backward()  # graph-free zero still backpropagates


def test_minted_replay_replaces_the_ar_item():
    trainer, args, model = _setup("ar,nar", ar_replay_fraction=1.0, ar_minted_pack="unused", nar_window_frames=16)
    codes = torch.randint(0, P.CODEC_SIZE, (30,))
    pack = YuE2MintedPack(None, records=[MintedRecord(codes=codes, style="edm", lyrics="", song_id="m0", is_validation=None)])
    pack.tokenize(TOK)
    trainer._minted_train = pack
    batch = _batch(abc=True)
    plans = [trainer._plan(args, batch, 0, 24, random.Random(0))]
    trainer._apply_minted_replay(args, plans, [random.Random(1)], batch)
    prefix = P.build_prefix(pack.item(0).text_ids["off"], "off", None)
    assert plans[0].ar_ids == prefix + P.codec_to_ids(codes.tolist()) + [P.MUSIC_END]
    assert plans[0].ar_target_start == len(prefix)
    net = _network(model, args)
    loss, logs = _step(trainer, args, model, net, batch)
    assert torch.isfinite(loss) and "loss/ar_ce" in logs


def _slice(batch: dict, j: int) -> dict:
    return {k: (v[j : j + 1] if isinstance(v, (torch.Tensor, list)) else v) for k, v in batch.items()}


def test_batch_of_two_without_swap_matches_single_items(monkeypatch):
    trainer, args, model = _setup("ar,nar", nar_window_frames=0, timestep_sampling="uniform")
    net = _network(model, args, randomize_up=True)
    batch = _batch(bsz=2, abc=True)
    noise = torch.randn(batch["latents"].shape, generator=torch.Generator().manual_seed(5))
    # the same per-item plan RNG and t whether an item is alone or second in the batch
    t_all = torch.tensor([0.3, 0.7])
    first = {"i": 0}
    monkeypatch.setattr(T, "item_rng", lambda seed, step, micro, rank, i: random.Random(100 + first["i"] + i))
    monkeypatch.setattr(T, "sample_t", lambda tr, a, bsz, ts, dev: t_all[first["i"] : first["i"] + bsz].to(dev))

    def run(b, nz):
        net.zero_grad(set_to_none=True)
        loss, logs = trainer.process_batch(
            args, _Accelerator(), model, net, b, b["latents"], nz, None, torch.bfloat16, torch.float32, None, 0
        )
        loss.backward()
        grads = {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in net.named_parameters()}
        return loss.detach(), logs, grads

    loss2, logs2, grads2 = run(batch, noise)
    singles = []
    for j in range(2):
        first["i"] = j
        singles.append(run(_slice(batch, j), noise[j : j + 1]))
    assert all("loss/ar_ce" in logs and "loss/flow" in logs for _, logs, _ in singles)
    assert singles[0][1]["loss/flow"] != pytest.approx(singles[1][1]["loss/flow"])  # the items differ
    # every term is a per-item mean, so the batch equals the mean of the single items
    assert loss2.item() == pytest.approx((singles[0][0].item() + singles[1][0].item()) / 2, rel=1e-5)
    for key in ("loss/flow", "loss/ar_ce", "loss/ar_kl"):
        assert logs2[key] == pytest.approx((singles[0][1][key] + singles[1][1][key]) / 2, rel=1e-5, abs=1e-7), key
    for name, g in grads2.items():
        ref = (singles[0][2][name] + singles[1][2][name]) / 2
        assert torch.allclose(g, ref, rtol=1e-4, atol=1e-7), name
    assert any(g.abs().sum() > 0 for g in grads2.values())


def _swap_setup(monkeypatch, branches, ar_n=1, nar_n=1):
    factory = FakeOffloaderFactory()
    monkeypatch.setattr(yue2_model, "create_offloader", factory)
    trainer, args, model = _setup(branches, ar_blocks_to_swap=ar_n, nar_blocks_to_swap=nar_n, nar_window_frames=16)
    trainer.on_transformer_loaded(args, _Accelerator(), model)
    model.enable_block_swap(max(ar_n, nar_n), BlockSwapConfig(device=torch.device("cpu"), supports_backward=True))
    model.move_to_device_except_swap_blocks("cpu")
    model.prepare_block_swap_before_forward()
    return trainer, args, model, factory


def test_phase_order_under_block_swap(monkeypatch):
    trainer, args, model, factory = _swap_setup(monkeypatch, "ar,nar")
    net = _network(model, args)
    for step in range(2):
        loss, _ = _step(trainer, args, model, net, _batch(abc=True), global_step=step)
        loss.backward()
    ar, nar = factory.created["yue2-ar"], factory.created["yue2-nar"]
    # per step: KL base pass (no grad), NAR context prefill (no grad), then the one AR grad pass
    assert ar.passes() == [True, True, False] * 2
    assert nar.passes() == [False] * 2

    # a second grad pass on a swapped stack before backward is refused (the trainer rejects bs>1 up front)
    with pytest.raises(RuntimeError, match="second grad forward"):
        _step(trainer, args, model, net, _batch(bsz=2, abc=True))


def test_nar_only_with_ar_swapped_uses_a_forward_only_ar_offloader(monkeypatch):
    trainer, args, model, factory = _swap_setup(monkeypatch, "nar", ar_n=1, nar_n=0)
    assert not factory.created["yue2-ar"].supports_backward and "yue2-nar" not in factory.created
    net = _network(model, args)
    loss, _ = _step(trainer, args, model, net, _batch(bsz=2))  # bs>1 is fine: no trained stack swaps
    loss.backward()
    assert set(factory.created["yue2-ar"].passes()) == {True}


# endregion

# region caches, validation, metadata


def _write_song(cache_dir: Path, key: str, frames: int, abc: bool = False, start: int = 0, song_frames=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    item = ItemInfo(key, "style", (frames, 64), (frames,), frame_count=frames)
    item.frame_pos = start
    item.song_start_frame = start
    item.song_frames = song_frames or (start + frames)
    item.truncated = False
    item.latent_cache_path = str(cache_dir / f"{key}_{start:06d}-{frames:06d}_yue2.safetensors")
    item.text_encoder_output_cache_path = str(cache_dir / f"{key}_yue2_te.safetensors")
    save_latent_cache_yue2(item, torch.randn(frames, 64, generator=g), torch.randint(0, P.CODEC_SIZE, (frames,), generator=g))
    texts, negs = _texts(style=f"style {key}", lyrics=f"lyrics of {key}")
    meta = {"yue2_tokenizer": TOK.fingerprint, "yue2_song_id": key, "yue2_instrumental_lyrics": "[instrumental]"}
    abc_ids = torch.tensor(TOK.encode(ABC_TEXT)) if abc else None
    save_text_encoder_output_cache_yue2(item, texts, negs, abc_ids, "full" if abc else None, meta)
    return item


class _ScheduleFree:
    def __init__(self):
        self.calls = []

    def eval(self):
        self.calls.append("eval")

    def train(self):
        self.calls.append("train")


def test_validation_is_deterministic_under_dropout_and_restores_state(tmp_path):
    items = [_write_song(tmp_path, f"v{i}", 30 + 4 * i, abc=i == 0, seed=i) for i in range(2)]
    trainer, args, model = _setup("ar,nar", nar_window_frames=16, validate_every_n_steps=1)
    trainer._validation_items = items
    net = _network(model, args, randomize_up=True, rank_dropout="0.5", module_dropout="0.5")
    net.train()
    opt = _ScheduleFree()
    trainer.on_train_start(args, _Accelerator(), net, model, opt)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    r1 = trainer._validator.run(trainer, args, _Accelerator(), model, net, opt)
    r2 = trainer._validator.run(trainer, args, _Accelerator(), model, net, opt)
    assert r1 == r2
    assert torch.equal(torch.get_rng_state(), state)
    assert opt.calls == ["eval", "train", "eval", "train"]
    assert {"val/flow", "val/flow@0.2", "val/flow@0.5", "val/flow@0.8", "val/ar_ce", "val/ar_kl", "val/total"} <= set(r1)
    assert all(m.training for m in net.modules())


def test_best_checkpoint_carries_training_metadata(tmp_path):
    items = [_write_song(tmp_path, "v0", 30)]
    trainer, args, model = _setup("nar", nar_window_frames=16, validate_every_n_steps=1, save_best_validation=True)
    args.output_dir = str(tmp_path / "out")
    trainer._validation_items = items
    net = _network(model, args)
    trainer.on_train_start(args, _Accelerator(), net, model, torch.optim.AdamW(net.parameters(), lr=1e-3))
    trainer.on_post_optimizer_step(args, _Accelerator(), net, model, True, 0)
    path = tmp_path / "out" / "yue2test-best.safetensors"
    assert path.is_file()
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        keys = list(f.keys())
    assert meta["modelspec.architecture"] == "YuE2-3B/lora" and meta["modelspec.resolution"] == "48000x2"
    assert json.loads(meta["ss_network_args"])["branches"] == "nar"
    assert meta["ss_steps"] == "1" and meta["ss_yue2_branches"] == "nar" and "torch.optim.adamw.AdamW" in meta["ss_optimizer"]
    assert keys and all(k.startswith("lora_unet_nar_") for k in keys)


def test_extra_metadata_keys():
    trainer, args, _ = _setup("ar,nar", timestep_sampling="beta")
    meta = trainer.extra_metadata(args)
    for key in (
        "ss_yue2_protocol",
        "ss_yue2_branches",
        "ss_yue2_nar_context",
        "ss_yue2_nar_cond_end",
        "ss_yue2_window_frames",
        "ss_yue2_cot",
        "ss_yue2_caption_dropout_scope",
        "ss_yue2_loss_weights",
        "ss_yue2_t_embed_dtype",
        "ss_yue2_timestep_sampling",
        "ss_yue2_instrumental_lyrics",
        "ss_yue2_first_timestep_chance",
        "ss_yue2_ar_pad_multiple",
        "ss_yue2_text_cache_version",
    ):
        assert key in meta, key
    assert meta["ss_yue2_timestep_sampling"] == "beta(2.0,2.0)" and meta["ss_yue2_loss_weights"] == "1.0,1.0,0.2"


def test_scale_weight_norms_runs_with_the_split_layout():
    trainer, args, model = _setup("ar,nar", nar_window_frames=16)
    net = _network(model, args, randomize_up=True)
    assert net.layout == "split"
    loss, _ = _step(trainer, args, model, net, _batch())
    loss.backward()
    keys_scaled, mean_norm, max_norm = net.apply_max_norm_regularization(1e-3, "cpu")
    assert keys_scaled > 0 and max_norm <= 1e-3 + 1e-6


def test_minted_only_validation_ranks_by_minted_ce():
    trainer, args, model = _setup("ar", validate_every_n_steps=1)
    codes = torch.randint(0, P.CODEC_SIZE, (30,), generator=torch.Generator().manual_seed(3))
    pack = YuE2MintedPack(None, records=[MintedRecord(codes=codes, style="edm", lyrics="", song_id="m0", is_validation=None)])
    pack.tokenize(TOK)
    trainer._minted_val = pack
    net = _network(model, args)
    trainer.on_train_start(args, _Accelerator(), net, model, None)
    assert trainer._validator is not None and not trainer._validator.batches
    results = trainer._validator.run(trainer, args, _Accelerator(), model, net, None)
    assert results["val/minted_ce"] > 0 and results["val/total"] == results["val/minted_ce"]

    # minted songs validate only the AR branch
    trainer, args, model = _setup("nar", validate_every_n_steps=1)
    trainer._minted_val = pack
    trainer.on_train_start(args, _Accelerator(), _network(model, args), model, None)
    assert trainer._validator is None


# endregion

# region exports, checkpoint rotation, resume


def _base_lora_file(tmp_path: Path, model) -> Path:
    """A NAR base LoRA with random blocks and an I/O diff with a bias delta."""
    args = _args(train_branches="nar")
    T.YuE2NetworkTrainer().handle_model_specific_args(args)
    base = _network(model, args, randomize_up=True)
    sd = {k: v.detach().float().contiguous() for k, v in base.state_dict().items()}
    g = torch.Generator().manual_seed(11)
    sd["lora_unet_nar_vae2llm.diff"] = 0.01 * torch.randn(CFG.hidden_size, 64, generator=g)
    sd["lora_unet_nar_vae2llm.diff_b"] = 0.01 * torch.randn(CFG.hidden_size, generator=g)
    path = tmp_path / "base_lora.safetensors"
    save_file(sd, str(path))
    return path


def test_exported_base_weights_use_their_merge_multiplier(tmp_path):
    base_path = _base_lora_file(tmp_path, _model(1))
    trainer, args, model = _setup(
        "nar",
        base_weights=[str(base_path)],
        base_weights_multiplier=[0.5],
        yue2_export_include_base_weights=True,
        yue2_export_formats="comfy",
    )
    net = _network(model, args)  # zero lora_up: the export holds only the base LoRA
    native = trainer._export_native(args, net)
    sd, meta = lora_formats.load_lora_file(str(base_path))
    expected = lora_formats.weight_deltas(lora_formats.to_native(sd, meta))
    got = lora_formats.weight_deltas(native)
    assert "nar.vae2llm" in expected and len(expected) > 1
    for path, (w, b) in expected.items():
        assert torch.allclose(got[path][0], 0.5 * w, atol=1e-6), path
        if b is not None:
            assert torch.allclose(got[path][1], 0.5 * b, atol=1e-7), path

    # hf cannot hold the included I/O bias delta: rejected at startup, not at the first save
    args = _args(base_weights=[str(base_path)], yue2_export_include_base_weights=True, yue2_export_formats="hf")
    with pytest.raises(ValueError, match="bias delta"):
        T.YuE2NetworkTrainer().handle_model_specific_args(args)


def test_a_failed_export_is_logged_and_training_continues(tmp_path, monkeypatch, caplog):
    trainer, args, model = _setup("nar", yue2_export_formats="fl,comfy")
    args.output_dir = str(tmp_path)
    net = _network(model, args)

    def broken(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(lora_formats, "native_to_fl", broken)
    with caplog.at_level("ERROR"):
        trainer.on_post_save(args, _Accelerator(), net, model, "yue2test.safetensors", None, {}, False)
    assert "fl export" in caplog.text and "boom" in caplog.text
    assert (tmp_path / "yue2test.comfy.safetensors").is_file()


@pytest.mark.parametrize("unit", ["steps", "epochs"])
def test_rotated_checkpoints_lose_their_exports(tmp_path, unit):
    from musubi_tuner.utils import train_utils

    if unit == "steps":
        kw = {"save_every_n_steps": 2, "save_last_n_steps": 2}
        names = [train_utils.get_step_ckpt_name("yue2test", n) for n in (2, 4, 6)]
    else:
        kw = {"save_every_n_epochs": 1, "save_last_n_epochs": 1}
        names = [train_utils.get_epoch_ckpt_name("yue2test", n) for n in (1, 2, 3)]
    trainer, args, model = _setup("nar", yue2_export_formats="comfy,fl", **kw)
    args.output_dir = str(tmp_path)
    net = _network(model, args)
    for name in names:
        trainer.on_post_save(args, _Accelerator(), net, model, name, None, {}, False)
    stems = [name[: -len(".safetensors")] for name in names]
    # steps: step 6 removes step 2 (last 2 steps kept); epochs: epoch 2 removes 1, epoch 3 removes 2
    removed = stems[:1] if unit == "steps" else stems[:2]
    for stem in stems:
        for suffix in (".comfy", ".fl-nar"):
            assert (tmp_path / f"{stem}{suffix}.safetensors").is_file() == (stem not in removed), (stem, suffix)


class _StateAccelerator(_Accelerator):
    def __init__(self):
        self.save_hooks, self.load_hooks = [], []

    def register_save_state_pre_hook(self, hook):
        self.save_hooks.append(hook)

    def register_load_state_pre_hook(self, hook):
        self.load_hooks.append(hook)

    def save_state(self, output_dir):
        for hook in self.save_hooks:
            hook([], [], output_dir)

    def load_state(self, input_dir):
        for hook in self.load_hooks:
            hook([], input_dir)


def test_best_validation_survives_resume(tmp_path):
    trainer, args, model = _setup("nar", save_best_validation=True, validate_every_n_steps=1)
    args.output_dir = str(tmp_path / "out")
    net = _network(model, args)
    trainer._best_val, trainer._best_step = 0.25, 7
    acc = _StateAccelerator()
    trainer._register_hooks_and_resume(args, acc, net)
    acc.save_state(str(tmp_path / "state"))
    assert json.loads((tmp_path / "state" / T.BEST_STATE_FILE).read_text()) == {"best_val": 0.25, "best_step": 7}

    resumed, args2, _ = _setup("nar", save_best_validation=True, validate_every_n_steps=1, resume=str(tmp_path / "state"))
    args2.output_dir = str(tmp_path / "out")
    resumed._register_hooks_and_resume(args2, _StateAccelerator(), net)
    assert (resumed._best_val, resumed._best_step) == (0.25, 7)
    # a worse validation after the resume does not overwrite the -best file
    resumed._validator = SimpleNamespace(run=lambda *a: {"val/total": 0.5})
    resumed._validate(args2, _Accelerator(), net, model, 10)
    assert resumed._best_step == 7 and not (tmp_path / "out" / "yue2test-best.safetensors").exists()


def test_best_checkpoint_shares_the_base_session_id():
    trainer, args, _ = _setup("nar")
    session_id, started = trainer._init_session(args)
    trainer.on_train_start(args, _Accelerator(), None, None, None)
    meta = trainer._build_save_metadata(args, 1, 1)
    assert meta["ss_session_id"] == str(session_id) and meta["ss_training_started_at"] == str(started)


# endregion

# region sampling hooks


def _fake_sampling_modules(monkeypatch, written):
    sampling = types.ModuleType("musubi_tuner.yue2.yue2_sampling")

    @dataclasses.dataclass
    class YuE2SongRequest:
        style: str = ""
        lyrics: str = ""
        cot: str = "full"
        seed: int = 0
        seconds: float = None
        text_ids: list = None

    def decode_latents(vae, latents, core_frames=1024, halo_frames=16):
        return vae.decode_tiled(latents.float().T[None], output_device="cpu")[0].clamp(-1, 1)

    def reconstruct(model, vae, prefix, codes, seed, steps=32, state_dtype=torch.bfloat16, grad_ctx=torch.no_grad, **kwargs):
        assert not torch.is_inference_mode_enabled() and kwargs["nar_context"] == "codes"
        with grad_ctx():
            ids = torch.tensor(prefix + P.codec_to_ids(codes.tolist()) + [P.MUSIC_END])[None]
            kv = model.ar_forward(model.embed(ids), return_kv=True, return_hidden=False).kv
            lat = model.nar_forward(torch.randn(1, len(codes), 64), torch.tensor([0.5]), kv, ids.shape[1])
            wave = vae.decode(lat.float().transpose(1, 2))[0].clamp(-1, 1)
        return SimpleNamespace(waveform=wave, latents=lat[0])

    def render_song(model, vae, tokenizer, request, grad_ctx=torch.no_grad):
        assert isinstance(request, YuE2SongRequest) and request.text_ids
        return SimpleNamespace(waveform=torch.zeros(2, 1920 * 5))

    sampling.YuE2SongRequest = YuE2SongRequest
    sampling.decode_latents = decode_latents
    sampling.reconstruct = reconstruct
    sampling.render_song = render_song
    audio_io = types.ModuleType("musubi_tuner.yue2.yue2_audio_io")

    def write_audio(path, waveform, sample_rate=48000, fmt="flac", metadata=None):
        written.append((path, tuple(waveform.shape), sample_rate, fmt))
        Path(path).write_bytes(b"fLaC")

    audio_io.write_audio = write_audio
    for name, module in (("yue2_sampling", sampling), ("yue2_audio_io", audio_io)):
        monkeypatch.setitem(sys.modules, f"musubi_tuner.yue2.{name}", module)
        monkeypatch.setattr(yue2_pkg, name, module, raising=False)


def test_sampling_under_no_grad_then_a_training_step(tmp_path, monkeypatch):
    written = []
    _fake_sampling_modules(monkeypatch, written)
    items = [_write_song(tmp_path, "v0", 40, abc=True)]
    trainer, args, model = _setup("ar,nar", nar_window_frames=16, sample_seconds=1.0)
    trainer._validation_items = items
    prompts = tmp_path / "prompts.json"
    (tmp_path / "lyrics.txt").write_text("[verse]\nhello", encoding="utf-8")
    prompts.write_text(
        json.dumps([{"prompt": "rock"}, {"prompt": "pop", "mode": "render", "lyrics_file": "lyrics.txt", "seed": 3}])
    )
    args.sample_prompts = str(prompts)
    monkeypatch.setattr(T, "load_yue2_tokenizer", lambda tokenizer, dit: TOK)
    monkeypatch.setattr(T, "load_yue2_vae", lambda *a, **k: FakeYuE2VAE(decoder_only=True))
    params, resources = trainer.prepare_sampling(args, _Accelerator(), torch.float32)
    assert params[0]["yue2_sample"]["mode"] == "reconstruct" and len(params[0]["yue2_sample"]["codes"]) == 25
    assert params[1]["yue2_sample"]["text_ids"] == P.text_ids(TOK, "pop", "[verse]\nhello", "off")

    net = _network(model, args)
    for param in params:
        gen = torch.Generator().manual_seed(param.get("seed", 0))
        with torch.no_grad():
            sample = trainer.do_inference(
                _Accelerator(), args, param, resources, None, model, 1.0, 32, 256, 256, 1, gen, False, 1.0, None
            )
        trainer.save_sample(_Accelerator(), args, param, sample, str(tmp_path), f"s{param['enum']}", 0)
    names = sorted(os.path.basename(p) for p, *_ in written)
    assert names == ["s0.flac", "s0_gt.flac", "s1.flac"]
    assert written[0][1][0] == 2 and written[0][2] == 48000

    loss, _ = _step(trainer, args, model, net, _batch(abc=True))
    loss.backward()  # no inference tensors leaked into training


# endregion

# region end-to-end CPU runs of the real training loop

RUNNER = r"""
import json, sys
import torch
from musubi_tuner.hv_train_network import setup_parser_common
from musubi_tuner import yue2_train_network as T
from musubi_tuner.yue2.yue2_model import YuE2Config

cfg = YuE2Config(**json.loads(sys.argv[1]))
parser = T.yue2_setup_parser(setup_parser_common())
args = parser.parse_args(sys.argv[2:])
args.dit_dtype = "bfloat16"
args.vae_dtype = "float32"
trainer = T.YuE2NetworkTrainer()
trainer.model_config = cfg
losses, logs = [], []
orig = trainer.process_batch
def process_batch(*a, **k):
    loss, metrics = orig(*a, **k)
    losses.append(loss.item())
    logs.append(metrics)
    return loss, metrics
trainer.process_batch = process_batch
trainer.train(args)
print("YUE2_RESULT=" + json.dumps({"losses": losses, "logs": logs, "best": trainer._best_step}))
"""


def _tiny_checkpoint(path: Path) -> str:
    torch.manual_seed(0)
    model = YuE2Model(CFG)
    save_file({k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}, str(path))
    return str(path)


def _e2e_dataset(tmp_path: Path, batch_size=1) -> str:
    audio, cache = tmp_path / "audio", tmp_path / "cache"
    audio.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    for i in range(3):
        _write_song(cache, f"song{i}", 40, abc=i == 1, seed=i)
    config = tmp_path / "dataset.toml"
    config.write_text(
        toml.dumps(
            {
                "general": {"batch_size": batch_size},
                "datasets": [{"audio_directory": str(audio), "cache_directory": str(cache), "validation_split": 0.34}],
            }
        )
    )
    return str(config)


def _run_e2e(tmp_path: Path, name: str, extra: list[str]) -> dict:
    ckpt = tmp_path / "tiny.safetensors"
    if not ckpt.exists():
        _tiny_checkpoint(ckpt)
    config = _e2e_dataset(tmp_path)
    out = tmp_path / name
    cmd = [
        sys.executable,
        "-c",
        RUNNER,
        json.dumps(dataclasses.asdict(CFG)),
        "--dataset_config",
        config,
        "--dit",
        str(ckpt),
        "--output_dir",
        str(out),
        "--output_name",
        name,
        "--sdpa",
        "--mixed_precision",
        "bf16",
        "--max_train_steps",
        "3",
        "--learning_rate",
        "1e-3",
        "--network_dim",
        "4",
        "--network_alpha",
        "4",
        "--seed",
        "42",
        "--optimizer_type",
        "AdamW",
        "--max_data_loader_n_workers",
        "0",
        "--nar_window_frames",
        "16",
        "--gradient_checkpointing",
    ] + extra
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("YUE2_RESULT=")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-6000:]
    result = json.loads(lines[-1][len("YUE2_RESULT=") :])
    result["out"] = out
    return result


def test_end_to_end_joint_training_run_is_reproducible(tmp_path):
    extra = [
        "--train_branches",
        "ar,nar",
        "--validate_every_n_steps",
        "2",
        "--save_best_validation",
        "--gradient_checkpointing_cpu_offload",
    ]
    a = _run_e2e(tmp_path, "run_a", extra)
    b = _run_e2e(tmp_path, "run_b", extra)
    assert len(a["losses"]) == 3 and all(torch.isfinite(torch.tensor(a["losses"])))
    assert a["losses"] == b["losses"]  # derived planning RNG + seeded noise
    assert all("loss/ar_ce" in logs and "loss/flow" in logs for logs in a["logs"])
    final = a["out"] / "run_a.safetensors"
    best = a["out"] / "run_a-best.safetensors"
    assert final.is_file() and best.is_file() and a["best"] == 2
    with safe_open(str(final), framework="pt") as f:
        meta = f.metadata()
        keys = list(f.keys())
    assert meta["modelspec.architecture"] == "YuE2-3B/lora" and meta["ss_yue2_branches"] == "ar,nar"
    assert any(k.startswith("lora_unet_ar_blocks_") for k in keys) and any(k.startswith("lora_unet_nar_blocks_") for k in keys)
    # the saved LoRA loads back through the from-weights path
    with safe_open(str(final), framework="pt") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    net = lora_yue2.create_arch_network_from_weights(1.0, sd, unet=YuE2Model(CFG), for_inference=True)
    assert len(net.unet_loras) == len({k.split(".")[0] for k in keys})


def _sampling_assets(tmp_path: Path) -> list[str]:
    from yue2_fakes import make_tiny_tokenizer_json, tiny_vae_configs, tiny_vae_state_dict

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    encoder, decoder = tiny_vae_configs()
    (vae_dir / "config.json").write_text(json.dumps({"encoder_config": encoder, "decoder_config": decoder}))
    save_file(tiny_vae_state_dict(), str(vae_dir / "model.safetensors"))
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(make_tiny_tokenizer_json())
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps(
            [
                {"prompt": "pop", "mode": "reconstruct", "seconds": 1.0, "seed": 1, "sample_steps": 4},
                {"prompt": "rock", "lyrics": "la la", "mode": "render", "seconds": 1.0, "seed": 2, "sample_steps": 4},
            ]
        )
    )
    return ["--vae", str(vae_dir), "--tokenizer", str(tokenizer), "--sample_prompts", str(prompts)]


def test_end_to_end_nar_run_with_sampling_exports_and_d50_guard(tmp_path):
    extra = [
        "--yue2_export_formats",
        "comfy,hf,fl",
        "--timestep_sampling",
        "beta",
        "--sample_every_n_steps",
        "2",
        "--sample_at_first",
    ]
    result = _run_e2e(tmp_path, "nar", extra + _sampling_assets(tmp_path))
    assert len(result["losses"]) == 3
    for suffix in ("nar.safetensors", "nar.comfy.safetensors", "nar.hf-nar.safetensors", "nar.fl-nar.safetensors"):
        assert (result["out"] / suffix).is_file(), suffix
    samples = sorted(p.name for p in (result["out"] / "sample").glob("*.flac"))
    # step 0 and step 2 (the base samples again at the end of epoch 1, which is also step 2); one ground truth
    assert len(samples) >= 5 and sum(name.endswith("_gt.flac") for name in samples) == 1, samples

    # batch_size 2 with a swapped trained stack is rejected before the model loads
    config = _e2e_dataset(tmp_path, batch_size=2)
    args = _args(dataset_config=config, nar_blocks_to_swap=1, train_branches="nar", max_data_loader_n_workers=0)
    trainer = T.YuE2NetworkTrainer()
    trainer.handle_model_specific_args(args)
    with pytest.raises(ValueError, match="batch_size=1"):
        trainer._build_dataset(args)


def test_end_to_end_train_io_full_exports_rotation_and_best_state(tmp_path):
    extra = [
        "--train_io",
        "full",
        "--yue2_export_formats",
        "comfy,fl",
        "--save_every_n_steps",
        "1",
        "--save_last_n_steps",
        "1",
        "--save_state",
        "--validate_every_n_steps",
        "1",
        "--save_best_validation",
    ]
    result = _run_e2e(tmp_path, "rot", extra)
    out = result["out"]
    files = {p.name for p in out.glob("*.safetensors")}
    exports = {n for n in files if any(n.endswith(s + ".safetensors") for s in T.EXPORT_SUFFIXES)}
    mains = {n[: -len(".safetensors")] for n in files - exports}
    # step 1 was rotated out with its exports; every remaining checkpoint, -best included, has its exports
    assert "rot-step00000001" not in mains and {"rot-step00000002", "rot", "rot-best"} <= mains, files
    assert {n.split(".")[0] for n in exports} == mains, files
    with safe_open(str(out / "rot.fl-nar.safetensors"), framework="pt") as f:
        assert any(k.endswith(".diff_b") for k in f.keys())  # the trained I/O bias delta is exported
    states = [p for p in out.iterdir() if p.is_dir() and p.name.endswith("-state")]
    assert states and all((s / T.BEST_STATE_FILE).is_file() for s in states), states


# endregion
