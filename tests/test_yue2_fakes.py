import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import FakeOffloader, FakeOffloaderFactory, FakeTokenizer, FakeYuE2VAE, load_fake_yue2_vae  # noqa: E402

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, ModelOffloader  # noqa: E402
from musubi_tuner.yue2 import yue2_protocol as p  # noqa: E402

N = 6


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return x + torch.tanh(self.linear(x))


def _blocks(device="cpu"):
    torch.manual_seed(0)
    return nn.ModuleList([_Block() for _ in range(N)]).to(device)


def _run(off, blocks, x):
    for i, block in enumerate(blocks):
        off.wait_for_block(i)
        x = block(x)
        off.submit_move_blocks_forward(blocks, i)
    return x


# region fake offloader schedule (CPU)


@pytest.mark.parametrize("swap", [1, 2, 3, 4])
def test_forward_only_pass_restores_layout(swap):
    blocks = _blocks()
    off = FakeOffloader("t", blocks, N, swap, supports_backward=False)
    off.prepare_block_devices_before_forward(blocks)
    for _ in range(3):
        _run(off, blocks, torch.randn(2, 8))
        assert off.resident == off.initial_layout
    assert off.passes() == [True, True, True]


@pytest.mark.parametrize("swap", [1, 2, 3, 4])
def test_training_pass_and_backward_restore_layout(swap):
    blocks = _blocks()
    off = FakeOffloader("t", blocks, N, swap, supports_backward=True)
    off.prepare_block_devices_before_forward(blocks)
    for _ in range(2):
        out = _run(off, blocks, torch.randn(2, 8))
        assert off.resident == set(range(swap, N))
        out.square().mean().backward()
        assert off.resident == off.initial_layout
    # a forward-only pass on the backward-capable offloader after backward
    off.set_forward_only(True)
    with torch.no_grad():
        _run(off, blocks, torch.randn(2, 8))
    assert off.resident == off.initial_layout
    assert off.passes() == [False, False, True]
    off.remove_hooks()


@pytest.mark.parametrize("swap", [1, 3])
def test_pass_before_backward_is_detected(swap):
    blocks = _blocks()
    off = FakeOffloader("t", blocks, N, swap, supports_backward=True)
    off.prepare_block_devices_before_forward(blocks)
    _run(off, blocks, torch.randn(2, 8))
    # a second grad pass before backward
    with pytest.raises(AssertionError, match="not resident"):
        _run(off, blocks, torch.randn(2, 8))
    # a no-grad pass after a grad pass (blocks 0..S-1 are still on CPU until backward)
    off2 = FakeOffloader("t", _blocks(), N, swap, supports_backward=False)
    off2.prepare_block_devices_before_forward(blocks)
    off2.set_forward_only(False)
    _run(off2, blocks, torch.randn(2, 8))
    off2.set_forward_only(True)
    with pytest.raises(AssertionError, match="not resident"):
        _run(off2, blocks, torch.randn(2, 8))
    off.remove_hooks()


def test_unprepared_and_no_swap():
    blocks = _blocks()
    off = FakeOffloader("t", blocks, N, 2, supports_backward=False)
    with pytest.raises(AssertionError, match="prepare"):
        off.wait_for_block(0)
    idle = FakeOffloader("t", blocks, N, 0, supports_backward=True)
    _run(idle, blocks, torch.randn(1, 8))
    assert idle.events == []


def test_factory_builds_from_config():
    factory = FakeOffloaderFactory()
    blocks = _blocks()
    cfg = BlockSwapConfig(device=torch.device("cpu"), supports_backward=True)
    off = factory("yue2-nar", blocks, N, 2, cfg)
    assert factory.created["yue2-nar"] is off
    assert off.supports_backward and not off.forward_only and off.blocks_to_swap == 2
    off.remove_hooks()


# endregion

# region fake offloader vs the real ModelOffloader (CUDA)


def _real_resident(blocks):
    return {i for i, b in enumerate(blocks) if b.linear.weight.device.type == "cuda"}


def _flush(off):
    for idx in list(off.futures.keys()):
        off._wait_blocks_move(idx)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the real ModelOffloader needs CUDA")
@pytest.mark.parametrize("swap", [1, 2, 3, 4])
def test_fake_matches_real_model_offloader(swap):
    dev = torch.device("cuda")
    real_blocks = _blocks(dev)
    fake_blocks = _blocks()
    real = ModelOffloader("real", real_blocks, N, swap, True, dev)
    fake = FakeOffloader("fake", fake_blocks, N, swap, supports_backward=True)
    real.prepare_block_devices_before_forward(real_blocks)
    fake.prepare_block_devices_before_forward(fake_blocks)
    assert _real_resident(real_blocks) == fake.resident == set(range(N - swap))

    x = torch.randn(2, 8)
    for off in (real, fake):
        off.set_forward_only(True)
    with torch.no_grad():
        ref = _run(real, real_blocks, x.to(dev))
        _run(fake, fake_blocks, x)
    _flush(real)
    assert _real_resident(real_blocks) == fake.resident

    for off in (real, fake):
        off.set_forward_only(False)
    out_real = _run(real, real_blocks, x.to(dev))
    out_fake = _run(fake, fake_blocks, x)
    _flush(real)
    assert _real_resident(real_blocks) == fake.resident == set(range(swap, N))

    out_real.square().mean().backward()
    out_fake.square().mean().backward()
    _flush(real)
    assert _real_resident(real_blocks) == fake.resident == set(range(N - swap))

    for off in (real, fake):
        off.set_forward_only(True)
    with torch.no_grad():
        again = _run(real, real_blocks, x.to(dev))
        _run(fake, fake_blocks, x)
    _flush(real)
    assert _real_resident(real_blocks) == fake.resident
    torch.testing.assert_close(again, ref)
    fake.remove_hooks()


# endregion

# region fake VAE and tokenizer


def test_fake_vae_shapes_and_chunking():
    vae = FakeYuE2VAE()
    torch.manual_seed(1)
    frames = 37
    audio = torch.randn(2, frames * p.HOP)
    full = vae.encode_mean(audio[None])[0].T
    assert full.shape == (frames, p.LATENT_DIM)
    chunked = vae.encode_mean_chunked(audio, chunk_frames=8, overlap_frames=1)
    torch.testing.assert_close(chunked, full)
    # a segment encoded alone differs from the slice of the whole-record encode at its edges
    seg = vae.encode_mean(audio[None, :, 10 * p.HOP : 20 * p.HOP])[0].T
    assert not torch.allclose(seg, full[10:20])
    torch.testing.assert_close(seg[1:-1], full[11:19])
    wave = vae.decode(full.T[None])
    assert wave.shape == (1, 2, p.HOP * frames - 64)
    torch.testing.assert_close(vae.decode_tiled(full.T[None]), wave)
    with pytest.raises(ValueError):
        vae.encode_mean(torch.randn(1, 2, p.HOP + 1))


def test_fake_vae_loader_and_tokenizer():
    assert load_fake_yue2_vae(None, None).source_dtype == "float32"
    assert load_fake_yue2_vae(source_dtype="float16").fingerprint != load_fake_yue2_vae().fingerprint
    with pytest.raises(ValueError):
        load_fake_yue2_vae(source_dtype="float16", allow_fp16_source=False)
    tok = FakeTokenizer()
    ids = tok.encode("héllo <abc>")
    assert all(0 <= i < p.EOD for i in ids)
    assert tok.decode(ids) == "héllo <abc>"


# endregion
