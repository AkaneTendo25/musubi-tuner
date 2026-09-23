import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import tiny_vae_configs, tiny_vae_state_dict  # noqa: E402

from musubi_tuner.yue2.yue2_checkpoint import load_yue2_vae  # noqa: E402
from musubi_tuner.yue2.yue2_vae import HOP, YuE2VAE  # noqa: E402


@pytest.fixture(scope="module")
def vae():
    model = YuE2VAE(*tiny_vae_configs())
    model.load_state_dict_any(tiny_vae_state_dict())
    return model


def _audio(frames, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.3 * torch.randn(1, 2, frames * HOP, generator=g)


def test_legacy_and_parametrized_keys_load(vae):
    sd = tiny_vae_state_dict()
    assert any(k.endswith("weight_g") for k in sd)
    renamed = {
        k.replace(".weight_g", ".parametrizations.weight.original0").replace(".weight_v", ".parametrizations.weight.original1"): v
        for k, v in sd.items()
    }
    other = YuE2VAE(*tiny_vae_configs())
    other.load_state_dict_any(renamed)
    for k, v in other.state_dict().items():
        assert torch.equal(v, sd[k])
    with pytest.raises(ValueError, match="float32"):
        other.load_state_dict_any({k: v.half() for k, v in sd.items()})


def test_encode_shape_and_decode_length(vae):
    lat = vae.encode_mean(_audio(7))
    assert lat.shape == (1, 64, 7) and lat.dtype == torch.float32
    wav = vae.decode(lat)
    assert wav.shape == (1, 2, 7 * HOP - 64)
    assert vae.natural_output_length(7) == 7 * HOP - 64
    with pytest.raises(ValueError):
        vae.encode_mean(torch.zeros(1, 2, HOP + 5))


def test_decode_tiled_equals_decode(vae):
    lat = torch.randn(1, 64, 23, generator=torch.Generator().manual_seed(1))
    full = vae.decode(lat)
    halo = vae.required_halo(5)
    tiled = vae.decode_tiled(lat, core_frames=5, halo_frames=halo)
    assert tiled.shape == full.shape
    # exact up to fp32 convolution-algorithm rounding
    assert (tiled - full).abs().max().item() / full.abs().max().item() < 1e-5
    assert halo > 0
    with pytest.raises(ValueError, match="halo_frames"):
        vae.decode_tiled(lat, core_frames=5, halo_frames=halo - 1)


def test_chunked_encode_matches_full(vae):
    frames = 40
    audio = _audio(frames, seed=2)[0]
    full = vae.encode_mean(audio[None])[0].T
    rf = vae.encoder_receptive_frames()
    chunked = vae.encode_mean_chunked(audio, chunk_frames=7, overlap_frames=rf + 1)
    assert chunked.shape == (frames, 64)
    assert (chunked - full).abs().max().item() < 1e-4
    short = vae.encode_mean_chunked(audio, chunk_frames=7, overlap_frames=0)
    assert (short - full).abs().max().item() > 1e-4  # without overlap the chunk edges differ


def test_reference_parity(vae):
    pytest.importorskip("transformers")
    from yue2_ref.modeling_vae import YuE2VAE as RefVAE, YuE2VAEConfig

    enc, dec = tiny_vae_configs()
    ref = RefVAE(
        YuE2VAEConfig(encoder_config=enc, decoder_config=dec, decode_core_frames=5, decode_halo_frames=vae.required_halo(5))
    )
    ref.load_state_dict(tiny_vae_state_dict(), strict=True)
    audio = _audio(9, seed=4)
    assert (ref.encode(audio) - vae.encode_mean(audio)).abs().max().item() < 1e-6
    lat = torch.randn(1, 64, 12)
    assert (ref.decode(lat) - vae.decode(lat)).abs().max().item() < 1e-6
    assert (
        ref.decode_tiled(lat) - vae.decode_tiled(lat, core_frames=5, halo_frames=vae.required_halo(5))
    ).abs().max().item() < 1e-6
    assert ref.required_halo(5) == vae.required_halo(5)


def test_fp16_source_rejected_and_fingerprint(tmp_path):
    enc, dec = tiny_vae_configs()
    sd = tiny_vae_state_dict()
    f32 = tmp_path / "f32.safetensors"
    f16 = tmp_path / "f16.safetensors"
    save_file(sd, str(f32))
    save_file({k: v.half() for k, v in sd.items()}, str(f16))
    with pytest.raises(ValueError, match="not float32"):
        load_yue2_vae(str(f16), None, vae_configs=(enc, dec), allow_fp16_source=False)
    a = load_yue2_vae(str(f32), None, vae_configs=(enc, dec), allow_fp16_source=False)
    b = load_yue2_vae(str(f16), None, vae_configs=(enc, dec))
    assert a.source_dtype == "float32" and b.source_dtype == "float16"
    assert a.fingerprint != b.fingerprint and a.fingerprint.startswith("sha256:")
    assert all(p.dtype == torch.float32 for p in b.parameters())
    with pytest.raises(ValueError, match="--vae"):
        load_yue2_vae(None, None)
