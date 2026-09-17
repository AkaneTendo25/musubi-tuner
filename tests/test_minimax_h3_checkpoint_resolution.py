from pathlib import Path

import pytest

from musubi_tuner.minimax_h3.model_loader import resolve_transformer_checkpoint


def checkpoint_name(family: str, int8_convrot: bool) -> str:
    suffix = "pruned_int8_convrot" if int8_convrot else "bf16"
    return f"minimax_h3_{family}_{suffix}.safetensors"


@pytest.mark.parametrize("int8_convrot", [False, True])
def test_explicit_ref2va_checkpoint_accepts_fl2va_conditioning(tmp_path: Path, int8_convrot: bool):
    checkpoint = tmp_path / checkpoint_name("ref2va", int8_convrot)
    checkpoint.touch()

    assert resolve_transformer_checkpoint(checkpoint, "fl2va", int8_convrot=int8_convrot) == checkpoint


@pytest.mark.parametrize("mode", ["ref2va", "ref2va_omni"])
@pytest.mark.parametrize("int8_convrot", [False, True])
def test_explicit_fl2va_checkpoint_still_rejects_reference_conditioning(tmp_path: Path, mode: str, int8_convrot: bool):
    checkpoint = tmp_path / checkpoint_name("fl2va", int8_convrot)
    checkpoint.touch()

    with pytest.raises(ValueError, match="cannot load"):
        resolve_transformer_checkpoint(checkpoint, mode, int8_convrot=int8_convrot)


@pytest.mark.parametrize("mode,family", [("fl2va", "fl2va"), ("ref2va", "ref2va"), ("ref2va_omni", "ref2va")])
@pytest.mark.parametrize("int8_convrot", [False, True])
@pytest.mark.parametrize("subdir", ["", "diffusion_models"])
def test_directory_defaults_remain_mode_and_precision_specific(
    tmp_path: Path, mode: str, family: str, int8_convrot: bool, subdir: str
):
    directory = tmp_path / subdir
    directory.mkdir(exist_ok=True)
    for checkpoint_family in ("fl2va", "ref2va"):
        for quantized in (False, True):
            (directory / checkpoint_name(checkpoint_family, quantized)).touch()

    assert resolve_transformer_checkpoint(tmp_path, mode, int8_convrot=int8_convrot) == directory / checkpoint_name(
        family, int8_convrot
    )


@pytest.mark.parametrize("int8_convrot", [False, True])
def test_fl2va_directory_does_not_fall_back_to_ref2va(tmp_path: Path, int8_convrot: bool):
    (tmp_path / checkpoint_name("ref2va", int8_convrot)).touch()

    with pytest.raises(FileNotFoundError, match="could not resolve the fl2va"):
        resolve_transformer_checkpoint(tmp_path, "fl2va", int8_convrot=int8_convrot)
