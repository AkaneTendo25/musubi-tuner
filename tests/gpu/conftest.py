"""Skip gates and shared paths for the YuE2 real-weight GPU tests.

Tests that need released weights take the ``yue2_weights`` fixture; it skips unless ``YUE2_WEIGHTS_DIR`` points at a
directory and CUDA is available. Layout under ``YUE2_WEIGHTS_DIR`` (Hugging Face repo ids with ``/`` -> ``__``):
``m-a-p__YuE2-3B/model.safetensors`` (+ ``config.json``, ``qwen.tiktoken``, the ``yue2_infer-*.whl`` reference),
``m-a-p__YuE2-Vae/``, ``Comfy-Org__YuE2/checkpoints/yue2_3b_{bf16,int8_convrot}.safetensors``,
``m-a-p__MERT-v2-FullSong/`` and ``Mothersuperior__yue2-mothersuperior-realaudio-tokenizer-v4/``.
Audio for the data-driven checks comes from ``YUE2_DATA_DIR/jamendolyrics`` (``mp3/``, ``lyrics/``).
"""

import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@dataclass(frozen=True)
class YuE2Weights:
    root: Path
    hf: Path
    hf_config: Path
    tiktoken: Path
    comfy_bf16: Path
    comfy_int8: Path
    vae_dir: Path
    mert_dir: Path
    ms_dir: Path
    wheel: Optional[Path]

    def require(self, *paths: Path) -> None:
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            pytest.skip(f"missing YuE2 weights: {missing}")


def weights_from_env() -> Optional[YuE2Weights]:
    root = os.environ.get("YUE2_WEIGHTS_DIR")
    if not root or not os.path.isdir(root):
        return None
    root = Path(root)
    hf_dir = root / "m-a-p__YuE2-3B"
    wheels = sorted(glob.glob(str(hf_dir / "yue2_infer-*.whl")))
    ckpt = root / "Comfy-Org__YuE2" / "checkpoints"
    return YuE2Weights(
        root=root,
        hf=hf_dir / "model.safetensors",
        hf_config=hf_dir / "config.json",
        tiktoken=hf_dir / "qwen.tiktoken",
        comfy_bf16=ckpt / "yue2_3b_bf16.safetensors",
        comfy_int8=ckpt / "yue2_3b_int8_convrot.safetensors",
        vae_dir=root / "m-a-p__YuE2-Vae",
        mert_dir=root / "m-a-p__MERT-v2-FullSong",
        ms_dir=root / "Mothersuperior__yue2-mothersuperior-realaudio-tokenizer-v4",
        wheel=Path(wheels[-1]) if wheels else None,
    )


@pytest.fixture(scope="session")
def yue2_weights() -> YuE2Weights:
    weights = weights_from_env()
    if weights is None:
        pytest.skip("YUE2_WEIGHTS_DIR is not set")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return weights


@pytest.fixture(scope="session")
def yue2_data_dir() -> Path:
    root = os.environ.get("YUE2_DATA_DIR")
    path = Path(root) / "jamendolyrics" if root else None
    if path is None or not (path / "mp3").is_dir():
        pytest.skip("YUE2_DATA_DIR/jamendolyrics is not available")
    return path


@pytest.fixture(scope="session")
def reference_wheel(yue2_weights: YuE2Weights):
    """The official ``yue2_infer`` package, imported from its wheel (zipimport; nothing is installed)."""
    if yue2_weights.wheel is None:
        pytest.skip("yue2_infer wheel not found next to the HF checkpoint")
    wheel = str(yue2_weights.wheel)
    if wheel not in sys.path:
        sys.path.insert(0, wheel)
    import yue2  # noqa: F401  (official package name)

    return yue2
