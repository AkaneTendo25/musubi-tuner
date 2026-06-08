"""Inference-only vendored copy of Meta's ImageBind (facebookresearch/ImageBind).

Vendored from the ``facebookresearch/ImageBind`` tree so the ``imagebind`` reward
plugin no longer depends on external code. Only the pieces needed for
in-process multimodal-similarity scoring are kept:

  - ``imagebind_model.imagebind_huge`` (the OUT_EMBED_DIM=1024 "huge" model) +
    its supporting preprocessors / trunk / heads, and
  - ``data.load_and_transform_{vision,video,audio,text}_data`` transforms.

Deviations from upstream (inference-only, training deps stripped):
  - ``data.py`` keeps the torch/torchvision/torchaudio transforms, but its heavy
    ``imagebind.*`` absolute imports are rewritten to relative imports.
  - The two ``timm.models.layers`` symbols (``trunc_normal_``, ``DropPath``) are
    inlined verbatim in ``_timm_compat.py`` so ``timm`` is no longer a dependency.
    They are init-/train-time only, so a loaded checkpoint's eval forward pass is
    identical to the timm-based version.
  - The video-FILE transform (``data.load_and_transform_video_data``) no longer uses
    ``pytorchvideo``; it decodes with ``torchvision.io.read_video`` and reimplements the
    clip/frame sampling (``ConstantClipsPerVideoSampler`` / ``UniformTemporalSubsample`` /
    ``ShortSideScale``) in pure torch. Clip timepoints + spatial/normalize steps match
    upstream exactly, but the decoder backend differs, so video embeddings are
    APPROXIMATE (not bit-identical) relative to the pytorchvideo path. Text/image/audio
    paths are unaffected and remain exact.
  - ``multimodal_preprocessors.SimpleTokenizer`` no longer depends on ``iopath``;
    the bundled bpe vocab is opened with a plain ``gzip.open`` (the bpe file is
    vendored alongside this package at ``bpe/bpe_simple_vocab_16e6.txt.gz``).
  - ``imagebind_model.imagebind_huge`` no longer auto-downloads the checkpoint;
    the reward plugin passes an explicit ``checkpoint_path``.

Dependency surface (inference): torch, torchvision, torchaudio, einops, ftfy,
regex, numpy. NO timm / pytorchvideo / fvcore / iopath / matplotlib / mayavi /
cartopy deps.

License: see ``LICENSE`` in this directory (CC-BY-NC-SA 4.0, Meta Platforms).
"""

from . import data  # noqa: F401
from .models import imagebind_model  # noqa: F401
from .models.imagebind_model import ModalityType, imagebind_huge  # noqa: F401

__all__ = ["data", "imagebind_model", "imagebind_huge", "ModalityType"]
