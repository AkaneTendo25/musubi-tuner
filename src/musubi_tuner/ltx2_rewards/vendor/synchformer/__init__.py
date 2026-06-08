"""Inference-only vendored copy of Synchformer (audio-video synchronization model).

Vendored verbatim (architecture) from github.com/v-iashin/Synchformer,
with the training-only / download deps stripped:

  - ``timm`` removed: the two helpers actually used (``trunc_normal_``, ``to_2tuple``)
    are inlined in ``_timm_compat.py``.
  - ``omegaconf`` removed: the MotionFormer ``divided_224_16x4.yaml`` config is inlined
    as a plain attribute-dict (``_MFORMER_CFG`` in ``motionformer.py``).
  - ``requests`` / ``tqdm`` checkpoint auto-download removed: the Synchformer state dict
    is loaded explicitly from a local ``checkpoint_path`` by the reward plugin.
  - The unused ``resnet.py`` audio extractor (an alternative not referenced by the
    ``Synchformer`` class, and with broken upstream imports) is NOT vendored.

Dependency surface: torch, torchaudio, torchvision, torio, einops, transformers, numpy.
NO timm / omegaconf / requests / tqdm / librosa / cv2 / decord / av / sklearn.

Public API mirrors the source: ``Synchformer`` (model) and ``make_class_grid``.
"""

from .synchformer import Synchformer, make_class_grid

__all__ = ["Synchformer", "make_class_grid"]
