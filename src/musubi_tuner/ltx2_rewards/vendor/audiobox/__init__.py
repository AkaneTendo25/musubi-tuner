"""Inference-only vendored copy of audiobox-aesthetics (Meta/FAIR, v0.0.4).

Vendored verbatim (architecture + scoring path) from the ``audiobox_aesthetics``
PyPI package, with the training / CLI / download surface stripped:

  - ``infer.py``: keeps ``read_wav`` / ``make_inference_batch`` / ``AesPredictor`` /
    ``initialize_predictor`` byte-for-byte. Drops ``tqdm`` (CLI progress bar), the
    ``json`` ``load_dataset`` / ``main_predict`` CLI helpers, and the
    ``from .utils import load_model`` HF/S3 auto-downloader (replaced with a plain
    local-path check). The ``AesMultiOutput.from_pretrained`` HF fallback is dropped;
    a local ``checkpoint_path`` is required.
  - ``model/aes.py``: verbatim except the ``huggingface_hub.PyTorchModelHubMixin``
    base class is removed (it only adds ``from_pretrained`` / ``save_pretrained``;
    removing it leaves the ``nn.Module`` weights and ``forward`` identical, so scores
    are bit-identical).
  - ``model/wavlm.py`` and ``model/utils.py``: verbatim, no changes.

Dependency surface: torch, torchaudio, numpy. NO huggingface_hub / safetensors /
submitit / rich / tqdm / requests / fire.

Public API mirrors the source: ``initialize_predictor`` (and ``AesPredictor``).
"""

from .infer import AesPredictor, initialize_predictor

__all__ = ["AesPredictor", "initialize_predictor"]
