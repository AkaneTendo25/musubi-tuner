from __future__ import annotations

from pathlib import Path

from musubi_tuner.minimax_h3.load_options import H3LoadOptions


def create_backend(*, model: Path, device: str | None, options: H3LoadOptions):
    """Adapt the vendored MiniMax implementation to Musubi.

    Import the config, model, VAE, and encoder classes from ``vendor/official``, validate a
    strict checkpoint load, then return an object satisfying ``H3Backend``.
    """
    from musubi_tuner.minimax_h3.backend import H3BackendUnavailableError

    del model, device, options
    raise H3BackendUnavailableError(
        "MiniMax H3 runtime support is unavailable; install the upstream source under "
        "musubi_tuner/minimax_h3/vendor/official and provide the minimax_h3/integration.py adapter"
    )
