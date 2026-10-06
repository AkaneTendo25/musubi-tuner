"""Safetensors state-dict loading for the released SR checkpoints.

The open-source release ships checkpoints as flat ``.safetensors`` state dicts
(one tensor per key, no pickled containers): the DiT as a single merged
``model.safetensors``, each latent upscaler with the EMA weights already
extracted at conversion time, and the KVAE as a ``{prefix}.safetensors``
sidecar. This module is the single load point for that format.
"""

from __future__ import annotations

import torch
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file

SAFETENSORS_SUFFIX = ".safetensors"


def is_safetensors_path(path: str) -> bool:
    """Return ``True`` when ``path`` is a safetensors file."""
    return path.rstrip("/").endswith(SAFETENSORS_SUFFIX)


def load_safetensors_state_dict(
    path: str,
    map_location: str | torch.device | None = None,
    key_prefix: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load a flat safetensors state dict from a local path.

    Args:
        path: Local ``.safetensors`` file.
        map_location: Target device for the loaded tensors (default CPU).
        key_prefix: Load only the tensors whose name starts with it and strip
            it from their names — one model out of a file that holds several.
            The other tensors are never read.

    Returns:
        The state dict with tensors on ``map_location``.

    Raises:
        KeyError: If ``key_prefix`` matches no tensor in the file.
    """
    device = str(map_location) if map_location is not None else "cpu"
    if key_prefix is None:
        return safetensors_load_file(path, device=device)
    with safe_open(path, framework="pt", device=device) as weights:
        names = [name for name in list(weights.keys()) if name.startswith(key_prefix)]
        state_dict = {name.removeprefix(key_prefix): weights.get_tensor(name) for name in names}
    if not state_dict:
        msg = f"{path} holds no tensors under the {key_prefix!r} prefix"
        raise KeyError(msg)
    return state_dict
