"""Write the KVAE sidecar pair the VAE builders load from.

The builders read a KVAE as ``{prefix}.yaml`` (architecture + latent scaling)
next to ``{prefix}.safetensors``. Published checkpoints keep the same weights
under other file names and the architecture in JSON, so a sidecar pair is
written beside them once: the YAML, and a relative symlink to the weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict

SIDECAR_STEM = "kvae"


class KvaeArchitecture(BaseModel):
    """What a KVAE sidecar YAML carries."""

    model_config = ConfigDict(frozen=True)

    scaling_factor: float
    encoder_params: dict[str, Any]
    decoder_params: dict[str, Any]


def write_kvae_sidecar(directory: Path, weights_name: str, architecture: KvaeArchitecture) -> str:
    """Make ``directory`` loadable as a KVAE sidecar pair and return its prefix.

    Existing sidecar files are kept, so repeated calls (and concurrent runs
    sharing a Hub snapshot) agree on one pair.

    Args:
        directory: Directory holding the weights file.
        weights_name: The weights file name inside ``directory``.
        architecture: Latent scaling and encoder / decoder parameters.

    Returns:
        The sidecar prefix (``{directory}/kvae``).
    """
    prefix = directory / SIDECAR_STEM
    sidecar_yaml, sidecar_weights = prefix.with_suffix(".yaml"), prefix.with_suffix(".safetensors")
    if not sidecar_weights.exists():
        sidecar_weights.symlink_to(weights_name)
    if not sidecar_yaml.exists():
        sidecar = {
            "scaling_factor": architecture.scaling_factor,
            "model": {
                "encoder_params": architecture.encoder_params,
                "decoder_params": architecture.decoder_params,
            },
        }
        sidecar_yaml.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        logger.info("Wrote KVAE sidecar config {} for {}", sidecar_yaml, directory / weights_name)
    return str(prefix)
