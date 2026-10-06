"""Hugging Face Hub references in SR model paths.

The three model-path settings (``sr.checkpoint_path``, ``sr.vae_path``,
``sr.latent_upscaler_config``) accept, in place of a local path:

* a Hub repo id (``namespace/name``) — a ``Kandinsky6SRPipeline`` Diffusers
  bundle such as ``kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers``
  (each setting takes its own component of it), or a native single-model repo;
* one component of a bundle repo: ``namespace/name/transformer``,
  ``namespace/name/vae`` or ``namespace/name/latent_upscaler``;
* a local bundle directory or one of its component directories.

The KVAE and LU settings also accept a native local directory: the sidecar
pair / bank YAML inside it is discovered the same way as inside a snapshot.
A repo is snapshot-downloaded into the standard Hub cache (reused across
runs; for a bundle only the needed components are fetched) and the reference
resolves to the matching local path inside the snapshot. Private repos use
the ambient Hub auth (``hf auth login`` or the ``HF_TOKEN`` env var).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

from huggingface_hub import snapshot_download
from loguru import logger

from musubi_tuner.kandinsky6.runtime.sr.pipeline.diffusers_bundle import (
    COMPONENTS,
    LATENT_UPSCALER,
    MODEL_INDEX,
    SCHEDULER,
    TRANSFORMER,
    VAE,
    is_bundle_dir,
    is_component_dir,
    kvae_sidecar_from_component,
)
from musubi_tuner.kandinsky6.runtime.sr.pipeline.kvae_sidecar import KvaeArchitecture, write_kvae_sidecar

# ``namespace/name`` with the Hub's allowed characters — and exactly one slash,
# so any deeper relative path never matches.
HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


# Files of a bundle repo each component needs: the DiT also takes its sampler settings.
COMPONENT_FILE_PATTERNS = {
    TRANSFORMER: (f"{TRANSFORMER}/*", f"{SCHEDULER}/*"),
    VAE: (f"{VAE}/*",),
    LATENT_UPSCALER: (f"{LATENT_UPSCALER}/*",),
}


class HubReference(NamedTuple):
    """A Hub repo, optionally narrowed to one component folder of a Diffusers bundle."""

    repo_id: str
    component: str | None


def is_hf_repo_id(value: str) -> bool:
    """Return ``True`` for a Hub repo id that is not an existing local path."""
    return bool(HF_REPO_ID_RE.match(value)) and not Path(value).exists()


def parse_hub_reference(value: str) -> HubReference | None:
    """Read ``namespace/name`` or ``namespace/name/<component>``; ``None`` for anything local.

    An existing path always wins, and so does a KVAE sidecar prefix that only
    looks like a component reference (``weights/kvae/vae`` next to ``vae.yaml``).
    """
    if is_hf_repo_id(value):
        return HubReference(value, None)
    repo_id, _, component = value.rpartition("/")
    if component not in COMPONENTS or not HF_REPO_ID_RE.match(repo_id):
        return None
    if Path(value).exists() or Path(f"{value}.yaml").exists():
        return None
    return HubReference(repo_id, component)


def is_hub_reference(value: str) -> bool:
    """Whether ``value`` names a Hub repo or a bundle component in one, rather than a local path."""
    return parse_hub_reference(value) is not None


def download_hf_repo(repo_id: str, allow_patterns: tuple[str, ...] | None = None) -> Path:
    """Snapshot-download a Hub repo, or only the files matching ``allow_patterns`` (cached)."""
    logger.info("Resolving Hugging Face repo '{}' ({}; cached)...", repo_id, allow_patterns or "whole snapshot")
    patterns = list(allow_patterns) if allow_patterns is not None else None
    return Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns))


def download_bundle_component(repo_id: str, component: str) -> Path | None:
    """Fetch one component of a bundle repo; ``None`` when the repo is not a Diffusers bundle.

    Only ``model_index.json`` is fetched to tell a bundle from a native repo,
    so a native repo costs one extra metadata request and nothing else.
    """
    if not is_bundle_dir(download_hf_repo(repo_id, (MODEL_INDEX,))):
        return None
    root = download_hf_repo(repo_id, (MODEL_INDEX, *COMPONENT_FILE_PATTERNS[component]))
    return root / component


def resolve_component_dir(value: str, component: str) -> Path | None:
    """Return the local ``component`` directory a bundle reference stands for.

    Args:
        value: A model reference in any accepted form.
        component: The component the setting loads (``transformer`` / ``vae`` /
            ``latent_upscaler``).

    Returns:
        The component directory, or ``None`` when ``value`` is not a bundle
        reference (a native repo, file or directory).

    Raises:
        ValueError: If ``value`` names another component, a component of a
            repo that is not a bundle, or a bundle without that component.
    """
    reference = parse_hub_reference(value)
    if reference is None:
        path = Path(value)
        if is_bundle_dir(path):
            return existing_component(path / component, value)
        return path if is_component_dir(path) else None
    if reference.component not in (None, component):
        msg = f"{value!r} names the {reference.component} component, but the {component} is loaded from it"
        raise ValueError(msg)
    component_dir = download_bundle_component(reference.repo_id, component)
    if component_dir is None:
        if reference.component is None:
            return None
        msg = f"{value!r}: {reference.repo_id} is not a Diffusers bundle (it has no {MODEL_INDEX})"
        raise ValueError(msg)
    return existing_component(component_dir, value)


def existing_component(component_dir: Path, reference: str) -> Path:
    """Return ``component_dir``, refusing a bundle that was exported without that component."""
    if not is_component_dir(component_dir):
        msg = f"the Diffusers bundle {reference!r} has no {component_dir.name} component"
        raise ValueError(msg)
    return component_dir


def resolve_checkpoint_reference(value: str) -> str:
    """Resolve a DiT checkpoint reference to a local path.

    A bundle (repo, directory or its ``transformer`` component) resolves to the
    ``transformer`` component directory; the sampler settings are read from the
    sibling ``scheduler`` folder. A native repo id becomes its snapshot
    directory (``model.safetensors`` + ``config.yaml`` at the root); any other
    local path passes through.

    Raises:
        ValueError: If the reference names another component of a bundle.
    """
    component_dir = resolve_component_dir(value, TRANSFORMER)
    if component_dir is not None:
        return str(component_dir)
    if is_hf_repo_id(value):
        return str(download_hf_repo(value))
    return value


def bundle_reference(checkpoint_reference: str, resolved_checkpoint: str) -> str | None:
    """Return the bundle a resolved DiT came from, as a reference the other settings can take.

    Args:
        checkpoint_reference: The checkpoint reference as configured.
        resolved_checkpoint: What :func:`resolve_checkpoint_reference` returned for it.

    Returns:
        The repo id or the local bundle root, or ``None`` for a native
        checkpoint (and for a lone ``transformer`` directory outside a bundle).
    """
    if not is_component_dir(Path(resolved_checkpoint)):
        return None
    reference = parse_hub_reference(checkpoint_reference)
    if reference is not None:
        return reference.repo_id
    root = Path(resolved_checkpoint).parent
    return str(root) if is_bundle_dir(root) else None


def has_sidecar_weights(yaml_file: Path) -> bool:
    """Whether ``yaml_file`` is the config half of a ``{prefix}.yaml`` + weights pair."""
    return yaml_file.with_suffix(".safetensors").exists() or yaml_file.with_suffix(".ckpt").exists()


# Latent scaling of the KVAE-3D-2.0-t4s16 latent space (the SR sidecar value;
# the public kvae checkpoints ship no scaling factor in their config.json).
KVAE_T4S16_SCALING_FACTOR = 0.910344004631042
HF_KVAE_CONFIG = "config.json"
HF_KVAE_WEIGHTS = "model.safetensors"


def sidecar_from_hf_kvae(directory: Path) -> str | None:
    """Turn a public-KVAE checkout (``config.json`` + ``model.safetensors``) into a sidecar pair.

    The published ``kandinskylab/KVAE-3D-2.0-*`` repos ship the architecture in
    ``config.json`` (``model.encoder_params`` / ``model.decoder_params`` — the
    same layout as the sidecar YAML's ``model:`` section) and the weights in
    ``model.safetensors`` with the same key scheme. Writing ``kvae.yaml`` next
    to a ``kvae.safetensors`` symlink makes it a regular sidecar pair; a
    missing ``scaling_factor`` falls back to the t4s16 value.

    Returns:
        The sidecar prefix, or ``None`` when ``directory`` is not in that format.
    """
    config_file, weights_file = directory / HF_KVAE_CONFIG, directory / HF_KVAE_WEIGHTS
    if not (config_file.is_file() and weights_file.is_file()):
        return None
    config = json.loads(config_file.read_text())
    model = config.get("model", config)
    if "encoder_params" not in model or "decoder_params" not in model:
        msg = f"{config_file} has no model.encoder_params / model.decoder_params"
        raise ValueError(msg)
    scaling_factor = config.get("scaling_factor", model.get("scaling_factor"))
    if scaling_factor is None:
        scaling_factor = KVAE_T4S16_SCALING_FACTOR
        logger.warning(
            "{} carries no scaling_factor; using the KVAE-3D-2.0-t4s16 value {}", config_file, scaling_factor
        )
    architecture = KvaeArchitecture(
        scaling_factor=float(scaling_factor),
        encoder_params=model["encoder_params"],
        decoder_params=model["decoder_params"],
    )
    return write_kvae_sidecar(directory, weights_file.name, architecture)


def find_kvae_sidecar_prefix(directory: Path, label: str) -> str:
    """Return the single ``{prefix}`` of a ``{prefix}.yaml`` + weights pair under ``directory``.

    A public-KVAE checkout (``config.json`` + ``model.safetensors``, see
    :func:`sidecar_from_hf_kvae`) is converted into such a pair first.

    Raises:
        ValueError: If the directory holds no (or more than one) sidecar pair.
    """
    prefixes = sorted(
        str(yaml_file.with_suffix("")) for yaml_file in directory.rglob("*.yaml") if has_sidecar_weights(yaml_file)
    )
    if not prefixes:
        converted = sidecar_from_hf_kvae(directory)
        if converted is not None:
            return converted
    if len(prefixes) != 1:
        msg = f"Expected exactly one KVAE sidecar pair in {label}, found: {prefixes or 'none'}"
        raise ValueError(msg)
    return prefixes[0]


def find_lu_bank_yaml(directory: Path, label: str) -> str:
    """Return the single bank YAML (a YAML without same-stem weights) under ``directory``.

    Raises:
        ValueError: If the directory holds no (or more than one) bank YAML.
    """
    banks = sorted(str(yaml_file) for yaml_file in directory.rglob("*.yaml") if not has_sidecar_weights(yaml_file))
    if len(banks) != 1:
        msg = f"Expected exactly one LU bank YAML in {label}, found: {banks or 'none'}"
        raise ValueError(msg)
    return banks[0]


def resolve_kvae_reference(value: str) -> str:
    """Resolve a KVAE reference to its sidecar prefix.

    Accepts a Diffusers bundle (repo, directory or its ``vae`` component), the
    prefix itself (``.../kvae_20_v2``, passed through), a local directory
    holding the ``{prefix}.yaml`` + ``{prefix}.safetensors`` / ``.ckpt`` pair,
    or a Hub repo id whose snapshot holds it — or, in either of the last two
    forms, a public KVAE checkout (``config.json`` + ``model.safetensors``,
    e.g. ``kandinskylab/KVAE-3D-2.0-t4s16``).

    Raises:
        ValueError: If the directory holds no (or more than one) sidecar pair,
            or the reference names another component of a bundle.
    """
    component_dir = resolve_component_dir(value, VAE)
    if component_dir is not None:
        return kvae_sidecar_from_component(component_dir)
    if is_hf_repo_id(value):
        return find_kvae_sidecar_prefix(download_hf_repo(value), value)
    if Path(value).is_dir():
        return find_kvae_sidecar_prefix(Path(value), value)
    return value


def resolve_lu_bank_reference(value: str) -> str:
    """Resolve an LU bank reference to the bank YAML path or a bundle's bank component.

    Accepts a Diffusers bundle (repo, directory or its ``latent_upscaler``
    component — resolved to that component directory), the YAML path itself
    (passed through), a local directory holding the single bank YAML, or a Hub
    repo id whose snapshot holds it. Relative ``checkpoint:`` entries then
    resolve against the YAML's directory as usual.

    Raises:
        ValueError: If the directory holds no (or more than one) bank YAML, or
            the reference names another component of a bundle.
    """
    component_dir = resolve_component_dir(value, LATENT_UPSCALER)
    if component_dir is not None:
        return str(component_dir)
    if is_hf_repo_id(value):
        return find_lu_bank_yaml(download_hf_repo(value), value)
    if Path(value).is_dir():
        return find_lu_bank_yaml(Path(value), value)
    return value
