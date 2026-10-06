"""Read a ``Kandinsky6SRPipeline`` Diffusers bundle with the native loaders.

The released models are Diffusers bundles::

    model_index.json
    transformer/      config.json + diffusion_pytorch_model.safetensors   (the SR DiT)
    scheduler/        scheduler_config.json                               (π-Flow or flow matching)
    vae/              config.json + diffusion_pytorch_model.safetensors   (the KVAE)
    latent_upscaler/  config.json + diffusion_pytorch_model.safetensors   (x2 and x4, one file)

The weights are the native tensors under other file names; what differs is
where the configuration lives. This module rebuilds the three things the
native loaders consume — the training config that used to sit next to the DiT,
the KVAE sidecar pair, and the latent-upscaler bank conf — from a local bundle.
Downloading is :mod:`musubi_tuner.kandinsky6.runtime.sr.pipeline.hub`'s job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from musubi_tuner.kandinsky6.runtime.sr.pipeline.kvae_sidecar import KvaeArchitecture, write_kvae_sidecar

MODEL_INDEX = "model_index.json"
COMPONENT_CONFIG = "config.json"
COMPONENT_WEIGHTS = "diffusion_pytorch_model.safetensors"
SCHEDULER_CONFIG = "scheduler_config.json"

TRANSFORMER = "transformer"
VAE = "vae"
LATENT_UPSCALER = "latent_upscaler"
SCHEDULER = "scheduler"
COMPONENTS = (TRANSFORMER, VAE, LATENT_UPSCALER)

KVAE_NAME = "video-kvae"
PIFLOW_SCHEDULER_CLASS = "PiflowScheduler"
FLOW_MATCHING_SCHEDULER_CLASS = "FlowMatchEulerDiscreteScheduler"
# Keys of ``transformer/config.json`` that are not DiT constructor parameters.
SR_PARAMS_KEY = "sr_params"
ATTRIBUTE_OVERRIDES_KEY = "attribute_overrides"
# Flat KVAE fields of a published bundle: the encoder takes these under the same names...
KVAE_ENCODER_FIELDS = (
    "in_channels",
    "ch",
    "ch_mult",
    "num_res_blocks",
    "resolution",
    "z_channels",
    "double_z",
    "temporal_compress_times",
    "fix_pxs",
    "norm_type",
    "downsample_version",
    "temporal_compress_start_level",
    "padding_mode",
)
# ...and the decoder takes these, native parameter name -> flat field.
KVAE_DECODER_FIELDS = {
    "out_ch": "out_channels",
    "ch": "decoder_ch",
    "ch_mult": "decoder_ch_mult",
    "num_res_blocks": "num_res_blocks",
    "resolution": "resolution",
    "z_channels": "z_channels",
    "temporal_compress_times": "temporal_compress_times",
    "norm_type": "norm_type",
    "temporal_compress_start_level": "temporal_compress_start_level",
    "padding_mode": "padding_mode",
}
TRAINER_PARAM_NAMES = (
    "lq_noise_scale",
    "lq_noise_type",
    "lq_channel_noise_scale",
    "cap_noise_timestep",
    "scheduler_scale",
)


def is_bundle_dir(path: Path) -> bool:
    """Whether ``path`` is the root of a Diffusers bundle."""
    return (path / MODEL_INDEX).is_file()


def is_component_dir(path: Path) -> bool:
    """Whether ``path`` is one model component of a bundle (config + weights)."""
    return (path / COMPONENT_CONFIG).is_file() and (path / COMPONENT_WEIGHTS).is_file()


def read_component_config(component_dir: Path) -> dict[str, Any]:
    """Load a component's ``config.json`` without the Diffusers bookkeeping keys (``_class_name`` ...)."""
    config = json.loads((component_dir / COMPONENT_CONFIG).read_text())
    return {key: value for key, value in config.items() if not key.startswith("_")}


def int_keyed(mapping: dict[str, Any]) -> dict[int, Any]:
    """Undo JSON's stringification of resolution keys (``"512"`` -> ``512``)."""
    return {int(key): value for key, value in mapping.items()}


def piflow_section(scheduler_dir: Path) -> dict[str, Any] | None:
    """The ``trainer.piflow`` section a π-Flow scheduler config stands for; ``None`` for flow matching.

    The scheduler config is what tells a distilled DiT from a flow-matching
    one, so it is required: guessing would run a distilled model with the
    wrong sampler.

    Raises:
        FileNotFoundError: If the scheduler config is missing.
        ValueError: If the scheduler is neither π-Flow nor flow-matching Euler.
    """
    config_file = scheduler_dir / SCHEDULER_CONFIG
    if not config_file.is_file():
        msg = f"{config_file} is missing: the DiT's sampler settings live next to its transformer component"
        raise FileNotFoundError(msg)
    config = json.loads(config_file.read_text())
    scheduler_class = config.get("_class_name")
    if scheduler_class == FLOW_MATCHING_SCHEDULER_CLASS:
        return None
    if scheduler_class != PIFLOW_SCHEDULER_CLASS:
        msg = f"{config_file}: unsupported scheduler {scheduler_class!r} for the SR DiT"
        raise ValueError(msg)
    return {
        "nfe": int(config["nfe"]),
        "dx_num_grid_points": int(config["n_grid"]),
        "shift": float(config["shift"]),
        "num_policy_substeps": int(config["num_policy_substeps"]),
        "final_step_size_scale": float(config["final_step_size_scale"]),
        "eps": float(config["eps"]),
    }


def native_out_visual_dim(bundle_out_dim: int, piflow: dict[str, Any] | None, transformer_dir: Path) -> int:
    """Fold the bundle's ``n_grid``-wide π-Flow output head back to the DiT's own width."""
    if piflow is None:
        return bundle_out_dim
    n_grid = piflow["dx_num_grid_points"]
    if bundle_out_dim % n_grid:
        msg = f"{transformer_dir}: out_visual_dim={bundle_out_dim} is not a multiple of the π-Flow grid ({n_grid})"
        raise ValueError(msg)
    return bundle_out_dim // n_grid


def training_config_from_transformer(transformer_dir: Path) -> DictConfig:
    """Rebuild the training config the native loaders read next to a DiT checkpoint.

    The converter spreads it over ``transformer/config.json`` (DiT parameters,
    SR generation parameters, behavioural overrides) and the sibling
    ``scheduler/scheduler_config.json`` (the π-Flow sampler settings).

    Args:
        transformer_dir: The bundle's ``transformer`` component directory.

    Returns:
        A config with the native ``vae`` / ``dit`` / ``common`` / ``trainer`` sections.

    Raises:
        FileNotFoundError: If the sibling scheduler config is missing.
        ValueError: If the scheduler is unsupported, or a π-Flow output head is
            not a multiple of the grid size.
    """
    dit_params = read_component_config(transformer_dir)
    sr_params = dit_params.pop(SR_PARAMS_KEY)
    attribute_overrides = dit_params.pop(ATTRIBUTE_OVERRIDES_KEY, None) or {}
    piflow = piflow_section(transformer_dir.parent / SCHEDULER)
    dit_params["out_visual_dim"] = native_out_visual_dim(int(dit_params["out_visual_dim"]), piflow, transformer_dir)
    dit_params["attention_params"] = int_keyed(dit_params["attention_params"])
    trainer: dict[str, Any] = {"params": {name: sr_params[name] for name in TRAINER_PARAM_NAMES}}
    if piflow is not None:
        trainer["piflow"] = piflow
    return OmegaConf.create(
        {
            "vae": {"name": KVAE_NAME},
            "dit": {"params": dit_params, "attribute_overrides": attribute_overrides},
            "common": {
                "visual_size": sr_params["visual_size"],
                "fps": sr_params["fps"],
                "scale_factor": int_keyed(sr_params["scale_factor"]),
            },
            "trainer": trainer,
        }
    )


def flat_kvae_architecture(config: dict[str, Any], vae_dir: Path) -> KvaeArchitecture:
    """Read the flat KVAE fields of a published bundle into the encoder / decoder parameter sets.

    The fields shared by both halves are listed once; the decoder's width and
    multipliers carry a ``decoder_`` prefix and its output channels are
    ``out_channels``.

    Raises:
        ValueError: If a field is missing.
    """
    required = sorted({*KVAE_ENCODER_FIELDS, *KVAE_DECODER_FIELDS.values()})
    missing = [name for name in required if name not in config]
    if missing:
        msg = f"{vae_dir}: the KVAE config.json has no {', '.join(missing)}"
        raise ValueError(msg)
    return KvaeArchitecture(
        scaling_factor=float(config["scaling_factor"]),
        encoder_params={name: config[name] for name in KVAE_ENCODER_FIELDS},
        decoder_params={native: config[flat] for native, flat in KVAE_DECODER_FIELDS.items()},
    )


def kvae_architecture(vae_dir: Path) -> KvaeArchitecture:
    """Read the KVAE architecture of a bundle's ``vae`` component, in either spelling.

    The published bundles list the architecture as flat fields; the bundles
    written by this package's converter nest it under ``encoder_config`` /
    ``decoder_config`` in the native parameter names.

    Raises:
        ValueError: If the component is not a KVAE, or a flat field is missing.
    """
    config = read_component_config(vae_dir)
    if config.get("vae_type") != KVAE_NAME or "scaling_factor" not in config:
        msg = f"{vae_dir} is not a KVAE component: its config.json has no vae_type={KVAE_NAME!r} / scaling_factor"
        raise ValueError(msg)
    if "encoder_config" in config and "decoder_config" in config:
        return KvaeArchitecture(
            scaling_factor=float(config["scaling_factor"]),
            encoder_params=config["encoder_config"],
            decoder_params=config["decoder_config"],
        )
    return flat_kvae_architecture(config, vae_dir)


def kvae_sidecar_from_component(vae_dir: Path) -> str:
    """Make the bundle's ``vae`` component loadable as a KVAE sidecar pair.

    Args:
        vae_dir: The bundle's ``vae`` component directory.

    Returns:
        The sidecar prefix the VAE builders take.

    Raises:
        ValueError: If the component is not a KVAE.
    """
    return write_kvae_sidecar(vae_dir, COMPONENT_WEIGHTS, kvae_architecture(vae_dir))


# Published banks store the 2x upscaler at index 0 and the 4x upscaler at index 1.
_LU_STATE_INDEX = {"2x": "0", "4x": "1"}


def lu_state_prefix(target_scale: str) -> str:
    """Key prefix of one upscaler inside the bank's shared weights file.

    The Diffusers state dict uses ``ModuleList`` indices, so ``2x`` is
    ``_models.0.`` and ``4x`` is ``_models.1.``.
    """
    try:
        index = _LU_STATE_INDEX[target_scale]
    except KeyError:
        msg = f"latent upscaler target_scale must be one of {tuple(_LU_STATE_INDEX)}, got {target_scale!r}"
        raise ValueError(msg) from None
    return f"_models.{index}."


def lu_bank_conf_from_component(lu_dir: Path) -> DictConfig:
    """Build the latent-upscaler bank conf from the bundle's ``latent_upscaler`` component.

    Every upscaler of the bank lives in the one weights file, under its
    ``_models.{index}.`` prefix (``2x`` at 0, ``4x`` at 1); each entry points
    at that file and names its prefix.

    Args:
        lu_dir: The bundle's ``latent_upscaler`` component directory.

    Returns:
        A conf in the bank-YAML layout (``latent_upscaler.models``).

    Raises:
        ValueError: If the component is not a latent-upscaler bank.
    """
    config = read_component_config(lu_dir)
    if "models" not in config:
        msg = f"{lu_dir} is not a latent-upscaler bank component: its config.json has no models list"
        raise ValueError(msg)
    weights = str(lu_dir / COMPONENT_WEIGHTS)
    entries = [
        {
            "target_scale": item["target_scale"],
            "model": item["model"],
            "checkpoint": weights,
            "state_prefix": lu_state_prefix(item["target_scale"]),
        }
        for item in config["models"]
    ]
    return OmegaConf.create({"latent_upscaler": {"enabled": True, "models": entries}})
