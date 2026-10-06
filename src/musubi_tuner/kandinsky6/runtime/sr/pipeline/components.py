"""Load SR inference components (DiT, compiled KVAE, latent upscaler).

Resolves the configuration and weights from local filesystem paths: the
components of a ``Kandinsky6SRPipeline`` Diffusers bundle (the released
format, see :mod:`musubi_tuner.kandinsky6.runtime.sr.pipeline.diffusers_bundle`), or native
checkpoints — a training YAML next to the DiT weights (a merged file or a
sharded ``model/`` directory), a KVAE sidecar pair and a latent-upscaler bank
YAML. Single GPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from musubi_tuner.kandinsky6.runtime.sr import constants as ksr_constants
from musubi_tuner.kandinsky6.runtime.sr.core.algo.cached_text_embs_utils import load_cached_empty_text_embeds
from musubi_tuner.kandinsky6.runtime.sr.core.algo.checkpoint import (
    CheckpointConfig,
    gather_distributed_model_state_dict,
    join_path,
    load,
    path_exists,
    validate_local_path,
)
from musubi_tuner.kandinsky6.runtime.sr.core.algo.latent_upscaler import LatentUpscalerBank, load_latent_upscaler
from musubi_tuner.kandinsky6.runtime.sr.core.algo.piflow_sampler import build_piflow_sampler_params
from musubi_tuner.kandinsky6.runtime.sr.core.algo.safetensors_io import is_safetensors_path, load_safetensors_state_dict
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.compiled_kvae import build_compiled_kvae
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.compiled_kvae_v2_magi import build_magi_compiled_kvae
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.dit import get_dit
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.dx_dit import DXDiTWrapper
from musubi_tuner.kandinsky6.runtime.sr.pipeline.config import VaeBackend
from musubi_tuner.kandinsky6.runtime.sr.pipeline.diffusers_bundle import (
    COMPONENT_WEIGHTS,
    is_component_dir,
    lu_bank_conf_from_component,
    training_config_from_transformer,
)
from musubi_tuner.kandinsky6.runtime.sr.pipeline.hub import (
    bundle_reference,
    resolve_checkpoint_reference,
    resolve_kvae_reference,
    resolve_lu_bank_reference,
)

_MISSING = object()


class SRParams(BaseModel):
    """SR generation parameters extracted from the training config."""

    lq_channel_noise_scale: float = 0.0
    lq_noise_scale: float = 0.7
    lq_noise_type: str = "ddpm"
    cap_noise_timestep: bool = False
    scheduler_scale: float = 5.0
    scale_factor: dict[int, list[float]]
    visual_size: list[int]
    fps: int = 24


class SRComponents(BaseModel):
    """Container for all SR pipeline components."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dit: Any
    vae: Any
    latent_upscaler: Any
    cached_text_embeds: dict[str, Any] | None
    sr_params: SRParams


def resolve_cuda_device(device: str) -> torch.device:
    """Return ``device`` with an explicit index: a bare ``"cuda"`` means the current device."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def scale_factor_for(components: Any) -> tuple[float, ...]:
    """Return the RoPE scale_factor tuple for the model's visual size.

    Args:
        components: Anything exposing ``sr_params`` (loaded components or the
            built pipeline).
    """
    visual_size = components.sr_params.visual_size[0]
    return tuple(components.sr_params.scale_factor[visual_size])


def resolve_kvae_path(vae_path: str) -> tuple[str, None]:
    """Return the local KVAE checkpoint prefix without copying or downloading it."""
    validate_local_path(vae_path, "KVAE path")
    return vae_path, None


def resolve_vae_source(vae_path: str | None) -> str:
    """Validate the KVAE sidecar prefix to load the VAE from.

    The path must be provided explicitly (YAML ``sr.vae_path`` or CLI
    ``--vae-path``); the ``vae.checkpoint_path`` baked into the checkpoint's
    config is ignored.

    Args:
        vae_path: The configured VAE sidecar prefix.

    Returns:
        Sidecar prefix to load the VAE from.

    Raises:
        ValueError: If no path was configured.
    """
    if not vae_path:
        msg = (
            "vae_path is not set and the checkpoint is not a Diffusers bundle that carries the VAE — "
            "set sr.vae_path in the config YAML or pass --vae-path"
        )
        raise ValueError(msg)
    return vae_path


def build_vae_for_backend(vae_conf: Any, vae_name: str, vae_backend: VaeBackend) -> torch.nn.Module:
    """Build the video-kvae VAE with the requested compilation backend.

    Args:
        vae_conf: The training config's ``vae`` section (``name`` +
            ``checkpoint_path`` already resolved to a local prefix).
        vae_name: ``vae_conf.name`` (passed separately for logging/dispatch).
        vae_backend: ``"torch"`` — per-block/leaf ``torch.compile``
            (``compiled_kvae.py``); ``"magi"`` — MagiCompiler static graphs
            per temporal segment (``compiled_kvae_v2_magi.py``).

    Returns:
        The built VAE (eval/bfloat16 handled by the builders or the caller).

    Raises:
        ValueError: For an unknown backend or an unknown VAE name.
    """
    if vae_name != "video-kvae":
        msg = f"Unsupported vae.name {vae_name!r}; only 'video-kvae' is supported."
        raise ValueError(msg)

    if vae_backend == "magi":
        return build_magi_compiled_kvae(vae_conf)
    if vae_backend != "torch":
        msg = f"Unknown vae_backend {vae_backend!r}; expected 'torch' or 'magi'."
        raise ValueError(msg)
    return build_compiled_kvae(vae_conf)


def _find_local_training_config(search_dir: Path) -> Path | None:
    """Locate a training config YAML in ``search_dir`` (then ``config.yaml``).

    Returns:
        The YAML path, or ``None`` if none is found.
    """
    if not search_dir.is_dir():
        return None
    yaml_files = sorted(search_dir.glob("*.yaml"))
    if yaml_files:
        return yaml_files[0]
    fallback = search_dir / "config.yaml"
    return fallback if fallback.exists() else None


def _config_search_locations(checkpoint_path: str) -> list[str]:
    """Ordered local candidate dirs that may hold the config.

    Searched most-specific first: the directory the checkpoint points at (e.g.
    the ``model/`` shard dir), the step dir, then the experiment dir. This finds
    the YAML whether it sits inside the step dir, next to its shards, or one
    level up — no assumption about a single fixed location.
    """
    raw = checkpoint_path.rstrip("/")
    # Directory the path refers to: parent of a .pt file, else the dir itself.
    given = raw.rsplit("/", 1)[0] if raw.endswith(".pt") else raw
    # Step dir = given dir without a trailing ``model`` shard segment.
    step = given[: -len("/model")] if given.endswith("/model") else given
    experiment = step.rsplit("/", 1)[0]
    out: list[str] = []
    for cand in (given, step, experiment):
        if cand and cand not in out:
            out.append(cand)
    return out


def _resolve_config(checkpoint_path: str) -> Any:
    """Load the merged training config (OmegaConf) for a checkpoint.

    A Diffusers bundle's ``transformer`` component carries it as JSON (rebuilt
    by :func:`training_config_from_transformer`); a native checkpoint has a
    YAML in or above its directory.

    Raises:
        FileNotFoundError: If no YAML is found in any candidate location.
    """
    validate_local_path(checkpoint_path, "checkpoint path")
    if is_component_dir(Path(checkpoint_path)):
        logger.info("Loading training config from the Diffusers transformer component {}", checkpoint_path)
        return training_config_from_transformer(Path(checkpoint_path))
    locations = _config_search_locations(checkpoint_path)
    for location in locations:
        found = _find_local_training_config(Path(location))
        yaml_local = str(found) if found is not None else None
        if yaml_local is not None:
            logger.info("Loading training config from {} (matched {})", yaml_local, location)
            return OmegaConf.load(yaml_local)
    searched = ", ".join(locations)
    msg = f"No training config YAML found. Searched: {searched}"
    raise FileNotFoundError(msg)


def resolve_dit_state_dict(
    checkpoint_path: str,
    ckpt_config: CheckpointConfig,
    device: str,
) -> dict[str, torch.Tensor]:
    """Load the DiT state dict, handling single-file and sharded layouts.

    Accepts five local ``checkpoint_path`` shapes:

    * a Diffusers ``transformer`` component dir → its
      ``diffusion_pytorch_model.safetensors`` (the open-source release format);
    * a ``.safetensors`` file → loaded directly;
    * a ``.pt`` file → loaded directly as a merged checkpoint;
    * a step dir (``.../step_N``) → ``<step>/model.safetensors`` is used when
      present, then ``<step>/model.pt``, else the sharded ``<step>/model/``
      directory is merged online;
    * the shard dir itself (``.../step_N/model`` or ``.../step_N/model/``) →
      merged online directly (no extra ``model`` segment appended).

    Args:
        checkpoint_path: ``.safetensors`` / ``.pt`` file, step dir, or
            ``model/`` shard dir.
        ckpt_config: Checkpoint config carrying the shard loading thread count.
        device: Map-location device for loaded tensors.

    Returns:
        The merged DiT state dict.
    """
    base = checkpoint_path.rstrip("/")
    points_at_shard_dir = base.endswith("/model")

    if is_component_dir(Path(base)):
        base = join_path(base, COMPONENT_WEIGHTS)
    if is_safetensors_path(base):
        logger.info("Loading safetensors checkpoint {}", base)
        return load_safetensors_state_dict(base, map_location=device)

    if base.endswith(".pt"):
        single: str | None = base
    elif points_at_shard_dir:
        single = None  # already the shard dir; no merged single file to probe
    else:
        merged_safetensors = join_path(base, "model.safetensors")
        if path_exists(merged_safetensors):
            logger.info("Loading merged safetensors checkpoint {}", merged_safetensors)
            return load_safetensors_state_dict(merged_safetensors, map_location=device)
        single = join_path(base, "model.pt")
    if single is not None and path_exists(single):
        logger.info("Loading merged checkpoint file {}", single)
        return load(single, map_location=device, weights_only=False)

    model_dir = base if points_at_shard_dir else join_path(base, "model")
    logger.info("Merging sharded checkpoint from {}", model_dir)
    return gather_distributed_model_state_dict(model_dir, ckpt_config, map_location=device)


def _require_sr_param(trainer_params: Any, name: str) -> Any:
    """Read a required SR generation param, failing loudly when omitted."""
    value = getattr(trainer_params, name, _MISSING)
    if value is _MISSING:
        msg = f"trainer config has no SR generation param '{name}' (no silent default is applied)"
        raise ValueError(msg)
    return value


def _extract_sr_params(conf: Any) -> SRParams:
    """Extract SR generation parameters from the merged training config."""
    trainer_params = conf.trainer.params
    scale_factor = {int(k): list(v) for k, v in dict(conf.common.scale_factor).items()}
    return SRParams(
        lq_channel_noise_scale=_require_sr_param(trainer_params, "lq_channel_noise_scale"),
        lq_noise_scale=_require_sr_param(trainer_params, "lq_noise_scale"),
        lq_noise_type=getattr(trainer_params, "lq_noise_type", "ddpm"),
        cap_noise_timestep=getattr(trainer_params, "cap_noise_timestep", False),
        scheduler_scale=_require_sr_param(trainer_params, "scheduler_scale"),
        scale_factor=scale_factor,
        visual_size=list(conf.common.visual_size),
        fps=getattr(conf.common, "fps", 24),
    )


def _load_cached_embeds(conf: Any) -> dict[str, torch.Tensor] | None:
    """Load cached empty-caption text embeddings, or ``None`` for text-free models."""
    cached = load_cached_empty_text_embeds(conf, local_rank=0)
    if cached is not None:
        return cached
    metrics_conf = getattr(conf, "metrics", None)
    if metrics_conf is not None:
        return load_cached_empty_text_embeds(metrics_conf, local_rank=0)
    return None


def _apply_model_path_overrides(conf: Any, vae_path: str) -> None:
    """Override the training-baked VAE path with the resolved one.

    The training config YAML next to the DiT checkpoint stores an absolute
    training-machine VAE path that rarely exists at inference time, so the
    resolved CLI/default path wins here. The training config's
    ``latent_upscaler`` section is ignored entirely — LU architectures come
    from the bank YAML (see ``_load_lu_bank_conf``).

    Args:
        conf: Loaded OmegaConf training config (mutated in place).
        vae_path: kvae sidecar prefix to load instead of
            ``conf.vae.checkpoint_path``.
    """
    OmegaConf.set_struct(conf, value=False)
    conf.vae.checkpoint_path = vae_path


def resolve_lu_bank_checkpoints(conf: Any, bank_dir: Path) -> None:
    """Resolve relative LU ``checkpoint:`` entries against the bank YAML's directory.

    A release bank ships its ``.safetensors`` files next to the YAML with
    plain-filename ``checkpoint:`` entries, so they must load no matter the
    process's working directory. Absolute paths are left untouched. Symlinks
    are NOT dereferenced: in a Hugging Face snapshot every file is a symlink
    into ``blobs/``, so following the YAML's link would resolve siblings
    against the wrong directory.

    Args:
        conf: Loaded bank conf (mutated in place).
        bank_dir: Directory the bank YAML was loaded from.
    """
    models = getattr(getattr(conf, "latent_upscaler", None), "models", None)
    if models is None:
        return
    for entry in models:
        checkpoint = str(entry.checkpoint)
        if Path(checkpoint).is_absolute():
            continue
        entry.checkpoint = str(bank_dir / checkpoint)


def _load_lu_bank_conf(config_path: str | None) -> Any | None:
    """Load the LU bank conf, or return ``None`` when the LU is disabled.

    The bank — a Diffusers bundle's ``latent_upscaler`` component or a bank
    YAML — is the single source of truth for the LU architectures, checkpoints,
    and ``use_ema``. ``None`` or the literal string ``"none"`` disables the
    latent upscaler entirely — every run then takes the pixel path. Relative
    ``checkpoint:`` entries of a YAML resolve against its own directory (see
    :func:`resolve_lu_bank_checkpoints`).

    Args:
        config_path: The bundle component directory, the bank YAML path,
            ``"none"``, or ``None``.

    Returns:
        OmegaConf wrapper ready for ``load_latent_upscaler``, or ``None``.
    """
    if config_path is None or config_path.lower() == "none":
        return None
    if is_component_dir(Path(config_path)):
        return lu_bank_conf_from_component(Path(config_path))
    conf = OmegaConf.load(config_path)
    resolve_lu_bank_checkpoints(conf, Path(config_path).absolute().parent)
    return conf


def split_lu_bank_by_scales(conf: Any, eager_scales: tuple[str, ...] | None) -> tuple[Any | None, dict[str, Any]]:
    """Split a bank conf into the entries to load NOW and per-scale lazy specs.

    ``eager_scales`` is the configured ``lu_load_scales``: entries whose ``target_scale``
    is listed load at startup; every other entry becomes a single-entry conf
    that :class:`LazyLatentUpscalerBank` materializes on the first run at that
    scale. An x2-only host thus never spends memory on the 1.45B-param x4
    cascade, yet an unexpected x4 run still works — it just pays the load on
    first use instead of at startup.

    Args:
        conf: Loaded bank conf (``_load_lu_bank_conf`` result, not ``None``).
        eager_scales: Scales to load at startup; ``None`` = all (nothing lazy).

    Returns:
        ``(eager_conf, lazy_specs)`` — ``eager_conf`` is ``None`` when no entry
        is eager; ``lazy_specs`` maps ``"4x"``-style keys to single-entry confs.
    """
    if eager_scales is None:
        return conf, {}
    eager_entries = []
    lazy_specs: dict[str, Any] = {}
    for entry in conf.latent_upscaler.models:
        scale_key = str(entry.get("target_scale", "4x"))
        if scale_key in eager_scales:
            eager_entries.append(entry)
        else:
            lazy_specs[scale_key] = OmegaConf.create({"latent_upscaler": {"enabled": True, "models": [entry]}})
    if lazy_specs:
        logger.info(
            "lu_load_scales={}: bank entries {} stay cold (lazy-loaded on first use)",
            eager_scales,
            sorted(lazy_specs),
        )
    if not eager_entries:
        return None, lazy_specs
    eager_conf = OmegaConf.create({"latent_upscaler": {"enabled": True, "models": eager_entries}})
    return eager_conf, lazy_specs


class LazyLatentUpscalerBank(LatentUpscalerBank):
    """A bank whose cold entries load on the first ``for_scale`` request.

    Startup loads only the configured eager entries; the rest are kept as
    single-entry confs and materialized through the same
    ``load_latent_upscaler`` path (frozen, bf16, EMA selection) the eager
    entries took — so a lazily loaded LU is byte-identical to an eager one.
    ``scales`` reports hot AND cold entries: every listed scale is servable,
    a cold one just pays its load on first use.
    """

    def __init__(
        self,
        lazy_specs: dict[str, Any],
        device: str,
        vae_scaling_factor: float,
    ) -> None:
        """Store the cold-entry specs and everything needed to load them later."""
        super().__init__()
        self.lazy_specs = dict(lazy_specs)
        self.lazy_device = device
        self.lazy_vae_scaling_factor = vae_scaling_factor

    def for_scale(self, scale: int) -> Any | None:
        """Return the LU for ``scale``, materializing a cold entry on first use."""
        module = super().for_scale(scale)
        if module is not None:
            return module
        key = f"{int(scale)}x"
        spec = self.lazy_specs.pop(key, None)
        if spec is None:
            return None
        logger.info("Lazy-loading the {} latent upscaler (kept cold at startup by lu_load_scales)...", key)
        load_t0 = time.perf_counter()
        loaded = load_latent_upscaler(
            spec,
            device=self.lazy_device,
            vae_scaling_factor=self.lazy_vae_scaling_factor,
        )
        module = loaded.for_scale(int(scale)) if isinstance(loaded, LatentUpscalerBank) else loaded
        if module is None:
            msg = f"Lazy spec for {key} produced no {key} latent upscaler"
            raise ValueError(msg)
        self[key] = module
        logger.info("Lazy-loaded the {} latent upscaler in {:.1f}s", key, time.perf_counter() - load_t0)
        return module

    @property
    def scales(self) -> tuple[int, ...]:
        """All servable upscale factors — loaded entries plus cold (lazy) ones."""
        keys = set(self._modules) | set(self.lazy_specs)
        return tuple(sorted(int(key.removesuffix("x")) for key in keys))


def _apply_instruct_type_override(dit: Any, override: str | None) -> None:
    """Override the loaded DiT's ``instruct_type`` attribute in place.

    Mirrors the reference ``infer_sr_tiling.py --instruct-type``: only the
    behavioural attribute changes — the trained architecture (channel layout)
    stays put — so ``override`` must be channel-compatible with it. ``None`` is a
    no-op (keep the config's value).

    Args:
        dit: The loaded DiT model (mutated in place).
        override: New ``instruct_type`` (e.g. ``"noise"``), or ``None`` to skip.
    """
    current = getattr(dit, "instruct_type", None)
    if override is None or override == current:
        return
    logger.info("Overriding dit.instruct_type: {!r} -> {!r} (architecture stays as trained)", current, override)
    dit.instruct_type = override


def resolve_model_references(
    checkpoint_path: str, vae_path: str | None, latent_upscaler_config: str | None
) -> tuple[str, str | None, str | None]:
    """Turn the three model references into local paths the loaders take.

    Hub references are downloaded, bundle references resolve to their
    components, plain local paths pass through. A Diffusers bundle given as
    the checkpoint also supplies the VAE and the latent upscalers when those
    are unset; ``"none"`` keeps the upscalers disabled.

    Raises:
        ValueError: If the upscaler setting is unset and the checkpoint is not a bundle.
    """
    resolved_checkpoint = resolve_checkpoint_reference(checkpoint_path)
    bundle = bundle_reference(checkpoint_path, resolved_checkpoint)
    vae_path = vae_path or bundle
    latent_upscaler_config = latent_upscaler_config or bundle
    if latent_upscaler_config is None:
        msg = (
            "latent_upscaler_config is not set and the checkpoint is not a Diffusers bundle that carries the "
            "upscalers — set sr.latent_upscaler_config / pass --latent-upscaler-config (or 'none' to disable them)"
        )
        raise ValueError(msg)
    if vae_path:
        vae_path = resolve_kvae_reference(vae_path)
    if latent_upscaler_config.lower() != "none":
        latent_upscaler_config = resolve_lu_bank_reference(latent_upscaler_config)
    return resolved_checkpoint, vae_path, latent_upscaler_config


def load_training_config(checkpoint_path: str, vae_backend: VaeBackend) -> tuple[Any, str]:
    """Load the training config next to the DiT checkpoint and pin the VAE compression factors.

    The config decides which VAE the model runs with; its compression factors
    feed every latent<->pixel size computation, so they are set process-wide
    before any geometry math.

    Returns:
        ``(config, vae_name)``.

    Raises:
        ValueError: If the config's VAE is not ``video-kvae``.
    """
    conf = _resolve_config(checkpoint_path)
    vae_name = str(getattr(getattr(conf, "vae", None), "name", None) or "")
    if vae_name != "video-kvae":
        raise ValueError(f"Kandinsky SR supports only 'video-kvae', got: {vae_name!r}")
    ksr_constants.set_vae_factors(vae_name)
    logger.info(
        "VAE from model config: '{}' (spatial {}x, temporal {}x, backend {})",
        vae_name,
        ksr_constants.VAE_SPATIAL_FACTOR,
        ksr_constants.VAE_TEMPORAL_FACTOR,
        vae_backend,
    )
    return conf, vae_name


def build_dit(conf: Any, device: str) -> torch.nn.Module:
    """Instantiate the DiT architecture the training config describes (weights not loaded yet).

    π-Flow DX checkpoints are self-describing via ``trainer.piflow``. With a
    multi-grid head the DiT must be built as ``DXDiTWrapper`` — its output head
    is ``n_grid``x wider than a plain DiT's, so the strict weight load would
    fail on the head shape otherwise.
    """
    piflow_conf = getattr(getattr(conf, "trainer", None), "piflow", None)
    piflow_n_grid = int(piflow_conf.dx_num_grid_points) if piflow_conf is not None else 0
    if piflow_n_grid > 1:
        dit = DXDiTWrapper(
            conf.dit.params,
            out_visual_dim=int(conf.dit.params.out_visual_dim),
            n_grid=piflow_n_grid,
        )
        return dit.to(device).eval()
    return get_dit(conf.dit.params).to(device).eval()


def attach_piflow_sampler(dit: torch.nn.Module, conf: Any) -> None:
    """Route a π-Flow checkpoint through the DX few-step sampler at its trained nfe.

    ``dit.piflow_params`` makes ``generate_sample_sr`` ignore ``num_steps``
    for such checkpoints.
    """
    piflow_conf = getattr(getattr(conf, "trainer", None), "piflow", None)
    if piflow_conf is None:
        return
    dit.piflow_params = build_piflow_sampler_params(piflow_conf, n_grid=int(piflow_conf.dx_num_grid_points))
    logger.info(
        "π-Flow checkpoint: DX sampler at trained nfe={} (n_grid={}, shift={}); --num-steps is ignored",
        dit.piflow_params["nfe"],
        dit.piflow_params["n_grid"],
        dit.piflow_params["shift"],
    )


def merged_dit_overrides(conf: Any, dit_overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Combine the overrides a Diffusers bundle stores with the explicitly configured ones (these win)."""
    stored = getattr(conf.dit, "attribute_overrides", None)
    bundled = OmegaConf.to_container(stored, resolve=True) if stored is not None else {}
    return {**bundled, **(dit_overrides or {})}


def apply_dit_overrides(dit: torch.nn.Module, dit_overrides: dict[str, Any] | None) -> None:
    """Apply behavioural ``setattr`` overrides to the loaded DiT (the trained architecture stays put)."""
    for key, value in (dit_overrides or {}).items():
        if getattr(dit, key, _MISSING) == value:
            logger.info("dit.{} is already {!r} — override is a no-op", key, value)
            continue
        logger.warning("Overriding dit.{} = {!r}", key, value)
        setattr(dit, key, value)


def load_dit(
    conf: Any,
    checkpoint_path: str,
    device: str,
    *,
    dit_overrides: dict[str, Any] | None = None,
    instruct_type_override: str | None = "noise",
) -> torch.nn.Module:
    """Build the DiT, load its weights strictly, and apply the post-load overrides.

    Args:
        conf: The training config (``load_training_config``).
        checkpoint_path: Local Diffusers ``transformer`` component, DiT step
            dir or ``.safetensors`` / ``.pt`` file; a sharded ``model/`` dir is
            merged online.
        device: Target CUDA device.
        dit_overrides: Post-load attribute overrides applied via ``setattr``
            (e.g. ``{"visual_cond": True}``), on top of the ones a bundle stores.
        instruct_type_override: Post-load ``instruct_type`` override; ``None``
            keeps the checkpoint's value.
    """
    logger.info("Building DiT...")
    dit = build_dit(conf, device)
    started = time.perf_counter()
    state_dict = resolve_dit_state_dict(checkpoint_path, CheckpointConfig(), device)
    dit.load_state_dict(state_dict, strict=True)
    logger.info("DiT loaded ({} keys) in {:.1f}s", len(state_dict), time.perf_counter() - started)
    del state_dict
    _apply_instruct_type_override(dit, instruct_type_override)
    apply_dit_overrides(dit, merged_dit_overrides(conf, dit_overrides))
    attach_piflow_sampler(dit, conf)
    return dit


def load_kvae(conf: Any, vae_name: str, vae_path: str | None, vae_backend: VaeBackend, device: str) -> torch.nn.Module:
    """Build the compiled video KVAE from ``vae_path`` (sidecar prefix or local checkout) in bfloat16.

    The training config's ``vae`` section is pointed at the local weights
    first, because the VAE builders read their checkpoint path from it.
    """
    resolved_vae_path = resolve_vae_source(vae_path)
    local_vae_path, _ = resolve_kvae_path(resolved_vae_path)
    _apply_model_path_overrides(conf, local_vae_path)
    logger.info("Building compiled VAE ({}, backend={}) from {}...", vae_name, vae_backend, resolved_vae_path)
    return build_vae_for_backend(conf.vae, vae_name, vae_backend).eval().to(device, dtype=torch.bfloat16)


def load_latent_upscaler_bank(
    latent_upscaler_config: str | None,
    lu_load_scales: tuple[str, ...] | None,
    device: str,
    vae_scaling_factor: float,
) -> torch.nn.Module | Any | None:
    """Load the latent-upscaler bank: eager entries now, the rest lazily on first use.

    Args:
        latent_upscaler_config: A bundle's ``latent_upscaler`` component or
            an LU bank YAML — the single source of truth for the LU
            architectures, checkpoints, and ``use_ema``. ``None`` / ``"none"``
            disables the LU: every run then takes the pixel path.
        lu_load_scales: Bank entries (by ``target_scale``) loaded now; ``None``
            = every entry.
        device: Target CUDA device.
        vae_scaling_factor: The KVAE latent scaling the upscalers expect.

    Returns:
        A bank (eager and/or lazy), a single upscaler, or ``None`` when disabled.
    """
    lu_bank_conf = _load_lu_bank_conf(latent_upscaler_config)
    if lu_bank_conf is None:
        logger.info("Latent upscaler disabled — every run takes the pixel path")
        return None
    lu_conf, lu_lazy_specs = split_lu_bank_by_scales(lu_bank_conf, lu_load_scales)
    latent_upscaler = (
        load_latent_upscaler(lu_conf, device=device, vae_scaling_factor=vae_scaling_factor)
        if lu_conf is not None
        else None
    )
    if lu_lazy_specs:
        # Wrap eager modules and cold specs into one bank: cold scales load on
        # their first for_scale request through the same load_latent_upscaler
        # path, so a lazily loaded LU is identical to an eager one.
        lazy_bank = LazyLatentUpscalerBank(lu_lazy_specs, device, vae_scaling_factor)
        if isinstance(latent_upscaler, LatentUpscalerBank):
            for key, module in latent_upscaler.items():
                lazy_bank[key] = module
        elif latent_upscaler is not None:
            lazy_bank[str(getattr(latent_upscaler, "target_scale", "4x"))] = latent_upscaler
        latent_upscaler = lazy_bank
    log_latent_upscaler(latent_upscaler)
    return latent_upscaler


def log_latent_upscaler(latent_upscaler: Any) -> None:
    """Log what the latent-upscaler bank holds after loading."""
    if latent_upscaler is None:
        logger.info("Latent upscaler disabled — every run takes the pixel path")
    elif isinstance(latent_upscaler, LazyLatentUpscalerBank):
        logger.info(
            "Latent upscaler bank ready (loaded={}, lazy={})",
            sorted(key for key, _ in latent_upscaler.items()),
            sorted(latent_upscaler.lazy_specs),
        )
    elif isinstance(latent_upscaler, LatentUpscalerBank):
        logger.info("Latent upscaler bank loaded (scales={})", latent_upscaler.scales)
    else:
        logger.info("Latent upscaler loaded (target_scale={})", getattr(latent_upscaler, "target_scale", "?"))


def load_text_embeds(dit: torch.nn.Module, conf: Any) -> dict[str, torch.Tensor] | None:
    """Load the cached empty-caption embeddings for a text-conditioned DiT; ``None`` for a text-free one."""
    if not dit.use_text:
        # Text-free DiT: no text encoder and no cached empty-caption embeddings
        # are loaded or used; generate_sample_sr builds placeholder embeds.
        logger.info("Text-free DiT (use_text=False): skipping text model and cached text embeds")
        return None
    cached_text_embeds = _load_cached_embeds(conf)
    logger.info("Cached text embeds: {}", "loaded" if cached_text_embeds is not None else "none")
    return cached_text_embeds


def load_sr_components(  # noqa: PLR0913
    checkpoint_path: str,
    vae_path: str | None,
    device: str,
    latent_upscaler_config: str | None,
    *,
    vae_backend: VaeBackend = "torch",
    dit_overrides: dict[str, Any] | None = None,
    instruct_type_override: str | None = "noise",
    lu_load_scales: tuple[str, ...] | None = None,
) -> SRComponents:
    """Load DiT, compiled VAE, and frozen latent upscaler onto a single GPU.

    The training config of the checkpoint carries ``vae.name``
    (``"video-kvae"``), and the VAE class and compression factors follow from
    it. The three models are loaded by :func:`load_dit`, :func:`load_kvae` and
    :func:`load_latent_upscaler_bank`.

    Args:
        checkpoint_path: A ``Kandinsky6SRPipeline`` Diffusers bundle (Hugging
            Face repo id or local directory), or a native DiT: step dir,
            ``.safetensors`` / ``.pt`` file, or repo id.
        vae_path: A bundle or its ``vae`` component, a video-kvae sidecar
            prefix, local checkout, or repo id. ``None`` takes the VAE of the
            checkpoint's bundle (required for a native checkpoint).
        device: Target CUDA device (e.g. ``cuda:0``).
        latent_upscaler_config: A bundle or its ``latent_upscaler`` component,
            an LU bank YAML (or its dir / repo id); ``"none"`` disables the
            LU, ``None`` takes the upscalers of the checkpoint's bundle.
        vae_backend: VAE compilation backend, ``"torch"`` or ``"magi"``.
        dit_overrides: Post-load DiT attribute overrides (see :func:`load_dit`).
        instruct_type_override: Post-load ``instruct_type`` override.
        lu_load_scales: Bank entries loaded eagerly; ``None`` = all.

    Returns:
        Fully loaded :class:`SRComponents`.
    """
    # Pin the process's current CUDA device so helper tensors created on the
    # default ``"cuda"`` (e.g. the nabla mask in nablaT_v2_doc) land on the
    # requested GPU instead of cuda:0 — otherwise a non-zero --device mismatches.
    if device.startswith("cuda"):
        torch.cuda.set_device(resolve_cuda_device(device))

    checkpoint_path, vae_path, latent_upscaler_config = resolve_model_references(
        checkpoint_path, vae_path, latent_upscaler_config
    )
    conf, vae_name = load_training_config(checkpoint_path, vae_backend)
    logger.info("Paths -> DiT: {} | VAE ({}): {} | LU: {}", checkpoint_path, vae_name, vae_path, latent_upscaler_config)
    sr_params = _extract_sr_params(conf)
    logger.info("SR params: {}", sr_params.model_dump())

    dit = load_dit(
        conf,
        checkpoint_path,
        device,
        dit_overrides=dit_overrides,
        instruct_type_override=instruct_type_override,
    )
    vae = load_kvae(conf, vae_name, vae_path, vae_backend, device)
    latent_upscaler = load_latent_upscaler_bank(
        latent_upscaler_config, lu_load_scales, device, vae.config.scaling_factor
    )
    return SRComponents(
        dit=dit,
        vae=vae,
        latent_upscaler=latent_upscaler,
        cached_text_embeds=load_text_embeds(dit, conf),
        sr_params=sr_params,
    )
