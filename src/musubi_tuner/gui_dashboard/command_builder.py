"""Convert ProjectConfig into CLI argument lists for subprocess launch."""

from __future__ import annotations

import hashlib
import copy
import shlex
import sys
import tempfile
from pathlib import Path

from musubi_tuner.gui_dashboard.cli_defaults import (
    get_ltx2_training_network_module_default,
    get_ltx2_training_output_dir_default,
)
from musubi_tuner.model_defaults import (
    DEFAULT_GEMMA_ROOT_NAME,
    DEFAULT_LTX2_CHECKPOINT_NAME,
    DEFAULT_MODEL_DIR_NAME,
)
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.toml_export import (
    _write_slider_toml,
    build_slider_toml_path,
    conditioning_recipe_is_active,
    export_conditioning_toml,
    export_dataset_toml,
)


def _conditioning_recipe_for_mode(recipe, mode: str):
    """Return a launch-effective recipe without mutating the stored GUI recipe.

    The GUI keeps hidden modality settings so switching modes is reversible. The launcher must still
    ignore blocks that the current mode cannot apply.
    """
    if recipe is None:
        return None
    effective = copy.deepcopy(recipe)
    if mode == "video":
        effective.audio.is_generated = True
        effective.audio.conditions = []
    elif mode == "audio":
        effective.video.is_generated = True
        effective.video.conditions = []
    return effective


def _find_script(name: str) -> str:
    """Find a script in the musubi_tuner package."""
    import musubi_tuner

    pkg_dir = Path(musubi_tuner.__file__).parent
    script = pkg_dir / name
    if script.exists():
        return str(script)
    raise FileNotFoundError(f"Script not found: {name}")


def _default_model_dir(config: ProjectConfig) -> Path:
    return Path(config.model_dir) if config.model_dir else Path.cwd() / DEFAULT_MODEL_DIR_NAME


def _effective_ltx2_checkpoint(config: ProjectConfig, explicit: str) -> str:
    return explicit or config.default_ltx2_checkpoint or str(_default_model_dir(config) / DEFAULT_LTX2_CHECKPOINT_NAME)


def _effective_gemma_root(config: ProjectConfig, explicit: str, gemma_safetensors: str) -> str:
    if gemma_safetensors:
        return ""
    return explicit or config.default_gemma_root or str(_default_model_dir(config) / DEFAULT_GEMMA_ROOT_NAME)


def _effective_gemma_safetensors(config: ProjectConfig, explicit: str, explicit_gemma_root: str = "") -> str:
    # If the user has explicitly set gemma_root on this section AND it points to
    # a real directory, suppress the default_gemma_safetensors fallback so that
    # gemma_root takes priority.
    if explicit_gemma_root and Path(explicit_gemma_root).is_dir():
        return explicit or ""
    return explicit or config.default_gemma_safetensors or ""


def _effective_output_dir(explicit: str) -> str:
    return explicit or get_ltx2_training_output_dir_default()


def _default_dataset_cache_directory(config: ProjectConfig) -> str:
    datasets = list(config.dataset.datasets or []) + list(config.dataset.validation_datasets or [])
    for entry in datasets:
        if getattr(entry, "cache_directory", ""):
            return str(entry.cache_directory)
        if getattr(entry, "directory", ""):
            return str(Path(entry.directory) / "cache")
        if getattr(entry, "jsonl_file", ""):
            return str(Path(entry.jsonl_file).parent / "cache")
    return ""


def _effective_cache_preview_input(config: ProjectConfig) -> str:
    return config.caching.cache_preview_input or _default_dataset_cache_directory(config)


def _effective_cache_preview_output(config: ProjectConfig) -> str:
    if config.caching.cache_preview_output:
        return config.caching.cache_preview_output
    if config.project_dir:
        return str(Path(config.project_dir) / "cache_preview")
    return "cache_preview"


def _effective_network_module(explicit: str) -> str:
    return explicit or get_ltx2_training_network_module_default()


def _append_weight_noise_args(cmd: list[str], config_section) -> None:
    mode = str(getattr(config_section, "weight_noise_mode", "none") or "none").lower()
    if mode == "none":
        return
    cmd += ["--weight_noise_mode", str(mode)]
    cmd += ["--weight_noise_scale", str(getattr(config_section, "weight_noise_scale", 0.01))]


def _append_ltx2_performance_args(cmd: list[str], config_section) -> None:
    flags = (
        "ltx2_fused_norm_rope",
        "ltx2_fused_norm_rope_backward",
        "ltx2_compact_av_cross_adaln",
        "ltx2_fp8_placement_scope",
        "ltx2_attn_auto_dispatch",
        "ltx2_flash_attn_4",
        "ltx2_prompt_kv_checkpoint",
        "ltx2_padded_prompt_trim",
        "ltx2_partial_gradient_checkpointing",
        "ltx2_compile_inner_blocks",
        "ltx2_bounded_activation_offload",
        "ltx2_block_swap_async_backward",
        "ltx2_block_swap_trainable_ring",
        "ltx2_validate_training_tensors",
    )
    for name in flags:
        if getattr(config_section, name, False):
            cmd.append(f"--{name}")

    values = (
        ("ltx2_compact_av_cross_adaln_min_tokens", "ltx2_compact_av_cross_adaln"),
        ("ltx2_prompt_kv_checkpoint_max_mb", "ltx2_prompt_kv_checkpoint"),
        ("ltx2_activation_offload_max_inflight", "ltx2_bounded_activation_offload"),
        ("ltx2_activation_offload_keep_trailing", "ltx2_bounded_activation_offload"),
        ("ltx2_activation_offload_min_mb", "ltx2_bounded_activation_offload"),
    )
    for name, enabled_by in values:
        value = getattr(config_section, name, None)
        if getattr(config_section, enabled_by, False) and value is not None:
            cmd += [f"--{name}", str(value)]

    xm_k = int(getattr(config_section, "ltx2_xm_k", 1) or 1)
    if xm_k > 1:
        cmd += ["--ltx2_xm_k", str(xm_k)]


def _training_requests_lycoris(t, network_module: str) -> bool:
    return (
        "lycoris" in (network_module or "").lower()
        or bool(getattr(t, "lycoris_config", ""))
        or bool(getattr(t, "lycoris_algo", ""))
        or getattr(t, "lycoris_factor", None) is not None
        or getattr(t, "lycoris_conv_dim", None) is not None
        or getattr(t, "lycoris_conv_alpha", None) is not None
        or getattr(t, "lycoris_dropout", None) is not None
        or str(getattr(t, "lora_target_preset", "") or "").lower() == "lycoris"
    )


def _effective_training_network_module(t) -> str:
    network_module = _effective_network_module(t.network_module or "")
    if _training_requests_lycoris(t, network_module):
        return "lycoris.kohya"
    return network_module


def _training_network_args_parts(t) -> list[str]:
    network_args_parts = _split_cli_args(t.network_args)

    _append_network_arg(network_args_parts, "rank_dropout", t.rank_dropout)
    _append_network_arg(network_args_parts, "module_dropout", t.module_dropout)
    _append_network_arg(network_args_parts, "use_dora", t.use_dora)
    _append_network_arg(network_args_parts, "use_dora_oft", t.use_dora_oft)
    _append_network_arg(network_args_parts, "use_oft", getattr(t, "use_oft", False))
    _append_network_arg(network_args_parts, "use_rslora", getattr(t, "use_rslora", False))
    _append_network_arg(network_args_parts, "adaptive_rank", t.adaptive_rank)
    _append_network_arg(network_args_parts, "adaptive_rank_target", t.adaptive_rank_target)
    _append_network_arg(network_args_parts, "adaptive_rank_min_rank", t.adaptive_rank_min_rank)
    _append_network_arg(network_args_parts, "adaptive_rank_init_rank", t.adaptive_rank_init_rank)
    _append_network_arg(network_args_parts, "adaptive_rank_quantile", t.adaptive_rank_quantile)
    _append_network_arg(network_args_parts, "adaptive_rank_weight", t.adaptive_rank_weight)
    _append_network_arg(network_args_parts, "algo", getattr(t, "lycoris_algo", "") or None)
    _append_network_arg(network_args_parts, "factor", getattr(t, "lycoris_factor", None))
    _append_network_arg(network_args_parts, "conv_dim", getattr(t, "lycoris_conv_dim", None))
    _append_network_arg(network_args_parts, "conv_alpha", getattr(t, "lycoris_conv_alpha", None))
    _append_network_arg(network_args_parts, "dropout", getattr(t, "lycoris_dropout", None))

    return network_args_parts


def _generated_sample_prompts_path(config: ProjectConfig) -> Path:
    return Path(config.project_dir) / "sample_prompts.generated.txt"


def _effective_training_sample_prompts(config: ProjectConfig) -> str:
    training = config.training
    if training.sample_prompts:
        return training.sample_prompts
    if training.sample_prompts_text.strip():
        output_path = _generated_sample_prompts_path(config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(training.sample_prompts_text, encoding="utf-8")
        return str(output_path)
    return ""


def _effective_full_finetune_sample_prompts(config: ProjectConfig) -> str:
    training = config.training
    full_finetune = getattr(config, "full_finetune", None)
    if full_finetune is None:
        return _effective_training_sample_prompts(config)
    if full_finetune.sample_prompts:
        return full_finetune.sample_prompts
    if full_finetune.sample_prompts_text.strip():
        output_path = _generated_sample_prompts_path(config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_finetune.sample_prompts_text, encoding="utf-8")
        return str(output_path)
    if training.sample_prompts:
        return training.sample_prompts
    if training.sample_prompts_text.strip():
        return _effective_training_sample_prompts(config)
    return ""


def _effective_caching_sample_prompts(config: ProjectConfig) -> str:
    caching = config.caching
    if caching.sample_prompts:
        return caching.sample_prompts
    return _effective_training_sample_prompts(config)


def _split_cli_args(raw: str) -> list[str]:
    if not raw:
        return []
    parts = shlex.split(raw, posix=False)
    normalized: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
            normalized.append(part[1:-1])
        else:
            normalized.append(part)
    return normalized


def _cli_args_has_key(args: list[str], key: str) -> bool:
    prefix = f"{key}="
    return any(part == key or part.startswith(prefix) for part in args)


def _is_qapollo_optimizer_type(optimizer_type: str | None) -> bool:
    opt = (optimizer_type or "").lower()
    return opt in {
        "qapollo",
        "q_apollo",
        "qapollo_adamw",
        "qapolloadamw",
        "q_apollo_adamw",
        "apollo_torch.qapolloadamw",
        "apollo_torch.q_apollo.adamw",
    } or opt.startswith("apollo_torch.q_apollo.")


def _remote_stage_orchestrator_config_path(config: ProjectConfig) -> Path:
    project_key = config.project_dir or config.name or "default"
    digest = hashlib.sha1(project_key.encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"ltx2_remote_stage_orchestrator_{digest}.json"


def _write_remote_stage_orchestrator_config(config: ProjectConfig) -> Path:
    path = _remote_stage_orchestrator_config_path(config)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


def _append_network_arg(args_parts: list[str], key: str, value) -> None:
    prefix = f"{key}="
    if any(part.startswith(prefix) for part in args_parts):
        return
    if isinstance(value, bool):
        if value:
            args_parts.append(f"{key}=true")
        return
    if value is None:
        return
    args_parts.append(f"{key}={value}")


def _append_optional(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd += [flag, str(value)]


def _accelerate_executable() -> str:
    executable_name = "accelerate.exe" if sys.platform == "win32" else "accelerate"
    executable = Path(sys.executable).parent / executable_name
    if executable.exists():
        return str(executable)
    repo_venv_executable = (
        Path(__file__).resolve().parents[3] / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / executable_name
    )
    if repo_venv_executable.exists():
        return str(repo_venv_executable)
    return "accelerate"


def _accelerate_launch_prefix(mixed_precision: str, extra_args: str) -> list[str]:
    extra = _split_cli_args(extra_args)
    cmd = [
        _accelerate_executable(),
        "launch",
        *extra,
        "--mixed_precision",
        mixed_precision,
    ]
    return cmd


def _ltx2_timestep_sampling(value: str) -> str:
    return "shifted_logit_normal" if value in ("", "sigma") else value


def _append_key_value_args(args_parts: list[str], raw: str) -> None:
    args_parts.extend(_split_cli_args(raw))


def _append_audio_metrics_args(cmd: list[str], training) -> None:
    if not training.audio_metrics:
        return
    cmd.append("--audio_metrics")
    args_parts: list[str] = []
    _append_key_value_args(args_parts, training.audio_metrics_args)
    if training.audio_metrics_mel_metrics:
        args_parts.append("mel_metrics=true")
        if training.audio_metrics_mel_compute_every != 100:
            args_parts.append(f"mel_compute_every={training.audio_metrics_mel_compute_every}")
    if training.audio_metrics_clap_similarity:
        args_parts.append("clap_similarity=true")
    if training.audio_metrics_clap_fad:
        args_parts.append("clap_fad=true")
        if training.audio_metrics_clap_fad_min_samples != 513:
            args_parts.append(f"clap_fad_min_samples={training.audio_metrics_clap_fad_min_samples}")
    if training.audio_metrics_av_onset_alignment:
        args_parts.append("av_onset_alignment=true")
    if training.audio_metrics_av_desync:
        args_parts.append("av_desync=true")
        args_parts.append(f"av_desync_checkpoint={training.audio_metrics_av_desync_checkpoint}")
        if training.audio_metrics_av_desync_device != "cpu":
            args_parts.append(f"av_desync_device={training.audio_metrics_av_desync_device}")
        if training.audio_metrics_av_desync_max_length_s != 8.0:
            args_parts.append(f"av_desync_max_length_s={training.audio_metrics_av_desync_max_length_s}")
    if args_parts:
        cmd += ["--audio_metrics_args"] + args_parts


def _compile_dynamic_value(value) -> str | None:
    if value in (None, False, ""):
        return None
    if value is True:
        return "true"
    normalized = str(value).lower()
    if normalized in {"true", "false", "auto"}:
        return normalized
    return None


def _append_compile_args(cmd: list[str], t) -> None:
    if getattr(t, "compile", False):
        cmd.append("--compile")
        if getattr(t, "compile_backend", ""):
            cmd += ["--compile_backend", t.compile_backend]
        if getattr(t, "compile_mode", ""):
            cmd += ["--compile_mode", t.compile_mode]
        inductor_config = getattr(t, "inductor_config", "")
        if inductor_config:
            tokens = [tok for tok in inductor_config.replace(",", " ").split() if tok]
            if tokens:
                cmd += ["--inductor_config", *tokens]
        compile_dynamic = _compile_dynamic_value(getattr(t, "compile_dynamic", None))
        if compile_dynamic:
            cmd += ["--compile_dynamic", compile_dynamic]
        if getattr(t, "compile_fullgraph", False):
            cmd.append("--compile_fullgraph")
        if getattr(t, "compile_cache_size_limit", None) is not None:
            cmd += ["--compile_cache_size_limit", str(t.compile_cache_size_limit)]
        if getattr(t, "compile_auto_cache_size_limit", False):
            cmd.append("--compile_auto_cache_size_limit")
        if getattr(t, "compile_fallback_to_eager", False):
            cmd.append("--compile_fallback_to_eager")
        if getattr(t, "compile_cudagraph_mark_step", False):
            cmd.append("--compile_cudagraph_mark_step")

    if getattr(t, "dynamo_backend", "NO") != "NO":
        cmd += ["--dynamo_backend", t.dynamo_backend]
        if getattr(t, "dynamo_mode", None):
            cmd += ["--dynamo_mode", t.dynamo_mode]
        if getattr(t, "dynamo_fullgraph", False):
            cmd.append("--dynamo_fullgraph")
        if getattr(t, "dynamo_dynamic", False):
            cmd.append("--dynamo_dynamic")
        if getattr(t, "dynamo_use_regional_compilation", False):
            cmd.append("--dynamo_use_regional_compilation")


def _append_dataloader_args(cmd: list[str], t) -> None:
    if getattr(t, "max_data_loader_n_workers", None) is not None:
        cmd += ["--max_data_loader_n_workers", str(t.max_data_loader_n_workers)]
    if getattr(t, "persistent_data_loader_workers", False):
        cmd.append("--persistent_data_loader_workers")
    if getattr(t, "dataloader_pin_memory", False):
        cmd.append("--dataloader_pin_memory")
    if getattr(t, "dataloader_prefetch_factor", None) is not None:
        cmd += ["--dataloader_prefetch_factor", str(t.dataloader_prefetch_factor)]


def _append_cuda_args(cmd: list[str], t) -> None:
    if getattr(t, "cuda_allow_tf32", False):
        cmd.append("--cuda_allow_tf32")
    if getattr(t, "cuda_cudnn_benchmark", False):
        cmd.append("--cuda_cudnn_benchmark")
    if getattr(t, "cuda_memory_fraction", None) is not None:
        cmd += ["--cuda_memory_fraction", str(t.cuda_memory_fraction)]


def _append_h2d_block_swap_args(cmd: list[str], settings) -> None:
    if not getattr(settings, "block_swap_h2d_only", False):
        return
    cmd.append("--block_swap_h2d_only")
    ring_size = int(getattr(settings, "block_swap_ring_size", 2) or 2)
    if ring_size != 2:
        cmd += ["--block_swap_ring_size", str(ring_size)]


def _append_advanced_cli_args(cmd: list[str], settings) -> None:
    """Append persisted advanced flags shared by LoRA and full-FT commands."""
    if getattr(settings, "debug_dataset", False):
        cmd.append("--debug_dataset")
    if getattr(settings, "save_precision", None):
        cmd += ["--save_precision", str(settings.save_precision)]
    if getattr(settings, "log_grad_metrics", False):
        cmd.append("--log_grad_metrics")

    if getattr(settings, "int4_convrot_awq_calibration", False):
        cmd.append("--int4_convrot_awq_calibration")
    if getattr(settings, "int4_convrot_awq_scales", ""):
        cmd += ["--int4_convrot_awq_scales", str(settings.int4_convrot_awq_scales)]
    if float(getattr(settings, "int4_convrot_awq_alpha", 0.25)) != 0.25:
        cmd += ["--int4_convrot_awq_alpha", str(settings.int4_convrot_awq_alpha)]
    if getattr(settings, "int4_convrot_activation_calibration_report", ""):
        cmd += [
            "--int4_convrot_activation_calibration_report",
            str(settings.int4_convrot_activation_calibration_report),
        ]
    if int(getattr(settings, "int4_convrot_activation_calibration_batches", 1)) != 1:
        cmd += [
            "--int4_convrot_activation_calibration_batches",
            str(settings.int4_convrot_activation_calibration_batches),
        ]
    if getattr(settings, "int4_convrot_activation_calibration_regex", ""):
        cmd += [
            "--int4_convrot_activation_calibration_regex",
            str(settings.int4_convrot_activation_calibration_regex),
        ]
    if int(getattr(settings, "int4_convrot_activation_calibration_max_rows", 128)) != 128:
        cmd += [
            "--int4_convrot_activation_calibration_max_rows",
            str(settings.int4_convrot_activation_calibration_max_rows),
        ]
    if int(getattr(settings, "int4_convrot_activation_calibration_max_layers", 0)) != 0:
        cmd += [
            "--int4_convrot_activation_calibration_max_layers",
            str(settings.int4_convrot_activation_calibration_max_layers),
        ]
    if getattr(settings, "int4_convrot_activation_calibration_only", False):
        cmd.append("--int4_convrot_activation_calibration_only")


def build_cache_latents_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_latents.py."""
    toml_path = export_dataset_toml(config)
    c = config.caching
    t = config.training
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, c.ltx2_checkpoint)
    sample_prompts = _effective_caching_sample_prompts(config)

    cmd = [
        sys.executable,
        "-u",
        _find_script("ltx2_cache_latents.py"),
        "--dataset_config",
        str(toml_path),
        "--ltx2_checkpoint",
        ltx2_checkpoint,
        "--ltx2_mode",
        c.ltx2_mode,
    ]

    if c.vae_dtype:
        cmd += ["--vae_dtype", c.vae_dtype]
    if c.device:
        cmd += ["--device", c.device]
    if c.skip_existing:
        cmd.append("--skip_existing")
    if c.atomic_cache_writes:
        cmd.append("--atomic_cache_writes")
    if c.keep_cache:
        cmd.append("--keep_cache")
    if c.num_workers is not None:
        cmd += ["--num_workers", str(c.num_workers)]
    if getattr(c, "cache_distributed", False):
        cmd.append("--cache_distributed")
    if c.cpu_staged_checkpoint_loading:
        cmd.append("--cpu_staged_checkpoint_loading")
    if getattr(c, "video_decode_backend", None):
        cmd += ["--video_decode_backend", c.video_decode_backend]
    if getattr(c, "video_decode_device", None):
        cmd += ["--video_decode_device", c.video_decode_device]
    if c.vae_chunk_size is not None:
        cmd += ["--vae_chunk_size", str(c.vae_chunk_size)]
    if c.vae_spatial_tile_size is not None:
        cmd += ["--vae_spatial_tile_size", str(c.vae_spatial_tile_size)]
    if c.vae_spatial_tile_overlap is not None:
        cmd += ["--vae_spatial_tile_overlap", str(c.vae_spatial_tile_overlap)]
    if c.vae_temporal_tile_size is not None:
        cmd += ["--vae_temporal_tile_size", str(c.vae_temporal_tile_size)]
    if c.vae_temporal_tile_overlap is not None:
        cmd += ["--vae_temporal_tile_overlap", str(c.vae_temporal_tile_overlap)]

    # Reference (V2V)
    if c.reference_frames != 1:
        cmd += ["--reference_frames", str(c.reference_frames)]
    if c.reference_downscale != 1:
        cmd += ["--reference_downscale", str(c.reference_downscale)]
    if getattr(c, "reference_temporal_scale", 1) != 1:
        cmd += ["--reference_temporal_scale", str(c.reference_temporal_scale)]

    # Audio source options
    if c.ltx2_mode in ("av", "audio"):
        cmd += ["--ltx2_audio_source", c.ltx2_audio_source]
        if c.ltx2_audio_source == "audio_files" and c.ltx2_audio_dir:
            cmd += ["--ltx2_audio_dir", c.ltx2_audio_dir]
            if c.ltx2_audio_ext:
                cmd += ["--ltx2_audio_ext", c.ltx2_audio_ext]
        if c.ltx2_audio_dtype:
            cmd += ["--ltx2_audio_dtype", c.ltx2_audio_dtype]
        if c.audio_video_latent_channels is not None:
            cmd += ["--audio_video_latent_channels", str(c.audio_video_latent_channels)]
        if c.audio_video_latent_dtype:
            cmd += ["--audio_video_latent_dtype", c.audio_video_latent_dtype]
        if c.audio_only_target_resolution is not None:
            cmd += ["--audio_only_target_resolution", str(c.audio_only_target_resolution)]
        if c.audio_only_target_fps is not None:
            cmd += ["--audio_only_target_fps", str(c.audio_only_target_fps)]
        if c.audio_only_sequence_resolution != 64:
            cmd += ["--audio_only_sequence_resolution", str(c.audio_only_sequence_resolution)]

    # I2V latent precaching
    if c.precache_sample_latents and sample_prompts:
        cmd.append("--precache_sample_latents")
        cmd += ["--sample_prompts", sample_prompts]
        cmd += ["--ltx_version", t.ltx_version]
        cmd += ["--sample_sampling_preset", t.sample_sampling_preset]
        _append_optional(cmd, "--height", t.height)
        _append_optional(cmd, "--width", t.width)
        _append_optional(cmd, "--sample_num_frames", t.sample_num_frames)
        if c.sample_latents_cache:
            cmd += ["--sample_latents_cache", c.sample_latents_cache]

    if c.save_dataset_manifest:
        cmd += ["--save_dataset_manifest", c.save_dataset_manifest]
    if getattr(c, "preserve_audio_timing", False):
        cmd.append("--preserve_audio_timing")

    cmd += _split_cli_args(c.cache_latents_extra_args)
    return cmd


def build_cache_text_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_text_encoder_outputs.py."""
    toml_path = export_dataset_toml(config)
    c = config.caching
    t = config.training
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, c.ltx2_checkpoint)
    gemma_safetensors = _effective_gemma_safetensors(config, c.gemma_safetensors, c.gemma_root)
    gemma_root = _effective_gemma_root(config, c.gemma_root, gemma_safetensors)
    sample_prompts = _effective_caching_sample_prompts(config)

    cmd = [
        sys.executable,
        "-u",
        _find_script("ltx2_cache_text_encoder_outputs.py"),
        "--dataset_config",
        str(toml_path),
        "--ltx2_checkpoint",
        ltx2_checkpoint,
        "--ltx2_mode",
        c.ltx2_mode,
    ]

    if gemma_root:
        cmd += ["--gemma_root", gemma_root]
    if gemma_safetensors:
        cmd += ["--gemma_safetensors", gemma_safetensors]
    if c.ltx2_text_encoder_checkpoint:
        cmd += ["--ltx2_text_encoder_checkpoint", c.ltx2_text_encoder_checkpoint]
    if c.mixed_precision != "no":
        cmd += ["--mixed_precision", c.mixed_precision]
    if c.skip_existing:
        cmd.append("--skip_existing")
    if c.atomic_cache_writes:
        cmd.append("--atomic_cache_writes")
    if c.keep_cache:
        cmd.append("--keep_cache")
    if c.num_workers is not None:
        cmd += ["--num_workers", str(c.num_workers)]
    if getattr(c, "cache_distributed", False):
        cmd.append("--cache_distributed")
    if c.cpu_staged_checkpoint_loading:
        cmd.append("--cpu_staged_checkpoint_loading")
    if c.gemma_load_in_8bit:
        cmd.append("--gemma_load_in_8bit")
    if c.gemma_load_in_4bit:
        cmd.append("--gemma_load_in_4bit")
        cmd += ["--gemma_bnb_4bit_quant_type", c.gemma_bnb_4bit_quant_type]
    if c.gemma_bnb_4bit_disable_double_quant:
        cmd.append("--gemma_bnb_4bit_disable_double_quant")
    if c.gemma_bnb_4bit_compute_dtype != "auto":
        cmd += ["--gemma_bnb_4bit_compute_dtype", c.gemma_bnb_4bit_compute_dtype]
    if c.gemma_fp8_weight_offload:
        cmd.append("--gemma_fp8_weight_offload")
    else:
        cmd.append("--no-gemma_fp8_weight_offload")

    # Precaching
    if c.precache_sample_prompts and sample_prompts:
        cmd.append("--precache_sample_prompts")
        cmd += ["--sample_prompts", sample_prompts]
        cmd += ["--ltx_version", t.ltx_version]
        cmd += ["--sample_sampling_preset", t.sample_sampling_preset]
        if t.sample_use_default_negative_prompt is True:
            cmd.append("--sample_use_default_negative_prompt")
        elif t.sample_use_default_negative_prompt is False:
            cmd.append("--no-sample_use_default_negative_prompt")
        _append_optional(cmd, "--guidance_scale", t.guidance_scale)
        _append_optional(cmd, "--video_cfg_scale", t.video_cfg_scale)
        _append_optional(cmd, "--audio_cfg_scale", t.audio_cfg_scale)
        if c.sample_prompts_cache:
            cmd += ["--sample_prompts_cache", c.sample_prompts_cache]
    if c.precache_preservation_prompts:
        cmd.append("--precache_preservation_prompts")
        if c.preservation_prompts_cache:
            cmd += ["--preservation_prompts_cache", c.preservation_prompts_cache]
        if c.blank_preservation:
            cmd.append("--blank_preservation")
        if c.dop:
            cmd.append("--dop")
            if c.dop_class_prompt:
                cmd += ["--dop_class_prompt", c.dop_class_prompt]
            dop_cache_args: list[str] = []
            _append_key_value_args(dop_cache_args, c.dop_args)
            if c.dop_mode != "fixed":
                dop_cache_args.append(f"mode={c.dop_mode}")
            if c.dop_trigger:
                dop_cache_args.append(f"trigger={c.dop_trigger}")
            if c.dop_replacements:
                dop_cache_args.append(f"replace={c.dop_replacements}")
            if c.dop_prompt_bank:
                dop_cache_args.append(f"bank={c.dop_prompt_bank}")
            if dop_cache_args:
                cmd += ["--dop_args"] + dop_cache_args

    if c.cache_before_connector:
        cmd.append("--cache_before_connector")

    cmd += _split_cli_args(c.cache_text_extra_args)
    return cmd


def build_cache_preview_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_preview.py."""
    c = config.caching
    cmd = [
        sys.executable,
        "-u",
        _find_script("ltx2_cache_preview.py"),
        "--input",
        _effective_cache_preview_input(config),
        "--output",
        _effective_cache_preview_output(config),
        "--fps",
        str(c.cache_preview_fps),
    ]

    if c.cache_preview_decode:
        cmd += ["--checkpoint", _effective_ltx2_checkpoint(config, c.ltx2_checkpoint)]
        if c.cache_preview_device:
            cmd += ["--device", c.cache_preview_device]
        if c.cache_preview_dtype:
            cmd += ["--dtype", c.cache_preview_dtype]
    else:
        cmd.append("--no_decode")

    if c.cache_preview_stats:
        cmd.append("--stats")
    if c.cache_preview_fail_on_error:
        cmd.append("--fail_on_error")
    if c.cache_preview_check_source:
        cmd.append("--check_source")
    if c.cache_preview_require_companions.strip():
        cmd += ["--require_companions", c.cache_preview_require_companions.strip()]
    cmd += ["--av_duration_tolerance", str(c.cache_preview_av_duration_tolerance)]
    if c.cache_preview_limit is not None:
        cmd += ["--limit", str(c.cache_preview_limit)]
    return cmd


def build_inference_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_generate_video.py."""
    s = config.inference
    gemma_safetensors = _effective_gemma_safetensors(config, s.gemma_safetensors, s.gemma_root)
    gemma_root = _effective_gemma_root(config, s.gemma_root, gemma_safetensors)

    cmd = [
        sys.executable,
        "-u",
        _find_script("ltx2_generate_video.py"),
        "--ltx2_checkpoint",
        s.ltx2_checkpoint,
        "--ltx2_mode",
        s.ltx2_mode,
    ]

    if s.vae:
        cmd += ["--vae", s.vae]
    if s.vae_dtype:
        cmd += ["--vae_dtype", s.vae_dtype]
    if s.device:
        cmd += ["--device", s.device]
    if gemma_root:
        cmd += ["--gemma_root", gemma_root]
    if gemma_safetensors:
        cmd += ["--gemma_safetensors", gemma_safetensors]

    # LoRA
    if s.lora_weight:
        cmd += ["--lora_weight"] + _split_cli_args(s.lora_weight)
        cmd += ["--lora_multiplier", str(s.lora_multiplier)]
    if s.include_patterns:
        cmd += ["--include_patterns"] + _split_cli_args(s.include_patterns)
    if s.exclude_patterns:
        cmd += ["--exclude_patterns"] + _split_cli_args(s.exclude_patterns)

    # Prompt
    if s.prompt:
        cmd += ["--prompt", s.prompt]
    if s.negative_prompt:
        cmd += ["--negative_prompt", s.negative_prompt]
    if s.from_file:
        cmd += ["--from_file", s.from_file]

    # Sampling params
    cmd += ["--sampling_preset", s.sampling_preset]
    if s.sample_sigma_schedule != "auto":
        cmd += ["--sample_sigma_schedule", s.sample_sigma_schedule]
    if s.sample_sampler != "auto":
        cmd += ["--sample_sampler", s.sample_sampler]
    if s.use_default_negative_prompt is True:
        cmd.append("--use_default_negative_prompt")
    elif s.use_default_negative_prompt is False:
        cmd.append("--no-use_default_negative_prompt")
    _append_optional(cmd, "--height", s.height)
    _append_optional(cmd, "--width", s.width)
    _append_optional(cmd, "--frame_count", s.frame_count)
    _append_optional(cmd, "--frame_rate", s.frame_rate)
    _append_optional(cmd, "--sample_steps", s.sample_steps)
    _append_optional(cmd, "--guidance_scale", s.guidance_scale)
    if s.cfg_scale is not None:
        cmd += ["--cfg_scale", str(s.cfg_scale)]
    cmd += ["--discrete_flow_shift", str(s.discrete_flow_shift)]
    if s.seed is not None:
        cmd += ["--seed", str(s.seed)]
    _append_optional(cmd, "--video_cfg_scale", s.video_cfg_scale)
    _append_optional(cmd, "--audio_cfg_scale", s.audio_cfg_scale)
    _append_optional(cmd, "--video_modality_scale", s.video_modality_scale)
    _append_optional(cmd, "--audio_modality_scale", s.audio_modality_scale)
    _append_optional(cmd, "--video_rescale_scale", s.video_rescale_scale)
    _append_optional(cmd, "--audio_rescale_scale", s.audio_rescale_scale)
    if s.stg_scale is not None:
        cmd += ["--stg_scale", str(s.stg_scale)]
    if s.stg_blocks:
        cmd += ["--stg_blocks"] + _split_cli_args(s.stg_blocks)
    if s.stg_mode:
        cmd += ["--stg_mode", s.stg_mode]
    if s.rescale_scale is not None:
        cmd += ["--rescale_scale", str(s.rescale_scale)]
    if s.av_bimodal_cfg is True:
        cmd.append("--av_bimodal_cfg")
    elif s.av_bimodal_cfg is False:
        cmd.append("--no-av_bimodal_cfg")
    _append_optional(cmd, "--av_bimodal_scale", s.av_bimodal_scale)

    # Precision
    if s.mixed_precision != "no":
        cmd += ["--mixed_precision", s.mixed_precision]
    cmd += ["--attn_mode", s.attn_mode]
    if s.flash_attn:
        cmd.append("--flash_attn")
    if s.flash3:
        cmd.append("--flash3")
    if getattr(s, "cudnn_attn", False):
        cmd.append("--cudnn_attn")
    if s.sdpa:
        cmd.append("--sdpa")
    if s.xformers:
        cmd.append("--xformers")
    if s.fp8_base:
        cmd.append("--fp8_base")
    if s.fp8_scaled:
        cmd.append("--fp8_scaled")
    if getattr(s, "fp8_keep_blocks", ""):
        cmd += ["--fp8_keep_blocks", s.fp8_keep_blocks]
    if s.fp8_w8a8:
        cmd.append("--fp8_w8a8")
    if s.w8a8_mode != "int8":
        cmd += ["--w8a8_mode", s.w8a8_mode]
    if s.fp8_w8a8 and s.w8a8_mode == "int8" and getattr(s, "w8a8_backend", None):
        cmd += ["--w8a8_backend", s.w8a8_backend]
    if s.fp8_upcast:
        cmd.append("--fp8_upcast")
    if s.fp8_upcast_stochastic:
        cmd.append("--fp8_upcast_stochastic")
    if s.fp8_upcast_seed != 0:
        cmd += ["--fp8_upcast_seed", str(s.fp8_upcast_seed)]
    if s.nf4_base:
        cmd.append("--nf4_base")
    if s.nf4_block_size != 64:
        cmd += ["--nf4_block_size", str(s.nf4_block_size)]
    if s.loftq_init:
        cmd.append("--loftq_init")
    if s.loftq_iters != 2:
        cmd += ["--loftq_iters", str(s.loftq_iters)]
    if s.awq_calibration:
        cmd.append("--awq_calibration")
    if s.awq_alpha != 0.25:
        cmd += ["--awq_alpha", str(s.awq_alpha)]
    if s.awq_num_batches != 8:
        cmd += ["--awq_num_batches", str(s.awq_num_batches)]
    if s.network_dim:
        cmd += ["--network_dim", str(s.network_dim)]
    if s.split_attn_target:
        cmd += ["--split_attn_target"] + _split_cli_args(s.split_attn_target)
    if s.split_attn_mode:
        cmd += ["--split_attn_mode", s.split_attn_mode]
    if s.split_attn_chunk_size:
        cmd += ["--split_attn_chunk_size", str(s.split_attn_chunk_size)]
    if s.ffn_chunk_target:
        cmd += ["--ffn_chunk_target"] + _split_cli_args(s.ffn_chunk_target)
    if s.ffn_chunk_size:
        cmd += ["--ffn_chunk_size", str(s.ffn_chunk_size)]

    # Gemma quantization
    if s.gemma_load_in_8bit:
        cmd.append("--gemma_load_in_8bit")
    if s.gemma_load_in_4bit:
        cmd.append("--gemma_load_in_4bit")
        if s.gemma_bnb_4bit_quant_type != "nf4":
            cmd += ["--gemma_bnb_4bit_quant_type", s.gemma_bnb_4bit_quant_type]
    if s.gemma_bnb_4bit_disable_double_quant:
        cmd.append("--gemma_bnb_4bit_disable_double_quant")
    if s.gemma_fp8_weight_offload:
        cmd.append("--gemma_fp8_weight_offload")
    else:
        cmd.append("--no-gemma_fp8_weight_offload")

    # Memory
    if s.offloading:
        cmd.append("--sample_with_offloading")
    if s.blocks_to_swap is not None:
        cmd += ["--blocks_to_swap", str(s.blocks_to_swap)]
    if s.use_pinned_memory_for_block_swap:
        cmd.append("--use_pinned_memory_for_block_swap")

    # Conditioning
    if not s.sample_i2v_token_timestep_mask:
        cmd.append("--no-sample_i2v_token_timestep_mask")
    if s.reference_downscale != 1:
        cmd += ["--reference_downscale", str(s.reference_downscale)]
    if getattr(s, "reference_temporal_scale", 1) != 1:
        cmd += ["--reference_temporal_scale", str(s.reference_temporal_scale)]
    if s.reference_frames != 1:
        cmd += ["--reference_frames", str(s.reference_frames)]
    if s.sample_include_reference:
        cmd.append("--sample_include_reference")
    if s.reference_image:
        cmd += ["--reference_image", s.reference_image]
    if s.reference_video:
        cmd += ["--reference_video", s.reference_video]
    if getattr(s, "reference_audio", ""):
        cmd += ["--reference_audio", s.reference_audio]
    if getattr(s, "ic_lora_strategy", "auto") not in ("", "auto"):
        cmd += ["--ic_lora_strategy", s.ic_lora_strategy]
    if getattr(s, "av_cross_attention_mode", "both") not in ("", "both"):
        cmd += ["--av_cross_attention_mode", s.av_cross_attention_mode]
    if getattr(s, "inpaint_mask", ""):
        cmd += ["--inpaint_mask", s.inpaint_mask]
    if getattr(s, "inpaint_source", ""):
        cmd += ["--inpaint_source", s.inpaint_source]
    if getattr(s, "inpaint_invert", None) is True:
        cmd.append("--inpaint_invert")
    elif getattr(s, "inpaint_invert", None) is False:
        cmd.append("--no-inpaint_invert")
    if getattr(s, "inpaint_threshold", None) is not None:
        cmd += ["--inpaint_threshold", str(s.inpaint_threshold)]
    if getattr(s, "extend_prefix", ""):
        cmd += ["--extend_prefix", s.extend_prefix]
    if getattr(s, "extend_prefix_frames", None) is not None:
        cmd += ["--extend_prefix_frames", str(s.extend_prefix_frames)]
    if getattr(s, "outpaint_region", ""):
        cmd += ["--outpaint_region", s.outpaint_region]
    if getattr(s, "outpaint_source", ""):
        cmd += ["--outpaint_source", s.outpaint_source]
    if getattr(s, "drive_video", ""):
        cmd += ["--drive_video", s.drive_video]
    if getattr(s, "drive_audio", ""):
        cmd += ["--drive_audio", s.drive_audio]
    if getattr(s, "audio_prefix", ""):
        cmd += ["--audio_prefix", s.audio_prefix]
    if getattr(s, "audio_prefix_seconds", None) is not None:
        cmd += ["--audio_prefix_seconds", str(s.audio_prefix_seconds)]
    if getattr(s, "audio_inpaint_audio", ""):
        cmd += ["--audio_inpaint_audio", s.audio_inpaint_audio]
    if getattr(s, "audio_inpaint_interval", ""):
        cmd += ["--audio_inpaint_interval", s.audio_inpaint_interval]

    # Audio / decode
    if s.sample_disable_audio:
        cmd.append("--sample_disable_audio")
    if s.sample_audio_only:
        cmd.append("--sample_audio_only")
    if s.sample_merge_audio:
        cmd.append("--sample_merge_audio")
    if not s.sample_audio_subprocess:
        cmd.append("--no-sample_audio_subprocess")
    if s.sample_two_stage:
        cmd.append("--sample_two_stage")
        if s.spatial_upsampler_path:
            cmd += ["--spatial_upsampler_path", s.spatial_upsampler_path]
        if getattr(s, "temporal_upsampler_path", ""):
            cmd += ["--temporal_upsampler_path", s.temporal_upsampler_path]
        if s.distilled_lora_path:
            cmd += ["--distilled_lora_path", s.distilled_lora_path]
        if s.sample_stage2_steps != 3:
            cmd += ["--sample_stage2_steps", str(s.sample_stage2_steps)]
        if s.sample_stage1_distilled_lora_multiplier is not None:
            cmd += [
                "--sample_stage1_distilled_lora_multiplier",
                str(s.sample_stage1_distilled_lora_multiplier),
            ]
        if s.sample_stage2_distilled_lora_multiplier is not None:
            cmd += [
                "--sample_stage2_distilled_lora_multiplier",
                str(s.sample_stage2_distilled_lora_multiplier),
            ]
    if s.sample_tiled_vae:
        cmd.append("--sample_tiled_vae")
    if s.sample_vae_tile_size != 512:
        cmd += ["--sample_vae_tile_size", str(s.sample_vae_tile_size)]
    if s.sample_vae_tile_overlap != 64:
        cmd += ["--sample_vae_tile_overlap", str(s.sample_vae_tile_overlap)]
    if s.sample_vae_temporal_tile_size != 0:
        cmd += ["--sample_vae_temporal_tile_size", str(s.sample_vae_temporal_tile_size)]
    if s.sample_vae_temporal_tile_overlap != 8:
        cmd += ["--sample_vae_temporal_tile_overlap", str(s.sample_vae_temporal_tile_overlap)]
    if s.sample_disable_flash_attn:
        cmd.append("--sample_disable_flash_attn")

    # Precached inputs
    if s.use_precached_sample_prompts:
        cmd.append("--use_precached_sample_prompts")
    if s.sample_prompts_cache:
        cmd += ["--sample_prompts_cache", s.sample_prompts_cache]
    if s.use_precached_sample_latents:
        cmd.append("--use_precached_sample_latents")
    if s.sample_latents_cache:
        cmd += ["--sample_latents_cache", s.sample_latents_cache]

    # Output
    if s.output_dir:
        cmd += ["--output_dir", s.output_dir]
    if s.output_name:
        cmd += ["--output_name", s.output_name]

    cmd += _split_cli_args(s.extra_args)
    return cmd


def _append_ltx2_conditioning(cmd: list[str], t, config, recipe_filename: str = "conditioning_config.toml") -> None:
    """Emit the LTX-2 intrinsic / directional / recipe conditioning args (shared by the LoRA and
    full-finetune builders). A conditioning recipe (--ltx2_conditioning_config) is authoritative and
    mutually exclusive with the individual conditioning flags: when set, only the recipe is emitted and
    the flags it owns are skipped. Keyframe / video-anchor (orthogonal latent guides) always apply.
    --lora_target_preset / --ic_lora_strategy are LoRA-only and emitted by the LoRA builder, not here.

    Recipe source priority: the structured GUI recipe (serialized to a TOML here) wins; otherwise the
    explicit ltx2_conditioning_config path; otherwise the individual flags below.
    """
    # The structured recipe is built once on the (training-scoped) Conditioning tab and applies to both
    # LoRA and full fine-tune, so always source it from config.training rather than the section `t`
    # (full_finetune inherits the field but the GUI never writes it there).
    builder_recipe = _conditioning_recipe_for_mode(
        getattr(config.training, "conditioning_recipe", None), str(getattr(t, "ltx2_mode", "video") or "video")
    )
    if conditioning_recipe_is_active(builder_recipe):
        recipe_path = export_conditioning_toml(config, builder_recipe, recipe_filename)
        cmd += ["--ltx2_conditioning_config", str(recipe_path)]
        recipe = "recipe"  # truthy sentinel: the recipe owns the conditioning flags below
    else:
        recipe = str(getattr(t, "ltx2_conditioning_config", "") or "").strip()
        if recipe:
            cmd += ["--ltx2_conditioning_config", recipe]
        else:
            cmd += ["--ltx2_first_frame_conditioning_p", str(t.ltx2_first_frame_conditioning_p)]

    if getattr(t, "keyframe_endpoint_training", False):
        cmd += ["--keyframe_endpoint_training"]
        cmd += ["--keyframe_first_frame_p", str(getattr(t, "keyframe_first_frame_p", 1.0))]
        cmd += ["--keyframe_last_frame_p", str(getattr(t, "keyframe_last_frame_p", 1.0))]
        cmd += ["--keyframe_random_interior_p", str(getattr(t, "keyframe_random_interior_p", 0.0))]
        cmd += ["--keyframe_max_random_interior", str(int(getattr(t, "keyframe_max_random_interior", 0)))]
    if getattr(t, "ltx2_graded_conditioning", False):
        cmd += ["--ltx2_graded_conditioning"]
    if getattr(t, "ltx2_causal_temporal_attention", False):
        cmd += ["--ltx2_causal_temporal_attention"]
    if getattr(t, "ltx2_soft_av_alignment", False):
        cmd += ["--ltx2_soft_av_alignment"]
        cmd += ["--ltx2_soft_av_alignment_sigma", str(getattr(t, "ltx2_soft_av_alignment_sigma", 1.0))]
    if getattr(t, "video_anchor_training", False):
        cmd += ["--video_anchor_training"]
        cmd += ["--video_anchor_probability", str(getattr(t, "video_anchor_probability", 0.5))]
        cmd += ["--video_anchor_count", str(int(getattr(t, "video_anchor_count", 1)))]
        cmd += ["--video_anchor_strategy", str(getattr(t, "video_anchor_strategy", "endpoints_random"))]
        cmd += ["--video_anchor_strength", str(getattr(t, "video_anchor_strength", 1.0))]
    if not recipe and getattr(t, "ltx2_spatial_crop", False):
        # Region is dataset-level (spatial_crop_region); only the master flag / probability / invert.
        cmd += ["--ltx2_spatial_crop"]
        cmd += ["--ltx2_spatial_crop_p", str(getattr(t, "ltx2_spatial_crop_probability", 0.0))]
        if getattr(t, "ltx2_spatial_crop_invert", False):
            cmd += ["--ltx2_spatial_crop_invert"]
    if not recipe and getattr(t, "ltx2_inpaint_mask", False):
        # Mask comes from the dataset's loss_mask_directory; only the flags are CLI args.
        cmd += ["--ltx2_inpaint_mask"]
        cmd += ["--ltx2_inpaint_mask_p", str(getattr(t, "ltx2_inpaint_mask_probability", 0.0))]
        if getattr(t, "ltx2_inpaint_mask_invert", False):
            cmd += ["--ltx2_inpaint_mask_invert"]
        cmd += ["--ltx2_inpaint_mask_threshold", str(getattr(t, "ltx2_inpaint_mask_threshold", 0.5))]
    if not recipe and getattr(t, "ltx2_audio_inpaint_mask", False):
        # Audio mask comes from the dataset's audio_cond_mask_directory; only the flags are CLI args.
        cmd += ["--ltx2_audio_inpaint_mask"]
        cmd += ["--ltx2_audio_inpaint_mask_p", str(getattr(t, "ltx2_audio_inpaint_mask_probability", 0.0))]
        if getattr(t, "ltx2_audio_inpaint_mask_invert", False):
            cmd += ["--ltx2_audio_inpaint_mask_invert"]
        cmd += ["--ltx2_audio_inpaint_mask_threshold", str(getattr(t, "ltx2_audio_inpaint_mask_threshold", 0.5))]
    _ext_prefix = int(getattr(t, "ltx2_extend_prefix_frames", 0) or 0)
    _ext_suffix = int(getattr(t, "ltx2_extend_suffix_frames", 0) or 0)
    if not recipe and (_ext_prefix > 0 or _ext_suffix > 0):
        cmd += ["--ltx2_extend_prefix_frames", str(_ext_prefix)]
        cmd += ["--ltx2_extend_suffix_frames", str(_ext_suffix)]
        cmd += ["--ltx2_extend_p", str(getattr(t, "ltx2_extend_probability", 1.0))]
        # Optional independent per-side probabilities (omit for the shared single draw).
        _ext_pp = getattr(t, "ltx2_extend_prefix_p", None)
        if _ext_pp is not None and str(_ext_pp).strip() != "":
            cmd += ["--ltx2_extend_prefix_p", str(_ext_pp)]
        _ext_sp = getattr(t, "ltx2_extend_suffix_p", None)
        if _ext_sp is not None and str(_ext_sp).strip() != "":
            cmd += ["--ltx2_extend_suffix_p", str(_ext_sp)]
    _aext_prefix = int(getattr(t, "ltx2_audio_extend_prefix_frames", 0) or 0)
    _aext_suffix = int(getattr(t, "ltx2_audio_extend_suffix_frames", 0) or 0)
    if not recipe and (_aext_prefix > 0 or _aext_suffix > 0):
        cmd += ["--ltx2_audio_extend_prefix_frames", str(_aext_prefix)]
        cmd += ["--ltx2_audio_extend_suffix_frames", str(_aext_suffix)]
        cmd += ["--ltx2_audio_extend_p", str(getattr(t, "ltx2_audio_extend_probability", 1.0))]
        _aext_pp = getattr(t, "ltx2_audio_extend_prefix_p", None)
        if _aext_pp is not None and str(_aext_pp).strip() != "":
            cmd += ["--ltx2_audio_extend_prefix_p", str(_aext_pp)]
        _aext_sp = getattr(t, "ltx2_audio_extend_suffix_p", None)
        if _aext_sp is not None and str(_aext_sp).strip() != "":
            cmd += ["--ltx2_audio_extend_suffix_p", str(_aext_sp)]
    _train_direction = str(getattr(t, "ltx2_train_direction", "joint") or "joint")
    if not recipe and _train_direction != "joint":
        # Directional AV training (requires --ltx2_mode av); "joint" emits nothing (default).
        cmd += ["--ltx2_train_direction", _train_direction]


def build_training_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for training via accelerate launch."""
    toml_path = export_dataset_toml(config)
    t = config.training
    # Conditioning recipe (authoritative + mutually exclusive). When set, emit only
    # --ltx2_conditioning_config and skip the conditioning flags it owns (--lora_target_preset,
    # --ic_lora_strategy / --ic_lora_ref_probability, and the intrinsic + directional flags below).
    # Active when an explicit recipe path is set OR the structured GUI builder is active (both lower to
    # --ltx2_conditioning_config in _append_ltx2_conditioning); truthy string for the `if not` gates.
    _cond_recipe = str(getattr(t, "ltx2_conditioning_config", "") or "").strip() or (
        "builder"
        if conditioning_recipe_is_active(
            _conditioning_recipe_for_mode(
                getattr(config.training, "conditioning_recipe", None), str(getattr(t, "ltx2_mode", "video") or "video")
            )
        )
        else ""
    )
    # Only a REFERENCE recipe derives the LoRA target preset (from its IC-LoRA strategy); an intrinsic-only
    # or directional recipe does not, so the user's --lora_target_preset must still apply for those.
    _builder_recipe = _conditioning_recipe_for_mode(
        getattr(config.training, "conditioning_recipe", None), str(getattr(t, "ltx2_mode", "video") or "video")
    )
    if conditioning_recipe_is_active(_builder_recipe):
        _recipe_derives_preset = any(c.type == "reference" for c in _builder_recipe.video.conditions) or any(
            c.type == "reference" for c in _builder_recipe.audio.conditions
        )
    else:
        # An external recipe file cannot be introspected here; preserve the prior "recipe governs preset" behavior.
        _recipe_derives_preset = bool(str(getattr(t, "ltx2_conditioning_config", "") or "").strip())
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, t.ltx2_checkpoint)
    gemma_safetensors = _effective_gemma_safetensors(config, t.gemma_safetensors, t.gemma_root)
    gemma_root = _effective_gemma_root(config, t.gemma_root, gemma_safetensors)
    network_module = _effective_training_network_module(t)
    sample_prompts = _effective_training_sample_prompts(config)
    network_args_parts = _training_network_args_parts(t)

    # Use accelerate launch. The Python module form is equivalent to the
    # console script and avoids PATH issues with Windows virtualenvs.
    cmd = _accelerate_launch_prefix(t.mixed_precision, t.accelerate_extra_args)
    cmd.append(_find_script("ltx2_train_network.py"))

    # Dataset
    if t.config_file:
        cmd += ["--config_file", t.config_file]
    if t.dataset_manifest:
        cmd += ["--dataset_manifest", t.dataset_manifest]
    elif t.dataset_config:
        cmd += ["--dataset_config", t.dataset_config]
    else:
        cmd += ["--dataset_config", str(toml_path)]

    # Model
    cmd += ["--ltx2_checkpoint", ltx2_checkpoint]
    if t.vae:
        cmd += ["--vae", t.vae]
    if gemma_root:
        cmd += ["--gemma_root", gemma_root]
    if gemma_safetensors:
        cmd += ["--gemma_safetensors", gemma_safetensors]
    cmd += ["--ltx2_mode", t.ltx2_mode]
    if t.ltx_version != "2.3":
        cmd += ["--ltx_version", t.ltx_version]
    if t.ltx_version_check_mode != "warn":
        cmd += ["--ltx_version_check_mode", t.ltx_version_check_mode]
    if t.vae_dtype:
        cmd += ["--vae_dtype", t.vae_dtype]
    if t.fp8_base:
        cmd.append("--fp8_base")
    if t.fp8_scaled:
        cmd.append("--fp8_scaled")
    if getattr(t, "fp8_keep_blocks", ""):
        cmd += ["--fp8_keep_blocks", t.fp8_keep_blocks]
    if t.flash_attn:
        cmd.append("--flash_attn")
    if t.flash3:
        cmd.append("--flash3")
    if getattr(t, "cudnn_attn", False):
        cmd.append("--cudnn_attn")
    if t.sdpa:
        cmd.append("--sdpa")
    if t.sage_attn:
        cmd.append("--sage_attn")
    if t.xformers:
        cmd.append("--xformers")
    if t.gemma_load_in_8bit:
        cmd.append("--gemma_load_in_8bit")
    if t.gemma_load_in_4bit:
        cmd.append("--gemma_load_in_4bit")
        if t.gemma_bnb_4bit_quant_type != "nf4":
            cmd += ["--gemma_bnb_4bit_quant_type", t.gemma_bnb_4bit_quant_type]
    if t.gemma_bnb_4bit_disable_double_quant:
        cmd.append("--gemma_bnb_4bit_disable_double_quant")
    if t.gemma_bnb_use_local_rank:
        cmd.append("--gemma_bnb_use_local_rank")
    if t.gemma_fp8_weight_offload:
        cmd.append("--gemma_fp8_weight_offload")
    else:
        cmd.append("--no-gemma_fp8_weight_offload")
    if t.ltx2_audio_only_model:
        cmd.append("--ltx2_audio_only_model")
    _append_ltx2_performance_args(cmd, t)

    # Quantization
    if t.nf4_base:
        cmd.append("--nf4_base")
        if t.nf4_block_size != 32:
            cmd += ["--nf4_block_size", str(t.nf4_block_size)]
    if t.loftq_init:
        cmd.append("--loftq_init")
        if t.loftq_iters != 2:
            cmd += ["--loftq_iters", str(t.loftq_iters)]
    if t.fp8_w8a8:
        cmd.append("--fp8_w8a8")
        if t.w8a8_mode != "int8":
            cmd += ["--w8a8_mode", t.w8a8_mode]
    if getattr(t, "w8a8_backend", None) and (
        (t.fp8_w8a8 and t.w8a8_mode == "int8")
        or getattr(t, "int8_base", False)
        or getattr(t, "int8_base_dynamic", False)
        or getattr(t, "int8_convrot_base", False)
        or getattr(t, "int8_convrot_dynamic", False)
    ):
        cmd += ["--w8a8_backend", t.w8a8_backend]
    if getattr(t, "int8_base", False):
        cmd.append("--int8_base")
    if getattr(t, "int8_base_dynamic", False):
        cmd.append("--int8_base_dynamic")
    if getattr(t, "int8_convrot_base", False):
        cmd.append("--int8_convrot_base")
    if getattr(t, "int8_convrot_dynamic", False):
        cmd.append("--int8_convrot_dynamic")
    if getattr(t, "int8_convrot_groupsize", "auto") not in (None, "", "auto"):
        cmd += ["--int8_convrot_groupsize", str(t.int8_convrot_groupsize)]
    if getattr(t, "int8_convrot_no_mse_clip", False):
        cmd.append("--int8_convrot_no_mse_clip")
    if getattr(t, "int8_convrot_quality_report", ""):
        cmd += ["--int8_convrot_quality_report", str(t.int8_convrot_quality_report)]
    if getattr(t, "w4a4g4", False):
        cmd.append("--w4a4g4")
        if str(getattr(t, "w4a4g4_container", "auto") or "auto") != "auto":
            cmd += ["--w4a4g4_container", str(t.w4a4g4_container)]
    if getattr(t, "w4a4g8", False):
        cmd.append("--w4a4g8")
    if getattr(t, "w4a8", False):
        cmd.append("--w4a8")
    # Stabilizer rank applies to the int4 a4-forward modes (w4a4g4 / w4a4g8), not w4a8.
    if (getattr(t, "w4a4g4", False) or getattr(t, "w4a4g8", False)) and int(getattr(t, "w4a4g4_stabilizer_rank", 0) or 0) > 0:
        cmd += ["--w4a4g4_stabilizer_rank", str(int(t.w4a4g4_stabilizer_rank))]
    if getattr(t, "int4_convrot_groupsize", "auto") not in (None, "", "auto"):
        cmd += ["--int4_convrot_groupsize", str(t.int4_convrot_groupsize)]
    if getattr(t, "int4_convrot_no_mse_clip", False):
        cmd.append("--int4_convrot_no_mse_clip")
    if getattr(t, "int4_convrot_quality_report", ""):
        cmd += ["--int4_convrot_quality_report", str(t.int4_convrot_quality_report)]
    if int(getattr(t, "int4_convrot_scale_refine_steps", 0) or 0) > 0:
        cmd += ["--int4_convrot_scale_refine_steps", str(int(t.int4_convrot_scale_refine_steps))]
    if int(getattr(t, "int4_convrot_group_scales", 0) or 0) > 0:
        cmd += ["--int4_convrot_group_scales", str(int(t.int4_convrot_group_scales))]
    if getattr(t, "int4_convrot_group_ratio_q8", False):
        cmd.append("--int4_convrot_group_ratio_q8")
    if str(getattr(t, "int4_convrot_compare_group_scales", "") or "").strip():
        cmd += ["--int4_convrot_compare_group_scales", str(t.int4_convrot_compare_group_scales).strip()]
    if getattr(t, "convrot_policy", ""):
        cmd += ["--convrot_policy", str(t.convrot_policy)]
    if getattr(t, "int8_fused_quant", False):
        cmd.append("--int8_fused_quant")
    if t.awq_calibration:
        cmd.append("--awq_calibration")
        if t.awq_alpha != 0.25:
            cmd += ["--awq_alpha", str(t.awq_alpha)]
        if t.awq_num_batches != 8:
            cmd += ["--awq_num_batches", str(t.awq_num_batches)]
    if t.quantize_device:
        cmd += ["--quantize_device", t.quantize_device]
    if getattr(t, "preserve_audio_timing", False):
        cmd.append("--preserve_audio_timing")

    # LoRA / Network
    cmd += ["--network_module", network_module]
    if t.network_dim is not None:
        cmd += ["--network_dim", str(t.network_dim)]
    if t.network_alpha != 1:
        cmd += ["--network_alpha", str(t.network_alpha)]
    if not _recipe_derives_preset or getattr(t, "lora_target_preset_manual", False):
        # A reference recipe normally derives the preset from its strategy, so the flag is omitted and the
        # engine auto-selects it. lora_target_preset_manual overrides that: emit the chosen preset
        # explicitly (the engine honors an explicit preset and still derives the strategy from the recipe).
        # Intrinsic-only / directional recipes (and no recipe) always emit the user's preset.
        cmd += ["--lora_target_preset", t.lora_target_preset]
    if t.train_connectors:
        cmd.append("--train_connectors")
    if network_args_parts:
        cmd += ["--network_args"] + network_args_parts
    if t.network_weights:
        cmd += ["--network_weights", t.network_weights]
    if t.network_freeze_surplus_modules:
        cmd.append("--network_freeze_surplus_modules")
    if t.network_dropout is not None:
        cmd += ["--network_dropout", str(t.network_dropout)]
    if t.scale_weight_norms is not None:
        cmd += ["--scale_weight_norms", str(t.scale_weight_norms)]
    if t.dim_from_weights:
        cmd.append("--dim_from_weights")
    if t.frozen_network_weights:
        cmd += ["--frozen_network_weights"] + _split_cli_args(t.frozen_network_weights)
    if t.frozen_network_multiplier:
        cmd += ["--frozen_network_multiplier"] + _split_cli_args(t.frozen_network_multiplier)
    if t.base_weights:
        cmd += ["--base_weights"] + _split_cli_args(t.base_weights)
    if t.base_weights_multiplier:
        cmd += ["--base_weights_multiplier"] + _split_cli_args(t.base_weights_multiplier)
    if t.lycoris_config:
        cmd += ["--lycoris_config", t.lycoris_config]
    if t.lycoris_quantized_base_check_mode != "warn":
        cmd += ["--lycoris_quantized_base_check_mode", t.lycoris_quantized_base_check_mode]
    if t.init_lokr_norm is not None:
        cmd += ["--init_lokr_norm", str(t.init_lokr_norm)]
    if t.caption_dropout_rate > 0:
        cmd += ["--caption_dropout_rate", str(t.caption_dropout_rate)]
    if t.video_caption_dropout_rate > 0:
        cmd += ["--video_caption_dropout_rate", str(t.video_caption_dropout_rate)]
    if t.audio_caption_dropout_rate > 0:
        cmd += ["--audio_caption_dropout_rate", str(t.audio_caption_dropout_rate)]
    if not t.save_original_lora:
        cmd.append("--no_save_original_lora")
    if not _cond_recipe and t.ic_lora_strategy != "auto":
        cmd += ["--ic_lora_strategy", t.ic_lora_strategy]
    if not _cond_recipe and getattr(t, "ic_lora_ref_probability", 1.0) != 1.0:
        cmd += ["--ic_lora_ref_probability", str(t.ic_lora_ref_probability)]
    if t.av_cross_attention_mode != "both":
        cmd += ["--av_cross_attention_mode", t.av_cross_attention_mode]
    if t.av_multi_ref:
        cmd.append("--av_multi_ref")
    if t.audio_ref_use_negative_positions:
        cmd.append("--audio_ref_use_negative_positions")
    if t.audio_ref_mask_cross_attention_to_reference:
        cmd.append("--audio_ref_mask_cross_attention_to_reference")
    if t.audio_ref_mask_reference_from_text_attention:
        cmd.append("--audio_ref_mask_reference_from_text_attention")
    if t.audio_ref_identity_guidance_scale != 0.0:
        cmd += ["--audio_ref_identity_guidance_scale", str(t.audio_ref_identity_guidance_scale)]
    if t.av_bimodal_cfg:
        cmd.append("--av_bimodal_cfg")
    if t.av_bimodal_scale != 3.0:
        cmd += ["--av_bimodal_scale", str(t.av_bimodal_scale)]

    # Optimizer
    cmd += ["--learning_rate", str(t.learning_rate)]
    if t.optimizer_type:
        cmd += ["--optimizer_type", t.optimizer_type]
    if t.optimizer_args:
        cmd += ["--optimizer_args"] + _split_cli_args(t.optimizer_args)
    cmd += ["--lr_scheduler", t.lr_scheduler]
    cmd += ["--lr_warmup_steps", str(t.lr_warmup_steps)]
    if t.lr_decay_steps is not None:
        cmd += ["--lr_decay_steps", str(t.lr_decay_steps)]
    if t.lr_scheduler_num_cycles is not None:
        cmd += ["--lr_scheduler_num_cycles", str(t.lr_scheduler_num_cycles)]
    if t.lr_scheduler_power is not None:
        cmd += ["--lr_scheduler_power", str(t.lr_scheduler_power)]
    if t.lr_scheduler_min_lr_ratio is not None:
        cmd += ["--lr_scheduler_min_lr_ratio", str(t.lr_scheduler_min_lr_ratio)]
    if t.lr_scheduler_type:
        cmd += ["--lr_scheduler_type", t.lr_scheduler_type]
    if t.lr_scheduler_args:
        cmd += ["--lr_scheduler_args"] + _split_cli_args(t.lr_scheduler_args)
    if t.lr_scheduler_timescale is not None:
        cmd += ["--lr_scheduler_timescale", str(t.lr_scheduler_timescale)]
    cmd += ["--gradient_accumulation_steps", str(t.gradient_accumulation_steps)]
    if t.accumulation_group_by != "none":
        cmd += ["--accumulation_group_by", t.accumulation_group_by]
        cmd += ["--accumulation_group_remainder", t.accumulation_group_remainder]
    cmd += ["--max_grad_norm", str(t.max_grad_norm)]
    _append_weight_noise_args(cmd, t)
    if t.audio_lr is not None:
        cmd += ["--audio_lr", str(t.audio_lr)]
    if t.lr_args:
        cmd += ["--lr_args"] + _split_cli_args(t.lr_args)
    if t.lr_group_warmup_args:
        cmd += ["--lr_group_warmup_args"] + _split_cli_args(t.lr_group_warmup_args)
    if t.lr_group_scheduler_args:
        cmd += ["--lr_group_scheduler_args"] + _split_cli_args(t.lr_group_scheduler_args)
    if t.audio_dim is not None:
        cmd += ["--audio_dim", str(t.audio_dim)]
    if t.audio_alpha is not None:
        cmd += ["--audio_alpha", str(t.audio_alpha)]
    for modality in ("video", "audio", "cross_modal"):
        for suffix in ("rank_dropout", "module_dropout", "max_grad_norm", "weight_decay"):
            name = f"{modality}_{suffix}"
            value = getattr(t, name, None)
            if value is not None:
                cmd += [f"--{name}", str(value)]

    # Schedule
    if t.max_train_epochs is not None:
        cmd += ["--max_train_epochs", str(t.max_train_epochs)]
    else:
        cmd += ["--max_train_steps", str(t.max_train_steps)]
    cmd += ["--timestep_sampling", _ltx2_timestep_sampling(t.timestep_sampling)]
    cmd += ["--discrete_flow_shift", str(t.discrete_flow_shift)]
    cmd += ["--weighting_scheme", t.weighting_scheme]
    if t.seed is not None:
        cmd += ["--seed", str(t.seed)]
    if t.guidance_scale is not None:
        cmd += ["--guidance_scale", str(t.guidance_scale)]
    if t.sigmoid_scale is not None:
        cmd += ["--sigmoid_scale", str(t.sigmoid_scale)]
    if t.logit_mean is not None:
        cmd += ["--logit_mean", str(t.logit_mean)]
    if t.logit_std is not None:
        cmd += ["--logit_std", str(t.logit_std)]
    if t.mode_scale is not None:
        cmd += ["--mode_scale", str(t.mode_scale)]
    if t.min_timestep is not None:
        cmd += ["--min_timestep", str(t.min_timestep)]
    if t.max_timestep is not None:
        cmd += ["--max_timestep", str(t.max_timestep)]

    # Advanced timestep
    if t.shifted_logit_mode:
        cmd += ["--shifted_logit_mode", t.shifted_logit_mode]
    if t.shifted_logit_eps != 1e-3:
        cmd += ["--shifted_logit_eps", str(t.shifted_logit_eps)]
    if t.shifted_logit_uniform_prob != 0.1:
        cmd += ["--shifted_logit_uniform_prob", str(t.shifted_logit_uniform_prob)]
    if t.shifted_logit_shift is not None:
        cmd += ["--shifted_logit_shift", str(t.shifted_logit_shift)]
    if t.shifted_logit_clamp_auto_shift:
        cmd.append("--shifted_logit_clamp_auto_shift")
    if t.shifted_logit_min_shift != 0.95:
        cmd += ["--shifted_logit_min_shift", str(t.shifted_logit_min_shift)]
    if t.shifted_logit_max_shift != 2.05:
        cmd += ["--shifted_logit_max_shift", str(t.shifted_logit_max_shift)]
    if t.preserve_distribution_shape:
        cmd.append("--preserve_distribution_shape")
    if t.num_timestep_buckets is not None:
        cmd += ["--num_timestep_buckets", str(t.num_timestep_buckets)]
    if t.show_timesteps:
        cmd += ["--show_timesteps", t.show_timesteps]
    if not t.log_timestep_distribution_tensorboard:
        cmd.append("--disable_timestep_distribution_tensorboard")
    elif t.log_timestep_distribution_tensorboard:
        cmd.append("--log_timestep_distribution_tensorboard")
    if t.log_timestep_distribution_interval != 100:
        cmd += ["--log_timestep_distribution_interval", str(t.log_timestep_distribution_interval)]

    # Memory
    if t.ltx2_model_parallel:
        cmd.append("--ltx2_model_parallel")
        if t.ltx2_model_parallel_devices:
            cmd += ["--ltx2_model_parallel_devices", t.ltx2_model_parallel_devices]
        if t.ltx2_model_parallel_splits:
            cmd += ["--ltx2_model_parallel_splits", t.ltx2_model_parallel_splits]
        if t.ltx2_mp_profile_transfers:
            cmd.append("--ltx2_mp_profile_transfers")
        if t.ltx2_mp_profile_log_every != 20:
            cmd += ["--ltx2_mp_profile_log_every", str(t.ltx2_mp_profile_log_every)]
        if t.ltx2_mp_activation_codec != "none":
            cmd += ["--ltx2_mp_activation_codec", t.ltx2_mp_activation_codec]
        if t.ltx2_mp_grad_codec != "none":
            cmd += ["--ltx2_mp_grad_codec", t.ltx2_mp_grad_codec]
        if t.ltx2_mp_int8_block_size != 256:
            cmd += ["--ltx2_mp_int8_block_size", str(t.ltx2_mp_int8_block_size)]
    if t.ltx2_remote_stage:
        cmd.append("--ltx2_remote_stage")
        if t.ltx2_remote_stage_specs:
            cmd += ["--ltx2_remote_stage_specs", t.ltx2_remote_stage_specs]
        else:
            cmd += ["--ltx2_remote_stage_host", t.ltx2_remote_stage_host]
            cmd += ["--ltx2_remote_stage_port", str(t.ltx2_remote_stage_port)]
            cmd += ["--ltx2_remote_stage_split", str(t.ltx2_remote_stage_split)]
        if t.ltx2_remote_stage_timeout != 600.0:
            cmd += ["--ltx2_remote_stage_timeout", str(t.ltx2_remote_stage_timeout)]
        if t.ltx2_remote_stage_codec != "none":
            cmd += ["--ltx2_remote_stage_codec", t.ltx2_remote_stage_codec]
        if t.ltx2_remote_stage_grad_codec != "none":
            cmd += ["--ltx2_remote_stage_grad_codec", t.ltx2_remote_stage_grad_codec]
        if t.ltx2_remote_stage_int8_block_size != 256:
            cmd += ["--ltx2_remote_stage_int8_block_size", str(t.ltx2_remote_stage_int8_block_size)]
        if not t.ltx2_remote_stage_metadata_cache:
            cmd.append("--no-ltx2_remote_stage_metadata_cache")
        if t.ltx2_remote_stage_metadata_cache_size != 8:
            cmd += ["--ltx2_remote_stage_metadata_cache_size", str(t.ltx2_remote_stage_metadata_cache_size)]
        if t.ltx2_remote_stage_aq_key_mode != "sample":
            cmd += ["--ltx2_remote_stage_aq_key_mode", t.ltx2_remote_stage_aq_key_mode]
        if not t.ltx2_remote_stage_aq_stochastic:
            cmd.append("--no-ltx2_remote_stage_aq_stochastic")
        if t.ltx2_remote_stage_aq_cache_size != 0:
            cmd += ["--ltx2_remote_stage_aq_cache_size", str(t.ltx2_remote_stage_aq_cache_size)]
        if t.ltx2_remote_stage_trainable:
            cmd.append("--ltx2_remote_stage_trainable")
            if t.ltx2_remote_stage_trainable_scope != "auto":
                cmd += ["--ltx2_remote_stage_trainable_scope", t.ltx2_remote_stage_trainable_scope]
            if t.ltx2_remote_stage_learning_rate is not None:
                cmd += ["--ltx2_remote_stage_learning_rate", str(t.ltx2_remote_stage_learning_rate)]
            if t.ltx2_remote_stage_weight_decay != 0.01:
                cmd += ["--ltx2_remote_stage_weight_decay", str(t.ltx2_remote_stage_weight_decay)]
            if t.ltx2_remote_stage_max_grad_norm != 0.0:
                cmd += ["--ltx2_remote_stage_max_grad_norm", str(t.ltx2_remote_stage_max_grad_norm)]
            if t.ltx2_remote_stage_checkpoint_dir:
                cmd += ["--ltx2_remote_stage_checkpoint_dir", t.ltx2_remote_stage_checkpoint_dir]
        if t.ltx2_remote_stage_prune_local_blocks:
            cmd.append("--ltx2_remote_stage_prune_local_blocks")
    if t.blocks_to_swap is not None:
        cmd += ["--blocks_to_swap", str(t.blocks_to_swap)]
    if t.gradient_checkpointing or t.blockwise_checkpointing:
        cmd.append("--gradient_checkpointing")
    if t.gradient_checkpointing_cpu_offload:
        cmd.append("--gradient_checkpointing_cpu_offload")
    if t.split_attn:
        cmd.append("--split_attn")
    if t.split_attn_target:
        cmd += ["--split_attn_target", t.split_attn_target]
    if t.split_attn_mode:
        cmd += ["--split_attn_mode", t.split_attn_mode]
    if t.split_attn_chunk_size is not None:
        cmd += ["--split_attn_chunk_size", str(t.split_attn_chunk_size)]
    if t.blockwise_checkpointing:
        cmd.append("--blockwise_checkpointing")
    if t.blocks_to_checkpoint is not None:
        cmd += ["--blocks_to_checkpoint", str(t.blocks_to_checkpoint)]
    if t.full_fp16:
        cmd.append("--full_fp16")
    if t.full_bf16:
        cmd.append("--full_bf16")
    if t.ffn_chunk_target:
        cmd += ["--ffn_chunk_target", t.ffn_chunk_target]
    if t.ffn_chunk_size:
        cmd += ["--ffn_chunk_size", str(t.ffn_chunk_size)]
    if t.use_pinned_memory_for_block_swap:
        cmd.append("--use_pinned_memory_for_block_swap")
    _append_h2d_block_swap_args(cmd, t)
    if t.img_in_txt_in_offloading:
        cmd.append("--img_in_txt_in_offloading")

    # Compile
    _append_compile_args(cmd, t)

    # CUDA
    _append_cuda_args(cmd, t)
    if t.disable_numpy_memmap:
        cmd.append("--disable_numpy_memmap")
    if t.ddp_timeout is not None:
        cmd += ["--ddp_timeout", str(t.ddp_timeout)]
    if t.ddp_gradient_as_bucket_view:
        cmd.append("--ddp_gradient_as_bucket_view")
    if t.ddp_static_graph:
        cmd.append("--ddp_static_graph")
    if t.ddp_find_unused_parameters:
        cmd.append("--ddp_find_unused_parameters")

    # Sampling
    if t.sample_every_n_steps:
        cmd += ["--sample_every_n_steps", str(t.sample_every_n_steps)]
    if t.sample_every_n_epochs:
        cmd += ["--sample_every_n_epochs", str(t.sample_every_n_epochs)]
    if sample_prompts:
        cmd += ["--sample_prompts", sample_prompts]
    if t.precache_sample_prompts:
        cmd.append("--precache_sample_prompts")
    if t.use_precached_sample_prompts:
        cmd.append("--use_precached_sample_prompts")
    if t.sample_prompts_cache:
        cmd += ["--sample_prompts_cache", t.sample_prompts_cache]
    if t.caption_field:
        cmd += ["--caption_field", t.caption_field]
    if t.use_precached_sample_latents:
        cmd.append("--use_precached_sample_latents")
    if t.sample_latents_cache:
        cmd += ["--sample_latents_cache", t.sample_latents_cache]
    cmd += ["--sample_sampling_preset", t.sample_sampling_preset]
    if t.sample_sigma_schedule != "auto":
        cmd += ["--sample_sigma_schedule", t.sample_sigma_schedule]
    if t.sample_sampler != "auto":
        cmd += ["--sample_sampler", t.sample_sampler]
    if t.sample_use_default_negative_prompt is True:
        cmd.append("--sample_use_default_negative_prompt")
    elif t.sample_use_default_negative_prompt is False:
        cmd.append("--no-sample_use_default_negative_prompt")
    _append_optional(cmd, "--height", t.height)
    _append_optional(cmd, "--width", t.width)
    _append_optional(cmd, "--sample_num_frames", t.sample_num_frames)
    _append_optional(cmd, "--video_cfg_scale", t.video_cfg_scale)
    _append_optional(cmd, "--audio_cfg_scale", t.audio_cfg_scale)
    _append_optional(cmd, "--video_modality_scale", t.video_modality_scale)
    _append_optional(cmd, "--audio_modality_scale", t.audio_modality_scale)
    _append_optional(cmd, "--stg_scale", t.stg_scale)
    if t.stg_blocks:
        cmd += ["--stg_blocks"] + _split_cli_args(t.stg_blocks)
    if t.stg_mode:
        cmd += ["--stg_mode", t.stg_mode]
    _append_optional(cmd, "--rescale_scale", t.rescale_scale)
    _append_optional(cmd, "--video_rescale_scale", t.video_rescale_scale)
    _append_optional(cmd, "--audio_rescale_scale", t.audio_rescale_scale)
    if t.sample_with_offloading:
        cmd.append("--sample_with_offloading")
    if t.sample_merge_audio:
        cmd.append("--sample_merge_audio")
    if t.sample_disable_audio:
        cmd.append("--sample_disable_audio")
    if t.sample_at_first:
        cmd.append("--sample_at_first")
    if t.sample_tiled_vae:
        cmd.append("--sample_tiled_vae")
    if t.sample_vae_tile_size is not None:
        cmd += ["--sample_vae_tile_size", str(t.sample_vae_tile_size)]
    if t.sample_vae_tile_overlap is not None:
        cmd += ["--sample_vae_tile_overlap", str(t.sample_vae_tile_overlap)]
    if t.sample_vae_temporal_tile_size is not None:
        cmd += ["--sample_vae_temporal_tile_size", str(t.sample_vae_temporal_tile_size)]
    if t.sample_vae_temporal_tile_overlap is not None:
        cmd += ["--sample_vae_temporal_tile_overlap", str(t.sample_vae_temporal_tile_overlap)]
    if t.sample_two_stage:
        cmd.append("--sample_two_stage")
        if t.spatial_upsampler_path:
            cmd += ["--spatial_upsampler_path", t.spatial_upsampler_path]
        if getattr(t, "temporal_upsampler_path", ""):
            cmd += ["--temporal_upsampler_path", t.temporal_upsampler_path]
        if t.distilled_lora_path:
            cmd += ["--distilled_lora_path", t.distilled_lora_path]
        if t.sample_stage2_steps != 3:
            cmd += ["--sample_stage2_steps", str(t.sample_stage2_steps)]
        if t.sample_stage1_distilled_lora_multiplier is not None:
            cmd += [
                "--sample_stage1_distilled_lora_multiplier",
                str(t.sample_stage1_distilled_lora_multiplier),
            ]
        if t.sample_stage2_distilled_lora_multiplier is not None:
            cmd += [
                "--sample_stage2_distilled_lora_multiplier",
                str(t.sample_stage2_distilled_lora_multiplier),
            ]
    if t.sample_audio_only:
        cmd.append("--sample_audio_only")
    if t.sample_disable_flash_attn:
        cmd.append("--sample_disable_flash_attn")
    if not t.sample_i2v_token_timestep_mask:
        cmd.append("--no-sample_i2v_token_timestep_mask")
    if not t.sample_audio_subprocess:
        cmd.append("--no-sample_audio_subprocess")
    if t.sample_include_reference:
        cmd.append("--sample_include_reference")
    if t.reference_downscale != 1:
        cmd += ["--reference_downscale", str(t.reference_downscale)]
    if getattr(t, "reference_temporal_scale", 1) != 1:
        cmd += ["--reference_temporal_scale", str(t.reference_temporal_scale)]
    if t.reference_frames != 1:
        cmd += ["--reference_frames", str(t.reference_frames)]

    # Validation
    if t.validate_every_n_steps is not None:
        cmd += ["--validate_every_n_steps", str(t.validate_every_n_steps)]
    if t.validate_every_n_epochs is not None:
        cmd += ["--validate_every_n_epochs", str(t.validate_every_n_epochs)]
    if t.offload_optimizer_during_validation:
        cmd.append("--offload_optimizer_during_validation")

    # Output
    cmd += ["--output_dir", _effective_output_dir(t.output_dir)]
    if t.output_name:
        cmd += ["--output_name", t.output_name]
    if t.save_every_n_epochs:
        cmd += ["--save_every_n_epochs", str(t.save_every_n_epochs)]
    if t.save_every_n_steps:
        cmd += ["--save_every_n_steps", str(t.save_every_n_steps)]
    if t.save_last_n_epochs is not None:
        cmd += ["--save_last_n_epochs", str(t.save_last_n_epochs)]
    if t.save_last_n_steps is not None:
        cmd += ["--save_last_n_steps", str(t.save_last_n_steps)]
    if t.save_last_n_epochs_state is not None:
        cmd += ["--save_last_n_epochs_state", str(t.save_last_n_epochs_state)]
    if t.save_last_n_steps_state is not None:
        cmd += ["--save_last_n_steps_state", str(t.save_last_n_steps_state)]
    if t.save_state:
        cmd.append("--save_state")
    if t.save_state_on_train_end:
        cmd.append("--save_state_on_train_end")
    if (t.save_state or t.save_state_on_train_end) and t.save_state_mode != "full":
        cmd += ["--save_state_mode", t.save_state_mode]
    if t.save_checkpoint_metadata:
        cmd.append("--save_checkpoint_metadata")
    if t.no_metadata:
        cmd.append("--no_metadata")
    if t.no_convert_to_comfy:
        cmd.append("--no_convert_to_comfy")
    if t.log_with:
        cmd += ["--log_with", t.log_with]
    if t.log_with and t.logging_dir:
        cmd += ["--logging_dir", t.logging_dir]
    if t.log_prefix:
        cmd += ["--log_prefix", t.log_prefix]
    if t.log_tracker_name:
        cmd += ["--log_tracker_name", t.log_tracker_name]
    if t.log_tracker_config:
        cmd += ["--log_tracker_config", t.log_tracker_config]
    if t.log_config:
        cmd.append("--log_config")
    if t.wandb_run_name:
        cmd += ["--wandb_run_name", t.wandb_run_name]
    if t.wandb_api_key:
        cmd += ["--wandb_api_key", t.wandb_api_key]
    if t.log_cuda_memory_every_n_steps is not None:
        cmd += ["--log_cuda_memory_every_n_steps", str(t.log_cuda_memory_every_n_steps)]
    if t.resume:
        cmd += ["--resume", t.resume]
    if t.autoresume:
        cmd.append("--autoresume")
    if t.reset_optimizer:
        cmd.append("--reset_optimizer")
    if t.reset_optimizer_params:
        cmd.append("--reset_optimizer_params")
    if t.reset_dataloader:
        cmd.append("--reset_dataloader")
    if t.training_comment:
        cmd += ["--training_comment", t.training_comment]
    if t.loss_type != "mse":
        cmd += ["--loss_type", t.loss_type]
    if t.loss_type in ("huber", "smooth_l1") and t.huber_delta != 1.0:
        cmd += ["--huber_delta", str(t.huber_delta)]

    # Metadata
    if t.metadata_title:
        cmd += ["--metadata_title", t.metadata_title]
    if t.metadata_author:
        cmd += ["--metadata_author", t.metadata_author]
    if t.metadata_description:
        cmd += ["--metadata_description", t.metadata_description]
    if t.metadata_license:
        cmd += ["--metadata_license", t.metadata_license]
    if t.metadata_tags:
        cmd += ["--metadata_tags", t.metadata_tags]
    if t.metadata_reso:
        cmd += ["--metadata_reso", t.metadata_reso]
    if t.metadata_arch:
        cmd += ["--metadata_arch", t.metadata_arch]

    # HuggingFace upload
    if t.huggingface_repo_id:
        cmd += ["--huggingface_repo_id", t.huggingface_repo_id]
    if t.huggingface_repo_type:
        cmd += ["--huggingface_repo_type", t.huggingface_repo_type]
    if t.huggingface_path_in_repo:
        cmd += ["--huggingface_path_in_repo", t.huggingface_path_in_repo]
    if t.huggingface_token:
        cmd += ["--huggingface_token", t.huggingface_token]
    if t.huggingface_repo_visibility:
        cmd += ["--huggingface_repo_visibility", t.huggingface_repo_visibility]
    if t.save_state_to_huggingface:
        cmd.append("--save_state_to_huggingface")
    if t.resume_from_huggingface:
        cmd.append("--resume_from_huggingface")
    if t.async_upload:
        cmd.append("--async_upload")

    # CREPA
    if t.crepa:
        cmd.append("--crepa")
        args_parts = []
        _append_key_value_args(args_parts, t.crepa_args)
        if t.crepa_mode != "backbone":
            args_parts.append(f"mode={t.crepa_mode}")
        if t.crepa_student_block_idx != 16:
            args_parts.append(f"student_block_idx={t.crepa_student_block_idx}")
        if t.crepa_mode == "backbone" and t.crepa_teacher_block_idx != 32:
            args_parts.append(f"teacher_block_idx={t.crepa_teacher_block_idx}")
        if t.crepa_mode == "dino" and t.crepa_dino_model != "dinov2_vitb14":
            args_parts.append(f"dino_model={t.crepa_dino_model}")
        if t.crepa_lambda != 0.1:
            args_parts.append(f"lambda_crepa={t.crepa_lambda}")
        if t.crepa_tau != 1.0:
            args_parts.append(f"tau={t.crepa_tau}")
        if t.crepa_num_neighbors != 2:
            args_parts.append(f"num_neighbors={t.crepa_num_neighbors}")
        if t.crepa_schedule != "constant":
            args_parts.append(f"schedule={t.crepa_schedule}")
        if t.crepa_warmup_steps != 0:
            args_parts.append(f"warmup_steps={t.crepa_warmup_steps}")
        if not t.crepa_normalize:
            args_parts.append("normalize=false")
        if t.crepa_cutoff_step != 0:
            args_parts.append(f"cutoff_step={t.crepa_cutoff_step}")
        if t.crepa_similarity_threshold is not None:
            args_parts.append(f"similarity_threshold={t.crepa_similarity_threshold}")
            if t.crepa_similarity_ema_decay != 0.99:
                args_parts.append(f"similarity_ema_decay={t.crepa_similarity_ema_decay}")
            if t.crepa_threshold_mode != "permanent":
                args_parts.append(f"threshold_mode={t.crepa_threshold_mode}")
        if args_parts:
            cmd += ["--crepa_args"] + args_parts

    # Self-Flow
    if t.self_flow:
        cmd.append("--self_flow")
        args_parts = []
        _append_key_value_args(args_parts, t.self_flow_args)
        if t.self_flow_teacher_mode != "ema":
            args_parts.append(f"teacher_mode={t.self_flow_teacher_mode}")
        if t.self_flow_student_block_idx != 16:
            args_parts.append(f"student_block_idx={t.self_flow_student_block_idx}")
        if t.self_flow_teacher_block_idx != 32:
            args_parts.append(f"teacher_block_idx={t.self_flow_teacher_block_idx}")
        if t.self_flow_student_block_ratio is not None:
            args_parts.append(f"student_block_ratio={t.self_flow_student_block_ratio}")
        if t.self_flow_teacher_block_ratio is not None:
            args_parts.append(f"teacher_block_ratio={t.self_flow_teacher_block_ratio}")
        if t.self_flow_student_block_stochastic_range != 0:
            args_parts.append(f"student_block_stochastic_range={t.self_flow_student_block_stochastic_range}")
        if t.self_flow_lambda != 0.1:
            args_parts.append(f"lambda_self_flow={t.self_flow_lambda}")
        if t.self_flow_lambda_audio != 0.0:
            args_parts.append(f"lambda_audio={t.self_flow_lambda_audio}")
        if t.self_flow_mask_ratio != 0.1:
            args_parts.append(f"mask_ratio={t.self_flow_mask_ratio}")
        if t.self_flow_frame_level_mask:
            args_parts.append("frame_level_mask=true")
        if t.self_flow_mask_focus_loss:
            args_parts.append("mask_focus_loss=true")
        if t.self_flow_max_loss != 0.0:
            args_parts.append(f"max_loss={t.self_flow_max_loss}")
        if t.self_flow_teacher_momentum != 0.999:
            args_parts.append(f"teacher_momentum={t.self_flow_teacher_momentum}")
        if not t.self_flow_dual_timestep:
            args_parts.append("dual_timestep=false")
        if t.self_flow_projector_lr is not None:
            args_parts.append(f"projector_lr={t.self_flow_projector_lr}")
        if t.self_flow_projector_activation != "silu":
            args_parts.append(f"projector_activation={t.self_flow_projector_activation}")
        if getattr(t, "self_flow_temporal_mode", "off") != "off":
            args_parts.append(f"temporal_mode={t.self_flow_temporal_mode}")
        if getattr(t, "self_flow_lambda_temporal", 0.0) != 0.0:
            args_parts.append(f"lambda_temporal={t.self_flow_lambda_temporal}")
        if getattr(t, "self_flow_lambda_delta", 0.0) != 0.0:
            args_parts.append(f"lambda_delta={t.self_flow_lambda_delta}")
        if getattr(t, "self_flow_temporal_tau", 1.0) != 1.0:
            args_parts.append(f"temporal_tau={t.self_flow_temporal_tau}")
        if getattr(t, "self_flow_num_neighbors", 2) != 2:
            args_parts.append(f"num_neighbors={t.self_flow_num_neighbors}")
        if getattr(t, "self_flow_temporal_granularity", "frame") != "frame":
            args_parts.append(f"temporal_granularity={t.self_flow_temporal_granularity}")
        if getattr(t, "self_flow_patch_spatial_radius", 0) != 0:
            args_parts.append(f"patch_spatial_radius={t.self_flow_patch_spatial_radius}")
        if getattr(t, "self_flow_patch_match_mode", "hard") != "hard":
            args_parts.append(f"patch_match_mode={t.self_flow_patch_match_mode}")
        if getattr(t, "self_flow_delta_num_steps", 1) != 1:
            args_parts.append(f"delta_num_steps={t.self_flow_delta_num_steps}")
        if getattr(t, "self_flow_motion_weighting", "none") != "none":
            args_parts.append(f"motion_weighting={t.self_flow_motion_weighting}")
        if getattr(t, "self_flow_motion_weight_strength", 0.0) != 0.0:
            args_parts.append(f"motion_weight_strength={t.self_flow_motion_weight_strength}")
        if getattr(t, "self_flow_temporal_schedule", "constant") != "constant":
            args_parts.append(f"temporal_schedule={t.self_flow_temporal_schedule}")
        if getattr(t, "self_flow_temporal_warmup_steps", 0) != 0:
            args_parts.append(f"temporal_warmup_steps={t.self_flow_temporal_warmup_steps}")
        if getattr(t, "self_flow_temporal_max_steps", 0) != 0:
            args_parts.append(f"temporal_max_steps={t.self_flow_temporal_max_steps}")
        if getattr(t, "self_flow_schedule_end_weight", 0.0) != 0.0:
            args_parts.append(f"schedule_end_weight={t.self_flow_schedule_end_weight}")
        if getattr(t, "self_flow_schedule_power", 1.0) != 1.0:
            args_parts.append(f"schedule_power={t.self_flow_schedule_power}")
        if getattr(t, "self_flow_schedule_cutoff_step", 0) != 0:
            args_parts.append(f"schedule_cutoff_step={t.self_flow_schedule_cutoff_step}")
        if getattr(t, "self_flow_similarity_cutoff", None) is not None:
            args_parts.append(f"similarity_cutoff={t.self_flow_similarity_cutoff}")
        if getattr(t, "self_flow_similarity_ema_decay", 0.99) != 0.99:
            args_parts.append(f"similarity_ema_decay={t.self_flow_similarity_ema_decay}")
        if getattr(t, "self_flow_similarity_cutoff_mode", "permanent") != "permanent":
            args_parts.append(f"similarity_cutoff_mode={t.self_flow_similarity_cutoff_mode}")
        if getattr(t, "self_flow_offload_teacher_features", False):
            args_parts.append("offload_teacher_features=true")
        if args_parts:
            cmd += ["--self_flow_args"] + args_parts

    # HFATO (ViBe)
    if t.hfato:
        cmd.append("--hfato")
        args_parts = []
        _append_key_value_args(args_parts, t.hfato_args)
        if t.hfato_scale_factor != 0.5:
            args_parts.append(f"scale_factor={t.hfato_scale_factor}")
        if t.hfato_interpolation != "trilinear":
            args_parts.append(f"interpolation={t.hfato_interpolation}")
        if t.hfato_probability != 1.0:
            args_parts.append(f"probability={t.hfato_probability}")
        if args_parts:
            cmd += ["--hfato_args"] + args_parts

    # Latent temporal objectives
    if t.latent_temporal_weighting:
        cmd.append("--latent_temporal_weighting")
        args_parts = []
        _append_key_value_args(args_parts, t.latent_temporal_weighting_args)
        if t.latent_temporal_weighting_alpha != 0.5:
            args_parts.append(f"alpha={t.latent_temporal_weighting_alpha}")
        if t.latent_temporal_weighting_mode != "log":
            args_parts.append(f"mode={t.latent_temporal_weighting_mode}")
        if t.latent_temporal_weighting_normalize != "mean":
            args_parts.append(f"normalize={t.latent_temporal_weighting_normalize}")
        if t.latent_temporal_weighting_clip_min != 0.5:
            args_parts.append(f"clip_min={t.latent_temporal_weighting_clip_min}")
        if t.latent_temporal_weighting_clip_max != 2.0:
            args_parts.append(f"clip_max={t.latent_temporal_weighting_clip_max}")
        if args_parts:
            cmd += ["--latent_temporal_weighting_args"] + args_parts

    if t.latent_delta_loss:
        cmd.append("--latent_delta_loss")
        args_parts = []
        _append_key_value_args(args_parts, t.latent_delta_loss_args)
        if t.latent_delta_loss_weight != 0.03:
            args_parts.append(f"weight={t.latent_delta_loss_weight}")
        if t.latent_delta_loss_order != "1":
            args_parts.append(f"order={t.latent_delta_loss_order}")
        if t.latent_delta_loss_target != "x0":
            args_parts.append(f"target={t.latent_delta_loss_target}")
        if t.latent_delta_loss_sigma_min != 0.0:
            args_parts.append(f"sigma_min={t.latent_delta_loss_sigma_min}")
        if t.latent_delta_loss_sigma_max != 1.0:
            args_parts.append(f"sigma_max={t.latent_delta_loss_sigma_max}")
        if t.latent_delta_loss_second_order_weight != 0.5:
            args_parts.append(f"second_order_weight={t.latent_delta_loss_second_order_weight}")
        if t.latent_delta_loss_type != "mse":
            args_parts.append(f"loss_type={t.latent_delta_loss_type}")
        if t.latent_delta_loss_huber_delta != 1.0:
            args_parts.append(f"huber_delta={t.latent_delta_loss_huber_delta}")
        if args_parts:
            cmd += ["--latent_delta_loss_args"] + args_parts

    # Preservation
    if t.blank_preservation:
        cmd.append("--blank_preservation")
        args_parts = []
        _append_key_value_args(args_parts, t.blank_preservation_args)
        if t.blank_preservation_multiplier != 1.0:
            args_parts.append(f"multiplier={t.blank_preservation_multiplier}")
        if args_parts:
            cmd += ["--blank_preservation_args"] + args_parts
    if t.dop:
        cmd.append("--dop")
        args_parts = []
        _append_key_value_args(args_parts, t.dop_args)
        if t.dop_class:
            args_parts.append(f"class={t.dop_class}")
        if t.dop_multiplier != 1.0:
            args_parts.append(f"multiplier={t.dop_multiplier}")
        if t.dop_mode != "fixed":
            args_parts.append(f"mode={t.dop_mode}")
        if t.dop_trigger:
            args_parts.append(f"trigger={t.dop_trigger}")
        if t.dop_replacements:
            args_parts.append(f"replace={t.dop_replacements}")
        if t.dop_prompt_bank:
            args_parts.append(f"bank={t.dop_prompt_bank}")
        if t.dop_loss_type != "mse":
            args_parts.append(f"loss={t.dop_loss_type}")
        if t.dop_huber_delta != 1.0:
            args_parts.append(f"huber_delta={t.dop_huber_delta}")
        if t.dop_relative_eps != 1e-6:
            args_parts.append(f"relative_eps={t.dop_relative_eps}")
        if t.dop_temporal_weight != 0.0:
            args_parts.append(f"temporal_weight={t.dop_temporal_weight}")
        if t.dop_inside_weight != 1.0:
            args_parts.append(f"inside_weight={t.dop_inside_weight}")
        if t.dop_outside_weight != 1.0:
            args_parts.append(f"outside_weight={t.dop_outside_weight}")
        if t.dop_timestep_bins != 0:
            args_parts.append(f"timestep_bins={t.dop_timestep_bins}")
        if t.dop_timestep_min != 0.0:
            args_parts.append(f"timestep_min={t.dop_timestep_min}")
        if t.dop_timestep_max != 1.0:
            args_parts.append(f"timestep_max={t.dop_timestep_max}")
        if t.dop_timestep_weight != "none":
            args_parts.append(f"timestep_weight={t.dop_timestep_weight}")
        if t.dop_adaptive_target != 0.0:
            args_parts.append(f"adaptive_target={t.dop_adaptive_target}")
        if t.dop_adaptive_ema != 0.95:
            args_parts.append(f"adaptive_ema={t.dop_adaptive_ema}")
        if t.dop_adaptive_rate != 0.1:
            args_parts.append(f"adaptive_rate={t.dop_adaptive_rate}")
        if t.dop_adaptive_min != 0.0:
            args_parts.append(f"adaptive_min={t.dop_adaptive_min}")
        if t.dop_adaptive_max != 100.0:
            args_parts.append(f"adaptive_max={t.dop_adaptive_max}")
        if t.dop_adaptive_warmup != 0:
            args_parts.append(f"adaptive_warmup={t.dop_adaptive_warmup}")
        if t.dop_microbatch != 0:
            args_parts.append(f"microbatch={t.dop_microbatch}")
        if t.dop_anchor_size != 0:
            args_parts.append(f"anchor_size={t.dop_anchor_size}")
        if t.dop_anchor_interval != 0:
            args_parts.append(f"anchor_interval={t.dop_anchor_interval}")
        if t.dop_anchor_weight != 1.0:
            args_parts.append(f"anchor_weight={t.dop_anchor_weight}")
        if t.dop_anchor_path:
            args_parts.append(f"anchor_path={t.dop_anchor_path}")
        if not t.dop_strict_cache:
            args_parts.append("strict_cache=false")
        if args_parts:
            cmd += ["--dop_args"] + args_parts
    if t.prior_divergence:
        cmd.append("--prior_divergence")
        args_parts = []
        _append_key_value_args(args_parts, t.prior_divergence_args)
        if t.prior_divergence_multiplier != 0.1:
            args_parts.append(f"multiplier={t.prior_divergence_multiplier}")
        if args_parts:
            cmd += ["--prior_divergence_args"] + args_parts
    if t.use_precached_preservation:
        cmd.append("--use_precached_preservation")
    if t.preservation_prompts_cache:
        cmd += ["--preservation_prompts_cache", t.preservation_prompts_cache]

    # TARP / DCR
    if t.tarp:
        cmd.append("--tarp")
        args_parts = []
        _append_key_value_args(args_parts, t.tarp_args)
        if t.tarp_window_multiplier != 3:
            args_parts.append(f"window_multiplier={t.tarp_window_multiplier}")
        if args_parts:
            cmd += ["--tarp_args"] + args_parts
    if t.dcr:
        cmd.append("--dcr")
        args_parts = []
        _append_key_value_args(args_parts, t.dcr_args)
        if not t.dcr_reference_detach:
            args_parts.append("reference_detach=false")
        if args_parts:
            cmd += ["--dcr_args"] + args_parts
    if t.av_cross_grad_surgery:
        cmd.append("--av_cross_grad_surgery")
        args_parts = []
        _append_key_value_args(args_parts, t.av_cross_grad_surgery_args)
        if args_parts:
            cmd += ["--av_cross_grad_surgery_args"] + args_parts
    if t.av_attention_loss_weighting:
        cmd.append("--av_attention_loss_weighting")
        if t.av_attention_loss_max != 1.5:
            cmd += ["--av_attention_loss_max", str(t.av_attention_loss_max)]
        if t.av_attention_loss_warmup_steps != 400:
            cmd += ["--av_attention_loss_warmup_steps", str(t.av_attention_loss_warmup_steps)]

    # Audio Metrics
    _append_audio_metrics_args(cmd, t)

    # TREAD token routing
    if t.tread:
        cmd.append("--tread")
        args_parts = []
        _append_key_value_args(args_parts, t.tread_args)
        if t.tread_target != "video":
            args_parts.append(f"target={t.tread_target}")
        if t.tread_selection_ratio != 0.5:
            args_parts.append(f"selection_ratio={t.tread_selection_ratio}")
        if t.tread_start_layer_idx is not None:
            args_parts.append(f"start_layer_idx={t.tread_start_layer_idx}")
        if t.tread_end_layer_idx is not None:
            args_parts.append(f"end_layer_idx={t.tread_end_layer_idx}")
        if args_parts:
            cmd += ["--tread_args"] + args_parts

    # Differential guidance
    if t.differential_guidance:
        cmd.append("--differential_guidance")
        if t.differential_guidance_scale != 3.0:
            cmd += ["--differential_guidance_scale", str(t.differential_guidance_scale)]
        if t.differential_guidance_schedule != "constant":
            cmd += ["--differential_guidance_schedule", t.differential_guidance_schedule]
        if t.differential_guidance_start_scale != 1.0:
            cmd += ["--differential_guidance_start_scale", str(t.differential_guidance_start_scale)]
        if t.differential_guidance_end_scale != 1.0:
            cmd += ["--differential_guidance_end_scale", str(t.differential_guidance_end_scale)]
        if t.differential_guidance_warmup_steps != 0:
            cmd += ["--differential_guidance_warmup_steps", str(t.differential_guidance_warmup_steps)]
        if t.differential_guidance_hold_steps != 0:
            cmd += ["--differential_guidance_hold_steps", str(t.differential_guidance_hold_steps)]
        if t.differential_guidance_decay_steps != 0:
            cmd += ["--differential_guidance_decay_steps", str(t.differential_guidance_decay_steps)]
        if t.differential_guidance_timestep_mode != "none":
            cmd += ["--differential_guidance_timestep_mode", t.differential_guidance_timestep_mode]
        if t.differential_guidance_timestep_floor != 1.0:
            cmd += ["--differential_guidance_timestep_floor", str(t.differential_guidance_timestep_floor)]
        if t.differential_guidance_normalize_residual:
            cmd.append("--differential_guidance_normalize_residual")
        if t.differential_guidance_residual_clip != 0.0:
            cmd += ["--differential_guidance_residual_clip", str(t.differential_guidance_residual_clip)]
        if t.differential_guidance_adaptive_target_norm != 0.0:
            cmd += [
                "--differential_guidance_adaptive_target_norm",
                str(t.differential_guidance_adaptive_target_norm),
            ]
        if t.differential_guidance_adaptive_target_ratio != 0.0:
            cmd += [
                "--differential_guidance_adaptive_target_ratio",
                str(t.differential_guidance_adaptive_target_ratio),
            ]
        if t.differential_guidance_adaptive_ema != 0.95:
            cmd += ["--differential_guidance_adaptive_ema", str(t.differential_guidance_adaptive_ema)]
        if t.differential_guidance_adaptive_rate != 0.1:
            cmd += ["--differential_guidance_adaptive_rate", str(t.differential_guidance_adaptive_rate)]
        if t.differential_guidance_adaptive_min != 0.25:
            cmd += ["--differential_guidance_adaptive_min", str(t.differential_guidance_adaptive_min)]
        if t.differential_guidance_adaptive_max != 4.0:
            cmd += ["--differential_guidance_adaptive_max", str(t.differential_guidance_adaptive_max)]

    # Audio features
    if t.av_curriculum_mode != "none":
        cmd += ["--av_curriculum_mode", t.av_curriculum_mode]
        if t.av_curriculum_mode == "alternating":
            cmd += ["--av_curriculum_interval_steps", str(t.av_curriculum_interval_steps)]
            if t.av_curriculum_start_modality != "video":
                cmd += ["--av_curriculum_start_modality", t.av_curriculum_start_modality]
        elif t.av_curriculum_mode == "two_stage":
            cmd += ["--av_curriculum_stage1_steps", str(t.av_curriculum_stage1_steps)]
            if t.av_curriculum_stage1_policy != "video":
                cmd += ["--av_curriculum_stage1_policy", t.av_curriculum_stage1_policy]
            if t.av_curriculum_stage2_policy != "joint":
                cmd += ["--av_curriculum_stage2_policy", t.av_curriculum_stage2_policy]
    if t.audio_loss_balance_mode != "none":
        cmd += ["--audio_loss_balance_mode", t.audio_loss_balance_mode]
        if t.audio_loss_balance_mode == "inv_freq":
            if t.audio_loss_balance_beta != 0.01:
                cmd += ["--audio_loss_balance_beta", str(t.audio_loss_balance_beta)]
            if t.audio_loss_balance_eps != 0.05:
                cmd += ["--audio_loss_balance_eps", str(t.audio_loss_balance_eps)]
            if t.audio_loss_balance_min != 0.05:
                cmd += ["--audio_loss_balance_min", str(t.audio_loss_balance_min)]
            if t.audio_loss_balance_max != 4.0:
                cmd += ["--audio_loss_balance_max", str(t.audio_loss_balance_max)]
        if t.audio_loss_balance_ema_init != 1.0:
            cmd += ["--audio_loss_balance_ema_init", str(t.audio_loss_balance_ema_init)]
        if t.audio_loss_balance_mode == "ema_mag":
            if t.audio_loss_balance_target_ratio != 0.33:
                cmd += ["--audio_loss_balance_target_ratio", str(t.audio_loss_balance_target_ratio)]
            if t.audio_loss_balance_ema_decay != 0.99:
                cmd += ["--audio_loss_balance_ema_decay", str(t.audio_loss_balance_ema_decay)]
        if t.audio_loss_balance_mode == "uncertainty" and t.uncertainty_lr is not None:
            cmd += ["--uncertainty_lr", str(t.uncertainty_lr)]
        if t.audio_loss_balance_mode == "ogm_ge":
            if t.ogm_ge_alpha != 0.3:
                cmd += ["--ogm_ge_alpha", str(t.ogm_ge_alpha)]
            if t.ogm_ge_noise_std != 0.0:
                cmd += ["--ogm_ge_noise_std", str(t.ogm_ge_noise_std)]
    if t.independent_audio_timestep:
        cmd.append("--independent_audio_timestep")
    if t.audio_silence_regularizer:
        cmd.append("--audio_silence_regularizer")
        if t.audio_silence_regularizer_weight != 1.0:
            cmd += ["--audio_silence_regularizer_weight", str(t.audio_silence_regularizer_weight)]
    if t.audio_supervision_mode != "off":
        cmd += ["--audio_supervision_mode", t.audio_supervision_mode]
        if t.audio_supervision_warmup_steps != 50:
            cmd += ["--audio_supervision_warmup_steps", str(t.audio_supervision_warmup_steps)]
        if t.audio_supervision_check_interval != 50:
            cmd += ["--audio_supervision_check_interval", str(t.audio_supervision_check_interval)]
        if t.audio_supervision_min_ratio != 0.9:
            cmd += ["--audio_supervision_min_ratio", str(t.audio_supervision_min_ratio)]
    if t.audio_dop:
        cmd.append("--audio_dop")
        args_parts = []
        _append_key_value_args(args_parts, t.audio_dop_args)
        if t.audio_dop_multiplier != 0.5:
            args_parts.append(f"multiplier={t.audio_dop_multiplier}")
        if args_parts:
            cmd += ["--audio_dop_args"] + args_parts
    if t.audio_bucket_strategy:
        cmd += ["--audio_bucket_strategy", t.audio_bucket_strategy]
    if t.audio_bucket_interval is not None:
        cmd += ["--audio_bucket_interval", str(t.audio_bucket_interval)]
    if t.audio_only_sequence_resolution != 64:
        cmd += ["--audio_only_sequence_resolution", str(t.audio_only_sequence_resolution)]
    if t.min_audio_batches_per_accum > 0:
        cmd += ["--min_audio_batches_per_accum", str(t.min_audio_batches_per_accum)]
    if t.audio_batch_probability is not None:
        cmd += ["--audio_batch_probability", str(t.audio_batch_probability)]
    if t.cts_lambda_video_driven > 0:
        cmd += ["--cts_lambda_video_driven", str(t.cts_lambda_video_driven)]
    if t.cts_lambda_audio_driven > 0:
        cmd += ["--cts_lambda_audio_driven", str(t.cts_lambda_audio_driven)]
    if t.modality_freeze_check_interval > 0:
        cmd += ["--modality_freeze_check_interval", str(t.modality_freeze_check_interval)]
        if t.modality_freeze_ratio_threshold != 0.5:
            cmd += ["--modality_freeze_ratio_threshold", str(t.modality_freeze_ratio_threshold)]
        if t.modality_freeze_warmup_steps != 100:
            cmd += ["--modality_freeze_warmup_steps", str(t.modality_freeze_warmup_steps)]
        if t.modality_freeze_ema_decay != 0.99:
            cmd += ["--modality_freeze_ema_decay", str(t.modality_freeze_ema_decay)]

    # Loss weighting
    if t.video_loss_weight != 1.0:
        cmd += ["--video_loss_weight", str(t.video_loss_weight)]
    if t.audio_loss_weight != 1.0:
        cmd += ["--audio_loss_weight", str(t.audio_loss_weight)]

    # Misc
    if t.separate_audio_buckets:
        cmd.append("--separate_audio_buckets")
    _append_dataloader_args(cmd, t)
    _append_ltx2_conditioning(cmd, t, config)

    if t.use_ema:
        cmd.append("--use_ema")
        if t.ema_decay != 0.9999:
            cmd += ["--ema_decay", str(t.ema_decay)]
        if t.ema_update_after_step != 100:
            cmd += ["--ema_update_after_step", str(t.ema_update_after_step)]
        if t.ema_update_every != 1:
            cmd += ["--ema_update_every", str(t.ema_update_every)]
        if t.save_ema_only:
            cmd.append("--save_ema_only")
        if t.ema_cpu_offload:
            cmd.append("--ema_cpu_offload")
    _append_advanced_cli_args(cmd, t)

    cmd += _split_cli_args(t.extra_args)
    return cmd


def build_full_finetune_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for full-parameter LTX-2 fine-tuning via accelerate launch."""
    toml_path = export_dataset_toml(config)
    t = config.full_finetune
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, t.ltx2_checkpoint)
    gemma_safetensors = _effective_gemma_safetensors(config, t.gemma_safetensors, t.gemma_root)
    gemma_root = _effective_gemma_root(config, t.gemma_root, gemma_safetensors)
    sample_prompts = _effective_full_finetune_sample_prompts(config)

    cmd = _accelerate_launch_prefix(t.mixed_precision, t.accelerate_extra_args)
    cmd.append(_find_script("ltx2_train.py"))

    # Dataset
    if t.config_file:
        cmd += ["--config_file", t.config_file]
    if t.dataset_manifest:
        cmd += ["--dataset_manifest", t.dataset_manifest]
    elif t.dataset_config:
        cmd += ["--dataset_config", t.dataset_config]
    else:
        cmd += ["--dataset_config", str(toml_path)]

    # Model
    cmd += ["--ltx2_checkpoint", ltx2_checkpoint]
    if t.vae:
        cmd += ["--vae", t.vae]
    if gemma_root:
        cmd += ["--gemma_root", gemma_root]
    if gemma_safetensors:
        cmd += ["--gemma_safetensors", gemma_safetensors]
    cmd += ["--ltx2_mode", t.ltx2_mode]
    if t.ltx_version != "2.3":
        cmd += ["--ltx_version", t.ltx_version]
    if t.ltx_version_check_mode != "warn":
        cmd += ["--ltx_version_check_mode", t.ltx_version_check_mode]
    if t.vae_dtype:
        cmd += ["--vae_dtype", t.vae_dtype]
    if t.fp8_base:
        cmd.append("--fp8_base")
    if t.fp8_scaled:
        cmd.append("--fp8_scaled")
    if getattr(t, "fp8_keep_blocks", ""):
        cmd += ["--fp8_keep_blocks", t.fp8_keep_blocks]
    if t.nf4_base:
        cmd.append("--nf4_base")
    if getattr(t, "nf4_block_size", 32) != 32:
        cmd += ["--nf4_block_size", str(t.nf4_block_size)]
    if t.flash_attn:
        cmd.append("--flash_attn")
    if t.flash3:
        cmd.append("--flash3")
    if getattr(t, "cudnn_attn", False):
        cmd.append("--cudnn_attn")
    if t.sdpa:
        cmd.append("--sdpa")
    if t.sage_attn:
        cmd.append("--sage_attn")
    if t.xformers:
        cmd.append("--xformers")
    if t.gemma_load_in_8bit:
        cmd.append("--gemma_load_in_8bit")
    if t.gemma_load_in_4bit:
        cmd.append("--gemma_load_in_4bit")
        if t.gemma_bnb_4bit_quant_type != "nf4":
            cmd += ["--gemma_bnb_4bit_quant_type", t.gemma_bnb_4bit_quant_type]
    if t.gemma_bnb_4bit_disable_double_quant:
        cmd.append("--gemma_bnb_4bit_disable_double_quant")
    if t.gemma_bnb_use_local_rank:
        cmd.append("--gemma_bnb_use_local_rank")
    if t.gemma_fp8_weight_offload:
        cmd.append("--gemma_fp8_weight_offload")
    else:
        cmd.append("--no-gemma_fp8_weight_offload")
    if t.ltx2_audio_only_model:
        cmd.append("--ltx2_audio_only_model")
    _append_ltx2_performance_args(cmd, t)

    # Optimizer
    cmd += ["--learning_rate", str(t.learning_rate)]
    if t.optimizer_type:
        cmd += ["--optimizer_type", t.optimizer_type]
    optimizer_args = _split_cli_args(t.optimizer_args)
    if _is_qapollo_optimizer_type(t.optimizer_type) and not _cli_args_has_key(optimizer_args, "optim_bits"):
        optimizer_args.append(f"optim_bits={getattr(t, 'qapollo_optim_bits', 8)}")
    if optimizer_args:
        cmd += ["--optimizer_args"] + optimizer_args
    if t.base_optimizer_args:
        cmd += ["--base_optimizer_args"] + _split_cli_args(t.base_optimizer_args)
    cmd += ["--lr_scheduler", t.lr_scheduler]
    cmd += ["--lr_warmup_steps", str(t.lr_warmup_steps)]
    if t.lr_decay_steps is not None:
        cmd += ["--lr_decay_steps", str(t.lr_decay_steps)]
    if t.lr_scheduler_num_cycles is not None:
        cmd += ["--lr_scheduler_num_cycles", str(t.lr_scheduler_num_cycles)]
    if t.lr_scheduler_power is not None:
        cmd += ["--lr_scheduler_power", str(t.lr_scheduler_power)]
    if t.lr_scheduler_min_lr_ratio is not None:
        cmd += ["--lr_scheduler_min_lr_ratio", str(t.lr_scheduler_min_lr_ratio)]
    if t.lr_scheduler_type:
        cmd += ["--lr_scheduler_type", t.lr_scheduler_type]
    if t.lr_scheduler_args:
        cmd += ["--lr_scheduler_args"] + _split_cli_args(t.lr_scheduler_args)
    if t.lr_scheduler_timescale is not None:
        cmd += ["--lr_scheduler_timescale", str(t.lr_scheduler_timescale)]
    cmd += ["--gradient_accumulation_steps", str(t.gradient_accumulation_steps)]
    if t.accumulation_group_by != "none":
        cmd += ["--accumulation_group_by", t.accumulation_group_by]
        cmd += ["--accumulation_group_remainder", t.accumulation_group_remainder]
    cmd += ["--max_grad_norm", str(t.max_grad_norm)]
    _append_weight_noise_args(cmd, t)
    if t.lr_args:
        cmd += ["--lr_args"] + _split_cli_args(t.lr_args)
    if t.lr_group_warmup_args:
        cmd += ["--lr_group_warmup_args"] + _split_cli_args(t.lr_group_warmup_args)

    # Schedule
    if t.max_train_epochs is not None:
        cmd += ["--max_train_epochs", str(t.max_train_epochs)]
    else:
        cmd += ["--max_train_steps", str(t.max_train_steps)]
    cmd += ["--timestep_sampling", _ltx2_timestep_sampling(t.timestep_sampling)]
    cmd += ["--discrete_flow_shift", str(t.discrete_flow_shift)]
    if t.seed is not None:
        cmd += ["--seed", str(t.seed)]
    if t.guidance_scale is not None:
        cmd += ["--guidance_scale", str(t.guidance_scale)]
    if t.sigmoid_scale is not None:
        cmd += ["--sigmoid_scale", str(t.sigmoid_scale)]
    if t.logit_mean is not None:
        cmd += ["--logit_mean", str(t.logit_mean)]
    if t.logit_std is not None:
        cmd += ["--logit_std", str(t.logit_std)]
    if t.mode_scale is not None:
        cmd += ["--mode_scale", str(t.mode_scale)]
    if t.min_timestep is not None:
        cmd += ["--min_timestep", str(t.min_timestep)]
    if t.max_timestep is not None:
        cmd += ["--max_timestep", str(t.max_timestep)]
    if t.shifted_logit_mode:
        cmd += ["--shifted_logit_mode", t.shifted_logit_mode]
    if t.shifted_logit_eps != 1e-3:
        cmd += ["--shifted_logit_eps", str(t.shifted_logit_eps)]
    if t.shifted_logit_uniform_prob != 0.1:
        cmd += ["--shifted_logit_uniform_prob", str(t.shifted_logit_uniform_prob)]
    if t.shifted_logit_shift is not None:
        cmd += ["--shifted_logit_shift", str(t.shifted_logit_shift)]
    if t.shifted_logit_clamp_auto_shift:
        cmd.append("--shifted_logit_clamp_auto_shift")
    if t.shifted_logit_min_shift != 0.95:
        cmd += ["--shifted_logit_min_shift", str(t.shifted_logit_min_shift)]
    if t.shifted_logit_max_shift != 2.05:
        cmd += ["--shifted_logit_max_shift", str(t.shifted_logit_max_shift)]
    if t.preserve_distribution_shape:
        cmd.append("--preserve_distribution_shape")
    if t.num_timestep_buckets is not None:
        cmd += ["--num_timestep_buckets", str(t.num_timestep_buckets)]
    if t.show_timesteps:
        cmd += ["--show_timesteps", t.show_timesteps]

    # Memory and execution
    if t.blocks_to_swap is not None:
        cmd += ["--blocks_to_swap", str(t.blocks_to_swap)]
    if t.gradient_checkpointing or t.blockwise_checkpointing:
        cmd.append("--gradient_checkpointing")
    if t.gradient_checkpointing_cpu_offload:
        cmd.append("--gradient_checkpointing_cpu_offload")
    if t.blockwise_checkpointing:
        cmd.append("--blockwise_checkpointing")
    if t.blocks_to_checkpoint is not None:
        cmd += ["--blocks_to_checkpoint", str(t.blocks_to_checkpoint)]
    if t.full_fp16:
        cmd.append("--full_fp16")
    if t.full_bf16:
        cmd.append("--full_bf16")
    if t.fused_backward_pass:
        cmd.append("--fused_backward_pass")
    if t.mem_eff_save:
        cmd.append("--mem_eff_save")
    if t.async_checkpoint_save:
        cmd.append("--async_checkpoint_save")
    if t.ffn_chunk_target:
        cmd += ["--ffn_chunk_target", t.ffn_chunk_target]
    if t.ffn_chunk_size:
        cmd += ["--ffn_chunk_size", str(t.ffn_chunk_size)]
    if t.use_pinned_memory_for_block_swap:
        cmd.append("--use_pinned_memory_for_block_swap")
    if t.img_in_txt_in_offloading:
        cmd.append("--img_in_txt_in_offloading")
    if t.ltx2_finetune_block_swap_mode != "default":
        cmd += ["--ltx2_finetune_block_swap_mode", t.ltx2_finetune_block_swap_mode]
    if t.ltx2_finetune_block_swap_mask != "all":
        cmd += ["--ltx2_finetune_block_swap_mask", t.ltx2_finetune_block_swap_mask]
    if t.freeze_early_blocks:
        cmd += ["--freeze_early_blocks", str(t.freeze_early_blocks)]
    if t.freeze_block_indices:
        cmd += ["--freeze_block_indices", t.freeze_block_indices]
    if t.block_lr_scales:
        cmd += ["--block_lr_scales"] + _split_cli_args(t.block_lr_scales)
    if t.non_block_lr_scale != 1.0:
        cmd += ["--non_block_lr_scale", str(t.non_block_lr_scale)]
    if t.attn_geometry_lr_scale != 1.0:
        cmd += ["--attn_geometry_lr_scale", str(t.attn_geometry_lr_scale)]
    if t.freeze_attn_geometry:
        cmd.append("--freeze_attn_geometry")
    if t.freeze_audio_params:
        cmd.append("--freeze_audio_params")
    if t.audio_param_lr_scale != 1.0:
        cmd += ["--audio_param_lr_scale", str(t.audio_param_lr_scale)]

    # fp8_gemm full-FT (FP8 GEMM for attn/FFN Linears; sm_89+).
    if getattr(t, "fp8_gemm", False):
        cmd.append("--fp8_gemm")
        if getattr(t, "fp8_gemm_targets", "video") != "video":
            cmd += ["--fp8_gemm_targets", t.fp8_gemm_targets]
        if getattr(t, "fp8_gemm_grad_dtype", "e4m3") != "e4m3":
            cmd += ["--fp8_gemm_grad_dtype", t.fp8_gemm_grad_dtype]
        if getattr(t, "fp8_gemm_min_numel", 16384) != 16384:
            cmd += ["--fp8_gemm_min_numel", str(t.fp8_gemm_min_numel)]
        if not getattr(t, "fp8_gemm_compile", True):
            cmd.append("--no-fp8_gemm_compile")
        if getattr(t, "fp8_gemm_scaling", "tensor") != "tensor":
            cmd += ["--fp8_gemm_scaling", t.fp8_gemm_scaling]

    # int8 weight-only full-FT (resident int8 weights + SR update).
    if getattr(t, "int8_weights", False):
        cmd.append("--int8_weights")
        if getattr(t, "int8_weights_targets", "video") != "video":
            cmd += ["--int8_weights_targets", t.int8_weights_targets]
        if getattr(t, "int8_weights_min_numel", 16384) != 16384:
            cmd += ["--int8_weights_min_numel", str(t.int8_weights_min_numel)]
        if getattr(t, "int8_weights_group_size", 0) != 0:
            cmd += ["--int8_weights_group_size", str(t.int8_weights_group_size)]
        if getattr(t, "int8_weights_outlier_quantile", 1.0) != 1.0:
            cmd += ["--int8_weights_outlier_quantile", str(t.int8_weights_outlier_quantile)]
        if getattr(t, "int8_weights_sparse_ratio", 0.0) != 0.0:
            cmd += ["--int8_weights_sparse_ratio", str(t.int8_weights_sparse_ratio)]
        if getattr(t, "int8_weights_convrot", ""):
            cmd += ["--int8_weights_convrot", str(t.int8_weights_convrot)]
        if getattr(t, "int8_weights_w8a8", False):
            cmd.append("--int8_weights_w8a8")
        if getattr(t, "int8_weights_prequant", ""):
            cmd += ["--int8_weights_prequant", str(t.int8_weights_prequant)]

    if getattr(t, "preserve_audio_timing", False):
        cmd.append("--preserve_audio_timing")

    # Compile / CUDA
    _append_compile_args(cmd, t)
    _append_cuda_args(cmd, t)

    # Q-GaLore
    if t.qgalore_full_ft:
        cmd.append("--qgalore_full_ft")
        cmd += ["--qgalore_targets", t.qgalore_targets]
        cmd += ["--qgalore_rank", str(t.qgalore_rank)]
        cmd += ["--qgalore_update_proj_gap", str(t.qgalore_update_proj_gap)]
        cmd += ["--qgalore_scale", str(t.qgalore_scale)]
        if t.qgalore_proj_type != "std":
            cmd += ["--qgalore_proj_type", t.qgalore_proj_type]
        cmd += ["--qgalore_proj_bits", str(t.qgalore_proj_bits)]
        cmd += ["--qgalore_proj_group_size", str(t.qgalore_proj_group_size)]
        cmd += ["--qgalore_weight_bits", str(t.qgalore_weight_bits)]
        cmd += ["--qgalore_weight_group_size", str(t.qgalore_weight_group_size)]
        cmd += ["--qgalore_min_weight_numel", str(t.qgalore_min_weight_numel)]
        if t.qgalore_max_modules is not None:
            cmd += ["--qgalore_max_modules", str(t.qgalore_max_modules)]
        if not t.qgalore_proj_quant:
            cmd.append("--no-qgalore_proj_quant")
        if not t.qgalore_stochastic_round:
            cmd.append("--no-qgalore_stochastic_round")
        qgalore_load_device = getattr(t, "qgalore_load_device", "cuda") or "cuda"
        if qgalore_load_device != "cuda":
            cmd += ["--qgalore_load_device", qgalore_load_device]
        if not t.qgalore_dequantize_save:
            cmd.append("--no-qgalore_dequantize_save")
        if t.qgalore_streaming_dequantize_save:
            cmd.append("--qgalore_streaming_dequantize_save")
            if t.qgalore_streaming_dequantize_device != "cpu":
                cmd += ["--qgalore_streaming_dequantize_device", t.qgalore_streaming_dequantize_device]
        if t.qgalore_cos_threshold != 0.4:
            cmd += ["--qgalore_cos_threshold", str(t.qgalore_cos_threshold)]
        if t.qgalore_gamma_proj != 2.0:
            cmd += ["--qgalore_gamma_proj", str(t.qgalore_gamma_proj)]
        if t.qgalore_queue_size != 5:
            cmd += ["--qgalore_queue_size", str(t.qgalore_queue_size)]
        if t.qgalore_svd_method != "full":
            cmd += ["--qgalore_svd_method", t.qgalore_svd_method]
        if t.qgalore_svd_oversampling != 32:
            cmd += ["--qgalore_svd_oversampling", str(t.qgalore_svd_oversampling)]
        if t.qgalore_svd_niter != 1:
            cmd += ["--qgalore_svd_niter", str(t.qgalore_svd_niter)]

    # APOLLO
    _optimizer_type = (t.optimizer_type or "").lower()
    if _optimizer_type in {
        "apollo",
        "apollo_adamw",
        "apolloadamw",
        "qapollo",
        "q_apollo",
        "qapollo_adamw",
        "qapolloadamw",
        "q_apollo_adamw",
    } or _optimizer_type.startswith("apollo_torch."):
        cmd += ["--apollo_rank", str(t.apollo_rank)]
        cmd += ["--apollo_update_proj_gap", str(t.apollo_update_proj_gap)]
        cmd += ["--apollo_scale", str(t.apollo_scale)]
        if t.apollo_proj != "random":
            cmd += ["--apollo_proj", t.apollo_proj]
        if t.apollo_proj_type != "std":
            cmd += ["--apollo_proj_type", t.apollo_proj_type]
        if t.apollo_scale_type != "channel":
            cmd += ["--apollo_scale_type", t.apollo_scale_type]
        if t.apollo_update_rule != "apollo":
            cmd += ["--apollo_update_rule", t.apollo_update_rule]

    # Sampling
    if t.sample_every_n_steps:
        cmd += ["--sample_every_n_steps", str(t.sample_every_n_steps)]
    if t.sample_every_n_epochs:
        cmd += ["--sample_every_n_epochs", str(t.sample_every_n_epochs)]
    if sample_prompts:
        cmd += ["--sample_prompts", sample_prompts]
    if t.precache_sample_prompts:
        cmd.append("--precache_sample_prompts")
    if t.use_precached_sample_prompts:
        cmd.append("--use_precached_sample_prompts")
    if t.sample_prompts_cache:
        cmd += ["--sample_prompts_cache", t.sample_prompts_cache]
    if t.use_precached_sample_latents:
        cmd.append("--use_precached_sample_latents")
    if t.sample_latents_cache:
        cmd += ["--sample_latents_cache", t.sample_latents_cache]
    cmd += ["--sample_sampling_preset", t.sample_sampling_preset]
    if t.sample_sigma_schedule != "auto":
        cmd += ["--sample_sigma_schedule", t.sample_sigma_schedule]
    if t.sample_sampler != "auto":
        cmd += ["--sample_sampler", t.sample_sampler]
    _append_optional(cmd, "--height", t.height)
    _append_optional(cmd, "--width", t.width)
    _append_optional(cmd, "--sample_num_frames", t.sample_num_frames)
    _append_optional(cmd, "--video_cfg_scale", t.video_cfg_scale)
    _append_optional(cmd, "--audio_cfg_scale", t.audio_cfg_scale)
    if t.sample_with_offloading:
        cmd.append("--sample_with_offloading")
    if t.sample_merge_audio:
        cmd.append("--sample_merge_audio")
    if t.sample_disable_audio:
        cmd.append("--sample_disable_audio")
    if t.sample_at_first:
        cmd.append("--sample_at_first")
    if t.sample_tiled_vae:
        cmd.append("--sample_tiled_vae")
    if t.sample_vae_tile_size is not None:
        cmd += ["--sample_vae_tile_size", str(t.sample_vae_tile_size)]
    if t.sample_vae_tile_overlap is not None:
        cmd += ["--sample_vae_tile_overlap", str(t.sample_vae_tile_overlap)]
    if t.sample_vae_temporal_tile_size is not None:
        cmd += ["--sample_vae_temporal_tile_size", str(t.sample_vae_temporal_tile_size)]
    if t.sample_vae_temporal_tile_overlap is not None:
        cmd += ["--sample_vae_temporal_tile_overlap", str(t.sample_vae_temporal_tile_overlap)]
    if t.sample_two_stage:
        cmd.append("--sample_two_stage")
        if t.spatial_upsampler_path:
            cmd += ["--spatial_upsampler_path", t.spatial_upsampler_path]
        if getattr(t, "temporal_upsampler_path", ""):
            cmd += ["--temporal_upsampler_path", t.temporal_upsampler_path]
    if t.sample_audio_only:
        cmd.append("--sample_audio_only")
    if t.sample_disable_flash_attn:
        cmd.append("--sample_disable_flash_attn")
    if not t.sample_i2v_token_timestep_mask:
        cmd.append("--no-sample_i2v_token_timestep_mask")
    if not t.sample_audio_subprocess:
        cmd.append("--no-sample_audio_subprocess")
    if t.sample_include_reference:
        cmd.append("--sample_include_reference")
    if t.reference_downscale != 1:
        cmd += ["--reference_downscale", str(t.reference_downscale)]
    if getattr(t, "reference_temporal_scale", 1) != 1:
        cmd += ["--reference_temporal_scale", str(t.reference_temporal_scale)]
    if t.reference_frames != 1:
        cmd += ["--reference_frames", str(t.reference_frames)]

    # Validation
    if t.validate_every_n_steps is not None:
        cmd += ["--validate_every_n_steps", str(t.validate_every_n_steps)]
    if t.validate_every_n_epochs is not None:
        cmd += ["--validate_every_n_epochs", str(t.validate_every_n_epochs)]
    if t.validation_dataset_config:
        cmd += ["--validation_dataset_config", t.validation_dataset_config]
    if t.validation_extra_configs:
        cmd += ["--validation_extra_configs"] + _split_cli_args(t.validation_extra_configs)
    if t.num_validation_batches is not None:
        cmd += ["--num_validation_batches", str(t.num_validation_batches)]
    if t.validation_timesteps:
        cmd += ["--validation_timesteps", t.validation_timesteps]
    if t.offload_optimizer_during_validation:
        cmd.append("--offload_optimizer_during_validation")

    # Output
    cmd += ["--output_dir", _effective_output_dir(t.output_dir)]
    if t.output_name:
        cmd += ["--output_name", t.output_name]
    if t.save_every_n_epochs:
        cmd += ["--save_every_n_epochs", str(t.save_every_n_epochs)]
    if t.save_every_n_steps:
        cmd += ["--save_every_n_steps", str(t.save_every_n_steps)]
    if t.save_last_n_epochs is not None:
        cmd += ["--save_last_n_epochs", str(t.save_last_n_epochs)]
    if t.save_last_n_steps is not None:
        cmd += ["--save_last_n_steps", str(t.save_last_n_steps)]
    if t.save_state:
        cmd.append("--save_state")
    if t.save_state_on_train_end:
        cmd.append("--save_state_on_train_end")
    if (t.save_state or t.save_state_on_train_end) and t.save_state_mode != "full":
        cmd += ["--save_state_mode", t.save_state_mode]
    if t.no_final_save:
        cmd.append("--no_final_save")
    if t.save_checkpoint_metadata:
        cmd.append("--save_checkpoint_metadata")
    if t.no_metadata:
        cmd.append("--no_metadata")
    if t.no_convert_to_comfy:
        cmd.append("--no_convert_to_comfy")
    if t.save_comfy_format:
        cmd.append("--save_comfy_format")
    if t.save_merged_checkpoint:
        cmd.append("--save_merged_checkpoint")
    if t.log_with:
        cmd += ["--log_with", t.log_with]
    if t.log_with and t.logging_dir:
        cmd += ["--logging_dir", t.logging_dir]
    if t.log_prefix:
        cmd += ["--log_prefix", t.log_prefix]
    if t.log_tracker_name:
        cmd += ["--log_tracker_name", t.log_tracker_name]
    if t.log_tracker_config:
        cmd += ["--log_tracker_config", t.log_tracker_config]
    if t.log_config:
        cmd.append("--log_config")
    if t.wandb_run_name:
        cmd += ["--wandb_run_name", t.wandb_run_name]
    if t.wandb_api_key:
        cmd += ["--wandb_api_key", t.wandb_api_key]
    if t.log_cuda_memory_every_n_steps is not None:
        cmd += ["--log_cuda_memory_every_n_steps", str(t.log_cuda_memory_every_n_steps)]
    if t.resume:
        cmd += ["--resume", t.resume]
    if t.autoresume:
        cmd.append("--autoresume")
    if t.reset_optimizer:
        cmd.append("--reset_optimizer")
    if t.reset_optimizer_params:
        cmd.append("--reset_optimizer_params")
    if t.reset_dataloader:
        cmd.append("--reset_dataloader")
    if t.training_comment:
        cmd += ["--training_comment", t.training_comment]
    if t.loss_type != "mse":
        cmd += ["--loss_type", t.loss_type]
    if t.loss_type in ("huber", "smooth_l1") and t.huber_delta != 1.0:
        cmd += ["--huber_delta", str(t.huber_delta)]

    # EMA and instrumentation
    if t.use_ema:
        cmd.append("--use_ema")
        if t.ema_decay != 0.9999:
            cmd += ["--ema_decay", str(t.ema_decay)]
        if t.ema_update_after_step != 100:
            cmd += ["--ema_update_after_step", str(t.ema_update_after_step)]
        if t.ema_update_every != 1:
            cmd += ["--ema_update_every", str(t.ema_update_every)]
        if t.save_ema_only:
            cmd.append("--save_ema_only")
        if t.ema_cpu_offload:
            cmd.append("--ema_cpu_offload")
    if t.log_weight_drift_every:
        cmd += ["--log_weight_drift_every", str(t.log_weight_drift_every)]
        cmd += ["--weight_drift_target", t.weight_drift_target]
        cmd += ["--weight_drift_top_k", str(t.weight_drift_top_k)]
    if t.log_grad_norm_every:
        cmd += ["--log_grad_norm_every", str(t.log_grad_norm_every)]
        cmd += ["--grad_norm_target", t.grad_norm_target]
        cmd += ["--grad_norm_top_k", str(t.grad_norm_top_k)]
    if t.log_output_drift_every:
        cmd += ["--log_output_drift_every", str(t.log_output_drift_every)]
        cmd += ["--output_drift_batches", str(t.output_drift_batches)]
        cmd += ["--output_drift_timestep", str(t.output_drift_timestep)]

    # TREAD token routing
    if t.tread:
        cmd.append("--tread")
        args_parts = []
        _append_key_value_args(args_parts, t.tread_args)
        if t.tread_target != "video":
            args_parts.append(f"target={t.tread_target}")
        if t.tread_selection_ratio != 0.5:
            args_parts.append(f"selection_ratio={t.tread_selection_ratio}")
        if t.tread_start_layer_idx is not None:
            args_parts.append(f"start_layer_idx={t.tread_start_layer_idx}")
        if t.tread_end_layer_idx is not None:
            args_parts.append(f"end_layer_idx={t.tread_end_layer_idx}")
        if args_parts:
            cmd += ["--tread_args"] + args_parts

    _append_dataloader_args(cmd, t)

    # Conditioning (recipe / intrinsic / directional). --lora_target_preset / --ic_lora_strategy are
    # LoRA-only and not emitted for full fine-tune; a recipe's reference still derives the strategy.
    _append_ltx2_conditioning(cmd, t, config, "conditioning_config.fft.toml")

    # Gradient noise-scale probe (diagnostic): when > 0 the trainer skips training and estimates the
    # critical batch B_simple, then exits. param_frac only matters while probing.
    if t.noise_scale_probe and t.noise_scale_probe > 0:
        cmd += ["--noise_scale_probe", str(t.noise_scale_probe)]
        cmd += ["--noise_scale_probe_rounds", str(t.noise_scale_probe_rounds)]
        if t.noise_scale_probe_param_frac != 1.0:
            cmd += ["--noise_scale_probe_param_frac", str(t.noise_scale_probe_param_frac)]

    _append_advanced_cli_args(cmd, t)

    cmd += _split_cli_args(t.extra_args)
    return cmd


def _remote_stage_server_arglist(config: ProjectConfig) -> list[str]:
    r = config.remote_stage_server
    ltx2_checkpoint = (
        r.ltx2_checkpoint or config.default_ltx2_checkpoint or str(_default_model_dir(config) / DEFAULT_LTX2_CHECKPOINT_NAME)
    )

    cmd: list[str] = [
        "--ltx2_checkpoint",
        ltx2_checkpoint,
        "--bind",
        r.bind,
        "--port",
        str(r.port),
        "--device",
        r.device,
        "--dtype",
        r.dtype,
        "--split",
        str(r.split),
    ]

    if r.end != -1:
        cmd += ["--end", str(r.end)]
    if r.load_device:
        cmd += ["--load_device", r.load_device]
    if r.trainable:
        cmd.append("--trainable")
        if r.trainable_scope != "auto":
            cmd += ["--trainable_scope", r.trainable_scope]
        if r.learning_rate is not None:
            cmd += ["--learning_rate", str(r.learning_rate)]
        if r.weight_decay != 0.01:
            cmd += ["--weight_decay", str(r.weight_decay)]
        if r.max_grad_norm != 0.0:
            cmd += ["--max_grad_norm", str(r.max_grad_norm)]
    if r.prune_non_stage_blocks:
        cmd.append("--prune_non_stage_blocks")
    if r.stage_only_device_placement:
        cmd.append("--stage_only_device_placement")
    if r.full_model_device_placement:
        cmd.append("--full_model_device_placement")
    if r.block_only_load:
        cmd.append("--block_only_load")
    if r.network_module:
        cmd += ["--network_module", r.network_module]
    if r.network_dim is not None:
        cmd += ["--network_dim", str(r.network_dim)]
    if r.network_alpha is not None:
        cmd += ["--network_alpha", str(r.network_alpha)]
    if r.network_dropout is not None:
        cmd += ["--network_dropout", str(r.network_dropout)]
    if r.network_args:
        cmd += ["--network_args"] + _split_cli_args(r.network_args)
    if r.network_weights:
        cmd += ["--network_weights", r.network_weights]
    if r.network_lr is not None:
        cmd += ["--network_lr", str(r.network_lr)]
    if r.ltx2_mode:
        cmd += ["--ltx2_mode", r.ltx2_mode]
    if r.ltx2_audio_only_model:
        cmd.append("--ltx2_audio_only_model")
    if r.attn_mode:
        cmd += ["--attn_mode", r.attn_mode]
    if r.fp8_scaled:
        cmd.append("--fp8_scaled")
    if r.fp8_w8a8:
        cmd.append("--fp8_w8a8")
        if r.w8a8_mode != "int8":
            cmd += ["--w8a8_mode", r.w8a8_mode]
    if r.fp8_w8a8 and r.w8a8_mode == "int8" and getattr(r, "w8a8_backend", None):
        cmd += ["--w8a8_backend", r.w8a8_backend]
    if r.fp8_upcast:
        cmd.append("--fp8_upcast")
    if r.fp8_upcast_stochastic:
        cmd.append("--fp8_upcast_stochastic")
    if r.fp8_upcast_seed != 0:
        cmd += ["--fp8_upcast_seed", str(r.fp8_upcast_seed)]
    if r.fp8_keep_blocks:
        cmd += ["--fp8_keep_blocks", r.fp8_keep_blocks]
    if r.nf4_base:
        cmd.append("--nf4_base")
    if r.nf4_block_size != 32:
        cmd += ["--nf4_block_size", str(r.nf4_block_size)]
    if r.split_attn_target:
        cmd += ["--split_attn_target", r.split_attn_target]
    if r.split_attn_mode:
        cmd += ["--split_attn_mode", r.split_attn_mode]
    if r.split_attn_chunk_size:
        cmd += ["--split_attn_chunk_size", str(r.split_attn_chunk_size)]
    if r.ffn_chunk_target:
        cmd += ["--ffn_chunk_target", r.ffn_chunk_target]
    if r.ffn_chunk_size:
        cmd += ["--ffn_chunk_size", str(r.ffn_chunk_size)]
    if r.quantize_device:
        cmd += ["--quantize_device", r.quantize_device]
    if r.int8_block_size != 256:
        cmd += ["--int8_block_size", str(r.int8_block_size)]
    if r.log_level and r.log_level != "INFO":
        cmd += ["--log_level", r.log_level]

    cmd += _split_cli_args(r.extra_args)
    return cmd


def build_remote_stage_server_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_remote_stage_server.py."""
    return [
        sys.executable,
        "-u",
        _find_script("ltx2_remote_stage_server.py"),
        *_remote_stage_server_arglist(config),
    ]


def build_remote_stage_launcher_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_remote_stage_launcher.py."""
    launcher_config_path = _write_remote_stage_orchestrator_config(config)
    return [
        sys.executable,
        "-u",
        _find_script("ltx2_remote_stage_launcher.py"),
        "--project_config_json",
        str(launcher_config_path),
    ]


def build_slider_training_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for slider LoRA training via accelerate launch.

    Shared settings (model, LoRA, optimizer, memory, output) are inherited
    from the training config.  Only slider-specific values (steps, output name,
    slider config, latent dims) come from ``config.slider``.
    """
    s = config.slider
    t = config.training
    slider_toml = _write_slider_toml(config, build_slider_toml_path(config))
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, t.ltx2_checkpoint)
    gemma_safetensors = _effective_gemma_safetensors(config, t.gemma_safetensors, t.gemma_root)
    gemma_root = _effective_gemma_root(config, t.gemma_root, gemma_safetensors)
    network_module = _effective_training_network_module(t)
    network_args_parts = _training_network_args_parts(t)

    cmd = _accelerate_launch_prefix(t.mixed_precision, s.accelerate_extra_args)
    cmd.append(_find_script("ltx2_train_slider.py"))

    # Slider config
    cmd += ["--slider_config", str(slider_toml)]

    # Model — from training config
    cmd += ["--ltx2_checkpoint", ltx2_checkpoint]
    if gemma_root:
        cmd += ["--gemma_root", gemma_root]
    if gemma_safetensors:
        cmd += ["--gemma_safetensors", gemma_safetensors]
    if t.ltx2_mode:
        cmd += ["--ltx2_mode", t.ltx2_mode]
    if t.ltx_version:
        cmd += ["--ltx_version", t.ltx_version]
    if t.vae_dtype:
        cmd += ["--vae_dtype", t.vae_dtype]
    if s.mode == "ic_reference":
        cmd += ["--lora_target_preset", "v2v"]
        cmd += ["--ic_lora_strategy", "v2v"]
    elif t.lora_target_preset:
        cmd += ["--lora_target_preset", t.lora_target_preset]
    if t.fp8_base:
        cmd.append("--fp8_base")
    if t.fp8_scaled:
        cmd.append("--fp8_scaled")
    if getattr(t, "fp8_keep_blocks", ""):
        cmd += ["--fp8_keep_blocks", t.fp8_keep_blocks]
    if t.flash_attn:
        cmd.append("--flash_attn")
    if t.flash3:
        cmd.append("--flash3")
    if getattr(t, "cudnn_attn", False):
        cmd.append("--cudnn_attn")
    if t.sdpa:
        cmd.append("--sdpa")
    if t.sage_attn:
        cmd.append("--sage_attn")
    if t.xformers:
        cmd.append("--xformers")
    if t.gemma_load_in_8bit:
        cmd.append("--gemma_load_in_8bit")
    if t.gemma_load_in_4bit:
        cmd.append("--gemma_load_in_4bit")
    if t.gemma_bnb_use_local_rank:
        cmd.append("--gemma_bnb_use_local_rank")
    if t.gemma_fp8_weight_offload:
        cmd.append("--gemma_fp8_weight_offload")
    else:
        cmd.append("--no-gemma_fp8_weight_offload")

    # Text mode latent dimensions — slider-specific
    if s.mode == "text":
        cmd += ["--latent_frames", str(s.latent_frames)]
        cmd += ["--latent_height", str(s.latent_height)]
        cmd += ["--latent_width", str(s.latent_width)]
    else:
        # First-frame conditioning probability — only meaningful for reference
        # (image/video pair) sliders, read by the reference training step.
        cmd += ["--ltx2_first_frame_conditioning_p", str(t.ltx2_first_frame_conditioning_p)]
    if s.guidance_strength != 1.0:
        cmd += ["--guidance_strength", str(s.guidance_strength)]
    if s.sample_slider_range != "-2,-1,0,1,2":
        cmd += ["--sample_slider_range", s.sample_slider_range]

    # LoRA — from training config
    cmd += ["--network_module", network_module]
    if t.network_dim is not None:
        cmd += ["--network_dim", str(t.network_dim)]
    if t.network_alpha != 1:
        cmd += ["--network_alpha", str(t.network_alpha)]
    if network_args_parts:
        cmd += ["--network_args"] + network_args_parts

    # Additional LoRA options inherited from the training config.
    if t.network_weights:
        cmd += ["--network_weights", t.network_weights]
    if t.network_dropout is not None:
        cmd += ["--network_dropout", str(t.network_dropout)]
    if t.scale_weight_norms is not None:
        cmd += ["--scale_weight_norms", str(t.scale_weight_norms)]

    # Optimizer — from training config
    cmd += ["--learning_rate", str(t.learning_rate)]
    if t.optimizer_type:
        cmd += ["--optimizer_type", t.optimizer_type]
    if t.optimizer_args:
        cmd += ["--optimizer_args"] + _split_cli_args(t.optimizer_args)
    cmd += ["--gradient_accumulation_steps", str(t.gradient_accumulation_steps)]
    cmd += ["--max_grad_norm", str(t.max_grad_norm)]

    # LR scheduler — from training config
    cmd += ["--lr_scheduler", t.lr_scheduler]
    cmd += ["--lr_warmup_steps", str(t.lr_warmup_steps)]
    if t.lr_decay_steps is not None:
        cmd += ["--lr_decay_steps", str(t.lr_decay_steps)]
    if t.lr_scheduler_num_cycles is not None:
        cmd += ["--lr_scheduler_num_cycles", str(t.lr_scheduler_num_cycles)]
    if t.lr_scheduler_power is not None:
        cmd += ["--lr_scheduler_power", str(t.lr_scheduler_power)]
    if t.lr_scheduler_min_lr_ratio is not None:
        cmd += ["--lr_scheduler_min_lr_ratio", str(t.lr_scheduler_min_lr_ratio)]
    if t.lr_scheduler_type:
        cmd += ["--lr_scheduler_type", t.lr_scheduler_type]
    if t.lr_scheduler_args:
        cmd += ["--lr_scheduler_args"] + _split_cli_args(t.lr_scheduler_args)
    if t.lr_scheduler_timescale is not None:
        cmd += ["--lr_scheduler_timescale", str(t.lr_scheduler_timescale)]

    # Schedule — slider override for steps
    cmd += ["--max_train_steps", str(s.max_train_steps)]
    if t.seed is not None:
        cmd += ["--seed", str(t.seed)]

    # Timestep / sigma distribution — slider reads these via getattr in its loop.
    # Excludes timestep_sampling/discrete_flow_shift/weighting_scheme (slider hardcodes
    # shifted_logit_normal sampling) and guidance_scale (slider uses its own guidance_strength).
    # --logit_mean is intentionally NOT forwarded: it belongs to the standard
    # logit_normal weighting scheme. The slider's shifted_logit_normal sampler takes
    # its mean from the per-sequence-length shift (or --shifted_logit_shift), so
    # logit_mean would be inert for sliders. --shifted_logit_shift is the equivalent.
    if t.logit_std is not None:
        cmd += ["--logit_std", str(t.logit_std)]
    if t.min_timestep is not None:
        cmd += ["--min_timestep", str(t.min_timestep)]
    if t.max_timestep is not None:
        cmd += ["--max_timestep", str(t.max_timestep)]
    if t.shifted_logit_mode:
        cmd += ["--shifted_logit_mode", t.shifted_logit_mode]
    if t.shifted_logit_eps != 1e-3:
        cmd += ["--shifted_logit_eps", str(t.shifted_logit_eps)]
    if t.shifted_logit_uniform_prob != 0.1:
        cmd += ["--shifted_logit_uniform_prob", str(t.shifted_logit_uniform_prob)]
    if t.shifted_logit_shift is not None:
        cmd += ["--shifted_logit_shift", str(t.shifted_logit_shift)]
    if t.shifted_logit_clamp_auto_shift:
        cmd.append("--shifted_logit_clamp_auto_shift")
    if t.shifted_logit_min_shift != 0.95:
        cmd += ["--shifted_logit_min_shift", str(t.shifted_logit_min_shift)]
    if t.shifted_logit_max_shift != 2.05:
        cmd += ["--shifted_logit_max_shift", str(t.shifted_logit_max_shift)]

    # Memory — from training config
    if t.blocks_to_swap is not None:
        cmd += ["--blocks_to_swap", str(t.blocks_to_swap)]
    if t.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if t.use_pinned_memory_for_block_swap:
        cmd.append("--use_pinned_memory_for_block_swap")
    _append_h2d_block_swap_args(cmd, t)

    # Compile / dynamo — from training config
    if t.compile:
        cmd.append("--compile")
        if t.compile_backend:
            cmd += ["--compile_backend", t.compile_backend]
        if t.compile_mode:
            cmd += ["--compile_mode", t.compile_mode]
        inductor_config = getattr(t, "inductor_config", "")
        if inductor_config:
            tokens = [tok for tok in inductor_config.replace(",", " ").split() if tok]
            if tokens:
                cmd += ["--inductor_config", *tokens]
        compile_dynamic = _compile_dynamic_value(t.compile_dynamic)
        if compile_dynamic:
            cmd += ["--compile_dynamic", compile_dynamic]
        if t.compile_fullgraph:
            cmd.append("--compile_fullgraph")
        if t.compile_cache_size_limit is not None:
            cmd += ["--compile_cache_size_limit", str(t.compile_cache_size_limit)]
    if t.dynamo_backend != "NO":
        cmd += ["--dynamo_backend", t.dynamo_backend]
        if t.dynamo_mode:
            cmd += ["--dynamo_mode", t.dynamo_mode]
        if t.dynamo_fullgraph:
            cmd.append("--dynamo_fullgraph")
        if t.dynamo_dynamic:
            cmd.append("--dynamo_dynamic")

    # CUDA performance — from training config
    if t.cuda_allow_tf32:
        cmd.append("--cuda_allow_tf32")
    if t.cuda_cudnn_benchmark:
        cmd.append("--cuda_cudnn_benchmark")
    if t.cuda_memory_fraction is not None:
        cmd += ["--cuda_memory_fraction", str(t.cuda_memory_fraction)]

    # Output — dir from training, name from slider
    cmd += ["--output_dir", _effective_output_dir(t.output_dir)]
    if s.output_name:
        cmd += ["--output_name", s.output_name]
    if t.save_every_n_steps:
        cmd += ["--save_every_n_steps", str(t.save_every_n_steps)]
    if t.save_last_n_steps is not None:
        cmd += ["--save_last_n_steps", str(t.save_last_n_steps)]
    if t.save_last_n_steps_state is not None:
        cmd += ["--save_last_n_steps_state", str(t.save_last_n_steps_state)]
    if t.save_state:
        cmd.append("--save_state")
    if t.save_state_on_train_end:
        cmd.append("--save_state_on_train_end")
    if (t.save_state or t.save_state_on_train_end) and t.save_state_mode != "full":
        cmd += ["--save_state_mode", t.save_state_mode]
    if t.resume:
        cmd += ["--resume", t.resume]
    if t.reset_optimizer:
        cmd.append("--reset_optimizer")
    if t.reset_optimizer_params:
        cmd.append("--reset_optimizer_params")
    if t.autoresume:
        cmd.append("--autoresume")

    # Model metadata — embedded in saved safetensors, useful for distribution
    if t.metadata_title:
        cmd += ["--metadata_title", t.metadata_title]
    if t.metadata_author:
        cmd += ["--metadata_author", t.metadata_author]
    if t.metadata_description:
        cmd += ["--metadata_description", t.metadata_description]
    if t.metadata_license:
        cmd += ["--metadata_license", t.metadata_license]
    if t.metadata_tags:
        cmd += ["--metadata_tags", t.metadata_tags]
    if t.metadata_reso:
        cmd += ["--metadata_reso", t.metadata_reso]
    if t.metadata_arch:
        cmd += ["--metadata_arch", t.metadata_arch]

    # Sampling — inherited from training config (same pattern as build_training_cmd)
    sample_prompts = _effective_training_sample_prompts(config)
    if t.sample_every_n_steps:
        cmd += ["--sample_every_n_steps", str(t.sample_every_n_steps)]
    if t.sample_every_n_epochs:
        cmd += ["--sample_every_n_epochs", str(t.sample_every_n_epochs)]
    if sample_prompts:
        cmd += ["--sample_prompts", sample_prompts]
    if t.sample_at_first:
        cmd.append("--sample_at_first")
    cmd += ["--sample_sampling_preset", t.sample_sampling_preset]
    if t.sample_sigma_schedule != "auto":
        cmd += ["--sample_sigma_schedule", t.sample_sigma_schedule]
    if t.sample_sampler != "auto":
        cmd += ["--sample_sampler", t.sample_sampler]
    if t.sample_use_default_negative_prompt is True:
        cmd.append("--sample_use_default_negative_prompt")
    elif t.sample_use_default_negative_prompt is False:
        cmd.append("--no-sample_use_default_negative_prompt")
    if t.sample_with_offloading:
        cmd.append("--sample_with_offloading")
    _append_optional(cmd, "--height", t.height)
    _append_optional(cmd, "--width", t.width)
    _append_optional(cmd, "--sample_num_frames", t.sample_num_frames)
    _append_optional(cmd, "--video_cfg_scale", t.video_cfg_scale)

    # Logging — inherited from training config (same pattern as build_training_cmd)
    if t.log_with:
        cmd += ["--log_with", t.log_with]
    if t.log_with and t.logging_dir:
        cmd += ["--logging_dir", t.logging_dir]
    if t.log_prefix:
        cmd += ["--log_prefix", t.log_prefix]
    if t.log_tracker_name:
        cmd += ["--log_tracker_name", t.log_tracker_name]
    if t.log_tracker_config:
        cmd += ["--log_tracker_config", t.log_tracker_config]
    if t.log_config:
        cmd.append("--log_config")
    if t.wandb_run_name:
        cmd += ["--wandb_run_name", t.wandb_run_name]
    if t.wandb_api_key:
        cmd += ["--wandb_api_key", t.wandb_api_key]
    if t.log_cuda_memory_every_n_steps is not None:
        cmd += ["--log_cuda_memory_every_n_steps", str(t.log_cuda_memory_every_n_steps)]
    if t.save_checkpoint_metadata:
        cmd.append("--save_checkpoint_metadata")
    if t.no_metadata:
        cmd.append("--no_metadata")
    if t.no_convert_to_comfy:
        cmd.append("--no_convert_to_comfy")
    if not t.save_original_lora:
        cmd.append("--no_save_original_lora")

    cmd += _split_cli_args(s.extra_args)
    return cmd


def _rl_common_model_args(config: ProjectConfig) -> list[str]:
    """Model + LoRA + memory args shared by both RL phases (inherited from training config).

    Mirrors the slider builder: RL is a standalone driver that reuses the supervised
    setup, so model/precision/network/memory all come from ``config.training``.
    """
    t = config.training
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, t.ltx2_checkpoint)
    gemma_safetensors = _effective_gemma_safetensors(config, t.gemma_safetensors, t.gemma_root)
    gemma_root = _effective_gemma_root(config, t.gemma_root, gemma_safetensors)
    network_module = _effective_training_network_module(t)
    network_args_parts = _training_network_args_parts(t)

    args: list[str] = ["--ltx2_checkpoint", ltx2_checkpoint]
    if gemma_root:
        args += ["--gemma_root", gemma_root]
    if gemma_safetensors:
        args += ["--gemma_safetensors", gemma_safetensors]
    if t.ltx2_mode:
        args += ["--ltx2_mode", t.ltx2_mode]
    args += ["--network_module", network_module]
    if t.network_dim is not None:
        args += ["--network_dim", str(t.network_dim)]
    if t.network_alpha != 1:
        args += ["--network_alpha", str(t.network_alpha)]
    if t.lora_target_preset:
        args += ["--lora_target_preset", t.lora_target_preset]
    if network_args_parts:
        args += ["--network_args"] + network_args_parts
    # warm-start LoRA: Phase A `old` snapshot; Phase B `default`==`old` snapshot (the cache invariant)
    if t.network_weights:
        args += ["--network_weights", t.network_weights]
    if t.fp8_base:
        args.append("--fp8_base")
    if t.fp8_scaled:
        args.append("--fp8_scaled")
    if getattr(t, "fp8_keep_blocks", ""):
        args += ["--fp8_keep_blocks", t.fp8_keep_blocks]
    if t.flash_attn:
        args.append("--flash_attn")
    if t.gemma_load_in_8bit:
        args.append("--gemma_load_in_8bit")
    if t.gemma_load_in_4bit:
        args.append("--gemma_load_in_4bit")
    if t.gemma_fp8_weight_offload:
        args.append("--gemma_fp8_weight_offload")
    else:
        args.append("--no-gemma_fp8_weight_offload")
    if t.blocks_to_swap is not None:
        args += ["--blocks_to_swap", str(t.blocks_to_swap)]
    if t.use_pinned_memory_for_block_swap:
        args.append("--use_pinned_memory_for_block_swap")
    _append_h2d_block_swap_args(args, t)
    if t.gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if t.seed is not None:
        args += ["--seed", str(t.seed)]
    return args


def _rl_sample_dim_args(rl) -> list[str]:
    """Sample-dimension flags shared by Phase A and Phase B (online)."""
    args: list[str] = []
    if rl.sample_width != 768:
        args += ["--sample_width", str(rl.sample_width)]
    if rl.sample_height != 512:
        args += ["--sample_height", str(rl.sample_height)]
    if rl.sample_frames != 49:
        args += ["--sample_frames", str(rl.sample_frames)]
    if rl.sample_cfg != 1.0:
        args += ["--sample_cfg", str(rl.sample_cfg)]
    if rl.sample_steps != 20:
        args += ["--sample_steps", str(rl.sample_steps)]
    return args


def _rl_reward_args(rl) -> list[str]:
    """Reward-spec flags shared by both phases."""
    args = ["--reward_fn", rl.reward_fn]
    reward_args_parts = _split_cli_args(rl.reward_args)
    if reward_args_parts:
        args += ["--reward_args"] + reward_args_parts
    plugin_parts = _split_cli_args(getattr(rl, "reward_plugins", ""))
    if plugin_parts:
        args += ["--reward_plugins"] + plugin_parts
    return args


def build_rl_cache_rollouts_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_rollouts.py (RL Phase A: rollout generation + scoring)."""
    rl = config.rl
    cmd = [
        sys.executable,
        "-u",
        _find_script("ltx2_cache_rollouts.py"),
    ]
    cmd += _rl_common_model_args(config)
    cmd += ["--rl_rollout_cache", rl.rl_rollout_cache]
    cmd += ["--rl_prompts", rl.rl_prompts]
    cmd += _rl_reward_args(rl)
    cmd += ["--rl_group_size", str(rl.rl_group_size)]
    cmd += _rl_sample_dim_args(rl)
    if rl.rl_save_old_lora:
        cmd += ["--rl_save_old_lora", rl.rl_save_old_lora]
    # ppo (trajectory-faithful DDPO) needs the SDE sampler so the per-step trajectory is cached.
    if getattr(rl, "rl_loss", "nft") == "ppo":
        cmd += ["--rl_sde_sampler"]
        if rl.rl_sde_eta != 1.0:
            cmd += ["--rl_sde_eta", str(rl.rl_sde_eta)]
    cmd += _split_cli_args(rl.extra_args)
    return cmd


def build_rl_train_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_train_rl.py (RL Phase B: NFT cache-replay / online loop).

    Mirrors the slider builder: optimizer + schedule + output inherit from the training
    config, RL-specific values come from ``config.rl``.
    """
    rl = config.rl
    t = config.training
    cmd = _accelerate_launch_prefix(t.mixed_precision, rl.accelerate_extra_args)
    cmd.append(_find_script("ltx2_train_rl.py"))
    cmd += _rl_common_model_args(config)

    # Optimizer — from training config
    cmd += ["--learning_rate", str(t.learning_rate)]
    if t.optimizer_type:
        cmd += ["--optimizer_type", t.optimizer_type]
    if t.optimizer_args:
        cmd += ["--optimizer_args"] + _split_cli_args(t.optimizer_args)
    cmd += ["--lr_scheduler", t.lr_scheduler]
    if t.lr_warmup_steps:
        cmd += ["--lr_warmup_steps", str(t.lr_warmup_steps)]
    cmd += ["--max_grad_norm", str(t.max_grad_norm)]

    rl_loss = getattr(rl, "rl_loss", "nft") or "nft"
    _refl = rl_loss == "refl"

    # Rollout source: refl always generates inline (differentiable backprop straight through the
    # sampler — there is no cache to replay); the policy-gradient rules use offline cache replay
    # (default) or inline online generation.
    if rl.online or _refl:
        if not _refl:
            cmd.append("--rl_online")
        if rl.rl_prompts:
            cmd += ["--rl_prompts", rl.rl_prompts]
        cmd += _rl_reward_args(rl)
        cmd += ["--rl_group_size", str(rl.rl_group_size)]
        cmd += _rl_sample_dim_args(rl)
        if rl.rl_dump_cache and not _refl:
            cmd += ["--rl_dump_cache", rl.rl_dump_cache]
    else:
        cmd += ["--rl_rollout_cache", rl.rl_rollout_cache]

    # Update rule + its hyperparameters (default nft is implicit, so emit only when switched)
    if rl_loss != "nft":
        cmd += ["--rl_loss", rl_loss]
    if rl_loss == "rwr" and rl.rwr_temperature != 1.0:
        cmd += ["--rwr_temperature", str(rl.rwr_temperature)]
    if rl_loss == "dpo" and rl.dpo_beta != 5.0:
        cmd += ["--dpo_beta", str(rl.dpo_beta)]
    if _refl:
        # Differentiable-reward backprop: online-only; --nft_kl_beta weights reference-x0 MSE.
        if rl.refl_grad_steps != 1:
            cmd += ["--refl_grad_steps", str(rl.refl_grad_steps)]
        if rl.refl_renoise_samples != 1:
            cmd += ["--refl_renoise_samples", str(rl.refl_renoise_samples)]
        if rl.refl_reward_weight != 1.0:
            cmd += ["--refl_reward_weight", str(rl.refl_reward_weight)]
        if rl.refl_av:
            cmd += ["--refl_av"]
    if rl_loss == "ppo":
        # ppo is trajectory-faithful DDPO: needs the SDE-sampled trajectory (Phase A
        # --rl_sde_sampler for offline; the same flag on inline generation for online).
        if rl.online:
            cmd += ["--rl_sde_sampler"]
        if rl.ppo_clip_eps != 0.2:
            cmd += ["--ppo_clip_eps", str(rl.ppo_clip_eps)]
        if rl.rl_sde_eta != 1.0:
            cmd += ["--rl_sde_eta", str(rl.rl_sde_eta)]

    # NFT loss coefficients
    if rl.nft_beta_mix != 1.0:
        cmd += ["--nft_beta_mix", str(rl.nft_beta_mix)]
    if rl.nft_kl_beta != 1e-4:
        cmd += ["--nft_kl_beta", str(rl.nft_kl_beta)]
    if rl.nft_adv_clip_max != 5.0:
        cmd += ["--nft_adv_clip_max", str(rl.nft_adv_clip_max)]

    # Loop schedule
    if rl.rl_timesteps_per_sample != 1:
        cmd += ["--rl_timesteps_per_sample", str(rl.rl_timesteps_per_sample)]
    if rl.rl_max_steps != 0:
        cmd += ["--rl_max_steps", str(rl.rl_max_steps)]
    if rl.rl_decay_type != 1:
        cmd += ["--rl_decay_type", str(rl.rl_decay_type)]
    if rl.rl_log_vram:
        cmd.append("--rl_log_vram")

    # Output — dir from training, name from RL
    cmd += ["--output_dir", _effective_output_dir(t.output_dir)]
    if rl.output_name:
        cmd += ["--output_name", rl.output_name]

    cmd += _split_cli_args(rl.extra_args)
    return cmd


def build_cache_dino_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_dino_features.py."""
    toml_path = export_dataset_toml(config)
    c = config.caching
    t = config.training

    cmd = [
        sys.executable,
        "-u",
        _find_script("ltx2_cache_dino_features.py"),
        "--dataset_config",
        str(toml_path),
        "--dino_model",
        t.crepa_dino_model,  # Use training model setting, not caching
        "--dino_batch_size",
        str(c.dino_batch_size),
    ]

    if c.device:
        cmd += ["--device", c.device]
    if c.dino_repo_path:
        cmd += ["--dino_repo_path", c.dino_repo_path]
    if c.torch_hub_dir:
        cmd += ["--torch_hub_dir", c.torch_hub_dir]
    if c.skip_existing:
        cmd.append("--skip_existing")
    if c.atomic_cache_writes:
        cmd.append("--atomic_cache_writes")

    cmd += _split_cli_args(c.cache_dino_extra_args)
    return cmd
