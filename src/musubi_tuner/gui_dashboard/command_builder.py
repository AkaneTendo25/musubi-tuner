"""Convert ProjectConfig into CLI argument lists for subprocess launch."""

from __future__ import annotations

import hashlib
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
    export_dataset_toml,
)


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


def _effective_gemma_safetensors(config: ProjectConfig, explicit: str) -> str:
    return explicit or config.default_gemma_safetensors or ""


def _effective_output_dir(explicit: str) -> str:
    return explicit or get_ltx2_training_output_dir_default()


def _effective_network_module(explicit: str) -> str:
    return explicit or get_ltx2_training_network_module_default()


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


def _compile_dynamic_value(value) -> str | None:
    if value in (None, False, ""):
        return None
    if value is True:
        return "true"
    normalized = str(value).lower()
    if normalized in {"true", "false", "auto"}:
        return normalized
    return None


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

    cmd += _split_cli_args(c.cache_latents_extra_args)
    return cmd


def build_cache_text_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_text_encoder_outputs.py."""
    toml_path = export_dataset_toml(config)
    c = config.caching
    t = config.training
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, c.ltx2_checkpoint)
    gemma_safetensors = _effective_gemma_safetensors(config, c.gemma_safetensors)
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

    if c.cache_before_connector:
        cmd.append("--cache_before_connector")

    cmd += _split_cli_args(c.cache_text_extra_args)
    return cmd


def build_inference_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_generate_video.py."""
    s = config.inference
    gemma_safetensors = _effective_gemma_safetensors(config, s.gemma_safetensors)
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
    if s.reference_frames != 1:
        cmd += ["--reference_frames", str(s.reference_frames)]
    if s.sample_include_reference:
        cmd.append("--sample_include_reference")
    if s.reference_image:
        cmd += ["--reference_image", s.reference_image]
    if s.reference_video:
        cmd += ["--reference_video", s.reference_video]

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


def build_training_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for training via accelerate launch."""
    toml_path = export_dataset_toml(config)
    t = config.training
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, t.ltx2_checkpoint)
    gemma_safetensors = _effective_gemma_safetensors(config, t.gemma_safetensors)
    gemma_root = _effective_gemma_root(config, t.gemma_root, gemma_safetensors)
    network_module = _effective_network_module(t.network_module or "")
    sample_prompts = _effective_training_sample_prompts(config)
    network_args_parts = _split_cli_args(t.network_args)

    _append_network_arg(network_args_parts, "rank_dropout", t.rank_dropout)
    _append_network_arg(network_args_parts, "module_dropout", t.module_dropout)
    _append_network_arg(network_args_parts, "use_dora", t.use_dora)
    _append_network_arg(network_args_parts, "adaptive_rank", t.adaptive_rank)
    _append_network_arg(network_args_parts, "adaptive_rank_target", t.adaptive_rank_target)
    _append_network_arg(network_args_parts, "adaptive_rank_min_rank", t.adaptive_rank_min_rank)
    _append_network_arg(network_args_parts, "adaptive_rank_init_rank", t.adaptive_rank_init_rank)
    _append_network_arg(network_args_parts, "adaptive_rank_quantile", t.adaptive_rank_quantile)
    _append_network_arg(network_args_parts, "adaptive_rank_weight", t.adaptive_rank_weight)

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
    if t.awq_calibration:
        cmd.append("--awq_calibration")
        if t.awq_alpha != 0.25:
            cmd += ["--awq_alpha", str(t.awq_alpha)]
        if t.awq_num_batches != 8:
            cmd += ["--awq_num_batches", str(t.awq_num_batches)]
    if t.quantize_device:
        cmd += ["--quantize_device", t.quantize_device]

    # LoRA / Network
    cmd += ["--network_module", network_module]
    if t.network_dim is not None:
        cmd += ["--network_dim", str(t.network_dim)]
    if t.network_alpha != 1:
        cmd += ["--network_alpha", str(t.network_alpha)]
    cmd += ["--lora_target_preset", t.lora_target_preset]
    if t.train_connectors:
        cmd.append("--train_connectors")
    if network_args_parts:
        cmd += ["--network_args"] + network_args_parts
    if t.network_weights:
        cmd += ["--network_weights", t.network_weights]
    if t.network_dropout is not None:
        cmd += ["--network_dropout", str(t.network_dropout)]
    if t.scale_weight_norms is not None:
        cmd += ["--scale_weight_norms", str(t.scale_weight_norms)]
    if t.dim_from_weights:
        cmd.append("--dim_from_weights")
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
    if t.ic_lora_strategy != "auto":
        cmd += ["--ic_lora_strategy", t.ic_lora_strategy]
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
    if t.audio_lr is not None:
        cmd += ["--audio_lr", str(t.audio_lr)]
    if t.lr_args:
        cmd += ["--lr_args"] + _split_cli_args(t.lr_args)
    if t.lr_group_warmup_args:
        cmd += ["--lr_group_warmup_args"] + _split_cli_args(t.lr_group_warmup_args)
    if t.audio_dim is not None:
        cmd += ["--audio_dim", str(t.audio_dim)]
    if t.audio_alpha is not None:
        cmd += ["--audio_alpha", str(t.audio_alpha)]

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
    if t.gradient_checkpointing:
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
    if t.img_in_txt_in_offloading:
        cmd.append("--img_in_txt_in_offloading")

    # Compile
    if t.compile:
        cmd.append("--compile")
        if t.compile_backend:
            cmd += ["--compile_backend", t.compile_backend]
        if t.compile_mode:
            cmd += ["--compile_mode", t.compile_mode]
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

    # CUDA
    if t.cuda_allow_tf32:
        cmd.append("--cuda_allow_tf32")
    if t.cuda_cudnn_benchmark:
        cmd.append("--cuda_cudnn_benchmark")
    if t.cuda_memory_fraction is not None:
        cmd += ["--cuda_memory_fraction", str(t.cuda_memory_fraction)]
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
        if t.self_flow_teacher_mode != "base":
            args_parts.append(f"teacher_mode={t.self_flow_teacher_mode}")
        if t.self_flow_student_block_idx != 16:
            args_parts.append(f"student_block_idx={t.self_flow_student_block_idx}")
        if t.self_flow_teacher_block_idx != 32:
            args_parts.append(f"teacher_block_idx={t.self_flow_teacher_block_idx}")
        if t.self_flow_student_block_ratio != 0.3:
            args_parts.append(f"student_block_ratio={t.self_flow_student_block_ratio}")
        if t.self_flow_teacher_block_ratio != 0.7:
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
        if t.hfato_interpolation != "bilinear":
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
    if t.audio_metrics:
        cmd.append("--audio_metrics")
        args_parts = []
        _append_key_value_args(args_parts, t.audio_metrics_args)
        if t.audio_metrics_mel_metrics:
            args_parts.append("mel_metrics=true")
            if t.audio_metrics_mel_compute_every != 100:
                args_parts.append(f"mel_compute_every={t.audio_metrics_mel_compute_every}")
        if t.audio_metrics_clap_similarity:
            args_parts.append("clap_similarity=true")
        if t.audio_metrics_av_onset_alignment:
            args_parts.append("av_onset_alignment=true")
        if args_parts:
            cmd += ["--audio_metrics_args"] + args_parts

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

    # Audio features
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
    if t.max_data_loader_n_workers is not None:
        cmd += ["--max_data_loader_n_workers", str(t.max_data_loader_n_workers)]
    if t.persistent_data_loader_workers:
        cmd.append("--persistent_data_loader_workers")
    cmd += ["--ltx2_first_frame_conditioning_p", str(t.ltx2_first_frame_conditioning_p)]

    if getattr(t, "keyframe_endpoint_training", False):
        cmd += ["--keyframe_endpoint_training"]
        cmd += ["--keyframe_first_frame_p", str(getattr(t, "keyframe_first_frame_p", 1.0))]
        cmd += ["--keyframe_last_frame_p", str(getattr(t, "keyframe_last_frame_p", 1.0))]
        cmd += ["--keyframe_random_interior_p", str(getattr(t, "keyframe_random_interior_p", 0.0))]
        cmd += ["--keyframe_max_random_interior", str(int(getattr(t, "keyframe_max_random_interior", 0)))]
    if getattr(t, "video_anchor_training", False):
        cmd += ["--video_anchor_training"]
        cmd += ["--video_anchor_probability", str(getattr(t, "video_anchor_probability", 0.5))]
        cmd += ["--video_anchor_count", str(int(getattr(t, "video_anchor_count", 1)))]
        cmd += ["--video_anchor_strategy", str(getattr(t, "video_anchor_strategy", "endpoints_random"))]

    cmd += _split_cli_args(t.extra_args)
    return cmd


def build_full_finetune_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for full-parameter LTX-2 fine-tuning via accelerate launch."""
    toml_path = export_dataset_toml(config)
    t = config.full_finetune
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, t.ltx2_checkpoint)
    gemma_safetensors = _effective_gemma_safetensors(config, t.gemma_safetensors)
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
    if t.fp8_keep_blocks:
        cmd += ["--fp8_keep_blocks", t.fp8_keep_blocks]
    if t.flash_attn:
        cmd.append("--flash_attn")
    if t.flash3:
        cmd.append("--flash3")
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
    if t.gradient_checkpointing:
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
    gemma_safetensors = _effective_gemma_safetensors(config, t.gemma_safetensors)
    gemma_root = _effective_gemma_root(config, t.gemma_root, gemma_safetensors)

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
    if s.guidance_strength != 1.0:
        cmd += ["--guidance_strength", str(s.guidance_strength)]
    if s.sample_slider_range != "-2,-1,0,1,2":
        cmd += ["--sample_slider_range", s.sample_slider_range]

    # LoRA — from training config
    if t.network_dim is not None:
        cmd += ["--network_dim", str(t.network_dim)]
    if t.network_alpha != 1:
        cmd += ["--network_alpha", str(t.network_alpha)]

    # Optimizer — from training config
    cmd += ["--learning_rate", str(t.learning_rate)]
    if t.optimizer_type:
        cmd += ["--optimizer_type", t.optimizer_type]
    if t.optimizer_args:
        cmd += ["--optimizer_args"] + _split_cli_args(t.optimizer_args)
    cmd += ["--gradient_accumulation_steps", str(t.gradient_accumulation_steps)]
    cmd += ["--max_grad_norm", str(t.max_grad_norm)]

    # Schedule — slider override for steps
    cmd += ["--max_train_steps", str(s.max_train_steps)]
    if t.seed is not None:
        cmd += ["--seed", str(t.seed)]

    # Memory — from training config
    if t.blocks_to_swap is not None:
        cmd += ["--blocks_to_swap", str(t.blocks_to_swap)]
    if t.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    # Output — dir from training, name from slider
    cmd += ["--output_dir", _effective_output_dir(t.output_dir)]
    if s.output_name:
        cmd += ["--output_name", s.output_name]
    if t.save_every_n_steps:
        cmd += ["--save_every_n_steps", str(t.save_every_n_steps)]

    cmd += _split_cli_args(s.extra_args)
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
