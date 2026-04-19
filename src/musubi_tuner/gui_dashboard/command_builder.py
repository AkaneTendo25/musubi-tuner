"""Convert ProjectConfig into CLI argument lists for subprocess launch."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from musubi_tuner.gui_dashboard.cli_defaults import (
    get_ltx2_training_network_module_default,
    get_ltx2_training_output_dir_default,
)
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.toml_export import (
    _write_slider_toml,
    build_slider_toml_path,
    export_dataset_toml,
)

DEFAULT_MODEL_DIR_NAME = "models"
DEFAULT_LTX2_CHECKPOINT_NAME = "ltx-2-19b-dev.safetensors"
DEFAULT_GEMMA_ROOT_NAME = "gemma-3-12b-it"


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


def _effective_caching_sample_prompts(config: ProjectConfig) -> str:
    caching = config.caching
    if caching.sample_prompts:
        return caching.sample_prompts
    return _effective_training_sample_prompts(config)


def _split_cli_args(raw: str) -> list[str]:
    if not raw:
        return []
    return shlex.split(raw, posix=False)


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


def build_cache_latents_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_latents.py."""
    toml_path = export_dataset_toml(config)
    c = config.caching
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, c.ltx2_checkpoint)
    sample_prompts = _effective_caching_sample_prompts(config)

    cmd = [
        sys.executable,
        "-u",
        _find_script("ltx2_cache_latents.py"),
        "--dataset_config", str(toml_path),
        "--ltx2_checkpoint", ltx2_checkpoint,
        "--ltx2_mode", c.ltx2_mode,
    ]

    if c.vae_dtype:
        cmd += ["--vae_dtype", c.vae_dtype]
    if c.device:
        cmd += ["--device", c.device]
    if c.skip_existing:
        cmd.append("--skip_existing")
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
        if c.audio_only_sequence_resolution != 64:
            cmd += ["--audio_only_sequence_resolution", str(c.audio_only_sequence_resolution)]

    # I2V latent precaching
    if c.precache_sample_latents and sample_prompts:
        cmd.append("--precache_sample_latents")
        cmd += ["--sample_prompts", sample_prompts]
        if c.sample_latents_cache:
            cmd += ["--sample_latents_cache", c.sample_latents_cache]

    if c.quantize_device:
        cmd += ["--quantize_device", c.quantize_device]
    if c.save_dataset_manifest:
        cmd += ["--save_dataset_manifest", c.save_dataset_manifest]

    return cmd


def build_cache_text_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_text_encoder_outputs.py."""
    toml_path = export_dataset_toml(config)
    c = config.caching
    ltx2_checkpoint = _effective_ltx2_checkpoint(config, c.ltx2_checkpoint)
    gemma_safetensors = _effective_gemma_safetensors(config, c.gemma_safetensors)
    gemma_root = _effective_gemma_root(config, c.gemma_root, gemma_safetensors)
    sample_prompts = _effective_caching_sample_prompts(config)

    cmd = [
        sys.executable,
        "-u",
        _find_script("ltx2_cache_text_encoder_outputs.py"),
        "--dataset_config", str(toml_path),
        "--ltx2_checkpoint", ltx2_checkpoint,
        "--ltx2_mode", c.ltx2_mode,
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

    # Precaching
    if c.precache_sample_prompts and sample_prompts:
        cmd.append("--precache_sample_prompts")
        cmd += ["--sample_prompts", sample_prompts]
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

    return cmd


def build_inference_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_generate_video.py."""
    s = config.inference

    cmd = [
        sys.executable,
        "-u",
        _find_script("ltx2_generate_video.py"),
        "--ltx2_checkpoint", s.ltx2_checkpoint,
        "--gemma_root", s.gemma_root,
        "--ltx2_mode", s.ltx2_mode,
    ]

    # LoRA
    if s.lora_weight:
        cmd += ["--lora_weight", s.lora_weight]
        cmd += ["--lora_multiplier", str(s.lora_multiplier)]

    # Prompt
    if s.prompt:
        cmd += ["--prompt", s.prompt]
    if s.negative_prompt:
        cmd += ["--negative_prompt", s.negative_prompt]
    if s.from_file:
        cmd += ["--from_file", s.from_file]

    # Sampling params
    cmd += ["--height", str(s.height)]
    cmd += ["--width", str(s.width)]
    cmd += ["--frame_count", str(s.frame_count)]
    cmd += ["--frame_rate", str(s.frame_rate)]
    cmd += ["--sample_steps", str(s.sample_steps)]
    cmd += ["--guidance_scale", str(s.guidance_scale)]
    if s.cfg_scale is not None:
        cmd += ["--cfg_scale", str(s.cfg_scale)]
    cmd += ["--discrete_flow_shift", str(s.discrete_flow_shift)]
    if s.seed is not None:
        cmd += ["--seed", str(s.seed)]

    # Precision
    if s.mixed_precision != "no":
        cmd += ["--mixed_precision", s.mixed_precision]
    cmd += ["--attn_mode", s.attn_mode]
    if s.fp8_base:
        cmd.append("--fp8_base")
    if s.fp8_scaled:
        cmd.append("--fp8_scaled")

    # Gemma quantization
    if s.gemma_load_in_8bit:
        cmd.append("--gemma_load_in_8bit")
    if s.gemma_load_in_4bit:
        cmd.append("--gemma_load_in_4bit")

    # Memory
    if s.offloading:
        cmd.append("--offloading")
    if s.blocks_to_swap is not None:
        cmd += ["--blocks_to_swap", str(s.blocks_to_swap)]

    # Output
    if s.output_dir:
        cmd += ["--output_dir", s.output_dir]
    if s.output_name:
        cmd += ["--output_name", s.output_name]

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
    _append_network_arg(network_args_parts, "adaptive_rank", t.adaptive_rank)
    _append_network_arg(network_args_parts, "adaptive_rank_target", t.adaptive_rank_target)
    _append_network_arg(network_args_parts, "adaptive_rank_min_rank", t.adaptive_rank_min_rank)
    _append_network_arg(network_args_parts, "adaptive_rank_init_rank", t.adaptive_rank_init_rank)
    _append_network_arg(network_args_parts, "adaptive_rank_quantile", t.adaptive_rank_quantile)
    _append_network_arg(network_args_parts, "adaptive_rank_weight", t.adaptive_rank_weight)

    # Use accelerate launch
    cmd = [
        sys.executable, "-u", "-m", "accelerate.commands.launch",
        "--mixed_precision", t.mixed_precision,
        "--num_processes", "1",
        "--num_machines", "1",
        _find_script("ltx2_train_network.py"),
    ]

    # Dataset
    if t.dataset_manifest:
        cmd += ["--dataset_manifest", t.dataset_manifest]
    else:
        cmd += ["--dataset_config", str(toml_path)]

    # Model
    cmd += ["--ltx2_checkpoint", ltx2_checkpoint]
    if gemma_root:
        cmd += ["--gemma_root", gemma_root]
    if gemma_safetensors:
        cmd += ["--gemma_safetensors", gemma_safetensors]
    cmd += ["--ltx2_mode", t.ltx2_mode]
    if t.ltx_version != "2.0":
        cmd += ["--ltx_version", t.ltx_version]
    if t.ltx_version_check_mode != "warn":
        cmd += ["--ltx_version_check_mode", t.ltx_version_check_mode]
    if t.fp8_base:
        cmd.append("--fp8_base")
    if t.fp8_scaled:
        cmd.append("--fp8_scaled")
    if t.flash_attn:
        cmd.append("--flash_attn")
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
    if t.gemma_bnb_4bit_disable_double_quant:
        cmd.append("--gemma_bnb_4bit_disable_double_quant")
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
        cmd.append("--no-save_original_lora")
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
    cmd += ["--max_grad_norm", str(t.max_grad_norm)]
    if t.audio_lr is not None:
        cmd += ["--audio_lr", str(t.audio_lr)]
    if t.lr_args:
        cmd += ["--lr_args"] + _split_cli_args(t.lr_args)
    if t.audio_dim is not None:
        cmd += ["--audio_dim", str(t.audio_dim)]
    if t.audio_alpha is not None:
        cmd += ["--audio_alpha", str(t.audio_alpha)]

    # Schedule
    if t.max_train_epochs is not None:
        cmd += ["--max_train_epochs", str(t.max_train_epochs)]
    else:
        cmd += ["--max_train_steps", str(t.max_train_steps)]
    cmd += ["--timestep_sampling", t.timestep_sampling]
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
    if t.preserve_distribution_shape:
        cmd.append("--preserve_distribution_shape")
    if t.num_timestep_buckets is not None:
        cmd += ["--num_timestep_buckets", str(t.num_timestep_buckets)]

    # Memory
    if t.blocks_to_swap is not None:
        cmd += ["--blocks_to_swap", str(t.blocks_to_swap)]
    if t.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if t.gradient_checkpointing_cpu_offload:
        cmd.append("--gradient_checkpointing_cpu_offload")
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
        if t.compile_dynamic:
            cmd.append("--compile_dynamic")
        if t.compile_fullgraph:
            cmd.append("--compile_fullgraph")
        if t.compile_cache_size_limit is not None:
            cmd += ["--compile_cache_size_limit", str(t.compile_cache_size_limit)]

    # CUDA
    if t.cuda_allow_tf32:
        cmd.append("--cuda_allow_tf32")
    if t.cuda_cudnn_benchmark:
        cmd.append("--cuda_cudnn_benchmark")
    if t.cuda_memory_fraction is not None:
        cmd += ["--cuda_memory_fraction", str(t.cuda_memory_fraction)]

    # Sampling
    if t.sample_every_n_steps:
        cmd += ["--sample_every_n_steps", str(t.sample_every_n_steps)]
    if t.sample_every_n_epochs:
        cmd += ["--sample_every_n_epochs", str(t.sample_every_n_epochs)]
    if sample_prompts:
        cmd += ["--sample_prompts", sample_prompts]
    if t.use_precached_sample_prompts:
        cmd.append("--use_precached_sample_prompts")
    if t.sample_prompts_cache:
        cmd += ["--sample_prompts_cache", t.sample_prompts_cache]
    if t.use_precached_sample_latents:
        cmd.append("--use_precached_sample_latents")
    if t.sample_latents_cache:
        cmd += ["--sample_latents_cache", t.sample_latents_cache]
    cmd += ["--height", str(t.height)]
    cmd += ["--width", str(t.width)]
    cmd += ["--sample_num_frames", str(t.sample_num_frames)]
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
    if t.logging_dir:
        cmd += ["--logging_dir", t.logging_dir]
    if t.log_prefix:
        cmd += ["--log_prefix", t.log_prefix]
    if t.log_tracker_name:
        cmd += ["--log_tracker_name", t.log_tracker_name]
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
        if args_parts:
            cmd += ["--crepa_args"] + args_parts

    # Self-Flow
    if t.self_flow:
        cmd.append("--self_flow")
        args_parts = []
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
        if t.hfato_scale_factor != 0.5:
            args_parts.append(f"scale_factor={t.hfato_scale_factor}")
        if t.hfato_interpolation != "bilinear":
            args_parts.append(f"interpolation={t.hfato_interpolation}")
        if t.hfato_probability != 1.0:
            args_parts.append(f"probability={t.hfato_probability}")
        if args_parts:
            cmd += ["--hfato_args"] + args_parts

    # Preservation
    if t.blank_preservation:
        cmd.append("--blank_preservation")
        args_parts = []
        if t.blank_preservation_multiplier != 1.0:
            args_parts.append(f"multiplier={t.blank_preservation_multiplier}")
        if args_parts:
            cmd += ["--blank_preservation_args"] + args_parts
    if t.dop:
        cmd.append("--dop")
        args_parts = []
        if t.dop_class:
            args_parts.append(f"class={t.dop_class}")
        if t.dop_multiplier != 1.0:
            args_parts.append(f"multiplier={t.dop_multiplier}")
        if args_parts:
            cmd += ["--dop_args"] + args_parts
    if t.prior_divergence:
        cmd.append("--prior_divergence")
        args_parts = []
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
        if t.tarp_window_multiplier != 3:
            cmd += ["--tarp_args", f"window_multiplier={t.tarp_window_multiplier}"]
    if t.dcr:
        cmd.append("--dcr")
        if not t.dcr_reference_detach:
            cmd += ["--dcr_args", "reference_detach=false"]

    # Audio Metrics
    if t.audio_metrics:
        cmd.append("--audio_metrics")
        args_parts = []
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
        if t.audio_dop_multiplier != 0.5:
            cmd += ["--audio_dop_args", f"multiplier={t.audio_dop_multiplier}"]
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
    cmd += ["--max_data_loader_n_workers", str(t.max_data_loader_n_workers)]
    if t.persistent_data_loader_workers:
        cmd.append("--persistent_data_loader_workers")
    cmd += ["--ltx2_first_frame_conditioning_p", str(t.ltx2_first_frame_conditioning_p)]

    return cmd


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

    cmd = [
        sys.executable, "-u", "-m", "accelerate.commands.launch",
        "--mixed_precision", t.mixed_precision,
        "--num_processes", "1",
        "--num_machines", "1",
        _find_script("ltx2_train_slider.py"),
    ]

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
    if t.flash_attn:
        cmd.append("--flash_attn")
    if t.gemma_load_in_8bit:
        cmd.append("--gemma_load_in_8bit")
    if t.gemma_load_in_4bit:
        cmd.append("--gemma_load_in_4bit")

    # Text mode latent dimensions — slider-specific
    if s.mode == "text":
        cmd += ["--latent_frames", str(s.latent_frames)]
        cmd += ["--latent_height", str(s.latent_height)]
        cmd += ["--latent_width", str(s.latent_width)]

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
        "--dataset_config", str(toml_path),
        "--dino_model", t.crepa_dino_model,  # Use training model setting, not caching
        "--dino_batch_size", str(c.dino_batch_size),
    ]

    if c.device:
        cmd += ["--device", c.device]
    if c.skip_existing:
        cmd.append("--skip_existing")

    return cmd
