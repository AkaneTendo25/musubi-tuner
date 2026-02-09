"""Convert ProjectConfig into CLI argument lists for subprocess launch."""

from __future__ import annotations

import sys
from pathlib import Path

from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.toml_export import (
    _toml_value,
    _write_slider_toml,
    _write_toml_fallback,
    build_dataset_toml_path,
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


def build_cache_latents_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_latents.py."""
    toml_path = export_dataset_toml(config)
    c = config.caching

    cmd = [
        sys.executable,
        _find_script("ltx2_cache_latents.py"),
        "--dataset_config", str(toml_path),
        "--ltx2_checkpoint", c.ltx2_checkpoint,
        "--ltx2_mode", c.ltx2_mode,
    ]

    if c.vae_dtype:
        cmd += ["--vae_dtype", c.vae_dtype]
    if c.device:
        cmd += ["--device", c.device]
    if c.skip_existing:
        cmd.append("--skip_existing")
    if c.vae_chunk_size is not None:
        cmd += ["--vae_chunk_size", str(c.vae_chunk_size)]
    if c.vae_spatial_tile_size is not None:
        cmd += ["--vae_spatial_tile_size", str(c.vae_spatial_tile_size)]

    # Audio source options
    if c.ltx2_mode in ("av", "audio"):
        cmd += ["--ltx2_audio_source", c.ltx2_audio_source]
        if c.ltx2_audio_source == "audio_files" and c.ltx2_audio_dir:
            cmd += ["--ltx2_audio_dir", c.ltx2_audio_dir]
            if c.ltx2_audio_ext:
                cmd += ["--ltx2_audio_ext", c.ltx2_audio_ext]

    return cmd


def build_cache_text_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_text_encoder_outputs.py."""
    toml_path = export_dataset_toml(config)
    c = config.caching

    cmd = [
        sys.executable,
        _find_script("ltx2_cache_text_encoder_outputs.py"),
        "--dataset_config", str(toml_path),
        "--ltx2_checkpoint", c.ltx2_checkpoint,
        "--gemma_root", c.gemma_root,
        "--ltx2_mode", c.ltx2_mode,
    ]

    if c.mixed_precision != "no":
        cmd += ["--mixed_precision", c.mixed_precision]
    if c.skip_existing:
        cmd.append("--skip_existing")
    if c.gemma_load_in_8bit:
        cmd.append("--gemma_load_in_8bit")
    if c.gemma_load_in_4bit:
        cmd.append("--gemma_load_in_4bit")
        cmd += ["--gemma_bnb_4bit_quant_type", c.gemma_bnb_4bit_quant_type]

    # Precaching
    if c.precache_sample_prompts and c.sample_prompts:
        cmd.append("--precache_sample_prompts")
        cmd += ["--sample_prompts", c.sample_prompts]
    if c.precache_preservation_prompts:
        cmd.append("--precache_preservation_prompts")
        if c.blank_preservation:
            cmd.append("--blank_preservation")
        if c.dop:
            cmd.append("--dop")
            if c.dop_class_prompt:
                cmd += ["--dop_class_prompt", c.dop_class_prompt]

    return cmd


def build_inference_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_generate_video.py."""
    s = config.inference

    cmd = [
        sys.executable,
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

    # Use accelerate launch
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision", t.mixed_precision,
        "--num_processes", "1",
        "--num_machines", "1",
        _find_script("ltx2_train_network.py"),
    ]

    # Dataset
    cmd += ["--dataset_config", str(toml_path)]

    # Model
    cmd += ["--ltx2_checkpoint", t.ltx2_checkpoint]
    if t.gemma_root:
        cmd += ["--gemma_root", t.gemma_root]
    cmd += ["--ltx2_mode", t.ltx2_mode]
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

    # LoRA
    cmd += ["--network_dim", str(t.network_dim)]
    cmd += ["--network_alpha", str(t.network_alpha)]
    cmd += ["--lora_target_preset", t.lora_target_preset]
    if t.network_args:
        cmd += ["--network_args"] + t.network_args.split()
    if t.network_weights:
        cmd += ["--network_weights", t.network_weights]
    if t.network_dropout is not None:
        cmd += ["--network_dropout", str(t.network_dropout)]
    if t.scale_weight_norms is not None:
        cmd += ["--scale_weight_norms", str(t.scale_weight_norms)]
    if t.dim_from_weights:
        cmd.append("--dim_from_weights")
    if t.base_weights:
        cmd += ["--base_weights"] + t.base_weights.split()
    if t.base_weights_multiplier:
        cmd += ["--base_weights_multiplier"] + t.base_weights_multiplier.split()

    # Optimizer
    cmd += ["--learning_rate", str(t.learning_rate)]
    cmd += ["--optimizer_type", t.optimizer_type]
    if t.optimizer_args:
        cmd += ["--optimizer_args"] + t.optimizer_args.split()
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
        cmd += ["--lr_scheduler_args"] + t.lr_scheduler_args.split()
    cmd += ["--gradient_accumulation_steps", str(t.gradient_accumulation_steps)]
    cmd += ["--max_grad_norm", str(t.max_grad_norm)]

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

    # Memory
    if t.blocks_to_swap is not None:
        cmd += ["--blocks_to_swap", str(t.blocks_to_swap)]
    if t.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
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
    if t.sample_prompts:
        cmd += ["--sample_prompts", t.sample_prompts]
    if t.use_precached_sample_prompts:
        cmd.append("--use_precached_sample_prompts")
    if t.sample_prompts_cache:
        cmd += ["--sample_prompts_cache", t.sample_prompts_cache]
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
    if t.sample_two_stage:
        cmd.append("--sample_two_stage")
    if t.sample_audio_only:
        cmd.append("--sample_audio_only")
    if t.sample_disable_flash_attn:
        cmd.append("--sample_disable_flash_attn")

    # Output
    if t.output_dir:
        cmd += ["--output_dir", t.output_dir]
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
    if t.no_convert_to_comfy:
        cmd.append("--no_convert_to_comfy")
    if t.log_with:
        cmd += ["--log_with", t.log_with]
    if t.logging_dir:
        cmd += ["--logging_dir", t.logging_dir]
    if t.wandb_run_name:
        cmd += ["--wandb_run_name", t.wandb_run_name]
    if t.resume:
        cmd += ["--resume", t.resume]
    if t.training_comment:
        cmd += ["--training_comment", t.training_comment]

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

    # GUI flag so training writes dashboard data
    cmd.append("--gui")

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

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision", t.mixed_precision,
        "--num_processes", "1",
        "--num_machines", "1",
        _find_script("ltx2_train_slider.py"),
    ]

    # Slider config
    cmd += ["--slider_config", str(slider_toml)]

    # Model — from training config
    cmd += ["--ltx2_checkpoint", t.ltx2_checkpoint]
    if t.gemma_root:
        cmd += ["--gemma_root", t.gemma_root]
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
    cmd += ["--network_dim", str(t.network_dim)]
    cmd += ["--network_alpha", str(t.network_alpha)]

    # Optimizer — from training config
    cmd += ["--learning_rate", str(t.learning_rate)]
    cmd += ["--optimizer_type", t.optimizer_type]
    if t.optimizer_args:
        cmd += ["--optimizer_args"] + t.optimizer_args.split()
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
    if t.output_dir:
        cmd += ["--output_dir", t.output_dir]
    if s.output_name:
        cmd += ["--output_name", s.output_name]
    if t.save_every_n_steps:
        cmd += ["--save_every_n_steps", str(t.save_every_n_steps)]

    return cmd


def build_cache_dino_cmd(config: ProjectConfig) -> list[str]:
    """Build CLI args for ltx2_cache_dino_features.py."""
    toml_path = export_dataset_toml(config)
    c = config.caching

    cmd = [
        sys.executable,
        _find_script("ltx2_cache_dino_features.py"),
        "--dataset_config", str(toml_path),
        "--dino_model", c.dino_model,
        "--dino_batch_size", str(c.dino_batch_size),
    ]

    if c.device:
        cmd += ["--device", c.device]
    if c.skip_existing:
        cmd.append("--skip_existing")

    return cmd
