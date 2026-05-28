"""Pydantic v2 models for GUI dashboard project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from musubi_tuner.gui_dashboard.cli_defaults import (
    get_ltx2_training_network_module_default,
    get_ltx2_training_output_dir_default,
)
from musubi_tuner.model_defaults import default_gemma_root_path, default_ltx2_checkpoint_path


SamplingPreset = Literal["legacy", "defaults", "ltx20", "ltx23", "ltx23_hq", "distilled_two_stage"]
SampleSigmaSchedule = Literal["auto", "ltx", "ltx23_distilled"]
SampleSampler = Literal["auto", "euler", "res_2s"]
LTX2BoundaryCodec = Literal["none", "int8", "int4"]
LTX2RemoteActivationCodec = Literal["none", "int8", "int4", "aq-int8", "aq-int4"]
LTX2RemoteAqKeyMode = Literal["sample", "sample_timestep", "sample_timestep_noise", "off"]
LTX2RemoteTrainableScope = Literal["auto", "lora", "blocks"]


class GeneralConfig(BaseModel):
    enable_bucket: bool = True
    bucket_no_upscale: bool = True


class DatasetEntry(BaseModel):
    type: Literal["video", "image", "audio"] = "video"
    directory: str = ""
    cache_directory: str = ""
    reference_cache_directory: str = ""
    extra_reference_cache_directories: str = ""
    reference_frames: Optional[int] = None
    reference_audio_cache_directory: str = ""
    extra_reference_audio_cache_directories: str = ""
    control_directory: str = ""
    extra_control_directories: str = ""
    reference_audio_directory: str = ""
    extra_reference_audio_directories: str = ""
    loss_mask_directory: str = ""
    default_loss_mask_path: str = ""
    loss_mask_use_alpha: bool = False
    loss_mask_invert: bool = False
    # Latent guides (directory-based, one guide of each type per item)
    latent_idx_guide_directory: str = ""
    latent_idx_guide_cache_directory: str = ""
    latent_idx_guide_frame_idx: int = 0
    latent_idx_guide_strength: float = 1.0
    keyframe_guide_directory: str = ""
    keyframe_guide_cache_directory: str = ""
    keyframe_guide_frame_idx: int = -1
    keyframe_guide_strength: float = 1.0
    # Multi-keyframe extras (semicolon-separated to keep the project JSON flat).
    # Each list must have the same number of entries; first entry corresponds to
    # the first extra keyframe. Empty falls back to single-keyframe behavior.
    keyframe_guide_extra_directories: str = ""
    keyframe_guide_extra_cache_directories: str = ""
    keyframe_guide_extra_frame_idxs: str = ""
    keyframe_guide_extra_strengths: str = ""
    jsonl_file: str = ""
    resolution_w: int = 768
    resolution_h: int = 512
    batch_size: int = 1
    num_repeats: int = 1
    caption_extension: str = ".txt"
    caption_field: str = ""
    # video-specific
    target_frames: int = 33
    frame_extraction: Literal["head", "chunk", "slide", "uniform", "full"] = "head"
    frame_sample: Optional[int] = None
    max_frames: Optional[int] = None
    frame_stride: Optional[int] = None
    source_fps: Optional[float] = None
    target_fps: Optional[float] = None


class DatasetConfig(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    datasets: list[DatasetEntry] = Field(default_factory=list)
    validation_datasets: list[DatasetEntry] = Field(default_factory=list)


class CachingConfig(BaseModel):
    ltx2_checkpoint: str = ""
    gemma_root: str = ""
    gemma_safetensors: str = ""
    ltx2_text_encoder_checkpoint: str = ""
    ltx2_mode: Literal["video", "av", "audio"] = "video"
    vae_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None
    device: Optional[str] = None
    skip_existing: bool = False
    atomic_cache_writes: bool = False
    keep_cache: bool = False
    num_workers: Optional[int] = None
    # VAE tiling
    vae_chunk_size: Optional[int] = None
    vae_spatial_tile_size: Optional[int] = None
    vae_spatial_tile_overlap: Optional[int] = None
    vae_temporal_tile_size: Optional[int] = None
    vae_temporal_tile_overlap: Optional[int] = None
    # Gemma quantization
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    gemma_load_in_8bit: bool = False
    gemma_load_in_4bit: bool = False
    gemma_bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    gemma_bnb_4bit_disable_double_quant: bool = False
    gemma_bnb_4bit_compute_dtype: Literal["auto", "fp16", "bf16", "fp32"] = "auto"
    gemma_fp8_weight_offload: bool = True
    # Text encoder precaching
    precache_sample_prompts: bool = False
    sample_prompts: str = ""
    sample_prompts_cache: str = ""
    precache_preservation_prompts: bool = False
    preservation_prompts_cache: str = ""
    # VAE I2V latent precaching
    precache_sample_latents: bool = False
    sample_latents_cache: str = ""
    blank_preservation: bool = False
    dop: bool = False
    dop_class_prompt: str = ""
    # Reference (V2V)
    reference_frames: int = 1
    reference_downscale: int = 1
    # Audio
    ltx2_audio_source: Literal["video", "audio_files"] = "video"
    ltx2_audio_dir: str = ""
    ltx2_audio_ext: str = ".wav"
    ltx2_audio_dtype: Optional[str] = None
    preserve_audio_timing: bool = False
    audio_video_latent_channels: Optional[int] = None
    audio_video_latent_dtype: Optional[str] = None
    audio_only_target_resolution: Optional[int] = None
    audio_only_target_fps: Optional[float] = None
    audio_only_sequence_resolution: int = 64
    # DINOv2 feature caching (for CREPA dino mode - model selection in training.crepa_dino_model)
    dino_batch_size: int = 16
    dino_repo_path: str = ""
    torch_hub_dir: str = ""
    # Connector LoRA
    cache_before_connector: bool = False
    # Dataset manifest
    save_dataset_manifest: str = ""
    # Raw CLI passthroughs, scoped to the individual cache commands.
    cache_latents_extra_args: str = ""
    cache_text_extra_args: str = ""
    cache_dino_extra_args: str = ""


class TrainingConfig(BaseModel):
    # Model
    ltx2_checkpoint: str = ""
    gemma_root: str = ""
    gemma_safetensors: str = ""
    config_file: str = ""
    dataset_config: str = ""
    ltx2_mode: Literal["video", "av", "audio"] = "video"
    ltx_version: Literal["2.0", "2.3"] = "2.3"
    ltx_version_check_mode: Literal["off", "warn", "error"] = "warn"
    fp8_base: bool = False
    fp8_scaled: bool = False
    fp8_keep_blocks: str = ""
    flash_attn: bool = False
    flash3: bool = False
    sdpa: bool = False
    sage_attn: bool = False
    xformers: bool = False
    gemma_load_in_8bit: bool = False
    gemma_load_in_4bit: bool = False
    gemma_bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    gemma_bnb_4bit_disable_double_quant: bool = False
    gemma_bnb_use_local_rank: bool = False
    gemma_fp8_weight_offload: bool = True
    ltx2_audio_only_model: bool = False
    vae_dtype: Optional[str] = None

    # Quantization
    nf4_base: bool = False
    nf4_block_size: int = 32
    loftq_init: bool = False
    loftq_iters: int = 2
    fp8_w8a8: bool = False
    w8a8_mode: Literal["int8", "fp8"] = "int8"
    awq_calibration: bool = False
    awq_alpha: float = 0.25
    awq_num_batches: int = 8
    quantize_device: Optional[str] = None

    # LoRA / Network
    network_module: Optional[str] = None
    network_dim: Optional[int] = None
    network_alpha: float = 1.0
    lora_target_preset: Literal[
        "t2v",
        "v2v",
        "video_sa",
        "video_sa_ff",
        "video_sa_ca_ff",
        "audio",
        "audio_v2a",
        "audio_ref_ic",
        "av_ic",
        "video_ref_only_av",
        "full",
        "lycoris",
    ] = "t2v"
    network_args: str = ""
    network_weights: str = ""
    network_dropout: Optional[float] = None
    scale_weight_norms: Optional[float] = None
    dim_from_weights: bool = False
    base_weights: str = ""
    base_weights_multiplier: str = ""
    lycoris_config: str = ""
    lycoris_quantized_base_check_mode: Literal["off", "warn", "error"] = "warn"
    init_lokr_norm: Optional[float] = None
    use_dora: bool = False
    rank_dropout: Optional[float] = None
    module_dropout: Optional[float] = None
    adaptive_rank: bool = False
    adaptive_rank_target: Optional[int] = None
    adaptive_rank_min_rank: Optional[int] = None
    adaptive_rank_init_rank: Optional[int] = None
    adaptive_rank_quantile: Optional[float] = None
    adaptive_rank_weight: Optional[float] = None
    caption_dropout_rate: float = 0.0
    video_caption_dropout_rate: float = 0.0
    audio_caption_dropout_rate: float = 0.0
    train_connectors: bool = False
    save_original_lora: bool = True
    ic_lora_strategy: Literal["auto", "none", "v2v", "audio_ref_ic", "av_ic", "video_ref_only_av"] = "auto"
    av_cross_attention_mode: Literal["both", "a2v_only", "v2a_only", "none"] = "both"
    av_multi_ref: bool = False
    audio_ref_use_negative_positions: bool = False
    audio_ref_mask_cross_attention_to_reference: bool = False
    audio_ref_mask_reference_from_text_attention: bool = False
    audio_ref_identity_guidance_scale: float = 0.0
    av_bimodal_cfg: bool = False
    av_bimodal_scale: float = 3.0

    # Optimizer
    learning_rate: float = 2e-6
    optimizer_type: str = ""
    optimizer_args: str = ""
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: Optional[int] = 0
    lr_scheduler_num_cycles: Optional[int] = 1
    lr_scheduler_power: Optional[float] = 1.0
    lr_scheduler_min_lr_ratio: Optional[float] = None
    lr_scheduler_type: str = ""
    lr_scheduler_args: str = ""
    lr_scheduler_timescale: Optional[int] = None
    gradient_accumulation_steps: int = 1
    accumulation_group_by: Literal["none", "frames", "bucket", "dataset"] = "none"
    accumulation_group_remainder: Literal["drop", "pad", "allow_mixed"] = "drop"
    max_grad_norm: float = 1.0
    audio_lr: Optional[float] = None
    lr_args: str = ""
    lr_group_warmup_args: str = ""
    audio_dim: Optional[int] = None
    audio_alpha: Optional[float] = None

    # Schedule
    max_train_steps: int = 1600
    max_train_epochs: Optional[int] = None
    timestep_sampling: str = "sigma"
    discrete_flow_shift: float = 1.0
    weighting_scheme: str = "none"
    seed: Optional[int] = None
    guidance_scale: Optional[float] = 1.0
    sigmoid_scale: Optional[float] = 1.0
    logit_mean: Optional[float] = 0.0
    logit_std: Optional[float] = 1.0
    mode_scale: Optional[float] = 1.29
    min_timestep: Optional[float] = None
    max_timestep: Optional[float] = None

    # Advanced timestep
    shifted_logit_mode: Optional[str] = None
    shifted_logit_eps: float = 1e-3
    shifted_logit_uniform_prob: float = 0.1
    shifted_logit_shift: Optional[float] = None
    shifted_logit_clamp_auto_shift: bool = False
    shifted_logit_min_shift: float = 0.95
    shifted_logit_max_shift: float = 2.05
    preserve_distribution_shape: bool = False
    num_timestep_buckets: Optional[int] = None
    show_timesteps: Optional[Literal["image", "console"]] = None
    log_timestep_distribution_tensorboard: bool = True
    log_timestep_distribution_interval: int = 100

    # Memory
    blocks_to_swap: Optional[int] = None
    gradient_checkpointing: bool = False
    gradient_checkpointing_cpu_offload: bool = False
    split_attn: bool = False
    split_attn_target: Optional[str] = None
    split_attn_mode: Optional[str] = None
    split_attn_chunk_size: Optional[int] = None
    blockwise_checkpointing: bool = False
    blocks_to_checkpoint: Optional[int] = None
    ltx2_model_parallel: bool = False
    ltx2_model_parallel_devices: str = ""
    ltx2_model_parallel_splits: str = ""
    ltx2_mp_profile_transfers: bool = False
    ltx2_mp_profile_log_every: int = 20
    ltx2_mp_activation_codec: LTX2BoundaryCodec = "none"
    ltx2_mp_grad_codec: LTX2BoundaryCodec = "none"
    ltx2_mp_int8_block_size: int = 256
    ltx2_remote_stage: bool = False
    ltx2_remote_stage_host: str = "127.0.0.1"
    ltx2_remote_stage_port: int = 7788
    ltx2_remote_stage_split: int = -1
    ltx2_remote_stage_specs: str = ""
    ltx2_remote_stage_timeout: float = 600.0
    ltx2_remote_stage_codec: LTX2RemoteActivationCodec = "none"
    ltx2_remote_stage_grad_codec: LTX2BoundaryCodec = "none"
    ltx2_remote_stage_int8_block_size: int = 256
    ltx2_remote_stage_metadata_cache: bool = True
    ltx2_remote_stage_metadata_cache_size: int = 8
    ltx2_remote_stage_aq_key_mode: LTX2RemoteAqKeyMode = "sample"
    ltx2_remote_stage_aq_stochastic: bool = True
    ltx2_remote_stage_aq_cache_size: int = 0
    ltx2_remote_stage_trainable: bool = False
    ltx2_remote_stage_trainable_scope: LTX2RemoteTrainableScope = "auto"
    ltx2_remote_stage_learning_rate: Optional[float] = None
    ltx2_remote_stage_weight_decay: float = 0.01
    ltx2_remote_stage_max_grad_norm: float = 0.0
    ltx2_remote_stage_checkpoint_dir: str = ""
    ltx2_remote_stage_prune_local_blocks: bool = False
    mixed_precision: str = "no"
    full_fp16: bool = False
    full_bf16: bool = False
    ffn_chunk_target: Optional[str] = None
    ffn_chunk_size: int = 0
    use_pinned_memory_for_block_swap: bool = False
    img_in_txt_in_offloading: bool = False

    # Compile
    compile: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    compile_dynamic: bool | str | None = False
    compile_fullgraph: bool = False
    compile_cache_size_limit: Optional[int] = None
    dynamo_backend: str = "NO"
    dynamo_mode: Optional[Literal["default", "reduce-overhead", "max-autotune"]] = None
    dynamo_fullgraph: bool = False
    dynamo_dynamic: bool = False

    # CUDA
    cuda_allow_tf32: bool = False
    cuda_cudnn_benchmark: bool = False
    cuda_memory_fraction: Optional[float] = None
    disable_numpy_memmap: bool = False
    ddp_timeout: Optional[int] = None
    ddp_gradient_as_bucket_view: bool = False
    ddp_static_graph: bool = False
    ddp_find_unused_parameters: bool = False

    # Sampling
    sample_every_n_steps: Optional[int] = None
    sample_every_n_epochs: Optional[int] = None
    sample_prompts: str = ""
    sample_prompts_text: str = ""
    precache_sample_prompts: bool = False
    use_precached_sample_prompts: bool = False
    sample_prompts_cache: str = ""
    use_precached_sample_latents: bool = False
    sample_latents_cache: str = ""
    caption_field: str = ""
    sample_sampling_preset: SamplingPreset = "defaults"
    sample_sigma_schedule: SampleSigmaSchedule = "auto"
    sample_sampler: SampleSampler = "auto"
    sample_use_default_negative_prompt: Optional[bool] = None
    height: Optional[int] = None
    width: Optional[int] = None
    sample_num_frames: Optional[int] = None
    video_cfg_scale: Optional[float] = None
    audio_cfg_scale: Optional[float] = None
    video_modality_scale: Optional[float] = None
    audio_modality_scale: Optional[float] = None
    stg_scale: Optional[float] = None
    stg_blocks: str = ""
    stg_mode: Optional[Literal["video", "audio", "both"]] = None
    rescale_scale: Optional[float] = None
    video_rescale_scale: Optional[float] = None
    audio_rescale_scale: Optional[float] = None
    sample_with_offloading: bool = False
    sample_merge_audio: bool = False
    sample_disable_audio: bool = False
    sample_at_first: bool = False
    sample_tiled_vae: bool = False
    sample_vae_tile_size: Optional[int] = 512
    sample_vae_tile_overlap: Optional[int] = 64
    sample_vae_temporal_tile_size: Optional[int] = 0
    sample_vae_temporal_tile_overlap: Optional[int] = 8
    sample_two_stage: bool = False
    spatial_upsampler_path: str = ""
    distilled_lora_path: str = ""
    sample_stage2_steps: int = 3
    sample_stage1_distilled_lora_multiplier: Optional[float] = None
    sample_stage2_distilled_lora_multiplier: Optional[float] = None
    sample_audio_only: bool = False
    sample_disable_flash_attn: bool = False
    sample_i2v_token_timestep_mask: bool = True
    sample_audio_subprocess: bool = True
    sample_include_reference: bool = False
    reference_downscale: int = 1
    reference_frames: int = 1

    # Validation
    validate_every_n_steps: Optional[int] = None
    validate_every_n_epochs: Optional[int] = None
    offload_optimizer_during_validation: bool = False

    # Output
    output_dir: str = ""
    output_name: str = "ltx2_lora"
    save_every_n_epochs: Optional[int] = None
    save_every_n_steps: Optional[int] = None
    save_last_n_epochs: Optional[int] = None
    save_last_n_steps: Optional[int] = None
    save_last_n_epochs_state: Optional[int] = None
    save_last_n_steps_state: Optional[int] = None
    save_state: bool = False
    save_state_on_train_end: bool = False
    save_checkpoint_metadata: bool = False
    no_metadata: bool = False
    no_convert_to_comfy: bool = False
    log_with: Optional[str] = None
    logging_dir: str = ""
    log_prefix: str = ""
    log_tracker_name: str = ""
    log_tracker_config: str = ""
    log_config: bool = False
    wandb_run_name: str = ""
    wandb_api_key: str = ""
    log_cuda_memory_every_n_steps: Optional[int] = None
    resume: str = ""
    autoresume: bool = False
    reset_optimizer: bool = False
    reset_optimizer_params: bool = False
    reset_dataloader: bool = False
    training_comment: str = ""
    loss_type: Literal["mse", "mae", "l1", "huber", "smooth_l1"] = "mse"
    huber_delta: float = 1.0

    # Metadata
    metadata_title: str = ""
    metadata_author: str = ""
    metadata_description: str = ""
    metadata_license: str = ""
    metadata_tags: str = ""
    metadata_reso: str = ""
    metadata_arch: str = ""

    # HuggingFace upload
    huggingface_repo_id: str = ""
    huggingface_repo_type: str = ""
    huggingface_path_in_repo: str = ""
    huggingface_token: str = ""
    huggingface_repo_visibility: str = ""
    save_state_to_huggingface: bool = False
    resume_from_huggingface: bool = False
    async_upload: bool = False

    # Dataset
    dataset_manifest: str = ""

    # Preservation
    blank_preservation: bool = False
    blank_preservation_args: str = ""
    blank_preservation_multiplier: float = 1.0
    dop: bool = False
    dop_args: str = ""
    dop_class: str = ""
    dop_multiplier: float = 1.0
    prior_divergence: bool = False
    prior_divergence_args: str = ""
    prior_divergence_multiplier: float = 0.1
    use_precached_preservation: bool = False
    preservation_prompts_cache: str = ""

    # TARP / DCR (arXiv:2603.18600)
    tarp: bool = False
    tarp_args: str = ""
    tarp_window_multiplier: int = 3
    dcr: bool = False
    dcr_args: str = ""
    dcr_reference_detach: bool = True
    av_cross_grad_surgery: bool = False
    av_cross_grad_surgery_args: str = ""
    av_attention_loss_weighting: bool = False
    av_attention_loss_max: float = 1.5
    av_attention_loss_warmup_steps: int = 400

    # Audio Metrics
    audio_metrics: bool = False
    audio_metrics_args: str = ""
    audio_metrics_mel_metrics: bool = False
    audio_metrics_mel_compute_every: int = 100
    audio_metrics_clap_similarity: bool = False
    audio_metrics_av_onset_alignment: bool = False

    # Token routing
    tread: bool = False
    tread_args: str = ""
    tread_target: Literal["video", "audio", "both"] = "video"
    tread_selection_ratio: float = 0.5
    tread_start_layer_idx: Optional[int] = None
    tread_end_layer_idx: Optional[int] = None

    # Differential Guidance
    differential_guidance: bool = False
    differential_guidance_scale: float = 3.0

    # CREPA
    crepa: bool = False
    crepa_args: str = ""
    crepa_mode: Literal["backbone", "dino"] = "backbone"
    crepa_student_block_idx: int = 16
    crepa_teacher_block_idx: int = 32
    crepa_dino_model: Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"] = "dinov2_vitb14"
    crepa_lambda: float = 0.1
    crepa_tau: float = 1.0
    crepa_num_neighbors: int = 2
    crepa_schedule: Literal["constant", "linear", "cosine"] = "constant"
    crepa_warmup_steps: int = 0
    crepa_normalize: bool = True
    crepa_cutoff_step: int = 0
    crepa_similarity_threshold: Optional[float] = None
    crepa_similarity_ema_decay: float = 0.99
    crepa_threshold_mode: Literal["permanent", "recoverable"] = "permanent"

    # Self-Flow
    self_flow: bool = False
    self_flow_args: str = ""
    self_flow_teacher_mode: Literal["base", "ema", "partial_ema"] = "base"
    self_flow_student_block_idx: int = 16
    self_flow_teacher_block_idx: int = 32
    self_flow_student_block_ratio: float = 0.3
    self_flow_teacher_block_ratio: float = 0.7
    self_flow_student_block_stochastic_range: int = 0
    self_flow_lambda: float = 0.1
    self_flow_lambda_audio: float = 0.0
    self_flow_mask_ratio: float = 0.1
    self_flow_frame_level_mask: bool = False
    self_flow_mask_focus_loss: bool = False
    self_flow_max_loss: float = 0.0
    self_flow_teacher_momentum: float = 0.999
    self_flow_dual_timestep: bool = True
    self_flow_projector_lr: Optional[float] = None
    self_flow_projector_activation: Literal["silu", "gelu"] = "silu"
    self_flow_temporal_mode: Literal["off", "frame", "delta", "hybrid"] = "off"
    self_flow_lambda_temporal: float = 0.0
    self_flow_lambda_delta: float = 0.0
    self_flow_temporal_tau: float = 1.0
    self_flow_num_neighbors: int = 2
    self_flow_temporal_granularity: Literal["frame", "patch"] = "frame"
    self_flow_patch_spatial_radius: int = 0
    self_flow_patch_match_mode: Literal["hard", "soft"] = "hard"
    self_flow_delta_num_steps: int = 1
    self_flow_motion_weighting: Literal["none", "teacher_delta"] = "none"
    self_flow_motion_weight_strength: float = 0.0
    self_flow_temporal_schedule: Literal["constant", "linear", "cosine"] = "constant"
    self_flow_temporal_warmup_steps: int = 0
    self_flow_temporal_max_steps: int = 0
    self_flow_offload_teacher_features: bool = False

    # HFATO (ViBe - High-Frequency Awareness Training Objective)
    hfato: bool = False
    hfato_args: str = ""
    hfato_scale_factor: float = 0.5
    hfato_interpolation: Literal["bilinear", "nearest", "bicubic"] = "bilinear"
    hfato_probability: float = 1.0

    # Latent temporal objectives
    latent_temporal_weighting: bool = False
    latent_temporal_weighting_args: str = ""
    latent_temporal_weighting_alpha: float = 0.5
    latent_temporal_weighting_mode: Literal["log", "linear"] = "log"
    latent_temporal_weighting_normalize: Literal["mean", "max", "none"] = "mean"
    latent_temporal_weighting_clip_min: float = 0.5
    latent_temporal_weighting_clip_max: float = 2.0
    latent_delta_loss: bool = False
    latent_delta_loss_args: str = ""
    latent_delta_loss_weight: float = 0.03
    latent_delta_loss_order: Literal["1", "2", "1+2", "both"] = "1"
    latent_delta_loss_target: Literal["x0", "velocity"] = "x0"
    latent_delta_loss_sigma_min: float = 0.05
    latent_delta_loss_sigma_max: float = 0.85
    latent_delta_loss_second_order_weight: float = 0.5
    latent_delta_loss_type: Literal["mse", "l1", "huber", "smooth_l1"] = "mse"
    latent_delta_loss_huber_delta: float = 1.0

    # Audio features
    audio_loss_balance_mode: Literal["none", "inv_freq", "ema_mag", "uncertainty", "ogm_ge"] = "none"
    audio_loss_balance_beta: float = 0.01
    audio_loss_balance_eps: float = 0.05
    audio_loss_balance_min: float = 0.05
    audio_loss_balance_max: float = 4.0
    audio_loss_balance_ema_init: float = 1.0
    audio_loss_balance_target_ratio: float = 0.33
    audio_loss_balance_ema_decay: float = 0.99
    uncertainty_lr: Optional[float] = None
    ogm_ge_alpha: float = 0.3
    ogm_ge_noise_std: float = 0.0
    independent_audio_timestep: bool = False
    audio_silence_regularizer: bool = False
    audio_silence_regularizer_weight: float = 1.0
    audio_supervision_mode: Literal["off", "warn", "error"] = "off"
    audio_supervision_warmup_steps: int = 50
    audio_supervision_check_interval: int = 50
    audio_supervision_min_ratio: float = 0.9
    audio_dop: bool = False
    audio_dop_args: str = ""
    audio_dop_multiplier: float = 0.5
    audio_bucket_strategy: Optional[str] = None
    audio_bucket_interval: Optional[float] = None
    audio_only_sequence_resolution: int = 64
    min_audio_batches_per_accum: int = 0
    audio_batch_probability: Optional[float] = None
    cts_lambda_video_driven: float = 0.0
    cts_lambda_audio_driven: float = 0.0
    modality_freeze_check_interval: int = 0
    modality_freeze_ratio_threshold: float = 0.5
    modality_freeze_warmup_steps: int = 100
    modality_freeze_ema_decay: float = 0.99

    # Loss weighting
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0

    # Audio timing (also applies to caching when set on training side)
    preserve_audio_timing: bool = False

    # Misc
    separate_audio_buckets: bool = False
    max_data_loader_n_workers: Optional[int] = None
    persistent_data_loader_workers: bool = False
    ltx2_first_frame_conditioning_p: float = 0.1
    # Endpoint-keyframe training (orthogonal to --ic_lora_strategy)
    keyframe_endpoint_training: bool = False
    keyframe_first_frame_p: float = 1.0
    keyframe_last_frame_p: float = 1.0
    keyframe_random_interior_p: float = 0.0
    keyframe_max_random_interior: int = 0
    # Video-anchor training (hard target-frame replacement)
    video_anchor_training: bool = False
    video_anchor_probability: float = 0.5
    video_anchor_count: int = 1
    video_anchor_strategy: Literal["endpoints", "random", "endpoints_random"] = "endpoints_random"
    accelerate_extra_args: str = ""
    extra_args: str = ""


class FullFinetuneConfig(TrainingConfig):
    model_config = ConfigDict(extra="ignore")

    # Defaults are tuned for real runs rather than dashboard smoke tests.
    learning_rate: float = 1e-6
    optimizer_type: str = "Adafactor"
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 500
    max_train_steps: int = 50000
    gradient_checkpointing: bool = True
    full_bf16: bool = True
    flash_attn: bool = True
    fused_backward_pass: bool = True
    mem_eff_save: bool = True
    output_name: str = "ltx2_full_ft"
    save_every_n_steps: Optional[int] = 1000
    network_module: Optional[str] = None

    # Full fine-tune optimizer/saving options.
    base_optimizer_args: str = ""
    no_final_save: bool = False
    save_comfy_format: bool = False
    save_merged_checkpoint: bool = False

    # FP8 GEMM full-FT (sm_89+, mutually exclusive with LoRA / qgalore_full_ft / fp8_scaled).
    fp8_gemm: bool = False
    fp8_gemm_targets: str = "video"
    fp8_gemm_grad_dtype: Literal["e4m3", "e5m2"] = "e4m3"
    fp8_gemm_min_numel: int = 16384
    fp8_gemm_compile: bool = True

    # int8 weight-only FFT (requires fused_backward_pass + factored optimizer).
    int8_weights: bool = False
    int8_weights_targets: str = "video"
    int8_weights_min_numel: int = 16384
    int8_weights_group_size: int = 0
    int8_weights_outlier_quantile: float = 1.0

    # Q-GaLore full fine-tune.
    qgalore_full_ft: bool = False
    qgalore_targets: str = "all"
    qgalore_rank: int = 256
    qgalore_update_proj_gap: int = 200
    qgalore_scale: float = 0.25
    qgalore_proj_type: str = "std"
    qgalore_proj_quant: bool = True
    qgalore_proj_bits: int = 4
    qgalore_proj_group_size: int = 256
    qgalore_weight_bits: int = 8
    qgalore_weight_group_size: int = 256
    qgalore_stochastic_round: bool = True
    qgalore_min_weight_numel: int = 16384
    qgalore_max_modules: Optional[int] = None
    qgalore_load_device: Literal["cuda", "cpu"] = "cuda"
    qgalore_cos_threshold: float = 0.4
    qgalore_gamma_proj: float = 2.0
    qgalore_queue_size: int = 5
    qgalore_svd_method: Literal["full", "lowrank"] = "lowrank"
    qgalore_svd_oversampling: int = 32
    qgalore_svd_niter: int = 1
    qgalore_dequantize_save: bool = True
    qgalore_streaming_dequantize_save: bool = False
    qgalore_streaming_dequantize_device: Literal["cpu", "cuda"] = "cpu"

    # APOLLO full fine-tune.
    apollo_rank: int = 256
    apollo_update_proj_gap: int = 200
    apollo_scale: float = 1.0
    apollo_proj: Literal["random", "svd"] = "random"
    apollo_proj_type: Literal["std", "reverse_std", "left", "right"] = "std"
    apollo_scale_type: Literal["channel", "tensor"] = "channel"
    apollo_update_rule: Literal["apollo", "fira"] = "apollo"
    qapollo_optim_bits: Literal[8, 32] = 8

    # Full fine-tune-only controls.
    ltx2_finetune_block_swap_mode: Literal["default", "linear", "full"] = "default"
    ltx2_finetune_block_swap_mask: str = "all"
    freeze_early_blocks: int = 0
    freeze_block_indices: str = ""
    block_lr_scales: str = ""
    non_block_lr_scale: float = 1.0
    attn_geometry_lr_scale: float = 1.0
    freeze_attn_geometry: bool = False
    freeze_audio_params: bool = False
    audio_param_lr_scale: float = 1.0

    # EMA and validation.
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_update_after_step: int = 100
    ema_update_every: int = 1
    save_ema_only: bool = False
    ema_cpu_offload: bool = False
    validation_dataset_config: str = ""
    validation_extra_configs: str = ""
    num_validation_batches: Optional[int] = None
    validation_timesteps: str = ""

    # Instrumentation.
    log_weight_drift_every: int = 0
    weight_drift_target: Literal["attn_norm_bias", "attn_geometry", "all_trainable"] = "all_trainable"
    weight_drift_top_k: int = 20
    log_grad_norm_every: int = 0
    grad_norm_target: Literal["attn_norm_bias", "attn_geometry", "all_trainable"] = "all_trainable"
    grad_norm_top_k: int = 20
    log_output_drift_every: int = 0
    output_drift_batches: int = 1
    output_drift_timestep: float = 500.0


class InferenceConfig(BaseModel):
    ltx2_checkpoint: str = ""
    vae: str = ""
    vae_dtype: Optional[Literal["bfloat16", "float16", "float32"]] = None
    device: Optional[str] = None
    gemma_root: str = ""
    gemma_safetensors: str = ""
    lora_weight: str = ""
    lora_multiplier: float = 1.0
    include_patterns: str = ""
    exclude_patterns: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    from_file: str = ""
    output_dir: str = "output"
    output_name: str = "ltx2_sample"
    sampling_preset: SamplingPreset = "defaults"
    sample_sigma_schedule: SampleSigmaSchedule = "auto"
    sample_sampler: SampleSampler = "auto"
    use_default_negative_prompt: Optional[bool] = None
    height: Optional[int] = None
    width: Optional[int] = None
    frame_count: Optional[int] = None
    frame_rate: Optional[float] = None
    sample_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    cfg_scale: Optional[float] = None
    discrete_flow_shift: float = 5.0
    seed: Optional[int] = None
    video_cfg_scale: Optional[float] = None
    audio_cfg_scale: Optional[float] = None
    video_modality_scale: Optional[float] = None
    audio_modality_scale: Optional[float] = None
    video_rescale_scale: Optional[float] = None
    audio_rescale_scale: Optional[float] = None
    stg_scale: Optional[float] = None
    stg_blocks: str = ""
    stg_mode: Optional[Literal["video", "audio", "both"]] = None
    rescale_scale: Optional[float] = None
    av_bimodal_cfg: Optional[bool] = None
    av_bimodal_scale: Optional[float] = None
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    ltx2_mode: Literal["video", "av", "audio"] = "video"
    attn_mode: str = "torch"
    flash_attn: bool = False
    flash3: bool = False
    sdpa: bool = False
    xformers: bool = False
    fp8_base: bool = False
    fp8_scaled: bool = False
    fp8_keep_blocks: str = ""
    fp8_w8a8: bool = False
    w8a8_mode: Literal["int8", "fp8"] = "int8"
    fp8_upcast: bool = False
    fp8_upcast_stochastic: bool = False
    fp8_upcast_seed: int = 0
    nf4_base: bool = False
    nf4_block_size: int = 64
    loftq_init: bool = False
    loftq_iters: int = 2
    awq_calibration: bool = False
    awq_alpha: float = 0.25
    awq_num_batches: int = 8
    network_dim: int = 0
    split_attn_target: str = ""
    split_attn_mode: str = ""
    split_attn_chunk_size: int = 0
    ffn_chunk_target: str = ""
    ffn_chunk_size: int = 0
    offloading: bool = False
    blocks_to_swap: Optional[int] = None
    use_pinned_memory_for_block_swap: bool = False
    gemma_load_in_8bit: bool = False
    gemma_load_in_4bit: bool = False
    gemma_bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    gemma_bnb_4bit_disable_double_quant: bool = False
    gemma_fp8_weight_offload: bool = True
    sample_i2v_token_timestep_mask: bool = True
    reference_downscale: int = 1
    reference_frames: int = 1
    sample_include_reference: bool = False
    reference_image: str = ""
    reference_video: str = ""
    sample_disable_audio: bool = False
    sample_audio_only: bool = False
    sample_merge_audio: bool = False
    sample_audio_subprocess: bool = True
    sample_two_stage: bool = False
    spatial_upsampler_path: str = ""
    distilled_lora_path: str = ""
    sample_stage2_steps: int = 3
    sample_stage1_distilled_lora_multiplier: Optional[float] = None
    sample_stage2_distilled_lora_multiplier: Optional[float] = None
    sample_tiled_vae: bool = False
    sample_vae_tile_size: int = 512
    sample_vae_tile_overlap: int = 64
    sample_vae_temporal_tile_size: int = 0
    sample_vae_temporal_tile_overlap: int = 8
    sample_disable_flash_attn: bool = False
    use_precached_sample_prompts: bool = False
    sample_prompts_cache: str = ""
    use_precached_sample_latents: bool = False
    sample_latents_cache: str = ""
    extra_args: str = ""


class RemoteStageServerConfig(BaseModel):
    ltx2_checkpoint: str = ""
    bind: str = "0.0.0.0"
    port: int = 7788
    device: str = "cuda:0"
    load_device: Optional[str] = None
    dtype: Literal["bfloat16", "float16", "float32", "bf16", "fp16", "fp32"] = "bfloat16"
    split: int = 0
    end: int = -1
    trainable: bool = False
    trainable_scope: LTX2RemoteTrainableScope = "auto"
    learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    max_grad_norm: float = 0.0
    prune_non_stage_blocks: bool = False
    stage_only_device_placement: bool = True
    full_model_device_placement: bool = False
    block_only_load: bool = True
    network_module: Optional[str] = None
    network_dim: Optional[int] = None
    network_alpha: Optional[float] = None
    network_dropout: Optional[float] = None
    network_args: str = ""
    network_weights: str = ""
    network_lr: Optional[float] = None
    ltx2_mode: Literal["video", "av", "audio"] = "video"
    ltx2_audio_only_model: bool = False
    attn_mode: Literal["torch", "sdpa", "flash", "flash3", "xformers"] = "sdpa"
    fp8_scaled: bool = False
    fp8_w8a8: bool = False
    w8a8_mode: Literal["int8", "fp8"] = "int8"
    fp8_upcast: bool = False
    fp8_upcast_stochastic: bool = False
    fp8_upcast_seed: int = 0
    fp8_keep_blocks: str = ""
    nf4_base: bool = False
    nf4_block_size: int = 32
    split_attn_target: Optional[str] = None
    split_attn_mode: Optional[str] = None
    split_attn_chunk_size: int = 0
    ffn_chunk_target: Optional[str] = None
    ffn_chunk_size: int = 0
    quantize_device: Optional[str] = None
    int8_block_size: int = 256
    log_level: str = "INFO"
    extra_args: str = ""


class RemoteStageLauncherConfig(BaseModel):
    ssh_user: str = ""
    ssh_port: int = 22
    ssh_extra_args: str = ""
    remote_root: str = ""
    remote_python: str = "python"
    ready_timeout: float = 120.0
    ready_poll_interval: float = 2.0
    log_level: str = "INFO"


class SliderTargetConfig(BaseModel):
    positive: str = ""
    negative: str = ""
    target_class: str = ""
    weight: float = 1.0


class SliderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")  # old projects may have fields that moved to TrainingConfig

    # Mode
    mode: Literal["text", "reference", "ic_reference"] = "text"
    reference_modality: Literal["video", "audio"] = "video"
    pos_cache_dir: str = ""
    neg_cache_dir: str = ""
    text_cache_dir: str = ""
    reference_cache_dir: str = ""

    # Targets (text-only mode)
    targets: list[SliderTargetConfig] = Field(default_factory=lambda: [SliderTargetConfig()])

    # Text mode settings
    guidance_strength: float = 1.0
    latent_frames: int = 1
    latent_height: int = 512
    latent_width: int = 768

    # Sampling
    sample_slider_range: str = "-2,-1,0,1,2"

    # Slider-specific overrides (empty = inherit from training config)
    max_train_steps: int = 500
    output_name: str = "ltx2_slider"
    accelerate_extra_args: str = ""
    extra_args: str = ""


class ProjectConfig(BaseModel):
    version: int = 2
    name: str = "New Project"
    project_dir: str = ""
    model_dir: str = ""  # directory where downloaded models are stored
    default_ltx2_checkpoint: str = ""
    default_gemma_root: str = ""
    default_gemma_safetensors: str = ""
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    full_finetune: FullFinetuneConfig = Field(default_factory=FullFinetuneConfig)
    remote_stage_launcher: RemoteStageLauncherConfig = Field(default_factory=RemoteStageLauncherConfig)
    remote_stage_server: RemoteStageServerConfig = Field(default_factory=RemoteStageServerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    slider: SliderConfig = Field(default_factory=SliderConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_sampling_key(cls, data):
        """Backward compat: rename old 'sampling' key to 'inference'."""
        if isinstance(data, dict) and "sampling" in data and "inference" not in data:
            data["inference"] = data.pop("sampling")
        if isinstance(data, dict):
            training = data.get("training")
            if training is None:
                data["training"] = {
                    "output_dir": get_ltx2_training_output_dir_default(),
                    "network_module": get_ltx2_training_network_module_default(),
                }
            elif isinstance(training, dict):
                training = dict(training)
                if not training.get("output_dir"):
                    training["output_dir"] = get_ltx2_training_output_dir_default()
                if not training.get("network_module"):
                    training["network_module"] = get_ltx2_training_network_module_default()
                data["training"] = training

            model_dir = data.get("model_dir") or None
            default_ltx = data.get("default_ltx2_checkpoint") or default_ltx2_checkpoint_path(model_dir)
            default_gemma = data.get("default_gemma_root") or default_gemma_root_path(model_dir)
            data["default_ltx2_checkpoint"] = default_ltx
            data["default_gemma_root"] = default_gemma

            full_finetune = data.get("full_finetune")
            if isinstance(full_finetune, dict):
                full_finetune = dict(full_finetune)
                if not full_finetune.get("output_dir"):
                    full_finetune["output_dir"] = get_ltx2_training_output_dir_default()
                data["full_finetune"] = full_finetune

            for section_name in ("caching", "training", "full_finetune", "remote_stage_server", "inference"):
                section = data.get(section_name)
                if section is None:
                    section = {}
                elif not isinstance(section, dict):
                    continue
                else:
                    section = dict(section)

                if section_name in ("training", "full_finetune") and not section.get("output_dir"):
                    section["output_dir"] = get_ltx2_training_output_dir_default()
                if not section.get("ltx2_checkpoint"):
                    section["ltx2_checkpoint"] = default_ltx
                if (
                    section_name in ("caching", "training", "full_finetune", "inference")
                    and not section.get("gemma_root")
                    and not section.get("gemma_safetensors")
                ):
                    section["gemma_root"] = default_gemma
                if section_name == "training":
                    has_old_sampling_values = any(
                        key in section and section.get(key) is not None for key in ("height", "width", "sample_num_frames")
                    )
                    if has_old_sampling_values and "sample_sampling_preset" not in section:
                        section["sample_sampling_preset"] = "legacy"
                elif section_name == "inference":
                    has_old_generation_values = any(
                        key in section and section.get(key) is not None
                        for key in ("height", "width", "frame_count", "frame_rate", "sample_steps", "guidance_scale")
                    )
                    if has_old_generation_values and "sampling_preset" not in section:
                        section["sampling_preset"] = "legacy"
                data[section_name] = section
            data["version"] = max(int(data.get("version") or 1), 2)
        return data

    def save(self, path: Optional[Path] = None):
        p = path or Path(self.project_dir) / "project.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
