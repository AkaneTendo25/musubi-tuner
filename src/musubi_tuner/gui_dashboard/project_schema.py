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


class GeneralConfig(BaseModel):
    enable_bucket: bool = True
    bucket_no_upscale: bool = True


class DatasetEntry(BaseModel):
    type: Literal["video", "image", "audio"] = "video"
    directory: str = ""
    cache_directory: str = ""
    reference_cache_directory: str = ""
    extra_reference_cache_directories: str = ""
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
    audio_video_latent_channels: Optional[int] = None
    audio_video_latent_dtype: Optional[str] = None
    audio_only_target_resolution: Optional[int] = None
    audio_only_target_fps: Optional[float] = None
    audio_only_sequence_resolution: int = 64
    # DINOv2 feature caching (for CREPA dino mode - model selection in training.crepa_dino_model)
    dino_batch_size: int = 16
    # Quantization device
    quantize_device: Optional[str] = None
    # Connector LoRA
    cache_before_connector: bool = False
    # Dataset manifest
    save_dataset_manifest: str = ""


class TrainingConfig(BaseModel):
    # Model
    ltx2_checkpoint: str = ""
    gemma_root: str = ""
    gemma_safetensors: str = ""
    ltx2_mode: Literal["video", "av", "audio"] = "video"
    ltx_version: Literal["2.0", "2.3"] = "2.3"
    ltx_version_check_mode: Literal["off", "warn", "error"] = "warn"
    fp8_base: bool = False
    fp8_scaled: bool = False
    flash_attn: bool = False
    sdpa: bool = False
    sage_attn: bool = False
    xformers: bool = False
    gemma_load_in_8bit: bool = False
    gemma_load_in_4bit: bool = False
    gemma_bnb_4bit_disable_double_quant: bool = False
    gemma_fp8_weight_offload: bool = True
    ltx2_audio_only_model: bool = False

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
        "t2v", "v2v", "video_sa", "video_sa_ff", "video_sa_ca_ff", "audio", "audio_ref_only_ic", "av_ic", "video_ref_only_av", "full", "lycoris"
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
    ic_lora_strategy: Literal["auto", "none", "v2v", "audio_ref_only_ic", "av_ic", "video_ref_only_av"] = "auto"
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
    preserve_distribution_shape: bool = False
    num_timestep_buckets: Optional[int] = None

    # Memory
    blocks_to_swap: Optional[int] = None
    gradient_checkpointing: bool = False
    gradient_checkpointing_cpu_offload: bool = False
    split_attn_target: Optional[str] = None
    split_attn_mode: Optional[str] = None
    split_attn_chunk_size: Optional[int] = None
    blockwise_checkpointing: bool = False
    blocks_to_checkpoint: Optional[int] = None
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
    compile_dynamic: bool = False
    compile_fullgraph: bool = False
    compile_cache_size_limit: Optional[int] = None

    # CUDA
    cuda_allow_tf32: bool = False
    cuda_cudnn_benchmark: bool = False
    cuda_memory_fraction: Optional[float] = None

    # Sampling
    sample_every_n_steps: Optional[int] = None
    sample_every_n_epochs: Optional[int] = None
    sample_prompts: str = ""
    sample_prompts_text: str = ""
    use_precached_sample_prompts: bool = False
    sample_prompts_cache: str = ""
    use_precached_sample_latents: bool = False
    sample_latents_cache: str = ""
    sample_sampling_preset: SamplingPreset = "defaults"
    sample_use_default_negative_prompt: Optional[bool] = None
    height: Optional[int] = None
    width: Optional[int] = None
    sample_num_frames: Optional[int] = None
    video_cfg_scale: Optional[float] = None
    audio_cfg_scale: Optional[float] = None
    video_modality_scale: Optional[float] = None
    audio_modality_scale: Optional[float] = None
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
    blank_preservation_multiplier: float = 1.0
    dop: bool = False
    dop_class: str = ""
    dop_multiplier: float = 1.0
    prior_divergence: bool = False
    prior_divergence_multiplier: float = 0.1
    use_precached_preservation: bool = False
    preservation_prompts_cache: str = ""

    # TARP / DCR (arXiv:2603.18600)
    tarp: bool = False
    tarp_window_multiplier: int = 3
    dcr: bool = False
    dcr_reference_detach: bool = True

    # Audio Metrics
    audio_metrics: bool = False
    audio_metrics_mel_metrics: bool = False
    audio_metrics_mel_compute_every: int = 100
    audio_metrics_clap_similarity: bool = False
    audio_metrics_av_onset_alignment: bool = False

    # CREPA
    crepa: bool = False
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

    # Self-Flow
    self_flow: bool = False
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
    hfato_scale_factor: float = 0.5
    hfato_interpolation: Literal["bilinear", "nearest", "bicubic"] = "bilinear"
    hfato_probability: float = 1.0

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

    # Misc
    separate_audio_buckets: bool = False
    max_data_loader_n_workers: int = 8
    persistent_data_loader_workers: bool = False
    ltx2_first_frame_conditioning_p: float = 0.1



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
    sample_two_stage: bool = False
    spatial_upsampler_path: str = ""
    distilled_lora_path: str = ""
    sample_stage2_steps: int = 3
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
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    slider: SliderConfig = Field(default_factory=SliderConfig)

    @model_validator(mode='before')
    @classmethod
    def _migrate_sampling_key(cls, data):
        """Backward compat: rename old 'sampling' key to 'inference'."""
        if isinstance(data, dict) and 'sampling' in data and 'inference' not in data:
            data['inference'] = data.pop('sampling')
        if isinstance(data, dict):
            training = data.get('training')
            if training is None:
                data['training'] = {
                    'output_dir': get_ltx2_training_output_dir_default(),
                    'network_module': get_ltx2_training_network_module_default(),
                }
            elif isinstance(training, dict):
                training = dict(training)
                if not training.get('output_dir'):
                    training['output_dir'] = get_ltx2_training_output_dir_default()
                if not training.get('network_module'):
                    training['network_module'] = get_ltx2_training_network_module_default()
                data['training'] = training

            model_dir = data.get('model_dir') or None
            default_ltx = data.get('default_ltx2_checkpoint') or default_ltx2_checkpoint_path(model_dir)
            default_gemma = data.get('default_gemma_root') or default_gemma_root_path(model_dir)
            data['default_ltx2_checkpoint'] = default_ltx
            data['default_gemma_root'] = default_gemma

            for section_name in ('caching', 'training', 'inference'):
                section = data.get(section_name)
                if section is None:
                    section = {}
                elif not isinstance(section, dict):
                    continue
                else:
                    section = dict(section)

                if not section.get('ltx2_checkpoint'):
                    section['ltx2_checkpoint'] = default_ltx
                if not section.get('gemma_root') and not section.get('gemma_safetensors'):
                    section['gemma_root'] = default_gemma
                if section_name == 'training':
                    has_old_sampling_values = any(
                        key in section and section.get(key) is not None
                        for key in ('height', 'width', 'sample_num_frames')
                    )
                    if has_old_sampling_values and 'sample_sampling_preset' not in section:
                        section['sample_sampling_preset'] = 'legacy'
                elif section_name == 'inference':
                    has_old_generation_values = any(
                        key in section and section.get(key) is not None
                        for key in ('height', 'width', 'frame_count', 'frame_rate', 'sample_steps', 'guidance_scale')
                    )
                    if has_old_generation_values and 'sampling_preset' not in section:
                        section['sampling_preset'] = 'legacy'
                data[section_name] = section
            data['version'] = max(int(data.get('version') or 1), 2)
        return data

    def save(self, path: Optional[Path] = None):
        p = path or Path(self.project_dir) / "project.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
