"""Pydantic v2 models for GUI dashboard project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GeneralConfig(BaseModel):
    enable_bucket: bool = True
    bucket_no_upscale: bool = True


class DatasetEntry(BaseModel):
    type: Literal["video", "image", "audio"] = "video"
    directory: str = ""
    cache_directory: str = ""
    reference_cache_directory: str = ""
    control_directory: str = ""
    jsonl_file: str = ""
    resolution_w: int = 768
    resolution_h: int = 512
    batch_size: int = 1
    num_repeats: int = 1
    caption_extension: str = ".txt"
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
    ltx2_mode: Literal["video", "av", "audio"] = "video"
    vae_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    device: str = "cuda"
    skip_existing: bool = True
    # VAE tiling
    vae_chunk_size: Optional[int] = None
    vae_spatial_tile_size: Optional[int] = None
    # Gemma quantization
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gemma_load_in_8bit: bool = False
    gemma_load_in_4bit: bool = False
    gemma_bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    # Text encoder precaching
    precache_sample_prompts: bool = False
    sample_prompts: str = ""
    precache_preservation_prompts: bool = False
    blank_preservation: bool = False
    dop: bool = False
    dop_class_prompt: str = ""
    # Audio
    ltx2_audio_source: Literal["video", "audio_files"] = "video"
    ltx2_audio_dir: str = ""
    ltx2_audio_ext: str = ".wav"
    # DINOv2 feature caching (for CREPA dino mode)
    dino_model: Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"] = "dinov2_vitb14"
    dino_batch_size: int = 16


class TrainingConfig(BaseModel):
    # Model
    ltx2_checkpoint: str = ""
    gemma_root: str = ""
    ltx2_mode: Literal["video", "av", "audio"] = "video"
    fp8_base: bool = False
    fp8_scaled: bool = False
    flash_attn: bool = True
    gemma_load_in_8bit: bool = False
    gemma_load_in_4bit: bool = False

    # LoRA
    network_dim: int = 16
    network_alpha: int = 16
    lora_target_preset: Literal["t2v", "v2v", "audio", "full"] = "t2v"
    network_args: str = ""
    network_weights: str = ""
    network_dropout: Optional[float] = None
    scale_weight_norms: Optional[float] = None
    dim_from_weights: bool = False
    base_weights: str = ""
    base_weights_multiplier: str = ""

    # Optimizer
    learning_rate: float = 1e-4
    optimizer_type: str = "adamw8bit"
    optimizer_args: str = ""
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_decay_steps: Optional[int] = None
    lr_scheduler_num_cycles: Optional[int] = None
    lr_scheduler_power: Optional[float] = None
    lr_scheduler_min_lr_ratio: Optional[float] = None
    lr_scheduler_type: str = ""
    lr_scheduler_args: str = ""
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Schedule
    max_train_steps: int = 1600
    max_train_epochs: Optional[int] = None
    timestep_sampling: str = "shifted_logit_normal"
    discrete_flow_shift: float = 1.0
    weighting_scheme: str = "none"
    seed: Optional[int] = None
    guidance_scale: Optional[float] = None
    sigmoid_scale: Optional[float] = None
    logit_mean: Optional[float] = None
    logit_std: Optional[float] = None
    mode_scale: Optional[float] = None
    min_timestep: Optional[float] = None
    max_timestep: Optional[float] = None

    # Memory
    blocks_to_swap: Optional[int] = None
    gradient_checkpointing: bool = True
    split_attn_target: Optional[str] = None
    split_attn_mode: Optional[str] = None
    split_attn_chunk_size: Optional[int] = None
    blockwise_checkpointing: bool = False
    blocks_to_checkpoint: Optional[int] = None
    mixed_precision: str = "bf16"
    ffn_chunk_target: Optional[str] = None
    ffn_chunk_size: int = 0
    use_pinned_memory_for_block_swap: bool = False
    img_in_txt_in_offloading: bool = False

    # Compile
    compile: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = ""
    compile_dynamic: bool = False
    compile_fullgraph: bool = False

    # CUDA
    cuda_allow_tf32: bool = False
    cuda_cudnn_benchmark: bool = False
    cuda_memory_fraction: Optional[float] = None

    # Sampling
    sample_every_n_steps: Optional[int] = None
    sample_every_n_epochs: Optional[int] = None
    sample_prompts: str = ""
    use_precached_sample_prompts: bool = False
    sample_prompts_cache: str = ""
    height: int = 512
    width: int = 768
    sample_num_frames: int = 45
    sample_with_offloading: bool = False
    sample_merge_audio: bool = False
    sample_disable_audio: bool = False
    sample_at_first: bool = False
    sample_tiled_vae: bool = False
    sample_vae_tile_size: Optional[int] = None
    sample_two_stage: bool = False
    sample_audio_only: bool = False
    sample_disable_flash_attn: bool = False

    # Output
    output_dir: str = ""
    output_name: str = "ltx2_lora"
    save_every_n_epochs: Optional[int] = None
    save_every_n_steps: Optional[int] = None
    save_last_n_epochs: Optional[int] = None
    save_last_n_steps: Optional[int] = None
    save_state: bool = False
    save_state_on_train_end: bool = False
    no_convert_to_comfy: bool = False
    log_with: Optional[str] = None
    logging_dir: str = ""
    wandb_run_name: str = ""
    resume: str = ""
    training_comment: str = ""

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

    # Loss weighting
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0

    # Misc
    separate_audio_buckets: bool = True
    max_data_loader_n_workers: int = 2
    persistent_data_loader_workers: bool = True
    ltx2_first_frame_conditioning_p: float = 0.1


class InferenceConfig(BaseModel):
    ltx2_checkpoint: str = ""
    gemma_root: str = ""
    lora_weight: str = ""
    lora_multiplier: float = 1.0
    prompt: str = ""
    negative_prompt: str = ""
    from_file: str = ""
    output_dir: str = "output"
    output_name: str = "ltx2_sample"
    height: int = 512
    width: int = 768
    frame_count: int = 45
    frame_rate: float = 25.0
    sample_steps: int = 20
    guidance_scale: float = 1.0
    cfg_scale: Optional[float] = None
    discrete_flow_shift: float = 5.0
    seed: Optional[int] = None
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    ltx2_mode: Literal["video", "av", "audio"] = "video"
    attn_mode: str = "torch"
    fp8_base: bool = False
    fp8_scaled: bool = False
    offloading: bool = False
    blocks_to_swap: Optional[int] = None
    gemma_load_in_8bit: bool = False
    gemma_load_in_4bit: bool = False


class SliderTargetConfig(BaseModel):
    positive: str = ""
    negative: str = ""
    target_class: str = ""
    weight: float = 1.0


class SliderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")  # old projects may have fields that moved to TrainingConfig

    # Mode
    mode: Literal["text", "reference"] = "text"

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
    version: int = 1
    name: str = "New Project"
    project_dir: str = ""
    model_dir: str = ""  # directory where downloaded models are stored
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
        return data

    def save(self, path: Optional[Path] = None):
        p = path or Path(self.project_dir) / "project.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
