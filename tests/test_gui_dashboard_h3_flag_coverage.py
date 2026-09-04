from pathlib import Path

from musubi_tuner.gui_dashboard.command_builder import (
    build_cache_latents_cmd,
    build_cache_text_cmd,
    build_inference_cmd,
    build_training_cmd,
)
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig


def _config(tmp_path: Path) -> ProjectConfig:
    config = ProjectConfig(project_dir=str(tmp_path))
    config.training.h3_model = "h3.safetensors"
    config.training.network_dim = 16
    config.caching.h3_video_vae = "vae.safetensors"
    config.caching.h3_audio_vae = "audio-vae.safetensors"
    config.caching.h3_text_encoder = "qwen.safetensors"
    config.caching.h3_tokenizer = "tokenizer"
    return config


def test_every_new_h3_training_control_emits_its_flag(tmp_path: Path) -> None:
    config = _config(tmp_path)
    values = {
        "h3_lora_targets": "main_blocks",
        "h3_audio_only_spatial_tokens": True,
        "h3_loss_mask_normalization": "full",
        "h3_guidance_audio_scale": 2.0,
        "reference_video_fps": 12.0,
        "h3_max_caption_tokens": 256,
        "h3_qwen_control_dropout_rate": 0.2,
        "h3_validation_field_probe": True,
        "h3_validation_rollout_probe": 2,
        "h3_validation_rollout_stop": 0.6,
        "h3_profile_steps": 4,
        "h3_rollout_supervision": True,
        "h3_rollout_teacher_config": "teacher.toml",
        "h3_rollout_teacher_privilege": "caption",
        "h3_rollout_probability": 0.7,
        "h3_rollout_steps": 4,
        "h3_rollout_window": 2,
        "h3_rollout_stop_shifted": True,
        "h3_rollout_fused_teacher": True,
        "h3_rollout_teacher_magnitude_weight": 1.5,
        "h3_rollout_field_floor": 3.0,
        "h3_guidance_scale_sigma_max": 0.9,
        "h3_validation_multipliers": "1,2",
        "h3_adapter_ema_decay": 0.9,
        "h3_validate_ema": True,
        "h3_term_grad_every": 5,
        "h3_adapter_stats_every": 6,
        "h3_train_sigma_bins": True,
        "h3_validation_bare_dataset_config": "bare.toml",
        "h3_validation_std": True,
        "h3_adapter_prompt_only": True,
        "h3_measured_variance_weighting": "variance.json",
        "h3_measured_variance_weight_max": 5.0,
        "h3_rollout_stop_min": 0.1,
        "h3_rollout_prefix": "teacher",
        "h3_rollout_null_anchor_weight": 0.3,
        "h3_rollout_field_cap": 1.5,
        "h3_rollout_field_floor_direction": "teacher",
        "h3_rollout_field_floor_sigma_max": 0.8,
        "h3_guidance_null_anchor_weight_end": 0.2,
        "h3_guidance_null_anchor_weight": 0.3,
        "h3_guidance_null_anchor_probability": 0.5,
        "h3_fused_elementwise": True,
        "h3_compile_attention": "opaque",
        "h3_swiglu_chunk_rows": 1024,
        "h3_checkpoint_keep": "attention",
        "h3_block_sparse_kv_fraction": 0.5,
        "h3_block_sparse_threshold": 0.8,
        "h3_block_sparse_start_block": 10,
        "h3_block_sparse_block_shape": "1,8,16",
    }
    for name, value in values.items():
        setattr(config.training, name, value)
    command = build_training_cmd(config)
    assert not {f"--{name}" for name in values}.difference(command)


def test_every_new_h3_cache_and_inference_control_emits_its_flag(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.caching.h3_loss_mask_pooling = "average"
    config.caching.h3_reference_video_fps = 12.0
    config.caching.h3_max_caption_tokens = 256
    config.caching.h3_keyframe_visuals = "first,last"
    config.caching.h3_qwen_control_dropout = True
    latent, text = build_cache_latents_cmd(config), build_cache_text_cmd(config)
    assert {"--h3_loss_mask_pooling", "--reference_video_fps"}.issubset(latent)
    assert {"--reference_video_fps", "--h3_max_caption_tokens", "--h3_keyframe_visuals", "--h3_qwen_control_dropout"}.issubset(text)

    config.inference.h3_guide_images = "0:first.png 24:last.png"
    config.inference.h3_guide_videos = "0:guide.mp4"
    config.inference.h3_guide_audios = "0:guide.wav"
    config.inference.h3_learned_context_multipliers = "1.0 0.5"
    config.inference.h3_null_guidance_scale = 4.0
    config.inference.h3_inspect = True
    inference = build_inference_cmd(config)
    assert inference.count("--guide_image") == 2
    assert inference.count("--h3_learned_context_multiplier") == 2
    assert {"--guide_video", "--guide_audio", "--h3_null_guidance_scale", "--inspect"}.issubset(inference)
