import sys
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musubi_tuner.gui_dashboard import toml_export
from musubi_tuner.gui_dashboard.command_builder import build_cache_latents_cmd, build_cache_text_cmd, build_inference_cmd, build_slider_training_cmd, build_training_cmd
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.validation import validate_process_config
from musubi_tuner.ltx2_generate_video import _apply_reference_conditioning_overrides


WINDOWS_VIDEO_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\videos"
WINDOWS_CACHE_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\cache"
WINDOWS_REF_CACHE_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_cache"
WINDOWS_REF_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\refs"
WINDOWS_REF_CACHE_DIR_2 = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_cache_b"
WINDOWS_REF_DIR_2 = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\refs_b"
WINDOWS_REF_AUDIO_CACHE_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_audio_cache"
WINDOWS_REF_AUDIO_CACHE_DIR_2 = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_audio_cache_b"
WINDOWS_REF_AUDIO_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_audio"
WINDOWS_REF_AUDIO_DIR_2 = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_audio_b"


def _build_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        name="GUI TOML Test",
        project_dir=str(tmp_path),
        dataset={
            "general": {
                "enable_bucket": True,
                "bucket_no_upscale": True,
            },
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                    "reference_cache_directory": WINDOWS_REF_CACHE_DIR,
                    "extra_reference_cache_directories": WINDOWS_REF_CACHE_DIR_2,
                    "reference_audio_cache_directory": WINDOWS_REF_AUDIO_CACHE_DIR,
                    "extra_reference_audio_cache_directories": WINDOWS_REF_AUDIO_CACHE_DIR_2,
                    "control_directory": WINDOWS_REF_DIR,
                    "extra_control_directories": WINDOWS_REF_DIR_2,
                    "reference_audio_directory": WINDOWS_REF_AUDIO_DIR,
                    "extra_reference_audio_directories": WINDOWS_REF_AUDIO_DIR_2,
                    "resolution_w": 768,
                    "resolution_h": 512,
                    "batch_size": 1,
                    "num_repeats": 1,
                    "caption_extension": ".txt",
                    "target_frames": 33,
                    "frame_extraction": "full",
                    "max_frames": 129,
                    "source_fps": 24.0,
                }
            ],
            "validation_datasets": [
                {
                    "type": "audio",
                    "directory": r"E:\audio\clips",
                    "cache_directory": r"E:\audio\cache",
                    "batch_size": 2,
                    "num_repeats": 1,
                    "caption_extension": ".txt",
                }
            ],
        },
    )


def test_toml_value_escapes_windows_paths():
    parsed = toml.loads(f"video_directory = {toml_export._toml_value(WINDOWS_VIDEO_DIR)}")
    assert parsed["video_directory"] == WINDOWS_VIDEO_DIR


def test_render_toml_fallback_handles_windows_paths(monkeypatch, tmp_path):
    config = _build_config(tmp_path)
    monkeypatch.setattr(toml_export, "tomli_w", None)

    rendered = toml_export.render_toml(toml_export.build_dataset_toml_document(config))
    parsed = toml.loads(rendered)

    dataset = parsed["datasets"][0]
    assert dataset["video_directory"] == WINDOWS_VIDEO_DIR
    assert dataset["cache_directory"] == WINDOWS_CACHE_DIR
    assert dataset["reference_cache_directories"] == [WINDOWS_REF_CACHE_DIR, WINDOWS_REF_CACHE_DIR_2]
    assert dataset["reference_directories"] == [WINDOWS_REF_DIR, WINDOWS_REF_DIR_2]
    assert dataset["reference_audio_cache_directories"] == [WINDOWS_REF_AUDIO_CACHE_DIR, WINDOWS_REF_AUDIO_CACHE_DIR_2]
    assert dataset["reference_audio_directories"] == [WINDOWS_REF_AUDIO_DIR, WINDOWS_REF_AUDIO_DIR_2]
    assert dataset["frame_extraction"] == "full"
    assert dataset["max_frames"] == 129
    assert dataset["source_fps"] == 24.0


def test_export_dataset_toml_writes_parseable_file_without_tomli_w(monkeypatch, tmp_path):
    config = _build_config(tmp_path)
    monkeypatch.setattr(toml_export, "tomli_w", None)

    output_path = toml_export.export_dataset_toml(config)
    parsed = toml.load(output_path)

    assert output_path.name == "dataset_config.toml"
    assert parsed["general"]["enable_bucket"] is True
    assert parsed["datasets"][0]["video_directory"] == WINDOWS_VIDEO_DIR
    assert parsed["validation_datasets"][0]["audio_directory"] == r"E:\audio\clips"


def test_dataset_cache_directory_defaults_to_dataset_cache_subdir(tmp_path):
    config = ProjectConfig(
        name="Default Cache Dir",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": r"E:\datasets\clips",
                    "cache_directory": "",
                }
            ]
        },
    )

    doc = toml_export.build_dataset_toml_document(config)

    assert doc["datasets"][0]["cache_directory"] == r"E:\datasets\clips\cache"


def test_dataset_toml_export_splits_extra_reference_dirs(tmp_path):
    config = ProjectConfig(
        name="Extra Reference Dirs",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                    "reference_cache_directory": WINDOWS_REF_CACHE_DIR,
                    "extra_reference_cache_directories": f"{WINDOWS_REF_CACHE_DIR_2}, {WINDOWS_REF_CACHE_DIR}",
                    "control_directory": WINDOWS_REF_DIR,
                    "extra_control_directories": f"{WINDOWS_REF_DIR_2}; {WINDOWS_REF_DIR}",
                }
            ]
        },
    )

    doc = toml_export.build_dataset_toml_document(config)

    assert doc["datasets"][0]["reference_cache_directories"] == [WINDOWS_REF_CACHE_DIR, WINDOWS_REF_CACHE_DIR_2]
    assert doc["datasets"][0]["reference_directories"] == [WINDOWS_REF_DIR, WINDOWS_REF_DIR_2]


def test_dataset_toml_export_splits_extra_control_dirs_without_reference_cache(tmp_path):
    config = ProjectConfig(
        name="Extra Control Dirs",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                    "control_directory": WINDOWS_REF_DIR,
                    "extra_control_directories": f"{WINDOWS_REF_DIR_2}; {WINDOWS_REF_CACHE_DIR_2}",
                }
            ]
        },
    )

    doc = toml_export.build_dataset_toml_document(config)

    assert doc["datasets"][0]["control_directory"] == WINDOWS_REF_DIR
    assert doc["datasets"][0]["extra_control_directories"] == f"{WINDOWS_REF_DIR_2}, {WINDOWS_REF_CACHE_DIR_2}"


def test_training_command_builder_includes_av_ic_modifier_args(tmp_path):
    config = ProjectConfig(
        name="AV IC Training Cmd",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                }
            ]
        },
        training={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_root": str(tmp_path / "gemma"),
            "mixed_precision": "bf16",
            "lora_target_preset": "av_ic",
            "ic_lora_strategy": "av_ic",
            "av_cross_attention_mode": "v2a_only",
            "av_multi_ref": True,
        },
    )

    cmd = build_training_cmd(config)

    assert "--ic_lora_strategy" in cmd
    assert cmd[cmd.index("--ic_lora_strategy") + 1] == "av_ic"
    assert "--av_cross_attention_mode" in cmd
    assert cmd[cmd.index("--av_cross_attention_mode") + 1] == "v2a_only"
    assert "--av_multi_ref" in cmd


def test_training_command_builder_omits_default_av_ic_modifier_args(tmp_path):
    config = ProjectConfig(
        name="Default AV IC Training Cmd",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                }
            ]
        },
        training={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_root": str(tmp_path / "gemma"),
            "mixed_precision": "bf16",
            "lora_target_preset": "av_ic",
            "ic_lora_strategy": "av_ic",
        },
    )

    cmd = build_training_cmd(config)

    assert "--av_cross_attention_mode" not in cmd
    assert "--av_multi_ref" not in cmd


def test_cache_text_command_builder_includes_explicit_fp8_weight_offload_flag(tmp_path):
    config = ProjectConfig(
        name="Cache Text Cmd",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                }
            ]
        },
        caching={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_safetensors": str(tmp_path / "gemma_fp8.safetensors"),
            "gemma_fp8_weight_offload": False,
        },
    )

    cmd = build_cache_text_cmd(config)

    assert "--no-gemma_fp8_weight_offload" in cmd
    assert "--gemma_fp8_weight_offload" not in cmd


def test_cache_latents_builder_includes_audio_only_overrides(tmp_path):
    config = _build_config(tmp_path)
    config.caching.ltx2_mode = "audio"
    config.caching.audio_video_latent_channels = 128
    config.caching.audio_video_latent_dtype = "float32"
    config.caching.audio_only_target_resolution = 768
    config.caching.audio_only_target_fps = 30.0

    cmd = build_cache_latents_cmd(config)

    assert "--audio_video_latent_channels" in cmd
    assert cmd[cmd.index("--audio_video_latent_channels") + 1] == "128"
    assert "--audio_video_latent_dtype" in cmd
    assert cmd[cmd.index("--audio_video_latent_dtype") + 1] == "float32"
    assert "--audio_only_target_resolution" in cmd
    assert cmd[cmd.index("--audio_only_target_resolution") + 1] == "768"
    assert "--audio_only_target_fps" in cmd
    assert cmd[cmd.index("--audio_only_target_fps") + 1] == "30.0"


def test_training_command_builder_includes_explicit_fp8_weight_offload_flag(tmp_path):
    config = ProjectConfig(
        name="Training Cmd",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                }
            ]
        },
        training={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_safetensors": str(tmp_path / "gemma_fp8.safetensors"),
            "gemma_fp8_weight_offload": False,
        },
    )

    cmd = build_training_cmd(config)

    assert "--no-gemma_fp8_weight_offload" in cmd
    assert "--gemma_fp8_weight_offload" not in cmd


def test_inference_command_builder_uses_gemma_safetensors_and_offload_flag(tmp_path):
    config = ProjectConfig(
        name="Inference Cmd",
        project_dir=str(tmp_path),
        default_gemma_safetensors=str(tmp_path / "default_gemma_fp8.safetensors"),
        inference={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "ltx2_mode": "video",
            "gemma_root": "",
            "gemma_safetensors": "",
            "gemma_fp8_weight_offload": False,
        },
    )

    cmd = build_inference_cmd(config)

    assert "--gemma_safetensors" in cmd
    assert cmd[cmd.index("--gemma_safetensors") + 1] == str(tmp_path / "default_gemma_fp8.safetensors")
    assert "--no-gemma_fp8_weight_offload" in cmd


def test_cache_text_validation_rejects_default_gemma_safetensors_with_bnb_quantization(tmp_path):
    config = ProjectConfig(
        name="Cache Text Validation",
        project_dir=str(tmp_path),
        default_gemma_safetensors=str(tmp_path / "default_gemma_fp8.safetensors"),
        caching={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_root": "",
            "gemma_safetensors": "",
            "gemma_load_in_8bit": True,
        },
    )

    report = validate_process_config("cache_text", config)

    assert report["ok"] is False
    assert "caching.gemma_safetensors" in report["field_errors"]


def test_training_validation_rejects_default_gemma_safetensors_with_bnb_quantization(tmp_path):
    config = ProjectConfig(
        name="Training Validation",
        project_dir=str(tmp_path),
        default_gemma_safetensors=str(tmp_path / "default_gemma_fp8.safetensors"),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                }
            ]
        },
        training={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_root": "",
            "gemma_safetensors": "",
            "gemma_load_in_4bit": True,
        },
    )

    report = validate_process_config("training", config)

    assert report["ok"] is False
    assert "training.gemma_safetensors" in report["field_errors"]


def test_inference_validation_rejects_default_gemma_safetensors_with_bnb_quantization(tmp_path):
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("a prompt", encoding="utf-8")
    config = ProjectConfig(
        name="Inference Validation",
        project_dir=str(tmp_path),
        default_gemma_safetensors=str(tmp_path / "default_gemma_fp8.safetensors"),
        inference={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "from_file": str(prompts_path),
            "gemma_root": "",
            "gemma_safetensors": "",
            "gemma_load_in_8bit": True,
        },
    )

    report = validate_process_config("inference", config)

    assert report["ok"] is False
    assert "inference.gemma_safetensors" in report["field_errors"]


def test_reference_conditioning_override_replaces_prompt_reference_fields(tmp_path):
    image_path = tmp_path / "ref.png"
    image_path.write_bytes(b"fake")
    prompts = [
        {
            "image_path": "old.png",
            "conditioning_latent": object(),
            "v2v_ref_path": "old.mp4",
            "v2v_ref_latent": object(),
        }
    ]

    ref_path, use_v2v = _apply_reference_conditioning_overrides(prompts, reference_image=str(image_path))

    assert ref_path == str(image_path)
    assert use_v2v is False
    assert prompts[0]["image_path"] == str(image_path)
    assert "conditioning_latent" not in prompts[0]
    assert "v2v_ref_path" not in prompts[0]
    assert "v2v_ref_latent" not in prompts[0]


def test_reference_video_override_replaces_prompt_reference_fields(tmp_path):
    video_path = tmp_path / "ref.mp4"
    video_path.write_bytes(b"fake")
    prompts = [
        {
            "image_path": "old.png",
            "conditioning_latent": object(),
            "v2v_ref_path": "old.mp4",
            "v2v_ref_latent": object(),
        }
    ]

    ref_path, use_v2v = _apply_reference_conditioning_overrides(prompts, reference_video=str(video_path))

    assert ref_path == str(video_path)
    assert use_v2v is True
    assert prompts[0]["v2v_ref_path"] == str(video_path)
    assert "image_path" not in prompts[0]
    assert "conditioning_latent" not in prompts[0]
    assert "v2v_ref_latent" not in prompts[0]


def test_slider_toml_export_writes_reference_slider_fields(tmp_path):
    config = ProjectConfig(
        name="Slider IC Export",
        project_dir=str(tmp_path),
        slider={
            "mode": "ic_reference",
            "reference_modality": "video",
            "pos_cache_dir": r"E:\slider\pos",
            "neg_cache_dir": r"E:\slider\neg",
            "text_cache_dir": r"E:\slider\text",
            "reference_cache_dir": r"E:\slider\ref",
            "sample_slider_range": "-3,-1,0,1,3",
        },
    )

    output_path = toml_export._write_slider_toml(config, toml_export.build_slider_toml_path(config))
    parsed = toml.load(output_path)

    assert parsed["mode"] == "ic_reference"
    assert parsed["reference_modality"] == "video"
    assert parsed["pos_cache_dir"] == r"E:\slider\pos"
    assert parsed["neg_cache_dir"] == r"E:\slider\neg"
    assert parsed["text_cache_dir"] == r"E:\slider\text"
    assert parsed["reference_cache_dir"] == r"E:\slider\ref"
    assert parsed["sample_slider_range"] == [-3.0, -1.0, 0.0, 1.0, 3.0]
    assert "targets" not in parsed


def test_slider_command_builder_includes_mode_relevant_training_args(tmp_path):
    config = ProjectConfig(
        name="Slider Cmd",
        project_dir=str(tmp_path),
        training={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_root": str(tmp_path / "gemma"),
            "mixed_precision": "bf16",
            "ltx2_mode": "video",
            "lora_target_preset": "v2v",
        },
        slider={
            "mode": "ic_reference",
            "pos_cache_dir": str(tmp_path / "pos"),
            "neg_cache_dir": str(tmp_path / "neg"),
            "text_cache_dir": str(tmp_path / "text"),
            "reference_cache_dir": str(tmp_path / "ref"),
        },
    )

    cmd = build_slider_training_cmd(config)

    assert "--ltx2_mode" in cmd
    assert cmd[cmd.index("--ltx2_mode") + 1] == "video"
    assert "--lora_target_preset" in cmd
    assert cmd[cmd.index("--lora_target_preset") + 1] == "v2v"


def test_slider_command_builder_includes_slider_cli_overrides(tmp_path):
    config = ProjectConfig(
        name="Slider Overrides",
        project_dir=str(tmp_path),
        training={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_root": str(tmp_path / "gemma"),
        },
        slider={
            "guidance_strength": 1.7,
            "sample_slider_range": "-4,-2,0,2,4",
        },
    )

    cmd = build_slider_training_cmd(config)

    assert "--guidance_strength" in cmd
    assert cmd[cmd.index("--guidance_strength") + 1] == "1.7"
    assert "--sample_slider_range" in cmd
    assert cmd[cmd.index("--sample_slider_range") + 1] == "-4,-2,0,2,4"


def test_inference_command_builder_includes_advanced_cli_flags(tmp_path):
    config = ProjectConfig(
        name="Inference Cmd",
        project_dir=str(tmp_path),
        inference={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "vae": str(tmp_path / "vae.safetensors"),
            "vae_dtype": "float16",
            "gemma_root": str(tmp_path / "gemma"),
            "lora_weight": '"a.safetensors" "b.safetensors"',
            "include_patterns": "double_blocks to_q",
            "exclude_patterns": "skip_me",
            "device": "cuda",
            "stg_scale": 1.0,
            "stg_blocks": "3 7 9",
            "stg_mode": "both",
            "rescale_scale": 0.7,
            "flash_attn": True,
            "fp8_w8a8": True,
            "w8a8_mode": "fp8",
            "fp8_upcast": True,
            "fp8_upcast_stochastic": True,
            "fp8_upcast_seed": 13,
            "nf4_base": True,
            "nf4_block_size": 128,
            "loftq_init": True,
            "loftq_iters": 4,
            "awq_calibration": True,
            "awq_alpha": 0.5,
            "awq_num_batches": 16,
            "network_dim": 32,
            "split_attn_target": "video audio",
            "split_attn_mode": "chunk",
            "split_attn_chunk_size": 256,
            "ffn_chunk_target": "video",
            "ffn_chunk_size": 128,
            "offloading": True,
            "blocks_to_swap": 4,
            "use_pinned_memory_for_block_swap": True,
            "gemma_load_in_4bit": True,
            "gemma_bnb_4bit_quant_type": "fp4",
            "gemma_bnb_4bit_disable_double_quant": True,
            "sample_i2v_token_timestep_mask": False,
            "reference_downscale": 2,
            "reference_frames": 5,
            "sample_include_reference": True,
            "reference_image": str(tmp_path / "ref.png"),
            "reference_video": str(tmp_path / "ref.mp4"),
            "sample_disable_audio": True,
            "sample_audio_only": True,
            "sample_merge_audio": True,
            "sample_two_stage": True,
            "spatial_upsampler_path": str(tmp_path / "upsampler.safetensors"),
            "distilled_lora_path": str(tmp_path / "distilled.safetensors"),
            "sample_stage2_steps": 5,
            "sample_tiled_vae": True,
            "sample_vae_tile_size": 768,
            "sample_vae_tile_overlap": 96,
            "sample_vae_temporal_tile_size": 16,
            "sample_vae_temporal_tile_overlap": 12,
            "sample_disable_flash_attn": True,
            "use_precached_sample_prompts": True,
            "sample_prompts_cache": str(tmp_path / "prompt_cache.pt"),
            "use_precached_sample_latents": True,
            "sample_latents_cache": str(tmp_path / "latents_cache.pt"),
        },
    )

    cmd = build_inference_cmd(config)

    assert "--device" in cmd
    assert "--vae" in cmd
    assert "--vae_dtype" in cmd
    assert "--lora_weight" in cmd and cmd[cmd.index("--lora_weight") + 1:cmd.index("--lora_multiplier")] == ["a.safetensors", "b.safetensors"]
    assert "--include_patterns" in cmd
    assert "--exclude_patterns" in cmd
    assert "--stg_scale" in cmd
    assert "--stg_blocks" in cmd
    assert "--rescale_scale" in cmd
    assert "--flash_attn" in cmd
    assert "--fp8_w8a8" in cmd
    assert "--w8a8_mode" in cmd
    assert "--nf4_base" in cmd
    assert "--loftq_init" in cmd
    assert "--awq_calibration" in cmd
    assert "--split_attn_target" in cmd
    assert "--ffn_chunk_target" in cmd
    assert "--sample_with_offloading" in cmd
    assert "--use_pinned_memory_for_block_swap" in cmd
    assert "--gemma_bnb_4bit_quant_type" in cmd
    assert "--gemma_bnb_4bit_disable_double_quant" in cmd
    assert "--no-sample_i2v_token_timestep_mask" in cmd
    assert "--reference_image" in cmd
    assert "--reference_video" in cmd
    assert "--sample_two_stage" in cmd
    assert "--spatial_upsampler_path" in cmd
    assert "--distilled_lora_path" in cmd
    assert "--sample_tiled_vae" in cmd
    assert "--sample_disable_flash_attn" in cmd
    assert "--use_precached_sample_prompts" in cmd
    assert "--sample_prompts_cache" in cmd
    assert "--use_precached_sample_latents" in cmd
    assert "--sample_latents_cache" in cmd
