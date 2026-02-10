"""Tests for CLI command builders: cache latents, cache text, training."""

import sys
from pathlib import Path

from musubi_tuner.gui_dashboard.command_builder import (
    build_cache_latents_cmd,
    build_cache_text_cmd,
    build_training_cmd,
)

from .conftest import MOCK_SCRIPTS, _make_config


class TestCacheLatentsCmd:
    def test_basic_video_mode(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_cache_latents_cmd(config)

        assert sys.executable == cmd[0]
        assert MOCK_SCRIPTS["ltx2_cache_latents.py"] in cmd
        assert "--dataset_config" in cmd
        assert "--ltx2_checkpoint" in cmd
        assert "/models/ltx2.safetensors" in cmd
        assert "--ltx2_mode" in cmd
        assert "video" in cmd
        assert "--skip_existing" in cmd
        assert "--vae_dtype" in cmd
        assert "bfloat16" in cmd

    def test_av_mode_with_audio_source(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.ltx2_mode": "av",
                "caching.ltx2_audio_source": "audio_files",
                "caching.ltx2_audio_dir": "/data/audio",
                "caching.ltx2_audio_ext": ".flac",
            },
        )
        cmd = build_cache_latents_cmd(config)
        assert "--ltx2_mode" in cmd
        idx = cmd.index("--ltx2_mode")
        assert cmd[idx + 1] == "av"
        assert "--ltx2_audio_source" in cmd
        assert "audio_files" in cmd
        assert "--ltx2_audio_dir" in cmd
        assert "/data/audio" in cmd
        assert "--ltx2_audio_ext" in cmd
        assert ".flac" in cmd

    def test_vae_tiling_options(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.vae_chunk_size": 16,
                "caching.vae_spatial_tile_size": 512,
            },
        )
        cmd = build_cache_latents_cmd(config)
        assert "--vae_chunk_size" in cmd
        assert "16" in cmd
        assert "--vae_spatial_tile_size" in cmd
        assert "512" in cmd

    def test_skip_existing_off(self, tmp_path):
        config = _make_config(str(tmp_path), **{"caching.skip_existing": False})
        cmd = build_cache_latents_cmd(config)
        assert "--skip_existing" not in cmd

    def test_writes_toml_file(self, tmp_path):
        config = _make_config(str(tmp_path))
        build_cache_latents_cmd(config)
        toml_path = Path(str(tmp_path)) / "dataset_config.toml"
        assert toml_path.exists()


class TestCacheTextCmd:
    def test_basic(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_cache_text_cmd(config)

        assert MOCK_SCRIPTS["ltx2_cache_text_encoder_outputs.py"] in cmd
        assert "--gemma_root" in cmd
        assert "/models/gemma" in cmd
        assert "--ltx2_mode" in cmd
        assert "video" in cmd

    def test_mixed_precision(self, tmp_path):
        config = _make_config(str(tmp_path), **{"caching.mixed_precision": "bf16"})
        cmd = build_cache_text_cmd(config)
        assert "--mixed_precision" in cmd
        assert "bf16" in cmd

    def test_no_mixed_precision(self, tmp_path):
        config = _make_config(str(tmp_path), **{"caching.mixed_precision": "no"})
        cmd = build_cache_text_cmd(config)
        assert "--mixed_precision" not in cmd

    def test_gemma_8bit(self, tmp_path):
        config = _make_config(str(tmp_path), **{"caching.gemma_load_in_8bit": True})
        cmd = build_cache_text_cmd(config)
        assert "--gemma_load_in_8bit" in cmd

    def test_gemma_4bit_with_quant_type(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.gemma_load_in_4bit": True,
                "caching.gemma_bnb_4bit_quant_type": "fp4",
            },
        )
        cmd = build_cache_text_cmd(config)
        assert "--gemma_load_in_4bit" in cmd
        assert "--gemma_bnb_4bit_quant_type" in cmd
        assert "fp4" in cmd

    def test_precache_sample_prompts(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.precache_sample_prompts": True,
                "caching.sample_prompts": "/data/prompts.txt",
            },
        )
        cmd = build_cache_text_cmd(config)
        assert "--precache_sample_prompts" in cmd
        assert "--sample_prompts" in cmd
        assert "/data/prompts.txt" in cmd

    def test_precache_preservation_with_dop(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.precache_preservation_prompts": True,
                "caching.blank_preservation": True,
                "caching.dop": True,
                "caching.dop_class_prompt": "woman",
            },
        )
        cmd = build_cache_text_cmd(config)
        assert "--precache_preservation_prompts" in cmd
        assert "--blank_preservation" in cmd
        assert "--dop" in cmd
        assert "--dop_class_prompt" in cmd
        assert "woman" in cmd


class TestTrainingCmd:
    def test_basic_training(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)

        # Should use accelerate launch
        assert "-m" in cmd
        assert "accelerate.commands.launch" in cmd
        assert MOCK_SCRIPTS["ltx2_train_network.py"] in cmd

        # Core args
        assert "--dataset_config" in cmd
        assert "--ltx2_checkpoint" in cmd
        assert "--learning_rate" in cmd
        assert "--optimizer_type" in cmd
        assert "--lr_scheduler" in cmd
        assert "--network_dim" in cmd
        assert "--network_alpha" in cmd
        assert "--output_dir" in cmd
        assert "--output_name" in cmd
        # GUI is no longer forced by command builder.
        assert "--gui" not in cmd

    def test_training_does_not_force_gui_flag(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)
        assert "--gui" not in cmd

    def test_lora_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.network_dim": 32,
                "training.network_alpha": 16,
                "training.lora_target_preset": "v2v",
            },
        )
        cmd = build_training_cmd(config)
        idx = cmd.index("--network_dim")
        assert cmd[idx + 1] == "32"
        idx = cmd.index("--network_alpha")
        assert cmd[idx + 1] == "16"
        idx = cmd.index("--lora_target_preset")
        assert cmd[idx + 1] == "v2v"

    def test_optimizer_args(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{"training.optimizer_args": "weight_decay=0.01 betas=0.9,0.999"},
        )
        cmd = build_training_cmd(config)
        idx = cmd.index("--optimizer_args")
        assert cmd[idx + 1] == "weight_decay=0.01"
        assert cmd[idx + 2] == "betas=0.9,0.999"

    def test_epochs_instead_of_steps(self, tmp_path):
        config = _make_config(str(tmp_path), **{"training.max_train_epochs": 10})
        cmd = build_training_cmd(config)
        assert "--max_train_epochs" in cmd
        assert "10" in cmd
        assert "--max_train_steps" not in cmd

    def test_steps_when_no_epochs(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)
        assert "--max_train_steps" in cmd
        assert "1600" in cmd
        assert "--max_train_epochs" not in cmd

    def test_memory_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.blocks_to_swap": 20,
                "training.gradient_checkpointing": True,
                "training.blockwise_checkpointing": True,
                "training.split_attn_target": "all",
                "training.split_attn_mode": "batch",
            },
        )
        cmd = build_training_cmd(config)
        assert "--blocks_to_swap" in cmd
        assert "20" in cmd
        assert "--gradient_checkpointing" in cmd
        assert "--blockwise_checkpointing" in cmd
        assert "--split_attn_target" in cmd
        assert "all" in cmd
        assert "--split_attn_mode" in cmd
        assert "batch" in cmd

    def test_sampling_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.sample_every_n_steps": 100,
                "training.sample_prompts": "/data/prompts.txt",
                "training.height": 480,
                "training.width": 640,
                "training.sample_num_frames": 25,
                "training.sample_with_offloading": True,
                "training.sample_merge_audio": True,
            },
        )
        cmd = build_training_cmd(config)
        assert "--sample_every_n_steps" in cmd
        assert "100" in cmd
        assert "--sample_prompts" in cmd
        assert "--height" in cmd
        assert "480" in cmd
        assert "--width" in cmd
        assert "640" in cmd
        assert "--sample_num_frames" in cmd
        assert "25" in cmd
        assert "--sample_with_offloading" in cmd
        assert "--sample_merge_audio" in cmd

    def test_preservation_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.blank_preservation": True,
                "training.blank_preservation_multiplier": 0.5,
                "training.dop": True,
                "training.dop_class": "woman",
                "training.dop_multiplier": 1.0,
                "training.prior_divergence": True,
                "training.prior_divergence_multiplier": 0.1,
            },
        )
        cmd = build_training_cmd(config)
        assert "--blank_preservation" in cmd
        assert "--blank_preservation_args" in cmd
        assert "multiplier=0.5" in cmd
        assert "--dop" in cmd
        assert "--dop_args" in cmd
        assert "class=woman" in cmd
        assert "--prior_divergence" in cmd

    def test_crepa_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.crepa": True,
                "training.crepa_student_block_idx": 12,
                "training.crepa_teacher_block_idx": 36,
                "training.crepa_lambda": 0.05,
                "training.crepa_tau": 2.0,
                "training.crepa_num_neighbors": 3,
                "training.crepa_schedule": "cosine",
                "training.crepa_warmup_steps": 100,
                "training.crepa_normalize": False,
            },
        )
        cmd = build_training_cmd(config)
        assert "--crepa" in cmd
        assert "--crepa_args" in cmd
        assert "student_block_idx=12" in cmd
        assert "teacher_block_idx=36" in cmd
        assert "lambda_crepa=0.05" in cmd
        assert "tau=2.0" in cmd
        assert "num_neighbors=3" in cmd
        assert "schedule=cosine" in cmd
        assert "warmup_steps=100" in cmd
        assert "normalize=false" in cmd

    def test_crepa_defaults_no_args(self, tmp_path):
        """When CREPA is enabled with all defaults, --crepa_args should not appear."""
        config = _make_config(str(tmp_path), **{"training.crepa": True})
        cmd = build_training_cmd(config)
        assert "--crepa" in cmd
        assert "--crepa_args" not in cmd

    def test_crepa_disabled_no_flag(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)
        assert "--crepa" not in cmd

    def test_loss_weighting(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.video_loss_weight": 0.5,
                "training.audio_loss_weight": 2.0,
            },
        )
        cmd = build_training_cmd(config)
        assert "--video_loss_weight" in cmd
        assert "0.5" in cmd
        assert "--audio_loss_weight" in cmd
        assert "2.0" in cmd

    def test_default_loss_weights_omitted(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)
        assert "--video_loss_weight" not in cmd
        assert "--audio_loss_weight" not in cmd

    def test_fp8_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.fp8_base": True,
                "training.fp8_scaled": True,
            },
        )
        cmd = build_training_cmd(config)
        assert "--fp8_base" in cmd
        assert "--fp8_scaled" in cmd

    def test_av_mode_training(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.ltx2_mode": "av",
                "training.separate_audio_buckets": True,
                "training.lora_target_preset": "audio",
            },
        )
        cmd = build_training_cmd(config)
        idx = cmd.index("--ltx2_mode")
        assert cmd[idx + 1] == "av"
        assert "--separate_audio_buckets" in cmd
        idx = cmd.index("--lora_target_preset")
        assert cmd[idx + 1] == "audio"

    def test_save_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.save_every_n_epochs": 5,
                "training.save_every_n_steps": 500,
                "training.save_state": True,
            },
        )
        cmd = build_training_cmd(config)
        assert "--save_every_n_epochs" in cmd
        assert "5" in cmd
        assert "--save_every_n_steps" in cmd
        assert "500" in cmd
        assert "--save_state" in cmd

    def test_logging_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.log_with": "tensorboard",
                "training.logging_dir": "/logs",
            },
        )
        cmd = build_training_cmd(config)
        assert "--log_with" in cmd
        assert "tensorboard" in cmd
        assert "--logging_dir" in cmd
        assert "/logs" in cmd

    def test_seed(self, tmp_path):
        config = _make_config(str(tmp_path), **{"training.seed": 42})
        cmd = build_training_cmd(config)
        assert "--seed" in cmd
        assert "42" in cmd

    def test_no_seed_when_none(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)
        assert "--seed" not in cmd

    def test_mixed_precision_in_accelerate(self, tmp_path):
        config = _make_config(str(tmp_path), **{"training.mixed_precision": "fp16"})
        cmd = build_training_cmd(config)
        # mixed_precision should be passed to accelerate launch
        mp_idx = cmd.index("--mixed_precision")
        assert cmd[mp_idx + 1] == "fp16"
        # It should come before the training script
        script_idx = cmd.index(MOCK_SCRIPTS["ltx2_train_network.py"])
        assert mp_idx < script_idx

    def test_ffn_chunk_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.ffn_chunk_target": "all",
                "training.ffn_chunk_size": 2048,
            },
        )
        cmd = build_training_cmd(config)
        assert "--ffn_chunk_target" in cmd
        assert "all" in cmd
        assert "--ffn_chunk_size" in cmd
        assert "2048" in cmd
