"""Regression contracts for porting this fork across upstream PR #950.

The upstream refactor moves trainer and dataset symbols into focused modules.
These tests pin LTX2-specific behavior that must survive that move without
requiring real checkpoints, media decoding, CUDA, or model downloads.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import random
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def _resolve_attr(attr_name: str, *module_names: str):
    for module_name in module_names:
        module = _optional_module(module_name)
        if module is not None and hasattr(module, attr_name):
            return getattr(module, attr_name)
    searched = ", ".join(module_names)
    raise AssertionError(f"{attr_name} was not exported from any of: {searched}")


def _item(item_key: str, cache_path: Path):
    ItemInfo = _resolve_attr("ItemInfo", "musubi_tuner.dataset.image_video_dataset")
    return ItemInfo(
        item_key=item_key,
        caption=f"caption for {item_key}",
        original_size=(64, 64),
        bucket_size=(64, 64, 1),
        frame_count=1,
        latent_cache_path=str(cache_path),
    )


def test_ltx2_symbols_survive_dataset_module_split():
    image_video_dataset = importlib.import_module("musubi_tuner.dataset.image_video_dataset")
    architectures = _optional_module("musubi_tuner.dataset.architectures")
    bucket = _optional_module("musubi_tuner.dataset.bucket")
    cache_io = _optional_module("musubi_tuner.dataset.cache_io")

    assert image_video_dataset.ARCHITECTURE_LTX2 == "ltx2"
    assert image_video_dataset.ARCHITECTURE_LTX2_FULL == "ltx2_v1"

    if architectures is not None:
        assert architectures.ARCHITECTURE_LTX2 == image_video_dataset.ARCHITECTURE_LTX2
        assert architectures.ARCHITECTURE_LTX2_FULL == image_video_dataset.ARCHITECTURE_LTX2_FULL

    bucket_selector = _resolve_attr("BucketSelector", "musubi_tuner.dataset.bucket", "musubi_tuner.dataset.image_video_dataset")
    batch_manager = _resolve_attr("BucketBatchManager", "musubi_tuner.dataset.bucket", "musubi_tuner.dataset.image_video_dataset")
    assert bucket is not None
    assert bucket_selector is bucket.BucketSelector
    assert image_video_dataset.BucketSelector is bucket.BucketSelector
    assert batch_manager is bucket.BucketBatchManager
    assert image_video_dataset.BucketBatchManager is bucket.BucketBatchManager

    for symbol in (
        "save_latent_cache_ltx2",
        "save_text_encoder_output_cache_ltx2",
        "save_text_encoder_output_cache_ltx2_gemma",
    ):
        resolved = _resolve_attr(symbol, "musubi_tuner.dataset.cache_io", "musubi_tuner.dataset.image_video_dataset")
        assert cache_io is not None
        assert resolved is getattr(cache_io, symbol)
        assert getattr(image_video_dataset, symbol) is getattr(cache_io, symbol)

    assert hasattr(image_video_dataset, "AudioDataset")
    assert hasattr(image_video_dataset, "DatasetGroup")


def test_ltx2_bucket_resolution_step_contract():
    BucketSelector = _resolve_attr("BucketSelector", "musubi_tuner.dataset.bucket", "musubi_tuner.dataset.image_video_dataset")
    ARCHITECTURE_LTX2 = _resolve_attr(
        "ARCHITECTURE_LTX2", "musubi_tuner.dataset.architectures", "musubi_tuner.dataset.image_video_dataset"
    )

    assert BucketSelector.resolve_resolution_steps(ARCHITECTURE_LTX2) == 32
    assert BucketSelector.resolve_resolution_steps(ARCHITECTURE_LTX2, reference_downscale=2) == 64

    selector = BucketSelector((640, 384), enable_bucket=False, architecture=ARCHITECTURE_LTX2, reference_downscale=2)
    assert selector.reso_steps == 64
    assert selector.bucket_resolutions == [(640, 384)]

    with pytest.raises(ValueError):
        BucketSelector((640, 400), enable_bucket=False, architecture=ARCHITECTURE_LTX2, reference_downscale=2)


def test_ltx2_cache_writers_preserve_keys_and_metadata(tmp_path: Path):
    save_latent_cache_ltx2 = _resolve_attr(
        "save_latent_cache_ltx2", "musubi_tuner.dataset.cache_io", "musubi_tuner.dataset.image_video_dataset"
    )
    save_text_encoder_output_cache_ltx2_gemma = _resolve_attr(
        "save_text_encoder_output_cache_ltx2_gemma",
        "musubi_tuner.dataset.cache_io",
        "musubi_tuner.dataset.image_video_dataset",
    )

    latent_item = _item("clip", tmp_path / "clip_ltx2.safetensors")
    latent = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2)
    save_latent_cache_ltx2(
        latent_item,
        latent,
        extra_tensors={
            "ltx2_virtual_num_frames_int32": torch.tensor(17, dtype=torch.int32),
            "audio_loss_mask": torch.ones(5, dtype=torch.float32),
        },
        atomic=True,
    )

    latent_sd = load_file(str(latent_item.latent_cache_path))
    assert set(latent_sd) == {
        "latents_1x2x2_float32",
        "ltx2_virtual_num_frames_int32",
        "audio_loss_mask",
    }
    with safe_open(str(latent_item.latent_cache_path), framework="pt") as f:
        assert f.metadata()["architecture"] == "ltx2_v1"
        assert f.metadata()["frame_count"] == "1"

    text_item = _item("clip", tmp_path / "clip_ltx2_te.safetensors")
    text_item.text_encoder_output_cache_path = str(tmp_path / "clip_ltx2_te.safetensors")
    save_text_encoder_output_cache_ltx2_gemma(
        text_item,
        video_prompt_embeds=torch.ones(3, 4),
        audio_prompt_embeds=torch.full((3, 2), 2.0),
        prompt_attention_mask=torch.tensor([1, 1, 0], dtype=torch.int64),
        video_features=torch.full((3, 5), 3.0),
        audio_features=torch.full((3, 6), 4.0),
        atomic=True,
    )

    text_sd = load_file(text_item.text_encoder_output_cache_path)
    assert text_sd["video_prompt_embeds_float32"].shape == (3, 4)
    assert text_sd["audio_prompt_embeds_float32"].shape == (3, 2)
    assert text_sd["text_float32"].shape == (3, 6)
    assert text_sd["prompt_attention_mask"].tolist() == [1, 1, 0]
    assert text_sd["text_mask"].tolist() == [1, 1, 0]
    assert text_sd["video_features_float32"].shape == (3, 5)
    assert text_sd["audio_features_float32"].shape == (3, 6)


def test_ltx2_bucket_batch_manager_renames_reference_and_guide_latents(tmp_path: Path):
    BucketBatchManager = _resolve_attr(
        "BucketBatchManager", "musubi_tuner.dataset.bucket", "musubi_tuner.dataset.image_video_dataset"
    )
    ARCHITECTURE_LTX2 = _resolve_attr(
        "ARCHITECTURE_LTX2", "musubi_tuner.dataset.architectures", "musubi_tuner.dataset.image_video_dataset"
    )

    def write_cache(path: Path, value: float):
        save_file({"latents_1x2x2_float32": torch.full((4, 1, 2, 2), value)}, str(path))

    latent_path = tmp_path / "sample_0064x0064x0001_ltx2.safetensors"
    text_path = tmp_path / "sample_ltx2_te.safetensors"
    ref_path = tmp_path / "sample_ref_ltx2.safetensors"
    latent_idx_path = tmp_path / "sample_latent_idx_ltx2.safetensors"
    keyframe_path = tmp_path / "sample_keyframe_ltx2.safetensors"
    extra_path = tmp_path / "sample_keyframe_extra_ltx2.safetensors"

    write_cache(latent_path, 1.0)
    write_cache(ref_path, 2.0)
    write_cache(latent_idx_path, 3.0)
    write_cache(keyframe_path, 4.0)
    write_cache(extra_path, 5.0)
    save_file(
        {
            "video_prompt_embeds_float32": torch.ones(3, 4),
            "prompt_attention_mask": torch.tensor([1, 1, 0], dtype=torch.int64),
        },
        str(text_path),
    )

    item = _item("sample", latent_path)
    item.text_encoder_output_cache_path = str(text_path)
    item.reference_latent_cache_path = str(ref_path)
    item.latent_idx_guide_cache_path = str(latent_idx_path)
    item.keyframe_guide_cache_path = str(keyframe_path)
    item.keyframe_guide_extra_cache_paths = [str(extra_path)]

    manager = BucketBatchManager(
        {(64, 64, 1): [item]},
        batch_size=1,
        architecture=ARCHITECTURE_LTX2,
        target_fps=25.0,
        latent_idx_guide_frame_idx=2,
        latent_idx_guide_strength=0.75,
        keyframe_guide_frame_idx=-1,
        keyframe_guide_strength=1.0,
        keyframe_guide_extras=[{"frame_idx": 4, "strength": 0.5}],
    )

    batch = manager[0]

    assert batch["latents"]["latents"].shape == (1, 4, 1, 2, 2)
    assert batch["latents"]["fps"].tolist() == [25.0]
    assert batch["ref_latents"]["latents"].shape == (1, 4, 1, 2, 2)
    assert torch.all(batch["ref_latents"]["latents"] == 2.0)

    latent_idx_guide = batch["latent_idx_guide_latents"]
    assert latent_idx_guide["frame_idx"] == 2
    assert latent_idx_guide["strength"] == 0.75
    assert torch.all(latent_idx_guide["latents"] == 3.0)

    keyframe_guide = batch["keyframe_guide_latents"]
    assert keyframe_guide["frame_idx"] == -1
    assert keyframe_guide["strength"] == 1.0
    assert torch.all(keyframe_guide["latents"] == 4.0)

    assert len(batch["keyframe_guide_extras"]) == 1
    assert batch["keyframe_guide_extras"][0]["frame_idx"] == 4
    assert batch["keyframe_guide_extras"][0]["strength"] == 0.5
    assert torch.all(batch["keyframe_guide_extras"][0]["latents"] == 5.0)

    assert batch["conditions"]["video_prompt_embeds"].shape == (1, 3, 4)
    assert batch["conditions"]["prompt_attention_mask"].tolist() == [[1, 1, 0]]
    assert batch["captions"] == ["caption for sample"]


def test_ltx2_image_ic_config_normalization_is_architecture_scoped():
    from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

    user_config = {
        "general": {
            "resolution": [64, 64],
            "batch_size": 1,
            "caption_extension": ".txt",
        },
        "datasets": [
            {
                "image_directory": "images",
                "reference_directory": "refs",
                "reference_cache_directory": "ref_cache",
                "cache_directory": "cache",
            }
        ],
    }
    args = argparse.Namespace(debug_dataset=None)

    generator = BlueprintGenerator(ConfigSanitizer())
    blueprint = generator.generate(user_config, args, architecture="ltx2")
    params = blueprint.dataset_group.datasets[0].params
    assert params.control_directory == "refs"
    assert params.reference_cache_directory == "ref_cache"
    assert not hasattr(params, "reference_directory")

    unchanged = BlueprintGenerator._normalize_runtime_specific_user_config(user_config, {"architecture": "hv"})
    assert unchanged["datasets"][0]["reference_directory"] == "refs"
    assert "control_directory" not in unchanged["datasets"][0]

    ambiguous = {
        "general": user_config["general"],
        "datasets": [
            {
                "image_directory": "images",
                "reference_directory": "refs",
                "control_directory": "controls",
                "reference_cache_directory": "ref_cache",
                "cache_directory": "cache",
            }
        ],
    }
    with pytest.raises(ValueError, match="sets both reference_directory and control_directory"):
        generator.generate(ambiguous, args, architecture="ltx2")


def test_ltx2_trainer_interface_contract_after_training_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = _optional_module("musubi_tuner.training.trainer_base")
    metadata = importlib.import_module("musubi_tuner.training.metadata")
    outputs = importlib.import_module("musubi_tuner.training.outputs")

    assert hasattr(hv_train_network, "NetworkTrainer")
    if trainer_base is None:
        pytest.skip("upstream training module split has not landed yet")

    assert hv_train_network.NetworkTrainer is hv_train_network.HunyuanVideoNetworkTrainer
    assert issubclass(hv_train_network.NetworkTrainer, trainer_base.NetworkTrainer)
    assert hv_train_network.NetworkTrainer is not trainer_base.NetworkTrainer
    assert hv_train_network.DiTOutput is trainer_base.DiTOutput
    assert hv_train_network.DiTOutput is outputs.DiTOutput
    assert hv_train_network.SS_METADATA_MINIMUM_KEYS is metadata.SS_METADATA_MINIMUM_KEYS
    assert trainer_base.SS_METADATA_MINIMUM_KEYS is metadata.SS_METADATA_MINIMUM_KEYS
    assert hv_train_network.SS_METADATA_KEY_BASE_MODEL_VERSION == metadata.SS_METADATA_KEY_BASE_MODEL_VERSION

    from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer

    import inspect

    call_dit_params = inspect.signature(LTX2NetworkTrainer.call_dit).parameters
    assert any(param.kind is inspect.Parameter.VAR_KEYWORD for param in call_dit_params.values())


def test_training_helper_modules_own_fork_extensions_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    accelerator_setup = importlib.import_module("musubi_tuner.training.accelerator_setup")
    parser_common = importlib.import_module("musubi_tuner.training.parser_common")
    runtime_utils = importlib.import_module("musubi_tuner.training.runtime_utils")
    sampling_prompts = importlib.import_module("musubi_tuner.training.sampling_prompts")
    timesteps = importlib.import_module("musubi_tuner.training.timesteps")

    assert hv_train_network.read_config_from_file is parser_common.read_config_from_file

    assert hv_train_network.clean_memory_on_device is accelerator_setup.clean_memory_on_device
    assert hv_train_network.collator_class is accelerator_setup.collator_class
    assert hv_train_network.prepare_accelerator is accelerator_setup.prepare_accelerator
    assert hv_train_network.configure_console_output_for_help is runtime_utils.configure_console_output_for_help
    assert hv_train_network._update_global_peak is runtime_utils.update_global_peak
    assert hv_train_network._log_vram is runtime_utils.log_vram
    assert hv_train_network._log_cuda_memory_stats is runtime_utils.log_cuda_memory_stats
    assert hv_train_network.offload_optimizer_state_during_validation is runtime_utils.offload_optimizer_state_during_validation

    assert hv_train_network.line_to_prompt_dict is sampling_prompts.line_to_prompt_dict
    assert hv_train_network.load_prompts is sampling_prompts.load_prompts
    assert hv_train_network.should_sample_images is sampling_prompts.should_sample_images

    assert hv_train_network.compute_density_for_timestep_sampling is timesteps.compute_density_for_timestep_sampling
    assert hv_train_network.get_sigmas is timesteps.get_sigmas
    assert hv_train_network.compute_loss_weighting_for_sd3 is timesteps.compute_loss_weighting_for_sd3

    parsed = sampling_prompts.line_to_prompt_dict("prompt text --fr 25 --v reference.mp4 --ra reference.wav --i image.png")
    assert parsed["frame_rate"] == 25.0
    assert parsed["v2v_ref_path"] == "reference.mp4"
    assert parsed["ref_audio_path"] == "reference.wav"
    assert parsed["image_path"] == "image.png"


def test_architecture_trainers_import_shared_symbols_from_split_modules():
    source_root = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner"
    trainer_files = [
        "fpack_train_network.py",
        "flux_2_train_network.py",
        "flux_kontext_train_network.py",
        "hv_1_5_train_network.py",
        "kandinsky5_train_network.py",
        "ltx2_train_network.py",
        "qwen_image_train_network.py",
        "wan_train_network.py",
        "zimage_train_network.py",
    ]
    parser_consumers = set(trainer_files) - {"ltx2_train_network.py"}

    for trainer_file in trainer_files:
        source = (source_root / trainer_file).read_text(encoding="utf-8")
        assert "from musubi_tuner.hv_train_network import" not in source
        assert "from musubi_tuner.training.trainer_base import NetworkTrainer" in source

        if trainer_file in parser_consumers:
            assert "from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common" in source


def test_support_modules_import_training_helpers_from_split_modules():
    source_root = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner"
    support_files = [
        "flux_2_train_network_self_flow.py",
        "gui_dashboard/cli_defaults.py",
        "kandinsky5_generate_video.py",
        "ltx2_args.py",
        "ltx2_cache_latents.py",
        "ltx2_cache_rollouts.py",
        "ltx2_cache_text_encoder_outputs.py",
        "ltx2_rl_rounds.py",
        "ltx2_sampling.py",
    ]

    for support_file in support_files:
        source = (source_root / support_file).read_text(encoding="utf-8")
        assert "from musubi_tuner.hv_train_network import" not in source


def test_legacy_training_drivers_import_helpers_from_split_modules():
    source_root = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner"
    driver_files = [
        "ltx2_estimate.py",
        "ltx2_train.py",
        "ltx2_train_rl.py",
        "ltx2_train_slider.py",
        "qwen_image_train.py",
        "zimage_train.py",
    ]

    for driver_file in driver_files:
        source = (source_root / driver_file).read_text(encoding="utf-8")
        assert "from musubi_tuner.hv_train_network import" not in source


def test_split_parser_common_preserves_legacy_common_cli_surface():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    parser_common = importlib.import_module("musubi_tuner.training.parser_common")

    def option_strings(parser):
        return {option for action in parser._actions for option in action.option_strings}

    assert hv_train_network.setup_parser_common is parser_common.setup_parser_common
    assert option_strings(parser_common.setup_parser_common()) == option_strings(hv_train_network.setup_parser_common())

    parsed = parser_common.setup_parser_common().parse_args(
        [
            "--dataset_manifest",
            "manifest.json",
            "--cuda_memory_fraction",
            "0.5",
            "--full_bf16",
            "--disable_timestep_distribution_tensorboard",
            "--log_timestep_distribution_interval",
            "7",
            "--reset_optimizer",
            "--differential_guidance",
        ]
    )
    assert parsed.dataset_manifest == Path("manifest.json")
    assert parsed.cuda_memory_fraction == 0.5
    assert parsed.full_bf16 is True
    assert parsed.log_timestep_distribution_tensorboard is False
    assert parsed.log_timestep_distribution_interval == 7
    assert parsed.reset_optimizer is True
    assert parsed.differential_guidance is True
    assert parser_common.setup_parser_common().parse_args([]).output_dir == "output"


def test_training_loss_helpers_own_masked_loss_contract_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    losses = importlib.import_module("musubi_tuner.training.losses")

    assert hv_train_network._per_element_loss is losses.per_element_loss
    assert losses._per_element_loss is losses.per_element_loss
    assert hv_train_network.apply_loss_mask is losses.apply_loss_mask

    pred = torch.tensor([[1.0, 3.0], [2.0, 5.0]]).unsqueeze(-1)
    tgt = torch.ones_like(pred)
    per_elem = losses.per_element_loss(pred, tgt, "mse")
    assert per_elem.squeeze(-1).tolist() == [[0.0, 4.0], [1.0, 16.0]]

    mask = torch.tensor([1.0, 0.0])
    masked_loss, metrics = losses.apply_loss_mask(per_elem, mask)
    assert torch.isclose(masked_loss, torch.tensor(2.0))
    assert metrics["mask_active"] == 0.5
    assert metrics["loss_unmasked"] == 5.25
    assert metrics["loss_masked"] == 2.0


def test_trainer_step_logging_is_owned_by_split_module_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    trainer_logging = importlib.import_module("musubi_tuner.training.trainer_logging")

    assert hv_train_network.NetworkTrainer.generate_step_logs is trainer_logging.generate_step_logs
    assert trainer_base.NetworkTrainer.generate_step_logs is trainer_logging.generate_step_logs

    class DummyScheduler:
        def get_last_lr(self):
            return [0.1, 0.2]

    class DummyAutomagicOptimizer:
        def get_avg_learning_rate(self):
            return 0.15

        def get_lr_tensor(self):
            return torch.tensor([0.1, 0.2])

    logs = hv_train_network.NetworkTrainer().generate_step_logs(
        argparse.Namespace(optimizer_type="automagic"),
        current_loss=1.25,
        avr_loss=0.75,
        lr_scheduler=DummyScheduler(),
        lr_descriptions=["video", "audio"],
        optimizer=DummyAutomagicOptimizer(),
        video_loss=0.9,
        audio_loss=0.35,
        mask_metrics={"video_mask_active": 0.5, "audio_mask_active": 0.25},
    )

    assert logs["loss/current"] == 1.25
    assert logs["loss/video"] == 0.9
    assert logs["loss/audio"] == 0.35
    assert logs["loss/video_mask_active"] == 0.5
    assert logs["loss/audio_mask_active"] == 0.25
    assert logs["lr/video"] == 0.1
    assert logs["lr/audio"] == 0.2
    assert logs["lr/automagic_avg"] == 0.15
    assert logs["lr/automagic_min"] == pytest.approx(0.1)
    assert logs["lr/automagic_max"] == pytest.approx(0.2)


def test_trainer_optimizer_setup_is_owned_by_split_module_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    optimizer_setup = importlib.import_module("musubi_tuner.training.optimizer_setup")

    assert hv_train_network.NetworkTrainer.get_optimizer is optimizer_setup.get_optimizer
    assert hv_train_network.NetworkTrainer.is_schedulefree_optimizer is optimizer_setup.is_schedulefree_optimizer
    assert hv_train_network.NetworkTrainer.get_dummy_scheduler is optimizer_setup.get_dummy_scheduler
    assert hv_train_network.NetworkTrainer.get_lr_scheduler is optimizer_setup.get_lr_scheduler
    assert trainer_base.NetworkTrainer.get_optimizer is optimizer_setup.get_optimizer
    assert trainer_base.NetworkTrainer.get_lr_scheduler is optimizer_setup.get_lr_scheduler

    trainer = hv_train_network.NetworkTrainer()
    param = torch.nn.Parameter(torch.zeros(()))
    args = argparse.Namespace(
        optimizer_type="badam",
        optimizer_args=["base_optimizer_type=AdamW"],
        base_optimizer_args=["weight_decay=0.01"],
        learning_rate=0.001,
    )

    optimizer_name, optimizer_args, optimizer, train_fn, eval_fn = trainer.get_optimizer(args, [param])

    assert optimizer_name == "torch.optim.adamw.AdamW"
    assert optimizer_args == "weight_decay=0.01"
    assert isinstance(optimizer, torch.optim.AdamW)
    assert args.optimizer_type == "badam"
    assert args.optimizer_args == ["base_optimizer_type=AdamW"]
    train_fn()
    eval_fn()

    automagic_args = argparse.Namespace(optimizer_type="automagic")
    assert trainer.is_schedulefree_optimizer(optimizer, automagic_args)
    dummy_scheduler = trainer.get_dummy_scheduler(optimizer)
    assert dummy_scheduler.get_last_lr() == [0.001]
    lr_scheduler = trainer.get_lr_scheduler(automagic_args, optimizer, num_processes=1)
    assert lr_scheduler.get_last_lr() == [0.001]

    with pytest.raises(ValueError, match="Invalid --optimizer_args entry"):
        trainer.get_optimizer(
            argparse.Namespace(optimizer_type="AdamW", optimizer_args=["not_key_value"], learning_rate=0.001),
            [torch.nn.Parameter(torch.ones(()))],
        )


def test_trainer_optimizer_aux_helpers_are_owned_by_split_module_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    optimizer_setup = importlib.import_module("musubi_tuner.training.optimizer_setup")
    GroupWarmupScheduler = importlib.import_module("musubi_tuner.modules.group_lr_scheduler").GroupWarmupScheduler

    assert hv_train_network.NetworkTrainer._enable_lycoris_fp8_forward_compat is optimizer_setup.enable_lycoris_fp8_forward_compat
    assert hv_train_network.NetworkTrainer._maybe_wrap_group_warmup_scheduler is optimizer_setup.maybe_wrap_group_warmup_scheduler
    assert hv_train_network.NetworkTrainer._prepare_network_optimizer_params is optimizer_setup.prepare_network_optimizer_params
    assert hv_train_network.NetworkTrainer._copy_optimizer_state_subset is optimizer_setup.copy_optimizer_state_subset
    assert (
        hv_train_network.NetworkTrainer._refresh_optimizer_after_adaptive_rank_prune
        is optimizer_setup.refresh_optimizer_after_adaptive_rank_prune
    )
    assert (
        hv_train_network.NetworkTrainer._register_optimizer_resume_safe_globals
        is optimizer_setup.register_optimizer_resume_safe_globals
    )
    assert trainer_base.NetworkTrainer._prepare_network_optimizer_params is optimizer_setup.prepare_network_optimizer_params

    trainer = hv_train_network.NetworkTrainer()

    class DummyLayer:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float8_e4m3fn), requires_grad=False)
            self.bias = torch.nn.Parameter(torch.ones(2, dtype=torch.float8_e4m3fn), requires_grad=False)

    class DummyLora:
        def __init__(self, layer):
            self.org_module = [layer]

    class DummyNetwork:
        def __init__(self, layer):
            self.loras = [DummyLora(layer)]

    layer = DummyLayer()
    network = DummyNetwork(layer)
    trainer._enable_lycoris_fp8_forward_compat(
        argparse.Namespace(network_module="lycoris.kohya", fp8_base=True, fp8_scaled=False, mixed_precision="bf16"),
        network,
    )
    assert layer.weight.dtype == torch.bfloat16
    assert layer.bias.dtype == torch.bfloat16
    assert network._lycoris_fp8_forward_compat_applied

    class DummyPreparedNetwork:
        def prepare_optimizer_params(self, *, unet_lr, audio_lr, lr_args):
            self.received = (unet_lr, audio_lr, lr_args)
            return ["params"], ["desc"]

    prepared_network = DummyPreparedNetwork()
    assert trainer._prepare_network_optimizer_params(
        argparse.Namespace(network_module="lora", learning_rate=0.001, audio_lr=0.002, lr_args=["a=b"]),
        prepared_network,
    ) == (["params"], ["desc"])
    assert prepared_network.received == (0.001, 0.002, ["a=b"])

    p1 = object()
    p2 = object()
    state = defaultdict(dict, {p1: {"keep": True}, p2: {"drop": True}})
    copied = trainer._copy_optimizer_state_subset(state, {id(p1)})
    assert isinstance(copied, defaultdict)
    assert copied == {p1: {"keep": True}}

    param1 = torch.nn.Parameter(torch.ones(()))
    param2 = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [param1], "lr": 1.0, "group_name": "video"},
            {"params": [param2], "lr": 1.0, "group_name": "audio"},
        ],
        lr=1.0,
    )

    class DummyScheduler:
        last_epoch = 1

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass

        def get_last_lr(self):
            return [group["lr"] for group in optimizer.param_groups]

        def step(self):
            self.last_epoch += 1

    base_scheduler = DummyScheduler()
    assert trainer._maybe_wrap_group_warmup_scheduler(base_scheduler, optimizer, 10, {}) is base_scheduler
    wrapped = trainer._maybe_wrap_group_warmup_scheduler(base_scheduler, optimizer, 10, {"audio": 5})
    assert isinstance(wrapped, GroupWarmupScheduler)
    assert wrapped.get_last_lr()[0] == pytest.approx(1.0)
    assert wrapped.get_last_lr()[1] == pytest.approx(2.0)

    trainer._register_optimizer_resume_safe_globals(argparse.Namespace(optimizer_type="AdamW"))

    old_param = torch.nn.Parameter(torch.ones(()))
    extra_param = torch.nn.Parameter(torch.full((), 2.0))
    new_param = torch.nn.Parameter(torch.full((), 3.0))
    prune_optimizer = torch.optim.SGD(
        [
            {"params": [old_param], "lr": 0.01, "group_name": "network"},
            {"params": [extra_param], "lr": 0.02, "group_name": "extra"},
        ],
        lr=0.01,
    )
    prune_optimizer.state[old_param]["old"] = torch.tensor(1.0)
    prune_optimizer.state[extra_param]["extra"] = torch.tensor(2.0)

    class PrunedNetwork:
        def prepare_optimizer_params(self, *, unet_lr, audio_lr, lr_args):
            return [{"params": [new_param], "lr": unet_lr, "group_name": "network"}], ["network"]

    class DummyUnwrapAccelerator:
        num_processes = 1

        def unwrap_model(self, network):
            return network

    scheduler_args = argparse.Namespace(
        network_module="lora",
        learning_rate=0.03,
        audio_lr=None,
        lr_args=None,
        optimizer_type="SGD",
        lr_scheduler="constant",
        lr_scheduler_type=None,
        lr_scheduler_args=None,
        lr_warmup_steps=0,
        lr_decay_steps=0,
        max_train_steps=10,
        lr_scheduler_num_cycles=1,
        lr_scheduler_power=1.0,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        lr_group_warmup_args=None,
    )
    refreshed_scheduler, descriptions = trainer._refresh_optimizer_after_adaptive_rank_prune(
        scheduler_args,
        DummyUnwrapAccelerator(),
        PrunedNetwork(),
        prune_optimizer,
        object(),
        old_network_param_ids={id(old_param)},
        global_step=0,
    )

    assert descriptions == ["network"]
    assert hasattr(refreshed_scheduler, "get_last_lr")
    assert [group["params"] for group in prune_optimizer.param_groups] == [[new_param], [extra_param]]
    assert old_param not in prune_optimizer.state
    assert extra_param in prune_optimizer.state


def test_trainer_default_hooks_are_owned_by_split_module_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    trainer_hooks = importlib.import_module("musubi_tuner.training.trainer_hooks")

    for name in (
        "is_model_parallel_enabled",
        "validate_model_parallel_setup",
        "enable_model_parallel_transformer",
        "place_network_for_model_parallel",
        "clip_grad_norm_for_model_parallel",
        "pre_train_hook",
        "compute_prior_divergence_addition",
        "preservation_backward",
        "compute_validation_extra_loss",
        "modify_video_loss_per_element",
        "modify_audio_loss_per_element",
        "compute_video_extra_loss",
        "apply_differential_guidance_target",
    ):
        assert getattr(hv_train_network.NetworkTrainer, name) is getattr(trainer_hooks, name)
        assert getattr(trainer_base.NetworkTrainer, name) is getattr(trainer_hooks, name)

    from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer

    assert LTX2NetworkTrainer.is_model_parallel_enabled is not trainer_hooks.is_model_parallel_enabled
    assert LTX2NetworkTrainer.pre_train_hook is not trainer_hooks.pre_train_hook

    trainer = hv_train_network.NetworkTrainer()
    args = argparse.Namespace(differential_guidance=False)
    pred = torch.tensor([2.0])
    target = torch.tensor([4.0])
    assert trainer.apply_differential_guidance_target(args, pred, target) is target

    guided = trainer.apply_differential_guidance_target(
        argparse.Namespace(differential_guidance=True, differential_guidance_scale=2.0),
        pred,
        target,
    )
    assert guided.tolist() == [6.0]

    with pytest.raises(ValueError, match="differential_guidance_scale"):
        trainer.apply_differential_guidance_target(
            argparse.Namespace(differential_guidance=True, differential_guidance_scale=float("nan")),
            pred,
            target,
        )

    per_elem = torch.ones(1)
    assert trainer.modify_video_loss_per_element(args, per_elem, {}, torch.float32) == (per_elem, {})
    assert trainer.modify_audio_loss_per_element(args, per_elem, {}, torch.float32) == (per_elem, {})
    assert trainer.compute_video_extra_loss(args, {}, torch.float32) == (None, {})
    assert trainer.preservation_backward(args, None, None, None, torch.float32) == {}
    assert trainer.compute_validation_extra_loss(args, None, None, None, {}, 1, torch.float32) == (None, {})

    class DummyAccelerator:
        def clip_grad_norm_(self, params, max_norm):
            return params, max_norm

    assert trainer.clip_grad_norm_for_model_parallel(
        argparse.Namespace(max_grad_norm=1.5),
        DummyAccelerator(),
        ["param"],
        None,
    ) == (["param"], 1.5)


def test_trainer_batch_state_accessors_are_owned_by_split_module_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    batch_state = importlib.import_module("musubi_tuner.training.batch_state")

    assert hv_train_network.NetworkTrainer.set_current_batch_latents_info is batch_state.set_current_batch_latents_info
    assert hv_train_network.NetworkTrainer.get_current_batch_latents_info is batch_state.get_current_batch_latents_info
    assert trainer_base.NetworkTrainer.set_current_batch_latents_info is batch_state.set_current_batch_latents_info
    assert trainer_base.NetworkTrainer.get_current_batch_latents_info is batch_state.get_current_batch_latents_info

    trainer = hv_train_network.NetworkTrainer()
    assert trainer.get_current_batch_latents_info() is None
    latents_info = {"latents": torch.ones(1), "fps": torch.tensor([25.0])}
    trainer.set_current_batch_latents_info(latents_info)
    assert trainer.get_current_batch_latents_info() is latents_info
    trainer.set_current_batch_latents_info(None)
    assert trainer.get_current_batch_latents_info() is None


def test_trainer_resume_helpers_are_owned_by_split_module_after_split(tmp_path: Path):
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    resume_utils = importlib.import_module("musubi_tuner.training.resume_utils")
    train_utils = importlib.import_module("musubi_tuner.utils.train_utils")

    assert hv_train_network.NetworkTrainer.resume_from_local_or_hf_if_specified is resume_utils.resume_from_local_or_hf_if_specified
    assert hv_train_network.NetworkTrainer._recover_global_step is resume_utils.recover_global_step
    assert hv_train_network.NetworkTrainer._state_dir_matches_output_name is resume_utils.state_dir_matches_output_name
    assert hv_train_network.NetworkTrainer._find_latest_state_dir is resume_utils.find_latest_state_dir
    assert trainer_base.NetworkTrainer._find_latest_state_dir is resume_utils.find_latest_state_dir

    trainer = hv_train_network.NetworkTrainer()
    assert trainer._state_dir_matches_output_name("model-state", "model")
    assert trainer._state_dir_matches_output_name("model-step000012-state", "model")
    assert not trainer._state_dir_matches_output_name("other-step000999-state", "model")

    def make_state_dir(name: str, *, metadata_step: int | None = None, scheduler_step: int | None = None, complete=True):
        state_dir = tmp_path / name
        state_dir.mkdir()
        if metadata_step is not None:
            train_utils.save_resume_metadata(str(state_dir), metadata_step, step_in_epoch=0, epoch=0)
        if scheduler_step is not None:
            torch.save({"last_epoch": scheduler_step}, state_dir / "scheduler.bin")
        if complete:
            (state_dir / "model.safetensors").write_bytes(b"model")
            (state_dir / "optimizer.bin").write_bytes(b"optimizer")
        return state_dir

    state5 = make_state_dir("model-step000005-state", metadata_step=5)
    state12 = make_state_dir("model-step000012-state", metadata_step=12)
    make_state_dir("other-step000999-state", metadata_step=999)
    make_state_dir("model-step000030-state", metadata_step=30, complete=False)
    scheduler_state = make_state_dir("model-000001-state", scheduler_step=7)

    latest = trainer._find_latest_state_dir(argparse.Namespace(output_dir=str(tmp_path), output_name="model"))
    assert Path(latest).name == "model-step000012-state"
    assert trainer._recover_global_step(str(state5)) == 5
    assert trainer._recover_global_step(str(scheduler_state)) == 7

    class DummyAccelerator:
        def __init__(self):
            self.loaded = None

        def load_state(self, path):
            self.loaded = path

    accelerator = DummyAccelerator()
    resume_args = argparse.Namespace(
        resume=str(state12),
        resume_from_huggingface=False,
        _autoresume_selected=False,
        optimizer_type="AdamW",
    )
    assert trainer.resume_from_local_or_hf_if_specified(accelerator, resume_args) == 12
    assert accelerator.loaded == str(state12)
    assert trainer._resume_state_dir == str(state12)

    autoresume_args = argparse.Namespace(
        resume=str(tmp_path / "model-step000030-state"),
        resume_from_huggingface=False,
        _autoresume_selected=True,
        optimizer_type="AdamW",
    )
    assert trainer.resume_from_local_or_hf_if_specified(accelerator, autoresume_args) == 0
    assert autoresume_args.resume is None
    assert trainer._resume_state_dir is None


def test_trainer_timestep_logging_helpers_are_owned_by_split_module_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    timestep_logging = importlib.import_module("musubi_tuner.training.timestep_logging")

    assert hv_train_network.NetworkTrainer._get_tensorboard_writer is timestep_logging.get_tensorboard_writer
    assert (
        hv_train_network.NetworkTrainer._should_log_timestep_distribution_to_tensorboard
        is timestep_logging.should_log_timestep_distribution_to_tensorboard
    )
    assert (
        hv_train_network.NetworkTrainer._get_timestep_distribution_logging_payload
        is timestep_logging.get_timestep_distribution_logging_payload
    )
    assert (
        hv_train_network.NetworkTrainer._prepare_timestep_distribution_values
        is timestep_logging.prepare_timestep_distribution_values
    )
    assert hv_train_network.NetworkTrainer._accumulate_timestep_distribution is timestep_logging.accumulate_timestep_distribution
    assert (
        hv_train_network.NetworkTrainer._log_timestep_distribution_histogram is timestep_logging.log_timestep_distribution_histogram
    )
    assert trainer_base.NetworkTrainer._get_tensorboard_writer is timestep_logging.get_tensorboard_writer

    class DummyWriter:
        def __init__(self):
            self.histograms = []
            self.scalars = []

        def add_histogram(self, tag, values, *, global_step, bins=None):
            self.histograms.append((tag, values.clone(), global_step, bins))

        def add_scalar(self, tag, value, global_step):
            self.scalars.append((tag, value, global_step))

    class DummyTracker:
        name = "tensorboard"

        def __init__(self, writer):
            self.writer = writer

    class DummyAccelerator:
        def __init__(self, writer, *, is_main_process=True, num_processes=1):
            self.trackers = [DummyTracker(writer)]
            self.is_main_process = is_main_process
            self.num_processes = num_processes

        def gather(self, values):
            return values

    trainer = hv_train_network.NetworkTrainer()
    writer = DummyWriter()
    accelerator = DummyAccelerator(writer)

    assert trainer._get_tensorboard_writer(accelerator) is writer
    assert trainer._should_log_timestep_distribution_to_tensorboard(
        argparse.Namespace(log_timestep_distribution_tensorboard=True),
        accelerator,
    )
    assert not trainer._should_log_timestep_distribution_to_tensorboard(
        argparse.Namespace(log_timestep_distribution_tensorboard=True),
        DummyAccelerator(writer, is_main_process=False),
    )

    timesteps = torch.tensor([1.0, float("nan"), float("inf"), 3.0])
    assert trainer._get_timestep_distribution_logging_payload(argparse.Namespace(), timesteps) == {"main": timesteps}
    values = trainer._prepare_timestep_distribution_values(timesteps, accelerator)
    assert values.tolist() == [1.0, 3.0]

    buffers = {}
    trainer._accumulate_timestep_distribution(buffers, "main", timesteps, accelerator)
    assert list(buffers) == ["main"]
    assert buffers["main"][0].tolist() == [1.0, 3.0]

    trainer._log_timestep_distribution_histogram(accelerator, 10, "timesteps/main", values)
    assert writer.histograms[0][0] == "timesteps/main"
    assert ("timesteps/main_mean", 2.0, 10) in writer.scalars
    assert ("timesteps/main_count", 2, 10) in writer.scalars


def test_trainer_model_helpers_are_owned_by_split_module_after_split(tmp_path: Path):
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    model_helpers = importlib.import_module("musubi_tuner.training.model_helpers")

    assert hv_train_network.NetworkTrainer.get_checkpoint_metadata is model_helpers.get_checkpoint_metadata
    assert hv_train_network.NetworkTrainer.post_save_checkpoint_hook is model_helpers.post_save_checkpoint_hook
    assert hv_train_network.NetworkTrainer.i2v_training.fget is model_helpers.i2v_training
    assert hv_train_network.NetworkTrainer.control_training.fget is model_helpers.control_training
    assert hv_train_network.NetworkTrainer._resolve_network_module is model_helpers.resolve_network_module
    assert hv_train_network.NetworkTrainer.convert_weight_keys is model_helpers.convert_weight_keys
    assert hv_train_network.NetworkTrainer.load_network_weights is model_helpers.load_network_weights
    assert trainer_base.NetworkTrainer.load_network_weights is model_helpers.load_network_weights

    from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer

    assert LTX2NetworkTrainer.get_checkpoint_metadata is not model_helpers.get_checkpoint_metadata
    assert LTX2NetworkTrainer.post_save_checkpoint_hook is not model_helpers.post_save_checkpoint_hook

    trainer = hv_train_network.NetworkTrainer()
    trainer._i2v_training = True
    trainer._control_training = False
    assert trainer.i2v_training is True
    assert trainer.control_training is False
    assert trainer.get_checkpoint_metadata(argparse.Namespace()) == {}
    assert trainer.post_save_checkpoint_hook(argparse.Namespace(), "file", "name", None) is None

    assert trainer._resolve_network_module("math").sqrt(4) == 2

    class ConverterModule:
        @staticmethod
        def convert_weight_keys(weights_sd):
            return {"converted": weights_sd["raw"] + 1}

    raw_weights = {"raw": torch.tensor([1.0])}
    converted = trainer.convert_weight_keys(raw_weights, ConverterModule)
    assert converted["converted"].tolist() == [2.0]

    lora_weights = {"lora_a": torch.tensor([3.0])}
    assert trainer.convert_weight_keys(lora_weights, object()) is lora_weights

    weights_path = tmp_path / "weights.pt"
    torch.save(lora_weights, weights_path)
    loaded = trainer.load_network_weights(str(weights_path), object())
    assert loaded["lora_a"].tolist() == [3.0]


def test_trainer_sampling_runtime_is_owned_by_split_module_after_split(tmp_path: Path):
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    sampling_runtime = importlib.import_module("musubi_tuner.training.sampling_runtime")

    assert hv_train_network.NetworkTrainer.sample_images is sampling_runtime.sample_images
    assert hv_train_network.NetworkTrainer.sample_image_inference is sampling_runtime.sample_image_inference
    assert trainer_base.NetworkTrainer.sample_images is sampling_runtime.sample_images
    assert trainer_base.NetworkTrainer.sample_image_inference is sampling_runtime.sample_image_inference

    trainer = hv_train_network.NetworkTrainer()
    trainer.sample_images(
        object(),
        argparse.Namespace(sample_at_first=False, sample_every_n_steps=None, sample_every_n_epochs=None),
        epoch=0,
        steps=0,
        vae=object(),
        transformer=object(),
        sample_parameters=[{"prompt": "unused"}],
        dit_dtype=torch.float32,
    )

    trainer.default_guidance_scale = 6.0
    trainer._i2v_training = True
    trainer._control_training = False
    trainer.sample_image_inference(
        argparse.Namespace(device=torch.device("cpu")),
        argparse.Namespace(output_name=None),
        transformer=object(),
        dit_dtype=torch.float32,
        vae=object(),
        save_dir=str(tmp_path),
        sample_parameter={},
        epoch=0,
        steps=0,
    )


def test_trainer_training_loop_is_owned_by_split_module_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    training_loop = importlib.import_module("musubi_tuner.training.training_loop")

    assert hv_train_network.NetworkTrainer.train is training_loop.train
    assert trainer_base.NetworkTrainer.train is training_loop.train


def test_trainer_bucketed_timestep_is_owned_by_split_module_after_split():
    hv_train_network = importlib.import_module("musubi_tuner.hv_train_network")
    trainer_base = importlib.import_module("musubi_tuner.training.trainer_base")
    timestep_sampling = importlib.import_module("musubi_tuner.training.timestep_sampling")

    assert hv_train_network.NetworkTrainer.get_bucketed_timestep is timestep_sampling.get_bucketed_timestep
    assert (
        hv_train_network.NetworkTrainer.get_noisy_model_input_and_timesteps is timestep_sampling.get_noisy_model_input_and_timesteps
    )
    assert hv_train_network.NetworkTrainer.show_timesteps is timestep_sampling.show_timesteps
    assert trainer_base.NetworkTrainer.get_bucketed_timestep is timestep_sampling.get_bucketed_timestep
    assert trainer_base.NetworkTrainer.get_noisy_model_input_and_timesteps is timestep_sampling.get_noisy_model_input_and_timesteps
    assert trainer_base.NetworkTrainer.show_timesteps is timestep_sampling.show_timesteps

    trainer = hv_train_network.NetworkTrainer()
    trainer.num_timestep_buckets = 4
    random.seed(1234)
    values = [trainer.get_bucketed_timestep() for _ in range(4)]
    assert trainer.timestep_range_pool == []
    assert all(0.0 <= value <= 1.0 for value in values)
    assert {min(int(value * 4), 3) for value in values} == {0, 1, 2, 3}

    trainer.num_timestep_buckets = 1
    assert 0.0 <= trainer.get_bucketed_timestep() <= 1.0

    from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer

    assert LTX2NetworkTrainer.get_noisy_model_input_and_timesteps is not timestep_sampling.get_noisy_model_input_and_timesteps

    latents = torch.zeros(2, 1, 1, 1)
    noise = torch.ones_like(latents)
    noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
        argparse.Namespace(
            timestep_sampling="uniform",
            min_timestep=None,
            max_timestep=None,
            preserve_distribution_shape=False,
        ),
        noise,
        latents,
        [0.25, 0.75],
        noise_scheduler=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert timesteps.tolist() == [251.0, 751.0]
    assert noisy_model_input[:, 0, 0, 0].tolist() == [0.25, 0.75]
