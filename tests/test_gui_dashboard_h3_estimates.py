from musubi_tuner.gui_dashboard.routers.stats import (
    _calculate_training_stats,
    _calculate_vram_stats,
    _estimate_training_step_time_sec,
    _gpu_time_coefficient,
)


def _config(**training_overrides):
    training = {
        "model_type": "minimax_h3",
        "gradient_checkpointing": True,
        "h3_gradient_checkpointing_blocks": None,
        "gradient_checkpointing_cpu_offload": True,
        "h3_reusable_activation_offload": True,
        "blocks_to_swap": 48,
        "block_swap_h2d_only": True,
        "block_swap_granularity": "block",
        "block_swap_ring_size": 2,
        "use_pinned_memory_for_block_swap": True,
        "h3_base_preservation_loss_weight": 0.02,
        "h3_base_preservation_probability": 1.0,
        "max_train_steps": 100,
    }
    training.update(training_overrides)
    return {
        "training": training,
        "dataset": {
            "datasets": [
                {
                    "type": "video",
                    "resolution_w": 832,
                    "resolution_h": 480,
                    "target_frames": 124,
                    "batch_size": 1,
                }
            ]
        },
    }


def test_h3_reference_step_estimate_matches_synchronized_workbox_range():
    seconds = _estimate_training_step_time_sec(_config())
    assert 25.0 <= seconds <= 31.0


def test_h3_sparse_preservation_reduces_average_time_not_to_zero():
    dense = _estimate_training_step_time_sec(_config())
    sparse = _estimate_training_step_time_sec(_config(h3_base_preservation_probability=0.25))
    disabled = _estimate_training_step_time_sec(_config(h3_base_preservation_loss_weight=0.0))
    assert disabled < sparse < dense


def test_h3_extra_forwards_and_transfer_features_change_time_directionally():
    base = _estimate_training_step_time_sec(_config(h3_base_preservation_loss_weight=0.0))
    guidance = _estimate_training_step_time_sec(_config(h3_base_preservation_loss_weight=0.0, h3_guidance_distillation_scale=4.0))
    bidirectional_swap = _estimate_training_step_time_sec(_config(h3_base_preservation_loss_weight=0.0, block_swap_h2d_only=False))
    assert guidance > base
    assert bidirectional_swap > base


def test_h3_gpu_coefficients_scale_the_same_configuration():
    fastest = _estimate_training_step_time_sec(_config(), "NVIDIA B200")
    reference = _estimate_training_step_time_sec(_config(), "NVIDIA H100 80GB HBM3")
    consumer = _estimate_training_step_time_sec(_config(), "NVIDIA GeForce RTX 4090")
    older = _estimate_training_step_time_sec(_config(), "NVIDIA GeForce RTX 3090")
    assert fastest < reference < consumer < older
    assert _gpu_time_coefficient("NVIDIA A100-SXM4-80GB") == 1.45


def test_h3_stats_identify_hardware_adjusted_source(monkeypatch):
    monkeypatch.setattr("musubi_tuner.gui_dashboard.routers.stats._detect_local_gpu_name", lambda: None)
    stats = _calculate_training_stats(_config(), None)
    assert stats is not None
    assert stats.estimated_time_source == "Hardware-adjusted estimate"


def test_h3_stats_include_detected_gpu_name(monkeypatch):
    monkeypatch.setattr(
        "musubi_tuner.gui_dashboard.routers.stats._detect_local_gpu_name",
        lambda: "NVIDIA GeForce RTX 4090",
    )
    stats = _calculate_training_stats(_config(), None)
    assert stats is not None
    assert stats.estimated_time_source == "Hardware-adjusted estimate (NVIDIA GeForce RTX 4090)"


def test_h3_sequential_dataset_batch_does_not_multiply_peak_vram():
    batch_one = _calculate_vram_stats(_config())
    config = _config()
    config["dataset"]["datasets"][0]["batch_size"] = 4
    batch_four = _calculate_vram_stats(config)
    assert batch_one is not None and batch_four is not None
    assert batch_four.peak_training_gb == batch_one.peak_training_gb
    assert batch_four.activations_gb == batch_one.activations_gb


def test_h3_server_vram_uses_h3_architecture_instead_of_ltx_constants():
    stats = _calculate_vram_stats(_config(blocks_to_swap=0, gradient_checkpointing_cpu_offload=False))

    assert stats is not None
    # 33.1B BF16 parameters are about 61.65 GiB. The former shared LTX path
    # incorrectly reported a 42 GiB base with 48 blocks and hidden size 4096.
    assert 61.0 <= stats.model_size_gb <= 62.0
    assert stats.activations_gb > 10.0


def test_h3_server_estimates_the_heaviest_training_dataset():
    config = _config(blocks_to_swap=0, gradient_checkpointing_cpu_offload=False)
    config["dataset"]["datasets"] = [
        {
            "type": "video",
            "resolution_w": 384,
            "resolution_h": 256,
            "target_frames": 22,
            "batch_size": 1,
        },
        {
            "type": "video",
            "resolution_w": 832,
            "resolution_h": 480,
            "target_frames": 124,
            "batch_size": 1,
        },
    ]
    mixed = _calculate_vram_stats(config)
    config["dataset"]["datasets"] = [config["dataset"]["datasets"][1]]
    largest = _calculate_vram_stats(config)

    assert mixed is not None and largest is not None
    assert mixed.peak_training_gb == largest.peak_training_gb
    assert mixed.activations_gb == largest.activations_gb


def test_h3_reference_conditioning_increases_estimated_packed_work():
    plain = _config(h3_base_preservation_loss_weight=0.0)
    conditioned = _config(h3_base_preservation_loss_weight=0.0)
    conditioned["caching"] = {"h3_task": "ref2va"}
    conditioned["dataset"]["datasets"][0].update(
        control_video_directory="references/video",
        control_audio_directory="references/audio",
        control_modality="av",
        reference_frames=124,
    )

    assert _estimate_training_step_time_sec(conditioned) > _estimate_training_step_time_sec(plain)
    assert _calculate_vram_stats(conditioned).activations_gb > _calculate_vram_stats(plain).activations_gb


def test_h3_image_reference_uses_its_own_short_edge_in_estimator():
    small = _config(h3_base_preservation_loss_weight=0.0, reference_image_short_edge=384)
    large = _config(h3_base_preservation_loss_weight=0.0, reference_image_short_edge=2048)
    for config in (small, large):
        config["caching"] = {"h3_task": "ref2va"}
        config["dataset"]["datasets"][0]["control_directory"] = "references/images"

    assert _estimate_training_step_time_sec(large) > _estimate_training_step_time_sec(small)
    assert _calculate_vram_stats(large).activations_gb > _calculate_vram_stats(small).activations_gb


def test_h3_crepa_accounts_for_projector_and_retained_activations():
    base = _calculate_vram_stats(_config(crepa=False))
    crepa = _calculate_vram_stats(_config(crepa=True, crepa_mode="backbone"))

    assert base is not None and crepa is not None
    assert crepa.breakdown["overhead"] > base.breakdown["overhead"] + 1.0
    assert crepa.peak_training_gb > base.peak_training_gb + 1.0


def test_h3_image_dataset_defaults_to_one_target_frame_in_estimator():
    config = _config(blocks_to_swap=0, gradient_checkpointing_cpu_offload=False)
    config["dataset"]["datasets"] = [
        {
            "type": "image",
            "resolution_w": 832,
            "resolution_h": 480,
            "batch_size": 1,
            # target_frames remains the legacy schema default and must not be
            # interpreted as a 124-frame H3 image target.
            "target_frames": 33,
        }
    ]
    stats = _calculate_vram_stats(config)

    assert stats is not None
    assert stats.activations_gb < 5.0
