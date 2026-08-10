from musubi_tuner.gui_dashboard.routers.stats import (
    _calculate_training_stats,
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
