import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musubi_tuner.gui_dashboard.routers.stats import (
    DatasetStats,
    _calculate_training_stats,
    _calculate_vram_stats,
)


def test_calculate_vram_stats_handles_nullable_numeric_fields():
    config = {
        "dataset": {
            "datasets": [
                {
                    "type": "video",
                    "resolution_w": 768,
                    "resolution_h": 512,
                    "target_frames": 33,
                    "batch_size": 1,
                }
            ]
        },
        "training": {
            "ltx2_mode": "av",
            "blocks_to_swap": None,
            "network_dim": None,
            "gradient_accumulation_steps": None,
            "ffn_chunk_size": None,
        },
    }

    stats = _calculate_vram_stats(config)

    assert stats is not None
    assert stats.breakdown["model"] > 0
    assert stats.breakdown["activations"] > 0
    assert "overhead" in stats.breakdown
    assert stats.breakdown["overhead"] >= 0


def test_calculate_training_stats_uses_dataset_batch_size_when_present():
    config = {
        "dataset": {"datasets": [{"type": "video", "batch_size": 4}]},
        "training": {
            "gradient_accumulation_steps": None,
            "network_dim": None,
            "max_train_steps": 1600,
        },
    }
    dataset_stats = DatasetStats(
        total_items=80,
        video_items=80,
        audio_items=0,
        avg_resolution=None,
        avg_frames=None,
        max_resolution=None,
        max_frames=None,
    )

    stats = _calculate_training_stats(config, dataset_stats)

    assert stats is not None
    assert stats.effective_batch_size == 4
    assert stats.steps_per_epoch == 20
    assert stats.total_epochs == 80
    assert stats.checkpoint_size_mb == 80
