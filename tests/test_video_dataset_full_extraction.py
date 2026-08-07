from __future__ import annotations

import logging

from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3


class _Datasource:
    has_control = False

    def __init__(self, *_args, **_kwargs):
        pass


def test_h3_full_extraction_does_not_round_unused_target_frames(monkeypatch, caplog):
    monkeypatch.setattr(image_video_dataset, "VideoDirectoryDatasource", _Datasource)

    with caplog.at_level(logging.WARNING):
        dataset = image_video_dataset.VideoDataset(
            resolution=(640, 352),
            caption_extension=".txt",
            batch_size=1,
            num_repeats=1,
            enable_bucket=True,
            bucket_no_upscale=False,
            frame_extraction="full",
            target_frames=[1],
            max_frames=9_999,
            video_directory="videos",
            cache_directory="cache",
            architecture=ARCHITECTURE_MINIMAX_H3,
        )

    assert dataset.target_frames is None
    assert "target_frames are rounded" not in caplog.text


def test_h3_full_extraction_never_produces_negative_frame_counts():
    resolve = image_video_dataset._full_extraction_frame_count

    assert resolve(1, 9_999, 5, 17) is None
    assert resolve(4, 9_999, 5, 17) is None
    assert resolve(5, 9_999, 5, 17) == 5
    assert resolve(24, 9_999, 5, 17) == 22
    assert resolve(73, 9_999, 5, 17) == 73
