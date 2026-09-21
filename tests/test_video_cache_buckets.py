"""Video caching must use the bucket flags recorded in the dataset config."""

import pytest

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_WAN
from musubi_tuner.dataset.image_video_dataset import VideoDataset


@pytest.mark.parametrize("architecture", [ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_WAN])
@pytest.mark.parametrize("enable_bucket,no_upscale", [(False, False), (True, False), (True, True)])
def test_video_cache_respects_bucket_configuration(tmp_path, architecture, enable_bucket, no_upscale):
    dataset = VideoDataset(
        resolution=(1792, 768),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=enable_bucket,
        bucket_no_upscale=no_upscale,
        target_frames=[5],
        video_directory=str(tmp_path),
        cache_directory=str(tmp_path),
        architecture=architecture,
    )
    assert list(dataset.retrieve_latent_cache_batches(1)) == []
    selector = dataset.datasource.bucket_selector
    if not enable_bucket:
        assert selector.bucket_resolutions == [(1792, 768)]
        assert selector.get_bucket_resolution((3840, 1080)) == (1792, 768)
    elif no_upscale:
        assert selector.get_bucket_resolution((320, 192)) == (320, 192)
    else:
        assert len(selector.bucket_resolutions) > 1
        width, height = selector.get_bucket_resolution((320, 192))
        assert width * height > 320 * 192
