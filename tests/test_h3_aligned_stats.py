import pytest

from musubi_tuner.gui_dashboard.routers.stats import _h3_dataset_rows


@pytest.mark.parametrize("aligned", ["0", "+0", "0;2"])
def test_aligned_av_stats_count_visual_rows_only(aligned):
    dataset = {"type": "video", "h3_target_modalities": "video", "source_reference_modalities": "av;audio;video"}
    caching = {"h3_task": "ref2va"}
    actual = _h3_dataset_rows({}, caching, {**dataset, "aligned_guide_indices": aligned})
    visual = _h3_dataset_rows({}, caching, {**dataset, "source_reference_modalities": "video;audio;video"})
    full = _h3_dataset_rows({}, caching, dataset)
    assert actual == visual < full


@pytest.mark.parametrize("aligned", ["unfinished", "0,0", "-1", "0,unfinished"])
def test_invalid_aligned_stats_remain_conservative(aligned):
    dataset = {"source_reference_modalities": "av", "type": "video"}
    caching = {"h3_task": "ref2va"}
    assert _h3_dataset_rows({}, caching, {**dataset, "aligned_guide_indices": aligned}) == _h3_dataset_rows({}, caching, dataset)
