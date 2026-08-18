"""Tests for name-based cache skipping (--faster_check) and its probe gate."""

import os

from musubi_tuner.dataset.datasources import ImageDatasource
from musubi_tuner.dataset.image_video_dataset import (
    BaseDataset,
    ItemInfo,
    verify_faster_cache_plan,
)

ARCH = "mmh3"


class StubDatasource(ImageDatasource):
    def __init__(self, item_keys: list[str]) -> None:
        super().__init__()
        self.item_keys = item_keys

    def is_indexable(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self.item_keys)

    def get_item_key(self, idx: int) -> str:
        return self.item_keys[idx]


class StubDataset:
    architecture = ARCH
    plan_faster_cache_check = BaseDataset.plan_faster_cache_check

    def __init__(self, item_keys: list[str]) -> None:
        self.datasource = StubDatasource(item_keys)


def _plan(item_keys: list[str], cache_names: list[str], text: bool = False, probe_samples: int = 8):
    dataset = StubDataset(item_keys)
    paths = [os.path.join("cache", name) for name in cache_names]
    return dataset, dataset.plan_faster_cache_check(paths, text=text, probe_samples=probe_samples)


def test_matches_a_cache_written_for_the_item():
    dataset, plan = _plan(["d/foo.png"], [f"foo_0832x0480_{ARCH}.safetensors"])

    assert plan is not None
    assert plan.cached_item_keys == {"d/foo.png"}
    plan.install_skip_filter()
    assert dataset.datasource.should_include_item("d/foo.png") is False


def test_matches_a_frame_range_cache():
    _, plan = _plan(["d/clip.mp4"], [f"clip_00000-039_0832x0480_{ARCH}.safetensors"])

    assert plan is not None
    assert plan.cached_item_keys == {"d/clip.mp4"}


def test_ignores_a_cache_left_by_a_removed_item_with_a_longer_stem():
    """`foo_1`'s leftover cache is not a cache for `foo`."""
    _, plan = _plan(["d/foo.png"], [f"foo_1_0832x0480_{ARCH}.safetensors"])

    assert plan is None


def test_ignores_a_cache_whose_tail_is_not_a_generated_one():
    _, plan = _plan(["d/foo.png"], [f"foo_edited_{ARCH}.safetensors"])

    assert plan is None


def test_ignores_items_whose_stem_is_ambiguous():
    """Two sources with one stem already collide on one cache path."""
    _, plan = _plan(["a/foo.png", "b/foo.png"], [f"foo_0832x0480_{ARCH}.safetensors"])

    assert plan is None


def test_text_caches_match_without_dimensions():
    _, plan = _plan(["d/foo.png"], [f"foo_{ARCH}_te.safetensors"], text=True)

    assert plan is not None
    assert plan.cached_item_keys == {"d/foo.png"}


def test_text_caches_do_not_match_a_latent_style_name():
    _, plan = _plan(["d/foo.png"], [f"foo_0832x0480_{ARCH}_te.safetensors"], text=True)

    assert plan is None


def test_probe_filter_admits_only_the_probed_items():
    dataset, plan = _plan(
        ["d/a.png", "d/b.png", "d/c.png"],
        [f"a_0832x0480_{ARCH}.safetensors", f"b_0832x0480_{ARCH}.safetensors", f"c_0832x0480_{ARCH}.safetensors"],
        probe_samples=1,
    )

    assert len(plan.probe_item_keys) == 1
    plan.install_probe_filter()
    admitted = [key for key in ["d/a.png", "d/b.png", "d/c.png"] if dataset.datasource.should_include_item(key)]
    assert admitted == sorted(plan.probe_item_keys)


def _item(item_key: str, cache_path: str) -> ItemInfo:
    item = ItemInfo(item_key, "caption", (832, 480), latent_cache_path=cache_path)
    return item


def _batches(items: list[ItemInfo]):
    def retrieve():
        yield ("bucket", items)

    return retrieve


def test_probe_accepts_caches_that_still_validate():
    dataset, plan = _plan(["d/foo.png"], [f"foo_0832x0480_{ARCH}.safetensors"])
    cache_path = os.path.normpath(os.path.join("cache", f"foo_0832x0480_{ARCH}.safetensors"))

    rejection = verify_faster_cache_plan(
        plan,
        _batches([_item("d/foo.png", cache_path)]),
        lambda item: item.latent_cache_path,
        {cache_path},
        lambda item, path: True,
        unwrap_batch=lambda batch: batch[1],
    )

    assert rejection is None
    # The probe filter is always cleared, so a fallback run sees every item.
    assert dataset.datasource.should_include_item("d/foo.png") is True


def test_probe_rejects_when_the_expected_cache_name_changed():
    """Config drift renames the cache, which name matching alone cannot see."""
    _, plan = _plan(["d/foo.png"], [f"foo_0832x0480_{ARCH}.safetensors"])
    stale = os.path.normpath(os.path.join("cache", f"foo_0832x0480_{ARCH}.safetensors"))
    expected = os.path.normpath(os.path.join("cache", f"foo_1024x0576_{ARCH}.safetensors"))

    rejection = verify_faster_cache_plan(
        plan,
        _batches([_item("d/foo.png", expected)]),
        lambda item: item.latent_cache_path,
        {stale},
        lambda item, path: True,
        unwrap_batch=lambda batch: batch[1],
    )

    assert rejection is not None
    assert "does not exist" in rejection


def test_probe_rejects_when_the_cache_contents_no_longer_match():
    _, plan = _plan(["d/foo.png"], [f"foo_0832x0480_{ARCH}.safetensors"])
    cache_path = os.path.normpath(os.path.join("cache", f"foo_0832x0480_{ARCH}.safetensors"))

    rejection = verify_faster_cache_plan(
        plan,
        _batches([_item("d/foo.png", cache_path)]),
        lambda item: item.latent_cache_path,
        {cache_path},
        lambda item, path: False,
        unwrap_batch=lambda batch: batch[1],
    )

    assert rejection is not None
    assert "current configuration" in rejection
