"""Tests for Offloader eviction scoring methods."""
import torch
import torch.nn as nn

from musubi_tuner.ltx_2.model.ltx2_custom_offloading_utils import Offloader


def _make_offloader(**kwargs) -> Offloader:
    """Create a minimal Offloader for testing eviction logic."""
    defaults = dict(
        block_type="test",
        num_blocks=10,
        blocks_to_swap=4,
        device=torch.device("cpu"),
        use_pinned_memory=False,
        debug=False,
        prefetch_window=1,
    )
    defaults.update(kwargs)
    return Offloader(**defaults)


class TestScoreForEviction:
    def test_already_processed_block_scores_high(self):
        off = _make_offloader(num_blocks=10)
        # Block 2, current is 5: already processed → score = 10 + (5 - 2) = 13
        assert off._score_for_eviction(2, 5, 10) == 13

    def test_future_block_scores_by_distance(self):
        off = _make_offloader(num_blocks=10)
        # Block 8, current is 5: not yet processed → score = 8 - 5 = 3
        assert off._score_for_eviction(8, 5, 10) == 3

    def test_current_block_scores_num_blocks(self):
        off = _make_offloader(num_blocks=10)
        # Block 5, current is 5: already processed → score = 10 + 0 = 10
        assert off._score_for_eviction(5, 5, 10) == 10

    def test_next_block_scores_one(self):
        off = _make_offloader(num_blocks=10)
        # Block 6, current is 5: score = 6 - 5 = 1
        assert off._score_for_eviction(6, 5, 10) == 1


class TestPickEvictionCandidate:
    def test_picks_furthest_block(self):
        off = _make_offloader(num_blocks=10)
        off.gpu_resident_blocks = {3, 6, 8}
        # current_block=5: block 3 → score 12, block 6 → score 1, block 8 → score 3
        victim = off._pick_eviction_candidate(5, 10)
        assert victim == 3

    def test_respects_protected_set(self):
        off = _make_offloader(num_blocks=10)
        off.gpu_resident_blocks = {3, 6, 8}
        # Protect block 3 — should pick block 8 (score 3) over block 6 (score 1)
        victim = off._pick_eviction_candidate(5, 10, protected={3})
        assert victim == 8

    def test_returns_none_when_all_protected(self):
        off = _make_offloader(num_blocks=10)
        off.gpu_resident_blocks = {3, 6}
        victim = off._pick_eviction_candidate(5, 10, protected={3, 6})
        assert victim is None

    def test_returns_none_when_no_residents(self):
        off = _make_offloader(num_blocks=10)
        off.gpu_resident_blocks = set()
        victim = off._pick_eviction_candidate(5, 10)
        assert victim is None

    def test_single_candidate(self):
        off = _make_offloader(num_blocks=10)
        off.gpu_resident_blocks = {7}
        victim = off._pick_eviction_candidate(5, 10)
        assert victim == 7

    def test_eviction_prefers_earlier_processed(self):
        """Earlier processed blocks have higher scores and should be evicted first."""
        off = _make_offloader(num_blocks=20)
        off.gpu_resident_blocks = {1, 2, 3, 15, 16}
        # current=10: blocks 1,2,3 are processed; 15,16 are future
        # Scores: 1→29, 2→28, 3→27, 15→5, 16→6
        victim = off._pick_eviction_candidate(10, 20)
        assert victim == 1  # highest score
