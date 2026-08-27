from __future__ import annotations

import pytest
import torch

from musubi_tuner.minimax_h3.cache import H3_DOP_CONFIG_CACHE_KEY
from musubi_tuner.minimax_h3.dop import dop_config_identity, rewrite_dop_caption
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser


def test_h3_dop_rewrite_removes_trigger_without_duplicating_class() -> None:
    assert rewrite_dop_caption("sks woman walking in a park", "sks", "woman") == "woman walking in a park"
    assert rewrite_dop_caption("portrait of sks", "sks", "a woman") == "portrait of a woman"


def test_h3_dop_rewrite_matches_only_a_standalone_trigger() -> None:
    with pytest.raises(ValueError, match="absent"):
        rewrite_dop_caption("asks a question", "sks", "woman")


def test_h3_dop_identity_is_stable_and_pair_specific() -> None:
    identity = dop_config_identity("sks", "woman")
    assert identity.dtype is torch.uint8
    assert identity.shape == (32,)
    assert torch.equal(identity, dop_config_identity(" sks ", " woman "))
    assert not torch.equal(identity, dop_config_identity("sks", "man"))


def test_h3_dop_identity_cache_key_declares_uint8_dtype() -> None:
    assert H3_DOP_CONFIG_CACHE_KEY.endswith("_uint8")


def test_h3_dop_defaults_are_disabled() -> None:
    args = create_parser().parse_args([])
    assert args.h3_dop_loss_weight == 0.0
    assert args.h3_dop_probability == 1.0


def test_h3_dop_requires_trigger_and_class() -> None:
    args = create_parser().parse_args(["--h3_dop_loss_weight", "0.02", "--h3_dop_trigger", "sks"])
    with pytest.raises(ValueError, match="h3_dop_class_prompt"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_dop_rejects_ref2va() -> None:
    args = create_parser().parse_args(
        [
            "--h3_training_mode",
            "ref2va",
            "--h3_dop_loss_weight",
            "0.02",
            "--h3_dop_trigger",
            "sks",
            "--h3_dop_class_prompt",
            "woman",
        ]
    )
    with pytest.raises(ValueError, match="not supported for Ref2VA"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)
