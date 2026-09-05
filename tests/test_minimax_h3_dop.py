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


def test_h3_dop_bare_mode_removes_the_trigger_and_tidies_the_seam() -> None:
    assert rewrite_dop_caption("sks woman walking in a park", "sks", "", "bare") == "woman walking in a park"
    assert rewrite_dop_caption("xyz style. A wide shot of a table.", "xyz style", "", "bare") == "A wide shot of a table."
    assert rewrite_dop_caption("a woman, sks, walks", "sks", "", "bare") == "a woman, walks"
    assert rewrite_dop_caption("First sentence. sks. Second sentence.", "sks", "", "bare") == "First sentence. Second sentence."
    assert rewrite_dop_caption("hello! sks! world", "sks", "", "bare") == "hello! world"
    assert rewrite_dop_caption("foo (sks) bar", "sks", "", "bare") == "foo bar"
    assert rewrite_dop_caption("sks... Next", "sks", "", "bare") == "Next"
    assert rewrite_dop_caption("one sks two sks three", "sks", "", "bare") == "one two three"
    assert rewrite_dop_caption("she waits... then sks", "sks", "", "bare") == "she waits... then"
    assert rewrite_dop_caption("A sks's portrait", "sks", "", "bare") == "A portrait"
    assert rewrite_dop_caption("sks\u2019s portrait", "sks", "", "bare") == "portrait"
    assert rewrite_dop_caption("Before -- sks -- after", "sks", "", "bare") == "Before -- after"
    assert rewrite_dop_caption("Before / sks / after", "sks", "", "bare") == "Before / after"
    assert rewrite_dop_caption("Before \u2014 sks \u2014 after", "sks", "", "bare") == "Before \u2014 after"
    assert rewrite_dop_caption("Before \u2013 sks \u2013 after", "sks", "", "bare") == "Before \u2013 after"
    with pytest.raises(ValueError, match="takes no class prompt"):
        rewrite_dop_caption("sks woman", "sks", "woman", "bare")
    with pytest.raises(ValueError, match="empty without its trigger"):
        rewrite_dop_caption("sks", "sks", "", "bare")


def test_h3_dop_identity_distinguishes_caption_modes_and_keeps_class_bytes() -> None:
    assert torch.equal(dop_config_identity("sks", "woman"), dop_config_identity("sks", "woman", "class"))
    # The v1 bytes of the class payload, which every existing DOP cache was written with.
    assert (
        bytes(dop_config_identity("sks", "woman").tolist()).hex()
        == "be36aa933847b786aed7d88341d85b1fd7627f7518ac751ffc7a4ee74d094ea3"
    )
    assert not torch.equal(dop_config_identity("sks", ""), dop_config_identity("sks", "", "bare"))
    with pytest.raises(ValueError, match="unknown DOP caption mode"):
        dop_config_identity("sks", "", "prefix")


def test_h3_dop_bare_mode_needs_only_the_trigger() -> None:
    args = create_parser().parse_args(
        ["--sdpa", "--h3_dop_loss_weight", "0.02", "--h3_dop_trigger", "sks", "--h3_dop_caption_mode", "bare"]
    )
    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)
    args = create_parser().parse_args(
        [
            "--h3_dop_loss_weight",
            "0.02",
            "--h3_dop_trigger",
            "sks",
            "--h3_dop_class_prompt",
            "woman",
            "--h3_dop_caption_mode",
            "bare",
        ]
    )
    with pytest.raises(ValueError, match="takes no --h3_dop_class_prompt"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_dop_rejects_whitespace_only_arguments() -> None:
    args = create_parser().parse_args(
        ["--sdpa", "--h3_dop_loss_weight", "0.02", "--h3_dop_trigger", "   ", "--h3_dop_caption_mode", "bare"]
    )
    with pytest.raises(ValueError, match="requires --h3_dop_trigger"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)
    args = create_parser().parse_args(
        ["--sdpa", "--h3_dop_loss_weight", "0.02", "--h3_dop_trigger", "sks", "--h3_dop_class_prompt", " "]
    )
    with pytest.raises(ValueError, match="h3_dop_class_prompt"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_dop_cache_parser_validates_the_caption_mode(tmp_path) -> None:
    from musubi_tuner import minimax_h3_cache_text_encoder_outputs as cache_script

    parser = cache_script.create_parser()
    base = ["--dataset_config", str(tmp_path / "d.toml"), "--text_encoder", str(tmp_path / "te"), "--task", "t2va"]
    args = parser.parse_args(base + ["--h3_dop_trigger", "sks", "--h3_dop_caption_mode", "bare"])
    assert args.h3_dop_caption_mode == "bare"
    # The mode checks live in main(); a bare mode with a class prompt is refused before anything loads.
    with pytest.raises(SystemExit):
        cache_script.main(base + ["--h3_dop_trigger", "sks", "--h3_dop_class_prompt", "woman", "--h3_dop_caption_mode", "bare"])
    with pytest.raises(SystemExit):
        cache_script.main(base + ["--h3_dop_trigger", "   ", "--h3_dop_caption_mode", "bare"])
