from argparse import Namespace

import pytest

from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.validation import (
    ValidationEventDeduplicator,
    add_validation_args,
    derive_validation_seed,
    should_validate,
    validate_validation_args,
)


def _args(**overrides):
    values = {
        "validation_dataset_config": "val.toml",
        "validate_at_start": False,
        "validate_every_n_steps": None,
        "validate_every_n_epochs": None,
        "validation_timestep_bins": 4,
        "validation_min_timestep": 0,
        "validation_max_timestep": 1000,
        "max_validation_items": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_validation_disabled_has_no_due_events():
    args = _args(validation_dataset_config=None, validate_at_start=True, validate_every_n_steps=1)
    assert not should_validate(args, 0, 0, at_start=True)
    assert not should_validate(args, 1, None)


def test_start_and_step_gating():
    args = _args(validate_at_start=True, validate_every_n_steps=3)
    assert should_validate(args, 0, 0, at_start=True)
    assert not should_validate(args, 0, None)
    assert not should_validate(args, 2, None)
    assert should_validate(args, 3, None)
    assert not should_validate(args, 3, 1)


def test_epoch_gate_overrides_steps_and_prevents_double_fire():
    args = _args(validate_every_n_steps=3, validate_every_n_epochs=2)
    assert not should_validate(args, 3, None)
    assert not should_validate(args, 3, 1)
    assert should_validate(args, 3, 2)


def test_validation_event_deduplication_preserves_distinct_epochs():
    events = ValidationEventDeduplicator()
    assert events.claim(0, 0, at_start=True)
    assert not events.claim(0, 0, at_start=True)
    assert events.claim(4, None)
    assert not events.claim(4, None)
    assert events.claim(4, 1)
    assert not events.claim(4, 1)
    assert events.claim(4, 2)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"validation_dataset_config": None, "validate_at_start": True}, "require --validation_dataset_config"),
        ({"validate_every_n_steps": None}, "requires a validation schedule"),
        ({"validate_every_n_steps": 0}, "steps must be at least 1"),
        ({"validate_every_n_epochs": 0}, "epochs must be at least 1"),
        ({"validation_timestep_bins": 0, "validate_at_start": True}, "bins must be at least 1"),
        ({"validation_min_timestep": 500, "validation_max_timestep": 500, "validate_at_start": True}, "bounds"),
        ({"validation_min_timestep": -1, "validate_at_start": True}, "bounds"),
        ({"validation_max_timestep": 1001, "validate_at_start": True}, "bounds"),
        ({"max_validation_items": 0, "validate_at_start": True}, "items must be at least 1"),
    ],
)
def test_invalid_validation_arguments(overrides, match):
    args = _args(**overrides)
    with pytest.raises(ValueError, match=match):
        validate_validation_args(args, supported=True)


def test_unsupported_trainer_rejects_requested_validation():
    with pytest.raises(ValueError, match="does not support"):
        validate_validation_args(_args(validate_at_start=True), supported=False)


def test_unsupported_trainer_rejects_validation_tuning_without_a_schedule():
    with pytest.raises(ValueError, match="does not support"):
        validate_validation_args(_args(validation_dataset_config=None, validation_timestep_bins=8), supported=False)


def test_unsupported_trainer_accepts_disabled_defaults():
    validate_validation_args(_args(validation_dataset_config=None), supported=False)


@pytest.mark.parametrize(
    "tuning",
    [
        {"validation_seed": 123},
        {"validation_timestep_bins": 8},
        {"validation_min_timestep": 100},
        {"validation_max_timestep": 900},
        {"max_validation_items": 2},
    ],
)
def test_supported_trainer_rejects_tuning_without_dataset(tuning):
    with pytest.raises(ValueError, match="require --validation_dataset_config"):
        validate_validation_args(_args(validation_dataset_config=None, **tuning), supported=True)


def test_common_parser_does_not_duplicate_validation_options():
    parser = setup_parser_common()
    add_validation_args(parser)
    assert list(parser._option_string_actions).count("--validation_dataset_config") == 1


def test_validation_seed_is_stable_and_item_specific():
    seed = derive_validation_seed(123, item_key="item-a", bin_index=2)
    assert seed == 428603486652777968
    assert seed == derive_validation_seed(123, item_key="item-a", bin_index=2)
    assert seed != derive_validation_seed(123, item_key="item-b", bin_index=2)
    assert seed != derive_validation_seed(123, item_key="item-a", bin_index=3)
    assert seed != derive_validation_seed(123, item_key="item-a", bin_index=2, stream="audio")
    assert seed != derive_validation_seed(123, dataset_index=0, bin_index=2)


def test_validation_seed_requires_exactly_one_valid_identity():
    with pytest.raises(ValueError, match="exactly one"):
        derive_validation_seed(1, bin_index=0)
    with pytest.raises(ValueError, match="exactly one"):
        derive_validation_seed(1, item_key="item", dataset_index=0, bin_index=0)
    with pytest.raises(ValueError, match="must not be empty"):
        derive_validation_seed(1, item_key="", bin_index=0)
    with pytest.raises(ValueError, match="non-negative"):
        derive_validation_seed(1, dataset_index=-1, bin_index=0)
    with pytest.raises(ValueError, match="non-negative"):
        derive_validation_seed(1, dataset_index=0, bin_index=-1)
    with pytest.raises(ValueError, match="must not be empty"):
        derive_validation_seed(1, dataset_index=0, bin_index=0, stream="")
