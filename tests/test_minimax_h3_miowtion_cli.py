import pytest

from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser


def _validate(*options: str):
    args = create_parser().parse_args(["--sdpa", *options])
    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)
    return args


def test_miowtion_predictor_is_opt_in():
    args = _validate()

    assert args.h3_miowtion_predictor is None


def test_miowtion_predictor_accepts_fixed_sparse_fraction():
    args = _validate(
        "--h3_block_sparse_kv_fraction",
        "0.1",
        "--h3_miowtion_predictor",
        "predictor.safetensors",
    )

    assert args.h3_miowtion_predictor == "predictor.safetensors"
    assert args.h3_block_sparse_kv_fraction == 0.1


def test_miowtion_predictor_requires_positive_sparse_fraction():
    with pytest.raises(ValueError, match="requires --h3_block_sparse_kv_fraction above 0"):
        _validate("--h3_miowtion_predictor", "predictor.safetensors")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--h3_block_sparse_threshold", "0.8"), "cannot be combined with --h3_block_sparse_threshold"),
        (("--h3_block_sparse_block_shape", "1,8,16"), "cannot be combined with --h3_block_sparse_block_shape"),
    ],
)
def test_miowtion_predictor_rejects_other_selection_rules(options, message):
    with pytest.raises(ValueError, match=message):
        _validate(
            "--h3_block_sparse_kv_fraction",
            "0.1",
            *options,
            "--h3_miowtion_predictor",
            "predictor.safetensors",
        )
