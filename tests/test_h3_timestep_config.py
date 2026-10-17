import pytest

from musubi_tuner.minimax_h3_train_network import create_parser, read_config_from_file


@pytest.mark.parametrize(
    "options,explicit",
    [
        ([], False),
        (["--timestep_sampling", "uniform"], True),
        (["--timestep_sampling=uniform"], True),
        (["--timestep_sam", "uniform"], True),
        (["--seed", "123"], False),
    ],
)
def test_config_detects_explicit_sampler_with_native_argparse(tmp_path, options, explicit):
    config = tmp_path / "train.toml"
    config.write_text("h3_training_mode = 'fl2va'\n", encoding="utf-8")
    argv = ["--config_file", str(config), "--sdpa", *options]
    parser = create_parser()
    args = read_config_from_file(parser.parse_args(argv), parser, argv)
    assert args._h3_timestep_sampling_explicit is explicit
    if explicit:
        assert args.timestep_sampling == "uniform"
