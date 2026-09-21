"""Required output paths fail before any trainer begins loading data or models."""

from argparse import Namespace

import pytest

from musubi_tuner.dataset import config_utils
from musubi_tuner.hidream_o1_train import HiDreamO1Trainer
from musubi_tuner.hv_train import FineTuningTrainer
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer
from musubi_tuner.qwen_image_train import QwenImageTrainer
from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.zimage_train import ZImageTrainer


TRAINERS = (
    NetworkTrainer,
    MiniMaxH3NetworkTrainer,
    FineTuningTrainer,
    HiDreamO1Trainer,
    QwenImageTrainer,
    ZImageTrainer,
)


def _args(**overrides) -> Namespace:
    values = dict(
        dataset_config="unused-dataset.toml",
        dit="unused-model.safetensors",
        output_dir="unused-output",
        output_name="test-run",
        cuda_allow_tf32=False,
        cuda_cudnn_benchmark=False,
        fp8_base=False,
        fp8_scaled=False,
        seed=1,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("trainer_class", TRAINERS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("missing", ("output_dir", "output_name"))
def test_missing_output_argument_fails_before_dataset_loading(monkeypatch, trainer_class, missing) -> None:
    def unexpected_dataset_load(*_args, **_kwargs):
        pytest.fail("dataset loading began before output argument validation")

    monkeypatch.setattr(config_utils, "load_user_config", unexpected_dataset_load)
    trainer = trainer_class()
    if isinstance(trainer, NetworkTrainer):
        monkeypatch.setattr(trainer, "_build_dataset", unexpected_dataset_load)

    with pytest.raises(ValueError, match=rf"^{missing} is required"):
        trainer.train(_args(**{missing: None}))


def test_valid_output_arguments_advance_common_trainer_to_next_stage(monkeypatch) -> None:
    trainer = NetworkTrainer()

    class ReachedSession(Exception):
        pass

    def reach_session(_args):
        raise ReachedSession

    monkeypatch.setattr(trainer, "handle_model_specific_args", lambda _args: None)
    monkeypatch.setattr(trainer, "_init_session", reach_session)
    args = _args(
        sage_attn=False,
        disable_numpy_memmap=False,
        show_timesteps=False,
    )

    with pytest.raises(ReachedSession):
        trainer.train(args)
