from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.reamplify import parse_reamplify_schedule, reamplify_target
from musubi_tuner.minimax_h3.training import H3ModelPrediction
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer


class _Hooks:
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")

    def __init__(self):
        self.save = None
        self.load = None

    def register_save_state_pre_hook(self, hook):
        self.save = hook

    def register_load_state_pre_hook(self, hook):
        self.load = hook

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def print(*_args, **_kwargs):
        pass


class _Network:
    def __init__(self):
        self.weights = {
            "lora_unet_block.lora_down.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "lora_unet_block.lora_up.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "lora_unet_block.alpha": torch.tensor(2.0),
        }

    def snapshot_weights(self, dtype):
        return {key: value.detach().to(dtype=dtype).clone() for key, value in self.weights.items()}


class _Teacher:
    def __init__(self):
        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)


def _args(**updates):
    values = dict(
        h3_reamplify_scale="2,3",
        h3_reamplify_after=2,
        h3_reamplify_probability=0.5,
        network_module="networks.lora_minimax_h3",
        network_args=None,
        network_dim=2,
        network_alpha=2.0,
        dit="models/h3.safetensors",
        h3_projection_scale=None,
        h3_repair_reference=None,
        h3_guidance_distillation_scale=None,
        h3_rollout_supervision=False,
        h3_teacher_matching=False,
        h3_two_teacher_loss_weight=0.0,
        h3_sparse_teacher_only=False,
        h3_caption_dropout_rate=0.0,
        h3_overlay_weights=None,
        base_weights=None,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def _trainer(args=None):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._validate_reamplify_args(args or _args())

    def frozen(self, _weights, _multiplier, _accelerator, _transformer, _label, _source):
        # Consume the global stream like a real LoRA constructor can; the caller
        # must isolate this from training RNG.
        torch.rand(())
        return _Teacher()

    trainer._frozen_lora_from_weights = MethodType(frozen, trainer)
    return trainer


def test_schedule_and_fp32_binned_target():
    assert parse_reamplify_schedule("2, 3") == (2.0, 3.0)
    with pytest.raises(ValueError, match="validation-ratio resolver"):
        parse_reamplify_schedule("auto")
    with pytest.raises(ValueError, match="greater than 1"):
        parse_reamplify_schedule("1,2")

    # BF16 rounds 257 to 256. Subtracting before widening would erase the field.
    stock = torch.tensor([[[[0.0]]], [[[0.0]]]], dtype=torch.bfloat16)
    prompt = torch.tensor([[[[257.0]]], [[[4.0]]]], dtype=torch.float32)
    empty = torch.tensor([[[[256.0]]], [[[1.0]]]], dtype=torch.float32)
    built = reamplify_target(
        H3ModelPrediction(stock, None),
        H3ModelPrediction(prompt, None),
        H3ModelPrediction(empty, None),
        (2.0, 3.0),
        torch.tensor([0.0, 1.0]),
    )
    assert built.video.float().flatten().tolist() == [2.0, 9.0]
    assert built.video.dtype == torch.bfloat16


def test_validation_is_default_off_and_rejects_unsupported_combinations():
    trainer = MiniMaxH3NetworkTrainer()
    trainer._validate_reamplify_args(_args(h3_reamplify_scale=None, h3_reamplify_after=0, h3_reamplify_probability=1.0))
    assert trainer._reamplify_schedule is None
    for update, message in (
        ({"h3_reamplify_after": 0}, "greater than 0"),
        ({"h3_reamplify_probability": 0.0}, "lie in"),
        ({"resume_as_new_shard": True}, "resume_as_new_shard"),
        ({"h3_projection_scale": 2.0}, "cannot be combined"),
        ({"h3_guidance_scale_range": "2,4"}, "cannot be combined"),
        ({"base_weights": ["merged.safetensors"]}, "cannot be combined"),
        ({"network_module": "custom.network"}, "stock"),
        ({"network_args": ["sds=True"]}, "SDS-LoRA"),
        ({"network_args": ["sds=TRUE"]}, "SDS-LoRA"),
        ({"network_args": ["sds=1"]}, "SDS-LoRA"),
    ):
        with pytest.raises(ValueError, match=message):
            MiniMaxH3NetworkTrainer()._validate_reamplify_args(_args(**update))


def test_snapshot_is_taken_at_boundary_before_draw_and_constructor_preserves_rng():
    args = _args()
    trainer = _trainer(args)
    hooks = _Hooks()
    network = _Network()
    before = torch.get_rng_state().clone()
    trainer._take_reamplify_snapshot(args, hooks, object(), network, 2)
    assert torch.equal(before, torch.get_rng_state())
    assert trainer._reamplify_snapshot_step == 2
    assert trainer._reamplify_snapshot_network.enabled is False
    network.weights["lora_unet_block.lora_up.weight"].add_(100)
    assert float(trainer._reamplify_snapshot_weights["lora_unet_block.lora_up.weight"].max()) == 7.0


def test_state_round_trip_restores_snapshot_and_probability_stream(tmp_path: Path):
    args = _args()
    hooks = _Hooks()
    network = _Network()
    trainer = _trainer(args)
    trainer._take_reamplify_snapshot(args, hooks, object(), network, 2)
    trainer._reamplify_probability_generator = torch.Generator().manual_seed(91)
    trainer._reamplify_last_global_step = 3
    expected_next = torch.rand((), generator=trainer._reamplify_probability_generator)
    trainer._reamplify_probability_generator.manual_seed(91)
    trainer._register_reamplify_state_hooks(args, hooks, object(), network)
    hooks.save([], [], tmp_path)

    restored = _trainer(args)
    load_hooks = _Hooks()
    restored._register_reamplify_state_hooks(args, load_hooks, object(), network)
    torch.manual_seed(999)
    load_hooks.load([], tmp_path)
    assert restored._reamplify_state_loaded
    assert restored._reamplify_snapshot_step == 2
    assert restored._reamplify_snapshot_network.enabled is False
    assert torch.equal(torch.rand((), generator=restored._reamplify_probability_generator), expected_next)


@pytest.mark.parametrize("mutation", ["nested_bool", "missing_rng", "step_relation", "snapshot_bytes"])
def test_state_rejects_corruption_before_restoring_runtime(tmp_path: Path, mutation: str):
    args = _args()
    hooks = _Hooks()
    network = _Network()
    trainer = _trainer(args)
    trainer._take_reamplify_snapshot(args, hooks, object(), network, 2)
    trainer._reamplify_probability_generator = torch.Generator().manual_seed(7)
    trainer._reamplify_last_global_step = 2
    trainer._register_reamplify_state_hooks(args, hooks, object(), network)
    hooks.save([], [], tmp_path)
    state_path = tmp_path / "h3_reamplify_state-rank0.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if mutation == "nested_bool":
        state["identity"]["after"] = True
    elif mutation == "missing_rng":
        state["generator"] = None
    elif mutation == "step_relation":
        state["completed_steps"] = 2
    else:
        snapshot = tmp_path / "h3_reamplify_snapshot-rank0.safetensors"
        snapshot.write_bytes(snapshot.read_bytes() + b"changed")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    restored = _trainer(args)
    load_hooks = _Hooks()
    restored._register_reamplify_state_hooks(args, load_hooks, object(), network)
    with pytest.raises((ValueError, RuntimeError)):
        load_hooks.load([], tmp_path)
    assert restored._reamplify_snapshot_network is None
    assert restored._reamplify_probability_generator is None


def test_preboundary_state_can_resume_at_exact_boundary(tmp_path: Path):
    args = _args()
    trainer = _trainer(args)
    trainer._reamplify_last_global_step = 1  # two completed warm-up steps
    hooks = _Hooks()
    network = _Network()
    trainer._register_reamplify_state_hooks(args, hooks, object(), network)
    hooks.save([], [], tmp_path)

    restored = _trainer(args)
    load_hooks = _Hooks()
    restored._register_reamplify_state_hooks(args, load_hooks, object(), network)
    load_hooks.load([], tmp_path)
    assert restored._reamplify_snapshot_step is None
    restored._take_reamplify_snapshot(args, load_hooks, object(), network, 2)
    assert restored._reamplify_snapshot_step == 2
