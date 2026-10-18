from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.reamplify import (
    REAMPLIFY_AUTO,
    parse_reamplify_schedule,
    reamplify_target,
    resolve_reamplify_schedule,
)
from musubi_tuner.minimax_h3.training import H3ModelPrediction
from musubi_tuner.minimax_h3.validation import validation_sigma_bins
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
    def reduce(value, reduction):
        assert reduction == "sum"
        return value

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


class _ReducedHooks(_Hooks):
    @staticmethod
    def reduce(value, reduction):
        assert reduction == "sum"
        # Simulate a second rank contributing one ratio in each bin.
        return value + torch.tensor([0.25, 1.0, 0.75, 1.0], dtype=value.dtype)


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
        h3_overlay_weights_multiplier=1.0,
        h3_overlay_training_only=False,
        h3_validate_without_overlay=False,
        base_weights=None,
        h3_validation_field_probe=False,
        h3_validate_ema=False,
        validation_timestep_bins=2,
        validation_min_timestep=0,
        validation_max_timestep=1000,
        validation_dataset_config=None,
        validation_seed=17,
        max_validation_items=4,
        seed=11,
        h3_training_mode="t2va",
        h3_observed_modality=None,
        h3_video_loss_weight=1.0,
        h3_audio_loss_weight=1.0,
        h3_image_timestep_sampling="auto",
        h3_image_flow_shift=None,
        timestep_sampling="uniform",
        h3_shift_video=12.0,
        h3_shift_audio=3.0,
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
    assert parse_reamplify_schedule("auto") == REAMPLIFY_AUTO
    assert resolve_reamplify_schedule({0: 0.5, 1: 2.0}, 2) == (2.0, 1.01)
    with pytest.raises(ValueError, match="every contiguous"):
        resolve_reamplify_schedule({0: 0.5}, 2)
    with pytest.raises(ValueError, match="every contiguous"):
        resolve_reamplify_schedule({False: 0.5}, 1)
    with pytest.raises(ValueError, match="finite in FP32"):
        resolve_reamplify_schedule({0: 1e-320}, 1)
    with pytest.raises(ValueError, match="greater than 1"):
        parse_reamplify_schedule("1,2")
    with pytest.raises(ValueError, match="finite in FP32"):
        parse_reamplify_schedule("1e100")

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
        ({"h3_reamplify_after": 0}, "requires --h3_overlay_weights"),
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

    for update, message in (
        ({}, "field_probe"),
        ({"h3_validation_field_probe": True, "validation_min_timestep": 1}, "full validation"),
        ({"h3_validation_field_probe": True, "h3_validate_ema": True}, "live adapter"),
    ):
        with pytest.raises(ValueError, match=message):
            MiniMaxH3NetworkTrainer()._validate_reamplify_args(_args(h3_reamplify_scale="auto", **update))


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


def test_external_after_zero_uses_v3_state_without_owned_snapshot(tmp_path: Path):
    overlay = tmp_path / "collapsed.safetensors"
    save_file({"lora_unet_block.lora_down.weight": torch.ones(1, 3)}, str(overlay))
    args = _args(h3_reamplify_after=0, h3_overlay_weights=str(overlay), h3_overlay_weights_multiplier=0.5)
    trainer = _trainer(args)
    hooks = _Hooks()
    network = _Network()
    trainer._take_reamplify_snapshot(args, hooks, object(), network, 0)
    assert trainer._reamplify_snapshot_step == 0
    assert trainer._reamplify_snapshot_network is None
    trainer._reamplify_probability_generator = torch.Generator().manual_seed(3)
    trainer._reamplify_last_global_step = 0
    trainer._register_reamplify_state_hooks(args, hooks, object(), network)
    hooks.save([], [], tmp_path)
    state = json.loads((tmp_path / "h3_reamplify_state-rank0.json").read_text(encoding="utf-8"))
    assert state["version"] == 3
    assert state["snapshot"] is None
    assert state["identity"]["composition_mode"] == "external"

    restored = _trainer(args)
    load_hooks = _Hooks()
    restored._register_reamplify_state_hooks(args, load_hooks, object(), network)
    load_hooks.load([], tmp_path)
    assert restored._reamplify_snapshot_step == 0


def test_external_overlay_mutation_is_rejected_before_resolution(tmp_path: Path):
    overlay = tmp_path / "collapsed.safetensors"
    save_file({"lora_unet_block.lora_down.weight": torch.ones(1, 3)}, str(overlay))
    args = _args(h3_reamplify_after=0, h3_overlay_weights=str(overlay))
    trainer = _trainer(args)
    save_file({"lora_unet_block.lora_down.weight": torch.zeros(1, 3)}, str(overlay))
    with pytest.raises(ValueError, match="changed after configuration"):
        trainer._resolve_reamplify_schedule(args)


def test_reamp_overlay_installs_converted_keys_from_the_bound_payload(tmp_path: Path):
    overlay = tmp_path / "diffusers.safetensors"
    save_file({"transformer.foo.lora_A.weight": torch.ones(1, 3)}, str(overlay))
    args = _args(h3_reamplify_after=0, h3_overlay_weights=str(overlay))
    trainer = _trainer(args)
    captured = {}

    def frozen(self, weights, multiplier, _accelerator, _transformer, _label, source):
        captured.update(weights=weights, multiplier=multiplier, source=source)
        return _Teacher()

    trainer._frozen_lora_from_weights = MethodType(frozen, trainer)
    trainer.install_overlay_weights(args, _Hooks(), object())
    assert set(captured["weights"]) == {"lora_unet_foo.lora_down.weight", "lora_unet_foo.alpha"}
    assert captured["source"] == str(overlay.resolve())


def test_overlay_state_context_restores_mixed_modules_after_error():
    class Module:
        def __init__(self, enabled):
            self.enabled = enabled

    class Overlay:
        text_encoder_loras = ()

        def __init__(self):
            self.unet_loras = (Module(True), Module(False))

        def set_enabled(self, enabled):
            for module in self.unet_loras:
                module.enabled = enabled

    trainer = MiniMaxH3NetworkTrainer()
    trainer._overlay_network = Overlay()
    with pytest.raises(RuntimeError):
        with trainer._overlay_enabled(True):
            assert [module.enabled for module in trainer._overlay_network.unet_loras] == [True, True]
            raise RuntimeError("probe")
    assert [module.enabled for module in trainer._overlay_network.unet_loras] == [True, False]


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


def test_auto_measurement_resolves_once_and_round_trips_provenance(tmp_path: Path):
    args = _args(h3_reamplify_scale="auto", h3_validation_field_probe=True)
    trainer = _trainer(args)
    hooks = _Hooks()
    network = _Network()
    bins = validation_sigma_bins(2)
    trainer._field_ratios = {0: [0.5, 0.5], 1: [0.25]}
    trainer._record_reamplify_field_measurement(args, hooks, network, bins, 1)
    assert trainer._reamplify_last_field_ratios == {0: 0.5, 1: 0.25}
    assert trainer._resolve_reamplify_schedule(args) == (2.0, 4.0)
    trainer._take_reamplify_snapshot(args, hooks, object(), network, 2)
    trainer._reamplify_probability_generator = torch.Generator().manual_seed(7)
    trainer._reamplify_last_global_step = 2
    trainer._register_reamplify_state_hooks(args, hooks, object(), network)
    hooks.save([], [], tmp_path)

    restored = _trainer(args)
    load_hooks = _Hooks()
    restored._register_reamplify_state_hooks(args, load_hooks, object(), network)
    load_hooks.load([], tmp_path)
    assert restored._reamplify_resolved_schedule == (2.0, 4.0)
    assert restored._reamplify_last_field_measurement["step"] == 1
    # A later validation cannot move a schedule that the snapshot already bound.
    restored._field_ratios = {0: [0.1], 1: [0.1]}
    restored._record_reamplify_field_measurement(args, load_hooks, network, bins, 2)
    assert restored._reamplify_resolved_schedule == (2.0, 4.0)


def test_auto_measurement_uses_global_bin_sums_and_counts():
    args = _args(h3_reamplify_scale="auto", h3_validation_field_probe=True)
    trainer = _trainer(args)
    trainer._field_ratios = {0: [0.75], 1: [0.25]}
    trainer._record_reamplify_field_measurement(args, _ReducedHooks(), _Network(), validation_sigma_bins(2), 1)
    assert trainer._reamplify_last_field_ratios == {0: 0.5, 1: 0.5}
    assert trainer._resolve_reamplify_schedule(args) == (2.0, 2.0)


def test_auto_runtime_rechecks_measurement_identity_before_snapshot():
    args = _args(h3_reamplify_scale="auto", h3_validation_field_probe=True)
    trainer = _trainer(args)
    hooks = _Hooks()
    network = _Network()
    trainer._field_ratios = {0: [0.5], 1: [0.25]}
    trainer._record_reamplify_field_measurement(args, hooks, network, validation_sigma_bins(2), 1)
    trainer._reamplify_last_field_measurement["bins"][0]["base_sigma"] = 0.123
    with pytest.raises(ValueError, match="no longer matches"):
        trainer._resolve_reamplify_schedule(args)
    assert trainer._reamplify_resolved_schedule is None

    trainer._reamplify_last_field_measurement["bins"][0]["base_sigma"] = validation_sigma_bins(2)[0].base_sigma
    trainer._reamplify_last_field_measurement["step"] = 2
    trainer._reamplify_last_field_measurement["adapter_sha256"] = "0" * 64
    trainer._resolve_reamplify_schedule(args)
    with pytest.raises(ValueError, match="adapter being frozen"):
        trainer._take_reamplify_snapshot(args, hooks, object(), network, 2)
    assert trainer._reamplify_snapshot_network is None


@pytest.mark.parametrize(
    ("setting", "changed"),
    [
        ("h3_observed_modality", "video"),
        ("h3_audio_loss_weight", 0.0),
        ("h3_image_timestep_sampling", "uniform"),
        ("h3_image_flow_shift", 1.5),
    ],
)
def test_auto_measurement_rejects_changed_validation_task_identity(setting: str, changed):
    args = _args(h3_reamplify_scale="auto", h3_validation_field_probe=True)
    trainer = _trainer(args)
    trainer._field_ratios = {0: [0.5], 1: [0.25]}
    trainer._record_reamplify_field_measurement(args, _Hooks(), _Network(), validation_sigma_bins(2), 1)
    setattr(args, setting, changed)
    with pytest.raises(ValueError, match="no longer matches"):
        trainer._resolve_reamplify_schedule(args)


def test_explicit_v1_state_remains_loadable(tmp_path: Path):
    args = _args()
    hooks = _Hooks()
    network = _Network()
    trainer = _trainer(args)
    trainer._take_reamplify_snapshot(args, hooks, object(), network, 2)
    trainer._reamplify_probability_generator = torch.Generator().manual_seed(4)
    trainer._reamplify_last_global_step = 2
    trainer._register_reamplify_state_hooks(args, hooks, object(), network)
    hooks.save([], [], tmp_path)
    state_path = tmp_path / "h3_reamplify_state-rank0.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["version"] = 1
    state["snapshot"].pop("adapter_sha256")
    for key in ("field_ratios", "field_measurement", "resolved_schedule"):
        state.pop(key)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    restored = _trainer(args)
    load_hooks = _Hooks()
    restored._register_reamplify_state_hooks(args, load_hooks, object(), network)
    load_hooks.load([], tmp_path)
    assert restored._reamplify_snapshot_step == 2
    assert restored._reamplify_resolved_schedule == (2.0, 3.0)


@pytest.mark.parametrize(
    "mutation",
    ["missing_bin", "future_measurement", "wrong_boundary_adapter", "bool_factor", "bin_geometry"],
)
def test_auto_state_rejects_malformed_measurement_before_install(tmp_path: Path, mutation: str):
    args = _args(h3_reamplify_scale="auto", h3_validation_field_probe=True)
    hooks = _Hooks()
    network = _Network()
    trainer = _trainer(args)
    bins = validation_sigma_bins(2)
    trainer._field_ratios = {0: [0.5], 1: [0.25]}
    trainer._record_reamplify_field_measurement(args, hooks, network, bins, 1)
    trainer._resolve_reamplify_schedule(args)
    trainer._take_reamplify_snapshot(args, hooks, object(), network, 2)
    trainer._reamplify_probability_generator = torch.Generator().manual_seed(5)
    trainer._reamplify_last_global_step = 2
    trainer._register_reamplify_state_hooks(args, hooks, object(), network)
    hooks.save([], [], tmp_path)
    state_path = tmp_path / "h3_reamplify_state-rank0.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if mutation == "missing_bin":
        state["field_ratios"].pop("1")
    elif mutation == "future_measurement":
        state["field_measurement"]["step"] = state["completed_steps"] + 1
    elif mutation == "wrong_boundary_adapter":
        state["field_measurement"]["step"] = state["snapshot_step"]
        state["field_measurement"]["adapter_sha256"] = "0" * 64
    elif mutation == "bool_factor":
        state["resolved_schedule"][0] = True
    else:
        state["field_measurement"]["bins"][0]["base_sigma"] = 0.125
    state_path.write_text(json.dumps(state), encoding="utf-8")

    restored = _trainer(args)
    load_hooks = _Hooks()
    restored._register_reamplify_state_hooks(args, load_hooks, object(), network)
    with pytest.raises(ValueError):
        load_hooks.load([], tmp_path)
    assert restored._reamplify_snapshot_network is None
    assert restored._reamplify_probability_generator is None


def test_auto_resume_rejects_changed_validation_task_identity(tmp_path: Path):
    args = _args(h3_reamplify_scale="auto", h3_validation_field_probe=True)
    hooks = _Hooks()
    network = _Network()
    trainer = _trainer(args)
    trainer._field_ratios = {0: [0.5], 1: [0.25]}
    trainer._record_reamplify_field_measurement(args, hooks, network, validation_sigma_bins(2), 1)
    trainer._resolve_reamplify_schedule(args)
    trainer._take_reamplify_snapshot(args, hooks, object(), network, 2)
    trainer._reamplify_probability_generator = torch.Generator().manual_seed(5)
    trainer._reamplify_last_global_step = 2
    trainer._register_reamplify_state_hooks(args, hooks, object(), network)
    hooks.save([], [], tmp_path)

    changed = _args(
        h3_reamplify_scale="auto",
        h3_validation_field_probe=True,
        h3_observed_modality="video",
    )
    restored = _trainer(changed)
    load_hooks = _Hooks()
    restored._register_reamplify_state_hooks(changed, load_hooks, object(), network)
    with pytest.raises(ValueError, match="field-measurement identity"):
        load_hooks.load([], tmp_path)
    assert restored._reamplify_snapshot_network is None
