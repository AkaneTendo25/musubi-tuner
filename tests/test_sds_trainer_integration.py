"""Run the production trainer's SDS routing without loading video-model dependencies."""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).parents[1] / "src/musubi_tuner/training/trainer_base.py"


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _method(name, namespace=None):
    method = copy.deepcopy(next(node for node in ast.walk(_tree()) if isinstance(node, ast.FunctionDef) and node.name == name))
    method.decorator_list = []
    namespace = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "option,value",
    [
        ("network_weights", "export.safetensors"),
        ("dim_from_weights", True),
        ("scale_weight_norms", 1.0),
        ("fused_backward_pass", True),
        ("h3_adapter_ema_decay", 0.9),
        ("h3_lora_fused_bf16", True),
        ("h3_convrot_int8_lora_fused", True),
    ],
)
def test_sds_rejects_incompatible_trainer_routes(option, value):
    validate = _method("_validate_sds_training_options")
    with pytest.raises(ValueError, match=option):
        validate(SimpleNamespace(**{option: value}))


def test_sds_allows_ordinary_training_options():
    _method("_validate_sds_training_options")(SimpleNamespace(h3_adapter_ema_decay=0, network_weights=None))


@pytest.mark.parametrize("sds_value", ["yes", "tru", "maybe"])
def test_custom_network_cannot_silently_ignore_malformed_sds(sds_value):
    import os

    calls = []
    module = SimpleNamespace(create_arch_network=lambda *args, **kwargs: calls.append(kwargs))
    method = _method(
        "_build_network",
        {
            "__file__": str(SOURCE),
            "sys": SimpleNamespace(path=[]),
            "os": os,
            "importlib": SimpleNamespace(import_module=lambda _: module),
        },
    )
    args = SimpleNamespace(network_module="custom", base_weights=None, network_args=[f"sds={sds_value}"])
    with pytest.raises(ValueError, match="sds"):
        method(SimpleNamespace(), args, SimpleNamespace(print=lambda *_: None), None, None, torch.float32)
    assert calls == []


@pytest.mark.parametrize("sync,skipped,expected", [(False, False, 0), (True, True, 0), (True, False, 1)])
def test_basis_callback_runs_only_after_successful_accumulated_update(sync, skipped, expected):
    node = next(
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Name) and child.id == "sds_optimizer_step" for child in ast.walk(node.test))
    )
    calls = []
    optimizer = object()
    namespace = {
        "sds_optimizer_step": lambda value: calls.append(value),
        "accelerator": SimpleNamespace(sync_gradients=sync, optimizer_step_was_skipped=skipped),
        "optimizer": optimizer,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), namespace)
    assert calls == [optimizer] * expected


def test_plain_lora_never_runs_sds_callback():
    node = next(
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Name) and child.id == "sds_optimizer_step" for child in ast.walk(node.test))
    )
    namespace = {"sds_optimizer_step": None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), namespace)


def test_sds_budget_uses_final_epoch_override_before_scheduler():
    from torch.utils.data import DataLoader

    events = []

    class Dataset:
        def __len__(self):
            return 5

        def __getitem__(self, index):
            return index

        def set_max_train_steps(self, steps):
            events.append(("dataset", steps))

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter])
    network = SimpleNamespace(
        prepare_optimizer_params=lambda **_: ([parameter], ["unet"]),
        configure_sds_training=lambda total: events.append(("sds", total)),
    )
    trainer = SimpleNamespace(
        extra_trainable_params=lambda *values: values[-1],
        get_optimizer=lambda *_: ("AdamW", "", optimizer, lambda: None, lambda: None),
        get_lr_scheduler=lambda args, *_: events.append(("scheduler", args.max_train_steps)),
    )
    args = SimpleNamespace(
        learning_rate=0.01,
        max_data_loader_n_workers=0,
        seed=42,
        persistent_data_loader_workers=False,
        max_train_epochs=3,
        max_train_steps=999,
        gradient_accumulation_steps=2,
    )
    method = _method(
        "_build_optimizer_and_dataloader",
        {
            "torch": SimpleNamespace(Generator=torch.Generator, utils=SimpleNamespace(data=SimpleNamespace(DataLoader=DataLoader))),
            "os": SimpleNamespace(cpu_count=lambda: 4),
            "math": math,
            "dataloader_extra_kwargs": lambda *_: {},
        },
    )
    accelerator = SimpleNamespace(print=lambda *_: None, num_processes=1)
    method(trainer, args, accelerator, network, Dataset(), None, None)
    assert args.max_train_steps == 9
    assert events == [("sds", 9), ("dataset", 9), ("scheduler", 9)]


@pytest.mark.parametrize("sds_enabled", [True, False])
def test_save_metadata_describes_export_and_preserves_training_arguments(sds_enabled):
    import json

    save = next(node for node in ast.walk(_tree()) if isinstance(node, ast.FunctionDef) and node.name == "save_model")
    start = next(
        index
        for index, node in enumerate(save.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "metadata_to_save" for target in node.targets)
    )
    original = {"ss_network_dim": "16", "ss_network_args": '{"sds":"True"}'}
    network_args = {"sds": "True", "sds_warmup_steps": "10", "sds_update_phases": "5", "exclude_patterns": "[]"}
    network = SimpleNamespace(
        sds_enabled=sds_enabled,
        sds_export_metadata=lambda: {"ss_network_dim": "32", "ss_network_alpha": "8", "ss_sds_training_rank": "16"},
    )
    namespace = {
        "args": SimpleNamespace(no_metadata=False),
        "metadata": original,
        "unwrapped_nw": network,
        "net_kwargs": network_args,
        "SS_METADATA_KEY_NETWORK_ARGS": "ss_network_args",
        "json": json,
    }
    statements = save.body[start : start + 3]
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), str(SOURCE), "exec"), namespace)
    saved = namespace["metadata_to_save"]
    assert original == {"ss_network_dim": "16", "ss_network_args": '{"sds":"True"}'}
    assert network_args["sds"] == "True"
    if sds_enabled:
        assert saved["ss_network_dim"] == "32"
        assert json.loads(saved["ss_network_args"]) == {"exclude_patterns": "[]"}
    else:
        assert saved == original
