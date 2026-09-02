"""Rank-reduced AdaLN in MiniMax H3 full fine-tuning.

The dense trainer can train the ``[out, rank]`` AdaLN projections the loader
produces under ``--h3_adaln_rank`` instead of the full ``[out, 2688]`` weights.
Because the AdaLN input ``silu(e(t))`` lies in the span of the fitted basis, a
plain SGD step on the reduced weight moves the block output as the same step on
the full weight would (to the accuracy of that basis, shown here on a tiny model), and the checkpoint is written in the pruned
layout the loader already reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import musubi_tuner.minimax_h3.model_loader as h3_model_loader
from musubi_tuner.minimax_h3.adaln_lowrank import (
    ADALN_INFIX,
    DEFAULT_TABLE_POINTS,
    TABLE_KEY,
    build_adaln_basis,
    build_timestep_table,
    make_adaln_split_hook,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.model_loader import load_transformer, validate_transformer_checkpoint
from musubi_tuner.minimax_h3_train import MiniMaxH3Trainer, create_parser
from musubi_tuner.modules.custom_offloading_utils import TrainableBlockRingOffloader

RANK = 8

_FP32_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
    "rope.",
    TABLE_KEY,
)


def _tiny_config(*, num_layers: int = 2) -> MiniMaxH3TransformerConfig:
    return MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=num_layers,
        num_refiner_layers=1,
        ffn_dim=64,
        in_channels=4,
        audio_in_channels=8,
        patch_size=(1, 2, 2),
        text_dim=12,
        freq_dim=8,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )


def _tiny_inputs(*, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    # 0.25 and 0.75 land on rows of the 4097-point table, so the pruned model
    # interpolates nothing and any difference comes from the projection alone.
    return {
        "video_hidden_states": torch.randn(1, 2, 16, dtype=dtype),
        "audio_hidden_states": torch.randn(1, 4, 8, dtype=dtype),
        "encoder_hidden_states": torch.randn(1, 3, 12, dtype=dtype),
        "timestep": torch.tensor([0.25, 0.75]),
        "timestep_indices": torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0]),
        "token_tags": torch.tensor([1, 1, 1, 2, 2, 2, 2, 0, 0]),
        "position_ids": torch.arange(27, dtype=torch.float32).reshape(9, 3) / 10,
        "video_indices": torch.tensor([7, 8]),
        "audio_indices": torch.tensor([3, 4, 5, 6]),
        "text_indices": torch.tensor([0, 1, 2]),
    }


def _released_state_dict(model: MiniMaxH3Transformer) -> dict[str, torch.Tensor]:
    """The released mixed-precision layout: BF16 blocks, float32 projections and tables."""
    return {
        name: tensor.detach().to(torch.float32 if name.startswith(_FP32_PREFIXES) else torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
    }


def _reduce(model: MiniMaxH3Transformer, rank: int) -> tuple[dict[str, torch.Tensor], MiniMaxH3TransformerConfig]:
    """Apply the loader's streaming reduction to an in-memory state dict."""
    basis = build_adaln_basis(model.time_embedder, rank, center=False)
    table = build_timestep_table(model.time_embedder, basis, DEFAULT_TABLE_POINTS)
    hook = make_adaln_split_hook(basis, table)
    reduced = {}
    for key, tensor in model.state_dict().items():
        keys, tensors = hook(key, tensor.detach().clone())
        if keys is None:
            reduced[key] = tensor.detach().clone()
        else:
            reduced.update(zip(keys, tensors))
    config = MiniMaxH3TransformerConfig(
        **{**model.config.__dict__, "adaln_t_table_size": DEFAULT_TABLE_POINTS, "time_embed_dim": rank}
    )
    return reduced, config


def _reduced_copy(model: MiniMaxH3Transformer, rank: int) -> MiniMaxH3Transformer:
    state_dict, config = _reduce(model, rank)
    reduced = MiniMaxH3Transformer(config)
    info = reduced.load_state_dict(state_dict, strict=True)
    assert not info.missing_keys and not info.unexpected_keys
    return reduced


def _adaln_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for name, parameter in model.named_parameters() if ADALN_INFIX in name]


def _infer_with_tiny_config(monkeypatch, config: MiniMaxH3TransformerConfig) -> None:
    real = h3_model_loader.infer_transformer_config
    monkeypatch.setattr(h3_model_loader, "infer_transformer_config", lambda path, _config=None: real(path, config))


def test_dense_trainer_accepts_adaln_rank_for_bf16_and_still_rejects_pruned_sources(tmp_path: Path):
    torch.manual_seed(0)
    trainer = MiniMaxH3Trainer()
    model = MiniMaxH3Transformer(_tiny_config(num_layers=1))
    bf16 = tmp_path / "minimax_h3_fl2va_bf16.safetensors"
    save_file(_released_state_dict(model), bf16)
    pruned_state, _ = _reduce(model, RANK)
    pruned = tmp_path / "minimax_h3_fl2va_pruned.safetensors"
    save_file(pruned_state, pruned)

    trainer._validate_full_finetune_args(create_parser().parse_args(["--sdpa", "--dit", str(bf16), "--h3_adaln_rank", "8"]))
    # Without a checkpoint to inspect the flag is accepted; the loader decides.
    trainer._validate_full_finetune_args(create_parser().parse_args(["--sdpa", "--h3_adaln_rank", "8"]))

    with pytest.raises(ValueError, match="already pruned"):
        trainer._validate_full_finetune_args(create_parser().parse_args(["--sdpa", "--dit", str(pruned), "--h3_adaln_rank", "8"]))
    with pytest.raises(ValueError, match="BF16 weights|pre-pruned"):
        trainer._validate_full_finetune_args(
            create_parser().parse_args(["--sdpa", "--int8_convrot_base", "--dit", str(bf16), "--h3_adaln_rank", "8"])
        )
    # A pruned dense checkpoint is continued by omitting the flag.
    trainer._validate_full_finetune_args(create_parser().parse_args(["--sdpa", "--dit", str(pruned)]))


def test_dense_module_trains_reduced_projections_and_keeps_the_table_frozen():
    torch.manual_seed(0)
    reduced = _reduced_copy(MiniMaxH3Transformer(_tiny_config()), RANK)
    args = create_parser().parse_args(["--sdpa", "--h3_adaln_rank", str(RANK)])
    trainer = MiniMaxH3Trainer()

    module = trainer._build_network(args, None, reduced, None, None)
    (group,), _ = module.prepare_optimizer_params(1.0)
    trained = {id(parameter) for parameter in group["params"]}

    assert reduced.time_embedder is None
    assert reduced.adaln_t_table.shape == (DEFAULT_TABLE_POINTS, RANK)
    assert not reduced.adaln_t_table.requires_grad
    assert id(reduced.adaln_t_table) not in trained
    projections = _adaln_parameters(reduced)
    assert len(projections) == 2 * len(reduced.blocks) + 2  # weight and bias per block, plus the final layer
    for parameter in projections:
        assert parameter.requires_grad
        assert parameter.dtype == torch.float32
        assert id(parameter) in trained
    weights = [parameter for parameter in projections if parameter.ndim == 2]
    assert all(weight.shape[1] == RANK for weight in weights)
    assert trainer.extra_metadata(args)["ss_h3_adaln_layout"] == "pruned"
    trainer._adaln_reduced = False
    assert "ss_h3_adaln_layout" not in trainer.extra_metadata(args)


def test_reduced_adaln_sgd_step_moves_the_output_like_the_full_weight_step():
    torch.manual_seed(0)
    full = MiniMaxH3Transformer(_tiny_config(num_layers=2))
    reduced = _reduced_copy(full, RANK)
    inputs = _tiny_inputs()
    video_target = torch.randn(1, 2, 16)
    audio_target = torch.randn(1, 4, 8)

    for model in (full, reduced):
        model.requires_grad_(False)
        for parameter in _adaln_parameters(model):
            parameter.requires_grad_(True)

    with torch.no_grad():
        before_full, before_reduced = full(**inputs), reduced(**inputs)
    torch.testing.assert_close(before_reduced.video, before_full.video, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(before_reduced.audio, before_full.audio, atol=1e-5, rtol=1e-5)

    def sgd_step(model: MiniMaxH3Transformer, lr: float) -> None:
        model.zero_grad(set_to_none=True)
        output = model(**inputs)
        loss = torch.nn.functional.mse_loss(output.video, video_target) + torch.nn.functional.mse_loss(output.audio, audio_target)
        loss.backward()
        with torch.no_grad():
            for parameter in _adaln_parameters(model):
                assert parameter.grad is not None
                parameter.sub_(lr * parameter.grad)

    sgd_step(full, 0.5)
    sgd_step(reduced, 0.5)
    with torch.no_grad():
        after_full, after_reduced = full(**inputs), reduced(**inputs)

    # The step did something visible ...
    assert (after_full.video - before_full.video).abs().max() > 1e-3
    assert (after_full.audio - before_full.audio).abs().max() > 1e-3
    # ... and the reduced parametrisation did the same thing.
    torch.testing.assert_close(after_reduced.video, after_full.video, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(after_reduced.audio, after_full.audio, atol=1e-5, rtol=1e-5)


def test_dense_pruned_checkpoint_round_trips_through_the_loader(monkeypatch, tmp_path: Path):
    torch.manual_seed(0)
    config = _tiny_config(num_layers=2)
    source = tmp_path / "minimax_h3_fl2va_bf16.safetensors"
    released = _released_state_dict(MiniMaxH3Transformer(config))
    save_file(released, source)
    _infer_with_tiny_config(monkeypatch, config)

    loaded = load_transformer(source, mode="fl2va", loading_device="cpu", adaln_rank=RANK)
    assert loaded.time_embedder is None
    assert loaded.adaln_t_table.shape == (DEFAULT_TABLE_POINTS, RANK)
    assert loaded.blocks[0].adaln_proj.linear.weight.dtype == torch.float32
    full_rows, full_columns = released["blocks.0.adaln_proj.linear.weight"].shape
    assert full_columns == config.time_embed_dim
    assert loaded.blocks[0].adaln_proj.linear.weight.shape == (full_rows, RANK)
    assert loaded.blocks[0].attn.qkv_proj.weight.dtype == torch.bfloat16

    trainer = MiniMaxH3Trainer()
    args = create_parser().parse_args(["--sdpa", "--dit", str(source), "--h3_adaln_rank", str(RANK)])
    module = trainer._build_network(args, None, loaded, None, None)
    saved = tmp_path / "h3_full.safetensors"
    module.save_weights(str(saved), torch.bfloat16, {"ss_h3_adaln_layout": trainer.extra_metadata(args)["ss_h3_adaln_layout"]})

    with safe_open(str(saved), framework="pt") as handle:
        keys = set(handle.keys())
        assert handle.metadata()["ss_h3_adaln_layout"] == "pruned"
    assert TABLE_KEY in keys
    assert not any(key.startswith("time_embedder.") for key in keys)
    assert keys == set(loaded.state_dict())
    validate_transformer_checkpoint(saved, config)

    reloaded = load_transformer(saved, mode="fl2va", loading_device="cpu")
    for key, tensor in loaded.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], tensor), key
    inputs = _tiny_inputs(dtype=torch.bfloat16)
    with torch.no_grad():
        first, second = loaded(**inputs), reloaded(**inputs)
    assert torch.equal(first.video, second.video)
    assert torch.equal(first.audio, second.audio)

    # The pruned result is a terminal layout: reducing it again is refused
    # before and inside the loader, and continuing it needs no flag.
    with pytest.raises(ValueError, match="already pruned"):
        load_transformer(saved, mode="fl2va", loading_device="cpu", adaln_rank=RANK)
    with pytest.raises(ValueError, match="already pruned"):
        trainer._validate_full_finetune_args(
            create_parser().parse_args(["--sdpa", "--dit", str(saved), "--h3_adaln_rank", str(RANK)])
        )
    resumed = create_parser().parse_args(["--sdpa", "--dit", str(saved)])
    trainer._validate_full_finetune_args(resumed)
    trainer._build_network(resumed, None, reloaded, None, None)
    assert trainer.extra_metadata(resumed)["ss_h3_adaln_layout"] == "pruned"


def test_default_dense_save_layout_is_unchanged_without_the_flag(monkeypatch, tmp_path: Path):
    torch.manual_seed(0)
    config = _tiny_config(num_layers=1)
    source = tmp_path / "minimax_h3_fl2va_bf16.safetensors"
    released = _released_state_dict(MiniMaxH3Transformer(config))
    save_file(released, source)
    _infer_with_tiny_config(monkeypatch, config)

    loaded = load_transformer(source, mode="fl2va", loading_device="cpu")
    args = create_parser().parse_args(["--sdpa", "--dit", str(source)])
    trainer = MiniMaxH3Trainer()
    module = trainer._build_network(args, None, loaded, None, None)
    saved = tmp_path / "h3_full.safetensors"
    module.save_weights(str(saved), torch.bfloat16, None)

    with safe_open(str(saved), framework="pt") as handle:
        written = {key: handle.get_tensor(key) for key in list(handle.keys())}
    assert set(written) == set(released)
    assert all(torch.equal(written[key], released[key]) for key in released)
    assert "ss_h3_adaln_layout" not in trainer.extra_metadata(args)


def test_trainable_ring_layout_coalesces_the_mixed_precision_reduced_block():
    """The ring packs every block tensor by its own dtype, so the float32
    reduced projections ride alongside the BF16 dense weights."""
    torch.manual_seed(0)
    reduced = _reduced_copy(MiniMaxH3Transformer(_tiny_config(num_layers=2)), RANK)
    for name, parameter in reduced.named_parameters():
        if ADALN_INFIX not in name:
            parameter.data = parameter.data.to(torch.bfloat16)

    layouts = []
    for block in reduced.blocks:
        entries = TrainableBlockRingOffloader._block_tensors(block)
        tensors = [entry[3] for entry in entries]
        layout, total = TrainableBlockRingOffloader._make_layout(tensors)
        layouts.append([(size, dtype, shape) for _, size, dtype, shape in layout])
        flat = torch.empty(total, dtype=torch.uint8)
        views = [flat[offset : offset + size].view(dtype).view(shape) for offset, size, dtype, shape in layout]
        for view, tensor in zip(views, tensors):
            view.copy_(tensor)
        for view, tensor in zip(views, tensors):
            assert torch.equal(view, tensor)
        assert {dtype for _, _, dtype, _ in layout} == {torch.bfloat16, torch.float32}
        assert any(shape[-1] == RANK and dtype == torch.float32 for _, _, dtype, shape in layout)
    assert layouts[0] == layouts[1]


def test_f32_reduced_projections_are_admitted_only_with_the_dense_trainers_metadata(monkeypatch, tmp_path: Path):
    """The released pruned checkpoints are F16 and stay F16-strict; the float32
    reduced weights of a dense fine-tune pass only because its file says so."""
    torch.manual_seed(0)
    config = _tiny_config(num_layers=1)
    source = tmp_path / "minimax_h3_fl2va_bf16.safetensors"
    save_file(_released_state_dict(MiniMaxH3Transformer(config)), source)
    _infer_with_tiny_config(monkeypatch, config)
    loaded = load_transformer(source, mode="fl2va", loading_device="cpu", adaln_rank=RANK)
    trainer = MiniMaxH3Trainer()
    args = create_parser().parse_args(["--sdpa", "--dit", str(source), "--h3_adaln_rank", str(RANK)])
    module = trainer._build_network(args, None, loaded, None, None)

    unlabelled = tmp_path / "unlabelled.safetensors"
    module.save_weights(str(unlabelled), torch.bfloat16, None)
    with pytest.raises(Exception, match="dtype F32, expected F16"):
        validate_transformer_checkpoint(unlabelled, config)

    labelled = tmp_path / "labelled.safetensors"
    module.save_weights(str(labelled), torch.bfloat16, trainer.extra_metadata(args))
    validate_transformer_checkpoint(labelled, config)
