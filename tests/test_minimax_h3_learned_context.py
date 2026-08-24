from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.learned_context import (
    COMFYUI_LEARNED_CONTEXT_KEY,
    H3_TEXT_HIDDEN_SIZE,
    apply_learned_context,
    compose_learned_context,
    load_learned_context,
    load_learned_context_sequence,
    prepend_learned_context,
)
from musubi_tuner.minimax_h3_train_learned_context import (
    H3LearnedContext,
    MiniMaxH3LearnedContextTrainer,
    create_parser,
)
from musubi_tuner import minimax_h3_train_learned_context as learned_context_training
from musubi_tuner.minimax_h3_generate_video import create_parser as create_generate_parser


def _context(rows: int = 3, *, requires_grad: bool = False) -> torch.Tensor:
    return (
        torch.arange(rows * H3_TEXT_HIDDEN_SIZE, dtype=torch.float32)
        .reshape(rows, H3_TEXT_HIDDEN_SIZE)
        .requires_grad_(requires_grad)
    )


def test_disabled_path_preserves_objects() -> None:
    hidden = torch.zeros((2, H3_TEXT_HIDDEN_SIZE))
    tags = torch.tensor([1, 0], dtype=torch.long)

    output_hidden, output_tags = prepend_learned_context(hidden, tags, None)

    assert output_hidden is hidden
    assert output_tags is tags


def test_prepend_preserves_order_tags_and_gradient() -> None:
    context = _context(requires_grad=True)
    hidden = torch.full((2, H3_TEXT_HIDDEN_SIZE), -1.0, dtype=torch.bfloat16)
    tags = torch.tensor([1, 0], dtype=torch.long)

    output_hidden, output_tags = prepend_learned_context(hidden, tags, context)
    output_hidden.float().sum().backward()

    assert output_hidden.shape == (5, H3_TEXT_HIDDEN_SIZE)
    torch.testing.assert_close(output_hidden[:3], context.detach().to(torch.bfloat16))
    torch.testing.assert_close(output_hidden[3:], hidden)
    torch.testing.assert_close(output_tags, torch.tensor([1, 1, 1, 1, 0]))
    assert context.grad is not None
    torch.testing.assert_close(context.grad, torch.ones_like(context))


def test_apply_does_not_mutate_conditioning() -> None:
    conditioning = {
        H3_TEXT_HIDDEN_KEY: torch.zeros((2, H3_TEXT_HIDDEN_SIZE)),
        H3_TEXT_TOKEN_TAGS_KEY: torch.tensor([1, 0], dtype=torch.long),
        "unrelated": torch.tensor(7),
    }

    output = apply_learned_context(conditioning, _context(1))

    assert output is not conditioning
    assert conditioning[H3_TEXT_HIDDEN_KEY].shape[0] == 2
    assert output[H3_TEXT_HIDDEN_KEY].shape[0] == 3
    assert output["unrelated"] is conditioning["unrelated"]


def test_replace_composition_uses_only_learned_rows() -> None:
    hidden = torch.full((3, H3_TEXT_HIDDEN_SIZE), -1.0)
    tags = torch.tensor([1, 1, 0], dtype=torch.long)
    context = _context(2)

    output_hidden, output_tags = compose_learned_context(hidden, tags, context, "replace")

    torch.testing.assert_close(output_hidden, context)
    torch.testing.assert_close(output_tags, torch.ones(2, dtype=torch.long))


def test_loads_canonical_context_artifact(tmp_path) -> None:
    expected = _context(2).to(torch.bfloat16)
    path = tmp_path / "context.safetensors"
    save_file({COMFYUI_LEARNED_CONTEXT_KEY: expected}, path)
    torch.testing.assert_close(load_learned_context(path), expected)


def test_rejects_ambiguous_or_malformed_artifact(tmp_path) -> None:
    unexpected = tmp_path / "unexpected.safetensors"
    save_file({COMFYUI_LEARNED_CONTEXT_KEY: _context(1), "extra": _context(1)}, unexpected)
    malformed = tmp_path / "malformed.safetensors"
    save_file({COMFYUI_LEARNED_CONTEXT_KEY: torch.zeros((2, 16))}, malformed)

    for path in (unexpected, malformed):
        try:
            load_learned_context(path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {path.name} to be rejected")


def test_training_module_exports_only_comfyui_key() -> None:
    module = H3LearnedContext(_context(2))

    state = module.snapshot_weights(torch.bfloat16)

    assert set(state) == {COMFYUI_LEARNED_CONTEXT_KEY}
    assert state[COMFYUI_LEARNED_CONTEXT_KEY].dtype == torch.bfloat16
    assert state[COMFYUI_LEARNED_CONTEXT_KEY].shape == (2, H3_TEXT_HIDDEN_SIZE)


def test_training_module_refuses_nonfinite_runtime_or_checkpoint() -> None:
    module = H3LearnedContext(_context(1))
    module.weight.data[0, 0] = torch.nan

    for operation in (module.on_step_start, lambda: module.snapshot_weights(torch.bfloat16)):
        try:
            operation()
        except FloatingPointError:
            pass
        else:
            raise AssertionError("non-finite learned context must fail immediately")


def test_training_module_runtime_disable_preserves_base_conditioning() -> None:
    module = H3LearnedContext(_context(2))
    hidden = torch.zeros((3, H3_TEXT_HIDDEN_SIZE))
    tags = torch.tensor([1, 1, 0], dtype=torch.long)

    module.set_enabled(False)
    output_hidden, output_tags = module.prepend(hidden, tags)

    assert output_hidden is hidden
    assert output_tags is tags


def test_multiple_contexts_compose_in_argument_order(tmp_path) -> None:
    first = torch.full((1, H3_TEXT_HIDDEN_SIZE), 1.0, dtype=torch.bfloat16)
    second = torch.full((2, H3_TEXT_HIDDEN_SIZE), 2.0, dtype=torch.bfloat16)
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    save_file({COMFYUI_LEARNED_CONTEXT_KEY: first}, first_path)
    save_file({COMFYUI_LEARNED_CONTEXT_KEY: second}, second_path)

    combined = load_learned_context_sequence((first_path, second_path))

    assert combined is not None
    torch.testing.assert_close(combined, torch.cat((first, second)))


def test_context_multipliers_scale_and_zero_disables_without_placeholder_tokens(tmp_path) -> None:
    first = torch.full((1, H3_TEXT_HIDDEN_SIZE), 2.0, dtype=torch.bfloat16)
    second = torch.full((2, H3_TEXT_HIDDEN_SIZE), 3.0, dtype=torch.bfloat16)
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    save_file({COMFYUI_LEARNED_CONTEXT_KEY: first}, first_path)
    save_file({COMFYUI_LEARNED_CONTEXT_KEY: second}, second_path)

    combined = load_learned_context_sequence((first_path, second_path), (-0.5, 0.0))

    assert combined is not None
    torch.testing.assert_close(combined, torch.full_like(first, -1.0))
    assert load_learned_context_sequence((first_path,), (0.0,)) is None


def test_accepts_h2d_only_block_swap_with_gradient_checkpointing() -> None:
    args = create_parser().parse_args(["--text_encoder", "qwen.safetensors", "--h3_learned_context_init_prompt", "water"])
    args.block_swap_h2d_only = True
    args.gradient_checkpointing = True

    MiniMaxH3LearnedContextTrainer()._validate_learned_context_args(args)


def test_learned_context_freezes_the_transformer() -> None:
    network = H3LearnedContext(torch.zeros(1, H3_TEXT_HIDDEN_SIZE))
    transformer = torch.nn.Linear(4, 4)

    network.apply_to(None, transformer)

    assert network.weight.requires_grad
    assert not any(parameter.requires_grad for parameter in transformer.parameters())


def test_trainer_build_network_freezes_the_transformer(tmp_path: Path) -> None:
    initializer = tmp_path / "initializer.safetensors"
    save_file({COMFYUI_LEARNED_CONTEXT_KEY: torch.zeros(1, H3_TEXT_HIDDEN_SIZE)}, initializer)
    args = create_parser().parse_args(["--h3_learned_context_init", str(initializer), "--gradient_checkpointing"])

    class Transformer(torch.nn.Linear):
        def enable_gradient_checkpointing(self, cpu_offload: bool) -> None:
            self.checkpointing_enabled = True
            self.cpu_offload = cpu_offload

    transformer = Transformer(4, 4)
    trainer = MiniMaxH3LearnedContextTrainer()

    trainer._build_network(args, None, transformer, None, torch.bfloat16)

    assert not any(parameter.requires_grad for parameter in transformer.parameters())
    assert transformer.checkpointing_enabled is True
    assert transformer.cpu_offload is False


def test_prepare_grad_configures_final_transformer() -> None:
    network = H3LearnedContext(torch.zeros(1, H3_TEXT_HIDDEN_SIZE))

    transformer = torch.nn.Linear(4, 4)
    network.prepare_grad_etc(transformer)

    assert not any(parameter.requires_grad for parameter in transformer.parameters())


def test_accepts_convrot_with_h2d_only_block_swap() -> None:
    args = create_parser().parse_args(["--text_encoder", "qwen.safetensors", "--h3_learned_context_init_prompt", "water"])
    args.block_swap_h2d_only = True
    args.gradient_checkpointing = True
    args.h3_convrot_int8 = True

    MiniMaxH3LearnedContextTrainer()._validate_learned_context_args(args)


def test_inference_cli_accepts_contexts_and_loras_together() -> None:
    args = create_generate_parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--prompt",
            "test",
            "--output",
            "out.mp4",
            "--h3_learned_context",
            "first.safetensors",
            "--lora_weight",
            "style.safetensors",
            "--h3_learned_context",
            "second.safetensors",
            "--h3_learned_context_multiplier",
            "-1",
        ]
    )

    assert args.h3_learned_context == [Path("first.safetensors"), Path("second.safetensors")]
    assert args.lora_weight == [Path("style.safetensors")]
    assert args.h3_learned_context_multiplier == [-1.0]


def test_initializes_directly_from_complete_qwen_prompt_output(monkeypatch) -> None:
    expected = _context(6).to(torch.bfloat16)
    closed = []

    class Encoder:
        def encode_prompt(self, prompt):
            assert prompt == "A precise initialization prompt"
            return {
                H3_TEXT_HIDDEN_KEY: expected,
                H3_TEXT_TOKEN_TAGS_KEY: torch.ones(expected.shape[0], dtype=torch.long),
            }

        def close(self):
            closed.append(True)

    monkeypatch.setattr(learned_context_training, "create_conditioning_encoder", lambda **_kwargs: Encoder())
    args = create_parser().parse_args(
        [
            "--text_encoder",
            "qwen.safetensors",
            "--h3_learned_context_init_prompt",
            "A precise initialization prompt",
        ]
    )
    trainer = MiniMaxH3LearnedContextTrainer()
    accelerator = SimpleNamespace(device=torch.device("cpu"))

    trainer._encode_prompt_initializer(args, accelerator)
    module = trainer._build_network(args, accelerator, None, None, torch.bfloat16)

    torch.testing.assert_close(module.weight.detach(), expected.float())
    assert closed == [True]


def test_documented_training_command_parses() -> None:
    args = create_parser().parse_args(
        [
            "--dataset_config",
            "examples/minimax_h3/learned_context/dataset.toml",
            "--dit",
            "model.safetensors",
            "--h3_training_mode",
            "fl2va",
            "--text_encoder",
            "qwen.safetensors",
            "--text_encoder_quantization",
            "nvfp4_awq",
            "--h3_text_encoder_blocks_to_stream",
            "50",
            "--h3_learned_context_init_prompt",
            "water rapidly rises and floods the entire scene",
            "--h3_learned_context_composition",
            "prepend",
            "--h3_shift_video",
            "12",
            "--h3_shift_audio",
            "3",
            "--h3_loss_balance",
            "modality",
            "--h3_video_loss_weight",
            "1",
            "--h3_audio_loss_weight",
            "1",
            "--learning_rate",
            "1e-3",
            "--max_train_steps",
            "1000",
            "--output_dir",
            "output",
            "--output_name",
            "water_context",
            "--sdpa",
            "--mixed_precision",
            "bf16",
            "--save_precision",
            "bf16",
            "--gradient_checkpointing",
            "--h3_convrot_int8",
            "--h3_convrot_int8_fwd",
            "int8",
            "--h3_convrot_int8_bwd",
            "int8",
            "--h3_adaln_rank",
            "16",
            "--blocks_to_swap",
            "30",
            "--block_swap_h2d_only",
            "--block_swap_ring_size",
            "2",
        ]
    )

    MiniMaxH3LearnedContextTrainer()._validate_learned_context_args(args)
    assert args.h3_learned_context_init_prompt.startswith("water rapidly")
    assert args.blocks_to_swap == 30
