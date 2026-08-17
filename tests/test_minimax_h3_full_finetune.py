import copy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint
from transformers import Adafactor

from musubi_tuner.minimax_h3_train import MiniMaxH3FullFinetuneModule, MiniMaxH3Trainer, create_parser
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, TrainableBlockRingOffloader


def test_h3_full_finetune_parser_has_memory_optimized_defaults():
    args = create_parser().parse_args(["--sdpa"])

    assert args.network_module is None
    assert args.mixed_precision == "bf16"
    assert args.full_bf16 is True
    assert args.gradient_checkpointing is True
    assert args.optimizer_type == "Adafactor"
    assert args.optimizer_args == ["scale_parameter=False", "relative_step=False", "warmup_init=False"]
    assert args.fused_backward_pass is True
    assert args.max_grad_norm == 0.0
    assert args.adafactor_triton is False
    assert args.block_swap_trainable_ring is False
    MiniMaxH3Trainer()._validate_full_finetune_args(args)


def test_h3_triton_adafactor_rejects_relative_step_mode():
    args = create_parser().parse_args(
        ["--sdpa", "--adafactor_triton", "--optimizer_args", "scale_parameter=True", "relative_step=True"]
    )

    with pytest.raises(ValueError, match="manual-LR Adafactor"):
        MiniMaxH3Trainer()._validate_full_finetune_args(args)


def test_h3_full_finetune_rejects_frozen_weight_and_lora_paths():
    trainer = MiniMaxH3Trainer()
    args = create_parser().parse_args(["--sdpa", "--fp8_base"])
    with pytest.raises(ValueError, match="ordinary BF16 weights"):
        trainer._validate_full_finetune_args(args)

    args = create_parser().parse_args(["--sdpa", "--network_weights", "adapter.safetensors"])
    with pytest.raises(ValueError, match="LoRA initialization"):
        trainer._validate_full_finetune_args(args)

    # Without this rejection the flag reaches handle_model_specific_args and
    # crashes on ``network_module=None``.endswith().
    args = create_parser().parse_args(["--sdpa", "--h3_lora_token_refiner"])
    with pytest.raises(ValueError, match="LoRA-only option"):
        trainer._validate_full_finetune_args(args)

    # A live overlay never reaches a dense checkpoint, so a finished full
    # fine-tune would not reproduce the field it trained inside.
    args = create_parser().parse_args(["--sdpa", "--h3_overlay_weights", "overlay.safetensors"])
    with pytest.raises(ValueError, match="LoRA-only option"):
        trainer._validate_full_finetune_args(args)


def test_h3_trainable_ring_contract_is_checked_before_cuda_setup():
    args = create_parser().parse_args(
        [
            "--sdpa",
            "--blocks_to_swap",
            "4",
            "--block_swap_trainable_ring",
            "--use_pinned_memory_for_block_swap",
        ]
    )
    config = BlockSwapConfig.from_args(args, torch.device("cuda"), supports_backward=True)
    assert config.trainable_ring is True
    assert config.ring_size == 2

    args.use_pinned_memory_for_block_swap = False
    with pytest.raises(ValueError, match="requires --use_pinned_memory_for_block_swap"):
        BlockSwapConfig.from_args(args, torch.device("cuda"), supports_backward=True)


def test_h3_fused_adafactor_hook_steps_and_releases_gradient():
    parameter = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32))
    optimizer = Adafactor(
        [parameter],
        lr=1e-2,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    args = SimpleNamespace(fused_backward_pass=True, adafactor_triton=False)
    accelerator = SimpleNamespace(sync_gradients=True)
    before = parameter.detach().clone()

    MiniMaxH3Trainer._install_fused_optimizer(args, accelerator, optimizer)
    parameter.square().sum().backward()

    assert parameter.grad is None
    assert not torch.equal(parameter, before)


def test_h3_full_module_streams_native_transformer_checkpoint(tmp_path):
    transformer = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4)).to(torch.bfloat16)
    module = MiniMaxH3FullFinetuneModule(transformer)
    output = tmp_path / "h3.safetensors"

    module.save_weights(str(output), torch.bfloat16, {"kind": "test"})
    saved = load_file(output)

    assert set(saved) == set(transformer.state_dict())
    assert all(tensor.dtype == torch.bfloat16 for tensor in saved.values() if tensor.is_floating_point())


def test_h3_mem_eff_save_can_be_turned_off_and_writes_the_same_checkpoint(tmp_path):
    assert create_parser().parse_args(["--sdpa"]).mem_eff_save is True
    assert create_parser().parse_args(["--sdpa", "--no_mem_eff_save"]).mem_eff_save is False

    transformer = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4)).to(torch.bfloat16)
    streamed = tmp_path / "streamed.safetensors"
    ordinary = tmp_path / "ordinary.safetensors"

    MiniMaxH3FullFinetuneModule(transformer, mem_eff_save=True).save_weights(str(streamed), torch.bfloat16, None)
    MiniMaxH3FullFinetuneModule(transformer, mem_eff_save=False).save_weights(str(ordinary), torch.bfloat16, None)

    left, right = load_file(streamed), load_file(ordinary)
    assert set(left) == set(right)
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_h3_full_module_can_skip_large_weight_write_for_diagnostics(tmp_path):
    transformer = torch.nn.Linear(3, 4).to(torch.bfloat16)
    output = tmp_path / "skipped.safetensors"

    MiniMaxH3FullFinetuneModule(transformer, save_weights=False).save_weights(str(output), None, None)

    assert not output.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_trainable_block_ring_recomputes_blocks_in_reverse_order():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8)) for _ in range(4)]
            )

        def forward(self, value, offloader=None):
            for index, block in enumerate(self.blocks):
                if offloader is not None:
                    offloader.wait_for_block(index)

                def run(hidden, block=block, index=index):
                    if offloader is not None and offloader.recompute_requires_wait:
                        offloader.wait_for_block(index)
                    return block(hidden)

                value = checkpoint(run, value, use_reentrant=False)
                if offloader is not None:
                    offloader.submit_move_blocks_forward(self.blocks, index)
            return value

    torch.manual_seed(123)
    reference = Toy().cuda()
    streamed = copy.deepcopy(reference).cpu()
    offloader = TrainableBlockRingOffloader(
        "toy",
        streamed.blocks,
        num_blocks=4,
        blocks_to_swap=2,
        supports_backward=True,
        device=torch.device("cuda"),
        ring_size=1,
        use_pinned_memory=True,
    )
    offloader.prepare_block_devices_before_forward(streamed.blocks)
    ref_input = torch.randn(2, 8, device="cuda", requires_grad=True)
    ring_input = ref_input.detach().clone().requires_grad_(True)

    ref_output = reference(ref_input)
    ring_output = streamed(ring_input, offloader)
    ref_output.square().sum().backward()
    ring_output.square().sum().backward()

    torch.testing.assert_close(ring_output, ref_output)
    torch.testing.assert_close(ring_input.grad, ref_input.grad)
    for ref_parameter, ring_parameter in zip(reference.parameters(), streamed.parameters()):
        torch.testing.assert_close(ring_parameter.grad, ref_parameter.grad)

    offloader.offload_to_cpu(streamed.blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_adafactor_updates_supported_bf16_matrix():
    pytest.importorskip("triton")
    from musubi_tuner.modules.adafactor_triton import patch_adafactor_triton

    torch.manual_seed(321)
    parameter = torch.nn.Parameter(torch.randn(64, 128, device="cuda", dtype=torch.bfloat16))
    before = parameter.detach().clone()
    parameter.grad = torch.randn_like(parameter)
    optimizer = Adafactor(
        [parameter],
        lr=1e-2,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    patch_adafactor_triton(optimizer)
    optimizer.step_param(parameter, optimizer.param_groups[0])

    state = optimizer.state[parameter]
    assert state["exp_avg_sq_row"].dtype == torch.float32
    assert state["exp_avg_sq_col"].dtype == torch.float32
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, before)
