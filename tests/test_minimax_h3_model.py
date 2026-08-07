import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file

import musubi_tuner.minimax_h3.model as h3_model
import musubi_tuner.minimax_h3.model_loader as h3_model_loader
from musubi_tuner.minimax_h3.int8_convrot import (
    enable_int8_convrot,
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
    rotate_activation,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.model_loader import (
    H3_FP8_OPTIMIZATION_EXCLUDE_KEYS,
    H3_FP8_OPTIMIZATION_TARGET_KEYS,
    build_block_swap_placement,
    resolve_transformer_checkpoint,
    validate_transformer_checkpoint,
)
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8


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


def test_h3_low_ram_block_swap_placement_matches_offloader_policy():
    target = torch.device("cuda", 0)
    placement, offloaded = build_block_swap_placement(
        target_device=target,
        num_blocks=10,
        blocks_to_swap=4,
        h2d_only=True,
    )

    assert offloaded == (1, 3, 6, 8)
    assert placement("blocks.1.attn.qkv_proj.weight", target).type == "cpu"
    assert placement("blocks.2.attn.qkv_proj.weight", torch.device("cpu")) == target
    assert placement("token_refiner.blocks.0.attn.qkv_proj.weight", torch.device("cpu")) == target
    assert placement("final_layer.video_out.weight", torch.device("cpu")) == target

    classic, classic_offloaded = build_block_swap_placement(
        target_device=target,
        num_blocks=10,
        blocks_to_swap=4,
        h2d_only=False,
    )
    assert classic_offloaded == (6, 7, 8, 9)
    assert classic("blocks.6.mlp.fc1.weight", target).type == "cpu"


def test_h3_attention_auto_dispatch_threshold_is_shape_aware():
    assert not h3_model._cudnn_auto_workload_is_large(1023, 4096, 128)
    assert not h3_model._cudnn_auto_workload_is_large(1024, 1024, 128)
    assert h3_model._cudnn_auto_workload_is_large(1449, 1449, 128)


def test_h3_attention_auto_dispatch_is_opt_in_and_sdpa_only():
    model = MiniMaxH3Transformer(_tiny_config())
    attentions = [module for module in model.modules() if isinstance(module, h3_model.MiniMaxH3Attention)]

    assert attentions
    assert not any(module.auto_dispatch for module in attentions)

    model.enable_attention_auto_dispatch()

    assert all(module.auto_dispatch for module in attentions)

    flash_model = MiniMaxH3Transformer(_tiny_config(), attention_mode="flash")
    with pytest.raises(ValueError, match="requires SDPA"):
        flash_model.enable_attention_auto_dispatch()


def test_h3_attention_auto_dispatch_keeps_cpu_on_ordinary_sdpa(monkeypatch):
    def forbidden_priority(*_args, **_kwargs):
        raise AssertionError("CPU attention must not enter the cuDNN priority context")

    monkeypatch.setattr(h3_model, "sdpa_kernel", forbidden_priority)
    module = h3_model.MiniMaxH3Attention(hidden_size=16, heads=2, head_dim=8, qk_norm_eps=1e-5)
    module.auto_dispatch = True

    output = module(torch.randn(1, 16, 16))

    assert output.shape == (1, 16, 16)


def test_h3_fused_qk_norm_rope_cpu_falls_back_exactly():
    torch.manual_seed(12)
    reference = MiniMaxH3Transformer(_tiny_config(num_layers=1))
    fused = MiniMaxH3Transformer(_tiny_config(num_layers=1))
    fused.load_state_dict(reference.state_dict())
    fused.requires_grad_(False)
    fused.enable_fused_qk_norm_rope()
    inputs = _tiny_inputs()

    expected = reference(**inputs)
    actual = fused(**inputs)

    torch.testing.assert_close(actual.video, expected.video)
    torch.testing.assert_close(actual.audio, expected.audio)
    assert all(module.fused_qk_norm_rope for module in fused.modules() if isinstance(module, h3_model.MiniMaxH3Attention))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SDPA")
def test_h3_attention_auto_dispatch_runs_cudnn_priority_forward_backward(monkeypatch):
    original_sdpa_kernel = h3_model.sdpa_kernel
    calls = []

    def recorded_priority(backends, *, set_priority=False):
        calls.append((backends, set_priority))
        return original_sdpa_kernel(backends, set_priority=set_priority)

    monkeypatch.setattr(h3_model, "sdpa_kernel", recorded_priority)
    monkeypatch.setattr(h3_model, "_CUDNN_AUTO_WORK_THRESHOLD", 1)
    monkeypatch.setattr(h3_model, "_CUDNN_AUTO_MIN_SEQUENCE", 1)
    module = h3_model.MiniMaxH3Attention(hidden_size=128, heads=1, head_dim=128, qk_norm_eps=1e-5).cuda().bfloat16()
    module.auto_dispatch = True
    hidden_states = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = module(hidden_states)
    output.float().square().mean().backward()

    assert calls and calls[0][1] is True
    assert torch.isfinite(output).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def _tiny_inputs(*, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
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


def test_native_h3_model_matches_released_full_checkpoint_key_count():
    with init_empty_weights(include_buffers=True):
        model = MiniMaxH3Transformer()
    state_dict = model.state_dict()

    assert len(state_dict) == 535
    assert state_dict["blocks.0.attn.qkv_proj.weight"].shape == (21504, 5376)
    assert state_dict["blocks.49.adaln_proj.linear.weight"].shape == (96768, 2688)
    assert state_dict["token_refiner.blocks.1.mlp.fc1.weight"].shape == (28672, 5376)
    assert state_dict["rope.inv_freq"].shape == (16,)


def test_native_h3_tiny_forward_and_backward():
    torch.manual_seed(1)
    config = _tiny_config()
    model = MiniMaxH3Transformer(config)
    output = model(**_tiny_inputs())

    assert output.video.shape == (1, 2, 16)
    assert output.audio.shape == (1, 4, 8)
    loss = output.video.square().mean() + output.audio.square().mean()
    loss.backward()
    gradient = model.blocks[0].attn.qkv_proj.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_h3_partial_gradient_checkpointing_targets_last_blocks(monkeypatch):
    model = MiniMaxH3Transformer(_tiny_config(num_layers=4))
    model.enable_gradient_checkpointing()
    model.set_gradient_checkpointing_blocks(2)
    original = model._checkpointed_block
    seen = []

    def recorded(block, *args):
        seen.append(next(index for index, candidate in enumerate(model.blocks) if candidate is block))
        return original(block, *args)

    monkeypatch.setattr(model, "_checkpointed_block", recorded)

    model(**_tiny_inputs())

    assert seen == [2, 3]


def test_h3_partial_gradient_checkpointing_validates_depth():
    model = MiniMaxH3Transformer(_tiny_config(num_layers=4))

    with pytest.raises(ValueError, match=r"\[0, 4\]"):
        model.set_gradient_checkpointing_blocks(5)


def test_h3_final_layer_projects_only_requested_media_rows():
    model = MiniMaxH3Transformer(_tiny_config(num_layers=0))
    seen = {}

    def capture(name):
        def hook(_module, inputs, _output):
            seen[name] = inputs[0].shape[1]

        return hook

    model.final_layer.video_out.register_forward_hook(capture("video"))
    model.final_layer.audio_out.register_forward_hook(capture("audio"))

    output = model(**_tiny_inputs())

    assert seen == {"video": 2, "audio": 4}
    assert output.video.shape[1] == 2
    assert output.audio.shape[1] == 4


def test_h3_regional_compile_forward_backward_parity():
    from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer

    torch.manual_seed(11)
    config = _tiny_config(num_layers=1)
    reference = MiniMaxH3Transformer(config)
    compiled = MiniMaxH3Transformer(config)
    compiled.load_state_dict(reference.state_dict())
    args = SimpleNamespace(
        compile_backend="eager",
        compile_mode="default",
        compile_dynamic="false",
        compile_fullgraph=True,
        compile_cache_size_limit=None,
        compile_auto_cache_size_limit=True,
        compile_fallback_to_eager=False,
        inductor_config=None,
    )
    trainer = MiniMaxH3NetworkTrainer()
    trainer.blocks_to_swap = 0
    compiled = trainer.compile_transformer(args, compiled)

    reference_inputs = _tiny_inputs()
    compiled_inputs = {name: value.detach().clone() for name, value in reference_inputs.items()}
    reference_inputs["encoder_hidden_states"].requires_grad_(True)
    compiled_inputs["encoder_hidden_states"].requires_grad_(True)
    expected = reference(**reference_inputs)
    actual = compiled(**compiled_inputs)
    (expected.video.square().mean() + expected.audio.square().mean()).backward()
    (actual.video.square().mean() + actual.audio.square().mean()).backward()

    torch.testing.assert_close(actual.video, expected.video)
    torch.testing.assert_close(actual.audio, expected.audio)
    torch.testing.assert_close(compiled_inputs["encoder_hidden_states"].grad, reference_inputs["encoder_hidden_states"].grad)


def test_h3_scaled_fp8_scope_and_forward_backward():
    torch.manual_seed(2)
    config = _tiny_config()
    model = MiniMaxH3Transformer(config)
    fp32_prefixes = (
        "video_patch_proj.",
        "audio_patch_proj.",
        "time_embedder.",
        "final_layer.video_out.",
        "final_layer.audio_out.",
        "rope.",
    )
    state_dict = {
        name: tensor.detach().to(torch.float32 if name.startswith(fp32_prefixes) else torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
    }
    optimize_state_dict_with_fp8(
        state_dict,
        calc_device=torch.device("cpu"),
        target_layer_keys=H3_FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_layer_keys=H3_FP8_OPTIMIZATION_EXCLUDE_KEYS,
    )
    apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.requires_grad_(False)

    scale_keys = sorted(key for key in state_dict if key.endswith(".scale_weight"))
    assert len(scale_keys) == config.num_layers * 5
    assert model.blocks[0].adaln_proj.linear.weight.dtype == torch.float8_e4m3fn
    assert model.blocks[0].adaln_proj.linear.scale_weight.dtype == torch.bfloat16
    assert model.blocks[0].norm1.weight.dtype == torch.bfloat16
    assert model.token_refiner.blocks[0].attn.qkv_proj.weight.dtype == torch.bfloat16
    assert model.condition_proj.weight.dtype == torch.bfloat16
    assert model.time_embedder.proj_in.weight.dtype == torch.float32

    inputs = _tiny_inputs(dtype=torch.bfloat16)
    inputs["encoder_hidden_states"].requires_grad_(True)
    output = model(**inputs)
    loss = output.video.float().square().mean() + output.audio.float().square().mean()
    loss.backward()

    assert torch.isfinite(output.video).all()
    assert torch.isfinite(output.audio).all()
    assert inputs["encoder_hidden_states"].grad is not None
    assert torch.isfinite(inputs["encoder_hidden_states"].grad).all()


def _quantize_tiny_h3_state(model: MiniMaxH3Transformer) -> dict[str, torch.Tensor]:
    fp32_prefixes = (
        "video_patch_proj.",
        "audio_patch_proj.",
        "time_embedder.",
        "final_layer.video_out.",
        "final_layer.audio_out.",
        "rope.",
    )
    state_dict = {
        name: tensor.detach().to(torch.float32 if name.startswith(fp32_prefixes) else torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
    }
    optimize_state_dict_with_fp8(
        state_dict,
        calc_device=torch.device("cpu"),
        target_layer_keys=H3_FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_layer_keys=H3_FP8_OPTIMIZATION_EXCLUDE_KEYS,
    )
    return state_dict


def _load_tiny_h3_scaled_fp8(config: MiniMaxH3TransformerConfig, state_dict: dict[str, torch.Tensor]) -> MiniMaxH3Transformer:
    model = MiniMaxH3Transformer(config)
    cloned_state = {name: tensor.detach().clone() for name, tensor in state_dict.items()}
    apply_fp8_monkey_patch(model, cloned_state, use_scaled_mm=False)
    model.load_state_dict(cloned_state, strict=True, assign=True)
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    return model


def test_h3_block_swap_uses_complete_offloader_lifecycle(monkeypatch):
    events = []

    class FakeOffloader:
        def prepare_block_devices_before_forward(self, blocks):
            events.append(("prepare", len(blocks)))

        def wait_for_block(self, block_index):
            events.append(("wait", block_index))

        def submit_move_blocks_forward(self, blocks, block_index):
            events.append(("submit", block_index, len(blocks)))

        def set_forward_only(self, value):
            events.append(("forward_only", value))

        def offload_to_cpu(self, blocks):
            events.append(("offload", len(blocks)))

    captured = {}

    def fake_create_offloader(block_type, blocks, num_blocks, blocks_to_swap, config):
        captured.update(
            block_type=block_type,
            blocks=blocks,
            num_blocks=num_blocks,
            blocks_to_swap=blocks_to_swap,
            config=config,
        )
        return FakeOffloader()

    monkeypatch.setattr(h3_model, "create_offloader", fake_create_offloader)
    config = _tiny_config(num_layers=4)
    model = MiniMaxH3Transformer(config)
    swap_config = BlockSwapConfig(device=torch.device("cpu"), supports_backward=True)

    model.enable_block_swap(2, swap_config)
    model.move_to_device_except_swap_blocks(torch.device("cpu"))
    model.prepare_block_swap_before_forward()
    model.switch_block_swap_for_inference()
    model.offload_block_swap_to_cpu()
    model.switch_block_swap_for_training()

    assert captured == {
        "block_type": "minimax-h3-block",
        "blocks": model.blocks,
        "num_blocks": 4,
        "blocks_to_swap": 2,
        "config": swap_config,
    }
    assert events[:6] == [
        ("prepare", 4),
        ("forward_only", True),
        ("prepare", 4),
        ("offload", 4),
        ("forward_only", False),
        ("prepare", 4),
    ]

    events.clear()
    model(**_tiny_inputs())
    assert events == [
        ("wait", 0),
        ("submit", 0, 4),
        ("wait", 1),
        ("submit", 1, 4),
        ("wait", 2),
        ("submit", 2, 4),
        ("wait", 3),
        ("submit", 3, 4),
    ]


def test_h3_block_swap_requires_two_resident_blocks(monkeypatch):
    monkeypatch.setattr(h3_model, "create_offloader", lambda *args, **kwargs: SimpleNamespace())
    model = MiniMaxH3Transformer(_tiny_config(num_layers=4))
    config = BlockSwapConfig(device=torch.device("cpu"), supports_backward=True)

    with pytest.raises(ValueError, match="cannot swap more than 2"):
        model.enable_block_swap(3, config)


def test_h3_layer_streaming_allows_every_block_to_be_offloaded(monkeypatch):
    captured = {}

    def fake_create_offloader(block_type, blocks, num_blocks, blocks_to_swap, config):
        captured.update(num_blocks=num_blocks, blocks_to_swap=blocks_to_swap, config=config)
        return SimpleNamespace()

    monkeypatch.setattr(h3_model, "create_offloader", fake_create_offloader)
    model = MiniMaxH3Transformer(_tiny_config(num_layers=4))
    config = BlockSwapConfig(
        device=torch.device("cuda"),
        supports_backward=True,
        h2d_only=True,
        granularity="layer",
    )

    model.enable_block_swap(4, config)

    assert captured == {"num_blocks": 4, "blocks_to_swap": 4, "config": config}


def test_h3_h2d_only_block_swap_requires_gradient_checkpointing():
    args = SimpleNamespace(
        block_swap_h2d_only=True,
        block_swap_ring_size=2,
        use_pinned_memory_for_block_swap=False,
        gradient_checkpointing=False,
    )

    with pytest.raises(ValueError, match="requires --gradient_checkpointing"):
        BlockSwapConfig.from_args(args, torch.device("cpu"), supports_backward=True)


def test_layer_granularity_requires_h2d_only():
    args = SimpleNamespace(
        block_swap_h2d_only=False,
        block_swap_ring_size=2,
        block_swap_granularity="layer",
        use_pinned_memory_for_block_swap=False,
        gradient_checkpointing=True,
    )

    with pytest.raises(ValueError, match="requires --block_swap_h2d_only"):
        BlockSwapConfig.from_args(args, torch.device("cuda"), supports_backward=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the real block offloaders")
@pytest.mark.parametrize(
    ("h2d_only", "ring_size", "use_pinned_memory", "granularity", "blocks_to_swap"),
    [
        (False, 2, False, "block", 2),
        (True, 1, False, "block", 2),
        (True, 2, False, "block", 2),
        (True, 2, True, "block", 2),
        (True, 1, False, "layer", 4),
        (True, 2, True, "layer", 4),
    ],
)
def test_h3_real_block_swap_forward_backward_parity(h2d_only, ring_size, use_pinned_memory, granularity, blocks_to_swap):
    torch.manual_seed(3)
    device = torch.device("cuda")
    config = _tiny_config(num_layers=4)
    source = MiniMaxH3Transformer(config)
    source_state = {name: tensor.detach().clone() for name, tensor in source.state_dict().items()}
    source_inputs = _tiny_inputs()
    del source

    reference = MiniMaxH3Transformer(config).to(device)
    reference.load_state_dict(source_state, strict=True)
    reference.requires_grad_(False)
    reference.enable_gradient_checkpointing()
    reference_inputs = {
        key: value.to(device).requires_grad_(key == "encoder_hidden_states") if value.is_floating_point() else value.to(device)
        for key, value in source_inputs.items()
    }
    reference_output = reference(**reference_inputs)
    reference_loss = reference_output.video.square().mean() + reference_output.audio.square().mean()
    reference_loss.backward()
    reference_video = reference_output.video.detach().clone()
    reference_audio = reference_output.audio.detach().clone()
    reference_gradient = reference_inputs["encoder_hidden_states"].grad.detach().clone()
    del reference, reference_inputs, reference_output, reference_loss
    torch.cuda.empty_cache()

    swapped = MiniMaxH3Transformer(config)
    swapped.load_state_dict(source_state, strict=True)
    swapped.requires_grad_(False)
    swapped.enable_gradient_checkpointing()
    args = SimpleNamespace(
        block_swap_h2d_only=h2d_only,
        block_swap_ring_size=ring_size,
        block_swap_granularity=granularity,
        use_pinned_memory_for_block_swap=use_pinned_memory,
        gradient_checkpointing=True,
    )
    swap_config = BlockSwapConfig.from_args(args, device, supports_backward=True)
    swapped.enable_block_swap(blocks_to_swap, swap_config)
    swapped.move_to_device_except_swap_blocks(device)
    swapped.prepare_block_swap_before_forward()
    swapped.switch_block_swap_for_inference()
    with torch.no_grad():
        sample_inputs = {key: value.to(device) for key, value in source_inputs.items()}
        sample_output = swapped(**sample_inputs)
    torch.testing.assert_close(sample_output.video, reference_video)
    torch.testing.assert_close(sample_output.audio, reference_audio)
    del sample_inputs, sample_output

    swapped.offload_block_swap_to_cpu()
    assert all(parameter.device.type == "cpu" for parameter in swapped.parameters())
    if hasattr(swapped.offloader, "ring_flat"):
        assert swapped.offloader.ring_flat is None
    swapped.move_to_device_except_swap_blocks(device)
    swapped.switch_block_swap_for_training()
    swapped_inputs = {
        key: value.to(device).requires_grad_(key == "encoder_hidden_states") if value.is_floating_point() else value.to(device)
        for key, value in source_inputs.items()
    }
    swapped_output = swapped(**swapped_inputs)
    swapped_loss = swapped_output.video.square().mean() + swapped_output.audio.square().mean()
    swapped_loss.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(swapped_output.video, reference_video)
    torch.testing.assert_close(swapped_output.audio, reference_audio)
    torch.testing.assert_close(swapped_inputs["encoder_hidden_states"].grad, reference_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the real block offloaders")
@pytest.mark.parametrize(
    ("h2d_only", "use_pinned_memory", "granularity", "blocks_to_swap"),
    [(False, False, "block", 2), (True, False, "block", 2), (True, True, "block", 2), (True, True, "layer", 4)],
)
def test_h3_scaled_fp8_block_swap_forward_backward_parity(h2d_only, use_pinned_memory, granularity, blocks_to_swap):
    torch.manual_seed(4)
    device = torch.device("cuda")
    config = _tiny_config(num_layers=4)
    source = MiniMaxH3Transformer(config)
    quantized_state = _quantize_tiny_h3_state(source)
    source_inputs = _tiny_inputs(dtype=torch.bfloat16)
    del source

    reference = _load_tiny_h3_scaled_fp8(config, quantized_state).to(device)
    reference_inputs = {
        key: value.to(device).requires_grad_(key == "encoder_hidden_states") if value.is_floating_point() else value.to(device)
        for key, value in source_inputs.items()
    }
    reference_output = reference(**reference_inputs)
    reference_loss = reference_output.video.square().mean() + reference_output.audio.square().mean()
    reference_loss.backward()
    reference_video = reference_output.video.detach().clone()
    reference_audio = reference_output.audio.detach().clone()
    reference_gradient = reference_inputs["encoder_hidden_states"].grad.detach().clone()
    del reference, reference_inputs, reference_output, reference_loss
    torch.cuda.empty_cache()

    swapped = _load_tiny_h3_scaled_fp8(config, quantized_state)
    args = SimpleNamespace(
        block_swap_h2d_only=h2d_only,
        block_swap_ring_size=2,
        block_swap_granularity=granularity,
        use_pinned_memory_for_block_swap=use_pinned_memory,
        gradient_checkpointing=True,
    )
    swap_config = BlockSwapConfig.from_args(args, device, supports_backward=True)
    swapped.enable_block_swap(blocks_to_swap, swap_config)
    swapped.move_to_device_except_swap_blocks(device)
    swapped.prepare_block_swap_before_forward()
    swapped.switch_block_swap_for_inference()
    with torch.no_grad():
        sample_inputs = {key: value.to(device) for key, value in source_inputs.items()}
        sample_output = swapped(**sample_inputs)
    torch.testing.assert_close(sample_output.video, reference_video)
    torch.testing.assert_close(sample_output.audio, reference_audio)
    del sample_inputs, sample_output

    swapped.offload_block_swap_to_cpu()
    assert all(parameter.device.type == "cpu" for parameter in swapped.parameters())
    if hasattr(swapped.offloader, "ring_flat"):
        assert swapped.offloader.ring_flat is None
    swapped.move_to_device_except_swap_blocks(device)
    swapped.switch_block_swap_for_training()
    swapped_inputs = {
        key: value.to(device).requires_grad_(key == "encoder_hidden_states") if value.is_floating_point() else value.to(device)
        for key, value in source_inputs.items()
    }
    swapped_output = swapped(**swapped_inputs)
    swapped_loss = swapped_output.video.square().mean() + swapped_output.audio.square().mean()
    swapped_loss.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(swapped_output.video, reference_video)
    torch.testing.assert_close(swapped_output.audio, reference_audio)
    torch.testing.assert_close(swapped_inputs["encoder_hidden_states"].grad, reference_gradient)


def test_h3_checkpoint_validator_accepts_exact_native_mixed_precision_layout(tmp_path: Path):
    config = _tiny_config()
    model = MiniMaxH3Transformer(config)
    fp32_prefixes = (
        "video_patch_proj.",
        "audio_patch_proj.",
        "time_embedder.",
        "final_layer.video_out.",
        "final_layer.audio_out.",
        "rope.",
    )
    state_dict = {
        name: tensor.detach().to(torch.float32 if name.startswith(fp32_prefixes) else torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
    }
    checkpoint_path = tmp_path / "minimax_h3_fl2va_bf16.safetensors"
    save_file(state_dict, checkpoint_path)

    tensor_count, parameter_count = validate_transformer_checkpoint(checkpoint_path, config)

    assert tensor_count == len(state_dict)
    assert parameter_count == sum(tensor.numel() for tensor in state_dict.values())
    assert resolve_transformer_checkpoint(checkpoint_path, "fl2va") == checkpoint_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="low-RAM split placement requires CUDA")
def test_h3_loader_streams_resident_and_offloaded_blocks_to_final_devices(monkeypatch, tmp_path: Path):
    config = _tiny_config(num_layers=2)
    model = MiniMaxH3Transformer(config)
    fp32_prefixes = (
        "video_patch_proj.",
        "audio_patch_proj.",
        "time_embedder.",
        "final_layer.video_out.",
        "final_layer.audio_out.",
        "rope.",
    )
    state_dict = {
        name: tensor.detach().to(torch.float32 if name.startswith(fp32_prefixes) else torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
    }
    checkpoint = tmp_path / "minimax_h3_fl2va_bf16.safetensors"
    save_file(state_dict, checkpoint)
    monkeypatch.setattr(h3_model_loader, "infer_transformer_config", lambda *_args, **_kwargs: config)

    loaded = h3_model_loader.load_transformer(
        checkpoint,
        mode="fl2va",
        loading_device="cpu",
        quantization_device="cuda:0",
        target_device="cuda:0",
        blocks_to_swap=1,
        block_swap_h2d_only=True,
    )

    assert loaded.blocks[0].attn.qkv_proj.weight.device.type == "cuda"
    assert loaded.blocks[1].attn.qkv_proj.weight.device.type == "cpu"
    assert loaded.token_refiner.blocks[0].attn.qkv_proj.weight.device.type == "cuda"
    assert loaded.final_layer.video_out.weight.device.type == "cuda"


def test_h3_pruned_int8_convrot_checkpoint_contract_and_adapter_gradient(tmp_path: Path):
    torch.manual_seed(11)
    config = _tiny_config(num_layers=1)
    config = MiniMaxH3TransformerConfig(**{**config.__dict__, "time_embed_dim": 4, "adaln_t_table_size": 5})
    model = MiniMaxH3Transformer(config)
    fp32_prefixes = (
        "video_patch_proj.",
        "audio_patch_proj.",
        "final_layer.video_out.",
        "final_layer.audio_out.",
        "rope.",
        "adaln_t_table",
    )
    state_dict = {}
    for name, tensor in model.state_dict().items():
        dtype = (
            torch.float32 if name.startswith(fp32_prefixes) else torch.float16 if ".adaln_proj.linear." in name else torch.bfloat16
        )
        state_dict[name] = tensor.detach().to(dtype).contiguous()

    quantized_name = "blocks.0.attn.qkv_proj"
    weight = state_dict[f"{quantized_name}.weight"].float()
    rotated = rotate_activation(weight, 4)
    scale = (rotated.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    state_dict[f"{quantized_name}.weight"] = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
    state_dict[f"{quantized_name}.weight_scale"] = scale.float()
    marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}
    state_dict[f"{quantized_name}.comfy_quant"] = torch.tensor(list(json.dumps(marker).encode()), dtype=torch.uint8)
    checkpoint = tmp_path / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    save_file(state_dict, checkpoint)

    tensor_count, _ = validate_transformer_checkpoint(checkpoint, config)
    assert tensor_count == len(state_dict)

    loaded_state, quantized_layers = load_comfy_int8_convrot_state_dict(checkpoint, device=torch.device("cpu"))
    loaded = MiniMaxH3Transformer(config).requires_grad_(False)
    assert prepare_int8_convrot_modules(loaded, loaded_state) == quantized_layers == 1
    info = loaded.load_state_dict(loaded_state, strict=True, assign=True)
    assert not info.missing_keys and not info.unexpected_keys
    assert enable_int8_convrot(loaded) == 1

    inputs = _tiny_inputs()
    inputs["encoder_hidden_states"].requires_grad_(True)
    output = loaded(**inputs)
    (output.video.square().mean() + output.audio.square().mean()).backward()
    assert inputs["encoder_hidden_states"].grad is not None
    assert torch.isfinite(inputs["encoder_hidden_states"].grad).all()
    assert loaded.blocks[0].attn.qkv_proj.weight.grad is None


def test_h3_checkpoint_directory_resolution_keeps_existing_bf16_default(tmp_path: Path):
    model_dir = tmp_path / "diffusion_models"
    model_dir.mkdir()
    full = model_dir / "minimax_h3_fl2va_bf16.safetensors"
    pruned = model_dir / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    full.touch()
    pruned.touch()

    assert resolve_transformer_checkpoint(tmp_path, "fl2va") == full
    assert resolve_transformer_checkpoint(tmp_path, "fl2va", int8_convrot=True) == pruned


def test_h3_attention_mask_defaults_to_none_and_changes_nothing():
    torch.manual_seed(1)
    model = MiniMaxH3Transformer(_tiny_config())
    inputs = _tiny_inputs()

    without = model(**inputs)
    allow_all = torch.ones(9, 9, dtype=torch.bool)
    with_open_mask = model(**inputs, attention_mask=allow_all)

    torch.testing.assert_close(without.video, with_open_mask.video)
    torch.testing.assert_close(without.audio, with_open_mask.audio)


def test_h3_attention_mask_actually_restricts_attention():
    torch.manual_seed(1)
    model = MiniMaxH3Transformer(_tiny_config())
    inputs = _tiny_inputs()
    causal = torch.tril(torch.ones(9, 9, dtype=torch.bool))

    open_output = model(**inputs)
    causal_output = model(**inputs, attention_mask=causal)

    assert not torch.allclose(open_output.video, causal_output.video)


def test_h3_attention_mask_cannot_re_expose_padding():
    # The padding mask is applied on top of any caller topology, so a mask that
    # permits everything still cannot let real tokens attend to padding.
    torch.manual_seed(1)
    model = MiniMaxH3Transformer(_tiny_config())
    inputs = _tiny_inputs()
    inputs["token_tags"] = torch.tensor([1, 1, 1, 2, 2, 2, 2, 0, -1])

    padded = model(**inputs)
    with_open_mask = model(**inputs, attention_mask=torch.ones(9, 9, dtype=torch.bool))

    torch.testing.assert_close(padded.video, with_open_mask.video)


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (torch.ones(9, 9, dtype=torch.float32), "boolean"),
        (torch.ones(4, 4, dtype=torch.bool), r"\[9, 9\]"),
    ],
)
def test_h3_attention_mask_is_validated(mask, message):
    model = MiniMaxH3Transformer(_tiny_config())

    with pytest.raises(ValueError, match=message):
        model(**_tiny_inputs(), attention_mask=mask)
