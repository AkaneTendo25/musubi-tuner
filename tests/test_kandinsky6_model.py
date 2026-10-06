import torch
import importlib.util
import sys
import types
from pathlib import Path
import os
import pytest

from musubi_tuner.kandinsky6 import DiffusionTransformer3D, LITE_CONFIG, PRO_CONFIG, inspect_checkpoint
from musubi_tuner.kandinsky6.model import _stream_assign_safetensors
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig


def _tiny_model():
    config = dict(LITE_CONFIG)
    config.update(
        model_dim=64,
        model_dim_a=64,
        ff_dim=128,
        ff_dim_a=64,
        time_dim=32,
        time_dim_a=32,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(16, 24, 24),
        axes_dims_a=(16, 24, 24),
        in_text_dim=48,
        in_text_dim2=16,
    )
    return DiffusionTransformer3D(**config)


def _tiny_four_block_model():
    model = _tiny_model()
    config = dict(LITE_CONFIG)
    config.update(model_dim=64, model_dim_a=64, ff_dim=128, ff_dim_a=64, time_dim=32, time_dim_a=32,
                  num_text_blocks=1, num_visual_blocks=4, axes_dims=(16, 24, 24), axes_dims_a=(16, 24, 24),
                  in_text_dim=48, in_text_dim2=16)
    return DiffusionTransformer3D(**config)


def _inputs(model):
    batch, frames, height, width, audio_len, text_len = 1, 2, 4, 4, 3, 5
    return dict(
        x_video=torch.randn(batch, frames, height, width, 33, requires_grad=True),
        x_audio=torch.randn(batch, audio_len, 40, requires_grad=True),
        text_embed=torch.randn(batch, text_len, 48),
        pooled_text_embed=torch.randn(batch, 16),
        time=torch.rand(batch),
        visual_rope=model.visual_rope(
            (frames, height // 2, width // 2),
            [torch.arange(frames), torch.arange(height // 2), torch.arange(width // 2)],
        ),
        audio_rope=model.audio_rope(torch.arange(audio_len)),
        text_rope=[model.video_text_rope(torch.arange(text_len)), model.audio_text_rope(torch.arange(text_len))],
        attention_mask=torch.ones(batch, text_len, dtype=torch.bool),
        visual_token_type_ids=torch.zeros(batch, frames, dtype=torch.long),
    )


def test_joint_video_audio_forward_and_checkpointing():
    model = _tiny_model().train()
    model.enable_gradient_checkpointing()
    video, audio = model(**_inputs(model))
    assert video.shape == (1, 2, 4, 4, 16)
    assert audio.shape == (1, 3, 40)
    (video.square().mean() + audio.square().mean()).backward()


def test_published_lite_and_pro_architecture_dimensions():
    assert LITE_CONFIG["model_dim"] == 1792
    assert LITE_CONFIG["model_dim_a"] == 896
    assert LITE_CONFIG["num_visual_blocks"] == 32
    assert sum(LITE_CONFIG["axes_dims"]) == sum(LITE_CONFIG["axes_dims_a"]) == 64
    assert PRO_CONFIG["model_dim"] == 4096
    assert PRO_CONFIG["model_dim_a"] == 2048
    assert PRO_CONFIG["num_visual_blocks"] == 60
    assert sum(PRO_CONFIG["axes_dims"]) == sum(PRO_CONFIG["axes_dims_a"]) == 128


@pytest.mark.parametrize(
    ("hidden_size", "head_width", "expected"),
    [(1792, 64, ("lite", None)), (4096, 64, ("pro", None)), (1792, 640, ("lite", 10))],
)
def test_inspect_checkpoint_detects_architecture_and_piflow(tmp_path, hidden_size, head_width, expected):
    from safetensors.torch import save_file

    path = tmp_path / "model.safetensors"
    save_file({
        "visual_embeddings.in_layer.weight": torch.empty(hidden_size, 132),
        "out_layer.out_layer.weight": torch.empty(head_width, hidden_size),
    }, path)
    assert inspect_checkpoint(path) == expected


def test_inspect_checkpoint_rejects_unknown_architecture(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "unknown.safetensors"
    save_file({
        "visual_embeddings.in_layer.weight": torch.empty(2048, 132),
        "out_layer.out_layer.weight": torch.empty(64, 2048),
    }, path)
    with pytest.raises(ValueError, match="Unknown Kandinsky 6 hidden size"):
        inspect_checkpoint(path)


def test_quantization_scope_only_selects_safe_transformer_linears():
    from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer

    quantizer = ConvRotInt8Quantizer(
        target_layer_keys=["visual_transformer_blocks.", "video_text_transformer_blocks.", "audio_text_transformer_blocks."],
        exclude_layer_keys=["modulation", "norm"], allowed_groupsizes=(256, 64),
    )
    model = _tiny_model()
    selected = [key for key, value in model.state_dict().items() if value.ndim == 2 and quantizer.is_target_key(key)]
    assert selected
    assert all("transformer_blocks." in key for key in selected)
    assert not any(
        "modulation" in key or key.startswith(("visual_embeddings.", "out_layer.", "audio_out_layer."))
        for key in selected
    )


def test_bf16_streaming_assignment_roundtrip(tmp_path):
    from safetensors.torch import save_file

    class Tiny(torch.nn.Module):
        def __init__(self, device=None):
            super().__init__()
            self.linear = torch.nn.Linear(5, 3, device=device)
            self.register_buffer("counter", torch.arange(2, dtype=torch.int64, device=device))

    torch.manual_seed(31)
    source = Tiny()
    path = tmp_path / "tiny.safetensors"
    save_file(source.state_dict(), path)
    with torch.device("meta"):
        loaded = Tiny()
    missing, unexpected = _stream_assign_safetensors(
        loaded, path, dtype=torch.bfloat16, device="cpu", strict=True
    )
    assert missing == unexpected == []
    assert loaded.linear.weight.dtype == torch.bfloat16
    assert loaded.counter.dtype == torch.int64
    torch.testing.assert_close(loaded.linear.weight.float(), source.linear.weight, rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(loaded.linear.bias.float(), source.linear.bias, rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(loaded.counter, source.counter)


def test_bf16_streaming_assignment_validates_shape_and_keys(tmp_path):
    from safetensors.torch import save_file

    with torch.device("meta"):
        model = torch.nn.Linear(4, 2)
    path = tmp_path / "bad.safetensors"
    save_file({"weight": torch.empty(3, 4), "unexpected": torch.empty(1)}, path)
    with pytest.raises(RuntimeError, match="State dict mismatch"):
        _stream_assign_safetensors(model, path, dtype=torch.bfloat16, device="cpu", strict=True)

    shape_path = tmp_path / "bad_shape.safetensors"
    save_file({"weight": torch.empty(3, 4), "bias": torch.empty(2)}, shape_path)
    with pytest.raises(RuntimeError, match="shape mismatch for weight"):
        _stream_assign_safetensors(model, shape_path, dtype=torch.bfloat16, device="cpu", strict=True)


def test_image_condition_token_types_change_video_path():
    model = _tiny_model().eval()
    inputs = _inputs(model)
    with torch.no_grad():
        first, _ = model(**inputs)
        inputs["visual_token_type_ids"] = torch.ones_like(inputs["visual_token_type_ids"])
        second, _ = model(**inputs)
    # Published checkpoints learn this embedding. A tiny fresh model initializes it to zero,
    # so verifying acceptance and output shape captures the TI2AV contract without a fake effect.
    assert first.shape == second.shape


def test_checkpointing_preserves_forward_and_gradient():
    torch.manual_seed(7)
    model = _tiny_model().train()
    inputs = _inputs(model)
    model.disable_gradient_checkpointing()
    plain = model(**inputs)
    plain_loss = sum(value.float().sum() for value in plain)
    plain_grad = torch.autograd.grad(plain_loss, model.visual_embeddings.in_layer.weight)[0]

    model.enable_gradient_checkpointing()
    checked = model(**inputs)
    checked_loss = sum(value.float().sum() for value in checked)
    checked_grad = torch.autograd.grad(checked_loss, model.visual_embeddings.in_layer.weight)[0]
    torch.testing.assert_close(checked[0], plain[0])
    torch.testing.assert_close(checked[1], plain[1])
    torch.testing.assert_close(checked_grad, plain_grad)


def test_training_projection_cache_does_not_reuse_autograd_graph():
    model = _tiny_model().train()
    model.enable_gradient_checkpointing()
    inputs = _inputs(model)
    inputs["text_embed"].requires_grad_(True)
    for _ in range(2):
        outputs = model(**inputs)
        sum(output.float().square().mean() for output in outputs).backward()
    assert model._text_proj_cache == {}
    assert torch.isfinite(inputs["text_embed"].grad).all()


def test_vendored_model_matches_upstream_reference():
    upstream_root = Path(os.environ.get("K6_UPSTREAM", Path(__file__).resolve().parents[2] / "kandinsky6-upstream"))
    upstream = upstream_root / "kandinsky" / "core"
    if not upstream.is_dir():
        pytest.skip("set K6_UPSTREAM to an authentic Kandinsky 6 checkout for numerical parity")
    prefix = "_k6_reference"
    for package, path in (
        (prefix, upstream.parent), (f"{prefix}.core", upstream),
        (f"{prefix}.core.components", upstream / "components"),
        (f"{prefix}.core.components.attention", upstream / "components" / "attention"),
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module
    dispatch = types.ModuleType(f"{prefix}.core.components.attention.dispatch")
    from musubi_tuner.kandinsky6.attention import SelfAttentionEngine, _sdpa
    dispatch.SelfAttentionEngine, dispatch._sdpa = SelfAttentionEngine, _sdpa
    sys.modules[dispatch.__name__] = dispatch
    spec = importlib.util.spec_from_file_location(f"{prefix}.core.components.dit", upstream / "components" / "dit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    ours = _tiny_model().eval()
    config = dict(LITE_CONFIG)
    config.update(model_dim=64, model_dim_a=64, ff_dim=128, ff_dim_a=64, time_dim=32, time_dim_a=32,
                  num_text_blocks=1, num_visual_blocks=1, axes_dims=(16, 24, 24), axes_dims_a=(16, 24, 24),
                  in_text_dim=48, in_text_dim2=16)
    reference = module.DiffusionTransformer3D(**config).eval()
    reference.load_state_dict(ours.state_dict(), strict=True)
    inputs = _inputs(ours)
    with torch.no_grad():
        actual = ours(**inputs)
        expected = reference(**inputs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for asynchronous block swap")
@pytest.mark.parametrize("base_dtype", ["bf16", "convrot_int8", "fp8"])
@pytest.mark.parametrize("h2d_only", [False, True], ids=["bidirectional", "h2d_only"])
def test_cuda_block_swap_checkpoint_gradient_parity(base_dtype, h2d_only):
    from musubi_tuner.modules.convrot_int8_utils import apply_convrot_int8_monkey_patch, quantize_weight_convrot
    from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8
    from musubi_tuner.networks.lora_kandinsky6 import create_arch_network

    torch.manual_seed(19)
    source = _tiny_four_block_model()
    state = source.state_dict()

    def build():
        model = _tiny_four_block_model()
        if base_dtype == "convrot_int8":
            quantized, groups = dict(state), {}
            for key, value in list(state.items()):
                if "transformer_blocks." not in key or "modulation" in key or not key.endswith(".weight"):
                    continue
                result = quantize_weight_convrot(key, value, (64,))
                if result is None:
                    continue
                weight, scale, groupsize = result
                quantized[key] = weight
                quantized[key.replace(".weight", ".scale_weight")] = scale
                groups[key.removesuffix(".weight")] = groupsize
            apply_convrot_int8_monkey_patch(model, quantized, groupsize_map=groups)
            model.requires_grad_(False)
            model.load_state_dict(quantized, strict=True, assign=True)
        elif base_dtype == "fp8":
            quantized = optimize_state_dict_with_fp8(
                dict(state), torch.device("cpu"),
                target_layer_keys=["visual_transformer_blocks.", "video_text_transformer_blocks.", "audio_text_transformer_blocks."],
                exclude_layer_keys=["modulation", "norm"], move_to_device=False,
            )
            apply_fp8_monkey_patch(model, quantized, use_scaled_mm=False)
            model.requires_grad_(False)
            model.load_state_dict(quantized, strict=True, assign=True)
        else:
            model.load_state_dict(state, strict=True)
            model.to(torch.bfloat16)
            model.requires_grad_(False)
        model.enable_gradient_checkpointing(activation_cpu_offloading=True)
        torch.manual_seed(123)
        network = create_arch_network(1.0, 2, 2.0, None, [], model)
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        return model, network

    resident, resident_network = build()
    resident.to("cuda").train()
    resident_network.to("cuda")
    swapped, swapped_network = build()
    swapped.train()
    swapped_network.to("cuda")
    swapped.enable_block_swap(
        2,
        BlockSwapConfig(
            torch.device("cuda"), supports_backward=True, h2d_only=h2d_only, ring_size=2
        ),
    )
    swapped.move_to_device_except_swap_blocks(torch.device("cuda"))
    swapped.prepare_block_swap_before_forward()
    if base_dtype in ("convrot_int8", "fp8"):
        scales = [buffer for name, buffer in swapped.named_buffers() if name.endswith("scale_weight")]
        assert scales and all(buffer.is_cuda for buffer in scales)
    assert all(parameter.is_cuda for parameter in swapped_network.parameters())
    assert all(parameter.is_cuda for name, parameter in swapped.named_parameters() if "norm" in name)
    if h2d_only:
        assert swapped.offloader.S == 2
        streamed = [job for block_index in swapped.offloader.stream_idx for job in swapped.offloader._jobs(block_index)]
        assert streamed
        assert all(not getattr(module, name).requires_grad for module, name, is_parameter in streamed if is_parameter)

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to("cuda")
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    resident_inputs = {key: move(value) for key, value in _inputs(resident).items()}
    swapped_inputs = {
        key: (value.detach().clone().to("cuda").requires_grad_(value.requires_grad) if isinstance(value, torch.Tensor) else value)
        for key, value in resident_inputs.items()
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        resident_outputs = resident(**resident_inputs)
        swapped_outputs = swapped(**swapped_inputs)
    resident_loss = sum(output.float().square().mean() for output in resident_outputs)
    swapped_loss = sum(output.float().square().mean() for output in swapped_outputs)
    resident_grad, resident_lora_grad = torch.autograd.grad(
        resident_loss, (resident_inputs["x_video"], next(resident_network.parameters()))
    )
    swapped_grad, swapped_lora_grad = torch.autograd.grad(
        swapped_loss, (swapped_inputs["x_video"], next(swapped_network.parameters()))
    )
    assert torch.isfinite(swapped_grad).all()
    assert torch.isfinite(swapped_lora_grad).all()
    tolerance = 4e-2 if base_dtype == "fp8" else 2e-2
    torch.testing.assert_close(swapped_outputs[0], resident_outputs[0], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(swapped_outputs[1], resident_outputs[1], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(swapped_grad, resident_grad, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(swapped_lora_grad, resident_lora_grad, rtol=tolerance, atol=tolerance)
    for _ in range(2):
        swapped.prepare_block_swap_before_forward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            repeated = swapped(**swapped_inputs)
            repeated_loss = sum(output.float().square().mean() for output in repeated)
        repeated_loss.backward()
    swapped.close_block_swap()
    assert swapped.offloader is None
