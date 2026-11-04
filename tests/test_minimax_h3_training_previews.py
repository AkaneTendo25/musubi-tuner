from contextlib import nullcontext
import os
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from musubi_tuner import minimax_h3_train_network as training
from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.inference import H3EncodedReferences
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.packing import MiniMaxH3ReferenceGeometry
from musubi_tuner.minimax_h3.one_frame import route_keeps_dit_rows, route_presents_qwen_images
from musubi_tuner.minimax_h3_train_learned_context import H3LearnedContext, MiniMaxH3LearnedContextTrainer
from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
from musubi_tuner.minimax_h3.cache import H3_CONDITIONING_TASK_IDS, H3_CONDITIONING_TASK_KEY, H3_KEYFRAME_VIDEO_ROWS_KEY


DEVICE = torch.device(os.environ.get("H3_TEST_DEVICE", "cpu"))


def tiny_model():
    return MiniMaxH3Transformer(
        MiniMaxH3TransformerConfig(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=32,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=64,
            in_channels=24,
            audio_in_channels=32,
            patch_size=(1, 2, 2),
            text_dim=5120,
            freq_dim=8,
            time_embed_hidden_dim=32,
            time_embed_dim=16,
            rope_freq_dim=2,
        )
    ).to(DEVICE)


class Decoder(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.shapes = []
        self.fail = fail

    def decode(self, latent):
        self.shapes.append(tuple(latent.shape))
        if self.fail:
            raise RuntimeError("decode failed")
        return latent[:, :3].sigmoid()


@pytest.mark.parametrize("mode", ["fl2va", "ref2va"])
@pytest.mark.parametrize("route", ["dual", "qwen_image_only", "dit_latent_only", "text_only"])
@pytest.mark.parametrize("composition", ["prepend", "replace"])
def test_training_hook_runs_real_tiny_model_for_all_custom_routes(tmp_path, monkeypatch, mode, route, composition):
    image = tmp_path / "control.png"
    Image.new("RGB", (32, 32)).save(image)
    prompts = tmp_path / "prompts.txt"
    media = f"--ci {image}" if mode == "fl2va" else f"--ref {image}"
    placement = "target_indices=-2;2,control_index=-5" if mode == "fl2va" else "target_indices=-2;2"
    prompts.write_text(f"caption --w 32 --h 32 --f 1 --s 3 --d 17 --of {placement} {media}\n")
    args = training.create_parser().parse_args(
        [
            "--sdpa",
            "--one_frame",
            "--h3_training_mode",
            mode,
            "--h3_reference_route",
            route,
            "--h3_target_noise_coupling",
            "shared",
            "--text_encoder",
            "external-qwen",
            "--vae",
            "external-vae",
            "--audio_vae",
            "external-audio",
            "--output_dir",
            str(tmp_path),
            "--sample_prompts",
            str(prompts),
            "--sample_at_first",
        ]
    )
    encoder_calls = []

    class Encoder:
        def encode_prompt(self, text, images=()):
            encoder_calls.append(len(images))
            return {
                H3_TEXT_HIDDEN_KEY: torch.zeros(3, 5120),
                H3_TEXT_TOKEN_TAGS_KEY: torch.tensor([1, 1, 0] if images else [1, 1, 1]),
            }

        def encode_reference_prompt(self, text, references):
            return self.encode_prompt(text, references)

        def close(self):
            pass

    monkeypatch.setattr(training, "create_conditioning_encoder", lambda **kwargs: Encoder())
    monkeypatch.setattr(training, "encode_keyframe_images", lambda path, images, device: [torch.zeros(1, 96) for _ in images])
    monkeypatch.setattr(training, "prepare_references", lambda *a: (object(),))
    monkeypatch.setattr(
        training,
        "encode_reference_media",
        lambda *a: H3EncodedReferences(
            (MiniMaxH3ReferenceGeometry(kind=0, num_latent_frames=1, latent_height=2, latent_width=2),),
            torch.zeros(1, 96),
            torch.empty(0, 32),
        ),
    )
    trainer = MiniMaxH3LearnedContextTrainer()
    trainer.blocks_to_swap = 0
    trainer._learned_context = H3LearnedContext(torch.full((2, 5120), 0.25))
    trainer._learned_context_composition = composition
    accelerator = SimpleNamespace(device=DEVICE, unwrap_model=lambda model: model, autocast=nullcontext)
    prepared = trainer.process_sample_prompts(args, accelerator, str(prompts))
    assert encoder_calls == [int(route_presents_qwen_images(route))]
    assert prepared[0]["one_frame_target_indices"] == (-2, 2)
    # An optimizer update after prompt preparation must reach the preview.
    with torch.no_grad():
        trainer._learned_context.weight.fill_(0.75)
    model = tiny_model().train()
    # Preserve mixed child states as well as the top-level training flag.
    model.blocks[0].eval()
    before_states = [module.training for module in model.modules()]
    calls = []

    def observe(module, args, kwargs):
        assert not module.training
        assert not torch.is_grad_enabled()
        calls.append({key: value.detach().clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)})

    model.register_forward_pre_hook(observe, with_kwargs=True)
    decoder = Decoder()
    cuda_rng = torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None
    rng = torch.get_rng_state().clone()
    trainer.sample_images(accelerator, args, None, 0, SimpleNamespace(video_decoder=decoder), model, prepared, torch.float32)
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert all(torch.equal(before, after) for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert [module.training for module in model.modules()] == before_states
    assert len(calls) == 2
    first = calls[0]
    torch.testing.assert_close(first["encoder_hidden_states"][0, :2], torch.full((2, 5120), 0.75, device=DEVICE))
    expected_tags = [1, 1, 0] if route_presents_qwen_images(route) else [1, 1, 1]
    if composition == "prepend":
        expected_tags = [1, 1] + expected_tags
    else:
        expected_tags = [1, 1] + [tag for tag in expected_tags if tag != 1]
    actual_tags = first["token_tags"][first["text_indices"]]
    assert actual_tags.tolist() == expected_tags
    target_rows = first["video_indices"][-2:]
    target_times = first["position_ids"][target_rows, 0]
    # Negative controls kept out of the DiT still shift all slot coordinates.
    origin = len(expected_tags) + (1 if mode == "ref2va" else 0)
    offset = 0 if mode == "ref2va" else 3
    expected_times = torch.tensor([origin + offset * 5 / 3, origin + (offset + 4) * 5 / 3], dtype=target_times.dtype, device=DEVICE)
    torch.testing.assert_close(target_times, expected_times)
    # Shared noise is visible at the actual model call, after packing.
    rows = first["video_hidden_states"]
    torch.testing.assert_close(rows[:, -2], rows[:, -1])
    assert len(decoder.shapes) == 2 and all(shape[2] == 1 for shape in decoder.shapes)
    outputs = sorted((tmp_path / "sample").glob("*.png"))
    assert len(outputs) == 2
    assert "index-2" in outputs[0].name and "index2" in outputs[1].name
    assert first["video_indices"].numel() == 2 + int(route_keeps_dit_rows(route))
    for output in outputs:
        metadata = json.loads(output.with_suffix(".json").read_text())
        assert metadata["learned_context_composition"] == composition
        assert metadata["learned_context_tokens"] == 2
        assert metadata["reference_route"] == route
        assert metadata["target_noise_coupling"] == "shared"
        assert metadata["target_indices"] == [-2, 2]
        assert metadata["model_evaluations"] == len(calls)


def test_training_hook_restores_rng_and_training_state_on_decode_failure(tmp_path):
    trainer = training.MiniMaxH3NetworkTrainer()
    trainer.blocks_to_swap = 0
    model = tiny_model().train()
    model.blocks[0].eval()
    states = [module.training for module in model.modules()]
    args = training.create_parser().parse_args(["--sdpa", "--output_dir", str(tmp_path), "--sample_at_first"])
    accelerator = SimpleNamespace(device=DEVICE, unwrap_model=lambda model: model, autocast=nullcontext)
    parameter = {
        H3_TEXT_HIDDEN_KEY: torch.zeros(2, 5120),
        H3_TEXT_TOKEN_TAGS_KEY: torch.ones(2, dtype=torch.long),
        "frame_count": 1,
        "height": 32,
        "width": 32,
        "sample_steps": 2,
    }
    cuda_rng = torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="decode failed"):
        trainer.sample_images(
            accelerator, args, None, 0, SimpleNamespace(video_decoder=Decoder(fail=True)), model, [parameter], torch.float32
        )
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert all(torch.equal(before, after) for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert [module.training for module in model.modules()] == states


@pytest.mark.parametrize("composition", ["prepend", "replace"])
def test_learned_context_actual_multimodal_training_backward(composition):
    model = tiny_model().train()
    trainer = MiniMaxH3LearnedContextTrainer()
    trainer._learned_context = H3LearnedContext(torch.randn(2, 5120)).to(DEVICE)
    trainer._learned_context_composition = composition
    trainer._learned_context.apply_to(None, model)
    trainer.dit_dtype = torch.float32
    trainer.backend = _NativeTrainingBackend(model, mode="fl2va")
    accelerator = SimpleNamespace(device=DEVICE, autocast=nullcontext)
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(3, 5120)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor([1, 0, 1])],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["i2va"])],
        H3_KEYFRAME_VIDEO_ROWS_KEY: [torch.randn(2, 96)],
    }
    inputs = SimpleNamespace(
        video=torch.randn(1, 24, 2, 2, 2, device=DEVICE),
        audio=torch.randn(1, 2, 32, 1, device=DEVICE),
        video_timestep=torch.tensor([0.4], device=DEVICE),
        audio_timestep=torch.tensor([0.7], device=DEVICE),
    )
    prediction = trainer._predict(accelerator, model, batch, inputs, conditioning="prompt", role="student")
    loss = prediction.video.square().mean() + prediction.audio.square().mean()
    loss.backward()
    gradient = trainer._learned_context.weight.grad
    assert torch.isfinite(loss) and gradient is not None
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
