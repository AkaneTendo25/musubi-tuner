from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from musubi_tuner.minimax_h3 import integration
from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.inference import H3EncodedReferences
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.one_frame import H3_REFERENCE_ROUTES, target_slot_output_paths
from musubi_tuner.minimax_h3.packing import MiniMaxH3ReferenceGeometry
from musubi_tuner.minimax_h3.request import H3GenerationRequest, H3Reference, ReferenceKind
import musubi_tuner.minimax_h3_generate_video as cli


@pytest.fixture
def native_assets(monkeypatch):
    # Only the unavailable checkpoint assets and VAE are substituted. Requests,
    # preparation, context composition, layouts, solver and PNG writes stay real.
    model = (
        MiniMaxH3Transformer(
            MiniMaxH3TransformerConfig(
                num_attention_heads=1,
                attention_head_dim=8,
                hidden_size=8,
                num_layers=1,
                num_refiner_layers=1,
                ffn_dim=16,
                in_channels=24,
                audio_in_channels=32,
                patch_size=(1, 2, 2),
                text_dim=5120,
                freq_dim=8,
                time_embed_hidden_dim=8,
                time_embed_dim=8,
                rope_freq_dim=1,
            )
        )
        .to(os.environ.get("H3_TEST_DEVICE", "cpu"))
        .eval()
    )
    calls, presentations, decoded, noise = [], [], [], []

    def observe(module, args, kwargs):
        calls.append(
            {
                key: kwargs[key].detach().cpu().clone()
                for key in ("position_ids", "video_indices", "encoder_hidden_states", "token_tags")
            }
        )

    model.register_forward_pre_hook(observe, with_kwargs=True)
    monkeypatch.setattr(integration._NativeGenerator, "_load_transformer", lambda self: (model, []))

    class Encoder:
        def encode_prompt(self, prompt, images=()):
            presentations.append(len(images))
            return {
                H3_TEXT_HIDDEN_KEY: torch.ones(3, 5120),
                H3_TEXT_TOKEN_TAGS_KEY: torch.tensor([0, 1, 1]) if images else torch.ones(3, dtype=torch.long),
            }

        def encode_reference_prompt(self, prompt, references):
            return self.encode_prompt(prompt, references)

        def close(self):
            pass

    monkeypatch.setattr(integration, "create_conditioning_encoder", lambda **kwargs: Encoder())

    def keyframes(path, images, device):
        return tuple(torch.zeros((image.height // 32) * (image.width // 32), 96) for image in images)

    monkeypatch.setattr(integration, "encode_keyframe_images", keyframes)

    def references(video_vae, audio_vae, prepared, device):
        geometries = tuple(MiniMaxH3ReferenceGeometry(0, 1, 2, 2) for _ in prepared)
        return H3EncodedReferences(geometries, torch.zeros(len(prepared), 96), torch.empty(0, 32))

    monkeypatch.setattr(integration, "encode_reference_media", references)

    class Decoder:
        def to(self, device):
            return self

        def eval(self):
            return self

        def decode(self, latent):
            assert torch.isfinite(latent).all()
            decoded.append(latent.detach().clone())
            return latent[:, :3].sigmoid()

    monkeypatch.setattr(integration, "load_video_vae_decoder", lambda *args: Decoder())
    from musubi_tuner.minimax_h3 import inference

    original = inference.couple_target_slot_noise

    def couple(value, coupling):
        result = original(value, coupling)
        noise.append(result.cpu().clone())
        return result

    monkeypatch.setattr(inference, "couple_target_slot_noise", couple)
    return calls, presentations, decoded, noise


def make_generator(request, two_pass=False):
    generator = integration.create_generator(
        model=Path("model"),
        text_encoder=Path("text"),
        tokenizer=Path("tokenizer"),
        video_vae=Path("vae"),
        audio_vae=None,
        device=os.environ.get("H3_TEST_DEVICE", "cpu"),
        dtype="bfloat16",
        request=request,
        height=64,
        width=64,
        num_inference_steps=4,
        first_pass_scale=0.5 if two_pass else 0,
        first_pass_steps=3,
        second_pass_strength=0.5,
        reference_image_short_edge=32,
    )
    generator.learned_context = torch.full((2, 5120), 0.25)
    return generator


@pytest.mark.parametrize("composition", ["prepend", "replace"])
@pytest.mark.parametrize("route", H3_REFERENCE_ROUTES)
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("two_pass", [False, True])
def test_real_native_custom_generation(native_assets, tmp_path, route, reference, two_pass, composition):
    calls, presentations, decoded, noise = native_assets
    control = tmp_path / "control.png"
    Image.new("RGB", (32, 32), "red").save(control)
    request = H3GenerationRequest(
        "custom",
        tmp_path / "out.png",
        frame_count_override=1,
        one_frame_target_indices=(-1, 1),
        target_noise_coupling="shared",
        reference_route=route,
        **(
            {"references": (H3Reference(control, ReferenceKind.IMAGE),)}
            if reference
            else {"condition_images": (control,), "one_frame_control_indices": (-3,)}
        ),
    )
    generator = make_generator(request, two_pass)
    generator.learned_context_composition = composition
    generator.generate(request)
    expected_calls = 4 if two_pass else 3
    assert len(calls) == expected_calls
    assert presentations == [1 if route in ("dual", "qwen_image_only") else 0]
    assert len(noise) == (2 if two_pass else 1)
    for value in noise:
        torch.testing.assert_close(value[:, :, 0], value[:, :, 1], rtol=0, atol=0)
    assert len(decoded) == 2 and all(value.shape[2] == 1 for value in decoded)
    for slot, path in enumerate(target_slot_output_paths(request.output, (-1, 1))):
        with Image.open(path) as image:
            assert image.size == (4, 4)
        metadata = json.loads(path.with_suffix(".png.json").read_text())
        assert metadata["model_evaluations"] == expected_calls
        assert metadata["target_index"] == (-1, 1)[slot]
        assert metadata["reference_route"] == route
        assert metadata["one_frame_control_indices"] == (None if reference else [-3])
    for call_index, call in enumerate(calls):
        hidden = call["encoder_hidden_states"][0]
        presented = route in ("dual", "qwen_image_only")
        text_rows = 5 if composition == "prepend" else (3 if presented else 2)
        assert hidden.shape == (text_rows, 5120)
        expected_tags = (
            ([1, 1, 0] if presented else [1, 1]) if composition == "replace" else ([1, 1, 0, 1, 1] if presented else [1] * 5)
        )
        assert call["token_tags"][:text_rows].tolist() == expected_tags
        torch.testing.assert_close(hidden[:2], torch.full((2, 5120), 0.25))
        # The last two video frames are the two generated slots. Ghost controls
        # keep negative origins and omitted references reserve the same span.
        rows_per_slot = 1 if two_pass and call_index < 2 else 4
        target_rows = call["video_indices"][-2 * rows_per_slot :]
        positions = call["position_ids"][target_rows, 0]
        assert positions[0].item() == pytest.approx(text_rows + (1 if reference else 2 * 5 / 3))
        assert positions[-1].item() - positions[0].item() == pytest.approx(2 * 5 / 3)


@pytest.mark.parametrize("from_file", [False, True])
def test_real_cli_custom_generation(native_assets, monkeypatch, tmp_path, from_file):
    control = tmp_path / "control.png"
    Image.new("RGB", (32, 32)).save(control)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(f"caption --f 1 --ci {control.as_posix()} --of target_indices=-1;1,control_index=-3 --o slots.png\n")
    monkeypatch.setattr(cli, "create_generator", lambda **kwargs: make_generator(kwargs["request"]))
    output = tmp_path / "outputs"
    monkeypatch.setattr(cli, "inspect_checkpoint", lambda path: None)
    mode_args = (
        ["--from_file", str(prompts), "--output", str(output)]
        if from_file
        else [
            "--prompt",
            "caption",
            "--frame_count",
            "1",
            "--condition_image",
            str(control),
            "--one_frame",
            "target_indices=-1;1,control_index=-3",
            "--output",
            str(output / "slots.png"),
        ]
    )
    cli.main(
        [
            "--model",
            "model",
            "--text_encoder",
            "text",
            "--vae",
            "vae",
            *mode_args,
            "--h3_target_noise_coupling",
            "shared",
            "--h3_reference_route",
            "text_only",
        ]
    )
    assert (output / "slots_000_index-1.png").exists()
    assert (output / "slots_001_index1.png.json").exists()


@pytest.mark.parametrize("route", H3_REFERENCE_ROUTES)
def test_real_native_scalar_image_metadata(native_assets, tmp_path, route):
    control = tmp_path / "control.png"
    Image.new("RGB", (32, 32)).save(control)
    request = H3GenerationRequest(
        "still",
        tmp_path / "still.png",
        frame_count_override=1,
        one_frame_target_index=24,
        condition_images=(control, control),
        one_frame_control_indices=(0, 48),
        reference_route=route,
    )
    make_generator(request).generate(request)
    assert request.output.exists()
    metadata = json.loads(request.output.with_suffix(".png.json").read_text())
    assert metadata["model_evaluations"] == 3
    assert metadata["one_frame_target_index"] == 24
    assert metadata["one_frame_target_indices"] is None
    for call in native_assets[0]:
        target_rows = call["video_indices"][-4:]
        assert torch.all(call["position_ids"][target_rows, 0] == 45)
