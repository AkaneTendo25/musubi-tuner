from types import SimpleNamespace
import importlib
from contextlib import nullcontext

import pytest
import torch

from musubi_tuner.dataset.cache_io import save_latent_cache_kandinsky6
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.kandinsky6_train_network import (
    Kandinsky6NetworkTrainer,
    build_ti2av_video_input,
    kandinsky6_setup_parser,
    pad_text_batch,
    park_dit_for_sampling,
    restore_dit_after_sampling_park,
    sample_canvas_size,
    sample_latent_frame_count,
    shifted_flow_sigma,
)
from musubi_tuner.training.parser_common import setup_parser_common


def test_shifted_flow_sigma_matches_schedule_endpoints_and_formula():
    uniform = torch.tensor([0.0, 0.25, 1.0])
    actual = shifted_flow_sigma(uniform, 3.0)
    assert torch.allclose(actual, torch.tensor([0.0, 0.5, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        shifted_flow_sigma(uniform, 0.0)


def test_sample_frame_count_uses_musubi_pixel_frames_and_vae_stride():
    assert sample_latent_frame_count(121) == (121, 31)
    assert sample_latent_frame_count(120) == (117, 30)
    with pytest.raises(ValueError, match="positive"):
        sample_latent_frame_count(0)


def test_sample_canvas_aligns_and_rejects_empty_dimensions():
    assert sample_canvas_size(130, 257) == (128, 256)
    with pytest.raises(ValueError, match="at least 16"):
        sample_canvas_size(15, 128)


def test_sampling_park_preserves_offloader_managed_blocks_across_repeated_samples(monkeypatch):
    class Copier:
        def __init__(self):
            self.syncs = 0

        def sync(self):
            self.syncs += 1

    class SwapManagedDiT:
        def __init__(self):
            self.offloader = SimpleNamespace(copier=Copier())
            self.non_block_device = torch.device("cuda")
            self.block_modulation_devices = [torch.device("cuda"), torch.device("cpu")]
            self.full_moves = []
            self.prepares = 0

        def to(self, device):
            self.full_moves.append(torch.device(device))
            self.non_block_device = torch.device(device)
            self.block_modulation_devices = [torch.device(device)] * 2

        def move_to_device_except_swap_blocks(self, device):
            self.non_block_device = torch.device(device)

        def prepare_block_swap_before_forward(self):
            self.prepares += 1
            assert self.block_modulation_devices[0].type == "cuda"

    monkeypatch.setattr("musubi_tuner.kandinsky6_train_network.clean_memory_on_device", lambda _device: None)
    dit = SwapManagedDiT()
    for _ in range(2):
        park_dit_for_sampling(dit, torch.device("cuda"))
        assert dit.non_block_device.type == "cpu"
        assert dit.block_modulation_devices[0].type == "cuda"
        restore_dit_after_sampling_park(dit, torch.device("cuda"))
        assert dit.non_block_device.type == "cuda"
    assert dit.full_moves == []
    assert dit.offloader.copier.syncs == 2
    assert dit.prepares == 2


def test_kandinsky6_bucket_uses_vae_patch_alignment():
    selector = BucketSelector((512, 512), architecture="k6")
    assert selector.reso_steps == 16


def test_kandinsky6_parser_supplies_complete_base_trainer_precision_and_attention_contract():
    args = kandinsky6_setup_parser(setup_parser_common()).parse_args([])
    assert args.mixed_precision == "bf16"
    assert args.sdpa is True
    assert args.fp8_scaled is False
    assert args.disable_numpy_memmap is False
    args.dit_dtype = "bfloat16"
    Kandinsky6NetworkTrainer().handle_model_specific_args(args)


def test_deprecated_checkout_flags_parse_without_affecting_bundled_runtime():
    parser = kandinsky6_setup_parser(setup_parser_common())
    args = parser.parse_args(["--upstream", "missing-upstream", "--sr_upstream", "missing-sr"])
    assert args.upstream == "missing-upstream"
    assert args.sr_upstream == "missing-sr"
    help_text = parser.format_help()
    assert "--upstream" not in help_text
    assert "--sr_upstream" not in help_text


@pytest.mark.parametrize(
    "module",
    [
        "musubi_tuner.kandinsky6.runtime.core.components.text_embedder",
        "musubi_tuner.kandinsky6.runtime.core.components.vae_audio",
        "musubi_tuner.kandinsky6.runtime.core.components.vae_video",
        "musubi_tuner.kandinsky6.runtime.core.algo.denoise_loop",
        "musubi_tuner.kandinsky6.runtime.core.algo.mux",
    ],
)
def test_sampling_deferred_imports_resolve_from_bundled_runtime(module):
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize("quant_flag", ["convrot_int8", "fp8_base"])
def test_quantized_training_rejects_base_weight_merge(quant_flag):
    args = kandinsky6_setup_parser(setup_parser_common()).parse_args([])
    args.dit_dtype = "bfloat16"
    args.base_weights = ["adapter.safetensors"]
    setattr(args, quant_flag, True)
    with pytest.raises(ValueError, match="cannot be merged"):
        Kandinsky6NetworkTrainer().handle_model_specific_args(args)


def test_kandinsky6_rejects_irrelevant_input_offload_flag():
    args = kandinsky6_setup_parser(setup_parser_common()).parse_args([])
    args.dit_dtype = "bfloat16"
    args.img_in_txt_in_offloading = True
    with pytest.raises(ValueError, match="not supported"):
        Kandinsky6NetworkTrainer().handle_model_specific_args(args)


def test_compile_uses_real_multimodal_block_containers(monkeypatch):
    trainer = Kandinsky6NetworkTrainer()
    trainer.blocks_to_swap = 0
    transformer = SimpleNamespace(
        video_text_transformer_blocks=object(),
        audio_text_transformer_blocks=object(),
        visual_transformer_blocks=object(),
    )
    seen = {}

    def fake_compile(args, model, containers, disable_linear):
        seen.update(containers=containers, disable_linear=disable_linear)
        return model

    monkeypatch.setattr("musubi_tuner.kandinsky6_train_network.model_utils.compile_transformer", fake_compile)
    args = SimpleNamespace(convrot_int8=False, fp8_base=False)
    assert trainer.compile_transformer(args, transformer) is transformer
    assert seen["containers"] == [
        transformer.video_text_transformer_blocks,
        transformer.audio_text_transformer_blocks,
        transformer.visual_transformer_blocks,
    ]


def test_sampling_rejects_nf4_qwen_before_loading_resources():
    trainer = Kandinsky6NetworkTrainer()
    args = SimpleNamespace(sample_prompts="prompts.txt", quantized_qwen=True)
    with pytest.raises(ValueError, match="bitsandbytes NF4 cannot be parked on CPU"):
        trainer.prepare_sampling(args, SimpleNamespace(device=torch.device("cuda")), torch.float16)


def test_pad_text_batch_preserves_valid_rows():
    rows = [torch.ones(2, 3), torch.full((1, 3), 2.0)]
    masks = [torch.tensor([True, True]), torch.tensor([True])]
    padded, mask = pad_text_batch(rows, masks, torch.device("cpu"), torch.float32)
    assert padded.shape == (2, 2, 3)
    assert torch.equal(mask, torch.tensor([[True, True], [True, False]]))
    assert torch.equal(padded[1, 1], torch.zeros(3))


def test_ti2av_input_appends_clean_reference_and_masks_only_reference():
    noisy = torch.randn(2, 3, 4, 5, 16)
    reference = torch.randn(2, 1, 4, 5, 16)
    model_input, token_types = build_ti2av_video_input(noisy, reference, visual_cond=True)
    assert model_input.shape == (2, 4, 4, 5, 33)
    assert torch.equal(model_input[:, -1, ..., :16], reference[:, 0])
    assert torch.all(model_input[:, :-1, ..., -1] == 0)
    assert torch.all(model_input[:, -1, ..., -1] == 1)
    assert torch.equal(token_types, torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]]))


def test_joint_loss_uses_video_and_only_present_audio_with_user_weight():
    trainer = Kandinsky6NetworkTrainer()
    output = SimpleNamespace(
        pred=torch.ones(2, 1),
        target=torch.zeros(2, 1),
        extra={
            "audio_pred": torch.tensor([[[2.0]], [[10.0]]]),
            "audio_target": torch.zeros(2, 1, 1),
            "audio_loss_weights": torch.tensor([0.5, 0.0]),
        },
    )
    loss, logs = trainer.compute_loss(None, output, None, None, None, torch.float32, 0)
    assert loss.item() == pytest.approx(3.0)  # video 1 + (audio MSE 4 * 0.5)
    assert logs == {"loss/video": 1.0, "loss/audio": 2.0}


def test_process_batch_converts_cache_layout_and_shares_sigma(monkeypatch):
    trainer = Kandinsky6NetworkTrainer()
    trainer._scheduler_scale = 1.0
    seen = {}

    def fake_call(args, accelerator, transformer, video, batch, video_noise, noisy_video, timesteps, dtype, **kwargs):
        seen.update(video=video, audio=kwargs["audio_latents"], image=kwargs["image_latent"], timesteps=timesteps)
        return SimpleNamespace(
            pred=video_noise,
            target=video_noise - video,
            extra={
                "audio_pred": kwargs["audio_noise"],
                "audio_target": kwargs["audio_noise"] - kwargs["audio_latents"],
                "audio_loss_weights": kwargs["audio_loss_weights"],
            },
        )

    monkeypatch.setattr(trainer, "call_dit", fake_call)
    batch = {
        "latents_audio": torch.zeros(1, 40, 6),
        "audio_present": torch.ones(1),
        "text_embeds": [torch.zeros(2, 3584)],
        "attention_mask": [torch.ones(2, dtype=torch.bool)],
        "pooled_embed": torch.zeros(1, 768),
        "timesteps": torch.tensor([0.25]),
    }
    args = SimpleNamespace(video_only=False, audio_loss_weight=1.0)
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    latents = torch.zeros(1, 16, 3, 4, 5)
    noise = torch.ones_like(latents)
    loss, _ = trainer.process_batch(
        args,
        accelerator,
        object(),
        None,
        batch,
        latents,
        noise,
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    assert seen["video"].shape == (1, 3, 4, 5, 16)
    assert seen["audio"].shape == (1, 6, 40)
    assert seen["image"] is None
    assert seen["timesteps"].item() == pytest.approx(250.0)
    assert loss.item() == pytest.approx(0.0)


def test_kandinsky6_cache_rejects_wrong_layout_before_writing(tmp_path):
    item = SimpleNamespace(item_key="x", latent_cache_path=tmp_path / "x.safetensors")
    with pytest.raises(ValueError, match=r"\[C,T,H,W\]"):
        save_latent_cache_kandinsky6(item, torch.zeros(1, 2, 3), torch.zeros(40, 4), True)


@pytest.mark.parametrize("with_reference", [False, True])
def test_training_tail_reference_reuses_upstream_frame_zero_rope(with_reference):
    seen = {}

    class RecordingDiT:
        patch_size = (1, 2, 2)
        visual_cond = True

        @staticmethod
        def visual_rope(shape, positions, scale):
            return positions[0].reshape(-1, 1, 1, 1).expand(*shape, 1)

        @staticmethod
        def audio_rope(positions):
            return positions

        video_text_rope = audio_rope
        audio_text_rope = audio_rope

        def __call__(self, **kwargs):
            seen.update(kwargs)
            return kwargs["x_video"][..., :16], kwargs["x_audio"]

    video = torch.zeros(1, 3, 4, 4, 16)
    audio = torch.zeros(1, 6, 40)
    image = torch.ones(1, 1, 4, 4, 16) if with_reference else None
    output = Kandinsky6NetworkTrainer().call_dit(
        SimpleNamespace(gradient_checkpointing=False),
        SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext),
        RecordingDiT(),
        video,
        {},
        torch.ones_like(video),
        video,
        torch.tensor([250.0]),
        torch.float32,
        audio_latents=audio,
        audio_noise=audio,
        noisy_audio=audio,
        text_embeds=torch.zeros(1, 2, 48),
        pooled_embed=torch.zeros(1, 16),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        image_latent=image,
        audio_loss_weights=torch.ones(1),
    )
    assert seen["visual_rope"][:, 0, 0, 0].tolist() == ([0, 1, 2, 0] if with_reference else [0, 1, 2])
    assert output.pred.shape == video.shape
    if with_reference:
        assert seen["visual_token_type_ids"].tolist() == [[0, 0, 0, 1]]
        assert torch.equal(seen["x_video"][:, -1, ..., :16], image[:, 0])
