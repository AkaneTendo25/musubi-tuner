from __future__ import annotations

from pathlib import Path

import av
import torch

from musubi_tuner.minimax_h3.audio_vae import MiniMaxH3AudioBigVGANDecoder
from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.inference import (
    H3GeneratedMedia,
    data_ward_euler_step,
    decode_latents_sequentially,
    denoise_t2va,
    resolve_canvas_size,
    save_av_mp4,
    shifted_flow_schedule,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.video_vae import MiniMaxH3VideoViTDecoder3d


def _tiny_transformer() -> MiniMaxH3Transformer:
    return MiniMaxH3Transformer(
        MiniMaxH3TransformerConfig(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=32,
            num_layers=1,
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
    ).eval()


def test_released_canvas_resolution() -> None:
    assert resolve_canvas_size("16:9") == (768, 1344)
    assert resolve_canvas_size("1:1") == (768, 768)
    assert resolve_canvas_size("9:16") == (1344, 768)


def test_shifted_schedule_has_terminal_zero_and_one_fewer_model_evaluations() -> None:
    sigmas, timesteps = shifted_flow_schedule(20, 12.0, torch.device("cpu"))

    assert sigmas.shape == (20,)
    assert timesteps.shape == (19,)
    assert sigmas[0].item() == 1.0
    assert sigmas[-1].item() == 0.0
    assert timesteps[0].item() == 0.0


def test_data_ward_euler_full_step_recovers_clean_sample() -> None:
    clean = torch.randn(2, 3)
    noise = torch.randn_like(clean)
    prediction = clean - noise

    result = data_ward_euler_step(
        noise,
        prediction,
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor(0.0),
    )

    torch.testing.assert_close(result, clean)


def test_tiny_joint_denoising_is_finite_and_shape_correct() -> None:
    transformer = _tiny_transformer()
    conditioning = {
        H3_TEXT_HIDDEN_KEY: torch.randn(3, 12),
        H3_TEXT_TOKEN_TAGS_KEY: torch.ones(3, dtype=torch.long),
    }

    video, audio = denoise_t2va(
        transformer,
        conditioning,
        height=32,
        width=32,
        frame_count=5,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
        show_progress=False,
    )

    assert video.shape == (1, 4, 2, 2, 2)
    assert audio.shape == (1, 2, 8, 8)
    assert torch.isfinite(video).all()
    assert torch.isfinite(audio).all()


def test_tiny_video_decoder_restores_patch_geometry() -> None:
    decoder = MiniMaxH3VideoViTDecoder3d(
        in_channels=4,
        patch_size=2,
        patch_size_t=1,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=6,
        num_register_tokens=1,
        ffn_mult=2,
        rope_dim_ratio=1.0,
    )

    result = decoder(torch.randn(1, 4, 2, 2, 2))

    assert result.shape == (1, 3, 2, 4, 4)
    assert torch.isfinite(result).all()


def test_tiny_audio_decoder_restores_hop_geometry() -> None:
    decoder = MiniMaxH3AudioBigVGANDecoder(
        in_channels=8,
        upsample_initial_channel=16,
        upsample_rates=(2,),
        upsample_kernel_sizes=(4,),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
    )

    result = decoder(torch.randn(2, 8, 4))

    assert result.shape == (2, 1, 8)
    assert torch.isfinite(result).all()


def test_sequential_decode_synchronizes_audio_to_video_duration() -> None:
    class _VideoDecoder(torch.nn.Module):
        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            del latents
            return torch.zeros(1, 3, 4, 8, 8)

    class _AudioDecoder(torch.nn.Module):
        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            del latents
            return torch.zeros(1, 2, 6000)

    media = decode_latents_sequentially(
        _VideoDecoder(),
        _AudioDecoder(),
        torch.zeros(1),
        torch.zeros(1),
        torch.device("cpu"),
    )

    assert media.video.shape[2] == 4
    assert media.audio.shape[-1] == round(4 / 24 * 32_000)


def test_av_mux_writes_playable_video_and_audio_streams(tmp_path: Path) -> None:
    output = tmp_path / "sample.mp4"
    media = H3GeneratedMedia(
        video=torch.rand(1, 3, 4, 32, 32),
        audio=torch.zeros(1, 2, 6400),
    )

    save_av_mp4(media, output, {"seed": 1})

    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].width == 32
        assert container.streams.video[0].height == 32
        assert container.streams.audio[0].rate == 32_000
    assert output.with_suffix(".mp4.json").exists()
