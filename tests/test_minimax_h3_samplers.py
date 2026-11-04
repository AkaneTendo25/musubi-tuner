from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3 import inference as sampling
from musubi_tuner.minimax_h3 import integration
from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.request import H3GenerationRequest, H3Reference, ReferenceKind


@pytest.mark.parametrize("schedule", ["linear", "beta", "beta57"])
@pytest.mark.parametrize("shift", [1.0, 3.0, 12.0, 1e-7])
def test_schedules_keep_noise_and_clean_endpoints(schedule, shift):
    sigma, timestep = sampling.shifted_flow_schedule(9, shift, torch.device("cpu"), base_schedule=schedule)
    assert sigma[0].item() == 1 and sigma[-1].item() == 0
    assert torch.isfinite(sigma).all() and torch.all(sigma[:-1] > sigma[1:])
    torch.testing.assert_close(timestep, 1 - sigma[:-1], rtol=0, atol=0)


def test_released_linear_grid_is_bit_identical():
    base = torch.linspace(1, 0, 20, dtype=torch.float32)
    for shift in (3.0, 12.0):
        expected = torch.unique_consecutive(shift * base / (1 + (shift - 1) * base))
        actual, _ = sampling.shifted_flow_schedule(20, shift, torch.device("cpu"))
        assert torch.equal(actual, expected)


def test_beta_inverse_against_analytic_arcsine_distribution():
    actual = sampling._beta_base(17, alpha=0.5, beta=0.5).double()
    quantiles = torch.linspace(1, 0, 17, dtype=torch.float64)
    expected = torch.sin(math.pi * quantiles / 2).square()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("shift", [0, -1, math.inf, math.nan])
def test_invalid_shifts_fail_before_schedule(shift):
    with pytest.raises(ValueError):
        sampling.shifted_flow_schedule(6, shift, torch.device("cpu"))


def test_sde_eta_zero_is_euler_and_preserves_generator():
    sample = torch.tensor([1.0, 2.0])
    clean = torch.tensor([0.2, -0.4])
    generator = torch.Generator().manual_seed(53)
    state = generator.get_state().clone()
    actual = sampling.data_ward_sde_step(sample, clean, 0.8, 0.4, 0, generator)
    torch.testing.assert_close(actual, 0.5 * sample + 0.5 * clean)
    assert torch.equal(state, generator.get_state())


def test_sde_noise_mixing_matches_independent_formula_and_terminal_draw_policy():
    sample, clean = torch.tensor([1.0, 2.0]), torch.tensor([0.2, -0.4])
    eta, sigma, following = 0.6, 0.8, 0.4
    expected_rng = torch.Generator().manual_seed(53)
    fresh = torch.randn(sample.shape, generator=expected_rng)
    implied = (sample - (1 - sigma) * clean) / sigma
    expected = (1 - following) * clean + following * (math.sqrt(1 - eta**2) * implied + eta * fresh)
    generator = torch.Generator().manual_seed(53)
    actual = sampling.data_ward_sde_step(sample, clean, sigma, following, eta, generator)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(generator.get_state(), expected_rng.get_state())
    state = generator.get_state().clone()
    torch.testing.assert_close(sampling.data_ward_sde_step(sample, clean, sigma, 0, eta, generator), clean)
    assert torch.equal(generator.get_state(), state)


def test_res2s_stage_and_weight_against_numerical_integral():
    sigma, following, c2 = 0.7, 0.2, 0.5
    coordinate = lambda value: math.log((1 - value) / value)
    start, end = coordinate(sigma), coordinate(following)
    stage = 1 / (1 + math.exp(start + c2 * (end - start)))
    assert sampling.res_2s_stage_sigma(sigma, following, c2) == pytest.approx(stage)
    h = end - start
    nodes = torch.linspace(0, h, 10001, dtype=torch.float64)
    weight = torch.trapezoid(nodes.exp() * nodes / h, nodes).item() / (c2 * math.expm1(h))
    sample, clean, second = torch.tensor([0.4]), torch.tensor([0.2]), torch.tensor([0.8])
    blended = clean + weight * (second - clean)
    expected = following / sigma * sample + (1 - following / sigma) * blended
    torch.testing.assert_close(sampling.data_ward_res_2s_step(sample, clean, second, sigma, following, c2), expected)


def test_res2m_matches_independent_history_ratio():
    sample, current, previous = torch.tensor([0.4]), torch.tensor([0.6]), torch.tensor([0.2])
    before, sigma, following = 0.85, 0.6, 0.25
    coordinate = lambda value: math.log((1 - value) / value)
    ratio = (coordinate(sigma) - coordinate(before)) / (coordinate(following) - coordinate(sigma))
    blended = current + (current - previous) / (2 * ratio)
    expected = following / sigma * sample + (1 - following / sigma) * blended
    torch.testing.assert_close(sampling.data_ward_res_2m_step(sample, current, previous, before, sigma, following), expected)


@pytest.mark.parametrize("c2", [0, -1, 1.1, math.nan, 1e-50])
def test_unsafe_stage_nodes_are_controlled_errors(c2):
    with pytest.raises(ValueError):
        sampling.data_ward_res_2s_step(torch.ones(1), torch.ones(1), torch.ones(1), 0.7, 0.2, c2)


def tiny_joint_model(in_channels=4, audio_in_channels=8):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = MiniMaxH3Transformer(
            MiniMaxH3TransformerConfig(
                num_attention_heads=2,
                attention_head_dim=16,
                hidden_size=32,
                num_layers=1,
                num_refiner_layers=1,
                ffn_dim=64,
                in_channels=in_channels,
                audio_in_channels=audio_in_channels,
                patch_size=(1, 2, 2),
                text_dim=12,
                freq_dim=8,
                time_embed_hidden_dim=32,
                time_embed_dim=16,
                rope_freq_dim=2,
            )
        ).eval()
        condition = {H3_TEXT_HIDDEN_KEY: torch.randn(3, 12), H3_TEXT_TOKEN_TAGS_KEY: torch.ones(3, dtype=torch.long)}
    return model, condition


@pytest.mark.parametrize("sampler", ["euler", "sde", "res_2s", "res_2m"])
@pytest.mark.parametrize("strength,ordinary_calls,stage_calls", [(None, 5, 8), (0.6, 3, 5), (0.2, 1, 1)])
def test_real_tiny_joint_model_all_solvers_are_finite_seeded_and_evaluate_stages(sampler, strength, ordinary_calls, stage_calls):
    model, condition = tiny_joint_model()
    calls = []
    hook = model.register_forward_pre_hook(lambda *_: calls.append(1))
    arguments = dict(
        height=32, width=32, frame_count=5, num_inference_steps=6, device=torch.device("cpu"), show_progress=False, sampler=sampler
    )
    if sampler == "sde":
        arguments["sde_eta"] = 0.4
    if strength is not None:
        arguments.update(
            denoise_strength=strength,
            init_video=torch.zeros(1, 4, 2, 2, 2),
            init_audio=torch.zeros(1, 2, 8, 8),
        )
    with sampling.count_model_evaluations(model) as counts:
        first = sampling.denoise_fl2va(model, condition, generator=torch.Generator().manual_seed(11), **arguments)
    hook.remove()
    expected_calls = stage_calls if sampler == "res_2s" else ordinary_calls
    assert len(calls) == expected_calls
    assert counts["model_evaluations"] == expected_calls
    second = sampling.denoise_fl2va(model, condition, generator=torch.Generator().manual_seed(11), **arguments)
    assert first[0].shape == (1, 4, 2, 2, 2) and first[1].shape == (1, 2, 8, 8)
    for actual, replay in zip(first, second):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, replay, rtol=0, atol=0)


def test_model_evaluation_counter_accumulates_passes_and_removes_hook_after_failure():
    model = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="injected failure"):
        with sampling.count_model_evaluations(model) as counts:
            for _ in range(3):
                model(torch.ones(1))
            raise RuntimeError("injected failure")
    model(torch.ones(1))
    assert counts == {"model_evaluations": 3}
    assert not model._forward_pre_hooks


@pytest.mark.parametrize(
    "sampler,two_pass,guidance,first_pass_lora,reference,expected_calls",
    [
        ("euler", False, False, False, False, 5),
        ("res_2s", False, False, False, False, 8),
        ("euler", True, False, False, False, 6),
        ("res_2s", True, False, False, False, 9),
        ("euler", True, True, False, False, 9),
        ("res_2s", True, True, False, False, 14),
        ("res_2s", True, True, True, False, 18),
        ("euler", False, True, False, True, 10),
        ("euler", True, True, False, True, 9),
    ],
)
def test_generator_reports_actual_forward_calls_in_saved_metadata(
    monkeypatch, tmp_path, sampler, two_pass, guidance, first_pass_lora, reference, expected_calls
):
    # Keep the native generation, two-pass, guidance and denoising paths real;
    # replace only external asset loading and media encoding.
    model, condition = tiny_joint_model(in_channels=24, audio_in_channels=32)
    generator = object.__new__(integration._NativeGenerator)
    generator.__dict__.update(
        device=torch.device("cpu"),
        lora_weights=(),
        height=64,
        width=64,
        first_pass_scale=0.5 if two_pass else 0.0,
        first_pass_steps=4,
        num_inference_steps=6,
        second_pass_strength=0.6,
        first_pass_lora=first_pass_lora,
        null_guidance_scale=2.0 if guidance else 0.0,
        _null_conditioning={H3_TEXT_HIDDEN_KEY: torch.zeros_like(condition[H3_TEXT_HIDDEN_KEY])},
        latent_upscaler=None,
        latent_upscale_scale=1.0,
        sampler=sampler,
        base_schedule="linear",
        video_shift=12.0,
        audio_shift=3.0,
        sde_eta=0.0,
        res_c2=0.5,
        video_vae=Path("unused-video-vae"),
        audio_vae=Path("unused-audio-vae"),
    )
    monkeypatch.setattr(generator, "_load_transformer", lambda: (model, []))
    monkeypatch.setattr(generator, "_encode_prompt", lambda *args, **kwargs: condition)
    monkeypatch.setattr(generator, "_prepare_keyframes", lambda *args: ([], ()))
    monkeypatch.setattr(generator, "_prepare_references", lambda *args: ())
    monkeypatch.setattr(generator, "_prepare_guides", lambda *args: ())
    monkeypatch.setattr(integration, "load_video_vae_decoder", lambda *args: None)
    monkeypatch.setattr(integration, "load_audio_vae_decoder", lambda *args: None)
    monkeypatch.setattr(integration, "decode_latents_sequentially", lambda *args: SimpleNamespace(fps=24, sample_rate=32000))
    saved = []
    monkeypatch.setattr(integration, "save_av_mp4", lambda media, path, metadata: saved.append(metadata))
    references = (H3Reference(tmp_path / "unused-reference.png", ReferenceKind.IMAGE),) if reference else ()
    request = H3GenerationRequest(
        "prompt",
        tmp_path / "output.mp4",
        frame_count_override=5,
        references=references,
        reference_route="text_only" if reference else "dual",
    )
    actual_calls = []
    hook = model.register_forward_pre_hook(lambda *_: actual_calls.append(1))
    generator.generate(request)
    # Counts belong to one generation even when a generator/model is reused.
    generator.generate(request)
    hook.remove()
    assert len(actual_calls) == 2 * expected_calls
    assert [metadata["model_evaluations"] for metadata in saved] == [expected_calls, expected_calls]
    assert not model._forward_pre_hooks
