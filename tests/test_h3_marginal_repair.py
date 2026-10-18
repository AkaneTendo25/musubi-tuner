from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.cache import H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY, H3_TEXT_HIDDEN_KEY
from musubi_tuner.minimax_h3.marginal_repair import (
    H3CaptionAnchorPool,
    align_anchor_hidden,
    caption_marginal_reference,
    guidance_recovered_target,
    interpolate_schedule,
    parse_scale_schedule,
    scale_at,
    unbiased_caption_repair_loss,
)
from musubi_tuner.minimax_h3.training import H3JointLoss, H3ModelPrediction
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser, read_config_from_file


def configured(*options):
    args = create_parser().parse_args(["--sdpa", *options])
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    return args, trainer


def test_reconstruction_formula_and_detached_teacher():
    reference = torch.tensor([[2.0, 4.0], [3.0, 5.0]], requires_grad=True)
    conditional = torch.tensor([[6.0, 8.0], [7.0, 9.0]])
    scales = torch.tensor([2.0, 4.0])
    distilled = reference.detach() + scales[:, None] * (conditional - reference.detach())
    recovered = guidance_recovered_target(distilled.requires_grad_(), reference, scales)
    torch.testing.assert_close(recovered, conditional)
    assert not recovered.requires_grad
    torch.testing.assert_close(guidance_recovered_target(distilled, reference, 1), distilled)


@pytest.mark.parametrize("spec", ["nan", "inf,2", "0.5", "2,3,4", "garbage"])
def test_invalid_scale(spec):
    with pytest.raises(ValueError):
        parse_scale_schedule(spec)


def test_schedules_and_no_extrapolation():
    sigma = torch.tensor([0.0, 0.4, 0.8, 0.95, 1.0])
    torch.testing.assert_close(interpolate_schedule((1, 3, 6), sigma), torch.tensor([1.0, 1.0, 3.0, 6.0, 6.0]))
    torch.testing.assert_close(interpolate_schedule((1, 3), sigma)[[0, 1, 3, 4]], torch.tensor([1.0, 1.0, 3.0, 3.0]))
    torch.testing.assert_close(scale_at((0, 4), sigma), (4 * sigma).clamp_min(1))
    with pytest.raises(ValueError, match="extrapolates"):
        guidance_recovered_target(torch.ones(1, 2), torch.zeros(1, 2), 4, torch.tensor([2.0]))


def test_caption_alignment_prediction_mean_and_sparse_expectation():
    empty = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    torch.testing.assert_close(
        align_anchor_hidden(torch.tensor([[1.0, 2.0]]), empty), torch.tensor([[1.0, 2.0], [20.0, 21.0], [30.0, 31.0]])
    )
    reference = caption_marginal_reference(
        [H3ModelPrediction(torch.tensor([1.0]), None), H3ModelPrediction(torch.tensor([5.0]), None)]
    )
    torch.testing.assert_close(reference.video, torch.tensor([3.0]))
    baseline = H3JointLoss(*(torch.tensor(value) for value in (2.0, 3.0, 0.0)), 1, 0)
    caption = H3JointLoss(*(torch.tensor(value) for value in (6.0, 7.0, 0.0)), 1, 0)
    active = unbiased_caption_repair_loss(baseline, caption, 0.25)
    # E[(1-I)b + I(b + (c-b)/p)] = c.
    expected = 0.75 * baseline.loss + 0.25 * active.loss
    torch.testing.assert_close(expected, caption.loss)


def test_caption_pool_skips_every_current_caption_duplicate_and_rehashes(tmp_path):
    paths = []
    current = torch.ones(2, 3)
    for index, hidden in enumerate((current, current.clone(), torch.full((1, 3), 2.0), torch.full((4, 3), 3.0))):
        path = tmp_path / f"caption-{index}.safetensors"
        save_file({H3_TEXT_HIDDEN_KEY: hidden}, path)
        paths.append(str(path))
    pool = H3CaptionAnchorPool(paths)
    initial = pool.identity
    selected = pool.draw(2, generator=__import__("random").Random(7), exclude=current)
    assert sorted(float(value.flatten()[0]) for value in selected) == [2.0, 3.0]
    save_file({H3_TEXT_HIDDEN_KEY: torch.full((4, 3), 4.0)}, paths[-1])
    assert pool.compute_identity(refresh=True) != initial


class _StateHooks:
    device = torch.device("cpu")
    process_index = 0
    num_processes = 1

    def register_save_state_pre_hook(self, hook):
        self.save = hook

    def register_load_state_pre_hook(self, hook):
        self.load = hook


def _caption_state_trainer(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"state-caption-{index}.safetensors"
        save_file({H3_TEXT_HIDDEN_KEY: torch.full((2, 3), float(index))}, path)
        paths.append(str(path))
    args, trainer = configured(
        "--h3_repair_reference",
        "caption",
        "--h3_repair_scale",
        "4",
        "--h3_repair_caption_anchors",
        "1",
        "--h3_repair_caption_probability",
        "0.5",
        "--h3_repair_reference_weight",
        "0.25",
    )
    trainer._repair_anchor_pool = H3CaptionAnchorPool(paths)
    trainer._repair_anchor_pool_identity = trainer._repair_anchor_pool.identity
    hooks = _StateHooks()
    trainer._register_caption_repair_state_hooks(args, hooks)
    return args, trainer, hooks, paths


def test_caption_state_roundtrip_restores_both_rngs_and_lazy_state(tmp_path):
    _, trainer, hooks, _ = _caption_state_trainer(tmp_path)
    hooks.save([], [], tmp_path)
    trainer._repair_caption_generator = torch.Generator().manual_seed(99)
    trainer._repair_anchor_random = __import__("random").Random(98)
    hooks.load([], tmp_path)
    assert trainer._repair_caption_generator is None
    assert trainer._repair_anchor_random is None

    trainer._repair_caption_generator = torch.Generator().manual_seed(7)
    trainer._repair_anchor_random = __import__("random").Random(8)
    hooks.save([], [], tmp_path)
    expected_torch = torch.rand((), generator=trainer._repair_caption_generator)
    expected_python = trainer._repair_anchor_random.random()
    torch.manual_seed(123456)
    __import__("random").seed(654321)
    hooks.load([], tmp_path)
    torch.testing.assert_close(torch.rand((), generator=trainer._repair_caption_generator), expected_torch)
    assert trainer._repair_anchor_random.random() == expected_python


def test_caption_state_rejects_pool_mutation_after_identity_was_cached(tmp_path):
    _, trainer, hooks, paths = _caption_state_trainer(tmp_path)
    hooks.save([], [], tmp_path)
    save_file({H3_TEXT_HIDDEN_KEY: torch.full((2, 3), 9.0)}, paths[-1])
    with pytest.raises(ValueError, match="pool changed"):
        hooks.load([], tmp_path)


@pytest.mark.parametrize("missing", ["activation_generator", "anchor_random", "caption_dropout_rate", "null_anchor_weight"])
def test_caption_state_rejects_missing_rng_fields_without_mutating_streams(tmp_path, missing):
    _, trainer, hooks, _ = _caption_state_trainer(tmp_path)
    hooks.save([], [], tmp_path)
    state_path = tmp_path / "h3_caption_repair_state-rank0.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state[missing]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    generator = torch.Generator().manual_seed(111)
    anchor_random = __import__("random").Random(222)
    trainer._repair_caption_generator = generator
    trainer._repair_anchor_random = anchor_random
    generator_state = generator.get_state().clone()
    random_state = anchor_random.getstate()
    with pytest.raises(ValueError, match="schema"):
        hooks.load([], tmp_path)
    torch.testing.assert_close(trainer._repair_caption_generator.get_state(), generator_state)
    assert trainer._repair_anchor_random.getstate() == random_state


def test_caption_state_v1_migrates_only_with_zero_caption_dropout(tmp_path):
    _, trainer, hooks, _ = _caption_state_trainer(tmp_path)
    hooks.save([], [], tmp_path)
    state_path = tmp_path / "h3_caption_repair_state-rank0.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["version"] = 1
    del state["caption_dropout_rate"]
    del state["null_anchor_weight"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    hooks.load([], tmp_path)

    trainer_args = SimpleNamespace(
        h3_repair_caption_anchors=1,
        h3_repair_caption_probability=0.5,
        h3_repair_reference_weight=0.25,
        h3_repair_null_anchor_weight=0.0,
        h3_caption_dropout_rate=0.5,
    )
    rejecting_hooks = _StateHooks()
    trainer._register_caption_repair_state_hooks(trainer_args, rejecting_hooks)
    with pytest.raises(ValueError, match="only zero caption dropout"):
        rejecting_hooks.load([], tmp_path)


def test_caption_state_v2_rejects_nonzero_repair_null_anchor(tmp_path):
    _, trainer, hooks, _ = _caption_state_trainer(tmp_path)
    hooks.save([], [], tmp_path)
    state_path = tmp_path / "h3_caption_repair_state-rank0.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["version"] = 2
    del state["null_anchor_weight"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    trainer_args = SimpleNamespace(
        h3_repair_caption_anchors=1,
        h3_repair_caption_probability=0.5,
        h3_repair_reference_weight=0.25,
        h3_repair_null_anchor_weight=0.5,
        h3_caption_dropout_rate=0.0,
    )
    rejecting_hooks = _StateHooks()
    trainer._register_caption_repair_state_hooks(trainer_args, rejecting_hooks)
    with pytest.raises(ValueError, match="v2 resume state"):
        rejecting_hooks.load([], tmp_path)


def _preservation_state_trainer(probability):
    args, trainer = configured(
        "--h3_repair_reference",
        "empty",
        "--h3_repair_scale",
        "4",
        "--h3_repair_preservation_probability",
        str(probability),
    )
    hooks = _StateHooks()
    trainer._register_repair_preservation_state_hooks(args, hooks)
    return args, trainer, hooks


def test_repair_preservation_state_restores_used_and_lazy_streams(tmp_path):
    _, trainer, hooks = _preservation_state_trainer(0.5)
    hooks.save([], [], tmp_path)
    trainer._repair_preservation_generator = torch.Generator().manual_seed(8)
    hooks.load([], tmp_path)
    assert trainer._repair_preservation_generator is None

    trainer._repair_preservation_generator = torch.Generator().manual_seed(19)
    hooks.save([], [], tmp_path)
    expected = torch.rand((), generator=trainer._repair_preservation_generator)
    torch.manual_seed(9876)
    hooks.load([], tmp_path)
    torch.testing.assert_close(torch.rand((), generator=trainer._repair_preservation_generator), expected)


def test_repair_preservation_zero_accepts_legacy_missing_state_but_rejects_existing_cycle(tmp_path):
    _, zero_trainer, zero_hooks = _preservation_state_trainer(0)
    zero_hooks.load([], tmp_path)
    assert zero_trainer._repair_preservation_generator is None
    _, active_trainer, active_hooks = _preservation_state_trainer(0.5)
    active_hooks.save([], [], tmp_path)
    with pytest.raises(ValueError, match="probability"):
        zero_hooks.load([], tmp_path)


@pytest.mark.parametrize("bad", [[], [-1], [256], [1, 2, 3]])
def test_repair_preservation_state_rejects_malformed_generator_without_mutation(tmp_path, bad):
    _, trainer, hooks = _preservation_state_trainer(0.5)
    hooks.save([], [], tmp_path)
    path = tmp_path / "h3_repair_preservation_state-rank0.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["generator"] = bad
    path.write_text(json.dumps(state), encoding="utf-8")
    current = torch.Generator().manual_seed(77)
    trainer._repair_preservation_generator = current
    before = current.get_state().clone()
    with pytest.raises(ValueError, match="generator state"):
        hooks.load([], tmp_path)
    torch.testing.assert_close(trainer._repair_preservation_generator.get_state(), before)


@pytest.mark.parametrize(
    "options,match",
    [
        (["--h3_repair_reference", "caption", "--h3_repair_scale", "4", "--h3_repair_caption_anchors", "0"], "at least"),
        (["--h3_repair_reference", "caption", "--h3_repair_scale", "4", "--h3_repair_caption_probability", "0"], "lie in"),
        (["--h3_repair_reference", "empty", "--h3_repair_scale", "4", "--h3_repair_reference_weight", "1"], "must be 0"),
        (
            [
                "--h3_repair_reference",
                "caption",
                "--h3_repair_scale",
                "4",
                "--h3_repair_reference_weight",
                "1",
                "--blocks_to_swap",
                "1",
                "--fused_backward_pass",
            ],
            "cannot use",
        ),
    ],
)
def test_caption_repair_validation(options, match):
    with pytest.raises(ValueError, match=match):
        configured(*options)


@pytest.mark.parametrize(
    "options,match",
    [
        (["--h3_repair_scale", "4"], "require"),
        (["--h3_repair_null_anchor_weight", "0.5"], "require"),
        (["--h3_repair_preservation_probability", "0.5"], "require"),
        (["--h3_repair_reference", "empty"], "requires"),
        (["--h3_repair_reference", "empty", "--h3_repair_scale", "4", "--h3_guidance_distillation_scale", "4"], "guidance"),
        (["--h3_repair_reference", "empty", "--h3_repair_scale", "4", "--h3_repair_sigma_weight", "0,1"], "positive"),
        (["--h3_repair_reference", "empty", "--h3_repair_scale", "4", "--h3_repair_target_multiplier", "2,2,2"], "extrapolates"),
        (["--h3_repair_reference", "empty", "--h3_repair_scale", "4", "--h3_repair_preservation_probability", "1"], "lie in"),
    ],
)
def test_early_validation(options, match):
    with pytest.raises(ValueError, match=match):
        configured(*options)


def test_toml_and_metadata(tmp_path):
    path = tmp_path / "repair.toml"
    path.write_text('h3_repair_reference="empty"\nh3_repair_scale="0.86,4.24"\nh3_repair_sigma_weight=[1.0,2.0,3.0]\n')
    parser = create_parser()
    argv = ["--config_file", str(path), "--sdpa"]
    args = read_config_from_file(parser.parse_args(argv), parser, argv)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    metadata = trainer.extra_metadata(args)
    assert trainer._repair_schedule == (0.86, 4.24)
    assert args.h3_repair_sigma_weight == (1, 2, 3)
    assert metadata["ss_h3_repair_reference"] == "empty"
    assert metadata["ss_h3_repair_scale"] == "0.86,4.24"
    defaults, _ = configured()
    assert defaults.h3_repair_reference is None


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor(0.5))
        self.enabled = True
        self.events = []

    def set_enabled(self, enabled):
        self.enabled = enabled

    def switch_block_swap_for_inference(self):
        self.events.append("inference")

    def switch_block_swap_for_training(self):
        self.events.append("training")


class TinyBackend:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def predict_training(self, transformer, batch, video, audio, video_sigma, audio_sigma, *, conditioning="prompt"):
        self.calls.append((conditioning, torch.is_grad_enabled(), transformer.enabled))
        if conditioning == "empty" and self.fail:
            raise RuntimeError("teacher failure")
        offset = 4.0 if conditioning == "prompt" else 0.0
        delta = transformer.adapter if transformer.enabled else 0
        return H3ModelPrediction(
            None if video is None else torch.ones_like(video) * (offset + delta),
            None if audio is None else torch.ones_like(audio) * (offset + delta),
        )


def run_step(args, trainer, *, audio=False, swap=False, fail=False, missing=False, data=0.0, backend=None):
    transformer = TinyTransformer()
    backend = TinyBackend(fail) if backend is None else backend
    trainer.backend = backend
    trainer.blocks_to_swap = 2 if swap else 0
    trainer.dit_dtype = torch.float32
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        unwrap_model=lambda x: x,
        autocast=nullcontext,
        print=lambda *a: None,
        backward=lambda loss: loss.backward(),
    )
    video = torch.full((1, 24, 2, 2, 2), data)
    batch = {
        "timesteps": [0.5],
        H3_TEXT_HIDDEN_KEY: [torch.ones(1, 8)],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(1, 8)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
    }
    if missing:
        batch.pop(H3_EMPTY_TEXT_HIDDEN_KEY)
    if audio:
        from musubi_tuner.minimax_h3.cache import H3_AUDIO_LATENTS_KEY

        batch[H3_AUDIO_LATENTS_KEY] = torch.full((1, 4, 1, 4), data)
    result = trainer.process_batch(
        args,
        accelerator,
        transformer,
        transformer,
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    return result, transformer, backend


def test_first_caption_activation_initializes_anchor_rng_before_shared_model_entry():
    class Pool:
        def draw(self, count, *, generator, exclude):
            assert count == 1
            return [torch.full((1, 8), 2.0)]

    class RandomBackend(TinyBackend):
        def predict_training(self, *values, **kwargs):
            self.calls.append(float(torch.rand(())))
            return super().predict_training(*values, **kwargs)

    args, trainer = configured(
        "--h3_repair_reference",
        "caption",
        "--h3_repair_scale",
        "4",
        "--h3_repair_caption_anchors",
        "1",
    )
    trainer._repair_anchor_pool = Pool()
    torch.manual_seed(404)
    backend = RandomBackend()
    run_step(args, trainer, backend=backend)
    # Every frozen branch forks the exact entry state and the live primary then
    # consumes that same draw. Lazy sampler initialization has already happened.
    assert backend.calls[0] == backend.calls[2] == backend.calls[4]


def test_caption_reference_auxiliary_leaves_classic_swap_ready_for_primary():
    class Pool:
        def draw(self, count, *, generator, exclude):
            return [torch.full((1, 8), 2.0)]

    args, trainer = configured(
        "--h3_repair_reference",
        "caption",
        "--h3_repair_scale",
        "4",
        "--h3_repair_caption_anchors",
        "1",
        "--h3_repair_reference_weight",
        "0.2",
    )
    trainer._repair_anchor_pool = Pool()
    (_, metrics), model, _ = run_step(args, trainer, swap=True)
    assert model.events == ["inference", "training"]
    assert metrics["loss/repair_reference"] > 0


@pytest.mark.parametrize("reference", ["empty", "caption"])
def test_caption_dropout_uses_current_teacher_and_reuses_primary_empty_graph(reference):
    class Pool:
        def draw(self, count, *, generator, exclude):
            return [torch.full((1, 8), 2.0)]

    options = [
        "--h3_repair_reference",
        reference,
        "--h3_repair_scale",
        "4",
        "--h3_caption_dropout_rate",
        "1",
    ]
    if reference == "caption":
        options += ["--h3_repair_caption_anchors", "1", "--h3_repair_reference_weight", "0.2"]
    args, trainer = configured(*options)
    trainer._repair_anchor_pool = Pool()
    (loss, metrics), model, backend = run_step(args, trainer)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.adapter.grad is not None and torch.isfinite(model.adapter.grad)
    # Empty repair: one frozen current/empty teacher plus the primary. Caption
    # repair: one frozen current teacher, one frozen caption, then the primary;
    # its reference auxiliary reuses that primary graph.
    expected = 2 if reference == "empty" else 3
    assert len(backend.calls) == expected
    assert backend.calls[0] == ("empty", False, False)
    assert backend.calls[-1] == ("empty", True, True)
    if reference == "caption":
        assert metrics["loss/repair_reference"] > 0


def test_caption_repair_dropout_trace_is_deterministic():
    traces = []
    for _ in range(2):
        torch.manual_seed(812)
        args, trainer = configured(
            "--h3_repair_reference",
            "empty",
            "--h3_repair_scale",
            "4",
            "--h3_caption_dropout_rate",
            "0.5",
        )
        (_, _), _, backend = run_step(args, trainer)
        traces.append(backend.calls)
    assert traces[0] == traces[1]


def test_repair_null_anchor_is_opportunistic_and_not_probability_scaled():
    class Pool:
        def draw(self, count, *, generator, exclude):
            return [torch.full((1, 8), 2.0)]

    class DistinctBackend(TinyBackend):
        def predict_training(self, transformer, batch, video, audio, video_sigma, audio_sigma, *, conditioning="prompt"):
            self.calls.append((conditioning, torch.is_grad_enabled(), transformer.enabled))
            if conditioning == "prompt":
                offset = 4.0
            elif transformer.enabled:
                offset = 0.0
            else:
                value = batch[H3_EMPTY_TEXT_HIDDEN_KEY]
                value = value[0] if isinstance(value, (list, tuple)) else value
                offset = float(value.mean())
            delta = transformer.adapter if transformer.enabled else 0
            return H3ModelPrediction(
                None if video is None else torch.ones_like(video) * (offset + delta),
                None if audio is None else torch.ones_like(audio) * (offset + delta),
            )

    args, trainer = configured(
        "--h3_repair_reference",
        "caption",
        "--h3_repair_scale",
        "4",
        "--h3_repair_caption_anchors",
        "1",
        "--h3_repair_caption_probability",
        "0.25",
        "--h3_repair_reference_weight",
        "0.2",
        "--h3_repair_null_anchor_weight",
        "0.5",
    )
    trainer._repair_anchor_pool = Pool()
    trainer._repair_caption_active = lambda *_: True
    (loss, metrics), model, backend = run_step(args, trainer, swap=True, backend=DistinctBackend())
    # The trainable empty prediction is .5 and the frozen base empty is 0.
    # Weight .5 therefore contributes .5 * .5^2, with no division by p=.25.
    assert metrics["loss/repair_null_anchor"] == pytest.approx(0.125)
    # The caption reference is 2 while the frozen base empty is 0, proving the
    # two auxiliary targets are not accidentally shared.
    assert metrics["loss/repair_reference"] == pytest.approx((0.2 / 0.25) * (0.5 - 2.0) ** 2)
    assert [call[:2] for call in backend.calls if call[1]] == [("empty", True), ("prompt", True)]
    assert model.events == ["inference", "training"]
    assert metrics["_loss_for_average"] == pytest.approx(metrics["loss/repair_video"])


def test_repair_null_anchor_never_buys_a_trainable_empty_forward():
    class Pool:
        def draw(self, count, *, generator, exclude):
            return [torch.full((1, 8), 2.0)]

    args, trainer = configured(
        "--h3_repair_reference",
        "caption",
        "--h3_repair_scale",
        "4",
        "--h3_repair_caption_anchors",
        "1",
        "--h3_repair_null_anchor_weight",
        "0.5",
    )
    trainer._repair_anchor_pool = Pool()
    (_, metrics), _, backend = run_step(args, trainer)
    assert len(backend.calls) == 3
    assert [call for call in backend.calls if call[1]] == [("prompt", True, True)]
    assert metrics["loss/repair_null_anchor"] == 0.0


def test_repair_null_anchor_reuses_dropped_primary_empty():
    args, trainer = configured(
        "--h3_repair_reference",
        "empty",
        "--h3_repair_scale",
        "4",
        "--h3_caption_dropout_rate",
        "1",
        "--h3_repair_null_anchor_weight",
        "0.5",
    )
    (loss, metrics), _, backend = run_step(args, trainer)
    assert len(backend.calls) == 2
    assert backend.calls[-1] == ("empty", True, True)
    assert metrics["loss/repair_null_anchor"] == pytest.approx(0.125)
    assert metrics["_loss_for_average"] == pytest.approx(float(loss.detach()) - 0.125)


def test_repair_preservation_replaces_whole_objective_without_rescaling_or_repair_draws():
    class Pool:
        def draw(self, count, *, generator, exclude):
            raise AssertionError("preservation steps must not draw caption anchors")

    class EntryStateBackend(TinyBackend):
        def __init__(self):
            super().__init__()
            self.entry_states = []

        def predict_training(self, *values, **kwargs):
            self.entry_states.append(torch.get_rng_state().clone())
            return super().predict_training(*values, **kwargs)

    args, trainer = configured(
        "--h3_repair_reference",
        "caption",
        "--h3_repair_scale",
        "4",
        "--h3_repair_caption_anchors",
        "1",
        "--h3_repair_caption_probability",
        "0.25",
        "--h3_repair_reference_weight",
        "0.2",
        "--h3_repair_null_anchor_weight",
        "0.5",
        "--h3_repair_preservation_probability",
        "0.25",
    )
    trainer._repair_anchor_pool = Pool()
    trainer._repair_preservation_active = lambda *_: True
    caption_generator = torch.Generator().manual_seed(31)
    anchor_random = __import__("random").Random(32)
    trainer._repair_caption_generator = caption_generator
    trainer._repair_anchor_random = anchor_random
    caption_state = caption_generator.get_state().clone()
    anchor_state = anchor_random.getstate()
    (loss, metrics), _, backend = run_step(args, trainer, swap=True, backend=EntryStateBackend())
    # Frozen current prompt is 4 and the live prompt is 4.5: MSE .25. The
    # preservation probability does not scale either objective.
    assert float(loss.detach()) == pytest.approx(0.25)
    assert metrics["loss/repair_preservation"] == pytest.approx(0.25)
    assert metrics["h3/repair_preservation_active"] == 1.0
    assert backend.calls == [("prompt", False, False), ("prompt", True, True)]
    torch.testing.assert_close(backend.entry_states[0], backend.entry_states[1])
    assert "loss/repair_reference" not in metrics
    assert "loss/repair_null_anchor" not in metrics
    torch.testing.assert_close(caption_generator.get_state(), caption_state)
    assert anchor_random.getstate() == anchor_state


def test_repair_preservation_inactive_step_keeps_repair_objective_and_reports_zero():
    args, trainer = configured(
        "--h3_repair_reference",
        "empty",
        "--h3_repair_scale",
        "4",
        "--h3_repair_preservation_probability",
        "0.25",
    )
    trainer._repair_preservation_active = lambda *_: False
    (loss, metrics), _, backend = run_step(args, trainer)
    assert float(loss.detach()) == pytest.approx(3.5**2)
    assert metrics["h3/repair_preservation_active"] == 0.0
    assert metrics["loss/repair_preservation"] == 0.0
    assert len(backend.calls) == 3


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_cpu_trainer_smoke_replaces_data_loss_and_updates_only_adapter(audio, swap):
    args, trainer = configured("--h3_repair_reference", "empty", "--h3_repair_scale", "4")
    (loss, metrics), model, backend = run_step(args, trainer, audio=audio, swap=swap)
    # Frozen g=4, u=0, recovered c=1; live prediction=4.5.
    assert loss.item() == pytest.approx(3.5**2)
    loss.backward()
    assert model.adapter.grad.item() == pytest.approx(7)
    optimizer = torch.optim.SGD([model.adapter], lr=0.01)
    optimizer.step()
    assert model.adapter.item() == pytest.approx(0.43)
    assert backend.calls == [("prompt", False, False), ("empty", False, False), ("prompt", True, True)]
    assert model.events == (["inference", "training"] if swap else [])
    assert "loss/repair_video" in metrics
    args2, trainer2 = configured("--h3_repair_reference", "empty", "--h3_repair_scale", "4")
    (changed, _), _, _ = run_step(args2, trainer2, audio=audio, data=50)
    torch.testing.assert_close(loss, changed)


def test_cpu_missing_cache_and_teacher_failure_restore_state(monkeypatch):
    import sys

    args, trainer = configured("--h3_repair_reference", "empty", "--h3_repair_scale", "4")
    with pytest.raises(KeyError, match="cache_guidance_empty"):
        run_step(args, trainer, missing=True)
    model = TinyTransformer()
    monkeypatch.setattr(sys.modules[__name__], "TinyTransformer", lambda: model)
    with pytest.raises(RuntimeError, match="teacher failure"):
        run_step(args, trainer, swap=True, fail=True)
    assert model.enabled
    assert model.events == ["inference", "training"]


def test_cpu_av_schedule_uses_each_modalities_shifted_sigma():
    args, trainer = configured(
        "--h3_repair_reference",
        "empty",
        "--h3_repair_scale",
        "1,4",
        "--h3_loss_balance",
        "modality",
        "--h3_repair_sigma_weight",
        "2,2",
    )
    (loss, metrics), _, _ = run_step(args, trainer, audio=True)
    # t=.5, shift=12 for video and 3 for audio: sigma=12/13 and 3/4.
    video_error = (4.5 - 4 / (1 + 4 * 12 / 13)) ** 2
    audio_error = (4.5 - 4 / (1 + 4 * 3 / 4)) ** 2
    assert metrics["loss/repair_video"] == pytest.approx(2 * video_error)
    assert metrics["loss/repair_audio"] == pytest.approx(2 * audio_error)
    assert loss.item() == pytest.approx(video_error + audio_error)


def test_default_off_keeps_single_forward():
    args, trainer = configured()
    (_, metrics), _, backend = run_step(args, trainer)
    assert backend.calls == [("prompt", True, True)]
    assert "loss/repair_video" not in metrics


def test_cpu_native_transformer_real_lora_optimizer_smoke():
    from musubi_tuner.minimax_h3.cache import (
        H3_CONDITIONING_TASK_IDS,
        H3_CONDITIONING_TASK_KEY,
        H3_TEXT_HIDDEN_KEY,
        H3_TEXT_TOKEN_TAGS_KEY,
    )
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
    from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
    from musubi_tuner.networks import lora_minimax_h3

    torch.manual_seed(42)
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=8,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    model = MiniMaxH3Transformer(config).float().requires_grad_(False)
    original = [(parameter, parameter.detach().clone()) for parameter in model.parameters()]
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, [], model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    args, trainer = configured("--h3_repair_reference", "empty", "--h3_repair_scale", "4")
    trainer.dit_dtype = torch.float32
    backend = _NativeTrainingBackend(model, mode="fl2va")
    trainer.backend = backend
    teacher_grad = []
    original_predict = backend.predict_training

    def observed_predict(*values, **kwargs):
        prediction = original_predict(*values, **kwargs)
        teacher_grad.append(prediction.video.requires_grad)
        return prediction

    backend.predict_training = observed_predict
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        unwrap_model=lambda x: x,
        autocast=nullcontext,
        print=lambda *a: None,
        backward=lambda loss: loss.backward(),
    )
    video = torch.randn(1, 4, 2, 2, 2)
    batch = {
        "timesteps": [0.5],
        H3_TEXT_HIDDEN_KEY: [torch.randn(2, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(2, dtype=torch.long)],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.randn(1, 8)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(1, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
    }
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.SGD(trainable, lr=0.01)
    loss, _ = trainer.process_batch(
        args, accelerator, model, network, batch, video, torch.randn_like(video), None, torch.float32, torch.float32, None, 0
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert teacher_grad == [False, False, True]
    assert all(parameter.grad is None for parameter, _ in original)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in trainable)
    optimizer.step()
    assert any(not torch.equal(parameter, saved) for parameter, saved in zip(trainable, before))
    assert all(torch.equal(parameter, saved) for parameter, saved in original)
