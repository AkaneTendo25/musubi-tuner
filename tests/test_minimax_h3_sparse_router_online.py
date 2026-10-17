import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.cache import (
    H3_ALIGNED_GUIDE_COUNT_KEY,
    H3_CONDITIONING_TASK_KEY,
    H3_REFERENCE_ALIGNED_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_ROUTE_KEY,
    H3_REFERENCE_TEMPORAL_CONTRACT_KEY,
    H3_REFERENCE_VIDEO_FPS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3Attention
from musubi_tuner.minimax_h3.sparse_attention import H3ReferenceSparseLayout, H3UniversalSparseRouter
from scripts.train_minimax_h3_sparse_router_online import (
    _atomic_torch_save,
    _atomic_write_json,
    _capture_training_state,
    _extend_record_chain,
    _json_compatible,
    _load_example,
    _paired_records,
    _record_cache_index,
    _repeat_training_records,
    _run_with_generator_rng,
    _restore_training_state,
    _resolved_generator_seeds,
    _select_training_records,
    _validate_disjoint_records,
    _validate_initial_router_keys,
    _validate_layer_sequence,
    _validate_options,
    _visual_latent_key,
    OnlineRouterTeacher,
)


def _records(count: int) -> list[tuple[Path, Path]]:
    return [(Path(f"latent-{index}"), Path(f"text-{index}")) for index in range(count)]


def test_single_gpu_continuation_covers_each_training_record_once() -> None:
    records = _records(128 + 3000)
    first, validation = _select_training_records(
        records,
        validation_examples=128,
        train_offset=0,
        num_shards=4,
        shard_index=0,
        exclude_shard_index=None,
        max_train_examples=750,
    )
    continuation, repeated_validation = _select_training_records(
        records,
        validation_examples=128,
        train_offset=0,
        num_shards=4,
        shard_index=0,
        exclude_shard_index=0,
        max_train_examples=2250,
    )

    assert validation == repeated_validation == records[:128]
    assert set(first).isdisjoint(continuation)
    assert set(first + continuation) == set(records[128:])


def test_router_training_epochs_are_deterministic_and_keep_first_pass_order() -> None:
    records = _records(7)

    first = _repeat_training_records(records, epochs=3, seed=1234)
    second = _repeat_training_records(records, epochs=3, seed=1234)

    assert first == second
    assert first[: len(records)] == records
    assert len(first) == 3 * len(records)
    assert all(set(first[start : start + len(records)]) == set(records) for start in (0, 7, 14))


def test_validation_rng_is_independent_of_training_progress() -> None:
    training_seed, validation_seed = _resolved_generator_seeds(3407, None)
    assert training_seed == 3408
    assert validation_seed == 1_003_410
    assert _resolved_generator_seeds(3407, 1777) == (3408, 1777)

    baseline = torch.Generator().manual_seed(validation_seed)
    finetuned = torch.Generator().manual_seed(validation_seed)
    training = torch.Generator().manual_seed(training_seed)
    torch.rand(128, generator=training)
    torch.testing.assert_close(torch.rand(16, generator=baseline), torch.rand(16, generator=finetuned))


def test_validation_record_is_bound_to_exactly_one_cache(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    record = (first / "sample_mmh3.safetensors", first / "sample_mmh3_te.safetensors")

    assert _record_cache_index(record, [first, second]) == 0
    with pytest.raises(ValueError, match="exactly one cache"):
        _record_cache_index(record, [tmp_path, first])


def test_record_chain_is_identical_across_streaming_boundaries() -> None:
    records = _records(7)

    uninterrupted = _extend_record_chain(None, records)
    first = _extend_record_chain(None, records[:3])
    resumed = _extend_record_chain(first, records[3:])

    assert resumed == uninterrupted
    assert resumed != _extend_record_chain(None, list(reversed(records)))


def test_router_optimizer_and_rng_resume_exactly(tmp_path: Path) -> None:
    def components(seed: int) -> tuple[torch.nn.Linear, torch.optim.Optimizer, torch.Generator]:
        torch.manual_seed(seed)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(3408)
        return model, optimizer, generator

    def update(model: torch.nn.Linear, optimizer: torch.optim.Optimizer, generator: torch.Generator) -> None:
        value = torch.randn((4, 3), generator=generator)
        target = torch.randn((4, 2), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(value), target).backward()
        optimizer.step()

    invariants = {"learning_rate": 2e-4, "num_layers": 50}
    full_model, full_optimizer, full_generator = components(11)
    for _ in range(6):
        update(full_model, full_optimizer, full_generator)

    split_model, split_optimizer, split_generator = components(11)
    for _ in range(3):
        update(split_model, split_optimizer, split_generator)
    state_path = tmp_path / "train-state.pt"
    record_chain = _extend_record_chain(None, _records(3))
    _atomic_torch_save(
        _capture_training_state(
            split_model,
            split_optimizer,
            split_generator,
            completed_examples=3,
            completed_optimizer_updates=150,
            invariants=invariants,
            record_chain_sha256=record_chain,
            parent_state_sha256=None,
        ),
        state_path,
    )

    resumed_model, resumed_optimizer, resumed_generator = components(999)
    completed, optimizer_updates, restored_chain = _restore_training_state(
        torch.load(state_path, weights_only=True),
        resumed_model,
        resumed_optimizer,
        resumed_generator,
        expected_invariants=invariants,
    )
    assert (completed, optimizer_updates, restored_chain) == (3, 150, record_chain)
    for _ in range(3):
        update(resumed_model, resumed_optimizer, resumed_generator)

    for full, resumed in zip(full_model.parameters(), resumed_model.parameters(), strict=True):
        torch.testing.assert_close(full, resumed, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.rand(8, generator=full_generator),
        torch.rand(8, generator=resumed_generator),
        rtol=0,
        atol=0,
    )


def test_router_resume_rejects_invalid_rng_state() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    generator = torch.Generator().manual_seed(1)
    state = _capture_training_state(
        model,
        optimizer,
        generator,
        completed_examples=0,
        completed_optimizer_updates=0,
        invariants={"num_layers": 50},
        record_chain_sha256=_extend_record_chain(None, []),
        parent_state_sha256=None,
    )
    state["training_generator_state"] = torch.ones(4, dtype=torch.int64)
    with pytest.raises(ValueError, match="RNG state"):
        _restore_training_state(
            state,
            model,
            optimizer,
            generator,
            expected_invariants={"num_layers": 50},
        )


def test_dedicated_generator_contains_backend_global_rng_draws() -> None:
    global_before = torch.random.get_rng_state()
    generator = torch.Generator().manual_seed(123)
    expected = torch.Generator().manual_seed(123)

    actual = _run_with_generator_rng(generator, torch.device("cpu"), lambda: torch.rand(5))

    torch.testing.assert_close(actual, torch.rand(5, generator=expected), rtol=0, atol=0)
    assert torch.equal(generator.get_state(), expected.get_state())
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_router_resume_rejects_changed_training_invariants(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    generator = torch.Generator().manual_seed(1)
    state = _capture_training_state(
        model,
        optimizer,
        generator,
        completed_examples=0,
        completed_optimizer_updates=0,
        invariants={"seed": 3407},
        record_chain_sha256=_extend_record_chain(None, []),
        parent_state_sha256=None,
    )

    with pytest.raises(ValueError, match="invariants differ"):
        _restore_training_state(
            state,
            model,
            optimizer,
            generator,
            expected_invariants={"seed": 9999},
        )


def test_explicit_validation_records_must_be_disjoint_by_cache_identity() -> None:
    training = [(Path("train/source_mmh3.safetensors"), Path("train/source_mmh3_te.safetensors"))]
    validation = [(Path("validation/source_mmh3.safetensors"), Path("validation/source_mmh3_te.safetensors"))]

    with pytest.raises(ValueError, match="caches overlap"):
        _validate_disjoint_records(training, validation)

    _validate_disjoint_records(training, _records(2))


def test_initial_router_accepts_only_legacy_mass_gate_extras() -> None:
    _validate_initial_router_keys(
        [],
        [
            "reference_mass_feature_weight",
            "reference_mass_feature_bias",
            "reference_mass_feature_mean",
            "reference_mass_feature_scale",
        ],
    )

    with pytest.raises(ValueError, match="missing=.*query_layer_projection"):
        _validate_initial_router_keys(["query_layer_projection"], [])
    with pytest.raises(ValueError, match="unexpected=.*unknown"):
        _validate_initial_router_keys([], ["reference_mass_feature_weight", "unknown"])


def test_cache_pairing_supports_image_and_video_latent_suffixes(tmp_path: Path) -> None:
    names = (
        ("sample_000884_1365x2048_mmh3.safetensors", "sample_000884_mmh3_te.safetensors"),
        (
            "multi_0017_00000-124_06008x0352_mmh3.safetensors",
            "multi_0017_00000-124_mmh3_te.safetensors",
        ),
    )
    for latent_name, text_name in names:
        (tmp_path / latent_name).touch()
        (tmp_path / text_name).touch()

    records = _paired_records([tmp_path], seed=7)

    assert {(latent.name, text.name) for latent, text in records} == set(names)


def test_cache_pairing_is_independent_of_file_creation_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    names = [f"sample_{index:02d}" for index in range(6)]
    for stem in names:
        (first / f"{stem}_mmh3.safetensors").touch()
        (first / f"{stem}_mmh3_te.safetensors").touch()
    for stem in reversed(names):
        (second / f"{stem}_mmh3_te.safetensors").touch()
        (second / f"{stem}_mmh3.safetensors").touch()

    first_records = _paired_records([first], seed=19)
    second_records = _paired_records([second], seed=19)

    assert [(latent.name, text.name) for latent, text in first_records] == [
        (latent.name, text.name) for latent, text in second_records
    ]


def test_visual_latent_selection_ignores_audio_target() -> None:
    assert (
        _visual_latent_key(
            ["audio_loss_mask", "latents_37x22x38_float32", "latents_audio_2x32x207_float32"],
            Path("video.safetensors"),
        )
        == "latents_37x22x38_float32"
    )


@pytest.mark.parametrize(
    "text_contract",
    [None, torch.tensor(1), torch.tensor(2), torch.tensor(1, dtype=torch.int32), torch.tensor([1])],
)
def test_online_cache_loader_preserves_reference_contract_metadata(tmp_path: Path, text_contract: torch.Tensor | None) -> None:
    latent_path = tmp_path / "sample_mmh3.safetensors"
    text_path = tmp_path / "sample_mmh3_te.safetensors"
    save_file(
        {
            "latents_1x2x2_float32": torch.zeros(1, 2, 2),
            H3_REFERENCE_KINDS_KEY: torch.tensor([1]),
            H3_REFERENCE_VIDEO_SHAPES_KEY: torch.tensor([[1, 2, 2]]),
            H3_REFERENCE_AUDIO_LENGTHS_KEY: torch.tensor([0]),
            H3_REFERENCE_VIDEO_ROWS_KEY: torch.zeros(4, 8),
            H3_REFERENCE_AUDIO_ROWS_KEY: torch.zeros(0, 2),
            H3_REFERENCE_ALIGNED_KEY: torch.tensor([True]),
            H3_ALIGNED_GUIDE_COUNT_KEY: torch.tensor(1),
            H3_REFERENCE_TEMPORAL_CONTRACT_KEY: torch.tensor(1),
        },
        latent_path,
    )
    save_file(
        {
            H3_TEXT_HIDDEN_KEY: torch.zeros(2, 8),
            H3_TEXT_TOKEN_TAGS_KEY: torch.tensor([1, 1]),
            H3_CONDITIONING_TASK_KEY: torch.tensor(4),
            H3_REFERENCE_ROUTE_KEY: torch.tensor(0),
            H3_REFERENCE_VIDEO_FPS_KEY: torch.tensor(24.0, dtype=torch.float64),
            **({H3_REFERENCE_TEMPORAL_CONTRACT_KEY: text_contract} if text_contract is not None else {}),
        },
        text_path,
    )

    if text_contract is not None and (
        text_contract.dtype != torch.int64 or text_contract.shape != torch.Size([]) or text_contract.item() != 1
    ):
        with pytest.raises(ValueError, match="conflicting mmh3_reference_temporal_contract"):
            _load_example(latent_path, text_path)
        return

    batch, clean = _load_example(latent_path, text_path)

    assert clean.shape == (1, 1, 2, 2)
    for key in (
        H3_REFERENCE_ALIGNED_KEY,
        H3_ALIGNED_GUIDE_COUNT_KEY,
        H3_REFERENCE_TEMPORAL_CONTRACT_KEY,
        H3_REFERENCE_ROUTE_KEY,
        H3_REFERENCE_VIDEO_FPS_KEY,
    ):
        assert key in batch


@pytest.mark.parametrize(
    ("latent_fingerprint", "text_fingerprint", "valid"),
    [(None, None, True), ("same", "same", True), ("latent", None, False), (None, "text", False), ("one", "two", False)],
)
def test_online_cache_loader_requires_matching_sample_fingerprints(
    tmp_path: Path,
    latent_fingerprint: str | None,
    text_fingerprint: str | None,
    valid: bool,
) -> None:
    latent_path = tmp_path / "sample_mmh3.safetensors"
    text_path = tmp_path / "sample_mmh3_te.safetensors"
    save_file(
        {
            "latents_1x2x2_float32": torch.zeros(1, 2, 2),
            H3_REFERENCE_KINDS_KEY: torch.tensor([0]),
            H3_REFERENCE_VIDEO_SHAPES_KEY: torch.tensor([[1, 2, 2]]),
            H3_REFERENCE_AUDIO_LENGTHS_KEY: torch.tensor([0]),
            H3_REFERENCE_VIDEO_ROWS_KEY: torch.zeros(1, 8),
            H3_REFERENCE_AUDIO_ROWS_KEY: torch.zeros(0, 2),
        },
        latent_path,
        metadata={"sample_fingerprint": latent_fingerprint} if latent_fingerprint is not None else None,
    )
    save_file(
        {
            H3_TEXT_HIDDEN_KEY: torch.zeros(2, 8),
            H3_TEXT_TOKEN_TAGS_KEY: torch.tensor([1, 1]),
            H3_CONDITIONING_TASK_KEY: torch.tensor(4),
        },
        text_path,
        metadata={"sample_fingerprint": text_fingerprint} if text_fingerprint is not None else None,
    )

    if valid:
        batch, clean = _load_example(latent_path, text_path)
        assert batch[H3_REFERENCE_KINDS_KEY].tolist() == [0]
        assert clean.shape == (1, 1, 2, 2)
    else:
        with pytest.raises(ValueError, match="do not describe the same sample"):
            _load_example(latent_path, text_path)


def test_online_teacher_requires_each_layer_exactly_once() -> None:
    _validate_layer_sequence([0, 1, 2], 3)

    with pytest.raises(RuntimeError, match="every H3 layer exactly once"):
        _validate_layer_sequence([0, 2], 3)


def test_native_h3_dense_capture_updates_only_router() -> None:
    torch.manual_seed(91)
    attention = MiniMaxH3Attention(hidden_size=16, heads=2, head_dim=8, qk_norm_eps=1e-5).requires_grad_(False)
    router = H3UniversalSparseRouter(
        num_layers=1,
        num_heads=2,
        head_dim=8,
        index_dim=4,
        block_size=128,
        layer_specific_projections=True,
        summary_subblocks=1,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-2)
    teacher = OnlineRouterTeacher(router, optimizer, samples_per_block=2, diagnostic_topk=1)
    attention.reference_sparse_layout = H3ReferenceSparseLayout(reference_start=2, reference_rows=8, rows_per_frame=2)
    attention.reference_sparse_layer_index = 0
    attention.set_reference_router_teacher(teacher)
    before = {name: value.detach().clone() for name, value in router.named_parameters()}

    output = attention(torch.randn(1, 12, 16))

    assert output.shape == (1, 12, 16)
    assert teacher.layer_indices == [0]
    assert teacher.reference_rows == 8
    assert any(not torch.equal(before[name], value) for name, value in router.named_parameters())
    assert all(parameter.grad is None for parameter in attention.parameters())


class _TeacherProbeRouter(torch.nn.Module):
    block_size = 128

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.teacher_mass = None

    def summarize(self, value: torch.Tensor) -> torch.Tensor:
        return value.unflatten(1, (-1, self.block_size)).mean(dim=2)

    def score_summaries(self, query: torch.Tensor, key: torch.Tensor, layer_index: int) -> torch.Tensor:
        blocks = query.shape[1]
        return self.bias + torch.zeros((1, query.shape[2], blocks, blocks), device=query.device)

    def teacher_kl_loss(self, scores: torch.Tensor, teacher_mass: torch.Tensor) -> torch.Tensor:
        self.teacher_mass = teacher_mass.detach().clone()
        return scores.square().mean()


def test_online_teacher_excludes_padded_reference_keys_from_dense_mass() -> None:
    router = _TeacherProbeRouter()
    optimizer = torch.optim.SGD(router.parameters(), lr=0.1)
    teacher = OnlineRouterTeacher(router, optimizer, samples_per_block=1, diagnostic_topk=1)
    teacher.reset(training=False)
    layout = H3ReferenceSparseLayout(reference_start=0, reference_rows=129, rows_per_frame=1)
    query = torch.ones(1, 129, 1, 2)
    key = -torch.ones_like(query)

    teacher(query, key, layout, layer_index=0)

    assert router.teacher_mass is not None
    mass = router.teacher_mass[0, 0]
    torch.testing.assert_close(mass.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(mass[:, 0], torch.full((2,), 128 / 129), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(mass[:, 1], torch.full((2,), 1 / 129), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"validation_examples": 0}, "validation examples"),
        ({"num_shards": 0}, "shard index"),
        ({"exclude_shard_index": 2}, "excluded shard index"),
        ({"save_every": 0}, "save interval"),
        ({"learning_rate": float("nan")}, "learning rate"),
        ({"learning_rate": float("inf")}, "learning rate"),
        ({"samples_per_block": 0}, "teacher sampling"),
        ({"adaln_rank": 0}, "AdaLN rank"),
        ({"epochs": 0}, "epochs"),
        ({"validation_seed": -1}, "validation seed"),
        ({"initial": Path("router.pt"), "resume_state": Path("state.pt")}, "mutually exclusive"),
        ({"expected_resume_sha256": "0" * 64}, "requires --resume-state"),
        ({"no_validation": True}, "requires --skip-final-validation"),
        (
            {"no_validation": True, "skip_final_validation": True, "validation_cache_dir": [Path("held-out")]},
            "forbids --validation-cache-dir",
        ),
    ],
)
def test_online_router_options_fail_before_model_loading(override: dict, message: str) -> None:
    values = {
        "max_train_examples": 10,
        "validation_examples": 2,
        "train_offset": 0,
        "num_shards": 2,
        "shard_index": 0,
        "exclude_shard_index": None,
        "save_every": 5,
        "learning_rate": 2e-4,
        "index_dim": 64,
        "summary_subblocks": 4,
        "samples_per_block": 4,
        "diagnostic_topk": 16,
        "adaln_rank": None,
        "epochs": 1,
        "validation_seed": None,
        "initial": None,
        "resume_state": None,
        "expected_resume_sha256": None,
        "no_validation": False,
        "skip_final_validation": False,
        "validation_cache_dir": None,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        _validate_options(SimpleNamespace(**values))


def test_online_router_outputs_are_written_atomically(tmp_path: Path) -> None:
    checkpoint = tmp_path / "router.pt"
    report = tmp_path / "router.json"

    _atomic_torch_save({"weight": torch.arange(3)}, checkpoint)
    _atomic_write_json({"validation": 128}, report)

    torch.testing.assert_close(torch.load(checkpoint, weights_only=True)["weight"], torch.arange(3))
    assert report.read_text(encoding="utf-8") == '{\n  "validation": 128\n}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_online_router_config_is_json_compatible() -> None:
    config = {
        "cache_dir": [Path("train-a"), Path("train-b")],
        "validation_cache_dir": [Path("validation")],
        "output": Path("router.pt"),
        "nested": (Path("checkpoint.pt"), None),
    }

    converted = _json_compatible(config)

    assert converted == {
        "cache_dir": ["train-a", "train-b"],
        "validation_cache_dir": ["validation"],
        "output": "router.pt",
        "nested": ["checkpoint.pt", None],
    }
    assert json.loads(json.dumps(converted)) == converted
