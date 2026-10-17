import json
import sys
from pathlib import Path

import pytest
import torch

from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.sparse_attention import H3ReferenceSparseConfig, H3UniversalSparseRouter
from scripts.train_minimax_h3_sparse_router import (
    _atomic_torch_save,
    _atomic_write_json,
    _load_record,
    _split_paths,
    _validate_record_set,
    main,
)


def _record(layer: int, *, width: int = 8) -> dict[str, torch.Tensor | int]:
    return {
        "layer": layer,
        "query_summary": torch.randn(1, 2, 1, width),
        "key_summary": torch.randn(1, 2, 1, width),
        "teacher_mass": torch.rand(1, 2, 2).add_(0.1),
    }


def test_router_cache_split_keeps_all_layers_from_one_source_together() -> None:
    paths = [Path(f"sample_{source * 50 + layer:06d}.pt") for source in range(4) for layer in range(50)]

    train, validation = _split_paths(paths, validation_fraction=0.25, seed=1234, num_layers=50)

    train_sources = {int(path.stem.rsplit("_", 1)[1]) // 50 for path in train}
    validation_sources = {int(path.stem.rsplit("_", 1)[1]) // 50 for path in validation}
    assert train_sources.isdisjoint(validation_sources)
    assert len(train) == 150
    assert len(validation) == 50


def test_router_cache_split_rejects_non_numeric_record_names() -> None:
    with pytest.raises(ValueError, match="numeric record index"):
        _split_paths(
            [Path("sample_bad.pt"), Path("sample_000050.pt")],
            validation_fraction=0.5,
            seed=1234,
            num_layers=50,
        )


def test_router_cache_split_rejects_incomplete_source_groups() -> None:
    paths = [Path(f"sample_{layer:06d}.pt") for layer in range(49)]
    paths += [Path(f"sample_{50 + layer:06d}.pt") for layer in range(50)]

    with pytest.raises(ValueError, match="exactly one record per layer"):
        _split_paths(paths, validation_fraction=0.5, seed=1234, num_layers=50)


def test_router_checkpoint_and_report_writes_are_atomic(tmp_path: Path) -> None:
    checkpoint = tmp_path / "router.pt"
    report = tmp_path / "router.pt.json"

    _atomic_torch_save({"weight": torch.arange(3)}, checkpoint)
    _atomic_write_json({"schema_version": 1}, report)

    torch.testing.assert_close(torch.load(checkpoint, weights_only=True)["weight"], torch.arange(3))
    assert json.loads(report.read_text(encoding="utf-8")) == {"schema_version": 1}
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.pop("teacher_mass"), "missing router-cache fields"),
        (lambda record: record.__setitem__("key_summary", torch.randn(1, 3, 1, 8)), "matching"),
        (lambda record: record.__setitem__("teacher_mass", torch.zeros(1, 2, 2)), "positive totals"),
        (lambda record: record["query_summary"].fill_(float("nan")), "finite"),
    ],
)
def test_router_record_validation_rejects_malformed_data(tmp_path: Path, mutation, message: str) -> None:
    path = tmp_path / "sample_000000.pt"
    record = _record(0)
    mutation(record)
    torch.save(record, path)

    with pytest.raises((TypeError, ValueError), match=message):
        _load_record(path)


def test_router_record_set_rejects_filename_and_geometry_mismatches(tmp_path: Path) -> None:
    first_path = tmp_path / "sample_000000.pt"
    second_path = tmp_path / "sample_000001.pt"
    torch.save(_record(0), first_path)
    torch.save(_record(0, width=16), second_path)
    first = _load_record(first_path)

    with pytest.raises(ValueError, match="layer field does not match"):
        _validate_record_set([first_path, second_path], first, num_layers=2, summary_subblocks=1)

    torch.save(_record(1, width=16), second_path)
    with pytest.raises(ValueError, match="geometry differs"):
        _validate_record_set([first_path, second_path], first, num_layers=2, summary_subblocks=1)


def test_router_trainer_writes_best_checkpoint_and_structured_report(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    generator = torch.Generator().manual_seed(1234)
    for source in range(2):
        for layer in range(2):
            torch.save(
                {
                    "layer": layer,
                    "query_summary": torch.randn(1, 2, 1, 128, generator=generator),
                    "key_summary": torch.randn(1, 2, 1, 128, generator=generator),
                    "teacher_mass": torch.rand(1, 2, 2, generator=generator),
                },
                cache / f"sample_{source * 2 + layer:06d}.pt",
            )
    output = tmp_path / "router.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_minimax_h3_sparse_router.py",
            "--cache-dir",
            str(cache),
            "--output",
            str(output),
            "--epochs",
            "2",
            "--num-layers",
            "2",
            "--index-dim",
            "4",
            "--summary-subblocks",
            "1",
            "--validation-fraction",
            "0.5",
            "--topk",
            "1",
            "--device",
            "cpu",
        ],
    )

    main()

    state = torch.load(output, map_location="cpu", weights_only=True)
    assert state["query_layer_projection"].shape == (2, 128, 4)
    report = json.loads(output.with_suffix(".pt.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["train_records"] == report["validation_records"] == 2
    assert report["best_epoch"] in (1, 2)
    assert len(report["history"]) == 2
    assert list(tmp_path.glob("*.tmp")) == []

    second_output = tmp_path / "router_second.pt"
    sys.argv[sys.argv.index(str(output))] = str(second_output)
    main()
    second_state = torch.load(second_output, map_location="cpu", weights_only=True)
    assert state.keys() == second_state.keys()
    for key in state:
        torch.testing.assert_close(state[key], second_state[key], rtol=0, atol=0)

    router = H3UniversalSparseRouter(
        num_layers=2,
        num_heads=1,
        head_dim=128,
        index_dim=4,
        block_size=128,
        layer_specific_projections=True,
    )
    router.load_state_dict(state, strict=True)
    transformer = MiniMaxH3Transformer(
        MiniMaxH3TransformerConfig(
            num_attention_heads=1,
            attention_head_dim=128,
            hidden_size=128,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=256,
            in_channels=1,
            audio_in_channels=2,
            patch_size=(1, 1, 1),
            text_dim=8,
            freq_dim=8,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=8,
        )
    )
    transformer.enable_reference_sparse_attention(H3ReferenceSparseConfig(layer_ranges=((0, 2),)), router=router)
    assert transformer.reference_sparse_router is router
