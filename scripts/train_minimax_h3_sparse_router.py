"""Train the frozen MiniMax H3 sparse-reference router from dense-teacher records.

Each ``sample_*.pt`` record must contain ``layer``, ``query_summary``,
``key_summary``, and ``teacher_mass``. Numeric record indices must use
``source_index * num_layers + layer_index`` so all layers from one source can
remain in the same split. Records are streamed from disk so the dense-teacher
cache does not have to fit in host memory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from musubi_tuner.minimax_h3.sparse_attention import H3UniversalSparseRouter


@dataclass(frozen=True)
class RouterTrainingConfig:
    num_layers: int
    index_dim: int
    summary_subblocks: int
    topk: int
    seed: int
    validation_fraction: float


def _atomic_torch_save(state: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _atomic_write_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_record(path: Path) -> dict[str, torch.Tensor | int]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    required = {"layer", "query_summary", "key_summary", "teacher_mass"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"{path} is missing router-cache fields: {sorted(missing)}")
    query = record["query_summary"]
    key = record["key_summary"]
    teacher = record["teacher_mass"]
    if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
        raise TypeError(f"{path} query/key summaries must be tensors")
    if not isinstance(teacher, torch.Tensor):
        raise TypeError(f"{path} teacher mass must be a tensor")
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError(f"{path} query/key summaries must have matching [B, blocks, heads, dim] shapes")
    if query.shape[0] != 1 or min(query.shape[1:]) <= 0:
        raise ValueError(f"{path} query/key summaries must contain one non-empty batch")
    if not query.is_floating_point() or not key.is_floating_point():
        raise TypeError(f"{path} query/key summaries must use floating-point tensors")
    expected_teacher = (query.shape[2], query.shape[1], key.shape[1])
    if teacher.shape != expected_teacher:
        raise ValueError(f"{path} teacher mass must have shape {expected_teacher}, got {tuple(teacher.shape)}")
    if not teacher.is_floating_point():
        raise TypeError(f"{path} teacher mass must use a floating-point tensor")
    if not torch.isfinite(query).all() or not torch.isfinite(key).all() or not torch.isfinite(teacher).all():
        raise ValueError(f"{path} router-cache tensors must be finite")
    if bool((teacher < 0).any()) or bool((teacher.sum(dim=-1) <= 0).any()):
        raise ValueError(f"{path} teacher mass rows must be non-negative with positive totals")
    if int(record["layer"]) < 0:
        raise ValueError(f"{path} has a negative layer index")
    return record


def _validate_record_set(
    paths: list[Path], first: dict[str, torch.Tensor | int], *, num_layers: int, summary_subblocks: int
) -> None:
    first_query = first["query_summary"]
    assert isinstance(first_query, torch.Tensor)
    expected_geometry = tuple(first_query.shape[2:])
    if expected_geometry[-1] % summary_subblocks:
        raise ValueError("cached summary width must be divisible by --summary-subblocks")
    for path in paths:
        record = first if path == paths[0] else _load_record(path)
        layer = int(record["layer"])
        if layer >= num_layers:
            raise ValueError(f"{path} layer index {layer} exceeds --num-layers {num_layers}")
        record_index = int(path.stem.rsplit("_", 1)[1])
        if layer != record_index % num_layers:
            raise ValueError(f"{path} layer field does not match its numeric record index")
        query = record["query_summary"]
        assert isinstance(query, torch.Tensor)
        if tuple(query.shape[2:]) != expected_geometry:
            raise ValueError(f"{path} head/summary geometry differs from the first router-cache record")


def _split_paths(paths: list[Path], validation_fraction: float, seed: int, num_layers: int) -> tuple[list[Path], list[Path]]:
    groups: dict[int, list[tuple[int, Path]]] = {}
    for path in paths:
        try:
            record_index = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"router-cache filename must end in a numeric record index: {path}") from error
        groups.setdefault(record_index // num_layers, []).append((record_index % num_layers, path))
    if len(groups) < 2:
        raise ValueError("router training needs at least two complete source-example groups")
    expected_layers = set(range(num_layers))
    for group_id, records in groups.items():
        layers = {layer for layer, _ in records}
        if len(records) != num_layers or layers != expected_layers:
            raise ValueError(f"router-cache source group {group_id} does not contain exactly one record per layer")
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    validation_count = max(1, round(len(group_ids) * validation_fraction))
    validation_count = min(validation_count, len(group_ids) - 1)
    validation_ids = set(group_ids[:validation_count])
    train = [path for group_id in group_ids if group_id not in validation_ids for _, path in groups[group_id]]
    validation = [path for group_id in group_ids if group_id in validation_ids for _, path in groups[group_id]]
    return train, validation


def _build_router(
    first: dict[str, torch.Tensor | int],
    *,
    num_layers: int,
    index_dim: int,
    summary_subblocks: int,
    device: torch.device,
) -> H3UniversalSparseRouter:
    query = first["query_summary"]
    assert isinstance(query, torch.Tensor)
    if query.shape[-1] % summary_subblocks:
        raise ValueError("cached summary width must be divisible by --summary-subblocks")
    return H3UniversalSparseRouter(
        num_layers=num_layers,
        num_heads=query.shape[2],
        head_dim=query.shape[3] // summary_subblocks,
        index_dim=index_dim,
        layer_specific_projections=True,
        summary_subblocks=summary_subblocks,
    ).to(device)


@torch.no_grad()
def _validate(
    router: H3UniversalSparseRouter,
    paths: list[Path],
    device: torch.device,
    topk: int,
) -> dict[str, float]:
    router.eval()
    losses: list[float] = []
    recalls: list[float] = []
    retained_mass: list[float] = []
    for path in paths:
        record = _load_record(path)
        query = record["query_summary"].to(device, non_blocking=True)
        key = record["key_summary"].to(device, non_blocking=True)
        teacher = record["teacher_mass"].to(device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
        scores = router.score_summaries(query, key, int(record["layer"]))
        losses.append(float(router.teacher_kl_loss(scores, teacher)))
        count = min(topk, scores.shape[-1])
        predicted = scores.topk(count, dim=-1).indices
        target = teacher.topk(count, dim=-1).indices
        recalls.append(float((predicted.unsqueeze(-1) == target.unsqueeze(-2)).any(dim=-1).float().mean()))
        normalized_teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        retained_mass.append(float(normalized_teacher.gather(-1, predicted).sum(dim=-1).mean()))
    return {
        "validation_kl": sum(losses) / len(losses),
        f"validation_top{topk}_recall": sum(recalls) / len(recalls),
        f"validation_top{topk}_teacher_mass": sum(retained_mass) / len(retained_mass),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-layers", type=int, default=50)
    parser.add_argument("--index-dim", type=int, default=64)
    parser.add_argument("--summary-subblocks", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if min(args.epochs, args.validation_every, args.topk, args.num_layers, args.index_dim, args.summary_subblocks) <= 0:
        parser.error("--epochs, --validation-every, --topk, --num-layers, --index-dim, and --summary-subblocks must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and non-negative")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must lie in (0, 1)")

    paths = sorted(args.cache_dir.glob("sample_*.pt"))
    if len(paths) < 2:
        raise RuntimeError(f"need at least two router-cache records in {args.cache_dir}")
    first = _load_record(paths[0])
    _validate_record_set(paths, first, num_layers=args.num_layers, summary_subblocks=args.summary_subblocks)
    train_paths, validation_paths = _split_paths(paths, args.validation_fraction, args.seed, args.num_layers)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    router = _build_router(
        first,
        num_layers=args.num_layers,
        index_dim=args.index_dim,
        summary_subblocks=args.summary_subblocks,
        device=device,
    )
    if args.initial is not None:
        router.load_state_dict(torch.load(args.initial, map_location=device, weights_only=True), strict=True)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    history: list[dict[str, float | int]] = []
    best_validation_kl = math.inf
    best_epoch = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    training_config = RouterTrainingConfig(
        num_layers=args.num_layers,
        index_dim=args.index_dim,
        summary_subblocks=args.summary_subblocks,
        topk=args.topk,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )

    print(f"router records: train={len(train_paths)} validation={len(validation_paths)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        router.train()
        rng.shuffle(train_paths)
        total_loss = 0.0
        for path in train_paths:
            record = _load_record(path)
            query = record["query_summary"].to(device, non_blocking=True)
            key = record["key_summary"].to(device, non_blocking=True)
            teacher = record["teacher_mass"].to(device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
            scores = router.score_summaries(query, key, int(record["layer"]))
            loss = router.teacher_kl_loss(scores, teacher)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())

        row: dict[str, float | int] = {"epoch": epoch, "train_kl": total_loss / len(train_paths)}
        if epoch % args.validation_every == 0 or epoch == args.epochs:
            row.update(_validate(router, validation_paths, device, args.topk))
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if "validation_kl" in row and float(row["validation_kl"]) < best_validation_kl:
            best_validation_kl = float(row["validation_kl"])
            best_epoch = epoch
            _atomic_torch_save(router.state_dict(), args.output)
        _atomic_write_json(
            {
                "schema_version": 1,
                "config": asdict(training_config),
                "train_records": len(train_paths),
                "validation_records": len(validation_paths),
                "best_epoch": best_epoch,
                "best_validation_kl": best_validation_kl if best_epoch else None,
                "history": history,
            },
            report_path,
        )

    print(f"wrote best checkpoint from epoch {best_epoch} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
