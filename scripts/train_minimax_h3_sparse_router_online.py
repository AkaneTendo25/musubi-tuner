"""Train an H3 reference router from dense Q/K over paired Ref2VA caches.

Each dense layer produces one router update and releases its teacher tensors;
Q/K and attention-mass tensors are not written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.minimax_h3.cache import (
    H3_ALIGNED_GUIDE_COUNT_KEY,
    H3_CONDITIONING_TASK_KEY,
    H3_REFERENCE_ALIGNED_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
    H3_REFERENCE_IMAGE_MAX_PIXELS_KEY,
    H3_REFERENCE_IMAGE_SHORT_EDGE_KEY,
    H3_REFERENCE_IMAGE_SIZE_MODE_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_ROUTE_KEY,
    H3_REFERENCE_TEMPORAL_CONTRACT_KEY,
    H3_REFERENCE_VIDEO_FPS_KEY,
    H3_REFERENCE_VIDEO_MAX_PIXELS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHORT_EDGE_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
from musubi_tuner.minimax_h3.model_loader import load_transformer
from musubi_tuner.minimax_h3.sparse_attention import H3ReferenceSparseConfig, H3UniversalSparseRouter


TRAINING_STATE_VERSION = 1
TRAINING_RECORD_CHAIN_DOMAIN = hashlib.sha256(b"h3-sra-router-record-chain-v1\n").hexdigest()
_T = TypeVar("_T")


def _run_with_generator_rng(generator: torch.Generator, device: torch.device, function: Callable[[], _T]) -> _T:
    """Run global Torch draws on a dedicated generator without changing global RNG state."""
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        if device.type == "cuda":
            torch.cuda.set_rng_state(generator.get_state(), device)
        else:
            torch.random.set_rng_state(generator.get_state())
        result = function()
        state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()
        generator.set_state(state)
        return result


def _tensor_with_prefix(tensors: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    matches = [value for key, value in tensors.items() if key == prefix or key.startswith(f"varlen_{prefix}_")]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one tensor for {prefix}, found {len(matches)}")
    return matches[0]


def _optional_tensor_with_prefix(tensors: dict[str, torch.Tensor], prefix: str) -> torch.Tensor | None:
    matches = [value for key, value in tensors.items() if key == prefix or key.startswith(f"varlen_{prefix}_")]
    if len(matches) > 1:
        raise ValueError(f"expected at most one tensor for {prefix}, found {len(matches)}")
    return matches[0] if matches else None


def _visual_latent_key(keys: list[str], source: Path) -> str:
    matches = [key for key in keys if key.startswith("latents_") and not key.startswith("latents_audio_")]
    if len(matches) != 1:
        raise ValueError(f"{source} has {len(matches)} visual target latent tensors")
    return matches[0]


def _paired_records(cache_dirs: list[Path], seed: int) -> list[tuple[Path, Path]]:
    records = []
    for cache_dir in cache_dirs:
        text_by_id = {path.name.split("_mmh3_te", 1)[0]: path for path in sorted(cache_dir.glob("*_mmh3_te.safetensors"))}
        for latent_path in sorted(cache_dir.glob("*_mmh3.safetensors")):
            latent_id = latent_path.stem.removesuffix("_mmh3")
            parts = latent_id.split("_")
            text_path = next(
                (text_by_id[source_id] for stop in range(len(parts), 0, -1) if (source_id := "_".join(parts[:stop])) in text_by_id),
                None,
            )
            if text_path is not None:
                records.append((latent_path, text_path))
    random.Random(seed).shuffle(records)
    if not records:
        raise RuntimeError("no paired H3 latent/text cache records found")
    return records


def _select_training_records(
    records: list[tuple[Path, Path]],
    *,
    validation_examples: int,
    train_offset: int,
    num_shards: int,
    shard_index: int,
    exclude_shard_index: int | None,
    max_train_examples: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    validation = records[:validation_examples]
    candidates = records[validation_examples + train_offset :]
    if exclude_shard_index is None:
        training = candidates[shard_index::num_shards]
    else:
        training = [record for index, record in enumerate(candidates) if index % num_shards != exclude_shard_index]
    return training[:max_train_examples], validation


def _record_identity(record: tuple[Path, Path]) -> tuple[str, str]:
    return record[0].name, record[1].name


def _validate_disjoint_records(training: list[tuple[Path, Path]], validation: list[tuple[Path, Path]]) -> None:
    overlap = {_record_identity(record) for record in training} & {_record_identity(record) for record in validation}
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(f"training and validation caches overlap at {example[0]} / {example[1]}")


def _record_cache_index(record: tuple[Path, Path], cache_dirs: list[Path]) -> int:
    latent = record[0].resolve()
    matches = [index for index, cache_dir in enumerate(cache_dirs) if latent.is_relative_to(cache_dir.resolve())]
    if len(matches) != 1:
        raise ValueError(f"validation record must belong to exactly one cache directory: {record[0]}")
    return matches[0]


def _validate_initial_router_keys(missing: list[str], unexpected: list[str]) -> None:
    legacy_mass_gate = {
        "reference_mass_feature_weight",
        "reference_mass_feature_bias",
        "reference_mass_feature_mean",
        "reference_mass_feature_scale",
    }
    incompatible_unexpected = [key for key in unexpected if key not in legacy_mass_gate]
    if missing or incompatible_unexpected:
        raise ValueError(f"incompatible initial router: missing={missing}, unexpected={incompatible_unexpected}")


def _repeat_training_records(records: list[tuple[Path, Path]], *, epochs: int, seed: int) -> list[tuple[Path, Path]]:
    repeated = list(records)
    for epoch in range(1, epochs):
        epoch_records = list(records)
        random.Random(seed + epoch).shuffle(epoch_records)
        repeated.extend(epoch_records)
    return repeated


def _atomic_torch_save(state: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_compatible(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _resolved_generator_seeds(seed: int, validation_seed: int | None) -> tuple[int, int]:
    return seed + 1, validation_seed if validation_seed is not None else seed + 1_000_003


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extend_record_chain(parent: str | None, records: list[tuple[Path, Path]]) -> str:
    digest = parent or TRAINING_RECORD_CHAIN_DOMAIN
    if len(digest) != 64:
        raise ValueError("record-chain parent must be a SHA-256 digest")
    try:
        bytes.fromhex(digest)
    except ValueError as error:
        raise ValueError("record-chain parent must be a SHA-256 digest") from error
    for latent_path, text_path in records:
        identity = f"{latent_path.name}\0{text_path.name}".encode()
        digest = hashlib.sha256(bytes.fromhex(digest) + identity).hexdigest()
    return digest


def _training_state_invariants(args: argparse.Namespace, model, router: H3UniversalSparseRouter) -> dict[str, object]:
    training_seed, _ = _resolved_generator_seeds(args.seed, args.validation_seed)
    return {
        "adaln_rank": args.adaln_rank,
        "base_precision": args.base_precision,
        "block_size": router.block_size,
        "diagnostic_topk": args.diagnostic_topk,
        "head_dim": model.config.attention_head_dim,
        "index_dim": args.index_dim,
        "layer_specific_projections": True,
        "learning_rate": args.learning_rate,
        "num_heads": model.config.num_attention_heads,
        "num_layers": len(model.blocks),
        "resolved_training_seed": training_seed,
        "samples_per_block": args.samples_per_block,
        "seed": args.seed,
        "summary_subblocks": args.summary_subblocks,
        "weight_decay": 1e-4,
    }


def _capture_training_state(
    router: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    *,
    completed_examples: int,
    completed_optimizer_updates: int,
    invariants: dict[str, object],
    record_chain_sha256: str,
    parent_state_sha256: str | None,
) -> dict[str, object]:
    return {
        "version": TRAINING_STATE_VERSION,
        "router": router.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_generator_state": training_generator.get_state(),
        "completed_examples": completed_examples,
        "completed_optimizer_updates": completed_optimizer_updates,
        "invariants": invariants,
        "record_chain_sha256": record_chain_sha256,
        "parent_state_sha256": parent_state_sha256,
    }


def _restore_training_state(
    state: object,
    router: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    *,
    expected_invariants: dict[str, object],
) -> tuple[int, int, str]:
    if not isinstance(state, dict) or state.get("version") != TRAINING_STATE_VERSION:
        raise ValueError("unsupported router training-state version")
    actual_invariants = state.get("invariants")
    if (
        isinstance(actual_invariants, dict)
        and "base_precision" in expected_invariants
        and "base_precision" not in actual_invariants
    ):
        actual_invariants = dict(actual_invariants)
        actual_invariants["base_precision"] = "fp8"
    if actual_invariants != expected_invariants:
        keys = sorted(set(expected_invariants) | set(actual_invariants or {}))
        mismatches = {
            key: {"expected": expected_invariants.get(key), "actual": (actual_invariants or {}).get(key)}
            for key in keys
            if expected_invariants.get(key) != (actual_invariants or {}).get(key)
        }
        raise ValueError(f"router training-state invariants differ: {mismatches}")
    completed_examples = state.get("completed_examples")
    completed_optimizer_updates = state.get("completed_optimizer_updates")
    record_chain_sha256 = state.get("record_chain_sha256")
    if not isinstance(completed_examples, int) or completed_examples < 0:
        raise ValueError("invalid completed example count in router training state")
    if not isinstance(completed_optimizer_updates, int) or completed_optimizer_updates < 0:
        raise ValueError("invalid optimizer update count in router training state")
    num_layers = expected_invariants.get("num_layers")
    if not isinstance(num_layers, int) or completed_optimizer_updates != completed_examples * num_layers:
        raise ValueError("router training-state optimizer update count is inconsistent")
    if not isinstance(record_chain_sha256, str):
        raise ValueError("missing record-chain digest in router training state")
    _extend_record_chain(record_chain_sha256, [])
    router.load_state_dict(state["router"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    generator_state = state.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8 or generator_state.ndim != 1:
        raise ValueError("invalid training-generator RNG state")
    # map_location=device also moves this byte tensor to CUDA, while
    # Generator.set_state requires a CPU ByteTensor for CUDA generators too.
    training_generator.set_state(generator_state.cpu())
    return completed_examples, completed_optimizer_updates, record_chain_sha256


def _validate_layer_sequence(layer_indices: list[int], num_layers: int) -> None:
    expected = list(range(num_layers))
    if layer_indices != expected:
        raise RuntimeError(
            f"online router teacher must observe every H3 layer exactly once; expected {expected}, got {layer_indices}"
        )


def _validate_options(args: argparse.Namespace) -> None:
    if args.max_train_examples < 0:
        raise ValueError("max train examples must be non-negative")
    if args.validation_examples <= 0:
        raise ValueError("validation examples must be positive")
    if getattr(args, "no_validation", False) and not getattr(args, "skip_final_validation", False):
        raise ValueError("--no-validation requires --skip-final-validation")
    if getattr(args, "no_validation", False) and getattr(args, "validation_cache_dir", None):
        raise ValueError("--no-validation forbids --validation-cache-dir")
    if args.train_offset < 0:
        raise ValueError("train offset must be non-negative")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    if args.exclude_shard_index is not None and not 0 <= args.exclude_shard_index < args.num_shards:
        raise ValueError("excluded shard index must be in [0, num_shards)")
    if args.save_every <= 0:
        raise ValueError("save interval must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if args.index_dim <= 0 or args.summary_subblocks <= 0:
        raise ValueError("router dimensions must be positive")
    if args.samples_per_block <= 0 or args.diagnostic_topk <= 0:
        raise ValueError("teacher sampling and diagnostic top-k must be positive")
    if args.adaln_rank is not None and args.adaln_rank <= 0:
        raise ValueError("AdaLN rank must be positive")
    if getattr(args, "epochs", 1) <= 0:
        raise ValueError("epochs must be positive")
    if getattr(args, "validation_seed", None) is not None and args.validation_seed < 0:
        raise ValueError("validation seed must be non-negative")
    if getattr(args, "resume_state", None) is not None and getattr(args, "initial", None) is not None:
        raise ValueError("--resume-state and --initial are mutually exclusive")
    expected_resume_sha256 = getattr(args, "expected_resume_sha256", None)
    if expected_resume_sha256 is not None:
        if getattr(args, "resume_state", None) is None:
            raise ValueError("--expected-resume-sha256 requires --resume-state")
        if len(expected_resume_sha256) != 64:
            raise ValueError("expected resume SHA-256 must be a 64-character digest")
        try:
            bytes.fromhex(expected_resume_sha256)
        except ValueError as error:
            raise ValueError("expected resume SHA-256 must be a hexadecimal digest") from error


def _load_example(latent_path: Path, text_path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    with safe_open(latent_path, framework="pt", device="cpu") as handle:
        latent_fingerprint = (handle.metadata() or {}).get("sample_fingerprint")
    with safe_open(text_path, framework="pt", device="cpu") as handle:
        text_fingerprint = (handle.metadata() or {}).get("sample_fingerprint")
    if (latent_fingerprint or text_fingerprint) and latent_fingerprint != text_fingerprint:
        raise ValueError(
            f"MiniMax H3 latent/text caches do not describe the same sample; rebuild both caches: {latent_path} and {text_path}"
        )
    latent = load_file(latent_path)
    text = load_file(text_path)
    latent_key = _visual_latent_key(list(latent), latent_path)
    batch = {
        H3_REFERENCE_KINDS_KEY: _tensor_with_prefix(latent, H3_REFERENCE_KINDS_KEY),
        H3_REFERENCE_VIDEO_SHAPES_KEY: _tensor_with_prefix(latent, H3_REFERENCE_VIDEO_SHAPES_KEY),
        H3_REFERENCE_AUDIO_LENGTHS_KEY: _tensor_with_prefix(latent, H3_REFERENCE_AUDIO_LENGTHS_KEY),
        H3_REFERENCE_VIDEO_ROWS_KEY: _tensor_with_prefix(latent, H3_REFERENCE_VIDEO_ROWS_KEY),
        H3_REFERENCE_AUDIO_ROWS_KEY: _tensor_with_prefix(latent, H3_REFERENCE_AUDIO_ROWS_KEY),
        H3_TEXT_HIDDEN_KEY: _tensor_with_prefix(text, H3_TEXT_HIDDEN_KEY),
        H3_TEXT_TOKEN_TAGS_KEY: _tensor_with_prefix(text, H3_TEXT_TOKEN_TAGS_KEY),
        H3_CONDITIONING_TASK_KEY: text[H3_CONDITIONING_TASK_KEY],
    }
    metadata_keys = (
        H3_REFERENCE_ALIGNED_KEY,
        H3_ALIGNED_GUIDE_COUNT_KEY,
        H3_REFERENCE_IMAGE_SHORT_EDGE_KEY,
        H3_REFERENCE_IMAGE_SIZE_MODE_KEY,
        H3_REFERENCE_IMAGE_MAX_PIXELS_KEY,
        H3_REFERENCE_VIDEO_SHORT_EDGE_KEY,
        H3_REFERENCE_VIDEO_MAX_PIXELS_KEY,
        H3_REFERENCE_VIDEO_FPS_KEY,
        H3_REFERENCE_TEMPORAL_CONTRACT_KEY,
        H3_REFERENCE_ROUTE_KEY,
    )
    for key in metadata_keys:
        values = [value for tensors in (latent, text) if (value := _optional_tensor_with_prefix(tensors, key)) is not None]
        if len(values) > 1 and (
            values[0].dtype != values[1].dtype or values[0].shape != values[1].shape or not torch.equal(values[0], values[1])
        ):
            raise ValueError(f"{latent_path} and {text_path} contain conflicting {key}")
        if values:
            batch[key] = values[0]
    return batch, latent[latent_key].unsqueeze(0)


class OnlineRouterTeacher:
    def __init__(
        self,
        router: H3UniversalSparseRouter,
        optimizer: torch.optim.Optimizer,
        *,
        samples_per_block: int,
        diagnostic_topk: int,
    ) -> None:
        self.router = router
        self.optimizer = optimizer
        self.samples_per_block = samples_per_block
        self.diagnostic_topk = diagnostic_topk
        self.training = True
        self.rows: list[dict[str, float]] = []
        self.layer_indices: list[int] = []
        self.reference_rows: int | None = None
        self.sequence_rows: int | None = None

    def reset(self, training: bool) -> None:
        self.training = training
        self.rows.clear()
        self.layer_indices.clear()
        self.reference_rows = None
        self.sequence_rows = None

    def __call__(self, query, key, layout, layer_index: int) -> None:
        self.layer_indices.append(layer_index)
        start = layout.reference_start
        rows = layout.reference_rows
        stop = start + rows
        if self.reference_rows is None:
            self.reference_rows = rows
            self.sequence_rows = query.shape[1]
        elif self.reference_rows != rows or self.sequence_rows != query.shape[1]:
            raise RuntimeError("H3 reference geometry changed between attention layers")
        block_size = self.router.block_size
        blocks = math.ceil(rows / block_size)
        query_ref = query[:, start:stop].detach()
        key_ref = key[:, start:stop].detach()
        padded_rows = blocks * block_size
        if padded_rows != rows:
            query_ref = torch.nn.functional.pad(query_ref, (0, 0, 0, 0, 0, padded_rows - rows))
            key_ref = torch.nn.functional.pad(key_ref, (0, 0, 0, 0, 0, padded_rows - rows))

        teacher = torch.empty((query.shape[2], blocks, blocks), device=query.device, dtype=torch.float32)
        scale = query.shape[-1] ** -0.5
        for query_block in range(blocks):
            first = query_block * block_size
            last = min(first + block_size, rows)
            count = min(self.samples_per_block, last - first)
            offsets = torch.linspace(first, last - 1, count, device=query.device).round().long()
            sampled_query = query_ref[0].index_select(0, offsets)
            logits = torch.einsum("shd,khd->hsk", sampled_query, key_ref[0]).float().mul_(scale)
            if padded_rows != rows:
                logits[..., rows:] = -torch.inf
            probabilities = logits.softmax(dim=-1)
            teacher[:, query_block] = probabilities.unflatten(-1, (blocks, block_size)).sum(dim=-1).mean(dim=1)

        with torch.enable_grad():
            query_summary = self.router.summarize(query_ref)
            key_summary = self.router.summarize(key_ref)
            scores = self.router.score_summaries(query_summary, key_summary, layer_index)
            loss = self.router.teacher_kl_loss(scores, teacher.unsqueeze(0))
            if self.training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
                self.optimizer.step()

        with torch.no_grad():
            topk = min(self.diagnostic_topk, blocks)
            selected = scores.detach()[0].topk(topk, dim=-1).indices
            selected_mask = torch.zeros_like(teacher, dtype=torch.bool).scatter_(2, selected, True)
            normalized = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            retained = normalized.masked_fill(~selected_mask, 0).sum(dim=-1).mean()
            target = teacher.topk(topk, dim=-1).indices
            recall = (selected.unsqueeze(-1) == target.unsqueeze(-2)).any(dim=-1).float().mean()
            self.rows.append({"loss": float(loss.detach()), "retained_mass": float(retained), "recall": float(recall)})

    def summary(self) -> dict[str, float]:
        return {key: sum(row[key] for row in self.rows) / len(self.rows) for key in ("loss", "retained_mass", "recall")}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the H3 sparse-reference router online from dense Q/K over paired Ref2VA caches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache-dir", type=Path, action="append", required=True, help="Repeatable H3 cache directory.")
    parser.add_argument(
        "--validation-cache-dir",
        type=Path,
        action="append",
        help="Repeatable disjoint validation cache directory; otherwise validation is split from --cache-dir.",
    )
    parser.add_argument("--dit", type=Path, required=True, help="Dense Ref2VA teacher checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output router state-dict path.")
    parser.add_argument("--initial", type=Path, help="Compatible router checkpoint used for initialization.")
    parser.add_argument(
        "--resume-state",
        type=Path,
        help="Atomic router/AdamW/training-RNG state from the immediately preceding streaming shard.",
    )
    parser.add_argument(
        "--expected-resume-sha256",
        help="Required SHA-256 of --resume-state; binds a streaming replay to the selected state chain.",
    )
    parser.add_argument(
        "--state-output",
        type=Path,
        help="Final resumable training state; defaults to OUTPUT with -train-state before its suffix.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=4000,
        help="Maximum unique training examples selected before --epochs repetition.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Deterministic passes over the selected training examples.")
    parser.add_argument("--validation-examples", type=int, default=128, help="Held-out prefix size after seeded shuffle.")
    parser.add_argument("--skip-final-validation", action="store_true", help="Skip the dense held-out pass.")
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Replay-only mode: do not reserve or load validation data; requires --skip-final-validation.",
    )
    parser.add_argument("--train-offset", type=int, default=0, help="Offset applied after removing validation records.")
    parser.add_argument("--num-shards", type=int, default=1, help="Stride for partitioning training records.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard selected for training.")
    parser.add_argument(
        "--exclude-shard-index",
        type=int,
        help="Select the complement of this shard instead of --shard-index; useful for sequential continuation.",
    )
    parser.add_argument("--save-every", type=int, default=100, help="Examples between atomic intermediate checkpoints.")
    parser.add_argument("--seed", type=int, default=3407, help="Record shuffle, router initialization, and noise seed.")
    parser.add_argument(
        "--validation-seed",
        type=int,
        help="Independent held-out noise seed; defaults to --seed + 1000003 and never advances during training.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--index-dim", type=int, default=64, help="Router projection dimension.")
    parser.add_argument("--adaln-rank", type=int, help="AdaLN rank of a pruned teacher checkpoint.")
    parser.add_argument("--summary-subblocks", type=int, default=4, help="Summaries retained per 128-row router block.")
    parser.add_argument("--samples-per-block", type=int, default=4, help="Dense teacher query rows sampled per block.")
    parser.add_argument("--diagnostic-topk", type=int, default=16, help="Top-k used for reported mass and recall.")
    parser.add_argument(
        "--base-precision",
        choices=("fp8", "bf16"),
        default="fp8",
        help="Dense teacher weight precision (default: fp8).",
    )
    return parser


def _load_dense_teacher(args: argparse.Namespace, device: torch.device):
    return load_transformer(
        args.dit,
        mode="ref2va",
        loading_device=device,
        quantization_device=device,
        fp8_scaled=args.base_precision == "fp8",
        adaln_rank=args.adaln_rank,
    )


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    _validate_options(args)
    resume_state_sha256 = None
    if args.resume_state is not None:
        resume_state_sha256 = _sha256_file(args.resume_state)
        if args.expected_resume_sha256 is not None and resume_state_sha256 != args.expected_resume_sha256.lower():
            raise ValueError(
                f"resume state SHA-256 mismatch: expected {args.expected_resume_sha256.lower()}, got {resume_state_sha256}"
            )
    state_output = args.state_output or args.output.with_name(f"{args.output.stem}-train-state{args.output.suffix}")
    torch.manual_seed(args.seed)
    records = _paired_records(args.cache_dir, args.seed)
    split_validation_examples = 0 if args.validation_cache_dir or args.no_validation else args.validation_examples
    unique_training, validation = _select_training_records(
        records,
        validation_examples=split_validation_examples,
        train_offset=args.train_offset,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        exclude_shard_index=args.exclude_shard_index,
        max_train_examples=args.max_train_examples,
    )
    if args.validation_cache_dir and not args.no_validation:
        validation = _paired_records(args.validation_cache_dir, args.seed)[: args.validation_examples]
        _validate_disjoint_records(unique_training, validation)
    training = _repeat_training_records(unique_training, epochs=args.epochs, seed=args.seed)
    if not validation and not args.no_validation:
        raise RuntimeError("validation split must be non-empty")
    print(
        json.dumps({"unique_train": len(unique_training), "train_updates": len(training), "validation": len(validation)}),
        flush=True,
    )

    device = torch.device("cuda")
    model = _load_dense_teacher(args, device).eval().requires_grad_(False)
    router = H3UniversalSparseRouter(
        num_layers=len(model.blocks),
        num_heads=model.config.num_attention_heads,
        head_dim=model.config.attention_head_dim,
        index_dim=args.index_dim,
        block_size=128,
        layer_specific_projections=True,
        summary_subblocks=args.summary_subblocks,
    ).to(device)
    if args.initial:
        missing, unexpected = router.load_state_dict(torch.load(args.initial, map_location=device, weights_only=True), strict=False)
        _validate_initial_router_keys(list(missing), list(unexpected))
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    training_seed, validation_seed = _resolved_generator_seeds(args.seed, args.validation_seed)
    training_generator = torch.Generator(device=device).manual_seed(training_seed)
    validation_generator = torch.Generator(device=device).manual_seed(validation_seed)
    invariants = _training_state_invariants(args, model, router)
    completed_examples = 0
    completed_optimizer_updates = 0
    parent_record_chain = None
    if args.resume_state is not None:
        resumed = torch.load(args.resume_state, map_location=device, weights_only=True)
        completed_examples, completed_optimizer_updates, parent_record_chain = _restore_training_state(
            resumed,
            router,
            optimizer,
            training_generator,
            expected_invariants=invariants,
        )
    teacher = OnlineRouterTeacher(
        router,
        optimizer,
        samples_per_block=args.samples_per_block,
        diagnostic_topk=args.diagnostic_topk,
    )
    # Request Ref2VA geometry construction while keeping every attention layer dense.
    model.enable_reference_sparse_attention(
        H3ReferenceSparseConfig(layer_ranges=((len(model.blocks), len(model.blocks) + 1),)), None
    )
    for block in model.blocks:
        block.attn.set_reference_router_teacher(teacher)
    backend = _NativeTrainingBackend(model, "ref2va")
    history = []

    def run_record(
        record: tuple[Path, Path],
        *,
        training_mode: bool,
        step: int,
        generator: torch.Generator,
    ) -> dict[str, float]:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        batch, clean = _load_example(*record)
        clean = clean.to(device=device, dtype=torch.bfloat16)
        sigma = torch.rand((), generator=generator, device=device).mul_(0.96).add_(0.02)
        noisy = (1 - sigma) * clean + sigma * torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
        teacher.reset(training_mode)
        with torch.no_grad():
            _run_with_generator_rng(
                generator,
                device,
                lambda: backend.predict_training(model, batch, noisy, None, sigma[None], sigma[None]),
            )
        torch.cuda.synchronize(device)
        _validate_layer_sequence(teacher.layer_indices, len(model.blocks))
        row = teacher.summary()
        row.update(
            {
                "step": step,
                "sigma": float(sigma),
                "source": record[0].stem,
                "elapsed_seconds": time.perf_counter() - started,
                "sequence_rows": teacher.sequence_rows,
                "reference_rows": teacher.reference_rows,
                "reference_count": int(batch[H3_REFERENCE_KINDS_KEY].numel()),
            }
        )
        return row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_output.parent.mkdir(parents=True, exist_ok=True)
    record_chain_sha256 = parent_record_chain or TRAINING_RECORD_CHAIN_DOMAIN
    for step, record in enumerate(training, 1):
        row = run_record(record, training_mode=True, step=step, generator=training_generator)
        history.append(row)
        print(json.dumps(row), flush=True)
        record_chain_sha256 = _extend_record_chain(record_chain_sha256, [record])
        if step % args.save_every == 0:
            global_step = completed_examples + step
            _atomic_torch_save(
                router.state_dict(),
                args.output.with_name(f"{args.output.stem}-step{global_step:06d}{args.output.suffix}"),
            )
            _atomic_torch_save(
                _capture_training_state(
                    router,
                    optimizer,
                    training_generator,
                    completed_examples=global_step,
                    completed_optimizer_updates=completed_optimizer_updates + step * len(model.blocks),
                    invariants=invariants,
                    record_chain_sha256=record_chain_sha256,
                    parent_state_sha256=resume_state_sha256,
                ),
                state_output.with_name(f"{state_output.stem}-step{global_step:06d}{state_output.suffix}"),
            )
    _atomic_torch_save(router.state_dict(), args.output)
    cumulative_examples = completed_examples + len(training)
    cumulative_optimizer_updates = completed_optimizer_updates + len(training) * len(model.blocks)
    _atomic_torch_save(
        _capture_training_state(
            router,
            optimizer,
            training_generator,
            completed_examples=cumulative_examples,
            completed_optimizer_updates=cumulative_optimizer_updates,
            invariants=invariants,
            record_chain_sha256=record_chain_sha256,
            parent_state_sha256=resume_state_sha256,
        ),
        state_output,
    )
    state_output_sha256 = _sha256_file(state_output)

    validation_rows = []
    if not args.skip_final_validation:
        for index, record in enumerate(validation, 1):
            row = run_record(record, training_mode=False, step=index, generator=validation_generator)
            row["cache_index"] = _record_cache_index(record, args.validation_cache_dir) if args.validation_cache_dir else 0
            validation_rows.append(row)
            print(json.dumps({"validation": row}), flush=True)
    validation_by_cache = {}
    validation_by_cache_and_reference_count = {}
    if validation_rows:
        cache_labels = [str(path.resolve()) for path in (args.validation_cache_dir or [])] or ["training_split"]
        for cache_index, cache_label in enumerate(cache_labels):
            cache_rows = [row for row in validation_rows if row["cache_index"] == cache_index]
            if not cache_rows:
                raise RuntimeError(f"held-out cache produced no selected validation rows: {cache_label}")
            validation_by_cache[str(cache_index)] = {
                "cache_dir": cache_label,
                "examples": len(cache_rows),
                **{key: sum(row[key] for row in cache_rows) / len(cache_rows) for key in ("loss", "retained_mass", "recall")},
            }
            reference_groups = {}
            for reference_count in sorted({row["reference_count"] for row in cache_rows}):
                reference_rows = [row for row in cache_rows if row["reference_count"] == reference_count]
                reference_groups[str(reference_count)] = {
                    "examples": len(reference_rows),
                    **{
                        key: sum(row[key] for row in reference_rows) / len(reference_rows)
                        for key in ("loss", "retained_mass", "recall")
                    },
                }
            validation_by_cache_and_reference_count[str(cache_index)] = reference_groups
    report = {
        "version": 1,
        "config": _json_compatible(vars(args))
        | {
            "resolved_training_seed": training_seed,
            "resolved_validation_seed": validation_seed,
            "resolved_state_output": str(state_output),
        },
        "counts": {
            "train": len(training),
            "unique_train": len(unique_training),
            "train_updates": len(training),
            "router_layers": len(model.blocks),
            "optimizer_updates": len(training) * len(model.blocks),
            "cumulative_train": cumulative_examples,
            "cumulative_optimizer_updates": cumulative_optimizer_updates,
            "validation": len(validation),
        },
        "continuation": {
            "resume_state": str(args.resume_state) if args.resume_state is not None else None,
            "resume_state_sha256": resume_state_sha256,
            "state_output": str(state_output),
            "state_output_sha256": state_output_sha256,
            "record_chain_sha256": record_chain_sha256,
        },
        "validation": (
            {key: sum(row[key] for row in validation_rows) / len(validation_rows) for key in ("loss", "retained_mass", "recall")}
            if validation_rows
            else None
        ),
        "validation_by_cache": validation_by_cache,
        "validation_by_cache_and_reference_count": validation_by_cache_and_reference_count,
        "history": history,
        "validation_history": validation_rows,
    }
    _atomic_write_json(report, args.output.with_suffix(".json"))
    print(json.dumps(report["validation"]), flush=True)


if __name__ == "__main__":
    main()
