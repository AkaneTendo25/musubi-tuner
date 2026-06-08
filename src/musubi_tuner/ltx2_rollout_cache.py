"""Offline rollout cache for LTX-2 RL (NFT/GRPO) post-training.

A rollout cache is a disk-backed per-epoch sample buffer: for
each prompt it stores K generated samples (clean x0 latents + reconstruction metadata),
their per-reward scores, and the routed group-relative advantages. Phase A (``ltx2_cache_rollouts``)
writes it; Phase B (``ltx2_train_rl``) replays it.

Layout::

    cache_dir/
      index.json                 # global meta + per-group entries (prompts, seeds, keys)
      group_00000.safetensors     # tensors for one prompt-group, stacked over K samples
      group_00001.safetensors
      ...

Per-group safetensors keys are namespaced: ``tensor::<name>`` (e.g. video_x0, positions,
sigmas, v_ctx, v_mask, conditioning/ref latents), ``score::<reward>`` (per-sample scores,
already higher-is-better), ``adv::<route>`` (per-sample/-token routed advantages).

The cache records the ``snapshot_hash`` of the ``old`` policy that generated it. Phase B
must assert the current ``old`` snapshot matches (``assert_snapshot``) — this is the
fixed-behavior-policy invariant: training a stale cache (whose advantages came from a
different policy) is off-policy with no importance correction, so it is forbidden.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = 1

_KEY_SEP = "::"


def compute_snapshot_hash(tensors: Iterable[torch.Tensor]) -> str:
    """Stable content hash of a policy's trainable tensors.

    Deterministic and collision-free for drift detection: any change to any weight changes
    the hash. Tensors are upcast to float32 bytes so bf16/fp16 adapters hash deterministically.
    Pass ``network.trainable_lora_params()`` (the canonical ordered list).
    """
    h = hashlib.sha256()
    for tensor in tensors:
        t = tensor.detach().to(torch.float32).cpu().contiguous()
        h.update(str(tuple(t.shape)).encode())
        if t.numel():
            h.update(t.numpy().tobytes())
    return h.hexdigest()


@dataclass
class RolloutCacheMeta:
    snapshot_hash: str
    group_size: int
    reward_names: List[str]
    route_map: Dict[str, str]  # reward_name -> route (video|audio|sync)
    sampler_settings: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION


class RolloutCacheWriter:
    """Accumulates per-prompt groups and writes a rollout cache (Phase A)."""

    def __init__(self, cache_dir: str | Path, meta: RolloutCacheMeta) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # Clear stale cache files from a previous run so a crashed/partial re-write can never leave
        # this run's index.json paired with another run's group files (silent cross-run mix).
        for stale in list(self.dir.glob("group_*.safetensors")) + [self.dir / "index.json"]:
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        self.meta = meta
        self._groups: List[Dict[str, Any]] = []

    def write_group(
        self,
        group_idx: int,
        *,
        prompt: str,
        seeds: List[int],
        tensors: Dict[str, torch.Tensor],
        reward_scores: Dict[str, List[float]],
        advantages: Dict[str, torch.Tensor],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, torch.Tensor] = {}
        for name, value in tensors.items():
            payload[f"tensor{_KEY_SEP}{name}"] = value.detach().cpu().contiguous()
        for name, scores in reward_scores.items():
            payload[f"score{_KEY_SEP}{name}"] = torch.tensor(scores, dtype=torch.float64)
        for route, adv in advantages.items():
            payload[f"adv{_KEY_SEP}{route}"] = adv.detach().cpu().contiguous()

        fname = f"group_{group_idx:05d}.safetensors"
        save_file(payload, str(self.dir / fname))
        self._groups.append(
            {
                "group_idx": group_idx,
                "file": fname,
                "prompt": prompt,
                "seeds": [int(s) for s in seeds],
                "tensor_keys": sorted(tensors),
                "reward_names": sorted(reward_scores),
                "adv_routes": sorted(advantages),
                "extra": extra or {},
            }
        )

    def finalize(self) -> Path:
        index = {
            "schema_version": self.meta.schema_version,
            "snapshot_hash": self.meta.snapshot_hash,
            "group_size": self.meta.group_size,
            "reward_names": self.meta.reward_names,
            "route_map": self.meta.route_map,
            "sampler_settings": self.meta.sampler_settings,
            "num_groups": len(self._groups),
            "groups": self._groups,
        }
        index_path = self.dir / "index.json"
        index_path.write_text(json.dumps(index, indent=2))
        return index_path


class SnapshotMismatchError(RuntimeError):
    """Raised when a rollout cache was generated by a different policy than the live `old`."""


class RolloutCacheReader:
    """Reads a rollout cache written by :class:`RolloutCacheWriter` (Phase B)."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        index_path = self.dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"no rollout cache index at {index_path}")
        self.index = json.loads(index_path.read_text())
        if self.index.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"rollout cache schema {self.index.get('schema_version')} != expected {SCHEMA_VERSION}")

    @property
    def snapshot_hash(self) -> str:
        return self.index["snapshot_hash"]

    @property
    def group_size(self) -> int:
        return self.index["group_size"]

    @property
    def reward_names(self) -> List[str]:
        return self.index["reward_names"]

    @property
    def route_map(self) -> Dict[str, str]:
        return self.index["route_map"]

    @property
    def sampler_settings(self) -> Dict[str, Any]:
        return self.index["sampler_settings"]

    def __len__(self) -> int:
        return self.index["num_groups"]

    def assert_snapshot(self, current_hash: str) -> None:
        """Raise unless ``current_hash`` matches the cache's snapshot (fixed-behavior-policy invariant)."""
        cached = self.index["snapshot_hash"]
        if current_hash != cached:
            raise SnapshotMismatchError(
                f"rollout cache snapshot mismatch: cache={cached[:12]}.. current={current_hash[:12]}.. — "
                "the `old` policy that generated this cache differs from the current one. "
                "Training a stale cache is off-policy with no importance correction; regenerate the cache."
            )

    def read_group(self, group_idx: int, device: str | torch.device = "cpu") -> Dict[str, Any]:
        entry = self.index["groups"][group_idx]
        # Integrity: the index is positional, so verify the entry's logical group_idx matches the
        # requested position (guards a sparse/out-of-order or mismatched index).
        if entry.get("group_idx", group_idx) != group_idx:
            raise ValueError(f"rollout cache index misaligned: position {group_idx} holds group_idx {entry.get('group_idx')!r}")
        payload = load_file(str(self.dir / entry["file"]), device=str(device))
        # Integrity: the loaded file must carry exactly the tensor keys the index recorded (catches a
        # stale/foreign group file paired with this index).
        _tprefix = "tensor" + _KEY_SEP
        _loaded_tkeys = sorted(k[len(_tprefix) :] for k in payload if k.startswith(_tprefix))
        if _loaded_tkeys != sorted(entry.get("tensor_keys", _loaded_tkeys)):
            raise ValueError(
                f"rollout cache group {group_idx} file '{entry['file']}' tensor keys {_loaded_tkeys} "
                f"!= index {sorted(entry.get('tensor_keys', []))} (corrupt/stale cache)"
            )
        tensors: Dict[str, torch.Tensor] = {}
        reward_scores: Dict[str, List[float]] = {}
        advantages: Dict[str, torch.Tensor] = {}
        for key, value in payload.items():
            kind, _, name = key.partition(_KEY_SEP)
            if kind == "tensor":
                tensors[name] = value
            elif kind == "score":
                reward_scores[name] = value.tolist()
            elif kind == "adv":
                advantages[name] = value
        return {
            "prompt": entry["prompt"],
            "seeds": entry["seeds"],
            "tensors": tensors,
            "reward_scores": reward_scores,
            "advantages": advantages,
            "extra": entry.get("extra", {}),
        }
