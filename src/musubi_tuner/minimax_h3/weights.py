from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class CheckpointInspectionError(ValueError):
    pass


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    component: str = "."

    @property
    def parameters(self) -> int:
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total


@dataclass(frozen=True)
class CheckpointInventory:
    root: str
    index_file: str | None
    shards: tuple[str, ...]
    tensors: int
    parameters: int
    dtypes: dict[str, int]
    prefixes: dict[str, int]
    config_files: tuple[str, ...]
    index_files: tuple[str, ...] = ()
    components: dict[str, int] | None = None
    unsupported_weight_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise CheckpointInspectionError(f"invalid safetensors header in {path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length > 100 * 1024 * 1024:
                raise CheckpointInspectionError(f"unreasonably large safetensors header in {path}")
            header = handle.read(header_length)
        value = json.loads(header)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointInspectionError(f"cannot read safetensors metadata from {path}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointInspectionError(f"safetensors header is not an object: {path}")
    return value


def tensor_metadata(path: Path, *, shard_name: str | None = None, component: str = ".") -> tuple[TensorMetadata, ...]:
    path = Path(path)
    metadata: list[TensorMetadata] = []
    for name, value in _read_safetensors_header(path).items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict) or "dtype" not in value or "shape" not in value:
            raise CheckpointInspectionError(f"invalid tensor entry {name!r} in {path}")
        metadata.append(
            TensorMetadata(
                name,
                str(value["dtype"]),
                tuple(int(v) for v in value["shape"]),
                shard_name or path.name,
                component,
            )
        )
    return tuple(metadata)


def _relative_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _component_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.as_posix() if relative.parts else "."


def _index_shards(index_path: Path, root: Path) -> tuple[Path, ...]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
    except (OSError, json.JSONDecodeError, KeyError, AttributeError, TypeError) as error:
        raise CheckpointInspectionError(f"invalid weight index {index_path}: {error}") from error

    root_resolved = root.resolve()
    shards = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise CheckpointInspectionError(f"invalid shard name in {index_path}: {name!r}")
        shard = (index_path.parent / name).resolve()
        if not shard.is_relative_to(root_resolved):
            raise CheckpointInspectionError(f"weight index references a shard outside the checkpoint root: {name}")
        shards.append(shard)
    return tuple(shards)


def _resolve_layout(source: Path) -> tuple[Path, tuple[Path, ...], tuple[tuple[Path, str], ...]]:
    source = Path(source)
    if source.is_file() and source.suffix == ".safetensors":
        source = source.resolve()
        return source.parent, (), ((source, "."),)
    if source.is_file() and source.name.endswith(".safetensors.index.json"):
        root, index_paths = source.parent, (source,)
    elif source.is_dir():
        root = source
        index_paths = tuple(sorted(root.rglob("*.safetensors.index.json")))
    else:
        raise FileNotFoundError(source)

    root = root.resolve()
    index_paths = tuple(path.resolve() for path in index_paths)

    shard_components: dict[Path, str] = {}
    for index_path in index_paths:
        component = _component_name(index_path.parent, root)
        for shard in _index_shards(index_path, root):
            previous = shard_components.get(shard)
            if previous is not None and previous != component:
                raise CheckpointInspectionError(f"checkpoint shard is referenced by multiple components: {shard}")
            shard_components[shard] = component

    # Include standalone safetensors components as well as indexed shards.
    for shard in sorted(root.rglob("*.safetensors")):
        shard_components.setdefault(shard.resolve(), _component_name(shard.parent, root))

    if not shard_components:
        raise CheckpointInspectionError(f"no safetensors weights found under {root}")
    missing = [str(path) for path in shard_components if not path.is_file()]
    if missing:
        raise CheckpointInspectionError(f"weight index references missing shard(s): {', '.join(missing)}")
    shards = tuple(sorted(shard_components.items(), key=lambda item: _relative_name(item[0], root)))
    return root, index_paths, shards


def inspect_checkpoint(source: Path) -> CheckpointInventory:
    root, index_paths, shards = _resolve_layout(Path(source))
    tensors: list[TensorMetadata] = []
    names: set[tuple[str, str]] = set()
    component_counts: Counter[str] = Counter()
    for shard, component in shards:
        for tensor in tensor_metadata(
            shard,
            shard_name=_relative_name(shard, root),
            component=component,
        ):
            scoped_name = (component, tensor.name)
            if scoped_name in names:
                raise CheckpointInspectionError(f"duplicate tensor key across shards in component {component!r}: {tensor.name}")
            names.add(scoped_name)
            tensors.append(tensor)
            component_counts[component] += 1

    dtype_counts = Counter(tensor.dtype for tensor in tensors)
    prefix_counts = Counter(tensor.name.split(".", 1)[0] for tensor in tensors)
    index_set = {path.resolve() for path in index_paths}
    configs = tuple(
        _relative_name(path, root)
        for path in sorted(root.rglob("*.json"))
        if path.resolve() not in index_set and (path.name == "model_index.json" or path.name.endswith("config.json"))
    )
    unsupported = tuple(
        _relative_name(path, root) for pattern in ("*.bin", "*.pt", "*.pth", "*.ckpt") for path in sorted(root.rglob(pattern))
    )
    index_names = tuple(_relative_name(path, root) for path in index_paths)
    return CheckpointInventory(
        root=str(root.resolve()),
        index_file=index_names[0] if len(index_names) == 1 else None,
        shards=tuple(_relative_name(path, root) for path, _ in shards),
        tensors=len(tensors),
        parameters=sum(tensor.parameters for tensor in tensors),
        dtypes=dict(sorted(dtype_counts.items())),
        prefixes=dict(prefix_counts.most_common()),
        config_files=configs,
        index_files=index_names,
        components=dict(sorted(component_counts.items())),
        unsupported_weight_files=unsupported,
    )
