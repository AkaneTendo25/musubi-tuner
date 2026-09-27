"""Frozen Miowtion v1 predictor bundles and target-video tile plans.

This is a behavior-level implementation of the MIT-licensed Miowtion Veda
predictor at commit 6dfda026.  It intentionally supports deployment only:
weights are frozen, plans must match the current target grid exactly, and
training/checkpoint formats are rejected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

MIOWTION_FORMAT = "miowtion-veda-predictor-v1"
TILE_SIZE = 128
_SCALE_SUFFIX = ".__scale"
_E4M3_MAX = 448.0


@dataclass(frozen=True, order=True)
class MiowtionTileShape:
    t: int
    h: int
    w: int

    def __post_init__(self) -> None:
        if min(self.t, self.h, self.w) < 1 or self.t * self.h * self.w != TILE_SIZE:
            raise ValueError(f"Miowtion tile shape must contain {TILE_SIZE} rows, got {self}")

    @classmethod
    def parse(cls, value: str) -> MiowtionTileShape:
        try:
            parts = tuple(int(part) for part in value.split("x"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid Miowtion tile shape {value!r}") from error
        if len(parts) != 3:
            raise ValueError(f"invalid Miowtion tile shape {value!r}")
        return cls(*parts)


@dataclass(frozen=True)
class MiowtionHeadGroup:
    shape: MiowtionTileShape
    heads: torch.Tensor


@dataclass(frozen=True)
class MiowtionTileLayout:
    gather_index: torch.Tensor
    inverse_slots: torch.Tensor
    valid_count: torch.Tensor
    slot_valid: torch.Tensor
    full_tile: torch.Tensor
    n_video_tiles: int
    n_tiles: int
    rows: int


@dataclass
class MiowtionPlan:
    geometry: str
    grid: tuple[int, int, int]
    shapes: tuple[MiowtionTileShape, ...]
    head_shape: tuple[tuple[int, ...], ...]
    target_start: int
    rows: int
    position_ids: torch.Tensor = field(repr=False)
    _layouts: dict[tuple[MiowtionTileShape, str], MiowtionTileLayout] = field(default_factory=dict, init=False, repr=False)

    def head_groups(self, layer: int, device: torch.device | str) -> tuple[MiowtionHeadGroup, ...]:
        if not 0 <= layer < len(self.head_shape):
            raise ValueError(f"Miowtion layer {layer} is outside [0, {len(self.head_shape)})")
        row = torch.tensor(self.head_shape[layer], dtype=torch.long, device=device)
        return tuple(
            MiowtionHeadGroup(self.shapes[index], torch.nonzero(row == index).flatten()) for index in sorted(set(row.tolist()))
        )

    def layout(self, shape: MiowtionTileShape, device: torch.device | str) -> MiowtionTileLayout:
        key = (shape, str(torch.device(device)))
        if key not in self._layouts:
            self._layouts[key] = _build_layout(self.position_ids, self.target_start, shape, device)
        return self._layouts[key]


class MiowtionLayerPredictor(nn.Module):
    def __init__(self, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.proj_q = nn.Parameter(torch.empty(num_heads, 3 * head_dim, head_dim))
        self.proj_k = nn.Parameter(torch.empty(num_heads, 3 * head_dim, head_dim))

    def embed(self, features: torch.Tensor, heads: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        mean = features[..., : self.head_dim]
        # Keep the frozen predictor CPU-resident during training. Only the
        # projections used by this head group are staged for the score call;
        # retaining all 50 layers on CUDA costs roughly 600 MiB for the
        # released bundle and can push a 24 GiB run over its limit.
        projection_heads = heads.to(device=projection.device)
        selected = projection.index_select(0, projection_heads).to(device=features.device, dtype=torch.float32)
        # Bundle storage may be bf16; scoring is deliberately fp32.
        return torch.matmul(features, selected) + mean

    def forward(self, query_features: torch.Tensor, key_features: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        query = self.embed(query_features, heads, self.proj_q)
        key = self.embed(key_features, heads, self.proj_k)
        return torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)


class MiowtionPredictor(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(MiowtionLayerPredictor(num_heads, head_dim) for _ in range(num_layers))
        self.num_layers, self.num_heads, self.head_dim = num_layers, num_heads, head_dim


@dataclass(frozen=True)
class MiowtionBundle:
    predictor: MiowtionPredictor
    plans: tuple[dict, ...]
    keep_fraction: float
    metadata: dict[str, str]

    @property
    def num_layers(self) -> int:
        return self.predictor.num_layers

    @property
    def num_heads(self) -> int:
        return self.predictor.num_heads

    @property
    def head_dim(self) -> int:
        return self.predictor.head_dim


def _positive_int(metadata: dict[str, str], name: str, path: Path) -> int:
    try:
        value = int(metadata[name])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: invalid or missing {name!r} metadata") from error
    if value < 1:
        raise ValueError(f"{path}: {name} must be positive, got {value}")
    return value


def _validated_plans(raw: str | None, layers: int, heads: int, path: Path) -> tuple[dict, ...]:
    try:
        decoded = json.loads(raw) if raw is not None else None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid plans JSON") from error
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError(f"{path}: plans metadata must be a non-empty object")
    plans: list[dict] = []
    for name, plan in decoded.items():
        if not isinstance(plan, dict) or plan.get("geometry") != name:
            raise ValueError(f"{path}: plan key {name!r} does not match its geometry")
        grid = plan.get("grid")
        rows = plan.get("head_shape")
        shapes = plan.get("shapes")
        if not isinstance(grid, list) or len(grid) != 3 or any(not isinstance(v, int) or v < 1 for v in grid):
            raise ValueError(f"{path}: plan {name!r} has invalid grid")
        if not isinstance(shapes, list) or not shapes:
            raise ValueError(f"{path}: plan {name!r} has no shapes")
        parsed = tuple(MiowtionTileShape.parse(value) for value in shapes)
        if not isinstance(rows, list) or len(rows) != layers:
            raise ValueError(f"{path}: plan {name!r} must contain {layers} layer rows")
        for layer, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != heads or any(type(v) is not int or not 0 <= v < len(parsed) for v in row):
                raise ValueError(f"{path}: plan {name!r} layer {layer} has invalid head shape indices")
            if len(set(row)) > 2:
                raise ValueError(f"{path}: plan {name!r} layer {layer} uses more than two tile shapes")
        plans.append({**plan, "_shapes": parsed})
    return tuple(plans)


def load_miowtion_bundle(path: str | Path, *, device: torch.device | str = "cpu") -> MiowtionBundle:
    """Load and strictly validate a frozen Miowtion v1 safetensors bundle."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
    if metadata.get("format") != MIOWTION_FORMAT:
        raise ValueError(f"{path}: expected format {MIOWTION_FORMAT!r}, got {metadata.get('format')!r}")
    layers = _positive_int(metadata, "num_layers", path)
    heads = _positive_int(metadata, "num_heads", path)
    dim = _positive_int(metadata, "head_dim", path)
    plans = _validated_plans(metadata.get("plans"), layers, heads, path)
    try:
        keep_fraction = float(metadata["keep_ratio"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: invalid or missing keep_ratio metadata") from error
    if not 0.0 < keep_fraction <= 1.0 or not math.isfinite(keep_fraction):
        raise ValueError(f"{path}: keep_ratio must lie in (0, 1], got {keep_fraction}")
    stored = metadata.get("dtype", "float32")
    if stored not in {"float32", "bfloat16", "float8_e4m3fn"}:
        raise ValueError(f"{path}: unsupported predictor dtype {stored!r}")
    if stored == "float8_e4m3fn":
        values = {k: v for k, v in tensors.items() if not k.endswith(_SCALE_SUFFIX)}
        scales = {k[: -len(_SCALE_SUFFIX)]: v for k, v in tensors.items() if k.endswith(_SCALE_SUFFIX)}
        if set(values) != set(scales):
            raise ValueError(f"{path}: every fp8 predictor tensor must have exactly one per-head scale")
        tensors = {name: value.float() * scales[name].float().view(-1, *([1] * (value.ndim - 1))) for name, value in values.items()}
        resident = torch.bfloat16
    else:
        resident = torch.float32 if stored == "float32" else torch.bfloat16
    model = MiowtionPredictor(layers, heads, dim).to(dtype=resident)
    try:
        model.load_state_dict(tensors, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{path}: predictor tensors disagree with declared geometry: {error}") from error
    model.requires_grad_(False).eval().to(device)
    return MiowtionBundle(model, plans, keep_fraction, metadata)


def build_miowtion_plan(bundle: MiowtionBundle, position_ids: torch.Tensor, target_start: int) -> MiowtionPlan:
    """Build a per-forward plan, requiring an exact bundle grid match."""
    if position_ids.ndim != 2 or position_ids.shape[1] != 3:
        raise ValueError(f"position_ids must have shape (rows, 3), got {tuple(position_ids.shape)}")
    rows = position_ids.shape[0]
    if not 0 <= target_start < rows:
        raise ValueError(f"target_start must lie in [0, {rows}), got {target_start}")
    target = position_ids[target_start:].detach().cpu()
    axes = [torch.unique(target[:, axis], sorted=True) for axis in range(3)]
    grid = tuple(int(axis.numel()) for axis in axes)
    if math.prod(grid) != target.shape[0] or torch.unique(target, dim=0).shape[0] != target.shape[0]:
        raise ValueError(f"target position_ids do not form a complete 3D grid: inferred {grid} for {target.shape[0]} rows")
    normalized = torch.stack([torch.searchsorted(axes[axis], target[:, axis].contiguous()) for axis in range(3)], dim=1)
    expected = torch.cartesian_prod(*(torch.arange(size) for size in grid))
    if not torch.equal(normalized, expected):
        raise ValueError("target position_ids must be in T,H,W raster order")
    matches = [plan for plan in bundle.plans if tuple(plan["grid"]) == grid]
    if len(matches) != 1:
        names = [plan["geometry"] for plan in matches]
        raise ValueError(f"target grid {grid} must match exactly one bundle plan, matched {names}")
    selected = matches[0]
    return MiowtionPlan(
        selected["geometry"],
        grid,
        selected["_shapes"],
        tuple(tuple(row) for row in selected["head_shape"]),
        target_start,
        rows,
        position_ids.detach().cpu(),
    )


def _build_layout(
    position_ids: torch.Tensor, target_start: int, shape: MiowtionTileShape, device: torch.device | str
) -> MiowtionTileLayout:
    rows = position_ids.shape[0]
    grid = tuple(int(torch.unique(position_ids[target_start:, axis]).numel()) for axis in range(3))
    t, h, w = grid
    tp, hp, wp = (math.ceil(size / extent) * extent for size, extent in zip(grid, (shape.t, shape.h, shape.w)))
    lattice = torch.full((tp, hp, wp), -1, dtype=torch.long)
    lattice[:t, :h, :w] = target_start + torch.arange(t * h * w).view(t, h, w)
    video = lattice.view(tp // shape.t, shape.t, hp // shape.h, shape.h, wp // shape.w, shape.w)
    video = video.permute(2, 4, 0, 1, 3, 5).reshape(-1, TILE_SIZE)
    video = torch.gather(video, 1, torch.argsort((video < 0).to(torch.int8), dim=1, stable=True))
    global_rows = torch.arange(target_start, dtype=torch.long)
    global_count = math.ceil(target_start / TILE_SIZE)
    global_tiles = torch.full((global_count * TILE_SIZE,), -1, dtype=torch.long)
    global_tiles[:target_start] = global_rows
    perm = torch.cat((video.flatten(), global_tiles))
    valid = perm >= 0
    counts = valid.view(-1, TILE_SIZE).sum(-1).to(torch.int32)
    inverse = torch.empty(rows, dtype=torch.long)
    inverse[perm[valid]] = torch.nonzero(valid).flatten()
    return MiowtionTileLayout(
        perm.clamp(min=0).to(device),
        inverse.to(device),
        counts.to(device),
        valid.to(device),
        (counts == TILE_SIZE).to(device),
        video.shape[0],
        counts.numel(),
        rows,
    )


@torch.no_grad()
def pool_miowtion_tiles(tensor: torch.Tensor, layout: MiowtionTileLayout) -> torch.Tensor:
    """Pool ``(B,H,N,D)`` tile ordered rows into exact mean/max/min features."""
    batch, heads, _, dim = tensor.shape
    tiles = tensor.detach().view(batch, heads, layout.n_tiles, TILE_SIZE, dim)
    valid = layout.slot_valid.view(1, 1, layout.n_tiles, TILE_SIZE, 1)
    count = layout.valid_count.clamp(min=1).view(1, 1, -1, 1).float()
    # Accumulate only the reduction in fp32. Casting the complete tiled tensor
    # used to retain a full-size fp32 temporary for both Q and K.
    mean = tiles.sum(dim=3, dtype=torch.float32) / count
    maximum = tiles.masked_fill(~valid, -torch.inf).amax(dim=3).float()
    minimum = tiles.masked_fill(~valid, torch.inf).amin(dim=3).float()
    nonempty = (layout.valid_count > 0).view(1, 1, -1, 1)
    return torch.where(nonempty, torch.cat((mean, maximum, minimum), dim=-1), 0.0)
