"""Low-rank factorization of the H3 AdaLN timestep projection.

Each transformer block owns ``adaln_proj.linear`` of shape ``[96768, 2688]``
(260M parameters, 13.0B across 50 blocks). Its input is ``a(t) = silu(e(t))``
where ``e(t)`` is the timestep embedding, so the whole 2688-wide input is a
smooth one-dimensional curve traced by the scalar timestep. That curve is
well approximated by a handful of basis directions, which lets the projection
be stored as ``[96768, rank]`` instead.

The factorization is

    a(t) ~= mean + U c(t),      U: [2688, rank],  c(t): [rank]
    W a(t) + b ~= (W U) c(t) + (b + W mean)

With a centered basis the ``W mean`` term must be folded into the bias; dropping
it is a silent correctness bug rather than an approximation. Fitting the basis
uncentered (``center=False``) makes the mean zero, so biases pass through
untouched -- which is what the streaming load hook uses, since it removes any
dependence on the order weights and biases arrive in. By rank 16 the uncentered
fit is no less accurate.

The contracted weight is stored in float32. The basis reproduces the modulation
far below bf16's own rounding step, so the storage dtype rather than the rank is
what bounds the result; storing bf16 would cap every rank at the same accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Conditioning rows are pinned at these timesteps, so the basis must represent
# them exactly rather than interpolate toward them from a uniform grid.
ANCHOR_TIMESTEPS: tuple[float, ...] = (0.0, 0.999, 1.0)

DEFAULT_GRID_POINTS = 4097
DEFAULT_ADALN_RANK = 16


@dataclass(frozen=True)
class AdaLNBasis:
    """Shared basis for the AdaLN timestep curve."""

    basis: torch.Tensor  # [input_dim, rank]
    mean: torch.Tensor  # [input_dim]

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def coefficients(self, activated: torch.Tensor) -> torch.Tensor:
        """Project ``silu(timestep_embedding)`` onto the basis."""
        if activated.shape[-1] != self.basis.shape[0]:
            raise ValueError(f"H3 AdaLN basis expects {self.basis.shape[0]} input features, got {activated.shape[-1]}")
        centered = activated.to(self.basis.dtype) - self.mean
        return centered @ self.basis


def default_timestep_grid(points: int = DEFAULT_GRID_POINTS, *, device=None) -> torch.Tensor:
    """Uniform [0, 1] grid unioned with the pinned conditioning timesteps."""
    if points < 2:
        raise ValueError("H3 AdaLN basis grid needs at least two points")
    grid = torch.linspace(0.0, 1.0, points, dtype=torch.float64, device=device)
    anchors = torch.tensor(ANCHOR_TIMESTEPS, dtype=torch.float64, device=device)
    return torch.unique(torch.cat((grid, anchors)), sorted=True)


def build_adaln_basis(
    time_embedder: torch.nn.Module,
    rank: int,
    *,
    grid: torch.Tensor | None = None,
    center: bool = True,
) -> AdaLNBasis:
    """Fit an orthonormal basis to ``silu(time_embedder(t))`` over the grid.

    The embedder is evaluated in float64 so the basis is not limited by the
    storage precision of the checkpoint it came from.

    With ``center=False`` the basis spans the raw curve rather than its
    deviation from the mean, so the mean is zero and no ``W @ mean`` term has to
    be folded into the projection's bias. That costs one basis direction at very
    low rank and nothing at all by rank 16, and it removes an entire class of
    bug: the bias no longer depends on its weight having been converted first.
    """
    if rank < 1:
        raise ValueError("H3 AdaLN rank must be at least 1")
    if grid is None:
        grid = default_timestep_grid()

    device = next(time_embedder.parameters()).device
    original_dtype = next(time_embedder.parameters()).dtype
    embedder = time_embedder.to(torch.float64)
    try:
        with torch.no_grad():
            activated = F.silu(embedder(grid.to(device=device, dtype=torch.float64)))
    finally:
        time_embedder.to(original_dtype)

    if activated.ndim != 2:
        raise ValueError(f"H3 timestep embedder must return [points, features], got {tuple(activated.shape)}")
    if rank > activated.shape[1]:
        raise ValueError(f"H3 AdaLN rank {rank} exceeds the {activated.shape[1]} available features")

    mean = activated.mean(dim=0) if center else torch.zeros_like(activated[0])
    _, _, vh = torch.linalg.svd(activated - mean, full_matrices=False)
    return AdaLNBasis(basis=vh[:rank].T.contiguous(), mean=mean.contiguous())


def factorize_adaln_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    basis: AdaLNBasis,
    *,
    chunk_rows: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contract one AdaLN projection onto the basis.

    Returns ``(W @ U, b + W @ mean)``. The contraction runs in float64 over row
    chunks so a full-precision copy of the original ``[96768, 2688]`` weight is
    never materialized alongside its product.
    """
    if weight.ndim != 2:
        raise ValueError(f"H3 AdaLN weight must be 2-D, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features != basis.basis.shape[0]:
        raise ValueError(f"H3 AdaLN weight has {in_features} input features, basis expects {basis.basis.shape[0]}")

    # Weights stream in on whatever device the loader quantizes on, so follow them.
    device = weight.device
    u = basis.basis.to(device=device, dtype=torch.float64)
    mean = basis.mean.to(device=device, dtype=torch.float64)
    reduced = torch.empty((out_features, basis.rank), dtype=torch.float64, device=device)
    shift = torch.empty((out_features,), dtype=torch.float64, device=device)

    for start in range(0, out_features, max(1, chunk_rows)):
        stop = min(start + max(1, chunk_rows), out_features)
        rows = weight[start:stop].to(torch.float64)
        reduced[start:stop] = rows @ u
        shift[start:stop] = rows @ mean

    folded = shift if bias is None else bias.to(torch.float64) + shift
    return reduced.to(weight.dtype), folded.to(weight.dtype if bias is None else bias.dtype)


def reconstruction_error(time_embedder: torch.nn.Module, basis: AdaLNBasis, *, grid: torch.Tensor | None = None) -> float:
    """Relative RMS error of the basis on the timestep curve."""
    if grid is None:
        grid = default_timestep_grid()
    device = next(time_embedder.parameters()).device
    original_dtype = next(time_embedder.parameters()).dtype
    embedder = time_embedder.to(torch.float64)
    try:
        with torch.no_grad():
            activated = F.silu(embedder(grid.to(device=device, dtype=torch.float64)))
    finally:
        time_embedder.to(original_dtype)

    centered = activated - basis.mean.to(activated.device)
    residual = centered - basis.coefficients(activated).to(activated.device) @ basis.basis.T.to(activated.device)
    return float((residual.pow(2).sum() / centered.pow(2).sum().clamp_min(1e-30)).sqrt())


TIME_EMBEDDER_PREFIX = "time_embedder."
ADALN_INFIX = ".adaln_proj.linear."
TABLE_KEY = "adaln_t_table"
DEFAULT_TABLE_POINTS = 4097


def read_time_embedder(checkpoint_path, *, embedder_factory) -> torch.nn.Module:
    """Rebuild the timestep embedder from a checkpoint, using its own dimensions."""
    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

    with MemoryEfficientSafeOpen(str(checkpoint_path)) as source:
        keys = [key for key in source.keys() if key.startswith(TIME_EMBEDDER_PREFIX)]
        if not keys:
            raise ValueError("checkpoint has no time_embedder; it is already pruned")
        state = {key[len(TIME_EMBEDDER_PREFIX) :]: source.get_tensor(key).to(torch.float32) for key in keys}
    hidden_dim, freq_dim = state["proj_in.weight"].shape
    embedder = embedder_factory(freq_dim, hidden_dim, state["proj_out.weight"].shape[0])
    embedder.load_state_dict(state)
    embedder.eval()
    return embedder


def build_timestep_table(embedder: torch.nn.Module, basis: AdaLNBasis, points: int) -> torch.Tensor:
    """Basis coefficients on the uniform grid the model interpolates between."""
    grid = torch.linspace(0.0, 1.0, points, dtype=torch.float64)
    original = next(embedder.parameters()).dtype
    embedder.to(torch.float64)
    try:
        with torch.no_grad():
            activated = F.silu(embedder(grid))
    finally:
        embedder.to(original)
    return basis.coefficients(activated)


def make_adaln_split_hook(basis: AdaLNBasis, table: torch.Tensor, *, storage_dtype=torch.float32):
    """A ``WeightTransformHooks.split_hook`` that reduces AdaLN as weights stream in.

    The timestep embedder is dropped and replaced by the interpolated table, and
    each ``adaln_proj`` weight is contracted onto the basis. The contraction runs
    before any downstream quantization, so it sees the checkpoint's full
    precision.

    Requires a basis built with ``center=False``: biases then pass through
    untouched, so the hook holds no state and does not care whether a bias
    streams before or after its weight (safetensors orders keys alphabetically,
    so it does).
    """
    if bool(basis.mean.any()):
        raise ValueError("H3 AdaLN streaming hook requires an uncentered basis (center=False)")

    def split_hook(key: str, tensor: torch.Tensor | None):
        if key.startswith(TIME_EMBEDDER_PREFIX):
            # One embedder tensor carries the table out; the rest simply vanish.
            if key == f"{TIME_EMBEDDER_PREFIX}proj_out.weight":
                return [TABLE_KEY], None if tensor is None else [table.to(storage_dtype).contiguous()]
            return [], None if tensor is None else []
        if key.endswith(f"{ADALN_INFIX}weight"):
            if tensor is None:
                return [key], None
            reduced, _ = factorize_adaln_weight(tensor.to(torch.float32), None, basis)
            # Store float32, not the checkpoint's bf16. The stored dtype is what
            # bounds the reduction: bf16's 8-bit mantissa puts a ~1e-3 relative
            # floor under the modulation, orders of magnitude above the basis
            # error, so rounding here discards the accuracy the rank buys and
            # makes every rank score alike. Raising only the matmul precision
            # does not help -- the error is committed at the cast. The reduced
            # weight is a negligible fraction of the model, and
            # MiniMaxH3AdaLNProjection derives its compute dtype from this
            # weight while callers cast the modulation back to the activation
            # dtype, so float32 costs one narrow matmul and no dtype mismatch.
            return [key], [reduced.to(storage_dtype).contiguous()]
        if key.endswith(f"{ADALN_INFIX}bias"):
            if tensor is None:
                return [key], None
            # Pure cast: the uncentered basis leaves the value alone, but the bias
            # must share the weight's storage dtype. F.linear dispatches to
            # addmm(bias, input, weight.T), so a mismatch surfaces as a confusing
            # "self and mat2" error that names the bias as `self`.
            return [key], [tensor.to(storage_dtype).contiguous()]
        return None, None  # everything else untouched

    return split_hook
