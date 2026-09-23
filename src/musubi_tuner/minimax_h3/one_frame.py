from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

# Several one-frame target slots in one packed sequence. Each slot is its own
# single-frame latent slice at an explicit pixel-frame index; the slices are
# stacked on the latent frame axis but never encoded or decoded together.
H3_TARGET_NOISE_COUPLINGS = ("independent", "shared")
# Where a one-frame or Ref2VA image condition is shown to the model: to the
# Qwen3-VL conditioner, as DiT reference/condition latent rows, both, or neither.
# ``dual`` is the released presentation and the default.
H3_REFERENCE_ROUTES = ("dual", "qwen_image_only", "dit_latent_only", "text_only")
H3_REFERENCE_ROUTE_IDS = {route: index for index, route in enumerate(H3_REFERENCE_ROUTES)}


def route_presents_qwen_images(route: str) -> bool:
    """Whether the route shows the conditioning images to the Qwen3-VL conditioner."""
    validate_reference_route(route)
    return route in ("dual", "qwen_image_only")


def route_keeps_dit_rows(route: str) -> bool:
    """Whether the route keeps the conditioning images as DiT latent rows."""
    validate_reference_route(route)
    return route in ("dual", "dit_latent_only")


def validate_reference_route(route: str) -> None:
    if route not in H3_REFERENCE_ROUTES:
        raise ValueError(f"MiniMax H3 reference route must be one of {', '.join(H3_REFERENCE_ROUTES)}, got {route!r}")


def validate_target_noise_coupling(coupling: str) -> None:
    if coupling not in H3_TARGET_NOISE_COUPLINGS:
        raise ValueError(
            f"MiniMax H3 target noise coupling must be one of {', '.join(H3_TARGET_NOISE_COUPLINGS)}, got {coupling!r}"
        )


def couple_target_slot_noise(noise: torch.Tensor, coupling: str) -> torch.Tensor:
    """Apply the target-slot noise coupling to ``[B, C, slots, H, W]`` noise.

    ``shared`` repeats slot 0's draw in every slot, so each slot keeps an exact
    N(0, 1) marginal while all slots start from the same sample. ``independent``
    and single-slot noise are returned unchanged.
    """
    validate_target_noise_coupling(coupling)
    if coupling == "independent" or noise.ndim != 5 or noise.shape[2] <= 1:
        return noise
    return noise[:, :, :1].expand_as(noise).clone()


def validate_target_slot_indices(
    target_indices: Sequence[int],
    control_indices: Sequence[int] = (),
    *,
    label: str = "one-frame",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate signed slot placement: non-empty targets, integers, all indices distinct."""
    targets = tuple(target_indices)
    controls = tuple(control_indices)
    if not targets:
        raise ValueError(f"MiniMax H3 {label} target indices must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (*targets, *controls)):
        raise ValueError(f"MiniMax H3 {label} target and control indices must be integers")
    everything = (*targets, *controls)
    if len(set(everything)) != len(everything):
        raise ValueError(
            f"MiniMax H3 {label} target and control indices must be distinct, got targets {list(targets)} "
            f"and controls {list(controls)}"
        )
    return targets, controls


def one_frame_origin_shift(target_indices: Sequence[int], control_indices: Sequence[int] = ()) -> int:
    """Pixel frames added to every one-frame index before it becomes a rotary time.

    The media block starts at the rotary origin set by the text rows. A negative
    index is shifted with every other index of the same sequence so the earliest
    one lands on that origin: relative placement between slots and controls is
    exact, and no media row takes a temporal coordinate inside the text block.
    Non-negative placements are unshifted.
    """
    values = (*target_indices, *control_indices)
    return max(0, -min(values)) if values else 0


def parse_one_frame_options(spec: str | None) -> tuple[int | tuple[int, ...], tuple[int, ...] | None]:
    """Parse one-frame placement in 24 fps pixel-frame indices.

    ``target_index=N`` returns ``N`` as an int and keeps the non-negative
    contract. ``target_indices=A;B;...`` returns a tuple of signed slot indices;
    with it, ``control_index`` values may be signed too and every index must be
    distinct.
    """

    def index(value: str, label: str, *, signed: bool) -> int:
        try:
            result = int(value)
        except ValueError as error:
            raise ValueError(f"MiniMax H3 --one_frame {label} must be an integer, got {value!r}") from error
        if result < 0 and not signed:
            raise ValueError(f"MiniMax H3 --one_frame {label} must be non-negative, got {result}")
        return result

    target_index = 0
    control_values: list[str] | None = None
    target_values: list[str] | None = None
    seen: set[str] = set()
    if not spec:
        return target_index, None
    for part in spec.split(","):
        key, separator, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value:
            raise ValueError(f"MiniMax H3 --one_frame options must be key=value, got {part!r}")
        if key not in {"target_index", "target_indices", "control_index"}:
            raise ValueError(f"MiniMax H3 --one_frame has unknown option {key!r}")
        if key in seen:
            raise ValueError(f"MiniMax H3 --one_frame has duplicate option {key!r}")
        seen.add(key)
        if key == "target_index":
            target_index = index(value, key, signed=False)
        elif key == "target_indices":
            target_values = [item.strip() for item in value.split(";")]
        else:
            control_values = [item.strip() for item in value.split(";")]
    if target_values is None:
        control_indices = (
            tuple(index(item, "control_index", signed=False) for item in control_values) if control_values is not None else None
        )
        return target_index, control_indices
    if "target_index" in seen:
        raise ValueError("MiniMax H3 --one_frame takes target_index or target_indices, not both")
    targets = tuple(index(item, "target_indices", signed=True) for item in target_values)
    controls = tuple(index(item, "control_index", signed=True) for item in control_values) if control_values is not None else None
    validate_target_slot_indices(targets, controls or (), label="--one_frame")
    return targets, controls


def target_slot_output_paths(output: Path, target_indices: Sequence[int]) -> tuple[Path, ...]:
    """One output file per target slot: ``<stem>_<slot>_index<index><suffix>``."""
    output = Path(output)
    return tuple(
        output.with_name(f"{output.stem}_{slot:03d}_index{int(index)}{output.suffix}") for slot, index in enumerate(target_indices)
    )


def fl_condition_entries(
    *,
    frame_count: int,
    condition_images: tuple[str | Path, ...] = (),
    first_frame: str | Path | None = None,
    last_frame: str | Path | None = None,
) -> tuple[tuple[str, Path], ...]:
    """Return FL2VA controls in packed and Qwen presentation order."""
    if frame_count != 1:
        if condition_images:
            raise ValueError("MiniMax H3 condition_images apply only to one-frame generation")
        return tuple((role, Path(path)) for role, path in (("first", first_frame), ("last", last_frame)) if path)
    if condition_images and (first_frame or last_frame):
        raise ValueError("MiniMax H3 one-frame controls use condition_images or first/last aliases, not both")
    paths = condition_images or tuple(path for path in (first_frame, last_frame) if path)
    return tuple((f"cond_{i:03d}", Path(path)) for i, path in enumerate(paths))
