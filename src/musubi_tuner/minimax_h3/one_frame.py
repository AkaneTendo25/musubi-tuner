from __future__ import annotations

from pathlib import Path


def parse_one_frame_options(spec: str | None) -> tuple[int, tuple[int, ...] | None]:
    """Parse one-frame placement in 24 fps pixel-frame indices."""

    def index(value: str, label: str) -> int:
        try:
            result = int(value)
        except ValueError as error:
            raise ValueError(f"MiniMax H3 --one_frame {label} must be an integer, got {value!r}") from error
        if result < 0:
            raise ValueError(f"MiniMax H3 --one_frame {label} must be non-negative, got {result}")
        return result

    target_index = 0
    control_indices = None
    seen: set[str] = set()
    if not spec:
        return target_index, control_indices
    for part in spec.split(","):
        key, separator, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value:
            raise ValueError(f"MiniMax H3 --one_frame options must be key=value, got {part!r}")
        if key not in {"target_index", "control_index"}:
            raise ValueError(f"MiniMax H3 --one_frame has unknown option {key!r}")
        if key in seen:
            raise ValueError(f"MiniMax H3 --one_frame has duplicate option {key!r}")
        seen.add(key)
        if key == "target_index":
            target_index = index(value, key)
        else:
            control_indices = tuple(index(item.strip(), key) for item in value.split(";"))
    return target_index, control_indices


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
