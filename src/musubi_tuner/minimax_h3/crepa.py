from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class H3CREPAConfig:
    student_block: int = 16
    teacher_block: int = 33
    weight: float = 0.05
    tau: float = 1.0
    neighbors: int = 2

    def validate(self, num_blocks: int | None = None) -> None:
        if self.student_block < 0 or self.teacher_block < 0:
            raise ValueError("CREPA block indices must be non-negative")
        if self.student_block >= self.teacher_block:
            raise ValueError("CREPA student_block must precede teacher_block")
        if num_blocks is not None and self.teacher_block >= num_blocks:
            raise ValueError(f"CREPA teacher_block must be below the transformer block count ({num_blocks})")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("CREPA weight must be finite and positive")
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("CREPA tau must be finite and positive")
        if self.neighbors < 0:
            raise ValueError("CREPA neighbors must be non-negative")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def parse_crepa_config(values: list[str] | None) -> H3CREPAConfig | None:
    if values is None:
        return None
    fields = {
        "student_block": int,
        "teacher_block": int,
        "weight": float,
        "tau": float,
        "neighbors": int,
    }
    parsed: dict[str, int | float] = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError(f"CREPA arguments must use key=value syntax, got {value!r}")
        key, raw = value.split("=", 1)
        if key not in fields:
            raise ValueError(f"unknown CREPA argument {key!r}; expected one of {', '.join(fields)}")
        if key in parsed:
            raise ValueError(f"duplicate CREPA argument {key!r}")
        try:
            parsed[key] = fields[key](raw)
        except ValueError as exc:
            raise ValueError(f"invalid CREPA value {value!r}") from exc
    config = H3CREPAConfig(**parsed)
    config.validate()
    return config


class H3CREPA(torch.nn.Module):
    """Temporal representation alignment over H3's generated video rows."""

    def __init__(self, hidden_size: int, config: H3CREPAConfig) -> None:
        super().__init__()
        self.config = config
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self._active = False
        self._target_indices: torch.Tensor | None = None
        self._frames = 0
        self._rows_per_frame = 0
        self._student: torch.Tensor | None = None
        self._teacher: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self, transformer: torch.nn.Module) -> None:
        blocks = getattr(transformer, "blocks", None)
        if blocks is None:
            raise TypeError("CREPA requires a transformer with a blocks sequence")
        self.config.validate(len(blocks))
        self._handles = [
            blocks[self.config.student_block].register_forward_hook(self._capture_student),
            blocks[self.config.teacher_block].register_forward_hook(self._capture_teacher),
        ]

    def set_layout(self, target_indices: torch.Tensor, frames: int, rows_per_frame: int) -> None:
        if frames <= 0 or rows_per_frame <= 0 or target_indices.numel() != frames * rows_per_frame:
            raise ValueError("CREPA target-video layout is inconsistent")
        self._target_indices = target_indices
        self._frames = frames
        self._rows_per_frame = rows_per_frame

    def begin_step(self, active: bool) -> None:
        self.clear_step()
        self._active = active

    def clear_step(self) -> None:
        self._active = False
        self._student = None
        self._teacher = None

    def _pool_target_video(self, output: torch.Tensor) -> torch.Tensor:
        if self._target_indices is None:
            raise RuntimeError("CREPA layout was not supplied before the transformer forward")
        indices = self._target_indices.to(output.device)
        rows = output.index_select(1, indices)
        return rows.reshape(rows.shape[0], self._frames, self._rows_per_frame, rows.shape[-1]).mean(dim=2)

    def _capture_student(self, _module, _inputs, output: torch.Tensor) -> None:
        if self._active and torch.is_grad_enabled():
            self._student = self._pool_target_video(output)

    def _capture_teacher(self, _module, _inputs, output: torch.Tensor) -> None:
        if self._active and torch.is_grad_enabled():
            self._teacher = self._pool_target_video(output).detach()

    def loss(self) -> tuple[torch.Tensor, dict[str, float]]:
        if self._student is None or self._teacher is None:
            raise RuntimeError("CREPA did not capture both configured transformer blocks")
        student = F.normalize(self.projector(self._student.float()), dim=-1)
        teacher = F.normalize(self._teacher.float(), dim=-1)
        frame_count = student.shape[1]
        similarities: list[torch.Tensor] = []
        weights: list[float] = []
        max_neighbor = min(self.config.neighbors, frame_count - 1)
        for offset in range(-max_neighbor, max_neighbor + 1):
            if offset < 0:
                left, right = student[:, -offset:], teacher[:, : frame_count + offset]
            elif offset > 0:
                left, right = student[:, : frame_count - offset], teacher[:, offset:]
            else:
                left, right = student, teacher
            similarities.append((left * right).sum(dim=-1).mean())
            weights.append(math.exp(-abs(offset) / self.config.tau))
        weight_tensor = student.new_tensor(weights)
        similarity = torch.stack(similarities).mul(weight_tensor).sum() / weight_tensor.sum()
        loss = -self.config.weight * similarity
        metrics = {
            "loss/crepa": float(loss.detach()),
            "crepa/alignment": float(similarity.detach()),
        }
        return loss, metrics

    def save_state(self, directory: str | Path) -> None:
        path = Path(directory) / "h3_crepa.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {key: value.detach().cpu() for key, value in self.projector.state_dict().items()},
            path,
            metadata={"h3_crepa_config": self.config.to_json()},
        )

    def load_state(self, directory: str | Path) -> bool:
        path = Path(directory) / "h3_crepa.safetensors"
        if not path.exists():
            return False
        with safe_open(path, framework="pt") as handle:
            saved_config = (handle.metadata() or {}).get("h3_crepa_config")
        if saved_config != self.config.to_json():
            raise ValueError("saved CREPA state does not match the requested configuration")
        self.projector.load_state_dict(load_file(path))
        return True
