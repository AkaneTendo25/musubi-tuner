from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

DINO_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class H3CREPAConfig:
    mode: str = "backbone"
    student_block: int = 16
    teacher_block: int = 33
    dino_model: str = "dinov2_vitb14"
    weight: float = 0.05
    tau: float = 1.0
    neighbors: int = 2
    schedule: str = "constant"
    warmup_steps: int = 0
    max_steps: int = 0
    normalize: bool = True
    cutoff_step: int = 0
    similarity_threshold: float | None = None
    similarity_ema_decay: float = 0.99
    threshold_mode: str = "permanent"

    def validate(self, num_blocks: int | None = None) -> None:
        if self.mode not in {"backbone", "dino"}:
            raise ValueError("CREPA mode must be backbone or dino")
        if self.mode == "dino" and self.dino_model not in DINO_DIMS:
            raise ValueError(f"unsupported CREPA dino_model {self.dino_model!r}")
        if self.student_block < 0 or self.teacher_block < 0:
            raise ValueError("CREPA block indices must be non-negative")
        if self.mode == "backbone" and self.student_block >= self.teacher_block:
            raise ValueError("CREPA student_block must precede teacher_block")
        if num_blocks is not None:
            if self.student_block >= num_blocks:
                raise ValueError(f"CREPA student_block must be below the transformer block count ({num_blocks})")
            if self.mode == "backbone" and self.teacher_block >= num_blocks:
                raise ValueError(f"CREPA teacher_block must be below the transformer block count ({num_blocks})")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("CREPA weight must be finite and positive")
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("CREPA tau must be finite and positive")
        if self.neighbors < 0:
            raise ValueError("CREPA neighbors must be non-negative")
        if self.schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("CREPA schedule must be constant, linear, or cosine")
        if self.warmup_steps < 0 or self.max_steps < 0 or self.cutoff_step < 0:
            raise ValueError("CREPA step controls must be non-negative")
        if self.schedule != "constant" and self.max_steps <= 0:
            raise ValueError("CREPA linear and cosine schedules require max_steps > 0")
        if self.max_steps and self.warmup_steps > self.max_steps:
            raise ValueError("CREPA warmup_steps cannot exceed max_steps")
        if self.similarity_threshold is not None and (
            not math.isfinite(self.similarity_threshold) or not 0 <= self.similarity_threshold <= 1
        ):
            raise ValueError("CREPA similarity_threshold must be finite and in [0, 1]")
        if not 0 <= self.similarity_ema_decay < 1:
            raise ValueError("CREPA similarity_ema_decay must be in [0, 1)")
        if self.threshold_mode not in {"permanent", "recoverable"}:
            raise ValueError("CREPA threshold_mode must be permanent or recoverable")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def parse_crepa_config(values: list[str] | None) -> H3CREPAConfig | None:
    if values is None:
        return None
    fields = {
        "mode": str,
        "student_block": int,
        "teacher_block": int,
        "dino_model": str,
        "weight": float,
        "tau": float,
        "neighbors": int,
        "schedule": str,
        "warmup_steps": int,
        "max_steps": int,
        "normalize": lambda value: _parse_bool(value),
        "cutoff_step": int,
        "similarity_threshold": float,
        "similarity_ema_decay": float,
        "threshold_mode": str,
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


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


class H3CREPA(torch.nn.Module):
    """Temporal representation alignment over H3's generated video rows."""

    def __init__(self, hidden_size: int, config: H3CREPAConfig) -> None:
        super().__init__()
        self.config = config
        output_size = hidden_size if config.mode == "backbone" else DINO_DIMS[config.dino_model]
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, output_size),
        )
        self._active = False
        self._target_indices: torch.Tensor | None = None
        self._frames = 0
        self._rows_per_frame = 0
        self._student: torch.Tensor | None = None
        self._teacher: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._step = 0
        self._effective_weight = config.weight
        self._similarity_ema: float | None = None
        self._cutoff_triggered = False
        self._cutoff_active = False

    def install(self, transformer: torch.nn.Module) -> None:
        blocks = getattr(transformer, "blocks", None)
        if blocks is None:
            raise TypeError("CREPA requires a transformer with a blocks sequence")
        self.config.validate(len(blocks))
        self._handles = [blocks[self.config.student_block].register_forward_hook(self._capture_student)]
        if self.config.mode == "backbone":
            self._handles.append(blocks[self.config.teacher_block].register_forward_hook(self._capture_teacher))

    def set_layout(self, target_indices: torch.Tensor, frames: int, rows_per_frame: int) -> None:
        if frames <= 0 or rows_per_frame <= 0 or target_indices.numel() != frames * rows_per_frame:
            raise ValueError("CREPA target-video layout is inconsistent")
        self._target_indices = target_indices
        self._frames = frames
        self._rows_per_frame = rows_per_frame

    def _scheduled_weight(self, step: int) -> float:
        config = self.config
        if config.warmup_steps and step < config.warmup_steps:
            return config.weight * step / config.warmup_steps
        if config.schedule == "constant" or not config.max_steps:
            return config.weight
        denominator = max(config.max_steps - config.warmup_steps, 1)
        progress = min(max((step - config.warmup_steps) / denominator, 0.0), 1.0)
        if config.schedule == "linear":
            return config.weight * (1.0 - progress)
        return config.weight * 0.5 * (1.0 + math.cos(math.pi * progress))

    def begin_step(self, active: bool, global_step: int = 0) -> None:
        self.clear_step()
        self._step = global_step
        hard_cutoff = self.config.cutoff_step > 0 and global_step >= self.config.cutoff_step
        threshold_cutoff = self._cutoff_triggered and self.config.threshold_mode == "permanent"
        self._cutoff_active = hard_cutoff or threshold_cutoff
        self._effective_weight = 0.0 if hard_cutoff or threshold_cutoff else self._scheduled_weight(global_step)
        self._active = active and self._effective_weight > 0

    @property
    def active(self) -> bool:
        return self._active

    def status_metrics(self) -> dict[str, float]:
        metrics = {
            "crepa/weight": self._effective_weight if self._active else 0.0,
            "crepa/cutoff": float(self._cutoff_active),
        }
        if self._similarity_ema is not None:
            metrics["crepa/alignment_ema"] = self._similarity_ema
        return metrics

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

    def _target_video_rows(self, output: torch.Tensor) -> torch.Tensor:
        if self._target_indices is None:
            raise RuntimeError("CREPA layout was not supplied before the transformer forward")
        rows = output.index_select(1, self._target_indices.to(output.device))
        return rows.reshape(rows.shape[0], self._frames, self._rows_per_frame, rows.shape[-1])

    def _capture_student(self, _module, _inputs, output: torch.Tensor) -> None:
        if self._active and torch.is_grad_enabled():
            self._student = self._target_video_rows(output)

    def _capture_teacher(self, _module, _inputs, output: torch.Tensor) -> None:
        if self._active and torch.is_grad_enabled():
            self._teacher = self._pool_target_video(output).detach()

    def loss(self, dino_features: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        if self._student is None or (self.config.mode == "backbone" and self._teacher is None):
            raise RuntimeError("CREPA did not capture its configured transformer features")
        student = self.projector(self._student.float())
        if self.config.mode == "dino":
            if dino_features is None:
                raise ValueError("CREPA mode=dino requires cached h3_dino_features")
            teacher = dino_features.to(device=student.device, dtype=student.dtype)
            if teacher.ndim != 4:
                raise ValueError("h3_dino_features must have shape [batch, frames, patches, channels]")
            if teacher.shape[0] != student.shape[0] or teacher.shape[-1] != student.shape[-1]:
                raise ValueError("cached DINO features do not match the CREPA batch or selected DINO model")
            if teacher.shape[1] != student.shape[1]:
                frame_indices = torch.linspace(0, teacher.shape[1] - 1, student.shape[1], device=teacher.device).long()
                teacher = teacher.index_select(1, frame_indices)
            if teacher.shape[2] != student.shape[2]:
                batch, frames, rows, channels = student.shape
                flattened = student.reshape(batch * frames, rows, channels).transpose(1, 2)
                flattened = F.interpolate(flattened, size=teacher.shape[2], mode="linear", align_corners=False)
                student = flattened.transpose(1, 2).reshape(batch, frames, teacher.shape[2], channels)
        else:
            student = student.mean(dim=2)
            teacher = self._teacher.float()
        if self.config.normalize:
            student = F.normalize(student, dim=-1)
            teacher = F.normalize(teacher, dim=-1)
        frame_count = student.shape[1]
        batch_size = student.shape[0]
        loss_by_batch = student.new_zeros(batch_size)
        alignment_sum = student.new_zeros(batch_size, frame_count)
        alignment_weight = student.new_zeros(batch_size, frame_count)
        self_sum = student.new_zeros(batch_size)
        comparison_count = 0
        max_neighbor = min(self.config.neighbors, frame_count - 1)

        def pair_similarity(left_frame: int, right_frame: int) -> torch.Tensor:
            value = (student[:, left_frame] * teacher[:, right_frame]).sum(dim=-1)
            return value.flatten(start_dim=1).mean(dim=1) if value.ndim > 1 else value

        for frame in range(frame_count):
            same = pair_similarity(frame, frame)
            loss_by_batch -= same
            self_sum += same
            alignment_sum[:, frame] += same
            alignment_weight[:, frame] += 1.0
            comparison_count += 1
            for distance in range(1, max_neighbor + 1):
                temporal_weight = math.exp(-distance / self.config.tau)
                for neighbor in (frame - distance, frame + distance):
                    if 0 <= neighbor < frame_count:
                        nearby = pair_similarity(frame, neighbor)
                        loss_by_batch -= temporal_weight * nearby
                        alignment_sum[:, frame] += temporal_weight * nearby
                        alignment_weight[:, frame] += temporal_weight
                        comparison_count += 1
        average_comparisons_per_frame = max(comparison_count / frame_count, 1.0)
        unweighted_loss = loss_by_batch.mean() / average_comparisons_per_frame
        similarity = (alignment_sum / alignment_weight.clamp_min(torch.finfo(student.dtype).eps)).mean()
        self_similarity = (self_sum / frame_count).mean()
        score = float(similarity.detach())
        threshold = self.config.similarity_threshold
        if threshold is not None:
            if self._similarity_ema is None:
                self._similarity_ema = score
            else:
                decay = self.config.similarity_ema_decay
                self._similarity_ema = decay * self._similarity_ema + (1.0 - decay) * score
        threshold_active = threshold is not None and self._similarity_ema >= threshold
        if threshold_active and self.config.threshold_mode == "permanent":
            self._cutoff_triggered = True
        self._cutoff_active = threshold_active
        effective_weight = 0.0 if threshold_active else self._effective_weight
        loss = effective_weight * unweighted_loss
        if not torch.isfinite(loss):
            logger.warning("CREPA loss is non-finite; skipping the auxiliary term")
            loss = similarity.new_zeros(())
        metrics: dict[str, float] = {
            "loss/crepa": float(loss.detach()),
            "crepa/alignment": float(similarity.detach()),
            "crepa/similarity_self": float(self_similarity.detach()),
            "crepa/weight": effective_weight,
            "crepa/cutoff": float(self._cutoff_active),
        }
        if self._similarity_ema is not None:
            metrics["crepa/alignment_ema"] = self._similarity_ema
        return loss, metrics

    def save_state(self, directory: str | Path) -> None:
        path = Path(directory) / "h3_crepa.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                **{key: value.detach().cpu() for key, value in self.projector.state_dict().items()},
                "_crepa_similarity_ema": torch.tensor(float("nan") if self._similarity_ema is None else self._similarity_ema),
                "_crepa_cutoff_triggered": torch.tensor(self._cutoff_triggered),
            },
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
        state = load_file(path)
        ema = state.pop("_crepa_similarity_ema", None)
        cutoff = state.pop("_crepa_cutoff_triggered", None)
        self.projector.load_state_dict(state)
        if ema is not None and torch.isfinite(ema):
            self._similarity_ema = float(ema)
        if cutoff is not None:
            self._cutoff_triggered = bool(cutoff)
        return True
