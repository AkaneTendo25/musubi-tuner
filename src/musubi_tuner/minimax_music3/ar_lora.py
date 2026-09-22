"""LoRA adapters for the MiniMax Music 3 autoregressive model."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.adapters_enabled = True
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base(inputs)
        if not self.adapters_enabled:
            return result
        update = F.linear(F.linear(inputs.float(), self.lora_A), self.lora_B)
        return (result.float() + update * self.scaling).to(result.dtype)


@contextmanager
def disable_adapters(adapters: dict[str, "LoRALinear"]):
    """Temporarily run the frozen base model without LoRA contributions."""
    previous = {name: adapter.adapters_enabled for name, adapter in adapters.items()}
    try:
        for adapter in adapters.values():
            adapter.adapters_enabled = False
        yield
    finally:
        for name, adapter in adapters.items():
            adapter.adapters_enabled = previous[name]


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> dict[str, LoRALinear]:
    model.requires_grad_(False)
    modules = dict(model.named_modules())
    adapters = {}
    for name, module in list(modules.items()):
        if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
            continue
        parent_name, child_name = name.rsplit(".", 1)
        wrapper = LoRALinear(module, rank, alpha)
        setattr(modules[parent_name], child_name, wrapper)
        adapters[name] = wrapper
    if not adapters:
        raise ValueError(f"No AR linear layers matched targets: {', '.join(targets)}")
    return adapters


def _config(path: Path) -> dict:
    config_path = path / "adapter_config.json" if path.is_dir() else path.with_suffix(".json")
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    if not path.is_dir():
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if "config" in metadata:
            return json.loads(metadata["config"])
    raise ValueError(f"AR LoRA configuration not found for {path}")


def _weights_path(path: Path) -> Path:
    if not path.is_dir():
        return path
    for name in ("adapter_model.safetensors", "ar_lora.safetensors"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise ValueError(f"AR LoRA weights not found in {path}")


def _normalize_key(key: str) -> str:
    if key.startswith("base_model.model."):
        key = key[len("base_model.model.") :]
    return key.replace(".default", "").removesuffix(".weight")


def load_and_merge_lora(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    config = _config(path)
    rank = int(config.get("rank", config.get("r")))
    alpha = float(config.get("alpha", config.get("lora_alpha", rank)))
    targets = tuple(config.get("targets", config.get("target_modules", DEFAULT_TARGETS)))
    adapters = inject_lora(model, rank, alpha, targets)
    state = {_normalize_key(key): value for key, value in load_file(_weights_path(path), device="cpu").items()}
    expected = set()
    for name, adapter in adapters.items():
        for suffix, parameter in (("lora_A", adapter.lora_A), ("lora_B", adapter.lora_B)):
            key = f"{name}.{suffix}"
            expected.add(key)
            if key not in state:
                raise ValueError(f"Missing AR LoRA tensor: {key}")
            parameter.data.copy_(state[key].to(parameter))
    extra = set(state) - expected
    if extra:
        raise ValueError(f"Unexpected AR LoRA tensors: {sorted(extra)[:3]}")
    merge_lora(model, adapters)


@torch.no_grad()
def load_lora_weights(adapters: dict[str, LoRALinear], path: str | Path) -> None:
    """Load adapter tensors without merging them into the frozen base model."""
    state = {_normalize_key(key): value for key, value in load_file(_weights_path(Path(path)), device="cpu").items()}
    expected = set()
    for name, adapter in adapters.items():
        for suffix, parameter in (("lora_A", adapter.lora_A), ("lora_B", adapter.lora_B)):
            key = f"{name}.{suffix}"
            expected.add(key)
            if key not in state:
                raise ValueError(f"Missing AR LoRA tensor: {key}")
            parameter.copy_(state[key].to(parameter))
    extra = set(state) - expected
    if extra:
        raise ValueError(f"Unexpected AR LoRA tensors: {sorted(extra)[:3]}")


@torch.no_grad()
def merge_lora(model: nn.Module, adapters: dict[str, LoRALinear]) -> None:
    modules = dict(model.named_modules())
    for name, adapter in adapters.items():
        delta = adapter.lora_B @ adapter.lora_A
        adapter.base.weight.data += delta * adapter.scaling
        parent_name, child_name = name.rsplit(".", 1)
        setattr(modules[parent_name], child_name, adapter.base)


def save_lora(adapters: dict[str, LoRALinear], output_dir: str | Path, rank: int, alpha: float) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, adapter in adapters.items():
        state[f"{name}.lora_A"] = adapter.lora_A.detach().cpu().contiguous()
        state[f"{name}.lora_B"] = adapter.lora_B.detach().cpu().contiguous()
    config = {"format": "musubi_ar_lora", "rank": rank, "alpha": alpha, "targets": list(DEFAULT_TARGETS)}
    save_file(state, output_dir / "adapter_model.safetensors", metadata={"config": json.dumps(config)})
    (output_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
