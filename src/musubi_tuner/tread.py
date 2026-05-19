"""TREAD token routing helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class MaskInfo:
    """Routing state for one token window."""

    mask: torch.BoolTensor
    ids_keep: torch.LongTensor
    ids_mask: torch.LongTensor
    ids_shuffle: torch.LongTensor
    ids_restore: torch.LongTensor


class TREADRouter:
    """Training-time token router used by TREAD."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._generators: dict[str, torch.Generator] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator:
        device_key = str(device)
        generator = self._generators.get(device_key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[device_key] = generator
        return generator

    @staticmethod
    def _importance(x: torch.Tensor) -> torch.Tensor:
        magnitudes = x.abs().sum(-1)
        minimum = magnitudes.min(dim=1, keepdim=True)[0]
        range_ = magnitudes.max(dim=1, keepdim=True)[0] - minimum
        return (magnitudes - minimum) / (range_ + 1e-8)

    @staticmethod
    def _normalize_force_keep(
        force_keep: Optional[torch.BoolTensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.BoolTensor:
        if force_keep is None:
            return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        force_keep = force_keep.to(device=device, dtype=torch.bool)
        if force_keep.dim() == 1:
            force_keep = force_keep.view(1, -1)
        elif force_keep.dim() > 2:
            force_keep = force_keep.reshape(force_keep.shape[0], -1)

        if force_keep.shape[0] == 1 and batch_size != 1 and force_keep.shape[1] == seq_len:
            force_keep = force_keep.expand(batch_size, seq_len)
        elif force_keep.shape[0] != batch_size or force_keep.shape[1] != seq_len:
            if force_keep.numel() == batch_size * seq_len:
                force_keep = force_keep.reshape(batch_size, seq_len)
            else:
                raise ValueError(f"force_keep mask has shape {tuple(force_keep.shape)}, expected ({batch_size}, {seq_len})")
        return force_keep

    @torch.no_grad()
    def get_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
        l1_reg: float = 0.0,
        inverse: bool = False,
        force_keep: Optional[torch.BoolTensor] = None,
    ) -> MaskInfo:
        batch_size, seq_len, _ = x.shape

        force_keep = self._normalize_force_keep(
            force_keep,
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
        )

        force_count = force_keep.sum(dim=1)
        base_keep = seq_len - int(round(seq_len * float(mask_ratio)))
        keep_budget = max(base_keep, int(force_count.max().item()) if force_count.numel() > 0 else 0, 1)

        score = self._importance(x)
        if inverse:
            score = 1.0 - score

        noise = torch.rand(
            score.shape,
            dtype=score.dtype,
            device=score.device,
            generator=self._get_generator(score.device),
        )
        mix = (1.0 - l1_reg) * noise + l1_reg * score
        mix = mix.masked_fill(force_keep, -1.0)

        ids_shuffle = torch.argsort(mix, dim=1)
        ids_keep = ids_shuffle[:, :keep_budget]
        ids_mask = ids_shuffle[:, keep_budget:]
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_keep, False)

        return MaskInfo(mask, ids_keep, ids_mask, ids_shuffle, ids_restore)

    def start_route(self, x: torch.Tensor, info: MaskInfo) -> torch.Tensor:
        gather_idx = info.ids_shuffle.to(device=x.device).unsqueeze(-1).expand_as(x)
        shuffled = torch.take_along_dim(x, gather_idx, dim=1)
        return shuffled[:, : info.ids_keep.size(1), :]

    def end_route(
        self,
        routed_x: torch.Tensor,
        info: MaskInfo,
        original_x: Optional[torch.Tensor] = None,
        mask_token: float | int = 0.0,
    ) -> torch.Tensor:
        batch_size, seq_len = info.mask.shape
        hidden_dim = routed_x.shape[2]

        ids_shuffle = info.ids_shuffle.to(device=routed_x.device)
        ids_restore = info.ids_restore.to(device=routed_x.device)

        shuffled = torch.empty(batch_size, seq_len, hidden_dim, device=routed_x.device, dtype=routed_x.dtype)
        shuffled[:, : routed_x.size(1), :] = routed_x

        if original_x is not None:
            original_x = original_x.to(device=routed_x.device, dtype=routed_x.dtype)
            original_shuffled = torch.take_along_dim(
                original_x,
                ids_shuffle.unsqueeze(-1).expand_as(original_x),
                dim=1,
            )
            shuffled[:, routed_x.size(1) :, :] = original_shuffled[:, routed_x.size(1) :, :]
        else:
            shuffled[:, routed_x.size(1) :, :].fill_(mask_token)

        gather_idx = ids_restore.unsqueeze(-1).expand_as(shuffled)
        return torch.take_along_dim(shuffled, gather_idx, dim=1)


def default_ltx_tread_route(ltx_version: str) -> dict[str, Any]:
    """Return the default TREAD route window for LTX training."""
    route: dict[str, Any] = {"selection_ratio": 0.5, "target": "video"}
    if str(ltx_version) == "2.3":
        route["start_layer_idx"] = 3
        route["end_layer_idx"] = -4
    else:
        route["start_layer_idx"] = 2
        route["end_layer_idx"] = -2
    return route


def parse_tread_args(
    raw_args: Any,
    *,
    total_layers: int,
    default_route: dict[str, Any],
) -> Optional[dict[str, list[dict[str, Any]]]]:
    """Parse musubi's single-route ``--tread_args key=value ...`` CLI form."""
    if raw_args is None or raw_args is False:
        return None

    route = dict(default_route)

    if isinstance(raw_args, str):
        raw_text = raw_args.strip()
        items = shlex.split(raw_text) if raw_text else []
    elif isinstance(raw_args, (list, tuple)):
        items = [str(item).strip() for item in raw_args if str(item).strip()]
    elif raw_args is True:
        items = []
    else:
        raise TypeError(f"tread must be a key=value list, string, or boolean flag, got: {type(raw_args)}")

    key_aliases = {
        "selection_ratio": "selection_ratio",
        "ratio": "selection_ratio",
        "start_layer_idx": "start_layer_idx",
        "start": "start_layer_idx",
        "end_layer_idx": "end_layer_idx",
        "end": "end_layer_idx",
        "target": "target",
        "modality": "target",
    }
    for item in items:
        if "=" not in item:
            raise ValueError(f"--tread_args expects key=value items, got: {item!r}")
        raw_key, raw_value = item.split("=", 1)
        key = key_aliases.get(raw_key.strip())
        if key is None:
            raise ValueError(
                f"Unknown --tread_args option {raw_key!r}. Supported keys: selection_ratio, start_layer_idx, end_layer_idx, target."
            )
        route[key] = raw_value.strip()

    selection_ratio = float(route["selection_ratio"])
    if not 0.0 <= selection_ratio < 1.0:
        raise ValueError(f"tread.selection_ratio must be in [0.0, 1.0), got {selection_ratio}")
    target = str(route.get("target", "video")).strip().lower()
    if target not in {"video", "audio", "both"}:
        raise ValueError(f"tread.target must be one of video, audio, both; got {target!r}")

    start_layer_idx = int(route["start_layer_idx"])
    end_layer_idx = int(route["end_layer_idx"])
    if start_layer_idx < 0:
        start_layer_idx += total_layers
    if end_layer_idx < 0:
        end_layer_idx += total_layers

    if not 0 <= start_layer_idx < total_layers:
        raise ValueError(f"tread.start_layer_idx resolved to {start_layer_idx}, but model has {total_layers} layers")
    if not 0 <= end_layer_idx < total_layers:
        raise ValueError(f"tread.end_layer_idx resolved to {end_layer_idx}, but model has {total_layers} layers")
    if end_layer_idx < start_layer_idx:
        raise ValueError(f"tread route has end_layer_idx {end_layer_idx} before start_layer_idx {start_layer_idx}")

    return {
        "routes": [
            {
                "selection_ratio": selection_ratio,
                "start_layer_idx": start_layer_idx,
                "end_layer_idx": end_layer_idx,
                "target": target,
            }
        ]
    }
