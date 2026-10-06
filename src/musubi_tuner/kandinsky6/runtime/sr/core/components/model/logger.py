"""Inference model diagnostic decorators."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from torch import nn

_T = TypeVar("_T")


def get_full_tensor(tensor: torch.Tensor, *, emergency: bool = False) -> torch.Tensor:
    """Materialize a DTensor into a regular tensor.

    Args:
        tensor: Input tensor, possibly a DTensor with a ``full_tensor`` method.
        emergency: If True, skip the distributed ``full_tensor`` call.

    Returns:
        Full materialized tensor.
    """
    if hasattr(tensor, "full_tensor") and not emergency:
        return tensor.full_tensor()
    return tensor


# Mapping tables for converting internal `model_status/` keys to ClearML `title/series` format.
# _EXACT_MAP and _SUFFIX_MAP handle non-layerwise (aggregated) keys.
# _EXTENDED_MAP and _EXTENDED_SUFFIX_MAP handle layerwise (`#`-suffixed) keys only
# (dispatched by `_remap_layerwise_key`). The same base key (e.g. "visual_norm_mean")
# may appear in both an extended and a non-extended map — no conflict because the
# `#` presence determines which path is taken.
_EXTENDED_MAP: dict[str, str] = {
    "visual_abs_mean": "utility/visual_hidden-abs_mean",
    "visual_norm_mean": "utility/visual_hidden-norm_mean",
}
_EXTENDED_SUFFIX_MAP: dict[str, str] = {
    "q_norm_mean": "utility/qkv_norm-q_mean",
    "k_norm_mean": "utility/qkv_norm-k_mean",
    "v_norm_mean": "utility/qkv_norm-v_mean",
    "q_before_rms_norm_mean": "utility/qk_before_rms-q_mean",
    "k_before_rms_norm_mean": "utility/qk_before_rms-k_mean",
}
_EXACT_MAP: dict[str, str] = {
    "visual_norm_mean": "utility/visual_norm-mean",
    "visual_norm_max": "utility/visual_norm-max",
}
_SUFFIX_MAP: dict[str, str] = {
    "softmax_norm_max": "utility/softmax_norm-max",
    "softmax_norm_mean": "utility/softmax_norm-mean",
    "softmax_norm_min": "utility/softmax_norm-min",
    "block_mask_sparsity_mean": "utility/block_mask_sparsity-mean",
}


def _remap_layerwise_key(base: str, idx: str) -> str | None:
    """Remap a layerwise (``#``-suffixed) key segment.

    Args:
        base: Key segment before the ``#`` separator.
        idx: Layer index string after the ``#`` separator.

    Returns:
        Remapped key string, or ``None`` if no mapping found.
    """
    if base in _EXTENDED_MAP:
        return f"{_EXTENDED_MAP[base]}#{idx}"
    for pattern, replacement in _EXTENDED_SUFFIX_MAP.items():
        if base.endswith(pattern):
            return f"{replacement}#{idx}"
    return None


def _remap_key(key: str) -> str:
    """Remap ``model_status/`` keys to ClearML-friendly ``title/series`` format.

    Keys that don't match any known pattern pass through unchanged.

    Args:
        key: Metric key from the aggregation loop, e.g. ``"model_status/visual_norm_mean"``.

    Returns:
        Remapped key, e.g. ``"visual_norm/mean"``.
    """
    if not key.startswith("model_status/"):
        return key
    suffix = key.removeprefix("model_status/")

    if "#" in suffix:
        base, idx = suffix.split("#", 1)
        return _remap_layerwise_key(base, idx) or key

    if suffix in _EXACT_MAP:
        return _EXACT_MAP[suffix]

    for pattern, replacement in _SUFFIX_MAP.items():
        if suffix.endswith(pattern):
            return replacement

    return key


class ModelLogger:
    """Diagnostic logging for transformer models via class decorators.

    Provides decorators that instrument ``__init__`` and ``forward`` methods
    to collect norm, gradient, and attention statistics.
    """

    USE_EXTENDED_LOGS: bool = False

    @classmethod
    def set_extended_logs(cls, *, value: bool = True) -> None:
        """Enable or disable extended per-layer logging.

        Args:
            value: If True, log per-layer weight norms, QKV norms, etc.
        """
        cls.USE_EXTENDED_LOGS = value

    @staticmethod
    def log_transformer(klass: type[_T]) -> type[_T]:
        """Class decorator that adds log collection to a top-level transformer.

        Wraps ``__init__`` to initialize ``log_dict`` and ``forward`` to
        collect logs from all child modules after each forward pass.
        Aggregated logs are accessible via the injected ``get_logs`` method.

        Args:
            klass: Transformer class to instrument.

        Returns:
            The same class with logging wrappers applied.
        """

        def log_for_init(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = func(self, *args, **kwargs)
                self.log_dict = {}
                return result

            return wrapper

        def collate_log(self: nn.Module, log_dict: dict[str, list[Any]]) -> dict[str, list[Any]]:
            """Recursively gather logs from child modules into ``log_dict``."""
            for module in self.__dict__["_modules"].values():
                if isinstance(module, nn.ModuleList):
                    log_dict.update(collate_log(module, log_dict))
                elif getattr(module, "get_logs", None) is not None:
                    for log_name, log_value in module.get_logs().items():
                        log_dict.setdefault(log_name, []).append(log_value)
            return log_dict

        def log_for_forward(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = func(self, *args, **kwargs)
                self.log_dict = collate_log(self, {})
                return result

            return wrapper

        def wrap(cls: type[_T]) -> type[_T]:
            def get_logs(self: Any) -> dict[str, Any]:
                log_dict = {f"model_status/{key}": value for key, value in self.log_dict.items()}
                keys = list(log_dict.keys())
                for key in keys:
                    values = torch.stack(log_dict[key])
                    if key.endswith("max"):
                        log_dict[key] = values.max().cpu().item()
                    elif key.endswith(("mean", "std")):
                        log_dict[key] = values.mean().cpu().item()
                    elif key.endswith("min"):
                        log_dict[key] = values.min().cpu().item()
                    elif key.endswith("#"):
                        for i, v in enumerate(values):
                            log_dict[key + f"{i}"] = v.cpu().item()
                        log_dict.pop(key, None)
                    else:
                        log_dict[key] = values
                return {_remap_key(k): v for k, v in log_dict.items()}

            cls.__init__ = log_for_init(cls.__init__)  # type: ignore[method-assign]
            cls.forward = log_for_forward(cls.forward)  # type: ignore[method-assign]
            cls.get_logs = get_logs  # type: ignore[attr-defined]
            return cls

        return wrap(klass)

    @staticmethod
    def log_transformer_block(klass: type[_T]) -> type[_T]:
        """Class decorator that adds log collection to a transformer block.

        Wraps ``forward`` to log visual hidden-state norms and collect logs
        from child modules. Extended mode also logs per-layer statistics
        (keys ending with ``#``).

        Args:
            klass: Transformer block class to instrument.

        Returns:
            The same class with logging wrappers applied.
        """

        def log_for_init(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = func(self, *args, **kwargs)
                self.log_dict = {}
                return result

            return wrapper

        def log_for_forward(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                visual_result = func(self, *args, **kwargs)
                for module_name, module in self.__dict__["_modules"].items():
                    if getattr(module, "get_logs", None) is not None:
                        for log_name, log_value in module.get_logs().items():
                            self.log_dict[f"{module_name}_{log_name}"] = log_value
                visual_norm = torch.norm(visual_result.detach(), dim=-1)
                self.log_dict["visual_norm_mean"] = visual_norm.mean()
                self.log_dict["visual_norm_max"] = visual_norm.max()

                # log hidden states blockwise
                if ModelLogger.USE_EXTENDED_LOGS:
                    visual_abs = visual_result.detach().abs()
                    # end name with '#' for layerwise logging
                    self.log_dict["visual_abs_mean#"] = visual_abs.mean(dim=-1).mean()
                    self.log_dict["visual_norm_mean#"] = visual_norm.mean()
                return visual_result

            return wrapper

        def wrap(cls: type[_T]) -> type[_T]:
            def get_logs(self: Any) -> dict[str, Any]:
                return self.log_dict.copy()

            cls.__init__ = log_for_init(cls.__init__)  # type: ignore[method-assign]
            cls.forward = log_for_forward(cls.forward)  # type: ignore[method-assign]
            cls.get_logs = get_logs  # type: ignore[attr-defined]
            return cls

        return wrap(klass)

    @staticmethod
    def log_attention(klass: type[_T]) -> type[_T]:
        """Class decorator that adds attention-specific diagnostic logging.

        Instruments ``get_qkv``, ``norm_qk``, ``scaled_dot_product_attention``,
        and ``attention_flex`` methods (when present) to log QKV norms,
        softmax statistics, and block-mask sparsity.

        Args:
            klass: Attention class to instrument.

        Returns:
            The same class with logging wrappers applied.
        """

        def log_for_init(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = func(self, *args, **kwargs)
                self.log_dict = {}
                return result

            return wrapper

        def log_qkv(
            func: Callable[..., Any],
            *,
            layerwise: bool = False,
        ) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                q, k, v = func(self, *args, **kwargs)
                if ModelLogger.USE_EXTENDED_LOGS:
                    q_norm, k_norm, v_norm = (
                        torch.norm(q.detach(), dim=-1),
                        torch.norm(k.detach(), dim=-1),
                        torch.norm(v.detach(), dim=-1),
                    )
                    postfix = "#" if layerwise else ""
                    self.log_dict["q_norm_mean" + postfix] = q_norm.mean()
                    self.log_dict["k_norm_mean" + postfix] = k_norm.mean()
                    self.log_dict["v_norm_mean" + postfix] = v_norm.mean()
                return q, k, v

            return wrapper

        def log_before_norm_qk(
            func: Callable[..., Any],
            *,
            layerwise: bool = False,
        ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
                q, k = func(self, *args, **kwargs)
                if ModelLogger.USE_EXTENDED_LOGS:
                    q_before, k_before = args[:2]
                    q_norm, k_norm = (
                        torch.norm(q_before.detach(), dim=-1),
                        torch.norm(k_before.detach(), dim=-1),
                    )
                    postfix = "#" if layerwise else ""
                    self.log_dict["q_before_rms_norm_mean" + postfix] = q_norm.mean()
                    self.log_dict["k_before_rms_norm_mean" + postfix] = k_norm.mean()
                return q, k

            return wrapper

        def log_sdpa(func: Callable[..., Any], *, layerwise: bool = False) -> Callable[..., torch.Tensor]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                result, softmax_lse, _ = func(self, *args, return_attn_probs=True, **kwargs)
                softmax_lse = softmax_lse.detach()
                postfix = "#" if layerwise else ""
                self.log_dict["softmax_norm_max" + postfix] = softmax_lse.max()
                self.log_dict["softmax_norm_mean" + postfix] = softmax_lse.mean()
                self.log_dict["softmax_norm_min" + postfix] = softmax_lse.min()
                return result

            return wrapper

        def log_for_sparsity(func: Callable[..., Any], *, layerwise: bool = False) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result, sparsity = func(self, *args, return_sparsity=True, **kwargs)
                postfix = "#" if layerwise else ""
                self.log_dict["block_mask_sparsity_mean" + postfix] = torch.tensor(sparsity, dtype=torch.float32)
                return result

            return wrapper

        def wrap(cls: type[_T]) -> type[_T]:
            def get_logs(self: Any) -> dict[str, Any]:
                return self.log_dict.copy()

            cls.__init__ = log_for_init(cls.__init__)  # type: ignore[method-assign]

            if hasattr(cls, "scaled_dot_product_attention"):
                cls.scaled_dot_product_attention = log_sdpa(  # type: ignore[attr-defined]
                    cls.scaled_dot_product_attention,  # type: ignore[attr-defined]
                    layerwise=False,
                )

            if hasattr(cls, "attention_flex"):
                cls.attention_flex = log_for_sparsity(  # type: ignore[attr-defined]
                    cls.attention_flex,  # type: ignore[attr-defined]
                    layerwise=False,
                )

            if hasattr(cls, "get_qkv"):
                cls.get_qkv = log_qkv(cls.get_qkv, layerwise=True)  # type: ignore[attr-defined]

            if hasattr(cls, "norm_qk"):
                cls.norm_qk = log_before_norm_qk(cls.norm_qk, layerwise=True)  # type: ignore[attr-defined]

            cls.get_logs = get_logs  # type: ignore[attr-defined]
            return cls

        return wrap(klass)

    @staticmethod
    def log_model_status(model: nn.Module, *, emergency: bool = False) -> dict[str, Any]:
        """Collect weight norms, gradient norms, and per-module logs from the model.

        Args:
            model: Model with a ``get_logs`` method (injected by ``log_transformer``).
            emergency: If True, skip distributed ops (useful after NaN in backward).

        Returns:
            Dict of diagnostic metrics with ClearML-friendly ``title/series`` keys.
        """
        params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        grads = [p.grad for p in model.parameters() if p.grad is not None and p.requires_grad]
        # NOTE: in case of exception triggered on NaN in backward, these objects
        # can become usual Tensor instead of DTensor
        weights_norm = get_full_tensor(nn.utils.get_total_norm(params), emergency=emergency).cpu().item()
        grad_norm = get_full_tensor(nn.utils.get_total_norm(grads), emergency=emergency).cpu().item()
        log_dict = model.get_logs()  # type: ignore[operator]
        log_dict.update({"utility/weights_norm_all": weights_norm, "grad_norm/value": grad_norm})
        if ModelLogger.USE_EXTENDED_LOGS:
            log_dict.update(
                {
                    f"utility/weights_norm_{n}": get_full_tensor(nn.utils.get_total_norm(p), emergency=emergency).cpu().item()
                    for n, p in model.named_parameters()
                    if p.requires_grad
                }
            )
        return log_dict

    @staticmethod
    def log_system_status(step_time: float) -> dict[str, float]:
        """Return step timing metric.

        Args:
            step_time: Wall-clock time of the last training step in seconds.

        Returns:
            Dict with ``step_time/seconds`` key.
        """
        return {"utility/step_time_seconds": step_time}
