"""BAdam (block-coordinate Adam) optimizer wrapper.

BAdam applies block-coordinate training on top of a base optimizer. Only the
current block and configured always-active parameters require gradients;
inactive optimizer state can be purged at each switch to keep memory bounded.

Wraps any standard ``torch.optim.Optimizer`` (AdamW, AdamW8bit, CAME8bit,
Adafactor, etc.) and exposes ``enable_gradient_release`` for a fused-like
per-parameter step path that frees gradient peak memory during backward.
"""

from __future__ import annotations

import random
import re
import warnings
from typing import Any, Iterable, Sequence

import torch


BADAM_OPTIMIZER_ALIASES = {"badam", "blockadam", "block_optimizer", "blockoptimizer"}
BADAM_RATIO_OPTIMIZER_ALIASES = {"badamratio", "badam_ratio", "blockadamratio"}


def is_badam_optimizer_type(optimizer_type: str) -> bool:
    return optimizer_type.lower() in BADAM_OPTIMIZER_ALIASES


def is_badam_ratio_optimizer_type(optimizer_type: str) -> bool:
    return optimizer_type.lower() in BADAM_RATIO_OPTIMIZER_ALIASES


def flatten_optimizer_params(params_or_groups: Sequence[Any]) -> list[torch.nn.Parameter]:
    """Flatten torch optimizer params/groups while preserving first occurrence order."""

    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for item in params_or_groups:
        if isinstance(item, dict):
            group_params = item.get("params", [])
        else:
            group_params = [item]
        for param in group_params:
            if not isinstance(param, torch.nn.Parameter):
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params


def normalize_block_prefixes(
    prefixes: Iterable[str | Iterable[str]],
) -> list[tuple[str, ...]]:
    normalized: list[tuple[str, ...]] = []
    for item in prefixes:
        if isinstance(item, str):
            group = (item,)
        else:
            group = tuple(prefix for prefix in item if isinstance(prefix, str))
        if not group:
            continue
        normalized.append(group)
    return normalized


def group_block_prefixes(
    block_prefixes: Sequence[tuple[str, ...]],
    group_size: int = 1,
) -> list[tuple[str, ...]]:
    """Merge consecutive BAdam block groups into larger active windows."""

    group_size = int(group_size)
    if group_size < 1:
        raise ValueError("BAdam block_group_size must be >= 1")
    if group_size == 1:
        return list(block_prefixes)

    grouped: list[tuple[str, ...]] = []
    for start in range(0, len(block_prefixes), group_size):
        merged: list[str] = []
        for group in block_prefixes[start : start + group_size]:
            merged.extend(group)
        grouped.append(tuple(merged))
    return grouped


def infer_indexed_block_prefixes(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> list[tuple[str, ...]]:
    """Infer transformer block prefixes by matching ``...blocks.N.`` parameter names.

    Works for any model whose blocks are named ``something_blocks.N.`` or
    ``blocks.N.`` (including LTX-2 ``transformer_blocks``, Wan ``blocks``, etc.).
    """

    prefixes_by_index: dict[int, str] = {}
    for name, _param in named_parameters:
        match = re.match(r"^(.*?blocks\.(\d+)\.)", name)
        if match is None:
            continue
        prefixes_by_index[int(match.group(2))] = match.group(1)

    return [(prefixes_by_index[index],) for index in sorted(prefixes_by_index)]


def infer_transformer_block_prefixes(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    include_embedding: bool = False,
    include_lm_head: bool = False,
) -> list[tuple[str, ...]]:
    """Infer block-coordinate prefixes from trainable transformer parameters."""

    indexed_blocks = infer_indexed_block_prefixes(named_parameters)
    if indexed_blocks:
        prefixes = list(indexed_blocks)
        block_flat = [prefix for group in prefixes for prefix in group]
        embed_prefixes: dict[str, str] = {}
        remainder: list[str] = []
        embed_pattern = re.compile(r"^(.*?embed[^.]*\.)")
        for name, _param in named_parameters:
            if _matches_any(name, block_flat):
                continue
            embed_match = embed_pattern.match(name)
            if embed_match is not None and include_embedding:
                embed_prefixes[embed_match.group(1)] = embed_match.group(1)
            else:
                remainder.append(name)
        if embed_prefixes:
            prefixes = [(prefix,) for prefix in embed_prefixes.values()] + prefixes
        if include_lm_head and remainder:
            prefixes.append(tuple(remainder))
    else:
        prefixes_by_index: dict[int, str] = {}
        remainder: list[str] = []
        embed_prefixes: dict[str, str] = {}
        layer_pattern = re.compile(r"^(.*?layers\.(\d+)\.)")
        embed_pattern = re.compile(r"^(.*?embed[^.]*\.)")
        for name, _param in named_parameters:
            layer_match = layer_pattern.match(name)
            if layer_match is not None:
                prefixes_by_index[int(layer_match.group(2))] = layer_match.group(1)
                continue
            embed_match = embed_pattern.match(name)
            if embed_match is not None and include_embedding:
                embed_prefixes[embed_match.group(1)] = embed_match.group(1)
                continue
            remainder.append(name)
        prefixes = [(prefix,) for prefix in embed_prefixes.values()]
        prefixes.extend((prefixes_by_index[index],) for index in sorted(prefixes_by_index))
        if include_lm_head and remainder:
            prefixes.append(tuple(remainder))

    return prefixes


def _matches_any(name: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        if prefix.startswith("re:"):
            if re.search(prefix[3:], name):
                return True
        elif prefix in name:
            return True
    return False


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"BAdam {name} must be a boolean value, got {value!r}")


class BlockOptimizer(torch.optim.Optimizer):
    """Block-coordinate wrapper for a base optimizer.

    The wrapper keeps the base optimizer's parameter groups stable so schedulers
    and Accelerate wrappers continue to see the same groups throughout training.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        block_prefixes: Sequence[tuple[str, ...]],
        *,
        switch_block_every: int = 100,
        switch_mode: str = "random",
        start_block: int | None = None,
        always_active_prefixes: Sequence[str] | None = None,
        include_non_block: bool = True,
        use_fp32_active_copy: bool = True,
        purge_inactive_state: bool = True,
        reset_state_on_switch: bool = True,
        bread_sgd_enabled: bool = False,
        bread_sgd_mode: str = "all",
        bread_sgd_window_blocks: int = 0,
        bread_sgd_lr_scale: float = 1.0,
        bread_sgd_use_sign: bool = False,
        verbose: int = 1,
        logger: Any | None = None,
    ) -> None:
        if switch_block_every < 1:
            raise ValueError("switch_block_every must be >= 1")
        if switch_mode not in {"random", "ascending", "descending", "fixed"}:
            raise ValueError("switch_mode must be random, ascending, descending, or fixed")
        if not block_prefixes:
            raise ValueError("BAdam requires at least one block prefix")

        self.base_optimizer = base_optimizer
        self.named_parameters_list = list(named_parameters)
        self.block_prefixes = list(block_prefixes)
        self.switch_block_every = int(switch_block_every)
        self.switch_mode = switch_mode
        self.always_active_prefixes = tuple(always_active_prefixes or ())
        self.include_non_block = _as_bool(include_non_block, name="include_non_block")
        self.use_fp32_active_copy = _as_bool(use_fp32_active_copy, name="use_fp32_active_copy")
        self.purge_inactive_state = _as_bool(purge_inactive_state, name="purge_inactive_state")
        self.reset_state_on_switch = _as_bool(reset_state_on_switch, name="reset_state_on_switch")
        if self.use_fp32_active_copy and not self.reset_state_on_switch:
            raise ValueError(
                "BAdam use_fp32_active_copy=True requires reset_state_on_switch=True. "
                "The base optimizer state is attached to temporary fp32 active copies "
                "and cannot be safely reused across block switches without remapping."
            )
        self.verbose = int(verbose)
        self.logger = logger
        self.global_step = 0
        self._random_order: list[int] = []
        self._active_param_ids: set[int] = set()
        self._lp_to_hp: dict[torch.nn.Parameter, torch.nn.Parameter] = {}
        self._hp_to_lp: dict[torch.nn.Parameter, torch.nn.Parameter] = {}
        self._active_source_group_indices: list[int] = []
        self._block_managed_param_ids = self._collect_block_managed_param_ids()
        self._gradient_release_enabled: bool = False
        self._gr_accelerator: Any = None
        self._gr_max_grad_norm: float = 0.0
        self._gr_fused_step_state: dict[str, Any] | None = None
        self._gr_hook_handles: list[Any] = []
        self.bread_sgd_enabled = _as_bool(bread_sgd_enabled, name="bread_sgd")
        self.bread_sgd_mode = str(bread_sgd_mode).lower()
        self.bread_sgd_window_blocks = int(bread_sgd_window_blocks)
        self.bread_sgd_lr_scale = float(bread_sgd_lr_scale)
        self.bread_sgd_use_sign = _as_bool(bread_sgd_use_sign, name="bread_sgd_use_sign")
        if not self.bread_sgd_enabled:
            self.bread_sgd_mode = "disabled"
        if self.bread_sgd_mode in {"off", "false", "0", "none"}:
            self.bread_sgd_mode = "disabled"
        if self.bread_sgd_mode in {"true", "1", "on", "enabled"}:
            self.bread_sgd_mode = "all"
        if self.bread_sgd_mode not in {"disabled", "all", "partial", "window"}:
            raise ValueError("bread_sgd_mode must be disabled, all, partial, or window")
        if self.bread_sgd_window_blocks < 0:
            raise ValueError("bread_sgd_window_blocks must be >= 0")
        if self.bread_sgd_mode == "window" and self.bread_sgd_window_blocks < 1:
            raise ValueError("bread_sgd_window_blocks must be >= 1 when bread_sgd_mode=window")
        self._bread_sgd_hook_handles: list[Any] = []
        self._bread_sgd_param_ids: set[int] = set()
        self._source_group_by_param_id: dict[int, dict[str, Any]] = {}

        fp32_params = [name for name, param in self.named_parameters_list if param.dtype == torch.float32]
        if fp32_params and self.use_fp32_active_copy:
            warnings.warn(
                "BAdam expects the model to be loaded in fp16/bf16 for memory savings, "
                f"but found fp32 trainable parameters: {fp32_params}",
                RuntimeWarning,
                stacklevel=2,
            )

        if start_block is not None:
            start_block = int(start_block)
            if start_block < 0 or start_block >= len(self.block_prefixes):
                raise ValueError(f"start_block={start_block} is outside {len(self.block_prefixes)} BAdam blocks")
            self.current_block_idx = start_block
        elif self.switch_mode == "descending":
            self.current_block_idx = len(self.block_prefixes) - 1
        elif self.switch_mode == "random":
            self.current_block_idx = self._pop_random_block()
        else:
            self.current_block_idx = 0

        super().__init__(base_optimizer.param_groups, base_optimizer.defaults)
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state
        self._source_group_by_param_id = self._build_source_group_lookup()

        self.switch_trainable_params(advance=False)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer,
            "global_step": self.global_step,
            "current_block_idx": self.current_block_idx,
            "random_order": self._random_order,
        }

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self.logger is None:
            return
        log_fn = getattr(self.logger, level, None)
        if callable(log_fn):
            log_fn(message, *args)

    def _collect_block_managed_param_ids(self) -> set[int]:
        managed: set[int] = set()
        flat_prefixes = [prefix for group in self.block_prefixes for prefix in group]
        for name, param in self.named_parameters_list:
            if _matches_any(name, flat_prefixes):
                managed.add(id(param))
        return managed

    def _pop_random_block(self) -> int:
        if not self._random_order:
            self._random_order = list(range(len(self.block_prefixes)))
            random.shuffle(self._random_order)
            if self.verbose >= 2:
                self._log("info", "BAdam next random block order: %s", self._random_order)
        return self._random_order.pop()

    def _advance_block_idx(self) -> None:
        if self.switch_mode == "random":
            self.current_block_idx = self._pop_random_block()
        elif self.switch_mode == "ascending":
            self.current_block_idx = (self.current_block_idx + 1) % len(self.block_prefixes)
        elif self.switch_mode == "descending":
            self.current_block_idx = (self.current_block_idx - 1) % len(self.block_prefixes)
        elif self.switch_mode == "fixed":
            return

    def _is_always_active(self, name: str, param: torch.nn.Parameter) -> bool:
        if _matches_any(name, self.always_active_prefixes):
            return True
        return self.include_non_block and id(param) not in self._block_managed_param_ids

    def _build_source_group_lookup(self) -> dict[int, dict[str, Any]]:
        lookup: dict[int, dict[str, Any]] = {}
        for group in self.param_groups:
            for param in group.get("params", []):
                if isinstance(param, torch.nn.Parameter):
                    lookup[id(param)] = group
        return lookup

    def _select_bread_sgd_param_ids(
        self,
        active_status: Sequence[tuple[str, torch.nn.Parameter, bool]],
    ) -> set[int]:
        """Select inactive parameters that receive BREAD's lightweight SGD update."""
        if not self.bread_sgd_enabled or self.bread_sgd_mode == "disabled":
            return set()

        if self.bread_sgd_mode == "all":
            return {id(param) for _name, param, is_active in active_status if not is_active}

        start = self.current_block_idx + 1
        stop = len(self.block_prefixes)
        if self.bread_sgd_mode == "window":
            stop = min(stop, start + self.bread_sgd_window_blocks)
        elif self.bread_sgd_window_blocks > 0:
            stop = min(stop, start + self.bread_sgd_window_blocks)
        if start >= stop:
            return set()

        correction_prefixes = tuple(prefix for group in self.block_prefixes[start:stop] for prefix in group)
        if not correction_prefixes:
            return set()

        return {id(param) for name, param, is_active in active_status if not is_active and _matches_any(name, correction_prefixes)}

    def switch_trainable_params(self, *, advance: bool = True) -> None:
        if advance:
            self._advance_block_idx()

        active_prefixes = self.block_prefixes[self.current_block_idx]
        active_ids: set[int] = set()
        active_names: list[str] = []
        inactive_count = 0
        active_status: list[tuple[str, torch.nn.Parameter, bool]] = []

        for name, param in self.named_parameters_list:
            is_active = self._is_always_active(name, param) or _matches_any(
                name,
                active_prefixes,
            )
            active_status.append((name, param, is_active))
            if is_active:
                active_ids.add(id(param))
                if self.verbose >= 2:
                    active_names.append(name)

        self._bread_sgd_param_ids = self._select_bread_sgd_param_ids(active_status)

        for name, param, is_active in active_status:
            has_bread_sgd = id(param) in self._bread_sgd_param_ids
            param.requires_grad_(is_active or has_bread_sgd)
            if not is_active:
                inactive_count += 1
            if not is_active and not has_bread_sgd:
                param.grad = None
            if not is_active and self.purge_inactive_state:
                self.base_optimizer.state.pop(param, None)

        self._active_param_ids = active_ids

        if self.use_fp32_active_copy:
            self._rebuild_active_fp32_param_groups()
        else:
            self.base_optimizer.param_groups = self.param_groups
            if self.reset_state_on_switch:
                self.base_optimizer.state.clear()

        if self._gradient_release_enabled:
            self._refresh_gradient_release_hooks()
        if self.bread_sgd_enabled and self.bread_sgd_mode != "disabled":
            self._refresh_bread_sgd_hooks()

        if self.verbose >= 1:
            self._log(
                "info",
                "BAdam active block %d/%d prefixes=%s active_params=%d inactive_params=%d bread_sgd_params=%d",
                self.current_block_idx,
                len(self.block_prefixes),
                list(active_prefixes),
                len(active_ids),
                inactive_count,
                len(self._bread_sgd_param_ids),
            )
        if active_names:
            self._log("info", "BAdam active parameter names: %s", active_names)

    def _clone_group_options(self, group: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in group.items() if key != "params"}

    def _rebuild_active_fp32_param_groups(self) -> None:
        self._lp_to_hp = {}
        self._hp_to_lp = {}
        self._active_source_group_indices = []
        active_groups: list[dict[str, Any]] = []

        for source_group_idx, group in enumerate(self.param_groups):
            active_hp_params: list[torch.nn.Parameter] = []
            for param in group.get("params", []):
                if id(param) not in self._active_param_ids:
                    continue
                hp_param = torch.nn.Parameter(
                    param.detach().clone().float(),
                    requires_grad=True,
                )
                self._lp_to_hp[param] = hp_param
                self._hp_to_lp[hp_param] = param
                active_hp_params.append(hp_param)

            if not active_hp_params:
                continue
            active_group = self._clone_group_options(group)
            active_group["params"] = active_hp_params
            active_group["_badam_source_group_idx"] = source_group_idx
            active_groups.append(active_group)
            self._active_source_group_indices.append(source_group_idx)

        if not active_groups:
            raise RuntimeError("BAdam active block has no optimizer parameters.")
        self.base_optimizer.param_groups = active_groups
        if self.reset_state_on_switch:
            self.base_optimizer.state.clear()

    def _sync_active_optimizer_lrs(self) -> None:
        if not self.use_fp32_active_copy:
            return
        for active_group in self.base_optimizer.param_groups:
            source_group_idx = active_group.get("_badam_source_group_idx")
            if source_group_idx is None:
                continue
            source_group = self.param_groups[int(source_group_idx)]
            for key in ("lr", "weight_decay"):
                if key in source_group:
                    active_group[key] = source_group[key]

    def _move_grads_to_hp(self) -> None:
        for lp_param, hp_param in self._lp_to_hp.items():
            if lp_param.grad is None:
                hp_param.grad = None
                continue
            hp_param.grad = lp_param.grad.detach().float()
            lp_param.grad = None

    def _copy_hp_to_lp(self) -> None:
        for hp_param, lp_param in self._hp_to_lp.items():
            lp_param.data.copy_(hp_param.detach().to(dtype=lp_param.dtype))

    def synchronize_for_checkpoint(self) -> None:
        if self.use_fp32_active_copy:
            self._copy_hp_to_lp()

    # --- gradient-release (fused-like) ---------------------------------------

    def enable_gradient_release(
        self,
        *,
        accelerator: Any = None,
        max_grad_norm: float = 0.0,
        fused_step_state: dict[str, Any] | None = None,
    ) -> None:
        """Enable per-parameter step via post-accumulate-grad hooks.

        Hooks release LP grad immediately after accumulate, run a single-param
        base-optimizer step on the HP fp32 copy, then write the updated weight
        back to LP and free HP grad. Net effect: gradient peak shrinks from a
        full active-block worth to per-tensor.

        Must be called AFTER ``accelerator.prepare()`` so hooks attach to the
        underlying parameters that backward will populate.
        """
        if not self.use_fp32_active_copy:
            raise RuntimeError(
                "BAdam gradient-release currently requires use_fp32_active_copy=True; "
                "the per-param step path operates on HP fp32 copies."
            )
        self._gradient_release_enabled = True
        self._gr_accelerator = accelerator
        self._gr_max_grad_norm = float(max_grad_norm or 0.0)
        self._gr_fused_step_state = fused_step_state
        self._refresh_gradient_release_hooks()
        if self.bread_sgd_enabled and self.bread_sgd_mode != "disabled":
            self._refresh_bread_sgd_hooks()
        if self.verbose >= 1:
            self._log(
                "info",
                "BAdam gradient-release enabled (max_grad_norm=%.4f, defer_aware=%s)",
                self._gr_max_grad_norm,
                fused_step_state is not None,
            )

    def _unregister_gradient_release_hooks(self) -> None:
        for handle in self._gr_hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._gr_hook_handles = []

    def _refresh_gradient_release_hooks(self) -> None:
        if not self._gradient_release_enabled:
            return
        self._unregister_gradient_release_hooks()
        for lp_param, hp_param in self._lp_to_hp.items():
            handle = lp_param.register_post_accumulate_grad_hook(self._make_grad_release_hook(lp_param, hp_param))
            self._gr_hook_handles.append(handle)

    def _unregister_bread_sgd_hooks(self) -> None:
        for handle in self._bread_sgd_hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._bread_sgd_hook_handles = []

    def _refresh_bread_sgd_hooks(self) -> None:
        """Register cheap SGD hooks on selected inactive params.

        Implements BREAD's "landscape correction" idea (Luo et al. 2025): when a
        block is frozen, applying a small on-the-fly SGD update during the same
        backward keeps frozen weights from drifting away from the active block's
        current optimum. Updates use the wrapper's current LR scaled by
        bread_sgd_lr_scale; grads are released immediately after the update.
        """
        if not self.bread_sgd_enabled or self.bread_sgd_mode == "disabled":
            return
        self._unregister_bread_sgd_hooks()
        for name, param in self.named_parameters_list:
            if id(param) not in self._bread_sgd_param_ids:
                continue
            handle = param.register_post_accumulate_grad_hook(self._make_bread_sgd_hook(param))
            self._bread_sgd_hook_handles.append(handle)

    def _make_bread_sgd_hook(self, param: torch.nn.Parameter):
        def hook(p: torch.Tensor) -> None:
            if p.grad is None:
                return
            group = self._source_group_by_param_id.get(id(param), self.param_groups[0])
            lr = float(group.get("lr", 0.0)) * self.bread_sgd_lr_scale
            if lr == 0.0:
                p.grad = None
                return
            with torch.no_grad():
                if self.bread_sgd_use_sign:
                    p.add_(p.grad.detach().sign(), alpha=-lr)
                else:
                    p.add_(p.grad.detach(), alpha=-lr)
            p.grad = None

        return hook

    def _make_grad_release_hook(
        self,
        lp_param: torch.nn.Parameter,
        hp_param: torch.nn.Parameter,
    ):
        def hook(p: torch.Tensor) -> None:
            if p.grad is None:
                return
            accel = self._gr_accelerator
            sync = True if accel is None else bool(getattr(accel, "sync_gradients", True))
            if sync and self._gr_max_grad_norm > 0 and accel is not None:
                accel.clip_grad_norm_(p, self._gr_max_grad_norm)
            hp_param.grad = p.grad.detach().float()
            p.grad = None
            if not sync:
                return
            fss = self._gr_fused_step_state
            if fss is not None and (bool(fss.get("defer_step")) or bool(fss.get("suspend_step"))):
                return
            self._step_single_hp_param(hp_param)
            lp_param.data.copy_(hp_param.detach().to(lp_param.dtype))
            hp_param.grad = None
            if fss is not None:
                fss["hook_stepped"] = True

        return hook

    def _step_single_hp_param(self, hp_param: torch.nn.Parameter) -> None:
        """Run base optimizer for exactly one HP parameter via temporary group swap."""
        if hp_param.grad is None:
            return
        saved_groups = self.base_optimizer.param_groups
        single_group: dict[str, Any] | None = None
        for group in saved_groups:
            if any(p is hp_param for p in group.get("params", [])):
                single_group = {**group, "params": [hp_param]}
                break
        if single_group is None:
            return
        self.base_optimizer.param_groups = [single_group]
        try:
            self.base_optimizer.step()
        finally:
            self.base_optimizer.param_groups = saved_groups

    def _flush_deferred_grads(self) -> None:
        """Apply optimizer step to any HP params that still hold grads."""
        any_pending = any(hp.grad is not None for hp in self._hp_to_lp)
        if not any_pending:
            return
        self.base_optimizer.step()
        for hp_param, lp_param in self._hp_to_lp.items():
            if hp_param.grad is not None:
                lp_param.data.copy_(hp_param.detach().to(lp_param.dtype))
                hp_param.grad = None

    def _clear_inactive_grads(self) -> None:
        for name, param in self.named_parameters_list:
            if id(param) not in self._active_param_ids and not self._is_always_active(
                name,
                param,
            ):
                param.grad = None

    def step(self, closure: Any | None = None) -> Any:
        if self._gradient_release_enabled:
            self._sync_active_optimizer_lrs()
            self._flush_deferred_grads()
            self._clear_inactive_grads()
            self.global_step += 1
            if self.global_step % self.switch_block_every == 0:
                self.switch_trainable_params(advance=True)
            return None
        self._clear_inactive_grads()
        self._sync_active_optimizer_lrs()
        if self.use_fp32_active_copy:
            self._move_grads_to_hp()
        if closure is None:
            loss = self.base_optimizer.step()
        else:
            loss = self.base_optimizer.step(closure=closure)
        if self.use_fp32_active_copy:
            self._copy_hp_to_lp()
            for hp_param in self._hp_to_lp:
                hp_param.grad = None
        self.global_step += 1
        if self.global_step % self.switch_block_every == 0:
            self.switch_trainable_params(advance=True)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        if not set_to_none:
            for name, param in self.named_parameters_list:
                if id(param) not in self._active_param_ids and not self._is_always_active(
                    name,
                    param,
                ):
                    param.grad = None

    def train(self) -> None:
        train_fn = getattr(self.base_optimizer, "train", None)
        if callable(train_fn):
            train_fn()

    def eval(self) -> None:
        eval_fn = getattr(self.base_optimizer, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def state_dict(self) -> dict[str, Any]:
        state = self.base_optimizer.state_dict()
        state["_badam_wrapper"] = {
            "global_step": self.global_step,
            "current_block_idx": self.current_block_idx,
            "random_order": list(self._random_order),
            "use_fp32_active_copy": self.use_fp32_active_copy,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        badam_state = state_dict.pop("_badam_wrapper", None)

        if isinstance(badam_state, dict):
            self.global_step = int(badam_state.get("global_step", self.global_step))
            current_block_idx = int(badam_state.get("current_block_idx", self.current_block_idx))
            if 0 <= current_block_idx < len(self.block_prefixes):
                self.current_block_idx = current_block_idx
            random_order = badam_state.get("random_order", [])
            if isinstance(random_order, list) and all(isinstance(item, int) for item in random_order):
                self._random_order = list(random_order)
        self.switch_trainable_params(advance=False)
        self.base_optimizer.load_state_dict(state_dict)
        self.state = self.base_optimizer.state


def create_badam_optimizer(
    args: Any,
    transformer: torch.nn.Module,
    trainable_params: Sequence[Any],
    base_optimizer: torch.optim.Optimizer,
    *,
    distributed_context: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> BlockOptimizer:
    """Wrap an already-created base optimizer with the BAdam block-coordinate wrapper."""

    distributed_context = distributed_context or {}
    world_size = int(distributed_context.get("world_size", 1) or 1)
    if world_size > 1 and not _as_bool(getattr(args, "badam_allow_distributed", False), name="allow_distributed"):
        raise ValueError(
            "BAdam changes requires_grad between blocks and is currently supported "
            "only for single-process training. Set badam_allow_distributed only "
            "after validating your distributed setup."
        )

    flat_trainable = flatten_optimizer_params(list(trainable_params))
    trainable_ids = {id(param) for param in flat_trainable}
    named_parameters = [(name, param) for name, param in transformer.named_parameters() if id(param) in trainable_ids]

    matched_ids = {id(param) for _name, param in named_parameters}
    unmatched_count = len(trainable_ids - matched_ids)
    allow_unmatched = _as_bool(getattr(args, "badam_allow_unmatched_params", False), name="allow_unmatched_params")
    if unmatched_count and not allow_unmatched:
        raise ValueError(
            "BAdam requires trainable parameters to be named by the transformer. "
            f"Found {unmatched_count} unmatched trainable parameter(s). This usually "
            "means LoRA/helper/text-encoder params are present; disable BAdam or set "
            "badam_allow_unmatched_params=true to keep them always active."
        )
    if unmatched_count and logger is not None:
        logger.warning(
            "BAdam: %d trainable parameter(s) are not named by the transformer and will remain always active with synthetic names.",
            unmatched_count,
        )
    if unmatched_count and allow_unmatched:
        matched_ids = {id(param) for _name, param in named_parameters}
        unmatched_params = [param for param in flat_trainable if id(param) not in matched_ids]
        named_parameters.extend((f"badam_unmatched.{index}", param) for index, param in enumerate(unmatched_params))
    if not named_parameters:
        raise ValueError(
            "BAdam found no trainable transformer parameters. It is intended for "
            "full or partial transformer fine-tuning, not pure LoRA parameter sets."
        )

    configured_prefixes = normalize_block_prefixes(getattr(args, "badam_block_prefixes", []) or [])
    prefix_mode = str(getattr(args, "badam_block_prefix_mode", "transformer_blocks")).lower()
    if configured_prefixes:
        block_prefixes = configured_prefixes
    elif prefix_mode in {"auto", "transformer_blocks"}:
        block_prefixes = infer_transformer_block_prefixes(
            named_parameters,
            include_embedding=_as_bool(getattr(args, "badam_include_embedding", False), name="include_embedding"),
            include_lm_head=_as_bool(getattr(args, "badam_include_lm_head", False), name="include_lm_head"),
        )
    else:
        block_prefixes = []
    if not block_prefixes:
        raise ValueError(
            "BAdam could not infer any transformer block prefixes from trainable parameters. Set badam_block_prefixes explicitly."
        )
    block_prefixes = group_block_prefixes(
        block_prefixes,
        int(getattr(args, "badam_block_group_size", 1)),
    )

    wrapper = BlockOptimizer(
        base_optimizer,
        named_parameters,
        block_prefixes,
        switch_block_every=int(getattr(args, "badam_switch_block_every", 100)),
        switch_mode=str(getattr(args, "badam_switch_mode", "random")).lower(),
        start_block=getattr(args, "badam_start_block", None),
        always_active_prefixes=(
            list(getattr(args, "badam_always_active_prefixes", []) or []) + list(getattr(args, "badam_active_modules", []) or [])
        ),
        include_non_block=_as_bool(getattr(args, "badam_include_non_block", True), name="include_non_block"),
        use_fp32_active_copy=_as_bool(getattr(args, "badam_use_fp32_active_copy", True), name="use_fp32_active_copy"),
        purge_inactive_state=_as_bool(getattr(args, "badam_purge_inactive_state", True), name="purge_inactive_state"),
        reset_state_on_switch=_as_bool(getattr(args, "badam_reset_state_on_switch", True), name="reset_state_on_switch"),
        bread_sgd_enabled=_as_bool(getattr(args, "badam_bread_sgd", False), name="bread_sgd"),
        bread_sgd_mode=str(getattr(args, "badam_bread_sgd_mode", "all")).lower(),
        bread_sgd_window_blocks=int(getattr(args, "badam_bread_sgd_window_blocks", 0)),
        bread_sgd_lr_scale=float(getattr(args, "badam_bread_sgd_lr_scale", 1.0)),
        bread_sgd_use_sign=_as_bool(getattr(args, "badam_bread_sgd_use_sign", False), name="bread_sgd_use_sign"),
        verbose=int(getattr(args, "badam_verbose", 1)),
        logger=logger,
    )
    if logger is not None:
        logger.info(
            "BAdam wrapped %s with %d block(s), switch_every=%d, mode=%s, bread_sgd=%s.",
            base_optimizer.__class__.__name__,
            len(block_prefixes),
            wrapper.switch_block_every,
            wrapper.switch_mode,
            wrapper.bread_sgd_mode if wrapper.bread_sgd_enabled else "off",
        )
    return wrapper
