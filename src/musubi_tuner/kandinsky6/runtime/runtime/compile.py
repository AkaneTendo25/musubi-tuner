"""DiT torch.compile strategies (no in-module decorators).

Baseline ``torch`` strategy mirrors kandinsky-5-inference
``dit_compile_wrapper(..., compile_type="torch")``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol

import torch
from pydantic import Field
from torch import nn

from musubi_tuner.kandinsky6.runtime.core.components.dit import (
    FusedTransformerDecoderBlock,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)

from .base import ConfigModel

logger = logging.getLogger("kandinsky")

CompileStrategyName = Literal["none", "torch"]


class AotiCompileConfig(ConfigModel):
    """Regional AOTInductor knobs for ``kandy export`` / ``compile_aoti``.

    Defaults match ``torch.compile(mode="max-autotune-no-cudagraphs")`` plus
    packaging with weights outside the SO. Prefer ``cudagraphs=false`` with
    MagCache + FA3 (graphs break on skip / custom-op callouts).
    """

    max_autotune: bool = True
    coordinate_descent_tuning: bool = True
    epilogue_fusion: bool = True
    shape_padding: bool = True
    package_constants_in_so: bool = False
    cudagraphs: bool = False
    force_same_precision: bool = False
    # None → the export CLI uses ``$KANDINSKY_HOME/export/inductor``.
    cache_dir: str | None = None

    def inductor_configs(self) -> dict[str, Any]:
        """Flat dict for ``torch._inductor.aoti_compile_and_package``."""
        cfg: dict[str, Any] = {
            "aot_inductor.package_constants_in_so": self.package_constants_in_so,
            "max_autotune": self.max_autotune,
            "coordinate_descent_tuning": self.coordinate_descent_tuning,
            "epilogue_fusion": self.epilogue_fusion,
            "shape_padding": self.shape_padding,
            "force_same_precision": self.force_same_precision,
        }
        if self.cudagraphs:
            cfg["triton.cudagraphs"] = True
        return cfg


class CompileConfig(ConfigModel):
    """DiT compile: runtime ``torch.compile`` + optional AOTI export settings.

    ``strategy`` — ``DiTCompiler`` at pipeline load (``none`` | ``torch``).
    Ignored when ``paths.dit_export`` is set (regional block AOTI replaces it).

    ``aoti`` — Inductor knobs for ``kandy export`` (always read by the export CLI).
    """

    strategy: CompileStrategyName = "torch"
    aoti: AotiCompileConfig = Field(default_factory=AotiCompileConfig)


class CompileStrategy(Protocol):
    name: str

    def apply(self, dit: nn.Module) -> nn.Module: ...


class NoneCompileStrategy:
    name = "none"

    def apply(self, dit: nn.Module) -> nn.Module:
        return dit


class TorchCompileStrategy:
    """Match K5 ``compile: torch``: fused inductor + decoder/encoder max-autotune.

    K5 encoder blocks use ``magi_compile``; here they get the same torch.compile
    settings as decoder blocks (portable stand-in without magi_compiler).
    """

    name = "torch"

    def apply(self, dit: nn.Module) -> nn.Module:
        for module in dit.modules():
            if isinstance(module, FusedTransformerDecoderBlock):
                install_forward(module, torch.compile(module.forward, backend="inductor"))
            elif isinstance(module, TransformerEncoderBlock):
                install_forward(
                    module,
                    torch.compile(
                        module.forward,
                        mode="max-autotune-no-cudagraphs",
                        dynamic=True,
                    ),
                )
            elif isinstance(module, TransformerDecoderBlock):
                install_forward(
                    module,
                    torch.compile(
                        module.forward,
                        mode="max-autotune-no-cudagraphs",
                        dynamic=True,
                    ),
                )
        return dit


_STRATEGIES: dict[str, type[CompileStrategy]] = {
    "none": NoneCompileStrategy,
    "torch": TorchCompileStrategy,
}


class DiTCompiler:
    """Apply a named compile strategy to a bare DiffusionTransformer3D."""

    def __init__(self, strategy: CompileStrategyName | str = "torch"):
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown compile strategy: {strategy!r}. Available: {sorted(_STRATEGIES)}")
        self.strategy_name = strategy
        self._strategy = _STRATEGIES[strategy]()

    def apply(self, dit: nn.Module) -> nn.Module:
        logger.info("compile strategy = %s", self.strategy_name)
        return self._strategy.apply(dit)


def apply_dit_compile(
    dit: nn.Module,
    strategy: CompileStrategyName | str = "torch",
) -> nn.Module:
    return DiTCompiler(strategy).apply(dit)


_ORIG_FORWARD = "_runtime_orig_forward"


def install_forward(module: nn.Module, forward) -> None:
    """Replace ``module.forward``, keeping the first Python forward so it can be restored."""
    if not hasattr(module, _ORIG_FORWARD):
        setattr(module, _ORIG_FORWARD, module.forward)
    module.forward = forward


def restore_execution(dit: nn.Module) -> None:
    """Drop torch.compile / AOTI forwards and put the original Python forwards back."""
    for module in dit.modules():
        original = getattr(module, _ORIG_FORWARD, None)
        if original is not None:
            module.forward = original
            delattr(module, _ORIG_FORWARD)
        if hasattr(module, "_aoti_compiled"):
            delattr(module, "_aoti_compiled")
    if hasattr(dit, "_aoti_block_package"):
        delattr(dit, "_aoti_block_package")
