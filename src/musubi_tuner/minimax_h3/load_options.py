from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum


class H3Quantization(str, Enum):
    NATIVE = "native"
    FP8 = "fp8"
    FP8_SCALED = "fp8_scaled"
    INT8 = "int8"


@dataclass(frozen=True)
class H3LoadOptions:
    """Musubi-owned loading options passed to the MiniMax adapter.

    The vendored source stays unmodified. The adapter decides which options the
    model supports and must reject unsupported modes.
    """

    dtype: str = "float16"
    fp8: bool = False
    fp8_scaled: bool = False
    int8: bool = False
    fp8_text_encoder: bool = False
    allow_prequantized_fp8: bool = False
    blocks_to_swap: int = 0
    attention_mode: str | None = None
    split_attention: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"unsupported H3 compute dtype: {self.dtype}")
        if self.int8 and (self.fp8 or self.fp8_scaled):
            raise ValueError("--int8 cannot be combined with --fp8 or --fp8_scaled")
        if self.allow_prequantized_fp8 and not (self.fp8 or self.fp8_scaled):
            raise ValueError("--allow_prequantized_fp8 requires --fp8 or --fp8_scaled")
        if self.blocks_to_swap < 0:
            raise ValueError("--blocks_to_swap must be non-negative")
        if self.attention_mode not in {None, "torch", "flash", "flash3", "xformers"}:
            raise ValueError(f"unsupported H3 attention mode: {self.attention_mode}")

    @property
    def quantization(self) -> H3Quantization:
        if self.int8:
            return H3Quantization.INT8
        if self.fp8_scaled:
            return H3Quantization.FP8_SCALED
        if self.fp8:
            return H3Quantization.FP8
        return H3Quantization.NATIVE

    @classmethod
    def from_namespace(cls, args: argparse.Namespace, *, dtype: str) -> H3LoadOptions:
        return cls(
            dtype=dtype,
            fp8=bool(getattr(args, "fp8", False) or getattr(args, "fp8_base", False)),
            fp8_scaled=bool(getattr(args, "fp8_scaled", False)),
            int8=bool(getattr(args, "int8", False)),
            fp8_text_encoder=bool(getattr(args, "fp8_text_encoder", False)),
            allow_prequantized_fp8=bool(getattr(args, "allow_prequantized_fp8", False)),
            blocks_to_swap=int(getattr(args, "blocks_to_swap", 0)),
            attention_mode=getattr(args, "attention_mode", None),
            split_attention=bool(getattr(args, "split_attn", False)),
        )


def add_h3_load_arguments(parser: argparse.ArgumentParser, *, include_text_encoder: bool = True) -> None:
    group = parser.add_argument_group("MiniMax H3 model loading")
    group.add_argument("--fp8", action="store_true", help="load the H3 transformer with unscaled FP8 weights")
    group.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="quantize supported H3 transformer Linear weights with Musubi's scaled FP8 path",
    )
    group.add_argument(
        "--int8",
        action="store_true",
        help="use the vendored H3 INT8 path when supported",
    )
    if include_text_encoder:
        group.add_argument(
            "--fp8_text_encoder",
            action="store_true",
            help="load the H3 text/understanding encoder in FP8 when supported",
        )
    group.add_argument(
        "--allow_prequantized_fp8",
        action="store_true",
        help="accept preconverted FP8 weights and their scale tensors instead of requantizing them",
    )
    group.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="number of H3 transformer blocks to swap to CPU",
    )
