"""Read GUI-visible defaults from the canonical LTX-2 training parser."""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Any

from musubi_tuner.hv_train_network import setup_parser_common
from musubi_tuner.ltx2_args import ltx2_setup_parser


@lru_cache(maxsize=1)
def get_ltx2_training_arg_defaults() -> dict[str, Any]:
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)

    defaults: dict[str, Any] = {}
    for action in parser._actions:
        if not getattr(action, "dest", None) or action.dest == "help":
            continue
        if action.default is argparse.SUPPRESS:
            continue
        defaults[action.dest] = action.default

    return defaults


def get_ltx2_training_arg_default(name: str, fallback: Any = None) -> Any:
    return get_ltx2_training_arg_defaults().get(name, fallback)


def get_ltx2_training_output_dir_default() -> str:
    value = get_ltx2_training_arg_default("output_dir", "")
    return value if isinstance(value, str) else str(value)


def get_ltx2_training_network_module_default() -> str:
    value = get_ltx2_training_arg_default("network_module", "")
    return value if isinstance(value, str) else str(value)
