import importlib
import sys
from types import ModuleType
from typing import Any, Optional


LOCAL_MAGIHUMAN_PACKAGE = "musubi_tuner.magihuman"


def resolve_magihuman_module_name(module_name: str) -> str:
    if module_name == "inference":
        return LOCAL_MAGIHUMAN_PACKAGE
    if module_name.startswith("inference."):
        return f"{LOCAL_MAGIHUMAN_PACKAGE}.{module_name[len('inference.'):]}"
    if module_name.startswith(f"{LOCAL_MAGIHUMAN_PACKAGE}.") or module_name == LOCAL_MAGIHUMAN_PACKAGE:
        return module_name
    return f"{LOCAL_MAGIHUMAN_PACKAGE}.{module_name}"


def import_magihuman_module(module_name: str) -> ModuleType:
    return importlib.import_module(resolve_magihuman_module_name(module_name))


def parse_magihuman_config(config_load_path: Optional[str] = None) -> Any:
    common_module = import_magihuman_module("common")

    original_argv = None
    if config_load_path is not None:
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0], "--config-load-path", config_load_path]

    try:
        return common_module.parse_config()
    finally:
        if original_argv is not None:
            sys.argv = original_argv
