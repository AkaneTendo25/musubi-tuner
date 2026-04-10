import importlib
import sys
from types import ModuleType
from typing import Any, Optional


LOCAL_MAGIHUMAN_PACKAGE = "musubi_tuner.magihuman"
DEFAULT_MAGIHUMAN_REPO_ENV = "MAGIHUMAN_REPO"


def resolve_magihuman_module_name(module_name: str) -> str:
    if module_name == "inference":
        return LOCAL_MAGIHUMAN_PACKAGE
    if module_name.startswith("inference."):
        return f"{LOCAL_MAGIHUMAN_PACKAGE}.{module_name[len('inference.'):]}"
    if module_name.startswith(f"{LOCAL_MAGIHUMAN_PACKAGE}.") or module_name == LOCAL_MAGIHUMAN_PACKAGE:
        return module_name
    return f"{LOCAL_MAGIHUMAN_PACKAGE}.{module_name}"


def resolve_magihuman_repo(repo_root: Optional[str] = None) -> str:
    del repo_root
    return LOCAL_MAGIHUMAN_PACKAGE


def ensure_magihuman_repo_on_path(repo_root: Optional[str] = None) -> str:
    del repo_root
    return LOCAL_MAGIHUMAN_PACKAGE


def import_magihuman_module(module_name: str, repo_root: Optional[str] = None) -> ModuleType:
    del repo_root
    return importlib.import_module(resolve_magihuman_module_name(module_name))


def parse_magihuman_config(repo_root: Optional[str] = None, config_load_path: Optional[str] = None) -> Any:
    del repo_root
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
