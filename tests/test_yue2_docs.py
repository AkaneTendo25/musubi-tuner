"""Docs and code comments name only YuE2 scripts that exist in the tree."""

import glob
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_REF = re.compile(r"\byue2_[a-z0-9_]+\.py\b")


def _existing_scripts() -> set[str]:
    names = set()
    for path in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True):
        if f"{os.sep}.git{os.sep}" not in path:
            names.add(os.path.basename(path))
    return names


def _missing_refs(text: str, existing: set[str]) -> set[str]:
    return {name for name in SCRIPT_REF.findall(text) if name not in existing}


def _scanned_files() -> list[str]:
    files = [os.path.join(REPO, "README.md")]
    files += glob.glob(os.path.join(REPO, "docs", "*.md"))
    files += glob.glob(os.path.join(REPO, "src", "musubi_tuner", "**", "*.py"), recursive=True)
    files += glob.glob(os.path.join(REPO, "tests", "gpu", "*.py"))
    files += glob.glob(os.path.join(REPO, "yue2_*.py"))
    return files


def test_missing_refs_detects_unknown_script():
    existing = {"yue2_train_network.py"}
    text = "run `yue2_train_network.py`, headers written by `yue2_transcribe_abc.py`"
    assert _missing_refs(text, existing) == {"yue2_transcribe_abc.py"}


def test_referenced_yue2_scripts_exist():
    existing = _existing_scripts()
    assert "yue2_train_network.py" in existing
    missing = {}
    for path in _scanned_files():
        with open(path, encoding="utf-8") as f:
            refs = _missing_refs(f.read(), existing)
        if refs:
            missing[os.path.relpath(path, REPO)] = sorted(refs)
    assert not missing, f"docs/comments name YuE2 scripts that do not exist: {missing}"
