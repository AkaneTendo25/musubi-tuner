#!/usr/bin/env python3
"""Install this repository's local Git hooks without changing core.hooksPath."""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def install(repo: Path) -> Path:
    repo = Path(git_output(repo, "rev-parse", "--show-toplevel"))
    source = repo / ".githooks" / "pre-commit"
    git_dir_text = git_output(repo, "rev-parse", "--path-format=absolute", "--git-dir")
    git_dir = Path(git_dir_text)
    destination = git_dir / "hooks" / "pre-commit"

    # Git may check the tracked shell script out with CRLF on Windows. Hooks
    # run under a POSIX shell, so always install the canonical LF form.
    source_bytes = source.read_bytes().replace(b"\r\n", b"\n")
    if destination.exists():
        if destination.read_bytes() == source_bytes:
            print(f"Git hook is already installed: {destination}")
            return destination
        raise RuntimeError(
            f"Refusing to overwrite a different pre-commit hook: {destination}\n"
            "Move or integrate that hook manually, then run this installer again."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source_bytes)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed Git hook: {destination}")
    return destination


def main() -> int:
    try:
        install(Path.cwd())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"install_git_hooks: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
