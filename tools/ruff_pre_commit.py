#!/usr/bin/env python3
"""Run Ruff against an exact materialization of the Git index."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MIN_RUFF = (0, 12, 10)
MAX_RUFF = (0, 16, 0)


def run_output(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def ruff_candidates(repo: Path) -> list[list[str]]:
    candidates: list[list[str]] = []
    for path in (repo / ".venv" / "Scripts" / "ruff.exe", repo / ".venv" / "bin" / "ruff"):
        if path.is_file():
            candidates.append([str(path)])

    path_ruff = shutil.which("ruff")
    if path_ruff and [path_ruff] not in candidates:
        candidates.append([path_ruff])

    uv = shutil.which("uv")
    if uv:
        # Resolve uv's already-installed Ruff without allowing a network sync.
        # The absolute executable remains valid when checks run in the snapshot.
        resolved = subprocess.run(
            [
                uv,
                "run",
                "--offline",
                "--no-sync",
                "python",
                "-c",
                "import shutil; print(shutil.which('ruff') or '')",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        uv_ruff = resolved.stdout.strip() if resolved.returncode == 0 else ""
        if uv_ruff and [uv_ruff] not in candidates:
            candidates.append([uv_ruff])
    return candidates


def version_of(command: list[str], repo: Path) -> tuple[int, int, int] | None:
    try:
        result = subprocess.run([*command, "--version"], cwd=repo, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"ruff (\d+)\.(\d+)\.(\d+)", result.stdout)
    return tuple(map(int, match.groups())) if match else None


def find_ruff(repo: Path) -> list[str]:
    rejected: list[str] = []
    for command in ruff_candidates(repo):
        version = version_of(command, repo)
        if version is not None and MIN_RUFF <= version < MAX_RUFF:
            return command
        if version is not None:
            rejected.append(f"{' '.join(command)} ({'.'.join(map(str, version))})")

    detail = f" Found incompatible versions: {', '.join(rejected)}." if rejected else ""
    raise RuntimeError(
        "Ruff >=0.12.10,<0.16 is required but was not found. "
        "Install the development dependencies with `uv sync` or `pip install --group dev`." + detail
    )


def check_index(repo: Path, ruff: list[str]) -> int:
    # checkout-index's --prefix is most portable as a path relative to the
    # repository (an absolute Windows drive path can be treated as a literal
    # relative prefix by Git for Windows).
    with tempfile.TemporaryDirectory(prefix=".ruff-pre-commit-", dir=repo) as temporary:
        snapshot = Path(temporary)
        if snapshot.resolve().parent != repo.resolve():
            raise RuntimeError(f"temporary snapshot escaped the repository: {snapshot}")
        prefix = snapshot.name + "/"
        subprocess.run(
            ["git", "checkout-index", "--all", f"--prefix={prefix}"],
            cwd=repo,
            check=True,
        )

        checks = ([*ruff, "check", "."], [*ruff, "format", "--check", "."])
        for command in checks:
            result = subprocess.run(command, cwd=snapshot)
            if result.returncode:
                print(
                    "pre-commit: Ruff rejected the staged snapshot. Fix and stage the reported files, then commit again.",
                    file=sys.stderr,
                )
                return result.returncode
    return 0


def main() -> int:
    try:
        repo = Path(run_output(Path.cwd(), "rev-parse", "--show-toplevel"))
        return check_index(repo, find_ruff(repo))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"pre-commit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
