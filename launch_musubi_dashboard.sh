#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${VENV_PY:-$REPO_ROOT/.venv/bin/python}"
PROJECT_FILE="${PROJECT_FILE:-$REPO_ROOT/projects/runpod/project.json}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

if [[ ! -x "$VENV_PY" ]]; then
  printf '[launch_musubi_dashboard] ERROR: missing local python: %s\n' "$VENV_PY" >&2
  printf '[launch_musubi_dashboard] Run: bash ./setup_runpod.sh\n' >&2
  exit 1
fi

cmd=(
  "$VENV_PY"
  -m
  musubi_tuner.gui_dashboard
  --host
  "$HOST"
  --port
  "$PORT"
)

if [[ -f "$PROJECT_FILE" ]]; then
  cmd+=(--project "$PROJECT_FILE")
fi

printf '[launch_musubi_dashboard] Starting dashboard on %s:%s\n' "$HOST" "$PORT"
exec "${cmd[@]}"
