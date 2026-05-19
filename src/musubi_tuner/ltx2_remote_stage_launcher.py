"""Launch remote LTX-2 stage servers over SSH from a saved dashboard config.

The dashboard writes a config snapshot to disk and then starts this launcher as
an ordinary local process. The launcher reads the full project config, parses
the remote-stage host list from the training section, and starts one SSH
session per remote stage.

This is intentionally deterministic and narrow:

- it does not discover hosts,
- it does not mutate the model config,
- it does not try to synchronize multiple trainers,
- and it keeps the SSH sessions attached so stopping the local launcher kills
  the remote stage servers as well.

The remote side is assumed to be a Windows host with PowerShell and Python
available. That matches the current LAN proof-of-concept setup. Support for
other remote shells can be added later without changing the training side.
"""

from __future__ import annotations

import argparse
import base64
import logging
import shlex
import socket
import subprocess
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from musubi_tuner.gui_dashboard.command_builder import build_remote_stage_server_cmd
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.ltx2_remote_stage import LTX2RemoteStageSpec, parse_ltx2_remote_stage_specs

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_config_json", required=True, help="Path to the dashboard project JSON snapshot.")
    return parser


def _powershell_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _powershell_encoded_command(script: str) -> str:
    return base64.b64encode(script.encode("utf-16le")).decode("ascii")


def _split_args(raw: str) -> list[str]:
    if not raw:
        return []
    return shlex.split(raw, posix=False)


def _load_project_config(path: str | Path) -> ProjectConfig:
    return ProjectConfig.load(Path(path))


def _training_spec_args(config: ProjectConfig) -> SimpleNamespace:
    t = config.training
    return SimpleNamespace(
        ltx2_remote_stage_specs=t.ltx2_remote_stage_specs,
        ltx2_remote_stage_host=t.ltx2_remote_stage_host,
        ltx2_remote_stage_port=t.ltx2_remote_stage_port,
        ltx2_remote_stage_split=t.ltx2_remote_stage_split,
    )


def _remote_power_shell_script(*, remote_root: str, remote_python: str, server_cmd: list[str]) -> str:
    python_cmd = " ".join(
        _powershell_quote(part)
        for part in [remote_python, *server_cmd[1:]]
    )
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "$env:PYTHONIOENCODING = 'utf-8'",
            "$env:PYTHONUTF8 = '1'",
            f"Set-Location -LiteralPath {_powershell_quote(remote_root)}",
            f"& {python_cmd}",
        ]
    )


def _ssh_target(host: str, ssh_user: str) -> str:
    return f"{ssh_user}@{host}" if ssh_user else host


def _build_ssh_cmd(
    *,
    host: str,
    ssh_user: str,
    ssh_port: int,
    ssh_extra_args: str,
    remote_script: str,
) -> list[str]:
    cmd = ["ssh", "-p", str(ssh_port), *_split_args(ssh_extra_args), _ssh_target(host, ssh_user), "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", _powershell_encoded_command(remote_script)]
    return cmd


def _forward_stream(prefix: str, stream) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            text = line.rstrip("\r\n")
            if text:
                print(f"[{prefix}] {text}", flush=True)
    except Exception as exc:
        logger.debug("Stopped forwarding %s: %s", prefix, exc)


def _wait_for_tcp_ready(host: str, port: int, timeout: float, poll_interval: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=min(5.0, poll_interval)):
                return True
        except OSError:
            time.sleep(poll_interval)
    return False


def _launch_remote_stage(config: ProjectConfig, spec: LTX2RemoteStageSpec) -> subprocess.Popen[str]:
    launcher = config.remote_stage_launcher
    remote_config = config.model_copy(deep=True)
    remote_config.remote_stage_server.port = int(spec.port)
    remote_config.remote_stage_server.split = int(spec.start_index)
    remote_config.remote_stage_server.end = -1 if spec.end_index is None else int(spec.end_index)

    server_cmd = build_remote_stage_server_cmd(remote_config)
    remote_python = launcher.remote_python or "python"
    remote_server_cmd = [remote_python, "-u", "-m", "musubi_tuner.ltx2_remote_stage_server", *server_cmd[3:]]
    remote_script = _remote_power_shell_script(
        remote_root=launcher.remote_root,
        remote_python=remote_python,
        server_cmd=remote_server_cmd,
    )
    ssh_cmd = _build_ssh_cmd(
        host=spec.host,
        ssh_user=launcher.ssh_user,
        ssh_port=launcher.ssh_port,
        ssh_extra_args=launcher.ssh_extra_args,
        remote_script=remote_script,
    )
    logger.info("Starting remote stage %s:%s -> %s", spec.host, spec.port, " ".join(ssh_cmd[:4]))
    return subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)


def _terminate_procs(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _load_project_config(args.project_config_json)
    launcher = config.remote_stage_launcher
    logging.basicConfig(level=getattr(logging, str(launcher.log_level or "INFO").upper(), logging.INFO), format="%(message)s")

    if not config.training.ltx2_remote_stage:
        raise SystemExit("Remote stage launcher requires training.ltx2_remote_stage to be enabled.")
    if not launcher.remote_root:
        raise SystemExit("Remote stage launcher requires remote_stage_launcher.remote_root.")
    if not launcher.remote_python:
        raise SystemExit("Remote stage launcher requires remote_stage_launcher.remote_python.")
    if shutil.which("ssh") is None:
        raise SystemExit("ssh executable was not found on PATH.")

    specs = parse_ltx2_remote_stage_specs(_training_spec_args(config), num_blocks=None)
    if not specs:
        raise SystemExit("Remote stage launcher requires at least one remote stage spec.")

    if not (0 < int(launcher.ssh_port) < 65536):
        raise SystemExit("remote_stage_launcher.ssh_port must be in 1..65535")
    if launcher.ready_timeout <= 0:
        raise SystemExit("remote_stage_launcher.ready_timeout must be greater than 0")
    if launcher.ready_poll_interval <= 0:
        raise SystemExit("remote_stage_launcher.ready_poll_interval must be greater than 0")

    procs: list[tuple[LTX2RemoteStageSpec, subprocess.Popen[str]]] = []
    for spec in specs:
        proc = _launch_remote_stage(config, spec)
        procs.append((spec, proc))
        if proc.stdout is not None:
            threading.Thread(
                target=_forward_stream,
                args=(f"{spec.host}:{spec.port}", proc.stdout),
                daemon=True,
            ).start()

    try:
        for spec, proc in procs:
            if not _wait_for_tcp_ready(spec.host, spec.port, launcher.ready_timeout, launcher.ready_poll_interval):
                raise SystemExit(f"Remote stage did not become ready: {spec.host}:{spec.port}")
            if proc.poll() is not None and proc.returncode not in (0, None):
                raise SystemExit(f"SSH process exited early for {spec.host}:{spec.port} with code {proc.returncode}")
            print(f"[{spec.host}:{spec.port}] ready", flush=True)

        print(f"All {len(procs)} remote stage(s) are ready.", flush=True)

        while True:
            exited = []
            for spec, proc in procs:
                code = proc.poll()
                if code is not None:
                    exited.append((spec, code))
            if exited:
                spec, code = exited[0]
                if code != 0:
                    raise SystemExit(f"Remote stage SSH process exited for {spec.host}:{spec.port} with code {code}")
                break
            time.sleep(2.0)
        return 0
    except KeyboardInterrupt:
        raise
    finally:
        _terminate_procs([proc for _, proc in procs])


if __name__ == "__main__":
    raise SystemExit(main())
