"""Round-loop driver for RL post-training: alternates Phase A (`ltx2_cache_rollouts.py`) and
Phase B (`ltx2_train_rl.py`) as subprocesses, warm-starting each round from the previous round's
LoRA and wiring the `old`-snapshot chain automatically (the snapshot-hash invariant cannot be
violated from here). Accepts the union of both phases' arguments and forwards each flag to the
phase that owns it.

  python ltx2_rl_rounds.py <phase A + phase B args...> ^
    --rl_rounds 10 ^
    --output_dir output --output_name my_lora_rl ^
    --rl_heldout_prompts heldout_prompts.txt        # optional: score every round on held-out

Outputs land in `<output_dir>/<output_name>_rounds/round_NN/` (rollout cache, `old` snapshot,
and the round's LoRA `<output_name>_rNN.safetensors`); `progress.log` in the rounds root records
one line per phase with the `ROUND_REWARD` of every Phase-A scoring pass, and the same per-round
train/held-out reward curves are written as TensorBoard scalars to `<rounds root>/tb`.
Re-running the same command resumes: rounds whose LoRA already exists are skipped. With `--rl_heldout_prompts` the
starting point (round 0) and every round's LoRA are scored on the held-out prompts with a fixed
seed and reduced group size; pick the round at the held-out peak (see the docs' Tips).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from musubi_tuner.ltx2_cache_rollouts import rollout_setup_parser
from musubi_tuner.ltx2_train_network import ltx2_setup_parser
from musubi_tuner.ltx2_train_rl import rl_setup_parser
from musubi_tuner.training.parser_common import setup_parser_common

# Flags the driver owns per round (stripped from the passthrough and re-injected with managed
# values): the warm-start chain, the cache/snapshot paths, the output naming, and the seed
# (Phase A gets seed+round so every round samples fresh rollouts).
_MANAGED = {
    "--network_weights",
    "--rl_rollout_cache",
    "--rl_save_old_lora",
    "--output_dir",
    "--output_name",
    "--seed",
}

_ROUND_REWARD_RE = re.compile(r"ROUND_REWARD (\S+) mean=([-+\d.eE]+) n=(\d+)")


def _build_phase_parsers() -> Tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """The real Phase A / Phase B parsers (same constructors the phase mains use)."""
    a = rollout_setup_parser(ltx2_setup_parser(setup_parser_common()))
    b = rl_setup_parser(ltx2_setup_parser(setup_parser_common()))
    return a, b


def _option_actions(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
    return {opt: action for action in parser._actions for opt in action.option_strings}


def split_argv(argv: List[str], parser: argparse.ArgumentParser, drop: set = frozenset()) -> List[str]:
    """Return the tokens of ``argv`` that belong to ``parser``'s options, skipping ``drop``.

    Handles ``--opt value``, ``--opt=value``, zero-arg store flags, and multi-value options
    (``nargs`` consuming values until the next ``--``-prefixed token).
    """
    table = _option_actions(parser)
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        base = tok.split("=", 1)[0]
        action = table.get(base)
        if action is None or base in drop:
            # skip the unknown/dropped option and its values
            n_values = 0
            if action is not None and "=" not in tok:
                n_values = _value_count(action, argv, i)
            i += 1 + n_values
            continue
        out.append(tok)
        if "=" not in tok:
            n_values = _value_count(action, argv, i)
            out.extend(argv[i + 1 : i + 1 + n_values])
            i += n_values
        i += 1
    return out


def _value_count(action: argparse.Action, argv: List[str], opt_index: int) -> int:
    """How many following tokens are values of ``action`` at ``argv[opt_index]``."""
    if action.nargs == 0:
        return 0
    if action.nargs in (None, 1, "?"):
        return 1 if opt_index + 1 < len(argv) and not argv[opt_index + 1].startswith("--") else 0
    # nargs '*' / '+' / int>1: consume until the next option-looking token
    n = 0
    j = opt_index + 1
    while j < len(argv) and not argv[j].startswith("--"):
        n += 1
        j += 1
    return n


def parse_round_rewards(text: str) -> List[Tuple[str, float, int]]:
    return [(m.group(1), float(m.group(2)), int(m.group(3))) for m in _ROUND_REWARD_RE.finditer(text)]


def _run(cmd: List[str], log_path: Path) -> Tuple[int, str]:
    """Run ``cmd`` streaming output to console and ``log_path``; return (rc, full output)."""
    chunks: List[str] = []
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            chunks.append(line)
        proc.wait()
    return proc.returncode, "".join(chunks)


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


class RoundsDriver:
    def __init__(self, args: argparse.Namespace, passthrough: List[str]) -> None:
        self.args = args
        here = Path(__file__).resolve().parent
        self.phase_a_script = str(here / "ltx2_cache_rollouts.py")
        self.phase_b_script = str(here / "ltx2_train_rl.py")
        parser_a, parser_b = _build_phase_parsers()
        self.tokens_a = split_argv(passthrough, parser_a, drop=_MANAGED)
        self.tokens_b = split_argv(passthrough, parser_b, drop=_MANAGED)
        # held-out scoring reuses the Phase-A command with its own prompts/group size/seed
        self.tokens_eval = split_argv(self.tokens_a, parser_a, drop=_MANAGED | {"--rl_prompts", "--rl_group_size"})
        self.mixed_precision = self._lookup(passthrough, "--mixed_precision", "bf16")
        self.root = Path(args.output_dir) / f"{args.output_name}_rounds"
        self.root.mkdir(parents=True, exist_ok=True)
        self.progress = self.root / "progress.log"
        self.base_seed = int(args.seed) if args.seed is not None else 42
        self._tb = None  # lazy TensorBoard writer for the per-round reward curves (root/tb)

    @staticmethod
    def _lookup(argv: List[str], opt: str, default: str) -> str:
        for i, tok in enumerate(argv):
            if tok == opt and i + 1 < len(argv):
                return argv[i + 1]
            if tok.startswith(opt + "="):
                return tok.split("=", 1)[1]
        return default

    def _log(self, line: str) -> None:
        stamped = f"[{_ts()}] {line}"
        print(stamped)
        with open(self.progress, "a", encoding="utf-8") as f:
            f.write(stamped + "\n")

    def _tb_log(self, kind: str, round_no: int, rewards: List[Tuple[str, float, int]]) -> None:
        """Append per-round reward scalars (``reward/<name>/<kind>`` vs round) to ``root/tb``.

        On a resumed run only freshly executed rounds are re-logged; earlier rounds live in the
        previous run's event file in the same directory, so the curve stays complete.
        """
        if self._tb is None:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(str(self.root / "tb"))
        for name, mean, _count in rewards:
            self._tb.add_scalar(f"reward/{name}/{kind}", mean, round_no)
        self._tb.flush()

    # ---- per-round paths -------------------------------------------------------------------
    def round_dir(self, n: int) -> Path:
        return self.root / f"round_{n:02d}"

    def round_lora(self, n: int) -> Path:
        return self.round_dir(n) / f"{self.args.output_name}_r{n:02d}.safetensors"

    # ---- commands ---------------------------------------------------------------------------
    def phase_a_cmd(self, n: int, warm: Optional[Path]) -> List[str]:
        rdir = self.round_dir(n)
        cmd = [sys.executable, self.phase_a_script, *self.tokens_a]
        if warm is not None:
            cmd += ["--network_weights", str(warm)]
        cmd += [
            "--rl_rollout_cache",
            str(rdir / "cache"),
            "--rl_save_old_lora",
            str(rdir / "old.safetensors"),
            "--seed",
            str(self.base_seed + n),
        ]
        return cmd

    def phase_b_cmd(self, n: int) -> List[str]:
        rdir = self.round_dir(n)
        return [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            "1",
            "--num_cpu_threads_per_process",
            "1",
            "--mixed_precision",
            self.mixed_precision,
            self.phase_b_script,
            *self.tokens_b,
            "--network_weights",
            str(rdir / "old.safetensors"),
            "--rl_rollout_cache",
            str(rdir / "cache"),
            "--output_dir",
            str(rdir),
            "--output_name",
            f"{self.args.output_name}_r{n:02d}",
            "--seed",
            str(self.base_seed),
        ]

    def eval_cmd(self, weights: Optional[Path], cache: Path) -> List[str]:
        cmd = [sys.executable, self.phase_a_script, *self.tokens_eval]
        if weights is not None:
            cmd += ["--network_weights", str(weights)]
        cmd += [
            "--rl_prompts",
            self.args.rl_heldout_prompts,
            "--rl_group_size",
            str(self.args.rl_heldout_group_size),
            "--rl_rollout_cache",
            str(cache),
            "--seed",
            str(self.args.rl_heldout_seed),
        ]
        return cmd

    # ---- stages ------------------------------------------------------------------------------
    def heldout_eval(self, tag: str, weights: Optional[Path]) -> None:
        if not self.args.rl_heldout_prompts:
            return
        marker = f"{tag} HELDOUT"
        if self.progress.exists() and marker in self.progress.read_text(encoding="utf-8"):
            return  # already scored on a previous (resumed) run
        cache = self.root / "_heldout_cache_tmp"
        rc, out = _run(self.eval_cmd(weights, cache), self.root / f"heldout_{tag.replace(' ', '_')}.log")
        shutil.rmtree(cache, ignore_errors=True)
        if rc != 0:
            raise SystemExit(f"held-out scoring failed for {tag} (rc={rc}); training artifacts are intact")
        rewards = parse_round_rewards(out)
        for name, mean, count in rewards or [("?", float("nan"), 0)]:
            self._log(f"{marker} ROUND_REWARD {name} mean={mean:.6f} n={count}")
        if rewards:
            self._tb_log("heldout", int(tag.split()[1]), rewards)

    def run(self) -> None:
        self._log(f"RL_ROUNDS START rounds={self.args.rl_rounds} output={self.root} heldout={self.args.rl_heldout_prompts or '-'}")
        warm: Optional[Path] = Path(self.args.network_weights) if self.args.network_weights else None
        self.heldout_eval("round 00", warm)  # baseline: the starting point before any RL
        for n in range(1, self.args.rl_rounds + 1):
            lora = self.round_lora(n)
            if lora.exists():
                self._log(f"round {n:02d} SKIP (exists: {lora.name})")
                warm = lora
                continue
            rdir = self.round_dir(n)
            rdir.mkdir(parents=True, exist_ok=True)
            rc, out = _run(self.phase_a_cmd(n, warm), rdir / "phase_a.log")
            if rc != 0:
                raise SystemExit(f"round {n}: Phase A failed (rc={rc}); see {rdir / 'phase_a.log'}")
            rewards = parse_round_rewards(out)
            for name, mean, count in rewards or [("?", float("nan"), 0)]:
                self._log(f"round {n:02d} A rc=0 ROUND_REWARD {name} mean={mean:.6f} n={count}")
            if rewards:
                self._tb_log("train", n, rewards)
            rc, _ = _run(self.phase_b_cmd(n), rdir / "phase_b.log")
            if rc != 0:
                raise SystemExit(f"round {n}: Phase B failed (rc={rc}); see {rdir / 'phase_b.log'}")
            self._log(f"round {n:02d} B rc=0 -> {lora.name}")
            if self.args.rl_delete_round_caches:
                shutil.rmtree(rdir / "cache", ignore_errors=True)
            warm = lora
            self.heldout_eval(f"round {n:02d}", lora)
        self._log("RL_ROUNDS_DONE")
        if self._tb is not None:
            self._tb.close()
        if self.args.rl_heldout_prompts:
            print("\nheld-out summary (pick the peak round; every round's LoRA is on disk):")
            for line in self.progress.read_text(encoding="utf-8").splitlines():
                if "HELDOUT" in line:
                    print(" ", line)


def build_driver_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RL round-loop driver: Phase A -> Phase B per round, warm-started.",
        add_help=True,
    )
    p.add_argument("--rl_rounds", type=int, required=True, help="number of generate->train rounds")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--network_weights", default=None, help="LoRA to refine; omit to start from the bare base model")
    p.add_argument("--seed", type=int, default=42, help="base seed; Phase A uses seed+round")
    p.add_argument(
        "--rl_heldout_prompts", default=None, help="optional held-out prompt file; scores round 0 and every round's LoRA"
    )
    p.add_argument("--rl_heldout_group_size", type=int, default=4)
    p.add_argument("--rl_heldout_seed", type=int, default=10042, help="fixed seed for all held-out scoring passes")
    p.add_argument(
        "--rl_delete_round_caches",
        action="store_true",
        help="delete each round's rollout cache after its Phase B (saves disk; disables cache reuse)",
    )
    return p


def main() -> None:
    driver_args, passthrough = build_driver_parser().parse_known_args()
    RoundsDriver(driver_args, passthrough).run()


if __name__ == "__main__":
    main()
