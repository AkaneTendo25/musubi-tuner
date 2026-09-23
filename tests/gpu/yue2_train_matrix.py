"""YuE2 training matrix on real weights. Subprocess driver with a JSON report.

Each row is one ``yue2_train_network`` run through a small runner that records per-step losses and logs, step times,
the LoRA gradient norm of every optimizer step (global optimizer pre-hook, before the update), the process peak
``torch.cuda.max_memory_allocated``, for compile rows the dynamo graph counters, and the validation losses (fixed t and
noise) on the training songs before the first and after the last step: every run must lower them for its trained
branches (``learn:`` checks, a learning signal that 20 noisy random-t steps cannot give). After the runs the checks
that the selected rows allow are evaluated and written into ``<out>/report_<time>.json``.

usage (from the repository root, PYTHONPATH=src):
  python tests/gpu/yue2_train_matrix.py --out DIR [--rows reduced|full|overfit|abc|NAME,...] [--steps N] [--songs N]
      [--cache_only] [--list]

Data: the first ``--songs`` JamendoLyrics songs (``YUE2_DATA_DIR/jamendolyrics``), cached under ``<out>/cache`` with
the YuE2 cache scripts (VAE m-a-p/YuE2-Vae, MERT-v2 + Mothersuperior head ``--head``). Weights: ``YUE2_WEIGHTS_DIR``.
Rows that need a missing optional package (flash_attn, xformers) are reported as skipped.
"""

import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# fixed-t validation losses on the training songs that must decrease over a run ("learn:" checks)
LEARN_KEYS = ("val/flow", "val/ar_ce")

RUNNER = r"""
import json, os, sys, time
import torch

if os.environ.get("YUE2_MATRIX_DETERMINISTIC") == "1":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True)

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner import yue2_train_network as T

parser = T.yue2_setup_parser(setup_parser_common())
args = parser.parse_args(sys.argv[1:])
args = read_config_from_file(args, parser)
args.dit_dtype = "bfloat16"
if args.vae_dtype is None:
    args.vae_dtype = "float32"
trainer = T.YuE2NetworkTrainer()
rec = {"losses": [], "logs": [], "t": [], "grad_norm": [], "lr": []}

def pre_step(optimizer, *_):
    total = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                total += float(p.grad.detach().float().pow(2).sum())
    rec["grad_norm"].append(total ** 0.5)
    rec["lr"].append([float(g["lr"]) for g in optimizer.param_groups])

from torch.optim.optimizer import register_optimizer_step_pre_hook
register_optimizer_step_pre_hook(pre_step)

orig = trainer.process_batch
def process_batch(*a, **k):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rec["t"].append(time.perf_counter())
    loss, metrics = orig(*a, **k)
    rec["losses"].append(loss.item())
    rec["logs"].append(metrics)
    return loss, metrics
trainer.process_batch = process_batch

dump = os.environ.get("YUE2_MATRIX_DUMP_START")
if dump:
    orig_start = trainer.on_train_start
    def on_train_start(args_, accelerator, network, transformer, optimizer):
        orig_start(args_, accelerator, network, transformer, optimizer)
        net = accelerator.unwrap_model(network)
        torch.save(
            {"network": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "optimizer": optimizer.state_dict()},
            dump,
        )
    trainer.on_train_start = on_train_start

# learning signal: validation losses (fixed t, fixed noise, centre windows) on the training songs before the first and
# after the last optimizer step; single-step losses at random t are too noisy to show a trend in a few steps
rec["fixed_eval"] = []
fixed = {}
start_hook = trainer.on_train_start
def on_train_start_eval(args_, accelerator, network, transformer, optimizer):
    start_hook(args_, accelerator, network, transformer, optimizer)
    fixed["v"] = T.YuE2Validator(trainer._train_items, args_)
    rec["fixed_eval"].append(fixed["v"].run(trainer, args_, accelerator, transformer, network, trainer._optimizer))
trainer.on_train_start = on_train_start_eval
post_hook = trainer.on_post_optimizer_step
def on_post_optimizer_step_eval(args_, accelerator, network, transformer, sync_gradients, global_step):
    post_hook(args_, accelerator, network, transformer, sync_gradients, global_step)
    if sync_gradients and global_step + 1 == args_.max_train_steps:
        rec["fixed_eval"].append(fixed["v"].run(trainer, args_, accelerator, transformer, network, trainer._optimizer))
trainer.on_post_optimizer_step = on_post_optimizer_step_eval

t0 = time.perf_counter()
trainer.train(args)
rec["wall"] = time.perf_counter() - t0
if torch.cuda.is_available():
    rec["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
    rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
try:
    from torch._dynamo.utils import counters
    rec["dynamo"] = {"unique_graphs": int(counters["stats"]["unique_graphs"]), "recompiles": dict(counters["recompiles"])}
except Exception as e:
    rec["dynamo"] = {"error": str(e)}
rec["backend"] = {
    "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
    "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
    "deterministic": torch.are_deterministic_algorithms_enabled(),
}
with open(os.environ["YUE2_MATRIX_RESULT"], "w") as f:
    json.dump(rec, f)
"""


class Paths:
    def __init__(self, out: str, head: str):
        w = os.environ.get("YUE2_WEIGHTS_DIR")
        d = os.environ.get("YUE2_DATA_DIR")
        if not w or not d:
            sys.exit("YUE2_WEIGHTS_DIR and YUE2_DATA_DIR must be set")
        self.out = os.path.abspath(out)
        self.data = os.path.join(d, "jamendolyrics")
        self.comfy = os.path.join(w, "Comfy-Org__YuE2", "checkpoints", "yue2_3b_bf16.safetensors")
        self.int8 = os.path.join(w, "Comfy-Org__YuE2", "checkpoints", "yue2_3b_int8_convrot.safetensors")
        self.vae = os.path.join(w, "m-a-p__YuE2-Vae")
        self.mert = os.path.join(w, "m-a-p__MERT-v2-FullSong")
        ms = os.path.join(w, "Mothersuperior__yue2-mothersuperior-realaudio-tokenizer-v4")
        self.head = os.path.join(ms, f"tokenizer_head_{head}.safetensors")
        self.companion = os.path.join(ms, "nar_lora_joint_v9_comfyui.safetensors")
        self.cache = os.path.join(self.out, "cache")
        self.runs = os.path.join(self.out, "runs")
        self.config = os.path.join(self.out, "dataset.toml")
        self.jsonl = os.path.join(self.out, "songs.jsonl")
        self.abc_cache = os.path.join(self.out, "cache_abc")
        self.abc_config = os.path.join(self.out, "dataset_abc.toml")
        self.abc_jsonl = os.path.join(self.out, "songs_abc.jsonl")


SYNTHETIC_ABC = (
    "%%yue2_abc_mode full\nX:1\nT:matrix\nM:4/4\nL:1/8\nQ:1/4=120\nK:C\n"
    '"C"CDEF GABc|"G"BAGF EDCB,|"Am"A,2C2 E2A2|"F"F4 "G"G4|\n'
    '"C"c2G2 E2C2|"F"FAcA "G"GBdB|"C"c8|]\n'
)


def tail(path: str, n: int = 40) -> str:
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-n:])


def run_logged(cmd: list[str], log: str, env=None) -> tuple[int, float]:
    t0 = time.perf_counter()
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env, cwd=REPO)
    return p.returncode, time.perf_counter() - t0


def run_cache_scripts(paths: Paths, config: str, suffix: str) -> dict:
    info = {}
    py = sys.executable
    for name, cmd in (
        (
            "latents",
            [
                py,
                "-m",
                "musubi_tuner.yue2_cache_latents",
                "--dataset_config",
                config,
                "--vae",
                paths.vae,
                "--mert_model",
                paths.mert,
                "--semantic_head",
                paths.head,
                "--num_workers",
                "2",
                "--skip_existing",
            ],
        ),
        (
            "text",
            [
                py,
                "-m",
                "musubi_tuner.yue2_cache_text_encoder_outputs",
                "--dataset_config",
                config,
                "--tokenizer",
                paths.comfy,
                "--skip_existing",
            ],
        ),
    ):
        log = os.path.join(paths.out, f"cache_{name}{suffix}.log")
        rc, dt = run_logged(cmd, log)
        info[name] = {"rc": rc, "seconds": round(dt, 1)}
        if rc:
            print(tail(log))
            sys.exit(f"caching {name}{suffix} failed")
    return info


def build_abc_cache(paths: Paths) -> dict:
    """First song with a synthetic full score (flavour from the ``%%yue2_abc_mode`` header), in its own cache."""
    rec = json.loads(open(paths.jsonl, encoding="utf-8").readline())
    rec["abc"] = SYNTHETIC_ABC
    with open(paths.abc_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    with open(paths.abc_config, "w") as f:
        f.write(
            f'[general]\nbatch_size = 1\n[[datasets]]\naudio_jsonl_file = "{paths.abc_jsonl}"\n'
            f'cache_directory = "{paths.abc_cache}"\nsegment_extraction = "full"\nmax_seconds = 400.0\n'
        )
    info = run_cache_scripts(paths, paths.abc_config, "_abc")
    from safetensors import safe_open

    te = glob.glob(os.path.join(paths.abc_cache, "*_yue2_te.safetensors"))
    if len(te) != 1:
        sys.exit(f"ABC cache: expected one text cache, found {len(te)}")
    with safe_open(te[0], "pt") as f:
        info["has_abc"] = int(f.get_tensor("yue2_has_abc_int64").item())
        info["abc_mode"] = int(f.get_tensor("yue2_abc_mode_int64").item())
        info["abc_ids"] = int(f.get_tensor("varlen_yue2_abc_int64").numel())
    print(f"[cache_abc] {json.dumps(info)}", flush=True)
    return info


def build_cache(paths: Paths, songs: int) -> dict:
    os.makedirs(paths.out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(paths.data, "JamendoLyrics.csv"), encoding="utf-8")))[:songs]
    with open(paths.jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            stem = r["Filepath"][:-4]
            lyrics = open(os.path.join(paths.data, "lyrics", stem + ".txt"), encoding="utf-8").read()
            rec = {
                "audio_path": os.path.join(paths.data, "mp3", r["Filepath"]),
                "style": f"{r['Genre'].lower()}, {r['Language'].lower()} vocals",
                "lyrics": lyrics,
                "song_id": stem,
            }
            f.write(json.dumps(rec) + "\n")
    with open(paths.config, "w") as f:
        f.write(
            f'[general]\nbatch_size = 1\n[[datasets]]\naudio_jsonl_file = "{paths.jsonl}"\ncache_directory = "{paths.cache}"\n'
            'segment_extraction = "full"\nmax_seconds = 400.0\n'
        )
    info = run_cache_scripts(paths, paths.config, "")
    from safetensors import safe_open

    frames = {}
    for path in sorted(glob.glob(os.path.join(paths.cache, "*_yue2.safetensors"))):
        with safe_open(path, "pt") as f:
            meta = f.metadata()
        frames[os.path.basename(path)] = int(meta.get("yue2_frames", 0))
    info["songs"] = frames
    print(f"[cache] {json.dumps(info)}", flush=True)
    return info


def rows(paths: Paths) -> dict[str, dict]:
    """name -> {args, groups, [expect_fail], [needs], [env], [steps]}. Common: bs 1, rank 16, lr 1e-4, seed 42, gc."""
    modes = {"nar": ["--train_branches", "nar"], "ar": ["--train_branches", "ar"], "joint": ["--train_branches", "ar,nar"]}
    precisions = {
        "bf16": [],
        "convrot": ["--convrot_int8"],
        "int8file": ["--dit", paths.int8],
        "fp8": ["--fp8_base", "--fp8_scaled"],
    }
    table: dict[str, dict] = {}
    for m, margs in modes.items():
        for p, pargs in precisions.items():
            table[f"{m}_{p}"] = {"args": margs + pargs, "groups": {"reduced", "full"}}
    for m, margs in modes.items():
        table[f"{m}_swap14"] = {"args": margs + ["--blocks_to_swap", "14"], "groups": {"full"}}
        table[f"{m}_gc_offload"] = {"args": margs + ["--gradient_checkpointing_cpu_offload"], "groups": {"full"}}
        table[f"{m}_compile"] = {"args": margs + ["--compile"], "groups": {"full"}}
        table[f"{m}_compile_swap14"] = {"args": margs + ["--compile", "--blocks_to_swap", "14"], "groups": {"full"}}
        table[f"{m}_flash"] = {"args": margs + ["--flash_attn"], "groups": {"full"}, "needs": "flash_attn"}
    table["joint_flash3"] = {"args": modes["joint"] + ["--flash3"], "groups": {"full"}, "needs": "flash_attn_interface"}
    table["joint_split_attn"] = {"args": modes["joint"] + ["--split_attn"], "groups": {"full"}}
    table["joint_split_attn_compile"] = {"args": modes["joint"] + ["--split_attn", "--compile"], "groups": {"full"}}
    table["joint_swap14_h2d"] = {"args": modes["joint"] + ["--blocks_to_swap", "14", "--block_swap_h2d_only"], "groups": {"full"}}
    table["joint_pad0"] = {"args": modes["joint"] + ["--ar_pad_multiple", "0"], "groups": {"full"}}
    table["joint_pad256"] = {"args": modes["joint"] + ["--ar_pad_multiple", "256"], "groups": {"full"}}
    table["joint_compile_pad256"] = {"args": modes["joint"] + ["--compile", "--ar_pad_multiple", "256"], "groups": {"full"}}
    table["joint_bs2_swap14"] = {
        "args": modes["joint"] + ["--blocks_to_swap", "14", "--dataset_config", paths.config.replace(".toml", "_bs2.toml")],
        "groups": {"full"},
        "expect_fail": "batch",
    }
    table["joint_bs2_swap0"] = {
        "args": modes["joint"] + ["--dataset_config", paths.config.replace(".toml", "_bs2.toml")],
        "groups": {"full"},
    }
    table["joint_companion_v9"] = {"args": modes["joint"] + ["--base_weights", paths.companion], "groups": {"full"}}
    table["nar_resume_a"] = {"args": modes["nar"] + ["--save_every_n_steps", "10", "--save_state"], "groups": {"full"}, "steps": 15}
    table["nar_resume_b"] = {"args": modes["nar"], "groups": {"full"}, "steps": 5, "resume_of": "nar_resume_a"}
    for i in (1, 2):
        table[f"nar_determ_{i}"] = {
            "args": modes["nar"],
            "groups": {"full"},
            "steps": 15,
            "env": {"YUE2_MATRIX_DETERMINISTIC": "1"},
        }
        table[f"nar_default_{i}"] = {"args": modes["nar"], "groups": {"full"}, "steps": 15}
    abc = ["--dataset_config", paths.abc_config, "--cot", "full", "--abc_dropout", "0"]
    table["joint_abc_cot_codec_abc"] = {
        "args": modes["joint"] + abc + ["--ar_ce_targets", "codec_abc"],
        "groups": {"full", "abc"},
        "steps": 5,
        "abc": True,
    }
    table["joint_abc_cot_codec"] = {
        "args": modes["joint"] + abc + ["--ar_ce_targets", "codec"],
        "groups": {"full", "abc"},
        "steps": 5,
        "abc": True,
    }
    # the NAR flow loss falls only ~14% in 150 steps at lr 1e-4 (measured); 1e-3 shows memorisation clearly
    table["nar_overfit"] = {
        "args": modes["nar"] + ["--learning_rate", "1e-3"],
        "groups": {"overfit"},
        "steps": 150,
        "one_song": True,
    }
    table["ar_overfit"] = {"args": modes["ar"], "groups": {"overfit"}, "steps": 150, "one_song": True}
    return table


def common_args(paths: Paths) -> list[str]:
    return [
        "--dataset_config",
        paths.config,
        "--dit",
        paths.comfy,
        "--vae",
        paths.vae,
        "--tokenizer",
        paths.comfy,
        "--sdpa",
        "--mixed_precision",
        "bf16",
        "--gradient_checkpointing",
        "--network_dim",
        "16",
        "--network_alpha",
        "16",
        "--learning_rate",
        "1e-4",
        "--optimizer_type",
        "AdamW",
        "--seed",
        "42",
        "--max_data_loader_n_workers",
        "0",
        "--max_grad_norm",
        "0",
    ]


def derived_configs(paths: Paths) -> None:
    text = open(paths.config).read()
    with open(paths.config.replace(".toml", "_bs2.toml"), "w") as f:
        f.write(text.replace("batch_size = 1", "batch_size = 2"))
    first = open(paths.jsonl, encoding="utf-8").readline()
    one = paths.jsonl.replace(".jsonl", "_one.jsonl")
    with open(one, "w", encoding="utf-8") as f:
        f.write(first)
    # the dataset trains on every cache in its cache_directory, so the one-song config gets a directory of its own
    one_cache = paths.cache + "_one"
    os.makedirs(one_cache, exist_ok=True)
    stem = json.loads(first)["song_id"]
    files = glob.glob(os.path.join(paths.cache, f"{glob.escape(stem)}_*yue2*.safetensors"))
    if not any(p.endswith("_yue2_te.safetensors") for p in files) or len(files) < 2:
        sys.exit(f"no cache files of {stem} in {paths.cache}")
    for path in files:
        link = os.path.join(one_cache, os.path.basename(path))
        if not os.path.exists(link):
            os.symlink(path, link)
    with open(paths.config.replace(".toml", "_one.toml"), "w") as f:
        f.write(text.replace(paths.jsonl, one).replace(f'"{paths.cache}"', f'"{one_cache}"'))


def summarize(rec: dict) -> dict:
    losses, t = rec["losses"], rec["t"]
    steps = [b - a for a, b in zip(t[2:], t[3:])]
    out = {"steps": len(losses), "wall_s": round(rec["wall"], 1), "losses": [round(x, 6) for x in losses]}
    if steps:
        out["s_per_it"] = round(sum(steps) / len(steps), 3)
    out["peak_alloc_gb"] = round(rec.get("peak_alloc_gb", 0.0), 2)
    out["peak_reserved_gb"] = round(rec.get("peak_reserved_gb", 0.0), 2)
    for key in sorted({k for logs in rec["logs"] for k in logs if k.startswith("loss/")}):
        vals = [logs[key] for logs in rec["logs"] if key in logs]
        out[key] = [round(v, 6) for v in vals]
    out["grad_norm"] = [round(g, 6) for g in rec["grad_norm"]]
    out["finite"] = all(math.isfinite(x) for x in losses) and all(math.isfinite(g) for g in rec["grad_norm"])
    out["dynamo"] = rec.get("dynamo")
    out["backend"] = rec.get("backend")
    out["fixed_eval"] = [{k: round(v, 6) for k, v in ev.items()} for ev in rec.get("fixed_eval", [])]
    return out


def run_row(paths: Paths, name: str, row: dict, steps: int, extra: tuple = ()) -> dict:
    out_dir = os.path.join(paths.runs, name)
    os.makedirs(out_dir, exist_ok=True)
    if row.get("needs") and importlib.util.find_spec(row["needs"]) is None:
        print(f"[{name}] skipped: {row['needs']} is not installed", flush=True)
        return {"skipped": f"{row['needs']} is not installed"}
    runner = os.path.join(paths.out, "runner.py")
    with open(runner, "w") as f:
        f.write(RUNNER)
    result = os.path.join(out_dir, "result.json")
    if os.path.exists(result):
        os.remove(result)
    env = dict(os.environ, YUE2_MATRIX_RESULT=result, **row.get("env", {}))
    args = list(row["args"]) + list(extra)
    if row.get("one_song"):
        args += ["--dataset_config", paths.config.replace(".toml", "_one.toml")]
    if row.get("resume_of"):
        state = sorted(glob.glob(os.path.join(paths.runs, row["resume_of"], "*-state")))
        if not state:
            return {"error": f"no saved state of {row['resume_of']}"}
        args += ["--resume", state[0]]
        env["YUE2_MATRIX_DUMP_START"] = os.path.join(out_dir, "start_state.pt")
    n = row.get("steps", steps)
    cmd = (
        [sys.executable, runner]
        + common_args(paths)
        + ["--output_dir", out_dir, "--output_name", name, "--max_train_steps", str(n)]
        + args
    )
    rc, dt = run_logged(cmd, os.path.join(out_dir, "train.log"), env=env)
    log_tail = tail(os.path.join(out_dir, "train.log"), 60)
    if row.get("expect_fail"):
        ok = rc != 0 and row["expect_fail"] in log_tail.lower()
        print(f"[{name}] expected failure {'seen' if ok else 'NOT seen'} (rc={rc}, {dt:.1f}s)", flush=True)
        last = [line for line in log_tail.splitlines() if "Error" in line][-1:] or log_tail.splitlines()[-1:]
        return {"expected_failure": ok, "rc": rc, "message": last}
    if rc or not os.path.exists(result):
        lines = "\n".join(f"[{name}] | {line}" for line in log_tail.splitlines()[-25:])
        print(f"[{name}] FAILED rc={rc} after {dt:.1f}s\n{lines}", flush=True)
        return {"error": f"rc={rc}", "log_tail": log_tail[-3000:]}
    summary = summarize(json.load(open(result)))
    brief = {k: summary[k] for k in ("steps", "s_per_it", "peak_alloc_gb", "finite") if k in summary}
    for key in ("loss/flow", "loss/ar_ce", "loss/ar_kl"):
        if key in summary:
            brief[key] = [summary[key][0], summary[key][-1]]
    if len(summary["fixed_eval"]) == 2:
        for key in LEARN_KEYS:
            if key in summary["fixed_eval"][0]:
                brief[f"fixed_{key}"] = [summary["fixed_eval"][0][key], summary["fixed_eval"][1][key]]
    print(f"[{name}] {json.dumps(brief)}", flush=True)
    return summary


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def evaluate(results: dict, paths: Paths) -> dict:
    checks: dict[str, dict] = {}

    def ok_run(name):
        r = results.get(name)
        return r if r and "losses" in r and r["losses"] else None

    for name, r in results.items():
        if r and "losses" in r:
            checks[f"finite:{name}"] = {"pass": r["finite"]}
        elif r and "expected_failure" in r:
            checks[f"expected_failure:{name}"] = {"pass": r["expected_failure"], "message": r["message"]}
        elif r and "error" in r:
            checks[f"run:{name}"] = {"pass": False, "error": r["error"]}
    for mode in ("nar", "ar", "joint"):
        base = ok_run(f"{mode}_bf16")
        if base is None:
            continue
        for p in ("convrot", "int8file", "fp8"):
            q = ok_run(f"{mode}_{p}")
            if q is not None:
                d = rel(q["losses"][0], base["losses"][0])
                checks[f"quant_step1:{mode}_{p}"] = {"pass": d < 0.03, "rel": round(d, 5)}
    pairs = [(f"{m}_{x}", f"{m}_bf16") for m in ("nar", "ar", "joint") for x in ("swap14", "gc_offload", "compile", "flash")] + [
        ("joint_swap14_h2d", "joint_bf16"),
        ("joint_pad256", "joint_pad0"),
        ("joint_split_attn", "joint_bf16"),
        ("joint_flash3", "joint_bf16"),
        ("joint_compile_pad256", "joint_pad0"),
    ]
    for a, b in pairs:
        ra, rb = ok_run(a), ok_run(b)
        if ra is None or rb is None:
            continue
        dl = rel(ra["losses"][0], rb["losses"][0])
        dg = rel(ra["grad_norm"][0], rb["grad_norm"][0]) if ra["grad_norm"] and rb["grad_norm"] else None
        checks[f"equiv_step1:{a}~{b}"] = {"pass": dl < 1e-3 and (dg is None or dg < 1e-2), "loss_rel": round(dl, 6), "grad_rel": dg}
    for name, key in (("nar_overfit", "loss/flow"), ("ar_overfit", "loss/ar_ce")):
        r = ok_run(name)
        if r is not None and key in r:
            vals = r[key]
            first, last = sum(vals[:10]) / len(vals[:10]), sum(vals[-10:]) / len(vals[-10:])
            checks[f"overfit:{name}"] = {"pass": last <= 0.8 * first, "first10": round(first, 5), "last10": round(last, 5)}
    ra, rc = ok_run("joint_abc_cot_codec_abc"), ok_run("joint_abc_cot_codec")
    if ra and rc and "loss/ar_ce" in ra and "loss/ar_ce" in rc:
        # same data and seed: only the CE targets differ, so the ABC tokens must change the step-1 CE, not the flow loss
        ce = rel(ra["loss/ar_ce"][0], rc["loss/ar_ce"][0])
        flow = rel(ra["loss/flow"][0], rc["loss/flow"][0]) if "loss/flow" in ra and "loss/flow" in rc else None
        checks["abc:codec_abc_vs_codec"] = {
            "pass": ce > 1e-4 and (flow is None or flow < 1e-3),
            "ce_rel": round(ce, 6),
            "flow_rel": flow,
        }
    for name, r in results.items():
        run = ok_run(name)
        if run is None:
            continue
        evals = run.get("fixed_eval") or []
        if len(evals) != 2:
            checks[f"learn:{name}"] = {"pass": False, "error": f"{len(evals)} fixed evaluations (need start and end)"}
            continue
        start, end = evals
        # the trained branches' fixed-t losses on the training songs must go down
        deltas = {key: round(end[key] - start[key], 6) for key in LEARN_KEYS if key in start and key in end}
        checks[f"learn:{name}"] = {"pass": bool(deltas) and all(d < 0 for d in deltas.values()), "delta": deltas}
    d1, d2 = ok_run("nar_determ_1"), ok_run("nar_determ_2")
    if d1 and d2:
        checks["determinism:deterministic"] = {"pass": d1["losses"] == d2["losses"]}
    f1, f2 = ok_run("nar_default_1"), ok_run("nar_default_2")
    if f1 and f2:
        worst = max(rel(a, b) for a, b in zip(f1["losses"], f2["losses"]))
        checks["determinism:default_kernels"] = {"pass": worst < 1e-2, "worst_rel": round(worst, 6)}
    if ok_run("nar_resume_b"):
        checks["resume"] = resume_check(paths)
    ok = all(c.get("pass") for c in checks.values())
    return {"all_pass": ok, "checks": checks}


def resume_check(paths: Paths) -> dict:
    """Network tensors and optimizer state restored by --resume equal the saved state."""
    import torch
    from safetensors.torch import load_file

    start = os.path.join(paths.runs, "nar_resume_b", "start_state.pt")
    state = sorted(glob.glob(os.path.join(paths.runs, "nar_resume_a", "*-state")))
    if not os.path.exists(start) or not state:
        return {"pass": False, "error": "missing start dump or saved state"}
    dumped = torch.load(start, map_location="cpu", weights_only=False)
    saved_net = None
    for cand in ("model.safetensors", "pytorch_model.bin"):
        path = os.path.join(state[0], cand)
        if os.path.exists(path):
            saved_net = load_file(path) if cand.endswith(".safetensors") else torch.load(path, map_location="cpu")
            break
    res = {"state_dir": os.path.basename(state[0])}
    if saved_net is None:
        return dict(res, **{"pass": False, "error": "no network weights in the state dir"})
    net = dumped["network"]
    diff_net = [k for k in net if k not in saved_net or not torch.equal(net[k], saved_net[k].to(net[k].dtype))]
    res["network_tensors"] = len(net)
    res["network_mismatch"] = diff_net[:5]
    opt_path = os.path.join(state[0], "optimizer.bin")
    opt_ok = None
    if os.path.exists(opt_path):
        saved_opt = torch.load(opt_path, map_location="cpu", weights_only=False)
        cur = dumped["optimizer"]
        opt_ok = [g["lr"] for g in saved_opt["param_groups"]] == [g["lr"] for g in cur["param_groups"]]
        for pid, st in saved_opt["state"].items():
            for k, v in st.items():
                other = cur["state"][pid][k]
                if torch.is_tensor(v) and not torch.equal(v.cpu(), other.cpu()):
                    opt_ok = False
        res["optimizer_and_lr_equal"] = opt_ok
    res["pass"] = not diff_net and opt_ok is not False
    return res


def main():
    parser = argparse.ArgumentParser(description="YuE2 training matrix on real weights")
    parser.add_argument("--out", default=os.path.join(REPO, "tests", "gpu", "out", "train_matrix"))
    parser.add_argument("--rows", default="reduced", help="reduced, full, overfit or a comma list of row names")
    parser.add_argument("--steps", type=int, default=30, help="steps per run (rows may override)")
    parser.add_argument("--songs", type=int, default=3)
    parser.add_argument("--head", default="joint_v4", help="Mothersuperior head: joint_v4, joint_v9, ...")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the row table and exit")
    parser.add_argument("--extra", default="", help='extra trainer arguments for every row (e.g. "--learning_rate 1e-3")')
    args = parser.parse_args()

    paths = Paths(args.out, args.head)
    table = rows(paths)
    if args.list:
        for name, row in table.items():
            print(name, sorted(row["groups"]), " ".join(row["args"]))
        return
    if args.rows in ("reduced", "full", "overfit", "abc"):
        names = [n for n, r in table.items() if args.rows in r["groups"]]
    else:
        names = [n.strip() for n in args.rows.split(",") if n.strip()]
        unknown = [n for n in names if n not in table]
        if unknown:
            sys.exit(f"unknown rows {unknown}; use --list")
    cache = build_cache(paths, args.songs)
    derived_configs(paths)
    if any(table[n].get("abc") for n in names):
        cache["abc"] = build_abc_cache(paths)
        if (cache["abc"]["has_abc"], cache["abc"]["abc_mode"]) != (1, 2):
            sys.exit(f"ABC cache has no full score: {cache['abc']}")
    if args.cache_only:
        return
    os.makedirs(paths.runs, exist_ok=True)
    extra = tuple(args.extra.split())
    results = {name: run_row(paths, name, table[name], args.steps, extra) for name in names}
    report = {"rows": names, "steps": args.steps, "cache": cache, "results": results, "evaluation": evaluate(results, paths)}
    try:
        import torch

        report["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    except Exception:
        pass
    path = os.path.join(paths.out, f"report_{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    ev = report["evaluation"]
    for key, c in ev["checks"].items():
        print(f"[check] {key}: {'PASS' if c.get('pass') else 'FAIL'} {json.dumps({k: v for k, v in c.items() if k != 'pass'})}")
    print(f"[matrix] all_pass={ev['all_pass']} report={path}", flush=True)
    sys.exit(0 if ev["all_pass"] else 1)


if __name__ == "__main__":
    main()
