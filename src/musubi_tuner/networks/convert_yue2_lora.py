"""Convert YuE2 LoRA files between musubi native, ComfyUI, yue2-lora-v1 (Starnodes / Studio) and fl-yue2-lora-v1.

Inputs may be any known format (native, ComfyUI / ai-toolkit, yue2-lora-v1, fl-yue2-lora-v1, Mothersuperior
safetensors or ``.pt``). ``--concat`` rank-concatenates companion adapters (e.g. a folded ``--base_weights``) into the
output. Mothersuperior full I/O weights need ``--base_model`` (the base checkpoint) to become diffs.
"""

import argparse
import logging
import os
from typing import Optional

import torch

from musubi_tuner.yue2 import yue2_lora_formats as formats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FROM_CHOICES = {
    "auto": None,
    "native": "native",
    "comfy": "comfy",
    "aitk": "aitk",
    "hf": "hf",
    "fl": "fl",
    "ms": "ms",
}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert YuE2 LoRA formats")
    parser.add_argument("--input", required=True, help="input LoRA (.safetensors or Mothersuperior .pt)")
    parser.add_argument("--output", required=True, help="output .safetensors (fl writes {stem}-ar / {stem}-nar files)")
    parser.add_argument("--to", choices=["native", "comfy", "hf", "fl"], required=True, help="output format")
    parser.add_argument("--from", dest="src_format", choices=list(FROM_CHOICES), default="auto", help="input format")
    parser.add_argument("--branch", choices=["both", "ar", "nar"], default="both", help="modules to keep (I/O counts as nar)")
    parser.add_argument("--concat", nargs="*", default=None, help="companion adapters, rank-concatenated into the output")
    parser.add_argument("--base_model", default=None, help="base checkpoint for Mothersuperior full I/O weights")
    parser.add_argument("--dtype", choices=list(DTYPES), default="fp32", help="output dtype")
    parser.add_argument("--verify", action="store_true", help="reload the output and compare every target delta")
    return parser


def _fmt_hint(choice: str, path: str) -> Optional[str]:
    fmt = FROM_CHOICES[choice]
    if fmt == "ms":
        return "ms_safetensors" if path.lower().endswith(".safetensors") else "ms_pt"
    return fmt


def load_native(path: str, src_format: str = "auto", base_io=None) -> formats.NativeLoRA:
    sd, metadata = formats.load_lora_file(path)
    native = formats.to_native(sd, metadata, layout_hint=_fmt_hint(src_format, path), base_io=base_io)
    full = [p for p, md in native.modules.items() if md.kind == "full"]
    if full:
        raise ValueError(f"{path}: {full} hold full replacement weights; pass --base_model to convert them to diffs")
    return native


def _output_paths(output: str, fmt: str, branches) -> dict:
    if fmt != "fl":
        return {None: output}
    stem, ext = os.path.splitext(output)
    return {b: f"{stem}-{b}{ext or '.safetensors'}" for b in branches}


def convert(args) -> list[str]:
    base_io = None
    if args.base_model:
        from musubi_tuner.yue2.yue2_checkpoint import read_base_io

        base_io = read_base_io(args.base_model)

    native = load_native(args.input, args.src_format, base_io)
    for companion in args.concat or []:
        logger.info(f"concatenating {companion}")
        native = formats.concat_rank(native, load_native(companion, "auto", base_io))
    native = formats.filter_branch(native, args.branch)
    if not native.modules:
        raise ValueError(f"no modules left for --branch {args.branch}")

    dtype = DTYPES[args.dtype]
    written = []
    if args.to == "native":
        sd = formats.native_state_dict(native, dtype=dtype)
        meta = {k: v for k, v in native.metadata.items() if isinstance(v, str)}
        meta["yue2_converted_from"] = native.source_format
        formats.save_lora_file(args.output, sd, meta)
        written.append(args.output)
    elif args.to == "comfy":
        sd, meta = formats.native_to_comfy(native, dtype=dtype)
        formats.save_lora_file(args.output, sd, meta)
        written.append(args.output)
    elif args.to == "hf":
        branches = {("ar" if formats.module_branch(p) == "ar" else "nar") for p in native.modules}
        if len(branches) > 1:
            raise ValueError("yue2-lora-v1 holds one branch; pass --branch nar or --branch ar")
        sd, meta = formats.native_to_hf(native, branch=branches.pop(), dtype=dtype)
        formats.save_lora_file(args.output, sd, meta)
        written.append(args.output)
    else:
        per_branch = formats.native_to_fl(native, dtype=dtype)
        paths = _output_paths(args.output, "fl", per_branch)
        for b, (sd, meta) in per_branch.items():
            formats.save_lora_file(paths[b], sd, meta)
            written.append(paths[b])

    for path in written:
        logger.info(f"wrote {path}")
    if args.verify:
        verify(native, written, dtype)
    return written


def verify(native: formats.NativeLoRA, written: list[str], dtype: torch.dtype) -> float:
    """Reload the written file(s) and compare every target delta with the source; raises on a mismatch."""
    expected = formats.weight_deltas(native)
    got = {}
    for path in written:
        sd, meta = formats.load_lora_file(path)
        got.update(formats.weight_deltas(formats.to_native(sd, meta)))
    if set(got) != set(expected):
        missing, extra = sorted(set(expected) - set(got))[:8], sorted(set(got) - set(expected))[:8]
        raise ValueError(f"verify: module sets differ: missing {missing}, extra {extra}")
    worst = 0.0
    for path, (w, b) in expected.items():
        gw, gb = got[path]
        err = (gw - w).abs().max().item()
        tol = 1e-5 if dtype == torch.float32 else 1e-2 * max(w.abs().max().item(), 1e-6)
        if b is not None or gb is not None:
            zb = torch.zeros(w.shape[0])
            err_b = ((gb if gb is not None else zb) - (b if b is not None else zb)).abs().max().item()
            err = max(err, err_b)
        worst = max(worst, err)
        if err > tol:
            raise ValueError(f"verify: {path} max abs diff {err:.3e} > {tol:.1e}")
    logger.info(f"verify: {len(expected)} modules match (max abs diff {worst:.3e})")
    return worst


def main(argv=None):
    args = setup_parser().parse_args(argv)
    convert(args)


if __name__ == "__main__":
    main()
