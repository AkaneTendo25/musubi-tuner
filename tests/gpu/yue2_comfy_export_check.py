"""Check a YuE2 LoRA exported for ComfyUI against the ComfyUI checkpoint.

Two levels:
1. header check (no ComfyUI needed): every exported module maps onto a tensor of the ComfyUI all-in-one checkpoint
   (``diffusion_model.*`` -> ``model.diffusion_model.*``, ``text_encoders.*`` unchanged) with matching shapes;
   module counts per branch (AR via the CLIP loader, NAR and I/O via the model loader).
2. with ``--comfy_root`` (a ComfyUI checkout that has YuE2 support): load the checkpoint with
   ``comfy.sd.load_checkpoint_guess_config``, apply the LoRA with ``comfy.sd.load_lora_for_models``, count the
   patched keys and the "lora key not loaded" / "NOT LOADED" warnings, and compare every patched weight (ComfyUI's
   own ``comfy.lora.calculate_weight`` in fp32) with ``base + delta`` of the musubi LoRA.

usage:
  python tests/gpu/yue2_comfy_export_check.py --checkpoint yue2_3b_bf16.safetensors [--lora my.safetensors]
      [--comfy_root /path/to/ComfyUI] [--out report.json]
Without ``--lora`` a random joint (AR + NAR + full I/O) LoRA is built on a meta model and exported.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.yue2.yue2_lora_formats import load_lora_file, native_to_comfy, to_native  # noqa: E402

LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".alpha", ".diff_b", ".diff")


def checkpoint_shapes(path: str) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def random_joint_lora(train_io: str = "full", rank: int = 4, seed: int = 0) -> tuple[dict, dict]:
    """A joint LoRA with random (non-zero) weights built on a meta YuE2-3B model: ``(state_dict, metadata)``."""
    from musubi_tuner.networks import lora_yue2
    from musubi_tuner.yue2.yue2_model import YuE2Config, YuE2Model

    with torch.device("meta"):
        model = YuE2Model(YuE2Config())
    network = lora_yue2.create_arch_network(1.0, rank, rank, None, [], model, branches="ar,nar", train_io=train_io)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)  # registers the modules
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in network.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.01)
    return {k: v.detach().clone() for k, v in network.state_dict().items()}, {}


def _module_of(key: str) -> Optional[tuple[str, str]]:
    for suffix in LORA_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    return None


def _target(name: str) -> str:
    return "model." + name if name.startswith("diffusion_model.") else name


def comfy_key_report(comfy_sd: dict, shapes: dict[str, tuple[int, ...]]) -> dict:
    """Map every exported key onto the checkpoint; shape-check lora up/down and diffs."""
    modules: dict[str, dict[str, torch.Tensor]] = {}
    stray = []
    for key, value in comfy_sd.items():
        parsed = _module_of(key)
        if parsed is None:
            stray.append(key)
            continue
        modules.setdefault(parsed[0], {})[parsed[1]] = value
    unmapped, shape_errors = [], []
    counts = {"ar": 0, "nar": 0, "io": 0}
    for name, parts in sorted(modules.items()):
        target = _target(name)
        weight = shapes.get(target + ".weight")
        if weight is None:
            unmapped.append(name)
            continue
        if name.startswith("text_encoders."):
            counts["ar"] += 1
        elif ".layers." in name:
            counts["nar"] += 1
        else:
            counts["io"] += 1
        if ".diff" in parts and tuple(parts[".diff"].shape) != weight:
            shape_errors.append((name, "diff", tuple(parts[".diff"].shape), weight))
        if ".diff_b" in parts and tuple(parts[".diff_b"].shape) != shapes.get(target + ".bias"):
            shape_errors.append((name, "diff_b", tuple(parts[".diff_b"].shape), shapes.get(target + ".bias")))
        if ".lora_up.weight" in parts:
            up, down = parts[".lora_up.weight"], parts[".lora_down.weight"]
            if (up.shape[0], down.shape[1]) != weight[:2] or up.shape[1] != down.shape[0]:
                shape_errors.append((name, "lora", (tuple(up.shape), tuple(down.shape)), weight))
    return {"modules": counts, "unmapped": unmapped, "shape_errors": shape_errors, "stray_keys": stray, "keys": len(comfy_sd)}


class _WarningCounter(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.not_loaded: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("lora key not loaded") or msg.startswith("NOT LOADED"):
            self.not_loaded.append(msg)


def expected_patches(native) -> dict[str, tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """``{comfy module name: (weight delta, bias delta or None)}`` of a canonical LoRA."""
    from musubi_tuner.yue2.yue2_lora_formats import _comfy_module_name, weight_deltas

    return {_comfy_module_name(path): deltas for path, deltas in weight_deltas(native).items()}


def comfy_load_report(comfy_root: str, checkpoint: str, comfy_sd: dict, native) -> dict:
    """Load the checkpoint and the LoRA through ComfyUI itself; count patches and unloaded keys, and compare every
    patched weight (``comfy.lora.calculate_weight`` on an fp32 copy of the base) with ``base + delta``."""
    sys.path.insert(0, comfy_root)
    import comfy.lora  # noqa: E402  (ComfyUI checkout)
    import comfy.sd  # noqa: E402
    import comfy.utils  # noqa: E402

    counter = _WarningCounter()
    logging.getLogger().addHandler(counter)
    try:
        model, clip, _vae, _ = comfy.sd.load_checkpoint_guess_config(checkpoint, output_vae=False, output_clip=True)
        new_model, new_clip = comfy.sd.load_lora_for_models(model, clip, comfy_sd, 1.0, 1.0)
    finally:
        logging.getLogger().removeHandler(counter)
    key_map = comfy.lora.model_lora_keys_unet(new_model.model, {})
    key_map = comfy.lora.model_lora_keys_clip(new_clip.cond_stage_model, key_map)

    missing, worst = [], 0.0
    expected = expected_patches(native)
    for name, (dw, db) in expected.items():
        target = key_map.get(name)
        if not isinstance(target, str):
            missing.append((name, "no key map entry"))
            continue
        for key, delta in ((target, dw), (target[: -len(".weight")] + ".bias", db)):
            if delta is None:
                continue
            patcher = new_model if key in new_model.patches else new_clip.patcher
            if key not in patcher.patches:
                missing.append((name, key))
                continue
            base = comfy.utils.get_attr(patcher.model, key).detach().float().cpu()
            patched = comfy.lora.calculate_weight(patcher.patches[key], base.clone(), key)
            err = ((patched - base - delta.float()).norm() / delta.float().norm().clamp_min(1e-30)).item()
            worst = max(worst, err)
    n_bias = sum(1 for _, db in expected.values() if db is not None)
    return {
        "model_patches": len(new_model.patches),
        "clip_patches": len(new_clip.patcher.patches),
        "expected_patches": len(expected) + n_bias,
        "not_loaded": counter.not_loaded,
        "missing_patches": missing,
        "worst_delta_rel": worst,
    }


def report_ok(report: dict, tol: float = 1e-4) -> bool:
    header = report["header"]
    ok = not header["unmapped"] and not header["shape_errors"] and not header["stray_keys"]
    comfy = report.get("comfy")
    if isinstance(comfy, dict):
        ok = ok and not comfy["not_loaded"] and not comfy["missing_patches"] and comfy["worst_delta_rel"] < tol
        ok = ok and comfy["model_patches"] + comfy["clip_patches"] == comfy["expected_patches"]
    return ok


def main():
    parser = argparse.ArgumentParser(description="Check a YuE2 ComfyUI LoRA export against the ComfyUI checkpoint")
    parser.add_argument("--checkpoint", required=True, help="ComfyUI yue2_3b_bf16.safetensors")
    parser.add_argument("--lora", default=None, help="LoRA in any YuE2 format (default: a random joint LoRA)")
    parser.add_argument("--base_model", default=None, help="base checkpoint for Mothersuperior full I/O weights")
    parser.add_argument("--comfy_root", default=None, help="ComfyUI checkout for the real load check")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args()

    if args.lora:
        sd, metadata = load_lora_file(args.lora)
        base_io = None
        if args.base_model:
            from musubi_tuner.yue2.yue2_checkpoint import read_base_io

            base_io = read_base_io(args.base_model)
        native = to_native(sd, metadata, base_io=base_io)
    else:
        sd, metadata = random_joint_lora()
        native = to_native(sd, metadata)
    comfy_sd, comfy_meta = native_to_comfy(native)
    report = {"lora": args.lora or "random joint", "header": comfy_key_report(comfy_sd, checkpoint_shapes(args.checkpoint))}
    if args.comfy_root:
        report["comfy"] = comfy_load_report(args.comfy_root, args.checkpoint, comfy_sd, native)
    else:
        report["comfy"] = "skipped: pass --comfy_root to load through ComfyUI"
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
    sys.exit(0 if report_ok(report) else 1)


if __name__ == "__main__":
    main()
