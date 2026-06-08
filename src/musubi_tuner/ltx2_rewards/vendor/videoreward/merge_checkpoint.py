"""One-time, OFFLINE merge of the released VideoReward (VideoAlign) LoRA checkpoint.

The released ``checkpoint-*/model.pth`` is an UNMERGED PEFT/LoRA state dict (keys
``base_model.model.*`` with ``base_layer`` + ``lora_A`` / ``lora_B``). At inference time
this forced a runtime ``peft`` dependency: the model had to be rebuilt with
``get_peft_model`` so the LoRA tensors had somewhere to load.

This helper performs the LoRA merge ONCE, ahead of time, so the runtime/inference path
(:mod:`.inferencer`) can load a plain merged checkpoint with NO ``peft`` import. It:

  1. Builds the PEFT-wrapped model + loads the unmerged LoRA exactly as the runtime used
     to (reusing :func:`.inferencer._build_model_and_processor` /
     :func:`.inferencer._load_model_from_checkpoint`).
  2. Calls ``peft`` ``model.merge_and_unload()`` -- a mathematically exact fold of
     ``W += (alpha / r) * B @ A`` into each base ``Linear`` -- and unwraps back to the
     plain :class:`.model.Qwen2VLRewardModelBT` (the value head ``rm_head`` is carried
     through untouched, it was excluded from LoRA).
  3. Saves the merged plain ``state_dict()`` (NEW transformers layout, no ``base_model.``
     prefix, no ``lora_`` keys) to ``<dst>/model.pth`` and copies the sibling
     ``model_config.json`` next to the merged ``checkpoint-*`` dir with
     ``peft_lora_config.lora_enable = false`` so the runtime knows the checkpoint is
     already merged.

``peft`` is imported ONLY here (via the reused inferencer builder); it is NOT on the
inference path. Run this once per released checkpoint, e.g.::

    python -m musubi_tuner.ltx2_rewards.vendor.videoreward.merge_checkpoint \
        --src /.../VideoReward \
        --dst /.../VideoReward-merged

Because the merge is exact, scores from the merged checkpoint are bit-identical to the
old PEFT-rebuilt path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

from .inferencer import (
    _build_model_and_processor,
    _load_configs_from_json,
    _load_model_from_checkpoint,
    _resolve_checkpoint_path,
)


def merge_checkpoint(src_dir, dst_dir, checkpoint_step=-1, dtype=torch.bfloat16):
    """Merge the LoRA in the released ``src_dir`` checkpoint into a plain ``dst_dir`` one.

    ``src_dir`` is the dir holding ``model_config.json`` + ``checkpoint-*`` subdirs (or a
    ``checkpoint-*`` dir directly, with ``model_config.json`` one level up); ``dst_dir`` is
    the merged parent dir to create (it will get ``model_config.json`` +
    ``checkpoint-<step>/model.pth``). Returns the path of the saved merged ``model.pth``.
    """
    config_path = os.path.join(src_dir, "model_config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.normpath(src_dir)), "model_config.json")

    _data_config, model_config, peft_lora_config, _inference_config = _load_configs_from_json(config_path)
    if not peft_lora_config.get("lora_enable", False):
        raise ValueError(
            f"merge_checkpoint: {config_path} has peft_lora_config.lora_enable=false; "
            "this checkpoint looks already merged, nothing to do."
        )

    # Build the PEFT-wrapped model and load the unmerged LoRA exactly as the old runtime did.
    model, _processor = _build_model_and_processor(model_config, peft_lora_config, dtype=dtype, wrap_lora=True)
    model, checkpoint_step = _load_model_from_checkpoint(model, src_dir, checkpoint_step, allow_unmerged=True)
    model.eval()

    # Exact LoRA fold: W += (alpha / r) * B @ A, then unwrap to the plain base model.
    merged = model.merge_and_unload()

    # Sanity: nothing PEFT/LoRA must survive into the merged state dict.
    merged_state_dict = merged.state_dict()
    leftover = [k for k in merged_state_dict if ("lora_" in k) or k.startswith("base_model.model.") or ("base_layer" in k)]
    if leftover:
        raise RuntimeError(
            f"merge_checkpoint: {len(leftover)} PEFT/LoRA keys survived merge_and_unload (e.g. {leftover[:3]}); "
            "refusing to save a non-merged checkpoint."
        )
    if "rm_head.weight" not in merged_state_dict:
        raise RuntimeError("merge_checkpoint: rm_head.weight missing from merged state dict; value head was lost.")

    # Write <dst>/checkpoint-<step>/model.pth + a lora_enable=false model_config.json.
    dst_ckpt_dir = os.path.join(dst_dir, f"checkpoint-{checkpoint_step}")
    os.makedirs(dst_ckpt_dir, exist_ok=True)
    merged_ckpt = os.path.join(dst_ckpt_dir, "model.pth")
    torch.save(merged_state_dict, merged_ckpt)

    with open(config_path, "r") as f:
        full_config = json.load(f)
    full_config.setdefault("peft_lora_config", {})["lora_enable"] = False
    with open(os.path.join(dst_dir, "model_config.json"), "w") as f:
        json.dump(full_config, f, indent=4)

    # Carry the tokenizer dir along if the source checkpoint shipped one.
    src_ckpt_dir = _resolve_checkpoint_path(src_dir, checkpoint_step)
    src_tok = os.path.join(src_ckpt_dir, "tokenizer")
    if os.path.isdir(src_tok):
        shutil.copytree(src_tok, os.path.join(dst_ckpt_dir, "tokenizer"), dirs_exist_ok=True)

    return merged_ckpt


def main():
    parser = argparse.ArgumentParser(description="One-time offline LoRA merge for the VideoReward checkpoint.")
    parser.add_argument("--src", required=True, help="Source VideoReward dir (model_config.json + checkpoint-* subdirs).")
    parser.add_argument("--dst", required=True, help="Destination dir for the merged, peft-free checkpoint.")
    parser.add_argument("--checkpoint_step", type=int, default=-1, help="Which checkpoint-<step> to merge (-1 = latest).")
    args = parser.parse_args()

    out = merge_checkpoint(args.src, args.dst, checkpoint_step=args.checkpoint_step)
    print(f"merged checkpoint written to: {out}")


if __name__ == "__main__":
    main()
