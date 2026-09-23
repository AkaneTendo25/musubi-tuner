"""Training-path parity of the YuE2 trainer against the official YuE2 model code.

One cached song, the validation (centre) window, fixed t in {0.03, 0.5, 0.97} and fixed noise:
- ours: ``YuE2NetworkTrainer._plan`` -> ``_context_kv`` (no-grad prefill) -> ``call_dit`` -> ``flow_loss`` on our model;
- reference: the official ``YuE2ForCausalLM.nar_velocity`` (vendored under ``tests/yue2_ref``) on the full sequence
  ``nar_prefix + codes[win] + MUSIC_END + NAR`` with its hybrid AR/NAR attention mask, and the flow loss written out as
  ``mean((v - (eps - z))^2)`` at ``x_t = t * eps + (1 - t) * z``.
AR: our chunked CE and KL(base || LoRA) against the written-out formulas on the official backbone's hidden states
(``-log p(target)`` and ``sum_v p_base (log p_base - log p_lora)``, averaged over the target rows), first with a freshly
initialised (zero-up) joint LoRA, then with random ``lora_up`` (std ``--lora_scale``): ours applies the LoRA unmerged,
the official model gets the same AR delta merged into its split q/k/v/gate/up weights, and the reference base pass is
the official model before the merge.
Criteria: flow rel. diff < 1e-2 per t; CE rel. diff < 1e-2 with both LoRAs; our KL at zero LoRA < 1e-6; with the random
LoRA our KL equals the written-out sum p_base (log p_base - log p_lora) on our model (< 1e-3), differs from the reverse
direction, and matches the reference (< 1e-2).

usage (repository root, PYTHONPATH=src, YUE2_WEIGHTS_DIR set):
  python tests/gpu/yue2_train_parity.py --latent_cache DIR/xxx_yue2.safetensors [--window 750] [--out report.json]
  python tests/gpu/yue2_train_parity.py --cache_dir DIR   (first *_yue2.safetensors in DIR)
"""

import argparse
import glob
import json
import math
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (os.path.join(ROOT, "src"), os.path.join(ROOT, "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from musubi_tuner.dataset.cache_io import YUE2_SEG_KEY  # noqa: E402
from musubi_tuner.yue2.yue2_protocol import CODEC_OFFSET, MUSIC_END  # noqa: E402

DEV = torch.device("cuda")

# region reference losses on the official model


def reference_hidden(model, ids: torch.Tensor) -> torch.Tensor:
    """Final-norm hidden states ``[L, H]`` of the official backbone (AR path, causal, positions 0..L-1)."""
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    hidden, _ = model.model(input_ids=ids, position_ids=positions, use_cache=False)
    return hidden[0]


def reference_ce_kl(model, hidden: torch.Tensor, base: torch.Tensor, labels: torch.Tensor, chunk: int = 512):
    """Mean ``-log p(label)`` and mean ``sum_v p_base (log p_base - log p)`` over the rows, fp32 log-softmax."""
    ce = kl = 0.0
    for s in range(0, labels.numel(), chunk):
        logp = model.lm_head(hidden[s : s + chunk]).float().log_softmax(-1)
        logb = model.lm_head(base[s : s + chunk]).float().log_softmax(-1)
        ce += float(-logp.gather(-1, labels[s : s + chunk, None]).sum())
        kl += float((logb.exp() * (logb - logp)).sum())
    return ce / labels.numel(), kl / labels.numel()


def reference_flow_loss(model, prefix: list[int], codes: list[int], z: torch.Tensor, t: float, eps: torch.Tensor) -> float:
    """``mean((v - (eps - z))^2)`` with ``v`` from ``nar_velocity`` on ``prefix + codes + MUSIC_END`` and the NAR block
    (START + ``len(z)`` frames + END; the NAR token ids are replaced by latent embeddings inside ``nar_velocity``)."""
    ar = prefix + [c + CODEC_OFFSET for c in codes] + [MUSIC_END]
    frames = z.shape[0]
    n_nar = frames + 2
    tokens = torch.tensor([ar + [0] * n_nar], device=DEV)
    ar_mask = torch.zeros_like(tokens, dtype=torch.bool)
    ar_mask[0, : len(ar)] = True
    nar_mask = ~ar_mask
    content = torch.zeros_like(ar_mask)
    content[0, len(ar) + 1 : len(ar) + 1 + frames] = True
    x_t = t * eps + (1.0 - t) * z
    v = model.nar_velocity(tokens, ar_mask, nar_mask, content, x_t, math.log(t / (1.0 - t)))
    return float(((v.float() - (eps - z)) ** 2).mean())


# endregion


class _Accelerator:
    device = DEV

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16)


def reference_model(weights: str):
    """The official ``YuE2ForCausalLM`` (``tests/yue2_ref``, vendored unmodified) with the HF checkpoint weights."""
    from yue2_ref.modeling_yue2 import YuE2Config as RefConfig, YuE2ForCausalLM

    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

    hf = os.path.join(weights, "m-a-p__YuE2-3B")
    with open(os.path.join(hf, "config.json")) as f:
        raw = json.load(f)
    cfg = RefConfig(**{k: v for k, v in raw.items() if k in RefConfig._inference_fields})
    with torch.device("meta"):
        model = YuE2ForCausalLM(cfg)
    sd = {}
    with MemoryEfficientSafeOpen(os.path.join(hf, "model.safetensors")) as f:
        for key in f.keys():
            sd[key] = f.get_tensor(key, device=DEV)
    model.load_state_dict(sd, strict=True, assign=True)
    return model.to(DEV).eval()


def batch_from_cache(latent_cache: str):
    from safetensors.torch import load_file

    from musubi_tuner.yue2.yue2_sampling import load_reconstruct_cache

    src = load_reconstruct_cache(latent_cache)
    if not src.text_ids:
        sys.exit(f"no text cache next to {latent_cache}")
    seg = load_file(latent_cache)[YUE2_SEG_KEY]
    batch = {
        "codes": torch.tensor([src.codes], dtype=torch.long),
        "yue2_seg": seg[None],
        "yue2_abc": [torch.tensor(src.abc_ids or [], dtype=torch.long)],
        "yue2_abc_mode": torch.tensor([{None: 0, "melody": 1, "full": 2}[src.abc_mode]]),
    }
    for cot in ("off", "melody", "full"):
        batch[f"yue2_text_{cot}"] = [torch.tensor(src.text_ids[cot], dtype=torch.long)]
        batch[f"yue2_neg_{cot}"] = [torch.tensor(src.neg_ids[cot], dtype=torch.long)]
    return batch, src.latents, src.codes


def main():
    parser = argparse.ArgumentParser(description="YuE2 training-path parity vs the official model code")
    parser.add_argument("--latent_cache", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--dit", default=None, help="our checkpoint (default: Comfy bf16 under YUE2_WEIGHTS_DIR)")
    parser.add_argument("--window", type=int, default=750, help="NAR window frames (centre window)")
    parser.add_argument("--ar_max_tokens", type=int, default=0, help="AR codes from the song start (0 = whole song)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lora_scale", type=float, default=0.15, help="std of the random AR lora_up for the KL check")
    parser.add_argument("--out", default=None)
    cli = parser.parse_args()

    weights = os.environ.get("YUE2_WEIGHTS_DIR")
    if not weights or not torch.cuda.is_available():
        sys.exit("needs YUE2_WEIGHTS_DIR and CUDA")
    latent_cache = cli.latent_cache or sorted(glob.glob(os.path.join(cli.cache_dir or ".", "*_yue2.safetensors")))[0]
    dit = cli.dit or os.path.join(weights, "Comfy-Org__YuE2", "checkpoints", "yue2_3b_bf16.safetensors")

    from musubi_tuner.hv_train_network import setup_parser_common
    from musubi_tuner.networks import lora_yue2
    from musubi_tuner.yue2 import yue2_lora_formats as lora_formats
    from musubi_tuner.yue2.yue2_checkpoint import load_yue2_model
    from musubi_tuner.yue2.yue2_training import ar_targets, chunked_ce_kl, flow_loss as our_flow_loss
    from musubi_tuner.yue2_train_network import YuE2NetworkTrainer, yue2_setup_parser

    args = yue2_setup_parser(setup_parser_common()).parse_args(
        ["--dit", dit, "--nar_window_frames", str(cli.window), "--ar_max_tokens", str(cli.ar_max_tokens), "--sdpa"]
    )
    trainer = YuE2NetworkTrainer()
    trainer.branches = ("ar", "nar")
    acc = _Accelerator()

    batch, latents, codes = batch_from_cache(latent_cache)
    total = latents.shape[0]
    plan = trainer._plan(args, batch, 0, total, random.Random(0), training=False)
    s, w = plan.window_start, plan.window_frames
    report = {
        "latent_cache": os.path.basename(latent_cache),
        "frames": total,
        "prefix_len": len(plan.nar_prefix),
        "window": [s, w],
        "ar_len": len(plan.ar_ids) if plan.ar_ids else None,
    }

    ours = load_yue2_model(dit, device=DEV, loading_device=DEV)
    ref = reference_model(weights)
    z = latents[s : s + w].to(DEV).float()
    g = torch.Generator().manual_seed(cli.seed)
    eps = torch.randn(w, 64, generator=g).to(DEV)

    flow = {}
    with torch.no_grad():
        with acc.autocast():
            kv = trainer._context_kv(args, ours, plan)
        for t in (0.03, 0.5, 0.97):
            x_t = (1.0 - t) * z + t * eps
            tt = torch.tensor([t], device=DEV)
            out = trainer.call_dit(
                args,
                acc,
                ours,
                z[None],
                batch,
                eps[None],
                x_t[None],
                tt * 1000.0 + 1.0,
                torch.bfloat16,
                kv=kv,
                rope_offset=plan.rope_offset,
                t_value=tt,
            )
            mine = float(our_flow_loss(out.pred[0], eps, z))
            with acc.autocast():
                theirs = reference_flow_loss(ref, plan.nar_prefix, codes[s : s + w], z, t, eps)
            flow[t] = {"ours": mine, "reference": theirs, "rel": abs(mine - theirs) / abs(theirs)}
        del kv
    report["flow"] = flow

    ar = {}
    if plan.ar_ids is None:
        ar["skipped"] = plan.ar_skipped_reason
    else:
        net = lora_yue2.create_arch_network(1.0, 16, 16, None, [], ours, branches="ar,nar")
        net.apply_to(None, ours, apply_text_encoder=False, apply_unet=True)
        net.to(DEV)
        ids = torch.tensor([plan.ar_ids], device=DEV)

        def our_losses():
            with torch.no_grad(), acc.autocast():
                net.set_enabled(False)
                base, n = trainer._ar_hidden(args, ours, plan.ar_ids, grad=False)
                net.set_enabled(True)
                hid, _ = trainer._ar_hidden(args, ours, plan.ar_ids, grad=False)
                pos, labels = ar_targets(plan.ar_ids, plan.ar_target_start, n)
                ce, kl = chunked_ce_kl(ours.ar.lm_head, hid[pos], labels, base[pos], args.ar_ce_chunk)
                # KL(base || LoRA) written out as sum_v p_base * (log p_base - log p_lora); the reverse direction too
                fwd = rev = 0.0
                for s in range(0, hid[pos].shape[0], 512):
                    lp = ours.ar.lm_head(hid[pos][s : s + 512]).float().log_softmax(-1)
                    lb = ours.ar.lm_head(base[pos][s : s + 512]).float().log_softmax(-1)
                    fwd += float((lb.exp() * (lb - lp)).sum())
                    rev += float((lp.exp() * (lp - lb)).sum())
            rows = labels.numel()
            return float(ce), float(kl), fwd / rows, rev / rows, int(rows)

        ce, kl_zero, _, _, rows = our_losses()
        ref_labels = ids[0, plan.ar_target_start :]
        with torch.no_grad(), acc.autocast():
            ref_base = reference_hidden(ref, ids)[plan.ar_target_start - 1 : -1]
            ref_ce, ref_kl_zero = reference_ce_kl(ref, ref_base, ref_base, ref_labels)

        # a non-zero AR LoRA: ours applies it unmerged, the official model gets the same delta merged. The
        # fl-yue2-lora-v1 export uses the checkpoint's split module names (q/k/v, gate/up) with the scale baked into
        # lora_up, so every pair merges into the matching weight as up @ down.
        g = torch.Generator().manual_seed(cli.seed)
        with torch.no_grad():
            for name, p in net.named_parameters():
                if "lora_up" in name:
                    p.copy_(cli.lora_scale * torch.randn(p.shape, generator=g).to(p))
        sd = {k: v.detach().float().cpu() for k, v in net.state_dict().items()}
        split_sd, _ = lora_formats.native_to_fl(lora_formats.filter_branch(lora_formats.to_native(sd, {}), "ar"))["ar"]
        params = dict(ref.named_parameters())
        merged = 0
        with torch.no_grad():
            for key in split_sd:
                if not key.endswith(".lora_down.weight"):
                    continue
                stem = key[: -len(".lora_down.weight")]
                w = params[stem + ".weight"]
                delta = split_sd[stem + ".lora_up.weight"].to(DEV) @ split_sd[key].to(DEV)
                w.copy_((w.float() + delta).to(w.dtype))
                merged += 1
        ce_l, kl_l, kl_fwd, kl_rev, _ = our_losses()
        with torch.no_grad(), acc.autocast():
            ref_hid = reference_hidden(ref, ids)[plan.ar_target_start - 1 : -1]
            ref_ce_l, ref_kl_l = reference_ce_kl(ref, ref_hid, ref_base, ref_labels)
        ar = {
            "ours_ce": ce,
            "reference_ce": ref_ce,
            "ce_rel": abs(ce - ref_ce) / abs(ref_ce),
            "ours_kl_zero_lora": kl_zero,
            "reference_kl_zero_lora": ref_kl_zero,
            "lora_scale": cli.lora_scale,
            "merged_modules": merged,
            "ours_ce_lora": ce_l,
            "reference_ce_lora": ref_ce_l,
            "ce_lora_rel": abs(ce_l - ref_ce_l) / abs(ref_ce_l),
            "ours_kl_lora": kl_l,
            "explicit_kl_base_lora": kl_fwd,
            "explicit_kl_lora_base": kl_rev,
            "reference_kl_lora": ref_kl_l,
            "kl_lora_rel": abs(kl_l - ref_kl_l) / abs(ref_kl_l),
            "targets": rows,
            "complete": plan.ar_complete,
        }
    report["ar"] = ar

    checks = {f"flow_t{t}": v["rel"] < 1e-2 for t, v in flow.items()}
    if "ce_rel" in ar:
        checks["ce"] = ar["ce_rel"] < 1e-2
        checks["kl_zero_lora"] = abs(ar["ours_kl_zero_lora"]) < 1e-6
        checks["merged_ar_modules"] = ar["merged_modules"] > 0
        checks["ce_lora"] = ar["ce_lora_rel"] < 1e-2
        checks["kl_lora_nonzero"] = ar["ours_kl_lora"] > 1e-3
        # direction: our KL is KL(base || LoRA), measurably different from the reverse
        checks["kl_direction"] = abs(ar["ours_kl_lora"] - kl_fwd) / kl_fwd < 1e-3 and abs(kl_fwd - kl_rev) / kl_fwd > 1e-2
        checks["kl_lora_vs_reference"] = ar["kl_lora_rel"] < 1e-2
    report["checks"] = checks
    report["pass"] = all(checks.values())
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
        with open(cli.out, "w") as f:
            f.write(text)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
