"""LTX-2 NFT/GRPO RL post-training — Phase A (offline rollout generation + scoring).

Standalone driver (mirrors ``ltx2_train_slider.py`` setup): loads the transformer + the ``old``
LoRA snapshot, generates K samples per prompt with that policy, scores them with the reward stack,
computes routed GRPO advantages, and writes a rollout cache (with the ``old`` snapshot hash) that
``ltx2_train_rl.py`` (Phase B) replays.

Scope: plain video LoRA, offline. Rewards that consume decoded media require the decode-to-temp
path; model-backed rewards may require their optional runtime libraries and checkpoints.
"""

import argparse
import os
import sys

import torch
from accelerate.utils import set_seed
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.hv_train_network import prepare_accelerator, read_config_from_file, setup_parser_common
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer, ltx2_setup_parser
from musubi_tuner.ltx2_rewards import RewardStack, load_reward_plugins, parse_reward_spec
from musubi_tuner.ltx2_rollout import generate_groups, score_groups, stack_media_needs, stack_needs_media
from musubi_tuner.ltx2_rollout_cache import RolloutCacheMeta, RolloutCacheWriter, compute_snapshot_hash
from musubi_tuner.ltx2_rl_generate import build_generate_fn, make_sigma_schedule, prepare_sampling_args
from musubi_tuner.utils import model_utils


def _read_prompts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [s for ln in f if (s := ln.strip()) and not s.startswith("#")]


class LTX2RolloutCacher:
    def __init__(self) -> None:
        self._net = LTX2NetworkTrainer()

    def run(self, args: argparse.Namespace) -> None:
        if args.seed is not None:
            set_seed(args.seed)
        self._net.handle_model_specific_args(args)
        accelerator = prepare_accelerator(args)
        device = accelerator.device

        dit_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)
        dit_weight_dtype = (
            (None if getattr(args, "fp8_scaled", False) else torch.float8_e4m3fn) if getattr(args, "fp8_base", False) else dit_dtype
        )

        blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
        self._net.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else device
        transformer = self._net.load_transformer(accelerator, args, args.dit, "torch", False, loading_device, dit_weight_dtype)
        transformer.eval().requires_grad_(False)
        if blocks_to_swap > 0:
            transformer.enable_block_swap(blocks_to_swap, device, supports_backward=False)
            # NOTE: defer move_to_device_except_swap_blocks until AFTER the text encoder is freed,
            # so Gemma (~24 GB) and the DiT never co-reside on the GPU (24 GB generation path).

        vae = self._net.load_vae(args, vae_dtype=torch.float16, vae_path=args.vae) if getattr(args, "vae", None) else None

        # Pre-encode all prompts with the text encoder (Gemma) while the DiT is still on CPU, then
        # FREE Gemma BEFORE moving the DiT to the GPU. Gemma (~24 GB) and the DiT never co-reside ->
        # generation peak fits a 24 GB card (requires --blocks_to_swap so the DiT loads to CPU first).
        prompts = _read_prompts(args.rl_prompts)
        prepare_sampling_args(args)
        te_dtype = self._net._build_text_encoder(args, accelerator)
        prompt_embeds_cache = {}
        for p in dict.fromkeys(prompts):  # unique prompts, order preserved
            prompt_embeds_cache[p] = self._net._encode_prompt_text(accelerator, p, te_dtype)
        self._net._cleanup_text_encoder(accelerator)
        logger.info("Encoded %d unique prompt(s); freed text encoder before generation", len(prompt_embeds_cache))
        if blocks_to_swap > 0:
            transformer.move_to_device_except_swap_blocks(device)  # DiT -> GPU now that Gemma is gone

        # build network + load the `old` LoRA snapshot (the rollout-generating policy)
        sys.path.append(os.path.dirname(__file__))
        import importlib

        network_module = importlib.import_module(args.network_module)
        net_kwargs = {}
        for net_arg in args.network_args or []:
            key, val = net_arg.split("=")
            net_kwargs[key] = val
        network = network_module.create_arch_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            vae,
            None,
            transformer,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        if getattr(args, "network_weights", None):
            network.load_weights(args.network_weights)
        network.set_dropout_enabled(False)  # deterministic policy (rollouts must be reproducible)
        network.to(device)

        # Hash the fp32 adapter (Phase B's EMA hashes fp32 too -> the snapshot invariant matches). The
        # adapter stays fp32; accelerator.autocast() in build_generate_fn casts it to the compute dtype
        # for the matmul, so no manual cast is needed and keeping fp32 is what makes the hash round-trip.
        snapshot_hash = compute_snapshot_hash(network.trainable_lora_params())
        if getattr(args, "rl_save_old_lora", None):
            # The snapshot hash is over the live trainable params, but save_weights() exports a
            # TRANSFORMED state dict for weight-transforming adapters (adaptive-rank lambda scaling,
            # DoRA magnitude, LoKr/DoKr factorization) -> the file Phase B reloads would not reproduce
            # this hash and assert_snapshot would wrongly fail. The RL snapshot path supports plain
            # LoRA only for now.
            _transforming = sorted(
                {
                    type(m).__name__
                    for m in network.modules()
                    if getattr(m, "adaptive_rank", False)
                    or any(tag in type(m).__name__ for tag in ("DoRA", "DoKr", "LoKr", "LoHa"))
                }
            )
            if _transforming:
                raise NotImplementedError(
                    f"--rl_save_old_lora supports plain LoRA only; found weight-transforming modules "
                    f"{_transforming}. Their saved state dict differs from the hashed live params, so "
                    "Phase B's assert_snapshot would fail. Use plain LoRA for RL, or extend "
                    "compute_snapshot_hash to hash the exported state dict."
                )
            network.save_weights(args.rl_save_old_lora, torch.float32, {"ltx2_rl_snapshot_hash": snapshot_hash})
            logger.info("Saved `old` snapshot LoRA (fp32) to %s", args.rl_save_old_lora)

        if getattr(args, "rl_sde_sampler", False) and float(getattr(args, "rl_sde_eta", 1.0)) <= 0.0:
            raise ValueError(
                "--rl_sde_sampler with --rl_sde_eta <= 0 records a deterministic trajectory that PPO "
                "cannot score (zero transition std). Use eta in (0, 1]."
            )

        # reward stack + routes (from --reward_fn, e.g. "iqa_quality:1.0,anti_noise:0.1" or "hpsv3:1.0").
        # --reward_args key=value ... are passed to every reward's setup() (e.g. hpsv3
        # checkpoint_path / config_path). Applied to each selected reward.
        reward_kwargs = {}
        for raw in getattr(args, "reward_args", None) or []:
            if "=" not in raw:
                raise ValueError(f"--reward_args entry '{raw}' must be key=value")
            key, val = raw.split("=", 1)
            reward_kwargs[key.strip()] = val
        if getattr(args, "reward_plugins", None):
            load_reward_plugins(args.reward_plugins)
        weights = parse_reward_spec(args.reward_fn)
        per_reward_args = {name: dict(reward_kwargs) for name in weights}
        reward_stack = RewardStack.from_spec(args.reward_fn, device=device, reward_args=per_reward_args)
        route_map = reward_stack.routes
        group_size = int(args.rl_group_size)
        needs_media = stack_needs_media(reward_stack)
        media_needs = stack_media_needs(reward_stack)

        num_steps = int(getattr(args, "sample_steps", 20) or 20)
        sigma_schedule = make_sigma_schedule(num_steps)
        generate_fn = build_generate_fn(
            self._net,
            args,
            accelerator,
            transformer,
            vae,
            dit_dtype,
            device,
            num_steps=num_steps,
            needs_media=needs_media,
            sigma_schedule=sigma_schedule,
            te_dtype=te_dtype,
            prompt_embeds=prompt_embeds_cache,  # Gemma already freed; use precomputed embeds
            media_needs=media_needs,
        )

        logger.info("Generating rollouts: %d prompts x K=%d, rewards=%s", len(prompts), group_size, list(reward_stack.weights))
        # Pass 1: generate all rollouts while the DiT is resident.
        gen_groups = generate_groups(prompts, group_size=group_size, seed_base=int(args.seed or 0), generate_fn=generate_fn)
        # Free the DiT + VAE before scoring so the (heavy) reward models never co-reside with them
        # (offline VRAM-flat invariant: at any moment either the generator OR a reward is on the GPU).
        self._free_generation_models(transformer, vae, device, blocks_to_swap)
        # Pass 2: load each reward once, score all groups in a single pass, per-group GRPO advantages.
        groups = score_groups(gen_groups, reward_stack, route_map)

        # Learning diagnostic (mirrors ltx2_train_rl.py's online path): mean reward of this cache's
        # rollouts under each reward, in the greppable ROUND_REWARD format. Lets a Phase-A-only
        # evaluation (held-out self-eval / cross-reward matrix) read the score without a Phase B step.
        _round_scores = {}
        for _g in groups:
            for _rname, _sc in _g.get("reward_scores", {}).items():
                _round_scores.setdefault(_rname, []).extend(_sc)
        for _rname, _vals in _round_scores.items():
            _mean = sum(_vals) / len(_vals) if _vals else 0.0
            logger.info("ROUND_REWARD %s mean=%.6f n=%d", _rname, _mean, len(_vals))

        meta = RolloutCacheMeta(
            snapshot_hash=snapshot_hash,
            group_size=group_size,
            reward_names=list(reward_stack.weights),
            route_map=route_map,
            sampler_settings={
                "num_steps": num_steps,
                "frame_rate": getattr(args, "frame_rate", 24.0),
                "width": args.sample_width,
                "height": args.sample_height,
                "frames": args.sample_frames,
                "cfg": getattr(args, "sample_cfg", 1.0),
                "seed_base": int(args.seed or 0),
                # PPO phase coupling: Phase B must score cached actions at the eta they were drawn with.
                "rl_sde_sampler": bool(getattr(args, "rl_sde_sampler", False)),
                "rl_sde_eta": float(getattr(args, "rl_sde_eta", 1.0)),
            },
        )
        writer = RolloutCacheWriter(args.rl_rollout_cache, meta)
        flagged_total = 0
        for g in tqdm(groups, desc="writing cache"):
            flagged_total += len(g["flagged"])
            writer.write_group(
                g["group_idx"],
                prompt=g["prompt"],
                seeds=g["seeds"],
                tensors=g["tensors"],
                reward_scores=g["reward_scores"],
                advantages=g["advantages"],
            )
        index_path = writer.finalize()
        logger.info(
            "Wrote rollout cache: %d groups -> %s (snapshot %s, %d zero-variance reward-groups flagged)",
            len(groups),
            index_path,
            snapshot_hash[:12],
            flagged_total,
        )
        if torch.cuda.is_available():
            logger.info("Phase A peak VRAM allocated: %.2f GB", torch.cuda.max_memory_allocated(device) / 1e9)

    def _free_generation_models(self, transformer, vae, device, blocks_to_swap) -> None:
        """Move the DiT + VAE off the GPU before reward scoring.

        Offline two-pass invariant: at any moment either the generator OR a reward model is on the
        GPU, never both — so HPSv3 (~17 GB) fits even on a 24 GB card once the DiT is unloaded.
        """
        import gc

        try:
            if blocks_to_swap and hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(torch.device("cpu"))
            else:
                transformer.to("cpu")
        except Exception as exc:  # best-effort: scoring may still fit even if this fails
            logger.warning("could not move transformer to CPU before scoring: %s", exc)
        try:
            if vae is not None:
                vae.to_device("cpu") if hasattr(vae, "to_device") else vae.to("cpu")
        except Exception as exc:
            logger.warning("could not move VAE to CPU before scoring: %s", exc)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                "Freed generation models before scoring; VRAM allocated now %.1f GB",
                torch.cuda.memory_allocated(device) / 1e9,
            )


def rollout_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--rl_rollout_cache", type=str, required=True, help="output rollout cache dir")
    parser.add_argument("--rl_prompts", type=str, required=True, help="text file, one prompt per line")
    parser.add_argument(
        "--reward_fn",
        type=str,
        default="iqa_quality:1.0,anti_noise:0.1",
        help="reward spec 'name:weight,...'",
    )
    parser.add_argument(
        "--reward_plugins",
        type=str,
        nargs="*",
        default=None,
        help="paths to custom-reward .py files (imported before --reward_fn is parsed, so their names are usable)",
    )
    parser.add_argument(
        "--reward_args",
        type=str,
        nargs="*",
        default=None,
        help="key=value args passed to each reward's setup() (e.g. checkpoint_path=... config_path=... for hpsv3)",
    )
    parser.add_argument("--rl_group_size", type=int, default=8, help="K samples per prompt (GRPO group size)")
    parser.add_argument("--sample_width", type=int, default=768)
    parser.add_argument("--sample_height", type=int, default=512)
    parser.add_argument("--sample_frames", type=int, default=49)
    parser.add_argument("--sample_cfg", type=float, default=1.0)
    parser.add_argument("--sample_steps", type=int, default=20, help="denoising steps per rollout sample")
    parser.add_argument(
        "--rl_sde_sampler",
        action="store_true",
        help="sample rollouts with the stochastic SDE sampler and cache the per-step trajectory "
        "(required for --rl_loss ppo in Phase B). Default: deterministic sampler, x0 only.",
    )
    parser.add_argument(
        "--rl_sde_eta",
        type=float,
        default=1.0,
        help="SDE per-step noise level in [0,1] (std=sigma_next*eta). 0=deterministic; 1=fully stochastic.",
    )
    parser.add_argument(
        "--rl_save_old_lora",
        type=str,
        default=None,
        help="save the `old` snapshot LoRA (fp32) here; load it as --network_weights in Phase B for the invariant",
    )
    return parser


def main() -> None:
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)
    parser = rollout_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    # LTX-2 single checkpoint serves as DiT + VAE (mirrors ltx2_train_slider.py:1650-1658)
    if getattr(args, "ltx2_checkpoint", None) is not None:
        if getattr(args, "dit", None) != args.ltx2_checkpoint:
            args.dit = args.ltx2_checkpoint
        if getattr(args, "vae", None) != args.ltx2_checkpoint:
            args.vae = args.ltx2_checkpoint
    LTX2RolloutCacher().run(args)


if __name__ == "__main__":
    main()
