"""Reusable argparse registration for LTX-2 NFT/GRPO RL post-training flags.

Mirrors the RL flags declared inline by the two RL drivers — ``ltx2_cache_rollouts.py``
(Phase A: ``rollout_setup_parser``) and ``ltx2_train_rl.py`` (Phase B: ``rl_setup_parser``) —
so any other surface (GUI command builder, third-party launchers, tests) can register the
SAME flags without importing the heavy driver modules.

This is a helper only: it does NOT modify the drivers. The drivers keep their own inline
parsers (a running GPU job depends on them); this module merely re-declares the identical
flags for reuse, following the same add-args pattern as
``ltx2_av_cross_grad_surgery.parse_av_cross_grad_surgery_args`` and
``ltx2_remote_stage.add_ltx2_remote_stage_args``.

Two registration helpers are exposed:

* ``add_rl_rollout_args`` — Phase A (``ltx2_cache_rollouts.py``) flags.
* ``add_rl_train_args``   — Phase B (``ltx2_train_rl.py``) flags.

``add_rl_args`` registers both (the train superset plus the Phase-A-only options) on one
parser, which is what generic callers usually want. Each helper is idempotent-friendly:
the shared flags (``--rl_rollout_cache``, ``--rl_prompts``, ``--reward_fn``, ``--reward_args``,
``--rl_group_size``, sample dims, ``--sample_steps``) are only added once even when both
phase helpers run on the same parser.
"""

from __future__ import annotations

import argparse


# --- defaults kept in sync with the drivers (single source of truth for callers) ---
DEFAULT_REWARD_FN = "iqa_quality:1.0,anti_noise:0.1"
DEFAULT_NFT_BETA_MIX = 1.0
DEFAULT_NFT_KL_BETA = 1e-4
DEFAULT_NFT_ADV_CLIP_MAX = 5.0
DEFAULT_RL_GROUP_SIZE = 8
DEFAULT_RL_TIMESTEPS_PER_SAMPLE = 1
DEFAULT_RL_MAX_STEPS = 0
DEFAULT_RL_DECAY_TYPE = 1
DEFAULT_SAMPLE_WIDTH = 768
DEFAULT_SAMPLE_HEIGHT = 512
DEFAULT_SAMPLE_FRAMES = 49
DEFAULT_SAMPLE_CFG = 1.0
DEFAULT_SAMPLE_STEPS = 20


def _existing_options(parser: argparse.ArgumentParser) -> set[str]:
    """Return the set of option strings already registered on ``parser``."""
    seen: set[str] = set()
    for action in parser._actions:
        seen.update(action.option_strings)
    return seen


def _add_shared_sampling_args(parser: argparse.ArgumentParser, seen: set[str]) -> None:
    """Sample-dimension flags shared by both Phase A and Phase B (online)."""
    if "--rl_group_size" not in seen:
        parser.add_argument(
            "--rl_group_size", type=int, default=DEFAULT_RL_GROUP_SIZE, help="K samples per prompt (GRPO group size)"
        )
    if "--sample_width" not in seen:
        parser.add_argument("--sample_width", type=int, default=DEFAULT_SAMPLE_WIDTH)
    if "--sample_height" not in seen:
        parser.add_argument("--sample_height", type=int, default=DEFAULT_SAMPLE_HEIGHT)
    if "--sample_frames" not in seen:
        parser.add_argument("--sample_frames", type=int, default=DEFAULT_SAMPLE_FRAMES)
    if "--sample_cfg" not in seen:
        parser.add_argument("--sample_cfg", type=float, default=DEFAULT_SAMPLE_CFG)
    if "--sample_steps" not in seen:
        parser.add_argument("--sample_steps", type=int, default=DEFAULT_SAMPLE_STEPS, help="denoising steps per rollout sample")


def _add_shared_sde_args(parser: argparse.ArgumentParser, seen: set[str]) -> None:
    """Stochastic-sampler flags for PPO (DDPO): Phase A samples each rollout with the SDE sampler
    (per-step Gaussian actions) and caches the trajectory; Phase B (``--rl_loss ppo``) scores the
    exact per-step importance ratio. ``--rl_sde_sampler`` is required in Phase A for ``--rl_loss ppo``.
    """
    if "--rl_sde_sampler" not in seen:
        parser.add_argument(
            "--rl_sde_sampler",
            action="store_true",
            help="Phase A: sample rollouts with the stochastic SDE sampler and cache the per-step "
            "trajectory (required for --rl_loss ppo). Default: deterministic sampler, x0 only.",
        )
    if "--rl_sde_eta" not in seen:
        parser.add_argument(
            "--rl_sde_eta",
            type=float,
            default=1.0,
            help="SDE per-step noise level in [0,1] (std = sigma_next*eta); 1 = fully stochastic. "
            "Must match between Phase A and Phase B.",
        )


def _add_shared_reward_args(parser: argparse.ArgumentParser, seen: set[str]) -> None:
    """Reward-spec flags shared by both phases."""
    if "--reward_fn" not in seen:
        parser.add_argument(
            "--reward_fn", type=str, default=DEFAULT_REWARD_FN, help="reward spec 'name:weight,...' (e.g. 'hpsv3:1.0')"
        )
    if "--reward_args" not in seen:
        parser.add_argument(
            "--reward_args",
            type=str,
            nargs="*",
            default=None,
            help="key=value args passed to each reward's setup() (e.g. checkpoint_path=... config_path=... for hpsv3)",
        )
    if "--reward_plugins" not in seen:
        parser.add_argument(
            "--reward_plugins",
            type=str,
            nargs="*",
            default=None,
            help="paths to custom-reward .py files (imported before --reward_fn is parsed, so their names are usable)",
        )


def add_rl_rollout_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register Phase A (``ltx2_cache_rollouts.py``) rollout-cache flags.

    Mirrors ``ltx2_cache_rollouts.rollout_setup_parser`` (kept identical: same names,
    defaults, and ``required`` semantics). Used to generate + score K rollouts per prompt
    and write a rollout cache.
    """
    seen = _existing_options(parser)
    if "--rl_rollout_cache" not in seen:
        parser.add_argument("--rl_rollout_cache", type=str, required=True, help="output rollout cache dir")
    if "--rl_prompts" not in seen:
        parser.add_argument("--rl_prompts", type=str, required=True, help="text file, one prompt per line")
    _add_shared_reward_args(parser, seen)
    _add_shared_sampling_args(parser, seen)
    _add_shared_sde_args(parser, seen)
    if "--rl_save_old_lora" not in seen:
        parser.add_argument(
            "--rl_save_old_lora",
            type=str,
            default=None,
            help="save the `old` snapshot LoRA (fp32) here; load it as --network_weights in Phase B for the invariant",
        )
    return parser


def add_rl_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register Phase B (``ltx2_train_rl.py``) NFT training-loop flags.

    Mirrors ``ltx2_train_rl.rl_setup_parser`` (kept identical: same names + defaults).
    Covers the offline cache-replay path plus the experimental online path.
    """
    seen = _existing_options(parser)
    if "--rl_rollout_cache" not in seen:
        parser.add_argument("--rl_rollout_cache", type=str, default=None, help="rollout cache dir (Phase A output; offline mode)")
    if "--rl_log_vram" not in seen:
        parser.add_argument(
            "--rl_log_vram",
            action="store_true",
            help="log peak GPU memory of the Phase-B NFT training step in isolation (resets the peak counter before the loop)",
        )
    if "--nft_beta_mix" not in seen:
        parser.add_argument(
            "--nft_beta_mix", type=float, default=DEFAULT_NFT_BETA_MIX, help="NFT pos/neg mix coefficient (config.beta)"
        )
    if "--nft_kl_beta" not in seen:
        parser.add_argument(
            "--nft_kl_beta",
            type=float,
            default=DEFAULT_NFT_KL_BETA,
            help="KL coefficient (config.train.beta) — distinct from beta_mix",
        )
    if "--nft_adv_clip_max" not in seen:
        parser.add_argument("--nft_adv_clip_max", type=float, default=DEFAULT_NFT_ADV_CLIP_MAX)
    if "--nft_video_modality_weight" not in seen:
        parser.add_argument(
            "--nft_video_modality_weight",
            type=float,
            default=1.0,
            help="AV mode: weight of the video NFT branch in compute_nft_loss",
        )
    if "--nft_audio_modality_weight" not in seen:
        parser.add_argument(
            "--nft_audio_modality_weight",
            type=float,
            default=1.0,
            help="AV mode: weight of the audio NFT branch in compute_nft_loss",
        )
    # update-rule selector + per-objective hyperparameters (mirror ltx2_train_rl.rl_setup_parser).
    # choices are inlined as literals so this helper stays import-light (no torch via ltx2_rl_objectives).
    if "--rl_loss" not in seen:
        parser.add_argument(
            "--rl_loss",
            type=str,
            default="nft",
            choices=["nft", "rwr", "dpo", "ppo"],
            help="RL update rule: nft (default, negative-aware) | rwr | dpo | ppo",
        )
    if "--rwr_temperature" not in seen:
        parser.add_argument("--rwr_temperature", type=float, default=1.0, help="rwr: softmax temperature over group advantages")
    if "--dpo_beta" not in seen:
        parser.add_argument(
            "--dpo_beta", type=float, default=5.0, help="dpo: Diffusion-DPO beta (scale-invariant; preference sharpness)"
        )
    if "--ppo_clip_eps" not in seen:
        parser.add_argument("--ppo_clip_eps", type=float, default=0.2, help="ppo: PPO clip epsilon")
    _add_shared_sde_args(parser, seen)
    if "--rl_timesteps_per_sample" not in seen:
        parser.add_argument(
            "--rl_timesteps_per_sample",
            type=int,
            default=DEFAULT_RL_TIMESTEPS_PER_SAMPLE,
            help="random timesteps trained per cached sample",
        )
    if "--rl_max_steps" not in seen:
        parser.add_argument(
            "--rl_max_steps",
            type=int,
            default=DEFAULT_RL_MAX_STEPS,
            help="0 = ~one pass over the cache (fixed-behavior-policy bound)",
        )
    if "--rl_decay_type" not in seen:
        parser.add_argument(
            "--rl_decay_type", type=int, default=DEFAULT_RL_DECAY_TYPE, help="EMA decay schedule for the `old` policy"
        )
    # online mode (generate rollouts inline instead of reading a cache)
    if "--rl_online" not in seen:
        parser.add_argument(
            "--rl_online", action="store_true", help="generate rollouts inline each round instead of reading a cache"
        )
    if "--rl_prompts" not in seen:
        parser.add_argument("--rl_prompts", type=str, default=None, help="prompts file (online mode)")
    _add_shared_reward_args(parser, seen)
    _add_shared_sampling_args(parser, seen)
    if "--rl_dump_cache" not in seen:
        parser.add_argument(
            "--rl_dump_cache",
            type=str,
            default=None,
            help="online mode: also write the inline rollouts here (equivalence harness)",
        )
    return parser


def add_rl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the full RL flag surface (Phase A + Phase B) on one parser.

    The shared flags are de-duplicated, so this is safe even though both phase helpers
    declare ``--rl_rollout_cache`` / ``--rl_prompts`` / reward / sample-dim flags. When a
    flag appears in both phases with different ``required``/``default`` semantics
    (``--rl_rollout_cache``, ``--rl_prompts``), the Phase-B (train) variant is used because
    it is the permissive one (``required=False``), keeping the combined parser usable for
    either driver.
    """
    add_rl_train_args(parser)
    add_rl_rollout_args(parser)
    return parser
