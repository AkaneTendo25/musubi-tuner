"""LTX-2 NFT/GRPO RL post-training — Phase B (offline cache replay).

Standalone driver (mirrors ``ltx2_train_slider.py``): instantiates ``LTX2NetworkTrainer`` for
setup only, then runs the RL loop — it does NOT use the supervised ``train()`` loop or ``call_dit``'s
target construction. The loop replays a rollout cache (written by ``ltx2_cache_rollouts.py``):
per group / per random timestep it re-noises the cached clean x0, runs THREE forwards on one frozen
fp8 base — ``default`` (grad), ``old`` (no_grad EMA via ``LoraEMA.swapped``), ``ref`` (no_grad,
LoRA disabled) — and applies ``compute_nft_loss``.

Video path: plain video LoRA, offline.

AV path (``--ltx2_mode av``): the cache additionally carries ``audio_x0`` + ``a_ctx`` + ``a_mask``
(written by the AV generate_fn). The loop re-noises the audio latent the same way, runs the three
forwards in AV mode (``model_input=[video, audio]`` so the AV transformer returns video+audio
predictions), and builds ``modality_states={"video": ..., "audio": ...}`` for ``compute_nft_loss``
with the cached ``adv_audio``. The block-swap 3-forward ordering is identical to the video path.
The video path is unchanged when not in AV mode.

Runtime details such as tensor shapes, block-swap method availability, frame_rate source, and
transformer output unpacking are model-dependent and should be confirmed against the target model.
"""

import argparse
import os
import random
import sys

import torch
from accelerate.utils import set_seed
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    prepare_accelerator,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer, ltx2_setup_parser
from musubi_tuner.ltx2_rl_objectives import RL_LOSS_CHOICES, compute_rl_objective
from musubi_tuner.ltx2_lora_ema import LoraEMA
from musubi_tuner.ltx2_rollout_cache import RolloutCacheReader
from musubi_tuner.ltx_2.utils import to_denoised
from musubi_tuner.utils import model_utils


def _flatten_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, C, F, H, W] -> [B, T, C] token layout expected by compute_nft_loss (T = F*H*W)."""
    if x.dim() == 3:
        return x  # already [B, T, C]
    if x.dim() != 5:
        raise ValueError(f"expected 5D [B,C,F,H,W] or 3D [B,T,C] latent, got {tuple(x.shape)}")
    b, c, f, h, w = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c).contiguous()


def _flatten_audio_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T, F] audio latent -> [B, T, C*F] token layout for compute_nft_loss.

    Tokens = the audio time axis T; channels = C*F_mel. fwd/old/ref/x0 are all flattened the
    same way, so the NFT reductions (over token + channel) stay consistent across the four.
    """
    if x.dim() == 3:
        return x  # already [B, T, C]
    if x.dim() != 4:
        raise ValueError(f"expected 4D [B,C,T,F] audio latent or 3D [B,T,C], got {tuple(x.shape)}")
    b, c, t, f = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, t, c * f).contiguous()


class LTX2RLTrainer:
    def __init__(self) -> None:
        self._net = LTX2NetworkTrainer()

    def train(self, args: argparse.Namespace) -> None:
        if args.seed is not None:
            set_seed(args.seed)
        self._net.handle_model_specific_args(args)
        is_av = getattr(self._net, "_ltx_mode", getattr(args, "ltx2_mode", "video")) == "av"

        accelerator = prepare_accelerator(args)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
        device = accelerator.device

        dit_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)
        dit_weight_dtype = (
            (None if getattr(args, "fp8_scaled", False) else torch.float8_e4m3fn) if getattr(args, "fp8_base", False) else dit_dtype
        )

        # --- model + LoRA setup (mirrors ltx2_train_slider.py:1138-1279) ---
        blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
        self._net.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else device
        transformer = self._net.load_transformer(accelerator, args, args.dit, "torch", False, loading_device, dit_weight_dtype)
        transformer.eval()
        transformer.requires_grad_(False)

        if blocks_to_swap > 0:
            transformer.enable_block_swap(
                blocks_to_swap,
                device,
                supports_backward=True,
                use_pinned_memory=getattr(args, "use_pinned_memory_for_block_swap", False),
            )
            transformer.move_to_device_except_swap_blocks(device)

        sys.path.append(os.path.dirname(__file__))
        import importlib

        network_module = importlib.import_module(args.network_module)
        net_kwargs = {}
        for net_arg in args.network_args or []:
            k, v = net_arg.split("=")
            net_kwargs[k] = v
        network = network_module.create_arch_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            None,
            None,
            transformer,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )
        if network is None:
            raise RuntimeError("failed to create LoRA network")
        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        # warm-start: the `default` adapter == the `old` snapshot that generated the cache
        if getattr(args, "network_weights", None):
            info = network.load_weights(args.network_weights)
            logger.info("Loaded warm-start LoRA from %s: %s", args.network_weights, info)

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
            try:
                network.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
            except TypeError:
                network.enable_gradient_checkpointing()

        # deterministic RL forwards (no LoRA dropout, even in train mode)
        network.set_dropout_enabled(False)

        trainable_params, _ = network.prepare_optimizer_params(unet_lr=args.learning_rate)
        _, _, optimizer, _, _ = NetworkTrainer().get_optimizer(args, trainable_params)
        lr_scheduler = NetworkTrainer().get_lr_scheduler(args, optimizer, accelerator.num_processes)

        if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
            transformer.to(dit_weight_dtype)
        if blocks_to_swap > 0:
            transformer = accelerator.prepare(transformer, device_placement=[False])
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(device)
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)
        network, optimizer, lr_scheduler = accelerator.prepare(network, optimizer, lr_scheduler)
        transformer.train()  # train mode; dropout already disabled via set_dropout_enabled
        accelerator.unwrap_model(network).prepare_grad_etc(transformer)

        net = accelerator.unwrap_model(network)
        unwrapped = accelerator.unwrap_model(transformer)

        # --- EMA (`old`) ---
        # `old` is the fixed behavior policy that generated the cache: it stays FROZEN equal to the
        # warm-start snapshot for the whole Phase-B run (ema.step() is never called), which is exactly
        # what the snapshot-hash invariant requires. `old` advances only across rounds via Phase A, so
        # --rl_decay_type is inert here (reserved for an online EMA variant).
        ema = LoraEMA(net, decay_type=int(getattr(args, "rl_decay_type", 1)), device="cpu")
        ema.sync_with_model()  # old == default == warm-start snapshot

        beta_mix = float(getattr(args, "nft_beta_mix", 1.0))
        kl_beta = float(getattr(args, "nft_kl_beta", 1e-4))
        adv_clip_max = float(getattr(args, "nft_adv_clip_max", 5.0))
        num_train_t = int(getattr(args, "rl_timesteps_per_sample", 1))
        # Update rule: NFT (default) or a pluggable non-NFT objective (rwr/dpo/ppo). All consume the
        # same `states` dict; only the final loss differs (ltx2_rl_objectives.compute_rl_objective).
        rl_loss = (getattr(args, "rl_loss", "nft") or "nft").lower()
        logger.info("RL update rule: %s", rl_loss)
        # Authentic PPO (DDPO): the importance ratio is the per-step SDE transition probability at the
        # sampled action; Phase B replays the cached trajectory step (video + audio in AV mode).
        _ppo = rl_loss == "ppo"
        if _ppo:
            if float(getattr(args, "rl_sde_eta", 1.0)) <= 0.0:
                raise ValueError(
                    "--rl_loss ppo requires --rl_sde_eta > 0: at eta=0 the SDE step is deterministic, "
                    "the per-step importance ratio saturates the log-ratio clamp, and the PPO policy "
                    "gradient is zero (no learning). Use eta in (0, 1] and match it in Phase A."
                )
            logger.info(
                "PPO: per-step ratio (eta=%.3f)%s",
                float(getattr(args, "rl_sde_eta", 1.0)),
                " [AV: video+audio]" if is_av else "",
            )
        # AV modality weights for the combined NFT loss (video + audio NFT branches).
        modality_weights = {"video": 1.0}
        if is_av:
            modality_weights = {
                "video": float(getattr(args, "nft_video_modality_weight", 1.0)),
                "audio": float(getattr(args, "nft_audio_modality_weight", 1.0)),
            }

        # --- rollout source: inline generation (online) OR cache replay (offline) ---
        # Both produce the SAME group dicts; the only difference is where the samples come from.
        if getattr(args, "rl_online", False):
            from musubi_tuner.ltx2_rewards import RewardStack, load_reward_plugins, parse_reward_spec
            from musubi_tuner.ltx2_rl_generate import build_generate_fn, make_sigma_schedule, prepare_sampling_args
            from musubi_tuner.ltx2_rollout import generate_and_score_groups, stack_media_needs, stack_needs_media

            gen_vae = self._net.load_vae(args, vae_dtype=torch.float16, vae_path=args.vae) if getattr(args, "vae", None) else None
            reward_kwargs = {}
            for raw in getattr(args, "reward_args", None) or []:
                if "=" not in raw:
                    raise ValueError(f"--reward_args entry '{raw}' must be key=value")
                key, val = raw.split("=", 1)
                reward_kwargs[key.strip()] = val
            if getattr(args, "reward_plugins", None):
                load_reward_plugins(args.reward_plugins)
            per_reward_args = {name: dict(reward_kwargs) for name in parse_reward_spec(args.reward_fn)}
            reward_stack = RewardStack.from_spec(args.reward_fn, device=device, reward_args=per_reward_args)
            needs_media = stack_needs_media(reward_stack)
            media_needs = stack_media_needs(reward_stack)
            num_steps = int(getattr(args, "sample_steps", 20) or 20)
            sigma_schedule = make_sigma_schedule(num_steps)
            prepare_sampling_args(args)
            te_dtype = self._net._build_text_encoder(args, accelerator)
            gen_fn = build_generate_fn(
                self._net,
                args,
                accelerator,
                transformer,
                gen_vae,
                dit_dtype,
                device,
                num_steps=num_steps,
                needs_media=needs_media,
                sigma_schedule=sigma_schedule,
                te_dtype=te_dtype,
                media_needs=media_needs,
            )
            with open(args.rl_prompts, encoding="utf-8") as f:
                prompts = [s for ln in f if (s := ln.strip()) and not s.startswith("#")]
            groups = generate_and_score_groups(
                prompts,
                group_size=int(args.rl_group_size),
                seed_base=int(args.seed or 0),
                generate_fn=gen_fn,
                reward_stack=reward_stack,
                route_map=reward_stack.routes,
            )
            self._net._cleanup_text_encoder(accelerator)
            frame_rate = float(getattr(args, "frame_rate", 24.0))
            logger.info("ONLINE: generated %d groups inline (snapshot %s)", len(groups), ema.snapshot_hash()[:12])
            # Learning diagnostic: mean reward of THIS round's rollouts (under the current policy).
            # Across warm-started rounds this should trend UP if the policy is improving on the reward.
            _round_scores = {}
            for _g in groups:
                for _rname, _sc in _g.get("reward_scores", {}).items():
                    _round_scores.setdefault(_rname, []).extend(_sc)
            for _rname, _vals in _round_scores.items():
                _mean = sum(_vals) / len(_vals) if _vals else 0.0
                logger.info("ROUND_REWARD %s mean=%.6f n=%d", _rname, _mean, len(_vals))
            if getattr(args, "rl_dump_cache", None):
                # equivalence harness: persist the inline rollouts so an offline run can replay the
                # IDENTICAL samples — isolating the cache round-trip from any generation difference.
                from musubi_tuner.ltx2_rollout_cache import RolloutCacheMeta, RolloutCacheWriter

                dump_meta = RolloutCacheMeta(
                    snapshot_hash=ema.snapshot_hash(),
                    group_size=int(args.rl_group_size),
                    reward_names=list(reward_stack.weights),
                    route_map=reward_stack.routes,
                    sampler_settings={"num_steps": num_steps, "frame_rate": frame_rate},
                )
                dump_writer = RolloutCacheWriter(args.rl_dump_cache, dump_meta)
                for grp in groups:
                    dump_writer.write_group(
                        grp["group_idx"],
                        prompt=grp["prompt"],
                        seeds=grp["seeds"],
                        tensors=grp["tensors"],
                        reward_scores=grp["reward_scores"],
                        advantages=grp["advantages"],
                    )
                dump_writer.finalize()
                logger.info("ONLINE: dumped inline rollouts to %s", args.rl_dump_cache)
        else:
            reader = RolloutCacheReader(args.rl_rollout_cache)
            reader.assert_snapshot(ema.snapshot_hash())  # cache must match the `old` policy
            groups = [reader.read_group(gi, device="cpu") for gi in range(len(reader))]
            frame_rate = float(reader.sampler_settings.get("frame_rate", getattr(args, "frame_rate", 24.0)))
            if _ppo:
                # PPO phase coupling: the importance ratio must be scored at the eta the cached actions
                # were drawn with — adopt the cache's eta rather than trusting two CLI flags to match.
                if reader.sampler_settings.get("rl_sde_sampler") is False:
                    raise ValueError(
                        "--rl_loss ppo requires an SDE-sampled rollout cache, but this cache was "
                        "generated without --rl_sde_sampler. Regenerate Phase A with --rl_sde_sampler."
                    )
                _cached_eta = reader.sampler_settings.get("rl_sde_eta")
                if _cached_eta is not None:
                    if abs(float(_cached_eta) - float(getattr(args, "rl_sde_eta", 1.0))) > 1e-9:
                        logger.warning(
                            "PPO: adopting the cache's --rl_sde_eta=%.4f (Phase B was launched with "
                            "%.4f); cached actions must be scored at the eta they were drawn with.",
                            float(_cached_eta),
                            float(getattr(args, "rl_sde_eta", 1.0)),
                        )
                    args.rl_sde_eta = float(_cached_eta)
                    if args.rl_sde_eta <= 0.0:
                        raise ValueError("rollout cache records --rl_sde_eta <= 0; unusable for PPO")
            logger.info("OFFLINE: rollout cache OK, %d groups, snapshot %s", len(groups), ema.snapshot_hash()[:12])

        one_pass = len(groups) * num_train_t
        max_steps = int(getattr(args, "rl_max_steps", 0)) or one_pass
        if max_steps > one_pass:
            # Fixed-behavior-policy bound: rollouts came from `old`, but `default` drifts after the
            # first step and there is NO importance ratio / PPO clip. Amortizing many steps over one
            # frozen cache silently trains off-policy (reward-hacking / collapse risk). The safe pattern
            # is to regenerate the cache (Phase A with the updated `old`) between rounds, ~one pass each.
            logger.warning(
                "rl_max_steps=%d exceeds one pass over the cache (%d = %d groups x %d timesteps). "
                "This over-amortizes a STALE cache with no off-policy correction; prefer regenerating "
                "the rollout cache between rounds instead of training many passes on one cache.",
                max_steps,
                one_pass,
                len(groups),
                num_train_t,
            )

        # Re-seed so the training-loop RNG (re-noise + timestep choice) is identical online vs offline,
        # independent of any RNG consumed during inline generation -> exact equivalence by construction.
        set_seed(int(args.seed or 0) + 7)

        # VRAM diagnostic: reset the peak counter HERE so the reported peak isolates the Phase-B
        # NFT training step (3 forwards + backward + optimizer) from any Phase-A generation that
        # ran earlier in this process (online mode). Gated on --rl_log_vram.
        _log_vram = bool(getattr(args, "rl_log_vram", False)) and torch.cuda.is_available()
        if _log_vram:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            logger.info(
                "VRAM[pre-NFT-loop]: allocated=%.2f GB reserved=%.2f GB",
                torch.cuda.memory_allocated(device) / 1e9,
                torch.cuda.memory_reserved(device) / 1e9,
            )

        # warn once if a modality branch is skipped for lack of a reward signal on its route.
        _warned_audio_null = [False]
        _warned_video_null = [False]
        global_step = 0
        # with --logging_dir, mirror the per-step scalars into TensorBoard (one run per output_name)
        tb_writer = None
        if accelerator.is_main_process and getattr(args, "logging_dir", None):
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(os.path.join(args.logging_dir, args.output_name or "ltx2_rl"))
        progress = tqdm(total=max_steps, desc="RL (NFT)")
        for group in groups:
            if global_step >= max_steps:
                break
            x0 = group["tensors"]["video_x0"].to(device=device, dtype=dit_dtype)  # [K, C, F, H, W]
            v_ctx = group["tensors"]["v_ctx"].to(device=device, dtype=dit_dtype)
            v_mask = group["tensors"].get("v_mask")
            if v_mask is not None:
                v_mask = v_mask.to(device=device)
            sigmas = group["tensors"]["sigmas"].to(device=device, dtype=torch.float32)  # [K, num_steps]
            adv = group["advantages"]["video"].to(device=device, dtype=torch.float32)  # [K]
            k = x0.shape[0]

            # Authentic PPO: pull the cached SDE trajectory (per-step state/action/sigmas) for this group.
            _traj = None
            if _ppo:
                tt = group["tensors"]
                if "traj_x_t" not in tt:
                    raise ValueError(
                        "--rl_loss ppo requires an SDE-sampled rollout cache, but this group has no "
                        "'traj_x_t'. Regenerate Phase A with --rl_sde_sampler."
                    )
                # x0_gen = the gen-time x0 prediction = the exact behavior policy pi_old.
                # The big per-step tensors stay CPU-resident — only the one subsampled step moves to
                # the GPU below, so PPO trajectory VRAM is O(1) per group instead of O(sample_steps).
                _traj = {
                    "x_t": tt["traj_x_t"],  # [K, S, C, F, H, W] (cpu)
                    "x_next": tt["traj_x_next"],  # [K, S, C, F, H, W] (cpu)
                    "x0_gen": tt["traj_x0_gen"],  # [K, S, C, F, H, W] (cpu)
                    "sigma": tt["traj_sigma"].to(device=device, dtype=torch.float32),  # [K, S]
                    "sigma_next": tt["traj_sigma_next"].to(device=device, dtype=torch.float32),  # [K, S]
                }
                # Only stochastic steps carry a usable importance ratio: the terminal sigma_next==0
                # step is deterministic (std 0 -> log-ratio saturates the clamp -> zero gradient).
                _S = int(_traj["sigma_next"].shape[1])
                _valid_steps = [i for i in range(_S) if float(_traj["sigma_next"][:, i].max()) > 0.0]
                if not _valid_steps:
                    raise ValueError(
                        "--rl_loss ppo: the cached trajectory has no stochastic steps (all sigma_next == 0); nothing to score."
                    )
                _traj["valid_steps"] = _valid_steps
                if is_av:
                    if "traj_audio_x_t" not in tt:
                        raise ValueError(
                            "AV --rl_loss ppo requires the audio trajectory ('traj_audio_x_t') in the "
                            "cache. Regenerate Phase A with --ltx2_mode av --rl_sde_sampler."
                        )
                    _traj["audio_x_t"] = tt["traj_audio_x_t"]  # [K,S,C,T,F] (cpu)
                    _traj["audio_x_next"] = tt["traj_audio_x_next"]  # (cpu)
                    _traj["audio_x0_gen"] = tt["traj_audio_x0_gen"]  # (cpu)

            # --- AV: pull the cached audio NFT targets (gated behind --ltx2_mode av) ---
            audio_x0 = None
            adv_audio = None
            if is_av:
                audio_x0_cpu = group["tensors"].get("audio_x0")
                if audio_x0_cpu is None:
                    raise ValueError(
                        "AV RL (--ltx2_mode av) requires 'audio_x0' in the rollout cache, but this group "
                        "has none. Regenerate the cache with ltx2_cache_rollouts --ltx2_mode av."
                    )
                audio_x0 = audio_x0_cpu.to(device=device, dtype=dit_dtype)  # [K, C, T, F]
                # Phase B feeds the FULL AV context (v_ctx = [video|audio] concat) to the transformer;
                # the wrapper re-splits it. a_ctx is cached for explicitness/inspection but the forward
                # uses v_ctx so the split matches generation exactly.
                adv_audio = group["advantages"].get("audio")
                if adv_audio is None:
                    raise ValueError("AV RL: cache group is missing the 'audio' advantage route")
                adv_audio = adv_audio.to(device=device, dtype=torch.float32)  # [K]

            # A modality whose advantage is all-zero (no reward routed to it / zero-variance / flagged)
            # carries no learning signal; training it would regress that LoRA toward `old` for nothing.
            _video_live = bool(adv.abs().max() > 0)
            _audio_live = bool(adv_audio.abs().max() > 0) if is_av else False
            if not _video_live and not _audio_live:
                if not _warned_video_null[0]:
                    logger.warning(
                        "RL: video advantage is all-zero (no active video-routed reward / zero-variance "
                        "group) and no live audio signal -> skipping such groups; check that --reward_fn "
                        "routes a reward to the trained modality."
                    )
                    _warned_video_null[0] = True
                continue

            for _ in range(num_train_t):
                if global_step >= max_steps:
                    break
                if _ppo:
                    # Replay one stochastic trajectory step (subsampled, DDPO-style): the forward runs on
                    # the cached state x_t at its sigma; the cached x_next is the action to score. No
                    # re-noising — x_t already lives on the policy's sampling path.
                    s_idx = random.choice(_traj["valid_steps"])
                    xt = _traj["x_t"][:, s_idx].to(device=device, dtype=dit_dtype)  # [K, C, F, H, W]
                    ppo_action = _traj["x_next"][:, s_idx].to(device=device, dtype=dit_dtype)  # [K, C, F, H, W]
                    ppo_x0_gen = _traj["x0_gen"][:, s_idx].to(device=device, dtype=dit_dtype)  # pi_old
                    sigma = _traj["sigma"][:, s_idx].clamp_min(1e-4)  # [K]
                    ppo_sigma_next = _traj["sigma_next"][:, s_idx]  # [K]
                    sigma_b = sigma.view(k, *([1] * (x0.dim() - 1)))
                    model_ts = sigma.view(k, 1).to(dtype=dit_dtype)
                else:
                    t_idx = random.randint(0, sigmas.shape[1] - 1)
                    sigma = sigmas[:, t_idx].clamp_min(1e-4)  # [K], in [0,1]
                    sigma_b = sigma.view(k, *([1] * (x0.dim() - 1)))
                    noise = torch.randn_like(x0)
                    # A scalar per-sample sigma noises ALL latent tokens (no per-token denoise mask). This
                    # is exact for pure text-to-video/-AV rollouts (no image/keyframe conditioning -> the
                    # denoise_mask is all-ones), which is what the RL generate path produces. Conditioned
                    # rollouts (clean conditioning tokens that must keep sigma=0) are NOT supported here.
                    xt = (1.0 - sigma_b) * x0 + sigma_b * noise  # rectified-flow noising (call_dit:2861)
                    model_ts = sigma.view(k, 1).to(dtype=dit_dtype)  # transformer expects [0,1] sigma

                # AV: re-noise the audio latent with the SAME sampled sigma (AV training shares the
                # sampled timestep across modalities unless --independent_audio_timestep, which the
                # RL loop does not use). audio_timestep is [K,1], matching do_inference's AV forward.
                xt_audio = None
                sigma_audio_b = None
                ppo_audio_action = None
                ppo_audio_x0_gen = None
                if is_av:
                    sigma_audio_b = sigma.view(k, *([1] * (audio_x0.dim() - 1)))
                    if _ppo:
                        # Replay the audio trajectory step aligned to the SAME denoise index s_idx + sigma.
                        xt_audio = _traj["audio_x_t"][:, s_idx].to(device=device, dtype=dit_dtype)  # [K,C,T,F]
                        ppo_audio_action = _traj["audio_x_next"][:, s_idx].to(device=device, dtype=dit_dtype)
                        ppo_audio_x0_gen = _traj["audio_x0_gen"][:, s_idx].to(device=device, dtype=dit_dtype)
                    else:
                        audio_noise = torch.randn_like(audio_x0)
                        xt_audio = (1.0 - sigma_audio_b) * audio_x0 + sigma_audio_b * audio_noise

                def _forward():
                    """Plain video forward -> video velocity prediction [K,C,F,H,W]."""
                    fa, fk = self._net.prepare_forward_inputs(
                        transformer,
                        args,
                        model_input=xt.to(dit_dtype),
                        model_timesteps=model_ts,
                        text_embeds=v_ctx,
                        text_mask=v_mask,
                        frame_rate=frame_rate,
                        transformer_options={},
                    )
                    with accelerator.autocast():  # fp32 LoRA x bf16 base needs autocast (like call_dit)
                        out = transformer(*fa, **fk)
                    return out[0] if isinstance(out, (list, tuple)) else out  # video output

                def _forward_av():
                    """AV forward -> (video_velocity [K,C,F,H,W], audio_velocity [K,C,T,F]).

                    Reuses prepare_forward_inputs (the supervised AV forward contract): model_input is
                    the [video, audio] list and audio_timestep is supplied, so the AV transformer returns
                    [video_pred, audio_pred] (do not hand-roll AV inputs — same path as call_dit).
                    """
                    fa, fk = self._net.prepare_forward_inputs(
                        transformer,
                        args,
                        model_input=[xt.to(dit_dtype), xt_audio.to(dit_dtype)],
                        model_timesteps=model_ts,
                        text_embeds=v_ctx,  # full [video|audio] concat; wrapper splits it
                        text_mask=v_mask,
                        frame_rate=frame_rate,
                        audio_timestep=model_ts,
                        transformer_options={},
                    )
                    with accelerator.autocast():
                        out = transformer(*fa, **fk)
                    if not isinstance(out, (list, tuple)) or len(out) != 2:
                        raise ValueError(
                            f"AV forward expected [video_pred, audio_pred], got {type(out).__name__} "
                            f"len={len(out) if isinstance(out, (list, tuple)) else 'n/a'}"
                        )
                    return out[0], out[1]  # (video, audio) output ordering

                # --- block-swap order: old + ref FIRST (inference), then default (training) ---
                if is_av:
                    if blocks_to_swap > 0:
                        unwrapped.switch_block_swap_for_inference()
                    with torch.no_grad():
                        if blocks_to_swap > 0:
                            unwrapped.prepare_block_swap_before_forward()
                        net.set_enabled(False)  # ref = bare base (DoRA-safe gate)
                        ref_v, ref_a = _forward_av()
                        ref_x0 = to_denoised(xt, ref_v, sigma_b).detach()
                        ref_a_x0 = to_denoised(xt_audio, ref_a, sigma_audio_b).detach()
                        net.set_enabled(True)
                        if blocks_to_swap > 0:
                            unwrapped.prepare_block_swap_before_forward()
                        with ema.swapped():  # old = EMA weights
                            old_v, old_a = _forward_av()
                            old_x0 = to_denoised(xt, old_v, sigma_b).detach()
                            old_a_x0 = to_denoised(xt_audio, old_a, sigma_audio_b).detach()
                    if blocks_to_swap > 0:
                        unwrapped.switch_block_swap_for_training()
                        unwrapped.prepare_block_swap_before_forward()
                    fwd_v, fwd_a = _forward_av()  # default (grad)
                    fwd_x0 = to_denoised(xt, fwd_v, sigma_b)
                    fwd_a_x0 = to_denoised(xt_audio, fwd_a, sigma_audio_b)

                    # audio branch is added only when the audio advantage carries a real signal. If no
                    # audio-routed reward scored (audio rewards 0.0 -> zero-variance group -> adv_audio
                    # == 0), training it would regress the audio LoRA toward `old` on a null signal; skip.
                    if _ppo:
                        # AV authentic PPO: exact per-step ratio per live modality at the cached actions.
                        states = {}
                        if _video_live:
                            states["video"] = {
                                "x_t": _flatten_to_tokens(xt),
                                "action": _flatten_to_tokens(ppo_action),
                                "x0_fwd": _flatten_to_tokens(fwd_x0),
                                "x0_old": _flatten_to_tokens(ppo_x0_gen),
                                "x0_ref": _flatten_to_tokens(ref_x0),
                                "sigma": sigma,
                                "sigma_next": ppo_sigma_next,
                                "adv": adv,
                            }
                        if _audio_live:
                            states["audio"] = {
                                "x_t": _flatten_audio_to_tokens(xt_audio),
                                "action": _flatten_audio_to_tokens(ppo_audio_action),
                                "x0_fwd": _flatten_audio_to_tokens(fwd_a_x0),
                                "x0_old": _flatten_audio_to_tokens(ppo_audio_x0_gen),
                                "x0_ref": _flatten_audio_to_tokens(ref_a_x0),
                                "sigma": sigma,
                                "sigma_next": ppo_sigma_next,
                                "adv": adv_audio,
                            }
                    else:
                        states = {}
                        if _video_live:
                            adv_v_per_token = adv.view(k, 1).expand(k, _flatten_to_tokens(x0).shape[1])
                            states["video"] = {
                                "fwd": _flatten_to_tokens(fwd_x0),
                                "old": _flatten_to_tokens(old_x0),
                                "ref": _flatten_to_tokens(ref_x0),
                                "x0": _flatten_to_tokens(x0),
                                "adv": adv_v_per_token,
                            }
                        if _audio_live:
                            adv_a_per_token = adv_audio.view(k, 1).expand(k, _flatten_audio_to_tokens(audio_x0).shape[1])
                            states["audio"] = {
                                "fwd": _flatten_audio_to_tokens(fwd_a_x0),
                                "old": _flatten_audio_to_tokens(old_a_x0),
                                "ref": _flatten_audio_to_tokens(ref_a_x0),
                                "x0": _flatten_audio_to_tokens(audio_x0),
                                "adv": adv_a_per_token,
                            }
                    if not _audio_live and not _warned_audio_null[0]:
                        logger.warning(
                            "AV RL: audio advantage is all-zero (no active audio reward / zero-variance) -> "
                            "skipping the audio branch (it would otherwise train audio on a null signal). "
                            "Wire an audio reward + waveform decode to train the audio modality."
                        )
                        _warned_audio_null[0] = True
                    if not _video_live and not _warned_video_null[0]:
                        logger.warning(
                            "AV RL: video advantage is all-zero (no active video-routed reward / "
                            "zero-variance) -> skipping the video branch for such groups."
                        )
                        _warned_video_null[0] = True
                else:
                    if blocks_to_swap > 0:
                        unwrapped.switch_block_swap_for_inference()
                    with torch.no_grad():
                        if blocks_to_swap > 0:
                            unwrapped.prepare_block_swap_before_forward()
                        net.set_enabled(False)  # ref = bare base (DoRA-safe gate)
                        ref_x0 = to_denoised(xt, _forward(), sigma_b).detach()
                        net.set_enabled(True)
                        if blocks_to_swap > 0:
                            unwrapped.prepare_block_swap_before_forward()
                        with ema.swapped():  # old = EMA weights
                            old_x0 = to_denoised(xt, _forward(), sigma_b).detach()
                    if blocks_to_swap > 0:
                        unwrapped.switch_block_swap_for_training()
                        unwrapped.prepare_block_swap_before_forward()
                    fwd_x0 = to_denoised(xt, _forward(), sigma_b)  # default (grad)

                    if _ppo:
                        # PPO: behavior policy pi_old is the cached gen-time x0. x_t/action/x0 share the
                        # token layout; the ratio is built by ltx2_rl_sde.step_log_ratio (compute_ppo_loss).
                        states = {
                            "video": {
                                "x_t": _flatten_to_tokens(xt),
                                "action": _flatten_to_tokens(ppo_action),
                                "x0_fwd": _flatten_to_tokens(fwd_x0),
                                "x0_old": _flatten_to_tokens(ppo_x0_gen),
                                "x0_ref": _flatten_to_tokens(ref_x0),
                                "sigma": sigma,
                                "sigma_next": ppo_sigma_next,
                                "adv": adv,
                            }
                        }
                    else:
                        adv_per_token = adv.view(k, 1).expand(k, _flatten_to_tokens(x0).shape[1])
                        states = {
                            "video": {
                                "fwd": _flatten_to_tokens(fwd_x0),
                                "old": _flatten_to_tokens(old_x0),
                                "ref": _flatten_to_tokens(ref_x0),
                                "x0": _flatten_to_tokens(x0),
                                "adv": adv_per_token,
                            }
                        }

                loss, info = compute_rl_objective(rl_loss, states, args=args, modality_weights=modality_weights)
                accelerator.backward(loss)
                if args.max_grad_norm:
                    accelerator.clip_grad_norm_(net.trainable_lora_params(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                _post = {"loss": float(loss.detach()), "policy": float(info["policy"]), "kl": float(info["kl"])}
                if is_av:
                    _post["pol_v"] = float(info.get("policy_video", 0.0))
                    _post["pol_a"] = float(info.get("policy_audio", 0.0))
                progress.set_postfix(**_post)
                if tb_writer is not None:
                    for _k, _v in _post.items():
                        tb_writer.add_scalar(f"rl/{_k}", _v, global_step)
                    tb_writer.add_scalar("rl/lr", lr_scheduler.get_last_lr()[0], global_step)

        progress.close()
        if tb_writer is not None:
            tb_writer.close()
        if _log_vram:
            torch.cuda.synchronize(device)
            logger.info(
                "VRAM[Phase-B NFT step PEAK over %d steps]: max_allocated=%.2f GB max_reserved=%.2f GB",
                global_step,
                torch.cuda.max_memory_allocated(device) / 1e9,
                torch.cuda.max_memory_reserved(device) / 1e9,
            )
        if accelerator.is_main_process:
            out_dir = args.output_dir
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"{args.output_name or 'ltx2_rl_lora'}.safetensors")
            net.save_weights(save_path, torch.float16, None)
            logger.info("Saved RL LoRA to %s (global_step=%d)", save_path, global_step)


def rl_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--rl_rollout_cache", type=str, default=None, help="rollout cache dir (Phase A output; offline mode)")
    parser.add_argument(
        "--rl_log_vram",
        action="store_true",
        help="log peak GPU memory of the Phase-B NFT training step in isolation (resets the peak counter before the loop)",
    )
    parser.add_argument("--nft_beta_mix", type=float, default=1.0, help="NFT pos/neg mix coefficient (config.beta)")
    parser.add_argument(
        "--nft_kl_beta", type=float, default=1e-4, help="KL coefficient (config.train.beta) — distinct from beta_mix"
    )
    parser.add_argument("--nft_adv_clip_max", type=float, default=5.0)
    parser.add_argument(
        "--nft_video_modality_weight", type=float, default=1.0, help="AV mode: weight of the video NFT branch in compute_nft_loss"
    )
    parser.add_argument(
        "--nft_audio_modality_weight", type=float, default=1.0, help="AV mode: weight of the audio NFT branch in compute_nft_loss"
    )
    # --- update rule selector: NFT (default) or a pluggable non-NFT objective ---
    parser.add_argument(
        "--rl_loss",
        type=str,
        default="nft",
        choices=list(RL_LOSS_CHOICES),
        help="RL update rule: nft (default, negative-aware) | rwr (advantage-weighted regression) | "
        "dpo (Diffusion-DPO on each group's best/worst) | ppo (PPO-clip surrogate on the x0 Gaussian). "
        "All share the rollout/reward/advantage machinery; only the final loss differs.",
    )
    parser.add_argument("--rwr_temperature", type=float, default=1.0, help="rwr: softmax temperature over group advantages")
    parser.add_argument(
        "--dpo_beta", type=float, default=5.0, help="dpo: Diffusion-DPO beta (scale-invariant; preference sharpness)"
    )
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2, help="ppo: PPO clip epsilon")
    parser.add_argument(
        "--rl_sde_sampler",
        action="store_true",
        help="online mode: sample rollouts with the stochastic SDE sampler and cache the trajectory.",
    )
    parser.add_argument(
        "--rl_sde_eta",
        type=float,
        default=1.0,
        help="ppo: SDE per-step noise level in [0,1] (std=sigma_next*eta); must match Phase A.",
    )
    parser.add_argument("--rl_timesteps_per_sample", type=int, default=1, help="random timesteps trained per cached sample")
    parser.add_argument("--rl_max_steps", type=int, default=0, help="0 = ~one pass over the cache (fixed-behavior-policy bound)")
    parser.add_argument(
        "--rl_decay_type",
        type=int,
        default=1,
        help="EMA decay schedule for an online `old` EMA; INERT in the offline pipeline (old stays "
        "frozen per round, advanced only by regenerating the cache).",
    )
    # online mode (generate rollouts inline instead of reading a cache; VRAM-flat via sequential offload)
    parser.add_argument("--rl_online", action="store_true", help="generate rollouts inline each round instead of reading a cache")
    parser.add_argument("--rl_prompts", type=str, default=None, help="prompts file (online mode)")
    parser.add_argument(
        "--reward_fn",
        type=str,
        default="iqa_quality:1.0,anti_noise:0.1",
        help="reward spec (online mode)",
    )
    parser.add_argument(
        "--reward_args",
        type=str,
        nargs="*",
        default=None,
        help="key=value args passed to each reward's setup() (online mode; e.g. checkpoint_path=... for hpsv3)",
    )
    parser.add_argument(
        "--reward_plugins",
        type=str,
        nargs="*",
        default=None,
        help="paths to custom-reward .py files (imported before --reward_fn is parsed, so their names are usable)",
    )
    parser.add_argument("--rl_group_size", type=int, default=8, help="K samples per prompt (online mode)")
    parser.add_argument("--sample_width", type=int, default=768)
    parser.add_argument("--sample_height", type=int, default=512)
    parser.add_argument("--sample_frames", type=int, default=49)
    parser.add_argument("--sample_cfg", type=float, default=1.0)
    parser.add_argument("--sample_steps", type=int, default=20, help="denoising steps per rollout sample (online mode)")
    parser.add_argument(
        "--rl_dump_cache", type=str, default=None, help="online mode: also write the inline rollouts here (equivalence harness)"
    )
    return parser


def main() -> None:
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)
    parser = rl_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    # LTX-2 single checkpoint serves as DiT + VAE (mirrors ltx2_train_slider.py:1650-1658)
    if getattr(args, "ltx2_checkpoint", None) is not None:
        if getattr(args, "dit", None) != args.ltx2_checkpoint:
            args.dit = args.ltx2_checkpoint
        if getattr(args, "vae", None) != args.ltx2_checkpoint:
            args.vae = args.ltx2_checkpoint
    if args.rl_max_steps == 0:
        args.rl_max_steps = 10**9  # resolved against cache size in train()
    LTX2RLTrainer().train(args)


if __name__ == "__main__":
    main()
