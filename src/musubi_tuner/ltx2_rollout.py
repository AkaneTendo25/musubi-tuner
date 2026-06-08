"""Rollout generation + scoring orchestration for LTX-2 RL (shared by offline and online).

``generate_and_score_groups`` is the single code path both modes use:
  - offline (``ltx2_cache_rollouts``): call it, then write the groups to a rollout cache;
  - online  (``ltx2_train_rl --rl_online``): call it inline and feed the groups to the NFT step.

The model-dependent generation is injected as ``generate_fn`` so this orchestration (scoring +
GRPO advantage routing + cache-group assembly) is testable without a model. ``generate_fn`` does
the sampling/decoding (with sequential model offload — that is what keeps online VRAM-flat).

Advantage pipeline per group (one prompt, K samples): for each reward, GRPO group-normalize the
scores, multiply by the reward weight, then route by the reward's declared route
(video | audio | sync); ``sync`` adds to BOTH branches. The order is normalize -> weight -> route
with explicit per-plugin routes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import torch

from musubi_tuner.ltx2_rewards import PerPromptStatTracker, RewardStack

# A generate_fn maps (prompt, seeds) -> list of K per-sample dicts. Each dict carries whatever
# the rewards ``need`` (e.g. decoded media path, prompt, seed) plus the tensors to cache
# (e.g. "video_x0", "v_ctx", "v_mask", "positions", "sigmas"). In AV mode it also carries the
# audio NFT targets ("audio_x0", "a_ctx", "a_mask") which round-trip through the cache too.
GenerateFn = Callable[[str, List[int]], List[Dict[str, Any]]]


def compute_routed_advantages(
    prompt: str,
    reward_scores: Dict[str, List[float]],
    weights: Dict[str, float],
    route_map: Dict[str, str],
    *,
    std_eps: float = 1e-4,
    zero_var_threshold: float = 1e-6,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
    """Per-reward GRPO-normalize -> weight -> route. Returns ``({"video":adv, "audio":adv}, flagged)``.

    ``flagged`` maps reward_name -> [prompt] for zero-variance groups (excluded with advantage 0).
    """
    if not reward_scores:
        raise ValueError("reward_scores is empty")
    k = len(next(iter(reward_scores.values())))
    adv_video = torch.zeros(k, dtype=torch.float64)
    adv_audio = torch.zeros(k, dtype=torch.float64)
    flagged: Dict[str, List[str]] = {}

    for name, scores in reward_scores.items():
        if len(scores) != k:
            raise ValueError(f"reward '{name}' has {len(scores)} scores, expected {k}")
        route = route_map.get(name)
        if route not in ("video", "audio", "sync"):
            raise ValueError(f"reward '{name}' has no valid route ({route!r}); explicit route required")
        tracker = PerPromptStatTracker(std_eps=std_eps, zero_var_threshold=zero_var_threshold)
        adv_np, group_flagged = tracker.compute([prompt] * k, scores)
        weighted = torch.as_tensor(adv_np, dtype=torch.float64) * float(weights.get(name, 1.0))
        if route in ("video", "sync"):
            adv_video = adv_video + weighted
        if route in ("audio", "sync"):
            adv_audio = adv_audio + weighted
        if group_flagged:
            flagged[name] = group_flagged

    return {"video": adv_video, "audio": adv_audio}, flagged


# Decoded-media keys are consumed by the reward during scoring (Phase A) and must NOT be
# written to the cache (the decoded pixel video is ~230MB/sample at real resolution; the
# cache stores only the latent NFT targets in "video_x0"/"audio_x0"). Phase B never needs them.
_SCORING_ONLY_KEYS = frozenset({"video", "audio_waveform", "video_file", "audio_file"})

# Reward ``needs`` entries that require generation to decode pixel/audio media. A reward
# declaring any of these (e.g. HPSv3 needs {"video"}; clap needs {"audio_waveform"}) forces
# ``decode_video=True`` so the decoded media is available for scoring; otherwise generation
# skips the VAE decode.
MEDIA_NEEDS = frozenset({"video", "video_file", "audio_waveform", "audio_file"})


def stack_needs_media(reward_stack) -> bool:
    """True if any selected reward needs decoded media (so generate_fn must decode it)."""
    return any(bool(frozenset(r.needs) & MEDIA_NEEDS) for r in reward_stack._rewards.values())


def stack_media_needs(reward_stack) -> frozenset:
    """Union of the selected rewards' media ``needs`` (subset of ``MEDIA_NEEDS``).

    Tells generate_fn WHICH media to materialize: ``{"video"}`` (decoded tensor, e.g. HPSv3),
    ``{"video_file","audio_file"}`` (mp4 + wav on disk, e.g. av_align), ``{"audio_waveform"}``
    (decoded waveform tensor, e.g. clap). A superset of what ``stack_needs_media`` answers as a bool.
    """
    out: set = set()
    for r in reward_stack._rewards.values():
        out |= frozenset(r.needs) & MEDIA_NEEDS
    return frozenset(out)


def _stack_sample_tensors(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stack tensor-valued, same-keyed fields across the K samples into ``[K, ...]`` tensors.

    Scoring-only media keys (see ``_SCORING_ONLY_KEYS``) are excluded so the cache stays lean.
    All other tensor fields round-trip — including the AV NFT targets ("audio_x0", "a_ctx",
    "a_mask") emitted by the AV generate_fn, so the audio branch of Phase B can read them.
    """
    if not samples:
        return {}
    tensor_keys = [key for key, val in samples[0].items() if isinstance(val, torch.Tensor) and key not in _SCORING_ONLY_KEYS]
    stacked: Dict[str, torch.Tensor] = {}
    for key in tensor_keys:
        present = [isinstance(s.get(key), torch.Tensor) for s in samples]
        if not all(present):
            missing = [i for i, ok in enumerate(present) if not ok]
            raise ValueError(
                f"sample tensor key '{key}' is on sample 0 but missing on samples {missing} of this "
                f"group ({len(samples)} total). Silently dropping a partially-present key would corrupt "
                f"the group's modality set (e.g. lose 'audio_x0' -> a misleading Phase-B failure); RL "
                f"rollouts must be symmetric. Regenerate the group deterministically."
            )
        stacked[key] = torch.stack([s[key].detach().cpu() for s in samples], dim=0)
    return stacked


def generate_groups(
    prompts: List[str],
    *,
    group_size: int,
    seed_base: int,
    generate_fn: GenerateFn,
) -> List[Dict[str, Any]]:
    """Pass 1: generate K samples per prompt (model resident). No scoring.

    Returns one dict per prompt: ``{group_idx, prompt, seeds, samples}``. The caller may free the
    generation model before scoring (``score_groups``) so the DiT and reward models never co-reside.
    """
    groups: List[Dict[str, Any]] = []
    for group_idx, prompt in enumerate(prompts):
        seeds = [seed_base + group_idx * group_size + k for k in range(group_size)]
        samples = generate_fn(prompt, seeds)
        if len(samples) != group_size:
            raise ValueError(f"generate_fn returned {len(samples)} samples for group_size={group_size}")
        groups.append({"group_idx": group_idx, "prompt": prompt, "seeds": seeds, "samples": samples})
    return groups


def score_groups(
    groups: List[Dict[str, Any]],
    reward_stack: RewardStack,
    route_map: Dict[str, str],
    *,
    std_eps: float = 1e-4,
    zero_var_threshold: float = 1e-6,
) -> List[Dict[str, Any]]:
    """Pass 2: score every group's samples in ONE reward pass, then per-group GRPO advantages.

    Rewards are scored with one ``score_groups`` call so each reward model loads exactly once.
    Ordinary scalar rewards use the default flatten/slice fallback; pairwise rewards can compare
    samples within each prompt group directly. Advantages are computed per group from that group's
    K scores.
    Returns cache-ready groups (``group_idx, prompt, seeds, tensors, reward_scores, advantages,
    flagged``) with the scoring-only media dropped.
    """
    weights = dict(reward_stack.weights)
    grouped_scores = reward_stack.score_groups(groups) if groups else {}
    out: List[Dict[str, Any]] = []
    for group_pos, g in enumerate(groups):
        reward_scores = {name: scores_by_group[group_pos] for name, scores_by_group in grouped_scores.items()}
        advantages, flagged = compute_routed_advantages(
            g["prompt"], reward_scores, weights, route_map, std_eps=std_eps, zero_var_threshold=zero_var_threshold
        )
        out.append(
            {
                "group_idx": g["group_idx"],
                "prompt": g["prompt"],
                "seeds": g["seeds"],
                "tensors": _stack_sample_tensors(g["samples"]),
                "reward_scores": reward_scores,
                "advantages": advantages,
                "flagged": flagged,
            }
        )
    return out


def generate_and_score_groups(
    prompts: List[str],
    *,
    group_size: int,
    seed_base: int,
    generate_fn: GenerateFn,
    reward_stack: RewardStack,
    route_map: Dict[str, str],
    std_eps: float = 1e-4,
    zero_var_threshold: float = 1e-6,
) -> List[Dict[str, Any]]:
    """Generate K samples per prompt, score them, and assemble cache-ready groups (one combined call).

    Composition of ``generate_groups`` + ``score_groups`` for callers that do not need to free the
    generation model between the two phases (online ``--rl_online``). Offline Phase A calls the two
    separately so it can unload the DiT before loading the reward models (avoids VRAM co-residence).
    """
    groups = generate_groups(prompts, group_size=group_size, seed_base=seed_base, generate_fn=generate_fn)
    return score_groups(groups, reward_stack, route_map, std_eps=std_eps, zero_var_threshold=zero_var_threshold)
