"""Alternative (non-NFT) RL update rules for LTX-2, pluggable via ``--rl_loss``.

``nft``/``rwr``/``dpo`` consume the SAME per-modality ``states`` dict that
``ltx2_nft_loss.compute_nft_loss`` does, so the rollout / reward-zoo / GRPO-advantage machinery is
shared and only the final loss differs:

    states[m] = {
        "fwd", "old", "ref", "x0": [B, T, C],   # denoised-x0 predictions + the cached clean x0
        "adv": [B, T],                            # per-token advantage (expanded uniformly per sample)
        "attn_w": [B, T] (optional),
    }

``fwd`` carries grad; ``old``/``ref``/``x0`` are detached (computed under no_grad in the train loop).
``ppo`` instead carries a per-step trajectory state (see ``compute_ppo_loss``).

Objectives
----------
- ``nft``  : negative-aware fine-tuning (the default; lives in ``ltx2_nft_loss.py``). Reinforces
             good samples toward their own x0 AND pushes away from bad ones.
- ``rwr``  : advantage-weighted regression — NFT's positive branch only. Softmax(adv/T) weights the
             regression of each sample's prediction toward its own clean x0 (+ a KL term). Good
             samples are pulled in; bad ones are merely down-weighted (not pushed away).
- ``dpo``  : Diffusion-DPO over the best/worst sample of each group (ranked by advantage). The policy
             should denoise the *winner* better than the frozen ``ref`` does, relative to the *loser*
             (pairwise logistic loss). Preference-based; uses the reward only to rank.
- ``ppo``  : PPO-clip policy gradient with the exact per-step SDE transition probability at the
             sampled action (DDPO/DPOK). Requires a Phase-A ``--rl_sde_sampler`` trajectory cache; its
             ``states`` carry the trajectory step (see ``compute_ppo_loss``).

Each returns ``(loss, info)`` exactly like ``compute_nft_loss``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from musubi_tuner.ltx2_nft_loss import compute_nft_loss
from musubi_tuner.ltx2_rl_sde import step_log_ratio

_DEFAULT_MODALITY_WEIGHTS = {"video": 1.0, "audio": 1.0}


def _calc(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Promote bf16/fp16 to fp32 for the loss math (match the NFT kernel); leave fp32+ untouched."""
    return t.float() if t is not None and t.dtype in (torch.bfloat16, torch.float16) else t


def _unpack(state: Dict[str, torch.Tensor]):
    """-> (fwd, old, ref, x0 [B,T,C], adv_s [B] per-sample, reduce_dims). adv is averaged over tokens
    (it was expanded uniformly per sample, so this just recovers the per-sample scalar)."""
    fwd, old, ref, x0 = _calc(state["fwd"]), _calc(state["old"]), _calc(state["ref"]), _calc(state["x0"])
    adv = _calc(state["adv"])
    adv_s = adv.mean(dim=1) if adv.ndim >= 2 else adv
    reduce = tuple(range(1, fwd.ndim))
    return fwd, old, ref, x0, adv_s, reduce


def _finish(modality_policy, modality_kl, weights, modality_states, *, kl_beta) -> Tuple[torch.Tensor, Dict[str, Any]]:
    weight_sum = sum(weights.get(m, 1.0) for m in modality_states)
    if weight_sum <= 0:
        raise ValueError("sum of modality weights must be > 0")
    policy = sum(weights.get(m, 1.0) * modality_policy[m] for m in modality_states) / weight_sum
    kl = sum(weights.get(m, 1.0) * modality_kl[m] for m in modality_states) / weight_sum
    loss = policy + kl_beta * kl
    info = {
        "policy": policy.detach(),
        "kl": kl.detach(),
        **{f"policy_{m}": modality_policy[m].detach() for m in modality_states},
        **{f"kl_{m}": modality_kl[m].detach() for m in modality_states},
    }
    return loss, info


def compute_rwr_loss(
    modality_states: Dict[str, Dict[str, torch.Tensor]],
    *,
    kl_beta: float = 1e-4,
    rwr_temperature: float = 1.0,
    modality_weights: Optional[Dict[str, float]] = None,
    **_ignored,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Advantage-weighted regression toward each sample's own clean x0 (NFT's positive branch only)."""
    if not modality_states:
        raise ValueError("modality_states is empty")
    weights = dict(_DEFAULT_MODALITY_WEIGHTS if modality_weights is None else modality_weights)
    mp, mk = {}, {}
    for m, state in modality_states.items():
        fwd, _old, ref, x0, adv_s, reduce = _unpack(state)
        recon = ((fwd - x0) ** 2).mean(dim=reduce)  # [B] policy reconstruction error (grad via fwd)
        # softmax weights over the group are a constant function of the (cached) advantages -> detach;
        # the gradient flows only through `recon`, pulling high-advantage samples toward their x0.
        w = torch.softmax(adv_s / max(float(rwr_temperature), 1e-6), dim=0).detach()
        # Normalize by the detached mean recon so the policy term is O(1) regardless of the latent /
        # reward scale (keeps the effective LR consistent across rewards and comparable to NFT/PPO).
        # `scale` is detached, so the direction (pull high-advantage samples toward x0) is unchanged.
        scale = recon.mean().detach().clamp_min(1e-8)
        mp[m] = (w * (recon / scale)).sum()
        mk[m] = ((fwd - ref) ** 2).mean(dim=reduce).mean()
    return _finish(mp, mk, weights, modality_states, kl_beta=kl_beta)


def compute_dpo_loss(
    modality_states: Dict[str, Dict[str, torch.Tensor]],
    *,
    dpo_beta: float = 5.0,
    modality_weights: Optional[Dict[str, float]] = None,
    **_ignored,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Diffusion-DPO on the best vs worst sample of each group (ranked by advantage)."""
    if not modality_states:
        raise ValueError("modality_states is empty")
    weights = dict(_DEFAULT_MODALITY_WEIGHTS if modality_weights is None else modality_weights)
    weight_sum = sum(weights.get(m, 1.0) for m in modality_states)
    if weight_sum <= 0:
        raise ValueError("sum of modality weights must be > 0")
    modality_loss: Dict[str, torch.Tensor] = {}
    for m, state in modality_states.items():
        fwd, _old, ref, x0, adv_s, reduce = _unpack(state)
        # No winner/loser pair: K<2, or all advantages equal (a zero-variance/flagged group -> adv all
        # 0, or genuine ties). argmax==argmin would give margin 0 -> a zero-gradient wasted step; skip
        # it explicitly (mirrors the audio-branch null-advantage guard in ltx2_train_rl.py).
        if adv_s.shape[0] < 2 or float(adv_s.max() - adv_s.min()) == 0.0:
            modality_loss[m] = (fwd * 0.0).sum()
            continue
        m_policy = ((fwd - x0) ** 2).mean(dim=reduce)  # [B] (grad via fwd)
        m_ref = ((ref - x0) ** 2).mean(dim=reduce)  # [B] (detached ref)
        margin = m_ref - m_policy  # >0 -> policy denoises this sample better than ref
        w_idx = int(torch.argmax(adv_s))
        l_idx = int(torch.argmin(adv_s))
        # Normalize the margin by the (detached) reference MSE scale so dpo_beta is scale-invariant:
        # raw latent-MSE margins are tiny, so a large fixed beta saturates logsigmoid and zeros the
        # gradient. want margin(winner) > margin(loser): minimize -log sigmoid(beta*Δmargin/scale).
        ref_scale = m_ref.mean().detach().clamp_min(1e-8)
        logit = float(dpo_beta) * (margin[w_idx] - margin[l_idx]) / ref_scale
        modality_loss[m] = -F.logsigmoid(logit)
    loss = sum(weights.get(m, 1.0) * modality_loss[m] for m in modality_states) / weight_sum
    info = {
        "policy": loss.detach(),
        "kl": torch.zeros((), device=loss.device, dtype=loss.dtype),
        **{f"policy_{m}": modality_loss[m].detach() for m in modality_states},
    }
    return loss, info


def compute_ppo_loss(
    modality_states: Dict[str, Dict[str, torch.Tensor]],
    *,
    eta: float,
    ppo_clip_eps: float = 0.2,
    kl_beta: float = 1e-4,
    adv_clip_max: float = 5.0,
    modality_weights: Optional[Dict[str, float]] = None,
    **_ignored,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """PPO-clip with the per-step Gaussian transition probability at the sampled denoising action
    (DDPO/DPOK).

    Each modality state carries one cached trajectory step plus the fresh forwards:
        ``x_t`` (state), ``action`` (the sampled ``x_next``), ``x0_fwd`` (policy x0, grad),
        ``x0_old`` (behavior-policy x0, detached), ``x0_ref`` (frozen ref x0, detached, optional),
        ``sigma``, ``sigma_next``, ``adv``.
    Only ``x0_fwd`` carries grad, so the clip acts on the transition probability ``pi(action | x_t)``
    (see ``ltx2_rl_sde``). Returns ``(loss, info)`` like the other objectives.
    """
    if not modality_states:
        raise ValueError("modality_states is empty")
    weights = dict(_DEFAULT_MODALITY_WEIGHTS if modality_weights is None else modality_weights)
    mp, mk = {}, {}
    for m, state in modality_states.items():
        x_t = _calc(state["x_t"])
        action = _calc(state["action"])
        x0_fwd = _calc(state["x0_fwd"])
        x0_old = _calc(state["x0_old"])
        adv = _calc(state["adv"])
        adv_s = adv.mean(dim=1) if adv.ndim >= 2 else adv  # per-sample scalar (steps share the rollout adv)
        log_ratio = step_log_ratio(action, x0_fwd, x0_old, x_t, state["sigma"], state["sigma_next"], eta)
        ratio = torch.exp(log_ratio)
        adv_c = adv_s.clamp(-float(adv_clip_max), float(adv_clip_max))
        unclipped = ratio * adv_c
        clipped = torch.clamp(ratio, 1.0 - float(ppo_clip_eps), 1.0 + float(ppo_clip_eps)) * adv_c
        mp[m] = -torch.min(unclipped, clipped).mean()
        ref = state.get("x0_ref")
        if ref is not None:
            reduce = tuple(range(1, x0_fwd.ndim))
            mk[m] = ((x0_fwd - _calc(ref)) ** 2).mean(dim=reduce).mean()
        else:
            mk[m] = torch.zeros((), device=x0_fwd.device, dtype=x0_fwd.dtype)
    return _finish(mp, mk, weights, modality_states, kl_beta=kl_beta)


# name -> (callable, kwarg-builder reading from args). NFT is included for a single dispatch point.
def compute_rl_objective(
    name: str,
    modality_states: Dict[str, Dict[str, torch.Tensor]],
    *,
    args,
    modality_weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Dispatch to the selected RL update rule, pulling its hyperparameters from ``args``."""
    name = (name or "nft").lower()
    if name == "nft":
        return compute_nft_loss(
            modality_states,
            beta_mix=float(getattr(args, "nft_beta_mix", 1.0)),
            kl_beta=float(getattr(args, "nft_kl_beta", 1e-4)),
            adv_clip_max=float(getattr(args, "nft_adv_clip_max", 5.0)),
            modality_weights=modality_weights,
        )
    if name == "rwr":
        return compute_rwr_loss(
            modality_states,
            kl_beta=float(getattr(args, "nft_kl_beta", 1e-4)),
            rwr_temperature=float(getattr(args, "rwr_temperature", 1.0)),
            modality_weights=modality_weights,
        )
    if name == "dpo":
        return compute_dpo_loss(
            modality_states,
            dpo_beta=float(getattr(args, "dpo_beta", 5.0)),
            modality_weights=modality_weights,
        )
    if name == "ppo":
        # PPO requires the SDE-sampled rollout trajectory in the cache (Phase A --rl_sde_sampler) so
        # Phase B can build the per-step states.
        return compute_ppo_loss(
            modality_states,
            eta=float(getattr(args, "rl_sde_eta", 1.0)),
            ppo_clip_eps=float(getattr(args, "ppo_clip_eps", 0.2)),
            kl_beta=float(getattr(args, "nft_kl_beta", 1e-4)),
            adv_clip_max=float(getattr(args, "nft_adv_clip_max", 5.0)),
            modality_weights=modality_weights,
        )
    raise ValueError(f"unknown --rl_loss {name!r} (choices: nft, rwr, dpo, ppo)")


RL_LOSS_CHOICES = ("nft", "rwr", "dpo", "ppo")
