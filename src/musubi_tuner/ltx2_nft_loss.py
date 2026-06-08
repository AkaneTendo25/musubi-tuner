"""NFT (Negative-aware Fine-Tuning) policy loss for LTX-2 RL.

For each modality the loss regresses the policy's denoised-x0 prediction toward an implicit
positive target (reinforce) or away from it toward a negative target (suppress), gated by the
group-relative advantage:

    pos = beta_mix*fwd + (1-beta_mix)*old        # used DIRECTLY as the x0 target (not xt - sigma*pred)
    neg = (1+beta_mix)*old - beta_mix*fwd
    wp  = |pos - x0|.mean(non-batch, keepdim).clip(1e-5)   # fp64 per-sample normalizers
    wn  = |neg - x0|.mean(non-batch, keepdim).clip(1e-5)
    r   = clamp((adv/adv_clip_max)/2 + 0.5, 0, 1)          # advantage -> [0,1] gate
    pos_pt = ((pos - x0)^2 / wp).mean(-1) ; neg_pt = ((neg - x0)^2 / wn).mean(-1)
    policy_pt = r*pos_pt + (1-r)*neg_pt
    policy_m  = (policy_pt*attn_w).sum(1)/attn_w.sum(1) / beta_mix * adv_clip_max
    kl_m      = ((fwd - ref)^2).mean(-1).mean(1)
    loss = policy.mean() + kl_beta * kl.mean()

CRITICAL: ``beta_mix`` (the pos/neg mix, default 1.0) and ``kl_beta`` (the KL coefficient,
default 1e-4) are DIFFERENT scalars. Tensors are the token layout ``[B, T, C]``;
the RL loop flattens the latent to tokens before calling this. ``wp``/``wn`` are computed in
float64 (chunked per modality by construction).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

_DEFAULT_MODALITY_WEIGHTS = {"video": 1.0, "audio": 1.0}


def compute_nft_loss(
    modality_states: Dict[str, Dict[str, torch.Tensor]],
    *,
    beta_mix: float = 1.0,
    kl_beta: float = 1e-4,
    adv_clip_max: float = 5.0,
    modality_weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute the NFT policy+KL loss.

    ``modality_states[m]`` holds tensors ``fwd``, ``old``, ``ref``, ``x0`` of shape ``[B, T, C]``
    (``fwd`` carries grad; the rest are detached), ``adv`` of shape ``[B, T]`` (per-token advantage),
    and optional ``attn_w`` ``[B, T]`` (defaults to ones). Returns ``(loss, info)``.
    """
    if not modality_states:
        raise ValueError("modality_states is empty")
    if beta_mix == 0:
        raise ValueError("beta_mix must be non-zero (it divides the policy term)")

    weights = dict(_DEFAULT_MODALITY_WEIGHTS if modality_weights is None else modality_weights)
    weight_sum = sum(weights.get(m, 1.0) for m in modality_states)
    if weight_sum <= 0:
        raise ValueError("sum of modality weights must be > 0")

    modality_policy: Dict[str, torch.Tensor] = {}
    modality_kl: Dict[str, torch.Tensor] = {}

    # Promote bf16/fp16 state tensors to fp32 for the squared-error + KL math; the RL loop passes
    # bf16, which truncates the tiny kl_beta(1e-4)-scaled KL signal. Tensors that
    # are already fp32+ are left untouched (do NOT downcast fp64). ``.float()`` is differentiable so a
    # grad-carrying tensor keeps its graph (autograd casts the gradient back on backward).
    def _calc(t):
        return t.float() if t is not None and t.dtype in (torch.bfloat16, torch.float16) else t

    for modality, state in modality_states.items():
        fwd = _calc(state["fwd"])
        old = _calc(state["old"])
        ref = _calc(state["ref"])
        x0 = _calc(state["x0"])
        adv = _calc(state["adv"])
        attn_w = _calc(state.get("attn_w"))

        pos = beta_mix * fwd + (1.0 - beta_mix) * old
        neg = (1.0 + beta_mix) * old - beta_mix * fwd

        reduce_dims = tuple(range(1, x0.ndim))
        wp = (pos.double() - x0.double()).abs().mean(dim=reduce_dims, keepdim=True).clip(min=1e-5)
        wn = (neg.double() - x0.double()).abs().mean(dim=reduce_dims, keepdim=True).clip(min=1e-5)

        adv_clamped = torch.clamp(adv, -adv_clip_max, adv_clip_max)
        r_token = torch.clamp((adv_clamped / adv_clip_max) / 2.0 + 0.5, 0.0, 1.0)

        pos_per_token = ((pos - x0) ** 2 / wp).mean(dim=-1)
        neg_per_token = ((neg - x0) ** 2 / wn).mean(dim=-1)
        policy_per_token = r_token * pos_per_token + (1.0 - r_token) * neg_per_token

        if attn_w is None:
            attn_w = torch.ones_like(policy_per_token)
        policy_m = (policy_per_token * attn_w).sum(dim=1) / attn_w.sum(dim=1) / beta_mix * adv_clip_max

        kl_per_token = ((fwd - ref) ** 2).mean(dim=-1)
        kl_m = kl_per_token.mean(dim=1)

        modality_policy[modality] = policy_m
        modality_kl[modality] = kl_m

    policy = sum(weights.get(m, 1.0) * modality_policy[m] for m in modality_states) / weight_sum
    kl = sum(weights.get(m, 1.0) * modality_kl[m] for m in modality_states) / weight_sum
    loss = policy.mean() + kl_beta * kl.mean()

    info = {
        "policy": policy.mean().detach(),
        "kl": kl.mean().detach(),
        **{f"policy_{m}": modality_policy[m].mean().detach() for m in modality_states},
        **{f"kl_{m}": modality_kl[m].mean().detach() for m in modality_states},
    }
    return loss, info
