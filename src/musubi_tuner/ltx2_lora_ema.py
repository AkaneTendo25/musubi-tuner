"""The NFT ``old`` policy as an EMA of the trainable LoRA.

The ``old`` policy that generates rollouts is an exponential moving average of the trainable
(``default``) adapter — held as plain tensors OUTSIDE the optimizer (so it is never trained) and
optionally CPU-offloaded (streamed per ``old`` forward) to keep the VRAM delta to ~tens of MB.

Snapshot/swap cover EVERY trainable adapter tensor (lora_down/up, DoRA magnitude vectors,
adaptive-rank lambdas) via ``network.trainable_lora_params()`` — not a hardcoded subset. The
``old`` forward uses ``network.swapped_weights`` (exception-safe: restores on any raise).

Decay schedule ``return_decay``: decay_type 1 (default) ramps ``min(step*0.001, 0.5)``.
"""

from __future__ import annotations

from typing import List

import torch

from musubi_tuner.ltx2_rollout_cache import compute_snapshot_hash


def return_decay(step: int, decay_type: int = 1) -> float:
    """EMA decay schedule. Higher decay => `old` lags `default` more."""
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"unsupported decay_type={decay_type}")
    if step < flat:
        return 0.0
    return float(min((step - flat) * uprate, uphold))


class LoraEMA:
    """EMA of the trainable LoRA params = the NFT ``old`` policy.

    Held on ``device`` ("cpu" to offload). ``sync_with_model`` snapshots the current ``default``
    (call at init and after each cache regeneration). ``step`` EMA-updates ``old`` toward
    ``default``. ``swapped()`` runs a forward with the ``old`` weights. ``snapshot_hash`` matches
    the value the rollout cache stores so Phase B can assert the fixed-behavior-policy invariant.

    NOTE (offline pipeline): ``ltx2_train_rl.py`` intentionally does NOT call ``step()`` during a
    Phase-B run — ``old`` stays frozen equal to the warm-start snapshot (the fixed behavior policy
    that generated the cache), which is exactly what ``snapshot_hash``/``assert_snapshot`` require.
    ``old`` advances only ACROSS rounds, by regenerating the cache from the updated policy (Phase A).
    The per-step ``step()``/``return_decay``/``decay_type`` machinery is kept for an online EMA
    variant and is inert in the current offline driver (so ``--rl_decay_type`` has no effect there).
    """

    def __init__(self, network, *, decay_type: int = 1, device: str | torch.device = "cpu") -> None:
        self.network = network
        self.decay_type = decay_type
        self.device = torch.device(device)
        self._ema: List[torch.Tensor] = []

    def sync_with_model(self) -> None:
        self._ema = [p.detach().to(self.device).clone() for p in self.network.trainable_lora_params()]

    @torch.no_grad()
    def step(self, global_step: int) -> float:
        """EMA-update: ``old = decay*old + (1-decay)*default``. Returns the decay used."""
        params = self.network.trainable_lora_params()
        if not self._ema:
            self.sync_with_model()
            return 0.0
        if len(self._ema) != len(params):
            raise ValueError(f"EMA has {len(self._ema)} tensors but network has {len(params)} trainable params")
        decay = return_decay(global_step, self.decay_type)
        for ema, param in zip(self._ema, params):
            ema.mul_(decay).add_(param.detach().to(self.device), alpha=1.0 - decay)
        return decay

    def swapped(self):
        """Context manager running a forward with the ``old`` (EMA) weights.

        Streams EMA tensors to each param's device first, then delegates to the network's
        exception-safe ``swapped_weights`` (which restores ``default`` on exit even on error).
        """
        params = self.network.trainable_lora_params()
        if len(self._ema) != len(params):
            raise ValueError("EMA not initialized / size mismatch; call sync_with_model() first")
        ema_on_device = [ema.to(param.device, dtype=param.dtype) for ema, param in zip(self._ema, params)]
        return self.network.swapped_weights(ema_on_device)

    def snapshot_hash(self) -> str:
        """Content hash of the current ``old`` weights (stored in the rollout cache)."""
        if not self._ema:
            raise ValueError("EMA not initialized; call sync_with_model() first")
        return compute_snapshot_hash(self._ema)

    def state_dict(self) -> dict:
        return {"ema": [t.cpu() for t in self._ema], "decay_type": self.decay_type}

    def load_state_dict(self, sd: dict) -> None:
        self._ema = [t.to(self.device) for t in sd["ema"]]
        self.decay_type = sd.get("decay_type", self.decay_type)
