"""Pure training helpers for the YuE2 trainer (CPU-testable, no trainer state).

Per-item planning (prefix choice, NAR window and context layout, AR ids and targets), derived per-item RNG, timestep
sampling, chunked AR cross-entropy / KL, the flow loss and the LoRA-dropout switch used by no-grad passes.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from musubi_tuner.yue2.yue2_protocol import (
    ABC_START,
    CONTEXT,
    EOD,
    MUSIC_END,
    build_negative_prefix,
    build_prefix,
    codec_to_ids,
    max_nar_window,
)

NAR_CONTEXTS = ("codes", "text_only")
TEXT_ONLY_ROPE_MODES = ("full", "compact")
AR_CE_TARGETS = ("codec", "codec_abc")
COT_CHOICES = ("auto", "off", "melody", "full")
ABC_MODE_NAMES = {0: None, 1: "melody", 2: "full"}


@dataclass
class ItemPlan:
    prefix: list[int]  # AR prefix (negative prefix when the caption was dropped)
    nar_prefix: list[int]  # positive prefix the NAR context is built from
    cot_used: str
    score_used: bool
    dropped_caption: bool
    window_start: int
    window_frames: int
    ctx_ids: list[int]  # NAR context prefill ids
    kv_visible: Optional[int]  # NAR sees K/V of ctx_ids[:kv_visible] (None: all)
    rope_offset: int  # first NAR position
    ar_ids: Optional[list[int]]  # AR CE ids (None: no AR loss for this item)
    ar_target_start: int
    ar_complete: bool  # ar_ids end with MUSIC_END
    ar_skipped_reason: Optional[str]


def item_rng(seed: int, global_step: int, micro_index: int, rank: int, item_index: int) -> random.Random:
    """Planning RNG derived from the item's coordinates: two fresh runs plan identically without extra state."""
    key = f"yue2:{seed}:{global_step}:{micro_index}:{rank}:{item_index}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def select_window(total: int, requested: int, prefix_len: int, rng: Optional[random.Random], training: bool) -> tuple[int, int]:
    """``(start, frames)`` of the NAR window: ``requested`` frames (0 = whole item) capped by ``max_nar_window``;
    random start in training, centred otherwise."""
    if total < 1:
        raise ValueError("YuE2 item has no frames")
    frames = total if requested <= 0 else min(requested, total)
    frames = min(frames, max_nar_window(prefix_len))
    span = total - frames
    if span <= 0:
        return 0, frames
    start = rng.randint(0, span) if training else span // 2
    return start, frames


def validation_windows(total: int, prefix_len: int, requested: int) -> tuple[int, int]:
    """Centre window used by the validator."""
    return select_window(total, requested, prefix_len, None, training=False)


def _want_cot(cot: str, abc_mode: Optional[str], has_abc: bool) -> str:
    if cot == "off" or not has_abc:
        return "off"
    if cot == "auto":
        return abc_mode or "off"
    return cot


def plan_item(
    *,
    texts: dict[str, Sequence[int]],
    negatives: dict[str, Sequence[int]],
    abc: Sequence[int],
    abc_mode: Optional[str],
    codes: Optional[Sequence[int]],
    total_frames: int,
    seg: Sequence[int],
    cot: str,
    abc_dropout: float,
    caption_dropout: float,
    nar_caption_dropout: float,
    window_frames: int,
    nar_context: str,
    codec_dropout: float,
    text_only_rope: str,
    ar_max_tokens: int,
    ar_ce_targets: str,
    train_ar: bool,
    rng: random.Random,
    training: bool = True,
) -> ItemPlan:
    """Plan one item. ``seg = (start_frame, song_frames, truncated)`` in audio-file coordinates.

    Draw order on ``rng`` is fixed: abc dropout, caption dropout, NAR caption dropout, window, codec dropout (codes).
    Validation (``training=False``) uses the score when present, never drops, and takes the centre window.
    """
    if nar_context not in NAR_CONTEXTS:
        raise ValueError(f"nar_context must be one of {NAR_CONTEXTS}, got {nar_context!r}")
    if ar_ce_targets not in AR_CE_TARGETS:
        raise ValueError(f"ar_ce_targets must be one of {AR_CE_TARGETS}, got {ar_ce_targets!r}")
    abc = [int(t) for t in abc]
    has_abc = len(abc) > 0
    want = _want_cot(cot, abc_mode if has_abc else None, has_abc)
    # all dropout draws are taken unconditionally, so the draw sequence does not depend on the item
    u_abc, u_caption, u_nar_caption = (rng.random(), rng.random(), rng.random()) if training else (1.0, 1.0, 1.0)

    score_used = want != "off" and u_abc >= abc_dropout
    cot_used = want if score_used else "off"
    score = abc if score_used else None
    nar_prefix = build_prefix(texts[cot_used], cot_used, score)

    dropped_caption = u_caption < caption_dropout
    prefix = build_negative_prefix(negatives[cot_used], cot_used, score) if dropped_caption else nar_prefix
    if u_nar_caption < nar_caption_dropout:
        nar_prefix = build_negative_prefix(negatives[cot_used], cot_used, score)

    start, frames = select_window(total_frames, window_frames, len(nar_prefix), rng, training)

    if nar_context == "codes":
        if codes is None:
            raise ValueError("--nar_context codes needs semantic codes in the latent cache (cache with --semantic_head)")
        ctx_ids = nar_prefix + codec_to_ids([int(c) for c in codes[start : start + frames]]) + [MUSIC_END]
        rope_offset = len(ctx_ids)
        kv_visible = None
        if training and rng.random() < codec_dropout:
            kv_visible = len(nar_prefix)
    else:
        ctx_ids = list(nar_prefix)
        kv_visible = None
        rope_offset = len(nar_prefix) + frames + 1 if text_only_rope == "full" else len(nar_prefix)
    if len(nar_prefix) + 2 * frames + 3 > CONTEXT:
        raise ValueError(f"YuE2 NAR sequence exceeds the {CONTEXT}-token context")

    ar_ids, ar_target_start, ar_complete, skipped = None, 0, False, None
    if train_ar:
        start_frame, song_frames, truncated = (int(v) for v in seg)
        if start_frame != 0:
            skipped = "mid_song_segment"
        elif codes is None:
            raise ValueError("training the AR branch needs semantic codes in the latent cache (cache with --semantic_head)")
        else:
            ar_ids, ar_target_start, ar_complete = ar_sequence(
                prefix, codes, total_frames, song_frames, bool(truncated), ar_max_tokens, ar_ce_targets
            )

    return ItemPlan(
        prefix=prefix,
        nar_prefix=nar_prefix,
        cot_used=cot_used,
        score_used=score_used,
        dropped_caption=dropped_caption,
        window_start=start,
        window_frames=frames,
        ctx_ids=ctx_ids,
        kv_visible=kv_visible,
        rope_offset=rope_offset,
        ar_ids=ar_ids,
        ar_target_start=ar_target_start,
        ar_complete=ar_complete,
        ar_skipped_reason=skipped,
    )


def ar_sequence(
    prefix: Sequence[int],
    codes: Sequence[int],
    total_frames: int,
    song_frames: int,
    truncated: bool,
    ar_max_tokens: int,
    ar_ce_targets: str,
) -> tuple[list[int], int, bool]:
    """``(ar_ids, target_start, complete)`` for an item that starts at song frame 0.

    ``MUSIC_END`` (the generation stop token) is appended only when the ids reach the true, untruncated song end.
    ``codec_abc`` targets start after ``ABC_START`` and fall back to ``codec`` when the prefix has none.
    """
    prefix = list(prefix)
    count = total_frames if ar_max_tokens <= 0 else min(total_frames, ar_max_tokens)
    complete = count == total_frames and total_frames == song_frames and not truncated
    ids = prefix + codec_to_ids([int(c) for c in codes[:count]]) + ([MUSIC_END] if complete else [])
    if len(ids) > CONTEXT:
        raise ValueError(f"YuE2 AR sequence of {len(ids)} tokens exceeds the {CONTEXT}-token context; set --ar_max_tokens")
    if ar_ce_targets == "codec_abc" and ABC_START in prefix:
        target_start = prefix.index(ABC_START) + 1
    else:
        target_start = len(prefix)
    return ids, target_start, complete


def ar_targets(ar_ids: Sequence[int], target_start: int, valid_len: int) -> tuple[slice, torch.Tensor]:
    """Hidden rows ``[target_start-1, valid_len-1)`` predict ``ar_ids[target_start:valid_len]``."""
    if not 0 < target_start < valid_len <= len(ar_ids):
        raise ValueError(f"invalid AR target range: start {target_start}, valid {valid_len}, ids {len(ar_ids)}")
    labels = torch.tensor(list(ar_ids[target_start:valid_len]), dtype=torch.long)
    return slice(target_start - 1, valid_len - 1), labels


def pad_right(ids: Sequence[int], multiple: int, pad_id: int = EOD) -> tuple[torch.Tensor, int]:
    """``(LongTensor[1, L'], L)`` with ``L'`` rounded up to ``multiple`` (exact under causal attention)."""
    ids = list(ids)
    n = len(ids)
    if multiple and multiple > 1:
        padded = -(-n // multiple) * multiple
        ids = ids + [pad_id] * (min(padded, CONTEXT) - n)
    return torch.tensor(ids, dtype=torch.long)[None], n


def sample_t(trainer, args, batch_size: int, pool, device) -> torch.Tensor:
    """Flow times ``t [B]`` in [0, 1] (t=1 is noise), fp32.

    ``beta``: Beta(a, b) clipped to ``[min_timestep, max_timestep] / 1000`` (default [0.02, 0.98]); every other method
    goes through the base ``sample_timesteps`` (``pool`` = the dataset's bucketed uniform draws or None). Then
    ``--first_timestep_chance`` forces t = 1 (``max_timestep`` if set) per item.
    """
    if args.timestep_sampling == "beta":
        a, b = parse_beta_ab(args.beta_timestep_ab)
        lo = (args.min_timestep if args.min_timestep is not None else 20) / 1000.0
        hi = (args.max_timestep if args.max_timestep is not None else 980) / 1000.0
        t = torch.distributions.Beta(torch.tensor(float(a)), torch.tensor(float(b))).sample((batch_size,))
        t = t.clamp(lo, hi).to(device)
    else:
        dummy = torch.zeros(batch_size, 1, 1, 1)
        t = trainer.sample_timesteps(args, batch_size, pool, dummy, device)
    t = t.float()
    chance = float(getattr(args, "first_timestep_chance", 0.0) or 0.0)
    if chance > 0:
        first = (args.max_timestep / 1000.0) if args.max_timestep is not None else 1.0
        mask = torch.rand(batch_size, device=t.device) < chance
        t = torch.where(mask, torch.full_like(t, first), t)
    return t


def parse_beta_ab(value) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        a, b = value
    else:
        parts = str(value).split(",")
        if len(parts) != 2:
            raise ValueError(f"--beta_timestep_ab must be 'a,b', got {value!r}")
        a, b = parts
    a, b = float(a), float(b)
    if a <= 0 or b <= 0:
        raise ValueError(f"--beta_timestep_ab values must be positive, got {a},{b}")
    return a, b


def _block_loss_sums(
    lm_head: nn.Module, h: torch.Tensor, y: torch.Tensor, h_base: Optional[torch.Tensor]
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Summed CE and KL(base || adapted) for one block of rows; full-vocabulary logits are materialised only here, one block at a time."""
    logits = lm_head(h).float()
    # nll(log_softmax) is F.cross_entropy spelled out: ignore_index -100 rows add 0, bad labels raise
    ce = F.nll_loss(F.log_softmax(logits, dim=-1), y, reduction="sum")
    if h_base is None:
        return ce, None
    with torch.no_grad():
        log_p = F.log_softmax(lm_head(h_base).float(), dim=-1)
    # sum over every (row, vocab) of p_base * (log p_base - log p_adapted)
    kl = (log_p.exp() * (log_p - F.log_softmax(logits, dim=-1))).sum()
    return ce, kl


def chunked_ce_kl(
    lm_head: nn.Module, hidden: torch.Tensor, labels: torch.Tensor, base_hidden: Optional[torch.Tensor], chunk: int = 512
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mean next-token CE over the ``N`` rows of ``hidden`` and, if ``base_hidden`` is given, mean KL(base || adapted).

    Rows go through the head in blocks of ``chunk`` (at least 1), so at most ``[chunk, V]`` fp32 logits are alive at
    a time. When a block needs grad it is checkpointed: backward recomputes its logits from the ``[chunk, H]`` input.
    The base side never gets grad. Returns fp32 scalars ``(ce, kl)``; ``kl`` is None without ``base_hidden``.
    """
    n = hidden.shape[0]
    if n == 0:
        raise ValueError("chunked_ce_kl needs at least one target row")
    step = max(1, int(chunk))
    block_fn = functools.partial(_block_loss_sums, lm_head)
    with_kl = base_hidden is not None
    base_blocks = torch.split(base_hidden, step) if with_kl else itertools.repeat(None)

    ce = torch.zeros((), dtype=torch.float32, device=hidden.device)
    kl = torch.zeros((), dtype=torch.float32, device=hidden.device) if with_kl else None
    for h, y, h_base in zip(torch.split(hidden, step), torch.split(labels.to(hidden.device), step), base_blocks):
        if h_base is not None:
            h_base = h_base.to(h.device)
        if torch.is_grad_enabled() and h.requires_grad:
            ce_part, kl_part = torch.utils.checkpoint.checkpoint(block_fn, h, y, h_base, use_reentrant=False)
        else:
            ce_part, kl_part = block_fn(h, y, h_base)
        ce = ce + ce_part
        if with_kl:
            kl = kl + kl_part
    return ce / n, (kl / n if with_kl else None)


def flow_loss(v_pred: torch.Tensor, eps: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Mean MSE of the velocity against ``eps - z`` over the content frames (START/END are never in ``v_pred``)."""
    return F.mse_loss(v_pred.float(), (eps - z).float())


def _modules_with_training_flag(network) -> list[nn.Module]:
    if network is None:
        return []
    return list(network.modules())


@contextlib.contextmanager
def lora_eval(network):
    """Every LoRA/Diff module of ``network`` in eval mode (dropouts off) inside the block, flags restored after."""
    modules = _modules_with_training_flag(network)
    flags = [m.training for m in modules]
    try:
        for m in modules:
            m.training = False
        yield
    finally:
        for m, flag in zip(modules, flags):
            m.training = flag


@contextlib.contextmanager
def optimizer_eval(optimizer):
    """``optimizer.eval()`` / ``train()`` around the block for optimizers that have them (schedule-free)."""
    has = optimizer is not None and callable(getattr(optimizer, "eval", None)) and callable(getattr(optimizer, "train", None))
    if has:
        optimizer.eval()
    try:
        yield
    finally:
        if has:
            optimizer.train()


def abc_mode_name(value) -> Optional[str]:
    """Cached ``yue2_abc_mode`` id (0/1/2) -> None / "melody" / "full"."""
    return ABC_MODE_NAMES[int(value)]
