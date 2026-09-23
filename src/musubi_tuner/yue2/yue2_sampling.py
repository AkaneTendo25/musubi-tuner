"""YuE2 inference: AR token sampling (ABC plan and semantic codes), the NAR midpoint ODE, and song rendering.

Adapted from the official YuE2 inference code ("upstream": ``yue2_infer`` 0.1.5, see THIRD_PARTY_NOTICES.md):
``sampling.py`` (``window_penalty``, ``distribution``, the eager ``generate_tokens`` loop), ``nar.py`` (``song_chunks``,
``CachedNAR.solve``, ``synthesize``) and ``pipeline.py`` (plan -> semantic -> synthesize -> decode). Seeds, dtypes and
the order of random draws are kept, so a seeded run reproduces the official token stream and latents on the same
kernels.

Nothing here enters ``torch.inference_mode``: the sampler also runs inside training, where inference tensors
would poison lazily allocated offloader buffers. Every entry point takes ``grad_ctx`` (default ``torch.no_grad``).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import torch

from musubi_tuner.yue2.yue2_model import yue2_t_embed_input
from musubi_tuner.yue2.yue2_protocol import (
    ABC_DEFAULTS,
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    DEFAULT_INSTRUMENTAL_LYRICS,
    EOD,
    FRAME_RATE,
    LATENT_DIM,
    MUSIC_END,
    ODE_STEPS,
    SAMPLE_RATE,
    SEMANTIC_DEFAULTS,
    SamplingParams,
    build_negative_prefix,
    build_prefix,
    chunk_ranges,
    default_cfg,
    generation_budget,
    ids_to_codec,
    negative_text_ids,
    normalize_prompt_fields,
    text_ids,
)

logger = logging.getLogger(__name__)

GradCtx = Callable[[], object]
PHASES = ("abc", "semantic")
NAR_CONTEXTS = ("codes", "text_only")
TEXT_ONLY_ROPE = ("full", "compact")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_dtype(value: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    dtypes = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp32": torch.float32, "float32": torch.float32}
    if value not in dtypes:
        raise ValueError(f"ODE state dtype must be bf16 or fp32, got {value!r}")
    return dtypes[value]


# region AR sampling


def window_penalty(logits: torch.Tensor, recent_ids: Sequence[int], penalty: float) -> torch.Tensor:
    """Sign-aware repetition penalty over a window: ``penalty ** count`` divides positive and multiplies negative
    logits (upstream ``sampling.window_penalty``)."""
    if penalty == 1.0 or len(recent_ids) == 0:
        return logits
    recent = torch.as_tensor(list(recent_ids), dtype=torch.long, device=logits.device).reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
    alpha = penalty**freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def distribution(
    logits: torch.Tensor, params: SamplingParams, history: Sequence[int], step: int, phase: str, legacy_off: bool = False
) -> torch.Tensor:
    """Masked, penalised, temperature-scaled, top-k then top-p filtered scores ``[1, V]`` (upstream semantics).

    ``abc`` allows the ordinary text vocabulary plus ``ABC_END``; ``semantic`` allows the codec range plus
    ``MUSIC_END``; the end token is masked before ``min_tokens``. ``legacy_off`` (cot=off) keeps the model dtype
    (bf16) for the whole computation and protects the top three tokens from top-p, as the release does.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores = scores + allowed
    if step < params.min_tokens:
        scores[..., end] = -torch.inf
    scores = window_penalty(scores, history[-params.penalty_window :], params.repetition_penalty)
    if params.temperature == 0:
        return scores
    if params.temperature != 1:
        scores = scores / params.temperature
    threshold = scores.topk(min(params.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if params.top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > params.top_p
        removed[..., : 3 if legacy_off else 1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


def generate_tokens(
    model,
    prefix: Sequence[int],
    params: SamplingParams,
    seed: int,
    phase: str,
    negative: Optional[Sequence[int]] = None,
    cfg_scale: float = 1.0,
    legacy_off: bool = False,
    on_token: Optional[Callable[[str, int], None]] = None,
    grad_ctx: GradCtx = torch.no_grad,
    timing: Optional[dict] = None,
) -> tuple[list[int], bool]:
    """Sample up to ``params.max_tokens`` tokens after ``prefix`` with an eager KV cache (two caches with CFG).

    Returns ``(ids, truncated)``: the generated ids without the end token, and ``True`` when the budget ran out
    before the end token. ``timing`` (optional dict) receives the upstream timing fields. The seed resets per call,
    as upstream does for each phase. The budget is not capped here (upstream raises); callers cap it with
    ``yue2_protocol.generation_budget``.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    prefix = [int(t) for t in prefix]
    if len(prefix) + params.max_tokens > CONTEXT:
        raise ValueError(f"YuE2 prefix ({len(prefix)}) + generation budget ({params.max_tokens}) exceeds {CONTEXT}")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None:
        negative = [int(t) for t in negative]
        if len(negative) + params.max_tokens > CONTEXT:
            raise ValueError(f"YuE2 negative prefix ({len(negative)}) + generation budget exceeds {CONTEXT}")

    with grad_ctx():
        device = torch.device(model.device)
        rng_device = device if device.type in ("cpu", "cuda") else torch.device("cpu")
        generator = torch.Generator(device=rng_device).manual_seed(int(seed))

        def prefill(ids):
            cache = model.new_kv_cache(len(ids) + params.max_tokens)
            return model.ar_prefill_into_cache(torch.tensor([ids], dtype=torch.long, device=device), cache), cache

        _synchronize(device)
        start = time.perf_counter()
        conditional, positive_cache = prefill(prefix)
        unconditional = negative_cache = None
        if cfg_scale != 1.0:
            unconditional, negative_cache = prefill(negative)
        _synchronize(device)
        prefill_seconds = time.perf_counter() - start

        history: list[int] = []
        first = None
        eos = False
        end = ABC_END if phase == "abc" else MUSIC_END
        for step in range(params.max_tokens):
            # upstream keeps the CFG arithmetic in the model dtype before the upcast in distribution()
            logits = conditional if cfg_scale == 1.0 else unconditional + cfg_scale * (conditional - unconditional)
            scores = distribution(logits, params, history, step, phase, legacy_off)
            if params.temperature == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                next_id = torch.multinomial(scores.softmax(-1), 1, generator=generator)
            token = int(next_id.item())
            if first is None:
                first = time.perf_counter() - start
            if on_token is not None:
                on_token(phase, token)
            if token == end:
                eos = True
                break
            history.append(token)
            if step + 1 < params.max_tokens:
                conditional = model.ar_decode_step(next_id, positive_cache)
                if negative_cache is not None:
                    unconditional = model.ar_decode_step(next_id, negative_cache)
        _synchronize(device)
        seconds = time.perf_counter() - start
        del positive_cache, negative_cache

    if timing is not None:
        count = len(history) + int(eos)
        timing.update(
            seconds=seconds,
            prefill_seconds=prefill_seconds,
            ttft_seconds=first,
            output_tokens=count,
            content_tokens=len(history),
            output_tps=count / seconds if seconds > 0 else 0.0,
            prefix_tokens=len(prefix),
            cfg_branches=1 if cfg_scale == 1 else 2,
            execution="eager",
        )
    return history, not eos


# endregion

# region NAR ODE


def song_chunks(
    prefix: Sequence[int], codes: Sequence[int], seed: int, context: int = CONTEXT
) -> list[tuple[list[int], torch.Tensor]]:
    """``[(ctx_ids, noise[a:b])]`` per original chunk: ``ctx = prefix + (codes[a:b] + CODEC_OFFSET) + [MUSIC_END]``;
    the noise ``[T, 64]`` is drawn once per song on CPU in fp32 from ``seed`` and sliced (upstream ``song_chunks``)."""
    prefix = [int(t) for t in prefix]
    codes = [int(c) for c in codes]
    if not prefix or not codes:
        raise ValueError("YuE2 synthesis needs a nonempty prefix and nonempty codes")
    if min(prefix) < 0 or min(codes) < 0 or max(codes) >= CODEC_SIZE:
        raise ValueError("Token ids are outside their allowed vocabulary")
    ranges = chunk_ranges(len(codes), len(prefix), context)
    noise = song_noise(len(codes), seed)
    return [(prefix + [c + CODEC_OFFSET for c in codes[a:b]] + [MUSIC_END], noise[a:b]) for a, b in ranges]


def song_noise(frames: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((frames, LATENT_DIM), dtype=torch.float32, device="cpu", generator=generator)


def solve_chunk(
    model,
    ctx_ids: Sequence[int],
    noise: torch.Tensor,
    steps: int = ODE_STEPS,
    state_dtype: Union[str, torch.dtype] = torch.bfloat16,
    kv_visible: Optional[int] = None,
    rope_offset: Optional[int] = None,
    t_embed_mode: str = "bf16",
    grad_ctx: GradCtx = torch.no_grad,
    on_step: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """Explicit midpoint ODE from t=1 (noise) to t=0 over one chunk; returns CPU fp32 latents ``[T, 64]``.

    The AR context K/V (``kv_visible`` positions) is prefilled once; the NAR RoPE starts at ``rope_offset``
    (default ``len(ctx_ids)``). ``state_dtype`` bf16 keeps the upstream state rounding.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if noise.ndim != 2 or noise.shape[1] != LATENT_DIM or noise.shape[0] < 1:
        raise ValueError(f"expected nonempty noise [frames, {LATENT_DIM}], got {tuple(noise.shape)}")
    dtype = _state_dtype(state_dtype)
    with grad_ctx():
        device = torch.device(model.device)
        ids = torch.tensor([list(ctx_ids)], dtype=torch.long, device=device)
        kv = model.ar_forward(model.embed(ids), return_kv=True, kv_visible=kv_visible, return_hidden=False).kv
        offset = len(ctx_ids) if rope_offset is None else int(rope_offset)

        def velocity(x: torch.Tensor, t: float) -> torch.Tensor:
            t_embed = yue2_t_embed_input(torch.tensor([t], dtype=torch.float64), t_embed_mode).to(device)
            return model.nar_forward(x[None], t_embed, kv, offset)[0]

        state = noise.to(device=device, dtype=dtype)
        dt = 1.0 / steps
        for step in range(steps):
            t = 1.0 - step * dt
            mid = state - velocity(state, t) * (dt / 2)
            state = state - velocity(mid, t - dt / 2) * dt
            if on_step is not None:
                on_step(step + 1, steps)
        result = state.float().cpu()
    if not torch.isfinite(result).all():
        raise FloatingPointError("YuE2 flow matching produced non-finite latents")
    return result


def synthesize_latents(
    model,
    prefix: Sequence[int],
    codes: Sequence[int],
    seed: int,
    steps: int = ODE_STEPS,
    state_dtype: Union[str, torch.dtype] = torch.bfloat16,
    nar_context: str = "codes",
    text_only_rope: str = "full",
    t_embed_mode: str = "bf16",
    context: int = CONTEXT,
    grad_ctx: GradCtx = torch.no_grad,
    on_step: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """CPU fp32 latents ``[T, 64]`` for ``codes``, solving the original chunks serially (upstream ``synthesize``).

    ``nar_context="codes"`` (protocol): each chunk attends to ``prefix + codes[a:b] + MUSIC_END``.
    ``"text_only"``: only the prefix K/V are visible; the NAR RoPE starts at ``len(prefix) + W + 1`` (``full``,
    positions as if the codes were present) or at ``len(prefix)`` (``compact``).
    """
    if nar_context not in NAR_CONTEXTS:
        raise ValueError(f"nar_context must be one of {NAR_CONTEXTS}, got {nar_context!r}")
    if text_only_rope not in TEXT_ONLY_ROPE:
        raise ValueError(f"text_only_rope must be one of {TEXT_ONLY_ROPE}, got {text_only_rope!r}")
    chunks = song_chunks(prefix, codes, seed, context)
    total = len(chunks) * steps
    outputs = []
    for index, (ctx, noise) in enumerate(chunks):
        progress = None
        if on_step is not None:

            def progress(done, n, index=index):
                on_step(index * n + done, total)

        if nar_context == "codes":
            kv_visible, rope_offset = None, None
        else:
            kv_visible = len(prefix)
            rope_offset = len(prefix) + len(noise) + 1 if text_only_rope == "full" else len(prefix)
            ctx = list(prefix)
        outputs.append(
            solve_chunk(
                model,
                ctx,
                noise,
                steps,
                state_dtype,
                kv_visible=kv_visible,
                rope_offset=rope_offset,
                t_embed_mode=t_embed_mode,
                grad_ctx=grad_ctx,
                on_step=progress,
            )
        )
    return torch.cat(outputs, dim=0)


# endregion

# region songs


@dataclass
class YuE2SongRequest:
    """One song. ``text_ids``/``neg_ids`` (``[EOD] + ...`` text parts, as in the text cache) and ``abc_ids`` skip
    tokenization; ``mode="reconstruct"`` needs ``codes`` and skips the AR."""

    style: str = ""
    lyrics: str = ""
    cot: str = "full"
    abc: Optional[str] = None
    seed: int = 831001
    seconds: Optional[float] = None
    cfg_scale: Optional[float] = None
    semantic: SamplingParams = SEMANTIC_DEFAULTS
    abc_params: SamplingParams = ABC_DEFAULTS
    ode_steps: int = ODE_STEPS
    state_dtype: str = "bf16"
    mode: str = "render"
    codes: Optional[Sequence[int]] = None
    text_ids: Optional[Sequence[int]] = None
    neg_ids: Optional[Sequence[int]] = None
    abc_ids: Optional[Sequence[int]] = None
    nar_context: str = "codes"
    text_only_rope: str = "full"
    t_embed_mode: str = "bf16"
    instrumental_lyrics: str = DEFAULT_INSTRUMENTAL_LYRICS
    vae_core_frames: int = 1024
    vae_halo_frames: int = 16
    ar_resident: bool = False

    def __post_init__(self):
        if self.cot not in ("off", "melody", "full"):
            raise ValueError(f"cot must be off, melody or full, got {self.cot!r}")
        if self.mode not in ("render", "reconstruct"):
            raise ValueError(f"mode must be render or reconstruct, got {self.mode!r}")
        if self.mode == "reconstruct" and not self.codes:
            raise ValueError("reconstruct mode needs codes")
        if self.seconds is not None and self.seconds <= 0:
            raise ValueError(f"seconds must be positive, got {self.seconds}")


@dataclass
class YuE2SongResult:
    waveform: torch.Tensor  # [2, S] fp32 on CPU, clamped to [-1, 1]
    latents: torch.Tensor  # [T, 64] fp32 on CPU
    codes: list[int]
    prefix: list[int]
    abc_text: Optional[str] = None
    abc_ids: Optional[list[int]] = None
    negative: Optional[list[int]] = None
    cot: str = "off"
    cfg_scale: float = 1.0
    seed: int = 0
    sample_rate: int = SAMPLE_RATE
    truncated: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return self.waveform.shape[-1] / self.sample_rate


def _capped_params(params: SamplingParams, prefix_len: int, negative_len: int, seconds: Optional[float]) -> SamplingParams:
    requested = params.max_tokens
    if seconds is not None:
        requested = min(requested, max(1, round(seconds * FRAME_RATE)))
    budget = generation_budget(prefix_len, negative_len, requested)
    return dataclasses.replace(params, max_tokens=budget, min_tokens=min(params.min_tokens, budget))


def decode_latents(vae, latents: torch.Tensor, core_frames: int = 1024, halo_frames: int = 16) -> torch.Tensor:
    """``latents [T, 64]`` -> waveform ``[2, S]`` fp32 on CPU, clamped to [-1, 1] (exact tiled decode)."""
    z = latents.float().T.unsqueeze(0).contiguous()
    audio = vae.decode_tiled(z, core_frames=core_frames, halo_frames=halo_frames, output_device="cpu")
    if not torch.isfinite(audio).all():
        raise FloatingPointError("YuE2 VAE produced non-finite audio")
    return audio[0].float().clamp(-1, 1)


def _request_text(tokenizer, req: YuE2SongRequest, cot: str) -> tuple[list[int], list[int]]:
    if req.text_ids is not None and cot == req.cot:
        text = [int(t) for t in req.text_ids]
    else:
        if tokenizer is None:
            raise ValueError("YuE2 sampling needs a tokenizer or pre-tokenized text_ids")
        style, lyrics = normalize_prompt_fields(req.style, req.lyrics, req.instrumental_lyrics)
        text = text_ids(tokenizer, style, lyrics, cot)
    if req.neg_ids is not None and cot == req.cot:
        neg = [int(t) for t in req.neg_ids]
    elif tokenizer is not None:
        neg = negative_text_ids(tokenizer, cot)
    else:
        neg = None
    return text, neg


def _request_abc_ids(tokenizer, req: YuE2SongRequest) -> Optional[list[int]]:
    if req.abc_ids is not None:
        return [int(t) for t in req.abc_ids]
    if req.abc is not None and req.abc.strip():
        return tokenizer.encode(req.abc)
    return None


def _set_ar_resident(model, on: bool) -> None:
    if hasattr(model, "set_ar_resident"):
        model.set_ar_resident(on)


def reconstruct(
    model,
    vae,
    prefix: Sequence[int],
    codes: Sequence[int],
    seed: int,
    steps: int = ODE_STEPS,
    state_dtype: Union[str, torch.dtype] = torch.bfloat16,
    nar_context: str = "codes",
    text_only_rope: str = "full",
    t_embed_mode: str = "bf16",
    vae_core_frames: int = 1024,
    vae_halo_frames: int = 16,
    grad_ctx: GradCtx = torch.no_grad,
) -> YuE2SongResult:
    """Held-out (or given) codes -> NAR latents -> audio; isolates the NAR from the AR."""
    codes = [int(c) for c in codes]
    timings = {}
    with grad_ctx():
        t0 = time.perf_counter()
        latents = synthesize_latents(
            model, prefix, codes, seed, steps, state_dtype, nar_context, text_only_rope, t_embed_mode, grad_ctx=grad_ctx
        )
        _synchronize(torch.device(model.device))
        timings["nar_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        waveform = decode_latents(vae, latents, vae_core_frames, vae_halo_frames)
        timings["vae_seconds"] = time.perf_counter() - t0
    return YuE2SongResult(waveform, latents, codes, [int(t) for t in prefix], seed=int(seed), timings=timings)


def render_song(
    model,
    vae,
    tokenizer,
    request: YuE2SongRequest,
    grad_ctx: GradCtx = torch.no_grad,
    on_token: Optional[Callable[[str, int], None]] = None,
) -> YuE2SongResult:
    """One song end to end: ABC plan (cot melody/full without a score) -> semantic codes (CFG with the protocol
    negative prefix) -> NAR ODE -> VAE. ``request.mode == "reconstruct"`` uses ``request.codes`` and skips the AR.

    Generation budgets are capped to the context (``generation_budget``) instead of raising; a reduction is recorded
    in ``result.budget``. ``request.seconds`` limits the semantic budget to ``round(seconds * 25)`` codes.
    """
    req = request
    cot = req.cot
    started = time.perf_counter()
    timings: dict = {}
    truncated = {"abc": False, "semantic": False}
    budget: dict = {}
    with grad_ctx():
        abc_ids = _request_abc_ids(tokenizer, req)
        abc_text = req.abc
        if req.mode == "reconstruct" and cot != "off" and abc_ids is None:
            if req.text_ids is not None:
                raise ValueError(f"reconstruct with cot={cot} needs the ABC ids of the song (or cot=off)")
            logger.warning(f"YuE2 reconstruct: no ABC for cot={cot}; using cot=off for the NAR context")
            cot = "off"
        text, neg = _request_text(tokenizer, req, cot)

        if req.mode == "reconstruct":
            codes = [int(c) for c in req.codes]
            if req.seconds is not None:
                codes = codes[: max(1, round(req.seconds * FRAME_RATE))]
            prefix = build_prefix(text, cot, abc_ids)
            result = reconstruct(
                model,
                vae,
                prefix,
                codes,
                req.seed,
                req.ode_steps,
                req.state_dtype,
                req.nar_context,
                req.text_only_rope,
                req.t_embed_mode,
                req.vae_core_frames,
                req.vae_halo_frames,
                grad_ctx=grad_ctx,
            )
            result.cot, result.abc_ids, result.abc_text = cot, abc_ids, abc_text
            result.timings["e2e_seconds"] = time.perf_counter() - started
            return result

        if req.ar_resident:
            _set_ar_resident(model, True)
        try:
            if cot != "off" and abc_ids is None:
                plan_prefix = build_prefix(text, cot, None)
                params = _capped_params(req.abc_params, len(plan_prefix), 0, None)
                if params.max_tokens < req.abc_params.max_tokens:
                    budget["abc"] = {"requested": req.abc_params.max_tokens, "used": params.max_tokens}
                    logger.warning(f"YuE2 ABC budget reduced to {params.max_tokens} tokens by the context")
                timings["abc"] = {}
                abc_ids, truncated["abc"] = generate_tokens(
                    model, plan_prefix, params, req.seed, "abc", on_token=on_token, grad_ctx=grad_ctx, timing=timings["abc"]
                )
                abc_text = tokenizer.decode(abc_ids) if tokenizer is not None else None
                if truncated["abc"]:
                    logger.warning(f"YuE2 ABC plan hit its budget ({params.max_tokens} tokens) before ABC_END")

            prefix = build_prefix(text, cot, abc_ids)
            cfg = default_cfg(cot) if req.cfg_scale is None else float(req.cfg_scale)
            negative = None
            if cfg != 1:
                if neg is None:
                    raise ValueError("CFG needs negative text ids (tokenizer or neg_ids)")
                negative = build_negative_prefix(neg, cot, abc_ids)
            params = _capped_params(req.semantic, len(prefix), len(negative or []), req.seconds)
            limit = (
                req.semantic.max_tokens if req.seconds is None else min(req.semantic.max_tokens, round(req.seconds * FRAME_RATE))
            )
            if params.max_tokens < max(1, limit):
                budget["semantic"] = {"requested": limit, "used": params.max_tokens}
                logger.warning(f"YuE2 semantic budget reduced to {params.max_tokens} tokens by the context (prefix {len(prefix)})")
            timings["semantic"] = {}
            tokens, truncated["semantic"] = generate_tokens(
                model,
                prefix,
                params,
                req.seed,
                "semantic",
                negative=negative,
                cfg_scale=cfg,
                legacy_off=cot == "off",
                on_token=on_token,
                grad_ctx=grad_ctx,
                timing=timings["semantic"],
            )
        finally:
            if req.ar_resident:
                _set_ar_resident(model, False)
        if not tokens:
            raise ValueError("YuE2 AR produced no music tokens")
        codes = ids_to_codec(tokens)

        result = reconstruct(
            model,
            vae,
            prefix,
            codes,
            req.seed,
            req.ode_steps,
            req.state_dtype,
            req.nar_context,
            req.text_only_rope,
            req.t_embed_mode,
            req.vae_core_frames,
            req.vae_halo_frames,
            grad_ctx=grad_ctx,
        )
    timings.update(result.timings)
    timings["e2e_seconds"] = time.perf_counter() - started
    result.abc_text, result.abc_ids, result.negative = abc_text, abc_ids, negative
    result.cot, result.cfg_scale = cot, cfg
    result.truncated, result.budget, result.timings = truncated, budget, timings
    return result


# endregion

# region reconstruct sources


@dataclass
class YuE2ReconstructSource:
    codes: list[int]
    latents: Optional[torch.Tensor]  # [T, 64] fp32 ground truth
    text_ids: dict  # cot -> list[int] ([EOD] + ...), empty without a text cache
    neg_ids: dict
    abc_ids: Optional[list[int]]
    abc_mode: Optional[str]
    text_cache_path: Optional[str]


def text_cache_path_for(latent_cache_path: str) -> str:
    """``{key}_{pos}-{n}_yue2.safetensors`` -> ``{key}_yue2_te.safetensors`` (dataset/audio_dataset.py naming)."""
    directory, name = os.path.split(latent_cache_path)
    stem = name[: -len(".safetensors")] if name.endswith(".safetensors") else name
    tokens = stem.split("_")
    if len(tokens) < 3 or tokens[-1] != "yue2":
        raise ValueError(f"not a YuE2 latent cache file name: {name}")
    return os.path.join(directory, "_".join(tokens[:-2]) + "_yue2_te.safetensors")


def load_reconstruct_cache(latent_cache_path: str, text_cache_path: Optional[str] = None) -> YuE2ReconstructSource:
    """Codes (and ground-truth latents) of a YuE2 latent cache, plus the cached text/negative/ABC ids of its record
    when the text cache exists."""
    from safetensors.torch import load_file

    from musubi_tuner.dataset.cache_io import (
        YUE2_ABC_KEY,
        YUE2_ABC_MODE_IDS,
        YUE2_ABC_MODE_KEY,
        YUE2_CODES_KEY,
        YUE2_HAS_ABC_KEY,
        YUE2_NEG_KEY_FMT,
        YUE2_TEXT_KEY_FMT,
    )

    sd = load_file(latent_cache_path)
    if YUE2_CODES_KEY not in sd:
        raise ValueError(f"{latent_cache_path} has no semantic codes ({YUE2_CODES_KEY}); cache with a semantic head")
    codes = [int(c) for c in sd[YUE2_CODES_KEY].tolist()]
    latents = next((v.float() for k, v in sd.items() if k.startswith("latents_")), None)

    text_cache_path = text_cache_path or text_cache_path_for(latent_cache_path)
    texts, negs, abc_ids, abc_mode = {}, {}, None, None
    if os.path.exists(text_cache_path):
        te = load_file(text_cache_path)
        for cot in ("off", "melody", "full"):
            for keyfmt, target in ((YUE2_TEXT_KEY_FMT, texts), (YUE2_NEG_KEY_FMT, negs)):
                key = keyfmt.format(cot=cot)
                if key in te:
                    target[cot] = [int(t) for t in te[key].tolist()]
        if int(te.get(YUE2_HAS_ABC_KEY, torch.tensor(0))) and YUE2_ABC_KEY in te:
            abc_ids = [int(t) for t in te[YUE2_ABC_KEY].tolist()]
            mode_id = int(te.get(YUE2_ABC_MODE_KEY, torch.tensor(0)))
            abc_mode = {v: k for k, v in YUE2_ABC_MODE_IDS.items()}.get(mode_id)
    else:
        text_cache_path = None
    return YuE2ReconstructSource(codes, latents, texts, negs, abc_ids, abc_mode, text_cache_path)


# endregion
