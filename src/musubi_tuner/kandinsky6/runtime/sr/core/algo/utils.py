"""Utility functions for SR video generation and evaluation."""

from __future__ import annotations

from typing import Any, Literal, Callable

import torch

from musubi_tuner.kandinsky6.runtime.sr.core.components.model.vae_io import (
    cast_to_module_dtype,
    decode_latent_to_uint8,
    encode_pixels_to_latent,
)
from musubi_tuner.kandinsky6.runtime.sr.core.algo.cached_text_embs_utils import replicate_cached_embeds
from musubi_tuner.kandinsky6.runtime.sr.core.algo.train_utils import degrade_lq_latent, get_sparse_params


X0_T_CLAMP = 1e-5


@torch.no_grad()
def get_model_prediction(  # noqa: PLR0913
    dit: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    text_embeds: dict[str, torch.Tensor],
    null_text_embeds: dict[str, torch.Tensor],
    visual_cu_seqlens: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    null_text_cu_seqlens: torch.Tensor,
    visual_rope_pos: list[torch.Tensor],
    text_rope_pos: torch.Tensor,
    null_text_rope_pos: torch.Tensor,
    scale_factor: tuple[float, ...],
    guidance_weight: float,
    sparse_params: Any | None = None,
) -> torch.Tensor:
    """Compute classifier-free guidance model prediction.

    The returned tensor has the same semantics as the raw model output
    (velocity when ``prediction_target="velocity"``, clean ``x_0`` when
    ``prediction_target="x0"``).  The caller is responsible for converting
    the prediction to a velocity if needed for the Euler step.

    Args:
        dit: DiT model used for inference.
        x: Visual latent tensor to denoise.
        t: Timestep tensor broadcast over the batch.
        text_embeds: Dict with ``text_embeds`` and ``pooled_embed`` keys.
        null_text_embeds: Same structure as ``text_embeds`` but for null captions.
        visual_cu_seqlens: Cumulative sequence lengths for visual tokens.
        text_cu_seqlens: Cumulative sequence lengths for text tokens.
        null_text_cu_seqlens: Cumulative sequence lengths for null text tokens.
        visual_rope_pos: RoPE position indices for visual tokens.
        text_rope_pos: RoPE position indices for text tokens.
        null_text_rope_pos: RoPE position indices for null text tokens.
        scale_factor: Per-axis RoPE frequency scaling.
        guidance_weight: CFG strength (1 = no guidance).
        sparse_params: Optional sparse attention parameters.

    Returns:
        CFG-combined model prediction (velocity or ``x_0``).
    """
    model_time = (t * 1000).to(dtype=x.dtype)
    motion_score = torch.full((1,), 900.0, device=x.device, dtype=x.dtype)
    cond_pred = dit(
        x,
        text_embeds["text_embeds"],
        text_embeds["pooled_embed"],
        model_time,
        visual_cu_seqlens,
        text_cu_seqlens,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=scale_factor,
        sparse_params=sparse_params,
        motion_score=motion_score,
    )
    if guidance_weight == 1.0:
        return cond_pred
    uncond_pred = dit(
        x,
        null_text_embeds["text_embeds"],
        null_text_embeds["pooled_embed"],
        model_time,
        visual_cu_seqlens,
        null_text_cu_seqlens,
        visual_rope_pos,
        null_text_rope_pos,
        scale_factor=scale_factor,
        sparse_params=sparse_params,
        motion_score=motion_score,
    )
    return uncond_pred + guidance_weight * (cond_pred - uncond_pred)


@torch.no_grad()
def generate(  # noqa: PLR0913
    img: torch.Tensor,
    model: torch.nn.Module,
    device: str | int,
    num_steps: int,
    text_embeds: dict[str, torch.Tensor],
    null_text_embeds: dict[str, torch.Tensor],
    visual_cu_seqlens: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    null_text_cu_seqlens: torch.Tensor,
    visual_rope_pos: list[torch.Tensor],
    text_rope_pos: torch.Tensor,
    null_text_rope_pos: torch.Tensor,
    scale_factor: tuple[float, ...],
    guidance_weight: float,
    scheduler_scale: float,
    first_frames: torch.Tensor | None = None,
    tp_mesh: Any | None = None,
    visual_cond_scheme: str = "pretrain",
    start_timestep: float = 1.0,
    prediction_target: str = "velocity",
    channelcat_drop_threshold: float = 0.0,
    rfg_scale: float = 1.0,
    progress_callback: Callable[[int], Any] | None = None,
) -> torch.Tensor:
    """Run the full denoising loop from start_timestep to 0.

    When ``prediction_target="x0"`` the model predicts clean ``x_0`` instead
    of velocity.  The velocity for the Euler step is recovered as
    ``v = (x_t - x_0_pred) / clamp(t, min=X0_T_CLAMP)`` to avoid numerical
    instability as ``t → 0``.

    Args:
        img: Initial latent tensor ``[T, H, W, C]`` (or wider for instruct).
        model: DiT model.
        device: Target CUDA device.
        num_steps: Number of denoising steps.
        text_embeds: Conditional text embeddings.
        null_text_embeds: Unconditional text embeddings.
        visual_cu_seqlens: Cumulative visual sequence lengths.
        text_cu_seqlens: Cumulative text sequence lengths.
        null_text_cu_seqlens: Cumulative null-text sequence lengths.
        visual_rope_pos: RoPE position indices for visual tokens.
        text_rope_pos: RoPE position indices for text tokens.
        null_text_rope_pos: RoPE position indices for null text tokens.
        scale_factor: Per-axis RoPE frequency scaling.
        guidance_weight: CFG strength.
        scheduler_scale: Timestep scheduler warp factor.
        first_frames: Optional I2V conditioning first frames.
        tp_mesh: Tensor-parallel device mesh (optional).
        visual_cond_scheme: ``"pretrain"`` or ``"i2v"`` injection strategy.
        start_timestep: Starting timestep (``1.0`` = full noise).
        prediction_target: ``"velocity"`` or ``"x0"``.
        channelcat_drop_threshold: When ``> 0``, zero out channel-cat LQ
            channels (and mask) once the denoising timestep drops below this
            value.  ``0`` = disabled (default).
        rfg_scale: Reference-Free Guidance strength on the anchor.  Only
            applied when ``model.instruct_type == "hybrid_anchor"`` and
            ``rfg_scale != 1.0`` — in that case a second forward is run with
            the anchor and anchor_mask channels zeroed, and predictions blend
            as ``v_uncond + rfg_scale * (v_cond - v_uncond)``.  ``1.0``
            (default) skips the second forward entirely.
    Returns:
        Denoised latent tensor (same leading dims as ``img``).
    """
    img = img.to(device)
    visual_cu_seqlens = visual_cu_seqlens.to(device)
    text_cu_seqlens = text_cu_seqlens.to(device)
    null_text_cu_seqlens = null_text_cu_seqlens.to(device)
    visual_rope_pos = [position.to(device) for position in visual_rope_pos]
    text_rope_pos = text_rope_pos.to(device)
    null_text_rope_pos = null_text_rope_pos.to(device)
    text_embeds = {key: value.to(device) for key, value in text_embeds.items()}
    null_text_embeds = {key: value.to(device) for key, value in null_text_embeds.items()}
    sparse_params = get_sparse_params(model, img, visual_cu_seqlens)

    timesteps = torch.linspace(start_timestep, 0, num_steps, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    if tp_mesh:
        tp_rank = tp_mesh["tp"].get_local_rank()
        tp_world_size = tp_mesh["tp"].size()
        img = torch.chunk(img, tp_world_size, dim=1)[tp_rank]
        if first_frames is not None:
            first_frames = torch.chunk(first_frames, tp_world_size, dim=1)[tp_rank]

    if model.visual_cond and first_frames is not None:
        first_frames = first_frames.to(device=img.device, dtype=img.dtype)

    out_channels: int = 0
    # Save channel-cat LQ for channelcat_drop_threshold scheduling.
    lq_cond_channels: torch.Tensor | None = None
    if channelcat_drop_threshold > 0 and img.shape[-1] > model.in_visual_dim:  # type: ignore
        lq_cond_channels = img[..., model.in_visual_dim :].clone()

    for timestep, timestep_diff in list(zip(timesteps[:-1], torch.diff(timesteps), strict=False)):
        time = timestep.unsqueeze(0).repeat(visual_cu_seqlens.shape[0] - 1)
        # Apply channel-cat LQ schedule: restore LQ+mask when t >= threshold,
        # zero them (like "noise" branch input) when t drops below threshold.
        if lq_cond_channels is not None:
            if timestep.item() >= channelcat_drop_threshold:
                img[..., model.in_visual_dim :] = lq_cond_channels
            else:
                img[..., model.in_visual_dim :] = 0
        if model.visual_cond and img.shape[-1] == model.in_visual_dim:
            # image-to-video / first-frame conditioning
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros([*img.shape[:-1], 1], dtype=img.dtype, device=img.device)
            if first_frames is not None:
                if visual_cond_scheme == "pretrain":
                    # inject first_frames inside additional channels
                    visual_cond[visual_cu_seqlens[:-1]] = first_frames
                elif visual_cond_scheme == "i2v":
                    # inject first_frames straight into latents, while visual_cond is zero and unused
                    img[visual_cu_seqlens[:-1]] = first_frames
                else:
                    msg = f"unknown visual_cond_scheme={visual_cond_scheme}"
                    raise ValueError(msg)
                # in both cases mark injected first frames
                visual_cond_mask[visual_cu_seqlens[:-1]] = 1

            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img

        v_cond = get_model_prediction(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_cu_seqlens,
            text_cu_seqlens,
            null_text_cu_seqlens,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            scale_factor,
            guidance_weight,
            sparse_params=sparse_params,
        )

        if model.instruct_type == "hybrid_anchor" and rfg_scale != 1.0:
            # Reference-Free Guidance: second forward with anchor + anchor_mask
            # zeroed (the "anchor-dropped" regime seen during reference-dropout
            # training). Blend per SparkVSR §3.3.
            model_input_uncond = model_input.clone()
            model_input_uncond[..., model.in_visual_dim :] = 0
            v_uncond = get_model_prediction(
                model,
                model_input_uncond,
                time,
                text_embeds,
                null_text_embeds,
                visual_cu_seqlens,
                text_cu_seqlens,
                null_text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                scale_factor,
                guidance_weight,
                sparse_params=sparse_params,
            )
            model_pred = v_uncond + rfg_scale * (v_cond - v_uncond)
        else:
            model_pred = v_cond
        out_channels = model_pred.shape[-1]

        if prediction_target == "x0":
            # Convert x0 prediction to velocity: v = (x_t - x0) / t
            t_clamped = timestep.clamp(min=X0_T_CLAMP)
            velocity = (img[..., :out_channels] - model_pred) / t_clamped
        else:
            velocity = model_pred

        # NOTE: update channel slice for instruct (only first model.in_visual_dim channels)
        img[..., :out_channels] += timestep_diff * velocity

        if progress_callback is not None:
            progress_callback()

    # NOTE: make sure that injected first frames stay unchanged in the end of I2V;
    # slicing not done here, since instruct does not work together with I2V
    if model.visual_cond and first_frames is not None and visual_cond_scheme == "i2v":
        img[visual_cu_seqlens[:-1]] = first_frames

    # NOTE: return channel slice for instruct
    return img[..., :out_channels]


def _encode_lq_videos(
    lq_videos: list[torch.Tensor],
    vae: torch.nn.Module,
    device: str | int,
) -> torch.Tensor:
    """VAE-encode a list of LQ videos into a stacked latent tensor.

    Args:
        lq_videos: List of ``[T, H, W, 3]`` float32 tensors in ``[0, 255]``.
        vae: Pre-trained VAE model (eval mode).
        device: Target CUDA device.

    Returns:
        Stacked latent tensor ``[sum(T'), H', W', C]`` scaled by
        ``vae.config.scaling_factor``.
    """
    latents: list[torch.Tensor] = []
    for lq in lq_videos:
        # [T, H, W, 3] → [1, 3, T, H, W]; normalize + encode handled by the helper
        lq_input = lq.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        lq_latent = encode_pixels_to_latent(vae, lq_input)
        # [1, C, T', H', W'] → [T', H', W', C]
        lq_latent = lq_latent.squeeze(0).permute(1, 2, 3, 0).float()
        lq_latent *= vae.config.scaling_factor  # type: ignore
        latents.append(lq_latent)
    return torch.cat(latents, dim=0)


@torch.no_grad()
def decode_latent_to_pixels(
    latent: torch.Tensor,
    vae: torch.nn.Module,
) -> torch.Tensor:
    """Decode a single latent tensor to pixel-space video via VAE.

    Args:
        latent: ``[T, H, W, C]`` float latent (not yet scaled by ``scaling_factor``).
        vae: Pre-trained VAE model (already on the correct device).

    Returns:
        ``[3, T, H, W]`` uint8 tensor in ``[0, 255]``.
    """
    # [T, H, W, C] → [1, C, T, H, W]
    x = latent.permute(3, 0, 1, 2).unsqueeze(0)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=latent.device.type == "cuda",
    ):
        pixels = decode_latent_to_uint8(vae, x)
    # [1, 3, T, H, W] → [3, T, H, W] uint8
    return pixels.squeeze(0)


def _build_initial_latent(  # noqa: PLR0913
    dit: torch.nn.Module,
    lq_latent: torch.Tensor,
    bs: int,
    duration: int,
    height: int,
    width: int,
    device: str | int,
    seed: int,
    lq_noise_scale: float,
    lq_noise_type: Literal["linear", "ddpm"],
    lq_channel_noise_scale: float,
    anchor_latent: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    *,
    anchor_free: bool = False,
) -> torch.Tensor:
    """Build the initial latent tensor for the SR denoising loop.

    Constructs ``[starting | cond | mask]`` depending on ``instruct_type``:

    - ``"noise"``: degraded LQ as starting point, zero LQ cond + zero mask.
    - ``"channel"``: random Gaussian noise as starting point, LQ cond + ones mask.
    - ``"hybrid"``: degraded LQ as starting point, LQ cond + ones mask.
    - ``"hybrid_anchor"``: degraded LQ as starting point, sparse HR anchor +
      sparse anchor_mask.  Requires ``anchor_latent`` and ``anchor_mask``.

    Args:
        dit: DiT model (reads ``instruct_type`` and ``in_visual_dim``).
        lq_latent: Scaled LQ latent ``[bs*T, H, W, C]``.
        bs: Batch size.
        duration: Number of temporal frames per sample.
        height: Latent height.
        width: Latent width.
        device: Target CUDA device.
        seed: RNG seed for noise generation.
        lq_noise_scale: Noise fraction mixed into the LQ starting point.
        lq_noise_type: ``"linear"`` or ``"ddpm"`` noising formula.
        lq_channel_noise_scale: Noise fraction mixed into the channel-cat
            conditioning (LQ for ``channel``/``hybrid``, anchor for
            ``hybrid_anchor``).
        anchor_latent: Pre-scaled sparse HR anchor ``[bs*T, H, W, C]`` (zeros
            in slots without anchor).  Required when
            ``instruct_type == "hybrid_anchor"``.
        anchor_mask: Sparse binary mask ``[bs*T, H, W, 1]`` flagging anchor
            slots.  Required when ``instruct_type == "hybrid_anchor"``.

    Returns:
        Initial latent tensor ready for the denoising loop (bs*T, H, W, 33).

    Raises:
        ValueError: If ``instruct_type`` is not supported, or if
            ``hybrid_anchor`` is missing ``anchor_latent`` / ``anchor_mask``.
    """
    lq_latent = lq_latent.to(device)
    lq_latent = cast_to_module_dtype(dit, lq_latent)
    if anchor_latent is not None:
        anchor_latent = anchor_latent.to(device)
        anchor_latent = cast_to_module_dtype(dit, anchor_latent)
    if anchor_mask is not None:
        anchor_mask = anchor_mask.to(device)
        anchor_mask = cast_to_module_dtype(dit, anchor_mask)

    if dit.instruct_type == "noise":
        g_noise = torch.Generator(device=device)
        g_noise.manual_seed(seed)
        degraded_lq = degrade_lq_latent(lq_latent, lq_noise_scale, lq_noise_type, generator=g_noise)
        if not getattr(dit, "visual_cond", False):
            # No conditioning channels in the model input layer (e.g. the
            # t2v-initialized KVAE SR DiT) — feed the bare C-channel latent,
            # mirroring prepare_noisy_input at train time.
            return degraded_lq
        zero_cond = torch.zeros_like(degraded_lq)
        zero_mask = torch.zeros([*degraded_lq.shape[:-1], 1], dtype=degraded_lq.dtype, device=device)
        return torch.cat([degraded_lq, zero_cond, zero_mask], dim=-1)

    if dit.instruct_type in ("channel", "hybrid"):
        if dit.instruct_type == "hybrid":
            g_noise = torch.Generator(device=device)
            g_noise.manual_seed(seed)
            starting = degrade_lq_latent(lq_latent.clone(), lq_noise_scale, lq_noise_type, generator=g_noise)
        else:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            starting = torch.randn(
                bs * duration,
                height,
                width,
                dit.in_visual_dim,  # type: ignore
                device=device,
                generator=g,
            )
        channel_lq = degrade_lq_latent(lq_latent, lq_channel_noise_scale, noise_type="linear")
        mask = torch.ones_like(lq_latent[..., :1])
        return torch.cat([starting, channel_lq, mask], dim=-1)

    if dit.instruct_type == "hybrid_anchor":
        if anchor_latent is None or anchor_mask is None:
            if not anchor_free:
                msg = "hybrid_anchor requires anchor_latent and anchor_mask"
                raise ValueError(msg)
            # Anchor-free inference (tiled real-world LQ has no HR ground truth):
            # a zeroed anchor + zeroed mask runs the model with no anchor signal,
            # equivalent to anchor_indices=[] for the packed-latent datasets.
            anchor_latent = torch.zeros_like(lq_latent)
            anchor_mask = torch.zeros_like(lq_latent[..., :1])
        g_noise = torch.Generator(device=device)
        g_noise.manual_seed(seed)
        starting = degrade_lq_latent(lq_latent.clone(), lq_noise_scale, lq_noise_type, generator=g_noise)
        anchor = degrade_lq_latent(anchor_latent, lq_channel_noise_scale, noise_type="linear")
        return torch.cat([starting, anchor, anchor_mask], dim=-1)

    msg = f"generate_sample_sr does not support instruct_type={dit.instruct_type!r}"
    raise ValueError(msg)


def _empty_text_embeds(
    bs: int,
    device: str | int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build placeholder text embeddings for a text-free DiT.

    A ``use_text=False`` DiT ignores every text input (encoder, cross-attention
    and pooled-text conditioning are all absent), so no real embeddings are
    needed. These zero-size tensors only satisfy the indexing and ``cu_seqlens``
    bookkeeping in ``generate_sample_sr`` / ``get_model_prediction`` without ever
    loading the cached empty-caption file.

    Args:
        bs: Batch size.
        device: Target CUDA device.

    Returns:
        Tuple of ``(text_embeds_dict, text_cu_seqlens)`` with empty sequences.
    """
    embeds = {
        "text_embeds": torch.zeros(0, 1, device=device),
        "pooled_embed": torch.zeros(bs, 1, device=device),
    }
    cu_seqlens = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    return embeds, cu_seqlens


def _encode_text(
    bs: int,
    device: str | int,
    text_embedder: Any | None = None,
    cached_text_embeds: dict[str, torch.Tensor] | None = None,
    *,
    use_text: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Encode text or use cached embeddings for SR generation.

    For SR we always use empty captions. Returns both text and null-text
    embeddings (identical when using cached embeds).

    Args:
        bs: Batch size.
        device: Target CUDA device.
        text_embedder: Text encoder (Qwen2.5-VL + CLIP). Can be ``None``
            when ``cached_text_embeds`` is provided.
        cached_text_embeds: Pre-computed empty-caption embeddings (bs=1)
            loaded from disk.
        use_text: Whether the DiT consumes text. When ``False`` placeholder
            zero embeddings are returned and neither a text encoder nor cached
            embeddings are required.

    Returns:
        Tuple of ``(text_embed, text_cu_seqlens, null_text_embed,
        null_text_cu_seqlens)``.

    Raises:
        ValueError: If a text DiT is given neither ``text_embedder`` nor
            ``cached_text_embeds``.
    """
    if not use_text:
        embeds, cu_seqlens = _empty_text_embeds(bs, device)
        return embeds, cu_seqlens, embeds, cu_seqlens

    if cached_text_embeds is not None:
        bs_text_embed, text_cu_seqlens = replicate_cached_embeds(cached_text_embeds, bs, device)
        return bs_text_embed, text_cu_seqlens, bs_text_embed, text_cu_seqlens

    if text_embedder is not None:
        empty_captions = [""] * bs
        bs_text_embed, text_cu_seqlens = text_embedder.encode(
            empty_captions,
            images=None,
            type_of_content="video",
        )
        bs_null_text_embed, null_text_cu_seqlens = text_embedder.encode(
            empty_captions,
            images=None,
            type_of_content="video",
        )
        bs_text_embed = {k: v.to(device) for k, v in bs_text_embed.items()}
        text_cu_seqlens = text_cu_seqlens.to(device)
        bs_null_text_embed = {k: v.to(device) for k, v in bs_null_text_embed.items()}
        null_text_cu_seqlens = null_text_cu_seqlens.to(device)
        return bs_text_embed, text_cu_seqlens, bs_null_text_embed, null_text_cu_seqlens

    msg = "generate_sample_sr requires either text_embedder or cached_text_embeds"
    raise ValueError(msg)


@torch.no_grad()
def generate_sample_sr(  # noqa: PLR0913
    *,
    dit: torch.nn.Module,
    vae: torch.nn.Module,
    scale_factor: tuple[float, ...] = (1.0, 1.0, 1.0),
    num_steps: int = 50,
    guidance_weight: float = 5.0,
    scheduler_scale: float = 5.0,
    seed: int = 42,
    device: str | int = "cuda",
    tp_mesh: dict[str, Any] | None = None,
    lq_noise_scale: float = 0.0,
    lq_noise_type: Literal["linear", "ddpm"] = "linear",
    lq_channel_noise_scale: float = 0.0,
    text_embedder: Any | None = None,
    cached_text_embeds: dict[str, torch.Tensor] | None = None,
    lq_videos: list[torch.Tensor] | None = None,
    lq_latents: torch.Tensor | None = None,
    n_samples: int | None = None,
    cap_noise_timestep: bool = False,
    vae_decode_batch: bool = False,
    prediction_target: str = "velocity",
    channelcat_drop_threshold: float = 0.0,
    anchor_latents: torch.Tensor | None = None,
    anchor_masks: torch.Tensor | None = None,
    anchor_free: bool = False,
    rfg_scale: float = 1.0,
) -> torch.Tensor:
    """Generate super-resolved video from LQ inputs (pixels or latents).

    Accepts either raw pixel videos (``lq_videos``) or pre-encoded latents
    (``lq_latents``).  When ``lq_latents`` is provided, VAE encoding is
    skipped and the latents are used directly.

    Builds the initial latent tensor, runs the denoising loop via
    ``generate()``, and decodes back to pixel space.

    Supports three SR instruct modes:

    - ``instruct_type="channel"``: random noise in HQ channels with LQ latent
        and mask concatenated as extra channels → ``[noise(C) | lq(C) | mask(1)]``.
    - ``instruct_type="noise"``: LQ latent is used directly as the starting
        point (replaces random noise) → ``[lq(C)]``, zero extra channels.
    - ``instruct_type="hybrid"``: LQ latent as starting point (like ``"noise"``)
        AND LQ latent concatenated as extra channels (like ``"channel"``) →
        ``[degraded_lq(C) | lq(C) | mask(1)]``.

    When ``lq_noise_scale > 0``, noise is mixed into the LQ latent used as
    the starting point (applies to ``"noise"`` and ``"hybrid"``).

    When ``lq_channel_noise_scale > 0``, noise is mixed into the LQ latent
    concatenated as extra channels (applies to ``"channel"`` and ``"hybrid"``).

    Args:
        dit: DiT model with ``instruct_type`` in
            ``{"channel", "noise", "hybrid"}``.
        vae: Pre-trained VAE model.
        scale_factor: Per-axis RoPE frequency scaling.
        num_steps: Number of diffusion denoising steps.
        guidance_weight: Classifier-free guidance strength.
        scheduler_scale: Timestep scheduler scaling factor.
        seed: RNG seed for noise initialization.
        device: Target CUDA device.
        tp_mesh: Tensor parallelism mesh (optional).
        lq_noise_scale: Noise fraction ``s`` mixed into LQ latent used as
            the noise/starting point (``0`` = disabled).  Applies to
            ``"noise"`` and ``"hybrid"`` modes.
        lq_noise_type: Noising formula — ``"linear"`` or ``"ddpm"``.
        lq_channel_noise_scale: Noise fraction mixed into LQ latent
            concatenated as extra channels (``0`` = disabled).  Applies to
            ``"channel"`` and ``"hybrid"`` modes.
        text_embedder: Text encoder (Qwen2.5-VL + CLIP). Can be ``None``
            when ``cached_text_embeds`` is provided.
        cached_text_embeds: Pre-computed empty-caption embeddings (bs=1)
            loaded from disk. When provided, ``text_embedder`` is not called.
        lq_videos: List of ``[T, H, W, 3]`` float32 tensors in ``[0, 255]``.
        lq_latents: Pre-encoded LQ latents (skips VAE encoding).
        n_samples: Required when ``lq_latents`` is provided.
        cap_noise_timestep: When ``True`` and ``instruct_type`` in
            ``("noise", "hybrid")``, starts the denoising loop from raw
            ``t = lq_noise_scale`` instead of ``t = 1.0``.  Matches the
            training-time behaviour when ``cap_noise_timestep`` is also set
            during training.  Default ``False`` preserves current behaviour.
        vae_decode_batch: When ``True``, decode all samples in a single VAE
            call instead of one-by-one. Faster but uses more GPU memory.
        prediction_target: ``"velocity"`` for velocity prediction (default),
            ``"x0"`` for clean-image prediction.
        channelcat_drop_threshold: When ``> 0``, zero out channel-cat LQ
            channels during denoising once ``t < threshold``.  ``0`` =
            disabled (default).
        anchor_latents: Pre-encoded sparse HR-anchor latents
            ``[bs*T', H', W', C]``, **already scaled** by VAE
            ``scaling_factor``.  Required for ``instruct_type="hybrid_anchor"``.
        anchor_masks: Sparse binary anchor mask ``[bs*T', H', W', 1]``.
            Required for ``instruct_type="hybrid_anchor"``.
        anchor_free: When ``True`` and ``instruct_type="hybrid_anchor"`` with
            no anchors provided, run anchor-free (zeroed anchor + mask) — the
            tiled path for GT-less real-world LQ videos.
        rfg_scale: Reference-Free Guidance strength on the anchor.  ``1.0``
            (default) runs a single cond forward per denoising step; values
            ``!= 1.0`` add a second uncond forward (anchor + anchor_mask
            zeroed) and blend ``v_uncond + rfg_scale * (v_cond - v_uncond)``.
            Only applies when ``instruct_type="hybrid_anchor"``.  Cost: +1
            DiT forward per denoising step when active.
    Returns:
        Generated SR videos as ``[bs, 3, T, H, W]`` uint8 tensor.

    Raises:
        ValueError: If neither ``text_embedder`` nor ``cached_text_embeds``
            is provided or if ``instruct_type`` is unsupported.
    """
    from musubi_tuner.kandinsky6.runtime.sr.pipeline.stages import run_stages

    piflow_params = getattr(dit, "piflow_params", None)
    if piflow_params is not None:
        if rfg_scale != 1.0:
            msg = "π-Flow checkpoints do not support Reference-Free Guidance (drop rfg_scale)"
            raise ValueError(msg)

    return run_stages(
        dit=dit,
        vae=vae,
        scale_factor=scale_factor,
        num_steps=num_steps,
        guidance_weight=guidance_weight,
        scheduler_scale=scheduler_scale,
        seed=seed,
        device=device,
        tp_mesh=tp_mesh,
        lq_noise_scale=lq_noise_scale,
        lq_noise_type=lq_noise_type,
        lq_channel_noise_scale=lq_channel_noise_scale,
        text_embedder=text_embedder,
        cached_text_embeds=cached_text_embeds,
        lq_videos=lq_videos,
        lq_latents=lq_latents,
        n_samples=n_samples,
        cap_noise_timestep=cap_noise_timestep,
        vae_decode_batch=vae_decode_batch,
        prediction_target=prediction_target,
        channelcat_drop_threshold=channelcat_drop_threshold,
        anchor_latents=anchor_latents,
        anchor_masks=anchor_masks,
        anchor_free=anchor_free,
        rfg_scale=rfg_scale,
        piflow_params=piflow_params,
    )
