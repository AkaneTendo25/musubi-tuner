"""Post-load warmup: compile nabla flex_attention kernels and the VAE decode.

Mirrors musubi_tuner.kandinsky6.runtime.sr/validate.py: strip the ``log_for_sparsity`` wrapper that
forces ``return_sparsity=True`` (a recompile cascade that ends in eager
flex_attention OOM), bump the dynamo recompile limit, then run a tiny flex
forward so fused kernels are cached before the first real tile. A best-effort
dummy VAE decode triggers the compiled-VAE compile.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch._dynamo as dynamo
from loguru import logger

from musubi_tuner.kandinsky6.runtime.sr import constants
from musubi_tuner.kandinsky6.runtime.sr.constants import SPARSE_RECOMPILE_LIMIT
from musubi_tuner.kandinsky6.runtime.sr.core.algo.train_utils import get_sparse_params
from musubi_tuner.kandinsky6.runtime.sr.core.algo.utils import generate_sample_sr
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.nn import MultiheadSelfAttention
from musubi_tuner.kandinsky6.runtime.sr.pipeline.sr_pipeline import (
    _closest_base_resolution,
    _tile_geometry,
    encode_lq_video_to_lr_latent,
    latent_path_padding,
    upscale_lr_latent_tile,
)


def silence_symbolic_shape_warnings() -> None:
    """Mute torch's benign symbolic-shape value-range warnings during compile.

    Under ``dynamic=True`` compilation, ``torch.utils._sympy.interp`` logs
    ``failed while executing pow_by_natural(...)`` whenever its value-range
    interpreter hits a reciprocal (``x ** -1``) — it falls back to a safe bound,
    so the messages are pure noise that floods the flex/VAE warmup.
    """
    logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)


def strip_log_for_sparsity() -> None:
    """Remove the ``log_for_sparsity`` wrapper from ``attention_flex`` if present."""
    cls = MultiheadSelfAttention
    if hasattr(cls.attention_flex, "__wrapped__"):
        cls.attention_flex = cls.attention_flex.__wrapped__  # type: ignore[method-assign]
        logger.info("Stripped log_for_sparsity wrapper from attention_flex")


def set_sparse_recompile_limit() -> None:
    """Raise the dynamo recompile limit to a defensive ceiling for sparse attention."""
    dynamo.config.suppress_errors = False
    if dynamo.config.recompile_limit < SPARSE_RECOMPILE_LIMIT:
        logger.info(
            "Bumping dynamo.config.recompile_limit {} -> {}",
            dynamo.config.recompile_limit,
            SPARSE_RECOMPILE_LIMIT,
        )
        dynamo.config.recompile_limit = SPARSE_RECOMPILE_LIMIT


def warmup_flex_attention(dit: torch.nn.Module, scale_factor: tuple[float, ...], device: int | str) -> None:
    """Trigger torch.compile for flex_attention with a tiny forward pass.

    Uses minimal inputs (1 frame, 64x64 latent) so the unfused fallback fits in
    memory; once compiled with ``dynamic=True`` the fused kernels are reused.

    Args:
        dit: The loaded DiT (parameters materialized on ``device``).
        scale_factor: RoPE frequency scaling for the target resolution.
        device: CUDA device for the dummy tensors.
    """
    ps = dit.patch_size
    # The dummy must resolve to a resolution the model has attention_params
    # for: 512-family. The latent size therefore depends on the VAE spatial
    # factor (32x32 -> 512x512 under the kvae's 16x) — a hardcoded size would
    # silently skip the warmup (get_sparse_params returns None).
    n_frames = 1
    h_latent = w_latent = 512 // constants.VAE_SPATIAL_FACTOR
    if dit.visual_cond or dit.instruct_type in ("channel", "hybrid", "hybrid_anchor"):
        channels = 2 * dit.in_visual_dim + 1
    else:
        channels = dit.in_visual_dim

    x = torch.randn(n_frames, h_latent, w_latent, channels, device=device, dtype=torch.bfloat16)
    visual_cu_seqlens = torch.tensor([0, n_frames], dtype=torch.int32, device=device)
    sparse_params = get_sparse_params(dit, x, visual_cu_seqlens)
    if sparse_params is None:
        return

    time = torch.tensor([500.0], device=device)
    visual_rope_pos = (
        torch.arange(n_frames, device=device),
        torch.arange(h_latent // ps[1], device=device),
        torch.arange(w_latent // ps[2], device=device),
    )
    if dit.use_text:
        text_len = 5
        text_embed = torch.randn(
            text_len, dit.text_embeddings.in_layer.in_features, device=device, dtype=torch.bfloat16
        )
        pooled_text_embed = torch.randn(
            1, dit.pooled_text_embeddings.in_layer.in_features, device=device, dtype=torch.bfloat16
        )
        text_cu_seqlens = torch.tensor([0, text_len], dtype=torch.int32, device=device)
        text_rope_pos = torch.arange(text_len, device=device)
    else:
        text_embed = pooled_text_embed = text_cu_seqlens = text_rope_pos = None

    dit.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dit(
            x,
            text_embed,
            pooled_text_embed,
            time,
            visual_cu_seqlens,
            text_cu_seqlens,
            visual_rope_pos,
            text_rope_pos,
            scale_factor=scale_factor,
            sparse_params=sparse_params,
        )
    torch.cuda.empty_cache()


def log_sampling_mode(components: Any, num_steps: int) -> None:
    """Log which denoiser the run uses: π-Flow DX few-step or flow Euler.

    A DiT carrying ``piflow_params`` always routes through the DX sampler;
    everything else is the ordinary flow Euler loop.
    """
    piflow_params = getattr(components.dit, "piflow_params", None)
    if piflow_params is not None:
        logger.info(
            "Sampling: π-Flow DX few-step — nfe={} (n_grid={}, shift={}, substeps={}; num_steps ignored)",
            piflow_params["nfe"],
            piflow_params["n_grid"],
            piflow_params["shift"],
            piflow_params["num_policy_substeps"],
        )
    else:
        logger.info("Sampling: flow Euler — {} steps ({} grid points)", num_steps - 1, num_steps)


def clip_base_resolution(components: Any, source_hw: tuple[int, int]) -> tuple[int, int]:
    """Return the trained base resolution ``(H, W)`` a clip of ``source_hw`` tiles into.

    The base is chosen by aspect ratio alone, so the pixel pre-upscale of the
    2.25x route (uniform scaling) does not change it.
    """
    return _closest_base_resolution(source_hw[0], source_hw[1], int(components.sr_params.visual_size[0]))


def compile_vae_decode(components: Any, device: int | str, bases: Sequence[tuple[int, int]] | None = None) -> None:
    """Compile the VAE decode for the given base resolutions (default: every trained one).

    torch.compile (and magi) build a graph the first time they see a shape,
    so the only way to compile ahead of a run is a forward of that shape. The
    decode tile is fixed by the model, not the clip: ``base / VAE_SPATIAL_FACTOR``
    latent pixels over ``MAX_NUM_FRAMES`` frames — 32x32, 32x48 and 48x32 for
    the kvae 512 family. A clip with another frame count only re-shapes the
    last temporal segment. Compiling one base is enough for one clip (see
    :func:`clip_base_resolution`); the magi backend pins a copy of the decoder
    weights per compiled graph, so all bases at once may not fit next to the
    DiT.

    Args:
        components: Anything exposing ``vae``, ``dit`` and ``sr_params`` (loaded
            components or the built pipeline); the DiT's ``in_visual_dim`` is
            the latent width the VAE decodes.
        device: CUDA device holding the VAE.
        bases: Base resolutions ``(H, W)`` to compile; ``None`` = all of
            ``RESOLUTIONS[visual_size]``.
    """
    vae, dit = components.vae, components.dit
    visual_size = int(components.sr_params.visual_size[0])
    t_latent = (constants.MAX_NUM_FRAMES - 1) // constants.VAE_TEMPORAL_FACTOR + 1
    dtype = next(vae.parameters(), torch.zeros(1, dtype=torch.bfloat16)).dtype
    for base_h, base_w in bases if bases is not None else constants.RESOLUTIONS[visual_size]:
        h, w = base_h // constants.VAE_SPATIAL_FACTOR, base_w // constants.VAE_SPATIAL_FACTOR
        logger.info("Compiling VAE decode for base {}x{}: {} frames at {}x{} latent", base_h, base_w, t_latent, h, w)
        # Exactly what the real path feeds the decoder: channels_last_3d under
        # bf16 autocast — the compiled artifact asserts those strides.
        z = torch.zeros(1, dit.in_visual_dim, t_latent, h, w, device=device, dtype=dtype)
        z = z.to(memory_format=torch.channels_last_3d)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            vae.decode(z)
    torch.cuda.empty_cache()


def warmup_target_pass(  # noqa: PLR0913
    components: Any,
    run_config: Any,
    source_hw: tuple[int, int],
    num_pixel_frames: int,
    lu: torch.nn.Module | None,
    lq_video: torch.Tensor | None = None,
) -> None:
    """One throwaway inference pass at the run's target tile shape (timing runs only).

    The first tile of a run pays one-time costs — flex/nabla compilation for
    the exact tile shape, cudnn autotune, allocator growth — so a timed run
    that should measure steady-state tiles warms them up here first: exactly
    the ops of the real loop on ONE synthetic tile (LU forward + DiT + KVAE
    decode on the LU path, DiT with its inner per-tile encode on the pixel
    path). It costs about one tile, so it has no value outside of timing.

    Args:
        components: Anything exposing ``dit``, ``vae``, ``sr_params`` and
            ``cached_text_embeds`` (loaded components or the built pipeline).
        run_config: The run's ``RunConfig`` (tiling scale, overlap, steps, seed, device).
        source_hw: Source pixel ``(H, W)`` of the upcoming run (after any pre-upscale).
        num_pixel_frames: Source frame count in pixel frames (``1 + 4k``).
        lu: The scale-resolved latent upscaler for this run, or ``None`` for the pixel path.
        lq_video: The run's ``[T, C, H, W]`` source video; on the LU path it is
            encoded once and discarded so the whole-clip KVAE encode is warm too.
    """
    sr_params = components.sr_params
    visual_size = int(sr_params.visual_size[0])
    if lu is not None:
        # The real run pads the source for the latent path; warm the shapes it will actually see.
        padding = latent_path_padding(components, run_config, source_hw)
        source_hw = padding.padded_hw(source_hw)
        lq_video = None if lq_video is None else padding.apply_to_video(lq_video)
    (base_h, base_w), tile_hw, _grid = _tile_geometry(
        source_hw[0], source_hw[1], visual_size, run_config.resolution_scale, run_config.overlap
    )
    if lu is not None and lq_video is not None:
        logger.info("Warmup: throwaway whole-clip vae.encode ({}x{})", *source_hw)
        encode_lq_video_to_lr_latent(lq_video, components.vae, run_config.device)

    if lu is not None:
        t_latent = (num_pixel_frames - 1) // constants.VAE_TEMPORAL_FACTOR + 1
        channels = int(getattr(lu, "in_channels", 0) or getattr(components.vae.config, "latent_channels", 64))
        tile = torch.randn(
            t_latent, channels, tile_hw[0] // constants.VAE_SPATIAL_FACTOR, tile_hw[1] // constants.VAE_SPATIAL_FACTOR
        )
        lq_kwargs: dict[str, Any] = {
            "lq_latents": upscale_lr_latent_tile(tile, lu, components.vae, run_config.device),
            "n_samples": 1,
        }
    else:
        lq_kwargs = {"lq_videos": [torch.rand(num_pixel_frames, base_h, base_w, 3) * 255.0]}
    logger.info(
        "Warmup: one throwaway tile at base {}x{} ({} path)", base_h, base_w, "LU" if lu is not None else "pixel"
    )
    with torch.no_grad():
        generate_sample_sr(
            **lq_kwargs,
            dit=components.dit,
            vae=components.vae,
            scale_factor=tuple(sr_params.scale_factor[visual_size]),
            # DiT/VAE shapes repeat across denoising steps: 2 grid points (1 full step) warm every kernel.
            num_steps=min(run_config.num_steps, 2),
            anchor_free=True,
            guidance_weight=1.0,
            scheduler_scale=sr_params.scheduler_scale,
            seed=run_config.seed,
            device=run_config.device,
            tp_mesh=None,
            lq_noise_scale=sr_params.lq_noise_scale,
            lq_noise_type=sr_params.lq_noise_type,
            lq_channel_noise_scale=sr_params.lq_channel_noise_scale,
            cap_noise_timestep=sr_params.cap_noise_timestep,
            cached_text_embeds=components.cached_text_embeds,
        )
    torch.cuda.empty_cache()


def warmup(dit: torch.nn.Module, scale_factor: tuple[float, ...], device: int | str) -> None:
    """Run the post-load warmup (strip wrapper, bump limit, flex_attention compile).

    Only the flex/nabla attention compile lives here — it is always needed
    (without it the first real forward pays the compilation); the VAE decode
    graphs are compiled next to it by :func:`compile_vae_decode`.
    """
    silence_symbolic_shape_warnings()
    strip_log_for_sparsity()
    needs_flex = getattr(dit, "attention_params", None) is not None and any(
        str(getattr(p, "type", "")).startswith("nabla") for p in dit.attention_params.values()
    )
    if needs_flex:
        set_sparse_recompile_limit()
        logger.info("Warming up flex_attention compiled kernels...")
        warmup_flex_attention(dit, scale_factor, device)
        logger.info("flex_attention warmup complete")
