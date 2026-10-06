"""Regional AOTI export for repeated DiT visual blocks.

Export + compile one ``FusedTransformerDecoderBlock`` (T2VA) or
``TransformerDecoderBlock`` (T2V), then bind the same ``.pt2`` onto every
``dit.visual_transformer_blocks[i]`` with that block's Python ``state_dict``
(``package_constants_in_so=False``, ``user_managed=True``).

AOTI uses Inductor ``max-autotune-no-cudagraphs`` (``max_autotune`` +
``coordinate_descent_tuning``; no CUDA graphs — MagCache/FA3 outside the SO).

Keeps the Python ``for blk in visual_transformer_blocks`` loop so future per-block offload
can ``.to()`` individual blocks; CacheDiT still wraps the bare DiT afterwards.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.export import ExportedProgram

from ..core.algo.prepare_latents import audio_latent_duration
from ..core.algo.prepare_ropes import compute_rope1d, compute_visual_rope
from ..core.components.dit import (
    DiffusionTransformer3D,
    FusedTransformerDecoderBlock,
    TransformerDecoderBlock,
)
from .compile import install_forward

logger = logging.getLogger("kandinsky")


def build_example_kwargs(  # noqa: PLR0913
    dit: DiffusionTransformer3D,
    *,
    height: int,
    width: int,
    latent_frames: int,
    scale_factor: tuple[float, ...],
    text_max_length: int,
    in_text_dim: int,
    in_text_dim2: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Synthetic DiT-level kwargs (geometry + precomputed RoPE)."""
    device = torch.device(device)
    dtype = torch.bfloat16
    duration = latent_frames
    h_lat = height // 8
    w_lat = width // 8
    pt, ph, pw = dit.patch_size
    h_patches = h_lat // ph
    w_patches = w_lat // pw
    t_rope = duration // pt

    c = dit.in_visual_dim
    in_ch = (2 * c + 1) if dit.visual_cond else c
    x_video = torch.randn(duration, h_lat, w_lat, in_ch, device=device, dtype=dtype)

    scale = (float(scale_factor[0]), float(scale_factor[1]), float(scale_factor[2]))
    visual_rope = compute_visual_rope(dit.visual_rope, (t_rope, h_patches, w_patches), scale, device=device)

    text_len = text_max_length
    text_embed = torch.randn(text_len, in_text_dim, device=device, dtype=dtype)
    pooled = torch.randn(1, in_text_dim2, device=device, dtype=dtype)
    time = torch.full((1,), 500.0, device=device, dtype=dtype)

    kwargs: dict[str, Any] = {
        "x_video": x_video,
        "x_audio": None,
        "text_embed": text_embed,
        "pooled_text_embed": pooled,
        "time": time,
        "visual_rope": visual_rope,
        "audio_rope": None,
        "text_rope": compute_rope1d(dit.text_rope, text_len, device=device)
        if not dit.is_multimodal
        else [
            compute_rope1d(dit.video_text_rope, text_len, device=device),
            compute_rope1d(dit.audio_text_rope, text_len, device=device),
        ],
    }

    if dit.is_multimodal:
        audio_dur = audio_latent_duration(duration)
        x_audio = torch.randn(audio_dur, dit.in_audio_dim, device=device, dtype=dtype)
        kwargs.update(
            {
                "x_audio": x_audio,
                "audio_rope": compute_rope1d(dit.audio_rope, audio_dur, device=device),
                "text_embed": [text_embed, text_embed],
                "pooled_text_embed": [pooled, pooled],
                "text_rope": [
                    compute_rope1d(dit.video_text_rope, text_len, device=device),
                    compute_rope1d(dit.audio_text_rope, text_len, device=device),
                ],
                "time": [time, time],
            }
        )

    return kwargs


def _to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    return obj


@contextmanager
def _sdpa_for_cpu_materialize(module: nn.Module):
    """FA3/FA2 are CUDA-only; example-arg encode may run on CPU-held DiT.

    ``SelfAttentionEngine`` is a plain object (not ``nn.Module``), so walk
    ``.attn`` attributes hanging off attention layers.
    """
    from ..core.components.attention.dispatch import SelfAttentionEngine, _sdpa

    restored: list[tuple[SelfAttentionEngine, Any]] = []
    for m in module.modules():
        attn = getattr(m, "attn", None)
        if isinstance(attn, SelfAttentionEngine) and attn._fn is not _sdpa:
            restored.append((attn, attn._fn))
            attn._fn = _sdpa
    try:
        yield
    finally:
        for attn, fn in restored:
            attn._fn = fn


@torch.no_grad()
def materialize_block_example_args(
    dit: DiffusionTransformer3D,
    dit_kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    """Run embed/encode path → positional args for ``visual_transformer_blocks[0].forward``."""
    dit.eval()
    pad = bool(getattr(dit, "text_token_padding", False))

    with _sdpa_for_cpu_materialize(dit):
        if dit.is_multimodal:
            text_embed = dit_kwargs["text_embed"]
            pooled = dit_kwargs["pooled_text_embed"]
            text_rope = dit_kwargs["text_rope"]
            time = dit_kwargs["time"]
            te_v, pe_v = text_embed[0], pooled[0]
            te_a, pe_a = text_embed[1], pooled[1]
            rope_v, rope_a = text_rope[0], text_rope[1]
            t_v, t_a = time[0], time[1]
            video_te, video_tm = dit._encode_text("video", te_v, pe_v, t_v, rope_v)
            audio_te, audio_tm = dit._encode_text("audio", te_a, pe_a, t_a, rope_a)
            vis_embed, _, vis_rope = dit._embed_visual(dit_kwargs["x_video"], dit_kwargs["visual_rope"])
            aud_embed, aud_rope = dit._embed_audio(dit_kwargs["x_audio"], dit_kwargs["audio_rope"])
            args: tuple[Any, ...] = (
                vis_embed,
                aud_embed,
                video_te,
                audio_te,
                (video_tm, audio_tm),
                vis_rope,
                aud_rope,
            )
            if pad:
                # Static key-padding mask (True=valid); shape matches runtime pad-to-max.
                attn_mask = torch.ones(1, int(video_te.shape[0]), dtype=torch.bool, device=video_te.device)
                args = (*args, attn_mask)
            return args

        te_in = dit_kwargs["text_embed"]
        pe_in = dit_kwargs["pooled_text_embed"]
        rope_in = dit_kwargs["text_rope"]
        t_in = dit_kwargs["time"]
        te, tm = dit._encode_t2v(te_in, pe_in, t_in, rope_in)
        vis_embed, _, vis_rope = dit._embed_visual(dit_kwargs["x_video"], dit_kwargs["visual_rope"])
        args = (vis_embed, te, tm, vis_rope)
        if pad:
            attn_mask = torch.ones(1, int(te.shape[0]), dtype=torch.bool, device=te.device)
            args = (*args, attn_mask)
        return args


def export_module(
    module: nn.Module,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    dynamic_shapes: Any | None = None,
) -> ExportedProgram:
    """``torch.export`` a module; skip Dynamo weight clones (OOM on large DiT)."""
    import torch._dynamo.utils as _dyn_utils
    import torch._dynamo.variables.builder as _builder

    _orig_cache = _builder.cache_real_value_when_export
    _orig_clone = _dyn_utils.clone_input
    _builder.cache_real_value_when_export = lambda *_a, **_k: None  # noqa: ARG005

    def _no_clone(x, *, dtype=None):  # noqa: ARG001
        return x

    _dyn_utils.clone_input = _no_clone  # type: ignore[assignment]
    module = module.eval()
    try:
        with torch.no_grad():
            return torch.export.export(
                module,
                args=args,
                kwargs=kwargs or {},
                strict=True,
                dynamic_shapes=dynamic_shapes,
            )
    finally:
        _builder.cache_real_value_when_export = _orig_cache
        _dyn_utils.clone_input = _orig_clone


def export_visual_block(
    dit: DiffusionTransformer3D,
    dit_kwargs: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    text_max_length: int | None = None,
) -> ExportedProgram:
    """Export ``visual_transformer_blocks[0]`` with production-shaped block inputs.

    Prefer ``dit.text_token_padding=True`` (pad text to ``text_embedder.max_length``) so
    cond/uncond share a static text length — no ``dynamic_shapes`` required.
    """
    del text_max_length  # kept for CLI compat; length comes from example te
    if len(dit.visual_transformer_blocks) < 1:
        raise ValueError("DiT has no visual_transformer_blocks to export")
    block = dit.visual_transformer_blocks[0]
    expected = FusedTransformerDecoderBlock if dit.is_multimodal else TransformerDecoderBlock
    if not isinstance(block, expected):
        raise TypeError(f"expected visual_transformer_blocks[0] to be {expected.__name__}, got {type(block)}")

    args = materialize_block_example_args(dit, dit_kwargs)
    if device is not None:
        device = torch.device(device)
        block = block.to(device)
        args = _to_device(args, device)

    # Static shapes only (K5 pad-to-max). Dynamic text_len was dropped.
    return export_module(block, args=args)


def _configure_aoti_compile_parallelism() -> int:
    """Size Inductor compile workers for fat CPU nodes.

    PyTorch defaults to ``min(32, ncpu)`` — leaves most of a 96-core box idle.
    Use half the usable CPUs (cap 64); pin BLAS/OMP to 1 per worker so the pool
    does not oversubscribe. Honors an existing ``TORCHINDUCTOR_COMPILE_THREADS``.
    """
    import os

    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    reported = os.cpu_count() or affinity
    ncpu = max(1, min(affinity, reported))
    if ncpu >= 64:
        recommended = min(64, ncpu // 2)
    else:
        recommended = min(32, ncpu)

    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        workers = max(1, int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"]))
    else:
        workers = recommended
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(workers)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "subprocess")

    import torch._inductor.config as inductor_config

    inductor_config.compile_threads = workers
    return workers


def _default_aoti_inductor_configs() -> dict[str, Any]:
    """Default Inductor knobs (= ``max-autotune-no-cudagraphs``, weights outside SO)."""
    return {
        "aot_inductor.package_constants_in_so": False,
        "max_autotune": True,
        "coordinate_descent_tuning": True,
        "epilogue_fusion": True,
        "shape_padding": True,
    }


def compile_aoti(
    ep: ExportedProgram,
    out_path: str | Path,
    *,
    inductor_configs: dict[str, Any] | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Package AOTInductor artifact with weights kept outside the SO.

    ``inductor_configs`` / ``cache_dir`` come from ``compile.aoti`` in the YAML
    (see ``AotiCompileConfig``); defaults match max-autotune-no-cudagraphs.
    """
    import os

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Match pipeline factory: TF32 for residual float32 matmuls so Inductor
    # autotune/codegen sees the same policy as runtime (avoids TF32 UserWarning).
    torch.set_float32_matmul_precision("high")
    # Persist autotune choices. The export CLI passes ``$KANDINSKY_HOME/export/inductor``
    # unless ``compile.aoti.cache_dir`` is set; a direct call keeps the cache next to the artifact.
    resolved_cache = cache_dir or str(out_path.parent / ".torch_inductor_cache")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", resolved_cache)
    workers = _configure_aoti_compile_parallelism()
    configs = dict(inductor_configs) if inductor_configs is not None else _default_aoti_inductor_configs()
    if configs.get("triton.cudagraphs"):
        logger.warning(
            "triton.cudagraphs=true is fragile with MagCache skip and FA3 custom ops; prefer compile.aoti.cudagraphs=false"
        )
    max_at = bool(configs.get("max_autotune", False))
    mode = "max-autotune" if max_at and configs.get("triton.cudagraphs") else "max-autotune-no-cudagraphs" if max_at else "default"
    logger.info(
        "AOTI compile_threads=%s mode=%s cache=%s",
        workers,
        mode,
        os.environ["TORCHINDUCTOR_CACHE_DIR"],
    )
    torch._inductor.aoti_compile_and_package(
        ep,
        package_path=str(out_path),
        inductor_configs=configs,
    )
    return out_path


def save_exported_program(ep: ExportedProgram, out_path: str | Path) -> Path:
    """Persist ``ExportedProgram`` (``torch.export.save``) alongside AOTI if needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, out_path)
    return out_path


def _aoti_device_index(module: nn.Module) -> int:
    param = next(module.parameters())
    if param.device.type != "cuda":
        return -1
    idx = param.device.index
    return torch.cuda.current_device() if idx is None else idx


def bind_aoti_to_visual_blocks(
    dit: nn.Module,
    package_path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Load one regional AOTI package onto every ``visual_transformer_blocks[i]``.

    ``device`` is the CUDA compute device (e.g. ``torch.device("cuda:0")``).
    Pass it explicitly when the DiT blocks may be on CPU at call time (e.g.
    with module offload). AOTI copies constants to its own CUDA allocation
    (``user_managed=False``) so the PyTorch-side params can move freely.
    """
    package_path = Path(package_path)
    if not package_path.is_file():
        raise FileNotFoundError(f"dit_export package not found: {package_path}")

    blocks = dit.visual_transformer_blocks
    if len(blocks) < 1:
        raise ValueError("DiT has no visual_transformer_blocks to bind")

    if device is not None:
        dev = torch.device(device)
        device_index = (
            (torch.cuda.current_device() if dev.index is None else dev.index)
            if dev.type == "cuda"
            else _aoti_device_index(blocks[0])
        )
    else:
        device_index = _aoti_device_index(blocks[0])

    for i, blk in enumerate(blocks):
        compiled = torch._inductor.aoti_load_package(
            str(package_path),
            device_index=device_index,
        )
        compiled.load_constants(
            dict(blk.state_dict()),
            check_full_update=True,
            user_managed=False,
            allow_h2d_copy=True,
        )

        def _forward(*args, _compiled=compiled, **kwargs):
            return _compiled(*args, **kwargs)

        install_forward(blk, _forward)
        blk._aoti_compiled = compiled  # type: ignore[attr-defined]

    dit._aoti_block_package = str(package_path)  # type: ignore[attr-defined]
    logger.info("bound AOTI to %d visual_transformer_blocks from %s", len(blocks), package_path)
    return dit


# Backward-compat alias used by older notes / scripts.
bind_aoti_to_dit = bind_aoti_to_visual_blocks
