"""Compiled (torch.compile-optimized) causal video KVAE.

SR inference with the KVAE is decode-bound: the eager segment-cached
``CachedDecoder3D`` runs ~1.5x slower than a fully compiled decode path
(178s vs 115s per 5s 512x768 video), while ``vae.encode`` is ~15s. Earlier
accelerated KVAE variants compiled only the encoder for dataset encoding, so
the decoder was everywhere the same eager loop.

Whole-graph compilation of the decoder is not feasible: the temporal segment
cache is a python dict of per-layer padding tensors mutated across chunks
(``None`` on the first segment, tensors afterwards). Instead the decoder is
compiled REGION-wise by default — whole ``CachedCausalResnetBlock3D`` forwards
plus the standalone convs/norms/upsamples between them — with the cache
plumbing staying eager, and its conv weights converted to ``channels_last_3d``.

Measured on the tiled SR pipeline (5s 512x768 video, 25 tiles, 7 steps):
``vae.decode`` 178.6s eager → 135s per-leaf compile → **90.8s** region compile
(compiled reference: 115s). Env knobs:

* ``KVAE_COMPILE=0`` — fully eager fallback;
* ``KVAE_COMPILE_DYNAMIC=1`` — compile dynamic symbolic temporal slices
  instead of the default static segment shapes (one graph per segment length
  and tile aspect, cached by inductor on disk). Static is the default because
  it is faster (x2 clip on an A100, torch 2.10: 45 s static vs 75-85 s dynamic)
  and torch <= 2.9's dynamo cannot build the dynamic guards at all;
* ``KVAE_COMPILE_DYNAMIC_MODE=regions`` — dynamic resblock regions (default
  of the dynamic mode);
* ``KVAE_COMPILE_DYNAMIC_MODE=inner_conv`` — dynamic stateless inner-conv
  wrapping (the conservative fallback);
* ``KVAE_COMPILE_RESBLOCKS=0`` — per-leaf granularity instead of regions;
* ``KVAE_COMPILE_ENCODER=1`` — also compile the encoder (measured SLOWER:
  14.5s → 22s, hence off);
* ``KVAE_COMPILE_AUTOTUNE=1`` — inductor max_autotune (measured no effect);
* ``KVAE_DECODE_SEG`` — pixel frames per decoder temporal segment (default 16;
  32 measured slightly slower).

The first decode pays a one-time compile cost (mirroring the compiled
VAE); shapes are stable across SR tile chunks, so no recompiles afterwards.
"""

from __future__ import annotations

import os

import torch
from loguru import logger
from omegaconf import OmegaConf

from musubi_tuner.kandinsky6.runtime.sr.core.components.model.vae_io import kvae_weights_path
from musubi_tuner.kandinsky6.runtime.sr.core.components.video_kvae.cached_layers import (
    CachedCausalConv3d,
    CachedCausalResnetBlock3D,
    CachedGroupNorm,
    CachedPXSDownsample,
    CachedPXSUpsample,
    CachedSpatialNorm3D,
)
from musubi_tuner.kandinsky6.runtime.sr.core.components.video_kvae.cached_model import CachedCausalVAE

# Compute-heavy leaves of CachedEncoder3D / CachedDecoder3D. Their forwards
# take the segment-cache dict as an argument — fullgraph=False lets dynamo
# break on the dict handling while still fusing the conv/norm math.  This
# list is used by the static-shape region path; dynamic mode has a narrower
# list below because cache-aware wrappers contain symbolic temporal slices.
# CachedCausalResnetBlock3D is intentionally NOT listed: it only orchestrates
# these leaves, so compiling it would just duplicate the same regions.
_COMPILED_LEAF_TYPES = (
    CachedCausalConv3d,
    CachedGroupNorm,
    CachedSpatialNorm3D,
    CachedPXSUpsample,
    CachedPXSDownsample,
)

_COMPILE_WRAPPED_FLAG = "_kvae_leaf_compiled"
_INNER_COMPILE_WRAPPED_FLAG = "_kvae_inner_conv_compiled"
_DYNAMIC_CACHE_EAGER_FLAG = "_kvae_dynamic_cache_eager"
_DYNAMIC_COMPILE_MODES = frozenset({"regions", "inner_conv"})

# Inductor autotune options for the leaf compiles (mirrors the settings the
# compiled VAE / encoder pieces use). Opt-in via
# ``KVAE_COMPILE_AUTOTUNE=1`` — autotune shaves per-kernel time but makes the
# one-time first-decode compilation take several minutes.
_AUTOTUNE_OPTIONS = {
    "max_autotune": True,
    "memory_planning": True,
    # deprecated in torch 2.13
    # "force_same_precision": True,
}


def kvae_compile_kwargs() -> dict:
    """Return the ``torch.compile`` kwargs for leaf wrapping.

    Static shapes by default (the faster mode, and the only one torch <= 2.9
    can compile); ``KVAE_COMPILE_DYNAMIC=1`` opts into dynamic symbolic shapes.
    ``KVAE_COMPILE_AUTOTUNE=1`` adds the inductor autotune options.
    """
    dynamic = os.environ.get("KVAE_COMPILE_DYNAMIC", "0") == "1"
    kwargs: dict = {"dynamic": dynamic, "fullgraph": False}
    if os.environ.get("KVAE_COMPILE_AUTOTUNE", "0") == "1":
        kwargs["options"] = dict(_AUTOTUNE_OPTIONS)
    return kwargs


def kvae_dynamic_compile_mode() -> str:
    """Return the dynamic KVAE compile boundary.

    ``regions`` compiles resblocks while graph-breaking around cache-aware
    wrappers.  ``inner_conv`` keeps the earlier conservative mode that only
    compiles stateless convolution kernels and GroupNorm leaves.
    """
    mode = os.environ.get("KVAE_COMPILE_DYNAMIC_MODE", "regions")
    if mode not in _DYNAMIC_COMPILE_MODES:
        valid = ", ".join(sorted(_DYNAMIC_COMPILE_MODES))
        raise ValueError(f"Unknown KVAE_COMPILE_DYNAMIC_MODE {mode!r}; expected one of: {valid}")
    return mode


def kvae_encoder_compile_enabled() -> bool:
    """Whether to also compile/channels-last the ENCODER (default off).

    Measured on the SR assess pipeline, compiling the encoder REGRESSED
    ``vae.encode`` 14.5s -> 22s per video (many small high-res convs; the
    per-segment wrapper overhead outweighs the fusion gains), so the encoder
    stays eager unless ``KVAE_COMPILE_ENCODER=1`` is set.
    """
    return os.environ.get("KVAE_COMPILE_ENCODER", "0") == "1"


def decode_split_list(latent_frames: int, temporal_compress: int, seg_pixel_frames: int) -> list[int]:
    """Build the decoder's temporal split list in PIXEL frame units.

    Mirrors ``CachedCausalVAE.decode``'s auto-split (first segment carries the
    extra causal frame) with a configurable segment size; the returned sizes
    are multiples of ``temporal_compress`` so the base decode's
    ``ceil(size / tc)`` conversion round-trips exactly.

    Args:
        latent_frames: ``z.shape[2]`` of the latent being decoded.
        temporal_compress: The VAE temporal compression factor.
        seg_pixel_frames: Target segment length in pixel frames (16 reproduces
            the eager default of ``16 // tc`` latent frames per segment).

    Returns:
        Pixel-frame segment sizes summing to ``latent_frames * tc``.
    """
    seg = max(1, seg_pixel_frames // temporal_compress)
    if latent_frames == 1:
        latent_split = [1]
    else:
        latent_split = [seg] * ((latent_frames - 1) // seg)
        if (latent_frames - 1) % seg:
            latent_split.append((latent_frames - 1) % seg)
        latent_split[0] += 1
    return [s * temporal_compress for s in latent_split]


def compile_kvae_leaves(module: torch.nn.Module, *, enabled: bool = True) -> int:
    """Wrap the compute-heavy leaf forwards of a KVAE half with ``torch.compile``.

    Compilation is lazy — wrapping executes no kernels, so this is safe to call
    on CPU at build time; inductor kicks in on the first real forward.

    Dynamic compilation uses a slightly different boundary from static
    compilation.  The cache-aware KVAE wrappers contain Python cache mutation
    and symbolic temporal slices; compiling those wrappers makes Inductor
    reason about ranges such as ``offset_in:`` and can produce invalid ranges
    for a later segment.  The default ``regions`` mode graph-breaks around
    those wrappers and compiles the surrounding resblock regions.  The
    ``inner_conv`` mode is an explicit conservative fallback that compiles
    only stateless ``SafeConv3d`` and GroupNorm kernels.  Static mode keeps the
    faster region compilation path, because every segment shape is fixed.

    Args:
        module: Encoder or decoder (any module tree) to walk.
        enabled: ``False`` makes this a strict no-op (eager fallback / tests).
            ``KVAE_COMPILE=0`` in the environment forces the no-op too — the
            escape hatch for entrypoints that hard-code KVAE compilation
            (run_inference.py / infer_sr_tiling.py).

    Returns:
        Number of leaf modules wrapped (0 when disabled or already wrapped).
    """
    if not enabled or os.environ.get("KVAE_COMPILE", "1") == "0":
        return 0
    compile_kwargs = kvae_compile_kwargs()
    dynamic = compile_kwargs.get("dynamic", True)

    def wrap(leaf: torch.nn.Module) -> int:
        if getattr(leaf, _COMPILE_WRAPPED_FLAG, False):
            return 0
        leaf.forward = torch.compile(leaf.forward, **compile_kwargs)
        setattr(leaf, _COMPILE_WRAPPED_FLAG, True)
        return 1

    def wrap_inner_conv(leaf: CachedCausalConv3d) -> int:
        """Compile only the stateless convolution below the causal cache."""
        conv = leaf.conv
        if getattr(conv, _INNER_COMPILE_WRAPPED_FLAG, False):
            return 0
        conv.forward = torch.compile(conv.forward, **compile_kwargs)
        setattr(conv, _INNER_COMPILE_WRAPPED_FLAG, True)
        return 1

    if dynamic:
        mode = kvae_dynamic_compile_mode()
        if mode == "inner_conv":
            # Do not compile CachedCausalConv3d / CachedPXS* /
            # CachedSpatialNorm3D as a whole: they contain cache-dependent
            # slices and Python shape arithmetic. Their convolution kernels
            # remain compiled below the eager cache plumbing. GroupNorm has
            # no shape-dependent indexing and is safe to compile as a leaf.
            wrapped = 0
            for leaf in module.modules():
                if isinstance(leaf, CachedCausalConv3d):
                    wrapped += wrap_inner_conv(leaf)
                elif isinstance(leaf, CachedGroupNorm):
                    wrapped += wrap(leaf)
            return wrapped

        # ``regions`` is the default dynamic boundary. Cache-aware wrappers
        # are deliberately graph-break points; otherwise Dynamo inlines their
        # symbolic temporal slices into the resblock and Inductor can emit an
        # invalid range such as [0:-1]. Resblocks themselves contain the
        # compute-heavy norm/activation/residual regions to compile. GroupNorm
        # is also wrapped so the disabled spatial-norm path keeps its math
        # compiled when it invokes the nested normalization module.
        cache_types = (
            CachedCausalConv3d,
            CachedPXSDownsample,
            CachedPXSUpsample,
            CachedSpatialNorm3D,
        )
        for leaf in module.modules():
            if isinstance(leaf, cache_types) and not getattr(leaf, _DYNAMIC_CACHE_EAGER_FLAG, False):
                leaf.forward = torch.compiler.disable(
                    leaf.forward,
                    recursive=False,
                    reason="KVAE dynamic cache/slice boundary",
                )
                setattr(leaf, _DYNAMIC_CACHE_EAGER_FLAG, True)

        wrapped = 0
        for leaf in module.modules():
            if isinstance(leaf, CachedGroupNorm):
                wrapped += wrap(leaf)
        if os.environ.get("KVAE_COMPILE_RESBLOCKS", "1") == "1":
            for block in module.modules():
                if isinstance(block, CachedCausalResnetBlock3D):
                    wrapped += wrap(block)
        return wrapped

    wrapped = 0
    inside_resblocks: set[int] = set()
    if os.environ.get("KVAE_COMPILE_RESBLOCKS", "1") == "1":
        # Region mode (DEFAULT): compile whole resblocks — norm+silu+conv
        # chains fuse into fewer, larger kernels with fewer graph breaks — and
        # skip their inner leaves (nesting compiled fns would just duplicate
        # the same regions). Measured on the SR assess pipeline this is the
        # single biggest decode win: 178.6s (eager) / 135s (per-leaf) → 90.8s
        # per 5s 512x768 video. Set KVAE_COMPILE_RESBLOCKS=0 for the per-leaf
        # granularity.
        for block in module.modules():
            if isinstance(block, CachedCausalResnetBlock3D):
                wrapped += wrap(block)
                inside_resblocks.update(id(sub) for sub in block.modules() if sub is not block)
    for leaf in module.modules():
        if isinstance(leaf, _COMPILED_LEAF_TYPES) and id(leaf) not in inside_resblocks:
            wrapped += wrap(leaf)
    return wrapped


class CompiledCachedCausalVAE(CachedCausalVAE):
    """``CachedCausalVAE`` with compiled leaf forwards and channels-last convs.

    Weight-compatible with the eager class (same submodule tree — only bound
    ``forward``s are wrapped), so ``init_from_ckpt`` and the ``vae.config``
    convention are inherited unchanged.
    """

    def compile_for_inference(self) -> None:
        """Convert decoder convs to ``channels_last_3d`` and wrap the leaves.

        Call AFTER weights are loaded and the model is cast to its inference
        dtype — compiled artifacts are specialized to parameter dtype/layout.
        The encoder stays eager by default (see
        :func:`kvae_encoder_compile_enabled` — compiling it measured SLOWER);
        set ``KVAE_COMPILE_ENCODER=1`` to include it.
        """
        self.decoder = torch.nn.utils.convert_conv3d_weight_memory_format(self.decoder, torch.channels_last_3d)
        n_dec = compile_kvae_leaves(self.decoder)
        n_enc = 0
        if kvae_encoder_compile_enabled():
            self.encoder = torch.nn.utils.convert_conv3d_weight_memory_format(self.encoder, torch.channels_last_3d)
            n_enc = compile_kvae_leaves(self.encoder)
        logger.info(
            "Compiled KVAE leaves: decoder={}, encoder={} (autotune={})",
            n_dec,
            n_enc,
            "on" if os.environ.get("KVAE_COMPILE_AUTOTUNE", "0") == "1" else "off",
        )

    # Many leaf instances share the same forward code object but differ in
    # channel widths, so dynamo accumulates one specialization per (instance,
    # segment-shape) — easily past the default recompile limit. Mirror the
    # Lift the limit around both halves.

    def encode(self, x: torch.Tensor, seg_len: int = 16):  # noqa: ANN201 — mirrors the eager base signature
        """Segment-cached encode under a lifted dynamo cache limit."""
        with torch._dynamo.utils.disable_cache_limit():  # noqa: SLF001 — no public API for this
            return super().encode(x, seg_len)

    def decode(self, z: torch.Tensor, split_list: list[int] | None = None):  # noqa: ANN201
        """Segment-cached decode under a lifted dynamo cache limit.

        ``KVAE_DECODE_SEG`` (pixel frames per temporal segment, default 16 —
        identical to the eager auto-split) enlarges the decoder segments:
        e.g. ``32`` halves the number of decoder invocations per call at the
        cost of ~2x decoder activation memory. Larger kernels utilize the GPU
        better; tune against available VRAM.
        """
        if split_list is None:
            tc = int(self.conf["enc"]["temporal_compress_times"])
            split_list = decode_split_list(z.shape[2], tc, int(os.environ.get("KVAE_DECODE_SEG", "16")))
        with torch._dynamo.utils.disable_cache_limit():  # noqa: SLF001 — no public API for this
            return super().decode(z, split_list)


def build_compiled_kvae(conf: object) -> CompiledCachedCausalVAE:
    """Build the compiled causal video KVAE from a ``vae:`` config section.

    Mirrors the ``video-kvae`` branch of ``model.vae.build_vae`` (sidecar
    ``{checkpoint_path}.yaml`` for the architecture + ``.ckpt`` weights,
    ``vae.config`` exposing ``scaling_factor``), then applies the inference
    compilation.

    Args:
        conf: Configuration with ``name == "video-kvae"`` and
            ``checkpoint_path`` attributes.

    Returns:
        Compiled KVAE in eval mode, bfloat16, with ``config`` attached.
    """
    vae_config = OmegaConf.load(f"{conf.checkpoint_path}.yaml")  # type: ignore[attr-defined]
    enc_params = vae_config.encoder_params if "encoder_params" in vae_config else vae_config.model.encoder_params
    dec_params = vae_config.decoder_params if "decoder_params" in vae_config else vae_config.model.decoder_params
    kvae = CompiledCachedCausalVAE(encoder_conf=enc_params, decoder_conf=dec_params)
    kvae.init_from_ckpt(kvae_weights_path(str(conf.checkpoint_path)))  # type: ignore[attr-defined]
    kvae = kvae.eval().bfloat16()
    kvae.config = vae_config
    kvae.compile_for_inference()
    return kvae
