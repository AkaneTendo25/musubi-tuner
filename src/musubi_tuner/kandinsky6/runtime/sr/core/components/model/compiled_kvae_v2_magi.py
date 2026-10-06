"""MagiCompiler-compiled causal video KVAE — the "magi" counterpart of compiled_kvae.py.

``compiled_kvae.py`` compiles the decoder region-wise because whole-graph
compilation chokes on the temporal segment cache: a nested python dict of
per-layer padding tensors mutated across segments (``None`` on the first
segment, tensors afterwards). This module makes each SEGMENT forward a pure
function instead: the cache dict is rebuilt inside the traced function from an
explicit tuple of tensors and flattened back on exit, so MagiCompiler can
compile one fully static graph per ``(first-vs-warm, latent shape)`` pair —
for the standard SR decode that is first (5 latent frames), mid (4) and tail
(remainder) segments, one graph each.

The layer code (``cached_layers.py`` / ``cached_enc_dec.py``) is reused
unchanged — only the cache plumbing at the segment boundary is functional.
``is None`` branches inside the layers resolve at trace time because the first
and warm segments compile as separate graphs. The encoder stays eager (same
policy and measurements as ``compiled_kvae.py``).

Env knobs:

* ``KVAE_MAGI=0`` — escape hatch: keep ``--vae-backend magi`` on the
  functional-eager fallback (same numerics, no compilation);
* ``KVAE_DECODE_SEG`` — pixel frames per decoder temporal segment (default 16,
  same as ``compiled_kvae.py``).

Compiled artifacts are cached per ``model_tag``
(``{magi_tag}_{first|next}_{t}x{h}x{w}``) and survive process restarts.
MagiCompiler must be >= v1.1.0 — see ``magi_patch.py``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from loguru import logger
from omegaconf import OmegaConf

from musubi_tuner.kandinsky6.runtime.sr.core.components.model.vae_io import kvae_weights_path
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.compiled_kvae import decode_split_list
from musubi_tuner.kandinsky6.runtime.sr.core.components.video_kvae.cached_layers import CachedSpatialNorm3D
from musubi_tuner.kandinsky6.runtime.sr.core.components.video_kvae.cached_model import CachedCausalVAE, DecoderOutput

if TYPE_CHECKING:
    from musubi_tuner.kandinsky6.runtime.sr.core.components.video_kvae.cached_enc_dec import CachedDecoder3D

CachePath = tuple[object, ...]
SegmentFn = Callable[[torch.Tensor, tuple[torch.Tensor, ...]], tuple[torch.Tensor, tuple[torch.Tensor, ...]]]

MAGI_IMPORT_HINT = (
    "magi_compiler is not installed (>= v1.1.0 required). Install with:\n"
    '  pip install "kandinsky-6-sr[magi]" "setuptools>=61.0" wheel versioningit\n'
    '  pip install --no-build-isolation "magi_compiler @ '
    'git+https://github.com/SandAI-org/MagiCompiler.git@v1.1.0"\n'
    "Or use the region-compiled KVAE (musubi_tuner.kandinsky6.runtime.sr.model.compiled_kvae) instead."
)


def spatial_norm_cache_paths(base: CachePath, norm: torch.nn.Module) -> list[CachePath]:
    """Cache paths of one ``Normalize3D`` slot (its optional add_conv only)."""
    if isinstance(norm, CachedSpatialNorm3D) and norm.add_conv:
        return [(*base, "add_conv", "padding")]
    return []


def resblock_cache_paths(base: CachePath, block: torch.nn.Module) -> list[CachePath]:
    """Tensor-cache paths of one ``CachedCausalResnetBlock3D``, in forward order."""
    paths = spatial_norm_cache_paths((*base, "norm1"), block.norm1)
    paths.append((*base, "conv1", "padding"))
    paths += spatial_norm_cache_paths((*base, "norm2"), block.norm2)
    paths.append((*base, "conv2", "padding"))
    if hasattr(block, "conv_shortcut"):
        paths.append((*base, "conv_shortcut", "padding"))
    return paths


def decoder_cache_conv_paths(decoder: CachedDecoder3D) -> list[CachePath]:
    """Enumerate every decoder cache slot that carries a tensor across segments.

    Follows ``CachedDecoder3D.forward``'s traversal order over the cache dict
    built by ``CachedCausalVAE.make_empty_cache('dec')``. Slots that the
    module tree never writes (e.g. ``conv_shortcut`` when the block uses a
    ``nin_shortcut``, ``up`` on non-temporal upsamples) are excluded, so after
    any segment every listed path holds a tensor.

    Args:
        decoder: The cached decoder whose module tree defines the used slots.

    Returns:
        Ordered nested-key paths into the segment cache dict.
    """
    paths: list[CachePath] = [("conv_in", "padding")]
    paths += resblock_cache_paths(("mid_1",), decoder.mid.block_1)
    paths += resblock_cache_paths(("mid_2",), decoder.mid.block_2)
    for i_level in reversed(range(decoder.num_resolutions)):
        for i_block in range(decoder.num_res_blocks + 1):
            paths += resblock_cache_paths((i_level, i_block), decoder.up[i_level].block[i_block])
        if i_level != 0 and decoder.up[i_level].upsample.temporal_compress:
            paths.append((i_level, "up", "padding"))
    paths += spatial_norm_cache_paths(("norm_out",), decoder.norm_out)
    paths.append(("conv_out", "padding"))
    return paths


def cache_leaf_parent(cache: dict, path: CachePath) -> dict:
    """Return the dict holding the leaf slot addressed by ``path``."""
    node = cache
    for key in path[:-1]:
        node = node[key]
    return node


def mark_cache_warm(node: object) -> None:
    """Flip every ``mean``/``var`` first-segment sentinel to its warm value."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("mean", "var") and value is None:
                node[key] = 1
            else:
                mark_cache_warm(value)
    elif isinstance(node, list):
        for item in node:
            mark_cache_warm(item)


def clone_cache_template(node: object) -> object:
    """Deep-copy the (dict/list/None) cache template with plain recursion.

    ``make_empty_cache`` reads the OmegaConf model config, which dynamo cannot
    trace through — so the template is built once outside the compiled region
    and cloned with this dynamo-friendly walk inside it.
    """
    if isinstance(node, dict):
        return {key: clone_cache_template(value) for key, value in node.items()}
    if isinstance(node, list):
        return [clone_cache_template(item) for item in node]
    return node


class MagiCachedCausalVAE(CachedCausalVAE):
    """``CachedCausalVAE`` whose decoder runs as per-segment static graphs.

    Weight-compatible with the eager class (same submodule tree), so
    ``init_from_ckpt`` and the ``vae.config`` convention are inherited
    unchanged. Until :meth:`compile_for_inference` is called the segment
    functions run as plain python — the functional-eager fallback used by the
    CPU tests and the ``KVAE_MAGI=0`` escape hatch.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Build the VAE and precompute the cache spec/template (see class doc)."""
        super().__init__(*args, **kwargs)
        self.cache_spec = decoder_cache_conv_paths(self.decoder)
        self.cache_template = self.make_empty_cache("dec")
        self.magi_tag = "ksr_kvae"
        self.magi_enabled = False
        self.segment_fns: dict[tuple, SegmentFn] = {}

    def build_segment_cache(self, flat: tuple[torch.Tensor, ...] | None) -> dict:
        """Reconstruct the segment cache dict from its flat tensor form.

        Args:
            flat: Tensors in ``cache_spec`` order for a warm segment, or
                ``None`` for the first segment (every slot stays ``None``).

        Returns:
            A cache dict accepted by ``CachedDecoder3D.forward``.
        """
        cache = clone_cache_template(self.cache_template)
        if flat is not None:
            mark_cache_warm(cache)
            for path, tensor in zip(self.cache_spec, flat, strict=True):
                cache_leaf_parent(cache, path)[path[-1]] = tensor
        return cache

    def flatten_segment_cache(self, cache: dict) -> tuple[torch.Tensor, ...]:
        """Extract the tensor cache slots in ``cache_spec`` order."""
        flat = tuple(cache_leaf_parent(cache, path)[path[-1]] for path in self.cache_spec)
        missing = [path for path, value in zip(self.cache_spec, flat, strict=True) if not isinstance(value, torch.Tensor)]
        if missing:
            msg = f"decoder cache slots not written this segment: {missing}"
            raise RuntimeError(msg)
        return flat

    def make_segment_fn(self, *, first: bool) -> SegmentFn:
        """Build the pure per-segment decode function (the magi compile target)."""

        def kvae_segment(z: torch.Tensor, cache_flat: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            cache = self.build_segment_cache(None if first else cache_flat)
            out = self.decoder(z, cache)
            return out, self.flatten_segment_cache(cache)

        return kvae_segment

    def segment_runner(self, z: torch.Tensor, *, first: bool) -> SegmentFn:
        """Return the (lazily magi-compiled) segment function for this shape."""
        key = (first, tuple(z.shape))
        fn = self.segment_fns.get(key)
        if fn is None:
            fn = self.make_segment_fn(first=first)
            if self.magi_enabled:
                from magi_compiler import magi_compile  # noqa: PLC0415 — optional dependency

                shape_tag = "x".join(str(s) for s in z.shape[2:])
                tag = f"{self.magi_tag}_{'first' if first else 'next'}_{shape_tag}"
                fn = magi_compile(fn, model_tag=tag, dynamic_arg_dims={"z": []})
                logger.info("KVAE magi segment graph registered: {} (input {})", tag, tuple(z.shape))
            self.segment_fns[key] = fn
        return fn

    def decode(self, z: torch.Tensor, split_list: list[int] | None = None) -> DecoderOutput:
        """Segment-cached decode threading the cache as explicit tensors.

        Args:
            z: Latent batch ``(B, C, T, H, W)``.
            split_list: Optional segment sizes in PIXEL frames (the base-class
                convention); defaults to ``decode_split_list`` with
                ``KVAE_DECODE_SEG``.

        Returns:
            Decoded pixels wrapped in ``DecoderOutput``.
        """
        tc = int(self.conf["enc"]["temporal_compress_times"])
        if split_list is None:
            split_list = decode_split_list(z.shape[2], tc, int(os.environ.get("KVAE_DECODE_SEG", "16")))
        latent_split = [math.ceil(size / tc) for size in split_list]

        recs = []
        cache_flat: tuple[torch.Tensor, ...] = ()
        for idx, chunk in enumerate(torch.split(z, latent_split, dim=2)):
            out, cache_flat = self.segment_runner(chunk, first=idx == 0)(chunk, cache_flat)
            recs.append(out)
        return DecoderOutput(sample=torch.cat(recs, dim=2))

    def compile_for_inference(self) -> None:
        """Enable magi compilation of the segment functions.

        Call AFTER weights are loaded and the model is cast to its inference
        dtype. Applies the ``_mark_static_shapes`` patch (shared with the
        magi VAE) and switches :meth:`segment_runner` to wrap new
        segment shapes with ``magi_compile``. ``KVAE_MAGI=0`` keeps the
        functional-eager fallback.
        """
        if os.environ.get("KVAE_MAGI", "1") == "0":
            logger.info("KVAE_MAGI=0 — magi wrapping disabled, functional-eager KVAE decode")
            return
        try:
            import magi_compiler  # noqa: F401, PLC0415 — availability probe
        except ImportError as exc:
            raise ImportError(MAGI_IMPORT_HINT) from exc

        from musubi_tuner.kandinsky6.runtime.sr.core.components.model.magi_patch import patch_magi_mark_static_shapes  # noqa: PLC0415

        patch_magi_mark_static_shapes(mark_owner=False)
        self.magi_enabled = True
        logger.info("KVAE magi compile enabled (tag prefix '{}', {} cache slots)", self.magi_tag, len(self.cache_spec))


def build_magi_compiled_kvae(conf: object) -> MagiCachedCausalVAE:
    """Build the MagiCompiler-compiled causal video KVAE from a ``vae:`` config.

    Mirrors ``compiled_kvae.build_compiled_kvae`` (sidecar
    ``{checkpoint_path}.yaml`` for the architecture + ``.ckpt`` weights,
    ``vae.config`` exposing ``scaling_factor``), swapping the region-compiled
    class for the magi segment-graph one. ``conf.magi_tag`` (default
    ``"ksr_kvae"``) keys the artifact cache.

    Args:
        conf: Configuration with ``name == "video-kvae"`` and
            ``checkpoint_path`` attributes.

    Returns:
        Magi-compiled KVAE in eval mode, bfloat16, with ``config`` attached.
    """
    if getattr(conf, "name", None) != "video-kvae":
        msg = f"Magi-compiled KVAE only supports 'video-kvae', got: {getattr(conf, 'name', None)}"
        raise ValueError(msg)
    vae_config = OmegaConf.load(f"{conf.checkpoint_path}.yaml")  # type: ignore[attr-defined]
    enc_params = vae_config.encoder_params if "encoder_params" in vae_config else vae_config.model.encoder_params
    dec_params = vae_config.decoder_params if "decoder_params" in vae_config else vae_config.model.decoder_params
    kvae = MagiCachedCausalVAE(encoder_conf=enc_params, decoder_conf=dec_params)
    kvae.init_from_ckpt(kvae_weights_path(str(conf.checkpoint_path)))  # type: ignore[attr-defined]
    kvae = kvae.eval().bfloat16()
    kvae.config = vae_config
    # KSR_MAGI_TAG namespaces the artifact cache.
    kvae.magi_tag = str(getattr(conf, "magi_tag", None) or os.environ.get("KSR_MAGI_TAG", "ksr_kvae"))
    kvae.compile_for_inference()
    return kvae
