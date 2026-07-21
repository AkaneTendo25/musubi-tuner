"""Ahead-of-time caching of the Cosmos3 reasoner (understanding) tower K/V.

The und ("reasoner") tower is a closed text-only sub-network.  In
``two_way_attention`` the causal branch reads only causal K/V::

    causal_res = _packed_sdpa(causal_q, causal_k, causal_v, is_causal=True)

so und activations never see gen tokens.  Combined with the packer placing text
first from mrope offset 0 (``sequence_packing._pack_text_tokens``), and with the
timestep embedding being applied only in ``_encode_vision``/``_encode_action``/
``_encode_sound``, the und K/V of every layer is a pure function of the text
token ids.  It does not depend on the noisy latent, the timestep, the video
resolution, or the frame count.

That makes it cacheable ahead of time and replayable for every training step and
every bucket.  Because the only consumer of an und activation is the *next*
layer's und K/V -- which is also cached -- no und hidden state is ever needed, so
the ~8B und parameters can be skipped at load time entirely.

How the substitution works
--------------------------
``get_all_seq`` scatters ``causal_seq`` and ``full_only_seq`` into a joint tensor
by absolute index, so writing cached und K/V into the causal slots lets the stock
``two_way_attention`` compute the correct gen output unchanged.  The und *output*
of attention is discarded: ``predict_text_tokens`` is False, so nothing
downstream reads und positions.

Patching happens at runtime so the vendored NVIDIA sources stay untouched,
matching how block swap is grafted on in ``cosmos3_utils.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Optional

import torch

from musubi_tuner.cosmos3.cosmos_framework.data.vfm.sequence_packing import (
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
)
from musubi_tuner.cosmos3.cosmos_framework.model.vfm.mot import attention as _attention_mod
from musubi_tuner.cosmos3.cosmos_framework.model.vfm.mot import unified_mot as _unified_mot
from musubi_tuner.cosmos3.cosmos_framework.model.vfm.utils.memory import (
    MemoryState,
    MemoryValue,
)

CACHE_FORMAT_VERSION = 1

# Weight-name fragments belonging to the und (reasoner) tower.  The generation
# tower carries the ``_moe_gen`` suffix; see the tower split documented on
# ``MoTDecoderLayer`` in unified_mot.py.
_UND_LAYER_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "mlp.",
    "input_layernorm",
    "post_attention_layernorm",
)


def is_und_layer_weight(key: str) -> bool:
    """True when *key* is a per-layer und (reasoner) tower weight.

    Only per-layer weights are considered.  Model-level und weights
    (``embed_tokens``, ``norm``, ``lm_head``) are excluded: ``norm`` is still
    applied to a length-0 und sequence in ``_impl_forward`` and so must remain
    materialized, and it is negligible in size.
    """
    if ".layers." not in key:
        return False
    if "_moe_gen" in key:
        return False
    return any(fragment in key for fragment in _UND_LAYER_SUFFIXES)


def text_ids_cache_key(input_ids: torch.Tensor, enable_fps_modulation: bool) -> str:
    """Stable cache key for a text token sequence.

    ``enable_fps_modulation`` is included because it selects float vs long mrope
    position ids for text tokens (see ``_pack_text_tokens``), which changes
    the RoPE applied to the cached keys.
    """
    ids = input_ids.detach().to("cpu", torch.int64).flatten().numpy().tobytes()
    digest = hashlib.sha256(ids)
    digest.update(b"|fps_mod=1" if enable_fps_modulation else b"|fps_mod=0")
    return digest.hexdigest()


@dataclasses.dataclass
class ReasonerKV:
    """Per-layer und K/V for one text sequence.

    ``keys`` are post-RoPE (applied in ``PackedAttentionMoT.forward`` before the
    K/V is captured) and ``values`` are un-rotated.  Both are stored unpadded,
    length ``num_text_tokens``.
    """

    keys: list[torch.Tensor]  # per layer, [N_und, num_kv_heads, head_dim]
    values: list[torch.Tensor]  # per layer, [N_und, num_kv_heads, head_dim]
    num_text_tokens: int
    cache_key: str

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def to(self, device, dtype=None) -> "ReasonerKV":
        return ReasonerKV(
            keys=[k.to(device=device, dtype=dtype or k.dtype) for k in self.keys],
            values=[v.to(device=device, dtype=dtype or v.dtype) for v in self.values],
            num_text_tokens=self.num_text_tokens,
            cache_key=self.cache_key,
        )

    def to_state_dict(self) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for i, (k, v) in enumerate(zip(self.keys, self.values)):
            sd[f"und_k.{i}"] = k
            sd[f"und_v.{i}"] = v
        return sd

    @staticmethod
    def from_state_dict(sd: dict[str, torch.Tensor], cache_key: str) -> "ReasonerKV":
        num_layers = sum(1 for key in sd if key.startswith("und_k."))
        keys = [sd[f"und_k.{i}"] for i in range(num_layers)]
        values = [sd[f"und_v.{i}"] for i in range(num_layers)]
        return ReasonerKV(
            keys=keys,
            values=values,
            num_text_tokens=int(keys[0].shape[0]) if keys else 0,
            cache_key=cache_key,
        )


@dataclasses.dataclass
class _CaptureSentinel(MemoryValue):
    """Non-None ``MemoryValue`` used purely to make the vendored attention emit
    ``kv_to_store``.  Carries no tensors."""


class ReasonerKVCapture(MemoryState):
    """``MemoryState`` that records und K/V per layer and nothing else.

    Uses the vendored producer path exactly as designed: returning a non-None
    ``MemoryValue`` from ``read_for_layer`` makes each attention block emit
    ``(gen_k, gen_v, und_k, und_v)``, which ``_impl_forward`` hands to
    ``write_for_layer``.
    """

    def __init__(self, num_layers: int, store_dtype: torch.dtype = torch.bfloat16):
        self._sentinel = _CaptureSentinel()
        self._store_dtype = store_dtype
        self.keys: list[Optional[torch.Tensor]] = [None] * num_layers
        self.values: list[Optional[torch.Tensor]] = [None] * num_layers

    def init(self, hidden_states: dict, device: torch.device) -> None:  # noqa: D102
        return None

    def read_for_layer(self, layer_idx: int) -> MemoryValue:  # noqa: D102
        return self._sentinel

    def write_for_layer(self, layer_idx: int, kv_to_store) -> None:  # noqa: D102
        _gen_k, _gen_v, und_k, und_v = kv_to_store
        # und_k/und_v arrive as [1, und_len, num_kv_heads, head_dim], already
        # sliced to the unpadded length by the producer.
        self.keys[layer_idx] = und_k[0].detach().to("cpu", self._store_dtype)
        self.values[layer_idx] = und_v[0].detach().to("cpu", self._store_dtype)

    def is_gen_only(self) -> bool:  # noqa: D102
        # False: the capture pass runs the real und tower to produce the K/V.
        return False

    def to_reasoner_kv(self, cache_key: str) -> ReasonerKV:
        missing = [i for i, k in enumerate(self.keys) if k is None]
        if missing:
            raise RuntimeError(
                f"Reasoner K/V capture incomplete: no K/V recorded for layers {missing}. "
                "The forward pass did not reach every decoder layer."
            )
        keys = [k for k in self.keys if k is not None]
        values = [v for v in self.values if v is not None]
        return ReasonerKV(
            keys=keys,
            values=values,
            num_text_tokens=int(keys[0].shape[0]),
            cache_key=cache_key,
        )


# --------------------------------------------------------------------------
# Runtime patching
# --------------------------------------------------------------------------

_ORIGINAL_DISPATCH_ATTENTION = None
_ORIGINAL_ATTENTION_FORWARD = None
_ORIGINAL_LAYER_FORWARD = None

# Incremented every time the cached-K/V branch is taken.  Because a correct
# replay is bit-identical to the stock forward, output comparison alone cannot
# distinguish "replay works" from "patch never engaged"; this counter can.
_REPLAY_CALLS = 0


def _patched_dispatch_attention(
    packed_query_states,
    packed_key_states,
    packed_value_states,
    attention_mask,
    natten_metadata=None,
    memory_value=None,
):
    """``dispatch_attention`` with the ``memory_value is None`` assert removed.

    The vendored ``dispatch_attention`` asserts that ``memory_value`` is None.
    Both capture and replay carry K/V outside the attention call itself, so the
    value is inert here and is dropped.
    """
    return _ORIGINAL_DISPATCH_ATTENTION(
        packed_query_states,
        packed_key_states,
        packed_value_states,
        attention_mask,
        natten_metadata=natten_metadata,
        memory_value=None,
    )


def _patched_attention_forward(
    self,
    pack,
    attention_mask,
    packed_position_embeddings,
    natten_metadata=None,
    memory_value=None,
):
    """``PackedAttentionMoT.forward`` with optional cached-und-K/V substitution.

    When ``self._cached_und_kv`` is set, the und projections are skipped and the
    cached K/V is written into the causal slots.  ``q_und`` is a zero tensor: the
    causal attention branch still runs but its output feeds only the und MLP,
    which the patched layer forward discards.
    """
    cached = getattr(self, "_cached_und_kv", None)
    if cached is None:
        return _ORIGINAL_ATTENTION_FORWARD(
            self,
            pack,
            attention_mask,
            packed_position_embeddings,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
        )

    global _REPLAY_CALLS
    _REPLAY_CALLS += 1

    cached_k, cached_v = cached
    und_len = int(pack["_num_causal_tokens"])
    if cached_k.shape[0] != und_len:
        raise RuntimeError(
            f"Cached reasoner K/V length {cached_k.shape[0]} does not match the packed "
            f"causal length {und_len}. The cache was built for a different text sequence."
        )

    gen_hidden = get_gen_seq(pack)
    q_gen_in = self.q_proj_moe_gen(gen_hidden)
    k_gen_in = self.k_proj_moe_gen(gen_hidden)
    v_gen_in = self.v_proj_moe_gen(gen_hidden)

    q_gen = q_gen_in.view(-1, self.num_attention_heads, self.head_dim)
    k_gen = k_gen_in.view(-1, self.num_key_value_heads, self.head_dim)
    v_gen = v_gen_in.view(-1, self.num_key_value_heads, self.head_dim)

    q_gen = self.q_norm_moe_gen(q_gen)
    k_gen = self.k_norm_moe_gen(k_gen)

    packed_cos, packed_sin = packed_position_embeddings
    q_gen_, k_gen_ = self._apply_rotary_pos_emb(
        q_gen,
        k_gen,
        get_gen_seq(packed_cos),
        get_gen_seq(packed_sin),
        unsqueeze_dim=1,
    )

    # Cached K/V is post-RoPE for keys and un-rotated for values, matching what
    # the producer stored.
    k_und_ = cached_k.to(device=k_gen_.device, dtype=k_gen_.dtype)
    v_und = cached_v.to(device=v_gen.device, dtype=v_gen.dtype)
    q_und_ = q_gen_.new_zeros(und_len, self.num_attention_heads, self.head_dim)

    packed_query_states_ = from_und_gen_splits(q_und_, q_gen_, pack)
    packed_key_states_ = from_und_gen_splits(k_und_, k_gen_, pack)
    packed_value_states_ = from_und_gen_splits(v_und, v_gen, pack)

    packed_attn_output, _ = self.dispatch_attention_fn(
        packed_query_states_,
        packed_key_states_,
        packed_value_states_,
        attention_mask,
        natten_metadata=natten_metadata,
        memory_value=None,
    )

    # Only the gen projection is applied; the und branch output is discarded by
    # the patched layer forward, and o_proj (und) is not loaded in replay mode.
    gen_seq = self.o_proj_moe_gen(get_gen_seq(packed_attn_output))
    und_seq = gen_seq.new_zeros(und_len, self.hidden_size)
    return from_und_gen_splits(und_seq, gen_seq, pack), None


def _patched_layer_forward(
    self,
    input,
    attention_mask,
    packed_position_embeddings,
    natten_metadata=None,
    memory_value=None,
    gen_only=False,
):
    """``MoTDecoderLayer._forward`` that skips the und tower when K/V is cached."""
    cached = getattr(self.self_attn, "_cached_und_kv", None)
    if cached is None:
        return _ORIGINAL_LAYER_FORWARD(
            self,
            input,
            attention_mask,
            packed_position_embeddings,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
            gen_only=gen_only,
        )

    gen_hidden = get_gen_seq(input)
    und_hidden = get_und_seq(input)

    pack_norm_out = from_und_gen_splits(
        und_hidden,  # passed through unnormalized; unused by the attention branch
        self.input_layernorm_moe_gen(gen_hidden),
        input,
    )

    pack_attn_out, _ = self.self_attn(
        pack_norm_out,
        attention_mask,
        packed_position_embeddings,
        natten_metadata=natten_metadata,
        memory_value=None,
    )

    residual_gen = gen_hidden + get_gen_seq(pack_attn_out)

    ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)
    gen_len = pack_attn_out["_num_full_tokens"]
    ln_out_gen_unpadded = ln_out_gen[:gen_len]

    mlp_out_gen_unpadded, lbl_metadata_gen = _unified_mot._run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)
    mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)

    lbl_metadata_dict = {}
    if lbl_metadata_gen is not None:
        lbl_metadata_dict["gen"] = lbl_metadata_gen

    mlp_out_gen_seq = residual_gen + mlp_out_gen
    # und carried through untouched; nothing downstream reads it because
    # predict_text_tokens is False.
    mlp_out_und_seq = und_hidden

    return from_und_gen_splits(mlp_out_und_seq, mlp_out_gen_seq, input), lbl_metadata_dict, None


def install_patches() -> None:
    """Install the runtime patches.  Idempotent."""
    global _ORIGINAL_DISPATCH_ATTENTION, _ORIGINAL_ATTENTION_FORWARD, _ORIGINAL_LAYER_FORWARD

    if _ORIGINAL_DISPATCH_ATTENTION is None:
        _ORIGINAL_DISPATCH_ATTENTION = _attention_mod.dispatch_attention
        _attention_mod.dispatch_attention = _patched_dispatch_attention
        # PackedAttentionMoT binds dispatch_attention at __init__ time
        # in its __init__, so already-built modules need rebinding too.
        _unified_mot.dispatch_attention = _patched_dispatch_attention

    if _ORIGINAL_ATTENTION_FORWARD is None:
        _ORIGINAL_ATTENTION_FORWARD = _unified_mot.PackedAttentionMoT.forward
        _unified_mot.PackedAttentionMoT.forward = _patched_attention_forward

    if _ORIGINAL_LAYER_FORWARD is None:
        _ORIGINAL_LAYER_FORWARD = _unified_mot.MoTDecoderLayer._forward
        _unified_mot.MoTDecoderLayer._forward = _patched_layer_forward


def rebind_dispatch(transformer: torch.nn.Module) -> None:
    """Point every attention module's bound ``dispatch_attention_fn`` at the patch."""
    for module in transformer.modules():
        if isinstance(module, _unified_mot.PackedAttentionMoT):
            module.dispatch_attention_fn = _patched_dispatch_attention


def cache_path_for(cache_dir, cache_key: str):
    """Content-addressed path for *cache_key*, sharded by the first two hex chars
    to keep any single directory small."""
    from pathlib import Path

    return Path(cache_dir) / cache_key[:2] / f"{cache_key}.safetensors"


def save_reasoner_kv(cache_dir, rkv: ReasonerKV, caption: Optional[str] = None) -> "object":
    """Write *rkv* to the content-addressed store.  Returns the path written."""
    from safetensors.torch import save_file

    path = cache_path_for(cache_dir, rkv.cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": str(CACHE_FORMAT_VERSION),
        "cache_key": rkv.cache_key,
        "num_layers": str(rkv.num_layers),
        "num_text_tokens": str(rkv.num_text_tokens),
    }
    if caption is not None:
        metadata["caption"] = caption[:1000]
    save_file(rkv.to_state_dict(), str(path), metadata=metadata)
    return path


class ReasonerKVStore:
    """Reads cached und K/V by content hash, with a small in-RAM LRU.

    Entries are a few MiB each, so a modest LRU removes almost all disk traffic
    for datasets whose captions repeat while keeping memory bounded.
    """

    def __init__(self, cache_dir, device=None, dtype=None, lru_size: int = 32):
        from collections import OrderedDict

        self.cache_dir = cache_dir
        self.device = device
        self.dtype = dtype
        self.lru_size = lru_size
        self._lru = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, cache_key: str) -> ReasonerKV:
        if cache_key in self._lru:
            self._lru.move_to_end(cache_key)
            self.hits += 1
            return self._lru[cache_key]

        from safetensors.torch import load_file

        path = cache_path_for(self.cache_dir, cache_key)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached reasoner K/V for key {cache_key} at {path}. "
                "Run cosmos3_cache_reasoner_kv.py over this dataset, and make sure the "
                "text-encoder cache it was built from matches the one training is using."
            )
        sd = load_file(str(path))
        rkv = ReasonerKV.from_state_dict(sd, cache_key)
        if self.device is not None or self.dtype is not None:
            rkv = rkv.to(self.device, self.dtype)

        self._lru[cache_key] = rkv
        self._lru.move_to_end(cache_key)
        while len(self._lru) > self.lru_size:
            self._lru.popitem(last=False)
        self.misses += 1
        return rkv


def sample_prompt_cache_entries(args, sample_prompts_path: str, vae_scale_factor_temporal: int):
    """Every (cond, uncond) token sequence that in-training sampling will request.

    Single source of truth shared by the cache producer and the trainer's
    startup verification, so the two cannot drift apart.  The expressions here
    mirror ``cosmos3_generate_video.sample_one`` (the ``use_system_prompt`` /
    ``add_*_template`` / ``negative_metadata_mode`` block) rather than being
    re-derived, because the cache key is the tokenizer output and any difference
    -- the system-prompt wrapper, the negative-prompt fallback, the fps -- yields
    a different key and a cache miss at sample time.

    The prompt dicts come from ``Cosmos3NetworkTrainer.process_sample_prompts``
    itself rather than from a reimplementation, because the defaults it applies
    are not obvious: ``line_to_prompt_dict`` never parses ``--n`` (that branch is
    commented out), so the negative prompt is always the default from
    ``neg_prompts.json`` regardless of what the prompt file says, and
    ``use_system_prompt`` is forced to False.  Calling it keeps this in lockstep.

    Returns a list of ``(ids_tensor, label)``.
    """
    from musubi_tuner.cosmos3 import cosmos3_utils
    from musubi_tuner.cosmos3_train_network import Cosmos3NetworkTrainer

    tokenizer_path = args.tokenizer if getattr(args, "tokenizer", None) is not None else args.dit
    tokenizer = cosmos3_utils.load_tokenizer(tokenizer_path, args.tokenizer_subfolder)

    # process_sample_prompts touches neither self nor accelerator.
    prompts = Cosmos3NetworkTrainer.process_sample_prompts(None, args, None, sample_prompts_path)

    entries = []
    for prompt in prompts:
        fps = float(prompt.get("fps", args.fps))
        negative_prompt = prompt.get("negative_prompt", "")
        use_system_prompt = bool(
            prompt.get(
                "use_system_prompt",
                getattr(args, "system_prompt", False) and not getattr(args, "no_system_prompt", False),
            )
        )
        add_resolution_template = bool(
            prompt.get("add_resolution_template", not getattr(args, "no_resolution_template", False))
        )
        add_duration_template = bool(
            prompt.get("add_duration_template", not getattr(args, "no_duration_template", False))
        )
        negative_metadata_mode = str(
            prompt.get(
                "negative_metadata_mode",
                getattr(args, "negative_metadata_mode", cosmos3_utils.DEFAULT_NEGATIVE_METADATA_MODE),
            )
        )

        width, height, frame_count = cosmos3_utils.normalize_sample_dimensions(
            prompt.get("width", 256),
            prompt.get("height", 256),
            prompt.get("frame_count", 1),
            vae_scale_factor_temporal,
        )

        cond_ids, uncond_ids = cosmos3_utils.tokenize_prompt(
            tokenizer,
            prompt=prompt.get("prompt", ""),
            negative_prompt=negative_prompt,
            num_frames=frame_count,
            height=height,
            width=width,
            fps=fps,
            use_system_prompt=use_system_prompt,
            add_resolution_template=add_resolution_template,
            add_duration_template=add_duration_template,
            negative_metadata_mode=negative_metadata_mode,
        )
        text = prompt.get("prompt", "")[:70]
        entries.append((torch.tensor(cond_ids, dtype=torch.int64), f"[sample cond] {text}"))
        entries.append((torch.tensor(uncond_ids, dtype=torch.int64), f"[sample uncond] {text}"))
    return entries


_ORIGINAL_RUN_TRANSFORMER = None
_AUTO_ATTACH = None  # (transformer, store, fps_modulation) or None


def install_auto_attach(transformer, store: "ReasonerKVStore", fps_modulation: bool) -> None:
    """Attach cached und K/V automatically inside ``run_transformer_for_sample``.

    Every path that drives the model -- the training step, in-training sampling,
    and standalone generation -- funnels through that one function, and each call
    carries the ``input_ids`` the cache is keyed on.  Hooking there covers all of
    them uniformly instead of threading an attach call through each caller, and
    it means the conditional (cond) and unconditional (uncond) halves of CFG each
    pick up their own cache entry without the sampler knowing anything about it.
    """
    global _ORIGINAL_RUN_TRANSFORMER, _AUTO_ATTACH
    from musubi_tuner.cosmos3 import cosmos3_utils

    _AUTO_ATTACH = (transformer, store, fps_modulation)

    if _ORIGINAL_RUN_TRANSFORMER is not None:
        return

    _ORIGINAL_RUN_TRANSFORMER = cosmos3_utils.run_transformer_for_sample

    def _wrapped(transformer_arg, input_ids, *args, **kwargs):
        if _AUTO_ATTACH is not None:
            target, active_store, fps_mod = _AUTO_ATTACH
            key = text_ids_cache_key(torch.as_tensor(input_ids), fps_mod)
            attach_cached_kv(target, active_store.get(key))
        return _ORIGINAL_RUN_TRANSFORMER(transformer_arg, input_ids, *args, **kwargs)

    cosmos3_utils.run_transformer_for_sample = _wrapped


def clear_auto_attach() -> None:
    global _AUTO_ATTACH
    _AUTO_ATTACH = None


def prune_und_layer_weights(model: torch.nn.Module) -> int:
    """Drop the per-layer und (reasoner) parameters from *model*.

    Must run on a meta-device model *before* weight loading.  Skipping
    those keys during load is not enough: ``_materialize_remaining_meta_tensors``
    would turn every unloaded meta parameter into a real empty on-device tensor,
    allocating the very memory this is meant to save.  Removing the parameter
    entries also removes them from ``state_dict()``, so the loader's strict
    missing-key check does not fire.

    Returns the number of parameter tensors removed.
    """
    removed = 0
    for module_prefix, module in model.named_modules():
        for name in list(module._parameters.keys()):
            full_name = f"{module_prefix}.{name}" if module_prefix else name
            if is_und_layer_weight(full_name):
                module._parameters[name] = None
                removed += 1
    return removed


def load_transformer_gen_only(
    model_path: str,
    transformer_subfolder: Optional[str],
    dtype,
    loading_device,
) -> tuple[torch.nn.Module, int]:
    """Load the transformer with the und (reasoner) tower omitted.

    Only valid together with a cached-K/V replay: with these weights absent the
    und pathway cannot be computed, so ``attach_cached_kv`` must supply K/V for
    every layer before any forward pass.
    """
    from musubi_tuner.cosmos3 import cosmos3_utils

    cosmos3_utils.validate_transformer_checkpoint_files(model_path, transformer_subfolder)
    with torch.device("meta"):
        model = cosmos3_utils.build_native_transformer(model_path, transformer_subfolder)
    removed = prune_und_layer_weights(model)
    cosmos3_utils.load_native_transformer_weights(
        model, model_path, transformer_subfolder, dtype, loading_device
    )
    cosmos3_utils._reinitialize_rotary_buffers(model, loading_device)
    cosmos3_utils.patch_transformer_for_training(model)
    model.eval()
    return model, removed


def attach_cached_kv(transformer: torch.nn.Module, reasoner_kv: Optional[ReasonerKV]) -> None:
    """Attach (or clear, when *reasoner_kv* is None) per-layer cached und K/V."""
    layers = [m for m in transformer.modules() if isinstance(m, _unified_mot.PackedAttentionMoT)]
    if reasoner_kv is None:
        for module in layers:
            module._cached_und_kv = None
        return

    if len(layers) != reasoner_kv.num_layers:
        raise RuntimeError(
            f"Cached reasoner K/V has {reasoner_kv.num_layers} layers but the model has "
            f"{len(layers)} attention blocks."
        )
    for module, k, v in zip(layers, reasoner_kv.keys, reasoner_kv.values):
        module._cached_und_kv = (k, v)
