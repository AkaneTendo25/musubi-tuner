"""LoRA training for YuE2 (m-a-p/YuE2-3B): AR (semantic codec LM), NAR (flow-matching acoustic stack) or both.

One trainer covers the three modes of ``--train_branches``. Per micro-batch every item is planned (prefix, NAR window,
context layout, AR ids), then all no-grad passes run (KL base pass, NAR context prefill), then one grad pass per stack
(AR cross-entropy/KL, NAR flow). Every musubi mechanism goes through the base trainer: two-list block swap, ConvRot
int8 / scaled fp8 bases, gradient checkpointing with activation CPU offload, per-block compile, attention backends,
sampling during training, save/resume. Adds validation losses with a best checkpoint and optional LoRA exports.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import time
from typing import Any, Optional

import torch
from accelerate import Accelerator

from musubi_tuner.dataset.architectures import ARCHITECTURE_YUE2, ARCHITECTURE_YUE2_FULL
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.cache_io import YUE2_LATENT_CACHE_VERSION, YUE2_TEXT_CACHE_VERSION
from musubi_tuner.hv_train_network import (
    DiTOutput,
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.networks import lora_yue2
from musubi_tuner.training.trainer_base import SS_METADATA_MINIMUM_KEYS, wandb_tracker_and_module
from musubi_tuner.utils import sai_model_spec, train_utils
from musubi_tuner.yue2 import yue2_lora_formats as lora_formats
from musubi_tuner.yue2.yue2_args import normalize_swap_counts, setup_parser_yue2_model, validate_yue2_model_args
from musubi_tuner.yue2.yue2_checkpoint import (
    CheckpointLayout,
    detect_layout,
    load_yue2_model,
    load_yue2_tokenizer,
    load_yue2_vae,
    read_base_io,
)
from musubi_tuner.yue2.yue2_model import YuE2Config, yue2_t_embed_input
from musubi_tuner.yue2.yue2_protocol import (
    ABC_DEFAULTS,
    COT_MODES,
    EOD,
    FRAME_RATE,
    MUSIC_END,
    ODE_STEPS,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
    SEMANTIC_DEFAULTS,
    YUE2_AUDIO_SPEC,
    build_negative_prefix,
    build_prefix,
    negative_text_ids,
    normalize_prompt_fields,
    text_ids,
)
from musubi_tuner.yue2.yue2_training import (
    AR_CE_TARGETS,
    COT_CHOICES,
    NAR_CONTEXTS,
    TEXT_ONLY_ROPE_MODES,
    ItemPlan,
    abc_mode_name,
    ar_sequence,
    ar_targets,
    chunked_ce_kl,
    flow_loss,
    item_rng,
    lora_eval,
    optimizer_eval,
    pad_right,
    parse_beta_ab,
    plan_item,
    sample_t,
)

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

YUE2_TIMESTEP_SAMPLING = ("uniform", "sigmoid", "shift", "logsnr", "beta")
YUE2_NETWORK_MODULES = ("networks.lora_yue2", "musubi_tuner.networks.lora_yue2")
SAMPLE_MODES = ("render", "reconstruct")
EXPORT_FORMATS = ("comfy", "hf", "fl")
EXPORT_SUFFIXES = (".comfy", ".hf-ar", ".hf-nar", ".fl-ar", ".fl-nar")
BEST_STATE_FILE = "yue2_best_validation.json"


def _export_formats(args) -> list[str]:
    return [f.strip() for f in (args.yue2_export_formats or "").split(",") if f.strip()]


def _base_multipliers(args) -> list[float]:
    """``--base_weights_multiplier`` padded with 1.0 to one value per ``--base_weights`` file."""
    n = len(args.base_weights or [])
    multipliers = [float(m) for m in (args.base_weights_multiplier or [])][:n]
    return multipliers + [1.0] * (n - len(multipliers))


def _parse_branches(value: str) -> tuple[str, ...]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts or any(p not in ("ar", "nar") for p in parts) or len(set(parts)) != len(parts):
        raise ValueError(f"--train_branches must be nar, ar or ar,nar; got {value!r}")
    return tuple(b for b in ("ar", "nar") if b in parts)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _item_inputs(batch: dict, i: int) -> dict[str, Any]:
    """Per-item planner inputs from a ``BucketBatchManager`` batch."""
    codes = batch.get("codes")
    return {
        "texts": {cot: batch[f"yue2_text_{cot}"][i].tolist() for cot in COT_MODES},
        "negatives": {cot: batch[f"yue2_neg_{cot}"][i].tolist() for cot in COT_MODES},
        "abc": batch["yue2_abc"][i].tolist(),
        "abc_mode": abc_mode_name(batch["yue2_abc_mode"][i]),
        "codes": None if codes is None else codes[i].tolist(),
        "seg": [int(v) for v in batch["yue2_seg"][i].tolist()],
    }


@dataclasses.dataclass
class YuE2SampleAudio:
    """A training sample: ``waveform [2, S]`` fp32 in [-1, 1] (+ the ground-truth decode of a reconstruct prompt)."""

    waveform: torch.Tensor
    sample_rate: int = SAMPLE_RATE
    meta: dict = dataclasses.field(default_factory=dict)
    ground_truth: Optional[torch.Tensor] = None


class YuE2SamplingResources(torch.nn.Module):
    """Sampling payload: the decoder-only fp32 VAE (a submodule, moved by the base) and the text tokenizer."""

    def __init__(self, vae: torch.nn.Module, tokenizer):
        super().__init__()
        self.vae = vae
        self.tokenizer = tokenizer


class YuE2Validator:
    """Validation losses on held-out songs: flow at fixed t with fixed noise on the centre window, AR CE/KL over the
    whole song with the score prefix when present, and CE on held-out minted songs."""

    def __init__(self, items: list, args: argparse.Namespace, minted_val=None):
        self.items = list(items)[: max(0, int(args.validation_max_items))]
        # same key normalisation as the training batches
        self.batches = [BucketBatchManager({(it.frame_count,): [it]}, 1)[0] for it in self.items]
        self.minted_val = minted_val
        self.timesteps = [float(t) for t in str(args.validation_timesteps).split(",") if t.strip()]
        if not self.timesteps or any(not 0.0 < t <= 1.0 for t in self.timesteps):
            raise ValueError(f"--validation_timesteps must be a comma list in (0, 1], got {args.validation_timesteps!r}")

    def __len__(self) -> int:
        return len(self.batches) + (len(self.minted_val) if self.minted_val is not None else 0)

    def run(self, trainer: "YuE2NetworkTrainer", args, accelerator, transformer, network, optimizer) -> dict[str, float]:
        model = accelerator.unwrap_model(transformer)
        net = accelerator.unwrap_model(network) if network is not None else None
        dev = accelerator.device
        train_ar, train_nar = "ar" in trainer.branches, "nar" in trainer.branches
        seed = int(args.validation_noise_seed)
        flows = {tv: [] for tv in self.timesteps}
        ces, kls, minted = [], [], []

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        try:
            with torch.no_grad(), lora_eval(net), optimizer_eval(optimizer), accelerator.autocast():
                model.begin_train_step()  # runs after backward: the grad-pass counters must be clean
                for j, batch in enumerate(self.batches):
                    latents = batch["latents"][0]
                    plan = trainer._plan(args, batch, 0, latents.shape[0], random.Random(seed + j), training=False)
                    if train_nar:
                        kv = trainer._context_kv(args, model, plan)
                        s, w = plan.window_start, plan.window_frames
                        z = latents[s : s + w].to(dev).float()
                        eps = torch.randn(w, z.shape[-1], generator=torch.Generator().manual_seed(seed + j)).to(dev)
                        for tv in self.timesteps:
                            x_t = (1.0 - tv) * z + tv * eps
                            t_emb = yue2_t_embed_input(torch.tensor([tv], dtype=torch.float64), args.t_embed_dtype)
                            v = model.nar_forward(x_t[None], t_emb.to(dev), kv, plan.rope_offset)
                            flows[tv].append(flow_loss(v[0], eps, z).item())
                        del kv
                    if train_ar and plan.ar_ids is not None:
                        ce, kl = trainer._ar_eval_losses(args, model, net, plan.ar_ids, plan.ar_target_start)
                        ces.append(ce)
                        if kl is not None:
                            kls.append(kl)
                if train_ar and self.minted_val is not None:
                    for k in range(len(self.minted_val)):
                        item = self.minted_val.item(k)
                        ids, target_start = trainer._minted_ar_ids(args, item, None)
                        ce, _ = trainer._ar_eval_losses(args, model, None, ids, target_start)
                        minted.append(ce)
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
            model.begin_train_step()

        results: dict[str, float] = {}
        total = 0.0
        if flows and any(flows.values()):
            per_t = {tv: sum(v) / len(v) for tv, v in flows.items() if v}
            for tv, value in per_t.items():
                results[f"val/flow@{tv:g}"] = value
            results["val/flow"] = sum(per_t.values()) / len(per_t)
            total += args.flow_loss_weight * results["val/flow"]
        if ces:
            results["val/ar_ce"] = sum(ces) / len(ces)
            total += args.ar_ce_weight * results["val/ar_ce"]
        if kls:
            results["val/ar_kl"] = sum(kls) / len(kls)
            total += args.ar_kl_weight * results["val/ar_kl"]
        if minted:
            results["val/minted_ce"] = sum(minted) / len(minted)
            if not (ces or kls or "val/flow" in results):
                # minted songs are the only validation data: rank checkpoints by their CE
                total = results["val/minted_ce"]
        if results:
            results["val/total"] = total
        return results


class YuE2NetworkTrainer(NetworkTrainer):
    audio_spec = YUE2_AUDIO_SPEC

    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1
        self.default_discrete_flow_shift = 1.0
        # model dimensions; tests replace this with a tiny config
        self.model_config = YuE2Config()
        self.branches: tuple[str, ...] = ("nar",)
        self._swap_counts = (0, 0)
        self._micro_index = 0
        self._validation_items: list = []
        self._train_items: list = []
        self._validator: Optional[YuE2Validator] = None
        self._minted_train = None
        self._minted_val = None
        self._minted_fingerprint: Optional[str] = None
        self._optimizer = None
        self._best_val: Optional[float] = None
        self._best_step: Optional[int] = None
        self._base_weights_merged_at_load = False
        self._base_weights_desc: Optional[str] = None
        self._checkpoint_layout: Optional[str] = None
        self._base_quant: Optional[str] = None
        self._convrot_int8_active = False
        self._attn_mode = "torch"
        self._cache_info: dict[str, Any] = {}
        self._dataset_group = None
        self._session_id: Optional[int] = None
        self._training_started_at: Optional[float] = None
        self._ar_skipped = 0
        self._epoch = 1
        self._gt_written: set[int] = set()

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_YUE2

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_YUE2_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = 1.0

        validate_yue2_model_args(args, training=True)
        self.branches = _parse_branches(args.train_branches)
        train_ar = "ar" in self.branches

        if args.network_module not in YUE2_NETWORK_MODULES:
            raise ValueError(f"YuE2 training requires --network_module networks.lora_yue2, got {args.network_module!r}")
        if args.dim_from_weights:
            raise ValueError("--dim_from_weights is not supported for YuE2; use --network_weights (any YuE2 LoRA format)")
        if args.timestep_sampling not in YUE2_TIMESTEP_SAMPLING:
            raise ValueError(f"YuE2 supports --timestep_sampling {', '.join(YUE2_TIMESTEP_SAMPLING)}, got {args.timestep_sampling}")
        if args.weighting_scheme != "none":
            raise ValueError("YuE2 supports only --weighting_scheme none")
        if args.timestep_sampling == "beta":
            parse_beta_ab(args.beta_timestep_ab)
            if args.show_timesteps:
                raise ValueError("--show_timesteps does not support --timestep_sampling beta")
            if args.num_timestep_buckets is not None and args.num_timestep_buckets > 1:
                raise ValueError("--num_timestep_buckets is not supported with --timestep_sampling beta")
        if args.nar_context not in NAR_CONTEXTS:
            raise ValueError(f"--nar_context must be one of {NAR_CONTEXTS}")
        if args.nar_window_frames < 0:
            raise ValueError("--nar_window_frames must be >= 0 (0 = whole song)")
        if args.ar_ce_chunk < 1:
            raise ValueError("--ar_ce_chunk must be >= 1")
        for name in (
            "abc_dropout",
            "caption_dropout",
            "nar_caption_dropout",
            "nar_codec_dropout",
            "ar_replay_fraction",
            "first_timestep_chance",
            "ar_minted_val_fraction",
        ):
            value = getattr(args, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name} must be in [0, 1], got {value}")
        if args.nar_codec_dropout > 0 and args.nar_context != "codes":
            raise ValueError("--nar_codec_dropout applies only to --nar_context codes")
        if args.ar_kl_weight > 0 and not train_ar:
            logger.warning("--ar_kl_weight has no effect without the ar branch; set to 0")
            args.ar_kl_weight = 0.0
        if args.ar_replay_fraction > 0 and (not train_ar or not args.ar_minted_pack):
            raise ValueError("--ar_replay_fraction needs the ar branch and --ar_minted_pack")
        if args.flow_loss_weight < 0 or args.ar_ce_weight < 0 or args.ar_kl_weight < 0:
            raise ValueError("YuE2 loss weights must be >= 0")
        if args.ar_pad_multiple is None:
            args.ar_pad_multiple = 256 if args.compile else 0
        if args.ar_pad_multiple < 0:
            raise ValueError("--ar_pad_multiple must be >= 0")
        formats = _export_formats(args)
        if any(f not in EXPORT_FORMATS for f in formats):
            raise ValueError(f"--yue2_export_formats must be a comma list of {EXPORT_FORMATS}")
        if args.yue2_export_include_base_weights and not args.base_weights:
            raise ValueError("--yue2_export_include_base_weights needs --base_weights")
        if args.validate_every_n_steps is not None and args.validate_every_n_steps < 1:
            raise ValueError("--validate_every_n_steps must be >= 1")
        if args.sample_prompts and args.vae is None and args.dit and os.path.isfile(args.dit):
            if detect_layout(args.dit)[0] != CheckpointLayout.COMFY:
                raise ValueError("YuE2 --sample_prompts needs a VAE: pass --vae (m-a-p/YuE2-Vae) or a ComfyUI --dit")
        if args.sample_ar_resident and not args.sample_prompts:
            logger.warning("--sample_ar_resident has no effect without --sample_prompts")

        self._swap_counts = normalize_swap_counts(args)
        self._inject_network_args(args)
        self._check_export_formats(args, formats)

    def _check_export_formats(self, args: argparse.Namespace, formats: list[str]) -> None:
        """Reject export formats that cannot hold what will be trained or included (else the first save would fail)."""
        net_kwargs = {k.strip(): v.strip() for k, _, v in (entry.partition("=") for entry in args.network_args)}
        time_embedder = lora_yue2._bool(net_kwargs.get("include_time_embedder", False))
        if "fl" in formats and time_embedder:
            raise ValueError("--yue2_export_formats fl has no time_embedder target; drop include_time_embedder or use comfy/hf")
        if "hf" not in formats:
            return
        if args.train_io == "full":
            raise ValueError("--yue2_export_formats hf cannot carry the I/O bias deltas of --train_io full; use comfy or fl")
        if args.yue2_export_include_base_weights:
            base_io = read_base_io(args.dit) if args.dit and os.path.isfile(args.dit) else None
            for path in args.base_weights:
                sd, metadata = lora_formats.load_lora_file(path)
                native = lora_formats.to_native(sd, metadata, base_io=base_io)
                for module, md in native.modules.items():
                    bias = md.bias_delta()
                    if bias is not None and bias.abs().max() > 0:
                        raise ValueError(
                            f"--base_weights {os.path.basename(path)} has an I/O bias delta ({module}) that"
                            " --yue2_export_formats hf cannot carry with --yue2_export_include_base_weights; use comfy or fl"
                        )

    def _inject_network_args(self, args: argparse.Namespace) -> None:
        """Trainer-level network flags become ``--network_args`` entries, so ``ss_network_args`` records them."""
        entries = list(args.network_args or [])
        present = {}
        for entry in entries:
            key, _, value = entry.partition("=")
            present[key.strip()] = value.strip()
        wanted = {"branches": ",".join(self.branches), "train_io": args.train_io, "ar_lr_ratio": str(args.ar_lr_ratio)}
        if args.io_lr is not None:
            wanted["io_lr"] = str(args.io_lr)
        for key, value in wanted.items():
            if key in present:
                given = present[key]
                same = given == value
                if not same and key in ("ar_lr_ratio", "io_lr"):
                    try:
                        same = float(given) == float(value)
                    except ValueError:
                        same = False
                if not same and key == "branches":
                    same = _parse_branches(given) == self.branches
                if not same:
                    raise ValueError(f"--network_args {key}={given} conflicts with the trainer flag value {value}")
                continue
            entries.append(f"{key}={value}")
        args.network_args = entries

    def _trained_swapped_stacks(self) -> list[str]:
        ar_n, nar_n = self._swap_counts
        return [name for name, n in (("ar", ar_n), ("nar", nar_n)) if n > 0 and name in self.branches]

    def _build_dataset(self, args):
        from musubi_tuner.dataset.audio_dataset import AudioDataset

        train_dataset_group, collator, current_epoch = super()._build_dataset(args)
        self._dataset_group = train_dataset_group
        self._validation_items, self._train_items = [], []
        for ds in train_dataset_group.datasets:
            if not isinstance(ds, AudioDataset):
                raise ValueError("YuE2 training needs audio datasets only (audio_directory / audio_jsonl_file)")
            self._validation_items.extend(ds.validation_items)
            seen = set()
            for bucket in ds.batch_manager.buckets.values():
                for item in bucket:
                    if item.latent_cache_path not in seen:
                        seen.add(item.latent_cache_path)
                        self._train_items.append(item)
        swapped = self._trained_swapped_stacks()
        if swapped and any(ds.batch_size > 1 for ds in train_dataset_group.datasets):
            raise ValueError(
                f"YuE2 with block swap on a trained stack ({', '.join(swapped)}) needs batch_size=1 in the dataset config;"
                " use --gradient_accumulation_steps for larger effective batches"
            )
        self._check_caches(args)
        if args.ar_minted_pack:
            self._load_minted_pack(args)
        return train_dataset_group, collator, current_epoch

    def _check_caches(self, args) -> None:
        """Header/metadata checks over every training and validation cache (fails before the model loads)."""
        from safetensors import safe_open

        need_codes = args.nar_context == "codes" or "ar" in self.branches
        missing_codes, cot_mismatch, tokenizers, lyrics_conv, mid_song = [], [], set(), set(), 0
        abc_modes = {"none": 0, "melody": 0, "full": 0}
        latent_meta: dict[str, str] = {}
        te_seen = set()
        heads = set()
        for item in self._train_items + self._validation_items:
            with safe_open(item.latent_cache_path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                meta = dict(f.metadata() or {})
                if not latent_meta:
                    latent_meta = meta
                seg = f.get_tensor("yue2_seg_int64").tolist() if "yue2_seg_int64" in keys else [0, 0, 0]
            if "codes_int64" in keys and meta.get("yue2_semantic_head"):
                heads.add(meta["yue2_semantic_head"])
            if need_codes and "codes_int64" not in keys:
                missing_codes.append(os.path.basename(item.latent_cache_path))
            if seg[0] != 0:
                mid_song += 1
            te = item.text_encoder_output_cache_path
            if te in te_seen:
                continue
            te_seen.add(te)
            with safe_open(te, framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
                mode = abc_mode_name(f.get_tensor("yue2_abc_mode_int64").item()) if "yue2_abc_mode_int64" in f.keys() else None
            abc_modes[mode or "none"] += 1
            if meta.get("yue2_tokenizer"):
                tokenizers.add(meta["yue2_tokenizer"])
            if meta.get("yue2_instrumental_lyrics") is not None:
                lyrics_conv.add(meta["yue2_instrumental_lyrics"])
            if args.cot in ("melody", "full") and mode is not None and mode != args.cot:
                cot_mismatch.append(f"{os.path.basename(te)}:{mode}")
        if missing_codes:
            raise ValueError(
                f"{len(missing_codes)} latent caches have no semantic codes (needed by --nar_context codes or the ar branch):"
                f" {missing_codes[:8]}; re-cache with --semantic_head"
            )
        if cot_mismatch:
            raise ValueError(f"--cot {args.cot} does not match the ABC flavour of {len(cot_mismatch)} records: {cot_mismatch[:8]}")
        if len(heads) > 1:
            raise ValueError(f"latent caches mix semantic heads {sorted(heads)}; re-cache with one --semantic_head")
        if len(tokenizers) > 1:
            raise ValueError(f"text caches were made with different tokenizers: {sorted(tokenizers)}")
        if lyrics_conv and lyrics_conv != {args.instrumental_lyrics}:
            raise ValueError(
                f"text caches use instrumental lyrics {sorted(lyrics_conv)}, the trainer --instrumental_lyrics is"
                f" {args.instrumental_lyrics!r}"
            )
        if "ar" in self.branches and mid_song:
            logger.warning(
                f"{mid_song} segments do not start at the song start and get no AR loss; use segment_extraction=full"
                " (or head) for AR training"
            )
        self._cache_info = {
            "tokenizer": next(iter(tokenizers)) if tokenizers else None,
            "abc_modes": abc_modes,
            "latent_meta": latent_meta,
            "mid_song": mid_song,
        }
        logger.info(
            f"YuE2 caches: {len(self._train_items)} training segments, {len(self._validation_items)} validation segments,"
            f" ABC flavours {abc_modes}"
        )

    def _load_minted_pack(self, args) -> None:
        from musubi_tuner.dataset.yue2_minted import YuE2MintedPack

        pack = YuE2MintedPack(args.ar_minted_pack)
        tokenizer = load_yue2_tokenizer(args.tokenizer, args.dit)
        cached = self._cache_info.get("tokenizer")
        if cached is not None and tokenizer.fingerprint != cached:
            raise ValueError(f"--tokenizer {tokenizer.fingerprint} differs from the text caches' tokenizer {cached}")
        pack.tokenize(tokenizer, args.instrumental_lyrics)
        self._minted_fingerprint = pack.fingerprint
        self._minted_train, self._minted_val = pack.split(args.ar_minted_val_fraction, args.seed or 0)
        if len(self._minted_train) == 0:
            raise ValueError(f"minted pack {args.ar_minted_pack} has no training songs after the validation split")
        logger.info(f"YuE2 minted replay: {len(self._minted_train)} train / {len(self._minted_val)} validation songs")

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading YuE2 VAE (decoder) from {vae_path or args.dit}")
        return load_yue2_vae(vae_path, args.dit, device="cpu", decoder_only=True, allow_fp16_source=True)

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        self._attn_mode = attn_mode
        lora_weights, multipliers = None, None
        if args.base_weights:
            # merged into the bf16 weights while streaming, before any quantization
            _, prequant = detect_layout(dit_path)
            if prequant:
                raise ValueError("--base_weights cannot be merged into a pre-quantized int8 checkpoint; use a bf16 checkpoint")
            base_io = read_base_io(dit_path)
            lora_weights, names = [], []
            for path in args.base_weights:
                sd, metadata = lora_formats.load_lora_file(path)
                lora_weights.append(
                    lora_yue2.convert_lora_state_dict(sd, metadata=metadata, target="native_fused", base_io=base_io)
                )
                names.append(f"{os.path.basename(path)}:{_file_sha256(path)}")
            multipliers = _base_multipliers(args)
            self._base_weights_desc = ",".join(names)
            logger.info(f"Merging --base_weights into the YuE2 weights at load: {', '.join(args.base_weights)}")
        needs_lm_head = "ar" in self.branches or bool(args.sample_prompts)
        model = load_yue2_model(
            dit_path,
            device=accelerator.device,
            loading_device=loading_device,
            dtype=torch.bfloat16,
            config=self.model_config,
            attn_mode=attn_mode,
            split_attn=split_attn,
            fp8_scaled=args.fp8_scaled,
            convrot_int8=args.convrot_int8,
            convrot_int8_bwd=args.convrot_int8_bwd,
            quantize_lm_head=args.quantize_lm_head,
            prequant_lm_head=args.prequant_lm_head,
            lm_head_needed=needs_lm_head,
            lora_weights=lora_weights,
            lora_multipliers=multipliers,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        self._base_weights_merged_at_load = lora_weights is not None
        self._checkpoint_layout = model.checkpoint_layout
        self._base_quant = model.base_quant
        self._convrot_int8_active = bool(model.is_convrot_int8)
        return model

    def on_transformer_loaded(self, args: argparse.Namespace, accelerator: Accelerator, transformer) -> None:
        model = transformer
        if args.convrot_int8_bwd == "int8":
            if not model.is_convrot_int8:
                raise ValueError("--convrot_int8_bwd int8 requires a ConvRot int8 base (--convrot_int8 or a pre-quantized file)")
            if torch.device(accelerator.device).type != "cuda":
                raise ValueError("--convrot_int8_bwd int8 requires a CUDA training device")
        ar_n, nar_n = self._swap_counts
        model.set_block_swap_plan(
            ar_n, nar_n, ar_backward="ar" in self.branches, nar_backward="nar" in self.branches, branches=self.branches
        )
        model.set_attention(self._attn_mode, args.split_attn, args.attn_query_tile, args.sdpa_gqa)
        if max(ar_n, nar_n) == 0:
            # the base moves the model only when swapping; apply the branch residency here
            model.move_to_device_except_swap_blocks(accelerator.device)

    def _prepare_with_accelerator(self, args, accelerator, transformer, network, *rest):
        out = super()._prepare_with_accelerator(args, accelerator, transformer, network, *rest)
        model = accelerator.unwrap_model(out[0])
        if not self.blocks_to_swap and model.cpu_resident_modules:
            # accelerator.prepare placed the whole model on the device; put the CPU-resident parts back
            model.move_to_device_except_swap_blocks(accelerator.device)
            clean_memory_on_device(accelerator.device)
        return out

    def convert_weight_keys(self, weights_sd: dict[str, torch.Tensor], network_module):
        return lora_yue2.convert_lora_state_dict(weights_sd)

    def merge_base_weights(self, args, accelerator, transformer, network_module, weight_dtype):
        if self._base_weights_merged_at_load:
            accelerator.print(f"all weights merged during the YuE2 load: {', '.join(args.base_weights)}")
            return
        super().merge_base_weights(args, accelerator, transformer, network_module, weight_dtype)

    def compile_transformer(self, args, transformer):
        from musubi_tuner.utils import model_utils

        model = transformer
        ar_n, nar_n = self._swap_counts
        disable_linear = bool(max(ar_n, nar_n)) or bool(model.is_convrot_int8)
        return model_utils.compile_transformer(args, model, [model.ar.blocks, model.nar.blocks], disable_linear=disable_linear)

    def scale_shift_latents(self, latents):
        return latents

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
        # rank-generic broadcast: [B,T,64] training latents and the 5-D --show_timesteps probe
        t = sample_t(self, args, noise.shape[0], timesteps, device)
        tb = t.view(-1, *([1] * (latents.ndim - 1))).to(latents.device)
        noisy = (1.0 - tb) * latents + tb * noise
        return noisy, t * 1000.0 + 1.0

    # endregion

    # region training step

    def _plan(self, args, batch: dict, i: int, total_frames: int, rng: random.Random, training: bool = True) -> ItemPlan:
        inputs = _item_inputs(batch, i)
        return plan_item(
            texts=inputs["texts"],
            negatives=inputs["negatives"],
            abc=inputs["abc"],
            abc_mode=inputs["abc_mode"],
            codes=inputs["codes"],
            total_frames=total_frames,
            seg=inputs["seg"],
            cot=args.cot,
            abc_dropout=args.abc_dropout if training else 0.0,
            caption_dropout=args.caption_dropout if training else 0.0,
            nar_caption_dropout=args.nar_caption_dropout if training else 0.0,
            window_frames=args.nar_window_frames,
            nar_context=args.nar_context,
            codec_dropout=args.nar_codec_dropout if training else 0.0,
            text_only_rope=args.nar_text_only_rope,
            ar_max_tokens=args.ar_max_tokens,
            ar_ce_targets=args.ar_ce_targets,
            train_ar="ar" in self.branches,
            rng=rng,
            training=training,
        )

    def _minted_ar_ids(self, args, item, negatives: Optional[dict]) -> tuple[list[int], int]:
        """AR ids of a minted song (no score: cot off); the negative prefix when ``negatives`` is given (dropped caption)."""
        if negatives is not None:
            prefix = build_negative_prefix(negatives["off"], "off", None)
        else:
            prefix = build_prefix(item.text_ids["off"], "off", None)
        codes = item.codes.tolist()
        ids, target_start, _ = ar_sequence(prefix, codes, len(codes), len(codes), False, args.ar_max_tokens, "codec")
        return ids, target_start

    def _apply_minted_replay(self, args, plans: list[ItemPlan], rngs: list[random.Random], batch: dict) -> None:
        """Replace the AR ids of an item by a minted song with probability ``--ar_replay_fraction``."""
        if self._minted_train is None or args.ar_replay_fraction <= 0:
            return
        for i, (plan, rng) in enumerate(zip(plans, rngs)):
            if rng.random() >= args.ar_replay_fraction:
                continue
            item = self._minted_train.sample(rng)
            negatives = {cot: batch[f"yue2_neg_{cot}"][i].tolist() for cot in COT_MODES} if plan.dropped_caption else None
            plan.ar_ids, plan.ar_target_start = self._minted_ar_ids(args, item, negatives)
            plan.ar_complete = plan.ar_ids[-1] == MUSIC_END
            plan.ar_skipped_reason = None

    def _context_kv(self, args, model, plan: ItemPlan) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """No-grad AR prefill of the NAR context; K/V detached. Only the visible part is prefilled (causal: identical)."""
        ids = plan.ctx_ids if plan.kv_visible is None else plan.ctx_ids[: plan.kv_visible]
        ids_, n = pad_right(ids, args.ar_pad_multiple, EOD)
        out = model.ar_forward(model.embed(ids_), return_kv=True, valid_len=n, return_hidden=False)
        return [(k.detach(), v.detach()) for k, v in out.kv]

    def _ar_hidden(self, args, model, ar_ids: list[int], grad: bool) -> tuple[torch.Tensor, int]:
        ids_, n = pad_right(ar_ids, args.ar_pad_multiple, EOD)
        emb = model.embed(ids_)
        if grad:
            emb.requires_grad_(True)
        return model.ar_forward(emb, valid_len=n).hidden[0], n

    def _ar_eval_losses(self, args, model, net, ar_ids: list[int], target_start: int) -> tuple[float, Optional[float]]:
        """No-grad CE (and KL against the base when ``net`` is given and the KL weight is set)."""
        hid, n = self._ar_hidden(args, model, ar_ids, grad=False)
        pos, labels = ar_targets(ar_ids, target_start, n)
        base = None
        if net is not None and args.ar_kl_weight > 0:
            net.set_enabled(False)
            try:
                base = self._ar_hidden(args, model, ar_ids, grad=False)[0][pos]
            finally:
                net.set_enabled(True)
        ce, kl = chunked_ce_kl(model.ar.lm_head, hid[pos], labels, base, args.ar_ce_chunk)
        return ce.item(), (kl.item() if kl is not None else None)

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        sample_resources,
        global_step,
    ):
        model = accelerator.unwrap_model(transformer)
        net = accelerator.unwrap_model(network) if network is not None else None
        dev = accelerator.device
        bsz, total_frames = latents.shape[0], latents.shape[1]
        model.begin_train_step()  # one backward follows every process_batch
        train_ar, train_nar = "ar" in self.branches, "nar" in self.branches

        micro = self._micro_index
        self._micro_index += 1
        rank = getattr(accelerator, "process_index", 0)
        rngs = [item_rng(args.seed or 0, global_step, micro, rank, i) for i in range(bsz)]
        plans = [self._plan(args, batch, i, total_frames, rngs[i]) for i in range(bsz)]
        self._apply_minted_replay(args, plans, rngs, batch)
        t = sample_t(self, args, bsz, batch.get("timesteps"), dev) if train_nar else None
        want_kl = train_ar and args.ar_kl_weight > 0 and net is not None

        # phase A: every no-grad pass of every item, LoRA dropouts off
        base_hidden: list[Optional[torch.Tensor]] = [None] * bsz
        kv: list[Optional[list]] = [None] * bsz
        with torch.no_grad(), accelerator.autocast(), lora_eval(net):
            for i, plan in enumerate(plans):
                if want_kl and plan.ar_ids is not None:
                    net.set_enabled(False)
                    try:
                        hid, _ = self._ar_hidden(args, model, plan.ar_ids, grad=False)
                    finally:
                        net.set_enabled(True)
                    base_hidden[i] = hid[plan.ar_target_start - 1 : len(plan.ar_ids) - 1]
                if train_nar:
                    kv[i] = self._context_kv(args, model, plan)

        # phase B: one grad pass per stack (AR, then NAR)
        ce_terms, kl_terms, flow_terms = [], [], []
        skipped = 0
        with accelerator.autocast():
            if train_ar:
                for i, plan in enumerate(plans):
                    if plan.ar_ids is None:
                        skipped += 1
                        continue
                    hid, n = self._ar_hidden(args, model, plan.ar_ids, grad=True)
                    pos, labels = ar_targets(plan.ar_ids, plan.ar_target_start, n)
                    ce, kl = chunked_ce_kl(model.ar.lm_head, hid[pos], labels, base_hidden[i], args.ar_ce_chunk)
                    ce_terms.append(ce)
                    if kl is not None:
                        kl_terms.append(kl)
                    base_hidden[i] = None
            if train_nar:
                for i, plan in enumerate(plans):
                    s, w = plan.window_start, plan.window_frames
                    z = latents[i, s : s + w].to(dev).float()
                    eps = noise[i, s : s + w].to(dev).float()
                    x_t = (1.0 - t[i]) * z + t[i] * eps
                    out = self.call_dit(
                        args,
                        accelerator,
                        transformer,
                        z[None],
                        batch,
                        eps[None],
                        x_t[None],
                        t[i : i + 1] * 1000.0 + 1.0,
                        network_dtype,
                        kv=kv[i],
                        rope_offset=plan.rope_offset,
                        t_value=t[i : i + 1],
                    )
                    kv[i] = None
                    flow_terms.append(flow_loss(out.pred[0], eps, z))

        self._ar_skipped += skipped
        extra = {
            "flow": torch.stack(flow_terms).mean() if flow_terms else None,
            "ce": torch.stack(ce_terms).mean() if ce_terms else None,
            "kl": torch.stack(kl_terms).mean() if kl_terms else None,
            "plans": plans,
            "t": t,
            "ar_skipped": skipped,
            "ref": latents,
        }
        output = DiTOutput(pred=latents, target=latents, extra=extra)
        timesteps = t * 1000.0 + 1.0 if t is not None else None
        return self.compute_loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step)

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        """NAR flow step for one item: ``kv`` (context K/V), ``rope_offset`` and ``t_value`` come in ``kwargs``."""
        model = accelerator.unwrap_model(transformer)
        dev = accelerator.device
        x = noisy_model_input.to(dev)
        if args.gradient_checkpointing:
            x.requires_grad_(True)
        t_value = kwargs.get("t_value")
        if t_value is None:
            t_value = (timesteps - 1.0) / 1000.0
        t_emb = yue2_t_embed_input(t_value, args.t_embed_dtype)  # shared with the sampler
        with accelerator.autocast():
            v = model.nar_forward(x, t_emb.to(dev), kwargs["kv"], kwargs["rope_offset"])
        return DiTOutput(pred=v, target=noise.to(dev) - latents.to(dev))

    def compute_loss(self, args, output: DiTOutput, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step):
        extra = output.extra
        terms = []
        logs: dict[str, float] = {}
        for key, weight, log_key in (
            ("flow", args.flow_loss_weight, "loss/flow"),
            ("ce", args.ar_ce_weight, "loss/ar_ce"),
            ("kl", args.ar_kl_weight, "loss/ar_kl"),
        ):
            value = extra.get(key)
            if value is None:
                continue
            logs[log_key] = value.detach().item()
            terms.append(weight * value)
        if terms:
            loss = torch.stack([term.float() for term in terms]).sum()
        else:
            # no supervised term in this micro-batch (e.g. AR-only on mid-song segments): a graph-free zero
            loss = extra["ref"].new_zeros((), dtype=torch.float32).requires_grad_(True)
        plans: list[ItemPlan] = extra.get("plans") or []
        if extra.get("t") is not None:
            logs["yue2/t"] = extra["t"].float().mean().item()
        if plans:
            logs["yue2/window"] = sum(p.window_frames for p in plans) / len(plans)
            logs["yue2/prefix_len"] = sum(len(p.prefix) for p in plans) / len(plans)
            logs["yue2/score_frac"] = sum(1.0 for p in plans if p.score_used) / len(plans)
        if "ar" in self.branches:
            logs["yue2/ar_skipped"] = float(extra.get("ar_skipped", 0))
        return loss, logs

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients: bool, global_step: int) -> None:
        if not sync_gradients:
            return
        self._micro_index = 0
        step = global_step + 1
        if self._validator is not None and args.validate_every_n_steps and step % args.validate_every_n_steps == 0:
            self._validate(args, accelerator, network, transformer, step)

    # endregion

    # region validation and metadata

    def on_epoch_end(self, args, accelerator, network, transformer, epoch: int) -> None:
        self._epoch = epoch + 1

    def _init_session(self, args):
        # the -best checkpoint carries the same session id and start time as the base checkpoints
        self._session_id, self._training_started_at = super()._init_session(args)
        return self._session_id, self._training_started_at

    def _register_hooks_and_resume(self, args, accelerator, network):
        # the best validation value travels with --save_state, so a resumed run does not overwrite a better -best file
        def save_best_hook(models, weights, output_dir):
            if accelerator.is_main_process and self._best_val is not None:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, BEST_STATE_FILE), "w", encoding="utf-8") as f:
                    json.dump({"best_val": self._best_val, "best_step": self._best_step}, f)

        def load_best_hook(models, input_dir):
            path = os.path.join(input_dir, BEST_STATE_FILE)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self._best_val, self._best_step = float(state["best_val"]), state.get("best_step")
                logger.info(f"YuE2 best validation restored: {self._best_val:.5f} at step {self._best_step}")

        accelerator.register_save_state_pre_hook(save_best_hook)
        accelerator.register_load_state_pre_hook(load_best_hook)
        super()._register_hooks_and_resume(args, accelerator, network)

    def on_train_start(self, args, accelerator, network, transformer, optimizer) -> None:
        self._optimizer = getattr(optimizer, "optimizer", optimizer)
        self._micro_index = 0
        if args.validate_every_n_steps:
            minted_val = self._minted_val if (self._minted_val is not None and len(self._minted_val) > 0) else None
            if "ar" not in self.branches:
                minted_val = None  # minted songs validate only the AR branch
            validator = YuE2Validator(self._validation_items, args, minted_val=minted_val)
            if len(validator) == 0:
                logger.warning(
                    "--validate_every_n_steps is set but there are no validation items (validation_split / is_validation)"
                )
            else:
                self._validator = validator
                logger.info(
                    f"YuE2 validation: {len(validator.batches)} items"
                    + (f" + {len(minted_val)} minted songs" if minted_val is not None else "")
                    + f" every {args.validate_every_n_steps} steps"
                )
                if not validator.batches:
                    logger.info("YuE2 validation has only minted songs: val/total is val/minted_ce")

    def _validate(self, args, accelerator, network, transformer, step: int) -> None:
        results = self._validator.run(self, args, accelerator, transformer, network, self._optimizer)
        if not results:
            return
        logger.info("YuE2 validation at step %d: %s", step, ", ".join(f"{k}={v:.5f}" for k, v in sorted(results.items())))
        total = results["val/total"]
        if self._best_val is None or total < self._best_val:
            self._best_val, self._best_step = total, step
            if args.save_best_validation and accelerator.is_main_process:
                self._save_best(args, accelerator, network, step)
        if self._best_step is not None:
            results["val/best_step"] = float(self._best_step)
        if len(accelerator.trackers) > 0:
            accelerator.log(results, step=step)

    def _save_best(self, args, accelerator, network, step: int) -> None:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, f"{args.output_name}-best.safetensors")
        save_dtype = train_utils.resolve_save_dtype(args.save_precision, False, False)
        metadata = self._build_save_metadata(args, step, self._epoch)
        net = accelerator.unwrap_model(network)
        net.save_weights(path, save_dtype, metadata)
        logger.info(f"YuE2 best validation checkpoint (step {step}): {path}")
        self._write_exports(args, net, os.path.splitext(path)[0], save_dtype)

    def _build_save_metadata(self, args, steps: int, epoch) -> dict[str, str]:
        """Training metadata as the base save writes it (``trainer_base`` builds it as a loop local)."""
        group = self._dataset_group
        num_items = group.num_train_items if group is not None else 0
        num_batches = len(group) if group is not None else 0
        per_epoch = max(1, math.ceil(num_batches / max(1, args.gradient_accumulation_steps)))
        optimizer = self._optimizer
        optimizer_name = f"{type(optimizer).__module__}.{type(optimizer).__name__}" if optimizer is not None else ""
        optimizer_args = ",".join(args.optimizer_args or [])
        net_kwargs = {}
        for net_arg in args.network_args or []:
            key, value = net_arg.split("=")
            net_kwargs[key] = value
        metadata = {
            "ss_session_id": self._session_id,
            "ss_training_started_at": self._training_started_at,
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_num_train_items": num_items,
            "ss_num_batches_per_epoch": num_batches,
            "ss_num_epochs": math.ceil(args.max_train_steps / per_epoch) if args.max_train_steps else None,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_base_model_version": self.architecture_full_name,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,
            "ss_network_alpha": args.network_alpha,
            "ss_network_dropout": args.network_dropout,
            "ss_mixed_precision": args.mixed_precision,
            "ss_seed": args.seed,
            "ss_training_comment": args.training_comment,
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if optimizer_args else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_fp8_base": bool(args.fp8_base),
            "ss_full_fp16": False,
            "ss_full_bf16": False,
            "ss_weighting_scheme": args.weighting_scheme,
            "ss_logit_mean": args.logit_mean,
            "ss_logit_std": args.logit_std,
            "ss_mode_scale": args.mode_scale,
            "ss_guidance_scale": args.guidance_scale,
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_sigmoid_scale": args.sigmoid_scale,
            "ss_discrete_flow_shift": args.discrete_flow_shift,
        }
        metadata.update(self.extra_metadata(args))
        if group is not None:
            metadata["ss_datasets"] = json.dumps([ds.get_metadata() for ds in group.datasets])
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)
        if args.dit is not None:
            metadata["ss_sd_model_name"] = os.path.basename(args.dit) if os.path.exists(args.dit) else args.dit
        if args.vae is not None:
            metadata["ss_vae_name"] = os.path.basename(args.vae) if os.path.exists(args.vae) else args.vae
        metadata["ss_training_finished_at"] = time.time()
        metadata["ss_steps"] = steps
        metadata["ss_epoch"] = epoch
        metadata = {k: str(v) for k, v in metadata.items()}
        if args.no_metadata:
            metadata = {k: metadata[k] for k in SS_METADATA_MINIMUM_KEYS if k in metadata}

        title = args.metadata_title if args.metadata_title is not None else args.output_name
        md_timesteps = None
        if args.min_timestep is not None or args.max_timestep is not None:
            md_timesteps = (args.min_timestep or 0, args.max_timestep if args.max_timestep is not None else 1000)
        metadata.update(
            sai_model_spec.build_metadata(
                None,
                self.architecture,
                time.time(),
                title,
                args.metadata_reso,
                args.metadata_author,
                args.metadata_description,
                args.metadata_license,
                args.metadata_tags,
                timesteps=md_timesteps,
                custom_arch=args.metadata_arch,
            )
        )
        return metadata

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        latent_meta = self._cache_info.get("latent_meta") or {}
        semantic = latent_meta.get("yue2_semantic_head")
        mert = latent_meta.get("yue2_semantic_mert")
        vae_dtype = latent_meta.get("yue2_vae_source_dtype")
        metadata = {
            "ss_yue2_protocol": PROTOCOL_VERSION,
            "ss_yue2_branches": ",".join(self.branches),
            "ss_yue2_lora_layout": getattr(self, "_network_layout", None),
            "ss_yue2_train_io": args.train_io,
            "ss_yue2_nar_context": args.nar_context,
            "ss_yue2_nar_codec_dropout": args.nar_codec_dropout,
            "ss_yue2_nar_text_only_rope": args.nar_text_only_rope,
            "ss_yue2_nar_cond_end": "len(prefix)",
            "ss_yue2_window_frames": args.nar_window_frames,
            "ss_yue2_cot": args.cot,
            "ss_yue2_abc_dropout": args.abc_dropout,
            "ss_yue2_caption_dropout": args.caption_dropout,
            "ss_yue2_caption_dropout_scope": "ar",
            "ss_yue2_nar_caption_dropout": args.nar_caption_dropout,
            "ss_yue2_ar_ce_targets": args.ar_ce_targets,
            "ss_yue2_ar_max_tokens": args.ar_max_tokens,
            "ss_yue2_loss_weights": f"{args.flow_loss_weight},{args.ar_ce_weight},{args.ar_kl_weight}",
            "ss_yue2_t_embed_dtype": args.t_embed_dtype,
            "ss_yue2_timestep_sampling": args.timestep_sampling
            + (f"({args.beta_timestep_ab})" if args.timestep_sampling == "beta" else ""),
            "ss_yue2_semantic_head": semantic,
            "ss_yue2_mert": mert,
            "ss_yue2_tokenizer": self._cache_info.get("tokenizer"),
            "ss_yue2_base_weights": self._base_weights_desc,
            "ss_yue2_base_quant": self._base_quant,
            "ss_yue2_checkpoint_layout": self._checkpoint_layout,
            "ss_yue2_latent_cache_version": latent_meta.get("yue2_cache_version", YUE2_LATENT_CACHE_VERSION),
            "ss_yue2_text_cache_version": YUE2_TEXT_CACHE_VERSION,
            "ss_yue2_instrumental_lyrics": args.instrumental_lyrics,
            "ss_yue2_abc_modes": json.dumps(self._cache_info.get("abc_modes") or {}),
            "ss_yue2_ar_replay": f"{args.ar_replay_fraction}:{self._minted_fingerprint}" if self._minted_fingerprint else None,
            "ss_yue2_first_timestep_chance": args.first_timestep_chance,
            "ss_yue2_ar_pad_multiple": args.ar_pad_multiple,
            "ss_yue2_vae_source_dtype": vae_dtype,
        }
        return {k: v for k, v in metadata.items() if v is not None}

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        network = super()._build_network(args, accelerator, transformer, vae, weight_dtype)
        if network is not None:
            self._network_layout = getattr(network, "layout", None)
        return network

    def on_post_save(self, args, accelerator, network, transformer, ckpt_name, save_dtype, metadata, force_sync_upload) -> None:
        stem = os.path.join(args.output_dir, os.path.splitext(ckpt_name)[0])
        self._write_exports(args, accelerator.unwrap_model(network), stem, save_dtype)
        self._remove_rotated_exports(args, ckpt_name)

    def _export_native(self, args, net) -> lora_formats.NativeLoRA:
        """The trained LoRA as ``NativeLoRA``; with ``--yue2_export_include_base_weights`` plus every base LoRA at the
        multiplier it was merged with at load."""
        sd = {k: v.detach().float().cpu() for k, v in net.state_dict().items()}
        native = lora_formats.to_native(sd, {})
        if args.yue2_export_include_base_weights:
            base_io = read_base_io(args.dit) if args.dit and os.path.isfile(args.dit) else None
            for path, multiplier in zip(args.base_weights, _base_multipliers(args)):
                bsd, bmeta = lora_formats.load_lora_file(path)
                base = lora_formats.to_native(bsd, bmeta, base_io=base_io)
                if multiplier != 1.0:
                    base = lora_formats.scale_branches(base, ar=multiplier, nar=multiplier)
                native = lora_formats.concat_rank(native, base)
        return native

    def _write_exports(self, args, net, stem: str, save_dtype) -> None:
        """``--yue2_export_formats`` files next to a saved checkpoint. A failed export is logged and never stops training."""
        formats = _export_formats(args)
        if not formats:
            return
        try:
            native = self._export_native(args, net)
        except Exception as e:
            logger.error(f"YuE2 LoRA exports of {stem} skipped: {type(e).__name__}: {e}")
            return
        dtype = save_dtype if save_dtype is not None else torch.float32
        written = []
        for fmt in formats:
            try:
                if fmt == "comfy":
                    out, meta = lora_formats.native_to_comfy(native, dtype=dtype)
                    lora_formats.save_lora_file(f"{stem}.comfy.safetensors", out, meta)
                elif fmt == "hf":
                    for branch in ("ar", "nar"):
                        part = lora_formats.filter_branch(native, branch)
                        if part.modules:
                            out, meta = lora_formats.native_to_hf(part, branch=branch, dtype=dtype)
                            lora_formats.save_lora_file(f"{stem}.hf-{branch}.safetensors", out, meta)
                elif fmt == "fl":
                    for branch, (out, meta) in lora_formats.native_to_fl(native, dtype=dtype).items():
                        lora_formats.save_lora_file(f"{stem}.fl-{branch}.safetensors", out, meta)
                written.append(fmt)
            except Exception as e:
                logger.error(f"YuE2 {fmt} export of {stem} failed: {type(e).__name__}: {e}")
        if written:
            logger.info(f"YuE2 LoRA exports written: {stem}.* ({', '.join(written)})")

    def _remove_rotated_exports(self, args, ckpt_name: str) -> None:
        """The base deletes only the main file of a rotated checkpoint (``--save_last_n_steps/epochs``); drop its exports."""
        name = re.escape(args.output_name)
        step = re.fullmatch(name + r"-step(\d+)\.safetensors", ckpt_name)
        epoch = re.fullmatch(name + r"-(\d+)\.safetensors", ckpt_name)
        old = None
        if step is not None and args.save_every_n_steps:
            remove_no = train_utils.get_remove_step_no(args, int(step.group(1)))
            old = None if remove_no is None else train_utils.get_step_ckpt_name(args.output_name, remove_no)
        elif epoch is not None and args.save_every_n_epochs:
            remove_no = train_utils.get_remove_epoch_no(args, int(epoch.group(1)))
            old = None if remove_no is None else train_utils.get_epoch_ckpt_name(args.output_name, remove_no)
        if old is None:
            return
        stem = os.path.join(args.output_dir, os.path.splitext(old)[0])
        for suffix in EXPORT_SUFFIXES:
            path = f"{stem}{suffix}.safetensors"
            if os.path.exists(path):
                logger.info(f"removing old YuE2 LoRA export: {path}")
                os.remove(path)

    # endregion

    # region sampling

    def prepare_sampling(self, args, accelerator, vae_dtype):
        if not args.sample_prompts:
            return None, None
        prompts = load_prompts(args.sample_prompts)
        if not prompts:
            raise ValueError(f"YuE2 sample prompt file is empty: {args.sample_prompts}")
        tokenizer = load_yue2_tokenizer(args.tokenizer, args.dit)
        base_dir = os.path.dirname(os.path.abspath(args.sample_prompts))
        for parameter in prompts:
            parameter["yue2_sample"] = self._prepare_sample(args, parameter, tokenizer, base_dir)
            parameter["sample_steps"] = parameter["yue2_sample"]["ode_steps"]
            parameter["frame_count"] = 1
        vae = self.load_vae(args, vae_dtype, args.vae)
        vae.requires_grad_(False)
        vae.eval()
        return prompts, YuE2SamplingResources(vae, tokenizer)

    def _read_rel(self, base_dir: str, path: str) -> str:
        path = path if os.path.isabs(path) else os.path.join(base_dir, path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _prepare_sample(self, args, parameter: dict, tokenizer, base_dir: str) -> dict:
        """Resolve one prompt dict (sample prompt keys) into the sampler inputs."""
        mode = parameter.get("mode", args.sample_mode)
        if mode not in SAMPLE_MODES:
            raise ValueError(f"YuE2 sample mode must be one of {SAMPLE_MODES}, got {mode!r}")
        seconds = float(parameter.get("seconds", args.sample_seconds))
        sample = {"mode": mode, "seconds": seconds, "ode_steps": int(parameter.get("sample_steps", ODE_STEPS))}
        sample["cfg_scale"] = parameter.get("cfg_scale")
        if mode == "reconstruct":
            sample.update(self._reconstruct_source(args, parameter, base_dir))
            return sample

        lyrics = parameter.get("lyrics")
        if parameter.get("lyrics_file"):
            lyrics = self._read_rel(base_dir, parameter["lyrics_file"])
        abc = parameter.get("abc")
        if parameter.get("abc_file"):
            abc = self._read_rel(base_dir, parameter["abc_file"])
        style, lyrics = normalize_prompt_fields(parameter.get("prompt", ""), lyrics, args.instrumental_lyrics)
        cot = parameter.get("cot") or ("full" if abc else "off")
        if cot not in COT_MODES:
            raise ValueError(f"YuE2 sample cot must be one of {COT_MODES}, got {cot!r}")
        semantic = dataclasses.replace(
            SEMANTIC_DEFAULTS,
            **{
                k: type(getattr(SEMANTIC_DEFAULTS, k))(parameter[k])
                for k in ("temperature", "top_p", "top_k", "repetition_penalty", "penalty_window")
                if k in parameter
            },
        )
        abc_params = dataclasses.replace(
            ABC_DEFAULTS,
            **{
                k[len("abc_") :]: type(getattr(ABC_DEFAULTS, k[len("abc_") :]))(parameter[k])
                for k in ("abc_temperature", "abc_top_p", "abc_top_k")
                if k in parameter
            },
        )
        abc_ids = tokenizer.encode(abc) if abc else None
        sample.update(
            style=style,
            lyrics=lyrics,
            cot=cot,
            abc=abc,
            semantic=semantic,
            abc_params=abc_params,
            text_ids=text_ids(tokenizer, style, lyrics, cot),
            neg_ids=negative_text_ids(tokenizer, cot),
            abc_ids=abc_ids,
        )
        return sample

    def _reconstruct_source(self, args, parameter: dict, base_dir: str) -> dict:
        """Codes + text ids of a cached song: ``reconstruct_cache`` (a latent cache file) or a validation item."""
        from safetensors.torch import load_file

        path = parameter.get("reconstruct_cache")
        if path:
            latent_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
            stem = os.path.basename(latent_path)[: -len(".safetensors")]
            item_key = "_".join(stem.split("_")[:-2])
            te_path = os.path.join(os.path.dirname(latent_path), f"{item_key}_{self.architecture}_te.safetensors")
        else:
            pool = self._validation_items or self._train_items
            if not pool:
                raise ValueError("YuE2 reconstruct samples need a reconstruct_cache or cached validation/training items")
            if not self._validation_items:
                logger.warning("YuE2 reconstruct sample uses a training item (no validation split)")
            item = pool[int(parameter.get("enum", 0)) % len(pool)]
            latent_path, te_path = item.latent_cache_path, item.text_encoder_output_cache_path
        latent_sd, te_sd = load_file(latent_path), load_file(te_path)
        if "codes_int64" not in latent_sd:
            raise ValueError(f"reconstruct sample needs semantic codes in {latent_path}")
        latents = next(v for k, v in latent_sd.items() if k.startswith("latents_"))
        codes = latent_sd["codes_int64"]
        frames = min(latents.shape[0], max(1, round(float(parameter.get("seconds", args.sample_seconds)) * FRAME_RATE)))
        abc = te_sd["varlen_yue2_abc_int64"].tolist()
        mode = abc_mode_name(te_sd["yue2_abc_mode_int64"].item()) if "yue2_abc_mode_int64" in te_sd else None
        cot = parameter.get("cot") or (mode if abc and mode else "off")
        if cot != "off" and not abc:
            cot = "off"
        texts = te_sd[f"varlen_yue2_text_{cot}_int64"].tolist()
        prefix = build_prefix(texts, cot, abc if cot != "off" else None)
        logger.info(f"YuE2 reconstruct sample from {os.path.basename(latent_path)}: {frames} frames, cot {cot}")
        return {
            "prefix": prefix,
            "codes": codes[:frames].clone(),
            "gt_latents": latents[:frames].float().clone(),
            "source": os.path.basename(latent_path),
            "cot": cot,
        }

    def round_sample_frame_count(self, frame_count: int) -> int:
        return frame_count

    def on_before_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        # sampling runs after a backward: the per-stack grad-pass counters must be clean
        accelerator.unwrap_model(transformer).begin_train_step()

    def on_after_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        accelerator.unwrap_model(transformer).begin_train_step()

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        from musubi_tuner.yue2 import yue2_sampling

        sample = sample_parameter["yue2_sample"]
        model = accelerator.unwrap_model(transformer)
        device = accelerator.device
        seed = int(generator.initial_seed())
        resources: YuE2SamplingResources = vae
        decoder = resources.vae
        decoder.to(device)
        state_dtype = torch.bfloat16 if args.sample_ode_state_dtype == "bf16" else torch.float32
        try:
            # the NAR conditioning of the preview follows the training configuration
            if sample["mode"] == "reconstruct":
                result = yue2_sampling.reconstruct(
                    model,
                    decoder,
                    sample["prefix"],
                    sample["codes"],
                    seed,
                    sample["ode_steps"],
                    state_dtype=state_dtype,
                    nar_context=args.nar_context,
                    text_only_rope=args.nar_text_only_rope,
                    t_embed_mode=args.t_embed_dtype,
                    grad_ctx=torch.no_grad,
                )
            else:
                request = self._song_request(args, sample, seed, state_dtype)
                result = yue2_sampling.render_song(model, decoder, resources.tokenizer, request, grad_ctx=torch.no_grad)
            ground_truth = None
            enum = int(sample_parameter.get("enum", 0))
            if sample["mode"] == "reconstruct" and enum not in self._gt_written:
                self._gt_written.add(enum)
                with torch.no_grad():
                    ground_truth = yue2_sampling.decode_latents(decoder, sample["gt_latents"])
        finally:
            decoder.to("cpu")
            clean_memory_on_device(device)
        waveform = result.waveform
        if waveform.ndim == 3:
            waveform = waveform[0]
        meta = {"mode": sample["mode"], "seed": seed, "steps": sample["ode_steps"]}
        if sample["mode"] == "reconstruct":
            meta["source"] = sample["source"]
        return YuE2SampleAudio(waveform=waveform.float().cpu(), meta=meta, ground_truth=ground_truth)

    def _song_request(self, args, sample: dict, seed: int, state_dtype: torch.dtype):
        from musubi_tuner.yue2 import yue2_sampling

        values = {
            "style": sample["style"],
            "lyrics": sample["lyrics"],
            "cot": sample["cot"],
            "abc": sample["abc"],
            "seed": seed,
            "seconds": sample["seconds"],
            "cfg_scale": sample["cfg_scale"],
            "semantic": sample["semantic"],
            "abc_params": sample["abc_params"],
            "ode_steps": sample["ode_steps"],
            "state_dtype": "bf16" if state_dtype == torch.bfloat16 else "fp32",
            "mode": "render",
            "text_ids": sample["text_ids"],
            "neg_ids": sample["neg_ids"],
            "abc_ids": sample["abc_ids"],
            "nar_context": args.nar_context,
            "text_only_rope": args.nar_text_only_rope,
            "t_embed_mode": args.t_embed_dtype,
            "instrumental_lyrics": args.instrumental_lyrics,
            "ar_resident": bool(args.sample_ar_resident),
        }
        fields = {f.name for f in dataclasses.fields(yue2_sampling.YuE2SongRequest)}
        return yue2_sampling.YuE2SongRequest(**{k: v for k, v in values.items() if k in fields})

    def save_sample(self, accelerator, args, sample_parameter, sample, save_dir: str, save_path: str, steps: int) -> None:
        from musubi_tuner.yue2.yue2_audio_io import write_audio

        path = os.path.join(save_dir, f"{save_path}.flac")
        metadata = {k: str(v) for k, v in sample.meta.items()}
        write_audio(path, sample.waveform, sample_rate=sample.sample_rate, fmt="flac", metadata=metadata)
        logger.info(f"YuE2 training sample: {path}")
        paths = [path]
        if sample.ground_truth is not None:
            gt_path = os.path.join(save_dir, f"{save_path}_gt.flac")
            write_audio(gt_path, sample.ground_truth, sample_rate=sample.sample_rate, fmt="flac", metadata=None)
            paths.append(gt_path)
        wandb_tracker, wandb = wandb_tracker_and_module(accelerator)
        if wandb_tracker is not None:
            prompt_idx = sample_parameter.get("enum", 0)
            wandb_tracker.log({f"sample_{prompt_idx}": wandb.Audio(path)}, step=steps)

    # endregion


def yue2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    setup_parser_yue2_model(parser, training=True)
    parser.add_argument("--train_branches", type=str, default="nar", help="trained stacks: nar (default), ar, or ar,nar")
    parser.add_argument(
        "--nar_context",
        type=str,
        default="codes",
        choices=NAR_CONTEXTS,
        help="NAR conditioning: codes (prefix + window codes + MUSIC_END, protocol-exact) or text_only (prefix K/V only)",
    )
    parser.add_argument(
        "--nar_codec_dropout", type=float, default=0.0, help="probability of hiding the codes from the NAR (codes mode)"
    )
    parser.add_argument(
        "--nar_text_only_rope",
        type=str,
        default="full",
        choices=TEXT_ONLY_ROPE_MODES,
        help="text_only NAR positions: full (inference layout, as if codes were present) or compact (right after the prefix)",
    )
    parser.add_argument(
        "--nar_window_frames",
        type=int,
        default=1500,
        help="NAR training window in frames (25/s); 0 = whole segment; always capped to the context",
    )
    parser.add_argument(
        "--cot",
        type=str,
        default="auto",
        choices=COT_CHOICES,
        help="score usage: auto (the record's ABC flavour when it has ABC, else off), off, melody, full",
    )
    parser.add_argument("--abc_dropout", type=float, default=0.5, help="probability of training without the ABC score")
    parser.add_argument(
        "--caption_dropout",
        type=float,
        default=0.0,
        help="probability of the protocol negative prefix for the AR branch (CFG training; the NAR context stays positive)",
    )
    parser.add_argument("--nar_caption_dropout", type=float, default=0.0, help="negative prefix for the NAR context (experiments)")
    parser.add_argument(
        "--ar_ce_targets",
        type=str,
        default="codec",
        choices=AR_CE_TARGETS,
        help="AR CE targets: codec tokens (default) or codec_abc (the ABC score tokens too)",
    )
    parser.add_argument("--ar_max_tokens", type=int, default=0, help="AR codes per item from the song start (0 = whole song)")
    parser.add_argument("--ar_ce_chunk", type=int, default=512, help="rows per checkpointed AR CE/KL chunk")
    parser.add_argument(
        "--ar_pad_multiple",
        type=int,
        default=None,
        help="right-pad AR passes to this multiple (exact under causal attention); default 256 with --compile, else 0",
    )
    parser.add_argument(
        "--ar_minted_pack",
        type=str,
        default=None,
        help="minted-corpus pack for AR replay: a ComfyUI-FL-YuE2 .pt kit or .json manifest, or a .jsonl with codes",
    )
    parser.add_argument(
        "--ar_replay_fraction", type=float, default=0.0, help="probability of replacing the AR item by a minted song"
    )
    parser.add_argument("--ar_minted_val_fraction", type=float, default=0.1, help="minted songs held out for val/minted_ce")
    parser.add_argument(
        "--first_timestep_chance", type=float, default=0.0, help="probability of forcing t=1 (pure noise) for an item"
    )
    parser.add_argument("--flow_loss_weight", type=float, default=1.0, help="weight of the NAR flow loss")
    parser.add_argument("--ar_ce_weight", type=float, default=1.0, help="weight of the AR cross-entropy")
    parser.add_argument("--ar_kl_weight", type=float, default=0.2, help="weight of KL(base || LoRA) on the AR logits")
    parser.add_argument("--ar_lr_ratio", type=float, default=1.0, help="AR LoRA learning rate = --learning_rate x this")
    parser.add_argument(
        "--train_io",
        type=str,
        default="none",
        choices=["none", "lora", "full"],
        help="NAR I/O modules (vae2llm, llm2vae): none, lora, or full (trainable diff)",
    )
    parser.add_argument("--io_lr", type=float, default=None, help="learning rate of the I/O modules (default: --learning_rate)")
    parser.add_argument(
        "--beta_timestep_ab",
        type=str,
        default="2.0,2.0",
        help="Beta(a,b) for --timestep_sampling beta, clipped to min/max_timestep",
    )
    parser.add_argument("--sample_ar_resident", action="store_true", help="during AR sampling keep the whole AR stack on the GPU")
    parser.add_argument("--validate_every_n_steps", type=int, default=None, help="run validation losses every N steps")
    parser.add_argument("--validation_timesteps", type=str, default="0.2,0.5,0.8", help="flow times of the validation flow loss")
    parser.add_argument("--validation_noise_seed", type=int, default=1234, help="seed of the fixed validation noise")
    parser.add_argument("--validation_max_items", type=int, default=8, help="validation segments used per run")
    parser.add_argument(
        "--save_best_validation", action="store_true", help="save {output_name}-best.safetensors when val/total improves"
    )
    parser.add_argument(
        "--sample_mode", type=str, default="reconstruct", choices=SAMPLE_MODES, help="default mode of sample prompts"
    )
    parser.add_argument("--sample_seconds", type=float, default=30.0, help="default length of sample prompts in seconds")
    parser.add_argument(
        "--yue2_export_formats", type=str, default="", help="also write each saved LoRA as: comma list of comfy, hf, fl"
    )
    parser.add_argument(
        "--yue2_export_include_base_weights",
        action="store_true",
        help="rank-concatenate --base_weights into the exported files",
    )
    choices = parser._option_string_actions["--timestep_sampling"].choices
    if "beta" not in choices:
        parser._option_string_actions["--timestep_sampling"].choices = list(choices) + ["beta"]
    parser.set_defaults(
        timestep_sampling="sigmoid",
        weighting_scheme="none",
        discrete_flow_shift=1.0,
        network_module="networks.lora_yue2",
        compile_cache_size_limit=32,
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = yue2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "float32"

    trainer = YuE2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
