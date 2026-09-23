"""YuE2 latent cache: per segment, the FP32 VAE posterior mean ``[T,64]`` and (with ``--semantic_head``) the semantic
codes ``[T]``.

Both are computed once per record (the whole song or JSONL excerpt) and sliced per segment, so a segment's edges see
their real neighbours: the VAE runs over the record in overlapping chunks, and the semantic tokenizer normalises
MERT features over the whole record. Codes and latents are aligned 1:1 (one code per 1920-sample frame).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import logging
import os
import time
from typing import Callable, Optional

import av
import torch
from safetensors.torch import load_file

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_YUE2
from musubi_tuner.dataset.audio_dataset import AudioDataset
from musubi_tuner.dataset.cache_io import YUE2_LATENT_CACHE_VERSION, save_latent_cache_yue2, yue2_latent_key, yue2_segment_info
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.yue2.cache_plan import latent_metadata_matches, plan_yue2_datasets
from musubi_tuner.yue2.yue2_checkpoint import load_yue2_vae
from musubi_tuner.yue2.yue2_protocol import FRAME_RATE, MERT_SAMPLE_RATE, YUE2_AUDIO_SPEC
from musubi_tuner.yue2.yue2_semantic import MERT_REPO, SemanticTokenizer, semantic_fingerprint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LATENT_DTYPES = ("float32", "bfloat16")
MERT_DTYPES = ("bfloat16", "float16", "float32")


class _RecordLRU:
    """Whole-record results keyed by (dataset_index, datasource_index); the segments of a record arrive together."""

    def __init__(self, maxsize: int = 2):
        self.maxsize = maxsize
        self.data: OrderedDict = OrderedDict()

    def get(self, key, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        value = compute()
        self.data[key] = value
        while len(self.data) > self.maxsize:
            self.data.popitem(last=False)
        return value

    def clear(self):
        self.data.clear()


def audio_stamp(path) -> str:
    st = os.stat(path)
    return f"{st.st_size}:{st.st_mtime_ns}"


def source_channels(path) -> int:
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream: {path}")
        return int(container.streams.audio[0].codec_context.channels)


def decode_mono24(dataset: AudioDataset, item: ItemInfo) -> torch.Tensor:
    """The item's record at 24 kHz as the mean of its channels (the downmix the semantic heads were trained on),
    cropped like the 48 kHz record. Mono files are decoded as mono, everything else as stereo and averaged."""
    channels = 1 if source_channels(item.audio_source.path) == 1 else 2
    wave = dataset.datasource.decode(item.datasource_index, sample_rate=MERT_SAMPLE_RATE, channels=channels)
    return wave.float().mean(0)


def latent_identity(item: ItemInfo, vae, args: argparse.Namespace) -> dict[str, str]:
    """Metadata a current latent cache carries, codes excluded (the --skip_existing check)."""
    start, song_frames, truncated = yue2_segment_info(item)
    return {
        "yue2_cache_version": YUE2_LATENT_CACHE_VERSION,
        "yue2_start_frame": str(start),
        "yue2_frames": str(item.frame_count),
        "yue2_song_frames": str(song_frames),
        "yue2_truncated": str(int(truncated)),
        # record offset in the song + record length: codes are normalised over the record
        "yue2_record": f"{start - item.frame_pos}+{item.record_frames}",
        "yue2_song_id": str(item.song_id),
        "yue2_audio": audio_stamp(item.audio_source.path),
        "yue2_vae": vae.fingerprint,
        "yue2_vae_source_dtype": vae.source_dtype,
        "yue2_vae_chunking": f"{args.vae_chunk_frames}/{args.vae_overlap_frames}",
        "yue2_latent_dtype": args.latent_dtype,
    }


def codes_identity(semantic_fp: Optional[dict[str, str]]) -> dict[str, str]:
    if semantic_fp is None:
        return {"yue2_has_codes": "0"}
    return {"yue2_has_codes": "1", **semantic_fp}


class _VAEIdentity:
    """What ``latent_identity`` needs from the VAE, kept after the model is unloaded."""

    def __init__(self, vae):
        self.fingerprint = vae.fingerprint
        self.source_dtype = vae.source_dtype


def show_audio_datasets(datasets: list[AudioDataset], num_workers: int) -> None:
    for i, dataset in enumerate(datasets):
        print(f"Dataset [{i}]")
        for key, batch in dataset.retrieve_latent_cache_batches(num_workers):
            for item in batch:
                print(
                    f"  {item.item_key}: segment {item.frame_pos}+{item.frame_count} of {item.record_frames} frames"
                    f" ({item.frame_count / FRAME_RATE:.1f}s), song start {item.song_start_frame}/{item.song_frames},"
                    f" truncated={item.truncated}, song_id={item.song_id}, cache {os.path.basename(item.latent_cache_path)}"
                )


def encode_datasets_yue2(datasets: list[AudioDataset], args: argparse.Namespace, device: torch.device) -> dict:
    """Runs the caching passes; returns a small summary (records, seconds of audio, timings, peak memory)."""
    latent_dtype = str_to_dtype(args.latent_dtype)
    use_codes = args.semantic_head is not None and not args.no_codes
    mert_dtype = None if args.mert_dtype == "float32" else str_to_dtype(args.mert_dtype)
    semantic_fp = semantic_fingerprint(args.semantic_head, args.mert_model, args.mert_revision) if use_codes else None
    summary = {
        "records": 0,
        "segments": 0,
        "audio_seconds": 0.0,
        "vae_seconds": 0.0,
        "semantic_seconds": 0.0,
        "codes_unique_min": None,
    }
    seen_records: set = set()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    logger.info(f"Loading YuE2 VAE from {args.vae or args.dit}")
    vae = load_yue2_vae(args.vae, args.dit, device=device, allow_fp16_source=args.allow_fp16_vae)
    vae_id = _VAEIdentity(vae)
    logger.info(f"YuE2 VAE: source dtype {vae.source_dtype}, fingerprint {vae.fingerprint[:23]}")

    def expected(item: ItemInfo, with_codes: bool = True) -> dict[str, str]:
        meta = latent_identity(item, vae_id, args)
        if with_codes:
            meta.update(codes_identity(semantic_fp))
        return meta

    def is_current(item: ItemInfo, with_codes: bool = True) -> bool:
        if not os.path.exists(item.latent_cache_path):
            return False
        if latent_metadata_matches(item.latent_cache_path, expected(item, with_codes)):
            logger.info(f"Skipping current YuE2 latent cache: {item.latent_cache_path}")
            return True
        logger.info(f"Rebuilding stale YuE2 latent cache: {item.latent_cache_path}")
        return False

    record_latents = _RecordLRU()
    record_codes = _RecordLRU()
    state = {"vae": vae, "semantic": None}

    def load_semantic():
        logger.info(f"Loading semantic tokenizer: head {args.semantic_head}, MERT {args.mert_model}")
        state["semantic"] = SemanticTokenizer(
            args.semantic_head, args.mert_model, args.mert_revision, device=device, dtype=mert_dtype
        )

    def whole_record_latents(item: ItemInfo) -> torch.Tensor:
        t0 = time.perf_counter()
        z = state["vae"].encode_mean_chunked(item.audio_content.to(device), args.vae_chunk_frames, args.vae_overlap_frames)
        summary["vae_seconds"] += time.perf_counter() - t0
        if z.shape[0] != item.record_frames:
            raise RuntimeError(f"VAE returned {z.shape[0]} frames for a {item.record_frames}-frame record: {item.item_key}")
        if not torch.isfinite(z).all():
            logger.warning(f"non-finite VAE latents: {item.item_key}")
        return z

    def whole_record_codes(item: ItemInfo) -> torch.Tensor:
        t0 = time.perf_counter()
        mono24 = decode_mono24(datasets[item.dataset_index], item)
        codes = state["semantic"].tokenize_frames(mono24, item.record_frames)
        summary["semantic_seconds"] += time.perf_counter() - t0
        if codes.shape != (item.record_frames,):
            raise RuntimeError(f"semantic tokenizer returned {tuple(codes.shape)} for {item.record_frames} frames: {item.item_key}")
        return codes

    def cached_latents(item: ItemInfo) -> Optional[torch.Tensor]:
        # the latents of a cache whose latent identity is current (only the codes are missing or stale)
        if not latent_metadata_matches(item.latent_cache_path, expected(item, with_codes=False)):
            return None
        return load_file(item.latent_cache_path)[yue2_latent_key(item.frame_count, latent_dtype)]

    def encode(batch: list[ItemInfo], latents_only: bool = False) -> None:
        for item in batch:
            record = (item.dataset_index, item.datasource_index)
            if record not in seen_records:
                seen_records.add(record)
                summary["records"] += 1
                summary["audio_seconds"] += item.record_frames / FRAME_RATE
            with_codes = use_codes and not latents_only
            z = cached_latents(item) if with_codes else None
            if z is None:
                if state["vae"] is None:
                    raise RuntimeError(f"latent cache is missing or stale after the VAE pass: {item.latent_cache_path}")
                full_z = record_latents.get(record, lambda: whole_record_latents(item))
                z = full_z[item.frame_pos : item.frame_pos + item.frame_count]
            codes = None
            if with_codes:
                full_codes = record_codes.get(record, lambda: whole_record_codes(item))
                codes = full_codes[item.frame_pos : item.frame_pos + item.frame_count]
                unique = int(codes.unique().numel())
                summary["codes_unique_min"] = (
                    unique if summary["codes_unique_min"] is None else min(summary["codes_unique_min"], unique)
                )
            meta = latent_identity(item, vae_id, args)
            meta.update(codes_identity(semantic_fp if with_codes else None))
            meta.pop("yue2_has_codes")  # written by save_latent_cache_yue2 from `codes`
            save_latent_cache_yue2(item, z, codes, meta, dtype=latent_dtype)
            summary["segments"] += 1
            logger.info(
                f"cached {item.item_key} [{item.frame_pos}+{item.frame_count}]"
                + (f", {int(codes.unique().numel())} unique codes" if codes is not None else "")
            )

    if use_codes and not args.sequential_models:
        load_semantic()
        cache_latents.encode_datasets(datasets, encode, args, supports_alpha=True, cache_is_current=is_current)
    else:
        # pass 1: VAE only; with --sequential_models the codes follow in pass 2 with the VAE unloaded
        cache_latents.encode_datasets(
            datasets,
            lambda batch: encode(batch, latents_only=True),
            args,
            supports_alpha=True,
            cache_is_current=lambda item: is_current(item, with_codes=not use_codes),
        )
        if use_codes:
            state["vae"] = None
            del vae
            record_latents.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            load_semantic()
            cache_latents.encode_datasets(datasets, encode, args, supports_alpha=True, cache_is_current=is_current)

    if state["semantic"] is not None:
        state["semantic"].unload()
    if device.type == "cuda":
        summary["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    minutes = summary["audio_seconds"] / 60
    logger.info(
        f"YuE2 latent cache: {summary['segments']} segments from {summary['records']} records ({minutes:.1f} min of audio)"
        f"; VAE {summary['vae_seconds']:.1f}s, semantic {summary['semantic_seconds']:.1f}s"
        + (f", {(summary['vae_seconds'] + summary['semantic_seconds']) / minutes:.2f}s per audio minute" if minutes else "")
        + (f", peak CUDA memory {summary['peak_memory_gib']:.2f} GiB" if "peak_memory_gib" in summary else "")
    )
    if summary["codes_unique_min"] is not None and summary["codes_unique_min"] < 8:
        logger.warning(
            f"some segments have fewer than 8 unique codes (min {summary['codes_unique_min']}); check the audio and the head"
        )
    return summary


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_latents.setup_parser_common(include_vae=False)
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="YuE2 VAE: m-a-p/YuE2-Vae directory or safetensors file (default: vae.* of a ComfyUI --dit)",
    )
    parser.add_argument(
        "--dit", type=str, default=None, help="ComfyUI all-in-one checkpoint to read vae.* from when --vae is not given"
    )
    parser.add_argument(
        "--allow_fp16_vae",
        action="store_true",
        help="accept a non-fp32 VAE source (the int8 all-in-one stores vae.* as fp16); encoded targets shift slightly",
    )
    parser.add_argument("--vae_chunk_frames", type=int, default=750, help="VAE encode chunk length in latent frames (30 s)")
    parser.add_argument(
        "--vae_overlap_frames",
        type=int,
        default=50,
        help="latent frames of context on each side of a VAE chunk (>= receptive field)",
    )
    parser.add_argument("--latent_dtype", type=str, default="float32", choices=LATENT_DTYPES, help="dtype of the cached latents")
    parser.add_argument(
        "--semantic_head",
        type=str,
        default=None,
        help="semantic tokenizer head (Mothersuperior tokenizer_head_*.safetensors or .pt); caches codes_int64 next to the"
        " latents. Required unless --no_codes",
    )
    parser.add_argument("--mert_model", type=str, default=MERT_REPO, help="MERT-v2-FullSong: hub id or local directory")
    parser.add_argument("--mert_revision", type=str, default=None, help="MERT revision (hub id only)")
    parser.add_argument(
        "--mert_dtype", type=str, default="bfloat16", choices=MERT_DTYPES, help="autocast dtype of MERT and the head on CUDA"
    )
    parser.add_argument(
        "--sequential_models",
        action="store_true",
        help="run the VAE and the semantic tokenizer in two passes so only one model is on the device at a time",
    )
    parser.add_argument("--no_codes", action="store_true", help="cache latents only (no semantic codes; NAR text_only training)")
    return parser


def main() -> None:
    args = setup_parser().parse_args()
    if args.disable_cudnn_backend:
        torch.backends.cudnn.enabled = False
    if args.semantic_head is None and not args.no_codes:
        raise ValueError("pass --semantic_head <tokenizer head> to cache semantic codes, or --no_codes for latents only")
    if args.vae_chunk_frames < 1 or args.vae_overlap_frames < 0:
        raise ValueError("--vae_chunk_frames must be >= 1 and --vae_overlap_frames >= 0")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_YUE2)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, audio_spec=YUE2_AUDIO_SPEC)
    datasets = dataset_group.datasets
    plan_yue2_datasets(datasets)  # audio datasets only, in group order

    if args.debug_mode is not None:
        show_audio_datasets(datasets, args.num_workers or 1)
        return

    encode_datasets_yue2(datasets, args, device)


if __name__ == "__main__":
    main()
