"""YuE2 text cache: token ids only (YuE2 has no separate text encoder; the prompt is part of the LM sequence).

One file per record: ``[EOD] + enc(prompt)`` for each cot mode, the matching negative (instruction-only) ids, and the
record's ABC ids with their flavour. Prefixes are assembled at train time.
"""

from __future__ import annotations

import argparse
import hashlib
import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_YUE2
from musubi_tuner.dataset.audio_dataset import SONG_ID_METADATA_KEY
from musubi_tuner.dataset.cache_io import YUE2_ABC_MODE_IDS, YUE2_TEXT_CACHE_VERSION, save_text_encoder_output_cache_yue2
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.yue2.cache_plan import YuE2Record, item_record, latent_metadata_matches, plan_yue2_datasets
from musubi_tuner.yue2.yue2_checkpoint import load_yue2_tokenizer
from musubi_tuner.yue2.yue2_protocol import (
    COT_MODES,
    DEFAULT_INSTRUMENTAL_LYRICS,
    EOD,
    PROTOCOL_VERSION,
    negative_text_ids,
    normalize_prompt_fields,
    text_ids,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_text_inputs(record: YuE2Record, instrumental_lyrics: str) -> tuple[str, str, str, int]:
    """(style, lyrics, abc, abc_mode id) as tokenized: normalised prompt fields, ABC text or "" and its flavour."""
    style, lyrics = normalize_prompt_fields(record.style, record.lyrics, instrumental_lyrics)
    abc = record.abc or ""
    abc_mode = YUE2_ABC_MODE_IDS[record.abc_mode if abc else None]
    return style, lyrics, abc, abc_mode


def text_identity(record: YuE2Record, tokenizer, instrumental_lyrics: str) -> dict[str, str]:
    """Metadata a current text cache carries (the --skip_existing check)."""
    style, lyrics, abc, abc_mode = record_text_inputs(record, instrumental_lyrics)
    return {
        "yue2_text_cache_version": YUE2_TEXT_CACHE_VERSION,
        "yue2_protocol": PROTOCOL_VERSION,
        "yue2_tokenizer": tokenizer.fingerprint,
        SONG_ID_METADATA_KEY: record.song_id,
        "yue2_style_sha": _sha(style),
        "yue2_lyrics_sha": _sha(lyrics),
        "yue2_abc_sha": _sha(abc),
        "yue2_abc_mode": str(abc_mode),
        "yue2_instrumental_lyrics": instrumental_lyrics,
    }


def encode_record(tokenizer, record: YuE2Record, instrumental_lyrics: str):
    """(texts, negatives, abc ids, abc_mode id) of one record."""
    style, lyrics, abc, abc_mode = record_text_inputs(record, instrumental_lyrics)
    texts = {cot: torch.tensor(text_ids(tokenizer, style, lyrics, cot), dtype=torch.int64) for cot in COT_MODES}
    negatives = {cot: torch.tensor(negative_text_ids(tokenizer, cot), dtype=torch.int64) for cot in COT_MODES}
    abc_ids = torch.tensor(tokenizer.encode(abc) if abc else [], dtype=torch.int64)
    if abc_ids.numel() and (int(abc_ids.min()) < 0 or int(abc_ids.max()) >= EOD):
        raise ValueError(f"ABC of {record.item_key} encodes outside the ordinary text vocabulary")
    if abc and abc_ids.numel() == 0:
        raise ValueError(f"ABC of {record.item_key} encodes to no tokens")
    return texts, negatives, abc_ids, abc_mode


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="YuE2 text tokenizer: a tokenizer .json, qwen.tiktoken, or a directory holding one"
        " (default: embedded in a ComfyUI --dit, else qwen.tiktoken next to a Hugging Face model.safetensors)",
    )
    parser.add_argument(
        "--dit", type=str, default=None, help="YuE2 checkpoint the tokenizer is taken from when --tokenizer is not given"
    )
    parser.add_argument(
        "--instrumental_lyrics",
        type=str,
        default=DEFAULT_INSTRUMENTAL_LYRICS,
        help=f"lyrics used for songs without lyrics (default: {DEFAULT_INSTRUMENTAL_LYRICS}); the trainer reads it from the cache",
    )
    parser.add_argument(
        "--debug_mode", type=str, default=None, choices=["console"], help="print the prompts and token counts, write nothing"
    )
    return parser


def main() -> None:
    args = setup_parser().parse_args()

    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_YUE2)
    # no audio spec: text caching never decodes audio
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = dataset_group.datasets
    plans = plan_yue2_datasets(datasets)

    tokenizer = load_yue2_tokenizer(args.tokenizer, args.dit)
    logger.info(f"YuE2 tokenizer: {tokenizer.backend}, {tokenizer.fingerprint[:23]}")

    if args.debug_mode is not None:
        for i, dataset in enumerate(datasets):
            print(f"Dataset [{i}]")
            for batch in dataset.retrieve_text_encoder_output_cache_batches(1):
                for item in batch:
                    record = item_record(plans, item)
                    style, lyrics, abc, abc_mode = record_text_inputs(record, args.instrumental_lyrics)
                    texts, negatives, abc_ids, _ = encode_record(tokenizer, record, args.instrumental_lyrics)
                    lengths = ", ".join(f"{cot} {len(texts[cot])}" for cot in COT_MODES)
                    print(f"  {item.item_key} (song {record.song_id}): text ids {lengths}; ABC {len(abc_ids)} ids, mode {abc_mode}")
                    print(f"    style: {style[:200]!r}")
                    print(f"    lyrics: {lyrics[:200]!r}")
        return

    def expected(item: ItemInfo) -> dict[str, str]:
        return text_identity(item_record(plans, item), tokenizer, args.instrumental_lyrics)

    def cache_is_current(item: ItemInfo) -> bool:
        return latent_metadata_matches(item.text_encoder_output_cache_path, expected(item))

    def encode(batch: list[ItemInfo]) -> None:
        for item in batch:
            record = item_record(plans, item)
            texts, negatives, abc_ids, abc_mode = encode_record(tokenizer, record, args.instrumental_lyrics)
            meta = expected(item)
            meta.pop("yue2_text_cache_version")  # written by the cache writer
            save_text_encoder_output_cache_yue2(item, texts, negatives, abc_ids, abc_mode, meta)

    all_cache_files, all_cache_paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)
    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files,
        all_cache_paths,
        encode,
        requires_content=False,
        cache_is_current=cache_is_current,
    )
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()
