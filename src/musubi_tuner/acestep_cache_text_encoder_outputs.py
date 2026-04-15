"""
Cache text encoder outputs for ACE-Step architecture.

This script encodes text prompts using an ACE-Step-compatible text encoder
and caches the embeddings for faster training.
"""

import argparse
import logging

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_ACESTEP,
    ItemInfo,
    save_text_encoder_output_cache_acestep,
)
from musubi_tuner.acestep import acestep_utils
from musubi_tuner.acestep.acestep_config import TEXT_ENCODER_MAX_LENGTH
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
_warned_missing_embed_tokens = False


def _get_embed_tokens_layer(text_encoder):
    if hasattr(text_encoder, "embed_tokens"):
        return text_encoder.embed_tokens
    if hasattr(text_encoder, "model") and hasattr(text_encoder.model, "embed_tokens"):
        return text_encoder.model.embed_tokens
    if hasattr(text_encoder, "get_input_embeddings"):
        return text_encoder.get_input_embeddings()
    return None


def encode_and_save_batch(
    tokenizer,
    text_encoder,
    batch: list[ItemInfo],
    device: torch.device,
    max_length: int,
    condition_encoder=None,
    cache_metadata: dict[str, str] | None = None,
):
    """Encode a batch of prompts and save their text encoder outputs.

    Args:
        tokenizer: ACE-Step-compatible tokenizer
        text_encoder: ACE-Step-compatible text encoder model
        batch: List of ItemInfo containing captions to encode
        device: Device to use for encoding
        max_length: Maximum token length
    """
    for item in batch:
        caption = item.caption
        lyrics = getattr(item, "lyrics", "")
        item_bpm = getattr(item, "bpm", None)
        item_key = getattr(item, "keyscale", None)
        item_timesig = getattr(item, "timesignature", None)
        item_duration = getattr(item, "duration", None)
        if isinstance(item_timesig, str):
            item_timesig = int(item_timesig) if item_timesig.isdigit() else None

        # Align with the original trainer flow by caching condition-encoder outputs
        # (not raw text encoder outputs) when DiT is provided.
        if condition_encoder is not None:
            # Text branch: prompt only. Lyrics are encoded separately for lyric branch.
            formatted_prompt = acestep_utils.format_text_for_acestep(
                caption,
                "",
                bpm=item_bpm,
                key=item_key,
                time_signature=item_timesig,
                duration=item_duration,
            )
            if "\n\n# Lyrics\n" in formatted_prompt:
                formatted_prompt = formatted_prompt.split("\n\n# Lyrics\n", 1)[0]

            text_inputs = tokenizer(
                formatted_prompt,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs["input_ids"].to(device)
            text_attention_mask = text_inputs["attention_mask"].to(device)

            with torch.no_grad():
                text_outputs = text_encoder(text_input_ids)
                if hasattr(text_outputs, "last_hidden_state"):
                    text_hidden_states = text_outputs.last_hidden_state
                elif isinstance(text_outputs, tuple):
                    text_hidden_states = text_outputs[0]
                else:
                    text_hidden_states = text_outputs

            # Format lyrics with language header like official handler
            lyrics_text = acestep_utils.format_lyrics_for_acestep(lyrics, language="unknown")
            lyric_inputs = tokenizer(
                lyrics_text,
                max_length=2048,  # Official uses 2048 for lyrics
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            lyric_input_ids = lyric_inputs["input_ids"].to(device)
            lyric_attention_mask = lyric_inputs["attention_mask"].to(device)

            with torch.no_grad():
                encoder_dtype = next(condition_encoder.parameters()).dtype
                text_hidden_states = text_hidden_states.to(device=device, dtype=encoder_dtype)
                text_attention_mask = text_attention_mask.to(device=device)

                embed_tokens = _get_embed_tokens_layer(text_encoder)
                if embed_tokens is not None:
                    lyric_hidden_states = embed_tokens(lyric_input_ids).to(device=device, dtype=encoder_dtype)
                else:
                    logger.warning("text_encoder has no embedding layer accessor; using zero lyric embeddings")
                    lyric_hidden_states = torch.zeros(
                        lyric_input_ids.shape[0],
                        lyric_input_ids.shape[1],
                        text_hidden_states.shape[-1],
                        device=device,
                        dtype=encoder_dtype,
                    )
                lyric_attention_mask = lyric_attention_mask.to(device=device)
                refer_audio_hidden = torch.zeros(1, 1, 64, device=device, dtype=encoder_dtype)
                refer_audio_order_mask = torch.zeros(1, device=device, dtype=torch.long)

                hidden_states, attention_mask = condition_encoder(
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
                    refer_audio_order_mask=refer_audio_order_mask,
                )
        else:
            # Fallback mode: cache prompt branch + lyric branch separately so the
            # trainer can still run the ACE-Step condition encoder correctly.
            formatted_prompt = acestep_utils.format_text_for_acestep(
                caption,
                "",
                bpm=item_bpm,
                key=item_key,
                time_signature=item_timesig,
                duration=item_duration,
            )
            if "\n\n# Lyrics\n" in formatted_prompt:
                formatted_prompt = formatted_prompt.split("\n\n# Lyrics\n", 1)[0]

            text_inputs = tokenizer(
                formatted_prompt,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)

            with torch.no_grad():
                text_outputs = text_encoder(text_input_ids)
                if hasattr(text_outputs, "last_hidden_state"):
                    hidden_states = text_outputs.last_hidden_state
                elif isinstance(text_outputs, tuple):
                    hidden_states = text_outputs[0]
                else:
                    hidden_states = text_outputs

            lyrics_text = acestep_utils.format_lyrics_for_acestep(lyrics, language="unknown")
            lyric_inputs = tokenizer(
                lyrics_text,
                max_length=2048,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            lyric_input_ids = lyric_inputs["input_ids"].to(device)
            lyric_attention_mask = lyric_inputs["attention_mask"].to(device)

            embed_tokens = _get_embed_tokens_layer(text_encoder)
            if embed_tokens is not None:
                with torch.no_grad():
                    lyric_hidden_states = embed_tokens(lyric_input_ids)
            else:
                global _warned_missing_embed_tokens
                if not _warned_missing_embed_tokens:
                    logger.warning("text_encoder has no embedding layer accessor; using zero lyric embeddings")
                    _warned_missing_embed_tokens = True
                lyric_hidden_states = torch.zeros(
                    lyric_input_ids.shape[0],
                    lyric_input_ids.shape[1],
                    hidden_states.shape[-1],
                    device=device,
                    dtype=hidden_states.dtype,
                )

        # Remove batch dimension and move to CPU
        hidden_states = hidden_states.squeeze(0).cpu()  # [L, D]
        attention_mask = attention_mask.squeeze(0).cpu()  # [L]

        # Trim to actual length based on attention mask
        actual_length = int(attention_mask.sum().item())
        hidden_states_trimmed = hidden_states[:actual_length]  # [actual_length, D]
        attention_mask_trimmed = attention_mask[:actual_length]  # [actual_length]

        lyric_hidden_states_trimmed = None
        lyric_attention_mask_trimmed = None
        if condition_encoder is None:
            lyric_hidden_states = lyric_hidden_states.squeeze(0).cpu()
            lyric_attention_mask = lyric_attention_mask.squeeze(0).cpu()
            lyric_length = int(lyric_attention_mask.sum().item())
            lyric_hidden_states_trimmed = lyric_hidden_states[:lyric_length]
            lyric_attention_mask_trimmed = lyric_attention_mask[:lyric_length]

        logger.debug(f"Saving text cache for {item.item_key}: {hidden_states_trimmed.shape}")

        save_text_encoder_output_cache_acestep(
            item,
            hidden_states_trimmed,
            attention_mask_trimmed,
            lyric_hidden_states=lyric_hidden_states_trimmed,
            lyric_attention_mask=lyric_attention_mask_trimmed,
            cache_metadata=cache_metadata,
        )


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = acestep_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_ACESTEP)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # Prepare cache files and paths
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(
        datasets
    )

    # Load ACE-Step-compatible tokenizer and text encoder
    logger.info(f"Loading text encoder from {args.text_encoder}")
    tokenizer, text_encoder = acestep_utils.load_text_encoder(
        args.text_encoder,
        device=str(device),
        dtype=torch.bfloat16,
    )
    text_encoder.eval()

    condition_encoder = None
    model = None
    cache_metadata = acestep_utils.build_acestep_text_cache_metadata(
        cache_kind="raw",
        text_encoder_path=args.text_encoder,
    )
    if args.dit is not None:
        logger.info(f"Loading ACE-Step DiT condition encoder from {args.dit}")
        model, _ = acestep_utils.load_acestep_model(
            args.dit,
            device=str(device),
            dtype=torch.bfloat16,
            attn_mode=args.attn_mode,
        )
        condition_encoder = model.encoder
        condition_encoder.eval()
        cache_metadata = acestep_utils.build_acestep_text_cache_metadata(
            cache_kind="conditioned",
            text_encoder_path=args.text_encoder,
            model_config=model.config,
            model_path=args.dit,
        )

    # Get max length from args
    max_length = args.max_length

    # Encode with the ACE-Step-compatible text encoder
    logger.info("Encoding prompts with text encoder")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal tokenizer, text_encoder, condition_encoder
        encode_and_save_batch(
            tokenizer,
            text_encoder,
            batch,
            device,
            max_length,
            condition_encoder=condition_encoder,
            cache_metadata=cache_metadata,
        )

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )

    # Clean up
    del tokenizer, text_encoder
    if model is not None:
        del model

    # Remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )

    logger.info("Done!")


def acestep_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ACE-Step specific arguments to the parser."""
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="ACE-Step-compatible text encoder checkpoint path or directory",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=TEXT_ENCODER_MAX_LENGTH,
        help=f"Maximum token length (default: {TEXT_ENCODER_MAX_LENGTH})",
    )
    parser.add_argument(
        "--dit",
        type=str,
        default=None,
        help="Optional ACE-Step DiT path to cache condition-encoder outputs like the original trainer",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="sdpa",
        help="Attention mode used when loading --dit (sdpa, flash_attn, eager)",
    )
    return parser


if __name__ == "__main__":
    main()
