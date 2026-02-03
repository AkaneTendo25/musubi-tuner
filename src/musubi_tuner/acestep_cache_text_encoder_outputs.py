"""
Cache text encoder outputs for ACE-Step 1.5 architecture.

This script encodes text prompts using ACE-Step's Qwen3-Embedding text encoder
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


def encode_and_save_batch(
    tokenizer,
    text_encoder,
    batch: list[ItemInfo],
    device: torch.device,
    max_length: int,
):
    """Encode a batch of prompts and save their text encoder outputs.

    Args:
        tokenizer: Qwen3 tokenizer
        text_encoder: Qwen3 text encoder model
        batch: List of ItemInfo containing captions to encode
        device: Device to use for encoding
        max_length: Maximum token length
    """
    for item in batch:
        caption = item.caption
        lyrics = getattr(item, "lyrics", "")

        # Encode using ACE-Step format
        hidden_states, attention_mask = acestep_utils.encode_text_for_acestep(
            tokenizer,
            text_encoder,
            caption,
            lyrics,
            max_length=max_length,
            device=str(device),
        )

        # Remove batch dimension and move to CPU
        hidden_states = hidden_states.squeeze(0).cpu()  # [L, D]
        attention_mask = attention_mask.squeeze(0).cpu()  # [L]

        # Trim to actual length based on attention mask
        actual_length = int(attention_mask.sum().item())
        hidden_states_trimmed = hidden_states[:actual_length]  # [actual_length, D]
        attention_mask_trimmed = attention_mask[:actual_length]  # [actual_length]

        logger.debug(f"Saving text cache for {item.item_key}: {hidden_states_trimmed.shape}")

        save_text_encoder_output_cache_acestep(item, hidden_states_trimmed, attention_mask_trimmed)


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

    # Load Qwen3 tokenizer and text encoder
    logger.info(f"Loading Qwen3 text encoder from {args.text_encoder}")
    tokenizer, text_encoder = acestep_utils.load_text_encoder(
        args.text_encoder,
        device=str(device),
        dtype=torch.bfloat16,
    )
    text_encoder.eval()

    # Get max length from args
    max_length = args.max_length

    # Encode with Qwen3 text encoder
    logger.info("Encoding prompts with Qwen3 text encoder")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal tokenizer, text_encoder
        encode_and_save_batch(tokenizer, text_encoder, batch, device, max_length)

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
        help="Qwen3-Embedding text encoder checkpoint path or directory",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=TEXT_ENCODER_MAX_LENGTH,
        help=f"Maximum token length (default: {TEXT_ENCODER_MAX_LENGTH})",
    )
    return parser


if __name__ == "__main__":
    main()
