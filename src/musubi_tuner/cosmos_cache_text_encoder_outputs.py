import argparse
from typing import Optional

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
import accelerate

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_COSMOS,
    ARCHITECTURE_COSMOS_FULL,
    ItemInfo,
    save_text_encoder_output_cache_common,
)
from musubi_tuner.utils.model_utils import dtype_to_str
from musubi_tuner.cosmos.text_encoder import CosmosTextEncoder

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    text_encoder: CosmosTextEncoder, batch: list[ItemInfo], device: torch.device, accelerator: Optional[accelerate.Accelerator]
):
    prompts = [item.caption for item in batch]

    with torch.no_grad():
        embeddings = text_encoder.encode(prompts)

    for i, item in enumerate(batch):
        emb = embeddings[i]  # [seq_len, 100352]
        sd = {}
        dt_str = dtype_to_str(emb.dtype)
        # Store as "varlen_qwen_{dtype}" to distinguish from T5
        sd[f"varlen_qwen_{dt_str}"] = emb.detach().cpu()
        save_text_encoder_output_cache_common(item, sd, ARCHITECTURE_COSMOS_FULL)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Cosmos text encoder model name or path. Uses Qwen2.5-VL architecture. "
        "(default: nvidia/Cosmos-Reason1-7B)",
    )
    parser.add_argument(
        "--text_encoder_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Text encoder dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--cosmos_text_len",
        type=int,
        default=512,
        help="Max text token length for Qwen encoder (default: 512)",
    )
    return parser


def cache(args: argparse.Namespace):
    te_dtype = torch.bfloat16 if args.text_encoder_dtype == "bfloat16" else torch.float16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading Cosmos text encoder: {args.text_encoder}")
    text_encoder = CosmosTextEncoder(
        model_name_or_path=args.text_encoder,
        device=device,
        dtype=te_dtype,
        max_length=args.cosmos_text_len,
    )

    def encode_fn(batch: list[ItemInfo]):
        encode_and_save_batch(text_encoder, batch, torch.device(device), None)

    cache_text_encoder_outputs.cache(args, ARCHITECTURE_COSMOS, encode_fn)

    logger.info("Done caching text encoder outputs.")


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = cache_text_encoder_outputs.setup_parser_common()
        setup_parser(parser)
        args = parser.parse_args()

    cache(args)


if __name__ == "__main__":
    main()
