import argparse
import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_MAGIHUMAN,
    ItemInfo,
    save_text_encoder_output_cache_magihuman,
)
from musubi_tuner.magihuman.model.t5_gemma import get_t5_gemma_embedding
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def resolve_bnb_4bit_compute_dtype(arg: str, weight_dtype: torch.dtype) -> torch.dtype | None:
    if arg == "auto":
        return weight_dtype
    if arg == "fp16":
        return torch.float16
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported bnb compute dtype: {arg}")


def magihuman_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Path to the local T5-Gemma encoder directory.")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", help="Weight dtype for the upstream T5-Gemma encoder.")
    parser.add_argument("--t5gemma_load_in_8bit", action="store_true", help="Load T5-Gemma encoder in 8-bit (bitsandbytes). CUDA only.")
    parser.add_argument("--t5gemma_load_in_4bit", action="store_true", help="Load T5-Gemma encoder in 4-bit (bitsandbytes). CUDA only.")
    parser.add_argument(
        "--t5gemma_bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="bitsandbytes 4-bit quant type for T5-Gemma.",
    )
    parser.add_argument(
        "--t5gemma_bnb_4bit_disable_double_quant",
        action="store_true",
        help="Disable bitsandbytes double quant for T5-Gemma 4-bit loading.",
    )
    parser.add_argument(
        "--t5gemma_bnb_4bit_compute_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Compute dtype for 4-bit T5-Gemma loading (auto uses --weight_dtype).",
    )
    return parser


def resolve_text_encoder_path(args) -> str:
    return args.text_encoder


def encode_and_save_batch(
    get_t5_gemma_embedding,
    text_encoder_path: str,
    device: torch.device,
    weight_dtype: torch.dtype,
    load_in_8bit: bool,
    load_in_4bit: bool,
    bnb_4bit_quant_type: str,
    bnb_4bit_use_double_quant: bool,
    bnb_4bit_compute_dtype: torch.dtype | None,
    batch: list[ItemInfo],
):
    for item in batch:
        embed = get_t5_gemma_embedding(
            item.caption,
            text_encoder_path,
            str(device),
            weight_dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )[0].cpu()
        save_text_encoder_output_cache_magihuman(item, embed)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = magihuman_setup_parser(parser)
    args = parser.parse_args()
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MAGIHUMAN)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    text_encoder_path = resolve_text_encoder_path(args)
    weight_dtype = str_to_dtype(args.weight_dtype)
    bnb_4bit_compute_dtype = resolve_bnb_4bit_compute_dtype(args.t5gemma_bnb_4bit_compute_dtype, weight_dtype)

    def encode(batch: list[ItemInfo]):
        encode_and_save_batch(
            get_t5_gemma_embedding,
            text_encoder_path,
            device,
            weight_dtype,
            bool(args.t5gemma_load_in_8bit),
            bool(args.t5gemma_load_in_4bit),
            args.t5gemma_bnb_4bit_quant_type,
            not bool(args.t5gemma_bnb_4bit_disable_double_quant),
            bnb_4bit_compute_dtype,
            batch,
        )

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode,
    )
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
