import argparse
import os

import torch
from tqdm import tqdm

from musubi_tuner.cosmos3 import cosmos3_utils
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_cosmos3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_COSMOS3, ItemInfo
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_item(tokenizer, item: ItemInfo, args: argparse.Namespace):
    width, height = item.bucket_size[0], item.bucket_size[1]
    frame_count = item.frame_count if item.frame_count is not None else 1
    cond_input_ids, _ = cosmos3_utils.tokenize_prompt(
        tokenizer,
        prompt=item.caption,
        negative_prompt=None,
        num_frames=frame_count,
        height=height,
        width=width,
        fps=args.fps,
        use_system_prompt=bool(args.system_prompt and not args.no_system_prompt),
        add_resolution_template=not args.no_resolution_template,
        add_duration_template=not args.no_duration_template,
    )
    save_text_encoder_output_cache_cosmos3(item, torch.tensor(cond_input_ids, dtype=torch.int64))


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = cosmos3_text_cache_setup_parser(parser)
    args = parser.parse_args()

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_COSMOS3)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    tokenizer_path = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = cosmos3_utils.load_tokenizer(tokenizer_path, args.tokenizer_subfolder)
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)

    for dataset_index, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{dataset_index}]")
        all_cache_files = all_cache_files_for_dataset[dataset_index]
        all_cache_paths = all_cache_paths_for_dataset[dataset_index]

        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            batch: list[ItemInfo]
            for item in batch:
                item.text_encoder_output_cache_path = dataset.get_text_encoder_output_cache_path(item)

            all_cache_paths.update([os.path.normpath(item.text_encoder_output_cache_path) for item in batch])
            if args.skip_existing:
                batch = [item for item in batch if os.path.normpath(item.text_encoder_output_cache_path) not in all_cache_files]
                if len(batch) == 0:
                    continue

            for item in batch:
                encode_and_save_item(tokenizer, item, args)

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def cosmos3_text_cache_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", type=str, required=True, help="Cosmos3 model repo/path used as tokenizer fallback")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer path/repo, defaults to --model")
    parser.add_argument("--tokenizer_subfolder", type=str, default="text_tokenizer", help="subfolder for Cosmos3 tokenizer")
    parser.add_argument("--fps", type=float, default=24.0, help="FPS used in Cosmos3 duration template and mRoPE")
    parser.add_argument("--system_prompt", action="store_true", help="enable the Cosmos3 system prompt wrapper")
    parser.add_argument("--no_system_prompt", action="store_true", help="disable the Cosmos3 system prompt wrapper")
    parser.add_argument("--no_resolution_template", action="store_true", help="disable the Cosmos3 resolution template")
    parser.add_argument("--no_duration_template", action="store_true", help="disable the Cosmos3 duration template")
    return parser


if __name__ == "__main__":
    main()
