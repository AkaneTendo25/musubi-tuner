import argparse
import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_MOVA, ItemInfo, save_text_encoder_output_cache_mova
from musubi_tuner.mova.text_encoder import encode_hidden_states, load_text_encoder, load_tokenizer
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    tokenizer,
    text_encoder,
    batch: list[ItemInfo],
    device: torch.device,
    max_length: int,
):
    prompts = [item.caption for item in batch]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device=device) for k, v in encoded.items()}

    with torch.no_grad():
        hidden_states = encode_hidden_states(text_encoder, encoded)

    for item, embed, attention_mask in zip(batch, hidden_states, encoded["attention_mask"]):
        valid_length = int(attention_mask.sum().item())
        save_text_encoder_output_cache_mova(item, embed[:valid_length].detach().cpu())


def mova_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="text encoder path or HF repo")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer path or HF repo, defaults to text_encoder")
    parser.add_argument("--text_encoder_subfolder", type=str, default=None, help="optional text encoder subfolder")
    parser.add_argument("--tokenizer_subfolder", type=str, default=None, help="optional tokenizer subfolder")
    parser.add_argument("--text_encoder_dtype", type=str, default="bfloat16", help="dtype for text encoder")
    parser.add_argument("--max_length", type=int, default=512, help="maximum prompt length")
    parser.add_argument("--trust_remote_code", action="store_true", help="allow trust_remote_code when loading tokenizer/text encoder")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = mova_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MOVA)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    text_encoder_path = args.text_encoder
    tokenizer_path = args.tokenizer if args.tokenizer is not None else args.text_encoder

    logger.info(
        f"Loading tokenizer: {tokenizer_path}"
        + (f" (subfolder={args.tokenizer_subfolder})" if args.tokenizer_subfolder is not None else "")
    )
    tokenizer = load_tokenizer(
        tokenizer_path,
        subfolder=args.tokenizer_subfolder,
        trust_remote_code=args.trust_remote_code,
    )

    text_encoder_dtype = str_to_dtype(args.text_encoder_dtype)
    logger.info(
        f"Loading text encoder: {text_encoder_path}"
        + (f" (subfolder={args.text_encoder_subfolder})" if args.text_encoder_subfolder is not None else "")
    )
    text_encoder = load_text_encoder(
        text_encoder_path,
        text_encoder_dtype,
        subfolder=args.text_encoder_subfolder,
        trust_remote_code=args.trust_remote_code,
    )
    text_encoder.eval().requires_grad_(False).to(device=device, dtype=text_encoder_dtype)

    def encode(batch: list[ItemInfo]):
        encode_and_save_batch(tokenizer, text_encoder, batch, device, args.max_length)

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
