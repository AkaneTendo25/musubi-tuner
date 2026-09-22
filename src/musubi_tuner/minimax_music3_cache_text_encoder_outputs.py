"""Cache native MiniMax Music 3 autoregressive conditioning."""

import torch
from musubi_tuner import cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_MUSIC3
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_minimax_music3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.minimax_music3.ar import generate_conditioning, load_ar


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser.add_argument("--ar_model", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    language_model, decoder, tokenizer = load_ar(args.ar_model, device=device)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_MINIMAX_MUSIC3)
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    existing, expected = cache_text_encoder_outputs.prepare_cache_files_and_paths(group.datasets)

    def encode(batch):
        for index, item in enumerate(batch):
            frames = max(1, round(item.frame_count / 44100 * 25))
            hidden = generate_conditioning(language_model, decoder, tokenizer, item.caption, item.lyrics, frames, args.seed + index)
            save_text_encoder_output_cache_minimax_music3(item, hidden)

    cache_text_encoder_outputs.process_text_encoder_batches(args.num_workers, args.skip_existing, 1, group.datasets, existing, expected, encode)
    cache_text_encoder_outputs.post_process_cache_files(group.datasets, existing, expected, args.keep_cache)


if __name__ == "__main__":
    main()
