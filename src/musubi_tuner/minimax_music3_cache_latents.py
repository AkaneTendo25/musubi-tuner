"""Cache MiniMax Music 3 targets with the official DAV encoder."""

import argparse
import logging

import torch

from musubi_tuner import cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_MUSIC3
from musubi_tuner.dataset.cache_io import save_latent_cache_minimax_music3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.minimax_music3.vocoder import load_official_dav_autoencoder

logger = logging.getLogger(__name__)


def main():
    parser = cache_latents.setup_parser_common()
    parser.add_argument("--dav", required=True, help="Official MiniMaxAI dav.pth (contains encoder)")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, args.dtype)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_MINIMAX_MUSIC3)
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    model = load_official_dav_autoencoder(args.dav, dtype=dtype).to(device)

    def encode(batch):
        for item in batch:
            waveform = item.content.unsqueeze(0).to(device=device, dtype=dtype)
            with torch.inference_mode():
                latent = model.encode(waveform, sample=False)[0]
            save_latent_cache_minimax_music3(item, latent)

    cache_latents.encode_datasets(group.datasets, encode, args, supports_alpha=True)


if __name__ == "__main__":
    main()
