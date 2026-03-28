import argparse
from typing import Optional

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

import logging

from musubi_tuner.dataset.image_video_dataset import (
    ItemInfo,
    save_latent_cache_common,
    ARCHITECTURE_COSMOS,
    ARCHITECTURE_COSMOS_FULL,
)
from musubi_tuner.utils.model_utils import str_to_dtype, dtype_to_str
from musubi_tuner.wan.modules.vae import WanVAE
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(vae: WanVAE, batch: list[ItemInfo]):
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    h, w = contents.shape[3], contents.shape[4]
    if h < 8 or w < 8:
        item = batch[0]
        raise ValueError(f"Image or video size too small: {item.item_key}, size: {item.original_size}")

    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
        latent = vae.encode(contents)
    latent = torch.stack(latent, dim=0)  # B, C, F, H, W
    latent = latent.to(vae.dtype)

    for i, item in enumerate(batch):
        lat = latent[i]  # C, F, H, W
        _, F, H, W = lat.shape
        dt_str = dtype_to_str(lat.dtype)
        sd = {f"latents_{F}x{H}x{W}_{dt_str}": lat.detach().cpu()}
        save_latent_cache_common(item, sd, ARCHITECTURE_COSMOS_FULL)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--vae", type=str, required=True, help="WAN VAE checkpoint path")
    parser.add_argument("--vae_dtype", type=str, default="float16", help="VAE dtype")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache VAE features on CPU")
    return parser


def cache(args: argparse.Namespace):
    vae_dtype = str_to_dtype(args.vae_dtype)

    assert args.vae is not None, "VAE path is required"

    logger.info(f"Loading VAE: {args.vae}")
    cache_device = torch.device("cpu") if args.vae_cache_cpu else None
    vae = WanVAE(vae_path=args.vae, device="cpu", dtype=vae_dtype, cache_device=cache_device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    def encode_fn(batch: list[ItemInfo]):
        encode_and_save_batch(vae, batch)

    cache_latents.cache(args, ARCHITECTURE_COSMOS, encode_fn)

    logger.info("Done caching latents.")


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = cache_latents.setup_parser_common()
        setup_parser(parser)
        args = parser.parse_args()

    cache(args)


if __name__ == "__main__":
    main()
