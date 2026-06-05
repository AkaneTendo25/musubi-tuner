import argparse

import torch

from musubi_tuner.cosmos3 import cosmos3_utils
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.cache_io import save_latent_cache_cosmos3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_COSMOS3, ItemInfo
from musubi_tuner.dataset.media_utils import load_audio
from musubi_tuner.utils.model_utils import str_to_dtype
import musubi_tuner.cache_latents as cache_latents

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _encode_sound_latent(sound_tokenizer, item: ItemInfo, sample_rate: int, channels: int, allow_missing_audio: bool):
    frame_count = item.frame_count or 1
    start_frame = item.frame_start or 0
    audio = None
    if item.source_path is not None:
        try:
            audio = load_audio(
                item.source_path,
                start_frame=start_frame,
                end_frame=start_frame + frame_count,
                sample_rate=sample_rate,
                channels=channels,
                video_fps=item.audio_fps,
            )
        except ValueError:
            audio = None

    if audio is None:
        if not allow_missing_audio:
            raise ValueError(f"Audio caching requires source audio for {item.item_key}")
        samples = int(round(frame_count / float(item.audio_fps or 24.0) * sample_rate))
        audio = torch.zeros((channels, samples), dtype=torch.float32).numpy()

    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(sound_tokenizer.device, dtype=sound_tokenizer.dtype)
    with torch.no_grad():
        return cosmos3_utils.encode_audio_to_latents(sound_tokenizer, audio_tensor)[0]


def encode_and_save_batch(vae, batch: list[ItemInfo], sound_tokenizer=None, args: argparse.Namespace | None = None):
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B,H,W,C -> B,F,H,W,C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0

    h, w = contents.shape[3], contents.shape[4]
    if h < 16 or w < 16:
        item = batch[0]
        raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    with torch.no_grad():
        latents = cosmos3_utils.encode_video_to_latents(vae, contents)

    for item, latent in zip(batch, latents):
        sound_latent = None
        if sound_tokenizer is not None:
            assert args is not None
            sound_latent = _encode_sound_latent(
                sound_tokenizer,
                item,
                sample_rate=args.sound_sample_rate,
                channels=args.sound_channels,
                allow_missing_audio=args.allow_missing_audio,
            )
        save_latent_cache_cosmos3(
            item,
            latent,
            sound_latent=sound_latent,
            sound_sample_rate=args.sound_sample_rate if sound_latent is not None and args is not None else None,
            sound_latent_fps=args.sound_latent_fps if sound_latent is not None and args is not None else None,
        )


def main():
    parser = cache_latents.setup_parser_common()
    parser = cosmos3_latent_cache_setup_parser(parser)
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_COSMOS3)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=int(args.fps)
        )
        return

    vae_source = args.vae if args.vae is not None else args.model
    assert vae_source is not None, "--vae or --model is required"

    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae = cosmos3_utils.load_vae(vae_source, args.vae_subfolder, dtype=vae_dtype, device=device)
    sound_tokenizer = None
    if args.cache_audio:
        sound_source = args.sound_tokenizer if args.sound_tokenizer is not None else args.model
        assert sound_source is not None, "--sound_tokenizer or --model is required for --cache_audio"
        sound_dtype = torch.bfloat16 if args.sound_dtype is None else str_to_dtype(args.sound_dtype)
        sound_tokenizer = cosmos3_utils.load_sound_tokenizer(
            sound_source,
            args.sound_tokenizer_subfolder,
            dtype=sound_dtype,
            device=device,
        )

    def encode(one_batch: list[ItemInfo]):
        encode_and_save_batch(vae, one_batch, sound_tokenizer=sound_tokenizer, args=args)

    cache_latents.encode_datasets(datasets, encode, args)


def cosmos3_latent_cache_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", type=str, default=None, help="Cosmos3 model repo/path used as VAE fallback")
    parser.add_argument("--vae_subfolder", type=str, default="vae", help="subfolder for Wan2.2/Cosmos3 VAE weights")
    parser.add_argument("--fps", type=float, default=24.0, help="preview FPS for debug video output")
    parser.add_argument("--cache_audio", action="store_true", help="also cache Cosmos3 AVAE audio latents from video audio")
    parser.add_argument("--sound_tokenizer", type=str, default=None, help="Cosmos3 model repo/path or AVAE path, defaults to --model")
    parser.add_argument("--sound_tokenizer_subfolder", type=str, default="sound_tokenizer", help="subfolder for AVAE weights")
    parser.add_argument("--sound_dtype", type=str, default=None, help="data type for AVAE, default is bfloat16")
    parser.add_argument("--sound_sample_rate", type=int, default=48000, help="AVAE audio sample rate")
    parser.add_argument("--sound_channels", type=int, default=2, help="AVAE audio channel count")
    parser.add_argument("--sound_latent_fps", type=float, default=25.0, help="AVAE latent FPS stored in cache metadata")
    parser.add_argument("--allow_missing_audio", action="store_true", help="cache silent audio if a source video has no audio stream")
    return parser


if __name__ == "__main__":
    main()
