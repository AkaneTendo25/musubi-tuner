import argparse
import logging
import os
from typing import Optional

import av
import numpy as np
import torch
import torchaudio

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_MOVA, ItemInfo, save_latent_cache_mova
from musubi_tuner.mova.audio_vae import load_audio_vae
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.wan.modules.vae import WanVAE


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def resolve_video_vae_path(vae_path: str, subfolder: Optional[str]) -> str:
    if subfolder:
        vae_path = os.path.join(vae_path, subfolder)
    if os.path.isdir(vae_path):
        for filename in ["diffusion_pytorch_model.safetensors", "model.safetensors", "pytorch_model.bin"]:
            candidate = os.path.join(vae_path, filename)
            if os.path.exists(candidate):
                return candidate
    return vae_path


def encode_video_latents(vae: WanVAE, batch: list[ItemInfo]) -> tuple[torch.Tensor, torch.Tensor]:
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0

    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
        latent = vae.encode(contents)
    latent = torch.stack(latent, dim=0).to(vae.dtype)

    first_frames = contents[:, :, 0:1]
    padding_frames = contents.shape[2] - 1
    conditioning_inputs = torch.cat(
        [first_frames, torch.zeros(contents.shape[0], 3, padding_frames, contents.shape[3], contents.shape[4], device=vae.device, dtype=vae.dtype)],
        dim=2,
    )
    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
        conditioning = vae.encode(conditioning_inputs)
    conditioning = torch.stack(conditioning, dim=0).to(vae.dtype)

    return latent, conditioning


def load_waveform_from_av(path: str) -> tuple[torch.Tensor, int]:
    container = av.open(path)
    frames = []
    sample_rate = None
    for frame in container.decode(audio=0):
        sample_rate = frame.sample_rate
        array = frame.to_ndarray()
        if array.ndim == 1:
            array = array[None, :]
        elif array.ndim == 2 and array.shape[0] > array.shape[1]:
            array = array.T
        frames.append(torch.from_numpy(array))
    container.close()

    if not frames or sample_rate is None:
        raise ValueError(f"No audio stream found in {path}")

    waveform = torch.cat(frames, dim=-1).to(torch.float32)
    return waveform, sample_rate


def load_waveform(path: str) -> tuple[torch.Tensor, int]:
    try:
        waveform, sample_rate = torchaudio.load(path)
        return waveform.to(torch.float32), int(sample_rate)
    except Exception:
        return load_waveform_from_av(path)


def crop_audio_segment(waveform: torch.Tensor, sample_rate: int, item: ItemInfo, target_sample_rate: int) -> torch.Tensor:
    target_fps = item.target_fps if item.target_fps is not None else item.video_fps
    if target_fps is None:
        raise ValueError(f"target_fps is missing for {item.item_key}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    start_frame = item.frame_start if item.frame_start is not None else 0
    frame_count = item.frame_count if item.frame_count is not None else 1
    start_sec = float(start_frame) / float(target_fps)
    duration_sec = float(frame_count) / float(target_fps)

    start_sample = int(round(start_sec * sample_rate))
    end_sample = int(round((start_sec + duration_sec) * sample_rate))
    waveform = waveform[:, start_sample:end_sample]

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    expected_samples = int(round(duration_sec * target_sample_rate))
    if waveform.shape[1] < expected_samples:
        waveform = torch.nn.functional.pad(waveform, (0, expected_samples - waveform.shape[1]))
    elif waveform.shape[1] > expected_samples:
        waveform = waveform[:, :expected_samples]

    return waveform


def encode_audio_latents(audio_vae, batch: list[ItemInfo]) -> torch.Tensor:
    waveforms = []
    for item in batch:
        source_path = item.audio_path if item.audio_path is not None else item.source_path
        if source_path is None:
            raise ValueError(f"audio source path is missing for {item.item_key}")
        waveform, sample_rate = load_waveform(source_path)
        waveform = crop_audio_segment(waveform, sample_rate, item, audio_vae.sample_rate)
        waveforms.append(waveform)

    waveform_batch = torch.stack(waveforms, dim=0)
    return audio_vae.encode(waveform_batch)


def encode_and_save_batch(vae: WanVAE, audio_vae, batch: list[ItemInfo]):
    latent, conditioning = encode_video_latents(vae, batch)
    audio_latent = encode_audio_latents(audio_vae, batch)

    for i, item in enumerate(batch):
        save_latent_cache_mova(
            item,
            latent[i].detach().cpu(),
            audio_latent[i].detach().cpu(),
            conditioning[i].detach().cpu(),
            video_fps=item.video_fps,
        )


def mova_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--video_vae_subfolder", type=str, default=None, help="optional video VAE subfolder")
    parser.add_argument("--audio_vae", type=str, default=None, help="audio VAE root path or HF repo, defaults to --vae")
    parser.add_argument("--audio_vae_subfolder", type=str, default="audio_vae", help="audio VAE subfolder")
    parser.add_argument("--audio_vae_type", type=str, default="dac", choices=["dac", "oobleck"], help="audio VAE type")
    parser.add_argument(
        "--audio_vae_model_spec",
        type=str,
        default=None,
        help="module spec for official DAC class, e.g. path/to/dac_vae.py:DAC",
    )
    return parser


def main():
    parser = cache_latents.setup_parser_common()
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

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16)
        return

    if args.vae is None:
        raise ValueError("--vae is required for MOVA latent caching")

    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    video_vae_path = resolve_video_vae_path(args.vae, args.video_vae_subfolder)
    logger.info(f"Loading Wan video VAE from {video_vae_path}")
    vae = WanVAE(vae_path=video_vae_path, device=device, dtype=vae_dtype)

    audio_vae_path = args.audio_vae if args.audio_vae is not None else args.vae
    logger.info(f"Loading MOVA audio VAE from {audio_vae_path}")
    audio_vae = load_audio_vae(
        audio_vae_path,
        subfolder=args.audio_vae_subfolder,
        device=device,
        dtype=vae_dtype,
        vae_type=args.audio_vae_type,
        model_spec=args.audio_vae_model_spec,
    )

    def encode(one_batch: list[ItemInfo]):
        encode_and_save_batch(vae, audio_vae, one_batch)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
