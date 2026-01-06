from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional

import torch
import torchaudio
from safetensors.torch import save_file

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTXV2, BaseDataset, ItemInfo
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_datasets(args: argparse.Namespace) -> List[BaseDataset]:
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LTXV2)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    return list(dataset_group.datasets)


def _audio_cache_path(item_info: ItemInfo) -> str:
    base_dir = os.path.dirname(item_info.latent_cache_path)
    base_name = os.path.basename(item_info.latent_cache_path)
    if not base_name.endswith("_ltxv2.safetensors"):
        return os.path.join(base_dir, f"{item_info.item_key}_ltxv2_audio.safetensors")
    return os.path.join(base_dir, base_name.replace("_ltxv2.safetensors", "_ltxv2_audio.safetensors"))


def _resolve_audio_path(item_info: ItemInfo, audio_dir: Optional[str], audio_ext: str) -> str:
    base = os.path.splitext(os.path.basename(item_info.item_key))[0]
    if audio_dir is not None:
        return os.path.join(audio_dir, base + audio_ext)
    return os.path.join(os.path.dirname(item_info.item_key), base + audio_ext)


def encode_and_save_audio_cache(
    encoder,
    processor,
    item_info: ItemInfo,
    *,
    audio_path: str,
    dtype: torch.dtype,
) -> None:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found for {item_info.item_key}: {audio_path}")

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.dim() != 2:
        raise ValueError(f"Unexpected waveform shape from {audio_path}: {tuple(waveform.shape)}")

    waveform = waveform.unsqueeze(0)  # [1, C, T]
    device = next(encoder.parameters()).device
    waveform = waveform.to(device=device, dtype=dtype)

    with torch.no_grad():
        mel = processor.waveform_to_mel(waveform, int(sample_rate))
        latents = encoder(mel)

    # latents: [1, C, T, F]
    latents = latents[0].detach().cpu().contiguous()
    time_steps = latents.shape[1]
    mel_bins = latents.shape[2]
    channels = latents.shape[0]

    dtype_str = (
        cache_latents.dtype_to_str(dtype)
        if hasattr(cache_latents, "dtype_to_str")
        else ("fp16" if dtype == torch.float16 else "bf16" if dtype == torch.bfloat16 else "fp32")
    )

    audio_lengths = torch.tensor(time_steps, dtype=torch.int32)
    int_dtype_str = cache_latents.dtype_to_str(audio_lengths.dtype) if hasattr(cache_latents, "dtype_to_str") else "int32"

    audio_cache_path = _audio_cache_path(item_info)
    os.makedirs(os.path.dirname(audio_cache_path), exist_ok=True)
    sd = {
        f"audio_latents_{time_steps}x{mel_bins}x{channels}_{dtype_str}": latents,
        f"audio_lengths_{int_dtype_str}": audio_lengths,
    }

    metadata = {
        "architecture": "ltxv2_v1",
        "format_version": "1.0.1",
    }

    save_file(sd, audio_cache_path, metadata=metadata)


def main() -> None:
    parser = cache_latents.setup_parser_common()
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing audio files (optional)")
    parser.add_argument("--audio_ext", type=str, default=".wav", help="Audio file extension (default: .wav)")
    parser.add_argument("--ltx2_checkpoint", type=str, required=True, help="Path to LTX-2 checkpoint (.safetensors)")
    parser.add_argument("--audio_dtype", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    datasets = _load_datasets(args)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_dtype = torch.float16 if args.audio_dtype is None else str_to_dtype(args.audio_dtype)

    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.audio_vae.model_configurator import AudioEncoderConfigurator, AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER
    from ltx_core.model.audio_vae.ops import AudioProcessor

    encoder = SingleGPUModelBuilder(
        model_path=str(args.ltx2_checkpoint),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=audio_dtype)
    encoder.eval()

    processor = AudioProcessor(
        sample_rate=int(getattr(encoder, "sample_rate", 16000)),
        mel_bins=int(getattr(encoder, "mel_bins", 64)),
        mel_hop_length=int(getattr(encoder, "mel_hop_length", 160)),
        n_fft=int(getattr(encoder, "n_fft", 1024)),
    ).to(device=device, dtype=audio_dtype)
    processor.eval()

    for ds in datasets:
        num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) - 1)
        for _bucket_key, batch in ds.retrieve_latent_cache_batches(num_workers):
            for item_info in batch:
                audio_cache_path = _audio_cache_path(item_info)
                if args.skip_existing and os.path.exists(audio_cache_path):
                    continue
                audio_path = _resolve_audio_path(item_info, args.audio_dir, args.audio_ext)
                encode_and_save_audio_cache(
                    encoder,
                    processor,
                    item_info,
                    audio_path=audio_path,
                    dtype=audio_dtype,
                )


if __name__ == "__main__":
    main()
