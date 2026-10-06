from __future__ import annotations

import logging
import math

import numpy as np
import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.audio_utils import AudioSpec, add_audio_tolerance_arguments, apply_audio_tolerance_arguments
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

K6_SAMPLE_RATE = 44100
K6_AUDIO_DOWNSAMPLE = 1024
K6_VIDEO_FPS = 24


def audio_samples_for_frames(frame_count: int) -> int:
    """Return the waveform window whose encoded grid spans the video duration."""
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    latent_frames = math.ceil(frame_count / K6_VIDEO_FPS * K6_SAMPLE_RATE / K6_AUDIO_DOWNSAMPLE)
    return latent_frames * K6_AUDIO_DOWNSAMPLE


def audio_latent_frames_for_video(frame_count: int) -> int:
    return audio_samples_for_frames(frame_count) // K6_AUDIO_DOWNSAMPLE


K6_AUDIO_SPEC = AudioSpec(
    sample_rate=K6_SAMPLE_RATE,
    channels=1,
    samples_per_crop=audio_samples_for_frames,
)


def _encode_pixels(video_vae, pixels: torch.Tensor) -> torch.Tensor:
    device = next(video_vae.parameters()).device
    dtype = next(video_vae.parameters()).dtype
    posterior = video_vae.encode(pixels.unsqueeze(0).to(device, dtype)).latent_dist
    return posterior.sample()[0] * video_vae.config.scaling_factor


def prepare_video_pixels(content: np.ndarray | list[np.ndarray]) -> torch.Tensor:
    data = np.stack(content) if isinstance(content, list) else np.asarray(content)
    if data.ndim == 3:
        data = data[None]
    if data.ndim != 4 or data.shape[-1] not in (3, 4):
        raise ValueError(f"Kandinsky6 content must be [F,H,W,C], got {data.shape}")
    data = data[..., :3]
    return torch.from_numpy(data).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)


def first_control_frame(content: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Return one external condition frame from directory/JSONL dataset layouts."""
    if isinstance(content, list):
        if not content:
            raise ValueError("Kandinsky6 image condition list is empty")
        content = content[0]
    frame = np.asarray(content)
    if frame.ndim == 4:
        if frame.shape[0] == 0:
            raise ValueError("Kandinsky6 image condition has no frames")
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Kandinsky6 image condition must be [H,W,C] or [F,H,W,C], got {frame.shape}")
    return frame


@torch.no_grad()
def encode_batch(video_vae, audio_vae, batch: list[ItemInfo]) -> None:
    from musubi_tuner.dataset.cache_io import save_latent_cache_kandinsky6

    for item in batch:
        if item.content is None:
            raise ValueError(f"Content not loaded for item {item.item_key}")
        latent = _encode_pixels(video_vae, prepare_video_pixels(item.content))  # [C,T,H,W]

        audio_latent = None
        audio_present = False
        if audio_vae is not None:
            if item.frame_count is None:
                # ImageDataset does not receive AudioSpec. Supply the one-frame silence
                # target locally so joint K6 batches retain a valid audio tensor which
                # audio_present=False masks out of the loss.
                waveform = torch.zeros(1, audio_samples_for_frames(1), dtype=torch.float32)
            else:
                if item.audio_content is None or item.audio_present is None:
                    raise ValueError(f"Audio was not loaded for audio-enabled video item {item.item_key}")
                waveform = item.audio_content
                audio_present = bool(item.audio_present)
            if waveform.ndim != 2 or waveform.shape[0] != 1:
                raise ValueError(f"Kandinsky6 waveform must be mono [1,L], got {tuple(waveform.shape)}")
            waveform = waveform.to(device=audio_vae.device, dtype=audio_vae.dtype)
            # Training targets use the deterministic posterior mean, matching upstream
            # FeaturesUtils.wrapped_encode. Keep [C_audio,A]; the trainer transposes to [A,C].
            audio_latent = audio_vae.encode_audio(waveform).mean[0] * audio_vae.scaling_factor
            expected_audio_frames = audio_latent_frames_for_video(int(item.frame_count or 1))
            if audio_latent.ndim != 2 or audio_latent.shape[-1] != expected_audio_frames:
                raise ValueError(
                    "Kandinsky6 audio VAE returned an unexpected latent grid: expected "
                    f"[C,{expected_audio_frames}], got {tuple(audio_latent.shape)}"
                )

        image_latent = None
        if item.control_content is not None:
            control = first_control_frame(item.control_content)
            control_pixels = prepare_video_pixels(control)
            image_latent = _encode_pixels(video_vae, control_pixels)
            if image_latent.shape[1] != 1:
                raise ValueError(f"Kandinsky6 image condition must encode to one latent frame, got {image_latent.shape[1]}")
            if image_latent.shape[2:] != latent.shape[2:]:
                raise ValueError(
                    "Kandinsky6 image condition must match the target latent canvas, got "
                    f"{tuple(image_latent.shape[2:])} versus {tuple(latent.shape[2:])}"
                )
        save_latent_cache_kandinsky6(
            item,
            video_latent=latent.cpu(),
            audio_latent=audio_latent.cpu(),
            audio_present=audio_present,
            image_latent=None if image_latent is None else image_latent.cpu(),
            metadata={
                "video_vae_scaling_factor": str(float(video_vae.config.scaling_factor)),
                "audio_vae_scaling_factor": str(float(audio_vae.scaling_factor)),
                "audio_sample_rate": str(K6_SAMPLE_RATE),
                "video_fps": str(K6_VIDEO_FPS),
            },
        )


def main() -> None:
    parser = cache_latents.setup_parser_common()
    vae_action = next(action for action in parser._actions if action.dest == "vae")
    vae_action.required = True
    vae_action.help = "Kandinsky6 HunyuanVideo VAE directory"
    parser.add_argument("--audio_vae", required=True, help="Kandinsky6 MMAudio VAE checkpoint")
    parser.add_argument("--audio_vae_scaling_factor", type=float, default=0.5302)
    parser.add_argument("--upstream", help="deprecated compatibility option; ignored because the runtime is bundled")
    parser.add_argument("--sr_upstream", help="deprecated compatibility option; ignored because the runtime is bundled")
    add_audio_tolerance_arguments(parser)
    args = parser.parse_args()

    try:
        from musubi_tuner.kandinsky6.runtime.core.components.vae_audio import build_audio_vae
        from musubi_tuner.kandinsky6.runtime.core.components.vae_video import build_vae
    except ImportError as exc:
        raise RuntimeError(
            "Kandinsky6 caching requires the optional Kandinsky6 dependencies; install `musubi-tuner[kandinsky6]`."
        ) from exc
    from musubi_tuner.dataset.architectures import ARCHITECTURE_KANDINSKY6

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        config_utils.load_user_config(args.dataset_config), args, architecture=ARCHITECTURE_KANDINSKY6
    )
    audio_spec = apply_audio_tolerance_arguments(K6_AUDIO_SPEC, args)
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, audio_spec=audio_spec).datasets
    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    video_vae = build_vae(args.vae, device=device)
    audio_vae = build_audio_vae(
        tod_vae_ckpt=args.audio_vae,
        device=device,
        need_vae_decoder=False,
        scaling_factor=args.audio_vae_scaling_factor,
    )
    cache_latents.encode_datasets(datasets, lambda batch: encode_batch(video_vae, audio_vae, batch), args)


if __name__ == "__main__":
    main()
