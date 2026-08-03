from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from multiprocessing import Value
from pathlib import Path

import torch
from accelerate import Accelerator

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.hv_train import get_sigmas
from musubi_tuner.hv_train_network import NetworkTrainer, read_config_from_file, setup_parser_common
from musubi_tuner.minimax_h3.architecture import VIDEO_FLOW_SHIFT
from musubi_tuner.minimax_h3.backend import H3TrainingBackend, create_training_backend
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.dataset import create_h3_dataset_group
from musubi_tuner.minimax_h3.training import (
    H3ModelPrediction,
    guidance_consistent_prediction,
    joint_velocity_loss,
    prepare_joint_noisy_inputs,
)
from musubi_tuner.training.accelerator_setup import collator_class
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)

_DIRECT_SIGMA_SAMPLING = {
    "uniform",
    "sigmoid",
    "shift",
    "flux_shift",
    "qwen_shift",
    "krea2_shift",
    "ideogram4_shift",
    "logsnr",
    "qinglong_flux",
    "qinglong_qwen",
    "flux2_shift",
}


class MiniMaxH3NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.backend: H3TrainingBackend | None = None

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MINIMAX_H3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MINIMAX_H3_FULL

    def _build_dataset(self, args):
        if args.num_timestep_buckets is not None:
            logger.info("Using timestep bucketing. Number of buckets: %s", args.num_timestep_buckets)
        self.num_timestep_buckets = args.num_timestep_buckets
        current_epoch = Value("i", 0)

        logger.info("Load dataset config from %s", args.dataset_config)
        user_config = config_utils.load_user_config(args.dataset_config)
        train_dataset_group, _ = create_h3_dataset_group(
            user_config,
            args,
            training=True,
            num_timestep_buckets=self.num_timestep_buckets,
            shared_epoch=current_epoch,
        )
        if train_dataset_group.num_train_items == 0:
            raise ValueError(
                "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
                " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
            )

        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, ds_for_collator)
        return train_dataset_group, collator, current_epoch

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = (
            torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = VIDEO_FLOW_SHIFT
        self.vae_frame_stride = 17

        if args.discrete_flow_shift <= 0:
            raise ValueError("MiniMax H3 --discrete_flow_shift must be positive")
        if not math.isclose(args.discrete_flow_shift, VIDEO_FLOW_SHIFT):
            logger.warning(
                "MiniMax H3 uses video flow shift %.1f; training requested %.4g",
                VIDEO_FLOW_SHIFT,
                args.discrete_flow_shift,
            )
        if args.h3_guidance_distillation_scale is not None and args.h3_guidance_distillation_scale <= 1.0:
            raise ValueError("--h3_guidance_distillation_scale must be greater than 1, or omitted for one-pass training")
        if args.fp8_base:
            # H3 supports only weight-only scaled FP8. Reuse the common
            # --fp8_base switch without exposing an H3-only parser field, and
            # prevent the base trainer from casting the mixed-precision shell
            # and norms directly to float8.
            args.fp8_scaled = True
        if args.blocks_to_swap is not None and args.blocks_to_swap < 0:
            raise ValueError("MiniMax H3 --blocks_to_swap must be non-negative")
        if args.block_swap_h2d_only and not args.use_pinned_memory_for_block_swap:
            logger.warning(
                "MiniMax H3 H2D-only block swap without pinned host memory uses staged copies and can be substantially slower; "
                "add --use_pinned_memory_for_block_swap for direct asynchronous transfers"
            )
        if args.compile:
            raise ValueError("MiniMax H3 training does not support compilation")
        if not args.sdpa:
            raise ValueError("MiniMax H3 training supports only --sdpa")
        if args.split_attn:
            raise ValueError("MiniMax H3 training does not support split attention")

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        del args, accelerator, sample_prompts
        raise ValueError("MiniMax H3 sampling during training requires a backend with native audio-video decoding")

    def do_inference(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("MiniMax H3 training inference must be provided by the selected backend")

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        del args, vae_dtype, vae_path
        raise RuntimeError("MiniMax H3 uses separate video and audio VAEs through its backend")

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: torch.dtype | None,
    ):
        if args.fp8_base and dit_weight_dtype is not None:
            raise ValueError("MiniMax H3 scaled FP8 loading requires dit_weight_dtype=None")
        self.backend = create_training_backend(
            model=Path(dit_path),
            device=str(loading_device),
            dtype=model_utils.dtype_to_str(self.dit_dtype),
            mode=args.h3_training_mode,
            attention_mode=attn_mode,
            split_attention=split_attn,
            fp8_scaled=bool(args.fp8_base),
            quantization_device=str(accelerator.device),
        )
        transformer = self.backend.get_training_transformer()
        if not isinstance(transformer, torch.nn.Module):
            raise TypeError("H3 backend get_training_transformer() must return a torch.nn.Module")
        return transformer

    def compile_transformer(self, args, transformer):
        del args, transformer
        raise RuntimeError("MiniMax H3 compilation is unavailable")

    def scale_shift_latents(self, latents):
        # H3 latent caches are written in the model's normalized latent space.
        return latents

    def _video_sigma(
        self,
        args: argparse.Namespace,
        noise_scheduler,
        timesteps: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if args.timestep_sampling in _DIRECT_SIGMA_SAMPLING:
            return ((timesteps.to(device=device, dtype=torch.float32) - 1.0) / 1000.0).clamp(0.0, 1.0)
        return get_sigmas(noise_scheduler, timesteps, device, n_dim=1, dtype=dtype).to(torch.float32)

    def _sample_weight(self, args: argparse.Namespace, sigma: torch.Tensor) -> torch.Tensor | None:
        if args.weighting_scheme == "sigma_sqrt":
            return sigma.clamp_min(1e-6).pow(-2.0)
        if args.weighting_scheme == "cosmap":
            return 2.0 / (math.pi * (1.0 - 2.0 * sigma + 2.0 * sigma.square()))
        return None

    def _predict(
        self,
        accelerator: Accelerator,
        transformer,
        batch,
        inputs,
        *,
        conditioning: str,
        gradient_checkpointing: bool,
    ) -> H3ModelPrediction:
        if self.backend is None:
            raise RuntimeError("H3 training backend is not loaded")
        video = inputs.video.to(device=accelerator.device, dtype=self.dit_dtype)
        audio = inputs.audio.to(device=accelerator.device, dtype=self.dit_dtype)
        if gradient_checkpointing:
            video.requires_grad_(True)
            audio.requires_grad_(True)
        with accelerator.autocast():
            prediction = self.backend.predict_training(
                transformer,
                batch,
                video,
                audio,
                inputs.video_timestep.to(accelerator.device),
                inputs.audio_timestep.to(accelerator.device),
                conditioning=conditioning,
            )
        if not isinstance(prediction, H3ModelPrediction):
            raise TypeError("H3 backend predict_training() must return H3ModelPrediction")
        return prediction

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        vae,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del network, network_dtype, vae, global_step
        if latents.shape[0] != 1:
            raise ValueError("MiniMax H3 training requires dataset batch_size = 1")
        if H3_AUDIO_LATENTS_KEY not in batch:
            raise KeyError(f"MiniMax H3 cache is missing {H3_AUDIO_LATENTS_KEY}")

        latents = latents.to(device=accelerator.device, dtype=dit_dtype)
        noise = noise.to(device=accelerator.device, dtype=dit_dtype)
        audio_latents = batch[H3_AUDIO_LATENTS_KEY].to(device=accelerator.device, dtype=dit_dtype)
        audio_noise = torch.randn_like(audio_latents)

        _, scheduler_timesteps = super().get_noisy_model_input_and_timesteps(
            args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
        )
        video_sigma = self._video_sigma(args, noise_scheduler, scheduler_timesteps, accelerator.device, dit_dtype)
        inputs = prepare_joint_noisy_inputs(latents, audio_latents, noise, audio_noise, video_sigma)

        if args.h3_guidance_distillation_scale is not None:
            missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
            if missing_empty:
                raise KeyError(
                    "guidance-consistent H3 training requires --cache_guidance_empty; missing " + ", ".join(missing_empty)
                )
            # The empty branch calibrates the distilled field but is not itself
            # optimized. Evaluate it first without retaining a 33B-model graph.
            with torch.no_grad():
                empty_prediction = self._predict(
                    accelerator,
                    transformer,
                    batch,
                    inputs,
                    conditioning="empty",
                    gradient_checkpointing=False,
                )
        prediction = self._predict(
            accelerator,
            transformer,
            batch,
            inputs,
            conditioning="prompt",
            gradient_checkpointing=args.gradient_checkpointing,
        )
        if args.h3_guidance_distillation_scale is not None:
            prediction = guidance_consistent_prediction(prediction, empty_prediction, args.h3_guidance_distillation_scale)

        result = joint_velocity_loss(
            prediction,
            inputs,
            video_mask=batch.get("video_loss_mask"),
            audio_mask=batch.get("audio_loss_mask"),
            sample_weight=self._sample_weight(args, video_sigma),
            balance=args.h3_loss_balance,
            video_weight=args.h3_video_loss_weight,
            audio_weight=args.h3_audio_loss_weight,
        )
        metrics = {
            "loss/video": float(result.video_loss.detach()),
            "loss/audio": float(result.audio_loss.detach()),
            "h3/sigma_video": float(inputs.video_sigma.mean().detach()),
            "h3/sigma_audio": float(inputs.audio_sigma.mean().detach()),
        }
        return result.loss, metrics

    def call_dit(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("MiniMax H3 uses its joint audio-video process_batch implementation")

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        return {
            "ss_h3_training_mode": args.h3_training_mode,
            "ss_h3_loss_balance": args.h3_loss_balance,
            "ss_h3_video_loss_weight": str(args.h3_video_loss_weight),
            "ss_h3_audio_loss_weight": str(args.h3_audio_loss_weight),
            "ss_h3_guidance_distillation_scale": str(args.h3_guidance_distillation_scale or "one_pass"),
        }


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Train a MiniMax H3 LoRA with synchronized video and audio flow matching"
    parser.add_argument(
        "--h3_training_mode",
        choices=("fl2va", "ref2va"),
        default="fl2va",
        help="select the first/last-frame base transformer or the separate reference-conditioned transformer",
    )
    parser.add_argument(
        "--h3_loss_balance",
        choices=("token", "modality"),
        default="modality",
        help="combine joint AV loss over all valid latent elements or equally by modality means",
    )
    parser.add_argument("--h3_video_loss_weight", type=float, default=1.0)
    parser.add_argument("--h3_audio_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--h3_guidance_distillation_scale",
        type=float,
        default=None,
        help="experimental: enable two-pass guidance-consistent training with a user-supplied distillation scale",
    )
    parser.set_defaults(
        network_module="networks.lora_minimax_h3",
        mixed_precision="bf16",
        timestep_sampling="shift",
        discrete_flow_shift=VIDEO_FLOW_SHIFT,
        vae_dtype="float32",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(setup_parser_common())


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    args = read_config_from_file(args, parser)
    args.dit_dtype = None
    trainer = MiniMaxH3NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
