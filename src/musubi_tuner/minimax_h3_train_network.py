from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from multiprocessing import Value
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner import convert_lora
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.hv_train import get_sigmas
from musubi_tuner.hv_train_network import NetworkTrainer, read_config_from_file, setup_parser_common
from musubi_tuner.minimax_h3.architecture import (
    AUDIO_FLOW_SHIFT,
    VIDEO_DIT_PATCH_SIZE,
    VIDEO_FLOW_SHIFT,
    align_frame_count,
)
from musubi_tuner.minimax_h3.assets import default_text_encoder_assets
from musubi_tuner.minimax_h3.backend import H3TrainingBackend, create_conditioning_encoder, create_training_backend
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.component_loader import load_audio_vae_decoder, load_video_vae_decoder
from musubi_tuner.minimax_h3.crepa import H3CREPA, H3CREPAConfig, parse_crepa_config
from musubi_tuner.minimax_h3.dataset import create_h3_dataset_group
from musubi_tuner.minimax_h3.inference import (
    decode_latents_sequentially,
    denoise_fl2va,
    encode_keyframe_images,
    prepare_keyframe_image,
    save_av_mp4,
)
from musubi_tuner.minimax_h3.masking import (
    audio_mask_to_rows,
    rows_to_latent_video_mask,
    sample_audio_mask,
    sample_video_mask,
    video_mask_to_rows,
)
from musubi_tuner.minimax_h3.packing import AUDIO_CHANNELS
from musubi_tuner.minimax_h3.references import REFERENCE_IMAGE_SHORT_EDGE
from musubi_tuner.minimax_h3.training import (
    H3ModelPrediction,
    contrastive_guidance_target,
    guidance_consistent_prediction,
    guidance_scale_for_sigma,
    joint_prediction_loss,
    joint_velocity_loss,
    prepare_joint_noisy_inputs,
    shift_sigma,
)
from musubi_tuner.minimax_h3.validation import (
    H3ValidationAccumulator,
    image_validation_sigma,
    masked_squared_error_sum,
    preserve_rng_state,
    seed_validation_forward,
    validation_sigma_bins,
)
from musubi_tuner.training.accelerator_setup import collator_class
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import LOSS_FOR_AVERAGE_KEY
from musubi_tuner.training.validation import derive_validation_seed
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)

_SAMPLE_KEYFRAME_ROWS = "_h3_keyframe_rows"
_SAMPLE_KEYFRAME_ANCHORS = "_h3_keyframe_anchors"

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


class _H3DecoderBundle(torch.nn.Module):
    def __init__(self, video_decoder: torch.nn.Module, audio_decoder: torch.nn.Module) -> None:
        super().__init__()
        self.video_decoder = video_decoder
        self.audio_decoder = audio_decoder


class _IndexedValidationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        dataset_index = self.indices[index]
        return dataset_index, self.dataset[dataset_index]


def _parse_keyframe_anchors(spec: str) -> tuple[int | str, ...]:
    """Parse a keyframe anchor spec into 'first'/'last' markers and frame indices."""
    if not spec:
        return ()
    anchors: list[int | str] = []
    for piece in spec.split(","):
        token = piece.strip()
        if token in ("first", "last"):
            anchors.append(token)
        elif token.lstrip("-").isdigit():
            anchors.append(int(token))
        else:
            raise ValueError(f"H3 keyframe anchor {token!r} must be 'first', 'last', or a latent frame index")
    return tuple(anchors)


class MiniMaxH3NetworkTrainer(NetworkTrainer):
    @staticmethod
    def _base_preservation_active(accelerator: Accelerator, probability: float) -> bool:
        """Draw one preservation decision shared by every distributed rank."""
        if probability >= 1.0:
            return True
        device = accelerator.device
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        # Draw on CPU before any replayed model branch. Every rank advances its
        # own seeded CPU stream once; rank zero's decision is then authoritative.
        # This keeps the CUDA replay untouched without restoring the Bernoulli
        # generator to the same position after every call.
        active = (torch.rand((), device="cpu") < probability).to(device=device)
        if distributed:
            torch.distributed.broadcast(active, src=0)
        return bool(active.item())

    supports_validation = True

    def __init__(self):
        super().__init__()
        self.backend: H3TrainingBackend | None = None
        self._crepa_config: H3CREPAConfig | None = None
        self._crepa: H3CREPA | None = None
        self._extension_video_frames = 0
        self._extension_audio_latents = 0
        self._extension_route = "condition_rows"
        self._frame_sigma_jitter = 0.0
        self._step_row_video_timestep = None
        self._spatial_density_jitter = 0.0
        self._step_spatial_density_scale = None
        self._keyframe_anchors: tuple[int | str, ...] = ()
        self._keyframe_random_count = 0
        self._mask_mode = "off"
        self._mask_audio = False
        self._mask_bounds = (0.25, 0.75)
        self._step_keyframes = None
        self._step_mask = None
        self._validation_dataloader = None

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MINIMAX_H3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MINIMAX_H3_FULL

    def convert_weight_keys(self, weights_sd: dict[str, torch.Tensor], network_module):
        del network_module
        if not weights_sd:
            return weights_sd
        first_key = next(iter(weights_sd))
        if first_key.startswith("lora_"):
            return weights_sd
        if first_key.startswith(("diffusion_model.", "transformer.")):
            logger.info("Converting MiniMax H3 base LoRA weights from Diffusers format")
            return convert_lora.convert_from_diffusers("lora_unet_", weights_sd)
        return weights_sd

    def load_network_weights(self, path: str, network_module_name: str) -> dict[str, torch.Tensor]:
        return self.convert_weight_keys(load_file(path), network_module_name)

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

    def _build_validation_dataloader(self, args, accelerator):
        validation_seed = args.validation_seed if args.validation_seed is not None else args.seed
        with preserve_rng_state():
            seed_validation_forward(validation_seed)
            validation_args = copy.copy(args)
            validation_args.h3_load_dino_features = False
            current_epoch = Value("i", 0)
            user_config = config_utils.load_user_config(args.validation_dataset_config)
            dataset_group, _ = create_h3_dataset_group(
                user_config,
                validation_args,
                training=True,
                num_timestep_buckets=None,
                shared_epoch=current_epoch,
            )
        if dataset_group.num_train_items == 0 or len(dataset_group) == 0:
            raise ValueError("MiniMax H3 validation dataset contains no cached items")
        item_count = len(dataset_group)
        if args.max_validation_items is not None:
            item_count = min(item_count, args.max_validation_items)
        indices = range(accelerator.process_index, item_count, accelerator.num_processes)
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(validation_seed)
        return torch.utils.data.DataLoader(
            _IndexedValidationDataset(dataset_group, indices),
            batch_size=None,
            num_workers=0,
            generator=loader_generator,
        )

    @torch.no_grad()
    def validate(
        self,
        accelerator,
        args,
        transformer,
        network,
        global_step,
        epoch,
    ) -> None:
        del network, epoch
        if self.backend is None:
            raise RuntimeError("H3 training backend is not loaded")
        # Per-step conditioning belongs to the training step that drew it. Left
        # in place it would make validation forward under the last batch's mask
        # or jitter schedule while scoring the full target, so the numbers would
        # not describe any coherent objective.
        self._step_mask = None
        self._step_row_video_timestep = None
        self._step_spatial_density_scale = None
        self._step_keyframes = None
        if self._validation_dataloader is None:
            self._validation_dataloader = self._build_validation_dataloader(args, accelerator)

        bins = validation_sigma_bins(
            args.validation_timestep_bins,
            minimum=args.validation_min_timestep / 1000.0,
            maximum=args.validation_max_timestep / 1000.0,
            video_shift=args.h3_shift_video,
            audio_shift=args.h3_shift_audio,
        )
        # Validation measures a fixed task, so a randomised observed modality is
        # reported as the joint objective rather than a different draw each run.
        observed = None if args.h3_observed_modality == "random" else args.h3_observed_modality
        video_weight = 0.0 if observed == "video" else args.h3_video_loss_weight
        audio_weight = 0.0 if observed == "audio" else args.h3_audio_loss_weight
        accumulator = H3ValidationAccumulator(
            len(bins),
            balance=args.h3_loss_balance,
            video_weight=video_weight,
            audio_weight=audio_weight,
        )
        validation_seed = args.validation_seed if args.validation_seed is not None else args.seed

        with preserve_rng_state():
            for dataset_index, batch in self._validation_dataloader:
                latents = self.get_primary_latents(batch)
                if latents.shape[0] != 1:
                    raise ValueError("MiniMax H3 validation requires dataset batch_size = 1")
                has_video = "latents" in batch or latents.ndim == 5
                has_audio = H3_AUDIO_LATENTS_KEY in batch
                video_source = batch.get("latents", latents if latents.ndim == 5 else None)
                video_latents = video_source.to(accelerator.device, dtype=self.dit_dtype) if has_video else None
                audio_latents = batch[H3_AUDIO_LATENTS_KEY].to(accelerator.device, dtype=self.dit_dtype) if has_audio else None
                is_image = has_video and not has_audio and video_latents.shape[2] == 1
                if observed is not None and not (has_video and has_audio):
                    raise ValueError("H3 observed-modality validation requires cached video and audio targets")

                for sigma_bin in bins:
                    if video_latents is not None:
                        video_noise_seed = derive_validation_seed(
                            validation_seed,
                            dataset_index=dataset_index,
                            bin_index=sigma_bin.index,
                            stream="video-noise",
                        )
                        seed_validation_forward(video_noise_seed)
                        video_noise = torch.randn_like(video_latents)
                    else:
                        video_noise = None
                    if audio_latents is not None:
                        audio_noise_seed = derive_validation_seed(
                            validation_seed,
                            dataset_index=dataset_index,
                            bin_index=sigma_bin.index,
                            stream="audio-noise",
                        )
                        seed_validation_forward(audio_noise_seed)
                        audio_noise = torch.randn_like(audio_latents)
                    else:
                        audio_noise = None

                    base_sigma = torch.tensor([sigma_bin.base_sigma], device=accelerator.device, dtype=torch.float32)
                    if is_image:
                        base_sigma = image_validation_sigma(
                            base_sigma,
                            latent_height=video_latents.shape[-2],
                            latent_width=video_latents.shape[-1],
                            flow_shift=args.h3_image_flow_shift,
                        )
                    inputs = prepare_joint_noisy_inputs(
                        video_latents,
                        audio_latents,
                        video_noise,
                        audio_noise,
                        base_sigma,
                        video_shift=1.0 if is_image else args.h3_shift_video,
                        audio_shift=1.0 if is_image else args.h3_shift_audio,
                        observed=observed,
                    )

                    forward_seed = derive_validation_seed(
                        validation_seed,
                        dataset_index=dataset_index,
                        bin_index=sigma_bin.index,
                        stream="model-forward",
                    )
                    seed_validation_forward(forward_seed)
                    if args.h3_guidance_distillation_scale is not None:
                        missing_empty = [
                            key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch
                        ]
                        if missing_empty:
                            raise KeyError("guidance-consistent H3 validation is missing " + ", ".join(missing_empty))
                        fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
                        with torch.random.fork_rng(devices=fork_devices):
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
                        gradient_checkpointing=False,
                    )
                    if args.h3_guidance_distillation_scale is not None:
                        prediction, loss_inputs = self._guidance_loss_inputs(args, prediction, empty_prediction, inputs)
                    else:
                        loss_inputs = inputs

                    video_sample_weight = self._sample_weight(args, inputs.video_sigma) if has_video else None
                    audio_sample_weight = self._sample_weight(args, inputs.audio_sigma) if has_audio else None
                    if prediction.video is not None and loss_inputs.video_target is not None and video_weight > 0:
                        total, count = masked_squared_error_sum(
                            prediction.video,
                            loss_inputs.video_target,
                            batch.get("video_loss_mask"),
                            sample_weight=video_sample_weight,
                        )
                        accumulator.add(sigma_bin.index, "video", total, count)
                    if prediction.audio is not None and loss_inputs.audio_target is not None and audio_weight > 0:
                        total, count = masked_squared_error_sum(
                            prediction.audio,
                            loss_inputs.audio_target,
                            batch.get("audio_loss_mask"),
                            sample_weight=audio_sample_weight,
                        )
                        accumulator.add(sigma_bin.index, "audio", total, count)

        reduced = accelerator.reduce(accumulator.reduction_tensor(device=accelerator.device), reduction="sum")
        accumulator.load_reduced_tensor(reduced)
        metrics = {f"val/{key}": value for key, value in accumulator.metrics().items()}
        if metrics and len(accelerator.trackers) > 0:
            accelerator.log(metrics, step=global_step)
        accelerator.print("MiniMax H3 validation: " + ", ".join(f"{key}={value:.6g}" for key, value in metrics.items()))

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = (
            torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        if args.h3_lora_token_refiner:
            if not args.network_module.endswith("lora_minimax_h3"):
                raise ValueError("--h3_lora_token_refiner requires --network_module networks.lora_minimax_h3")
            network_args = list(args.network_args or [])
            if any(value.startswith("h3_lora_token_refiner=") for value in network_args):
                raise ValueError(
                    "set H3 token-refiner targeting with --h3_lora_token_refiner, not a duplicate --network_args value"
                )
            network_args.append("h3_lora_token_refiner=true")
            args.network_args = network_args
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = 1.0
        self.vae_frame_stride = 17
        self._crepa_config = parse_crepa_config(args.crepa)
        args.h3_load_dino_features = self._crepa_config is not None and self._crepa_config.mode == "dino"
        args.h3_dino_model = self._crepa_config.dino_model if args.h3_load_dino_features else None
        if args.validation_dataset_config:
            validation_config = config_utils.load_user_config(args.validation_dataset_config)
            general_batch_size = int(validation_config.get("general", {}).get("batch_size", 1))
            validation_batch_sizes = [
                int(dataset.get("batch_size", general_batch_size)) for dataset in validation_config.get("datasets", [])
            ] or [general_batch_size]
            if any(batch_size != 1 for batch_size in validation_batch_sizes):
                raise ValueError("MiniMax H3 validation requires batch_size = 1 in --validation_dataset_config")

        # H3 owns its own flow shifts because video and audio ride different
        # schedules (12 and 3) off one shared unshifted coordinate. The common
        # sampler must therefore hand us that coordinate *unshifted*: applying
        # --discrete_flow_shift as well would shift video twice and leave audio
        # on a schedule the model was never trained for.
        if not math.isclose(args.discrete_flow_shift, 1.0):
            raise ValueError(
                "MiniMax H3 requires --discrete_flow_shift 1.0; set the per-modality shifts with "
                "--h3_shift_video / --h3_shift_audio instead (defaults 12.0 / 3.0)"
            )
        for name in ("h3_shift_video", "h3_shift_audio"):
            value = float(getattr(args, name))
            if not 0.01 <= value <= 100.0:
                raise ValueError(f"--{name} must be in [0.01, 100.0], got {value}")
        if args.h3_image_flow_shift is not None and args.h3_image_flow_shift <= 0:
            raise ValueError("MiniMax H3 --h3_image_flow_shift must be positive when specified")
        if args.h3_guidance_distillation_scale is not None and args.h3_guidance_distillation_scale <= 1.0:
            raise ValueError("--h3_guidance_distillation_scale must be greater than 1, or omitted for one-pass training")
        if args.h3_guidance_loss_form == "contrastive" and args.h3_guidance_distillation_scale is None:
            raise ValueError("--h3_guidance_loss_form contrastive requires --h3_guidance_distillation_scale")
        if not math.isfinite(args.h3_base_preservation_loss_weight) or args.h3_base_preservation_loss_weight < 0:
            raise ValueError("--h3_base_preservation_loss_weight must be finite and non-negative")
        if not math.isfinite(args.h3_base_preservation_probability) or not 0 < args.h3_base_preservation_probability <= 1:
            raise ValueError("--h3_base_preservation_probability must be finite and lie in (0, 1]")
        if args.h3_convrot_int8 and (args.fp8_base or args.int8_convrot_base):
            raise ValueError("--h3_convrot_int8 quantizes the BF16 checkpoint itself; drop --fp8_base/--int8_convrot_base")
        convrot_int8_active = args.h3_convrot_int8 or args.int8_convrot_base
        if args.h3_convrot_int8_bwd == "int8" and not convrot_int8_active:
            raise ValueError("--h3_convrot_int8_bwd int8 requires --h3_convrot_int8 or --int8_convrot_base")
        if args.h3_convrot_int8_fwd == "bf16" and not convrot_int8_active:
            raise ValueError("--h3_convrot_int8_fwd bf16 requires --h3_convrot_int8 or --int8_convrot_base")
        if args.h3_convrot_int8_fwd == "bf16" and args.h3_convrot_int8_bwd == "int8":
            raise ValueError("--h3_convrot_int8_fwd bf16 leaves no rotated activations for --h3_convrot_int8_bwd int8")
        if args.h3_convrot_int8_lora_fused and not (
            convrot_int8_active and args.h3_convrot_int8_fwd == "int8" and args.h3_convrot_int8_bwd == "int8"
        ):
            raise ValueError(
                "--h3_convrot_int8_lora_fused requires online or pre-quantized ConvRot INT8 weights with "
                "--h3_convrot_int8_fwd int8 and --h3_convrot_int8_bwd int8"
            )
        if not 0.0 <= args.h3_caption_dropout_rate <= 1.0:
            raise ValueError("--h3_caption_dropout_rate must lie in [0, 1]")
        if args.h3_extension_video_frames < 0 or args.h3_extension_audio_latents < 0:
            raise ValueError("H3 extension context lengths must be non-negative")
        self._extension_video_frames = args.h3_extension_video_frames
        self._extension_audio_latents = args.h3_extension_audio_latents
        self._extension_route = args.h3_extension_route
        self._frame_sigma_jitter = args.h3_frame_sigma_jitter
        if not 0.0 <= args.h3_frame_sigma_jitter <= 1.0:
            raise ValueError("--h3_frame_sigma_jitter must lie in [0, 1]")
        self._spatial_density_jitter = args.h3_spatial_density_jitter
        if not math.isfinite(args.h3_spatial_density_jitter) or args.h3_spatial_density_jitter < 0:
            raise ValueError("--h3_spatial_density_jitter must be finite and non-negative")
        self._keyframe_anchors = _parse_keyframe_anchors(args.h3_keyframe_anchors)
        self._keyframe_random_count = args.h3_keyframe_random_count
        if args.h3_keyframe_random_count < 0:
            raise ValueError("--h3_keyframe_random_count cannot be negative")
        if self._keyframe_anchors and args.h3_keyframe_random_count:
            raise ValueError("H3 keyframe anchors are either listed or drawn at random, not both")
        keyframes = bool(self._keyframe_anchors) or bool(args.h3_keyframe_random_count)
        if keyframes and (args.h3_extension_video_frames or args.h3_extension_audio_latents):
            raise ValueError("H3 keyframe conditioning and extension both claim the observed rows; enable only one")
        if keyframes and (args.h3_mask_mode != "off" or args.h3_mask_audio):
            raise ValueError("H3 keyframe conditioning and masked conditioning both claim the observed rows; enable only one")
        if keyframes and args.h3_training_mode != "fl2va":
            raise ValueError("H3 keyframe conditioning requires --h3_training_mode fl2va with --task t2va caches")
        self._mask_mode = args.h3_mask_mode
        self._mask_audio = args.h3_mask_audio
        self._mask_bounds = (args.h3_mask_min_fraction, args.h3_mask_max_fraction)
        if not 0.0 < args.h3_mask_min_fraction <= args.h3_mask_max_fraction <= 1.0:
            raise ValueError("H3 mask fractions must satisfy 0 < min <= max <= 1")
        masking = args.h3_mask_mode != "off" or args.h3_mask_audio
        if masking and (args.h3_extension_video_frames or args.h3_extension_audio_latents):
            raise ValueError("H3 masked conditioning and extension both claim the observed rows; enable only one")
        if masking and args.h3_training_mode != "fl2va":
            raise ValueError("H3 masked conditioning requires --h3_training_mode fl2va with --task t2va caches")
        if (args.h3_extension_video_frames or args.h3_extension_audio_latents) and args.h3_training_mode != "fl2va":
            raise ValueError("H3 extension training requires --h3_training_mode fl2va with --task t2va caches")
        # Jitter re-noises the whole video at per-frame levels, which silently
        # overwrites any row a conditioning mode pinned as observed and leaves
        # the row timesteps disagreeing with the noise actually applied.
        conditioning = (
            keyframes
            or masking
            or bool(args.h3_extension_video_frames or args.h3_extension_audio_latents)
            or args.h3_observed_modality is not None
        )
        if args.h3_frame_sigma_jitter > 0 and conditioning:
            raise ValueError(
                "--h3_frame_sigma_jitter re-noises every frame, so it cannot be combined with a conditioning mode "
                "that presents part of the target as observed (--h3_observed_modality, extension, keyframes, or masking)"
            )
        if args.h3_frame_sigma_jitter > 0 and args.weighting_scheme in {"sigma_sqrt", "cosmap"}:
            raise ValueError(
                f"--h3_frame_sigma_jitter cannot be combined with --weighting_scheme {args.weighting_scheme}: "
                "per-frame weighting is not supported"
            )
        if args.h3_guidance_distillation_scale is not None and float(getattr(args, "network_dropout", 0.0) or 0.0) > 0:
            raise ValueError(
                "H3 guidance-consistent training cannot replay --network_dropout across different prompt lengths; "
                "use rank_dropout or module_dropout instead"
            )
        if args.fp8_base and args.h3_adaln_rank is None:
            # AdaLN is ~39% of the transformer and is quantized by default, yet
            # measured against the BF16 reference the reduction is both smaller
            # and more faithful than quantizing it.
            logger.info(
                "MiniMax H3: --fp8_base quantizes the AdaLN projections. Reducing them instead with "
                "--h3_adaln_rank 16 is both smaller and closer to the BF16 reference; consider adding it."
            )
        if args.fp8_base:
            # H3 supports only weight-only scaled FP8. Reuse the common
            # --fp8_base switch without exposing an H3-only parser field, and
            # prevent the base trainer from casting the mixed-precision shell
            # and norms directly to float8.
            args.fp8_scaled = True
        if args.int8_convrot_base and args.fp8_base:
            raise ValueError("MiniMax H3 --int8_convrot_base cannot be combined with --fp8_base")
        if args.blocks_to_swap is not None and args.blocks_to_swap < 0:
            raise ValueError("MiniMax H3 --blocks_to_swap must be non-negative")
        if args.h3_gradient_checkpointing_blocks is not None:
            checkpoint_blocks = args.h3_gradient_checkpointing_blocks
            if not 0 <= checkpoint_blocks <= 50:
                raise ValueError("--h3_gradient_checkpointing_blocks must be in [0, 50]")
            if not args.gradient_checkpointing:
                raise ValueError("--h3_gradient_checkpointing_blocks requires --gradient_checkpointing")
            if checkpoint_blocks < 50 and (args.blocks_to_swap or 0) > 0:
                raise ValueError("partial H3 gradient checkpointing cannot be combined with block swap")
            if checkpoint_blocks < 50 and args.compile:
                raise ValueError("partial H3 gradient checkpointing cannot be combined with --compile")
        if args.h3_gradient_checkpointing_cpu_offload_pin_memory and not (
            args.gradient_checkpointing and args.gradient_checkpointing_cpu_offload
        ):
            raise ValueError(
                "--h3_gradient_checkpointing_cpu_offload_pin_memory requires "
                "--gradient_checkpointing and --gradient_checkpointing_cpu_offload"
            )
        if args.h3_reusable_activation_offload and not (args.gradient_checkpointing and args.gradient_checkpointing_cpu_offload):
            raise ValueError(
                "--h3_reusable_activation_offload requires --gradient_checkpointing and --gradient_checkpointing_cpu_offload"
            )
        if args.block_swap_h2d_only and not args.use_pinned_memory_for_block_swap:
            logger.warning(
                "MiniMax H3 H2D-only block swap without pinned host memory uses staged copies and can be substantially slower; "
                "add --use_pinned_memory_for_block_swap for direct asynchronous transfers"
            )
        if not (args.sdpa or args.flash_attn or args.flash3):
            raise ValueError("MiniMax H3 training requires --sdpa, --flash_attn, or --flash3")
        if args.h3_attn_auto_dispatch and not args.sdpa:
            raise ValueError("--h3_attn_auto_dispatch requires --sdpa")
        if getattr(args, "h3_int8_attention", "off") != "off" and args.compile:
            raise ValueError("--h3_int8_attention cannot currently be combined with --compile")
        if args.split_attn:
            raise ValueError("MiniMax H3 training does not support split attention")
        if args.sample_prompts:
            if args.h3_training_mode != "fl2va":
                raise ValueError(
                    "MiniMax H3 training-time sampling currently supports only FL2VA; "
                    "use minimax_h3_generate_video.py for Ref2VA samples"
                )
            required = {
                "--text_encoder": args.text_encoder,
                "--vae": args.vae,
                "--audio_vae": args.audio_vae,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError("MiniMax H3 sampling during training requires " + ", ".join(missing))

    def on_transformer_loaded(self, args, accelerator, transformer) -> None:
        transformer.set_gradient_checkpointing_blocks(args.h3_gradient_checkpointing_blocks)
        transformer.set_activation_cpu_offload_pin_memory(args.h3_gradient_checkpointing_cpu_offload_pin_memory)
        set_int8_attention_mode = getattr(transformer, "set_int8_attention_mode", None)
        if callable(set_int8_attention_mode):
            set_int8_attention_mode(getattr(args, "h3_int8_attention", "off"))
        if args.h3_reusable_activation_offload:
            transformer.enable_reusable_activation_offload()
        if args.h3_fused_qk_norm_rope:
            transformer.enable_fused_qk_norm_rope()
            if args.compile:
                logger.info(
                    "--h3_fused_qk_norm_rope requested with --compile: compiled blocks use Inductor fusion; "
                    "the explicit Triton kernel remains active for eager calls"
                )
        if getattr(args, "h3_fused_indexed_adaln", False):
            transformer.enable_fused_indexed_adaln()
            if args.compile:
                logger.warning("--h3_fused_indexed_adaln falls back to the Inductor path inside compiled blocks")
        if args.h3_convrot_int8_lora_fused:
            from musubi_tuner.modules.convrot_int8_utils import enable_convrot_int8_lora_fusion

            enabled = enable_convrot_int8_lora_fusion(transformer)
            if enabled == 0:
                raise RuntimeError("--h3_convrot_int8_lora_fused found no ConvRot INT8 Linear layers")

        sampler_state_path = "h3_timestep_sampler.json"

        def save_sampler_state(_models, _weights, output_dir):
            if accelerator.is_main_process:
                state = {
                    "num_timestep_buckets": self.num_timestep_buckets,
                    "timestep_range_pool": self.timestep_range_pool,
                }
                (Path(output_dir) / sampler_state_path).write_text(json.dumps(state), encoding="utf-8")

        def load_sampler_state(_models, input_dir):
            path = Path(input_dir) / sampler_state_path
            if not path.exists():
                # Older checkpoints did not save the partially consumed pool.
                # Starting a fresh cycle is safe, but cannot exactly reproduce
                # the pre-resume bucket ordering.
                self.timestep_range_pool = []
                return
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("num_timestep_buckets") != self.num_timestep_buckets:
                raise ValueError("saved H3 timestep sampler state does not match --num_timestep_buckets")
            pool = state.get("timestep_range_pool")
            if not isinstance(pool, list) or any(not isinstance(item, list) or len(item) != 2 for item in pool):
                raise ValueError(f"invalid H3 timestep sampler state: {path}")
            self.timestep_range_pool = [(float(lower), float(upper)) for lower, upper in pool]

        accelerator.register_save_state_pre_hook(save_sampler_state)
        accelerator.register_load_state_pre_hook(load_sampler_state)
        if self._crepa_config is None:
            return
        config = getattr(transformer, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise TypeError("MiniMax H3 CREPA requires transformer.config.hidden_size")
        self._crepa = H3CREPA(hidden_size, self._crepa_config)
        self._crepa.install(transformer)
        object.__setattr__(transformer, "_h3_crepa_controller", self._crepa)

        def save_crepa_state(_models, _weights, output_dir):
            if accelerator.is_main_process:
                self._crepa.save_state(output_dir)

        def load_crepa_state(_models, input_dir):
            if not self._crepa.load_state(input_dir):
                raise FileNotFoundError(f"CREPA resume state is missing: {Path(input_dir) / 'h3_crepa.safetensors'}")

        accelerator.register_save_state_pre_hook(save_crepa_state)
        accelerator.register_load_state_pre_hook(load_crepa_state)

    def extra_trainable_params(self, args, accelerator, network, transformer, trainable_params):
        if args is not None and args.h3_base_preservation_loss_weight > 0:
            if network is None:
                raise ValueError("--h3_base_preservation_loss_weight requires a trainable network")
            set_enabled = getattr(accelerator.unwrap_model(network), "set_enabled", None)
            if not callable(set_enabled):
                raise TypeError("H3 base-preservation loss requires a network with set_enabled()")
        del args, network, transformer
        if self._crepa is None:
            return trainable_params
        self._crepa.projector.to(device=accelerator.device, dtype=torch.float32)
        if not trainable_params or not isinstance(trainable_params[0], dict) or "params" not in trainable_params[0]:
            raise TypeError("MiniMax H3 CREPA requires the network optimizer parameters to use named parameter groups")
        groups = [dict(group) for group in trainable_params]
        groups[0]["params"] = [*groups[0]["params"], *self._crepa.projector.parameters()]
        return groups

    def extra_gradient_params(self) -> list[torch.nn.Parameter]:
        return [] if self._crepa is None else list(self._crepa.projector.parameters())

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        prompts = load_prompts(sample_prompts)
        logger.info("Encoding %d MiniMax H3 sampling prompt(s)", len(prompts))
        encoder = create_conditioning_encoder(
            text_encoder=Path(args.text_encoder),
            tokenizer=Path(args.tokenizer),
            task="t2va",
            device=str(accelerator.device),
            dtype="bfloat16",
            quantization=args.text_encoder_quantization,
            blocks_to_stream=args.h3_text_encoder_blocks_to_stream,
            nvfp4_scaled_mm=args.h3_nvfp4_scaled_mm,
        )
        prepared_images: list[list[Image.Image]] = []
        for prompt in prompts:
            height = prompt.get("height", 192)
            width = prompt.get("width", 320)
            images = []
            anchors = []
            if prompt.get("image_path"):
                with Image.open(prompt["image_path"]) as image:
                    images.append(prepare_keyframe_image(image, height, width, stretch=True))
                anchors.append("first")
            if prompt.get("end_image_path"):
                with Image.open(prompt["end_image_path"]) as image:
                    images.append(prepare_keyframe_image(image, height, width, stretch=False))
                anchors.append("last")
            prompt.update(encoder.encode_prompt(prompt.get("prompt", ""), images))
            prompt[_SAMPLE_KEYFRAME_ANCHORS] = tuple(anchors)
            prepared_images.append(images)
        encoder.close()
        del encoder
        gc.collect()
        clean_memory_on_device(accelerator.device)
        all_images = [image for images in prepared_images for image in images]
        encoded = iter(encode_keyframe_images(Path(args.vae), all_images, accelerator.device))
        for prompt, images in zip(prompts, prepared_images):
            rows = [next(encoded) for _ in images]
            prompt[_SAMPLE_KEYFRAME_ROWS] = torch.cat(rows) if rows else None
        return prompts

    def _generate_sample(
        self,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        decoder_bundle: _H3DecoderBundle,
        sample_parameter: dict,
    ):
        device = accelerator.device
        height = sample_parameter.get("height", 192)
        width = sample_parameter.get("width", 320)
        frame_count = align_frame_count(sample_parameter.get("frame_count", 124))
        sample_steps = sample_parameter.get("sample_steps", 20)
        seed = sample_parameter.get("seed", 42)
        generator = torch.Generator(device=device).manual_seed(seed)
        conditioning = {
            H3_TEXT_HIDDEN_KEY: sample_parameter[H3_TEXT_HIDDEN_KEY],
            H3_TEXT_TOKEN_TAGS_KEY: sample_parameter[H3_TEXT_TOKEN_TAGS_KEY],
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        denoise_started = time.perf_counter()
        video_latents, audio_latents = denoise_fl2va(
            transformer,
            conditioning,
            height=height,
            width=width,
            frame_count=frame_count,
            num_inference_steps=sample_steps,
            generator=generator,
            device=device,
            keyframe_rows=sample_parameter.get(_SAMPLE_KEYFRAME_ROWS),
            keyframe_anchors=sample_parameter.get(_SAMPLE_KEYFRAME_ANCHORS, ()),
            condition_seed=seed,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sample_metrics = {
            "joint_denoising": {
                "seconds": time.perf_counter() - denoise_started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
            }
        }
        video_latents = video_latents.cpu()
        audio_latents = audio_latents.cpu()

        block_swap_suspended = bool(self.blocks_to_swap)
        if block_swap_suspended:
            transformer.offload_block_swap_to_cpu()
        else:
            transformer.to("cpu")
        clean_memory_on_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        decode_started = time.perf_counter()
        try:
            media = decode_latents_sequentially(
                decoder_bundle.video_decoder,
                decoder_bundle.audio_decoder,
                video_latents,
                audio_latents,
                device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sample_metrics["sequential_av_decode"] = {
                "seconds": time.perf_counter() - decode_started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
            }
            self._last_sample_metrics = sample_metrics
            logger.info("MiniMax H3 training sample metrics: %s", sample_metrics)
        finally:
            if block_swap_suspended:
                transformer.move_to_device_except_swap_blocks(device)
                transformer.switch_block_swap_for_inference()
            else:
                transformer.to(device)
        return media, height, width, frame_count, sample_steps, seed

    def sample_image_inference(
        self,
        accelerator,
        args,
        transformer,
        dit_dtype,
        vae,
        save_dir,
        sample_parameter,
        epoch,
        steps,
    ):
        del dit_dtype
        media, height, width, frame_count, sample_steps, seed = self._generate_sample(
            accelerator,
            transformer,
            vae,
            sample_parameter,
        )
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        checkpoint = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        prompt_index = sample_parameter.get("enum", 0)
        prefix = "" if args.output_name is None else args.output_name + "_"
        output = Path(save_dir) / f"{prefix}{checkpoint}_{prompt_index:02d}_{timestamp}_{seed}.mp4"
        save_av_mp4(
            media,
            output,
            {
                "training_step": steps,
                "epoch": epoch,
                "prompt": sample_parameter.get("prompt", ""),
                "seed": seed,
                "height": height,
                "width": width,
                "frames": frame_count,
                "sigma_points": sample_steps,
                "model_evaluations": sample_steps - 1,
                "keyframe_anchors": list(sample_parameter.get(_SAMPLE_KEYFRAME_ANCHORS, ())),
                "metrics": self._last_sample_metrics,
            },
        )
        logger.info("Saved MiniMax H3 AV sample to %s", output)

    def do_inference(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("MiniMax H3 sampling uses its AV-aware sample_image_inference implementation")

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        del vae_dtype, vae_path
        logger.info("Loading MiniMax H3 video/audio decoders on CPU for sampling")
        return _H3DecoderBundle(
            load_video_vae_decoder(Path(args.vae), "cpu"),
            load_audio_vae_decoder(Path(args.audio_vae), "cpu"),
        )

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
        base_weight_paths = list(getattr(args, "base_weights", None) or [])
        base_lora_weights = [self.load_network_weights(path, "musubi_tuner.networks.lora_minimax_h3") for path in base_weight_paths]
        base_lora_multipliers = list(getattr(args, "base_weights_multiplier", None) or [])
        base_lora_multipliers.extend([1.0] * (len(base_lora_weights) - len(base_lora_multipliers)))
        base_lora_multipliers = base_lora_multipliers[: len(base_lora_weights)]
        backend_kwargs = dict(
            model=Path(dit_path),
            device=str(loading_device),
            dtype=model_utils.dtype_to_str(self.dit_dtype),
            mode=args.h3_training_mode,
            attention_mode=attn_mode,
            split_attention=split_attn,
            fp8_scaled=bool(args.fp8_base),
            adaln_rank=args.h3_adaln_rank,
            fp8_quantization_mode=args.h3_fp8_quantization_mode,
            convrot_int8=bool(args.h3_convrot_int8),
            convrot_int8_bwd=args.h3_convrot_int8_bwd,
            convrot_int8_fwd=args.h3_convrot_int8_fwd,
            quantization_device=str(accelerator.device),
            int8_convrot=bool(args.int8_convrot_base),
            target_device=str(accelerator.device),
            blocks_to_swap=int(getattr(args, "blocks_to_swap", 0) or 0),
            block_swap_h2d_only=bool(getattr(args, "block_swap_h2d_only", False)),
        )
        reference_image_short_edge = int(getattr(args, "reference_image_short_edge", REFERENCE_IMAGE_SHORT_EDGE))
        if reference_image_short_edge != REFERENCE_IMAGE_SHORT_EDGE:
            backend_kwargs["reference_image_short_edge"] = reference_image_short_edge
        if base_lora_weights:
            backend_kwargs["base_lora_weights"] = base_lora_weights
            backend_kwargs["base_lora_multipliers"] = base_lora_multipliers
        self.backend = create_training_backend(**backend_kwargs)
        transformer = self.backend.get_training_transformer()
        if not isinstance(transformer, torch.nn.Module):
            raise TypeError("H3 backend get_training_transformer() must return a torch.nn.Module")
        if args.h3_attn_auto_dispatch:
            transformer.enable_attention_auto_dispatch()
        if base_weight_paths:
            args.base_weights = None
            args.base_weights_multiplier = None
            accelerator.print("all H3 base weights merged during model loading: " + ", ".join(base_weight_paths))
        return transformer

    def compile_transformer(self, args, transformer):
        target_blocks = model_utils.resolve_compile_block_lists(transformer, ("blocks", "token_refiner.blocks"))
        count = sum(len(blocks) for blocks in target_blocks)
        logger.info("MiniMax H3: resolved %d regional torch.compile blocks", count)
        if count == 0:
            raise RuntimeError("--compile set but no H3 transformer blocks were resolved")
        return model_utils.compile_transformer(
            args,
            transformer,
            target_blocks,
            disable_linear=self.blocks_to_swap > 0,
        )

    def scale_shift_latents(self, latents):
        # H3 latent caches are written in the model's normalized latent space.
        return latents

    def _base_sigma(
        self,
        args: argparse.Namespace,
        noise_scheduler,
        timesteps: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Recover the *unshifted* schedule coordinate for this step.

        Both branches are unshifted only because ``--discrete_flow_shift`` is
        pinned to 1.0 (enforced in ``handle_model_specific_args``): the direct
        modes never apply it, and the scheduler branch builds
        ``FlowMatchDiscreteScheduler(shift=discrete_flow_shift)``. The chosen
        ``--timestep_sampling`` therefore only picks the *shape* of the base
        distribution; H3's own shifts are applied downstream.
        """
        if args.timestep_sampling in _DIRECT_SIGMA_SAMPLING:
            return ((timesteps.to(device=device, dtype=torch.float32) - 1.0) / 1000.0).clamp(0.0, 1.0)
        return get_sigmas(noise_scheduler, timesteps, device, n_dim=1, dtype=dtype).to(torch.float32)

    def _apply_frame_sigma_jitter(self, args, inputs, video_latents, video_noise, base_sigma, is_image):
        """Give each latent frame its own noise level around the shared schedule.

        One sigma per step supervises one point of the schedule per step. Drawing
        a nearby sigma per frame supervises a spread of the schedule in the same
        forward, which is worth most when data is scarce. The flow target
        ``x0 - noise`` does not depend on sigma, so only the noised input and the
        per-row timesteps change.
        """
        if self._frame_sigma_jitter <= 0 or inputs.video is None or is_image:
            return inputs, None
        frames = video_latents.shape[2]
        rows_per_frame_h, rows_per_frame_w = VIDEO_DIT_PATCH_SIZE[-2:]
        rows_per_frame = (video_latents.shape[-2] // rows_per_frame_h) * (video_latents.shape[-1] // rows_per_frame_w)
        base = float(base_sigma.reshape(-1)[0])
        epsilon = min(1e-4, self._frame_sigma_jitter * 0.5)
        lower = max(epsilon, base - self._frame_sigma_jitter)
        upper = min(1.0 - epsilon, base + self._frame_sigma_jitter)
        frame_base = lower + torch.rand(frames, device="cpu") * (upper - lower)
        frame_sigma = shift_sigma(frame_base, 1.0 if is_image else args.h3_shift_video)
        sigma = frame_sigma.to(device=video_latents.device, dtype=video_latents.dtype).view(1, 1, frames, 1, 1)
        noisy = (1.0 - sigma) * video_latents + sigma * video_noise
        row_timestep = (1.0 - frame_sigma).repeat_interleave(rows_per_frame)
        mean_sigma = frame_sigma.mean().reshape(1).to(device=inputs.video_sigma.device, dtype=inputs.video_sigma.dtype)
        return (
            replace(
                inputs,
                video=noisy,
                video_sigma=mean_sigma,
                video_timestep=1.0 - mean_sigma,
                video_frame_sigma=frame_sigma,
            ),
            row_timestep,
        )

    def _draw_spatial_density_scale(self):
        """Draw this step's spatial packing density.

        H3's spatial RoPE is area-normalized, so token spacing is fixed by the
        latent area and a single-resolution dataset teaches exactly one spacing.
        Perturbing the effective area per step synthesizes the range of spacings
        a multi-resolution dataset would supply, without re-caching anything.

        The factor is drawn log-uniformly so denser and sparser packing are
        equally likely, and one draw covers every spatial grid in the sequence so
        reference and target rows stay in coordinate correspondence.
        """
        if self._spatial_density_jitter <= 0:
            return None
        span = math.log1p(self._spatial_density_jitter)
        return float(torch.exp((torch.rand((), device="cpu") * 2 - 1) * span))

    def _resolve_keyframe_anchors(self, video):
        """Resolve this step's conditioning anchors.

        Returns ``(anchors, indices)``. ``anchors`` is what the packer receives
        and keeps ``"first"``/``"last"`` as themselves, because ``"last"`` names
        the final *pixel* frame while the integer ``frames - 1`` names the final
        latent window's start -- collapsing them would silently move the released
        anchor. ``indices`` says which latent frame supplies the content, where
        ``"last"`` does take the final window.
        """
        if video is None:
            return (), ()
        frames = video.shape[-3]
        if self._keyframe_random_count:
            count = min(self._keyframe_random_count, frames)
            drawn = sorted(int(index) for index in torch.randperm(frames, device="cpu")[:count])
            return tuple(drawn), tuple(drawn)
        anchors: list[int | str] = []
        indices: list[int] = []
        for anchor in self._keyframe_anchors:
            index = 0 if anchor == "first" else frames - 1 if anchor == "last" else int(anchor)
            if not 0 <= index < frames:
                raise ValueError(f"H3 keyframe anchor {anchor} is outside the {frames} target latent frames")
            anchors.append(anchor)
            indices.append(index)
        # Mirror the packer's identity rule: "first" and an explicit 0 name the
        # same coordinate and collide, while "last" is its own coordinate and
        # never collides with an index.
        identities = [0 if anchor == "first" else anchor for anchor in anchors]
        if len(set(identities)) != len(identities):
            raise ValueError("H3 keyframe anchors resolved to duplicate latent frames")
        order = sorted(range(len(indices)), key=lambda position: indices[position])
        return tuple(anchors[position] for position in order), tuple(indices[position] for position in order)

    @staticmethod
    def _clean_latents(noisy, target, sigma):
        """Recover x0 from the noised latents and their flow target."""
        shape = [1] * noisy.ndim
        shape[0] = sigma.shape[0]
        return noisy + sigma.to(device=noisy.device, dtype=noisy.dtype).view(shape) * target

    def _draw_step_mask(self, inputs, patch_size):
        """Draw one conditioning mask per step, shared by every forward it needs.

        The teacher, empty and trainable branches must all see the same observed
        region; drawing per forward would let them disagree about what is given.
        """
        if self._mask_mode == "off" and not self._mask_audio:
            return None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        video_rows = audio_rows = video_latent = audio_latent = None
        if self._mask_mode != "off" and inputs.video is not None:
            frames, height, width = inputs.video.shape[-3:]
            latent = sample_video_mask(
                mode=self._mask_mode,
                latent_frames=frames,
                latent_height=height,
                latent_width=width,
                generator=generator,
                minimum=self._mask_bounds[0],
                maximum=self._mask_bounds[1],
            )
            rows = video_mask_to_rows(latent, patch_size)
            # A patch counts as generated when any latent inside it is, so the
            # loss must score the whole patch rather than the drawn region.
            video_latent = rows_to_latent_video_mask(
                rows, latent_frames=frames, latent_height=height, latent_width=width, patch_size=patch_size
            )
            video_rows = ~rows
        if self._mask_audio and inputs.audio is not None:
            latent = sample_audio_mask(
                num_audio_latents=inputs.audio.shape[-1],
                generator=generator,
                minimum=self._mask_bounds[0],
                maximum=self._mask_bounds[1],
            )
            audio_rows = ~audio_mask_to_rows(latent, channels=AUDIO_CHANNELS)
            audio_latent = latent
        if video_rows is None and audio_rows is None:
            return None
        return SimpleNamespace(video_rows=video_rows, audio_rows=audio_rows, video_latent=video_latent, audio_latent=audio_latent)

    @staticmethod
    def _mask_to_loss(mask, target, generated, *, axis: int):
        """Restrict a modality's loss to the generated region."""
        if generated is None or target is None:
            return mask
        shape = [1] * target.ndim
        if generated.ndim == 1:
            shape[axis] = generated.shape[0]
        else:
            shape[-generated.ndim :] = list(generated.shape)
        keep = generated.to(device=target.device).view(shape).expand_as(target)
        if mask is None:
            return keep
        return keep & mask.to(device=target.device, dtype=torch.bool)

    @staticmethod
    def _clean_context(noisy, target, sigma, context_length: int, *, axis: int):
        """Recover the clean leading latents the observed context must present.

        The packed rows are already noised, so the context has to be rebuilt.
        H3's flow gives it exactly: ``x_t = (1 - s) * x0 + s * noise`` and
        ``target = x0 - noise`` imply ``x0 = x_t + s * target``, so no separate
        cache of the clean span is needed.
        """
        if noisy is None or target is None:
            raise ValueError("H3 extension needs both the noisy latents and their flow target")
        shape = [1] * noisy.ndim
        shape[0] = sigma.shape[0]
        clean = noisy + sigma.to(device=noisy.device, dtype=noisy.dtype).view(shape) * target
        index = [slice(None)] * noisy.ndim
        index[axis] = slice(0, context_length)
        return clean[tuple(index)].contiguous()

    @staticmethod
    def _extension_masked(mask, target, context_length: int, *, axis: int):
        """Drop the observed leading context from a modality's loss mask."""
        if not context_length or target is None:
            return mask
        length = target.shape[axis]
        if context_length >= length:
            raise ValueError(f"H3 extension context {context_length} covers the whole {length}-long target")
        keep = torch.ones(length, dtype=torch.bool, device=target.device)
        keep[:context_length] = False
        shape = [1] * target.ndim
        shape[axis] = length
        keep = keep.view(shape).expand_as(target)
        if mask is None:
            return keep
        return keep & mask.to(device=target.device, dtype=torch.bool)

    def _sample_weight(self, args: argparse.Namespace, sigma: torch.Tensor) -> torch.Tensor | None:
        if args.weighting_scheme == "sigma_sqrt":
            return sigma.clamp_min(1e-6).pow(-2.0)
        if args.weighting_scheme == "cosmap":
            return 2.0 / (math.pi * (1.0 - 2.0 * sigma + 2.0 * sigma.square()))
        return None

    @staticmethod
    def _guidance_loss_inputs(args, prediction, empty_prediction, inputs):
        video_sigma = inputs.video_frame_sigma if inputs.video_frame_sigma is not None else inputs.video_sigma
        video_scale = guidance_scale_for_sigma(
            args.h3_guidance_distillation_scale,
            video_sigma,
            args.h3_guidance_loss_schedule,
        )
        if inputs.video_frame_sigma is not None:
            video_scale = video_scale.reshape(1, 1, -1, 1, 1)
        audio_scale = guidance_scale_for_sigma(
            args.h3_guidance_distillation_scale,
            inputs.audio_sigma,
            args.h3_guidance_loss_schedule,
        )
        if args.h3_guidance_loss_form == "contrastive":
            target = contrastive_guidance_target(
                H3ModelPrediction(inputs.video_target, inputs.audio_target),
                empty_prediction,
                video_scale,
                audio_guidance_scale=audio_scale,
            )
            return prediction, replace(inputs, video_target=target.video, audio_target=target.audio)
        return (
            guidance_consistent_prediction(
                prediction,
                empty_prediction,
                video_scale,
                audio_guidance_scale=audio_scale,
            ),
            inputs,
        )

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
        video = inputs.video.to(device=accelerator.device, dtype=self.dit_dtype) if inputs.video is not None else None
        audio = inputs.audio.to(device=accelerator.device, dtype=self.dit_dtype) if inputs.audio is not None else None
        extension_kwargs = {}
        if self._step_row_video_timestep is not None:
            extension_kwargs["video_row_schedule"] = self._step_row_video_timestep
        if self._step_spatial_density_scale is not None:
            extension_kwargs["spatial_density_scale"] = self._step_spatial_density_scale
        anchors, anchor_indices = self._step_keyframes or ((), ())
        if anchors:
            extension_kwargs["condition_video_anchors"] = anchors
            extension_kwargs["extension_video_context"] = self._clean_latents(
                inputs.video, inputs.video_target, inputs.video_sigma
            ).index_select(-3, torch.tensor(anchor_indices, device=inputs.video.device))
        if self._step_mask is not None:
            if self._step_mask.video_rows is not None:
                extension_kwargs["observed_video_rows"] = self._step_mask.video_rows
                extension_kwargs["clean_video_latents"] = self._clean_latents(inputs.video, inputs.video_target, inputs.video_sigma)
            if self._step_mask.audio_rows is not None:
                extension_kwargs["observed_audio_rows"] = self._step_mask.audio_rows
                extension_kwargs["clean_audio_latents"] = self._clean_latents(inputs.audio, inputs.audio_target, inputs.audio_sigma)
        if self._extension_video_frames or self._extension_audio_latents:
            extension_kwargs["extension_route"] = self._extension_route
        if self._extension_video_frames:
            extension_kwargs["extension_video_frames"] = self._extension_video_frames
            extension_kwargs["extension_video_context"] = self._clean_context(
                inputs.video, inputs.video_target, inputs.video_sigma, self._extension_video_frames, axis=-3
            )
        if self._extension_audio_latents:
            extension_kwargs["extension_audio_latents"] = self._extension_audio_latents
            extension_kwargs["extension_audio_context"] = self._clean_context(
                inputs.audio, inputs.audio_target, inputs.audio_sigma, self._extension_audio_latents, axis=-1
            )
        with accelerator.autocast():
            prediction = self.backend.predict_training(
                transformer,
                batch,
                video,
                audio,
                inputs.video_timestep.to(accelerator.device),
                inputs.audio_timestep.to(accelerator.device),
                conditioning=conditioning,
                # Forwarded only when extension is active so a backend that does
                # not implement it keeps its existing signature.
                **extension_kwargs,
            )
        if not isinstance(prediction, H3ModelPrediction):
            raise TypeError("H3 backend predict_training() must return H3ModelPrediction")
        return prediction

    def get_primary_latents(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "latents" in batch:
            return batch["latents"]
        if H3_AUDIO_LATENTS_KEY in batch:
            return batch[H3_AUDIO_LATENTS_KEY]
        raise KeyError("MiniMax H3 cache contains neither video nor audio target latents")

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
        self._batch_backward_performed = False
        batch_size = int(latents.shape[0])
        if batch_size < 1:
            raise ValueError("MiniMax H3 training received an empty batch")
        preservation_active = args.h3_base_preservation_loss_weight > 0 and self._base_preservation_active(
            accelerator, args.h3_base_preservation_probability
        )
        if batch_size == 1:
            return self._process_single_batch(
                args,
                accelerator,
                transformer,
                network,
                batch,
                latents,
                noise,
                noise_scheduler,
                dit_dtype,
                network_dtype,
                vae,
                global_step,
                preservation_active_override=preservation_active,
            )

        # The released H3 transformer accepts one shared packed layout, while
        # prompts, references and task presentations are variable-length. Run
        # each packed item independently and backpropagate its scaled loss
        # immediately, so padding cannot leak through attention and only one
        # block-swap/checkpoint graph is alive at a time.
        losses: list[torch.Tensor] = []
        item_metrics: list[dict[str, float]] = []
        crepa_alignments: list[float] = []
        for index in range(batch_size):
            item_loss, metrics = self._process_single_batch(
                args,
                accelerator,
                transformer,
                network,
                self._slice_batch_item(batch, index, batch_size),
                latents[index : index + 1],
                noise[index : index + 1],
                noise_scheduler,
                dit_dtype,
                network_dtype,
                vae,
                global_step,
                preservation_active_override=preservation_active,
                crepa_update_similarity_threshold=False,
            )
            if "crepa/alignment" in metrics:
                crepa_alignments.append(metrics["crepa/alignment"])
            sync_context = (
                accelerator.no_sync(network if network is not None else transformer)
                if index + 1 < batch_size and getattr(accelerator, "num_processes", 1) > 1
                else nullcontext()
            )
            with sync_context:
                accelerator.backward(item_loss / batch_size)
            losses.append(item_loss.detach())
            item_metrics.append(metrics)

        averaged_metrics = self._average_batch_metrics(item_metrics)
        if self._crepa is not None and crepa_alignments:
            self._crepa.update_similarity_threshold(sum(crepa_alignments) / len(crepa_alignments))
            averaged_metrics["crepa/cutoff"] = float(self._crepa._cutoff_active)
            if self._crepa._similarity_ema is not None:
                averaged_metrics["crepa/alignment_ema"] = self._crepa._similarity_ema
        self._batch_backward_performed = True
        return torch.stack(losses).mean(), averaged_metrics

    def backward_loss(self, accelerator: Accelerator, loss: torch.Tensor) -> None:
        try:
            if getattr(self, "_batch_backward_performed", False):
                self._batch_backward_performed = False
                return
            super().backward_loss(accelerator, loss)
        finally:
            # Checkpoint recomputation needs the hook capture through backward,
            # but keeping it until the next step wastes the exact headroom used
            # by validation and sampling between optimizer steps.
            if self._crepa is not None:
                self._crepa.clear_step()

    @staticmethod
    def _average_batch_metrics(item_metrics: list[dict[str, float]]) -> dict[str, float]:
        metric_keys = set().union(*(metrics.keys() for metrics in item_metrics))
        return {
            key: sum(metrics[key] for metrics in item_metrics if key in metrics) / sum(key in metrics for metrics in item_metrics)
            for key in metric_keys
        }

    @staticmethod
    def _slice_batch_item(batch: dict, index: int, batch_size: int) -> dict:
        # BucketBatchManager stacks every fixed-size cache field on its first
        # dimension and leaves every varlen_ field as a list. Keep this rule in
        # one place; new shared metadata must remain scalar or opt into one of
        # those two dataset representations.
        item = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                item[key] = value[index : index + 1]
            elif isinstance(value, list) and len(value) == batch_size:
                item[key] = [value[index]]
            elif isinstance(value, tuple) and len(value) == batch_size:
                item[key] = (value[index],)
            else:
                item[key] = value
        return item

    def _process_single_batch(
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
        *,
        preservation_active_override: bool | None = None,
        crepa_update_similarity_threshold: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del network_dtype, vae
        has_video = "latents" in batch or latents.ndim == 5
        has_audio = H3_AUDIO_LATENTS_KEY in batch
        if not has_video and not has_audio:
            raise KeyError("MiniMax H3 cache contains no target modality")
        video_source = batch.get("latents", latents if latents.ndim == 5 else None)
        video_latents = video_source.to(device=accelerator.device, dtype=dit_dtype) if has_video else None
        audio_latents = batch[H3_AUDIO_LATENTS_KEY].to(device=accelerator.device, dtype=dit_dtype) if has_audio else None
        video_noise = noise.to(device=accelerator.device, dtype=dit_dtype) if video_latents is not None else None
        audio_noise = (
            noise.to(device=accelerator.device, dtype=dit_dtype)
            if audio_latents is not None and not has_video
            else torch.randn_like(audio_latents)
            if audio_latents is not None
            else None
        )
        is_image = has_video and not has_audio and video_latents.shape[2] == 1

        observed = args.h3_observed_modality
        if observed == "random":
            # One adapter covering joint generation, audio-driven video and
            # video-to-audio: the task is redrawn per step rather than fixed for
            # the run, so the model keeps all three rather than specialising.
            observed = (None, "video", "audio")[int(torch.randint(0, 3, (1,), device="cpu").item())]
        if observed is not None and not (has_video and has_audio):
            present = "video" if has_video else "audio" if has_audio else "neither"
            raise ValueError(
                f"--h3_observed_modality reads one modality while training the other, so batches must "
                f"carry both; this batch carries {present}. Cache the dataset with h3_target_mode = 'av'."
            )

        scheduler_args = args
        if is_image:
            patch_h, patch_w = VIDEO_DIT_PATCH_SIZE[-2:]
            latent_height, latent_width = video_latents.shape[-2:]
            if latent_height % patch_h or latent_width % patch_w:
                raise ValueError("MiniMax H3 image latent dimensions must be divisible by the spatial patch size")
            scheduler_args = copy.copy(args)
            if args.h3_image_flow_shift is None:
                # Use the common logit-normal density with a
                # resolution-aware shift for image batches.
                scheduler_args.timestep_sampling = "krea2_shift"
            else:
                scheduler_args.timestep_sampling = "shift"
                scheduler_args.discrete_flow_shift = args.h3_image_flow_shift

        _, scheduler_timesteps = super().get_noisy_model_input_and_timesteps(
            scheduler_args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
        )
        base_sigma = self._base_sigma(scheduler_args, noise_scheduler, scheduler_timesteps, accelerator.device, dit_dtype)
        inputs = prepare_joint_noisy_inputs(
            video_latents,
            audio_latents,
            video_noise,
            audio_noise,
            base_sigma,
            # Image sampling already returned its final shifted sigma. Video
            # batches instead receive H3's synchronized 12/3 shifts here.
            video_shift=1.0 if is_image else args.h3_shift_video,
            audio_shift=1.0 if is_image else args.h3_shift_audio,
            observed=observed,
        )

        inputs, self._step_row_video_timestep = self._apply_frame_sigma_jitter(
            args, inputs, video_latents, video_noise, base_sigma, is_image
        )
        self._step_spatial_density_scale = self._draw_spatial_density_scale()
        self._step_mask = self._draw_step_mask(inputs, tuple(VIDEO_DIT_PATCH_SIZE))
        # Drawn once per step for the same reason the mask is: the guidance and
        # base-preservation branches must condition on the same anchors as the
        # trainable branch, or the guidance correction inverts a different field.
        self._step_keyframes = self._resolve_keyframe_anchors(inputs.video)

        # H3 trains one item per step, so caption dropout is a single draw rather
        # than a per-sample mask. A dropped step trains the unconditional branch,
        # which is what gives the adapter a null prompt to contrast against at
        # inference; without it the model has never seen one.
        conditioning = "prompt"
        if args.h3_caption_dropout_rate > 0:
            missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
            if missing_empty:
                raise KeyError("--h3_caption_dropout_rate requires --cache_guidance_empty; missing " + ", ".join(missing_empty))
            if float(torch.rand((), device="cpu")) < args.h3_caption_dropout_rate:
                conditioning = "empty"

        # A dropped step is already unconditional, so there is no guided field to
        # invert and both branches would evaluate the same empty prompt.
        use_guidance = args.h3_guidance_distillation_scale is not None and conditioning == "prompt"
        if use_guidance:
            missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
            if missing_empty:
                raise KeyError(
                    "guidance-consistent H3 training requires --cache_guidance_empty; missing " + ", ".join(missing_empty)
                )
            # The empty branch calibrates the distilled field but is not itself
            # optimized. Evaluate it first without retaining its autograd graph.
            fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
            int8_context = getattr(transformer, "int8_attention_context", None)
            with (
                torch.random.fork_rng(devices=fork_devices),
                torch.no_grad(),
                int8_context(auxiliary=True) if callable(int8_context) else nullcontext(),
            ):
                empty_prediction = self._predict(
                    accelerator,
                    transformer,
                    batch,
                    inputs,
                    conditioning="empty",
                    gradient_checkpointing=False,
                )
        reference_prediction = None
        preservation_active = args.h3_base_preservation_loss_weight > 0 and (
            self._base_preservation_active(accelerator, args.h3_base_preservation_probability)
            if preservation_active_override is None
            else preservation_active_override
        )
        if preservation_active:
            if network is None:
                raise ValueError("--h3_base_preservation_loss_weight requires a trainable network")
            unwrapped_network = accelerator.unwrap_model(network)
            set_enabled = getattr(unwrapped_network, "set_enabled", None)
            if not callable(set_enabled):
                raise TypeError("H3 base-preservation loss requires a network with set_enabled()")
            fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
            # Restoring the RNG state makes the following trainable pass reuse
            # the stochastic conditioning rows sampled by the frozen branch.
            with torch.random.fork_rng(devices=fork_devices):
                set_enabled(False)
                try:
                    int8_context = getattr(transformer, "int8_attention_context", None)
                    with torch.no_grad(), int8_context(auxiliary=True) if callable(int8_context) else nullcontext():
                        reference_prediction = self._predict(
                            accelerator,
                            transformer,
                            batch,
                            inputs,
                            # Match the student's conditioning, or a dropped step
                            # would pull the unconditional branch toward the
                            # frozen base's conditional prediction.
                            conditioning=conditioning,
                            gradient_checkpointing=False,
                        )
                finally:
                    set_enabled(True)

        use_crepa = self._crepa is not None and has_video and not is_image and observed != "video"
        if self._crepa is not None:
            self._crepa.begin_step(use_crepa, global_step)
        try:
            raw_prediction = self._predict(
                accelerator,
                transformer,
                batch,
                inputs,
                conditioning=conditioning,
                gradient_checkpointing=args.gradient_checkpointing,
            )
        except Exception:
            if self._crepa is not None:
                self._crepa.clear_step()
            raise
        prediction = raw_prediction
        loss_inputs = inputs
        if use_guidance:
            prediction, loss_inputs = self._guidance_loss_inputs(args, prediction, empty_prediction, inputs)

        video_sample_weight = self._sample_weight(args, inputs.video_sigma) if has_video else None
        audio_sample_weight = self._sample_weight(args, inputs.audio_sigma) if has_audio else None
        video_weight = 0.0 if observed == "video" else args.h3_video_loss_weight
        audio_weight = 0.0 if observed == "audio" else args.h3_audio_loss_weight
        effective_video_mask = self._mask_to_loss(
            self._extension_masked(batch.get("video_loss_mask"), inputs.video_target, self._extension_video_frames, axis=-3),
            inputs.video_target,
            None if self._step_mask is None else self._step_mask.video_latent,
            axis=-3,
        )
        effective_audio_mask = self._mask_to_loss(
            self._extension_masked(batch.get("audio_loss_mask"), inputs.audio_target, self._extension_audio_latents, axis=-1),
            inputs.audio_target,
            None if self._step_mask is None else self._step_mask.audio_latent,
            axis=-1,
        )
        result = joint_velocity_loss(
            prediction,
            loss_inputs,
            # The observed context is given, not predicted, so it carries no
            # training signal and would otherwise dominate a short continuation.
            video_mask=effective_video_mask,
            audio_mask=effective_audio_mask,
            # Weighting keys on the shifted sigma the model actually saw for the
            # modality being generated, not the shared unshifted coordinate. An
            # observed modality sits at a pinned constant and would carry no
            # schedule information.
            video_sample_weight=video_sample_weight,
            audio_sample_weight=audio_sample_weight,
            balance=args.h3_loss_balance,
            # The observed modality is conditioning, not a target.
            video_weight=video_weight,
            audio_weight=audio_weight,
        )
        metrics = {
            "loss/video": float(result.video_loss.detach()),
            "loss/audio": float(result.audio_loss.detach()),
            "h3/sigma_video": float(inputs.video_sigma.mean().detach()),
            "h3/sigma_audio": float(inputs.audio_sigma.mean().detach()),
        }
        if result.video_elements == 0 and result.audio_elements == 0:
            metrics["h3/no_active_target"] = 1.0
        if args.h3_caption_dropout_rate > 0:
            # Only reported when the feature is on, so an existing run's metric
            # set is unchanged.
            metrics["h3/caption_dropped"] = float(conditioning == "empty")
        loss = result.loss
        base_preservation_term = None
        if reference_prediction is not None:
            preservation = joint_prediction_loss(
                raw_prediction,
                reference_prediction,
                video_mask=effective_video_mask,
                audio_mask=effective_audio_mask,
                video_sample_weight=video_sample_weight,
                audio_sample_weight=audio_sample_weight,
                balance=args.h3_loss_balance,
                video_weight=video_weight,
                audio_weight=audio_weight,
            )
            base_preservation_term = (
                args.h3_base_preservation_loss_weight / args.h3_base_preservation_probability
            ) * preservation.loss
            loss = loss + base_preservation_term
            metrics["loss/base_preservation"] = float(base_preservation_term.detach())
        if args.h3_base_preservation_loss_weight > 0:
            metrics["h3/base_preservation_active"] = float(preservation_active)
            metrics.setdefault("loss/base_preservation", 0.0)
        if use_crepa and self._crepa.active:
            crepa_loss, crepa_metrics = self._crepa.loss(
                batch.get("h3_dino_features"), update_similarity_threshold=crepa_update_similarity_threshold
            )
            loss = loss + crepa_loss
            metrics.update(crepa_metrics)
        elif use_crepa:
            metrics.update(self._crepa.status_metrics())
        if base_preservation_term is not None:
            metrics[LOSS_FOR_AVERAGE_KEY] = float((loss - base_preservation_term).detach())
        # Keep capture active until backward has completed. Non-reentrant
        # gradient checkpointing recomputes hooked blocks during backward and
        # requires the hook to perform the same tensor operations as forward.
        # begin_step() clears the captures before the next trainable pass.
        # Every consumer has run; nothing beyond this step may inherit the draw.
        self._step_mask = None
        self._step_row_video_timestep = None
        self._step_spatial_density_scale = None
        self._step_keyframes = None
        return loss, metrics

    def call_dit(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("MiniMax H3 uses its joint audio-video process_batch implementation")

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        return {
            "ss_h3_training_mode": args.h3_training_mode,
            "ss_h3_lora_token_refiner": str(args.h3_lora_token_refiner),
            "ss_h3_loss_balance": args.h3_loss_balance,
            "ss_h3_video_loss_weight": str(args.h3_video_loss_weight),
            "ss_h3_audio_loss_weight": str(args.h3_audio_loss_weight),
            "ss_h3_attn_auto_dispatch": str(args.h3_attn_auto_dispatch),
            "ss_h3_fused_indexed_adaln": str(args.h3_fused_indexed_adaln),
            "ss_h3_int8_attention": args.h3_int8_attention,
            "ss_h3_observed_modality": str(args.h3_observed_modality or "none"),
            "ss_h3_image_flow_shift": str(args.h3_image_flow_shift or "resolution_aware"),
            "ss_h3_guidance_distillation_scale": str(args.h3_guidance_distillation_scale or "one_pass"),
            "ss_h3_guidance_loss_form": args.h3_guidance_loss_form,
            "ss_h3_guidance_loss_schedule": args.h3_guidance_loss_schedule,
            "ss_h3_caption_dropout_rate": str(args.h3_caption_dropout_rate),
            "ss_h3_fp8_quantization_mode": args.h3_fp8_quantization_mode,
            "ss_h3_convrot_int8": str(args.h3_convrot_int8),
            "ss_h3_convrot_int8_bwd": args.h3_convrot_int8_bwd,
            "ss_h3_convrot_int8_fwd": args.h3_convrot_int8_fwd,
            "ss_h3_convrot_int8_lora_fused": str(args.h3_convrot_int8_lora_fused),
            "ss_h3_adaln_rank": str(args.h3_adaln_rank if args.h3_adaln_rank is not None else "full"),
            "ss_h3_reference_image_short_edge": str(args.reference_image_short_edge),
            "ss_h3_extension_video_frames": str(args.h3_extension_video_frames),
            "ss_h3_extension_audio_latents": str(args.h3_extension_audio_latents),
            "ss_h3_extension_route": args.h3_extension_route,
            "ss_h3_frame_sigma_jitter": str(args.h3_frame_sigma_jitter),
            "ss_h3_spatial_density_jitter": str(args.h3_spatial_density_jitter),
            "ss_h3_keyframe_anchors": args.h3_keyframe_anchors or "none",
            "ss_h3_keyframe_random_count": str(args.h3_keyframe_random_count),
            "ss_h3_mask_mode": args.h3_mask_mode,
            "ss_h3_mask_audio": str(args.h3_mask_audio),
            "ss_h3_base_preservation_loss_weight": str(args.h3_base_preservation_loss_weight),
            "ss_h3_base_preservation_probability": str(args.h3_base_preservation_probability),
            "ss_h3_shift_video": str(args.h3_shift_video),
            "ss_h3_shift_audio": str(args.h3_shift_audio),
            "ss_h3_timestep_sampling": args.timestep_sampling,
            "ss_h3_crepa": self._crepa_config.to_json() if self._crepa_config is not None else "disabled",
        }


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Train a MiniMax H3 LoRA with synchronized video and audio flow matching"
    parser.add_argument(
        "--h3_training_mode",
        choices=("fl2va", "ref2va", "ref2va_omni"),
        default="fl2va",
        help="select FL2VA, strict Ref2VA, or experimental zero-or-more-reference Ref2VA training",
    )
    parser.add_argument(
        "--h3_lora_token_refiner",
        action="store_true",
        help=(
            "also train LoRA adapters on the two H3 text token-refiner blocks; "
            "off by default and supported only by networks.lora_minimax_h3"
        ),
    )
    parser.add_argument("--text_encoder", type=str, help="Qwen3-VL H3 BF16 checkpoint used only for sampling prompts")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=default_text_encoder_assets(),
        help="H3 tokenizer/processor directory used for sampling; defaults to the metadata bundled with Musubi",
    )
    parser.add_argument("--audio_vae", type=str, help="MiniMax H3 audio VAE checkpoint used only for sampling")
    parser.add_argument(
        "--text_encoder_quantization",
        choices=("none", "int8", "nf4", "nvfp4_awq"),
        default="none",
        help="optional Qwen3-VL quantization while pre-encoding sampling prompts",
    )
    parser.add_argument(
        "--h3_text_encoder_blocks_to_stream",
        type=int,
        default=0,
        help="stream this many of the 50 frozen Qwen3-VL layers from CPU while encoding sample prompts (CUDA only)",
    )
    parser.add_argument(
        "--h3_nvfp4_scaled_mm",
        "--nvfp4_scaled_mm",
        action="store_true",
        help="use Blackwell W4A4 scaled_mm for a native NVFP4/AWQ sampling text encoder (PyTorch 2.10+)",
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
        "--h3_attn_auto_dispatch",
        action="store_true",
        help=(
            "prioritize cuDNN SDPA for large maskless CUDA BF16/FP16 attention shapes; "
            "short, masked, CPU, and FP32 workloads retain ordinary SDPA"
        ),
    )
    parser.add_argument(
        "--h3_observed_modality",
        type=str,
        default=None,
        choices=["video", "audio", "random"],
        help=(
            "Train one modality while the other is read as clean conditioning at the released "
            "transformer's own conditioning noise level. 'video' trains audio from video "
            "(video-to-audio / Foley); 'audio' trains video from audio; 'random' redraws the task "
            "each step across joint, video-observed and audio-observed, producing one adapter that "
            "keeps all three. Requires datasets that cache both modalities, and overrides the "
            "observed modality's loss weight to zero."
        ),
    )
    parser.add_argument(
        "--h3_image_flow_shift",
        type=float,
        default=None,
        help=(
            "override the default logit-normal, resolution-aware flow shift for image batches; "
            "video batches continue to use the released synchronized H3 schedule"
        ),
    )
    parser.add_argument(
        "--h3_guidance_distillation_scale",
        type=float,
        default=None,
        help="enable optional two-pass guidance-consistent training with an authoritative distillation scale",
    )
    parser.add_argument(
        "--h3_extension_video_frames",
        type=int,
        default=0,
        help=(
            "leading latent video frames observed as context instead of generated, training video extension. "
            "The observed span is packed as clean condition rows and removed from the loss"
        ),
    )
    parser.add_argument(
        "--h3_extension_audio_latents",
        type=int,
        default=0,
        help="leading audio latents observed as context instead of generated, training audio extension",
    )
    parser.add_argument(
        "--h3_frame_sigma_jitter",
        type=float,
        default=0.0,
        help=(
            "spread each latent frame's noise level around the step's shared schedule position by up to this much, "
            "so one forward supervises a range of the schedule instead of a single point"
        ),
    )
    parser.add_argument(
        "--h3_spatial_density_jitter",
        type=float,
        default=0.0,
        help=(
            "perturb the area normalization of the spatial RoPE grids by up to this fraction each step, drawn "
            "log-uniformly from [1/(1+j), 1+j], so fixed-resolution data still trains a range of token spacings. "
            "0 disables it"
        ),
    )
    parser.add_argument(
        "--h3_keyframe_anchors",
        type=str,
        default="",
        help=(
            "comma-separated conditioning frames given as clean context, each 'first', 'last', or a latent frame "
            "index, for example 'first,last' or '0,11,21'. Generalizes first/last keyframe conditioning to any set, "
            "training interpolation between arbitrary anchors"
        ),
    )
    parser.add_argument(
        "--h3_keyframe_random_count",
        type=int,
        default=0,
        help="draw this many distinct conditioning frames at random each step instead of listing them",
    )
    parser.add_argument(
        "--reference_image_short_edge",
        type=int,
        default=REFERENCE_IMAGE_SHORT_EDGE,
        help=(
            "short edge in pixels the Ref2VA reference caches were built with; it selects the matching reference "
            "cache keys and must equal the value given to latent caching"
        ),
    )
    parser.add_argument(
        "--h3_mask_mode",
        choices=("off", "box", "border", "segment"),
        default="off",
        help=(
            "procedural video conditioning mask drawn per step: box trains inpainting, border trains outpainting, "
            "segment hides a run of frames. The observed region is presented as clean context and excluded from the loss"
        ),
    )
    parser.add_argument(
        "--h3_mask_audio",
        action="store_true",
        help="also hide a contiguous run of audio latents, training audio inpainting alongside the video mask",
    )
    parser.add_argument(
        "--h3_mask_min_fraction",
        type=float,
        default=0.25,
        help="smallest fraction of each masked axis the generated region may cover",
    )
    parser.add_argument(
        "--h3_mask_max_fraction",
        type=float,
        default=0.75,
        help="largest fraction of each masked axis the generated region may cover",
    )
    parser.add_argument(
        "--h3_extension_route",
        choices=("condition_rows", "per_row_sigma"),
        default="condition_rows",
        help=(
            "how the observed context is presented. condition_rows duplicates it as clean rows, generalizing the "
            "released keyframe contract. per_row_sigma pins the observed rows inside the target block, costing no "
            "extra tokens but placing intra-block noise levels outside what the released weights have seen"
        ),
    )
    parser.add_argument(
        "--h3_caption_dropout_rate",
        type=float,
        default=0.0,
        help=(
            "probability of replacing the prompt with the cached empty conditioning for a step, training the "
            "unconditional branch; requires --cache_guidance_empty. Steps that drop the caption skip the "
            "guidance-consistent correction, which has nothing to invert without a prompt"
        ),
    )
    parser.add_argument(
        "--h3_guidance_loss_form",
        choices=("normalized", "contrastive"),
        default="normalized",
        help=(
            "normalized applies flow loss to the reconstructed conditional field; contrastive applies the equivalent "
            "scale-squared loss magnitude of a direct extrapolated target"
        ),
    )
    parser.add_argument(
        "--h3_guidance_loss_schedule",
        choices=("sigma", "constant"),
        default="sigma",
        help=(
            "sigma uses effective_scale = 1 + (configured_scale - 1) * modality_sigma; "
            "constant applies the configured guidance scale at every noise level"
        ),
    )
    parser.add_argument(
        "--h3_base_preservation_loss_weight",
        type=float,
        default=0.0,
        help=(
            "optional frozen-base prediction-preservation loss weight; adds one no-grad transformer forward on each "
            "batch selected by --h3_base_preservation_probability"
        ),
    )
    parser.add_argument(
        "--h3_base_preservation_probability",
        type=float,
        default=1.0,
        help=(
            "probability of evaluating the frozen-base preservation branch on a batch; active losses are divided by "
            "this probability to preserve the expected gradient, and the draw is synchronized across distributed ranks"
        ),
    )
    parser.add_argument(
        "--crepa",
        nargs="*",
        metavar="KEY=VALUE",
        default=None,
        help=(
            "enable temporal representation alignment; optional values: student_block=16 teacher_block=33 "
            "weight=0.05 tau=1 neighbors=2 schedule=constant warmup_steps=0 max_steps=0 normalize=true "
            "cutoff_step=0 similarity_ema_decay=0.99 threshold_mode=permanent"
        ),
    )
    parser.add_argument(
        "--int8_convrot_base",
        action="store_true",
        help="load the pruned Comfy INT8 ConvRot transformer for LoRA training",
    )
    parser.add_argument(
        "--h3_convrot_int8",
        action="store_true",
        help=(
            "quantize the released BF16 transformer to ConvRot INT8 as it loads, rather than reading a checkpoint "
            "that was quantized offline. Because the quantization happens after the weight transforms, it composes "
            "with --h3_adaln_rank and so quantizes a reduced AdaLN instead of the full-width projections the "
            "published pruned checkpoints carry"
        ),
    )
    parser.add_argument(
        "--h3_convrot_int8_bwd",
        choices=("bf16", "int8"),
        default="bf16",
        help="precision of the ConvRot backward pass; int8 is faster and coarser",
    )
    parser.add_argument(
        "--h3_convrot_int8_fwd",
        choices=("int8", "bf16"),
        default="int8",
        help=(
            "how the ConvRot forward evaluates its matmul. 'int8' rotates the activations and runs the fused "
            "INT8 kernel; 'bf16' undoes the rotation on the weight instead and hands the vendor GEMM an ordinary "
            "matrix. The stored weights and the arithmetic result are the same either way, so this trades "
            "quantized compute for a better-tuned kernel and is worth measuring on GPUs with fast BF16"
        ),
    )
    parser.add_argument(
        "--h3_convrot_int8_lora_fused",
        action="store_true",
        help=(
            "fuse the LoRA-up projection into the ConvRot INT8 dequantization epilogue; requires online ConvRot "
            "with INT8 forward and backward, and automatically falls back for LoRA dropout or split dimensions"
        ),
    )
    parser.add_argument(
        "--h3_fp8_quantization_mode",
        choices=("block", "channel", "tensor"),
        default="block",
        help=(
            "granularity of the scale that accompanies each FP8 weight. Block is the finest and the default; the "
            "measured difference between them is small because FP8 error is dominated by the mantissa rather than "
            "the scale"
        ),
    )
    parser.add_argument(
        "--h3_adaln_rank",
        type=int,
        default=None,
        help=(
            "reduce the AdaLN timestep projection to this rank while loading, shrinking the frozen base by ~13B "
            "parameters; the reduced weights stay in BF16 because they are no longer large enough to be worth quantizing"
        ),
    )
    parser.add_argument(
        "--h3_fused_qk_norm_rope",
        action="store_true",
        help=(
            "use the opt-in Triton kernel that fuses H3 per-head Q/K RMSNorm with split RoPE; "
            "unsupported shapes and torch.compile automatically use the eager/Inductor path"
        ),
    )
    parser.add_argument(
        "--h3_fused_indexed_adaln",
        action="store_true",
        help=(
            "use an opt-in Triton kernel that fuses each main-block RMSNorm with token-indexed AdaLN shift/scale; "
            "frozen LoRA bases use the fused forward/backward path and unsupported cases fall back safely"
        ),
    )
    parser.add_argument(
        "--h3_gradient_checkpointing_blocks",
        type=int,
        default=None,
        help=(
            "checkpoint only the last N of H3's 50 main blocks; default checkpoints all blocks. "
            "Lower values trade more VRAM for less recomputation and require resident eager blocks"
        ),
    )
    parser.add_argument(
        "--h3_gradient_checkpointing_cpu_offload_pin_memory",
        action="store_true",
        help=(
            "pin H3 CPU-offloaded checkpoint activations for faster transfers; requires substantial non-pageable host RAM "
            "and --gradient_checkpointing --gradient_checkpointing_cpu_offload"
        ),
    )
    parser.add_argument(
        "--h3_reusable_activation_offload",
        action="store_true",
        help=(
            "reuse pinned CPU buffers for checkpoint activations and prefetch them in reverse block order; "
            "requires --gradient_checkpointing --gradient_checkpointing_cpu_offload"
        ),
    )
    parser.add_argument(
        "--h3_int8_attention",
        choices=("off", "aux", "train"),
        default="off",
        help=(
            "experimental H3-owned INT8 attention: 'aux' applies it only to guidance/base-preservation teacher "
            "forwards, while 'train' also uses its optimized backward for the trainable forward; default 'off' "
            "leaves the selected SDPA/FlashAttention backend unchanged"
        ),
    )
    parser.add_argument(
        "--h3_shift_video",
        type=float,
        default=VIDEO_FLOW_SHIFT,
        help="exponential flow shift for the target video stream (H3 released schedule: 12.0)",
    )
    parser.add_argument(
        "--h3_shift_audio",
        type=float,
        default=AUDIO_FLOW_SHIFT,
        help="exponential flow shift for the target audio stream (H3 released schedule: 3.0)",
    )
    parser.set_defaults(
        network_module="networks.lora_minimax_h3",
        mixed_precision="bf16",
        # --timestep_sampling selects only the shape of the unshifted
        # coordinate; H3's per-modality shifts are applied on top of it. A
        # uniform base keeps usable sampling density at low sigma once the
        # video shift is applied.
        timestep_sampling="uniform",
        discrete_flow_shift=1.0,
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
