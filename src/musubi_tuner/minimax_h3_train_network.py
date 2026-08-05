from __future__ import annotations

import argparse
import copy
import gc
import logging
import math
import time
from collections.abc import Sequence
from multiprocessing import Value
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image

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
from musubi_tuner.minimax_h3.dataset import create_h3_dataset_group
from musubi_tuner.minimax_h3.inference import (
    decode_latents_sequentially,
    denoise_fl2va,
    encode_keyframe_images,
    prepare_keyframe_image,
    save_av_mp4,
)
from musubi_tuner.minimax_h3.training import (
    H3ModelPrediction,
    guidance_consistent_prediction,
    joint_prediction_loss,
    joint_velocity_loss,
    prepare_joint_noisy_inputs,
)
from musubi_tuner.training.accelerator_setup import collator_class
from musubi_tuner.training.sampling_prompts import load_prompts
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
        self.default_discrete_flow_shift = 1.0
        self.vae_frame_stride = 17

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
        if args.sample_prompts:
            required = {
                "--text_encoder": args.text_encoder,
                "--vae": args.vae,
                "--audio_vae": args.audio_vae,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError("MiniMax H3 sampling during training requires " + ", ".join(missing))

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
        self.backend = create_training_backend(
            model=Path(dit_path),
            device=str(loading_device),
            dtype=model_utils.dtype_to_str(self.dit_dtype),
            mode=args.h3_training_mode,
            attention_mode=attn_mode,
            split_attention=split_attn,
            fp8_scaled=bool(args.fp8_base),
            adaln_rank=args.h3_adaln_rank,
            quantization_device=str(accelerator.device),
            int8_convrot=bool(args.int8_convrot_base),
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
        video = inputs.video.to(device=accelerator.device, dtype=self.dit_dtype) if inputs.video is not None else None
        audio = inputs.audio.to(device=accelerator.device, dtype=self.dit_dtype) if inputs.audio is not None else None
        if gradient_checkpointing:
            if video is not None:
                video.requires_grad_(True)
            if audio is not None:
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
        del network_dtype, vae, global_step
        if latents.shape[0] != 1:
            raise ValueError("MiniMax H3 training requires dataset batch_size = 1")
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
            else torch.randn_like(audio_latents) if audio_latents is not None else None
        )
        is_image = has_video and not has_audio and video_latents.shape[2] == 1

        observed = args.h3_observed_modality
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

        if args.h3_guidance_distillation_scale is not None:
            missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
            if missing_empty:
                raise KeyError(
                    "guidance-consistent H3 training requires --cache_guidance_empty; missing " + ", ".join(missing_empty)
                )
            # The empty branch calibrates the distilled field but is not itself
            # optimized. Evaluate it first without retaining its autograd graph.
            with torch.no_grad():
                empty_prediction = self._predict(
                    accelerator,
                    transformer,
                    batch,
                    inputs,
                    conditioning="empty",
                    gradient_checkpointing=False,
                )
        reference_prediction = None
        if args.h3_base_preservation_loss_weight > 0:
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
                    with torch.no_grad():
                        reference_prediction = self._predict(
                            accelerator,
                            transformer,
                            batch,
                            inputs,
                            conditioning="prompt",
                            gradient_checkpointing=False,
                        )
                finally:
                    set_enabled(True)

        raw_prediction = self._predict(
            accelerator,
            transformer,
            batch,
            inputs,
            conditioning="prompt",
            gradient_checkpointing=args.gradient_checkpointing,
        )
        prediction = raw_prediction
        if args.h3_guidance_distillation_scale is not None:
            prediction = guidance_consistent_prediction(prediction, empty_prediction, args.h3_guidance_distillation_scale)

        sample_weight = self._sample_weight(
            args, inputs.audio_sigma if observed == "video" or not has_video else inputs.video_sigma
        )
        video_weight = 0.0 if observed == "video" else args.h3_video_loss_weight
        audio_weight = 0.0 if observed == "audio" else args.h3_audio_loss_weight
        result = joint_velocity_loss(
            prediction,
            inputs,
            video_mask=batch.get("video_loss_mask"),
            audio_mask=batch.get("audio_loss_mask"),
            # Weighting keys on the shifted sigma the model actually saw for the
            # modality being generated, not the shared unshifted coordinate. An
            # observed modality sits at a pinned constant and would carry no
            # schedule information.
            sample_weight=sample_weight,
            balance=args.h3_loss_balance,
            # The observed modality is conditioning, not a target.
            video_weight=video_weight,
            audio_weight=audio_weight,
        )
        # normalized:  MSE((g + (s - 1)u) / s, target)
        # contrastive: MSE(g, u + s(target - u))
        # The objectives have the same optimum and gradient direction. The
        # contrastive loss is exactly s^2 larger, so scaling the normalized
        # result reproduces it without another forward or target allocation.
        guidance_loss_multiplier = 1.0
        if args.h3_guidance_distillation_scale is not None and args.h3_guidance_loss_form == "contrastive":
            guidance_loss_multiplier = args.h3_guidance_distillation_scale**2
        metrics = {
            "loss/video": float((result.video_loss * guidance_loss_multiplier).detach()),
            "loss/audio": float((result.audio_loss * guidance_loss_multiplier).detach()),
            "h3/sigma_video": float(inputs.video_sigma.mean().detach()),
            "h3/sigma_audio": float(inputs.audio_sigma.mean().detach()),
        }
        loss = result.loss * guidance_loss_multiplier
        if reference_prediction is not None:
            preservation = joint_prediction_loss(
                raw_prediction,
                reference_prediction,
                video_mask=batch.get("video_loss_mask"),
                audio_mask=batch.get("audio_loss_mask"),
                sample_weight=sample_weight,
                balance=args.h3_loss_balance,
                video_weight=video_weight,
                audio_weight=audio_weight,
            )
            loss = loss + args.h3_base_preservation_loss_weight * preservation.loss
            metrics["loss/base_preservation"] = float(preservation.loss.detach())
        return loss, metrics

    def call_dit(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("MiniMax H3 uses its joint audio-video process_batch implementation")

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        return {
            "ss_h3_training_mode": args.h3_training_mode,
            "ss_h3_loss_balance": args.h3_loss_balance,
            "ss_h3_video_loss_weight": str(args.h3_video_loss_weight),
            "ss_h3_audio_loss_weight": str(args.h3_audio_loss_weight),
            "ss_h3_observed_modality": str(args.h3_observed_modality or "none"),
            "ss_h3_image_flow_shift": str(args.h3_image_flow_shift or "resolution_aware"),
            "ss_h3_guidance_distillation_scale": str(args.h3_guidance_distillation_scale or "one_pass"),
            "ss_h3_guidance_loss_form": args.h3_guidance_loss_form,
            "ss_h3_base_preservation_loss_weight": str(args.h3_base_preservation_loss_weight),
            "ss_h3_shift_video": str(args.h3_shift_video),
            "ss_h3_shift_audio": str(args.h3_shift_audio),
            "ss_h3_timestep_sampling": args.timestep_sampling,
        }


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Train a MiniMax H3 LoRA with synchronized video and audio flow matching"
    parser.add_argument(
        "--h3_training_mode",
        choices=("fl2va", "ref2va", "ref2va_omni"),
        default="fl2va",
        help="select FL2VA, strict Ref2VA, or experimental zero-or-more-reference Ref2VA training",
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
        "--h3_loss_balance",
        choices=("token", "modality"),
        default="modality",
        help="combine joint AV loss over all valid latent elements or equally by modality means",
    )
    parser.add_argument("--h3_video_loss_weight", type=float, default=1.0)
    parser.add_argument("--h3_audio_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--h3_observed_modality",
        type=str,
        default=None,
        choices=["video", "audio"],
        help=(
            "Train one modality while the other is read as clean conditioning at the released "
            "transformer's own conditioning noise level. 'video' trains audio from video "
            "(video-to-audio / Foley); 'audio' trains video from audio. Requires datasets that "
            "cache both modalities, and overrides the observed modality's loss weight to zero."
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
        "--h3_guidance_loss_form",
        choices=("normalized", "contrastive"),
        default="normalized",
        help=(
            "normalized applies flow loss to the reconstructed conditional field; contrastive applies the equivalent "
            "scale-squared loss magnitude of a direct extrapolated target"
        ),
    )
    parser.add_argument(
        "--h3_base_preservation_loss_weight",
        type=float,
        default=0.0,
        help=(
            "optional frozen-base prediction-preservation loss weight; adds one no-grad transformer forward per batch"
        ),
    )
    parser.add_argument(
        "--int8_convrot_base",
        action="store_true",
        help="load the pruned Comfy INT8 ConvRot transformer for LoRA training",
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
