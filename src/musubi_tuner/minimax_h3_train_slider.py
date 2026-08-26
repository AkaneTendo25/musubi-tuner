from __future__ import annotations

import argparse
import gc
import logging
import random
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from multiprocessing import Value
from pathlib import Path

import torch

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.minimax_h3.architecture import AUDIO_FLOW_SHIFT, VIDEO_FLOW_SHIFT
from musubi_tuner.minimax_h3.backend import create_conditioning_encoder
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.slider import (
    H3SliderConfig,
    H3SliderDataset,
    H3SliderTarget,
    load_h3_slider_config,
    slider_collator,
    slider_direction_targets,
)
from musubi_tuner.minimax_h3.training import (
    H3JointNoisyInputs,
    H3ModelPrediction,
    joint_prediction_loss,
    prepare_joint_noisy_inputs,
)
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer
from musubi_tuner.minimax_h3_train_network import setup_parser as setup_h3_parser
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)


class MiniMaxH3SliderTrainer(MiniMaxH3NetworkTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.slider_config: H3SliderConfig | None = None
        self._slider_conditioning: dict[str, dict[str, torch.Tensor]] = {}
        self._slider_null_conditioning: dict[str, dict[str, torch.Tensor]] = {}
        self._slider_sampling_network = None
        self._slider_anchor_scale_ema: float | None = None

    def _validate_args_and_init(self, args) -> bool:
        self.slider_config = load_h3_slider_config(args.slider_config)
        if args.dataset_config is None:
            # The base trainer's generic validation requires a value, while the
            # slider supplies its own dataset group below.
            args.dataset_config = args.slider_config
        if self.slider_config.mode == "ref2va" and args.h3_training_mode not in {"ref2va", "ref2va_omni"}:
            raise ValueError("ref2va slider mode requires --h3_training_mode ref2va or ref2va_omni")
        if self.slider_config.mode != "ref2va" and args.h3_training_mode != "fl2va":
            raise ValueError("text/reference slider modes require --h3_training_mode fl2va")
        conflicts = {
            "--h3_caption_dropout_rate": args.h3_caption_dropout_rate > 0,
            "--h3_observed_modality": args.h3_observed_modality is not None,
            "--h3_mask_mode": args.h3_mask_mode != "off",
            "--h3_mask_audio": bool(args.h3_mask_audio),
            "--h3_extension_video_frames": args.h3_extension_video_frames > 0,
            "--h3_extension_audio_latents": args.h3_extension_audio_latents > 0,
            "--crepa": bool(args.crepa),
            "--h3_fuse_frozen_teachers": self.slider_config.mode == "text" and bool(args.h3_fuse_frozen_teachers),
        }
        active = [name for name, enabled in conflicts.items() if enabled]
        if active:
            raise ValueError("H3 slider training cannot combine with ordinary auxiliary objectives: " + ", ".join(active))
        return super()._validate_args_and_init(args)

    def _build_dataset(self, args):
        del args
        if self.slider_config is None:
            raise RuntimeError("H3 slider config was not initialized")
        self.num_timestep_buckets = None
        current_epoch = Value("i", 0)
        dataset = H3SliderDataset(self.slider_config)
        return dataset, slider_collator, current_epoch

    def _encode_text_slider_prompts(self, args, accelerator) -> None:
        if self.slider_config is None or self.slider_config.mode != "text":
            return
        if not args.text_encoder:
            raise ValueError("text H3 sliders require --text_encoder")
        prompts = set()
        for target in self.slider_config.targets:
            prompts.update((target.positive, target.negative, target.target_class))
        prompts.update(anchor.prompt for anchor in self.slider_config.anchors)
        encoder = create_conditioning_encoder(
            text_encoder=Path(args.text_encoder),
            tokenizer=Path(args.tokenizer),
            task="t2va",
            device=str(accelerator.device),
            dtype="bfloat16",
            quantization=args.text_encoder_quantization,
            blocks_to_stream=args.h3_text_encoder_blocks_to_stream,
            nvfp4_scaled_mm=args.h3_nvfp4_scaled_mm,
            text_visual_max_pixels=args.h3_text_visual_max_pixels,
            max_caption_tokens=args.h3_max_caption_tokens,
        )
        try:
            self._slider_conditioning = {
                prompt: {key: value.detach().cpu() for key, value in encoder.encode_prompt(prompt).items()}
                for prompt in sorted(prompts)
            }
            self._slider_null_conditioning = {
                prompt: {key: value.detach().cpu() for key, value in encoder.encode_null_prompt(prompt).items()}
                for prompt in sorted({target.target_class for target in self.slider_config.targets})
            }
        finally:
            encoder.close()
            del encoder
            gc.collect()
            clean_memory_on_device(accelerator.device)
        logger.info("Cached %d unique H3 slider prompt presentations", len(self._slider_conditioning))

    def _prepare_sampling(self, args, accelerator, vae_dtype):
        self._encode_text_slider_prompts(args, accelerator)
        return super()._prepare_sampling(args, accelerator, vae_dtype)

    def get_primary_latents(self, batch: dict[str, object]) -> torch.Tensor:
        if batch.get("slider_mode") == "text":
            video = batch.get("latents")
            return video if isinstance(video, torch.Tensor) else batch[H3_AUDIO_LATENTS_KEY]
        positive = batch["positive"]
        if "latents" in positive:
            return positive["latents"]
        return positive[H3_AUDIO_LATENTS_KEY]

    @staticmethod
    def _network_multiplier(accelerator, network, value: float) -> None:
        unwrapped = accelerator.unwrap_model(network)
        setter = getattr(unwrapped, "set_multiplier", None)
        if not callable(setter):
            raise TypeError("H3 slider training requires a LoRA network with set_multiplier()")
        setter(float(value))

    @staticmethod
    def _prompt_batch(
        conditioning: dict[str, torch.Tensor],
        accelerator,
    ) -> dict[str, torch.Tensor]:
        return {
            H3_TEXT_HIDDEN_KEY: conditioning[H3_TEXT_HIDDEN_KEY].to(accelerator.device),
            H3_TEXT_TOKEN_TAGS_KEY: conditioning[H3_TEXT_TOKEN_TAGS_KEY].to(accelerator.device),
            H3_CONDITIONING_TASK_KEY: torch.tensor([H3_CONDITIONING_TASK_IDS["t2va"]], device=accelerator.device, dtype=torch.long),
        }

    @staticmethod
    def _prediction_targets(
        positive: H3ModelPrediction,
        neutral: H3ModelPrediction,
        negative: H3ModelPrediction,
        strength: float,
    ) -> tuple[H3ModelPrediction, H3ModelPrediction]:
        enhance_video = erase_video = None
        enhance_audio = erase_audio = None
        if neutral.video is not None:
            enhance_video, erase_video = slider_direction_targets(positive.video, neutral.video, negative.video, strength)
        if neutral.audio is not None:
            enhance_audio, erase_audio = slider_direction_targets(positive.audio, neutral.audio, negative.audio, strength)
        return H3ModelPrediction(enhance_video, enhance_audio), H3ModelPrediction(erase_video, erase_audio)

    def _slider_loss(self, args, prediction: H3ModelPrediction, target: H3ModelPrediction) -> torch.Tensor:
        result = joint_prediction_loss(
            prediction,
            target,
            balance=args.h3_loss_balance,
            video_weight=args.h3_video_loss_weight,
            audio_weight=args.h3_audio_loss_weight,
        )
        return result.loss

    def _guided_slider_loss(
        self,
        args,
        accelerator,
        prediction: H3ModelPrediction,
        target: H3ModelPrediction,
        empty_prediction: H3ModelPrediction | None,
        inputs: H3JointNoisyInputs,
    ) -> torch.Tensor:
        """Apply the regular H3 guidance objective to a slider prediction target."""
        plain_loss = self._slider_loss(args, prediction, target)
        if empty_prediction is None:
            return plain_loss
        target_inputs = replace(inputs, video_target=target.video, audio_target=target.audio)
        guided_prediction, guided_inputs = self._guidance_loss_inputs(
            args,
            prediction,
            empty_prediction,
            target_inputs,
            accelerator=accelerator,
        )
        guided_target = H3ModelPrediction(guided_inputs.video_target, guided_inputs.audio_target)
        guided_loss = self._slider_loss(args, guided_prediction, guided_target)
        probability = args.h3_guidance_distillation_probability
        if probability < 1.0:
            return plain_loss + (guided_loss - plain_loss) / probability
        return guided_loss

    def _cap_anchor_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.slider_config is None:
            raise RuntimeError("H3 slider config was not initialized")
        value = float(loss.detach())
        previous = self._slider_anchor_scale_ema
        multiplier = self.slider_config.anchor_cap_mult
        if previous is not None and multiplier > 0 and value > previous * multiplier and value > 0:
            loss = loss * (previous * multiplier / value)
        if previous is None:
            self._slider_anchor_scale_ema = value
        else:
            rate = 0.02 if multiplier <= 0 or value <= previous * multiplier else 0.002
            self._slider_anchor_scale_ema = previous + rate * (value - previous)
        return loss

    def _text_inputs(
        self,
        args,
        accelerator,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
    ) -> H3JointNoisyInputs:
        video_latents = batch.get("latents")
        audio_latents = batch.get(H3_AUDIO_LATENTS_KEY)
        if isinstance(video_latents, torch.Tensor):
            video_latents = video_latents.to(device=accelerator.device, dtype=torch.float32)
            video_noise = noise.to(device=accelerator.device, dtype=torch.float32)
        else:
            video_noise = None
        if isinstance(audio_latents, torch.Tensor):
            audio_latents = audio_latents.to(device=accelerator.device, dtype=torch.float32)
            audio_noise = (
                noise.to(device=accelerator.device, dtype=torch.float32)
                if video_latents is None
                else torch.randn_like(audio_latents)
            )
        else:
            audio_noise = None
        primary = video_latents if video_latents is not None else audio_latents
        primary_noise = video_noise if video_latents is not None else audio_noise
        _, scheduler_timesteps = self.get_noisy_model_input_and_timesteps(
            args,
            primary_noise,
            primary,
            None,
            noise_scheduler,
            accelerator.device,
            dit_dtype,
            return_noisy=False,
        )
        base_sigma = self._base_sigma(args, noise_scheduler, scheduler_timesteps, accelerator.device)
        image = video_latents is not None and video_latents.shape[-3] == 1
        return prepare_joint_noisy_inputs(
            video_latents,
            audio_latents,
            video_noise,
            audio_noise,
            base_sigma,
            video_shift=1.0 if image else VIDEO_FLOW_SHIFT,
            audio_shift=1.0 if image else AUDIO_FLOW_SHIFT,
        )

    def _prepare_plain_prediction(self) -> None:
        self._step_mask = None
        self._step_row_video_timestep = None
        self._step_spatial_density_scale = 1.0
        self._step_keyframes = ()
        self._step_guides = ()
        self._step_reference_modality = "av"
        self._step_qwen_control_dropout = False
        self._step_recipe = None

    def _predict_prompt(self, accelerator, transformer, prompt: str, inputs: H3JointNoisyInputs) -> H3ModelPrediction:
        self._prepare_plain_prediction()
        return self._predict(
            accelerator,
            transformer,
            self._prompt_batch(self._slider_conditioning[prompt], accelerator),
            inputs,
            conditioning="prompt",
        )

    def _predict_null_prompt(self, accelerator, transformer, prompt: str, inputs: H3JointNoisyInputs) -> H3ModelPrediction:
        self._prepare_plain_prediction()
        return self._predict(
            accelerator,
            transformer,
            self._prompt_batch(self._slider_null_conditioning[prompt], accelerator),
            inputs,
            conditioning="prompt",
        )

    def _run_text_target(
        self,
        args,
        accelerator,
        transformer,
        network,
        target: H3SliderTarget,
        inputs: H3JointNoisyInputs,
        loss_scale: float,
        *,
        guidance_active: bool,
        preservation_active: bool,
    ) -> tuple[float, float, float]:
        if self.slider_config is None:
            raise RuntimeError("H3 slider config was not initialized")
        self._network_multiplier(accelerator, network, 0.0)
        auxiliary_swap = bool(self.blocks_to_swap) and not getattr(self, "_block_swap_h2d_only", False)
        if auxiliary_swap:
            transformer.switch_block_swap_for_inference()
        try:
            int8_context = getattr(transformer, "int8_attention_context", None)
            with torch.no_grad(), int8_context(auxiliary=True) if callable(int8_context) else nullcontext():
                positive = self._predict_prompt(accelerator, transformer, target.positive, inputs)
                neutral = self._predict_null_prompt(accelerator, transformer, target.target_class, inputs)
                negative = self._predict_prompt(accelerator, transformer, target.negative, inputs)
                anchor_targets = {
                    anchor.prompt: self._predict_prompt(accelerator, transformer, anchor.prompt, inputs)
                    for anchor in self.slider_config.anchors
                }
        finally:
            if auxiliary_swap:
                transformer.switch_block_swap_for_training()
        enhance, erase = self._prediction_targets(positive, neutral, negative, self.slider_config.guidance_strength)

        direction_values: list[float] = []
        anchor_values: list[float] = []
        preservation_values: list[float] = []
        for multiplier, direction_target in ((1.0, enhance), (-1.0, erase)):
            self._network_multiplier(accelerator, network, multiplier)
            empty_prediction = None
            reference_prediction = None
            if guidance_active or preservation_active:
                auxiliary_swap = bool(self.blocks_to_swap) and not getattr(self, "_block_swap_h2d_only", False)
                if auxiliary_swap:
                    transformer.switch_block_swap_for_inference()
                try:
                    int8_context = getattr(transformer, "int8_attention_context", None)
                    fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
                    with (
                        torch.random.fork_rng(devices=fork_devices),
                        torch.no_grad(),
                        int8_context(auxiliary=True) if callable(int8_context) else nullcontext(),
                    ):
                        if guidance_active:
                            null_set_enabled = (
                                self._runtime_network_toggle(accelerator, network, "--h3_guidance_null_source frozen")
                                if args.h3_guidance_null_source == "frozen"
                                else None
                            )
                            if null_set_enabled is not None:
                                null_set_enabled(False)
                            try:
                                empty_prediction = self._predict_null_prompt(accelerator, transformer, target.target_class, inputs)
                            finally:
                                if null_set_enabled is not None:
                                    null_set_enabled(True)
                        if preservation_active:
                            set_enabled = self._runtime_network_toggle(accelerator, network, "--h3_base_preservation_loss_weight")
                            set_enabled(False)
                            try:
                                reference_prediction = self._predict_prompt(accelerator, transformer, target.target_class, inputs)
                            finally:
                                set_enabled(True)
                finally:
                    if auxiliary_swap:
                        transformer.switch_block_swap_for_training()

            prediction = self._predict_prompt(accelerator, transformer, target.target_class, inputs)
            direction_loss = self._guided_slider_loss(args, accelerator, prediction, direction_target, empty_prediction, inputs)
            preservation_loss = direction_loss.new_zeros(())
            if reference_prediction is not None:
                preservation_loss = self._slider_loss(args, prediction, reference_prediction)
                preservation_loss = (
                    args.h3_base_preservation_loss_weight / args.h3_base_preservation_probability
                ) * preservation_loss
            anchor_loss = direction_loss.new_zeros(())
            for prompt, anchor_target in anchor_targets.items():
                anchor_prediction = self._predict_prompt(accelerator, transformer, prompt, inputs)
                anchor_loss = anchor_loss + self._slider_loss(args, anchor_prediction, anchor_target)
            if anchor_targets:
                anchor_loss = self._cap_anchor_loss(anchor_loss / len(anchor_targets))
                anchor_loss = anchor_loss * self.slider_config.anchor_strength
            total = (direction_loss + anchor_loss + preservation_loss) * target.weight * loss_scale * 0.5
            accelerator.backward(total)
            direction_values.append(float(direction_loss.detach()))
            anchor_values.append(float(anchor_loss.detach()))
            preservation_values.append(float(preservation_loss.detach()))
        self._network_multiplier(accelerator, network, 1.0)
        return (
            sum(direction_values) / 2.0,
            sum(anchor_values) / 2.0,
            sum(preservation_values) / 2.0,
        )

    def _process_text_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.slider_config is None:
            raise RuntimeError("H3 slider config was not initialized")
        inputs = self._text_inputs(args, accelerator, batch, latents, noise, noise_scheduler, dit_dtype)
        targets = self.slider_config.targets
        chosen = targets if self.slider_config.batch_all_targets else (random.choice(targets),)
        guidance_active = args.h3_guidance_distillation_scale is not None and self._guidance_distillation_active(
            accelerator, args.h3_guidance_distillation_probability
        )
        preservation_active = args.h3_base_preservation_loss_weight > 0 and self._base_preservation_active(
            accelerator, args.h3_base_preservation_probability
        )
        direction = 0.0
        anchor = 0.0
        preservation = 0.0
        for target in chosen:
            target_direction, target_anchor, target_preservation = self._run_text_target(
                args,
                accelerator,
                transformer,
                network,
                target,
                inputs,
                1.0 / len(chosen),
                guidance_active=guidance_active,
                preservation_active=preservation_active,
            )
            direction += target_direction / len(chosen)
            anchor += target_anchor / len(chosen)
            preservation += target_preservation / len(chosen)
        self._batch_backward_performed = True
        metrics = {
            "loss/slider_direction": direction,
            "loss/slider_anchor": anchor,
            "loss/base_preservation": preservation,
        }
        if args.h3_guidance_distillation_probability < 1.0:
            metrics["h3/guidance_distillation_active"] = float(guidance_active)
        if args.h3_base_preservation_loss_weight > 0:
            metrics["h3/base_preservation_active"] = float(preservation_active)
        return torch.tensor(direction + anchor + preservation, device=accelerator.device), metrics

    def _process_paired_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        positive = batch["positive"]
        negative = batch["negative"]
        positive_latents = self.get_primary_latents({"positive": positive})
        negative_latents = self.get_primary_latents({"positive": negative})
        if positive_latents.shape != negative_latents.shape:
            raise ValueError("paired H3 slider primary latent shapes must match")
        # Each fork starts from the same global RNG state and restores it on
        # exit. The two branches therefore share sigma, auxiliary modality
        # noise, conditioning draws, and model dropout without perturbing the
        # surrounding trainer stream.
        fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
        preservation_active = getattr(args, "h3_base_preservation_loss_weight", 0.0) > 0 and self._base_preservation_active(
            accelerator, args.h3_base_preservation_probability
        )
        guidance_active = getattr(args, "h3_guidance_distillation_scale", None) is not None and self._guidance_distillation_active(
            accelerator, args.h3_guidance_distillation_probability
        )
        losses = []
        metrics = []
        for multiplier, item_batch, item_latents in (
            (1.0, positive, positive_latents),
            (-1.0, negative, negative_latents),
        ):
            self._network_multiplier(accelerator, network, multiplier)
            with torch.random.fork_rng(devices=fork_devices):
                loss, item_metrics = self._process_single_batch(
                    args,
                    accelerator,
                    transformer,
                    network,
                    item_batch,
                    item_latents,
                    noise,
                    noise_scheduler,
                    dit_dtype,
                    network_dtype,
                    vae,
                    global_step,
                    preservation_active_override=preservation_active,
                    guidance_active_override=guidance_active,
                    recipe_override=None,
                    qwen_control_dropout_override=False,
                )
                accelerator.backward(loss * 0.5)
            losses.append(loss.detach())
            metrics.append(item_metrics)
        self._network_multiplier(accelerator, network, 1.0)
        self._batch_backward_performed = True
        averaged = self._average_batch_metrics(metrics)
        averaged["loss/slider_positive"] = float(losses[0])
        averaged["loss/slider_negative"] = float(losses[1])
        return torch.stack(losses).mean(), averaged

    def process_batch(
        self,
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
    ):
        self._batch_backward_performed = False
        if batch.get("slider_mode") == "text":
            return self._process_text_batch(
                args, accelerator, transformer, network, batch, latents, noise, noise_scheduler, dit_dtype
            )
        return self._process_paired_batch(
            args,
            accelerator,
            transformer,
            network,
            batch,
            noise,
            noise_scheduler,
            dit_dtype,
            network_dtype,
            vae,
            global_step,
        )

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        metadata = super().extra_metadata(args)
        if self.slider_config is not None:
            metadata.update(
                {
                    "ss_h3_slider_mode": self.slider_config.mode,
                    "ss_h3_slider_target_modality": self.slider_config.target_modality,
                    "ss_h3_slider_guidance_strength": str(self.slider_config.guidance_strength),
                    "ss_h3_slider_range": ",".join(str(value) for value in self.slider_config.sample_slider_range),
                }
            )
        return metadata

    def on_before_sample_images(
        self,
        accelerator,
        args,
        epoch,
        steps,
        vae,
        transformer,
        network,
        sample_parameters,
        dit_dtype,
    ) -> None:
        del args, epoch, steps, vae, transformer, sample_parameters, dit_dtype
        self._slider_sampling_network = network
        self._network_multiplier(accelerator, network, 1.0)

    def sample_images(self, accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
        if self.slider_config is None or self._slider_sampling_network is None:
            return super().sample_images(accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype)
        original_name = args.output_name
        try:
            for multiplier in self.slider_config.sample_slider_range:
                self._network_multiplier(accelerator, self._slider_sampling_network, multiplier)
                suffix = f"slider_{multiplier:+g}".replace("+", "p").replace("-", "m").replace(".", "p")
                args.output_name = f"{original_name}_{suffix}" if original_name else suffix
                super().sample_images(accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype)
        finally:
            args.output_name = original_name
            self._network_multiplier(accelerator, self._slider_sampling_network, 1.0)

    def on_after_sample_images(
        self,
        accelerator,
        args,
        epoch,
        steps,
        vae,
        transformer,
        network,
        sample_parameters,
        dit_dtype,
    ) -> None:
        del args, epoch, steps, vae, transformer, sample_parameters, dit_dtype
        self._network_multiplier(accelerator, network, 1.0)
        self._slider_sampling_network = None


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = setup_h3_parser(parser)
    parser.add_argument("--slider_config", type=str, required=True, help="H3 slider TOML configuration")
    return parser


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(setup_parser_common())


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    args = read_config_from_file(args, parser)
    args.dit_dtype = None
    trainer = MiniMaxH3SliderTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
