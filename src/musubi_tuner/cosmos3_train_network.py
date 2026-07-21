import argparse
import copy
import os
import sys
import time
import wave
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator, PartialState

from musubi_tuner import cosmos3_generate_video
from musubi_tuner.cosmos3 import cosmos3_utils, reasoner_kv_cache
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_COSMOS3, ARCHITECTURE_COSMOS3_FULL
from musubi_tuner.hv_train_network import (
    DiTOutput,
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
    should_sample_images,
)
from musubi_tuner.hv_generate_video import save_images_grid, save_videos_grid
from musubi_tuner.training.timesteps import get_sigmas
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def save_audio_wave(audio: torch.Tensor, path: str, sample_rate: int = 48000) -> None:
    audio_np = audio.detach().cpu().float().clamp(-1.0, 1.0)
    if audio_np.dim() == 3:
        audio_np = audio_np[0]
    audio_np = np.transpose(audio_np.numpy(), (1, 0))
    pcm = (audio_np * 32767.0).round().astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


class Cosmos3NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 4
        self.reasoner_kv_store = None
        self.reasoner_fps_modulation = True

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_COSMOS3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_COSMOS3_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = torch.bfloat16 if getattr(args, "dit_dtype", None) is None else model_utils.str_to_dtype(args.dit_dtype)
        self._i2v_training = bool(args.i2v)
        self._audio_training = bool(args.audio)
        self._control_training = False
        self.default_guidance_scale = 6.0
        self.default_discrete_flow_shift = 10.0

        if getattr(args, "reasoner_kv_cache_dir", None):
            if not os.path.isdir(args.reasoner_kv_cache_dir):
                raise ValueError(
                    f"--reasoner_kv_cache_dir {args.reasoner_kv_cache_dir} does not exist. "
                    "Build it with cosmos3_cache_reasoner_kv.py first."
                )
            # Sample-prompt verification happens in load_transformer, once the
            # model config has told us the real fps-modulation setting that the
            # cache key depends on.
            self.reasoner_kv_store = reasoner_kv_cache.ReasonerKVStore(
                args.reasoner_kv_cache_dir,
                device=None,
                dtype=None,
                lru_size=args.reasoner_kv_cache_lru,
            )
            logger.info(f"Cosmos3 training: replaying reasoner K/V from {args.reasoner_kv_cache_dir}")

        if args.network_module is None:
            args.network_module = "musubi_tuner.networks.lora_cosmos3"

        if args.vae_scale_factor_temporal < 1:
            raise ValueError("--vae_scale_factor_temporal must be positive")

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        prompts = load_prompts(sample_prompts)
        default_negative_prompt = cosmos3_utils.load_default_negative_prompt()
        for prompt_dict in prompts:
            if "negative_prompt" not in prompt_dict:
                prompt_dict["negative_prompt"] = default_negative_prompt
            prompt_dict.setdefault("use_system_prompt", False)
            prompt_dict.setdefault("add_resolution_template", True)
            prompt_dict.setdefault("add_duration_template", True)
            prompt_dict.setdefault("negative_metadata_mode", cosmos3_utils.DEFAULT_NEGATIVE_METADATA_MODE)
        return prompts

    def _move_transformer_to_sampling_device(self, transformer, device: torch.device) -> None:
        if getattr(transformer, "blocks_to_swap", 0) and hasattr(transformer, "move_to_device_except_swap_blocks"):
            transformer.move_to_device_except_swap_blocks(device)
        else:
            transformer.to(device)
        if hasattr(transformer, "prepare_block_swap_before_forward"):
            transformer.prepare_block_swap_before_forward()

    def _offload_transformer_after_sampling_step(self, transformer, device: torch.device, offload: bool) -> None:
        if not offload:
            return
        transformer.to("cpu")
        clean_memory_on_device(device)

    def _use_vae_on_sampling_device(self, vae, device: torch.device, use_device: bool) -> None:
        if use_device:
            vae.to(device)
            clean_memory_on_device(device)

    def _offload_vae_after_sampling_step(self, vae, device: torch.device, offload: bool) -> None:
        if offload:
            vae.to("cpu")
            clean_memory_on_device(device)

    def on_before_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        self._cosmos3_sample_network_was_training = None
        self._cosmos3_sample_network_dtype = None
        if network is not None:
            unwrapped_network = accelerator.unwrap_model(network)
            self._cosmos3_sample_network_was_training = unwrapped_network.training
            first_param = next((p for p in unwrapped_network.parameters() if p.is_floating_point()), None)
            self._cosmos3_sample_network_dtype = first_param.dtype if first_param is not None else None
            unwrapped_network.eval()
            if dit_dtype is not None and self._cosmos3_sample_network_dtype is not None:
                unwrapped_network.to(dtype=dit_dtype)

    def on_after_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        was_training = getattr(self, "_cosmos3_sample_network_was_training", None)
        network_dtype = getattr(self, "_cosmos3_sample_network_dtype", None)
        if network is not None and was_training is not None:
            unwrapped_network = accelerator.unwrap_model(network)
            if network_dtype is not None:
                unwrapped_network.to(dtype=network_dtype)
            if was_training:
                unwrapped_network.train()
            else:
                unwrapped_network.eval()
        self._cosmos3_sample_network_was_training = None
        self._cosmos3_sample_network_dtype = None

    def sample_images(self, accelerator: Accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
        if not should_sample_images(args, steps, epoch):
            return

        logger.info("")
        logger.info(f"generating Cosmos3 sample images at step: {steps}")
        if sample_parameters is None:
            logger.error(f"No prompt file: {args.sample_prompts}")
            return

        distributed_state = PartialState()
        transformer = accelerator.unwrap_model(transformer)
        was_training = transformer.training
        transformer.eval()
        transformer.switch_block_swap_for_inference()
        accelerator_wrapped_forward = None
        if hasattr(transformer, "_original_forward"):
            accelerator_wrapped_forward = transformer.forward
            transformer.forward = transformer._original_forward
            logger.info("Cosmos3 sampling: using original DiT forward without Accelerate mixed-precision wrapper")

        offload_dit = accelerator.device.type == "cuda" and (
            bool(getattr(args, "offload_dit_during_sampling", False)) or (self.blocks_to_swap or 0) > 0
        )
        if offload_dit:
            logger.info("Cosmos3 sampling: offloading DiT between denoising and VAE/AVAE decode")
            transformer.to("cpu")
            clean_memory_on_device(accelerator.device)

        sound_tokenizer = None
        if getattr(args, "audio", False):
            sound_source = args.sound_tokenizer if args.sound_tokenizer is not None else args.dit
            sound_device = "cpu" if offload_dit else accelerator.device
            sound_tokenizer = cosmos3_utils.load_sound_tokenizer(
                sound_source,
                args.sound_tokenizer_subfolder,
                dtype=model_utils.str_to_dtype(getattr(args, "sound_dtype", "bfloat16")),
                device=sound_device,
            )

        save_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)

        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        except Exception:
            pass

        try:
            if distributed_state.num_processes <= 1:
                with torch.no_grad():
                    for sample_parameter in sample_parameters:
                        self.sample_image_inference(
                            accelerator,
                            args,
                            transformer,
                            dit_dtype,
                            vae,
                            sound_tokenizer,
                            save_dir,
                            sample_parameter,
                            epoch,
                            steps,
                        )
                        clean_memory_on_device(accelerator.device)
            else:
                per_process_params = []
                for i in range(distributed_state.num_processes):
                    per_process_params.append(sample_parameters[i :: distributed_state.num_processes])

                with torch.no_grad():
                    with distributed_state.split_between_processes(per_process_params) as sample_parameter_lists:
                        for sample_parameter in sample_parameter_lists[0]:
                            self.sample_image_inference(
                                accelerator,
                                args,
                                transformer,
                                dit_dtype,
                                vae,
                                sound_tokenizer,
                                save_dir,
                                sample_parameter,
                                epoch,
                                steps,
                            )
                            clean_memory_on_device(accelerator.device)
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

            if sound_tokenizer is not None:
                cosmos3_generate_video.move_sound_tokenizer(sound_tokenizer, "cpu")
            if offload_dit:
                self._move_transformer_to_sampling_device(transformer, accelerator.device)
            if accelerator_wrapped_forward is not None:
                transformer.forward = accelerator_wrapped_forward
            transformer.switch_block_swap_for_training()
            if was_training:
                transformer.train()
            clean_memory_on_device(accelerator.device)

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        sound_tokenizer=None,
        image_path=None,
        control_video_path=None,
    ):
        tokenizer_path = args.tokenizer if args.tokenizer is not None else args.dit
        tokenizer = cosmos3_utils.load_tokenizer(tokenizer_path, args.tokenizer_subfolder)

        device = accelerator.device
        latent_dtype = dit_dtype if dit_dtype is not None else self.dit_dtype
        scheduler = cosmos3_utils.load_scheduler(args.dit, flow_shift=discrete_flow_shift)

        sample_args = copy.copy(args)
        sample_args.output_type = "video"
        sample_args.flow_shift = discrete_flow_shift
        sample_args.system_prompt = False
        sample_args.no_system_prompt = getattr(args, "no_system_prompt", False)
        sample_args.no_resolution_template = getattr(args, "no_resolution_template", False)
        sample_args.no_duration_template = getattr(args, "no_duration_template", False)
        sample_args.negative_metadata_mode = cosmos3_utils.DEFAULT_NEGATIVE_METADATA_MODE
        sample_args.sound_latent_length = sample_parameter.get("sound_latent_length", None)

        prompt_args = copy.deepcopy(sample_parameter)
        prompt_args["width"] = width
        prompt_args["height"] = height
        prompt_args["frame_count"] = frame_count
        prompt_args["sample_steps"] = sample_steps
        prompt_args["guidance_scale"] = guidance_scale if cfg_scale is None else cfg_scale
        prompt_args["discrete_flow_shift"] = discrete_flow_shift
        if image_path is not None:
            prompt_args["image_path"] = image_path

        _, video, _, audio = cosmos3_generate_video.sample_one(
            sample_args,
            prompt_args,
            transformer,
            tokenizer,
            scheduler,
            vae,
            sound_tokenizer,
            latent_dtype,
            device,
        )
        clean_memory_on_device(accelerator.device)

        if audio is not None:
            return video.to(torch.float32).cpu(), audio.to(torch.float32).cpu()
        return video.to(torch.float32).cpu()

    def sample_image_inference(
        self, accelerator, args, transformer, dit_dtype, vae, sound_tokenizer, save_dir, sample_parameter, epoch, steps
    ):
        sample_steps = sample_parameter.get("sample_steps", 35)
        width = sample_parameter.get("width", 256)
        height = sample_parameter.get("height", 256)
        frame_count = sample_parameter.get("frame_count", 1)
        guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
        discrete_flow_shift = sample_parameter.get("discrete_flow_shift", self.default_discrete_flow_shift)
        seed = sample_parameter.get("seed")
        prompt: str = sample_parameter.get("prompt", "")
        cfg_scale = sample_parameter.get("cfg_scale", None)
        negative_prompt = sample_parameter.get("negative_prompt", None)
        if not negative_prompt:
            negative_prompt = cosmos3_utils.load_default_negative_prompt()

        width, height, frame_count = cosmos3_utils.normalize_sample_dimensions(
            width,
            height,
            frame_count,
            args.vae_scale_factor_temporal,
        )

        if self.i2v_training:
            image_path = sample_parameter.get("image_path", None)
            if image_path is None:
                logger.error("No image_path for i2v model / i2vモデルのサンプル画像生成にはimage_pathが必要です")
                return
        else:
            image_path = None

        control_video_path = None
        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            torch.seed()
            torch.cuda.seed()
            generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

        logger.info(f"prompt: {prompt}")
        logger.info(f"height: {height}")
        logger.info(f"width: {width}")
        logger.info(f"frame count: {frame_count}")
        logger.info(f"sample steps: {sample_steps}")
        logger.info(f"guidance scale: {guidance_scale}")
        logger.info(f"discrete flow shift: {discrete_flow_shift}")
        if seed is not None:
            logger.info(f"seed: {seed}")

        do_classifier_free_guidance = negative_prompt is not None
        if do_classifier_free_guidance:
            logger.info(f"negative prompt: {negative_prompt}")
            logger.info(f"cfg scale: {cfg_scale}")
        if self.i2v_training:
            logger.info(f"image path: {image_path}")

        has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
        was_train = transformer.training if not has_self_ref_orig_mod else True
        if not has_self_ref_orig_mod:
            transformer.eval()

        result = self.do_inference(
            accelerator,
            args,
            sample_parameter,
            vae,
            dit_dtype,
            transformer,
            discrete_flow_shift,
            sample_steps,
            width,
            height,
            frame_count,
            generator,
            do_classifier_free_guidance,
            guidance_scale,
            cfg_scale,
            sound_tokenizer=sound_tokenizer,
            image_path=image_path,
            control_video_path=control_video_path,
        )

        if not has_self_ref_orig_mod:
            transformer.train(was_train)

        audio = None
        if isinstance(result, tuple):
            video, audio = result
        else:
            video = result

        if video is None:
            logger.error("No video generated / 生成された動画がありません")
            return

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)
        save_path = (
            f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )

        wandb_tracker = None
        try:
            wandb_tracker = accelerator.get_tracker("wandb")
            try:
                import wandb
            except ImportError as exc:
                raise ImportError("No wandb / wandb がインストールされていないようです") from exc
        except Exception:
            wandb = None

        if video.shape[2] == 1:
            image_paths = save_images_grid(video, save_dir, save_path, n_rows=video.shape[0], create_subdir=False)
            if wandb_tracker is not None and wandb is not None:
                for image_path_out in image_paths:
                    wandb_tracker.log({f"sample_{prompt_idx}": wandb.Image(image_path_out)}, step=steps)
        else:
            video_path = os.path.join(save_dir, save_path) + ".mp4"
            save_videos_grid(video, video_path)
            if wandb_tracker is not None and wandb is not None:
                wandb_tracker.log({f"sample_{prompt_idx}": wandb.Video(video_path)}, step=steps)

        if audio is not None:
            audio_path = os.path.join(save_dir, save_path) + ".wav"
            save_audio_wave(audio, audio_path, sample_rate=48000)
            if wandb_tracker is not None and wandb is not None:
                wandb_tracker.log({f"sample_audio_{prompt_idx}": wandb.Audio(audio_path, sample_rate=48000)}, step=steps)

        vae.to("cpu")
        clean_memory_on_device(device)

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_source = args.vae if args.vae is not None else args.dit
        return cosmos3_utils.load_vae(vae_source, args.vae_subfolder, dtype=vae_dtype, device="cpu")

    def _verify_sample_prompts_cached(self, args: argparse.Namespace):
        """Fail at startup if any sample prompt lacks a cached reasoner K/V.

        Mirrors the tokenization that ``do_inference`` performs, so a mismatch in
        fps or template flags between caching and training surfaces here rather
        than at the first sample step.
        """
        entries = reasoner_kv_cache.sample_prompt_cache_entries(
            args, args.sample_prompts, args.vae_scale_factor_temporal
        )
        missing = []
        checked = 0
        for ids, label in entries:
            key = reasoner_kv_cache.text_ids_cache_key(ids, self.reasoner_fps_modulation)
            checked += 1
            if not reasoner_kv_cache.cache_path_for(args.reasoner_kv_cache_dir, key).exists():
                missing.append(label)

        if missing:
            raise ValueError(
                f"{len(missing)} of {checked} sample-prompt reasoner K/V entries are missing from "
                f"{args.reasoner_kv_cache_dir}:\n  "
                + "\n  ".join(missing[:10])
                + "\n\nCache them with:\n"
                f"  python -m musubi_tuner.cosmos3_cache_reasoner_kv --dit {args.dit} "
                f"--reasoner_kv_cache_dir {args.reasoner_kv_cache_dir} "
                f"--sample_prompts {args.sample_prompts} --fps {args.fps}\n"
                "(the fps and --no_*_template flags must match this training run)"
            )
        logger.info(f"Cosmos3 training: all {checked} sample-prompt reasoner K/V entries present")

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        dtype = self.dit_dtype if args.fp8_scaled else dit_weight_dtype
        if getattr(args, "reasoner_kv_cache_dir", None):
            # The reasoner (und) tower is replayed from cache, so its weights are
            # never read; skip loading them entirely.
            reasoner_kv_cache.install_patches()
            model, removed = reasoner_kv_cache.load_transformer_gen_only(
                dit_path, args.transformer_subfolder, dtype, loading_device
            )
            reasoner_kv_cache.rebind_dispatch(model)
            # Resolve the cache key's fps-modulation component from the real
            # config: it selects float vs long text mrope positions, which changes
            # the RoPE baked into the cached keys.
            self.reasoner_fps_modulation = bool(getattr(model.config, "enable_fps_modulation", True))
            # Keep cached K/V resident on the compute device in the model dtype,
            # otherwise every layer of every step pays a host-to-device copy.
            if self.reasoner_kv_store is not None:
                self.reasoner_kv_store.device = accelerator.device
                self.reasoner_kv_store.dtype = self.dit_dtype
                # Attach inside run_transformer_for_sample so the training step,
                # in-training sampling, and both CFG branches are all covered by
                # one hook.
                reasoner_kv_cache.install_auto_attach(model, self.reasoner_kv_store, self.reasoner_fps_modulation)
            if getattr(args, "sample_prompts", None):
                self._verify_sample_prompts_cached(args)
            logger.info(
                f"Cosmos3 training: reasoner K/V cache enabled; omitted {removed} und parameter tensors."
            )
        else:
            model = cosmos3_utils.load_transformer(dit_path, args.transformer_subfolder, dtype, loading_device)
        cosmos3_utils.set_attention_backend(model, attn_mode)
        if args.fp8_scaled:
            keep_fp8_weights_on_cpu = (self.blocks_to_swap or 0) > 0
            if keep_fp8_weights_on_cpu:
                logger.info("Cosmos3 training: keeping FP8 weights on CPU because block swap is enabled.")
            cosmos3_utils.apply_scaled_fp8(
                model,
                accelerator.device,
                move_to_device=accelerator.device.type == "cuda" and not keep_fp8_weights_on_cpu,
            )
        return model

    def compile_transformer(self, args, transformer):
        layers = cosmos3_utils.get_transformer_layers(transformer)
        return model_utils.compile_transformer(args, transformer, [layers], disable_linear=(self.blocks_to_swap or 0) > 0)

    def scale_shift_latents(self, latents):
        return latents

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
        noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
            args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
        )
        sound_latents = None
        sound_noise = None
        sound_noisy_model_input = None
        sound_target = None
        if self._audio_training:
            if "sound_latents" not in batch:
                raise ValueError(
                    "Cosmos3 audio training requires sound latents in the cache. "
                    "Re-run cosmos3_cache_latents.py with --cache_audio."
                )
            sound_latents = batch["sound_latents"]
            if not isinstance(sound_latents, torch.Tensor):
                raise ValueError("Cosmos3 audio training currently requires fixed-length cached sound_latents tensors.")
            sound_latents = sound_latents.to(device=accelerator.device, dtype=network_dtype)
            sound_noise = torch.randn_like(sound_latents)
            sound_sigmas = get_sigmas(noise_scheduler, timesteps, accelerator.device, n_dim=sound_latents.ndim, dtype=dit_dtype)
            sound_noisy_model_input = sound_sigmas * sound_noise + (1.0 - sound_sigmas) * sound_latents
            sound_target = sound_noise - sound_latents

        target_override = None
        if self.i2v_training:
            latents_for_target = latents.to(device=accelerator.device, dtype=network_dtype)
            noise_for_target = noise.to(device=accelerator.device, dtype=network_dtype)
            noisy_model_input, target_override = cosmos3_utils.apply_i2v_conditioning(
                latents_for_target,
                noisy_model_input.to(device=accelerator.device, dtype=network_dtype),
                noise_for_target,
            )

        output = self.call_dit(
            args,
            accelerator,
            transformer,
            latents,
            batch,
            noise,
            noisy_model_input,
            timesteps,
            network_dtype,
            target_override=target_override,
            sound_latents=sound_latents,
            sound_noise=sound_noise,
            sound_noisy_model_input=sound_noisy_model_input,
            sound_target=sound_target,
        )
        return self.compute_loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype)

    def compute_loss(
        self,
        args: argparse.Namespace,
        output: DiTOutput,
        timesteps: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        video_loss, _ = super().compute_loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype)
        if "sound_pred" not in output.extra:
            return video_loss, {}

        sound_pred = output.extra["sound_pred"].unsqueeze(-1).unsqueeze(-1)
        sound_target = output.extra["sound_target"].unsqueeze(-1).unsqueeze(-1)
        sound_output = DiTOutput(pred=sound_pred, target=sound_target)
        sound_loss, _ = super().compute_loss(args, sound_output, timesteps, noise_scheduler, dit_dtype, network_dtype)
        loss = video_loss + float(args.sound_loss_weight) * sound_loss
        return loss, {
            "loss/video": float(video_loss.detach().float().item()),
            "loss/sound": float(sound_loss.detach().float().item()),
        }

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        model = transformer
        device = accelerator.device

        latents = latents.to(device=device, dtype=network_dtype)
        noise = noise.to(device=device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=device, dtype=network_dtype)
        timesteps = timesteps.to(device=device)

        input_ids_batch = batch["input_ids"]
        bsize = latents.shape[0]
        preds = []

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            sound_noisy_model_input = kwargs.get("sound_noisy_model_input")
            if sound_noisy_model_input is not None:
                sound_noisy_model_input.requires_grad_(True)

        with accelerator.autocast():
            sound_preds = []
            sound_noisy_model_input = kwargs.get("sound_noisy_model_input")
            for i in range(bsize):
                input_ids = input_ids_batch[i] if isinstance(input_ids_batch, list) else input_ids_batch[i]
                # Cached und K/V is attached inside run_transformer_for_sample by
                # install_auto_attach, keyed on the input_ids passed below.
                outputs_i = cosmos3_utils.run_transformer_for_sample(
                    model,
                    input_ids=input_ids,
                    vision_tokens=noisy_model_input[i : i + 1],
                    timestep=timesteps[i],
                    has_image_condition=self.i2v_training,
                    fps=float(args.fps),
                    device=device,
                    vae_scale_factor_temporal=args.vae_scale_factor_temporal,
                    sound_tokens=sound_noisy_model_input[i] if sound_noisy_model_input is not None else None,
                    sound_fps=float(args.sound_latent_fps),
                    return_dict=sound_noisy_model_input is not None,
                )
                if sound_noisy_model_input is not None:
                    pred_i = outputs_i["preds_vision"][0]
                    sound_preds.append(outputs_i["preds_sound"][0])
                else:
                    pred_i = outputs_i
                preds.append(pred_i)

        model_pred = torch.cat(preds, dim=0)
        target = kwargs.get("target_override")
        if target is None:
            target = noise - latents
        extra = {}
        if sound_preds:
            extra["sound_pred"] = torch.stack(sound_preds, dim=0)
            extra["sound_target"] = kwargs["sound_target"]
        return DiTOutput(pred=model_pred, target=target, extra=extra)


def cosmos3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--transformer_subfolder", type=str, default="transformer", help="subfolder for Cosmos3 transformer")
    parser.add_argument("--vae_subfolder", type=str, default="vae", help="subfolder for Cosmos3 VAE")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer path/repo, defaults to --dit")
    parser.add_argument("--tokenizer_subfolder", type=str, default="text_tokenizer", help="subfolder for Cosmos3 tokenizer")
    parser.add_argument("--sound_tokenizer", type=str, default=None, help="sound tokenizer path/repo, defaults to --dit")
    parser.add_argument("--sound_tokenizer_subfolder", type=str, default="sound_tokenizer", help="subfolder for Cosmos3 AVAE")
    parser.add_argument("--sound_dtype", type=str, default="bfloat16", help="data type for Cosmos3 AVAE sample decoding")
    parser.add_argument("--fps", type=float, default=24.0, help="Cosmos3 prompt/template and mRoPE FPS")
    parser.add_argument("--i2v", action="store_true", help="train image-to-video style by anchoring the first latent frame")
    parser.add_argument("--audio", action="store_true", help="train Cosmos3 video+audio jointly from cached AVAE sound latents")
    parser.add_argument("--sound_latent_fps", type=float, default=25.0, help="Cosmos3 AVAE latent FPS for mRoPE sound positions")
    parser.add_argument("--sound_loss_weight", type=float, default=1.0, help="loss multiplier for Cosmos3 sound velocity loss")
    parser.add_argument("--no_system_prompt", action="store_true", help="disable the Cosmos3 system prompt wrapper")
    parser.add_argument("--no_resolution_template", action="store_true", help="disable the Cosmos3 resolution template")
    parser.add_argument("--no_duration_template", action="store_true", help="disable the Cosmos3 duration template")
    parser.add_argument(
        "--vae_scale_factor_temporal",
        type=int,
        default=4,
        help="temporal compression factor of the Cosmos3/Wan VAE",
    )
    parser.add_argument("--offload_dit_during_sampling", action="store_true", help="offload Cosmos3 DiT while VAE/AVAE sampling decode runs")
    parser.add_argument(
        "--reasoner_kv_cache_dir",
        type=str,
        default=None,
        help="replay pre-computed reasoner (und) tower K/V from this directory and skip loading the ~8B "
        "reasoner weights; build it with cosmos3_cache_reasoner_kv.py",
    )
    parser.add_argument(
        "--reasoner_kv_cache_lru",
        type=int,
        default=32,
        help="number of reasoner K/V entries to keep in RAM (each is a few MiB)",
    )
    return parser


def main():
    _configure_utf8_stdio()
    parser = setup_parser_common()
    parser = cosmos3_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"
    args.dit_dtype = None

    trainer = Cosmos3NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
