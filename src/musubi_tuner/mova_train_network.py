import argparse
import logging
import math
import os
import time
from typing import Optional

import torch
from accelerate import Accelerator
from PIL import Image

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_MOVA, ARCHITECTURE_MOVA_FULL
from musubi_tuner.hv_generate_video import resize_image_to_bucket
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.mova.audio_vae import load_audio_vae
from musubi_tuner.mova.generation import MovaVAEBundle, save_video_with_audio
from musubi_tuner.mova.interaction import DualTowerConditionalBridge
from musubi_tuner.mova.model_bundle import MovaModelBundle
from musubi_tuner.mova.text_encoder import encode_hidden_states, load_text_encoder, load_tokenizer
from musubi_tuner.mova.wan_audio_dit import WanAudioModel
from musubi_tuner.mova.wan_video_dit import WanModel
from musubi_tuner.utils import model_utils
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


def pad_prompt_embeds(prompt_embeds: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    max_len = max(t.shape[0] for t in prompt_embeds)
    dim = prompt_embeds[0].shape[-1]
    padded = torch.zeros((len(prompt_embeds), max_len, dim), device=device, dtype=dtype)
    for i, embed in enumerate(prompt_embeds):
        padded[i, : embed.shape[0]] = embed.to(device=device, dtype=dtype)
    return padded


class MovaNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.latest_video_loss: Optional[float] = None
        self.latest_audio_loss: Optional[float] = None

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MOVA

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MOVA_FULL

    @property
    def i2v_training(self) -> bool:
        return False

    @property
    def control_training(self) -> bool:
        return False

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.default_guidance_scale = 5.0
        self.latest_video_loss = None
        self.latest_audio_loss = None

        if args.alternate_split_timestep > 1:
            args.alternate_split_timestep /= 1000.0
        if not (0.0 < args.alternate_split_timestep < 1.0):
            raise ValueError("alternate_split_timestep must be in the range (0, 1] or [1, 1000]")

        if args.offload_inactive_dit and args.blocks_to_swap:
            raise ValueError("offload_inactive_dit cannot be used with blocks_to_swap for MOVA")

        if args.sample_prompts is not None:
            if args.text_encoder is None:
                raise ValueError("--text_encoder is required for MOVA sample generation")
            if args.vae is None:
                raise ValueError("--vae is required for MOVA sample generation")

        args.network_args = list(args.network_args) if args.network_args is not None else []
        if not any(arg.startswith("target_scope=") for arg in args.network_args):
            args.network_args.append(f"target_scope={args.lora_scope}")

    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        logs = super().generate_step_logs(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer=optimizer,
            keys_scaled=keys_scaled,
            mean_norm=mean_norm,
            maximum_norm=maximum_norm,
        )
        if self.latest_video_loss is not None:
            logs["loss/video"] = self.latest_video_loss
        if self.latest_audio_loss is not None:
            logs["loss/audio"] = self.latest_audio_loss
        return logs

    def sample_images(self, accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
        return super().sample_images(accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype)

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        tokenizer_path = args.tokenizer if args.tokenizer is not None else args.text_encoder
        tokenizer = load_tokenizer(
            tokenizer_path,
            subfolder=args.tokenizer_subfolder,
            trust_remote_code=args.trust_remote_code,
        )

        text_encoder_dtype = torch.bfloat16 if args.text_encoder_dtype is None else model_utils.str_to_dtype(args.text_encoder_dtype)
        text_encoder = load_text_encoder(
            args.text_encoder,
            text_encoder_dtype,
            subfolder=args.text_encoder_subfolder,
            trust_remote_code=args.trust_remote_code,
        )
        text_encoder.eval().requires_grad_(False).to(device=accelerator.device, dtype=text_encoder_dtype)

        prompt_cache: dict[str, torch.Tensor] = {}

        def encode_prompt(prompt: str) -> torch.Tensor:
            if prompt not in prompt_cache:
                encoded = tokenizer(
                    [prompt],
                    padding=True,
                    truncation=True,
                    max_length=args.text_encoder_max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device=accelerator.device) for k, v in encoded.items()}
                with torch.no_grad():
                    hidden_states = encode_hidden_states(text_encoder, encoded)[0]
                valid_length = int(encoded["attention_mask"][0].sum().item())
                prompt_cache[prompt] = hidden_states[:valid_length].detach().cpu()
            return prompt_cache[prompt]

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["t5_embeds"] = encode_prompt(prompt_dict.get("prompt", ""))

            negative_prompt = prompt_dict.get("negative_prompt")
            if negative_prompt is not None:
                prompt_dict_copy["negative_t5_embeds"] = encode_prompt(negative_prompt)

            sample_parameters.append(prompt_dict_copy)

        del text_encoder
        clean_memory_on_device(accelerator.device)
        return sample_parameters

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        resolved_path = resolve_video_vae_path(vae_path, args.video_vae_subfolder)
        logger.info(f"Loading Wan video VAE from {resolved_path}")
        video_vae = WanVAE(vae_path=resolved_path, device="cpu", dtype=vae_dtype)

        audio_vae_path = args.audio_vae if args.audio_vae is not None else vae_path
        logger.info(f"Loading MOVA audio VAE from {audio_vae_path}")
        audio_vae = load_audio_vae(
            audio_vae_path,
            subfolder=args.audio_vae_subfolder,
            device=torch.device("cpu"),
            dtype=vae_dtype,
            vae_type=args.audio_vae_type,
            model_spec=args.audio_vae_model_spec,
        )
        return MovaVAEBundle(video_vae, audio_vae)

    def load_transformer(
        self,
        accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        del attn_mode, split_attn, dit_weight_dtype

        model_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)

        logger.info(f"Loading MOVA visual tower from {dit_path}/{args.visual_subfolder}")
        video_dit = WanModel.from_pretrained(
            dit_path,
            subfolder=args.visual_subfolder,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )

        alternate_video_dit = None
        if args.alternate_visual_subfolder is not None:
            logger.info(f"Loading MOVA alternate visual tower from {dit_path}/{args.alternate_visual_subfolder}")
            alternate_video_dit = WanModel.from_pretrained(
                dit_path,
                subfolder=args.alternate_visual_subfolder,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
            )

        logger.info(f"Loading MOVA audio tower from {dit_path}/{args.audio_subfolder}")
        audio_dit = WanAudioModel.from_pretrained(
            dit_path,
            subfolder=args.audio_subfolder,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )

        logger.info(f"Loading MOVA bridge from {dit_path}/{args.bridge_subfolder}")
        dual_tower_bridge = DualTowerConditionalBridge.from_pretrained(
            dit_path,
            subfolder=args.bridge_subfolder,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )

        bundle = MovaModelBundle(
            video_dit=video_dit,
            audio_dit=audio_dit,
            dual_tower_bridge=dual_tower_bridge,
            alternate_video_dit=alternate_video_dit,
            condition_scale=args.condition_scale,
        )
        bundle.requires_grad_(False)

        target_device = torch.device(loading_device)
        bundle.to(device=target_device, dtype=model_dtype)

        if args.offload_inactive_dit and hasattr(bundle, "video_dit_2") and target_device.type != "cpu":
            logger.info("Offloading inactive alternate MOVA visual tower to CPU")
            bundle.video_dit_2.to("cpu")

        if args.fp8_scaled:
            logger.info("Optimizing MOVA modules to FP8")
            bundle.fp8_optimization(accelerator.device, move_to_device=loading_device == "cpu")

        return bundle

    def compile_transformer(self, args, transformer):
        target_blocks = [transformer.video_dit.blocks, transformer.audio_dit.blocks]
        if hasattr(transformer, "video_dit_2"):
            target_blocks.append(transformer.video_dit_2.blocks)
        target_blocks.append(list(transformer.dual_tower_bridge.audio_to_video_conditioners.values()))
        target_blocks.append(list(transformer.dual_tower_bridge.video_to_audio_conditioners.values()))
        return model_utils.compile_transformer(args, transformer, target_blocks, disable_linear=self.blocks_to_swap > 0)

    def _prepare_conditioning_latents(
        self,
        sample_parameter: dict,
        video_vae: WanVAE,
        width: int,
        height: int,
        frame_count: int,
        latents_shape: tuple[int, int, int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        conditioning_latents = torch.zeros(latents_shape, device=device, dtype=video_vae.dtype)
        image_path = sample_parameter.get("image_path")
        if image_path is None:
            return conditioning_latents

        image = Image.open(image_path).convert("RGB")
        image = resize_image_to_bucket(image, (width, height))
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(1).float()
        image = image / 127.5 - 1.0

        padding_frames = frame_count - 1
        conditioning_inputs = torch.cat(
            [
                image,
                torch.zeros(3, padding_frames, height, width, dtype=torch.float32),
            ],
            dim=1,
        )
        video_vae.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=video_vae.dtype), torch.no_grad():
            conditioning = video_vae.encode([conditioning_inputs.to(device=device, dtype=video_vae.dtype)])[0]
        conditioning = conditioning.unsqueeze(0).to(device=device, dtype=video_vae.dtype)
        return conditioning

    def _ensure_visual_towers(
        self,
        transformer: MovaModelBundle,
        active_towers: list[WanModel],
        device: torch.device,
        offload_inactive: bool,
    ) -> None:
        if not offload_inactive or not hasattr(transformer, "video_dit_2"):
            return

        active_ids = {id(tower) for tower in active_towers}
        moved = False
        for tower in transformer.iter_visual_towers():
            target_device = device if id(tower) in active_ids else torch.device("cpu")
            tower_device = next(tower.parameters()).device
            if tower_device != target_device:
                tower.to(target_device)
                moved = True

        if moved:
            clean_memory_on_device(device)

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
        image_path=None,
        control_video_path=None,
    ):
        del image_path, control_video_path

        model: MovaModelBundle = transformer
        vae_bundle: MovaVAEBundle = vae
        device = accelerator.device
        model_dtype = model.dtype if model.dtype.itemsize > 1 else dit_dtype
        video_fps = float(sample_parameter.get("video_fps", args.video_fps_default))

        visual_dit = model.video_dit
        video_channels = visual_dit.head.head.out_features // math.prod(visual_dit.patch_size)
        audio_channels = model.audio_dit.head.head.out_features // math.prod(model.audio_dit.patch_size)
        latent_video_length = (frame_count - 1) // self.vae_frame_stride + 1
        lat_h = height // 8
        lat_w = width // 8

        video_latents = torch.randn(
            (1, video_channels, latent_video_length, lat_h, lat_w),
            generator=generator,
            device=device,
            dtype=model_dtype,
        )

        audio_patch = model.audio_dit.patch_size[0]
        duration_sec = float(frame_count) / float(video_fps)
        audio_steps = max(1, int(round(duration_sec * float(model.dual_tower_bridge.audio_fps))))
        audio_length = max(audio_patch, audio_steps * audio_patch)
        audio_latents = torch.randn(
            (1, audio_channels, audio_length),
            generator=generator,
            device=device,
            dtype=model_dtype,
        )

        vae_bundle.to(device)
        conditioning_latents = self._prepare_conditioning_latents(
            sample_parameter,
            vae_bundle.video_vae,
            width,
            height,
            frame_count,
            tuple(video_latents.shape),
            device,
        ).to(device=device, dtype=model_dtype)

        prompt_embeds = sample_parameter["t5_embeds"].unsqueeze(0).to(device=device, dtype=model_dtype)
        negative_prompt_embeds = None
        if do_classifier_free_guidance and "negative_t5_embeds" in sample_parameter:
            negative_prompt_embeds = sample_parameter["negative_t5_embeds"].unsqueeze(0).to(device=device, dtype=model_dtype)

        video_scheduler = FlowMatchDiscreteScheduler(shift=discrete_flow_shift, reverse=True, solver="euler")
        audio_scheduler = FlowMatchDiscreteScheduler(shift=discrete_flow_shift, reverse=True, solver="euler")
        video_scheduler.set_timesteps(sample_steps, device=device)
        audio_scheduler.set_timesteps(sample_steps, device=device)

        cfg_strength = float(cfg_scale) if cfg_scale is not None else float(guidance_scale)
        fps_tensor = torch.tensor([video_fps], device=device, dtype=torch.float32)

        with torch.no_grad():
            for video_t, audio_t in zip(video_scheduler.timesteps, audio_scheduler.timesteps):
                t_value = float(video_t.item()) / 1000.0
                active_visual_dit = model.video_dit
                if hasattr(model, "video_dit_2") and t_value >= args.alternate_split_timestep:
                    active_visual_dit = model.video_dit_2
                self._ensure_visual_towers(model, [active_visual_dit], device, args.offload_inactive_dit)

                timestep = video_t.view(1).to(device=device, dtype=torch.float32)
                audio_timestep = audio_t.view(1).to(device=device, dtype=torch.float32)

                with accelerator.autocast():
                    video_pred, audio_pred = model.forward_model(
                        visual_dit=active_visual_dit,
                        video_latents=video_latents,
                        audio_latents=audio_latents,
                        conditioning_latents=conditioning_latents,
                        prompt_embeds=prompt_embeds,
                        timesteps=timestep,
                        audio_timesteps=audio_timestep,
                        video_fps=fps_tensor,
                    )
                    if negative_prompt_embeds is not None:
                        uncond_video_pred, uncond_audio_pred = model.forward_model(
                            visual_dit=active_visual_dit,
                            video_latents=video_latents,
                            audio_latents=audio_latents,
                            conditioning_latents=conditioning_latents,
                            prompt_embeds=negative_prompt_embeds,
                            timesteps=timestep,
                            audio_timesteps=audio_timestep,
                            video_fps=fps_tensor,
                        )
                        video_pred = uncond_video_pred + cfg_strength * (video_pred - uncond_video_pred)
                        audio_pred = uncond_audio_pred + cfg_strength * (audio_pred - uncond_audio_pred)

                video_latents = video_scheduler.step(video_pred, video_t, video_latents, return_dict=False)[0].to(dtype=model_dtype)
                audio_latents = audio_scheduler.step(audio_pred, audio_t, audio_latents, return_dict=False)[0].to(dtype=model_dtype)

        decoded_video = vae_bundle.decode_video(video_latents, frame_count)
        decoded_audio = vae_bundle.decode_audio(audio_latents, duration_sec=duration_sec)
        return {
            "video": decoded_video,
            "audio": decoded_audio,
            "audio_sample_rate": None if vae_bundle.audio_vae is None else vae_bundle.audio_vae.sample_rate,
            "video_fps": video_fps,
        }

    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
        sample_steps = sample_parameter.get("sample_steps", 20)
        width = sample_parameter.get("width", 256)
        height = sample_parameter.get("height", 256)
        frame_count = sample_parameter.get("frame_count", 81)
        guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
        discrete_flow_shift = sample_parameter.get("discrete_flow_shift", self.default_discrete_flow_shift)
        seed = sample_parameter.get("seed")
        prompt: str = sample_parameter.get("prompt", "")
        cfg_scale = sample_parameter.get("cfg_scale", None)
        negative_prompt = sample_parameter.get("negative_prompt", None)

        width = (width // 8) * 8
        height = (height // 8) * 8
        frame_count = (frame_count - 1) // self.vae_frame_stride * self.vae_frame_stride + 1

        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            torch.seed()
            if torch.cuda.is_available():
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
        )

        if not has_self_ref_orig_mod:
            transformer.train(was_train)

        if result is None or result.get("video") is None:
            logger.error("No video generated / 生成された動画がありません")
            return

        video = result["video"]
        audio = result.get("audio")
        audio_sample_rate = result.get("audio_sample_rate")
        video_fps = float(result.get("video_fps", args.video_fps_default))

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)
        save_path = (
            f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )
        video_path = os.path.join(save_dir, save_path) + ".mp4"
        save_video_with_audio(
            video,
            video_path,
            fps=video_fps,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            rescale=False,
        )

        wandb_tracker = None
        try:
            wandb_tracker = accelerator.get_tracker("wandb")
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
        except Exception:
            wandb = None

        if wandb_tracker is not None and wandb is not None:
            wandb_tracker.log({f"sample_{prompt_idx}": wandb.Video(video_path)}, step=steps)

        vae.to("cpu")
        clean_memory_on_device(device)

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator,
        transformer: MovaModelBundle,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        device = accelerator.device

        if "latents_audio" not in batch:
            raise ValueError("MOVA training requires audio latents in cache files (latents_audio)")

        if "t5" not in batch:
            raise ValueError("MOVA training requires text encoder cache files with varlen_t5 tensors")

        latents = latents.to(device=device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=device, dtype=network_dtype)
        audio_latents = batch["latents_audio"].to(device=device, dtype=network_dtype)

        conditioning_latents = batch.get("latents_image")
        if conditioning_latents is None:
            conditioning_latents = torch.zeros_like(latents)
        conditioning_latents = conditioning_latents.to(device=device, dtype=network_dtype)

        prompt_embeds = pad_prompt_embeds(batch["t5"], device=device, dtype=network_dtype)
        video_fps = batch.get("video_fps")
        if video_fps is None:
            video_fps = torch.full((latents.shape[0],), float(args.video_fps_default), device=device, dtype=torch.float32)
        else:
            video_fps = video_fps.to(device=device, dtype=torch.float32)

        t = ((timesteps.float() - 1.0) / 1000.0).clamp(0.0, 1.0)
        audio_noise = torch.randn_like(audio_latents)
        audio_sigma = t.view(-1, 1, 1).to(dtype=network_dtype)
        noisy_audio = (1.0 - audio_sigma) * audio_latents + audio_sigma * audio_noise

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            noisy_audio.requires_grad_(True)
            conditioning_latents.requires_grad_(True)
            prompt_embeds.requires_grad_(True)

        video_target = noise.to(device=device, dtype=network_dtype) - latents
        audio_target = audio_noise - audio_latents
        video_losses = torch.zeros((latents.shape[0],), device=device, dtype=torch.float32)
        audio_losses = torch.zeros_like(video_losses)

        tower_masks: list[tuple[WanModel, torch.Tensor]] = [(transformer.video_dit, torch.ones_like(t, dtype=torch.bool))]
        if hasattr(transformer, "video_dit_2"):
            low_mask = t < args.alternate_split_timestep
            high_mask = ~low_mask
            tower_masks = []
            if low_mask.any():
                tower_masks.append((transformer.video_dit, low_mask))
            if high_mask.any():
                tower_masks.append((transformer.video_dit_2, high_mask))

        active_towers = [visual_dit for visual_dit, mask in tower_masks if mask.any()]
        self._ensure_visual_towers(transformer, active_towers, device, args.offload_inactive_dit)

        for visual_dit, mask in tower_masks:
            if not mask.any():
                continue
            with accelerator.autocast():
                video_pred, audio_pred = transformer.forward_model(
                    visual_dit=visual_dit,
                    video_latents=noisy_model_input[mask],
                    audio_latents=noisy_audio[mask],
                    conditioning_latents=conditioning_latents[mask],
                    prompt_embeds=prompt_embeds[mask],
                    timesteps=timesteps[mask].to(device=device, dtype=torch.float32),
                    audio_timesteps=timesteps[mask].to(device=device, dtype=torch.float32),
                    video_fps=video_fps[mask],
                )

            video_loss = torch.nn.functional.mse_loss(video_pred.float(), video_target[mask].float(), reduction="none")
            audio_loss = torch.nn.functional.mse_loss(audio_pred.float(), audio_target[mask].float(), reduction="none")
            video_losses[mask] = video_loss.reshape(video_pred.shape[0], -1).mean(dim=1)
            audio_losses[mask] = audio_loss.reshape(audio_pred.shape[0], -1).mean(dim=1)

        total_loss = video_losses + float(args.audio_loss_weight) * audio_losses
        self.latest_video_loss = float(video_losses.mean().detach().item())
        self.latest_audio_loss = float(audio_losses.mean().detach().item())

        surrogate = torch.sqrt(torch.clamp(total_loss, min=1e-20)).view(-1, 1, 1, 1, 1).to(dtype=network_dtype)
        target = torch.zeros_like(surrogate)
        return surrogate, target


def mova_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="alias for --dit")
    parser.add_argument("--visual_subfolder", type=str, default="video_dit", help="primary visual tower subfolder")
    parser.add_argument("--alternate_visual_subfolder", type=str, default=None, help="secondary visual tower subfolder")
    parser.add_argument("--audio_subfolder", type=str, default="audio_dit", help="audio tower subfolder")
    parser.add_argument("--bridge_subfolder", type=str, default="dual_tower_bridge", help="bridge subfolder")
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
    parser.add_argument("--offload_inactive_dit", action="store_true", help="offload inactive alternate visual tower to CPU")
    parser.add_argument("--alternate_split_timestep", type=float, default=0.9, help="normalized timestep split for dual visual towers")
    parser.add_argument("--audio_loss_weight", type=float, default=1.0, help="weight for the audio loss term")
    parser.add_argument("--condition_scale", type=float, default=None, help="override cross-modal condition scale")
    parser.add_argument("--video_fps_default", type=float, default=16.0, help="fallback fps when cache does not contain video_fps")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for MOVA modules")
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder path or HF repo for sample generation")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer path or HF repo for sample generation")
    parser.add_argument("--text_encoder_subfolder", type=str, default=None, help="optional text encoder subfolder")
    parser.add_argument("--tokenizer_subfolder", type=str, default=None, help="optional tokenizer subfolder")
    parser.add_argument("--text_encoder_max_length", type=int, default=512, help="maximum prompt length for sample generation")
    parser.add_argument("--trust_remote_code", action="store_true", help="allow trust_remote_code when loading tokenizer/text encoder")
    parser.add_argument(
        "--lora_scope",
        type=str,
        default="official",
        choices=["official", "attention_plus_image", "all_linear"],
        help="LoRA target scope for MOVA",
    )
    parser.set_defaults(
        network_module="musubi_tuner.networks.lora_mova",
        network_dim=32,
        network_alpha=32,
        dit_dtype="bfloat16",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = mova_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.pretrained_model_path is not None and args.dit is None:
        args.dit = args.pretrained_model_path

    if args.fp8_base and not args.fp8_scaled:
        args.fp8_scaled = True
    if args.fp8_scaled and not args.fp8_base:
        args.fp8_base = True

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = MovaNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
