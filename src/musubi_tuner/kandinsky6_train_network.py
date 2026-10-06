import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from musubi_tuner.dataset.architectures import ARCHITECTURE_KANDINSKY6, ARCHITECTURE_KANDINSKY6_FULL
from musubi_tuner.dataset.audio_utils import AudioSpec
from musubi_tuner.hv_train_network import (
    DiTOutput,
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
    should_sample_images,
)
from musubi_tuner.kandinsky6 import PiFlowDiffusionTransformer3D, load_dit, load_dit_convrot_int8, load_dit_fp8
from musubi_tuner.training.audio_loss import add_audio_train_args, effective_audio_loss_weights
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)


def kandinsky6_samples_per_crop(frame_count: int) -> int:
    """Waveform samples on the released 24 fps / 44.1 kHz AV timeline."""
    latent_frames = math.ceil(frame_count * 44100 / 24 / 1024)
    return latent_frames * 1024


KANDINSKY6_AUDIO_SPEC = AudioSpec(
    sample_rate=44100,
    channels=1,
    samples_per_crop=kandinsky6_samples_per_crop,
)


def sample_latent_frame_count(pixel_frames: int) -> tuple[int, int]:
    """Snap a Musubi pixel-frame request to the VAE grid and return (pixel, latent)."""
    if pixel_frames < 1:
        raise ValueError("sample frame_count must be positive")
    pixel_frames = 1 + ((pixel_frames - 1) // 4) * 4
    return pixel_frames, (pixel_frames - 1) // 4 + 1


def sample_canvas_size(width: int, height: int) -> tuple[int, int]:
    """Snap a sample canvas to K6's 16-pixel grid and reject empty canvases."""
    width = int(width) // 16 * 16
    height = int(height) // 16 * 16
    if width < 16 or height < 16:
        raise ValueError(f"sample width and height must each be at least 16 pixels, got {width}x{height} after alignment")
    return width, height


def park_dit_for_sampling(transformer, device: torch.device) -> None:
    """Park non-swapped modules without invalidating offloader-owned block storage."""
    copier = getattr(getattr(transformer, "offloader", None), "copier", None)
    if copier is not None:
        copier.sync()
    if hasattr(transformer, "move_to_device_except_swap_blocks"):
        transformer.move_to_device_except_swap_blocks(torch.device("cpu"))
    else:
        transformer.to("cpu")
    clean_memory_on_device(device)


def restore_dit_after_sampling_park(transformer, device: torch.device) -> None:
    if hasattr(transformer, "move_to_device_except_swap_blocks"):
        transformer.move_to_device_except_swap_blocks(device)
    else:
        transformer.to(device)
    if hasattr(transformer, "prepare_block_swap_before_forward"):
        transformer.prepare_block_swap_before_forward()


@dataclass
class _SamplingResources:
    video_vae: object
    audio_vae: object
    vocoder: object
    text_embedder: object

    def to(self, device):
        for module in (self.video_vae, self.audio_vae, self.vocoder, self.text_embedder):
            if hasattr(module, "to"):
                module.to(device)
        return self


def shifted_flow_sigma(uniform: torch.Tensor, shift: float) -> torch.Tensor:
    """Map a uniform flow time through Kandinsky's rational schedule shift."""
    if shift <= 0:
        raise ValueError("scheduler_scale must be positive")
    return shift * uniform / (1.0 + (shift - 1.0) * uniform)


def pad_text_batch(rows, masks, device: torch.device, dtype: torch.dtype):
    """Pad collated variable-length text rows and return a boolean validity mask."""
    if isinstance(rows, torch.Tensor):
        rows = list(rows)
    if isinstance(masks, torch.Tensor):
        masks = list(masks)
    if not rows or len(rows) != len(masks):
        raise ValueError("Kandinsky 6 text embeddings and masks must contain one entry per item")
    max_length = max(row.shape[0] for row in rows)
    width = rows[0].shape[-1]
    padded = torch.zeros((len(rows), max_length, width), device=device, dtype=dtype)
    attention_mask = torch.zeros((len(rows), max_length), device=device, dtype=torch.bool)
    for index, (row, mask) in enumerate(zip(rows, masks, strict=True)):
        if row.ndim != 2 or row.shape[1] != width or mask.shape != (row.shape[0],):
            raise ValueError("Invalid Kandinsky 6 variable-length text cache entry")
        length = row.shape[0]
        padded[index, :length] = row.to(device=device, dtype=dtype)
        attention_mask[index, :length] = mask.to(device=device, dtype=torch.bool)
    return padded, attention_mask


def build_ti2av_video_input(noisy_video: torch.Tensor, image_latent: torch.Tensor | None, visual_cond: bool):
    """Build the upstream 33-channel visual input, appending a clean TI2AV reference frame."""
    batch, frames, height, width, channels = noisy_video.shape
    token_types = None
    if image_latent is not None:
        if image_latent.shape != (batch, 1, height, width, channels):
            raise ValueError(f"TI2AV image latent must be {(batch, 1, height, width, channels)}, got {tuple(image_latent.shape)}")
        noisy_video = torch.cat([noisy_video, image_latent], dim=1)
        token_types = torch.cat(
            [
                torch.zeros((batch, frames), device=noisy_video.device, dtype=torch.long),
                torch.ones((batch, 1), device=noisy_video.device, dtype=torch.long),
            ],
            dim=1,
        )
    if not visual_cond:
        return noisy_video, token_types
    condition = torch.zeros_like(noisy_video)
    mask = torch.zeros((*noisy_video.shape[:-1], 1), device=noisy_video.device, dtype=noisy_video.dtype)
    if image_latent is not None:
        mask[:, -1] = 1
    return torch.cat([noisy_video, condition, mask], dim=-1), token_types


class Kandinsky6NetworkTrainer(NetworkTrainer):
    audio_spec = KANDINSKY6_AUDIO_SPEC

    def __init__(self):
        super().__init__()
        self._task = "t2av"
        self._scheduler_scale = 1.0
        self._visual_rope_scale = (1.0, 2.0, 2.0)

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_KANDINSKY6

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_KANDINSKY6_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self._task = args.task
        self._scheduler_scale = float(args.scheduler_scale)
        self._visual_rope_scale = tuple(float(value) for value in args.visual_rope_scale)
        self.default_guidance_scale = float(args.guidance_scale)
        self.default_discrete_flow_shift = 1.0
        if args.mixed_precision != "bf16":
            raise ValueError("Kandinsky 6 training requires --mixed_precision bf16")
        if args.flash_attn or args.sage_attn or args.xformers or args.flash3 or args.split_attn:
            raise ValueError("Kandinsky 6 training currently supports SDPA attention only")
        if args.timestep_sampling != "uniform" or args.weighting_scheme != "none" or args.discrete_flow_shift != 1.0:
            raise ValueError(
                "Kandinsky 6 uses its own shifted uniform flow schedule; timestep_sampling must be uniform, "
                "weighting_scheme none, and discrete_flow_shift 1"
            )
        if args.dit_dtype not in (None, "bfloat16"):
            raise ValueError("Kandinsky 6 transformer weights must use bfloat16")
        if args.convrot_int8 and args.fp8_base:
            raise ValueError("--convrot_int8 and --fp8_base are mutually exclusive")
        if args.base_weights and (args.convrot_int8 or args.fp8_base):
            raise ValueError("--base_weights cannot be merged into a ConvRot INT8 or FP8 base; use a BF16 base")
        if args.img_in_txt_in_offloading:
            raise ValueError("--img_in_txt_in_offloading is not supported by Kandinsky 6; use --blocks_to_swap")

    @property
    def i2v_training(self) -> bool:
        return self._task == "ti2av"

    @property
    def control_training(self) -> bool:
        return False

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        del args, vae_dtype, vae_path
        return None

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
        del attn_mode, split_attn
        dtype = dit_weight_dtype or torch.bfloat16
        if args.convrot_int8:
            transformer = load_dit_convrot_int8(
                dit_path,
                model_variant=args.model_variant,
                device=loading_device,
                quant_device=accelerator.device,
                bwd_mode=args.convrot_int8_bwd,
                disable_numpy_memmap=args.disable_numpy_memmap,
            )
        elif args.fp8_base:
            transformer = load_dit_fp8(
                dit_path,
                model_variant=args.model_variant,
                device=loading_device,
                quant_device=accelerator.device,
                disable_numpy_memmap=args.disable_numpy_memmap,
            )
        else:
            transformer = load_dit(dit_path, model_variant=args.model_variant, dtype=dtype, device=loading_device)
        if isinstance(transformer, PiFlowDiffusionTransformer3D):
            raise ValueError("PiFlow distilled Kandinsky 6 checkpoints use grid outputs and are not trainable yet")
        if not transformer.is_multimodal:
            raise ValueError("Kandinsky 6 joint training requires a multimodal Lite or Pro checkpoint")
        return transformer

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(
            args,
            transformer,
            [
                transformer.video_text_transformer_blocks,
                transformer.audio_text_transformer_blocks,
                transformer.visual_transformer_blocks,
            ],
            disable_linear=bool(self.blocks_to_swap) or bool(args.convrot_int8) or bool(args.fp8_base),
        )

    def scale_shift_latents(self, latents):
        return latents

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        del args, accelerator
        return load_prompts(sample_prompts)

    def prepare_sampling(self, args, accelerator, vae_dtype):
        del vae_dtype
        if not args.sample_prompts:
            return None, None
        if args.quantized_qwen:
            raise ValueError(
                "--quantized_qwen is supported by the standalone text-cache command, but not by in-training "
                "sampling: bitsandbytes NF4 cannot be parked on CPU between samples. Use the unquantized text "
                "encoder for --sample_prompts."
            )
        required = {
            "--vae": args.vae,
            "--audio_vae": args.audio_vae,
            "--vocoder": args.vocoder,
            "--text_encoder_qwen": args.text_encoder_qwen,
            "--text_encoder_clip": args.text_encoder_clip,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Kandinsky 6 sampling requires {', '.join(missing)}")
        try:
            from musubi_tuner.kandinsky6.runtime.core.components.text_embedder import Kandinsky6TextEmbedder
            from musubi_tuner.kandinsky6.runtime.core.components.vae_audio import build_audio_vae, build_vocoder
            from musubi_tuner.kandinsky6.runtime.core.components.vae_video import build_vae
        except ImportError as exc:
            raise RuntimeError("Kandinsky 6 sampling dependencies are missing. Install Musubi with the kandinsky6 extra.") from exc
        # Keep sampling resources on CPU between samples. Each encoder is staged briefly;
        # decoding stays on CPU so the training DiT and its optimizer can remain resident.
        device = torch.device("cpu")
        resources = _SamplingResources(
            video_vae=build_vae(args.vae, device=device),
            audio_vae=build_audio_vae(
                tod_vae_ckpt=args.audio_vae,
                mode="44k",
                need_vae_encoder=True,
                need_vae_decoder=True,
                scaling_factor=args.audio_vae_scaling_factor,
                device=device,
            ),
            vocoder=build_vocoder(ckpt=args.vocoder, device=device),
            text_embedder=Kandinsky6TextEmbedder(
                args.text_encoder_qwen,
                args.text_encoder_clip,
                max_length=args.max_length,
                device=device,
                quantized_qwen=args.quantized_qwen,
                text_token_padding=False,
            ),
        )
        for module in (resources.video_vae, resources.audio_vae, resources.vocoder, resources.text_embedder):
            if hasattr(module, "requires_grad_"):
                module.requires_grad_(False)
            if hasattr(module, "eval"):
                module.eval()
        return self.process_sample_prompts(args, accelerator, args.sample_prompts), resources

    @torch.no_grad()
    def sample_images(self, accelerator, args, epoch, steps, resources, transformer, sample_parameters, dit_dtype):
        del dit_dtype
        if not should_sample_images(args, steps, epoch) or not sample_parameters:
            return
        from musubi_tuner.kandinsky6.runtime.core.algo.denoise_loop import denoise_loop
        from musubi_tuner.kandinsky6.runtime.core.algo.mux import mux_video_audio
        from musubi_tuner.kandinsky6.runtime.core.algo.postprocess_audio import postprocess_audio
        from musubi_tuner.kandinsky6.runtime.core.algo.postprocess_video import postprocess_video
        from musubi_tuner.kandinsky6.runtime.core.algo.prepare_latents import (
            append_i2va_tail_condition,
            audio_latent_duration,
            encode_i2va_first_frame,
            prepare_audio_latents,
            prepare_video_latents,
        )
        from musubi_tuner.kandinsky6.runtime.core.algo.prepare_ropes import compute_rope1d, compute_visual_rope
        from musubi_tuner.kandinsky6.runtime.core.types import LatentBundle

        transformer = accelerator.unwrap_model(transformer)
        was_training = transformer.training
        transformer.eval()
        transformer.switch_block_swap_for_inference()

        def offload_dit() -> None:
            park_dit_for_sampling(transformer, accelerator.device)

        def restore_dit() -> None:
            restore_dit_after_sampling_park(transformer, accelerator.device)

        offload_dit()
        save_dir = Path(args.output_dir) / "sample"
        save_dir.mkdir(parents=True, exist_ok=True)
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            for index, sample in enumerate(sample_parameters):
                prompt = sample.get("prompt", "")
                negative = sample.get("negative_prompt", "low quality, bad quality")
                width, height = sample_canvas_size(sample.get("width", 512), sample.get("height", 512))
                pixel_frames = int(sample.get("frame_count", 121))
                pixel_frames, video_frames = sample_latent_frame_count(pixel_frames)
                sample_steps = int(sample.get("sample_steps", 20))
                guidance = float(sample.get("guidance_scale", self.default_guidance_scale))
                seed = int(sample.get("seed", 42))
                if hasattr(transformer, "clear_text_proj_cache"):
                    transformer.clear_text_proj_cache()
                resources.text_embedder.to(accelerator.device)
                text, text_cu, attention_mask = resources.text_embedder.encode([prompt])
                null, null_cu, null_attention_mask = resources.text_embedder.encode([negative])
                text = {key: value.to(accelerator.device, torch.bfloat16) for key, value in text.items()}
                null = {key: value.to(accelerator.device, torch.bfloat16) for key, value in null.items()}
                text_length = text["text_embeds"].shape[-2]
                null_length = null["text_embeds"].shape[-2]
                if attention_mask is not None:
                    attention_mask = attention_mask.to(accelerator.device)
                if null_attention_mask is not None:
                    null_attention_mask = null_attention_mask.to(accelerator.device)
                resources.text_embedder.to("cpu")
                clean_memory_on_device(accelerator.device)

                latent_h, latent_w = height // 8, width // 8
                bundle = prepare_video_latents(
                    1, video_frames, latent_h, latent_w, transformer.in_visual_dim, seed, accelerator.device
                )
                bundle = LatentBundle(
                    video=bundle.video.reshape(1, video_frames, latent_h, latent_w, transformer.in_visual_dim),
                    audio=None,
                    video_cu_seqlens=bundle.video_cu_seqlens,
                    audio_cu_seqlens=None,
                )
                first_frames = None
                token_types = None
                generated_mask = None
                image_path = sample.get("image_path")
                if image_path:
                    resources.video_vae.to(accelerator.device)
                    first_frames, _, _ = encode_i2va_first_frame(
                        image_path, resources.video_vae, accelerator.device, height=height, width=width
                    )
                    resources.video_vae.to("cpu")
                    clean_memory_on_device(accelerator.device)
                    video, token_types, generated_mask = append_i2va_tail_condition(
                        bundle.video, first_frames, batch_size=1, video_duration=video_frames
                    )
                    bundle = LatentBundle(
                        video=video,
                        audio=None,
                        video_cu_seqlens=torch.tensor([0, video_frames + 1], device=accelerator.device, dtype=torch.int32),
                        audio_cu_seqlens=None,
                    )
                audio_frames = audio_latent_duration(video_frames, fps=24, audio_fps=44100, downsample_factor=1024)
                bundle = prepare_audio_latents(bundle, audio_frames, transformer.in_audio_dim, seed, accelerator.device)
                bundle = LatentBundle(
                    video=bundle.video,
                    audio=bundle.audio.reshape(1, audio_frames, transformer.in_audio_dim),
                    video_cu_seqlens=bundle.video_cu_seqlens,
                    audio_cu_seqlens=bundle.audio_cu_seqlens,
                )
                restore_dit()
                visual_rope = compute_visual_rope(
                    transformer.visual_rope,
                    (
                        video_frames // transformer.patch_size[0],
                        latent_h // transformer.patch_size[1],
                        latent_w // transformer.patch_size[2],
                    ),
                    self._visual_rope_scale,
                )
                if image_path:
                    visual_rope = torch.cat([visual_rope, visual_rope[:1]], dim=0)
                audio_rope = compute_rope1d(transformer.audio_rope, audio_frames)
                text_rope = [
                    compute_rope1d(transformer.video_text_rope, text_length),
                    compute_rope1d(transformer.audio_text_rope, text_length),
                ]
                null_rope = [
                    compute_rope1d(transformer.video_text_rope, null_length),
                    compute_rope1d(transformer.audio_text_rope, null_length),
                ]
                result = denoise_loop(
                    bundle,
                    transformer,
                    text,
                    null,
                    visual_rope,
                    audio_rope,
                    text_rope,
                    null_rope,
                    sample_steps,
                    guidance,
                    self._scheduler_scale,
                    first_frames=first_frames,
                    visual_cond_scheme="tail_cond_first_frame" if image_path else "pretrain",
                    attention_mask=attention_mask,
                    null_attention_mask=null_attention_mask,
                    visual_token_type_ids=token_types,
                    scale_factor=self._visual_rope_scale,
                )
                if generated_mask is not None:
                    result = LatentBundle(
                        video=result.video[:, generated_mask[0]],
                        audio=result.audio,
                        video_cu_seqlens=torch.tensor([0, video_frames], device=accelerator.device, dtype=torch.int32),
                        audio_cu_seqlens=result.audio_cu_seqlens,
                    )
                # The decoders are large enough that co-residency with the DiT defeats
                # training-time sampling on consumer GPUs. Offload the unwrapped DiT,
                # decode one modality at a time, then restore its block-swap placement.
                offload_dit()
                resources.video_vae.to(accelerator.device)
                frames = postprocess_video(result, resources.video_vae, bs=1)
                resources.video_vae.to("cpu")
                resources.audio_vae.to(accelerator.device)
                resources.vocoder.to(accelerator.device)
                audio = postprocess_audio(result, resources.audio_vae, resources.vocoder)
                resources.audio_vae.to("cpu")
                resources.vocoder.to("cpu")
                clean_memory_on_device(accelerator.device)
                suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
                stamp = time.strftime("%Y%m%d%H%M%S")
                output = save_dir / f"{args.output_name or 'sample'}_{suffix}_{index:02d}_{stamp}_{seed}.mp4"
                mux_video_audio(frames[0], audio[0], output, fps=24, audio_sample_rate=44100)
                logger.info("Saved Kandinsky 6 AV sample to %s", output)
        finally:
            resources.to("cpu")
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            restore_dit()
            transformer.switch_block_swap_for_training()
            transformer.train(was_training)
            clean_memory_on_device(accelerator.device)

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
        del args, noise_scheduler, dtype
        uniform = (
            torch.as_tensor(timesteps, device=device) if timesteps is not None else torch.rand(latents.shape[0], device=device)
        )
        sigma = shifted_flow_sigma(uniform, self._scheduler_scale)
        sigma_view = sigma.view(-1, 1, 1, 1, 1)
        noisy = ((1.0 - sigma_view) * latents.float() + sigma_view * noise.float()).to(latents.dtype)
        return noisy, sigma * 1000.0

    def _ropes(self, transformer, video, audio, text_length):
        _, frames, height, width, _ = video.shape
        patch_t, patch_h, patch_w = transformer.patch_size
        visual_shape = (frames // patch_t, height // patch_h, width // patch_w)
        visual_pos = [torch.arange(size, device=video.device) for size in visual_shape]
        visual_rope = transformer.visual_rope(visual_shape, visual_pos, self._visual_rope_scale)
        audio_rope = transformer.audio_rope(torch.arange(audio.shape[1], device=audio.device))
        text_positions = torch.arange(text_length, device=video.device)
        text_rope = [transformer.video_text_rope(text_positions), transformer.audio_text_rope(text_positions)]
        return visual_rope, audio_rope, text_rope

    def call_dit(
        self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
    ) -> DiTOutput:
        del batch
        audio_latents = kwargs.pop("audio_latents")
        audio_noise = kwargs.pop("audio_noise")
        noisy_audio = kwargs.pop("noisy_audio")
        text_embeds = kwargs.pop("text_embeds")
        pooled_embed = kwargs.pop("pooled_embed")
        attention_mask = kwargs.pop("attention_mask")
        image_latent = kwargs.pop("image_latent")
        audio_loss_weights = kwargs.pop("audio_loss_weights")
        if kwargs:
            raise TypeError(f"Unexpected Kandinsky 6 call_dit arguments: {sorted(kwargs)}")

        generated_video = noisy_model_input.to(accelerator.device)
        model_video, token_types = build_ti2av_video_input(generated_video, image_latent, transformer.visual_cond)
        visual_rope, audio_rope, text_rope = self._ropes(transformer, generated_video, noisy_audio, text_embeds.shape[1])
        if image_latent is not None:
            # The upstream tail reference reuses generated frame zero's temporal
            # position; its token type distinguishes it from the generated frame.
            visual_rope = torch.cat([visual_rope, visual_rope[:1]], dim=0)
        if args.gradient_checkpointing:
            model_video.requires_grad_(True)
            noisy_audio.requires_grad_(True)
            text_embeds.requires_grad_(True)
        with accelerator.autocast():
            video_pred, audio_pred = transformer(
                x_video=model_video.to(dtype=network_dtype),
                x_audio=noisy_audio.to(device=accelerator.device, dtype=network_dtype),
                text_embed=text_embeds,
                pooled_text_embed=pooled_embed,
                time=timesteps,
                visual_rope=visual_rope,
                audio_rope=audio_rope,
                text_rope=text_rope,
                attention_mask=attention_mask,
                visual_token_type_ids=token_types,
            )
        video_pred = video_pred[:, : latents.shape[1]]
        return DiTOutput(
            pred=video_pred,
            target=noise - latents,
            extra={"audio_pred": audio_pred, "audio_target": audio_noise - audio_latents, "audio_loss_weights": audio_loss_weights},
        )

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
        sample_resources,
        global_step,
    ):
        del network, sample_resources
        # Cache layout [B,C,T,H,W]/[B,C,A] -> upstream [B,T,H,W,C]/[B,A,C].
        video = latents.permute(0, 2, 3, 4, 1).contiguous()
        video_noise = noise.permute(0, 2, 3, 4, 1).contiguous()
        noisy_video, timesteps = self.get_noisy_model_input_and_timesteps(
            args, video_noise, video, batch.get("timesteps"), noise_scheduler, video.device, dit_dtype
        )
        sigma = (timesteps / 1000.0).view(-1, 1, 1)
        audio = batch["latents_audio"].to(video.device).transpose(1, 2).contiguous()
        audio_noise = torch.randn_like(audio)
        noisy_audio = ((1.0 - sigma) * audio.float() + sigma * audio_noise.float()).to(audio.dtype)
        text, text_mask = pad_text_batch(batch["text_embeds"], batch["attention_mask"], video.device, network_dtype)
        pooled = batch["pooled_embed"].to(device=video.device, dtype=network_dtype)
        image = batch.get("latents_image")
        if self.i2v_training:
            if image is None:
                raise ValueError("--task ti2av requires latents_image in every cache item")
            image = image.to(video.device).permute(0, 2, 3, 4, 1).contiguous()
        elif image is not None:
            raise ValueError("T2AV cache unexpectedly contains TI2AV image conditioning")
        audio_weights = effective_audio_loss_weights(batch["audio_present"], args).to(video.device)
        output = self.call_dit(
            args,
            accelerator,
            transformer,
            video,
            batch,
            video_noise,
            noisy_video,
            timesteps,
            network_dtype,
            audio_latents=audio,
            audio_noise=audio_noise,
            noisy_audio=noisy_audio,
            text_embeds=text,
            pooled_embed=pooled,
            attention_mask=text_mask,
            image_latent=image,
            audio_loss_weights=audio_weights,
        )
        return self.compute_loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step)

    def compute_loss(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step):
        del args, timesteps, noise_scheduler, dit_dtype, global_step
        video_loss = F.mse_loss(output.pred.float(), output.target.float())
        per_item_audio = (output.extra["audio_pred"].float() - output.extra["audio_target"].float()).pow(2)
        per_item_audio = per_item_audio.flatten(1).mean(1)
        weights = output.extra["audio_loss_weights"].to(per_item_audio)
        supervised = (weights > 0).sum().clamp_min(1)
        audio_loss = (per_item_audio * weights).sum() / supervised
        total = video_loss + audio_loss
        return total, {"loss/video": float(video_loss.detach()), "loss/audio": float(audio_loss.detach())}


def kandinsky6_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(
        timestep_sampling="uniform",
        weighting_scheme="none",
        discrete_flow_shift=1.0,
        network_module="networks.lora_kandinsky6",
        sdpa=True,
        mixed_precision="bf16",
        guidance_scale=4.0,
    )
    parser.add_argument("--task", choices=("t2av", "ti2av"), default="t2av")
    parser.add_argument("--model_variant", choices=("auto", "lite", "pro"), default="auto")
    parser.add_argument("--scheduler_scale", type=float, default=1.0)
    parser.add_argument("--visual_rope_scale", type=float, nargs=3, default=(1.0, 2.0, 2.0), metavar=("T", "H", "W"))
    parser.add_argument("--convrot_int8", action="store_true", help="stream and quantize frozen DiT Linear weights to ConvRot INT8")
    parser.add_argument("--convrot_int8_bwd", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--audio_vae", help="Kandinsky 6 MMAudio VAE checkpoint (required for sampling)")
    parser.add_argument("--audio_vae_scaling_factor", type=float, default=0.5302)
    parser.add_argument("--vocoder", help="Kandinsky 6 BigVGAN checkpoint (required for sampling)")
    parser.add_argument("--text_encoder_qwen", help="Qwen2.5-VL text encoder (required for sampling)")
    parser.add_argument("--text_encoder_clip", help="CLIP text encoder (required for sampling)")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--quantized_qwen", action="store_true")
    parser.add_argument("--upstream", help=argparse.SUPPRESS)  # deprecated compatibility no-op
    parser.add_argument("--sr_upstream", help=argparse.SUPPRESS)  # deprecated compatibility no-op
    add_audio_train_args(parser)
    return parser


def main() -> None:
    parser = kandinsky6_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    if not hasattr(args, "dit_dtype"):
        args.dit_dtype = None
    if args.mixed_precision is None:
        args.mixed_precision = "bf16"
    args.dit_dtype = "bfloat16" if args.dit_dtype is None else args.dit_dtype
    args.fp8_fast = getattr(args, "fp8_fast", False)
    if args.fp8_base:
        args.fp8_scaled = True
    Kandinsky6NetworkTrainer().train(args)


if __name__ == "__main__":
    main()
