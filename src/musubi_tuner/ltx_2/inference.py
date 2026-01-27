
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from musubi_tuner.hv_generate_video import save_images_grid, save_videos_grid
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)

STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


def _load_video_encoder(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from musubi_tuner.ltx_2.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

    encoder = SingleGPUModelBuilder(
        model_path=checkpoint_path,
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype)
    encoder.eval()
    return encoder


def _load_upsampler(upsampler_path: str, device: torch.device, dtype: torch.dtype):
    from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from musubi_tuner.ltx_2.model.upsampler.model_configurator import LatentUpsamplerConfigurator

    upsampler = SingleGPUModelBuilder(
        model_path=upsampler_path,
        model_class_configurator=LatentUpsamplerConfigurator,
    ).build(device=device, dtype=dtype)
    upsampler.eval()
    return upsampler


def _apply_distilled_lora(
    transformer: torch.nn.Module,
    lora_path: str,
    strength: float,
):
    from safetensors.torch import load_file
    from musubi_tuner.networks import lora_ltx2

    lora_sd = load_file(lora_path)
    lora_net = lora_ltx2.create_arch_network_from_weights(
        strength,
        lora_sd,
        unet=transformer,
        for_inference=True,
    )
    lora_net.set_multiplier(strength)
    lora_net.backup_weights()
    lora_net.pre_calculation()
    return lora_net

def sample_image_inference(
    trainer,
    accelerator,
    args,
    transformer,
    dit_dtype: torch.dtype,
    vae,
    save_dir: str,
    sample_parameter: Dict,
    epoch,
    steps,
):
    """LTX-2-specific sampling with proper frame/size rounding."""
    lora_count = trainer._ensure_lora_enabled_for_sampling(transformer)
    if lora_count:
        logger.info("Sampling: LoRA modules active in transformer: %s", lora_count)
        lora_stats = trainer._get_lora_norm_samples(transformer)
        for stat in lora_stats:
            logger.info("Sampling LoRA norm: %s", stat)
    else:
        logger.warning("Sampling: no LoRA modules detected on transformer")

    loaded_vae = False
    if vae is None or getattr(vae, "_deferred", False):
        vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
        vae = trainer._load_vae_impl(args, vae_dtype=vae_dtype, vae_path=args.vae)
        loaded_vae = True

    audio_decoder = None
    vocoder = None
    loaded_audio = False
    disable_audio_preview = bool(getattr(args, "sample_disable_audio", False))
    use_audio_subprocess = os.name == "nt"
    audio_only_preview = bool(getattr(args, "sample_audio_only", False))
    if audio_only_preview and getattr(args, "ltx_mode", "v") not in {"av", "a"}:
        raise ValueError("--sample_audio_only requires --ltx2_mode av or a")
    enable_audio_preview = (trainer._audio_video or audio_only_preview) and not disable_audio_preview
    if enable_audio_preview and getattr(args, "ltx_mode", "v") in {"av", "a"}:
        if use_audio_subprocess:
            logger.info("Sampling audio: using subprocess decoder on Windows; skipping in-process load")
        else:
            # Align with LTX-v2 audio decode path (bf16 by default).
            audio_dtype = torch.bfloat16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
            audio_device = accelerator.device
            try:
                if accelerator.device.type == "cuda" and next(transformer.parameters()).device.type == "cuda":
                    logger.info("Sampling audio: offloading transformer to CPU before audio decoder/vocoder load")
                    if hasattr(transformer, "move_to_device_except_swap_blocks"):
                        transformer.move_to_device_except_swap_blocks("cpu")
                    else:
                        transformer.to("cpu")
                    clean_memory_on_device(accelerator.device)
                audio_decoder, vocoder = trainer._load_audio_components(
                    args,
                    audio_dtype=audio_dtype,
                    checkpoint_path=args.ltx2_checkpoint,
                    device=audio_device,
                )
                loaded_audio = True
            except Exception as exc:
                logger.warning("Sampling audio decoder load failed; continuing without audio preview: %s", exc)
                audio_decoder, vocoder = None, None
                loaded_audio = False
            finally:
                if accelerator.device.type == "cuda" and next(transformer.parameters()).device.type != accelerator.device:
                    logger.info("Sampling audio: restoring transformer to GPU after audio decoder/vocoder load")
                    if hasattr(transformer, "move_to_device_except_swap_blocks"):
                        transformer.move_to_device_except_swap_blocks(accelerator.device)
                    else:
                        transformer.to(accelerator.device)
                    clean_memory_on_device(accelerator.device)

    sample_steps = sample_parameter.get("sample_steps", 20)
    width = sample_parameter.get("width", 768)
    height = sample_parameter.get("height", 512)
    frame_count = sample_parameter.get("frame_count", 45)
    guidance_scale = sample_parameter.get("guidance_scale", trainer.default_guidance_scale)
    discrete_flow_shift = sample_parameter.get("discrete_flow_shift", 5.0)
    seed = sample_parameter.get("seed")
    prompt: str = sample_parameter.get("prompt", "")
    cfg_scale = sample_parameter.get("cfg_scale", None)
    negative_prompt = sample_parameter.get("negative_prompt", None)

    spatial_factor = int(getattr(vae, "spatial_downsample_factor", 32))
    temporal_factor = int(getattr(vae, "temporal_downsample_factor", 8))
    width = (width // spatial_factor) * spatial_factor
    height = (height // spatial_factor) * spatial_factor
    frame_count = (frame_count - 1) // temporal_factor * temporal_factor + 1

    loaded_text_encoder = False
    if sample_parameter.get("prompt_embeds") is None:
        text_encoder_dtype = trainer._build_text_encoder(args, accelerator)
        prompt_embeds, prompt_mask = trainer._encode_prompt_text(accelerator, prompt, text_encoder_dtype)
        sample_parameter["prompt_embeds"] = prompt_embeds
        sample_parameter["prompt_attention_mask"] = prompt_mask
        if negative_prompt:
            neg_embeds, neg_mask = trainer._encode_prompt_text(
                accelerator, negative_prompt, text_encoder_dtype
            )
            sample_parameter["negative_prompt_embeds"] = neg_embeds
            sample_parameter["negative_prompt_attention_mask"] = neg_mask
        loaded_text_encoder = True
    elif negative_prompt and sample_parameter.get("negative_prompt_embeds") is None:
        text_encoder_dtype = trainer._build_text_encoder(args, accelerator)
        neg_embeds, neg_mask = trainer._encode_prompt_text(
            accelerator, negative_prompt, text_encoder_dtype
        )
        sample_parameter["negative_prompt_embeds"] = neg_embeds
        sample_parameter["negative_prompt_attention_mask"] = neg_mask
        loaded_text_encoder = True

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

    do_classifier_free_guidance = False
    if negative_prompt is not None:
        do_classifier_free_guidance = True
        logger.info(f"negative prompt: {negative_prompt}")
        logger.info(f"cfg scale: {cfg_scale}")

    has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
    was_train = transformer.training if not has_self_ref_orig_mod else True
    if not has_self_ref_orig_mod:
        transformer.eval()

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    prompt_idx = sample_parameter.get("enum", 0)
    save_path = (
        f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
    )
    wav_path = os.path.join(save_dir, save_path) + ".wav"

    use_two_stage = bool(getattr(args, "sample_two_stage", False))
    if use_two_stage and audio_only_preview:
        logger.warning("Sampling two-stage does not support audio-only preview; falling back to single-stage.")
        use_two_stage = False

    if use_two_stage:
        video, audio_waveform = do_two_stage_inference(
            trainer,
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
            audio_decoder=audio_decoder,
            vocoder=vocoder,
            offload_transformer_for_decode=bool(getattr(args, "sample_with_offloading", False)),
            transformer_offload_device=torch.device("cpu"),
            restore_transformer_device=True,
            audio_output_path=wav_path if enable_audio_preview else None,
            use_audio_subprocess=use_audio_subprocess,
            enable_audio_preview=enable_audio_preview,
            decode_video=not audio_only_preview,
            audio_only=audio_only_preview,
        )
    else:
        video, audio_waveform = do_inference(
            trainer,
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
            audio_decoder=audio_decoder,
            vocoder=vocoder,
            offload_transformer_for_decode=bool(getattr(args, "sample_with_offloading", False)),
            transformer_offload_device=torch.device("cpu"),
            restore_transformer_device=True,
            audio_output_path=wav_path if enable_audio_preview else None,
            use_audio_subprocess=use_audio_subprocess,
            enable_audio_preview=enable_audio_preview,
            decode_video=not audio_only_preview,
            audio_only=audio_only_preview,
        )

    if not has_self_ref_orig_mod:
        transformer.train(was_train)

    if video is None and not audio_only_preview:
        logger.error("No video generated / 生成された動画がありません")
        return

    wandb_tracker = None
    try:
        wandb_tracker = accelerator.get_tracker("wandb")
        try:
            import wandb
        except ImportError:
            raise ImportError("No wandb / wandb がインストールされていないようです")
    except:
        wandb = None

    video_path = None
    if video is not None:
        if video.shape[2] == 1:
            image_paths = save_images_grid(video, save_dir, save_path, create_subdir=False)
            if wandb_tracker is not None and wandb is not None:
                for image_path in image_paths:
                    wandb_tracker.log({f"sample_{prompt_idx}": wandb.Image(image_path)}, step=steps)
        else:
            video_path = os.path.join(save_dir, save_path) + ".mp4"
            save_videos_grid(video, video_path)
            if wandb_tracker is not None and wandb is not None:
                wandb_tracker.log({f"sample_{prompt_idx}": wandb.Video(video_path)}, step=steps)
    if audio_waveform is not None:
        wav_path = os.path.join(save_dir, save_path) + ".wav"
        sample_rate = int(getattr(vocoder, "output_sample_rate", 24000)) if vocoder is not None else 24000
        trainer._save_audio_wav(wav_path, audio_waveform, sample_rate)
        if getattr(args, "sample_merge_audio", False) and video_path is not None:
            merged_path = os.path.join(save_dir, save_path) + "_av.mp4"
            trainer._mux_video_audio(video_path, wav_path, merged_path)
    elif getattr(args, "sample_merge_audio", False) and video_path is not None:
        wav_path = os.path.join(save_dir, save_path) + ".wav"
        if os.path.exists(wav_path):
            merged_path = os.path.join(save_dir, save_path) + "_av.mp4"
            trainer._mux_video_audio(video_path, wav_path, merged_path)

    if loaded_text_encoder:
        sample_parameter.pop("prompt_embeds", None)
        sample_parameter.pop("prompt_attention_mask", None)
        sample_parameter.pop("negative_prompt_embeds", None)
        sample_parameter.pop("negative_prompt_attention_mask", None)
        trainer._cleanup_text_encoder(accelerator)
    if loaded_vae:
        vae.to_device("cpu")
        clean_memory_on_device(device)
    if loaded_audio:
        audio_decoder.to("cpu")
        vocoder.to("cpu")
        clean_memory_on_device(device)

def do_two_stage_inference(
    trainer,
    accelerator,
    args,
    sample_parameter: Dict,
    vae,
    dit_dtype: torch.dtype,
    transformer,
    discrete_flow_shift: float,
    sample_steps: int,
    width: int,
    height: int,
    frame_count: int,
    generator: torch.Generator,
    do_classifier_free_guidance: bool,
    guidance_scale: float,
    cfg_scale: Optional[float],
    image_path: Optional[str] = None,
    control_video_path: Optional[str] = None,
    audio_decoder: Optional[torch.nn.Module] = None,
    vocoder: Optional[torch.nn.Module] = None,
    offload_transformer_for_decode: bool = False,
    transformer_offload_device: Optional[torch.device] = None,
    restore_transformer_device: bool = True,
    audio_output_path: Optional[str] = None,
    use_audio_subprocess: bool = False,
    enable_audio_preview: bool = False,
    decode_video: bool = True,
    audio_only: bool = False,
):
    if not getattr(args, "sample_upsampler", None):
        raise ValueError("--sample_two_stage requires --sample_upsampler to be set")

    spatial_factor = int(getattr(vae, "spatial_downsample_factor", 32))
    stage1_width = max(spatial_factor, (width // 2) // spatial_factor * spatial_factor)
    stage1_height = max(spatial_factor, (height // 2) // spatial_factor * spatial_factor)

    _, _, stage1_latents, stage1_audio_latents = do_inference(
        trainer,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        stage1_width,
        stage1_height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=image_path,
        control_video_path=control_video_path,
        audio_decoder=audio_decoder,
        vocoder=vocoder,
        offload_transformer_for_decode=False,
        transformer_offload_device=transformer_offload_device,
        restore_transformer_device=True,
        audio_output_path=None,
        use_audio_subprocess=use_audio_subprocess,
        enable_audio_preview=enable_audio_preview,
        decode_video=False,
        decode_audio=False,
        audio_only=audio_only,
        return_latents=True,
    )

    device = next(transformer.parameters()).device
    upsample_dtype = torch.bfloat16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
    upsampler = _load_upsampler(args.sample_upsampler, device=device, dtype=upsample_dtype)
    video_encoder = _load_video_encoder(args.ltx2_checkpoint, device=device, dtype=upsample_dtype)
    try:
        from musubi_tuner.ltx_2.model.upsampler import upsample_video

        stage2_latents = upsample_video(stage1_latents, video_encoder, upsampler)
    finally:
        video_encoder.to("cpu")
        upsampler.to("cpu")
        clean_memory_on_device(device)

    stage2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=device, dtype=torch.float32)

    distilled_lora_path = getattr(args, "sample_distilled_lora", None)
    distilled_strength = float(getattr(args, "sample_distilled_lora_strength", 1.0) or 1.0)
    distilled_net = None
    try:
        if distilled_lora_path:
            logger.info("Sampling two-stage: applying distilled LoRA %s (strength=%.3f)", distilled_lora_path, distilled_strength)
            distilled_net = _apply_distilled_lora(transformer, distilled_lora_path, distilled_strength)

        video, audio_waveform = do_inference(
            trainer,
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
            False,
            guidance_scale,
            None,
            image_path=image_path,
            control_video_path=control_video_path,
            audio_decoder=audio_decoder,
            vocoder=vocoder,
            offload_transformer_for_decode=offload_transformer_for_decode,
            transformer_offload_device=transformer_offload_device,
            restore_transformer_device=restore_transformer_device,
            audio_output_path=audio_output_path,
            use_audio_subprocess=use_audio_subprocess,
            enable_audio_preview=enable_audio_preview,
            decode_video=decode_video,
            decode_audio=True,
            audio_only=audio_only,
            initial_latents=stage2_latents,
            initial_audio_latents=stage1_audio_latents,
            sigmas_override=stage2_sigmas,
            noise_scale=float(stage2_sigmas[0]),
            return_latents=False,
        )
    finally:
        if distilled_net is not None:
            distilled_net.restore_weights()

    return video, audio_waveform

def do_inference(
    trainer,
    accelerator,
    args,
    sample_parameter: Dict,
    vae,
    dit_dtype: torch.dtype,
    transformer,
    discrete_flow_shift: float,
    sample_steps: int,
    width: int,
    height: int,
    frame_count: int,
    generator: torch.Generator,
    do_classifier_free_guidance: bool,
    guidance_scale: float,
    cfg_scale: Optional[float],
    image_path: Optional[str] = None,
    control_video_path: Optional[str] = None,
    audio_decoder: Optional[torch.nn.Module] = None,
    vocoder: Optional[torch.nn.Module] = None,
    offload_transformer_for_decode: bool = False,
    transformer_offload_device: Optional[torch.device] = None,
    restore_transformer_device: bool = True,
    audio_output_path: Optional[str] = None,
    use_audio_subprocess: bool = False,
    enable_audio_preview: bool = False,
    decode_video: bool = True,
    decode_audio: bool = True,
    audio_only: bool = False,
    return_latents: bool = False,
    initial_latents: Optional[torch.Tensor] = None,
    initial_audio_latents: Optional[torch.Tensor] = None,
    sigmas_override: Optional[torch.Tensor] = None,
    noise_scale: Optional[float] = None,
):
    """Generate sample video during training using LTX-2 denoising loop"""
    from musubi_tuner.ltx_2.types import AudioLatentShape, VideoPixelShape
    from musubi_tuner.ltx_2.model.ltx2_scheduler import LTX2Scheduler, EulerDiffusionStep, X0PredictionWrapper

    transformer_device = next(transformer.parameters()).device
    if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
        trainer._ensure_fp8_buffers_on_device(transformer)
    transformer_offload_device = transformer_offload_device or torch.device("cpu")
    original_vae_device = getattr(vae, "device", torch.device("cpu"))
    original_vae_dtype = getattr(vae, "dtype", torch.float32)
    # Keep VAE off GPU during denoise when offloading is enabled.
    if not offload_transformer_for_decode:
        vae.to_device(transformer_device)
    vae.to_dtype(original_vae_dtype)

    # Get text embeddings
    prompt_embeds = sample_parameter.get("prompt_embeds")
    if prompt_embeds is None:
        raise ValueError("Sample parameter missing prompt embeddings")
    if prompt_embeds.dim() == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    prompt_embeds = prompt_embeds.to(device=transformer_device, dtype=dit_dtype)

    prompt_mask = sample_parameter.get("prompt_attention_mask")
    def _normalize_prompt_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 1:
            return mask.unsqueeze(0)
        if mask.dim() > 2:
            return mask.view(mask.shape[0], -1)
        return mask

    if do_classifier_free_guidance:
        negative_prompt_embeds = sample_parameter.get("negative_prompt_embeds")
        negative_prompt_mask = sample_parameter.get("negative_prompt_attention_mask")
        if negative_prompt_embeds is not None:
            if negative_prompt_embeds.dim() == 2:
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=transformer_device, dtype=dit_dtype
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_mask = _normalize_prompt_mask(prompt_mask)
            negative_prompt_mask = _normalize_prompt_mask(negative_prompt_mask)
            if prompt_mask is not None and negative_prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask], dim=0)
            elif prompt_mask is not None:
                logger.warning(
                    "Sampling: negative prompt mask missing; duplicating prompt mask."
                )
                prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)
        else:
            logger.warning(
                "Sampling: negative prompt embeddings missing; duplicating prompt embeds."
            )
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
            prompt_mask = _normalize_prompt_mask(prompt_mask)
            if prompt_mask is not None:
                prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)
    if prompt_mask is not None:
        prompt_mask = _normalize_prompt_mask(prompt_mask)
    if prompt_mask is not None:
        mask_len = prompt_mask.shape[-1]
        embed_len = prompt_embeds.shape[1]
        if mask_len != embed_len:
            logger.warning(
                "Sample prompt mask length %s != embeds length %s; aligning mask for sampling.",
                mask_len,
                embed_len,
            )
            if mask_len > embed_len:
                # padding_side="left" in the Gemma encoder, keep rightmost tokens.
                prompt_mask = prompt_mask[:, -embed_len:]
            else:
                pad = embed_len - mask_len
                prompt_mask = F.pad(prompt_mask, (pad, 0), value=1)
        if prompt_mask.shape[-1] != prompt_embeds.shape[1]:
            logger.warning(
                "Sample prompt mask still mismatched after alignment (mask=%s, embeds=%s); disabling mask for sampling.",
                prompt_mask.shape[-1],
                prompt_embeds.shape[1],
            )
            prompt_mask = None
    prompt_mask = prompt_mask.to(device=transformer_device, dtype=torch.int64) if prompt_mask is not None else None

    attention_overrides = []
    if getattr(args, "sample_disable_flash_attn", True):
        from musubi_tuner.ltx_2.model.transformer.attention import AttentionFunction

        logger.info("Sampling: disabling FlashAttention for preview")
        attention_overrides = trainer._override_attention_function(
            transformer, AttentionFunction.PYTORCH
        )
        if prompt_mask is not None:
            logger.info("Sampling: disabling prompt attention mask for preview")
            prompt_mask = None

    enable_audio_preview = bool(enable_audio_preview)
    if not enable_audio_preview and prompt_embeds.shape[-1] % 2 == 0:
        logger.warning(
            "Sampling: audio preview disabled; using video-only prompt embeddings (half of dim=%s).",
            prompt_embeds.shape[-1],
        )
        prompt_embeds = prompt_embeds[..., : prompt_embeds.shape[-1] // 2]

    # Setup LTX-2 specific scheduler and stepper
    if sigmas_override is not None:
        sigmas = sigmas_override.to(device=transformer_device, dtype=torch.float32)
    else:
        ltx2_scheduler = LTX2Scheduler(shift=1.0)  # LTX-2 uses shift=1.0 (no shift)
        sigmas = ltx2_scheduler.execute(steps=sample_steps).to(device=transformer_device, dtype=torch.float32)
    stepper = EulerDiffusionStep()

    # Calculate latent dimensions
    vae_scale_factor_temporal = getattr(vae, "temporal_downsample_factor", 4)
    vae_scale_factor_spatial = getattr(vae, "spatial_downsample_factor", 8)
    latent_frames = (frame_count - 1) // vae_scale_factor_temporal + 1
    latent_height = height // vae_scale_factor_spatial
    latent_width = width // vae_scale_factor_spatial
    in_channels = getattr(transformer, "in_channels", 128)

    # Initialize latents
    if initial_latents is None:
        latents = torch.randn(
            (1, int(in_channels), latent_frames, latent_height, latent_width),
            dtype=torch.float32,
            device=transformer_device,
            generator=generator,
        )
    else:
        latents = initial_latents.to(device=transformer_device, dtype=torch.float32)
        if noise_scale is None:
            noise_scale = float(sigmas[0].item()) if sigmas.numel() else 1.0
        latents = latents + torch.randn(
            latents.shape,
            device=transformer_device,
            dtype=latents.dtype,
            generator=generator,
        ) * float(noise_scale)

    audio_latents = None
    if enable_audio_preview:
        frame_rate = sample_parameter.get("frame_rate", 25)
        video_shape = VideoPixelShape(
            batch=1,
            frames=int(frame_count),
            height=int(height),
            width=int(width),
            fps=float(frame_rate),
        )
        audio_cfg = trainer._get_audio_preview_config(args, transformer)
        channels = int(audio_cfg["channels"])
        mel_bins = int(audio_cfg["mel_bins"])
        sample_rate = int(audio_cfg["sample_rate"])
        hop_length = int(audio_cfg["hop_length"])
        audio_downsample = int(audio_cfg["audio_latent_downsample_factor"])
        audio_shape = AudioLatentShape.from_video_pixel_shape(
            video_shape,
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_downsample,
        )
        audio_frames = max(int(audio_shape.frames), 1)
        if initial_audio_latents is None:
            audio_latents = torch.randn(
                (1, channels, audio_frames, mel_bins),
                dtype=torch.float32,
                device=transformer_device,
                generator=generator,
            )
        else:
            audio_latents = initial_audio_latents.to(device=transformer_device, dtype=torch.float32)
            if noise_scale is None:
                noise_scale = float(sigmas[0].item()) if sigmas.numel() else 1.0
            audio_latents = audio_latents + torch.randn(
                audio_latents.shape,
                device=transformer_device,
                dtype=audio_latents.dtype,
                generator=generator,
            ) * float(noise_scale)

    # Denoising loop using LTX-2 scheduler with sigmas
    with torch.no_grad():
        for step_idx in tqdm(range(len(sigmas) - 1), desc="LTX-2 preview", leave=False):
            sigma = sigmas[step_idx]

            # Expand for CFG if needed
            latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
            latent_model_input = latent_model_input.to(dtype=dit_dtype)

            audio_model_input = None
            if audio_latents is not None:
                audio_model_input = (
                    torch.cat([audio_latents, audio_latents], dim=0)
                    if do_classifier_free_guidance
                    else audio_latents
                )
                audio_model_input = audio_model_input.to(dtype=dit_dtype)

            # Prepare timestep (sigma in [0, 1])
            timestep_for_model = sigma.expand(latent_model_input.shape[0]).to(device=transformer_device, dtype=dit_dtype)

            # Model prediction
            if trainer._audio_video and audio_model_input is not None:
                model_input = [latent_model_input, audio_model_input]
            else:
                model_input = latent_model_input

            model_pred = transformer(
                model_input,
                timestep=timestep_for_model.unsqueeze(1),  # [B, 1] for per-token timesteps
                context=prompt_embeds,
                attention_mask=prompt_mask,
                frame_rate=sample_parameter.get("frame_rate", 25),
                transformer_options={},
                audio_only=audio_only,
            )

            audio_pred = None
            if isinstance(model_pred, (list, tuple)):
                video_pred, audio_pred = model_pred
            else:
                video_pred = model_pred

            # Apply guidance if needed
            if do_classifier_free_guidance:
                # Use cfg_scale for CFG when available, otherwise fall back to guidance_scale
                effective_cfg_scale = cfg_scale if cfg_scale is not None else guidance_scale
                noise_uncond, noise_cond = video_pred.chunk(2)
                video_pred = noise_uncond + effective_cfg_scale * (noise_cond - noise_uncond)
                if audio_pred is not None:
                    audio_uncond, audio_cond = audio_pred.chunk(2)
                    audio_pred = audio_uncond + effective_cfg_scale * (audio_cond - audio_uncond)

            # Convert velocity prediction to x0 (denoised sample)
            video_pred = video_pred.to(dtype=latents.dtype)
            video_x0 = X0PredictionWrapper.velocity_to_x0(latents, video_pred, sigma.item())

            # Euler step to next latent
            latents = stepper.step(latents, video_x0, sigmas, step_idx)

            if audio_pred is not None and audio_latents is not None:
                audio_pred = audio_pred.to(dtype=audio_latents.dtype)
                audio_x0 = X0PredictionWrapper.velocity_to_x0(audio_latents, audio_pred, sigma.item())
                audio_latents = stepper.step(audio_latents, audio_x0, sigmas, step_idx)

    if return_latents:
        if attention_overrides:
            trainer._restore_attention_function(attention_overrides)
        return None, None, latents, audio_latents

    if offload_transformer_for_decode and transformer_device != transformer_offload_device:
        if hasattr(transformer, "move_to_device_except_swap_blocks"):
            transformer.move_to_device_except_swap_blocks(transformer_offload_device)
        else:
            transformer.to(transformer_offload_device)
        logger.info("Sampling offload: moved transformer to CPU for VAE decode")
        trainer._cleanup_cuda(transformer_device)

    # Decode latents
    if not decode_video:
        video = None
    else:
        if offload_transformer_for_decode:
            logger.info("Sampling offload: moving VAE to GPU for decode")
            vae.to_device(transformer_device)
        else:
            vae.to_device(transformer_device)

        def _decode_video() -> torch.Tensor:
            with torch.no_grad():
                use_tiled_vae = getattr(args, "sample_tiled_vae", False)
                if use_tiled_vae:
                    from musubi_tuner.ltx_2.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
                    tile_size = getattr(args, "sample_vae_tile_size", 512)
                    tile_overlap = getattr(args, "sample_vae_tile_overlap", 64)
                    temporal_tile_size = getattr(args, "sample_vae_temporal_tile_size", 0)
                    temporal_tile_overlap = getattr(args, "sample_vae_temporal_tile_overlap", 8)

                    # Use configured temporal tiling, or 9999 frames (all at once) if disabled
                    effective_temporal_size = temporal_tile_size if temporal_tile_size > 0 else 9999
                    effective_temporal_overlap = temporal_tile_overlap if temporal_tile_size > 0 else 0

                    tiling_config = TilingConfig(
                        spatial_config=SpatialTilingConfig(
                            tile_size_in_pixels=tile_size,
                            tile_overlap_in_pixels=tile_overlap,
                        ),
                        temporal_config=TemporalTilingConfig(
                            tile_size_in_frames=effective_temporal_size,
                            tile_overlap_in_frames=effective_temporal_overlap,
                        ),
                    )
                    if temporal_tile_size > 0:
                        logger.info(
                            "Using tiled VAE decode (spatial=%dx%d, temporal=%d/%d)",
                            tile_size,
                            tile_overlap,
                            temporal_tile_size,
                            temporal_tile_overlap,
                        )
                    else:
                        logger.info(
                            "Using tiled VAE decode (spatial=%dx%d, no temporal tiling)",
                            tile_size,
                            tile_overlap,
                        )
                    out = vae.tiled_decode(latents.squeeze(0), tiling_config)
                    if out.dim() == 4:  # [C, T, H, W]
                        out = out.unsqueeze(0)  # [1, C, T, H, W]
                    return out

                out = vae.decode([latents.squeeze(0)])
                if isinstance(out, list) and out:
                    out = out[0]
                    if out.dim() == 4:  # [C, T, H, W]
                        out = out.unsqueeze(0)  # [1, C, T, H, W]
                return out

        try:
            video = _decode_video()
        except torch.OutOfMemoryError as exc:
            if not offload_transformer_for_decode:
                raise
            logger.warning("Sampling decode OOM on GPU; retrying on CPU: %s", exc)
            vae.to_device(original_vae_device)
            clean_memory_on_device(transformer_device)
            video = _decode_video()

    audio_waveform = None
    if audio_latents is not None and decode_audio:
        if use_audio_subprocess and audio_output_path:
            trainer._decode_audio_preview_subprocess(
                audio_latents=audio_latents,
                output_path=audio_output_path,
                checkpoint_path=args.ltx2_checkpoint,
            )
        elif audio_decoder is not None and vocoder is not None:
            if offload_transformer_for_decode:
                logger.info("Sampling offload: moving VAE back to CPU before audio decode")
                vae.to_device(original_vae_device)
                clean_memory_on_device(transformer_device)

            decode_device = transformer_device
            if decode_device.type == "cpu":
                logger.info("Sampling offload: decoding audio on CPU")
            audio_decoder.to(decode_device)
            vocoder.to(decode_device)
            with torch.no_grad():
                decode_dtype = torch.bfloat16
                audio_latents = audio_latents.to(device=decode_device, dtype=decode_dtype)
                decoded_audio = audio_decoder(audio_latents)
                audio_waveform = vocoder(decoded_audio).squeeze(0).float().cpu()
            audio_decoder.to("cpu")
            vocoder.to("cpu")
        else:
            logger.warning("Sampling: audio preview requested but no decoder/vocoder available; skipping audio decode.")

    if attention_overrides:
        trainer._restore_attention_function(attention_overrides)
    if offload_transformer_for_decode and restore_transformer_device and transformer_device != transformer_offload_device:
        if hasattr(transformer, "move_to_device_except_swap_blocks"):
            transformer.move_to_device_except_swap_blocks(transformer_device)
        else:
            transformer.to(transformer_device)
        logger.info("Sampling offload: restored transformer to GPU after decode")
        clean_memory_on_device(transformer_device)

    # Normalize to [0, 1]
    if video is not None:
        video = (video / 2 + 0.5).clamp(0, 1).to(torch.float32).to("cpu")

    # Restore VAE state
    vae.to_device(original_vae_device)
    vae.to_dtype(original_vae_dtype)

    return video, audio_waveform
