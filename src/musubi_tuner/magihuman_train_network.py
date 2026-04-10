import argparse
from contextlib import nullcontext
import gc
import logging
import math
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator, init_empty_weights
from PIL import Image

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_MAGIHUMAN, ARCHITECTURE_MAGIHUMAN_FULL
from musubi_tuner.hv_train_network import (
    FlowMatchDiscreteScheduler,
    NetworkTrainer,
    get_sigmas,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.magihuman import parse_magihuman_config
from musubi_tuner.magihuman.common import DataProxyConfig
from musubi_tuner.magihuman.infra.checkpoint import (
    load_model_checkpoint,
    load_model_checkpoint_state_dict,
    resolve_checkpoint_weight_files,
)
from musubi_tuner.magihuman.model.t5_gemma import get_t5_gemma_embedding
from musubi_tuner.magihuman.model.vae2_2 import get_vae2_2
from musubi_tuner.magihuman.model.dit import DiTModel, configure_attention_backend
from musubi_tuner.magihuman.model.dit.dit_module import BaseLinear, NativeMoELinear
from musubi_tuner.magihuman.pipeline.data_proxy import MagiDataProxy
from musubi_tuner.magihuman.pipeline.video_generate import EvalInput
from musubi_tuner.hv_generate_video import resize_image_to_bucket
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils import model_utils
from musubi_tuner.modules.fp8_optimization_utils import (
    calculate_fp8_maxval,
    load_safetensors_with_fp8_optimization,
    optimize_state_dict_with_fp8,
    quantize_weight,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _validate_magihuman_args(args: argparse.Namespace):
    sampling_requested = (
        args.sample_prompts is not None
        or args.sample_at_first
        or args.sample_every_n_steps is not None
        or args.sample_every_n_epochs is not None
    )
    if sampling_requested:
        if args.sample_prompts is None:
            raise ValueError("MagiHuman sampling requires --sample_prompts when sample scheduling flags are enabled.")
        if args.text_encoder is None:
            raise ValueError("MagiHuman sampling requires --text_encoder to encode sample prompts.")
        if args.vae is None:
            raise ValueError("MagiHuman sampling requires --vae to decode preview videos.")
        if args.audio_model is None:
            raise ValueError("MagiHuman sampling requires --audio_model to derive audio token geometry.")

    data_proxy_defaults = DataProxyConfig()
    if args.magihuman_coords_style != "v1" and args.magihuman_text_offset != data_proxy_defaults.text_offset:
        raise ValueError("--magihuman_text_offset only affects --magihuman_coords_style v1 in the current MagiHuman trainer.")
    if args.t5gemma_load_in_8bit and args.t5gemma_load_in_4bit:
        raise ValueError("Only one of --t5gemma_load_in_8bit or --t5gemma_load_in_4bit can be enabled.")


def _resolve_t5gemma_bnb_4bit_compute_dtype(args: argparse.Namespace, weight_dtype: torch.dtype) -> torch.dtype | None:
    arg = args.t5gemma_bnb_4bit_compute_dtype
    if arg == "auto":
        return weight_dtype
    if arg == "fp16":
        return torch.float16
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported T5-Gemma 4-bit compute dtype: {arg}")


def _apply_magihuman_cli_overrides(config, args: argparse.Namespace):
    config.evaluation_config.data_proxy_config = DataProxyConfig(
        t_patch_size=args.magihuman_t_patch_size,
        patch_size=args.magihuman_patch_size,
        frame_receptive_field=args.magihuman_frame_receptive_field,
        spatial_rope_interpolation=args.magihuman_spatial_rope_interpolation,
        text_offset=args.magihuman_text_offset,
        coords_style=args.magihuman_coords_style,
    )
    return config


def _apply_magihuman_fp8_buffers(model: torch.nn.Module, optimized_state_dict: dict[str, torch.Tensor]):
    scale_keys = [key for key in optimized_state_dict.keys() if key.endswith(".scale_weight")]
    if not scale_keys:
        return

    module_lookup = dict(model.named_modules())
    for scale_key in scale_keys:
        module_name = scale_key.rsplit(".scale_weight", 1)[0]
        module = module_lookup.get(module_name)
        if not isinstance(module, (BaseLinear, NativeMoELinear)):
            continue
        scale_tensor = optimized_state_dict[scale_key]
        if hasattr(module, "scale_weight"):
            continue
        module.register_buffer("scale_weight", torch.ones_like(scale_tensor))


_MAGIHUMAN_FP8_TARGET_KEYS = ("linear_qkv", "linear_proj", "up_gate_proj", "down_proj")
_MAGIHUMAN_FP8_EXCLUDE_KEYS = ("norm",)


def _is_magihuman_fp8_target_key(key: str) -> bool:
    return key.endswith(".weight") and any(pattern in key for pattern in _MAGIHUMAN_FP8_TARGET_KEYS) and not any(
        pattern in key for pattern in _MAGIHUMAN_FP8_EXCLUDE_KEYS
    )


def _load_magihuman_fp8_state_dict_chunked(
    model_files: list[str],
    calc_device: str | torch.device,
    move_to_device: bool,
    disable_numpy_memmap: bool,
    row_chunk_size: int = 2048,
) -> dict[str, torch.Tensor]:
    fp8_dtype = torch.float8_e4m3fn
    max_value = calculate_fp8_maxval(4, 3)
    min_value = -max_value
    calc_device = torch.device(calc_device)

    state_dict: dict[str, torch.Tensor] = {}
    optimized_count = 0

    for model_file in model_files:
        logger.info("Streaming MagiHuman FP8 quantization from %s with row_chunk_size=%s", model_file, row_chunk_size)
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as handle:
            for key in handle.keys():
                value = handle.get_tensor(key)
                if not _is_magihuman_fp8_target_key(key):
                    if move_to_device:
                        value = value.to(calc_device, non_blocking=False)
                    state_dict[key] = value
                    continue

                original_device = value.device
                original_dtype = value.dtype
                quantization_mode = "block"
                if value.ndim != 2:
                    quantization_mode = "tensor"
                elif value.shape[1] % 64 != 0:
                    quantization_mode = "channel"
                    logger.warning(
                        "Layer %s with shape %s is not divisible by block_size 64, fallback to per-channel quantization.",
                        key,
                        tuple(value.shape),
                    )

                scale_key = key.replace(".weight", ".scale_weight")
                if value.ndim == 2 and value.shape[0] > row_chunk_size:
                    quantized_chunks = []
                    scale_chunks = []
                    for start in range(0, value.shape[0], row_chunk_size):
                        end = min(start + row_chunk_size, value.shape[0])
                        chunk = value[start:end].to(calc_device, non_blocking=False)
                        quantized_chunk, scale_chunk = quantize_weight(
                            key,
                            chunk,
                            fp8_dtype,
                            max_value,
                            min_value,
                            quantization_mode,
                            64,
                        )

                        if not move_to_device:
                            quantized_chunk = quantized_chunk.to(original_device)
                            scale_chunk = scale_chunk.to(dtype=original_dtype, device=original_device)
                        else:
                            scale_chunk = scale_chunk.to(dtype=original_dtype, device=calc_device)

                        quantized_chunks.append(quantized_chunk)
                        scale_chunks.append(scale_chunk)

                        del chunk, quantized_chunk, scale_chunk
                        clean_memory_on_device(calc_device)

                    state_dict[key] = torch.cat(quantized_chunks, dim=0)
                    state_dict[scale_key] = torch.cat(scale_chunks, dim=0)
                    del quantized_chunks, scale_chunks
                else:
                    value = value.to(calc_device, non_blocking=False)
                    quantized_weight, scale_tensor = quantize_weight(
                        key,
                        value,
                        fp8_dtype,
                        max_value,
                        min_value,
                        quantization_mode,
                        64,
                    )

                    if not move_to_device:
                        quantized_weight = quantized_weight.to(original_device)
                        scale_tensor = scale_tensor.to(dtype=original_dtype, device=original_device)
                    else:
                        scale_tensor = scale_tensor.to(dtype=original_dtype, device=calc_device)

                    state_dict[key] = quantized_weight
                    state_dict[scale_key] = scale_tensor
                    del quantized_weight, scale_tensor

                optimized_count += 1
                del value
                clean_memory_on_device(calc_device)

    logger.info("Number of optimized Linear layers: %s", optimized_count)
    return state_dict


class MagiHumanNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.magihuman_config = None
        self.data_proxy: Optional[MagiDataProxy] = None
        self._last_noise_scale: Optional[torch.Tensor] = None
        self._audio_geometry_cache: dict[str, tuple[int, int]] = {}
        self._last_video_loss: Optional[float] = None
        self._last_audio_loss: Optional[float] = None

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MAGIHUMAN

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MAGIHUMAN_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = model_utils.str_to_dtype(args.dit_dtype) if args.dit_dtype is not None else None
        self._i2v_training = True
        self._control_training = False
        self.default_guidance_scale = 5.0
        self.default_discrete_flow_shift = 5.0

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
        if self._last_video_loss is not None:
            logs["loss/video"] = self._last_video_loss
        if self._last_audio_loss is not None:
            logs["loss/audio"] = self._last_audio_loss
        return logs

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        logger.info(f"cache T5-Gemma outputs for sample prompt file: {sample_prompts}")
        prompts = load_prompts(sample_prompts)
        weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
        bnb_4bit_compute_dtype = _resolve_t5gemma_bnb_4bit_compute_dtype(args, weight_dtype)
        sample_fps = parse_magihuman_config().evaluation_config.fps

        prompt_cache: dict[str, torch.Tensor] = {}
        for prompt_dict in prompts:
            for prompt in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                if prompt is None or prompt in prompt_cache:
                    continue
                logger.info(f"cache T5-Gemma output for prompt: {prompt}")
                prompt_cache[prompt] = get_t5_gemma_embedding(
                    prompt,
                    args.text_encoder,
                    str(accelerator.device),
                    weight_dtype,
                    load_in_8bit=bool(args.t5gemma_load_in_8bit),
                    load_in_4bit=bool(args.t5gemma_load_in_4bit),
                    bnb_4bit_quant_type=args.t5gemma_bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=not bool(args.t5gemma_bnb_4bit_disable_double_quant),
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                )[0].cpu()

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["t5gemma_embed"] = prompt_cache[prompt_dict.get("prompt", "")]
            negative_prompt = prompt_dict.get("negative_prompt", None)
            if negative_prompt is not None:
                prompt_dict_copy["negative_t5gemma_embed"] = prompt_cache[negative_prompt]

            if "fps" not in prompt_dict_copy:
                prompt_dict_copy["fps"] = sample_fps

            sample_parameters.append(prompt_dict_copy)

        return sample_parameters

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
        del control_video_path

        if self.data_proxy is None or self.magihuman_config is None:
            raise RuntimeError("MagiHuman inference requires an initialized data proxy and MagiHuman config.")

        device = accelerator.device
        cfg_scale = guidance_scale if cfg_scale is None else cfg_scale
        do_classifier_free_guidance = do_classifier_free_guidance and cfg_scale != 1.0

        scheduler = FlowMatchDiscreteScheduler(shift=discrete_flow_shift, reverse=True, solver="euler")
        scheduler.set_timesteps(sample_steps, device=device)
        timesteps = scheduler.timesteps
        sample_fps = int(sample_parameter.get("fps", parse_magihuman_config().evaluation_config.fps))

        video_channels = self.magihuman_config.evaluation_config.z_dim
        audio_channels = self.magihuman_config.arch_config.audio_in_channels
        t_stride, h_stride, w_stride = self.magihuman_config.evaluation_config.vae_stride
        if height % h_stride != 0 or width % w_stride != 0:
            raise ValueError(
                f"MagiHuman sampling requires height/width divisible by {(h_stride, w_stride)}, got {(height, width)}."
            )
        latent_frames = max(1, (frame_count - 1) // t_stride + 1)
        latent_height = max(1, height // h_stride)
        latent_width = max(1, width // w_stride)
        audio_token_length = self._get_audio_token_length(args.audio_model, frame_count, sample_fps)

        video_latents = torch.randn(
            (1, video_channels, latent_frames, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        audio_latents = torch.randn(
            (1, audio_token_length, audio_channels),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        image_latents = None
        if image_path is not None:
            self._temporarily_offload_transformer_for_sampling(transformer, device, args.sample_offload_transformer)
            try:
                vae.to(device)
                image_latents = self._encode_reference_image_latents(vae, image_path, width, height, device)
            finally:
                vae.to("cpu")
                clean_memory_on_device(device)
                self._restore_transformer_after_sampling_offload(transformer, device, args.sample_offload_transformer)

        prompt_embed = sample_parameter["t5gemma_embed"].to(device=device, dtype=torch.float32)
        negative_prompt_embed = None
        if do_classifier_free_guidance:
            negative_prompt_embed = sample_parameter["negative_t5gemma_embed"].to(device=device, dtype=torch.float32)

        for timestep in timesteps:
            noisy_video = video_latents
            if image_latents is not None:
                noisy_video = video_latents.clone()
                noisy_video[:, :, :1] = image_latents[:, :, :1]

            if do_classifier_free_guidance:
                batch_video = torch.cat([noisy_video, noisy_video], dim=0)
                batch_audio = torch.cat([audio_latents, audio_latents], dim=0)
                text_embeddings = [negative_prompt_embed, prompt_embed]
            else:
                batch_video = noisy_video
                batch_audio = audio_latents
                text_embeddings = [prompt_embed]

            video_pred, audio_pred = self._predict_joint_outputs(accelerator, transformer, batch_video, batch_audio, text_embeddings)
            video_pred = video_pred.to(torch.float32)
            audio_pred = audio_pred.to(torch.float32)

            if do_classifier_free_guidance:
                video_pred_uncond, video_pred_cond = video_pred.chunk(2)
                audio_pred_uncond, audio_pred_cond = audio_pred.chunk(2)
                video_pred = video_pred_uncond + cfg_scale * (video_pred_cond - video_pred_uncond)
                audio_pred = audio_pred_uncond + cfg_scale * (audio_pred_cond - audio_pred_uncond)

            video_latents = scheduler.step(video_pred, timestep, video_latents, return_dict=False)[0]
            audio_latents = scheduler.step(audio_pred, timestep, audio_latents, return_dict=False)[0]

            if image_latents is not None:
                video_latents[:, :, :1] = image_latents[:, :, :1]

        self._temporarily_offload_transformer_for_sampling(transformer, device, args.sample_offload_transformer)
        try:
            vae.to(device)
            decode_dtype = self._get_vae_dtype(vae)
            video_latents = video_latents.to(device=device, dtype=decode_dtype)
            with torch.no_grad():
                video = vae.decode(video_latents)
        finally:
            vae.to("cpu")
            clean_memory_on_device(device)
            self._restore_transformer_after_sampling_offload(transformer, device, args.sample_offload_transformer)
        video = (video / 2 + 0.5).clamp(0, 1)
        return video.cpu().float()

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        return get_vae2_2(vae_path, device="cpu", weight_dtype=vae_dtype)

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
        config = parse_magihuman_config()
        config = _apply_magihuman_cli_overrides(config, args)
        if not config.arch_config.local_attn_layers and config.evaluation_config.data_proxy_config.frame_receptive_field != -1:
            logger.warning(
                "Base MagiHuman training does not enable local-attention layers. "
                "Disabling frame_receptive_field=%s for this run.",
                config.evaluation_config.data_proxy_config.frame_receptive_field,
            )
            config.evaluation_config.data_proxy_config.frame_receptive_field = -1
        self.magihuman_config = config
        self.data_proxy = MagiDataProxy(config.evaluation_config.data_proxy_config)
        compute_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)

        configure_attention_backend(attn_mode, split_attn, compute_dtype)

        use_fp8 = bool(args.fp8_base or args.fp8_scaled)
        config.arch_config.params_dtype = compute_dtype if use_fp8 else (dit_weight_dtype or compute_dtype)
        config.arch_config.compute_dtype = compute_dtype

        if args.dit is not None:
            config.engine_config.load = args.dit

        use_fast_full_state_dict_load = (
            self.blocks_to_swap == 0
            and config.engine_config.load is not None
            and Path(config.engine_config.load).is_file()
            and not str(config.engine_config.load).endswith(".zst")
        )
        init_context = init_empty_weights() if use_fast_full_state_dict_load else nullcontext()
        with init_context:
            model = DiTModel(model_config=config.arch_config)
            if use_fast_full_state_dict_load and dit_weight_dtype is not None:
                model = model.to(dit_weight_dtype)

        if not use_fast_full_state_dict_load and loading_device is not None and str(loading_device) != "cpu":
            model = model.to(loading_device)
        if config.engine_config.load and use_fp8:
            disable_memmap = getattr(args, "disable_numpy_memmap", False)
            blocks_to_swap = getattr(args, "blocks_to_swap", 0) or 0
            fp8_quant_device = getattr(args, "magihuman_fp8_quant_device", "auto")
            if fp8_quant_device == "auto":
                quant_device = "cpu" if blocks_to_swap > 0 else accelerator.device
            elif fp8_quant_device == "cpu":
                quant_device = "cpu"
            elif fp8_quant_device == "cuda":
                quant_device = accelerator.device
            else:
                raise ValueError(f"Unsupported --magihuman_fp8_quant_device value: {fp8_quant_device}")
            logger.info(
                f"Applying MagiHuman fp8 optimization (scaled={args.fp8_scaled}, base={args.fp8_base}) on {quant_device}"
            )
            checkpoint_weight_files = resolve_checkpoint_weight_files(config.engine_config)
            has_zstd_shards = any(Path(weight_file + ".zst").exists() for weight_file in checkpoint_weight_files)
            if has_zstd_shards:
                logger.info("MagiHuman checkpoint contains .zst shards, falling back to in-memory FP8 conversion path")
                state_dict = load_model_checkpoint_state_dict(
                    config.engine_config,
                    device="cpu",
                    dtype=None,
                    disable_numpy_memmap=disable_memmap,
                )
                state_dict = optimize_state_dict_with_fp8(
                    state_dict,
                    calc_device=quant_device,
                    target_layer_keys=list(_MAGIHUMAN_FP8_TARGET_KEYS),
                    exclude_layer_keys=list(_MAGIHUMAN_FP8_EXCLUDE_KEYS),
                    quantization_mode="block",
                    block_size=64,
                    move_to_device=blocks_to_swap == 0,
                )
            else:
                logger.info("Using streaming safetensors FP8 load path for MagiHuman: %s", checkpoint_weight_files)
                move_to_device = blocks_to_swap == 0
                if torch.device(quant_device).type == "cuda" and not move_to_device:
                    logger.info("Using chunked CUDA FP8 quantization path for MagiHuman swap load")
                    state_dict = _load_magihuman_fp8_state_dict_chunked(
                        checkpoint_weight_files,
                        calc_device=quant_device,
                        move_to_device=move_to_device,
                        disable_numpy_memmap=disable_memmap,
                    )
                else:
                    state_dict = load_safetensors_with_fp8_optimization(
                        checkpoint_weight_files,
                        calc_device=quant_device,
                        target_layer_keys=list(_MAGIHUMAN_FP8_TARGET_KEYS),
                        exclude_layer_keys=list(_MAGIHUMAN_FP8_EXCLUDE_KEYS),
                        quantization_mode="block",
                        block_size=64,
                        move_to_device=move_to_device,
                        disable_numpy_memmap=disable_memmap,
                    )
            _apply_magihuman_fp8_buffers(model, state_dict)
            info = model.load_state_dict(state_dict, strict=False, assign=True)
            logger.info(f"Loaded MagiHuman fp8 weights: {info}")
            del state_dict
            gc.collect()
            clean_memory_on_device(accelerator.device)
        elif config.engine_config.load:
            model = load_model_checkpoint(
                model,
                config.engine_config,
                device="cpu" if use_fast_full_state_dict_load else loading_device,
                dtype=dit_weight_dtype,
                disable_numpy_memmap=getattr(args, "disable_numpy_memmap", False),
                prefer_full_state_dict=use_fast_full_state_dict_load,
            )
            gc.collect()
            clean_memory_on_device(accelerator.device)
        logger.info("Created MagiHuman DiT from vendored config and initialized the Musubi training proxy.")
        return model

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(args, transformer, [transformer.block.layers], disable_linear=self.blocks_to_swap > 0)

    def scale_shift_latents(self, latents):
        return latents

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps,
        noise_scheduler: FlowMatchDiscreteScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        noisy_model_input, sampled_timesteps = super().get_noisy_model_input_and_timesteps(
            args, noise, latents, timesteps, noise_scheduler, device, dtype
        )

        if args.timestep_sampling == "sigma":
            noise_scale = get_sigmas(noise_scheduler, sampled_timesteps, device, n_dim=1, dtype=torch.float32)
        else:
            noise_scale = ((sampled_timesteps.to(device=device, dtype=torch.float32) - 1.0) / 1000.0).clamp(0.0, 1.0)

        self._last_noise_scale = noise_scale
        return noisy_model_input, sampled_timesteps

    def _pad_text_features(self, embeddings: list[torch.Tensor], device: torch.device, dtype: torch.dtype):
        lengths = [int(embed.shape[0]) for embed in embeddings]
        max_length = max(lengths)
        hidden_size = embeddings[0].shape[-1]
        padded = torch.zeros((len(embeddings), max_length, hidden_size), device=device, dtype=dtype)
        for index, embed in enumerate(embeddings):
            padded[index, : embed.shape[0]] = embed.to(device=device, dtype=dtype)
        return padded, lengths

    def _build_audio_noisy_input(
        self,
        audio_latents: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if self._last_noise_scale is None:
            raise RuntimeError("Noise scale was not initialized before MagiHuman call_dit.")

        audio_noise = torch.randn_like(audio_latents)
        noise_scale = self._last_noise_scale.to(device=device, dtype=dtype).view(-1, 1, 1)
        noisy_audio = (1.0 - noise_scale) * audio_latents + noise_scale * audio_noise
        target_audio = audio_noise - audio_latents
        return noisy_audio, target_audio

    def _flatten_joint_prediction(
        self,
        video_pred: torch.Tensor,
        audio_pred: torch.Tensor,
        video_target: torch.Tensor,
        audio_target: torch.Tensor,
        conditioned_video_frames: int,
    ):
        if conditioned_video_frames > 0:
            video_pred = video_pred.clone()
            video_target = video_target.clone()
            video_pred[:, :, :conditioned_video_frames] = 0
            video_target[:, :, :conditioned_video_frames] = 0

        flat_pred = torch.cat([video_pred.flatten(start_dim=1), audio_pred.flatten(start_dim=1)], dim=1)
        flat_target = torch.cat([video_target.flatten(start_dim=1), audio_target.flatten(start_dim=1)], dim=1)
        return flat_pred, flat_target

    def _predict_joint_outputs(
        self,
        accelerator: Accelerator,
        transformer,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        text_embeddings: list[torch.Tensor],
    ):
        if self.data_proxy is None:
            raise RuntimeError("MagiHuman data proxy is not initialized. Transformer must be loaded first.")

        if video_latents.shape[0] > 1 and self.data_proxy.frame_receptive_field != -1:
            video_outputs = []
            audio_outputs = []
            for batch_index, text_embed in enumerate(text_embeddings):
                video_pred, audio_pred = self._predict_joint_outputs(
                    accelerator,
                    transformer,
                    video_latents[batch_index : batch_index + 1],
                    audio_latents[batch_index : batch_index + 1],
                    [text_embed],
                )
                video_outputs.append(video_pred)
                audio_outputs.append(audio_pred)
            return torch.cat(video_outputs, dim=0), torch.cat(audio_outputs, dim=0)

        padded_text, text_lengths = self._pad_text_features(text_embeddings, accelerator.device, torch.float32)
        video_x_t = video_latents.to(device=accelerator.device, dtype=torch.float32)
        audio_x_t = audio_latents.to(device=accelerator.device, dtype=torch.float32)
        if video_latents.requires_grad or audio_latents.requires_grad:
            video_x_t.requires_grad_(True)
            audio_x_t.requires_grad_(True)
            padded_text.requires_grad_(True)

        audio_lengths = [int(audio_x_t[i].shape[0]) for i in range(audio_x_t.shape[0])]
        eval_input = EvalInput(
            x_t=video_x_t,
            audio_x_t=audio_x_t,
            audio_feat_len=audio_lengths,
            txt_feat=padded_text,
            txt_feat_len=text_lengths,
        )
        packed_input = self.data_proxy.process_input(eval_input)
        with accelerator.autocast():
            token_pred = transformer(*packed_input)
        return self.data_proxy.process_output(token_pred)

    def _get_vae_dtype(self, vae) -> torch.dtype:
        inner_vae = getattr(vae, "vae", None)
        if inner_vae is not None:
            first_param = next(inner_vae.parameters(), None)
            if first_param is not None:
                return first_param.dtype

        if hasattr(vae, "dtype") and isinstance(vae.dtype, torch.dtype):
            return vae.dtype

        return torch.float32

    def _get_audio_token_length(self, audio_model_path: str, frame_count: int, fps: int) -> int:
        sample_rate, downsampling_ratio = self._get_audio_geometry(audio_model_path)
        duration_seconds = max(1.0 / fps, frame_count / fps)
        return max(1, math.ceil(duration_seconds * sample_rate / downsampling_ratio))

    def _get_audio_geometry(self, audio_model_path: str) -> tuple[int, int]:
        cached = self._audio_geometry_cache.get(audio_model_path)
        if cached is not None:
            return cached

        config_path = Path(audio_model_path) / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"MagiHuman audio model config was not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        try:
            sample_rate = int(config["sample_rate"])
            downsampling_ratio = int(config["model"]["pretransform"]["config"]["downsampling_ratio"])
        except KeyError as exc:
            raise ValueError(f"MagiHuman audio model config is missing required geometry fields: {config_path}") from exc
        geometry = (sample_rate, downsampling_ratio)
        self._audio_geometry_cache[audio_model_path] = geometry
        return geometry

    def _temporarily_offload_transformer_for_sampling(self, transformer, device: torch.device, enabled: bool):
        if not enabled or device.type == "cpu":
            return

        offloader = getattr(transformer, "offloader", None)
        if offloader is not None:
            for block_idx in list(offloader.futures.keys()):
                offloader.wait_for_block(block_idx)

        transformer.to("cpu")
        clean_memory_on_device(device)

    def _restore_transformer_after_sampling_offload(self, transformer, device: torch.device, enabled: bool):
        if not enabled or device.type == "cpu":
            return

        if getattr(transformer, "blocks_to_swap", 0):
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        else:
            transformer.to(device)

        clean_memory_on_device(device)

    def _encode_reference_image_latents(
        self,
        vae,
        image_path: str,
        width: int,
        height: int,
        device: torch.device,
    ) -> torch.Tensor:
        image = Image.open(image_path)
        image = resize_image_to_bucket(image, (width, height))
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).contiguous()
        image = image.to(device=device, dtype=self._get_vae_dtype(vae))
        image = image / 127.5 - 1.0
        with torch.no_grad():
            image_latents = vae.encode(image).to(torch.float32)
        return image_latents

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
    ):
        if self.data_proxy is None:
            raise RuntimeError("MagiHuman data proxy is not initialized. Transformer must be loaded first.")

        video_latents = latents.to(device=accelerator.device, dtype=torch.float32)
        noisy_video = noisy_model_input.to(device=accelerator.device, dtype=torch.float32)
        video_target = (noise - latents).to(device=accelerator.device, dtype=torch.float32)

        audio_latents = batch["latents_audio"].to(device=accelerator.device, dtype=torch.float32)
        noisy_audio, audio_target = self._build_audio_noisy_input(audio_latents, accelerator.device, torch.float32)

        image_latents = batch.get("latents_image")
        conditioned_video_frames = 0
        if image_latents is not None:
            image_latents = image_latents.to(device=accelerator.device, dtype=torch.float32)
            conditioned_video_frames = min(image_latents.shape[2], noisy_video.shape[2], 1)
            noisy_video = noisy_video.clone()
            noisy_video[:, :, :conditioned_video_frames] = image_latents[:, :, :conditioned_video_frames]

        if args.gradient_checkpointing:
            noisy_video.requires_grad_(True)
            noisy_audio.requires_grad_(True)
        text_embeddings = batch["t5gemma_embed"]
        video_pred, audio_pred = self._predict_joint_outputs(accelerator, transformer, noisy_video, noisy_audio, text_embeddings)

        video_pred = video_pred.to(network_dtype)
        audio_pred = audio_pred.to(network_dtype)
        video_target = video_target.to(network_dtype)
        audio_target = audio_target.to(network_dtype)

        if conditioned_video_frames > 0:
            video_pred_for_loss = video_pred.clone()
            video_target_for_loss = video_target.clone()
            video_pred_for_loss[:, :, :conditioned_video_frames] = 0
            video_target_for_loss[:, :, :conditioned_video_frames] = 0
        else:
            video_pred_for_loss = video_pred
            video_target_for_loss = video_target

        self._last_video_loss = F.mse_loss(video_pred_for_loss.float(), video_target_for_loss.float()).detach().item()
        self._last_audio_loss = F.mse_loss(audio_pred.float(), audio_target.float()).detach().item()

        model_pred, target = self._flatten_joint_prediction(
            video_pred,
            audio_pred,
            video_target,
            audio_target,
            conditioned_video_frames,
        )

        return model_pred, target


def magihuman_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    data_proxy_defaults = DataProxyConfig()
    parser.add_argument("--dit_dtype", type=str, default=None, help="data type for DiT, default is bfloat16")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument(
        "--magihuman_fp8_quant_device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used for one-time MagiHuman FP8 quantization. 'auto' uses CPU for swap runs and CUDA otherwise.",
    )
    parser.add_argument(
        "--audio_model",
        type=str,
        default=None,
        help="Path to the MagiHuman audio model directory. Required for preview sampling to match audio token geometry.",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Path to the local T5-Gemma encoder directory used for MagiHuman sample prompts.",
    )
    parser.add_argument("--t5gemma_load_in_8bit", action="store_true", help="Load T5-Gemma encoder in 8-bit (bitsandbytes). CUDA only.")
    parser.add_argument("--t5gemma_load_in_4bit", action="store_true", help="Load T5-Gemma encoder in 4-bit (bitsandbytes). CUDA only.")
    parser.add_argument(
        "--t5gemma_bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=("nf4", "fp4"),
        help="bitsandbytes 4-bit quant type for T5-Gemma loading.",
    )
    parser.add_argument(
        "--t5gemma_bnb_4bit_disable_double_quant",
        action="store_true",
        help="Disable bitsandbytes double quant for T5-Gemma 4-bit loading.",
    )
    parser.add_argument(
        "--t5gemma_bnb_4bit_compute_dtype",
        type=str,
        default="auto",
        choices=("auto", "fp16", "bf16", "fp32"),
        help="Compute dtype for 4-bit T5-Gemma loading (auto uses mixed precision dtype).",
    )
    parser.add_argument(
        "--magihuman_t_patch_size",
        type=int,
        default=data_proxy_defaults.t_patch_size,
        help="Temporal patch size used by the MagiHuman data proxy.",
    )
    parser.add_argument(
        "--magihuman_patch_size",
        type=int,
        default=data_proxy_defaults.patch_size,
        help="Spatial patch size used by the MagiHuman data proxy.",
    )
    parser.add_argument(
        "--magihuman_frame_receptive_field",
        type=int,
        default=data_proxy_defaults.frame_receptive_field,
        help="Local attention frame receptive field. Use -1 to disable local attention windows.",
    )
    parser.add_argument(
        "--magihuman_spatial_rope_interpolation",
        choices=("inter", "extra"),
        default=data_proxy_defaults.spatial_rope_interpolation,
        help="Spatial RoPE interpolation mode for MagiHuman token coordinates.",
    )
    parser.add_argument(
        "--magihuman_text_offset",
        type=int,
        default=data_proxy_defaults.text_offset,
        help="Text token offset for MagiHuman coordinate metadata. Effective only with --magihuman_coords_style v1.",
    )
    parser.add_argument(
        "--magihuman_coords_style",
        choices=("v1", "v2"),
        default=data_proxy_defaults.coords_style,
        help="Coordinate packing style for MagiHuman audio/text tokens.",
    )
    parser.add_argument(
        "--sample_offload_transformer",
        action="store_true",
        help="Temporarily move the MagiHuman DiT to CPU during VAE-only sampling stages to reduce VRAM usage.",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = magihuman_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    _validate_magihuman_args(args)

    trainer = MagiHumanNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
