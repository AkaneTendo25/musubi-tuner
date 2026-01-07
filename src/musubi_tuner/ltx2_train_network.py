"""LTX-2 LoRA Training Implementation."""

import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from accelerate import Accelerator
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

# LTX-2 latent normalization defaults.
# These are identity stats (mean=0, std=1). We keep them as a safe fallback and
# override them from the loaded VAE if it exposes per-channel statistics.
LTX2_LATENTS_MEAN = [0.0]
LTX2_LATENTS_STD = [1.0]

# Modules to keep in high precision for FP8 quantization
KEEP_FP8_HIGH_PRECISION_TOKENS = (
    "norm",
    "bias",
    "scale_shift_table",
    "patchify_proj",
    "proj_out",
    "adaln_single",
    "caption_projection",
    "layer_norm",
)


def detect_ltx2_dtype(model_path: str) -> torch.dtype:
    """Detect the data type of LTX-2 model weights"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LTX-2 weights must be a .safetensors file. Got: {model_path}")

    with MemoryEfficientSafeOpen(model_path) as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"Unable to detect LTX-2 dtype; no tensors found in {model_path}")

        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point():
                dtype = tensor.dtype
                break
        else:
            dtype = handle.get_tensor(keys[0]).dtype

    logger.info("Detected LTX-2 dtype: %s", dtype)
    return dtype


def detect_ltx2_config(model_path: str) -> Dict[str, Any]:
    """Infer LTX-2 model configuration from weights."""
    keys: List[str]
    with MemoryEfficientSafeOpen(model_path) as handle:
        keys = list(handle.keys())

        def find_key(suffix: str) -> Optional[str]:
            for key in keys:
                if key.endswith(suffix):
                    return key
            return None

        def get_shape(suffix: str) -> Optional[Tuple[int, ...]]:
            key = find_key(suffix)
            if key is None:
                return None
            return tuple(handle.get_tensor(key).shape)

        config: Dict[str, Any] = {}

        # Count transformer blocks
        block_indices = set()
        for key in keys:
            match = re.search(r"transformer_blocks\.(\d+)\.", key)
            if match:
                block_indices.add(int(match.group(1)))
        if block_indices:
            config["num_layers"] = max(block_indices) + 1

        # Infer attention dimensions
        attn2_shape = get_shape("transformer_blocks.0.attn2.to_k.weight")
        if attn2_shape is not None and len(attn2_shape) == 2:
            inner_dim, cross_dim = attn2_shape
            config["cross_attention_dim"] = cross_dim
            config["num_attention_heads"] = 32
            if inner_dim % config["num_attention_heads"] == 0:
                config["attention_head_dim"] = inner_dim // config["num_attention_heads"]
            else:
                logger.warning("Unable to evenly infer attention_head_dim from %s", attn2_shape)

        patchify_shape = get_shape("patchify_proj.weight")
        if patchify_shape is not None and len(patchify_shape) == 2:
            config["in_channels"] = patchify_shape[1]

        caption_shape = get_shape("caption_projection.linear_1.weight")
        if caption_shape is not None and len(caption_shape) == 2:
            config["caption_channels"] = caption_shape[1]

        # Audio-video specific fields
        audio_patchify_shape = get_shape("audio_patchify_proj.weight")
        audio_attn2_shape = get_shape("transformer_blocks.0.audio_attn2.to_k.weight")
        audio_caption_shape = get_shape("audio_caption_projection.linear_1.weight")
        if audio_patchify_shape is not None:
            config["audio_in_channels"] = audio_patchify_shape[1]
        if audio_attn2_shape is not None and len(audio_attn2_shape) == 2:
            audio_inner_dim, audio_cross_dim = audio_attn2_shape
            config["audio_cross_attention_dim"] = audio_cross_dim
            config["audio_num_attention_heads"] = 32
            if audio_inner_dim % config["audio_num_attention_heads"] == 0:
                config["audio_attention_head_dim"] = audio_inner_dim // config["audio_num_attention_heads"]
            else:
                logger.warning("Unable to evenly infer audio_attention_head_dim from %s", audio_attn2_shape)
        if audio_caption_shape is not None and len(audio_caption_shape) == 2:
            config["caption_channels"] = audio_caption_shape[1]

    return config


def load_ltx2_model(
    model_path: str,
    device: Union[str, torch.device] = "cpu",
    load_device: Union[str, torch.device] = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    attn_mode: str = "torch",
    audio_video: bool = False,
    **_: Any,
):
    """Load LTX-2 (video or audio-video) transformer

    Args:
        model_path: Path to safetensors model weights
        device: Target device for model
        load_device: Device to load weights into
        torch_dtype: Data type for model parameters
        attn_mode: Attention implementation (torch, flash, flash3, xformers)
        audio_video: If True, load LTXAV model; if False, load LTXV model
        **_: Additional arguments (ignored)

    Returns:
        Loaded LTX-2 transformer model
    """
    target_device = torch.device(device)
    load_device = torch.device(load_device)

    from musubi_tuner.networks.lora_ltx2 import load_ltx2_wrapper

    logger.info("Loading LTX-2 transformer via LTX2Wrapper: %s", model_path)
    model = load_ltx2_wrapper(
        model_path,
        device=load_device,
        dtype=torch_dtype or torch.float32,
        audio_video=audio_video,
        patch_size=1,
    )
    model = model.to(device=target_device)
    return model


class LTX2NetworkTrainer(NetworkTrainer):
    """Trainer for LTX-2 models with LoRA support"""

    def __init__(self) -> None:
        super().__init__()
        self._text_encoder = None
        self._dit_attn_mode: Optional[str] = None
        self._latent_norm_cache: Dict = {}

        # Initialize latent normalization
        mean = torch.tensor(LTX2_LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(LTX2_LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = std.clamp_min(1e-6)
        self._latent_norm_base: Tuple[torch.Tensor, torch.Tensor] = (mean, std.reciprocal())

        self._flow_target: str = "noise"  # LTX-2 predicts noise
        self._num_timesteps: int = 1000
        self._audio_video: bool = False
        self._ltx_mode: str = "video"
        self.default_guidance_scale = 1.0

    def _normalize_timesteps_for_model(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Normalize timesteps to the model's expected 0..1 sigma range."""
        if timesteps.numel() == 0:
            return timesteps

        return timesteps / 1000.0

    @staticmethod
    def _shifted_logit_normal_shift_for_sequence_length(
        seq_length: int,
        *,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        m = (max_shift - min_shift) / float(max_tokens - min_tokens)
        b = min_shift - m * float(min_tokens)
        return m * float(seq_length) + b

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler,  # noqa: ARG002
        device: torch.device,
        dtype: torch.dtype,
    ):
        if latents.dim() != 5:
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        if latents.device != device:
            latents = latents.to(device=device)
        if noise.device != device:
            noise = noise.to(device=device)

        batch_size = latents.shape[0]
        frames, height, width = latents.shape[2], latents.shape[3], latents.shape[4]
        seq_len = int(frames * height * width)

        shift = self._shifted_logit_normal_shift_for_sequence_length(seq_len)
        normal_samples = torch.randn((batch_size,), device=device, dtype=torch.float32) + float(shift)
        sigmas = torch.sigmoid(normal_samples)

        sigmas_expanded = sigmas.view(-1, 1, 1, 1, 1)
        noisy_model_input = (1.0 - sigmas_expanded) * latents.to(dtype=torch.float32) + sigmas_expanded * noise.to(
            dtype=torch.float32
        )

        timesteps_out = sigmas.to(device=device, dtype=torch.float32) * 1000.0
        return noisy_model_input, timesteps_out

    # ======== Model-specific properties and configuration ========

    @property
    def architecture(self) -> str:
        """Returns architecture identifier"""
        from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2

        return ARCHITECTURE_LTX2

    @property
    def architecture_full_name(self) -> str:
        """Returns full architecture name with version"""
        from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2_FULL

        return ARCHITECTURE_LTX2_FULL

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        """Handle LTX-2-specific command line arguments"""
        self.dit_dtype = detect_ltx2_dtype(args.ltx2_checkpoint)

        if self.dit_dtype == torch.float16:
            assert args.mixed_precision in ["fp16", "no"], "LTX-2 weights are fp16; mixed precision must be fp16 or no"
        elif self.dit_dtype == torch.bfloat16:
            assert args.mixed_precision in ["bf16", "no"], "LTX-2 weights are bf16; mixed precision must be bf16 or no"

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        ltx_mode = getattr(args, "ltx_mode", "video")
        if ltx_mode not in {"video", "av", "audio"}:
            raise ValueError(f"Invalid ltx_mode: {ltx_mode}")
        self._ltx_mode = ltx_mode
        self._audio_video = self._ltx_mode == "av"
        self.default_guidance_scale = 1.0

        args.weighting_scheme = "none"

    @property
    def i2v_training(self) -> bool:
        """LTX-2 doesn't currently support I2V conditioning"""
        return False

    @property
    def control_training(self) -> bool:
        """LTX-2 doesn't currently support control conditioning"""
        return False

    # ======== Model loading ========

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
        """Load LTX-2 transformer model

        Args:
            accelerator: HF Accelerator instance
            args: Training arguments
            dit_path: Path to LTX-2 weights
            attn_mode: Attention implementation
            split_attn: Whether to split attention (ignored for LTX-2)
            loading_device: Device to load weights to
            dit_weight_dtype: Weight data type

        Returns:
            Loaded LTX-2 transformer model
        """
        # Determine attention mode from args
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.flash3:
            attn_mode = "flash3"
        elif args.xformers:
            attn_mode = "xformers"
        else:
            attn_mode = "torch"

        self._dit_attn_mode = attn_mode

        transformer = load_ltx2_model(
            model_path=dit_path,
            device=accelerator.device,
            load_device=loading_device,
            torch_dtype=dit_weight_dtype,
            attn_mode=attn_mode,
            audio_video=self._audio_video,
        )

        transformer.eval()
        transformer.requires_grad_(False)

        return transformer

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        """Load VAE for LTX2"""
        logger.info(f"Loading VAE from {vae_path}")
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.video_vae import VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER

        class _LTX2VideoVAE(torch.nn.Module):
            def __init__(self, decoder: torch.nn.Module):
                super().__init__()
                self.decoder = decoder

                first_param = next(self.decoder.parameters())
                self.device = first_param.device
                self.dtype = first_param.dtype

                # LTX Video VAE configuration compresses frames by 8 (except the first frame) and spatial dims by 32.
                self.temporal_downsample_factor = 8
                self.spatial_downsample_factor = 32

                stats = getattr(self.decoder, "per_channel_statistics", None)
                self.latents_mean = None
                self.latents_std = None
                if stats is not None:
                    try:
                        self.latents_mean = stats.get_buffer("mean-of-means").detach().cpu()
                        self.latents_std = stats.get_buffer("std-of-means").detach().cpu()
                    except Exception:
                        self.latents_mean = None
                        self.latents_std = None

            def to_device(self, device: torch.device | str) -> None:
                self.device = torch.device(device)
                self.decoder.to(self.device)

            def to_dtype(self, dtype: torch.dtype) -> None:
                self.dtype = dtype
                self.decoder.to(dtype=dtype)

            def eval(self) -> None:
                self.decoder.eval()

            def requires_grad_(self, requires_grad: bool = True):
                self.decoder.requires_grad_(requires_grad)
                return self

            def decode(self, zs):
                outs = []
                for z in zs:
                    if z.dim() == 4:
                        z = z.unsqueeze(0)
                    z = z.to(device=self.device, dtype=self.dtype)
                    video = self.decoder(z)
                    outs.append(video.squeeze(0))
                return outs

        decoder = SingleGPUModelBuilder(
            model_path=str(vae_path),
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=torch.device("cpu"), dtype=vae_dtype)
        decoder.eval()
        decoder.requires_grad_(False)

        vae = _LTX2VideoVAE(decoder)
        self._update_latent_norm_base_from_vae(vae)
        return vae

    def _update_latent_norm_base_from_vae(self, vae) -> None:
        """Update latent normalization statistics from VAE config"""
        latents_mean = getattr(vae, "latents_mean", None)
        latents_std = getattr(vae, "latents_std", None)

        if latents_mean is None or latents_std is None:
            # Some VAE wrappers expose mean/std instead of latents_mean/latents_std
            latents_mean = getattr(vae, "mean", None)
            latents_std = getattr(vae, "std", None)

        if latents_mean is None or latents_std is None:
            config = getattr(vae, "config", None)
            if config is None:
                return
            latents_mean = getattr(config, "latents_mean", None)
            latents_std = getattr(config, "latents_std", None)

        if latents_mean is None or latents_std is None:
            return

        if isinstance(latents_mean, torch.Tensor):
            mean = latents_mean.to(dtype=torch.float32).view(1, -1, 1, 1, 1)
        else:
            mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)

        if isinstance(latents_std, torch.Tensor):
            std = latents_std.to(dtype=torch.float32).view(1, -1, 1, 1, 1).clamp_min(1e-6)
        else:
            std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1).clamp_min(1e-6)
        self._latent_norm_base = (mean, std.reciprocal())
        self._latent_norm_cache.clear()

    # ======== Training loop methods ========

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ) -> Tuple[object, torch.Tensor]:
        """Forward pass through LTX-2 (video or audio-video) model

        Args:
            args: Training arguments
            accelerator: HF Accelerator
            transformer: LTX-2 model
            latents: Video latents [B, 128, T, H, W]
            batch: Batch data including text embeddings
            noise: Noise tensor (same shape as latents)
            noisy_model_input: Noisy latents [B, 128, T, H, W]
            timesteps: Diffusion timesteps (normalized 0-1 for flow matching)
            network_dtype: Network precision

        Returns:
            Tuple of (model_prediction, target) for loss computation
        """
        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be a dict, got: {type(batch)}")

        if latents is None or not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected latents to be a torch.Tensor, got: {type(latents)}")
        if latents.dim() != 5:
            raise ValueError(f"Expected latents to be 5D [B, C, F, H, W], got shape: {tuple(latents.shape)}")
        in_channels = getattr(transformer, "in_channels", None)
        if in_channels is None and hasattr(transformer, "patchify_proj"):
            in_channels = getattr(getattr(transformer, "patchify_proj", None), "in_features", None)
        if in_channels is not None and latents.shape[1] != int(in_channels):
            raise ValueError(
                f"Latents channel mismatch: got {latents.shape[1]}, expected {int(in_channels)} (transformer.in_channels)"
            )
        if not torch.isfinite(latents).all():
            raise ValueError("Non-finite (NaN/Inf) detected in latents")

        if timesteps is None or not isinstance(timesteps, torch.Tensor):
            raise TypeError(f"Expected timesteps to be a torch.Tensor, got: {type(timesteps)}")

        conditions = batch.get("conditions")
        if conditions is not None:
            if not isinstance(conditions, dict):
                raise TypeError(f"Expected batch['conditions'] to be a dict, got: {type(conditions)}")
            if self._audio_video:
                video_prompt_embeds = conditions.get("video_prompt_embeds")
                audio_prompt_embeds = conditions.get("audio_prompt_embeds")
                if video_prompt_embeds is not None and audio_prompt_embeds is not None:
                    text_embeds = torch.cat([video_prompt_embeds, audio_prompt_embeds], dim=-1)
                else:
                    text_embeds = conditions.get("prompt_embeds")
            else:
                text_embeds = conditions.get("video_prompt_embeds")

            text_mask = conditions.get("prompt_attention_mask")
        else:
            text_embeds = batch.get("text")
            text_mask = batch.get("text_mask")

        if text_embeds is None:
            raise ValueError(
                "Cached text embeddings missing from batch. Expected either batch['conditions'] (official format) "
                "or 'text'/'text_mask' (legacy musubi format)."
            )

        if not isinstance(text_embeds, torch.Tensor):
            raise TypeError(f"Expected text embeddings to be a torch.Tensor, got: {type(text_embeds)}")
        if text_embeds.dim() != 3:
            raise ValueError(f"Expected text embeddings to be 3D [B, seq_len, hidden_dim], got shape: {tuple(text_embeds.shape)}")
        if text_embeds.shape[0] != latents.shape[0]:
            raise ValueError(f"Batch size mismatch: latents batch={latents.shape[0]} vs text batch={text_embeds.shape[0]}")

        text_embeds = text_embeds.to(device=accelerator.device, dtype=network_dtype)

        # Check for NaN values
        if torch.isnan(text_embeds).any():
            raise ValueError("NaN detected in cached text embeddings!")

        if text_mask is not None:
            if not isinstance(text_mask, torch.Tensor):
                raise TypeError(f"Expected text_mask to be a torch.Tensor, got: {type(text_mask)}")
            if text_mask.dim() != 2:
                raise ValueError(f"Expected text_mask to be 2D [B, seq_len], got shape: {tuple(text_mask.shape)}")
            if text_mask.shape[0] != latents.shape[0]:
                raise ValueError(f"Batch size mismatch: latents batch={latents.shape[0]} vs text_mask batch={text_mask.shape[0]}")
            text_mask = text_mask.to(device=accelerator.device)
            if args.gradient_checkpointing:
                text_mask = text_mask.to(torch.bool)

        # Move latents to device
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noise = noise.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)

        # Check for NaN in latents
        if torch.isnan(latents).any():
            raise ValueError("NaN detected in latents!")

        # Get frame rate from batch or use default
        frame_rate = batch.get("frame_rate", None)
        if frame_rate is None:
            latents_info = batch.get("latents")
            if isinstance(latents_info, dict):
                frame_rate = latents_info.get("fps", None)
        if frame_rate is None:
            frame_rate = 24
        if isinstance(frame_rate, torch.Tensor):
            frame_rate = frame_rate.item() if frame_rate.numel() == 1 else frame_rate[0].item()

        model_timesteps = timesteps.to(device=accelerator.device, dtype=network_dtype)

        model_timesteps = self._normalize_timesteps_for_model(model_timesteps)

        if model_timesteps.dim() == 0:
            model_timesteps = model_timesteps.unsqueeze(0)
        if model_timesteps.dim() == 1:
            model_timesteps = model_timesteps.unsqueeze(1)

        sigma = model_timesteps[:, 0]

        ref_latents = batch.get("ref_latents")
        if ref_latents is None:
            ref_latents = batch.get("reference_latents")
        if isinstance(ref_latents, dict):
            ref_latents = ref_latents.get("latents")

        if ref_latents is not None:
            if self._audio_video or self._ltx_mode != "video":
                raise ValueError("Reference latent conditioning is only supported for video-only LTX-2 training")
            if not isinstance(ref_latents, torch.Tensor):
                raise TypeError(f"Expected ref_latents to be a torch.Tensor, got: {type(ref_latents)}")
            if ref_latents.dim() != 5:
                raise ValueError(f"Expected ref_latents to be 5D [B, C, F, H, W], got shape: {tuple(ref_latents.shape)}")
            if ref_latents.shape[0] != latents.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: latents batch={latents.shape[0]} vs ref_latents batch={ref_latents.shape[0]}"
                )
            if ref_latents.shape[1] != latents.shape[1]:
                raise ValueError(f"Channel mismatch: latents C={latents.shape[1]} vs ref_latents C={ref_latents.shape[1]}")
            if ref_latents.shape[3] != latents.shape[3] or ref_latents.shape[4] != latents.shape[4]:
                raise ValueError(
                    f"Spatial mismatch: latents HxW={latents.shape[3]}x{latents.shape[4]} vs ref_latents HxW={ref_latents.shape[3]}x{ref_latents.shape[4]}"
                )

        caption_channels = getattr(transformer, "caption_channels", None)
        if caption_channels is None and hasattr(transformer, "caption_projection"):
            caption_channels = getattr(getattr(transformer, "caption_projection", None), "in_features", None)
        if caption_channels is not None:
            expected_last_dim = int(caption_channels) * (2 if self._audio_video else 1)
            if text_embeds.shape[-1] != expected_last_dim:
                raise ValueError(
                    f"Text embedding dim mismatch for {'LTXAV' if self._audio_video else 'LTXV'}: "
                    f"got {text_embeds.shape[-1]}, expected {expected_last_dim}. "
                    f"(caption_channels={caption_channels})"
                )

        first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
        if not (0.0 <= first_frame_p <= 1.0):
            raise ValueError(f"ltx2_first_frame_conditioning_p must be in [0,1]. Got: {first_frame_p}")

        video_conditioning_enabled = None
        if first_frame_p > 0.0:
            enable_conditioning = bool(torch.rand((), device=accelerator.device) < first_frame_p)
            if enable_conditioning:
                video_conditioning_enabled = torch.ones((latents.shape[0],), device=accelerator.device, dtype=torch.bool)

        model_noisy_video = noisy_model_input
        if video_conditioning_enabled is not None and model_noisy_video.shape[2] > 0:
            model_noisy_video = model_noisy_video.clone()
            model_noisy_video[video_conditioning_enabled, :, 0:1, :, :] = latents[video_conditioning_enabled, :, 0:1, :, :]

        if ref_latents is not None:
            from musubi_tuner.ltx_2.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
            from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
            from musubi_tuner.ltx_2.model.transformer.modality import Modality
            from musubi_tuner.ltx_2.types import SpatioTemporalScaleFactors, VideoLatentShape

            patchifier = VideoLatentPatchifier(patch_size=1)

            ref_latents = ref_latents.to(device=accelerator.device, dtype=network_dtype)
            ref_tokens = patchifier.patchify(ref_latents)
            target_tokens = patchifier.patchify(model_noisy_video)
            combined_tokens = torch.cat([ref_tokens, target_tokens], dim=1)

            bsz = combined_tokens.shape[0]
            ref_seq_len = ref_tokens.shape[1]
            target_seq_len = target_tokens.shape[1]

            height = int(ref_latents.shape[3])
            width = int(ref_latents.shape[4])

            ref_conditioning_mask = torch.ones((bsz, ref_seq_len), device=accelerator.device, dtype=torch.bool)

            target_conditioning_mask = torch.zeros((bsz, target_seq_len), device=accelerator.device, dtype=torch.bool)
            if video_conditioning_enabled is not None:
                first_frame_tokens = height * width
                if first_frame_tokens > 0:
                    target_conditioning_mask[video_conditioning_enabled, :first_frame_tokens] = True
            conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

            combined_timesteps = sigma.view(bsz, 1).expand(bsz, ref_seq_len + target_seq_len)
            combined_timesteps = torch.where(conditioning_mask, torch.zeros_like(combined_timesteps), combined_timesteps)

            frame_rate_v2v = frame_rate
            if frame_rate_v2v is None:
                frame_rate_v2v = 24

            ref_frames = int(ref_latents.shape[2])
            tgt_frames = int(latents.shape[2])

            ref_coords = patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz,
                    channels=int(ref_latents.shape[1]),
                    frames=ref_frames,
                    height=height,
                    width=width,
                ),
                device=accelerator.device,
            )
            ref_positions = get_pixel_coords(
                latent_coords=ref_coords,
                scale_factors=SpatioTemporalScaleFactors.default(),
                causal_fix=True,
            ).to(dtype=network_dtype)
            ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / float(frame_rate_v2v)

            tgt_coords = patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz,
                    channels=int(latents.shape[1]),
                    frames=tgt_frames,
                    height=height,
                    width=width,
                ),
                device=accelerator.device,
            )
            tgt_positions = get_pixel_coords(
                latent_coords=tgt_coords,
                scale_factors=SpatioTemporalScaleFactors.default(),
                causal_fix=True,
            ).to(dtype=network_dtype)
            tgt_positions[:, 0, ...] = tgt_positions[:, 0, ...] / float(frame_rate_v2v)

            combined_positions = torch.cat([ref_positions, tgt_positions], dim=2)

            video_modality = Modality(
                enabled=True,
                latent=combined_tokens,
                timesteps=combined_timesteps,
                positions=combined_positions,
                context=text_embeds,
                context_mask=text_mask,
            )

            perturbations = BatchedPerturbationConfig.empty(bsz)
            base_model = transformer.model if hasattr(transformer, "model") else transformer

            with accelerator.autocast():
                pred_tokens, _ = base_model(video_modality, None, perturbations)

            target_pred_tokens = pred_tokens[:, ref_seq_len:, :]
            target_velocity = patchifier.patchify(noise - latents)
            target_loss_mask = ~target_conditioning_mask

            out_v2v: Dict[str, Any] = {
                "video_pred": target_pred_tokens,
                "video_target": target_velocity,
                "video_loss_mask": target_loss_mask,
                "video_loss_weight": float(getattr(args, "video_loss_weight", 1.0)),
            }
            if out_v2v["video_loss_weight"] < 0.0:
                raise ValueError(f"video_loss_weight must be >= 0. Got: {out_v2v['video_loss_weight']}")

            return out_v2v, torch.tensor(0.0, device=accelerator.device)

        audio_latents = None
        audio_noise = None
        noisy_audio = None
        if self._ltx_mode == "av":
            audio_latents = batch.get("audio_latents")
            if isinstance(audio_latents, dict):
                if "latents" not in audio_latents:
                    raise ValueError("batch['audio_latents'] is a dict but missing key 'latents'")
                audio_latents = audio_latents["latents"]

            if audio_latents is None:
                raise ValueError(
                    "LTXAV training requires audio latents in batch['audio_latents']. "
                    "Run the LTX-2 latents caching script with --ltx_mode av and ensure *_ltx2_audio.safetensors exist."
                )
            if not isinstance(audio_latents, torch.Tensor):
                raise TypeError(f"Expected audio_latents to be a torch.Tensor, got: {type(audio_latents)}")
            if audio_latents.dim() != 4:
                raise ValueError(f"Expected audio_latents to be 4D [B, C, T, F], got shape: {tuple(audio_latents.shape)}")
            if audio_latents.shape[0] != latents.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: latents batch={latents.shape[0]} vs audio_latents batch={audio_latents.shape[0]}"
                )

            audio_latents = audio_latents.to(device=accelerator.device, dtype=network_dtype)
            audio_noise = torch.randn_like(audio_latents)
            sigma_audio = sigma.view(-1, 1, 1, 1)
            noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise
        elif self._ltx_mode == "audio":
            raise NotImplementedError(
                "Audio-only training mode is not implemented yet. "
                "It requires official LTXAV audio-only forward semantics and audio latent caching."
            )

        model_input = model_noisy_video
        if self._ltx_mode == "av":
            model_input = [model_noisy_video, noisy_audio]

        video_conditioning_mask_tokens = None
        video_loss_mask_frames = None
        if video_conditioning_enabled is not None:
            bsz, _c, frames, height, width = latents.shape
            seq_len = frames * height * width
            first_frame_tokens = height * width
            video_conditioning_mask_tokens = torch.zeros((bsz, seq_len), device=accelerator.device, dtype=torch.bool)
            if first_frame_tokens > 0:
                video_conditioning_mask_tokens[video_conditioning_enabled, :first_frame_tokens] = True
            transformer_options = {"patches_replace": {}, "video_conditioning_mask": video_conditioning_mask_tokens}

            video_loss_mask_frames = torch.ones((bsz, frames), device=accelerator.device, dtype=torch.bool)
            if frames > 0:
                video_loss_mask_frames[video_conditioning_enabled, 0] = False

        with accelerator.autocast():
            model_pred = transformer(
                model_input,
                timestep=model_timesteps,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=frame_rate,
                transformer_options=transformer_options if video_conditioning_mask_tokens is not None else {"patches_replace": {}},
            )

        video_pred = model_pred
        audio_pred = None
        if isinstance(model_pred, (list, tuple)):
            if len(model_pred) != 2:
                raise ValueError(f"Expected AV model to return [video_pred, audio_pred], got {len(model_pred)} outputs")
            video_pred, audio_pred = model_pred

        video_target = noise - latents

        out: Dict[str, Any] = {
            "video_pred": video_pred,
            "video_target": video_target,
            "video_loss_mask": video_loss_mask_frames,
            "video_loss_weight": float(getattr(args, "video_loss_weight", 1.0)),
        }

        if out["video_loss_weight"] < 0.0:
            raise ValueError(f"video_loss_weight must be >= 0. Got: {out['video_loss_weight']}")

        if self._ltx_mode == "av":
            if audio_pred is None:
                raise ValueError("AV mode expected an audio prediction but got None")
            audio_target = audio_noise - audio_latents

            audio_seq_len = int(audio_latents.shape[2])
            audio_loss_mask = torch.ones(
                (audio_latents.shape[0], audio_seq_len),
                device=accelerator.device,
                dtype=torch.bool,
            )

            audio_lengths = batch.get("audio_lengths")
            if isinstance(audio_lengths, dict):
                audio_lengths = audio_lengths.get("lengths")
            if isinstance(audio_lengths, torch.Tensor):
                if audio_lengths.dim() == 0:
                    audio_lengths = audio_lengths.view(1)
                if audio_lengths.dim() != 1:
                    raise ValueError(f"Expected audio_lengths to be 1D [B] or scalar, got shape: {tuple(audio_lengths.shape)}")
                if audio_lengths.numel() == 1 and audio_latents.shape[0] != 1:
                    audio_lengths = audio_lengths.expand(audio_latents.shape[0])
                if audio_lengths.shape[0] != audio_latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: audio_latents batch={audio_latents.shape[0]} vs audio_lengths batch={audio_lengths.shape[0]}"
                    )

                audio_lengths = audio_lengths.to(device=accelerator.device)
                if audio_lengths.dtype.is_floating_point:
                    audio_lengths = audio_lengths.to(dtype=torch.int64)
                else:
                    audio_lengths = audio_lengths.to(dtype=torch.int64)

                audio_lengths = audio_lengths.clamp(min=0, max=audio_seq_len)
                t = torch.arange(audio_seq_len, device=accelerator.device).view(1, -1)
                audio_loss_mask = t < audio_lengths.view(-1, 1)
            out.update(
                {
                    "audio_pred": audio_pred,
                    "audio_target": audio_target,
                    "audio_loss_mask": audio_loss_mask,
                    "audio_loss_weight": float(getattr(args, "audio_loss_weight", 1.0)),
                }
            )
            if out["audio_loss_weight"] < 0.0:
                raise ValueError(f"audio_loss_weight must be >= 0. Got: {out['audio_loss_weight']}")

        return out, torch.tensor(0.0, device=accelerator.device)

    def scale_shift_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Scale and shift latents for training (optional normalization)"""
        # LTX-2 typically doesn't require normalization, but can be enabled if needed
        return latents

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ) -> Optional[List[Dict]]:
        """Process sample prompts for inference preview during training"""
        prompts = load_prompts(sample_prompts)
        if not prompts:
            return None

        if self._text_encoder is None:
            if getattr(args, "gemma_root", None) is None:
                raise ValueError("--gemma_root is required for LTX-2 sample prompts")
            if getattr(args, "ltx2_checkpoint", None) is None:
                raise ValueError("--ltx2_checkpoint is required for LTX-2 sample prompts")
            from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
            from musubi_tuner.ltx_2.text_encoders.gemma.encoders.av_encoder import (
                AVGemmaTextEncoderModelConfigurator,
                AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            )
            from musubi_tuner.ltx_2.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root
            from musubi_tuner.ltx_2.text_encoders.gemma.encoders.video_only_encoder import (
                VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS,
                VideoGemmaTextEncoderModelConfigurator,
            )

            configurator = AVGemmaTextEncoderModelConfigurator if self._audio_video else VideoGemmaTextEncoderModelConfigurator
            key_ops = AV_GEMMA_TEXT_ENCODER_KEY_OPS if self._audio_video else VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS

            self._text_encoder = SingleGPUModelBuilder(
                model_path=str(args.ltx2_checkpoint),
                model_class_configurator=configurator,
                model_sd_ops=key_ops,
                module_ops=module_ops_from_gemma_root(args.gemma_root),
            ).build(device=accelerator.device, dtype=torch.float32)
            self._text_encoder.eval()

        def encode_prompt(prompt_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
            with accelerator.autocast(), torch.no_grad():
                out = self._text_encoder(prompt_text, padding_side="left")
                if self._audio_video:
                    embed = torch.cat([out.video_encoding, out.audio_encoding], dim=-1)
                else:
                    embed = out.video_encoding
                mask = out.attention_mask
            return embed.squeeze(0).detach().cpu(), mask.squeeze(0).detach().cpu()

        sample_parameters = []
        for prompt_data in prompts:
            prompt_text = prompt_data.get("prompt", "")
            prompt_embeds, prompt_mask = encode_prompt(prompt_text)
            param = {
                "prompt": prompt_text,
                "negative_prompt": prompt_data.get("negative_prompt", ""),
                "height": prompt_data.get("height", args.height),
                "width": prompt_data.get("width", args.width),
                "num_frames": prompt_data.get("num_frames", 45),
                "seed": prompt_data.get("seed", 0),
                "prompt_embeds": prompt_embeds,
                "prompt_attention_mask": prompt_mask,
            }

            negative_prompt = param["negative_prompt"]
            if negative_prompt:
                neg_embeds, neg_mask = encode_prompt(negative_prompt)
                param["negative_prompt_embeds"] = neg_embeds
                param["negative_prompt_attention_mask"] = neg_mask
            sample_parameters.append(param)

        return sample_parameters

    def do_inference(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
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
    ):
        """Generate sample video during training using LTX-2 denoising loop"""
        from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

        transformer_device = next(transformer.parameters()).device
        original_vae_device = getattr(vae, "device", torch.device("cpu"))
        original_vae_dtype = getattr(vae, "dtype", torch.float32)
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
        if prompt_mask is not None and prompt_mask.dim() == 1:
            prompt_mask = prompt_mask.unsqueeze(0)
        prompt_mask = prompt_mask.to(device=transformer_device, dtype=torch.int64) if prompt_mask is not None else None

        # Setup scheduler
        scheduler = FlowMatchDiscreteScheduler(shift=discrete_flow_shift or 1.0)
        scheduler.set_timesteps(sample_steps, device=transformer_device)
        timesteps = scheduler.timesteps

        # Calculate latent dimensions
        vae_scale_factor_temporal = getattr(vae, "temporal_downsample_factor", 4)
        vae_scale_factor_spatial = getattr(vae, "spatial_downsample_factor", 8)
        latent_frames = (frame_count - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial
        in_channels = getattr(transformer, "in_channels", 128)

        # Initialize latents
        latents = torch.randn(
            (1, int(in_channels), latent_frames, latent_height, latent_width),
            dtype=torch.float32,
            device=transformer_device,
            generator=generator,
        )

        # Denoising loop
        with torch.no_grad():
            for t in tqdm(timesteps, desc="LTX-2 preview", leave=False):
                # Expand for CFG if needed
                latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dtype=dit_dtype)

                # Prepare timestep
                timestep_tensor = t.expand(latent_model_input.shape[0]).to(device=transformer_device, dtype=dit_dtype)
                timestep_for_model = self._normalize_timesteps_for_model(timestep_tensor)

                # Model prediction
                # Handle LTXAV model input format
                if self._audio_video:
                    model_input = [latent_model_input]  # Video only for inference
                else:
                    model_input = latent_model_input

                model_pred = transformer(
                    model_input,
                    timestep=timestep_for_model.unsqueeze(1),  # [B, 1] for per-token timesteps
                    context=prompt_embeds,
                    attention_mask=prompt_mask,
                    frame_rate=sample_parameter.get("frame_rate", 25),
                    transformer_options={},
                )

                # Handle LTXAV output format
                if isinstance(model_pred, (list, tuple)):
                    model_pred = model_pred[0]  # Use video prediction

                # Apply guidance if needed
                if do_classifier_free_guidance:
                    noise_uncond, noise_cond = model_pred.chunk(2)
                    model_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

                model_pred = model_pred.to(dtype=latents.dtype)
                latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]

        # Decode latents
        with torch.no_grad():
            video = vae.decode([latents.squeeze(0)])
            if isinstance(video, list) and video:
                video = video[0]
                if video.dim() == 4:  # [C, T, H, W]
                    video = video.unsqueeze(0)  # [1, C, T, H, W]

        # Normalize to [0, 1]
        video = (video / 2 + 0.5).clamp(0, 1).to(torch.float32).to("cpu")

        # Restore VAE state
        vae.to_device(original_vae_device)
        vae.to_dtype(original_vae_dtype)

        return video


# ======== Argument parser setup ========


def ltx2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add LTX-2-specific arguments to parser"""

    parser.add_argument(
        "--ltx2_checkpoint",
        type=str,
        required=True,
        help="Path to LTX-2 checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--gemma_root",
        type=str,
        default=None,
        help="Local directory containing Gemma weights/tokenizer (used for sample prompts)",
    )
    parser.add_argument(
        "--network_module",
        type=str,
        default="musubi_tuner.networks.lora_ltx2",
        help="Network module for LoRA (default: lora_ltx2)",
    )
    parser.add_argument(
        "--ltx_mode",
        type=str,
        default="video",
        choices=["video", "av", "audio"],
        help="Training modality.",
    )
    parser.add_argument(
        "--video_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the video diffusion loss.",
    )
    parser.add_argument(
        "--audio_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the audio diffusion loss.",
    )

    parser.add_argument(
        "--ltx2_first_frame_conditioning_p",
        type=float,
        default=0.1,
        help="Probability of first-frame conditioning during training (keep frame 0 clean and set its timestep to 0).",
    )

    return parser


# ======== Main training entry point ========


def main() -> None:
    """Main training entry point"""
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if getattr(args, "dit", None) is not None and args.dit != args.ltx2_checkpoint:
        logger.warning("Ignoring --dit for LTX-2; using --ltx2_checkpoint instead")
    args.dit = args.ltx2_checkpoint

    if getattr(args, "vae", None) is not None and args.vae != args.ltx2_checkpoint:
        logger.warning("Ignoring --vae for LTX-2; using --ltx2_checkpoint instead")
    args.vae = args.ltx2_checkpoint

    if getattr(args, "weighting_scheme", None) not in {None, "none"}:
        logger.warning("Ignoring --weighting_scheme for LTX-2; forcing weighting_scheme=none")
    args.weighting_scheme = "none"

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = LTX2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
