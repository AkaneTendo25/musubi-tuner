"""
LTXV2 LoRA Training Implementation

LTXV2 is a diffusion transformer for video generation from the Lightricks team.
Supports both pure video (LTXV) and audio-video (LTXAV) models.
"""

import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import math
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, T5EncoderModel
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

# LTXV2 architecture constants
ARCHITECTURE_LTXV2 = "ltxv2"
ARCHITECTURE_LTXV2_FULL = "ltxv2_v1"

# LTXV2 latent normalization constants (from WAN VAE)
# Mean and std for 128-channel latents - broadcasting handled automatically
# For single channel: [0.0] broadcasts to [1, 1, 1, 1, 1] shape
LTXV2_LATENTS_MEAN = [0.0]
LTXV2_LATENTS_STD = [1.0]

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


def detect_ltxv2_dtype(model_path: str) -> torch.dtype:
    """Detect the data type of LTXV2 model weights"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LTXV2 weights must be a .safetensors file. Got: {model_path}")

    with MemoryEfficientSafeOpen(model_path) as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"Unable to detect LTXV2 dtype; no tensors found in {model_path}")

        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point():
                dtype = tensor.dtype
                break
        else:
            dtype = handle.get_tensor(keys[0]).dtype

    logger.info("Detected LTXV2 dtype: %s", dtype)
    return dtype


def detect_ltxv2_config(model_path: str) -> Dict[str, Any]:
    """Infer LTXV2 model configuration from weights."""
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


def load_ltxv2_model(
    model_path: str,
    device: Union[str, torch.device] = "cpu",
    load_device: Union[str, torch.device] = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    attn_mode: str = "torch",
    audio_video: bool = False,
    backend: str = "official",
    **_: Any,
):
    """Load LTXV2 or LTXAV model

    Args:
        model_path: Path to safetensors model weights
        device: Target device for model
        load_device: Device to load weights into
        torch_dtype: Data type for model parameters
        attn_mode: Attention implementation (torch, flash, flash3, xformers)
        audio_video: If True, load LTXAV model; if False, load LTXV model
        **_: Additional arguments (ignored)

    Returns:
        Loaded LTXV2 or LTXAV transformer model
    """
    target_device = torch.device(device)
    load_device = torch.device(load_device)

    from musubi_tuner.networks.lora_ltxv2 import load_official_ltxv2_wrapper

    if backend != "official":
        raise ValueError(f"Unsupported LTXV2 backend: {backend}. Only 'official' is supported.")

    logger.info("Loading LTXV2 transformer via OfficialLTXV2Wrapper: %s", model_path)
    model = load_official_ltxv2_wrapper(
        model_path,
        device=load_device,
        dtype=torch_dtype or torch.float32,
        audio_video=audio_video,
        patch_size=1,
    )
    model = model.to(device=target_device)
    return model


class LTXV2NetworkTrainer(NetworkTrainer):
    """Trainer for LTXV2 models with LoRA support"""

    def __init__(self) -> None:
        super().__init__()
        self._tokenizer: Optional[AutoTokenizer] = None
        self._text_encoder: Optional[T5EncoderModel] = None
        self._dit_attn_mode: Optional[str] = None
        self._latent_norm_cache: Dict = {}

        # Initialize latent normalization
        mean = torch.tensor(LTXV2_LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(LTXV2_LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = std.clamp_min(1e-6)
        self._latent_norm_base: Tuple[torch.Tensor, torch.Tensor] = (mean, std.reciprocal())

        self._flow_target: str = "noise"  # LTXV2 predicts noise
        self._num_timesteps: int = 1000
        self._audio_video: bool = False
        self._ltxv2_backend: str = "official"
        self._ltx_mode: str = "video"
        self._ltxv2_timestep_format: Optional[str] = None
        self.default_guidance_scale = 1.0

    def _normalize_timesteps_for_model(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Normalize timesteps to the model's expected 0..1 range.

        The training loop provides timesteps either from a scheduler (common) or pre-generated pools.
        Since upstream implementations differ, we support an explicit format flag.

        Supported:
        - flowmatch_1_1000: integer-like in [1, 1000]
        - legacy_0_1000: integer-like in [0, 1000]
        - sd3_0_1: already normalized in [0, 1]
        """
        if timesteps.numel() == 0:
            return timesteps

        ts = timesteps
        fmt = self._ltxv2_timestep_format

        if fmt == "flowmatch_1_1000":
            ts = (ts - 1.0) / 1000.0
        elif fmt == "legacy_0_1000":
            ts = ts / 1000.0
        elif fmt == "sd3_0_1":
            ts = ts
        else:
            raise ValueError(f"Unknown ltxv2_timestep_format: {fmt}")

        return ts

    # ======== Model-specific properties and configuration ========

    @property
    def architecture(self) -> str:
        """Returns architecture identifier"""
        return ARCHITECTURE_LTXV2

    @property
    def architecture_full_name(self) -> str:
        """Returns full architecture name with version"""
        return ARCHITECTURE_LTXV2_FULL

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        """Handle LTXV2-specific command line arguments"""
        self.dit_dtype = detect_ltxv2_dtype(args.ltxv2_model)

        if self.dit_dtype == torch.float16:
            assert args.mixed_precision in ["fp16", "no"], "LTXV2 weights are fp16; mixed precision must be fp16 or no"
        elif self.dit_dtype == torch.bfloat16:
            assert args.mixed_precision in ["bf16", "no"], "LTXV2 weights are bf16; mixed precision must be bf16 or no"

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        ltx_mode = getattr(args, "ltx_mode", None)
        if ltx_mode is None:
            ltx_mode = "av" if getattr(args, "ltxv2_audio_video", False) else "video"
        if ltx_mode not in {"video", "av", "audio"}:
            raise ValueError(f"Invalid ltx_mode: {ltx_mode}")
        self._ltx_mode = ltx_mode
        self._audio_video = self._ltx_mode == "av"

        self._ltxv2_backend = getattr(args, "ltxv2_backend", "official")
        self._ltxv2_timestep_format = getattr(args, "ltxv2_timestep_format", None)
        if self._ltxv2_timestep_format is None:
            raise ValueError("--ltxv2_timestep_format is required for LTXV2. Please set it explicitly.")
        self.default_guidance_scale = 1.0

    @property
    def i2v_training(self) -> bool:
        """LTXV2 doesn't currently support I2V conditioning"""
        return False

    @property
    def control_training(self) -> bool:
        """LTXV2 doesn't currently support control conditioning"""
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
        """Load LTXV2 transformer model

        Args:
            accelerator: HF Accelerator instance
            args: Training arguments
            dit_path: Path to LTXV2 weights
            attn_mode: Attention implementation
            split_attn: Whether to split attention (ignored for LTXV2)
            loading_device: Device to load weights to
            dit_weight_dtype: Weight data type

        Returns:
            Loaded LTXV2 transformer model
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

        transformer = load_ltxv2_model(
            model_path=dit_path,
            device=accelerator.device,
            load_device=loading_device,
            torch_dtype=dit_weight_dtype,
            attn_mode=attn_mode,
            audio_video=self._audio_video,
            backend=self._ltxv2_backend,
        )

        transformer.eval()
        transformer.requires_grad_(False)

        return transformer

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        """Load VAE for LTXV2 (compatible with WAN VAE)"""
        logger.info(f"Loading VAE from {vae_path}")
        try:
            from musubi_tuner.wan.modules.vae import WanVAE
        except ImportError:
            raise ImportError("WAN VAE not available. Ensure musubi-tuner is properly configured.")

        vae = WanVAE(vae_path=vae_path, device="cpu", dtype=vae_dtype)

        # Update latent normalization from VAE if available
        self._update_latent_norm_base_from_vae(vae)

        return vae

    def _update_latent_norm_base_from_vae(self, vae) -> None:
        """Update latent normalization statistics from VAE config"""
        latents_mean = getattr(vae, "latents_mean", None)
        latents_std = getattr(vae, "latents_std", None)

        if latents_mean is None or latents_std is None:
            # WanVAE wrapper exposes mean/std instead of latents_mean/latents_std
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

        mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1).clamp_min(1e-6)
        self._latent_norm_base = (mean, std.reciprocal())
        self._latent_norm_cache.clear()

    def on_load_text_encoder(
        self,
        args: argparse.Namespace,
        tokenizer,
        text_encoder,
    ) -> None:
        """Store text encoder and tokenizer for inference"""
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through LTXV2/LTXAV model

        Args:
            args: Training arguments
            accelerator: HF Accelerator
            transformer: LTXV2/LTXAV model
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

        text_embeds = batch.get("text")
        if text_embeds is None:
            text_embeds = batch.get("t5")
            if text_embeds is not None and not self._warned_legacy_text_cache:
                logger.warning("Using legacy cached text key 't5'/'t5_mask'. Please recache to 'text'/'text_mask' for LTXV2.")
                self._warned_legacy_text_cache = True
        if text_embeds is None:
            raise ValueError(
                "Cached text embeddings missing from batch. Expected 'text' (preferred) or legacy 't5'. "
                "Please run ltxv2_cache_text_encoder_outputs.py with the correct backend."
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
            raise ValueError("NaN detected in cached T5 embeddings!")

        text_mask = batch.get("text_mask")
        if text_mask is None:
            text_mask = batch.get("t5_mask")
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
        frame_rate = batch.get("frame_rate", 25)
        if isinstance(frame_rate, torch.Tensor):
            frame_rate = frame_rate.item() if frame_rate.numel() == 1 else frame_rate[0].item()

        model_timesteps = timesteps.to(device=accelerator.device, dtype=network_dtype)

        model_timesteps = self._normalize_timesteps_for_model(model_timesteps)

        # Ensure timesteps have correct shape [B] or [B, 1]
        if model_timesteps.dim() == 0:
            model_timesteps = model_timesteps.unsqueeze(0)
        if model_timesteps.dim() == 1:
            model_timesteps = model_timesteps.unsqueeze(1)  # [B, 1] for per-token timesteps

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

        # For LTXAV (audio-video), handle audio latents
        model_input = noisy_model_input
        if self._ltx_mode == "av":
            audio_latents = batch.get("audio_latents")
            if audio_latents is not None:
                if not isinstance(audio_latents, torch.Tensor):
                    raise TypeError(f"Expected audio_latents to be a torch.Tensor, got: {type(audio_latents)}")
                if audio_latents.dim() != 4:
                    raise ValueError(
                        f"Expected audio_latents to be 4D [B, C, T, F] (or similar), got shape: {tuple(audio_latents.shape)}"
                    )
                if audio_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs audio_latents batch={audio_latents.shape[0]}"
                    )
                audio_latents = audio_latents.to(device=accelerator.device, dtype=network_dtype)
                model_input = [noisy_model_input, audio_latents]
            else:
                raise NotImplementedError(
                    "LTXAV training requires audio latents in batch['audio_latents'] and matching text embeddings. "
                    "Current dataset pipeline does not provide audio latents."
                )
        elif self._ltx_mode == "audio":
            raise NotImplementedError(
                "Audio-only training mode is not implemented yet. "
                "It requires official LTXAV audio-only forward semantics and audio latent caching."
            )

        with accelerator.autocast():
            model_pred = transformer(
                model_input,
                timestep=model_timesteps,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=frame_rate,
                transformer_options={"patches_replace": {}},
            )

        # Handle output format for LTXAV / wrapper
        if isinstance(model_pred, (list, tuple)):
            model_pred = model_pred[0]

        target = noise - latents

        if self._ltx_mode in {"video", "av"}:
            weight = float(getattr(args, "video_loss_weight", 1.0))
            if weight < 0.0:
                raise ValueError(f"video_loss_weight must be >= 0. Got: {weight}")
            if weight != 1.0:
                scale = math.sqrt(weight)
                model_pred = model_pred * scale
                target = target * scale

        return model_pred, target

    def scale_shift_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Scale and shift latents for training (optional normalization)"""
        # LTXV2 typically doesn't require normalization, but can be enabled if needed
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

        if self._tokenizer is None:
            text_backend = getattr(args, "text_encoder_backend", "t5")
            tokenizer_name = getattr(args, "tokenizer", None) or getattr(args, "text_encoder", None)
            if text_backend == "gemma" and tokenizer_name == "google/t5-v1_1-xxl":
                tokenizer_name = args.text_encoder

            logger.info("Loading tokenizer for sample prompts: %s", tokenizer_name)
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self._tokenizer.padding_side = "right"
        if self._text_encoder is None:
            text_backend = getattr(args, "text_encoder_backend", "t5")
            logger.info("Loading text encoder for sample prompts: %s (backend=%s)", args.text_encoder, text_backend)
            if text_backend == "t5":
                self._text_encoder = T5EncoderModel.from_pretrained(
                    args.text_encoder,
                    torch_dtype=torch.float32,
                    device_map=str(accelerator.device),
                )
            elif text_backend == "gemma":
                try:
                    self._text_encoder = AutoModel.from_pretrained(
                        args.text_encoder,
                        torch_dtype=torch.float32,
                        device_map=str(accelerator.device),
                    )
                except Exception:
                    self._text_encoder = AutoModelForCausalLM.from_pretrained(
                        args.text_encoder,
                        torch_dtype=torch.float32,
                        device_map=str(accelerator.device),
                    )
            else:
                raise ValueError(f"Unsupported text_encoder_backend for LTXV2 sample prompts: {text_backend}")
            self._text_encoder.eval()

        text_max_length = getattr(args, "max_length", None)
        if text_max_length is None:
            text_max_length = getattr(args, "text_max_length", 256)

        def _encode_hidden_states(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            outputs = model(input_ids=ids, attention_mask=mask, output_hidden_states=False, return_dict=True)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return outputs.last_hidden_state
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                return outputs.hidden_states[-1]
            if hasattr(model, "model"):
                inner = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=False, return_dict=True)
                if hasattr(inner, "last_hidden_state") and inner.last_hidden_state is not None:
                    return inner.last_hidden_state
            raise RuntimeError("Unable to extract last hidden states from the selected text encoder model")

        def encode_prompt(prompt_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
            tokens = self._tokenizer(
                prompt_text,
                max_length=text_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device=accelerator.device)
            attention_mask = tokens["attention_mask"].to(device=accelerator.device)
            with accelerator.autocast(), torch.no_grad():
                hidden = _encode_hidden_states(self._text_encoder, input_ids, attention_mask)
            return hidden.detach().cpu(), attention_mask.detach().cpu()

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
        """Generate sample video during training using LTXV2 denoising loop"""
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
            for t in tqdm(timesteps, desc="LTXV2 preview", leave=False):
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


def ltxv2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add LTXV2-specific arguments to parser"""

    def _ensure_arg(name: str, *, required: bool = False, **updates) -> None:
        """Add or update argument in parser"""
        action = parser._option_string_actions.get(name)
        if action is None:
            parser.add_argument(name, required=required, **updates)
        else:
            if required:
                action.required = True
            for key, value in updates.items():
                setattr(action, key, value)

    _ensure_arg(
        "--ltxv2_model",
        required=True,
        type=str,
        help="Path to LTXV2 model weights (.safetensors)",
    )
    _ensure_arg(
        "--vae",
        required=True,
        type=str,
        help="Path to VAE weights directory (WAN VAE compatible)",
    )
    _ensure_arg(
        "--tokenizer",
        type=str,
        default="google/t5-v1_1-xxl",
        help="T5 tokenizer repo or path",
    )
    _ensure_arg(
        "--text_encoder",
        required=True,
        type=str,
        help="T5 text encoder weights (HF folder or safetensors)",
    )
    _ensure_arg(
        "--network_module",
        type=str,
        default="musubi_tuner.networks.lora_ltxv2",
        help="Network module for LoRA (default: lora_ltxv2)",
    )
    parser.add_argument(
        "--ltx_mode",
        type=str,
        default=None,
        choices=["video", "av", "audio"],
        help="Training modality. If not set, derives from --ltxv2_audio_video.",
    )
    parser.add_argument(
        "--ltxv2_audio_video",
        action="store_true",
        help="Use LTXAV model for audio-video generation",
    )
    parser.add_argument(
        "--ltxv2_backend",
        type=str,
        default="official",
        choices=["official"],
        help="Backend for loading LTXV2 transformer.",
    )

    parser.add_argument(
        "--ltxv2_timestep_format",
        type=str,
        default=None,
        required=True,
        choices=["flowmatch_1_1000", "legacy_0_1000", "sd3_0_1"],
        help="Timestep format fed to LTXV2 model. Must be set explicitly.",
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
        help="Weight applied to the audio diffusion loss (reserved; audio loss not implemented yet).",
    )

    return parser


# ======== Main training entry point ========


def main() -> None:
    """Main training entry point"""
    parser = setup_parser_common()
    parser = ltxv2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = LTXV2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
