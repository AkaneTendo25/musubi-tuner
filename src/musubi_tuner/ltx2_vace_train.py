"""
VACE training script for LTX-2.

Subclasses LTX2NetworkTrainer to add VACE context encoder support.
The base DiT stays frozen; only VACE parameters are trained (full or LoRA).

Supports:
  - Video VACE: spatial-temporal control hints for video generation
    (inpaint, outpaint, extension, depth/pose control, reference)
  - Audio VACE: temporal control hints for audio generation
    (audio inpainting, extension, reference-based generation)
  - Joint AV inpainting: shared temporal masks across both modalities
  - First-frame (I2V) + VACE: both use independent transformer_options
    entries (I2V modifies timesteps, VACE adds residual hints)

Usage:
    accelerate launch ltx2_vace_train.py \
        --ltx2_checkpoint /path/to/ltx-2.safetensors \
        --vace_layers "0,4,8,12,16,20,24,28,32,36,40,44" \
        --vace_scale 1.0 \
        --lora_target_preset vace \
        --dataset_config dataset.toml \
        ...

    For audio VACE, add --audio_vace_scale and set audio_vace_directory
    in the dataset TOML. For audio cross-attention in video VACE blocks,
    add --enable_audio_xattn_in_vace.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator

from musubi_tuner.ltx2_train_network import (
    LTX2NetworkTrainer,
    ltx2_setup_parser,
    apply_ltx2_tweaks,
)
from musubi_tuner.hv_train_network import setup_parser as setup_parser_common, read_config_from_file
from musubi_tuner.ltx_vace.vace_model import VaceLTXModel, AudioVaceLTXModel, DEFAULT_VACE_LAYERS

logger = logging.getLogger(__name__)


class LTX2VaceTrainer(LTX2NetworkTrainer):
    """Trainer for LTX-2 with VACE context encoder."""

    def __init__(self) -> None:
        super().__init__()
        self._vace_model: Optional[VaceLTXModel] = None
        self._vace_scale: float = 1.0
        self._vace_layers: Tuple[int, ...] = DEFAULT_VACE_LAYERS
        self._vace_freeze_dit: bool = True
        self._audio_vace_model: Optional[AudioVaceLTXModel] = None
        self._audio_vace_scale: float = 1.0
        self._audio_vace_layers: Tuple[int, ...] = DEFAULT_VACE_LAYERS
        self._enable_audio_xattn_in_vace: bool = False
        self._video_patchifier = None  # Set during load_transformer for correct VACE patchification

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        """Add VACE-specific argument handling on top of LTX-2's."""
        super().handle_model_specific_args(args)

        # Parse VACE layers
        vace_layers_str = getattr(args, "vace_layers", None)
        if vace_layers_str:
            self._vace_layers = tuple(int(x.strip()) for x in vace_layers_str.split(","))
        else:
            self._vace_layers = DEFAULT_VACE_LAYERS

        self._vace_scale = float(getattr(args, "vace_scale", 1.0))
        self._vace_freeze_dit = bool(getattr(args, "vace_freeze_dit", True))

        # Audio VACE args
        audio_vace_layers_str = getattr(args, "audio_vace_layers", None)
        if audio_vace_layers_str:
            self._audio_vace_layers = tuple(int(x.strip()) for x in audio_vace_layers_str.split(","))
        else:
            self._audio_vace_layers = DEFAULT_VACE_LAYERS

        self._audio_vace_scale = float(getattr(args, "audio_vace_scale", 1.0))
        self._enable_audio_xattn_in_vace = bool(getattr(args, "enable_audio_xattn_in_vace", False))

        # Default to vace_gaussian weighting if not explicitly set
        if not getattr(args, "weighting_scheme", None):
            args.weighting_scheme = "vace_gaussian"
            logger.info("Defaulting to 'vace_gaussian' weighting scheme")

        logger.info(
            "VACE config: layers=%s scale=%.2f freeze_dit=%s weighting=%s "
            "audio_xattn=%s audio_vace_layers=%s audio_vace_scale=%.2f",
            self._vace_layers,
            self._vace_scale,
            self._vace_freeze_dit,
            getattr(args, "weighting_scheme", "none"),
            self._enable_audio_xattn_in_vace,
            self._audio_vace_layers,
            self._audio_vace_scale,
        )

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype=None,
    ):
        """Load base DiT + initialize VACE model."""
        # Load base LTX-2 transformer
        transformer = super().load_transformer(
            accelerator, args, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype
        )

        # Store video patchifier for correct VACE token sequence length
        if hasattr(transformer, "_video_patchifier"):
            self._video_patchifier = transformer._video_patchifier

        # Get model dimensions from the loaded transformer
        model = transformer.model if hasattr(transformer, "model") else transformer
        dim = getattr(model, "inner_dim", 4096)
        context_dim = getattr(model, "cross_attention_dim", dim)

        # Determine VAE latent channels for vace_in_dim calculation
        # LTX-2 VAE compresses spatially by 32x, so mask channels = 32*32 = 1024
        in_channels = getattr(model, "in_channels", 128)
        from musubi_tuner.ltx_vace.vace_control_encoder import LTX2_VAE_SPATIAL_COMPRESSION
        mask_channels = LTX2_VAE_SPATIAL_COMPRESSION ** 2  # 32*32 = 1024
        base_vace_channels = 2 * in_channels + mask_channels

        # Account for spatial patch folding: patchifier folds patch_size^2 spatial
        # elements into the channel dim, so vace_in_dim scales by prod(patch_size).
        patch_size_product = 1
        if self._video_patchifier is not None:
            for ps in self._video_patchifier.patch_size:
                patch_size_product *= ps
        vace_in_dim = base_vace_channels * patch_size_product
        logger.info(
            "VACE model init: dim=%d context_dim=%d vace_in_dim=%d (base=%d * patch=%d) num_blocks=%d",
            dim, context_dim, vace_in_dim, base_vace_channels, patch_size_product, len(self._vace_layers),
        )

        # Determine audio_context_dim for video VACE (if audio x-attn enabled)
        audio_context_dim_for_vace = None
        if self._enable_audio_xattn_in_vace:
            audio_context_dim_for_vace = getattr(model, "audio_inner_dim", None)
            if audio_context_dim_for_vace is None:
                # Derive from audio_patchify_proj if available
                audio_patchify = getattr(model, "audio_patchify_proj", None)
                if audio_patchify is not None:
                    audio_context_dim_for_vace = audio_patchify.weight.shape[0]
            if audio_context_dim_for_vace is not None:
                logger.info("Enabling audio cross-attention in video VACE (audio_context_dim=%d)", audio_context_dim_for_vace)
            else:
                logger.warning("--enable_audio_xattn_in_vace set but model has no audio branch; ignoring")

        # Initialize VACE model
        self._vace_model = VaceLTXModel(
            vace_layers=self._vace_layers,
            vace_in_dim=vace_in_dim,
            latent_channels=in_channels,
            dim=dim,
            context_dim=context_dim,
            audio_context_dim=audio_context_dim_for_vace,
        )
        self._vace_model.initialize_input_proj_from_patchify_proj(model.patchify_proj)

        # Load VACE weights if provided
        vace_path = getattr(args, "vace_model_path", None)
        if vace_path and os.path.exists(vace_path):
            from safetensors.torch import load_file
            vace_state_dict = load_file(vace_path)
            self._vace_model.load_state_dict(vace_state_dict, strict=False)
            logger.info("Loaded VACE weights from %s", vace_path)
        else:
            logger.info("Initializing VACE model from scratch")

        # Move VACE model to the same device/dtype as transformer
        if dit_weight_dtype is not None:
            self._vace_model = self._vace_model.to(dtype=dit_weight_dtype)

        # Attach VACE model to LTXModel so forward() can access it
        ltx_model = transformer.model if hasattr(transformer, "model") else transformer
        ltx_model._vace_model = self._vace_model
        logger.info("Attached VACE model to LTXModel for forward pass integration")

        # --- Audio VACE model (optional) ---
        audio_vace_path = getattr(args, "audio_vace_model_path", None)
        audio_patchify = getattr(model, "audio_patchify_proj", None)
        if audio_patchify is None and audio_vace_path:
            logger.warning(
                "--audio_vace_model_path was set but loaded model has no audio branch; "
                "audio VACE will be skipped"
            )
        # Enable audio VACE if model has audio branch AND audio VACE is requested
        # (either via weights path or by the presence of audio_vace configs which
        # will be checked later during dataset setup)
        enable_audio_vace = audio_vace_path is not None or self._enable_audio_xattn_in_vace
        if audio_patchify is not None and enable_audio_vace:
            audio_in_channels = audio_patchify.weight.shape[1]
            audio_inner_dim = audio_patchify.weight.shape[0]
            # Determine audio cross-attention dim from model
            audio_cross_attn_dim = getattr(model, "audio_cross_attention_dim", audio_inner_dim)
            # Determine number of heads and d_head for audio
            audio_num_heads = getattr(model, "audio_num_attention_heads", 32)
            audio_d_head = audio_inner_dim // audio_num_heads

            audio_vace_in_dim = 2 * audio_in_channels + 1  # 257 for default 128-ch audio
            logger.info(
                "Audio VACE model init: dim=%d context_dim=%d vace_in_dim=%d num_blocks=%d",
                audio_inner_dim, audio_cross_attn_dim, audio_vace_in_dim, len(self._audio_vace_layers),
            )

            self._audio_vace_model = AudioVaceLTXModel(
                vace_layers=self._audio_vace_layers,
                vace_in_dim=audio_vace_in_dim,
                latent_channels=audio_in_channels,
                dim=audio_inner_dim,
                num_heads=audio_num_heads,
                d_head=audio_d_head,
                context_dim=audio_cross_attn_dim,
            )
            self._audio_vace_model.initialize_input_proj_from_audio_patchify_proj(audio_patchify)

            # Load audio VACE weights if provided
            if audio_vace_path and os.path.exists(audio_vace_path):
                from safetensors.torch import load_file
                audio_vace_sd = load_file(audio_vace_path)
                self._audio_vace_model.load_state_dict(audio_vace_sd, strict=False)
                logger.info("Loaded audio VACE weights from %s", audio_vace_path)
            else:
                logger.info("Initializing audio VACE model from scratch")

            if dit_weight_dtype is not None:
                self._audio_vace_model = self._audio_vace_model.to(dtype=dit_weight_dtype)

            ltx_model._audio_vace_model = self._audio_vace_model
            logger.info("Attached audio VACE model to LTXModel")

        return transformer

    def pre_train_hook(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer=None,
        network=None,
    ) -> None:
        """Freeze base DiT, prepare VACE model for training."""
        super().pre_train_hook(args, accelerator, transformer, network)

        if self._vace_freeze_dit and transformer is not None:
            # Freeze all base DiT parameters (excluding VACE models which are child modules)
            model = transformer.model if hasattr(transformer, "model") else transformer
            frozen_count = 0
            for name, param in model.named_parameters():
                if not name.startswith("_vace_model.") and not name.startswith("_audio_vace_model."):
                    param.requires_grad = False
                    frozen_count += 1
            logger.info("Frozen %d base DiT parameters (excluded VACE models)", frozen_count)

        if self._vace_model is not None:
            # Enable gradient checkpointing for VACE if requested
            if getattr(args, "gradient_checkpointing", False):
                self._vace_model.enable_gradient_checkpointing()

            # Move VACE to accelerator device
            self._vace_model = self._vace_model.to(accelerator.device)
            self._vace_model.train()

            # Make VACE parameters trainable
            for param in self._vace_model.parameters():
                param.requires_grad = True

            vace_params = sum(p.numel() for p in self._vace_model.parameters() if p.requires_grad)
            logger.info("VACE trainable parameters: %d (%.2f MB)", vace_params, vace_params * 2 / 1024 / 1024)

        if self._audio_vace_model is not None:
            if getattr(args, "gradient_checkpointing", False):
                self._audio_vace_model.enable_gradient_checkpointing()

            self._audio_vace_model = self._audio_vace_model.to(accelerator.device)
            self._audio_vace_model.train()

            for param in self._audio_vace_model.parameters():
                param.requires_grad = True

            audio_vace_params = sum(p.numel() for p in self._audio_vace_model.parameters() if p.requires_grad)
            logger.info("Audio VACE trainable parameters: %d (%.2f MB)", audio_vace_params, audio_vace_params * 2 / 1024 / 1024)

    def get_vace_trainable_params(self) -> list[torch.nn.Parameter]:
        """Return VACE trainable parameters for optimizer registration."""
        params = []
        if self._vace_model is not None:
            params.extend(p for p in self._vace_model.parameters() if p.requires_grad)
        if self._audio_vace_model is not None:
            params.extend(p for p in self._audio_vace_model.parameters() if p.requires_grad)
        return params

    def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False):
        """Save VACE model weights alongside each LoRA checkpoint."""
        super().post_save_checkpoint_hook(args, ckpt_file, ckpt_name, accelerator, force_sync_upload)
        if self._vace_model is not None:
            from safetensors.torch import save_file
            vace_path = ckpt_file.replace(".safetensors", "_vace.safetensors")
            state_dict = {k: v.contiguous().cpu() for k, v in self._vace_model.state_dict().items()}
            save_file(state_dict, vace_path)
            logger.info("Saved VACE weights: %s", vace_path)
        if self._audio_vace_model is not None:
            from safetensors.torch import save_file
            audio_vace_path = ckpt_file.replace(".safetensors", "_audio_vace.safetensors")
            state_dict = {k: v.contiguous().cpu() for k, v in self._audio_vace_model.state_dict().items()}
            save_file(state_dict, audio_vace_path)
            logger.info("Saved audio VACE weights: %s", audio_vace_path)

    def _pre_transformer_call_hook(
        self,
        transformer_options: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        model_input,
        model_timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask,
        accelerator: Accelerator,
        network_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """Pass VACE context to transformer_options for processing inside LTXModel.forward().

        VACE hint computation now happens inside LTXModel.forward() where it has
        access to the DiT's preprocessed video_args (projected tokens, AdaLN
        embeddings, RoPE). This hook just prepares and passes the raw VACE tokens.
        """
        transformer_options = dict(transformer_options)

        # Extract VACE latents from batch
        vace_latents = batch.get("vace_latents")
        if isinstance(vace_latents, dict):
            vace_latents = vace_latents.get("latents")

        if vace_latents is not None and self._vace_model is not None:
            vace_latents = vace_latents.to(device=accelerator.device, dtype=network_dtype)

            # Patchify: (B, C, F, H, W) -> (B, seq_len, C * prod(patch_size))
            from musubi_tuner.ltx_vace.vace_control_encoder import patchify_vace_context
            vace_tokens = patchify_vace_context(vace_latents, patchifier=self._video_patchifier)

            # Pass to LTXModel.forward() which computes hints using preprocessed video_args
            transformer_options["vace_context"] = vace_tokens
            transformer_options["vace_scale"] = self._vace_scale

        # Audio VACE context — build in token space (post-patchify)
        audio_vace_latents = batch.get("audio_vace_latents")
        if isinstance(audio_vace_latents, dict):
            audio_vace_latents = audio_vace_latents.get("latents")

        audio_vace_mask = batch.get("audio_vace_mask")

        if audio_vace_latents is not None and self._audio_vace_model is not None:
            audio_vace_latents = audio_vace_latents.to(device=accelerator.device, dtype=network_dtype)

            # Patchify raw 4D latents to tokens: (B, C, T, F) -> (B, T, C*F)
            from musubi_tuner.ltx_vace.vace_control_encoder import (
                patchify_audio_latents_for_vace, prepare_audio_vace_context,
            )
            audio_tokens = patchify_audio_latents_for_vace(audio_vace_latents)

            # Build temporal mask (B, T, 1)
            if audio_vace_mask is not None:
                temporal_mask = audio_vace_mask.to(device=accelerator.device, dtype=network_dtype)
                if temporal_mask.dim() == 2:  # (B, T) -> (B, T, 1)
                    temporal_mask = temporal_mask.unsqueeze(-1)
            else:
                # Default: all-ones (generate everything)
                temporal_mask = torch.ones(
                    audio_tokens.shape[0], audio_tokens.shape[1], 1,
                    device=accelerator.device, dtype=network_dtype,
                )

            # Build audio VACE context in token space: (B, T, 2*audio_in_channels + 1)
            audio_vace_context = prepare_audio_vace_context(audio_tokens, temporal_mask)

            transformer_options["audio_vace_context"] = audio_vace_context
            transformer_options["audio_vace_scale"] = self._audio_vace_scale

        return transformer_options


def vace_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add VACE-specific arguments."""
    vace_group = parser.add_argument_group("VACE Training")
    vace_group.add_argument(
        "--vace_model_path",
        type=str,
        default=None,
        help="Path to pre-trained VACE model weights (.safetensors). If not provided, initializes from scratch.",
    )
    vace_group.add_argument(
        "--vace_scale",
        type=float,
        default=1.0,
        help="VACE hint injection scale (0.0 = no control, 1.0 = full control). Default: 1.0",
    )
    vace_group.add_argument(
        "--vace_layers",
        type=str,
        default=None,
        help=(
            "Comma-separated list of DiT block indices for VACE hint injection. "
            "Default: every 4th block (0,4,8,...,44)."
        ),
    )
    vace_group.add_argument(
        "--vace_freeze_dit",
        action="store_true",
        default=True,
        help="Freeze base DiT parameters during VACE training (default: True).",
    )
    vace_group.add_argument(
        "--no_vace_freeze_dit",
        action="store_false",
        dest="vace_freeze_dit",
        help="Do NOT freeze base DiT parameters (train DiT + VACE jointly).",
    )
    vace_group.add_argument(
        "--enable_audio_xattn_in_vace",
        action="store_true",
        default=False,
        help="Add cross-attention from video VACE blocks to audio DiT hidden states.",
    )
    vace_group.add_argument(
        "--audio_vace_model_path",
        type=str,
        default=None,
        help="Path to pre-trained audio VACE model weights (.safetensors).",
    )
    vace_group.add_argument(
        "--audio_vace_scale",
        type=float,
        default=1.0,
        help="Audio VACE hint injection scale. Default: 1.0",
    )
    vace_group.add_argument(
        "--audio_vace_layers",
        type=str,
        default=None,
        help=(
            "Comma-separated list of DiT block indices for audio VACE hint injection. "
            "Default: same as video VACE layers."
        ),
    )
    vace_group.add_argument(
        "--vace_lr",
        type=float,
        default=None,
        help="Separate learning rate for VACE parameters. Default: same as --learning_rate.",
    )
    return parser


def main() -> None:
    """VACE training entry point."""
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)
    parser = vace_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    # Apply LTX-2 tweaks
    if hasattr(args, "ltx_mode"):
        short_map = {"v": "video", "a": "audio", "va": "av"}
        if args.ltx_mode in short_map:
            args.ltx_mode = short_map[args.ltx_mode]

    # VACE uses its own optional timestep weighting, so don't force LTX-2's default override.
    args.allow_custom_weighting_scheme = True
    apply_ltx2_tweaks(args)

    # Default to vace LoRA preset if not specified
    if not getattr(args, "lora_target_preset", None):
        args.lora_target_preset = "vace"
        logger.info("Defaulting to 'vace' LoRA target preset")

    trainer = LTX2VaceTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
