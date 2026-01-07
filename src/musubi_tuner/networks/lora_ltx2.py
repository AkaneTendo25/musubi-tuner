from __future__ import annotations

import ast
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora
from musubi_tuner.ltx_2.components.patchifiers import get_pixel_coords
from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
from musubi_tuner.ltx_2.model.transformer.modality import Modality
from musubi_tuner.ltx_2.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LTX2Wrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        patch_size: int = 1,
        audio_patch_size: int = 1,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal_audio: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        from musubi_tuner.ltx_2.components.patchifiers import AudioPatchifier, VideoLatentPatchifier

        self._video_patchifier = VideoLatentPatchifier(patch_size=patch_size)
        self._audio_patchifier = AudioPatchifier(
            patch_size=audio_patch_size,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
            is_causal=is_causal_audio,
        )
        self.patch_size = patch_size

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        if hasattr(self.model, "enable_gradient_checkpointing"):
            return self.model.enable_gradient_checkpointing(activation_cpu_offloading)
        if hasattr(self.model, "set_gradient_checkpointing"):
            self.model.set_gradient_checkpointing(True)
            if hasattr(self.model, "activation_cpu_offloading"):
                self.model.activation_cpu_offloading = activation_cpu_offloading
            return None
        raise AttributeError("Underlying LTX2 model does not support gradient checkpointing")

    def disable_gradient_checkpointing(self):
        if hasattr(self.model, "disable_gradient_checkpointing"):
            return self.model.disable_gradient_checkpointing()
        if hasattr(self.model, "set_gradient_checkpointing"):
            self.model.set_gradient_checkpointing(False)
            if hasattr(self.model, "activation_cpu_offloading"):
                self.model.activation_cpu_offloading = False
            return None
        raise AttributeError("Underlying LTX2 model does not support gradient checkpointing")

    def enable_block_swap(
        self, blocks_to_swap: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False
    ):
        return self.model.enable_block_swap(blocks_to_swap, device, supports_backward, use_pinned_memory)

    def move_to_device_except_swap_blocks(self, device: torch.device):
        return self.model.move_to_device_except_swap_blocks(device)

    def prepare_block_swap_before_forward(self):
        return self.model.prepare_block_swap_before_forward()

    def switch_block_swap_for_inference(self):
        if hasattr(self.model, "switch_block_swap_for_inference"):
            return self.model.switch_block_swap_for_inference()
        return None

    def switch_block_swap_for_training(self):
        if hasattr(self.model, "switch_block_swap_for_training"):
            return self.model.switch_block_swap_for_training()
        return None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(
        self,
        x,
        *,
        timestep,
        context,
        attention_mask=None,
        frame_rate: int = 25,
        transformer_options=None,
        **kwargs,
    ):
        if isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError("Expected x to be [video_latents, audio_latents] for AV mode")
            video_latents, audio_latents = x
        else:
            video_latents, audio_latents = x, None

        if not isinstance(video_latents, torch.Tensor) or video_latents.dim() != 5:
            raise ValueError(f"Expected video latents shape [B, C, F, H, W], got: {getattr(video_latents, 'shape', None)}")

        bsz, vch, vframes, vheight, vwidth = video_latents.shape

        if isinstance(timestep, torch.Tensor):
            ts = timestep
        else:
            ts = torch.tensor(timestep, device=video_latents.device, dtype=video_latents.dtype)
        if ts.dim() == 0:
            ts = ts.view(1)
        if ts.dim() == 2 and ts.shape[1] == 1:
            sigma = ts[:, 0]
        elif ts.dim() == 1:
            sigma = ts
        else:
            raise ValueError(f"Unexpected timestep shape: {tuple(ts.shape)}")

        video_tokens = self._video_patchifier.patchify(video_latents)
        video_seq_len = video_tokens.shape[1]
        video_timesteps = sigma.view(bsz, 1).expand(bsz, video_seq_len)

        video_conditioning_mask = None
        if isinstance(transformer_options, dict):
            video_conditioning_mask = transformer_options.get("video_conditioning_mask")
        if video_conditioning_mask is not None:
            if not isinstance(video_conditioning_mask, torch.Tensor):
                raise TypeError(f"Expected video_conditioning_mask to be a torch.Tensor, got: {type(video_conditioning_mask)}")
            if video_conditioning_mask.shape != (bsz, video_seq_len):
                raise ValueError(
                    f"video_conditioning_mask shape mismatch: got {tuple(video_conditioning_mask.shape)}, expected {(bsz, video_seq_len)}"
                )
            video_conditioning_mask = video_conditioning_mask.to(device=video_tokens.device, dtype=torch.bool)
            video_timesteps = torch.where(video_conditioning_mask, torch.zeros_like(video_timesteps), video_timesteps)

        latent_coords = self._video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                batch=bsz,
                channels=vch,
                frames=vframes,
                height=vheight,
                width=vwidth,
            ),
            device=video_latents.device,
        )
        video_positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).to(dtype=video_latents.dtype)
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / float(frame_rate)

        video_context = context
        audio_context = context
        if audio_latents is not None and isinstance(context, torch.Tensor) and context.shape[-1] % 2 == 0:
            half = context.shape[-1] // 2
            video_context = context[..., :half]
            audio_context = context[..., half:]

        video_modality = Modality(
            enabled=True,
            latent=video_tokens,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
            context_mask=attention_mask,
        )

        audio_modality = None
        audio_shape = None
        if audio_latents is not None:
            if not isinstance(audio_latents, torch.Tensor) or audio_latents.dim() != 4:
                raise ValueError(f"Expected audio latents shape [B, C, T, F], got: {getattr(audio_latents, 'shape', None)}")

            absz, ach, at, af = audio_latents.shape
            if absz != bsz:
                raise ValueError(f"Batch mismatch: video B={bsz}, audio B={absz}")

            audio_tokens = self._audio_patchifier.patchify(audio_latents)
            audio_seq_len = audio_tokens.shape[1]
            audio_timesteps = sigma.view(bsz, 1).expand(bsz, audio_seq_len)

            audio_shape = AudioLatentShape(batch=bsz, channels=ach, frames=at, mel_bins=af)
            audio_positions = self._audio_patchifier.get_patch_grid_bounds(audio_shape, device=audio_latents.device)

            audio_modality = Modality(
                enabled=True,
                latent=audio_tokens,
                timesteps=audio_timesteps,
                positions=audio_positions.to(dtype=audio_latents.dtype),
                context=audio_context,
                context_mask=attention_mask,
            )

        perturbations = BatchedPerturbationConfig.empty(bsz)
        video_pred_tokens, audio_pred_tokens = self.model(video_modality, audio_modality, perturbations)

        video_pred = self._video_patchifier.unpatchify(
            video_pred_tokens,
            output_shape=VideoLatentShape(
                batch=bsz,
                channels=vch,
                frames=vframes,
                height=vheight,
                width=vwidth,
            ),
        )

        if audio_latents is None:
            return video_pred

        audio_pred = self._audio_patchifier.unpatchify(audio_pred_tokens, output_shape=audio_shape)
        return [video_pred, audio_pred]


def load_ltx2_transformer(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    audio_video: bool = False,
) -> nn.Module:
    from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from musubi_tuner.ltx_2.model.transformer.model_configurator import (
        LTXModelConfigurator,
        LTXVideoOnlyModelConfigurator,
        LTXV_MODEL_COMFY_RENAMING_MAP,
    )

    configurator = LTXModelConfigurator if audio_video else LTXVideoOnlyModelConfigurator
    return SingleGPUModelBuilder(
        model_path=str(model_path),
        model_class_configurator=configurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=device, dtype=dtype)


def load_ltx2_wrapper(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    audio_video: bool = False,
    patch_size: int = 1,
) -> nn.Module:
    model = load_ltx2_transformer(model_path, device=device, dtype=dtype, audio_video=audio_video)
    return LTX2Wrapper(model, patch_size=patch_size)


LTX2_TARGET_REPLACE_MODULES = [
    "BasicAVTransformerBlock",
    "PixArtAlphaTextProjection",
]


def _build_exclude_patterns(raw_patterns: Optional[str], audio_video: bool = False) -> List[str]:
    if raw_patterns is None:
        patterns: List[str] = []
    else:
        patterns = ast.literal_eval(raw_patterns)
        if not isinstance(patterns, list):
            raise ValueError("exclude_patterns must evaluate to a list")

    patterns.extend(
        [
            r".*(norm|norm_out|scale_shift_table|patchify_proj|proj_out).*",
            r".*patchifier.*",
            r".*adaln_single\.emb.*",
        ]
    )

    if audio_video:
        patterns.extend(
            [
                r".*audio_scale_shift_table.*",
                r".*audio_patchify_proj.*",
                r".*audio_proj_out.*",
                r".*audio_norm_out.*",
                r".*audio_adaln_single\.emb.*",
                r".*a_patchifier.*",
                r".*av_ca.*adaln_single\.emb.*",
            ]
        )

    return patterns


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    audio_video = kwargs.pop("audio_video", False)
    if not audio_video and unet is not None:
        audio_video = unet.__class__.__name__ == "LTXAVModel" or hasattr(unet, "audio_patchify_proj")

    kwargs["exclude_patterns"] = _build_exclude_patterns(kwargs.get("exclude_patterns"), audio_video=audio_video)

    return lora.create_network(
        LTX2_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    audio_video = kwargs.pop("audio_video", False)
    if not audio_video:
        audio_video = any("audio_" in k for k in weights_sd.keys())
        if not audio_video and unet is not None:
            audio_video = unet.__class__.__name__ == "LTXAVModel" or hasattr(unet, "audio_patchify_proj")

    kwargs["exclude_patterns"] = _build_exclude_patterns(kwargs.get("exclude_patterns"), audio_video=audio_video)

    return lora.create_network_from_weights(
        LTX2_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
