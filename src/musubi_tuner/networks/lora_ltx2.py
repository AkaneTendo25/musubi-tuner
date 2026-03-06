from __future__ import annotations

import ast
import logging
import types
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


def _patch_lora_load_state_dict_for_audio(network: lora.LoRANetwork) -> lora.LoRANetwork:
    original = network.load_state_dict

    def _filter_audio_keys(keys: List[str]) -> List[str]:
        return [k for k in keys if "audio_" not in k]

    def _load_state_dict(self, state_dict, strict: bool = True):
        result = original(state_dict, strict=False)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        non_audio_missing = _filter_audio_keys(missing)
        non_audio_unexpected = _filter_audio_keys(unexpected)
        if non_audio_missing:
            raise RuntimeError(
                f"Missing non-audio LoRA keys in state_dict: {non_audio_missing[:10]}"
            )
        if non_audio_unexpected:
            raise RuntimeError(
                f"Unexpected non-audio LoRA keys in state_dict: {non_audio_unexpected[:10]}"
            )
        if missing and not non_audio_missing:
            logger.warning(
                "LTX2 LoRA: missing audio keys in checkpoint; initializing audio LoRA weights."
            )
        if unexpected and not non_audio_unexpected:
            logger.warning(
                "LTX2 LoRA: unexpected audio keys in checkpoint; ignoring audio LoRA weights."
            )
        try:
            incompatible = torch.nn.modules.module._IncompatibleKeys(  # type: ignore[attr-defined]
                missing_keys=non_audio_missing,
                unexpected_keys=non_audio_unexpected,
            )
            return incompatible
        except Exception:
            return result

    network.load_state_dict = types.MethodType(_load_state_dict, network)
    return network


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

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False, **kwargs):
        if hasattr(self.model, "enable_gradient_checkpointing"):
            # LTX2 core model supports blocks_to_checkpoint when provided.
            weight_cpu_offloading = kwargs.get("weight_cpu_offloading", False)
            blocks_to_checkpoint = kwargs.get("blocks_to_checkpoint", None)
            return self.model.enable_gradient_checkpointing(
                activation_cpu_offloading,
                weight_cpu_offloading=weight_cpu_offloading,
                blocks_to_checkpoint=blocks_to_checkpoint,
            )
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
        self, blocks_to_swap: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False, swap_norms: bool = False
    ):
        return self.model.enable_block_swap(blocks_to_swap, device, supports_backward, use_pinned_memory, swap_norms=swap_norms)

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
        audio_only: bool = False,
        video_enabled: Optional[bool] = None,
        audio_enabled: Optional[bool] = None,
        **kwargs,
    ):
        if isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError("Expected x to be [video_latents, audio_latents] for AV mode")
            video_latents, audio_latents = x
        else:
            video_latents, audio_latents = x, None

        model_type = getattr(self.model, "model_type", None)
        model_video_enabled = bool(model_type.is_video_enabled()) if model_type is not None else True
        model_audio_enabled = bool(model_type.is_audio_enabled()) if model_type is not None else True

        if audio_only:
            if audio_latents is None:
                raise ValueError("audio_only=True requires audio_latents")
            if video_latents is None and model_video_enabled:
                in_channels = getattr(self.model, "in_channels", None)
                if in_channels is None:
                    raise ValueError("audio_only=True requires model.in_channels to create dummy video latents")
                bsz = int(audio_latents.shape[0])
                video_latents = torch.zeros(
                    (bsz, int(in_channels), 1, 1, 1),
                    device=audio_latents.device,
                    dtype=audio_latents.dtype,
                )

        if not model_audio_enabled and audio_latents is not None:
            raise ValueError("Audio latents were provided but the loaded model has no audio branch")
        if not model_video_enabled and not audio_only:
            raise ValueError("Loaded audio-only transformer requires audio_only=True")

        if model_video_enabled:
            if not isinstance(video_latents, torch.Tensor) or video_latents.dim() != 5:
                raise ValueError(f"Expected video latents shape [B, C, F, H, W], got: {getattr(video_latents, 'shape', None)}")
        elif video_latents is not None and (not isinstance(video_latents, torch.Tensor) or video_latents.dim() != 5):
            raise ValueError(f"Expected video latents shape [B, C, F, H, W], got: {getattr(video_latents, 'shape', None)}")

        ref_latents = video_latents if isinstance(video_latents, torch.Tensor) else audio_latents
        if not isinstance(ref_latents, torch.Tensor):
            raise ValueError("Expected at least one latent tensor (video or audio) to be present")

        bsz = int(ref_latents.shape[0])
        vch = vframes = vheight = vwidth = None
        if isinstance(video_latents, torch.Tensor):
            _, vch, vframes, vheight, vwidth = video_latents.shape

        def _to_timestep(ts_value, *, name: str) -> torch.Tensor:
            if isinstance(ts_value, torch.Tensor):
                ts = ts_value
            else:
                ts = torch.tensor(ts_value, device=ref_latents.device, dtype=ref_latents.dtype)
            if ts.dim() == 0:
                ts = ts.view(1, 1)
            elif ts.dim() == 1:
                ts = ts.view(-1, 1)
            elif ts.dim() == 2:
                pass
            else:
                raise ValueError(f"Unexpected {name} shape: {tuple(ts.shape)}")
            if ts.shape[0] == 1 and bsz != 1:
                ts = ts.expand(bsz, ts.shape[1])
            if ts.shape[0] != bsz:
                raise ValueError(f"Expected {name} batch size {bsz}, got {ts.shape[0]}")
            return ts.to(device=ref_latents.device, dtype=ref_latents.dtype)

        timestep_video = _to_timestep(timestep, name="timestep")
        audio_timestep = kwargs.get("audio_timestep")
        timestep_audio = _to_timestep(audio_timestep, name="audio_timestep") if audio_timestep is not None else timestep_video

        video_tokens = None
        video_timesteps = None
        video_positions = None
        if model_video_enabled:
            video_tokens = self._video_patchifier.patchify(video_latents)
            video_seq_len = video_tokens.shape[1]
            if timestep_video.shape[1] == 1:
                video_timesteps = timestep_video.expand(bsz, video_seq_len)
            elif timestep_video.shape[1] == video_seq_len:
                video_timesteps = timestep_video
            else:
                raise ValueError(
                    f"timestep shape mismatch for video tokens: got {tuple(timestep_video.shape)}, "
                    f"expected second dim 1 or {video_seq_len}"
                )

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
        if (
            model_video_enabled
            and not audio_only
            and audio_latents is not None
            and isinstance(context, torch.Tensor)
        ):
            split_video_dim = getattr(self.model, "cross_attention_dim", None)
            split_audio_dim = getattr(self.model, "audio_cross_attention_dim", None)
            if (
                isinstance(split_video_dim, int)
                and isinstance(split_audio_dim, int)
                and split_video_dim > 0
                and split_audio_dim > 0
                and (split_video_dim + split_audio_dim) == context.shape[-1]
            ):
                video_context = context[..., :split_video_dim]
                audio_context = context[..., split_video_dim : split_video_dim + split_audio_dim]
            elif context.shape[-1] % 2 == 0:
                half = context.shape[-1] // 2
                video_context = context[..., :half]
                audio_context = context[..., half:]

        video_modality = None
        if model_video_enabled:
            video_self_attention_mask = None
            if isinstance(transformer_options, dict):
                video_self_attention_mask = transformer_options.get("self_attention_mask")
            video_modality = Modality(
                enabled=(not audio_only if video_enabled is None else bool(video_enabled)),
                latent=video_tokens,
                timesteps=video_timesteps,
                positions=video_positions,
                context=video_context,
                sigma=sigma,
                context_mask=attention_mask,
                attention_mask=video_self_attention_mask,
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
            if timestep_audio.shape[1] == 1:
                audio_timesteps = timestep_audio.expand(bsz, audio_seq_len)
            elif timestep_audio.shape[1] == audio_seq_len:
                audio_timesteps = timestep_audio
            else:
                raise ValueError(
                    f"audio_timestep shape mismatch for audio tokens: got {tuple(timestep_audio.shape)}, "
                    f"expected second dim 1 or {audio_seq_len}"
                )

            audio_shape = AudioLatentShape(batch=bsz, channels=ach, frames=at, mel_bins=af)
            audio_positions = self._audio_patchifier.get_patch_grid_bounds(audio_shape, device=audio_latents.device)

            audio_modality = Modality(
                enabled=(True if audio_enabled is None else bool(audio_enabled)),
                latent=audio_tokens,
                timesteps=audio_timesteps,
                positions=audio_positions.to(dtype=audio_latents.dtype),
                context=audio_context,
                sigma=audio_sigma,
                context_mask=attention_mask,
            )

        perturbations = BatchedPerturbationConfig.empty(bsz)
        video_pred_tokens, audio_pred_tokens = self.model(video_modality, audio_modality, perturbations)

        if model_video_enabled:
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
        elif isinstance(video_latents, torch.Tensor):
            video_pred = torch.zeros_like(video_latents)
        else:
            channel_count = int(getattr(self.model, "in_channels", 1) or 1)
            video_pred = torch.zeros(
                (bsz, channel_count, 1, 1, 1),
                device=ref_latents.device,
                dtype=ref_latents.dtype,
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


# Target module class names to search for LoRA-applicable layers
# Only BasicAVTransformerBlock is targeted - it contains all attention modules:
#   - attn1 (video self-attention), attn2 (video cross-attention to text)
#   - audio_attn1, audio_attn2 (audio attention)
#   - audio_to_video_attn, video_to_audio_attn (cross-modal attention)
#   - ff, audio_ff (feed-forward, if included via patterns)
LTX2_TARGET_REPLACE_MODULES = [
    "BasicAVTransformerBlock",
]

# LoRA target presets for different training modes
# These patterns match layers inside BasicAVTransformerBlock.
#
# Available modules (from official LTX-2 docs):
#
# VIDEO MODULES:
#   - attn1.to_k, attn1.to_q, attn1.to_v, attn1.to_out.0  (video self-attention)
#   - attn2.to_k, attn2.to_q, attn2.to_v, attn2.to_out.0  (video cross-attention to text)
#   - ff.net.0.proj, ff.net.2                             (video feed-forward)
#
# AUDIO MODULES:
#   - audio_attn1.to_k, audio_attn1.to_q, etc.            (audio self-attention)
#   - audio_attn2.to_k, audio_attn2.to_q, etc.            (audio cross-attention to text)
#   - audio_ff.net.0.proj, audio_ff.net.2                 (audio feed-forward)
#
# CROSS-MODAL MODULES:
#   - audio_to_video_attn.to_k, etc.                      (audio-to-video cross-attention)
#   - video_to_audio_attn.to_k, etc.                      (video-to-audio cross-attention)
#
# Using short patterns like "to_k" matches ALL attention modules across video, audio,
# and cross-modal branches. This is the recommended approach for audio-video training.

# t2v: Text-to-video (attention only)
# Official LTX-2 trainer default. Trains all attention projections (Q, K, V, Out)
# across video, audio, and cross-modal attention blocks.
LTX2_INCLUDE_PATTERNS_T2V = [
    r".*\.to_k$",
    r".*\.to_q$",
    r".*\.to_v$",
    r".*\.to_out\.0$",
]

# v2v: Video-to-video / IC-LoRA (attention + feed-forward)
# Recommended for IC-LoRA and reference-based generation. Adds FFN layers
# for more expressive capacity when learning from reference videos.
LTX2_INCLUDE_PATTERNS_V2V = [
    r".*\.to_k$",
    r".*\.to_q$",
    r".*\.to_v$",
    r".*\.to_out\.0$",
    r".*\.ff\.net\.0\.proj$",
    r".*\.ff\.net\.2$",
    r".*\.audio_ff\.net\.0\.proj$",
    r".*\.audio_ff\.net\.2$",
]

# audio: Audio-only LoRA (audio attention/FFN + audio-side cross-modal)
# Targets audio self/cross-attn, audio FFN, and video_to_audio_attn (audio queries video).
# Excludes audio_to_video_attn to avoid altering the video branch.
LTX2_INCLUDE_PATTERNS_AUDIO = [
    r".*\.audio_attn1\.to_k$",
    r".*\.audio_attn1\.to_q$",
    r".*\.audio_attn1\.to_v$",
    r".*\.audio_attn1\.to_out\.0$",
    r".*\.audio_attn2\.to_k$",
    r".*\.audio_attn2\.to_q$",
    r".*\.audio_attn2\.to_v$",
    r".*\.audio_attn2\.to_out\.0$",
    r".*\.audio_ff\.net\.0\.proj$",
    r".*\.audio_ff\.net\.2$",
    r".*\.video_to_audio_attn\.to_k$",
    r".*\.video_to_audio_attn\.to_q$",
    r".*\.video_to_audio_attn\.to_v$",
    r".*\.video_to_audio_attn\.to_out\.0$",
]

# full: All linear layers in transformer blocks
# Maximum expressiveness, but larger LoRA file and more VRAM usage.
LTX2_INCLUDE_PATTERNS_FULL = None  # None means no filtering, all Linear layers matched

# Mapping from preset name to include patterns
LTX2_LORA_TARGET_PRESETS = {
    "t2v": LTX2_INCLUDE_PATTERNS_T2V,
    "v2v": LTX2_INCLUDE_PATTERNS_V2V,
    "audio": LTX2_INCLUDE_PATTERNS_AUDIO,
    "full": LTX2_INCLUDE_PATTERNS_FULL,
}

# Default preset (for backwards compatibility)
LTX2_DEFAULT_INCLUDE_PATTERNS = LTX2_INCLUDE_PATTERNS_T2V


def _build_exclude_patterns(raw_patterns: Optional[str], audio_video: bool = False) -> List[str]:  # noqa: ARG001
    """Build exclude patterns list, including connector exclusions."""
    patterns: List[str] = [
        r".*text_embedding_projection\.aggregate_embed.*",
        r".*text_embedding_projection\.video_aggregate_embed.*",
        r".*text_embedding_projection\.audio_aggregate_embed.*",
        r".*embeddings_connector\..*",
        r".*audio_embeddings_connector\..*",
    ]
    if raw_patterns is None:
        return patterns
    user_patterns = ast.literal_eval(raw_patterns)
    if not isinstance(user_patterns, list):
        raise ValueError("exclude_patterns must evaluate to a list")
    patterns.extend(user_patterns)
    return patterns


def _get_include_patterns_for_preset(preset: Optional[str]) -> Optional[List[str]]:
    """Get include patterns for a given preset name."""
    if preset is None:
        return LTX2_DEFAULT_INCLUDE_PATTERNS
    if preset not in LTX2_LORA_TARGET_PRESETS:
        raise ValueError(
            f"Unknown lora_target_preset: {preset!r}. "
            f"Valid presets: {list(LTX2_LORA_TARGET_PRESETS.keys())}"
        )
    return LTX2_LORA_TARGET_PRESETS[preset]


def compute_loftq_from_state_dict(
    state_dict: dict,
    loftq_config: dict,
    network_dim: int,
    target_layer_keys: Optional[List[str]] = None,
    exclude_layer_keys: Optional[List[str]] = None,
) -> Dict[str, tuple]:
    """Pre-compute LoftQ (lora_A, lora_B) from full-precision weights in a state dict.

    Must be called BEFORE NF4 quantization, while weights are still full-precision.

    Returns a dict mapping ``lora_unet_<module_path>`` → ``(lora_A, lora_B)``.
    """
    from tqdm import tqdm
    from musubi_tuner.modules.loftq_init import loftq_initialize
    from musubi_tuner.modules.nf4_optimization_utils import quantize_nf4_block, dequantize_nf4_block

    num_iterations = loftq_config.get("num_iterations", 1)
    block_size = loftq_config.get("block_size", 64)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Find target weight keys (same filtering as NF4 quantization)
    target_keys = []
    for key in state_dict:
        if not key.endswith(".weight"):
            continue
        is_target = target_layer_keys is None or any(p in key for p in target_layer_keys)
        is_excluded = exclude_layer_keys is not None and any(p in key for p in exclude_layer_keys)
        if is_target and not is_excluded:
            w = state_dict[key]
            if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[1] % block_size == 0:
                target_keys.append(key)

    loftq_data: Dict[str, tuple] = {}
    for key in tqdm(target_keys, desc="LoftQ SVD init"):
        weight = state_dict[key]
        # Build lora_name matching the convention in lora.py's create_modules
        module_path = key.rsplit(".weight", 1)[0]
        lora_name = f"lora_unet_{module_path}".replace(".", "_")
        try:
            lora_A, lora_B = loftq_initialize(
                weight,
                quantize_fn=quantize_nf4_block,
                dequantize_fn=dequantize_nf4_block,
                lora_rank=network_dim,
                block_size=block_size,
                num_iterations=num_iterations,
                device=device,
            )
            loftq_data[lora_name] = (lora_A.cpu(), lora_B.cpu())
        except Exception as e:
            logger.warning("LoftQ init failed for %s: %s", module_path, e)
            continue

    logger.info("LoftQ initialization computed for %d modules", len(loftq_data))
    return loftq_data


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

    # Handle lora_target_preset: use preset patterns unless include_patterns is explicitly set
    lora_target_preset = kwargs.pop("lora_target_preset", None)
    if kwargs.get("include_patterns") is None:
        preset_patterns = _get_include_patterns_for_preset(lora_target_preset)
        kwargs["include_patterns"] = preset_patterns
        if lora_target_preset is not None:
            logger.info(f"Using LoRA target preset '{lora_target_preset}' with patterns: {preset_patterns}")
    else:
        if lora_target_preset is not None:
            logger.warning(
                f"Both lora_target_preset='{lora_target_preset}' and include_patterns are set. "
                "Using explicit include_patterns, ignoring preset."
            )

    # Handle LoftQ: loftq_data is pre-computed from full-precision weights
    # before NF4 quantization (passed via kwargs from the training script)
    kwargs.pop("loftq_config", None)  # consumed upstream, not needed here
    loftq_data = kwargs.pop("loftq_data", None)
    if loftq_data is not None:
        module_kwargs = kwargs.get("module_kwargs", None) or {}
        module_kwargs["loftq_data"] = loftq_data
        kwargs["module_kwargs"] = module_kwargs

    net = lora.create_network(
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
    return _patch_lora_load_state_dict_for_audio(net)


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

    net = lora.create_network_from_weights(
        LTX2_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
    return _patch_lora_load_state_dict_for_audio(net)
