"""Preservation & regularization techniques for LTX-2 LoRA training.

Implements three optional techniques (from ai-toolkit via diffusion-pipe):
1. Blank Prompt Preservation  -- regularise LoRA to not change blank-prompt output
2. Differential Output Preservation (DOP) -- regularise LoRA to not change class-prompt output
3. Prior Divergence -- encourage LoRA output to diverge from base model on training prompts
"""

import argparse
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arg parsing helper (same format as --optimizer_args: key=value pairs)
# ---------------------------------------------------------------------------

DEFAULT_PRESERVATION_CACHE = "ltx2_preservation_cache.pt"


def parse_preservation_args(raw_args: Optional[List[str]]) -> Dict[str, str]:
    """Parse ``key=value`` list into a dict.  Returns empty dict for None/[]."""
    if not raw_args:
        return {}
    result: Dict[str, str] = {}
    for item in raw_args:
        if "=" not in item:
            raise ValueError(f"Expected key=value format, got: {item!r}")
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def _resolve_default_preservation_cache(args: argparse.Namespace) -> str:
    """Resolve default cache path from dataset config (same pattern as sample prompts)."""
    from musubi_tuner.dataset import config_utils
    from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2

    if not getattr(args, "dataset_config", None):
        raise ValueError("--dataset_config is required to resolve the preservation cache directory")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_LTX2)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = dataset_group.datasets
    if not datasets:
        raise ValueError("No datasets available to resolve preservation cache directory")
    cache_dir = getattr(datasets[0], "cache_directory", None)
    if not cache_dir:
        raise ValueError("First dataset has no cache_directory; set cache_directory in dataset config")
    return os.path.join(cache_dir, DEFAULT_PRESERVATION_CACHE)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreservationConfig:
    blank_preservation: bool = False
    blank_multiplier: float = 1.0
    blank_embed: Optional[torch.Tensor] = field(default=None, repr=False)
    blank_mask: Optional[torch.Tensor] = field(default=None, repr=False)

    dop: bool = False
    dop_multiplier: float = 1.0
    dop_class_prompt: str = ""
    dop_embed: Optional[torch.Tensor] = field(default=None, repr=False)
    dop_mask: Optional[torch.Tensor] = field(default=None, repr=False)

    prior_divergence: bool = False
    prior_divergence_multiplier: float = 0.1

    audio_dop: bool = False
    audio_dop_multiplier: float = 1.0

    @property
    def any_active(self) -> bool:
        return self.blank_preservation or self.dop or self.prior_divergence or self.audio_dop

    @property
    def needs_text_encoding(self) -> bool:
        return self.blank_preservation or self.dop


# ---------------------------------------------------------------------------
# Helper class
# ---------------------------------------------------------------------------

class PreservationHelper:
    def __init__(self, config: PreservationConfig) -> None:
        self.config = config

    # -- block swap compatibility ----------------------------------------

    @staticmethod
    def _prepare_block_swap(transformer: torch.nn.Module, accelerator: Accelerator) -> None:
        """Reset block-swap device placement before an extra forward pass.

        When ``--blocks_to_swap`` is active the offloader tracks which
        transformer blocks sit on GPU vs CPU.  After the main fwd/bwd the
        placement may be stale, so we must reset it before each preservation
        forward to avoid device-mismatch errors.
        """
        unwrapped = accelerator.unwrap_model(transformer)
        if hasattr(unwrapped, "prepare_block_swap_before_forward"):
            # Suppress verbose offloader logs during preservation forwards
            offload_logger = logging.getLogger("musubi_tuner.ltx_2.model.transformer.offloading_utils")
            prev_level = offload_logger.level
            offload_logger.setLevel(logging.WARNING)
            try:
                unwrapped.prepare_block_swap_before_forward()
            finally:
                offload_logger.setLevel(prev_level)

    # -- encode prompts --------------------------------------------------

    def encode_prompts(
        self,
        trainer: Any,  # LTX2NetworkTrainer
        args: argparse.Namespace,
        accelerator: Accelerator,
    ) -> None:
        """Load embeddings from cache or Gemma.  No-op if nothing to encode."""
        cfg = self.config
        if not cfg.needs_text_encoding:
            return

        # Try loading from precached file first
        if getattr(args, "use_precached_preservation", False):
            cache_path = getattr(args, "preservation_prompts_cache", None)
            if not cache_path:
                cache_path = _resolve_default_preservation_cache(args)
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(
                    f"Precached preservation embeddings not found: {cache_path}\n"
                    "Run ltx2_cache_text_encoder_outputs.py with --precache_preservation_prompts first."
                )
            self._load_from_cache(cache_path)
            return

        # Fall back to loading Gemma and encoding live
        text_encoder_dtype = trainer._build_text_encoder(args, accelerator)

        # In AV mode, _encode_prompt_text returns concatenated video+audio embeddings.
        # Preservation only regularises the video branch, so keep only the video half.
        av_mode = getattr(trainer, "_audio_video", False)

        if cfg.blank_preservation:
            embed, mask = trainer._encode_prompt_text(accelerator, "", text_encoder_dtype)
            if av_mode and embed.shape[-1] % 2 == 0:
                embed = embed[..., : embed.shape[-1] // 2]
            cfg.blank_embed = embed
            cfg.blank_mask = mask
            logger.info("Preservation: encoded blank prompt  embed=%s (av_mode=%s)", tuple(embed.shape), av_mode)

        if cfg.dop:
            embed, mask = trainer._encode_prompt_text(accelerator, cfg.dop_class_prompt, text_encoder_dtype)
            if av_mode and embed.shape[-1] % 2 == 0:
                embed = embed[..., : embed.shape[-1] // 2]
            cfg.dop_embed = embed
            cfg.dop_mask = mask
            logger.info("Preservation: encoded DOP class prompt %r  embed=%s (av_mode=%s)", cfg.dop_class_prompt, tuple(embed.shape), av_mode)

        # unload text encoder
        trainer._text_encoder = None
        clean_memory_on_device(accelerator.device)
        gc.collect()
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Preservation: text encoder unloaded")

    def _load_from_cache(self, cache_path: str) -> None:
        """Load preservation embeddings from a precached .pt file."""
        cfg = self.config
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        logger.info("Preservation: loading precached embeddings from %s (version=%s)", cache_path, payload.get("version"))

        if cfg.blank_preservation:
            if "blank_embed" in payload and "blank_mask" in payload:
                cfg.blank_embed = payload["blank_embed"]
                cfg.blank_mask = payload["blank_mask"]
                logger.info("Preservation: loaded blank prompt  embed=%s", tuple(cfg.blank_embed.shape))
            else:
                logger.warning("Preservation cache missing blank embeddings — blank_preservation will be skipped.")

        if cfg.dop:
            if "dop_embed" in payload and "dop_mask" in payload:
                cfg.dop_embed = payload["dop_embed"]
                cfg.dop_mask = payload["dop_mask"]
                cached_class = payload.get("dop_class_prompt", "")
                if cached_class != cfg.dop_class_prompt:
                    logger.warning(
                        "DOP class prompt mismatch: cache has %r, training uses %r. Using cached embeddings.",
                        cached_class, cfg.dop_class_prompt,
                    )
                logger.info("Preservation: loaded DOP class prompt %r  embed=%s", cached_class, tuple(cfg.dop_embed.shape))
            else:
                logger.warning("Preservation cache missing DOP embeddings — DOP will be skipped.")

    # -- prior divergence (no-grad fwd, LoRA OFF) -------------------------

    def compute_prior_divergence(
        self,
        trainer: Any,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        accelerator: Accelerator,
        dit_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
    ) -> torch.Tensor:
        """No-grad forward with LoRA OFF using training batch text embeddings.
        Returns prior_pred tensor (detached)."""
        self._prepare_block_swap(transformer, accelerator)
        network.set_multiplier(0.0)
        try:
            with torch.no_grad(), accelerator.autocast():
                prior_pred = transformer(
                    dit_inputs["model_input"],
                    timestep=dit_inputs["model_timesteps"],
                    context=dit_inputs["text_embeds"],
                    attention_mask=dit_inputs["text_mask"],
                    frame_rate=dit_inputs["frame_rate"],
                    transformer_options=dit_inputs["transformer_options"],
                )
            if isinstance(prior_pred, (list, tuple)):
                prior_pred = prior_pred[0]  # video only
            return prior_pred.detach()
        finally:
            network.set_multiplier(1.0)

    # -- preservation backward (blank / DOP) ------------------------------

    def compute_preservation_backward(
        self,
        technique: str,  # "blank" or "dop"
        trainer: Any,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        accelerator: Accelerator,
        dit_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
    ) -> float:
        """Two-forward preservation: (a) no-grad LoRA OFF -> prior, (b) with-grad LoRA ON -> pres.
        MSE(pres, prior) * mult, separate backward.  Returns loss float."""
        cfg = self.config
        if technique == "blank":
            embed, mask, mult = cfg.blank_embed, cfg.blank_mask, cfg.blank_multiplier
        elif technique == "dop":
            embed, mask, mult = cfg.dop_embed, cfg.dop_mask, cfg.dop_multiplier
        else:
            raise ValueError(f"Unknown preservation technique: {technique}")

        if embed is None or mask is None:
            return 0.0

        device = accelerator.device
        # Prepare embeddings: expand to batch size
        bsz = dit_inputs["model_timesteps"].shape[0]
        pres_embed = embed.unsqueeze(0).expand(bsz, -1, -1).to(device=device, dtype=network_dtype)
        pres_mask = mask.unsqueeze(0).expand(bsz, -1).to(device=device)

        # In AV mode model_input may be [video, audio].  Preservation only
        # regularises the video branch, so extract video-only to avoid the
        # wrapper trying to split our video-only embeddings as if they were
        # concatenated video+audio.
        model_input = dit_inputs["model_input"]
        if isinstance(model_input, (list, tuple)):
            model_input = model_input[0]  # video tensor only

        # Build inputs with preservation embeddings
        pres_inputs = {
            "model_input": model_input,
            "model_timesteps": dit_inputs["model_timesteps"],
            "text_embeds": pres_embed,
            "text_mask": pres_mask,
            "frame_rate": dit_inputs["frame_rate"],
            "transformer_options": dit_inputs["transformer_options"],
        }

        # (a) no-grad forward, LoRA OFF -> prior_pred
        self._prepare_block_swap(transformer, accelerator)
        network.set_multiplier(0.0)
        try:
            with torch.no_grad(), accelerator.autocast():
                prior_pred = transformer(
                    pres_inputs["model_input"],
                    timestep=pres_inputs["model_timesteps"],
                    context=pres_inputs["text_embeds"],
                    attention_mask=pres_inputs["text_mask"],
                    frame_rate=pres_inputs["frame_rate"],
                    transformer_options=pres_inputs["transformer_options"],
                )
            if isinstance(prior_pred, (list, tuple)):
                prior_pred = prior_pred[0]
            prior_pred = prior_pred.detach()
        finally:
            network.set_multiplier(1.0)

        # (b) with-grad forward, LoRA ON -> pres_pred
        self._prepare_block_swap(transformer, accelerator)
        with accelerator.autocast():
            pres_pred = transformer(
                pres_inputs["model_input"],
                timestep=pres_inputs["model_timesteps"],
                context=pres_inputs["text_embeds"],
                attention_mask=pres_inputs["text_mask"],
                frame_rate=pres_inputs["frame_rate"],
                transformer_options=pres_inputs["transformer_options"],
            )
        if isinstance(pres_pred, (list, tuple)):
            pres_pred = pres_pred[0]

        # (c) MSE loss * multiplier
        pres_loss = F.mse_loss(pres_pred.float(), prior_pred.float()) * mult

        # (d) NaN guard — skip backward if loss is non-finite
        if not torch.isfinite(pres_loss):
            logger.warning("Preservation %s loss is non-finite (%.4g), skipping backward.", technique, pres_loss.item())
            del prior_pred, pres_pred, pres_loss, pres_embed, pres_mask, pres_inputs
            clean_memory_on_device(device)
            return float("nan")

        # (e) separate backward
        accelerator.backward(pres_loss)

        loss_val = pres_loss.detach().item()

        # cleanup
        del prior_pred, pres_pred, pres_loss, pres_embed, pres_mask, pres_inputs
        clean_memory_on_device(device)

        return loss_val

    # -- audio DOP (preserve audio predictions on non-audio steps) ----------

    def compute_audio_dop_backward(
        self,
        trainer: Any,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        accelerator: Accelerator,
        av_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
    ) -> float:
        """Two-forward audio DOP: run transformer with AV inputs, compare audio predictions
        between LoRA OFF (prior) and LoRA ON.  MSE on audio branch only × multiplier.
        Returns loss float."""
        mult = self.config.audio_dop_multiplier
        device = accelerator.device

        # (a) no-grad forward, LoRA OFF -> extract audio prior
        self._prepare_block_swap(transformer, accelerator)
        network.set_multiplier(0.0)
        try:
            with torch.no_grad(), accelerator.autocast():
                prior_pred = transformer(
                    av_inputs["model_input"],
                    timestep=av_inputs["model_timesteps"],
                    audio_timestep=av_inputs["audio_timestep"],
                    context=av_inputs["text_embeds"],
                    attention_mask=av_inputs["text_mask"],
                    frame_rate=av_inputs["frame_rate"],
                    transformer_options=av_inputs["transformer_options"],
                )
            if not isinstance(prior_pred, (list, tuple)) or len(prior_pred) < 2:
                logger.warning("Audio DOP: transformer did not return [video, audio] — skipping.")
                return 0.0
            audio_prior = prior_pred[1].detach()
            del prior_pred
        finally:
            network.set_multiplier(1.0)

        # (b) with-grad forward, LoRA ON -> extract audio prediction
        self._prepare_block_swap(transformer, accelerator)
        with accelerator.autocast():
            lora_pred = transformer(
                av_inputs["model_input"],
                timestep=av_inputs["model_timesteps"],
                audio_timestep=av_inputs["audio_timestep"],
                context=av_inputs["text_embeds"],
                attention_mask=av_inputs["text_mask"],
                frame_rate=av_inputs["frame_rate"],
                transformer_options=av_inputs["transformer_options"],
            )
        if not isinstance(lora_pred, (list, tuple)) or len(lora_pred) < 2:
            logger.warning("Audio DOP: transformer did not return [video, audio] — skipping.")
            del audio_prior
            clean_memory_on_device(device)
            return 0.0
        audio_lora = lora_pred[1]
        del lora_pred

        # (c) MSE on audio predictions × multiplier
        adop_loss = F.mse_loss(audio_lora.float(), audio_prior.float()) * mult

        # (d) NaN guard
        if not torch.isfinite(adop_loss):
            logger.warning("Audio DOP loss is non-finite (%.4g), skipping backward.", adop_loss.item())
            del audio_prior, audio_lora, adop_loss
            clean_memory_on_device(device)
            return float("nan")

        # (e) separate backward
        accelerator.backward(adop_loss)

        loss_val = adop_loss.detach().item()

        # cleanup
        del audio_prior, audio_lora, adop_loss
        clean_memory_on_device(device)

        return loss_val
