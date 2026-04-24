"""
MMControl training entry point for LTX-2.

MMControl is implemented as a thin, explicit training surface over the existing
residual-hint control machinery on this branch. A frozen LTX-2 DiT receives
trainable visual and audio bypass hints through residual injection. This module
changes the defaults and names to MMControl:

  - even-numbered DiT layers by default,
  - separate visual/audio guidance scale aliases,
  - optional audio bypass initialization from scratch,
  - MMControl checkpoint names for full-branch training.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict

from safetensors.torch import save_file

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.ltx2_train_network import (
    LTX2NetworkTrainer,
    apply_ltx2_tweaks,
    ltx2_setup_parser,
)
from musubi_tuner.ltx2_vace_train import LTX2VaceTrainer, vace_setup_parser

logger = logging.getLogger(__name__)

MMCONTROL_DEFAULT_LAYERS = tuple(range(0, 48, 2))


def _layers_to_arg(layers: tuple[int, ...]) -> str:
    return ",".join(str(layer) for layer in layers)


class LTX2MMControlTrainer(LTX2VaceTrainer):
    """Trainer for MMControl-style dual-stream bypass control."""

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        mm_layers = getattr(args, "mmcontrol_layers", None)
        if mm_layers and not getattr(args, "vace_layers", None):
            args.vace_layers = mm_layers
        elif not getattr(args, "vace_layers", None):
            args.vace_layers = _layers_to_arg(MMCONTROL_DEFAULT_LAYERS)

        if mm_layers and not getattr(args, "audio_vace_layers", None):
            args.audio_vace_layers = mm_layers
        elif not getattr(args, "audio_vace_layers", None):
            args.audio_vace_layers = args.vace_layers

        visual_guidance = getattr(args, "visual_guidance_scale", None)
        if visual_guidance is not None:
            args.vace_scale = visual_guidance

        audio_guidance = getattr(args, "audio_guidance_scale", None)
        if audio_guidance is not None:
            args.audio_vace_scale = audio_guidance

        visual_model_path = getattr(args, "visual_control_model_path", None)
        if visual_model_path and not getattr(args, "vace_model_path", None):
            args.vace_model_path = visual_model_path

        audio_model_path = getattr(args, "audio_control_model_path", None)
        if audio_model_path and not getattr(args, "audio_vace_model_path", None):
            args.audio_vace_model_path = audio_model_path

        if getattr(args, "enable_audio_mmcontrol", False):
            args.enable_audio_vace = True

        super().handle_model_specific_args(args)
        logger.info(
            "MMControl config: visual_layers=%s audio_layers=%s "
            "visual_scale=%.2f audio_scale=%.2f audio_branch=%s",
            self._vace_layers,
            self._audio_vace_layers,
            self._vace_scale,
            self._audio_vace_scale,
            self._enable_audio_vace,
        )

    def get_checkpoint_metadata(self, args: argparse.Namespace) -> Dict[str, Any]:
        md = super().get_checkpoint_metadata(args)
        md["ss_mmcontrol_training"] = True
        md["ss_mmcontrol_visual_layers"] = _layers_to_arg(self._vace_layers)
        md["ss_mmcontrol_audio_layers"] = _layers_to_arg(self._audio_vace_layers)
        return md

    def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False):
        """Save MMControl branch weights using MMControl-specific suffixes."""
        LTX2NetworkTrainer.post_save_checkpoint_hook(
            self,
            args,
            ckpt_file,
            ckpt_name,
            accelerator,
            force_sync_upload=force_sync_upload,
        )
        if not self._train_vace_full_model:
            return
        if self._vace_model is not None:
            visual_path = ckpt_file.replace(".safetensors", "_mmcontrol_visual.safetensors")
            state_dict = {
                k: v.contiguous().cpu()
                for k, v in self._vace_model.state_dict().items()
            }
            save_file(state_dict, visual_path)
            logger.info("Saved MMControl visual branch weights: %s", visual_path)
        if self._audio_vace_model is not None:
            audio_path = ckpt_file.replace(".safetensors", "_mmcontrol_audio.safetensors")
            state_dict = {
                k: v.contiguous().cpu()
                for k, v in self._audio_vace_model.state_dict().items()
            }
            save_file(state_dict, audio_path)
            logger.info("Saved MMControl audio branch weights: %s", audio_path)


def mmcontrol_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("MMControl Training")
    group.add_argument(
        "--mmcontrol_layers",
        type=str,
        default=None,
        help=(
            "Comma-separated DiT block indices for MMControl hint injection. "
            "Default: all even layers, 0,2,4,...,46."
        ),
    )
    group.add_argument(
        "--visual_guidance_scale",
        type=float,
        default=None,
        help="Alias for --vace_scale; controls visual branch residual strength.",
    )
    group.add_argument(
        "--audio_guidance_scale",
        type=float,
        default=None,
        help="Alias for --audio_vace_scale; controls audio branch residual strength.",
    )
    group.add_argument(
        "--enable_audio_mmcontrol",
        action="store_true",
        default=False,
        help="Initialize/train the MMControl audio bypass branch from scratch.",
    )
    group.add_argument(
        "--visual_control_model_path",
        type=str,
        default=None,
        help="Alias for --vace_model_path.",
    )
    group.add_argument(
        "--audio_control_model_path",
        type=str,
        default=None,
        help="Alias for --audio_vace_model_path.",
    )
    return parser


def main() -> None:
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)
    parser = vace_setup_parser(parser)
    parser = mmcontrol_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if hasattr(args, "ltx_mode"):
        short_map = {"v": "video", "a": "audio", "va": "av"}
        if args.ltx_mode in short_map:
            args.ltx_mode = short_map[args.ltx_mode]

    args.allow_custom_weighting_scheme = True
    apply_ltx2_tweaks(args)

    if getattr(args, "dit", None) is not None and args.dit != args.ltx2_checkpoint:
        logger.warning("Ignoring --dit for LTX-2; using --ltx2_checkpoint instead")
    args.dit = args.ltx2_checkpoint

    if getattr(args, "vae", None) is not None and args.vae != args.ltx2_checkpoint:
        logger.warning("Ignoring --vae for LTX-2; using --ltx2_checkpoint instead")
    args.vae = args.ltx2_checkpoint

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    if not getattr(args, "lora_target_preset", None):
        args.lora_target_preset = "vace"
        logger.info("Defaulting to 'vace' LoRA target preset for MMControl bypass modules")

    if getattr(args, "lora_target_preset", None) is not None:
        if args.network_args is None:
            args.network_args = []
        if not any(arg.startswith("lora_target_preset=") for arg in args.network_args):
            args.network_args.append(f"lora_target_preset={args.lora_target_preset}")
            logger.info("Using LoRA target preset: %s", args.lora_target_preset)

    trainer = LTX2MMControlTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
