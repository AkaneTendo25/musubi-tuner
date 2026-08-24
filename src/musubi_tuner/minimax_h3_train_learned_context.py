from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.backend import create_conditioning_encoder
from musubi_tuner.minimax_h3.learned_context import (
    COMFYUI_LEARNED_CONTEXT_KEY,
    H3_TEXT_HIDDEN_SIZE,
    apply_learned_context,
    compose_learned_context,
    load_learned_context,
    prepend_learned_context,
    validate_learned_context,
)
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer
from musubi_tuner.minimax_h3_train_network import setup_parser as setup_h3_parser
from musubi_tuner.utils import async_save
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)


class H3LearnedContext(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(validate_learned_context(weight).detach().to(torch.float32).clone())
        self.enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def apply_to(self, _text_encoder, transformer, **_kwargs) -> None:
        # The learned rows are the only trainable parameters. LoRA networks
        # freeze the base inside apply_to(); this lightweight network must do
        # the same or autograd allocates gradients for the entire 33B model.
        if transformer is not None:
            transformer.requires_grad_(False)

    def enable_gradient_checkpointing(self) -> None:
        pass

    def prepare_optimizer_params(self, unet_lr: float, **_kwargs):
        return [{"params": [self.weight], "lr": unet_lr}], ["learned_context"]

    def prepare_grad_etc(self, transformer) -> None:
        transformer.requires_grad_(False)
        self.train()

    def on_epoch_start(self, _transformer) -> None:
        self.train()

    def on_step_start(self) -> None:
        if not torch.isfinite(self.weight).all():
            raise FloatingPointError("H3 learned-context parameter became non-finite")

    def get_trainable_params(self):
        return (self.weight,)

    def apply_max_norm_regularization(self, *_args, **_kwargs):
        raise ValueError("--scale_weight_norms is not supported by H3 learned-context training")

    def prepend(self, hidden: torch.Tensor, tags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.weight if self.enabled else None
        return prepend_learned_context(hidden, tags, context)

    def snapshot_weights(self, dtype: torch.dtype | None):
        export_dtype = torch.bfloat16 if dtype is None else dtype
        if not torch.isfinite(self.weight).all():
            raise FloatingPointError("refusing to save a non-finite H3 learned-context parameter")
        snapshot = self.weight.detach().to(device="cpu", dtype=export_dtype, copy=True).contiguous()
        return {COMFYUI_LEARNED_CONTEXT_KEY: snapshot}

    def save_weights(self, file: str, dtype: torch.dtype | None, metadata: dict | None) -> None:
        async_save.write_state_dict_file(self.snapshot_weights(dtype), file, metadata)


class MiniMaxH3LearnedContextTrainer(MiniMaxH3NetworkTrainer):
    def __init__(self) -> None:
        super().__init__()
        self._learned_context: H3LearnedContext | None = None
        self._learned_context_composition = "prepend"

    @staticmethod
    def _initial_weight(args: argparse.Namespace) -> torch.Tensor:
        prompt_weight = getattr(args, "_h3_learned_context_prompt_weight", None)
        if prompt_weight is not None:
            return validate_learned_context(prompt_weight)
        if args.h3_learned_context_init is not None:
            return load_learned_context(args.h3_learned_context_init)
        raise ValueError("use --h3_learned_context_init_prompt for a new context or --h3_learned_context_init to continue one")

    def _validate_learned_context_args(self, args: argparse.Namespace) -> None:
        initializers = (
            args.h3_learned_context_init is not None,
            args.h3_learned_context_init_prompt is not None,
        )
        if sum(initializers) != 1:
            raise ValueError("choose exactly one of --h3_learned_context_init_prompt or --h3_learned_context_init")
        if args.h3_learned_context_init_prompt is not None and not args.text_encoder:
            raise ValueError("--h3_learned_context_init_prompt requires --text_encoder")
        if args.network_weights or args.dim_from_weights or args.base_weights:
            raise ValueError("H3 learned-context training uses --h3_learned_context_init, not LoRA weight arguments")
        if args.network_args or args.network_dropout is not None or args.scale_weight_norms:
            raise ValueError("H3 learned-context training does not accept LoRA network/dropout/max-norm options")
        if args.h3_base_preservation_loss_weight > 0:
            raise ValueError("base-preservation loss is not supported for H3 learned-context training")
        if args.h3_guidance_distillation_scale is not None:
            raise ValueError("guidance-distillation is not supported for H3 learned-context training")
        if args.h3_caption_dropout_rate > 0:
            raise ValueError("caption dropout is not supported for H3 learned-context training")
        if args.save_precision not in (None, "bf16", "float", "fp32"):
            raise ValueError("H3 learned contexts can be saved as BF16 or FP32")

    def _validate_args_and_init(self, args) -> bool:
        self._validate_learned_context_args(args)
        return super()._validate_args_and_init(args)

    def _prepare_accelerator_and_dtypes(self, args):
        result = super()._prepare_accelerator_and_dtypes(args)
        if result[0].num_processes != 1:
            raise ValueError("H3 learned-context training currently supports one process/GPU")
        return result

    @staticmethod
    def _encode_prompt_initializer(args, accelerator) -> None:
        logger.info("Encoding learned-context initializer with frozen Qwen before loading the transformer")
        encoder = create_conditioning_encoder(
            text_encoder=Path(args.text_encoder),
            tokenizer=Path(args.tokenizer),
            task="t2va",
            device=str(accelerator.device),
            dtype="bfloat16",
            quantization=args.text_encoder_quantization,
            blocks_to_stream=args.h3_text_encoder_blocks_to_stream,
            nvfp4_scaled_mm=args.h3_nvfp4_scaled_mm,
            text_visual_max_pixels=args.h3_text_visual_max_pixels,
        )
        try:
            conditioning = encoder.encode_prompt(args.h3_learned_context_init_prompt)
            prompt_weight = conditioning[H3_TEXT_HIDDEN_KEY].detach().to(device="cpu")
        finally:
            encoder.close()
            del encoder
            gc.collect()
            clean_memory_on_device(accelerator.device)
        args._h3_learned_context_prompt_weight = prompt_weight

    def _prepare_sampling(self, args, accelerator, vae_dtype):
        if args.h3_learned_context_init_prompt is not None:
            self._encode_prompt_initializer(args, accelerator)
        return super()._prepare_sampling(args, accelerator, vae_dtype)

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        del accelerator, vae, weight_dtype
        self._learned_context = H3LearnedContext(self._initial_weight(args))
        self._learned_context_composition = args.h3_learned_context_composition
        self._learned_context.apply_to(None, transformer)
        if transformer is not None:
            if args.gradient_checkpointing:
                transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
        return self._learned_context

    def _predict(self, accelerator, transformer, batch, inputs, *, conditioning: str):
        if self._learned_context is None:
            raise RuntimeError("H3 learned context is not initialized")
        hidden_key = H3_TEXT_HIDDEN_KEY
        tags_key = H3_TEXT_TOKEN_TAGS_KEY
        if conditioning == "empty":
            from musubi_tuner.minimax_h3.cache import H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY

            hidden_key, tags_key = H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY
        hidden = batch[hidden_key]
        tags = batch[tags_key]
        container = list if isinstance(hidden, list) else tuple if isinstance(hidden, tuple) else None
        if container is not None:
            if len(hidden) != 1 or len(tags) != 1:
                raise ValueError("H3 learned-context training requires batch size one")
            context = self._learned_context.weight if self._learned_context.enabled else None
            composed_hidden, composed_tags = compose_learned_context(hidden[0], tags[0], context, self._learned_context_composition)
            conditioned_batch = {**batch, hidden_key: container((composed_hidden,)), tags_key: container((composed_tags,))}
        elif hidden.ndim == 3 and hidden.shape[0] == 1 and tags.ndim == 2 and tags.shape[0] == 1:
            context = self._learned_context.weight if self._learned_context.enabled else None
            composed_hidden, composed_tags = compose_learned_context(hidden[0], tags[0], context, self._learned_context_composition)
            conditioned_batch = {**batch, hidden_key: composed_hidden[None], tags_key: composed_tags[None]}
        else:
            context = self._learned_context.weight if self._learned_context.enabled else None
            composed_hidden, composed_tags = compose_learned_context(hidden, tags, context, self._learned_context_composition)
            conditioned_batch = {**batch, hidden_key: composed_hidden, tags_key: composed_tags}
        return super()._predict(accelerator, transformer, conditioned_batch, inputs, conditioning=conditioning)

    def _generate_sample(self, accelerator, transformer, decoder_bundle, sample_parameter):
        if self._learned_context is None:
            raise RuntimeError("H3 learned context is not initialized")
        conditioning = apply_learned_context(
            {
                H3_TEXT_HIDDEN_KEY: sample_parameter[H3_TEXT_HIDDEN_KEY],
                H3_TEXT_TOKEN_TAGS_KEY: sample_parameter[H3_TEXT_TOKEN_TAGS_KEY],
            },
            self._learned_context.weight.detach(),
            self._learned_context_composition,
        )
        return super()._generate_sample(
            accelerator,
            transformer,
            decoder_bundle,
            {**sample_parameter, **conditioning},
        )

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        metadata = super().extra_metadata(args)
        metadata.update(
            {
                "ss_training_type": "h3_learned_context",
                "ss_h3_learned_context_tokens": str(
                    self._learned_context.weight.shape[0] if self._learned_context is not None else "unknown"
                ),
                "ss_h3_learned_context_hidden_size": str(H3_TEXT_HIDDEN_SIZE),
                "ss_h3_learned_context_format": COMFYUI_LEARNED_CONTEXT_KEY,
                "ss_h3_learned_context_composition": args.h3_learned_context_composition,
            }
        )
        return metadata


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = setup_h3_parser(parser)
    parser.description = "Train a ComfyUI-compatible MiniMax H3 learned context"
    parser.add_argument("--h3_learned_context_init", type=Path, help="saved H3 learned context to continue training")
    parser.add_argument(
        "--h3_learned_context_init_prompt",
        type=str,
        help="encode this prompt once with frozen Qwen and use its complete hidden-state sequence as initialization",
    )
    parser.add_argument(
        "--h3_learned_context_composition",
        choices=("prepend", "replace"),
        default="prepend",
        help="prepend the context to the cached caption, or replace the caption with the context",
    )
    parser.set_defaults(
        network_module=None,
        network_dim=None,
        network_alpha=None,
        learning_rate=1e-4,
        optimizer_type="AdamW",
        lr_scheduler="constant",
        gradient_checkpointing=True,
        output_name="h3_learned_context",
        save_precision="bf16",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(setup_parser_common())


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    args = read_config_from_file(args, parser)
    args.dit_dtype = None
    MiniMaxH3LearnedContextTrainer().train(args)


if __name__ == "__main__":
    main()
