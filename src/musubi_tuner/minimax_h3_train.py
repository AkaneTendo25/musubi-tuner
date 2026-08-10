from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence

import torch

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer
from musubi_tuner.minimax_h3_train_network import setup_parser as setup_h3_parser
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file

logger = logging.getLogger(__name__)


class MiniMaxH3FullFinetuneModule(torch.nn.Module):
    """Adapter between the shared trainer lifecycle and dense H3 parameters."""

    def __init__(self, transformer: torch.nn.Module, *, save_weights: bool = True) -> None:
        super().__init__()
        self.transformer = transformer
        self._save_weights = save_weights

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def prepare_optimizer_params(self, unet_lr: float, **_kwargs):
        name_and_params = [(name, parameter) for name, parameter in self.transformer.named_parameters() if parameter.requires_grad]
        if not name_and_params:
            raise ValueError("no trainable MiniMax H3 transformer parameters were found")
        return [{"params": [parameter for _, parameter in name_and_params], "lr": unet_lr}], ["transformer"]

    def prepare_grad_etc(self, _transformer) -> None:
        self.train()

    def on_epoch_start(self, _transformer) -> None:
        self.train()

    def on_step_start(self) -> None:
        pass

    def get_trainable_params(self):
        return (parameter for parameter in self.transformer.parameters() if parameter.requires_grad)

    def apply_max_norm_regularization(self, *_args, **_kwargs):
        raise ValueError("--scale_weight_norms is not supported by MiniMax H3 full fine-tuning")

    def save_weights(self, file: str, dtype: torch.dtype | None, metadata: dict | None) -> None:
        if not self._save_weights:
            logger.warning("Skipping MiniMax H3 dense checkpoint write because --debug_no_save_weights is set")
            return
        offloader = getattr(self.transformer, "offloader", None)
        synchronize = getattr(offloader, "synchronize", None)
        if callable(synchronize):
            synchronize()
        state_dict = self.transformer.state_dict()
        if any("_orig_mod." in key for key in state_dict):
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
        if dtype not in (None, torch.bfloat16):
            raise ValueError("MiniMax H3 dense checkpoints can only be saved in native BF16 precision")
        mem_eff_save_file(state_dict, file, metadata)


class MiniMaxH3Trainer(MiniMaxH3NetworkTrainer):
    """Full-parameter BF16 MiniMax H3 trainer with fused Adafactor updates."""

    def _validate_full_finetune_args(self, args: argparse.Namespace) -> None:
        if args.mixed_precision != "bf16" or not args.full_bf16:
            raise ValueError("MiniMax H3 full fine-tuning requires --mixed_precision bf16 and --full_bf16")
        if args.full_fp16:
            raise ValueError("MiniMax H3 full fine-tuning does not support --full_fp16")
        if args.fp8_base or args.int8_convrot_base or args.h3_convrot_int8:
            raise ValueError(
                "MiniMax H3 full fine-tuning requires ordinary BF16 weights; FP8 and INT8 bases are frozen-weight paths"
            )
        if args.h3_adaln_rank is not None:
            raise ValueError(
                "MiniMax H3 full fine-tuning cannot use --h3_adaln_rank because it changes the checkpoint architecture"
            )
        if args.h3_convrot_int8_lora_fused:
            raise ValueError("--h3_convrot_int8_lora_fused is a LoRA-only option")
        if args.block_swap_h2d_only:
            raise ValueError("MiniMax H3 full fine-tuning cannot use frozen-weight --block_swap_h2d_only")
        if args.block_swap_granularity != "block":
            raise ValueError("MiniMax H3 full fine-tuning supports block-granular swap only")
        if args.base_weights or args.network_weights or args.dim_from_weights:
            raise ValueError("MiniMax H3 full fine-tuning does not accept LoRA initialization or merge weights")
        if args.network_args or args.network_dropout is not None or args.scale_weight_norms:
            raise ValueError("MiniMax H3 full fine-tuning does not accept LoRA network/dropout/max-norm options")
        if args.h3_base_preservation_loss_weight > 0:
            raise ValueError("--h3_base_preservation_loss_weight is not supported by MiniMax H3 full fine-tuning")
        if args.crepa is not None:
            raise ValueError("--crepa is not currently part of the dense H3 checkpoint")
        if args.save_precision not in (None, "bf16"):
            raise ValueError("MiniMax H3 dense checkpoints must be saved as BF16 (omit --save_precision or use bf16)")
        if args.fused_backward_pass:
            if args.optimizer_type.lower() != "adafactor":
                raise ValueError("--fused_backward_pass currently requires --optimizer_type Adafactor for dense H3")
            if args.max_grad_norm != 0.0:
                raise ValueError(
                    "--fused_backward_pass requires --max_grad_norm 0 so each gradient can be stepped and freed immediately"
                )
        if args.adafactor_triton and not args.fused_backward_pass:
            raise ValueError("--adafactor_triton requires --fused_backward_pass")
        if args.adafactor_triton:
            optimizer_args = {}
            for item in args.optimizer_args or []:
                key, separator, value = item.partition("=")
                if not separator:
                    raise ValueError(f"invalid --optimizer_args entry {item!r}; expected key=value")
                optimizer_args[key] = value.lower()
            if optimizer_args.get("scale_parameter") != "false" or optimizer_args.get("relative_step") != "false":
                raise ValueError(
                    "--adafactor_triton requires manual-LR Adafactor arguments: "
                    "--optimizer_args scale_parameter=False relative_step=False warmup_init=False"
                )
        if args.block_swap_trainable_ring and not (args.blocks_to_swap or 0):
            raise ValueError("--block_swap_trainable_ring requires --blocks_to_swap")

    def _validate_args_and_init(self, args) -> bool:
        self._validate_full_finetune_args(args)
        return super()._validate_args_and_init(args)

    def _prepare_accelerator_and_dtypes(self, args):
        result = super()._prepare_accelerator_and_dtypes(args)
        accelerator = result[0]
        if accelerator.num_processes != 1:
            raise ValueError("MiniMax H3 block-swapped full fine-tuning currently supports one process/GPU")
        return result

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        del accelerator, vae, weight_dtype
        transformer.requires_grad_(True)
        transformer.train()
        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
        return MiniMaxH3FullFinetuneModule(transformer, save_weights=not args.debug_no_save_weights)

    def on_train_start(self, args, accelerator, network, transformer, optimizer) -> None:
        del args, network, transformer, optimizer
        self._dense_step_started_at = time.perf_counter()
        self._dense_step_metrics: dict[str, float] = {}
        if accelerator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(accelerator.device)

    def on_post_optimizer_step(
        self,
        args,
        accelerator,
        network,
        transformer,
        sync_gradients: bool,
        global_step: int,
    ) -> None:
        del args, network, transformer
        if not sync_gradients:
            return
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
        elapsed = time.perf_counter() - self._dense_step_started_at
        metrics = {"dense_step_seconds": elapsed}
        if accelerator.device.type == "cuda":
            metrics.update(
                {
                    "dense_peak_allocated_gib": torch.cuda.max_memory_allocated(accelerator.device) / (1024**3),
                    "dense_peak_reserved_gib": torch.cuda.max_memory_reserved(accelerator.device) / (1024**3),
                }
            )
        self._dense_step_metrics = metrics
        logger.info(
            "MiniMax H3 dense step %d: %.3fs, peak allocated %.3f GiB, peak reserved %.3f GiB",
            global_step + 1,
            elapsed,
            metrics.get("dense_peak_allocated_gib", 0.0),
            metrics.get("dense_peak_reserved_gib", 0.0),
        )
        self._dense_step_started_at = time.perf_counter()
        if accelerator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(accelerator.device)

    def extra_step_logs(self, args: argparse.Namespace, logs: dict) -> dict:
        del args, logs
        return getattr(self, "_dense_step_metrics", {})

    def _prepare_with_accelerator(
        self,
        args,
        accelerator,
        transformer,
        network,
        optimizer,
        train_dataloader,
        lr_scheduler,
        weight_dtype,
        dit_dtype,
        dit_weight_dtype,
    ):
        del weight_dtype, dit_weight_dtype
        blocks_to_swap = self.blocks_to_swap or 0
        if args.compile:
            transformer = self.compile_transformer(args, transformer)
            network.transformer = transformer

        if blocks_to_swap > 0:
            network = accelerator.prepare(network, device_placement=[False])
            unwrapped = accelerator.unwrap_model(network)
            unwrapped.transformer.move_to_device_except_swap_blocks(accelerator.device)
            unwrapped.transformer.prepare_block_swap_before_forward()
            unwrapped.transformer.switch_block_swap_for_training()
        else:
            network = accelerator.prepare(network)

        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        transformer = accelerator.unwrap_model(network).transformer
        transformer.train()
        self._install_fused_optimizer(args, accelerator, optimizer)
        return transformer, network, optimizer, train_dataloader, lr_scheduler, network, dit_dtype

    @staticmethod
    def _install_fused_optimizer(args, accelerator, optimizer) -> None:
        if not args.fused_backward_pass:
            return
        if args.adafactor_triton:
            from musubi_tuner.modules.adafactor_triton import patch_adafactor_triton

            patch_adafactor_triton(optimizer)
            logger.info("MiniMax H3 dense Adafactor uses the Triton 2D BF16 fast path")
        else:
            from musubi_tuner.modules.adafactor_fused import patch_adafactor_fused

            patch_adafactor_fused(optimizer)
            logger.info("MiniMax H3 dense Adafactor uses fused per-parameter backward updates")

        def make_hook(parameter: torch.nn.Parameter, param_group: dict):
            def grad_hook(_tensor: torch.Tensor) -> None:
                if not accelerator.sync_gradients:
                    return
                optimizer.step_param(parameter, param_group)
                parameter.grad = None

            return grad_hook

        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:
                    parameter.register_post_accumulate_grad_hook(make_hook(parameter, param_group))

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        metadata = super().extra_metadata(args)
        metadata.update(
            {
                "ss_training_type": "full_finetune",
                "ss_fused_backward_pass": str(args.fused_backward_pass),
                "ss_adafactor_triton": str(args.adafactor_triton),
                "ss_block_swap_trainable_ring": str(args.block_swap_trainable_ring),
            }
        )
        return metadata


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = setup_h3_parser(parser)
    parser.description = "Full-parameter BF16 training for MiniMax H3 FL2VA and Ref2VA"
    parser.add_argument("--full_fp16", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--full_bf16", action="store_true", help="train MiniMax H3 weights directly in BF16")
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help="step and clear each Adafactor parameter from its post-accumulate gradient hook",
    )
    parser.add_argument(
        "--adafactor_triton",
        action="store_true",
        help="use fused Triton Adafactor kernels for supported contiguous 2D BF16 weights",
    )
    parser.add_argument(
        "--block_swap_trainable_ring",
        action="store_true",
        help="stream trainable blocks through a coalesced pinned-memory GPU ring; requires fused backward",
    )
    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help="stream the full checkpoint tensor-by-tensor instead of cloning the model in host memory",
    )
    parser.add_argument("--debug_no_save_weights", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(
        network_module=None,
        mixed_precision="bf16",
        full_bf16=True,
        optimizer_type="Adafactor",
        optimizer_args=["scale_parameter=False", "relative_step=False", "warmup_init=False"],
        learning_rate=1e-6,
        lr_scheduler="constant_with_warmup",
        max_grad_norm=0.0,
        gradient_checkpointing=True,
        fused_backward_pass=True,
        mem_eff_save=True,
        output_name="minimax_h3_full_finetune",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(setup_parser_common())


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    args = read_config_from_file(args, parser)
    args.dit_dtype = None
    MiniMaxH3Trainer().train(args)


if __name__ == "__main__":
    main()
