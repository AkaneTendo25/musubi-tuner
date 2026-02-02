#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from multiprocessing import Value
from typing import Optional

import toml
import torch
from accelerate import Accelerator
from tqdm import tqdm
from safetensors.torch import save_file

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.hv_train_network import (
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_MINIMUM_KEYS,
    clean_memory_on_device,
    collator_class,
    compute_loss_weighting_for_sd3,
    prepare_accelerator,
    read_config_from_file,
    set_seed,
    setup_parser_common,
    should_sample_images,
)
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.utils import huggingface_utils, model_utils, sai_model_spec, train_utils
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer, ltx2_setup_parser

import copy
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EMAModel:
    """Exponential Moving Average of model weights.

    Maintains shadow weights that are updated as:
        ema_weights = decay * ema_weights + (1 - decay) * model_weights

    Reference: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    Similar to diffusers EMAModel but simplified for our use case.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update, more smoothing)
            update_after_step: Start EMA decay after this many steps (warmup period
                              where shadow = current weights)
            update_every: Update EMA every N steps
            device: Device to store EMA weights (None = same as model)
        """
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0
        self._decay_started = False

        # Create shadow parameters (initialized from model)
        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                if device is not None:
                    self.shadow_params[name] = param.data.clone().to(device)
                else:
                    self.shadow_params[name] = param.data.clone()

    def _copy_to_shadow(self, model: torch.nn.Module) -> None:
        """Copy current model weights to shadow (used during warmup)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params and param.requires_grad:
                    self.shadow_params[name].copy_(param.data.to(self.shadow_params[name].device))

    def update(self, model: torch.nn.Module) -> None:
        """Update EMA weights.

        During warmup (step < update_after_step): shadow = current weights
        After warmup: shadow = decay * shadow + (1-decay) * current
        """
        self.step += 1

        # During warmup period, just copy current weights to shadow
        # This ensures EMA starts from a reasonable state
        if self.step <= self.update_after_step:
            self._copy_to_shadow(model)
            return

        # Check if this is an update step
        if (self.step - self.update_after_step) % self.update_every != 0:
            return

        # Perform EMA update: shadow = decay * shadow + (1 - decay) * current
        # Using lerp_: shadow.lerp_(current, 1-decay) = shadow*(1-(1-decay)) + current*(1-decay)
        #            = shadow*decay + current*(1-decay)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params and param.requires_grad:
                    self.shadow_params[name].lerp_(param.data.to(self.shadow_params[name].device), 1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> dict:
        """Apply EMA weights to model, returning original weights for restoration."""
        original_params = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    original_params[name] = param.data.clone()
                    param.data.copy_(self.shadow_params[name].to(param.device))
        return original_params

    def restore(self, model: torch.nn.Module, original_params: dict) -> None:
        """Restore original weights to model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "step": self.step,
            "shadow_params": {k: v.cpu() for k, v in self.shadow_params.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.step = state_dict["step"]
        for name, param in state_dict["shadow_params"].items():
            if name in self.shadow_params:
                self.shadow_params[name].copy_(param.to(self.shadow_params[name].device))


def _masked_mse(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    weighting: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(tgt, torch.Tensor):
        pred = pred.to(device=tgt.device, dtype=dtype)
    else:
        pred = pred.to(dtype=dtype)
    per_elem = torch.nn.functional.mse_loss(pred, tgt, reduction="none")
    if weighting is not None:
        w = weighting
        if isinstance(w, torch.Tensor) and w.dim() != per_elem.dim():
            while w.dim() > per_elem.dim() and w.shape[-1] == 1:
                w = w.squeeze(-1)
        per_elem = per_elem * w
    if mask is None:
        return per_elem.mean()

    mask = mask.to(device=per_elem.device)
    if per_elem.dim() == 5 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
    elif per_elem.dim() == 5 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)
    elif per_elem.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif per_elem.dim() == 3 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1)

    mask_f = mask.to(dtype=per_elem.dtype)
    denom = mask_f.mean()
    if denom.item() == 0:
        return per_elem.mean()
    return (per_elem * mask_f).div(denom).mean()


def ltx2_finetune_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help="Use fused backward pass for Adafactor optimizer",
    )
    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help=(
            "Enable memory efficient saving (saving states requires normal saving, so it takes same amount of memory "
            "even with this option enabled)"
        ),
    )
    # EMA arguments
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use Exponential Moving Average of model weights for more stable training",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate (higher = slower update, more smoothing). Default: 0.9999",
    )
    parser.add_argument(
        "--ema_update_after_step",
        type=int,
        default=100,
        help="Start EMA updates after this many steps. Default: 100",
    )
    parser.add_argument(
        "--ema_update_every",
        type=int,
        default=1,
        help="Update EMA every N steps. Default: 1",
    )
    parser.add_argument(
        "--save_ema_only",
        action="store_true",
        help="When using EMA, only save EMA weights (not training weights)",
    )
    parser.add_argument(
        "--ema_cpu_offload",
        action="store_true",
        help="Store EMA shadow weights on CPU to save GPU memory (slower updates but no extra VRAM)",
    )
    # Validation arguments
    parser.add_argument(
        "--validation_dataset_config",
        type=str,
        default=None,
        help="Path to validation dataset config (TOML). If not set, validation is disabled.",
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation every N training steps",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=1,
        help="Run validation every N epochs. Default: 1",
    )
    parser.add_argument(
        "--num_validation_batches",
        type=int,
        default=None,
        help="Number of validation batches to use (None = all)",
    )
    return parser


def main() -> None:
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)
    parser = ltx2_finetune_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = LTX2NetworkTrainer()

    if args.dataset_config is None:
        raise ValueError("dataset_config is required / dataset_configが必要です")
    if args.ltx2_checkpoint is None:
        raise ValueError("path to LTX-2 checkpoint is required / LTX-2チェックポイントのパスが必要です")

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

    trainer.handle_model_specific_args(args)

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)
    session_id = random.randint(0, 2**32)
    training_started_at = time.time()

    accelerator = prepare_accelerator(args)
    if args.mixed_precision is None:
        args.mixed_precision = accelerator.mixed_precision

    # sample prompts (optional)
    sample_parameters = None
    vae = None
    if args.sample_prompts or getattr(args, "precache_sample_prompts", False) or getattr(args, "use_precached_sample_prompts", False):
        sample_prompt_path = args.sample_prompts or ""
        sample_parameters = trainer.process_sample_prompts(args, accelerator, sample_prompt_path)
        vae = trainer.load_vae(args, vae_dtype=model_utils.str_to_dtype(args.vae_dtype), vae_path=args.vae)
        vae.requires_grad_(False)
        vae.eval()

    # datasets
    current_epoch = Value("i", 0)
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=trainer.architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.dataset_group,
        training=True,
        num_timestep_buckets=args.num_timestep_buckets,
        shared_epoch=current_epoch,
    )

    if train_dataset_group.num_train_items == 0:
        raise ValueError(
            "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
            " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
        )

    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, ds_for_collator)
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # Validation dataset (optional)
    val_dataloader = None
    if args.validation_dataset_config is not None:
        logger.info("Loading validation dataset from: %s", args.validation_dataset_config)
        val_user_config = config_utils.load_user_config(args.validation_dataset_config)
        val_blueprint = blueprint_generator.generate(val_user_config, args, architecture=trainer.architecture)
        val_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            val_blueprint.dataset_group,
            training=False,  # validation mode
            num_timestep_buckets=args.num_timestep_buckets,
            shared_epoch=current_epoch,
        )
        if val_dataset_group.num_train_items > 0:
            val_collator = collator_class(current_epoch, val_dataset_group if args.max_data_loader_n_workers == 0 else None)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset_group,
                batch_size=1,
                shuffle=False,
                collate_fn=val_collator,
                num_workers=n_workers,
                persistent_workers=False,
            )
            logger.info("Validation dataset loaded with %d items", val_dataset_group.num_train_items)
        else:
            logger.warning("Validation dataset has no items, validation disabled")

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )

    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # model
    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    trainer.blocks_to_swap = blocks_to_swap
    loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

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

    transformer = trainer.load_transformer(
        accelerator=accelerator,
        args=args,
        dit_path=args.ltx2_checkpoint,
        attn_mode=attn_mode,
        split_attn=bool(getattr(args, "split_attn", False)),
        loading_device=loading_device,
        dit_weight_dtype=None,
    )

    transformer.train()
    transformer.requires_grad_(True)

    # Clean up memory after model loading
    clean_memory_on_device(accelerator.device)

    if blocks_to_swap > 0:
        logger.info(
            "enable swap %s blocks to CPU from device: %s", blocks_to_swap, accelerator.device
        )
        transformer.enable_block_swap(
            blocks_to_swap,
            accelerator.device,
            supports_backward=True,
            use_pinned_memory=getattr(args, "use_pinned_memory_for_block_swap", False),
        )
        transformer.move_to_device_except_swap_blocks(accelerator.device)

    if args.gradient_checkpointing:
        blocks_to_ckpt = getattr(args, "blocks_to_checkpoint", -1)
        if getattr(args, "blockwise_checkpointing", False):
            transformer.enable_gradient_checkpointing(
                args.gradient_checkpointing_cpu_offload,
                weight_cpu_offloading=True,
                blocks_to_checkpoint=blocks_to_ckpt,
            )
            if args.use_pinned_memory_for_block_swap and hasattr(transformer, "transformer_blocks"):
                for block in transformer.transformer_blocks:
                    if hasattr(block, "use_pinned_memory"):
                        block.use_pinned_memory = True
        else:
            transformer.enable_gradient_checkpointing(
                args.gradient_checkpointing_cpu_offload,
                blocks_to_checkpoint=blocks_to_ckpt,
            )

    # optimizer
    name_and_params = list(transformer.named_parameters())
    params_to_optimize = [{"params": [p for _, p in name_and_params], "lr": args.learning_rate}]
    param_names = [[n for n, _ in name_and_params]]
    optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = trainer.get_optimizer(
        args, params_to_optimize
    )

    # lr scheduler
    lr_scheduler = trainer.get_lr_scheduler(args, optimizer, accelerator.num_processes)

    # prepare accelerator
    if blocks_to_swap > 0:
        transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
        accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
    else:
        transformer = accelerator.prepare(transformer)

    if args.compile:
        transformer = trainer.compile_transformer(args, transformer)
        transformer.__dict__["_orig_mod"] = transformer

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # Prepare validation dataloader if exists
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    trainer.resume_from_local_or_hf_if_specified(accelerator, args)

    # Initialize EMA after model is prepared
    ema_model = None
    ema_state_path = os.path.join(args.output_dir, "ema_state.pt") if args.output_dir else None
    if args.use_ema:
        ema_device = torch.device("cpu") if args.ema_cpu_offload else None
        logger.info(
            "Initializing EMA with decay=%.6f, update_after_step=%d, update_every=%d, device=%s",
            args.ema_decay, args.ema_update_after_step, args.ema_update_every,
            "cpu" if args.ema_cpu_offload else "same as model"
        )
        if not args.ema_cpu_offload:
            logger.warning(
                "EMA shadow weights will be stored on GPU, increasing VRAM usage. "
                "Use --ema_cpu_offload to store EMA on CPU and save GPU memory."
            )
        ema_model = EMAModel(
            accelerator.unwrap_model(transformer),
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_step,
            update_every=args.ema_update_every,
            device=ema_device,
        )
        # Try to load EMA state if resuming
        if args.resume and ema_state_path and os.path.exists(ema_state_path):
            logger.info("Loading EMA state from: %s", ema_state_path)
            ema_state = torch.load(ema_state_path, map_location="cpu", weights_only=True)
            ema_model.load_state_dict(ema_state)
            logger.info("EMA state loaded (step=%d)", ema_model.step)

    if args.fused_backward_pass:
        import musubi_tuner.modules.adafactor_fused as adafactor_fused

        adafactor_fused.patch_adafactor_fused(optimizer)

        for param_group, param_name_group in zip(optimizer.param_groups, param_names):
            for parameter, param_name in zip(param_group["params"], param_name_group):
                if parameter.requires_grad:

                    def create_grad_hook(p_name, p_group):
                        def grad_hook(tensor: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                            optimizer.step_param(tensor, p_group)
                            tensor.grad = None

                        return grad_hook

                    parameter.register_post_accumulate_grad_hook(create_grad_hook(param_name, param_group))

    # scheduler
    noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

    num_train_items = train_dataset_group.num_train_items
    metadata = {
        "ss_session_id": session_id,
        "ss_training_started_at": training_started_at,
        "ss_output_name": args.output_name,
        "ss_learning_rate": args.learning_rate,
        "ss_num_train_items": num_train_items,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": None,
        "ss_gradient_checkpointing": args.gradient_checkpointing,
        "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
        "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
        "ss_max_train_steps": args.max_train_steps,
        "ss_lr_warmup_steps": args.lr_warmup_steps,
        "ss_lr_scheduler": args.lr_scheduler,
        SS_METADATA_KEY_BASE_MODEL_VERSION: trainer.architecture_full_name,
        "ss_mixed_precision": args.mixed_precision,
        "ss_seed": args.seed,
        "ss_training_comment": args.training_comment,
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": args.max_grad_norm,
        "ss_fp8_base": bool(getattr(args, "fp8_base", False)),
        "ss_full_fp16": bool(getattr(args, "full_fp16", False)),
        "ss_full_bf16": bool(getattr(args, "full_bf16", False)),
        "ss_weighting_scheme": args.weighting_scheme,
        "ss_logit_mean": args.logit_mean,
        "ss_logit_std": args.logit_std,
        "ss_mode_scale": args.mode_scale,
        "ss_guidance_scale": args.guidance_scale,
        "ss_timestep_sampling": args.timestep_sampling,
        "ss_sigmoid_scale": args.sigmoid_scale,
        "ss_discrete_flow_shift": args.discrete_flow_shift,
        "ss_ltx_mode": args.ltx_mode,
        "ss_split_av_passes": bool(args.split_av_passes),
        "ss_video_loss_weight": args.video_loss_weight,
        "ss_audio_loss_weight": args.audio_loss_weight,
        "ss_use_ema": args.use_ema,
        "ss_ema_decay": args.ema_decay if args.use_ema else None,
    }

    datasets_metadata = []
    for dataset in train_dataset_group.datasets:
        datasets_metadata.append(dataset.get_metadata())

    metadata["ss_datasets"] = json.dumps(datasets_metadata)

    if args.ltx2_checkpoint is not None:
        logger.info("set LTX-2 model name for metadata: %s", args.ltx2_checkpoint)
        sd_model_name = args.ltx2_checkpoint
        if os.path.exists(sd_model_name):
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if args.vae is not None:
        logger.info("set VAE model name for metadata: %s", args.vae)
        vae_name = args.vae
        if os.path.exists(vae_name):
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name

    metadata = {k: str(v) for k, v in metadata.items()}

    minimum_metadata = {}
    for key in SS_METADATA_MINIMUM_KEYS:
        if key in metadata:
            minimum_metadata[key] = metadata[key]

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "fine-tuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_utils.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

    epoch_to_start = 0
    global_step = 0
    loss_recorder = train_utils.LossRecorder()
    del train_dataset_group

    def save_model(
        ckpt_name: str, unwrapped_model, steps, epoch_no, force_sync_upload=False, use_memory_efficient_saving=False
    ):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        metadata["ss_training_finished_at"] = str(time.time())
        metadata["ss_steps"] = str(steps)
        metadata["ss_epoch"] = str(epoch_no)

        metadata_to_save = minimum_metadata if args.no_metadata else metadata

        title = args.metadata_title if args.metadata_title is not None else args.output_name
        if args.min_timestep is not None or args.max_timestep is not None:
            min_time_step = args.min_timestep if args.min_timestep is not None else 0
            max_time_step = args.max_timestep if args.max_timestep is not None else 1000
            md_timesteps = (min_time_step, max_time_step)
        else:
            md_timesteps = None

        sai_metadata = sai_model_spec.build_metadata(
            args.metadata_reso,
            title=title,
            author=args.metadata_author,
            description=args.metadata_description,
            license=args.metadata_license,
            tags=args.metadata_tags,
            timesteps=md_timesteps,
            custom_arch=args.metadata_arch,
        )
        metadata_to_save.update(sai_metadata)

        save_model_ref = getattr(unwrapped_model, "_orig_mod", None) or unwrapped_model
        state_dict = save_model_ref.state_dict()
        if use_memory_efficient_saving or args.mem_eff_save:
            mem_eff_save_file(state_dict, ckpt_file, metadata_to_save)
        else:
            save_file(state_dict, ckpt_file, metadata_to_save)

        if args.huggingface_repo_id is not None:
            huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def remove_model(old_ckpt_name: str) -> None:
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

    def save_ema_model(ckpt_name: str, steps: int, epoch_no: int) -> None:
        """Save EMA weights as a separate checkpoint."""
        if ema_model is None:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        ema_ckpt_file = os.path.join(args.output_dir, ckpt_name.replace(".safetensors", "_ema.safetensors"))

        accelerator.print(f"\nsaving EMA checkpoint: {ema_ckpt_file}")

        # Create state dict from EMA shadow params
        unwrapped = accelerator.unwrap_model(transformer)
        save_model_ref = getattr(unwrapped, "_orig_mod", None) or unwrapped
        ema_state_dict = {}
        for name, param in save_model_ref.named_parameters():
            if name in ema_model.shadow_params:
                ema_state_dict[name] = ema_model.shadow_params[name].cpu()
            else:
                ema_state_dict[name] = param.data.cpu()

        # Add non-parameter state (buffers)
        for name, buf in save_model_ref.named_buffers():
            ema_state_dict[name] = buf.cpu()

        ema_metadata = metadata.copy()
        ema_metadata["ss_is_ema"] = "True"
        ema_metadata["ss_ema_decay"] = str(args.ema_decay)
        ema_metadata["ss_steps"] = str(steps)
        ema_metadata["ss_epoch"] = str(epoch_no)

        if args.mem_eff_save:
            mem_eff_save_file(ema_state_dict, ema_ckpt_file, ema_metadata)
        else:
            save_file(ema_state_dict, ema_ckpt_file, ema_metadata)

    def save_ema_state() -> None:
        """Save EMA state for resume functionality."""
        if ema_model is None or ema_state_path is None:
            return
        if not accelerator.is_main_process:
            return
        os.makedirs(os.path.dirname(ema_state_path), exist_ok=True)
        torch.save(ema_model.state_dict(), ema_state_path)
        logger.info("EMA state saved to: %s (step=%d)", ema_state_path, ema_model.step)

    def run_validation(step: int, epoch: int) -> dict:
        """Run validation and return metrics."""
        if val_dataloader is None:
            return {}

        accelerator.print(f"\nRunning validation at step {step}...")
        transformer.eval()

        # Apply EMA weights for validation if available
        original_params = None
        if ema_model is not None:
            original_params = ema_model.apply_to(accelerator.unwrap_model(transformer))

        val_losses = []
        val_video_losses = []
        val_audio_losses = []
        num_batches = 0
        max_batches = args.num_validation_batches

        with torch.no_grad():
            for batch in val_dataloader:
                if max_batches is not None and num_batches >= max_batches:
                    break

                latents = batch["latents"]
                if isinstance(latents, dict):
                    latents_tensor = latents["latents"]
                else:
                    latents_tensor = latents

                latents_tensor = trainer.scale_shift_latents(latents_tensor)
                noise = torch.randn_like(latents_tensor)

                noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                    args,
                    noise,
                    latents_tensor,
                    batch["timesteps"],
                    noise_scheduler,
                    accelerator.device,
                    trainer.dit_dtype,
                )

                weighting = compute_loss_weighting_for_sd3(
                    args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, trainer.dit_dtype
                )

                model_pred, target = trainer.call_dit(
                    args,
                    accelerator,
                    transformer,
                    latents_tensor,
                    batch,
                    noise,
                    noisy_model_input,
                    timesteps,
                    trainer.dit_dtype,
                )

                dict_output = isinstance(model_pred, dict)
                if dict_output:
                    out = model_pred
                    if out.get("_skip_step"):
                        continue

                    video_pred = out["video_pred"]
                    video_target = out["video_target"]
                    video_loss_mask = out.get("video_loss_mask")
                    video_loss = _masked_mse(
                        video_pred, video_target, video_loss_mask,
                        weighting=weighting, dtype=trainer.dit_dtype
                    )
                    val_video_losses.append(video_loss.item())

                    audio_pred = out.get("audio_pred")
                    audio_target = out.get("audio_target")
                    if audio_pred is not None and audio_target is not None:
                        audio_loss_mask = out.get("audio_loss_mask")
                        audio_loss = _masked_mse(
                            audio_pred, audio_target, audio_loss_mask,
                            weighting=weighting, dtype=trainer.dit_dtype
                        )
                        val_audio_losses.append(audio_loss.item())
                        val_losses.append(video_loss.item() * args.video_loss_weight + audio_loss.item() * args.audio_loss_weight)
                    else:
                        val_losses.append(video_loss.item())
                else:
                    if isinstance(target, torch.Tensor):
                        model_pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
                    loss = torch.nn.functional.mse_loss(model_pred, target)
                    val_losses.append(loss.item())

                num_batches += 1

        # Restore original weights if EMA was applied
        if original_params is not None:
            ema_model.restore(accelerator.unwrap_model(transformer), original_params)

        transformer.train()

        # Compute average metrics
        val_metrics = {}
        if val_losses:
            val_metrics["val_loss"] = sum(val_losses) / len(val_losses)
        if val_video_losses:
            val_metrics["val_video_loss"] = sum(val_video_losses) / len(val_video_losses)
        if val_audio_losses:
            val_metrics["val_audio_loss"] = sum(val_audio_losses) / len(val_audio_losses)

        if val_metrics:
            accelerator.print(f"Validation metrics: {val_metrics}")
            accelerator.log(val_metrics, step=step)

        return val_metrics

    def should_validate(step: int, epoch: int, is_epoch_end: bool) -> bool:
        """Check if validation should run."""
        if val_dataloader is None:
            return False
        if args.validate_every_n_steps is not None and step > 0 and step % args.validate_every_n_steps == 0:
            return True
        if is_epoch_end and args.validate_every_n_epochs is not None and (epoch + 1) % args.validate_every_n_epochs == 0:
            return True
        return False

    # For --sample_at_first
    if should_sample_images(args, global_step, epoch=0):
        optimizer_eval_fn()
        trainer.sample_images(
            accelerator,
            args,
            0,
            global_step,
            accelerator.device,
            vae,
            transformer,
            sample_parameters,
        )
        optimizer_train_fn()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    metadata["ss_num_epochs"] = str(num_train_epochs)

    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {num_train_items}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
    accelerator.print(
        f"  total train batch size (with parallel & accumulation) / 総バッチサイズ: {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
    )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    optimizer_train_fn()

    for epoch in range(epoch_to_start, num_train_epochs):
        current_epoch.value = epoch + 1
        metadata["ss_epoch"] = str(epoch + 1)
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                latents = batch["latents"]
                if isinstance(latents, dict):
                    if "latents" not in latents:
                        raise ValueError("batch['latents'] is a dict but missing key 'latents'")
                    latents_tensor = latents["latents"]
                else:
                    latents_tensor = latents

                latents_tensor = trainer.scale_shift_latents(latents_tensor)
                noise = torch.randn_like(latents_tensor)

                noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                    args,
                    noise,
                    latents_tensor,
                    batch["timesteps"],
                    noise_scheduler,
                    accelerator.device,
                    trainer.dit_dtype,
                )

                weighting = compute_loss_weighting_for_sd3(
                    args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, trainer.dit_dtype
                )

                model_pred, target = trainer.call_dit(
                    args,
                    accelerator,
                    transformer,
                    latents_tensor,
                    batch,
                    noise,
                    noisy_model_input,
                    timesteps,
                    trainer.dit_dtype,
                )

                dict_output = isinstance(model_pred, dict)
                if dict_output:
                    out = model_pred
                    if out.get("_skip_step"):
                        logger.warning(
                            "Skipping step due to non-finite tensor (%s).",
                            out.get("skip_reason", "unknown"),
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    video_pred = out["video_pred"]
                    video_target = out["video_target"]
                    video_loss_mask = out.get("video_loss_mask")
                    video_loss = _masked_mse(
                        video_pred,
                        video_target,
                        video_loss_mask,
                        weighting=weighting,
                        dtype=trainer.dit_dtype,
                    )
                    video_weight = float(out.get("video_loss_weight", 1.0))
                    loss = video_loss * video_weight

                    audio_pred = out.get("audio_pred")
                    audio_target = out.get("audio_target")
                    audio_loss_mask = out.get("audio_loss_mask")
                    if audio_pred is not None and audio_target is not None:
                        audio_loss = _masked_mse(
                            audio_pred,
                            audio_target,
                            audio_loss_mask,
                            weighting=weighting,
                            dtype=trainer.dit_dtype,
                        )
                        audio_weight = float(out.get("audio_loss_weight", 1.0))
                        loss = loss + audio_loss * audio_weight
                else:
                    if isinstance(target, torch.Tensor):
                        model_pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
                    else:
                        model_pred = model_pred.to(dtype=trainer.dit_dtype)
                    loss = torch.nn.functional.mse_loss(model_pred, target, reduction="none")
                    if weighting is not None:
                        w = weighting
                        if isinstance(w, torch.Tensor) and w.dim() != loss.dim():
                            while w.dim() > loss.dim() and w.shape[-1] == 1:
                                w = w.squeeze(-1)
                        loss = loss * w
                    loss = loss.mean()

                accelerator.backward(loss)
                if not args.fused_backward_pass:
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                current_loss = loss.item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)

                # Update EMA weights
                if ema_model is not None:
                    ema_model.update(accelerator.unwrap_model(transformer))

                # Update progress bar with current metrics
                current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else args.learning_rate
                logs = {"loss": current_loss, "lr": current_lr}
                if dict_output:
                    if "video_pred" in out:
                        logs["v_loss"] = video_loss.item()
                    if audio_pred is not None:
                        logs["a_loss"] = audio_loss.item()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # Run validation at step intervals
                if should_validate(global_step, epoch, is_epoch_end=False):
                    optimizer_eval_fn()
                    run_validation(global_step, epoch)
                    optimizer_train_fn()

                if should_sample_images(args, global_step, epoch=None):
                    optimizer_eval_fn()
                    trainer.sample_images(
                        accelerator,
                        args,
                        None,
                        global_step,
                        accelerator.device,
                        vae,
                        transformer,
                        sample_parameters,
                    )
                    optimizer_train_fn()

                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    ckpt_name = train_utils.get_step_ckpt_name(args.output_name, global_step)
                    if args.save_ema_only and ema_model is not None:
                        save_ema_model(ckpt_name, global_step, epoch + 1)
                    else:
                        save_model(
                            ckpt_name,
                            accelerator.unwrap_model(transformer),
                            global_step,
                            epoch + 1,
                        )
                        if ema_model is not None:
                            save_ema_model(ckpt_name, global_step, epoch + 1)
                    remove_step_no = train_utils.get_remove_step_no(args, global_step)
                    if remove_step_no is not None:
                        remove_model(train_utils.get_step_ckpt_name(args.output_name, remove_step_no))
                        # Also remove old EMA checkpoint if exists
                        if ema_model is not None:
                            old_ema_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no).replace(".safetensors", "_ema.safetensors")
                            remove_model(old_ema_name)
                    if args.save_state:
                        train_utils.save_and_remove_state_stepwise(args, accelerator, global_step)
                        save_ema_state()

                if global_step >= args.max_train_steps:
                    break

        # Run validation at epoch end
        if should_validate(global_step, epoch, is_epoch_end=True):
            optimizer_eval_fn()
            run_validation(global_step, epoch + 1)
            optimizer_train_fn()

        if args.save_every_n_epochs is not None and (epoch + 1) % args.save_every_n_epochs == 0:
            ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch + 1)
            if args.save_ema_only and ema_model is not None:
                # Only save EMA weights
                save_ema_model(ckpt_name, global_step, epoch + 1)
            else:
                # Save training weights
                save_model(
                    ckpt_name,
                    accelerator.unwrap_model(transformer),
                    global_step,
                    epoch + 1,
                )
                # Also save EMA if enabled
                if ema_model is not None:
                    save_ema_model(ckpt_name, global_step, epoch + 1)
            remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
            if remove_epoch_no is not None:
                remove_model(train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch_no))
            if args.save_state:
                train_utils.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)
                save_ema_state()

        if should_sample_images(args, global_step, epoch=epoch + 1):
            optimizer_eval_fn()
            trainer.sample_images(
                accelerator,
                args,
                epoch + 1,
                global_step,
                accelerator.device,
                vae,
                transformer,
                sample_parameters,
            )
            optimizer_train_fn()

        if global_step >= args.max_train_steps:
            break

    metadata["ss_training_finished_at"] = str(time.time())
    optimizer_eval_fn()

    # Final validation
    if val_dataloader is not None:
        accelerator.print("\nRunning final validation...")
        run_validation(global_step, num_train_epochs)

    if accelerator.is_main_process and (args.save_state or args.save_state_on_train_end):
        train_utils.save_state_on_train_end(args, accelerator)
        save_ema_state()

    # Save final model
    final_ckpt_name = f"{args.output_name}.safetensors"
    if args.save_ema_only and ema_model is not None:
        # Only save EMA weights as final model
        save_ema_model(final_ckpt_name, global_step, num_train_epochs)
    else:
        # Save training weights
        save_model(
            final_ckpt_name,
            accelerator.unwrap_model(transformer),
            global_step,
            num_train_epochs,
            force_sync_upload=True,
            use_memory_efficient_saving=args.mem_eff_save,
        )
        # Also save EMA if enabled
        if ema_model is not None:
            save_ema_model(final_ckpt_name, global_step, num_train_epochs)

    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()
