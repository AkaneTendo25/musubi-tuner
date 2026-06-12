"""Shared NetworkTrainer training loop."""

import importlib
import json
import logging
import math
import os
import random
import sys
import time
from multiprocessing import Value

import accelerate
import toml
import torch
from accelerate.utils import set_seed
from tqdm import tqdm

import musubi_tuner.networks.lora as lora_module
from musubi_tuner.audio_loss_balance import (
    compute_ema_magnitude_audio_weight,
    compute_inverse_frequency_audio_weight,
    compute_uncertainty_weighted_loss,
    update_audio_presence_ema,
    update_loss_ema,
)
from musubi_tuner.cross_task_synergy import compute_cross_task_synergy_losses
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.audio_quota_sampler import (
    build_audio_sampler,
    split_concat_indices_by_audio,
    sync_dataset_group_epoch_without_loading,
)
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.modality_freezer import ModalityFreezer
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.ogm_ge import compute_ogm_ge_coefficients, maybe_add_ogm_ge_gradient_noise
from musubi_tuner.training.accelerator_setup import clean_memory_on_device, collator_class, prepare_accelerator
from musubi_tuner.training.losses import apply_loss_mask, per_element_loss as _per_element_loss
from musubi_tuner.training.metadata import (
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_MINIMUM_KEYS,
)
from musubi_tuner.training.outputs import unpack_dit_output as _unpack_dit_output
from musubi_tuner.training.runtime_utils import (
    log_cuda_memory_stats as _log_cuda_memory_stats,
    log_vram as _log_vram,
    offload_optimizer_state_during_validation,
    update_global_peak as _update_global_peak,
)
from musubi_tuner.training.sampling_prompts import should_sample_images
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3
from musubi_tuner.utils import huggingface_utils, model_utils, sai_model_spec, train_utils

logger = logging.getLogger("musubi_tuner.hv_train_network")


def train(self, args):
    if torch.cuda.is_available():
        if args.cuda_memory_fraction is not None:
            if not (0.0 < args.cuda_memory_fraction <= 1.0):
                raise ValueError("--cuda_memory_fraction must be in (0, 1]")
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
            logger.info("Set per-process CUDA memory fraction to %.4f", args.cuda_memory_fraction)
        if args.cuda_allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 on CUDA / CUDAでTF32を有効化しました")
        if args.cuda_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark / cuDNNベンチマークを有効化しました")

    # check required arguments
    if args.dataset_config is None and getattr(args, "dataset_manifest", None) is None:
        raise ValueError("dataset_config or dataset_manifest is required / dataset_configまたはdataset_manifestが必要です")
    if args.dit is None:
        raise ValueError("path to DiT model is required / DiTモデルのパスが必要です")
    assert not args.fp8_scaled or args.fp8_base, "fp8_scaled requires fp8_base / fp8_scaledはfp8_baseが必要です"

    if args.sage_attn:
        raise ValueError(
            "SageAttention doesn't support training currently. Please use `--sdpa` or `--xformers` etc. instead."
            " / SageAttentionは現在学習をサポートしていないようです。`--sdpa`や`--xformers`などの他のオプションを使ってください"
        )

    if args.disable_numpy_memmap:
        logger.info(
            "Disabling numpy memory mapping for model loading (for Wan, FramePack and Qwen-Image). This may lead to higher memory usage but can speed up loading in some cases."
            " / モデル読み込み時のnumpyメモリマッピングを無効にします（Wan、FramePack、Qwen-Imageでのみ有効）。これによりメモリ使用量が増える可能性がありますが、場合によっては読み込みが高速化されることがあります"
        )

    # check model specific arguments
    self.handle_model_specific_args(args)

    # show timesteps for debugging
    if args.show_timesteps:
        self.show_timesteps(args)
        return

    session_id = random.randint(0, 2**32)
    training_started_at = time.time()
    # setup_logging(args, reset=True)

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

    loss_diag_enabled = os.getenv("LTX2_LOSS_DIAG", "0") == "1"
    loss_diag_every = int(os.getenv("LTX2_LOSS_DIAG_EVERY", "10"))
    audio_loss_balance_mode = str(getattr(args, "audio_loss_balance_mode", "none") or "none").lower()
    audio_loss_balance_beta = float(getattr(args, "audio_loss_balance_beta", 0.01))
    audio_loss_balance_eps = float(getattr(args, "audio_loss_balance_eps", 0.05))
    audio_loss_balance_min = float(getattr(args, "audio_loss_balance_min", 0.05))
    audio_loss_balance_max = float(getattr(args, "audio_loss_balance_max", 4.0))
    audio_presence_ema = float(getattr(args, "audio_loss_balance_ema_init", 1.0))
    audio_presence_ema = min(max(audio_presence_ema, 1e-6), 1.0)
    audio_loss_balance_target_ratio = float(getattr(args, "audio_loss_balance_target_ratio", 0.33))
    audio_loss_balance_ema_decay = float(getattr(args, "audio_loss_balance_ema_decay", 0.99))
    audio_loss_ema = max(float(getattr(args, "audio_loss_balance_ema_init", 1.0)), 1e-6)
    video_loss_ema = max(float(getattr(args, "audio_loss_balance_ema_init", 1.0)), 1e-6)
    if audio_loss_balance_mode == "inv_freq":
        logger.info(
            "Audio inverse-frequency weighting enabled: beta=%.4f eps=%.4f min=%.4f max=%.4f ema_init=%.4f",
            audio_loss_balance_beta,
            audio_loss_balance_eps,
            audio_loss_balance_min,
            audio_loss_balance_max,
            audio_presence_ema,
        )
    elif audio_loss_balance_mode == "ema_mag":
        logger.info(
            "Audio EMA-magnitude balancing enabled: target_ratio=%.4f ema_decay=%.4f min=%.4f max=%.4f ema_init=%.4f",
            audio_loss_balance_target_ratio,
            audio_loss_balance_ema_decay,
            audio_loss_balance_min,
            audio_loss_balance_max,
            audio_loss_ema,
        )
    elif audio_loss_balance_mode == "ogm_ge":
        logger.info(
            "OGM-GE balancing enabled: alpha=%.4f noise_std=%.4f",
            float(getattr(args, "ogm_ge_alpha", 0.3)),
            float(getattr(args, "ogm_ge_noise_std", 0.0)),
        )

    # Uncertainty weighting: learnable log-variance scalars (Kendall et al., CVPR 2018)
    uncertainty_log_var_video = None
    uncertainty_log_var_audio = None
    if audio_loss_balance_mode == "uncertainty":
        uncertainty_log_var_video = torch.nn.Parameter(torch.zeros(1))
        uncertainty_log_var_audio = torch.nn.Parameter(torch.zeros(1))
        logger.info("Uncertainty weighting enabled: learnable log-variance scalars initialized to 0.0")

    # G2D-style modality freezing
    modality_freezer = None
    freeze_check_interval = int(getattr(args, "modality_freeze_check_interval", 0) or 0)
    if freeze_check_interval > 0:
        modality_freezer = ModalityFreezer(
            check_interval=freeze_check_interval,
            ratio_threshold=float(getattr(args, "modality_freeze_ratio_threshold", 0.5)),
            warmup_steps=int(getattr(args, "modality_freeze_warmup_steps", 100)),
            ema_decay=float(getattr(args, "modality_freeze_ema_decay", 0.99)),
        )
        logger.info(
            "Modality freezer enabled: check_interval=%d ratio_threshold=%.2f warmup=%d ema_decay=%.4f",
            modality_freezer.check_interval,
            modality_freezer.ratio_threshold,
            modality_freezer.warmup_steps,
            modality_freezer.ema_decay,
        )

    # Load dataset config
    if args.num_timestep_buckets is not None:
        logger.info(f"Using timestep bucketing. Number of buckets: {args.num_timestep_buckets}")
    self.num_timestep_buckets = args.num_timestep_buckets  # None or int, None makes all the behavior same as before
    if (
        bool(getattr(args, "ltx2_remote_stage", False))
        and str(getattr(args, "ltx2_remote_stage_codec", "none") or "none").lower().startswith("aq-")
        and str(getattr(args, "ltx2_remote_stage_aq_key_mode", "sample") or "sample").lower() != "off"
    ):
        # AQ remote-stage compression needs stable sample identity for delta caches.
        os.environ["LTX2_COLLECT_BATCH_ITEM_KEYS"] = "1"

    current_epoch = Value("i", 0)  # shared between processes

    validation_dataset_group = None
    validation_dataloader = None
    if getattr(args, "dataset_manifest", None) is not None:
        logger.info("Load dataset manifest from %s", args.dataset_manifest)
        dataset_manifest = config_utils.load_dataset_manifest(args.dataset_manifest)
        manifest_architecture = dataset_manifest.get("architecture")
        if manifest_architecture is not None and manifest_architecture != self.architecture:
            raise ValueError(
                f"dataset manifest architecture mismatch: expected '{self.architecture}', got '{manifest_architecture}'"
            )

        train_dataset_group = config_utils.generate_dataset_group_by_manifest(
            dataset_manifest,
            split="train",
            training=True,
            num_timestep_buckets=self.num_timestep_buckets,
            shared_epoch=current_epoch,
            reference_downscale=getattr(args, "reference_downscale", 1),
        )
        if train_dataset_group is None:
            raise ValueError("dataset manifest contains no training datasets")

        validation_dataset_group = config_utils.generate_dataset_group_by_manifest(
            dataset_manifest,
            split="validation",
            training=True,
            num_timestep_buckets=self.num_timestep_buckets,
            shared_epoch=current_epoch,
            reference_downscale=getattr(args, "reference_downscale", 1),
        )
    else:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=self.architecture)
        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.dataset_group,
            training=True,
            num_timestep_buckets=self.num_timestep_buckets,
            shared_epoch=current_epoch,
            reference_downscale=getattr(args, "reference_downscale", 1),
        )
        if user_config.get("validation_datasets"):
            logger.info("Load validation datasets from dataset config")
            validation_user_config = {
                "general": user_config.get("general", {}),
                "datasets": user_config.get("validation_datasets", []),
            }
            validation_blueprint = blueprint_generator.generate(validation_user_config, args, architecture=self.architecture)
            validation_dataset_group = config_utils.generate_dataset_group_by_blueprint(
                validation_blueprint.dataset_group,
                training=True,
                num_timestep_buckets=self.num_timestep_buckets,
                shared_epoch=current_epoch,
                reference_downscale=getattr(args, "reference_downscale", 1),
            )

    if train_dataset_group.num_train_items == 0:
        raise ValueError(
            "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
            " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
        )

    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, ds_for_collator)
    validation_collator = None
    if validation_dataset_group is not None:
        if validation_dataset_group.num_train_items == 0:
            raise ValueError(
                "No validation items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
            )
        ds_for_val_collator = validation_dataset_group if args.max_data_loader_n_workers == 0 else None
        validation_collator = collator_class(current_epoch, ds_for_val_collator)

    # prepare accelerator
    logger.info("preparing accelerator")
    accelerator = prepare_accelerator(args)
    if args.mixed_precision is None:
        args.mixed_precision = accelerator.mixed_precision
        logger.info(f"mixed precision set to {args.mixed_precision} / mixed precisionを{args.mixed_precision}に設定")
    is_main_process = accelerator.is_main_process
    model_parallel = self.is_model_parallel_enabled(args)
    if model_parallel:
        self.validate_model_parallel_setup(args, accelerator)

    # prepare dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # HunyuanVideo: bfloat16 or float16, Wan2.1: bfloat16
    dit_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)
    if getattr(args, "nf4_base", False):
        dit_weight_dtype = None  # NF4: quantized at load time, no dtype override needed
    elif args.fp8_base:
        dit_weight_dtype = None if args.fp8_scaled else torch.float8_e4m3fn
    else:
        dit_weight_dtype = dit_dtype
    logger.info(f"DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

    # GUI dashboard metrics writer. Enable either via legacy --gui or the
    # GUI process manager's environment flag.
    gui_metrics = None
    dashboard_metrics_enabled = getattr(args, "gui", False) or os.getenv("MUSUBI_DASHBOARD_METRICS") == "1"

    # get embedding for sampling images
    vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
    sample_parameters = None
    vae = None
    if (
        args.sample_prompts
        or getattr(args, "precache_sample_prompts", False)
        or getattr(args, "use_precached_sample_prompts", False)
    ):
        sample_prompt_path = args.sample_prompts or ""
        sample_parameters = self.process_sample_prompts(args, accelerator, sample_prompt_path)

        # Load VAE model for sampling images: VAE is loaded to cpu to save gpu memory
        vae = self.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
        vae.requires_grad_(False)
        vae.eval()

    # load DiT model
    blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
    self.blocks_to_swap = blocks_to_swap
    loading_device = "cpu" if blocks_to_swap > 0 or model_parallel else accelerator.device

    # Reset VRAM tracking for spike analysis
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info("[VRAM_TRACE] Reset peak memory stats before DiT loading")
    _log_vram("BEFORE DiT loading", logger)

    logger.info(f"Loading DiT model from {args.dit}")
    if args.sdpa:
        attn_mode = "torch"
    elif args.flash_attn:
        attn_mode = "flash"
    elif args.sage_attn:
        attn_mode = "sageattn"
    elif args.xformers:
        attn_mode = "xformers"
    elif args.flash3:
        attn_mode = "flash3"
    else:
        raise ValueError(
            "either --sdpa, --flash-attn, --flash3, --sage-attn or --xformers must be specified / --sdpa, --flash-attn, --flash3, --sage-attn, --xformersのいずれかを指定してください"
        )
    transformer = self.load_transformer(accelerator, args, args.dit, attn_mode, args.split_attn, loading_device, dit_weight_dtype)
    transformer.eval()
    transformer.requires_grad_(False)
    _log_vram("AFTER load_transformer (model on CPU)", logger)

    if model_parallel:
        self.enable_model_parallel_transformer(args, accelerator, transformer)
        _log_vram("AFTER enable_model_parallel_transformer", logger)

    if blocks_to_swap > 0:
        logger.info(
            f"enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}, use pinned memory: {args.use_pinned_memory_for_block_swap}"
        )
        transformer.enable_block_swap(
            blocks_to_swap,
            accelerator.device,
            supports_backward=True,
            use_pinned_memory=args.use_pinned_memory_for_block_swap,
            swap_norms=getattr(args, "swap_norms", False),
        )
        _log_vram("AFTER enable_block_swap (offloader created)", logger)
        transformer.move_to_device_except_swap_blocks(accelerator.device)
        _log_vram("AFTER move_to_device_except_swap_blocks #1 (18 blocks to GPU)", logger)

    # load network model for differential training
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    accelerator.print("import network module:", args.network_module)
    network_module: lora_module = importlib.import_module(args.network_module)  # actual module may be different

    if args.base_weights is not None:
        # if base_weights is specified, merge the weights to DiT model
        for i, weight_path in enumerate(args.base_weights):
            if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                multiplier = 1.0
            else:
                multiplier = args.base_weights_multiplier[i]

            accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

            weights_sd = self.load_network_weights(weight_path, network_module)
            module = network_module.create_arch_network_from_weights(multiplier, weights_sd, unet=transformer, for_inference=True)
            module.merge_to(None, transformer, weights_sd, weight_dtype, "cpu")

        accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

    # prepare network
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            if "=" not in net_arg:
                raise ValueError(f"Invalid --network_args entry (expected key=value): {net_arg}")
            key, value = net_arg.split("=", 1)
            net_kwargs[key] = value

    # Inject pre-computed LoftQ data if available (computed during model loading).
    # Use a separate dict so loftq_data (tensors) doesn't end up in net_kwargs
    # which gets JSON-serialized for metadata.
    _loftq_net_kwargs = dict(net_kwargs)
    from musubi_tuner.ltx2_train_network import load_ltx2_model

    _loftq_data = getattr(load_ltx2_model, "_loftq_data", None)
    if _loftq_data is not None:
        _loftq_net_kwargs["loftq_data"] = _loftq_data
        load_ltx2_model._loftq_data = None  # consume it

    if args.dim_from_weights:
        logger.info(f"Loading network from weights: {args.dim_from_weights}")
        weights_sd = self.load_network_weights(args.dim_from_weights, network_module)
        network, _ = network_module.create_arch_network_from_weights(1, weights_sd, unet=transformer)
    else:
        # We use the name create_arch_network for compatibility with LyCORIS
        if hasattr(network_module, "create_arch_network"):
            network = network_module.create_arch_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                None,
                transformer,
                neuron_dropout=args.network_dropout,
                **_loftq_net_kwargs,
            )
        else:
            # LyCORIS compatibility
            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                None,
                transformer,
                **_loftq_net_kwargs,
            )
    if network is None:
        return
    _log_vram("AFTER LoRA network creation", logger)

    if hasattr(network_module, "prepare_network"):
        network.prepare_network(args)

    # apply network to DiT
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    _log_vram("AFTER network.apply_to (LoRA applied to transformer)", logger)

    if args.network_weights is not None:
        # FIXME consider alpha of weights: this assumes that the alpha is not changed
        weights_sd = self.load_network_weights(args.network_weights, network_module)
        info = network.load_state_dict(weights_sd, False)
        accelerator.print(f"load network weights from {args.network_weights}: {info}")

    # LyCORIS + FP8 backend compatibility:
    # keep most base model in FP8, but upcast adapted base layers that LyCORIS touches.
    self._enable_lycoris_fp8_forward_compat(args, network)

    if args.gradient_checkpointing:
        blocks_to_ckpt = getattr(args, "blocks_to_checkpoint", -1)
        if getattr(args, "blockwise_checkpointing", False):
            transformer.enable_gradient_checkpointing(
                args.gradient_checkpointing_cpu_offload, weight_cpu_offloading=True, blocks_to_checkpoint=blocks_to_ckpt
            )
            if hasattr(transformer, "transformer_blocks"):
                total_blocks = len(transformer.transformer_blocks)
                if blocks_to_ckpt is None or int(blocks_to_ckpt) == -1:
                    ckpt_start = 0
                else:
                    ckpt_start = max(0, total_blocks - int(blocks_to_ckpt))
                logger.info(
                    "Blockwise checkpointing: blocks_to_checkpoint=%s (range %s..%s of %s).",
                    blocks_to_ckpt,
                    ckpt_start,
                    max(0, total_blocks - 1),
                    total_blocks,
                )
            if args.use_pinned_memory_for_block_swap and hasattr(transformer, "transformer_blocks"):
                # LTX-2 blockwise checkpointing uses per-block use_pinned_memory for CPU<->GPU transfers.
                for block in transformer.transformer_blocks:
                    if hasattr(block, "use_pinned_memory"):
                        block.use_pinned_memory = True
        else:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload, blocks_to_checkpoint=blocks_to_ckpt)
        try:
            network.enable_gradient_checkpointing(
                args.gradient_checkpointing_cpu_offload,
                weight_cpu_offloading=bool(getattr(args, "blockwise_checkpointing", False)),
                blocks_to_checkpoint=blocks_to_ckpt,
            )
        except TypeError:
            network.enable_gradient_checkpointing()

    # prepare optimizer, data loader etc.
    accelerator.print("prepare optimizer, data loader etc.")

    trainable_params, lr_descriptions = self._prepare_network_optimizer_params(args, network)

    optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(args, trainable_params)

    # Add uncertainty weighting log-variance params to optimizer
    if uncertainty_log_var_video is not None:
        uncertainty_lr = float(getattr(args, "uncertainty_lr", None) or args.learning_rate)
        uncertainty_log_var_video = uncertainty_log_var_video.to(device=accelerator.device)
        uncertainty_log_var_audio = uncertainty_log_var_audio.to(device=accelerator.device)
        optimizer.add_param_group(
            {
                "params": [uncertainty_log_var_video, uncertainty_log_var_audio],
                "lr": uncertainty_lr,
                "weight_decay": 0.0,
                "group_name": "uncertainty",
            }
        )
        logger.info("Added uncertainty log-variance params to optimizer (lr=%.2e)", uncertainty_lr)
        self._refresh_prodigy_plus_late_param_group_state(optimizer)

    def set_trainer_train_mode() -> None:
        optimizer_train_fn()
        self.training = True

    def set_trainer_eval_mode() -> None:
        optimizer_eval_fn()
        self.training = False

    # prepare dataloader

    # num workers for data loader: if 0, persistent_workers is not available
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

    train_audio_sampler, train_audio_sampler_mode, train_audio_sampler_stats = build_audio_sampler(
        dataset_group=train_dataset_group,
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        min_audio_batches_per_accum=int(getattr(args, "min_audio_batches_per_accum", 0) or 0),
        audio_batch_probability=getattr(args, "audio_batch_probability", None),
        seed=int(args.seed),
    )

    if train_audio_sampler_mode == "quota":
        logger.info(
            "Audio quota sampler enabled: min_audio_batches_per_accum=%d, accumulation_steps=%d, "
            "audio_batches=%d, non_audio_batches=%d",
            train_audio_sampler_stats["min_audio_batches_per_accum"],
            train_audio_sampler_stats["accumulation_steps"],
            train_audio_sampler_stats["audio_batches"],
            train_audio_sampler_stats["non_audio_batches"],
        )
    elif train_audio_sampler_mode == "probability":
        logger.info(
            "Audio probability sampler enabled: audio_batch_probability=%.3f, audio_batches=%d, non_audio_batches=%d",
            train_audio_sampler_stats["audio_batch_probability"],
            train_audio_sampler_stats["audio_batches"],
            train_audio_sampler_stats["non_audio_batches"],
        )

    if train_audio_sampler is None:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=False,
            sampler=train_audio_sampler,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )
    if validation_dataset_group is not None:
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset_group,
            batch_size=1,
            shuffle=False,
            collate_fn=validation_collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

    # calculate max_train_steps
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    # send max_train_steps to train_dataset_group
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # prepare training model. accelerator does some magic here

    # experimental feature: train the model with gradients in fp16/bf16
    # Stochastic rounding is now supported via copy_stochastic in optimizer_utils.py
    network_dtype = torch.float32
    if args.full_fp16:
        assert args.mixed_precision == "fp16", (
            "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        )
        accelerator.print("enable full fp16 training.")
        network_dtype = weight_dtype
        network.to(network_dtype)
    elif args.full_bf16:
        assert args.mixed_precision == "bf16", (
            "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        )
        accelerator.print("enable full bf16 training.")
        network_dtype = weight_dtype
        network.to(network_dtype)

    if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
        logger.info(f"casting model to {dit_weight_dtype}")
        transformer.to(dit_weight_dtype)
    if model_parallel:
        self.place_network_for_model_parallel(args, accelerator, transformer, network)
    _log_vram("BEFORE accelerator.prepare(transformer)", logger)

    if blocks_to_swap > 0:
        transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
        _log_vram("AFTER accelerator.prepare(transformer) with device_placement=[False]", logger)
        accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        _log_vram("AFTER move_to_device_except_swap_blocks #2", logger)
        accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        _log_vram("AFTER prepare_block_swap_before_forward", logger)
    elif model_parallel:
        transformer = accelerator.prepare(transformer, device_placement=[False])
        _log_vram("AFTER accelerator.prepare(transformer) with model-parallel device_placement=[False]", logger)
    else:
        transformer = accelerator.prepare(transformer)
        _log_vram("AFTER accelerator.prepare(transformer) without block swap", logger)

    if args.compile:
        transformer = self.compile_transformer(args, transformer)
        transformer.__dict__["_orig_mod"] = transformer  # for annoying accelerator checks

    # Set up pre-train hooks (CREPA, Self-Flow, etc.) BEFORE creating the LR scheduler.
    # This guarantees optimizer.param_groups is finalized before scheduler init.
    # Otherwise torch LR schedulers can fail with:
    #   ValueError: zip() argument 2 is shorter than argument 1
    self.pre_train_hook(args, accelerator, transformer=transformer, network=network)
    if hasattr(self, "_crepa") and self._crepa is not None:
        crepa_params = self._crepa.get_trainable_params()
        if crepa_params:
            optimizer.add_param_group({"params": crepa_params, "lr": args.learning_rate})
            accelerator.print(f"CREPA: added {sum(p.numel() for p in crepa_params):,} projector params to optimizer")
    if hasattr(self, "_self_flow") and self._self_flow is not None:
        self_flow_params = self._self_flow.get_trainable_params()
        if self_flow_params:
            projector_lr = getattr(getattr(self._self_flow, "config", None), "projector_lr", None)
            effective_projector_lr = float(projector_lr) if projector_lr is not None else float(args.learning_rate)
            optimizer.add_param_group({"params": self_flow_params, "lr": effective_projector_lr})
            accelerator.print(
                f"Self-Flow: added {sum(p.numel() for p in self_flow_params):,} projector params to optimizer "
                f"(lr={effective_projector_lr:g})"
            )
    self._refresh_prodigy_plus_late_param_group_state(optimizer)

    # prepare lr_scheduler (must happen after all optimizer param groups are added)
    lr_scheduler = self.get_lr_scheduler(args, optimizer, accelerator.num_processes)

    if model_parallel:
        network = accelerator.prepare(network, device_placement=[False])
        if validation_dataloader is not None:
            optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
                optimizer, train_dataloader, validation_dataloader, lr_scheduler
            )
        else:
            optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    elif validation_dataloader is not None:
        network, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
            network, optimizer, train_dataloader, validation_dataloader, lr_scheduler
        )
    else:
        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)
    training_model = network

    if args.gradient_checkpointing:
        transformer.train()
    else:
        transformer.eval()

    accelerator.unwrap_model(network).prepare_grad_etc(transformer)
    self._current_call_network = accelerator.unwrap_model(network)

    if args.full_fp16:
        # patch accelerator for fp16 training
        # def patch_accelerator_for_fp16_training(accelerator):
        org_unscale_grads = accelerator.scaler._unscale_grads_

        def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
            return org_unscale_grads(optimizer, inv_scale, found_inf, True)

        accelerator.scaler._unscale_grads_ = _unscale_grads_replacer

    # before resuming make hook for saving/loading to save/load the network weights only
    def save_model_hook(models, weights, output_dir):
        # pop weights of other models than network to save only network weights
        # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
        if accelerator.is_main_process:  # or args.deepspeed:
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                if len(weights) > i:
                    weights.pop(i)
            # print(f"save model hook: {len(weights)} weights will be saved")

            unwrapped_network = accelerator.unwrap_model(network)
            if hasattr(unwrapped_network, "build_adaptive_rank_runtime_state"):
                try:
                    adaptive_rank_runtime_state = unwrapped_network.build_adaptive_rank_runtime_state()
                    if adaptive_rank_runtime_state is not None:
                        runtime_state_path = unwrapped_network._adaptive_rank_runtime_state_path(output_dir)
                        with open(runtime_state_path, "w", encoding="utf-8") as f:
                            json.dump(adaptive_rank_runtime_state, f, indent=2, sort_keys=True)
                except Exception as e:
                    logger.warning(f"Failed to save adaptive rank runtime state: {e}")

            # Save CREPA projector into state directory so it matches the optimizer state
            if hasattr(self, "_crepa") and self._crepa is not None:
                try:
                    from safetensors.torch import save_file

                    proj_sd = self._crepa.state_dict()
                    if proj_sd:
                        proj_file = os.path.join(output_dir, "crepa_projector.safetensors")
                        save_file(proj_sd, proj_file)
                    train_sd = self._crepa.training_state_dict()
                    if train_sd:
                        state_file = os.path.join(output_dir, "crepa_state.safetensors")
                        save_file(train_sd, state_file)
                except Exception as e:
                    logger.warning(f"Failed to save CREPA state to state dir: {e}")
            if hasattr(self, "_self_flow") and self._self_flow is not None:
                try:
                    from safetensors.torch import save_file

                    proj_sd = self._self_flow.state_dict()
                    if proj_sd:
                        proj_file = os.path.join(output_dir, "self_flow_projector.safetensors")
                        save_file(proj_sd, proj_file)
                    teacher_sd = self._self_flow.teacher_state_dict()
                    if teacher_sd:
                        teacher_file = os.path.join(output_dir, "self_flow_teacher_ema.safetensors")
                        save_file(teacher_sd, teacher_file)
                except Exception as e:
                    logger.warning(f"Failed to save Self-Flow projector to state dir: {e}")

            # Save uncertainty weighting log-variance params
            if uncertainty_log_var_video is not None:
                try:
                    from safetensors.torch import save_file

                    save_file(
                        {"log_var_video": uncertainty_log_var_video.data, "log_var_audio": uncertainty_log_var_audio.data},
                        os.path.join(output_dir, "uncertainty_log_vars.safetensors"),
                    )
                except Exception as e:
                    logger.warning(f"Failed to save uncertainty log-variance params: {e}")

    def load_model_hook(models, input_dir):
        # remove models except network
        remove_indices = []
        for i, model in enumerate(models):
            if not isinstance(model, type(accelerator.unwrap_model(network))):
                remove_indices.append(i)
        for i in reversed(remove_indices):
            models.pop(i)
        # print(f"load model hook: {len(models)} models will be loaded")

        if models:
            try:
                runtime_state_path = None
                adaptive_rank_runtime_state = None
                for model in models:
                    unwrapped_model = accelerator.unwrap_model(model)
                    if not hasattr(unwrapped_model, "load_adaptive_rank_runtime_state"):
                        continue
                    runtime_state_path = unwrapped_model._adaptive_rank_runtime_state_path(input_dir)
                    if not os.path.exists(runtime_state_path):
                        continue
                    with open(runtime_state_path, "r", encoding="utf-8") as f:
                        adaptive_rank_runtime_state = json.load(f)
                    break

                if adaptive_rank_runtime_state is not None:
                    for model in models:
                        unwrapped_model = accelerator.unwrap_model(model)
                        if hasattr(unwrapped_model, "load_adaptive_rank_runtime_state"):
                            unwrapped_model.load_adaptive_rank_runtime_state(adaptive_rank_runtime_state)
                    logger.info("Loaded adaptive rank runtime state from %s", runtime_state_path)
            except Exception as e:
                logger.warning(f"Failed to load adaptive rank runtime state: {e}")

        if hasattr(self, "_crepa") and self._crepa is not None:
            try:
                from safetensors.torch import load_file

                proj_file = os.path.join(input_dir, "crepa_projector.safetensors")
                if os.path.exists(proj_file):
                    self._crepa.load_state_dict(load_file(proj_file))
                    logger.info("CREPA: loaded projector state from %s", proj_file)

                state_file = os.path.join(input_dir, "crepa_state.safetensors")
                if os.path.exists(state_file):
                    self._crepa.load_training_state_dict(load_file(state_file))
            except Exception as e:
                logger.warning(f"Failed to load CREPA state from checkpoint dir: {e}")

        if hasattr(self, "_self_flow") and self._self_flow is not None:
            try:
                from safetensors.torch import load_file

                proj_file = os.path.join(input_dir, "self_flow_projector.safetensors")
                if os.path.exists(proj_file):
                    self._self_flow.load_state_dict(load_file(proj_file))
                    logger.info("Self-Flow: loaded projector state from %s", proj_file)

                teacher_file = os.path.join(input_dir, "self_flow_teacher_ema.safetensors")
                if os.path.exists(teacher_file):
                    self._self_flow.load_teacher_state_dict(load_file(teacher_file))
                    logger.info("Self-Flow: loaded EMA teacher state from %s", teacher_file)
            except Exception as e:
                logger.warning(f"Failed to load Self-Flow state from checkpoint dir: {e}")

        # Load uncertainty weighting log-variance params
        if uncertainty_log_var_video is not None:
            try:
                from safetensors.torch import load_file

                lv_file = os.path.join(input_dir, "uncertainty_log_vars.safetensors")
                if os.path.exists(lv_file):
                    lv_sd = load_file(lv_file)
                    uncertainty_log_var_video.data.copy_(lv_sd["log_var_video"])
                    uncertainty_log_var_audio.data.copy_(lv_sd["log_var_audio"])
                    logger.info(
                        "Loaded uncertainty log-variance params: video=%.4f, audio=%.4f",
                        uncertainty_log_var_video.item(),
                        uncertainty_log_var_audio.item(),
                    )
            except Exception as e:
                logger.warning(f"Failed to load uncertainty log-variance params: {e}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # autoresume: find latest state in output_dir if --autoresume is set and --resume is not
    if getattr(args, "autoresume", False) and not args.resume:
        latest = self._find_latest_state_dir(args)
        if latest:
            logger.info(f"autoresume: found latest state directory: {latest}")
            args.resume = latest
            args._autoresume_selected = True
        else:
            logger.info("autoresume: no saved state found in output_dir, starting from scratch")
            args._autoresume_selected = False
    else:
        args._autoresume_selected = False

    # resume from local or huggingface — must be after num_update_steps_per_epoch is known

    # save param_groups before resume so we can restore them if --reset_optimizer_params
    inner_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    if getattr(args, "reset_optimizer_params", False):
        saved_param_groups = [{k: v for k, v in pg.items() if k != "params"} for pg in inner_optimizer.param_groups]

    # load resume metadata (for mid-epoch skip) before accelerator.load_state
    resume_metadata = None
    if args.resume:
        resume_metadata = train_utils.load_resume_metadata(args.resume)

    initial_global_step = self.resume_from_local_or_hf_if_specified(accelerator, args)
    if dashboard_metrics_enabled and accelerator.is_main_process:
        from musubi_tuner.gui_dashboard import create_metrics_writer

        gui_metrics = create_metrics_writer(args.output_dir, reset=initial_global_step == 0)

    # apply optimizer/scheduler resets after resume
    if initial_global_step > 0:
        if getattr(args, "reset_optimizer", False):
            inner_optimizer.state.clear()
            accelerator.print("reset optimizer state (cleared momentum/variance)")

        if getattr(args, "reset_optimizer_params", False):
            for pg, saved in zip(inner_optimizer.param_groups, saved_param_groups):
                for k, v in saved.items():
                    pg[k] = v
            accelerator.print("reset optimizer param groups to CLI values")

        if getattr(args, "reset_optimizer", False) or getattr(args, "reset_optimizer_params", False):
            # reset lr to base value so the new scheduler starts from the correct base
            # (scheduler __init__ uses current group['lr'], not initial_lr)
            for pg in inner_optimizer.param_groups:
                if "initial_lr" in pg:
                    pg["lr"] = pg["initial_lr"]
                    del pg["initial_lr"]
            new_inner_scheduler = self.get_lr_scheduler(args, inner_optimizer, accelerator.num_processes)
            # scheduler restarts from step 0 (fresh warmup/decay)
            # resume_metadata.json tracks the real global_step for checkpoint recovery
            # replace the inner scheduler while keeping the AcceleratedScheduler wrapper
            # (the wrapper gates stepping on sync_gradients for gradient accumulation)
            if hasattr(lr_scheduler, "scheduler"):
                lr_scheduler.scheduler = new_inner_scheduler
            else:
                lr_scheduler = new_inner_scheduler
            accelerator.print("recreated LR scheduler (restarting schedule from step 0)")

    # calculate epoch and mid-epoch skip
    steps_to_skip_in_epoch = 0
    if initial_global_step > 0 and resume_metadata is not None and resume_metadata.get("global_step", 0) > 0:
        saved_epoch = resume_metadata.get("epoch", 1)
        step_in_epoch = resume_metadata.get("step_in_epoch", 0)
        if step_in_epoch > 0:
            # mid-epoch checkpoint: resume in the same epoch, skip processed batches
            epoch_to_start = max(saved_epoch - 1, 0)
            if not getattr(args, "reset_dataloader", False):
                steps_to_skip_in_epoch = step_in_epoch
        else:
            # epoch-end checkpoint: epoch is complete, start from next
            epoch_to_start = saved_epoch
    else:
        epoch_to_start = initial_global_step // num_update_steps_per_epoch if initial_global_step > 0 else 0
    if gui_metrics is not None:
        gui_metrics.update_status(
            step=initial_global_step,
            max_steps=args.max_train_steps,
            epoch=epoch_to_start,
            max_epochs=num_train_epochs,
            status="starting",
        )

    # 学習する
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num train items / 学習画像、動画数: {train_dataset_group.num_train_items}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")
    if initial_global_step > 0:
        msg = f"  resuming from step {initial_global_step}, epoch {epoch_to_start + 1}/{num_train_epochs}"
        if steps_to_skip_in_epoch > 0:
            msg += f", skipping {steps_to_skip_in_epoch} batches in epoch"
        accelerator.print(msg)

    # TODO refactor metadata creation and move to util
    metadata = {
        "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
        "ss_training_started_at": training_started_at,  # unix timestamp
        "ss_output_name": args.output_name,
        "ss_learning_rate": args.learning_rate,
        "ss_num_train_items": train_dataset_group.num_train_items,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": num_train_epochs,
        "ss_gradient_checkpointing": args.gradient_checkpointing,
        "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
        "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
        "ss_max_train_steps": args.max_train_steps,
        "ss_lr_warmup_steps": args.lr_warmup_steps,
        "ss_lr_scheduler": args.lr_scheduler,
        SS_METADATA_KEY_BASE_MODEL_VERSION: self.architecture_full_name,
        # "ss_network_module": args.network_module,
        # "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
        # "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
        SS_METADATA_KEY_NETWORK_MODULE: args.network_module,
        SS_METADATA_KEY_NETWORK_DIM: args.network_dim,
        SS_METADATA_KEY_NETWORK_ALPHA: args.network_alpha,
        "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
        "ss_mixed_precision": args.mixed_precision,
        "ss_seed": args.seed,
        "ss_training_comment": args.training_comment,  # will not be updated after training
        # "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": args.max_grad_norm,
        "ss_fp8_base": bool(args.fp8_base),
        "ss_nf4_base": bool(getattr(args, "nf4_base", False)),
        "ss_loftq_init": bool(getattr(args, "loftq_init", False)),
        "ss_awq_calibration": bool(getattr(args, "awq_calibration", False)),
        # "ss_fp8_llm": bool(args.fp8_llm), # remove this because this is only for HuanyuanVideo TODO set architecure dependent metadata
        "ss_full_fp16": bool(args.full_fp16),
        "ss_full_bf16": bool(args.full_bf16),
        "ss_weighting_scheme": args.weighting_scheme,
        "ss_logit_mean": args.logit_mean,
        "ss_logit_std": args.logit_std,
        "ss_mode_scale": args.mode_scale,
        "ss_guidance_scale": args.guidance_scale,
        "ss_timestep_sampling": args.timestep_sampling,
        "ss_sigmoid_scale": args.sigmoid_scale,
        "ss_discrete_flow_shift": args.discrete_flow_shift,
        "ss_ltx_version": getattr(args, "ltx_version", None),
        "ss_shifted_logit_mode": getattr(args, "shifted_logit_mode", None),
        "ss_shifted_logit_eps": getattr(args, "shifted_logit_eps", None),
        "ss_shifted_logit_uniform_prob": getattr(args, "shifted_logit_uniform_prob", None),
        "ss_audio_lr": getattr(args, "audio_lr", None),
        "ss_lr_args": json.dumps(getattr(args, "lr_args", None)) if getattr(args, "lr_args", None) else None,
        "ss_audio_dim": getattr(args, "audio_dim", None),
        "ss_audio_alpha": getattr(args, "audio_alpha", None),
        "ss_video_caption_dropout_rate": getattr(args, "video_caption_dropout_rate", 0.0),
        "ss_audio_caption_dropout_rate": getattr(args, "audio_caption_dropout_rate", 0.0),
    }

    datasets_metadata = []
    # tag_frequency = {}  # merge tag frequency for metadata editor # TODO support tag frequency
    for dataset in train_dataset_group.datasets:
        dataset_metadata = dataset.get_metadata()
        datasets_metadata.append(dataset_metadata)

    metadata["ss_datasets"] = json.dumps(datasets_metadata)

    # add extra args
    if args.network_args:
        # metadata["ss_network_args"] = json.dumps(net_kwargs)
        metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(net_kwargs)

    # model name and hash
    # calculate hash takes time, so we omit it for now
    if args.dit is not None:
        # logger.info(f"calculate hash for DiT model: {args.dit}")
        logger.info(f"set DiT model name for metadata: {args.dit}")
        sd_model_name = args.dit
        if os.path.exists(sd_model_name):
            # metadata["ss_sd_model_hash"] = model_utils.model_hash(sd_model_name)
            # metadata["ss_new_sd_model_hash"] = model_utils.calculate_sha256(sd_model_name)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if args.vae is not None:
        # logger.info(f"calculate hash for VAE model: {args.vae}")
        logger.info(f"set VAE model name for metadata: {args.vae}")
        vae_name = args.vae
        if os.path.exists(vae_name):
            # metadata["ss_vae_hash"] = model_utils.model_hash(vae_name)
            # metadata["ss_new_vae_hash"] = model_utils.calculate_sha256(vae_name)
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name

    metadata = {k: str(v) for k, v in metadata.items()}

    # make minimum metadata for filtering
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
            "network_train" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_utils.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=initial_global_step,
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )

    global_step = initial_global_step
    noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
    timestep_tb_buffers = None
    timestep_tb_interval = max(1, int(getattr(args, "log_timestep_distribution_interval", 100) or 100))
    try:
        if self._should_log_timestep_distribution_to_tensorboard(args, accelerator):
            timestep_tb_buffers = {}
    except Exception as e:
        logger.warning(f"Disabling TensorBoard timestep distribution logging due to initialization failure: {e}")
        timestep_tb_buffers = None

    loss_recorder = train_utils.LossRecorder()
    if initial_global_step > 0 and getattr(self, "_resume_state_dir", None):
        _meta = train_utils.load_resume_metadata(self._resume_state_dir)
        if _meta and "loss_avg" in _meta:
            loss_recorder.prefill(_meta["loss_avg"], _meta.get("loss_count", 0))
            accelerator.print(f"  restored loss average: {_meta['loss_avg']:.4f} (from {_meta.get('loss_count', 0)} steps)")
    if train_audio_sampler is None:
        del train_dataset_group

    # function for saving/removing
    save_dtype = dit_dtype

    def save_model(ckpt_name: str, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
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
            None,
            self.architecture,
            time.time(),
            title,
            args.metadata_reso,
            args.metadata_author,
            args.metadata_description,
            args.metadata_license,
            args.metadata_tags,
            timesteps=md_timesteps,
            custom_arch=args.metadata_arch,
        )

        metadata_to_save.update(sai_metadata)

        # Architecture-specific metadata (e.g. v2v/IC-LoRA info for LTX-2)
        extra_md = self.get_checkpoint_metadata(args)
        if extra_md:
            metadata_to_save.update({k: str(v) for k, v in extra_md.items()})

        unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

        if bool(getattr(args, "ltx2_remote_stage", False)) and bool(getattr(args, "ltx2_remote_stage_trainable", False)):
            remote_checkpoint_dir = getattr(args, "ltx2_remote_stage_checkpoint_dir", None) or args.output_dir
            try:
                from musubi_tuner.ltx2_remote_stage import save_ltx2_remote_stage_state

                responses = save_ltx2_remote_stage_state(
                    transformer,
                    checkpoint_dir=remote_checkpoint_dir,
                    checkpoint_name=ckpt_name,
                )
                for response in responses:
                    path = response.get("path")
                    if path:
                        accelerator.print(f"saved remote stage checkpoint: {path}")
            except Exception as e:
                logger.warning("Failed to save remote LTX-2 stage checkpoint: %s", e)

        # Call post-save hook for architecture-specific processing
        self.post_save_checkpoint_hook(
            args,
            ckpt_file,
            ckpt_name,
            accelerator,
            force_sync_upload,
            unwrapped_nw=unwrapped_nw,
        )

        upload_original = (not getattr(args, "convert_to_comfy", True)) or getattr(args, "save_original_lora", True)
        if args.huggingface_repo_id is not None and upload_original:
            huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        if getattr(args, "save_checkpoint_metadata", False):
            from datetime import datetime

            _md = {
                "step": steps,
                "epoch": epoch_no,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                _md["loss"] = loss.detach().item()
            except Exception:
                pass
            if loss_recorder.loss_list:
                _md["loss_avg"] = loss_recorder.moving_average
            try:
                _md["lr"] = float(lr_scheduler.get_last_lr()[0])
            except Exception:
                pass
            if video_loss_value is not None:
                _md["loss_video"] = video_loss_value
            if audio_loss_value is not None:
                _md["loss_audio"] = audio_loss_value
            train_utils.save_checkpoint_metadata(ckpt_file, _md)

    def remove_model(old_ckpt_name):
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)
        if getattr(args, "convert_to_comfy", True):
            comfy_old_ckpt_file = old_ckpt_file.replace(".safetensors", ".comfy.safetensors")
            if os.path.exists(comfy_old_ckpt_file):
                accelerator.print(f"removing old Comfy checkpoint: {comfy_old_ckpt_file}")
                os.remove(comfy_old_ckpt_file)
        train_utils.remove_checkpoint_metadata(old_ckpt_file)

    def handle_dashboard_stop_request(global_step: int, epoch: int, step_in_epoch: int) -> bool:
        if not train_utils.dashboard_stop_requested():
            return False

        if train_utils.dashboard_stop_mode() == "force":
            accelerator.print("\nDashboard force stop requested; exiting without saving interrupt state.")
            if gui_metrics is not None and accelerator.is_main_process:
                gui_metrics.update_status(
                    step=global_step,
                    max_steps=args.max_train_steps,
                    epoch=epoch + 1,
                    max_epochs=num_train_epochs,
                    status="stopped",
                )
                gui_metrics.close()
            train_utils.clear_dashboard_stop_request()
            accelerator.end_training()
            return True

        if global_step <= 0:
            accelerator.print("\nDashboard stop requested before training steps completed; exiting without saving state.")
            if gui_metrics is not None and accelerator.is_main_process:
                gui_metrics.update_status(
                    step=global_step,
                    max_steps=args.max_train_steps,
                    epoch=epoch + 1,
                    max_epochs=num_train_epochs,
                    status="stopped",
                )
                gui_metrics.close()
            train_utils.clear_dashboard_stop_request()
            accelerator.end_training()
            return True

        accelerator.print("\nDashboard stop requested; saving interrupt state and exiting training.")
        set_trainer_eval_mode()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dir = train_utils.save_state_on_interrupt(
                args,
                accelerator,
                global_step=global_step,
                epoch=epoch + 1,
                step_in_epoch=step_in_epoch,
            )
            train_utils.update_resume_metadata(
                state_dir,
                {
                    "loss_avg": loss_recorder.moving_average,
                    "loss_count": len(loss_recorder.loss_list),
                    "interrupted": True,
                },
            )
            if gui_metrics is not None:
                gui_metrics.log_event("interrupt_state", global_step, path=state_dir)
                gui_metrics.update_status(
                    step=global_step,
                    max_steps=args.max_train_steps,
                    epoch=epoch + 1,
                    max_epochs=num_train_epochs,
                    status="stopped",
                )
                gui_metrics.close()
        train_utils.clear_dashboard_stop_request()
        accelerator.end_training()
        return True

    def run_validation(step: int, epoch_no: int | None = None) -> None:
        if validation_dataloader is None:
            return

        set_trainer_eval_mode()
        network.eval()
        transformer_was_training = transformer.training
        transformer.eval()

        total_loss = 0.0
        total_count = 0
        val_audio_presence_ema = float(audio_presence_ema)
        val_audio_loss_ema = float(audio_loss_ema)
        val_video_loss_ema = float(video_loss_ema)
        validation_self_flow_network = None
        validation_self_flow_prev_training = None
        if bool(getattr(args, "self_flow", False)):
            validation_self_flow_network = getattr(self, "_self_flow_network", None)
            if validation_self_flow_network is not None:
                validation_self_flow_prev_training = bool(getattr(validation_self_flow_network, "training", False))
                validation_self_flow_network.training = True

        try:
            with torch.no_grad():
                val_iter = validation_dataloader
                if accelerator.is_local_main_process:
                    val_iter = tqdm(
                        validation_dataloader,
                        desc="validation",
                        smoothing=0,
                        leave=False,
                    )
                for val_step, batch in enumerate(val_iter):
                    latents = batch["latents"]
                    if isinstance(latents, dict):
                        if "latents" not in latents:
                            raise ValueError("batch['latents'] is a dict but missing key 'latents'")
                        self.set_current_batch_latents_info(latents)
                        latents_tensor = latents["latents"]
                    else:
                        self.set_current_batch_latents_info(None)
                        latents_tensor = latents

                    latents_tensor = self.scale_shift_latents(latents_tensor)
                    noise = torch.randn_like(latents_tensor)

                    # HFATO: degrade latents before noise addition, keep clean for loss
                    _hfato_config = getattr(self, "_hfato_config", None)
                    if _hfato_config is not None and latents_tensor.dim() == 5:
                        import random as _hfato_rand

                        if _hfato_rand.random() < _hfato_config.probability:
                            from musubi_tuner.hfato import degrade_latents

                            batch["_hfato"] = {"clean_latents": latents_tensor}
                            latents_tensor = degrade_latents(
                                latents_tensor,
                                _hfato_config.scale_factor,
                                _hfato_config.interpolation,
                            )

                    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                        args,
                        noise,
                        latents_tensor,
                        batch["timesteps"],
                        noise_scheduler,
                        accelerator.device,
                        dit_dtype,
                    )

                    weighting = compute_loss_weighting_for_sd3(
                        args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype
                    )

                    self._current_train_global_step = global_step
                    model_pred, target = _unpack_dit_output(
                        self.call_dit(
                            args,
                            accelerator,
                            transformer,
                            latents_tensor,
                            batch,
                            noise,
                            noisy_model_input,
                            timesteps,
                            network_dtype,
                        )
                    )

                    dict_output = isinstance(model_pred, dict)
                    out = model_pred if dict_output else None
                    _loss_type = getattr(args, "loss_type", "mse")
                    _huber_delta = getattr(args, "huber_delta", 1.0)
                    video_pred = None
                    video_loss = None
                    audio_loss = None
                    video_weight = None
                    audio_weight = None

                    if dict_output:
                        if out.get("_skip_step"):
                            logger.warning(
                                "Skipping step due to non-finite tensor (%s).",
                                out.get("skip_reason", "unknown"),
                            )
                            optimizer.zero_grad(set_to_none=True)
                            continue

                        def _masked_loss(
                            pred: torch.Tensor,
                            tgt: torch.Tensor,
                            mask: torch.Tensor | None,
                            *,
                            tag: str | None = None,
                        ) -> torch.Tensor:
                            if isinstance(tgt, torch.Tensor):
                                pred = pred.to(device=tgt.device, dtype=network_dtype)
                            else:
                                pred = pred.to(dtype=network_dtype)
                            per_elem = _per_element_loss(pred, tgt, _loss_type, _huber_delta)
                            if weighting is not None:
                                w = weighting
                                if isinstance(w, torch.Tensor) and w.dim() != per_elem.dim():
                                    while w.dim() > per_elem.dim() and w.shape[-1] == 1:
                                        w = w.squeeze(-1)
                                per_elem = per_elem * w
                            if tag == "video":
                                per_elem, _ = self.modify_video_loss_per_element(args, per_elem, out, network_dtype)
                            elif tag == "audio":
                                per_elem, _ = self.modify_audio_loss_per_element(args, per_elem, out, network_dtype)
                            loss, _ = apply_loss_mask(per_elem, mask)
                            return loss

                        video_pred = out["video_pred"]
                        video_target = out["video_target"]
                        video_loss_mask = out.get("video_loss_mask")
                        _hfato_data = out.get("_hfato")
                        if _hfato_data is not None:
                            from musubi_tuner.hfato import hfato_x0_loss

                            video_loss = hfato_x0_loss(
                                video_pred.to(dtype=network_dtype),
                                _hfato_data["noisy"].to(device=video_pred.device, dtype=network_dtype),
                                _hfato_data["clean"].to(device=video_pred.device, dtype=network_dtype),
                                _hfato_data["sigma"].to(device=video_pred.device),
                                video_loss_mask,
                            )
                        else:
                            video_loss = _masked_loss(video_pred, video_target, video_loss_mask, tag="video")

                        audio_pred = out.get("audio_pred")
                        audio_target = out.get("audio_target")
                        audio_loss_mask = out.get("audio_loss_mask")
                        has_audio_loss = audio_pred is not None and audio_target is not None

                        if (
                            audio_loss_balance_mode == "uncertainty"
                            and has_audio_loss
                            and uncertainty_log_var_video is not None
                            and uncertainty_log_var_audio is not None
                        ):
                            audio_loss = _masked_loss(audio_pred, audio_target, audio_loss_mask, tag="audio")
                            loss = compute_uncertainty_weighted_loss(
                                video_loss,
                                audio_loss,
                                uncertainty_log_var_video,
                                uncertainty_log_var_audio,
                            )
                        elif audio_loss_balance_mode == "ogm_ge" and has_audio_loss:
                            audio_loss = _masked_loss(audio_pred, audio_target, audio_loss_mask, tag="audio")
                            ogm_ge_state = compute_ogm_ge_coefficients(
                                float(video_loss.detach().item()),
                                float(audio_loss.detach().item()),
                                alpha=float(getattr(args, "ogm_ge_alpha", 0.3)),
                            )
                            video_weight = float(ogm_ge_state.video_coeff)
                            audio_weight = float(ogm_ge_state.audio_coeff)
                            loss = video_loss * ogm_ge_state.video_coeff + audio_loss * ogm_ge_state.audio_coeff
                        else:
                            video_weight = float(out.get("video_loss_weight", 1.0))
                            loss = video_loss * video_weight
                            if audio_loss_balance_mode == "ema_mag":
                                video_loss_item = max(float(video_loss.detach().item()), 1e-12)
                                val_video_loss_ema = update_loss_ema(
                                    loss_ema=val_video_loss_ema,
                                    loss_value=video_loss_item,
                                    ema_decay=audio_loss_balance_ema_decay,
                                )
                            if audio_loss_balance_mode == "inv_freq":
                                val_audio_presence_ema = update_audio_presence_ema(
                                    audio_presence_ema=val_audio_presence_ema,
                                    balance_beta=audio_loss_balance_beta,
                                    has_audio_loss=has_audio_loss,
                                )
                            if has_audio_loss:
                                audio_loss = _masked_loss(audio_pred, audio_target, audio_loss_mask, tag="audio")
                                audio_weight = float(out.get("audio_loss_weight", 1.0))
                                if audio_loss_balance_mode == "inv_freq":
                                    audio_weight = compute_inverse_frequency_audio_weight(
                                        base_audio_weight=audio_weight,
                                        audio_presence_ema=val_audio_presence_ema,
                                        balance_eps=audio_loss_balance_eps,
                                        balance_min=audio_loss_balance_min,
                                        balance_max=audio_loss_balance_max,
                                    )
                                elif audio_loss_balance_mode == "ema_mag":
                                    audio_loss_item = max(float(audio_loss.detach().item()), 1e-12)
                                    val_audio_loss_ema = update_loss_ema(
                                        loss_ema=val_audio_loss_ema,
                                        loss_value=audio_loss_item,
                                        ema_decay=audio_loss_balance_ema_decay,
                                    )
                                    audio_weight = compute_ema_magnitude_audio_weight(
                                        base_audio_weight=audio_weight,
                                        audio_loss_ema=val_audio_loss_ema,
                                        video_loss_ema=val_video_loss_ema,
                                        target_audio_ratio=audio_loss_balance_target_ratio,
                                        balance_min=audio_loss_balance_min,
                                        balance_max=audio_loss_balance_max,
                                    )
                                loss = loss + audio_loss * audio_weight

                        video_extra_loss, _ = self.compute_video_extra_loss(args, out, network_dtype)
                        if video_extra_loss is not None:
                            loss = loss + video_extra_loss
                    else:
                        if isinstance(target, torch.Tensor):
                            model_pred = model_pred.to(device=target.device, dtype=network_dtype)
                        else:
                            model_pred = model_pred.to(dtype=network_dtype)
                        loss = _per_element_loss(model_pred, target, _loss_type, _huber_delta)
                        if weighting is not None:
                            loss = loss * weighting
                        loss = loss.mean()

                    if dict_output and video_pred is not None:
                        _prior_div = self.compute_prior_divergence_addition(
                            args, accelerator, transformer, network, video_pred, network_dtype
                        )
                        if _prior_div is not None:
                            loss = loss + _prior_div

                    if hasattr(self, "_crepa") and self._crepa is not None:
                        self._crepa.on_step(step)
                        try:
                            num_latent_frames = int(latents_tensor.shape[2]) if latents_tensor.dim() >= 3 else 0
                            conditions = batch.get("conditions", {})
                            dino_features = conditions.get("dino_features") if isinstance(conditions, dict) else None
                            crepa_loss = self._crepa.compute_loss(
                                num_latent_frames,
                                dino_features=dino_features,
                                update_cutoff=False,
                            )
                            if crepa_loss is not None:
                                loss = loss + crepa_loss
                        finally:
                            self._crepa.cleanup_step()

                    if hasattr(self, "compute_self_flow_addition"):
                        if hasattr(self, "_self_flow") and self._self_flow is not None:
                            self._self_flow.on_step(step)
                        try:
                            self_flow_loss, _ = self.compute_self_flow_addition(
                                args,
                                accelerator,
                                transformer,
                                network,
                                network_dtype,
                            )
                        except Exception as e:
                            logger.warning("Self-Flow loss computation failed during validation: %s", e)

                    if dict_output and out is not None:
                        cts_data = out.get("_cts")
                        if cts_data is not None:
                            try:
                                cts_loss, _ = compute_cross_task_synergy_losses(
                                    transformer=transformer,
                                    accelerator=accelerator,
                                    noisy_video=cts_data["noisy_video"],
                                    clean_video=cts_data["clean_video"],
                                    video_target=out["video_target"],
                                    video_timesteps=cts_data["video_timesteps"],
                                    video_loss_mask=out.get("video_loss_mask"),
                                    noisy_audio=cts_data["noisy_audio"],
                                    clean_audio=cts_data["clean_audio"],
                                    audio_target=out.get("audio_target"),
                                    audio_timesteps=cts_data["audio_timesteps"],
                                    audio_loss_mask=out.get("audio_loss_mask"),
                                    text_embeds=cts_data["text_embeds"],
                                    text_mask=cts_data["text_mask"],
                                    frame_rate=cts_data["frame_rate"],
                                    transformer_options=cts_data["transformer_options"],
                                    lambda_video_driven=cts_data["lambda_video_driven"],
                                    lambda_audio_driven=cts_data["lambda_audio_driven"],
                                )
                                if cts_loss is not None:
                                    loss = loss + cts_loss
                            except Exception as e:
                                logger.warning("Cross-Task Synergy loss failed during validation: %s", e)

                    unwrapped_network = accelerator.unwrap_model(network)
                    if hasattr(unwrapped_network, "compute_adaptive_rank_loss"):
                        adaptive_rank_loss = unwrapped_network.compute_adaptive_rank_loss()
                        if adaptive_rank_loss is not None:
                            loss = loss + adaptive_rank_loss

                    try:
                        validation_extra_loss, _ = self.compute_validation_extra_loss(
                            args,
                            accelerator,
                            transformer,
                            network,
                            batch,
                            step,
                            network_dtype,
                        )
                        if validation_extra_loss is not None:
                            loss = loss + validation_extra_loss
                    finally:
                        if hasattr(self, "_last_dit_inputs"):
                            self._last_dit_inputs = None

                    if loss_diag_enabled and global_step % max(loss_diag_every, 1) == 0:
                        weight_stats = ""
                        if isinstance(weighting, torch.Tensor):
                            weight_stats = (
                                f" weighting(mean={weighting.float().mean().item():.6f}"
                                f" min={weighting.float().min().item():.6f}"
                                f" max={weighting.float().max().item():.6f})"
                            )
                        logger.info(
                            "LOSS_DIAG step=%s video_loss=%s video_weight=%s audio_loss=%s audio_weight=%s total=%s%s",
                            global_step,
                            f"{video_loss.item():.6f}" if isinstance(video_loss, torch.Tensor) else "n/a",
                            f"{video_weight:.3f}" if isinstance(video_weight, float) else "n/a",
                            f"{audio_loss.item():.6f}" if isinstance(audio_loss, torch.Tensor) else "n/a",
                            f"{audio_weight:.3f}" if isinstance(audio_weight, float) else "n/a",
                            f"{loss.item():.6f}" if isinstance(loss, torch.Tensor) else str(loss),
                            weight_stats,
                        )

                    total_loss += loss.detach().item()
                    total_count += 1
        finally:
            if validation_self_flow_network is not None and validation_self_flow_prev_training is not None:
                validation_self_flow_network.training = validation_self_flow_prev_training

        loss_stats = torch.tensor([total_loss, total_count], device=accelerator.device)
        if accelerator.num_processes > 1:
            loss_stats = accelerator.gather(loss_stats)
            total_loss = float(loss_stats[:, 0].sum().item())
            total_count = int(loss_stats[:, 1].sum().item())

        if accelerator.is_main_process:
            avg_loss = total_loss / max(total_count, 1)
            if len(accelerator.trackers) > 0:
                accelerator.log({"val_loss": avg_loss}, step=step)
            if gui_metrics is not None:
                gui_metrics.log_event("validation", step, val_loss=avg_loss, epoch=epoch_no)
            log_msg = f"validation loss: {avg_loss:.6f}"
            if epoch_no is not None:
                log_msg += f" (epoch {epoch_no})"
            logger.info(log_msg)

        if transformer_was_training:
            transformer.train()
        else:
            transformer.eval()
        network.train()
        set_trainer_train_mode()

    # For --sample_at_first (skip on resume — samples were already generated)
    if global_step == 0 and should_sample_images(args, global_step, epoch=0):
        set_trainer_eval_mode()
        with offload_optimizer_state_during_validation(
            optimizer,
            accelerator,
            bool(getattr(args, "offload_optimizer_during_validation", False)),
            logger=logger,
        ):
            self.sample_images(accelerator, args, 0, global_step, vae, transformer, sample_parameters, dit_dtype)
        set_trainer_train_mode()
    if len(accelerator.trackers) > 0:
        # log empty object to commit the sample images to wandb
        accelerator.log({}, step=0)

    # training loop

    # log device and dtype for each model
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    first_param = next(iter(unwrapped_transformer.parameters()), None)
    logger.info(
        f"DiT dtype: {first_param.dtype if first_param is not None else None}, device: {first_param.device if first_param is not None else accelerator.device}"
    )

    clean_memory_on_device(accelerator.device)

    # pre_train_hook and CREPA param group already called before resume (above)

    set_trainer_train_mode()  # Set training mode

    for epoch in range(epoch_to_start, num_train_epochs):
        accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
        current_epoch.value = epoch + 1
        if train_audio_sampler is not None:
            sync_dataset_group_epoch_without_loading(train_dataset_group, epoch + 1, logger=logger)
            audio_indices, non_audio_indices = split_concat_indices_by_audio(train_dataset_group)
            if len(audio_indices) == 0:
                raise ValueError(
                    f"No audio-bearing batches available at epoch {epoch + 1} while "
                    f"{'--min_audio_batches_per_accum' if train_audio_sampler_mode == 'quota' else '--audio_batch_probability'} is enabled."
                )
            if hasattr(train_audio_sampler, "update_groups"):
                train_audio_sampler.update_groups(audio_indices, non_audio_indices)
            if hasattr(train_audio_sampler, "set_epoch"):
                train_audio_sampler.set_epoch(epoch)

        metadata["ss_epoch"] = str(epoch + 1)

        accelerator.unwrap_model(network).on_epoch_start(transformer)
        _prev_step_end_time = time.perf_counter()

        for step, batch in enumerate(train_dataloader):
            if handle_dashboard_stop_request(global_step, epoch, step):
                return

            # mid-epoch resume: skip batches already processed before checkpoint
            if steps_to_skip_in_epoch > 0:
                steps_to_skip_in_epoch -= 1
                continue

            _step_start_time = time.perf_counter()
            _data_wait_time = max(0.0, _step_start_time - _prev_step_end_time)
            # VRAM spike tracing for first iteration
            _is_first_step = epoch == epoch_to_start and step == 0
            if _is_first_step:
                _log_vram("FIRST_ITER: before batch processing", logger)
            # torch.compiler.cudagraph_mark_step_begin() # for cudagraphs
            if (
                args.log_cuda_memory_every_n_steps is not None
                and args.log_cuda_memory_every_n_steps > 0
                and accelerator.device.type == "cuda"
                and step % args.gradient_accumulation_steps == 0
                and global_step % args.log_cuda_memory_every_n_steps == 0
            ):
                _update_global_peak()  # Capture peak before reset for global tracking
                torch.cuda.reset_peak_memory_stats()

            latents = batch["latents"]
            if isinstance(latents, dict):
                if "latents" not in latents:
                    raise ValueError("batch['latents'] is a dict but missing key 'latents'")
                self.set_current_batch_latents_info(latents)
                latents_tensor = latents["latents"]
            else:
                self.set_current_batch_latents_info(None)
                latents_tensor = latents
            latents_shape = tuple(latents_tensor.shape)

            with accelerator.accumulate(training_model):
                accelerator.unwrap_model(network).on_step_start(
                    global_step=global_step,
                    max_train_steps=args.max_train_steps,
                )

                latents_tensor = self.scale_shift_latents(latents_tensor)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents_tensor)

                # HFATO: degrade latents before noise addition, keep clean for loss
                _hfato_config = getattr(self, "_hfato_config", None)
                if _hfato_config is not None and latents_tensor.dim() == 5:
                    import random as _hfato_rand

                    if _hfato_rand.random() < _hfato_config.probability:
                        from musubi_tuner.hfato import degrade_latents

                        batch["_hfato"] = {"clean_latents": latents_tensor}
                        latents_tensor = degrade_latents(
                            latents_tensor,
                            _hfato_config.scale_factor,
                            _hfato_config.interpolation,
                        )

                # calculate model input and timesteps
                noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                    args,
                    noise,
                    latents_tensor,
                    batch["timesteps"],
                    noise_scheduler,
                    accelerator.device,
                    dit_dtype,
                )

                weighting = compute_loss_weighting_for_sd3(
                    args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype
                )

                if _is_first_step:
                    _log_vram("FIRST_ITER: BEFORE call_dit (forward pass)", logger)
                self._current_train_global_step = global_step
                model_pred, target = _unpack_dit_output(
                    self.call_dit(
                        args,
                        accelerator,
                        transformer,
                        latents_tensor,
                        batch,
                        noise,
                        noisy_model_input,
                        timesteps,
                        network_dtype,
                    )
                )
                if timestep_tb_buffers is not None:
                    payload = self._get_timestep_distribution_logging_payload(args, timesteps)
                    for name, ts_values in payload.items():
                        if ts_values is None:
                            continue
                        self._accumulate_timestep_distribution(timestep_tb_buffers, name, ts_values, accelerator)
                if _is_first_step:
                    _log_vram("FIRST_ITER: AFTER call_dit (forward pass)", logger)
                dict_output = isinstance(model_pred, dict)
                video_loss_value = None  # For tracking in wandb/tensorboard
                audio_loss_value = None  # For tracking in wandb/tensorboard
                audio_weight_effective_value = None
                audio_presence_ema_value = None
                audio_loss_ema_value = None
                video_loss_ema_value = None
                ogm_ge_state = None
                grad_norm_total_value = None
                grad_norm_video_value = None  # Per-modality gradient norms
                grad_norm_audio_value = None
                audio_diagnostics = {}  # Per-batch audio quality diagnostics (negligible cost)
                mask_metrics: dict[str, float] = {}
                latent_temporal_metrics: dict[str, float] = {}
                _loss_type = getattr(args, "loss_type", "mse")
                _huber_delta = getattr(args, "huber_delta", 1.0)

                if dict_output:
                    out = model_pred

                    def _masked_loss(
                        pred: torch.Tensor,
                        tgt: torch.Tensor,
                        mask: torch.Tensor | None,
                        *,
                        tag: str | None = None,
                    ) -> torch.Tensor:
                        if isinstance(tgt, torch.Tensor):
                            pred = pred.to(device=tgt.device, dtype=network_dtype)
                        else:
                            pred = pred.to(dtype=network_dtype)
                        per_elem = _per_element_loss(pred, tgt, _loss_type, _huber_delta)
                        if weighting is not None:
                            w = weighting
                            if isinstance(w, torch.Tensor) and w.dim() != per_elem.dim():
                                while w.dim() > per_elem.dim() and w.shape[-1] == 1:
                                    w = w.squeeze(-1)
                            per_elem = per_elem * w
                        if tag == "video":
                            per_elem, modifier_metrics = self.modify_video_loss_per_element(args, per_elem, out, network_dtype)
                            for k, v in modifier_metrics.items():
                                mask_metrics[k] = v
                        elif tag == "audio":
                            per_elem, modifier_metrics = self.modify_audio_loss_per_element(args, per_elem, out, network_dtype)
                            for k, v in modifier_metrics.items():
                                mask_metrics[k] = v
                        loss, metrics = apply_loss_mask(per_elem, mask)
                        if tag is not None and metrics:
                            for k, v in metrics.items():
                                mask_metrics[f"{tag}_{k}"] = v
                        return loss

                    video_pred = out["video_pred"]
                    video_target = out["video_target"]
                    video_target_for_loss = self.apply_differential_guidance_target(args, video_pred, video_target)
                    video_loss_mask = out.get("video_loss_mask")
                    _hfato_data = out.get("_hfato")
                    if _hfato_data is not None:
                        from musubi_tuner.hfato import hfato_x0_loss

                        video_loss = hfato_x0_loss(
                            video_pred.to(dtype=network_dtype),
                            _hfato_data["noisy"].to(device=video_pred.device, dtype=network_dtype),
                            _hfato_data["clean"].to(device=video_pred.device, dtype=network_dtype),
                            _hfato_data["sigma"].to(device=video_pred.device),
                            video_loss_mask,
                        )
                    else:
                        video_loss = _masked_loss(video_pred, video_target_for_loss, video_loss_mask, tag="video")

                    audio_pred = out.get("audio_pred")
                    audio_target = out.get("audio_target")
                    audio_loss_mask = out.get("audio_loss_mask")
                    has_audio_loss = audio_pred is not None and audio_target is not None

                    if audio_loss_balance_mode == "uncertainty" and has_audio_loss:
                        # Uncertainty weighting: learnable log-variance scalars replace manual weights
                        video_loss_value = video_loss.detach().item()
                        audio_loss_raw = _masked_loss(audio_pred, audio_target, audio_loss_mask, tag="audio")
                        audio_loss_value = audio_loss_raw.detach().item()
                        loss = compute_uncertainty_weighted_loss(
                            video_loss,
                            audio_loss_raw,
                            uncertainty_log_var_video,
                            uncertainty_log_var_audio,
                        )
                    elif audio_loss_balance_mode == "ogm_ge" and has_audio_loss:
                        audio_loss = _masked_loss(audio_pred, audio_target, audio_loss_mask, tag="audio")
                        video_loss_value = video_loss.detach().item()
                        audio_loss_value = audio_loss.detach().item()
                        ogm_ge_state = compute_ogm_ge_coefficients(
                            video_loss_value,
                            audio_loss_value,
                            alpha=float(getattr(args, "ogm_ge_alpha", 0.3)),
                        )
                        loss = video_loss * ogm_ge_state.video_coeff + audio_loss * ogm_ge_state.audio_coeff
                        audio_weight_effective_value = ogm_ge_state.audio_coeff
                    else:
                        # Standard weighting path (none / inv_freq / ema_mag)
                        video_weight = float(out.get("video_loss_weight", 1.0))
                        loss = video_loss * video_weight
                        if audio_loss_balance_mode == "ema_mag":
                            video_loss_item = max(float(video_loss.detach().item()), 1e-12)
                            video_loss_ema = update_loss_ema(
                                loss_ema=video_loss_ema,
                                loss_value=video_loss_item,
                                ema_decay=audio_loss_balance_ema_decay,
                            )
                            video_loss_ema_value = video_loss_ema
                        # Capture video loss for logging (only if weight > 0)
                        if video_weight > 0:
                            video_loss_value = video_loss.detach().item()
                        if audio_loss_balance_mode == "inv_freq":
                            audio_presence_ema = update_audio_presence_ema(
                                audio_presence_ema=audio_presence_ema,
                                balance_beta=audio_loss_balance_beta,
                                has_audio_loss=has_audio_loss,
                            )
                            audio_presence_ema_value = audio_presence_ema
                        if has_audio_loss:
                            audio_loss = _masked_loss(audio_pred, audio_target, audio_loss_mask, tag="audio")
                            audio_weight = float(out.get("audio_loss_weight", 1.0))
                            if audio_loss_balance_mode == "inv_freq":
                                audio_weight = compute_inverse_frequency_audio_weight(
                                    base_audio_weight=audio_weight,
                                    audio_presence_ema=audio_presence_ema,
                                    balance_eps=audio_loss_balance_eps,
                                    balance_min=audio_loss_balance_min,
                                    balance_max=audio_loss_balance_max,
                                )
                            elif audio_loss_balance_mode == "ema_mag":
                                audio_loss_item = max(float(audio_loss.detach().item()), 1e-12)
                                audio_loss_ema = update_loss_ema(
                                    loss_ema=audio_loss_ema,
                                    loss_value=audio_loss_item,
                                    ema_decay=audio_loss_balance_ema_decay,
                                )
                                audio_loss_ema_value = audio_loss_ema
                                audio_weight = compute_ema_magnitude_audio_weight(
                                    base_audio_weight=audio_weight,
                                    audio_loss_ema=audio_loss_ema,
                                    video_loss_ema=video_loss_ema,
                                    target_audio_ratio=audio_loss_balance_target_ratio,
                                    balance_min=audio_loss_balance_min,
                                    balance_max=audio_loss_balance_max,
                                )
                            audio_weight_effective_value = audio_weight
                            loss = loss + audio_loss * audio_weight
                            audio_loss_value = audio_loss.detach().item() if audio_weight > 0 else None

                    latent_temporal_extra_loss, latent_temporal_metrics = self.compute_video_extra_loss(args, out, network_dtype)
                    if latent_temporal_extra_loss is not None:
                        loss = loss + latent_temporal_extra_loss

                    # --- Audio diagnostics (per-batch, negligible cost) ---
                    if has_audio_loss and audio_pred is not None and audio_target is not None:
                        with torch.no_grad():
                            ap = audio_pred.detach().float()
                            at = audio_target.detach().float()

                            # Task 2: Audio latent statistics — detect collapse/explosion
                            audio_diagnostics["audio_latent/pred_mean"] = ap.mean().item()
                            audio_diagnostics["audio_latent/pred_std"] = ap.std().item()
                            audio_diagnostics["audio_latent/pred_absmax"] = ap.abs().max().item()

                            # Task 3: Latent-space SNR (dB)
                            target_power = (at**2).mean()
                            error_power = ((at - ap) ** 2).mean()
                            if error_power > 0:
                                audio_diagnostics["audio_latent/snr_db"] = (10.0 * torch.log10(target_power / error_power)).item()

                            # Task 1: Timestep-stratified audio loss
                            audio_sigma = out.get("audio_sigma")
                            if audio_sigma is not None:
                                sigma = audio_sigma.detach().float()
                                # Per-sample MSE (reduce over C, T, F)
                                per_sample = ((ap - at) ** 2).mean(dim=list(range(1, ap.dim())))
                                high_mask = sigma > 0.5
                                mid_mask = (sigma >= 0.1) & (sigma <= 0.5)
                                low_mask = sigma < 0.1
                                if high_mask.any():
                                    audio_diagnostics["loss_a/sigma_high"] = per_sample[high_mask].mean().item()
                                if mid_mask.any():
                                    audio_diagnostics["loss_a/sigma_mid"] = per_sample[mid_mask].mean().item()
                                if low_mask.any():
                                    audio_diagnostics["loss_a/sigma_low"] = per_sample[low_mask].mean().item()

                    # Task 4: Audio/video loss ratio
                    if video_loss_value is not None and audio_loss_value is not None and video_loss_value > 0:
                        audio_diagnostics["loss/audio_video_ratio"] = audio_loss_value / video_loss_value

                    # Extended audio metrics (standalone module, no-op when off)
                    if getattr(self, "_audio_metrics", None) is not None:
                        self._audio_metrics.on_step(global_step)
                        audio_diagnostics.update(
                            self._audio_metrics.compute_latent_metrics(
                                ap,
                                at,
                                video_pred=out.get("video_pred"),
                                video_target=out.get("video_target"),
                            )
                        )
                        if self._audio_metrics.should_compute_mel(global_step):
                            _mel_decoder = getattr(self, "_get_audio_decoder_for_metrics", lambda: None)()
                            if _mel_decoder is not None:
                                audio_diagnostics.update(
                                    self._audio_metrics.compute_mel_metrics(
                                        out.get("audio_pred"),
                                        out.get("audio_target"),
                                        _mel_decoder,
                                    )
                                )

                else:
                    if isinstance(target, torch.Tensor):
                        model_pred = model_pred.to(device=target.device, dtype=network_dtype)
                        target = self.apply_differential_guidance_target(args, model_pred, target)
                    else:
                        model_pred = model_pred.to(dtype=network_dtype)
                    loss = _per_element_loss(model_pred, target, _loss_type, _huber_delta)

                if not dict_output and weighting is not None:
                    loss = loss * weighting
                # loss = loss.mean([1, 2, 3])
                # # min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation etc.
                # loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

                if not dict_output:
                    loss = loss.mean()  # mean loss over all elements in batch

                _prior_div_value = None
                if dict_output:
                    _prior_div = self.compute_prior_divergence_addition(
                        args, accelerator, transformer, network, video_pred, network_dtype
                    )
                    if _prior_div is not None:
                        _prior_div_value = _prior_div.detach().item()
                        loss = loss + _prior_div

                # CREPA loss — must be added before backward (shares computation graph)
                _crepa_value = None
                crepa_metrics = {}
                if hasattr(self, "_crepa") and self._crepa is not None:
                    self._crepa.on_step(global_step)
                    num_latent_frames = latents_tensor.shape[2]
                    dino_features = batch.get("conditions", {}).get("dino_features", None)
                    crepa_loss = self._crepa.compute_loss(num_latent_frames, dino_features=dino_features)
                    if crepa_loss is not None:
                        _crepa_value = crepa_loss.detach().item()
                        loss = loss + crepa_loss
                    crepa_metrics = self._crepa.get_metrics()
                    self._crepa.cleanup_step()

                # Self-Flow loss
                self_flow_metrics = {}
                if hasattr(self, "compute_self_flow_addition"):
                    if hasattr(self, "_self_flow") and self._self_flow is not None:
                        self._self_flow.on_step(global_step)
                    try:
                        self_flow_loss, self_flow_metrics = self.compute_self_flow_addition(
                            args,
                            accelerator,
                            transformer,
                            network,
                            network_dtype,
                        )
                        if self_flow_loss is not None:
                            loss = loss + self_flow_loss
                    except Exception as e:
                        logger.warning("Self-Flow loss computation failed: %s", e)

                # Cross-Task Synergy auxiliary losses
                cts_metrics = {}
                if dict_output:
                    cts_data = out.get("_cts")
                    if cts_data is not None:
                        try:
                            cts_loss, cts_metrics = compute_cross_task_synergy_losses(
                                transformer=transformer,
                                accelerator=accelerator,
                                noisy_video=cts_data["noisy_video"],
                                clean_video=cts_data["clean_video"],
                                video_target=out["video_target"],
                                video_timesteps=cts_data["video_timesteps"],
                                video_loss_mask=out.get("video_loss_mask"),
                                noisy_audio=cts_data["noisy_audio"],
                                clean_audio=cts_data["clean_audio"],
                                audio_target=out.get("audio_target"),
                                audio_timesteps=cts_data["audio_timesteps"],
                                audio_loss_mask=out.get("audio_loss_mask"),
                                text_embeds=cts_data["text_embeds"],
                                text_mask=cts_data["text_mask"],
                                frame_rate=cts_data["frame_rate"],
                                transformer_options=cts_data["transformer_options"],
                                lambda_video_driven=cts_data["lambda_video_driven"],
                                lambda_audio_driven=cts_data["lambda_audio_driven"],
                            )
                            if cts_loss is not None:
                                loss = loss + cts_loss
                        except Exception as e:
                            logger.warning("Cross-Task Synergy loss failed: %s", e)

                adaptive_rank_metrics = {}
                unwrapped_network = accelerator.unwrap_model(network)
                if hasattr(unwrapped_network, "compute_adaptive_rank_loss"):
                    adaptive_rank_loss = unwrapped_network.compute_adaptive_rank_loss()
                    if adaptive_rank_loss is not None:
                        loss = loss + adaptive_rank_loss
                        adaptive_rank_metrics = unwrapped_network.get_adaptive_rank_metrics()
                        adaptive_rank_metrics["loss/adaptive_rank"] = adaptive_rank_loss.detach().item()

                if _is_first_step:
                    _log_vram("FIRST_ITER: BEFORE backward", logger)
                accelerator.backward(loss)
                if _is_first_step:
                    _log_vram("FIRST_ITER: AFTER backward", logger)

                if dict_output and ogm_ge_state is not None:
                    maybe_add_ogm_ge_gradient_noise(
                        accelerator.unwrap_model(network),
                        video_coeff=ogm_ge_state.video_coeff,
                        audio_coeff=ogm_ge_state.audio_coeff,
                        noise_std_scale=float(getattr(args, "ogm_ge_noise_std", 0.0)),
                    )

                pres_losses = self.preservation_backward(args, accelerator, transformer, network, network_dtype)
                if _prior_div_value is not None:
                    pres_losses["loss/prior_div"] = _prior_div_value
                if _crepa_value is not None:
                    pres_losses["loss/crepa"] = _crepa_value
                if crepa_metrics:
                    pres_losses.update(crepa_metrics)
                if latent_temporal_metrics:
                    pres_losses.update(latent_temporal_metrics)
                if self_flow_metrics:
                    pres_losses.update(self_flow_metrics)
                if cts_metrics:
                    pres_losses.update(cts_metrics)
                if adaptive_rank_metrics:
                    pres_losses.update(adaptive_rank_metrics)

                # DEBUG: Check if LoRA parameters have gradients (requires LTX2_DEBUG env var)
                if os.environ.get("LTX2_DEBUG", "0") == "1":
                    unwrapped_net = accelerator.unwrap_model(network)
                    lora_modules = getattr(unwrapped_net, "unet_loras", [])
                    if lora_modules:
                        sample_loras = lora_modules[:3]
                        for lora in sample_loras:
                            logger.info(
                                f"[DEBUG] LoRA {lora.lora_name}: "
                                f"up_norm={lora.lora_up.weight.norm().item():.4f}, "
                                f"down_norm={lora.lora_down.weight.norm().item():.4f}"
                            )
                            up_grad = lora.lora_up.weight.grad
                            down_grad = lora.lora_down.weight.grad

                            up_stat = "None"
                            if up_grad is not None:
                                up_norm = up_grad.norm().item()
                                up_nan = torch.isnan(up_grad).any().item()
                                up_inf = torch.isinf(up_grad).any().item()
                                up_stat = f"norm={up_norm:.6f} nan={up_nan} inf={up_inf}"

                            down_stat = "None"
                            if down_grad is not None:
                                down_norm = down_grad.norm().item()
                                down_nan = torch.isnan(down_grad).any().item()
                                down_inf = torch.isinf(down_grad).any().item()
                                down_stat = f"norm={down_norm:.6f} nan={down_nan} inf={down_inf}"

                            logger.info(f"[DEBUG] LoRA Grad {lora.lora_name}:\n  UP  : {up_stat}\n  DOWN: {down_stat}")

                if accelerator.sync_gradients:
                    # self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                    state = accelerate.PartialState()
                    if state.distributed_type != accelerate.DistributedType.NO:
                        for param in network.parameters():
                            if param.grad is not None:
                                param.grad = accelerator.reduce(param.grad, reduction="mean")
                        if hasattr(self, "_crepa") and self._crepa is not None:
                            for param in self._crepa.get_trainable_params():
                                if param.grad is not None:
                                    param.grad = accelerator.reduce(param.grad, reduction="mean")
                        if hasattr(self, "_self_flow") and self._self_flow is not None:
                            for param in self._self_flow.get_trainable_params():
                                if param.grad is not None:
                                    param.grad = accelerator.reduce(param.grad, reduction="mean")

                    params_to_clip = list(accelerator.unwrap_model(network).get_trainable_params())
                    if hasattr(self, "_crepa") and self._crepa is not None:
                        params_to_clip.extend(self._crepa.get_trainable_params())
                    if hasattr(self, "_self_flow") and self._self_flow is not None:
                        params_to_clip.extend(self._self_flow.get_trainable_params())

                    if len(accelerator.trackers) > 0 or gui_metrics is not None:
                        total_grad_sq = torch.zeros(1, device=accelerator.device)
                        for param in params_to_clip:
                            if param.grad is not None:
                                grad_sq = param.grad.detach().float().pow(2).sum()
                                total_grad_sq += grad_sq.to(device=total_grad_sq.device, non_blocking=True)
                        grad_norm_total_value = total_grad_sq.sqrt().item()

                    # Per-modality gradient norm tracking (accumulate on GPU, sync once)
                    if (len(accelerator.trackers) > 0 or gui_metrics is not None) and dict_output:
                        unwrapped_net = accelerator.unwrap_model(network)
                        lora_modules = getattr(unwrapped_net, "unet_loras", None)
                        if lora_modules:
                            video_grad_sq = torch.zeros(1, device=accelerator.device)
                            audio_grad_sq = torch.zeros(1, device=accelerator.device)
                            for lora in lora_modules:
                                is_audio = "audio_" in lora.lora_name
                                for param in lora.parameters():
                                    if param.grad is not None:
                                        g_sq = param.grad.detach().float().pow(2).sum()
                                        if is_audio:
                                            audio_grad_sq += g_sq.to(device=audio_grad_sq.device, non_blocking=True)
                                        else:
                                            video_grad_sq += g_sq.to(device=video_grad_sq.device, non_blocking=True)
                            grad_norm_video_value = video_grad_sq.sqrt().item()
                            grad_norm_audio_value = audio_grad_sq.sqrt().item()

                    if args.max_grad_norm != 0.0:
                        if model_parallel:
                            self.clip_grad_norm_for_model_parallel(args, accelerator, params_to_clip, optimizer)
                        else:
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if _is_first_step:
                    _log_vram("FIRST_ITER: BEFORE optimizer.step", logger)
                optimizer.step()
                if (
                    accelerator.sync_gradients
                    and bool(getattr(args, "ltx2_remote_stage", False))
                    and bool(getattr(args, "ltx2_remote_stage_trainable", False))
                ):
                    try:
                        from musubi_tuner.ltx2_remote_stage import optimizer_step_ltx2_remote_stage

                        optimizer_step_ltx2_remote_stage(transformer)
                    except Exception as e:
                        logger.warning("Remote LTX-2 stage optimizer step failed: %s", e)
                        raise
                if _is_first_step:
                    _log_vram("FIRST_ITER: AFTER optimizer.step", logger)
                if accelerator.sync_gradients and hasattr(self, "_self_flow") and self._self_flow is not None:
                    try:
                        # Use stored network ref: may be LoRA network or transformer (full fine-tuning).
                        _sf_net = getattr(self, "_self_flow_network", None) or (
                            accelerator.unwrap_model(network) if network is not None else None
                        )
                        if _sf_net is not None:
                            self._self_flow.update_teacher(_sf_net)
                    except Exception as e:
                        logger.warning("Self-Flow EMA update failed: %s", e)
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if (
                    accelerator.sync_gradients
                    and bool(getattr(args, "ltx2_remote_stage", False))
                    and bool(getattr(args, "ltx2_remote_stage_trainable", False))
                ):
                    try:
                        from musubi_tuner.ltx2_remote_stage import zero_grad_ltx2_remote_stage

                        zero_grad_ltx2_remote_stage(transformer)
                    except Exception as e:
                        logger.warning("Remote LTX-2 stage zero_grad failed: %s", e)
                        raise
                _prev_step_end_time = time.perf_counter()
                if _is_first_step:
                    _log_vram("FIRST_ITER: AFTER zero_grad (end of first step)", logger)

            if args.scale_weight_norms:
                keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                    args.scale_weight_norms, accelerator.device
                )
                max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
            else:
                keys_scaled, mean_norm, maximum_norm = None, None, None

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if global_step == 0:
                    progress_bar.reset()  # exclude first step from progress bar, because it may take long due to initializations
                progress_bar.update(1)
                global_step += 1
                unwrapped_network = accelerator.unwrap_model(network)
                finalized_this_step = False
                if hasattr(unwrapped_network, "maybe_finalize_adaptive_rank"):
                    old_network_param_ids = {id(param) for param in unwrapped_network.get_trainable_params()}
                    finalize_report = unwrapped_network.maybe_finalize_adaptive_rank(
                        global_step=global_step,
                        max_train_steps=args.max_train_steps,
                    )
                    if finalize_report is not None:
                        lr_scheduler, lr_descriptions = self._refresh_optimizer_after_adaptive_rank_prune(
                            args,
                            accelerator,
                            network,
                            optimizer,
                            lr_scheduler,
                            old_network_param_ids=old_network_param_ids,
                            global_step=global_step,
                            recovery_config=finalize_report.get("recovery_config"),
                        )
                        finalized_this_step = True
                        logger.info(
                            "adaptive rank finalize at step %s: modules=%s rank_sum=%s->%s recovery=%s",
                            finalize_report["step"],
                            finalize_report["finalized_module_count"],
                            finalize_report["rank_sum_before"],
                            finalize_report["rank_sum_after"],
                            "on" if finalize_report.get("recovery_config") is not None else "off",
                        )
                if not finalized_this_step and hasattr(unwrapped_network, "maybe_hard_prune_adaptive_rank"):
                    old_network_param_ids = {id(param) for param in unwrapped_network.get_trainable_params()}
                    hard_prune_report = unwrapped_network.maybe_hard_prune_adaptive_rank(
                        global_step=global_step,
                        max_train_steps=args.max_train_steps,
                    )
                    if hard_prune_report is not None:
                        lr_scheduler, lr_descriptions = self._refresh_optimizer_after_adaptive_rank_prune(
                            args,
                            accelerator,
                            network,
                            optimizer,
                            lr_scheduler,
                            old_network_param_ids=old_network_param_ids,
                            global_step=global_step,
                        )
                        logger.info(
                            "adaptive rank hard prune at step %s: modules=%s rank_sum=%s->%s",
                            hard_prune_report["step"],
                            hard_prune_report["pruned_module_count"],
                            hard_prune_report["rank_sum_before"],
                            hard_prune_report["rank_sum_after"],
                        )
                if not finalized_this_step and hasattr(unwrapped_network, "maybe_reallocate_adaptive_rank_estimate"):
                    reallocate_report = unwrapped_network.maybe_reallocate_adaptive_rank_estimate(
                        global_step=global_step,
                        max_train_steps=args.max_train_steps,
                    )
                    if reallocate_report is not None:
                        logger.info(
                            "adaptive rank estimate reallocation at step %s: changed=%s fixed_budget=%.3f remaining_budget=%.3f",
                            reallocate_report["step"],
                            reallocate_report["changed_module_count"],
                            reallocate_report["fixed_budget"],
                            reallocate_report["remaining_budget"],
                        )
                if timestep_tb_buffers is not None and (global_step == 1 or global_step % timestep_tb_interval == 0):
                    for name, chunks in timestep_tb_buffers.items():
                        if not chunks:
                            continue
                        values = torch.cat(list(chunks), dim=0)
                        tag = "timestep/used_values" if name == "main" else f"timestep/used_values_{name}"
                        self._log_timestep_distribution_histogram(
                            accelerator,
                            global_step,
                            tag,
                            values,
                        )
                        chunks.clear()
                if (
                    args.log_cuda_memory_every_n_steps is not None
                    and args.log_cuda_memory_every_n_steps > 0
                    and accelerator.device.type == "cuda"
                    and global_step % args.log_cuda_memory_every_n_steps == 0
                ):
                    _log_cuda_memory_stats(f"step_{global_step}", latents_shape=latents_shape)

                # G2D modality freezer: update loss EMA and check freeze state
                if modality_freezer is not None:
                    modality_freezer.update_losses(video_loss_value, audio_loss_value)
                    modality_freezer.maybe_update_freeze(
                        global_step,
                        accelerator.unwrap_model(network),
                    )

                # to avoid calling optimizer_eval_fn() too frequently, we call it only when we need to sample images or save the model
                should_sampling = should_sample_images(args, global_step, epoch=None)
                should_saving = args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0

                if should_sampling or should_saving:
                    set_trainer_eval_mode()
                    if should_sampling:
                        with offload_optimizer_state_during_validation(
                            optimizer,
                            accelerator,
                            bool(getattr(args, "offload_optimizer_during_validation", False)),
                            logger=logger,
                        ):
                            self.sample_images(accelerator, args, None, global_step, vae, transformer, sample_parameters, dit_dtype)
                        if gui_metrics is not None:
                            gui_metrics.log_event("sample", global_step)

                    if should_saving:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_utils.get_step_ckpt_name(args.output_name, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)
                            if gui_metrics is not None:
                                gui_metrics.log_event("checkpoint", global_step)

                            if args.save_state:
                                train_utils.save_and_remove_state_stepwise(
                                    args, accelerator, global_step, epoch=epoch + 1, step_in_epoch=step + 1
                                )
                                _state_dir = os.path.join(
                                    args.output_dir,
                                    train_utils.STEP_STATE_NAME.format(args.output_name, global_step),
                                )
                                train_utils.update_resume_metadata(
                                    _state_dir,
                                    {
                                        "loss_avg": loss_recorder.moving_average,
                                        "loss_count": len(loss_recorder.loss_list),
                                    },
                                )

                            remove_step_no = train_utils.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no)
                                remove_model(remove_ckpt_name)
                    set_trainer_train_mode()

            current_loss = loss.detach().item()
            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            if dict_output:
                logs["loss_v"] = video_loss_value if video_loss_value is not None else "n/a"
                logs["loss_a"] = audio_loss_value if audio_loss_value is not None else "n/a"
                if audio_weight_effective_value is not None:
                    logs["audio_w"] = audio_weight_effective_value
                if audio_presence_ema_value is not None:
                    logs["audio_p"] = audio_presence_ema_value
                if "video_mask_active" in mask_metrics:
                    logs["mv_act"] = round(mask_metrics["video_mask_active"], 3)
                if "audio_mask_active" in mask_metrics:
                    logs["ma_act"] = round(mask_metrics["audio_mask_active"], 3)
            progress_bar.set_postfix(**logs)

            if args.scale_weight_norms:
                progress_bar.set_postfix(**{**max_mean_logs, **logs})

            if len(accelerator.trackers) > 0:
                logs = self.generate_step_logs(
                    args,
                    current_loss,
                    avr_loss,
                    lr_scheduler,
                    lr_descriptions,
                    optimizer,
                    keys_scaled,
                    mean_norm,
                    maximum_norm,
                    video_loss=video_loss_value,
                    audio_loss=audio_loss_value,
                    mask_metrics=mask_metrics if mask_metrics else None,
                )
                if audio_weight_effective_value is not None:
                    logs["loss/audio_weight_effective"] = audio_weight_effective_value
                if ogm_ge_state is not None:
                    logs["ogm_ge/video_coeff"] = ogm_ge_state.video_coeff
                    logs["ogm_ge/audio_coeff"] = ogm_ge_state.audio_coeff
                    logs["ogm_ge/discrepancy"] = ogm_ge_state.discrepancy
                if audio_presence_ema_value is not None:
                    logs["loss/audio_presence_ema"] = audio_presence_ema_value
                if audio_loss_ema_value is not None:
                    logs["loss/audio_loss_ema"] = audio_loss_ema_value
                if video_loss_ema_value is not None:
                    logs["loss/video_loss_ema"] = video_loss_ema_value
                if grad_norm_video_value is not None:
                    logs["grad_norm/video"] = grad_norm_video_value
                if grad_norm_audio_value is not None:
                    logs["grad_norm/audio"] = grad_norm_audio_value
                    if grad_norm_video_value is not None and grad_norm_video_value > 0:
                        logs["grad_norm/audio_video_ratio"] = grad_norm_audio_value / grad_norm_video_value
                if grad_norm_total_value is not None:
                    logs["grad_norm/total"] = grad_norm_total_value
                if uncertainty_log_var_video is not None:
                    lv_v = uncertainty_log_var_video.detach().item()
                    lv_a = uncertainty_log_var_audio.detach().item()
                    logs["uncertainty/log_var_video"] = lv_v
                    logs["uncertainty/log_var_audio"] = lv_a
                    logs["uncertainty/precision_video"] = math.exp(-lv_v)
                    logs["uncertainty/precision_audio"] = math.exp(-lv_a)
                if modality_freezer is not None:
                    # Encode state as numeric: 0=both active, 1=audio frozen, -1=video frozen
                    state_map = {"both": 0, "audio_frozen": 1, "video_frozen": -1}
                    logs["modality_freeze/state"] = state_map.get(modality_freezer.state, 0)
                    logs["modality_freeze/video_loss_ema"] = modality_freezer.video_loss_ema
                    logs["modality_freeze/audio_loss_ema"] = modality_freezer.audio_loss_ema
                if pres_losses:
                    logs.update(pres_losses)
                if audio_diagnostics:
                    logs.update(audio_diagnostics)
                accelerator.log(logs, step=global_step)

                # Log automagic LR histogram directly to tracker
                if args.optimizer_type.lower() == "automagic" and optimizer is not None:
                    lr_tensor = optimizer.get_lr_tensor()
                    if lr_tensor is not None and lr_tensor.mean() > 0:
                        for tracker in accelerator.trackers:
                            if tracker.name == "tensorboard":
                                tracker.writer.add_histogram("lr/automagic_lrs", lr_tensor, global_step)
                            elif tracker.name == "wandb":
                                import wandb

                                tracker.log({"lr/automagic_lrs": wandb.Histogram(lr_tensor.cpu().numpy())}, step=global_step)

            # GUI dashboard per-step metrics
            if gui_metrics is not None:
                step_time = time.perf_counter() - _step_start_time
                gui_metrics.log(
                    step=global_step,
                    epoch=epoch,
                    loss=current_loss,
                    avr_loss=avr_loss,
                    loss_v=video_loss_value,
                    loss_a=audio_loss_value,
                    grad_norm=grad_norm_total_value,
                    grad_norm_v=grad_norm_video_value,
                    grad_norm_a=grad_norm_audio_value,
                    lr=lr_scheduler.get_last_lr()[0],
                    step_time=step_time,
                    data_wait_time=_data_wait_time,
                )
                gui_metrics.update_status(
                    step=global_step,
                    max_steps=args.max_train_steps,
                    epoch=epoch + 1,
                    max_epochs=num_train_epochs,
                    status="training",
                )

            if (
                validation_dataloader is not None
                and args.validate_every_n_steps is not None
                and global_step % args.validate_every_n_steps == 0
            ):
                with offload_optimizer_state_during_validation(
                    optimizer,
                    accelerator,
                    bool(getattr(args, "offload_optimizer_during_validation", False)),
                    logger=logger,
                ):
                    run_validation(global_step)

            _prev_step_end_time = time.perf_counter()
            if handle_dashboard_stop_request(global_step, epoch, step + 1):
                return

            if global_step >= args.max_train_steps:
                break

        # ensure skip counter doesn't carry into next epoch (e.g. if dataset shrunk)
        steps_to_skip_in_epoch = 0

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        if (
            validation_dataloader is not None
            and args.validate_every_n_epochs is not None
            and (epoch + 1) % args.validate_every_n_epochs == 0
        ):
            with offload_optimizer_state_during_validation(
                optimizer,
                accelerator,
                bool(getattr(args, "offload_optimizer_during_validation", False)),
                logger=logger,
            ):
                run_validation(global_step, epoch_no=epoch + 1)

        accelerator.wait_for_everyone()

        # save model at the end of epoch if needed
        set_trainer_eval_mode()
        if args.save_every_n_epochs is not None:
            saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
            if is_main_process and saving:
                ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch + 1)
                save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
                if remove_epoch_no is not None:
                    remove_ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch_no)
                    remove_model(remove_ckpt_name)

                if args.save_state:
                    train_utils.save_and_remove_state_on_epoch_end(
                        args, accelerator, epoch + 1, global_step=global_step, step_in_epoch=0
                    )
                    _state_dir = os.path.join(
                        args.output_dir,
                        train_utils.EPOCH_STATE_NAME.format(args.output_name, epoch + 1),
                    )
                    train_utils.update_resume_metadata(
                        _state_dir,
                        {
                            "loss_avg": loss_recorder.moving_average,
                            "loss_count": len(loss_recorder.loss_list),
                        },
                    )

        offload_epoch_sample_optimizer = bool(getattr(args, "offload_optimizer_during_validation", False)) and (
            should_sample_images(args, global_step, epoch=epoch + 1)
        )
        with offload_optimizer_state_during_validation(
            optimizer,
            accelerator,
            offload_epoch_sample_optimizer,
            logger=logger,
        ):
            self.sample_images(accelerator, args, epoch + 1, global_step, vae, transformer, sample_parameters, dit_dtype)
        set_trainer_train_mode()

        # end of epoch

    # metadata["ss_epoch"] = str(num_train_epochs)
    metadata["ss_training_finished_at"] = str(time.time())

    if gui_metrics is not None:
        gui_metrics.update_status(
            step=global_step,
            max_steps=args.max_train_steps,
            epoch=num_train_epochs,
            max_epochs=num_train_epochs,
            status="completed",
        )
        gui_metrics.close()

    if is_main_process:
        network = accelerator.unwrap_model(network)

    accelerator.end_training()
    set_trainer_eval_mode()

    if is_main_process and (args.save_state or args.save_state_on_train_end):
        train_utils.save_state_on_train_end(args, accelerator, global_step=global_step, epoch=num_train_epochs)
        _state_dir = os.path.join(
            args.output_dir,
            train_utils.LAST_STATE_NAME.format(args.output_name),
        )
        train_utils.update_resume_metadata(
            _state_dir,
            {
                "loss_avg": loss_recorder.moving_average,
                "loss_count": len(loss_recorder.loss_list),
            },
        )

    if is_main_process:
        ckpt_name = train_utils.get_last_ckpt_name(args.output_name)
        save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

        logger.info("model saved.")
