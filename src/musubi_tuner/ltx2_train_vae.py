#!/usr/bin/env python3
"""Train or fine-tune the LTX-2 video VAE.

The default mode is decoder-only fine-tuning. This keeps the diffusion model's
latent space stable while adapting pixel reconstruction quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors import safe_open
from tqdm import tqdm

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2, AudioDataset, BaseDataset, ItemInfo
from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
from musubi_tuner.ltx_2.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.utils.train_utils import load_resume_metadata, save_resume_metadata


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PreparedBatch:
    pixels: torch.Tensor
    masks: Optional[torch.Tensor]
    original_frames: int


def _load_ltx2_config_metadata(path: str) -> str:
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    if metadata is None or "config" not in metadata:
        raise ValueError(f"{path} does not contain LTX-2 config metadata")
    return metadata["config"]


def _amp_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "xpu"} and dtype in {torch.float16, torch.bfloat16}:
        try:
            from torch.amp import autocast as torch_autocast  # type: ignore[attr-defined]

            return torch_autocast(device_type=device.type, dtype=dtype)
        except (ImportError, AttributeError):
            from torch.cuda.amp import autocast as torch_autocast

            return torch_autocast(dtype=dtype)
    return nullcontext()


def _load_dataset_split(args: argparse.Namespace, *, validation: bool = False) -> list[BaseDataset]:
    user_config = config_utils.load_user_config(args.dataset_config)
    if validation:
        validation_entries = user_config.get("validation_datasets")
        if not validation_entries:
            return []
        user_config = {
            "general": user_config.get("general", {}),
            "datasets": validation_entries,
        }

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LTX2)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.dataset_group,
        reference_downscale=getattr(args, "reference_downscale", 1),
    )

    datasets: list[BaseDataset] = []
    for dataset in dataset_group.datasets:
        if isinstance(dataset, AudioDataset):
            logger.warning("Skipping audio dataset for video VAE training: %s", getattr(dataset, "audio_directory", None))
            continue
        if getattr(dataset, "cache_only", False):
            raise ValueError("VAE training needs source images/videos; cache_only datasets are not supported")
        datasets.append(dataset)
    return datasets


def _iter_raw_batches(datasets: Sequence[BaseDataset], num_workers: int, *, shuffle_datasets: bool) -> Iterable[list[ItemInfo]]:
    dataset_order = list(datasets)
    if shuffle_datasets:
        random.shuffle(dataset_order)

    for dataset in dataset_order:
        for _key, batch in dataset.retrieve_latent_cache_batches(num_workers):
            if batch:
                yield batch


def _content_to_tensor(content) -> torch.Tensor:
    if isinstance(content, list):
        frames = [torch.from_numpy(frame) for frame in content]
        return torch.stack(frames, dim=0)

    tensor = torch.from_numpy(content)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected image/video content with shape HWC or FHWC, got {tuple(tensor.shape)}")
    return tensor


def _mask_to_tensor(mask, frame_count: int) -> torch.Tensor:
    mask_tensor = torch.from_numpy(mask).to(dtype=torch.float32)
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    if mask_tensor.ndim != 3:
        raise ValueError(f"Expected loss mask with shape HW or FHW, got {tuple(mask_tensor.shape)}")
    if mask_tensor.shape[0] < frame_count:
        pad = frame_count - int(mask_tensor.shape[0])
        mask_tensor = torch.cat([mask_tensor, mask_tensor[-1:].expand(pad, -1, -1)], dim=0)
    elif mask_tensor.shape[0] > frame_count:
        mask_tensor = mask_tensor[:frame_count]
    return mask_tensor.clamp(0.0, 1.0)


def prepare_batch(batch: list[ItemInfo], device: torch.device, dtype: torch.dtype) -> PreparedBatch:
    contents = [_content_to_tensor(item.content) for item in batch]
    frame_counts = {int(content.shape[0]) for content in contents}
    if len(frame_counts) != 1:
        raise ValueError(f"Batch contains mixed frame counts: {sorted(frame_counts)}")
    original_frames = frame_counts.pop()

    has_masks = any(getattr(item, "loss_mask_content", None) is not None for item in batch)
    masks: list[torch.Tensor] = []
    if has_masks:
        for item, content in zip(batch, contents, strict=True):
            mask = getattr(item, "loss_mask_content", None)
            if mask is None:
                mask_tensor = torch.ones((content.shape[0], content.shape[1], content.shape[2]), dtype=torch.float32)
            else:
                mask_tensor = _mask_to_tensor(mask, int(content.shape[0]))
            masks.append(mask_tensor)

    pixels = torch.stack(contents, dim=0).permute(0, 4, 1, 2, 3).contiguous()
    pixels = pixels.to(device=device, dtype=dtype)
    pixels = pixels / 127.5 - 1.0

    frames = int(pixels.shape[2])
    remainder = (frames - 1) % 8
    if remainder != 0:
        pad = 8 - remainder
        pixels = torch.cat([pixels, pixels[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        if has_masks:
            padded_masks = []
            for mask_tensor in masks:
                padded_masks.append(torch.cat([mask_tensor, mask_tensor[-1:].expand(pad, -1, -1)], dim=0))
            masks = padded_masks

    mask_batch = None
    if has_masks:
        mask_batch = torch.stack(masks, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
        if mask_batch.shape[2] != pixels.shape[2]:
            raise ValueError(f"Mask frame count mismatch: mask={tuple(mask_batch.shape)} pixels={tuple(pixels.shape)}")

    return PreparedBatch(pixels=pixels, masks=mask_batch, original_frames=original_frames)


def _elementwise_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor, loss_type: str, eps: float) -> torch.Tensor:
    diff = recon.float() - target.float()
    if loss_type == "l1":
        return diff.abs()
    if loss_type == "mse":
        return diff.square()
    if loss_type == "charbonnier":
        return torch.sqrt(diff.square() + eps * eps)
    raise ValueError(f"Unsupported reconstruction loss: {loss_type}")


def _masked_mean(loss_map: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return loss_map.mean()
    mask = mask.to(device=loss_map.device, dtype=loss_map.dtype)
    denom = mask.sum() * loss_map.shape[1]
    return (loss_map * mask).sum() / denom.clamp_min(1.0)


def compute_vae_loss(
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    batch: PreparedBatch,
    args: argparse.Namespace,
    train_encoder: bool,
    reference_encoder: Optional[torch.nn.Module] = None,
    tiling_config: Optional[TilingConfig] = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    pixels = batch.pixels
    mask = batch.masks

    if train_encoder:
        latents = encoder(pixels)
    else:
        with torch.no_grad():
            if tiling_config is not None and hasattr(encoder, "tiled_encode"):
                latents = encoder.tiled_encode(pixels, tiling_config)
            else:
                latents = encoder(pixels)

    timestep = None
    if bool(getattr(decoder, "timestep_conditioning", False)):
        timestep = torch.full(
            (latents.shape[0],),
            float(args.decode_timestep),
            device=latents.device,
            dtype=latents.dtype,
        )

    recon = decoder(latents, timestep=timestep)
    target = pixels
    valid_frames = min(int(batch.original_frames), int(recon.shape[2]), int(target.shape[2]))
    recon = recon[:, :, :valid_frames]
    target = target[:, :, :valid_frames]
    if mask is not None:
        mask = mask[:, :, :valid_frames]
    if recon.shape[2] != target.shape[2]:
        frames = min(int(recon.shape[2]), int(target.shape[2]))
        recon = recon[:, :, :frames]
        target = target[:, :, :frames]
        if mask is not None:
            mask = mask[:, :, :frames]
    if recon.shape[-2:] != target.shape[-2:]:
        height = min(int(recon.shape[-2]), int(target.shape[-2]))
        width = min(int(recon.shape[-1]), int(target.shape[-1]))
        recon = recon[..., :height, :width]
        target = target[..., :height, :width]
        if mask is not None:
            mask = mask[..., :height, :width]

    recon_loss = _masked_mean(
        _elementwise_reconstruction_loss(recon, target, args.reconstruction_loss, args.charbonnier_eps),
        mask,
    )
    total_loss = recon_loss * float(args.reconstruction_loss_weight)

    temporal_loss = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)
    if float(args.temporal_loss_weight) > 0 and recon.shape[2] > 1:
        recon_delta = recon[:, :, 1:].float() - recon[:, :, :-1].float()
        target_delta = target[:, :, 1:].float() - target[:, :, :-1].float()
        temporal_loss = (recon_delta - target_delta).abs().mean()
        total_loss = total_loss + temporal_loss * float(args.temporal_loss_weight)

    latent_reg_loss = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)
    if train_encoder and reference_encoder is not None and float(args.latent_regularization_weight) > 0:
        with torch.no_grad():
            ref_latents = reference_encoder(pixels)
        latent_reg_loss = F.mse_loss(latents.float(), ref_latents.float())
        total_loss = total_loss + latent_reg_loss * float(args.latent_regularization_weight)

    metrics = {
        "loss": float(total_loss.detach().float().cpu()),
        "recon_loss": float(recon_loss.detach().float().cpu()),
        "temporal_loss": float(temporal_loss.detach().float().cpu()),
        "latent_reg_loss": float(latent_reg_loss.detach().float().cpu()),
    }
    return total_loss, metrics, recon.detach(), target.detach()


def _build_tiling_config(args: argparse.Namespace) -> Optional[TilingConfig]:
    spatial_config = None
    temporal_config = None
    if args.vae_spatial_tile_size is not None:
        spatial_config = SpatialTilingConfig(
            tile_size_in_pixels=args.vae_spatial_tile_size,
            tile_overlap_in_pixels=args.vae_spatial_tile_overlap,
        )
    if args.vae_temporal_tile_size is not None:
        temporal_config = TemporalTilingConfig(
            tile_size_in_frames=args.vae_temporal_tile_size,
            tile_overlap_in_frames=args.vae_temporal_tile_overlap,
        )
    if spatial_config is None and temporal_config is None:
        return None
    return TilingConfig(spatial_config=spatial_config, temporal_config=temporal_config)


def _set_decoder_runtime_options(decoder: torch.nn.Module, args: argparse.Namespace) -> None:
    if hasattr(decoder, "decode_timestep"):
        decoder.decode_timestep = float(args.decode_timestep)
    if hasattr(decoder, "decode_noise_scale"):
        decoder.decode_noise_scale = 0.0 if args.disable_decode_noise else float(args.decode_noise_scale)
    if args.vae_chunk_size is not None and hasattr(decoder, "set_chunk_size_for_causal_conv_3d"):
        decoder.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)


def _set_encoder_runtime_options(encoder: torch.nn.Module, args: argparse.Namespace) -> None:
    if args.vae_chunk_size is not None and hasattr(encoder, "set_chunk_size_for_causal_conv_3d"):
        encoder.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)


def load_vae_components(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    *,
    load_reference_encoder: bool,
) -> tuple[torch.nn.Module, torch.nn.Module, Optional[torch.nn.Module]]:
    logger.info("Loading LTX-2 VAE encoder from %s", args.vae)
    encoder = SingleGPUModelBuilder(
        model_path=str(args.vae),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype)

    logger.info("Loading LTX-2 VAE decoder from %s", args.vae)
    decoder = SingleGPUModelBuilder(
        model_path=str(args.vae),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype)

    reference_encoder = None
    if load_reference_encoder:
        logger.info("Loading frozen reference encoder for latent regularization")
        reference_encoder = SingleGPUModelBuilder(
            model_path=str(args.vae),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=dtype)
        reference_encoder.eval()
        reference_encoder.requires_grad_(False)
        _set_encoder_runtime_options(reference_encoder, args)

    _set_encoder_runtime_options(encoder, args)
    _set_decoder_runtime_options(decoder, args)
    return encoder, decoder, reference_encoder


def _build_optimizer(args: argparse.Namespace, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if args.optimizer_type.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer_type: {args.optimizer_type}")
    return torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )


def _build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    scheduler_name = args.lr_scheduler.lower()
    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    if args.max_train_steps is None or args.max_train_steps <= 0:
        raise ValueError(f"--max_train_steps is required when --lr_scheduler={args.lr_scheduler}")

    warmup_steps = max(0, int(args.lr_warmup_steps))
    total_steps = max(1, int(args.max_train_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_name == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _state_dict_with_ltx2_prefix(
    *,
    encoder: Optional[torch.nn.Module],
    decoder: torch.nn.Module,
    save_encoder: bool,
    save_dtype: Optional[torch.dtype],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}

    def add_tensor(key: str, tensor: torch.Tensor) -> None:
        tensor = tensor.detach()
        if save_dtype is not None and tensor.dtype.is_floating_point:
            tensor = tensor.to(dtype=save_dtype)
        tensors[key] = tensor.contiguous()

    if save_encoder:
        if encoder is None:
            raise ValueError("save_encoder=True but encoder is None")
        for name, tensor in encoder.state_dict().items():
            if name.startswith("per_channel_statistics."):
                continue
            add_tensor(f"vae.encoder.{name}", tensor)

    for name, tensor in decoder.state_dict().items():
        if name.startswith("per_channel_statistics."):
            continue
        add_tensor(f"vae.decoder.{name}", tensor)

    stats = getattr(decoder, "per_channel_statistics", None)
    if stats is not None:
        for name, tensor in stats.state_dict().items():
            add_tensor(f"vae.per_channel_statistics.{name}", tensor)

    return tensors


def save_vae_checkpoint(
    *,
    accelerator: Accelerator,
    args: argparse.Namespace,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    config_json: str,
    output_path: str,
    global_step: int,
    epoch: int,
    val_loss: Optional[float] = None,
) -> None:
    if not accelerator.is_main_process:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    unwrapped_encoder = accelerator.unwrap_model(encoder)
    unwrapped_decoder = accelerator.unwrap_model(decoder)
    save_dtype = str_to_dtype(args.save_dtype) if args.save_dtype is not None else None
    save_encoder = args.save_format == "full"

    tensors = _state_dict_with_ltx2_prefix(
        encoder=unwrapped_encoder,
        decoder=unwrapped_decoder,
        save_encoder=save_encoder,
        save_dtype=save_dtype,
    )
    metadata = {
        "config": config_json,
        "format": "ltx2_video_vae",
        "ss_training_script": "ltx2_train_vae.py",
        "ss_train_target": args.train_target,
        "ss_save_format": args.save_format,
        "ss_global_step": str(global_step),
        "ss_epoch": str(epoch),
        "ss_learning_rate": str(args.learning_rate),
        "ss_reconstruction_loss": args.reconstruction_loss,
        "ss_reconstruction_loss_weight": str(args.reconstruction_loss_weight),
        "ss_temporal_loss_weight": str(args.temporal_loss_weight),
        "ss_latent_regularization_weight": str(args.latent_regularization_weight),
        "ss_vae_dtype": args.vae_dtype,
    }
    if args.save_dtype is not None:
        metadata["ss_save_dtype"] = args.save_dtype
    if val_loss is not None:
        metadata["ss_validation_loss"] = str(val_loss)

    logger.info("Saving VAE checkpoint to %s", output_path)
    safetensors_utils.mem_eff_save_file(tensors, output_path, metadata=metadata)

    sidecar = os.path.splitext(output_path)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metadata.items() if k != "config"}, f, indent=2)


def _tensor_to_pil(frame: torch.Tensor):
    from PIL import Image

    frame = (((frame.float() + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    frame = frame.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(frame)


def save_preview(
    *,
    args: argparse.Namespace,
    target: torch.Tensor,
    recon: torch.Tensor,
    output_dir: str,
    global_step: int,
) -> None:
    from PIL import Image, ImageDraw

    if args.preview_samples <= 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    target = target.detach().cpu()
    recon = recon.detach().cpu()
    sample_count = min(int(args.preview_samples), int(target.shape[0]), int(recon.shape[0]))
    frame_count = min(int(target.shape[2]), int(recon.shape[2]))
    frame_indices = sorted(set([0, frame_count // 2, frame_count - 1]))

    tiles = []
    labels = []
    for sample_idx in range(sample_count):
        for frame_idx in frame_indices:
            tiles.append(_tensor_to_pil(target[sample_idx, :, frame_idx]))
            labels.append(f"s{sample_idx} f{frame_idx} target")
            tiles.append(_tensor_to_pil(recon[sample_idx, :, frame_idx]))
            labels.append(f"s{sample_idx} f{frame_idx} recon")

    if not tiles:
        return

    max_width = max(64, int(args.preview_tile_width))
    resized_tiles = []
    for tile in tiles:
        if tile.width > max_width:
            scale = max_width / float(tile.width)
            tile = tile.resize((max_width, max(1, int(tile.height * scale))), Image.LANCZOS)
        resized_tiles.append(tile)

    tile_w = max(tile.width for tile in resized_tiles)
    tile_h = max(tile.height for tile in resized_tiles)
    label_h = 18
    cols = max(1, len(frame_indices) * 2)
    rows = math.ceil(len(resized_tiles) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for idx, tile in enumerate(resized_tiles):
        row = idx // cols
        col = idx % cols
        x = col * tile_w
        y = row * (tile_h + label_h)
        canvas.paste(tile, (x, y + label_h))
        draw.text((x + 4, y + 2), labels[idx], fill=(235, 235, 235))

    preview_path = os.path.join(output_dir, f"{args.output_name}-step{global_step:08d}-preview.jpg")
    canvas.save(preview_path, quality=92)
    logger.info("Saved preview: %s", preview_path)


@torch.no_grad()
def evaluate(
    *,
    args: argparse.Namespace,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    datasets: Sequence[BaseDataset],
    device: torch.device,
    dtype: torch.dtype,
    tiling_config: Optional[TilingConfig],
    max_batches: int,
) -> Optional[float]:
    if not datasets or max_batches <= 0:
        return None

    was_encoder_training = encoder.training
    was_decoder_training = decoder.training
    encoder.eval()
    decoder.eval()

    losses: list[float] = []
    for batch_idx, batch in enumerate(
        _iter_raw_batches(datasets, args.max_data_loader_n_workers, shuffle_datasets=False)
    ):
        if batch_idx >= max_batches:
            break
        prepared = prepare_batch(batch, device, dtype)
        with _amp_context(device, dtype):
            loss, _metrics, _recon, _target = compute_vae_loss(
                encoder=encoder,
                decoder=decoder,
                batch=prepared,
                args=args,
                train_encoder=False,
                reference_encoder=None,
                tiling_config=tiling_config,
            )
        losses.append(float(loss.detach().float().cpu()))

    if was_encoder_training:
        encoder.train()
    if was_decoder_training:
        decoder.train()

    if not losses:
        return None
    return sum(losses) / len(losses)


def train(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if accelerator.num_processes != 1:
        raise ValueError("ltx2_train_vae.py currently supports single-process training only")

    if args.vae is None:
        args.vae = args.ltx2_checkpoint
    config_json = _load_ltx2_config_metadata(args.vae)
    vae_dtype = str_to_dtype(args.vae_dtype)
    device = accelerator.device

    train_datasets = _load_dataset_split(args, validation=False)
    val_datasets = _load_dataset_split(args, validation=True)
    if not train_datasets:
        raise ValueError("No image/video datasets found for VAE training")

    train_encoder = args.train_target == "encoder_decoder"
    load_reference_encoder = train_encoder and float(args.latent_regularization_weight) > 0
    encoder, decoder, reference_encoder = load_vae_components(
        args,
        device,
        vae_dtype,
        load_reference_encoder=load_reference_encoder,
    )

    encoder.train(train_encoder)
    encoder.requires_grad_(train_encoder)
    decoder.train(True)
    decoder.requires_grad_(True)

    trainable_parameters = list(decoder.parameters())
    if train_encoder:
        trainable_parameters.extend(list(encoder.parameters()))

    optimizer = _build_optimizer(args, (p for p in trainable_parameters if p.requires_grad))
    scheduler = _build_scheduler(args, optimizer)

    if train_encoder:
        encoder, decoder, optimizer, scheduler = accelerator.prepare(encoder, decoder, optimizer, scheduler)
    else:
        decoder, optimizer, scheduler = accelerator.prepare(decoder, optimizer, scheduler)

    resume_global_step = 0
    resume_epoch = 0
    if args.resume is not None:
        logger.info("Loading accelerator state from %s", args.resume)
        accelerator.load_state(args.resume)
        resume_metadata = load_resume_metadata(args.resume)
        if resume_metadata is not None:
            resume_global_step = int(resume_metadata.get("global_step", 0))
            resume_epoch = int(resume_metadata.get("epoch", 0))
            logger.info("Resumed metadata: global_step=%d epoch=%d", resume_global_step, resume_epoch)

    tiling_config = _build_tiling_config(args)
    if tiling_config is not None and train_encoder:
        logger.warning("Encoder tiling is disabled while training the encoder because tiled_encode is intended for no-grad use")
        tiling_config = None

    global_step = resume_global_step
    epoch = resume_epoch
    moving_loss: Optional[float] = None
    last_recon = None
    last_target = None

    os.makedirs(args.output_dir, exist_ok=True)
    preview_dir = os.path.join(args.output_dir, "vae_previews")

    if args.max_train_epochs is None:
        max_epochs = 1 if args.max_train_steps is None else 1_000_000_000
    else:
        max_epochs = int(args.max_train_epochs)
    progress_total = None
    if args.max_train_steps is not None and args.max_train_steps > 0:
        progress_total = max(0, int(args.max_train_steps) - int(global_step))
    progress = tqdm(total=progress_total, disable=not accelerator.is_main_process, desc="steps")

    while epoch < max_epochs:
        epoch += 1
        for raw_batch in _iter_raw_batches(
            train_datasets,
            args.max_data_loader_n_workers,
            shuffle_datasets=bool(args.shuffle_datasets),
        ):
            prepared = prepare_batch(raw_batch, device, vae_dtype)
            with accelerator.accumulate(decoder):
                with _amp_context(device, vae_dtype):
                    loss, metrics, recon, target = compute_vae_loss(
                        encoder=encoder,
                        decoder=decoder,
                        batch=prepared,
                        args=args,
                        train_encoder=train_encoder,
                        reference_encoder=reference_encoder,
                        tiling_config=tiling_config,
                    )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    last_recon = recon
                    last_target = target
                    current_loss = metrics["loss"]
                    moving_loss = current_loss if moving_loss is None else moving_loss * 0.95 + current_loss * 0.05
                    progress.update(1)
                    progress.set_postfix(loss=f"{current_loss:.4f}", avg=f"{moving_loss:.4f}")

                    if args.log_every_n_steps > 0 and global_step % args.log_every_n_steps == 0:
                        logger.info(
                            "step=%d epoch=%d loss=%.6f recon=%.6f temporal=%.6f latent_reg=%.6f lr=%.3e",
                            global_step,
                            epoch,
                            metrics["loss"],
                            metrics["recon_loss"],
                            metrics["temporal_loss"],
                            metrics["latent_reg_loss"],
                            scheduler.get_last_lr()[0],
                        )

                    should_save_step = args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0
                    if should_save_step:
                        val_loss = evaluate(
                            args=args,
                            encoder=encoder,
                            decoder=decoder,
                            datasets=val_datasets,
                            device=device,
                            dtype=vae_dtype,
                            tiling_config=tiling_config,
                            max_batches=args.validation_batches,
                        )
                        if val_loss is not None:
                            logger.info("validation loss at step %d: %.6f", global_step, val_loss)
                        ckpt_path = os.path.join(args.output_dir, f"{args.output_name}-step{global_step:08d}.safetensors")
                        save_vae_checkpoint(
                            accelerator=accelerator,
                            args=args,
                            encoder=encoder,
                            decoder=decoder,
                            config_json=config_json,
                            output_path=ckpt_path,
                            global_step=global_step,
                            epoch=epoch,
                            val_loss=val_loss,
                        )
                        if last_recon is not None and last_target is not None:
                            save_preview(
                                args=args,
                                target=last_target,
                                recon=last_recon,
                                output_dir=preview_dir,
                                global_step=global_step,
                            )
                        if args.save_state:
                            state_dir = os.path.join(args.output_dir, f"{args.output_name}-step{global_step:08d}-state")
                            accelerator.save_state(state_dir)
                            save_resume_metadata(state_dir, global_step, 0, epoch)

                    if args.max_train_steps is not None and global_step >= args.max_train_steps:
                        break

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

        if args.save_every_n_epochs > 0 and epoch % args.save_every_n_epochs == 0:
            val_loss = evaluate(
                args=args,
                encoder=encoder,
                decoder=decoder,
                datasets=val_datasets,
                device=device,
                dtype=vae_dtype,
                tiling_config=tiling_config,
                max_batches=args.validation_batches,
            )
            if val_loss is not None:
                logger.info("validation loss at epoch %d: %.6f", epoch, val_loss)
            ckpt_path = os.path.join(args.output_dir, f"{args.output_name}-epoch{epoch:06d}.safetensors")
            save_vae_checkpoint(
                accelerator=accelerator,
                args=args,
                encoder=encoder,
                decoder=decoder,
                config_json=config_json,
                output_path=ckpt_path,
                global_step=global_step,
                epoch=epoch,
                val_loss=val_loss,
            )
            if last_recon is not None and last_target is not None:
                save_preview(args=args, target=last_target, recon=last_recon, output_dir=preview_dir, global_step=global_step)

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    progress.close()
    final_path = os.path.join(args.output_dir, f"{args.output_name}.safetensors")
    val_loss = evaluate(
        args=args,
        encoder=encoder,
        decoder=decoder,
        datasets=val_datasets,
        device=device,
        dtype=vae_dtype,
        tiling_config=tiling_config,
        max_batches=args.validation_batches,
    )
    save_vae_checkpoint(
        accelerator=accelerator,
        args=args,
        encoder=encoder,
        decoder=decoder,
        config_json=config_json,
        output_path=final_path,
        global_step=global_step,
        epoch=epoch,
        val_loss=val_loss,
    )
    if last_recon is not None and last_target is not None:
        save_preview(args=args, target=last_target, recon=last_recon, output_dir=preview_dir, global_step=global_step)

    accelerator.wait_for_everyone()
    logger.info("VAE training complete: %s", final_path)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the LTX-2 video VAE")

    parser.add_argument("--dataset_config", type=str, required=True, help="Dataset TOML with image/video datasets")
    parser.add_argument("--ltx2_checkpoint", type=str, default=None, help="Base LTX-2 checkpoint. Used when --vae is omitted.")
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint. Defaults to --ltx2_checkpoint.")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_name", type=str, default="ltx2_vae")
    parser.add_argument("--train_target", type=str, default="decoder", choices=["decoder", "encoder_decoder"])
    parser.add_argument("--save_format", type=str, default="full", choices=["full", "decoder"],
                        help="full saves frozen/trained encoder plus decoder; decoder saves decoder-only.")

    parser.add_argument("--vae_dtype", type=str, default="bfloat16", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--save_dtype", type=str, default=None, choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--max_train_epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--optimizer_type", type=str, default="adamw", choices=["adamw"])
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--lr_warmup_steps", type=int, default=0)

    parser.add_argument("--reconstruction_loss", type=str, default="charbonnier", choices=["l1", "mse", "charbonnier"])
    parser.add_argument("--reconstruction_loss_weight", type=float, default=1.0)
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--temporal_loss_weight", type=float, default=0.05)
    parser.add_argument("--latent_regularization_weight", type=float, default=0.0,
                        help="Only used with --train_target encoder_decoder; keeps new encoder latents near base encoder latents.")

    parser.add_argument("--decode_timestep", type=float, default=0.05)
    parser.add_argument("--decode_noise_scale", type=float, default=0.025)
    parser.add_argument("--disable_decode_noise", action="store_true",
                        help="Disable decoder latent noise injection for deterministic reconstruction training.")
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="Set CausalConv3d temporal chunk size")
    parser.add_argument("--vae_spatial_tile_size", type=int, default=None,
                        help="No-grad encoder tiling size in pixels. Decoder tiling is not used for training.")
    parser.add_argument("--vae_spatial_tile_overlap", type=int, default=64)
    parser.add_argument("--vae_temporal_tile_size", type=int, default=None,
                        help="No-grad encoder tiling size in frames. Decoder tiling is not used for training.")
    parser.add_argument("--vae_temporal_tile_overlap", type=int, default=24)

    parser.add_argument("--max_data_loader_n_workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--shuffle_datasets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference_downscale", type=int, default=1)
    parser.add_argument("--debug_dataset", action="store_true")

    parser.add_argument("--validation_batches", type=int, default=4)
    parser.add_argument("--save_every_n_steps", type=int, default=0)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--preview_samples", type=int, default=1)
    parser.add_argument("--preview_tile_width", type=int, default=256)
    parser.add_argument("--save_state", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Accelerate state directory to resume from")

    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    if args.vae is None and args.ltx2_checkpoint is None:
        raise ValueError("Specify --vae or --ltx2_checkpoint")
    train(args)


if __name__ == "__main__":
    main()
