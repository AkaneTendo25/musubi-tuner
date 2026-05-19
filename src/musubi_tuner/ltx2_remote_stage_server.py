"""Run an experimental LTX-2 remote transformer-stage server.

This process owns one contiguous transformer-block range for remote pipeline
training. A multi-PC run starts one server per remote GPU/PC, then the trainer
connects to all servers through ``--ltx2_remote_stage_specs``.

Minimal three-PC example:

PC-A:
  python -m musubi_tuner.ltx2_remote_stage_server --ltx2_checkpoint /models/ltx.safetensors \
    --split 12 --end 24 --bind 0.0.0.0 --port 17810 --device cuda:0 \
    --dtype bfloat16 --ltx2_mode video --attn_mode sdpa --fp8_scaled \
    --trainable --learning_rate 1e-5 --weight_decay 0.0 \
    --network_module networks.lora_ltx2 --network_dim 4 --network_alpha 4 \
    --network_args lora_target_preset=t2v --prune_non_stage_blocks

PC-B uses ``--split 24 --end 36``. PC-C uses ``--split 36 --end 48``.
The trainer then uses:
  --ltx2_remote_stage_specs "pc-a:17810:12:24;pc-b:17810:24:36;pc-c:17810:36:48"

Small remote PC recipe:
  Use ``--block_only_load --prune_non_stage_blocks --stage_only_device_placement
  --load_device cpu`` on remote middle/suffix stages. In that mode the server
  loads only checkpoint tensors for its owned ``transformer_blocks[start:end]``.
  Shared LTX tensors are intentionally left on meta and never consume remote CPU
  RAM or VRAM. This is the mode intended for machines like 16 GB RAM + 8 GB
  VRAM.

Memory ownership rule:
  The local first stage owns shared input/output modules and local prefix
  blocks. Remote middle/suffix stages own only their transformer block ranges.
  Do not use ``--block_only_load`` for a standalone full-model process because
  shared modules are not materialized there.

The current trainer coordinates every remote hop. Multi-PC specs split memory
ownership across machines, but many small stages add extra round trips and local
autograd boundaries. For memory-focused experiments, start with one large remote
suffix such as ``12:48``; add more remote stages only after measuring the
tradeoff.

Security: the protocol is unauthenticated pickle-over-TCP. Use trusted LAN/VPN
or tunnels only.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm

from musubi_tuner.ltx2_model_loading import load_ltx2_model
from musubi_tuner.ltx2_remote_stage import (
    DEFAULT_REMOTE_STAGE_PORT,
    LTX2RemoteStageServer,
    REMOTE_STAGE_TRAINABLE_SCOPE_AUTO,
    REMOTE_STAGE_TRAINABLE_SCOPES,
    prune_ltx2_blocks_to_range,
)
from musubi_tuner.modules.nf4_optimization_utils import DEFAULT_NF4_BLOCK_SIZE

logger = logging.getLogger(__name__)

_REMOTE_BLOCK_KEY_RE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.")


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(mapping)}")
    return mapping[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ltx2_checkpoint", "--checkpoint", dest="ltx2_checkpoint", required=True)
    parser.add_argument("--bind", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_REMOTE_STAGE_PORT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--load_device",
        type=str,
        default=None,
        help=(
            "Device used while loading the checkpoint. Defaults to CPU when --prune_non_stage_blocks is set, "
            "otherwise --device. Use CPU on small remote GPUs so only owned blocks are moved to VRAM."
        ),
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--split", type=int, required=True, help="First transformer block index owned by this server")
    parser.add_argument("--end", type=int, default=-1, help="Exclusive final transformer block index owned by this server")
    parser.add_argument("--trainable", action="store_true", help="Enable server-owned optimizer updates for this block range")
    parser.add_argument(
        "--trainable_scope",
        type=str,
        default=REMOTE_STAGE_TRAINABLE_SCOPE_AUTO,
        choices=REMOTE_STAGE_TRAINABLE_SCOPES,
        help=(
            "Remote trainable parameter scope when --trainable is set. "
            "'lora' requires --network_module and trains only that adapter; "
            "'blocks' trains owned base block weights; 'auto' chooses lora when --network_module is set, otherwise blocks."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=None, help="AdamW learning rate when --trainable is set")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay when --trainable is set")
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Clip remote stage gradients before optimizer.step")
    parser.add_argument(
        "--prune_non_stage_blocks",
        action="store_true",
        help="Replace non-owned transformer blocks with placeholders after load to reduce per-server VRAM.",
    )
    parser.add_argument(
        "--stage_only_device_placement",
        action="store_true",
        help=(
            "Keep non-stage/shared modules on --load_device and move only transformer blocks [--split, --end) "
            "to --device. This is the intended mode for small remote GPUs; use with --load_device cpu."
        ),
    )
    parser.add_argument(
        "--full_model_device_placement",
        action="store_true",
        help=(
            "Move the entire loaded model to --device. This is mostly for compatibility/debugging and can OOM "
            "small cards because shared non-block LTX tensors are large."
        ),
    )
    parser.add_argument(
        "--block_only_load",
        action="store_true",
        help=(
            "Remote-stage memory-saving mode: load only checkpoint tensors for transformer blocks [--split, --end). "
            "Shared input/output modules stay on meta and are never materialized on CPU/GPU. Use this for "
            "middle/suffix remote stages on memory-limited machines. The first/last trainer stage must "
            "own the shared modules that turn latents/text into block inputs and block outputs back into model output."
        ),
    )
    parser.add_argument("--network_module", type=str, default=None, help="Optional LoRA network module owned by this remote stage")
    parser.add_argument("--network_dim", type=int, default=None, help="Remote LoRA rank")
    parser.add_argument("--network_alpha", type=float, default=None, help="Remote LoRA alpha")
    parser.add_argument("--network_dropout", type=float, default=None, help="Remote LoRA dropout")
    parser.add_argument("--network_args", type=str, nargs="*", default=None, help="Remote LoRA network args, key=value")
    parser.add_argument("--network_weights", type=str, default=None, help="Optional remote LoRA weights to load")
    parser.add_argument("--network_lr", type=float, default=None, help="Remote LoRA learning rate. Defaults to --learning_rate.")
    parser.add_argument("--ltx2_mode", "--ltx_mode", dest="ltx_mode", choices=["video", "av", "audio", "v", "a", "va"], default="video")
    parser.add_argument("--ltx2_audio_only_model", action="store_true")
    parser.add_argument("--attn_mode", choices=["torch", "sdpa", "flash", "flash3", "xformers"], default="torch")
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--fp8_w8a8", action="store_true")
    parser.add_argument("--w8a8_mode", type=str, default="int8")
    parser.add_argument("--fp8_upcast", action="store_true")
    parser.add_argument("--fp8_upcast_stochastic", action="store_true")
    parser.add_argument("--fp8_upcast_seed", type=int, default=0)
    parser.add_argument("--fp8_keep_blocks", type=str, default=None)
    parser.add_argument("--nf4_base", action="store_true")
    parser.add_argument("--nf4_block_size", type=int, default=DEFAULT_NF4_BLOCK_SIZE)
    parser.add_argument("--split_attn_target", type=str, default=None)
    parser.add_argument("--split_attn_mode", type=str, default=None)
    parser.add_argument("--split_attn_chunk_size", type=int, default=0)
    parser.add_argument("--ffn_chunk_target", type=str, default=None)
    parser.add_argument("--ffn_chunk_size", type=int, default=0)
    parser.add_argument("--quantize_device", type=str, default=None)
    parser.add_argument("--int8_block_size", type=int, default=256, help="Block size for remote-stage low-bit codecs")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def _parse_network_args(raw_args: list[str] | None) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    for raw_arg in raw_args or []:
        if "=" not in raw_arg:
            raise ValueError(f"Invalid --network_args entry (expected key=value): {raw_arg}")
        key, value = raw_arg.split("=", 1)
        kwargs[key] = value
    return kwargs


def _load_network_weights(path: str) -> dict[str, torch.Tensor]:
    if os.path.splitext(path)[1] == ".safetensors":
        return load_file(path)
    return torch.load(path, map_location="cpu")


def _apply_remote_config_overrides(
    config: dict,
    *,
    model_path: str,
    attn_mode: str,
    split_attn_target: str | None,
    split_attn_mode: str | None,
    split_attn_chunk_size: int,
    ffn_chunk_target: str | None,
    ffn_chunk_size: int,
) -> dict:
    config = dict(config)
    transformer_config = dict(config.get("transformer", {}))
    config["transformer"] = transformer_config

    attn_mode = (attn_mode or "torch").lower()
    attn_type = None
    if attn_mode in {"xformers", "xformers-attn"}:
        attn_type = "xformers"
    elif attn_mode in {"flash3", "flash_attention_3"}:
        attn_type = "flash_attention_3"
    elif attn_mode in {"flash", "flash_attention_2"}:
        attn_type = "flash_attention_2"
    elif attn_mode in {"torch", "sdpa"}:
        attn_type = "pytorch"
    if attn_type is not None:
        transformer_config["attention_type"] = attn_type
    if split_attn_target is not None:
        transformer_config["split_attn_target"] = split_attn_target
    if split_attn_mode is not None:
        transformer_config["split_attn_mode"] = split_attn_mode
    if split_attn_chunk_size is not None:
        transformer_config["split_attn_chunk_size"] = int(split_attn_chunk_size)
    if ffn_chunk_target is not None:
        transformer_config["ffn_chunk_target"] = ffn_chunk_target
    if ffn_chunk_size is not None:
        transformer_config["ffn_chunk_size"] = int(ffn_chunk_size)

    if not transformer_config.get("apply_gated_attention", False):
        with safe_open(model_path, framework="pt") as handle:
            if any("to_gate_logits" in key for key in handle.keys()):
                transformer_config["apply_gated_attention"] = True
                logger.info("Auto-detected gated attention from checkpoint keys")
    return config


def _resolve_ltx2_configurator(*, audio_video: bool, audio_only_model: bool):
    from musubi_tuner.ltx_2.model.transformer.model_configurator import (
        LTXAudioOnlyModelConfigurator,
        LTXModelConfigurator,
        LTXVideoOnlyModelConfigurator,
    )

    if audio_only_model and not audio_video:
        raise ValueError("audio_only_model=True requires audio_video=True")
    if audio_only_model:
        return LTXAudioOnlyModelConfigurator, "audio-only"
    if audio_video:
        return LTXModelConfigurator, "audio-video"
    return LTXVideoOnlyModelConfigurator, "video-only"


def _owned_block_key(
    key: str,
    *,
    start: int,
    end: int,
    renamer,
) -> tuple[str, int] | None:
    renamed_key = renamer.apply_to_key(key)
    normalized_key = renamed_key if renamed_key is not None else key
    match = _REMOTE_BLOCK_KEY_RE.search(normalized_key)
    if match is None:
        return None
    block_index = int(match.group(1))
    if start <= block_index < end:
        return normalized_key, block_index
    return None


def _load_ltx2_remote_block_only_model(
    *,
    model_path: str,
    dtype: torch.dtype,
    start: int,
    end: int | None,
    audio_video: bool,
    audio_only_model: bool,
    attn_mode: str,
    split_attn_target: str | None,
    split_attn_mode: str | None,
    split_attn_chunk_size: int,
    ffn_chunk_target: str | None,
    ffn_chunk_size: int,
) -> torch.nn.Module:
    """Load only the transformer blocks owned by this remote process.

    This is intentionally narrower than ``load_ltx2_model``. A pure remote
    pipeline stage receives already-built TransformerArgs, runs a contiguous
    block range, and returns TransformerArgs. It does not need patchify,
    caption projection, output projection, or any other shared LTX tensors.

    Memory contract:
      * CPU RAM contains only owned block tensors while loading.
      * GPU VRAM contains only owned block tensors after
        ``_move_remote_stage_blocks_to_device`` runs.
      * Non-owned blocks are replaced by Identity placeholders so global block
        indices stay stable for LoRA naming and protocol validation.

    Do not use this mode for the local first stage or a standalone full model
    forward; shared modules are deliberately left unmaterialized.
    """

    from musubi_tuner.ltx_2.loader.sft_loader import SafetensorsModelStateDictLoader
    from musubi_tuner.ltx_2.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
    from musubi_tuner.networks.lora_ltx2 import LTX2Wrapper
    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

    model_path = os.fspath(model_path)
    loader = SafetensorsModelStateDictLoader()
    config = loader.metadata(model_path)
    config = _apply_remote_config_overrides(
        config,
        model_path=model_path,
        attn_mode=attn_mode,
        split_attn_target=split_attn_target,
        split_attn_mode=split_attn_mode,
        split_attn_chunk_size=split_attn_chunk_size,
        ffn_chunk_target=ffn_chunk_target,
        ffn_chunk_size=ffn_chunk_size,
    )
    configurator, variant = _resolve_ltx2_configurator(audio_video=audio_video, audio_only_model=audio_only_model)
    logger.info("LTX-2 remote block-only model variant: %s", variant)

    with torch.device("meta"):
        base_model = configurator.from_config(config)
    blocks = getattr(base_model, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("remote block-only load requires transformer_blocks")
    total_blocks = len(blocks)
    resolved_end = total_blocks if end is None else int(end)
    if start < 0 or resolved_end <= start or resolved_end > total_blocks:
        raise ValueError(f"invalid remote block-only range {start}:{resolved_end} for {total_blocks} blocks")
    expected_model_keys = set(base_model.state_dict().keys())

    sd: dict[str, torch.Tensor] = {}
    loaded_blocks: set[int] = set()
    skipped_unexpected = 0
    total_bytes = 0
    with MemoryEfficientSafeOpen(model_path) as handle:
        keys = []
        for key in handle.keys():
            owned = _owned_block_key(
                key,
                start=start,
                end=resolved_end,
                renamer=LTXV_MODEL_COMFY_RENAMING_MAP,
            )
            if owned is not None:
                if owned[0] in expected_model_keys:
                    keys.append((key, owned[0], owned[1]))
                else:
                    skipped_unexpected += 1
        if not keys:
            raise RuntimeError(f"no checkpoint tensors found for remote block range {start}:{resolved_end}")
        for raw_key, normalized_key, block_index in tqdm(
            keys,
            desc=f"Loading remote LTX-2 blocks {start}:{resolved_end}",
            leave=False,
        ):
            tensor = handle.get_tensor(raw_key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            sd[normalized_key] = tensor
            loaded_blocks.add(block_index)
            total_bytes += int(tensor.numel() * tensor.element_size())

    incompatible = base_model.load_state_dict(sd, strict=False, assign=True)
    del sd
    gc.collect()

    missing_owned = [
        key
        for key in incompatible.missing_keys
        if _REMOTE_BLOCK_KEY_RE.search(key)
        and start <= int(_REMOTE_BLOCK_KEY_RE.search(key).group(1)) < resolved_end
    ]
    if missing_owned:
        preview = ", ".join(missing_owned[:8])
        raise RuntimeError(
            f"remote block-only load left {len(missing_owned)} owned block tensors on meta; first missing: {preview}"
        )
    if incompatible.unexpected_keys:
        logger.warning("Remote block-only load ignored %d unexpected keys", len(incompatible.unexpected_keys))

    replaced = 0
    for idx in range(total_blocks):
        if start <= idx < resolved_end:
            continue
        blocks[idx] = torch.nn.Identity()
        replaced += 1

    logger.info(
        "Remote block-only load complete: blocks=%d:%d loaded_blocks=%s tensors=%d bytes=%.2f GiB replaced=%d",
        start,
        resolved_end,
        ",".join(str(idx) for idx in sorted(loaded_blocks)),
        len(keys),
        total_bytes / (1024**3),
        replaced,
    )
    if skipped_unexpected:
        logger.info("Remote block-only load skipped %d checkpoint keys not used by %s model", skipped_unexpected, variant)
    return LTX2Wrapper(base_model, patch_size=1)


def _build_remote_network(args: argparse.Namespace, model: torch.nn.Module, device: torch.device):
    if not args.network_module:
        return None, None

    try:
        network_module = importlib.import_module(args.network_module)
    except ModuleNotFoundError:
        if str(args.network_module).startswith("networks."):
            network_module = importlib.import_module(f"musubi_tuner.{args.network_module}")
        else:
            sys.path.append(os.path.dirname(__file__))
            network_module = importlib.import_module(args.network_module)
    net_kwargs = _parse_network_args(args.network_args)
    if hasattr(network_module, "create_arch_network"):
        network = network_module.create_arch_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            None,
            None,
            model,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )
    else:
        network = network_module.create_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            None,
            None,
            model,
            **net_kwargs,
        )
    if network is None:
        raise RuntimeError(f"network module {args.network_module!r} returned None")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    if args.network_weights is not None:
        info = network.load_state_dict(_load_network_weights(args.network_weights), strict=False)
        logger.info("Loaded remote network weights from %s: %s", args.network_weights, info)
    network.to(device)
    network.train()
    network.requires_grad_(bool(args.trainable))
    lr = args.network_lr if args.network_lr is not None else args.learning_rate
    optimizer_params = None
    if args.trainable and lr is not None and hasattr(network, "prepare_optimizer_params"):
        optimizer_params, _ = network.prepare_optimizer_params(
            unet_lr=float(lr),
            audio_lr=None,
            lr_args=None,
        )
    logger.info(
        "Remote LoRA network enabled: module=%s params=%d lr=%s",
        args.network_module,
        sum(param.numel() for param in network.parameters()),
        lr,
    )
    return network, optimizer_params


def _move_remote_stage_blocks_to_device(
    model: torch.nn.Module,
    *,
    start: int,
    end: int | None,
    device: torch.device,
) -> None:
    base_model = model.model if hasattr(model, "model") else model
    blocks = getattr(base_model, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("remote stage model must expose transformer_blocks")
    end = len(blocks) if end is None else int(end)
    for block in blocks[int(start) : end]:
        block.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Moved remote-owned transformer blocks %d:%d to %s", int(start), end, device)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    if args.trainable_scope != REMOTE_STAGE_TRAINABLE_SCOPE_AUTO and not args.trainable:
        raise ValueError("--trainable_scope requires --trainable")

    mode = str(args.ltx_mode).lower()
    audio_video = mode in {"av", "audio", "a", "va"}
    audio_only_model = bool(args.ltx2_audio_only_model or mode in {"audio", "a"})
    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)
    stage_only_device_placement = bool(args.stage_only_device_placement or args.prune_non_stage_blocks)
    if bool(args.full_model_device_placement):
        stage_only_device_placement = False
    load_device = torch.device(args.load_device or ("cpu" if stage_only_device_placement else args.device))
    if stage_only_device_placement and load_device.type == "cuda":
        raise ValueError(
            "--stage_only_device_placement requires a non-CUDA --load_device, normally --load_device cpu. "
            "Loading shared/non-stage LTX tensors on CUDA can OOM small remote GPUs before pruning."
        )
    end_index = None if int(args.end) < 0 else int(args.end)
    stage_load_range = (
        (int(args.split), end_index)
        if bool(args.prune_non_stage_blocks or args.network_module)
        else None
    )
    if args.block_only_load:
        if not stage_only_device_placement:
            raise ValueError("--block_only_load requires --stage_only_device_placement or --prune_non_stage_blocks")
        if bool(args.fp8_scaled or args.nf4_base or args.fp8_w8a8):
            raise ValueError(
                "--block_only_load currently supports ordinary fp16/bf16/fp32 block weights only. "
                "Use a non-quantized checkpoint path first; fp8/nf4 block-only loading needs separate validation."
            )
        if args.network_module and not args.prune_non_stage_blocks:
            raise ValueError("--block_only_load with a remote LoRA/network requires --prune_non_stage_blocks")

    logger.info(
        "Loading remote LTX-2 stage model on %s; execution device=%s; stage_only_device_placement=%s",
        load_device,
        device,
        stage_only_device_placement,
    )
    if args.block_only_load:
        model = _load_ltx2_remote_block_only_model(
            model_path=args.ltx2_checkpoint,
            dtype=dtype,
            start=int(args.split),
            end=end_index,
            audio_video=audio_video,
            audio_only_model=audio_only_model,
            attn_mode=args.attn_mode,
            split_attn_target=args.split_attn_target,
            split_attn_mode=args.split_attn_mode,
            split_attn_chunk_size=int(args.split_attn_chunk_size or 0),
            ffn_chunk_target=args.ffn_chunk_target,
            ffn_chunk_size=int(args.ffn_chunk_size or 0),
        )
    else:
        model = load_ltx2_model(
            model_path=args.ltx2_checkpoint,
            device=load_device,
            load_device=load_device,
            torch_dtype=dtype,
            attn_mode=args.attn_mode,
            audio_video=audio_video,
            audio_only_model=audio_only_model,
            split_attn_target=args.split_attn_target,
            split_attn_mode=args.split_attn_mode,
            split_attn_chunk_size=int(args.split_attn_chunk_size or 0),
            ffn_chunk_target=args.ffn_chunk_target,
            ffn_chunk_size=int(args.ffn_chunk_size or 0),
            fp8_scaled=bool(args.fp8_scaled),
            fp8_w8a8=bool(args.fp8_w8a8),
            w8a8_mode=args.w8a8_mode,
            fp8_upcast=bool(args.fp8_upcast),
            fp8_upcast_stochastic=bool(args.fp8_upcast_stochastic),
            fp8_upcast_seed=int(args.fp8_upcast_seed),
            fp8_keep_blocks=args.fp8_keep_blocks,
            nf4_base=bool(args.nf4_base),
            nf4_block_size=int(args.nf4_block_size),
            quantize_device=args.quantize_device,
            transformer_block_load_range=stage_load_range,
        )
    model.train()
    model.requires_grad_(False)
    if args.prune_non_stage_blocks or args.network_module:
        prune_ltx2_blocks_to_range(
            model,
            keep_start=int(args.split),
            keep_end=end_index if end_index is not None else len(model.transformer_blocks),
            label="server-pre-network",
        )
    network, network_optimizer_params = _build_remote_network(args, model, device)
    if stage_only_device_placement:
        _move_remote_stage_blocks_to_device(model, start=int(args.split), end=end_index, device=device)
    elif load_device != device:
        model.to(device)

    server = LTX2RemoteStageServer(
        model=model,
        split_index=int(args.split),
        end_index=end_index,
        device=device,
        bind=args.bind,
        port=int(args.port),
        int8_block_size=int(args.int8_block_size),
        trainable=bool(args.trainable),
        learning_rate=args.network_lr if args.network_lr is not None else args.learning_rate,
        weight_decay=float(args.weight_decay),
        max_grad_norm=float(args.max_grad_norm),
        autocast_dtype=dtype,
        trainable_network=network,
        network_optimizer_params=network_optimizer_params,
        prune_non_stage_blocks=bool(args.prune_non_stage_blocks),
        trainable_scope=args.trainable_scope,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
