import argparse
import hashlib
from functools import lru_cache
from io import BytesIO
from typing import Any, Callable, Optional
import logging
import safetensors.torch
import torch

logger = logging.getLogger(__name__)


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def dtype_to_str(dtype: torch.dtype) -> str:
    # get name of the dtype
    dtype_name = str(dtype).split(".")[-1]
    return dtype_name


def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    Convert a string to a torch.dtype

    Args:
        s: string representation of the dtype
        default_dtype: default dtype to return if s is None

    Returns:
        torch.dtype: the corresponding torch.dtype

    Raises:
        ValueError: if the dtype is not supported

    Examples:
        >>> str_to_dtype("float32")
        torch.float32
        >>> str_to_dtype("fp32")
        torch.float32
        >>> str_to_dtype("float16")
        torch.float16
        >>> str_to_dtype("fp16")
        torch.float16
        >>> str_to_dtype("bfloat16")
        torch.bfloat16
        >>> str_to_dtype("bf16")
        torch.bfloat16
        >>> str_to_dtype("fp8")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fn")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fnuz")
        torch.float8_e4m3fnuz
        >>> str_to_dtype("fp8_e5m2")
        torch.float8_e5m2
        >>> str_to_dtype("fp8_e5m2fnuz")
        torch.float8_e5m2fnuz
    """
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")


@lru_cache(maxsize=1)
def _known_dtype_strs() -> tuple[str, ...]:
    """All dtype strings dtype_to_str() can emit, longest first.

    Sorting by length keeps multi-underscore names (e.g. "float8_e4m3fn") ahead of their
    prefixes ("float8...") so the longest matching suffix wins.
    """
    names = set()
    for attr in dir(torch):
        try:
            obj = getattr(torch, attr)
        except Exception:
            continue
        if isinstance(obj, torch.dtype):
            names.add(dtype_to_str(obj))
    return tuple(sorted(names, key=len, reverse=True))


def remove_dtype_suffix(name: str) -> str:
    """Remove a trailing ``_<dtype>`` suffix (as written by dtype_to_str) from a cache key.

    Robust to dtype names that contain underscores such as ``float8_e4m3fn``; a plain
    ``rsplit("_", 1)`` would only drop the final ``fn`` segment. Returns ``name`` unchanged
    if it does not end with a known dtype suffix.
    """
    for dtype_str in _known_dtype_strs():
        suffix = "_" + dtype_str
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [to_device(elem, device) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x: Any) -> Any:
    """
    Recursively moves torch.Tensor objects (and containers thereof) to CPU.

    Args:
        x: A torch.Tensor, or a (possibly nested) list, tuple, or dict containing tensors.

    Returns:
        The same structure as x, with all torch.Tensor objects moved to CPU.
        Non-tensor objects are returned unchanged.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_cpu(elem) for elem in x)
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


def create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    """
    Create a wrapper function that offloads inputs to CPU before calling the original function
    and moves outputs back to the specified device.

    Args:
        func: The original function to wrap.
        device: The device to move outputs back to.

    Returns:
        A wrapped function that offloads inputs to CPU and moves outputs back to the specified device.
    """

    def wrapper(orig_func: Callable) -> Callable:
        def custom_forward(*inputs):
            nonlocal device, orig_func
            cuda_inputs = to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            return to_cpu(outputs)

        return custom_forward

    return wrapper(func)


def disable_linear_from_compile(module: torch.nn.Module):
    """Monkey-patch to disable torch.compile for all Linear layers (if the class name ends with 'Linear') in the given module."""
    for sub_module in module.modules():
        # if isinstance(sub_module, torch.nn.Linear):
        if sub_module.__class__.__name__.endswith("Linear"):
            if not hasattr(sub_module, "_forward_before_disable_compile"):
                sub_module._forward_before_disable_compile = sub_module.forward
                sub_module._eager_forward = torch._dynamo.disable()(sub_module.forward)
            sub_module.forward = sub_module._eager_forward  # override forward to disable compile


def compile_dynamic_arg(args: argparse.Namespace) -> bool | None:
    value = getattr(args, "compile_dynamic", None)
    if value is None or isinstance(value, bool):
        return value
    return {"true": True, "false": False, "auto": None}[value.lower()]


def unwrap_compile_module(module: torch.nn.Module) -> torch.nn.Module:
    """Unwrap common distributed/compile wrappers without an Accelerator dependency."""
    unwrapped = module
    seen: set[int] = set()
    for _ in range(8):
        if id(unwrapped) in seen:
            break
        seen.add(id(unwrapped))
        candidate = getattr(unwrapped, "module", None)
        if isinstance(candidate, torch.nn.Module) and candidate is not unwrapped:
            unwrapped = candidate
            continue
        candidate = getattr(unwrapped, "_orig_mod", None)
        if isinstance(candidate, torch.nn.Module) and candidate is not unwrapped:
            unwrapped = candidate
            continue
        break
    return unwrapped


def resolve_compile_block_lists(
    module: torch.nn.Module,
    block_attr_names: tuple[str, ...] = ("transformer_blocks",),
) -> list[torch.nn.ModuleList | list[torch.nn.Module]]:
    roots = [unwrap_compile_module(module)]
    wrapped = getattr(roots[0], "model", None)
    if isinstance(wrapped, torch.nn.Module) and wrapped is not roots[0]:
        roots.append(wrapped)

    resolved: list[torch.nn.ModuleList | list[torch.nn.Module]] = []
    seen: set[int] = set()
    for root in roots:
        for path in block_attr_names:
            value: Any = root
            for part in path.split("."):
                value = getattr(value, part, None)
                if value is None:
                    break
            if isinstance(value, (torch.nn.ModuleList, list)) and id(value) not in seen:
                resolved.append(value)
                seen.add(id(value))
    return resolved


def _collect_compile_targets(target_blocks):
    targets = []
    for blocks in target_blocks:
        for index, block in enumerate(blocks):
            if not isinstance(block, torch.nn.Module):
                continue
            if hasattr(block, "_hf_hook"):
                logger.info("Skipping compile target %s because it has an HF offload hook", index)
                continue
            targets.append((blocks, index, block))
    return targets


def _configure_compile_cache(args: argparse.Namespace, target_count: int) -> None:
    limit = getattr(args, "compile_cache_size_limit", None)
    if getattr(args, "compile_auto_cache_size_limit", False) and target_count:
        automatic = target_count * 2
        limit = (
            max(int(getattr(torch._dynamo.config, "cache_size_limit", 0) or 0), automatic)
            if limit is None
            else max(limit, automatic)
        )
    if limit is not None:
        torch._dynamo.config.cache_size_limit = limit
        logger.info("Set torch._dynamo.config.cache_size_limit to %s", limit)


def _parse_compile_config_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    for value_type in (int, float):
        try:
            return value_type(raw)
        except ValueError:
            pass
    return raw


def _apply_inductor_config(args: argparse.Namespace) -> None:
    overrides = getattr(args, "inductor_config", None)
    if not overrides:
        return
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    for token in overrides:
        if "=" not in token:
            raise ValueError(f"--inductor_config expects KEY=VALUE, got {token!r}")
        key, raw = token.split("=", 1)
        parts = key.strip().split(".")
        root = inductor_config if hasattr(inductor_config, parts[0]) else dynamo_config
        target = root
        for part in parts[:-1]:
            target = getattr(target, part, None)
            if target is None:
                break
        if target is None or not hasattr(target, parts[-1]):
            logger.warning("--inductor_config: skipping unknown key %s", key)
            continue
        setattr(target, parts[-1], _parse_compile_config_value(raw))


def compile_transformer(
    args: argparse.Namespace,
    transformer: torch.nn.Module,
    target_blocks: list[torch.nn.ModuleList | list[torch.nn.Module]],
    disable_linear: bool,
) -> torch.nn.Module:
    targets = _collect_compile_targets(target_blocks)
    if not targets:
        raise RuntimeError("torch.compile was requested, but no target blocks were found")
    if disable_linear:
        for _, _, block in targets:
            disable_linear_from_compile(block)

    _apply_inductor_config(args)
    _configure_compile_cache(args, len(targets))
    dynamic = compile_dynamic_arg(args)
    logger.info(
        "Compiling %d transformer blocks: backend=%s mode=%s dynamic=%s fullgraph=%s",
        len(targets),
        args.compile_backend,
        args.compile_mode,
        dynamic,
        args.compile_fullgraph,
    )
    compiled = []
    try:
        for blocks, index, block in targets:
            blocks[index] = torch.compile(
                block,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=dynamic,
                fullgraph=args.compile_fullgraph,
            )
            compiled.append((blocks, index, block))
    except Exception as error:
        for blocks, index, block in reversed(compiled):
            blocks[index] = block
        if getattr(args, "compile_fallback_to_eager", False):
            logger.warning("torch.compile failed; restored eager blocks: %s", error)
            return transformer
        raise
    return transformer
