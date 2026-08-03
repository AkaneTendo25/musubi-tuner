from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import torch
from accelerate import init_empty_weights
from safetensors import safe_open

from musubi_tuner.minimax_h3.audio_vae import MiniMaxH3AudioDecoderModel, MiniMaxH3AudioEncoderModel
from musubi_tuner.minimax_h3.video_vae import MiniMaxH3VideoDecoderModel, MiniMaxH3VideoEncoderModel

logger = logging.getLogger(__name__)

_VIDEO_FILENAME = "minimax_h3_video_vae_fp16.safetensors"
_AUDIO_FILENAME = "minimax_h3_audio_vae_fp32.safetensors"
_TEXT_FILENAME = "qwen3vl_32b_minimax_h3_bf16.safetensors"


def _resolve_component(source: Path, subdirectory: str, filename: str) -> Path:
    source = Path(source)
    if source.is_file():
        return source
    candidates = (source / subdirectory / filename, source / filename)
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if len(matches) != 1:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"could not resolve {filename}; searched: {searched}")
    return matches[0]


def resolve_video_vae_checkpoint(source: Path) -> Path:
    return _resolve_component(source, "vae", _VIDEO_FILENAME)


def resolve_audio_vae_checkpoint(source: Path) -> Path:
    return _resolve_component(source, "vae", _AUDIO_FILENAME)


def resolve_text_encoder_checkpoint(source: Path) -> Path:
    return _resolve_component(source, "text_encoders", _TEXT_FILENAME)


def _metadata(path: Path, key: str) -> dict:
    with safe_open(path, framework="pt") as handle:
        serialized = (handle.metadata() or {}).get(key)
    if serialized is None:
        raise TypeError(f"{path.name} is missing required safetensors metadata {key!r}")
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} metadata {key!r} must be a JSON object")
    return value


def _video_key(source_key: str) -> str:
    if not source_key.startswith("encoder.down."):
        return source_key
    level, rest = source_key.removeprefix("encoder.down.").split(".", 1)
    rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
    rest = rest.replace("downsample.", "downsamplers.0.", 1)
    return f"encoder.down_blocks.{level}.{rest}"


def _load_subset(
    model: torch.nn.Module,
    path: Path,
    *,
    device: torch.device,
    select_key: Callable[[str], str | None],
    allowed_ignored_prefixes: tuple[str, ...],
    required_dtype: torch.dtype | tuple[torch.dtype, ...],
    allowed_ignored_keys: frozenset[str] = frozenset(),
) -> torch.nn.Module:
    expected = set(model.state_dict())
    state_dict: dict[str, torch.Tensor] = {}
    ignored = 0
    with safe_open(path, framework="pt", device=str(device)) as handle:
        for source_key in handle.keys():  # noqa: SIM118 - safetensors.safe_open is not iterable
            target_key = select_key(source_key)
            if target_key is None:
                if source_key not in allowed_ignored_keys and not source_key.startswith(allowed_ignored_prefixes):
                    raise ValueError(f"{path.name} contains unexpected key {source_key!r}")
                ignored += 1
                continue
            if target_key not in expected:
                raise ValueError(f"{path.name} maps {source_key!r} to unexpected key {target_key!r}")
            if target_key in state_dict:
                raise ValueError(f"{path.name} maps more than one tensor to {target_key!r}")
            tensor = handle.get_tensor(source_key)
            allowed_dtypes = required_dtype if isinstance(required_dtype, tuple) else (required_dtype,)
            if tensor.dtype not in allowed_dtypes:
                names = ", ".join(str(dtype) for dtype in allowed_dtypes)
                raise ValueError(f"{source_key}: expected one of {names}, got {tensor.dtype}")
            state_dict[target_key] = tensor
    missing = sorted(expected - set(state_dict))
    if missing:
        raise ValueError(f"{path.name} is missing {len(missing)} encoder tensor(s), examples: {missing[:5]}")
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"strict component load failed: {info}")
    model.requires_grad_(False).eval()
    logger.info(
        "Loaded %d tensors from %s on %s; validated %d known decoder tensors",
        len(state_dict),
        path,
        device,
        ignored,
    )
    return model


def load_video_vae_encoder(source: Path, device: str | torch.device) -> MiniMaxH3VideoEncoderModel:
    path = resolve_video_vae_checkpoint(source)
    metadata = _metadata(path, "minimax_h3_video_vae")
    if metadata.get("vae_clip_length") != 17 or metadata.get("vae_token_drop") != 3:
        raise ValueError("unsupported MiniMax H3 video VAE temporal geometry")
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3VideoEncoderModel(
            tuple(metadata["latents_mean"]),
            tuple(metadata["latents_std"]),
        )

    def select(source_key: str) -> str | None:
        if source_key.startswith(("encoder.", "quant_conv.")):
            return _video_key(source_key)
        return None

    model = _load_subset(
        model,
        path,
        device=torch.device(device),
        select_key=select,
        allowed_ignored_prefixes=("decoder.", "post_quant_conv."),
        allowed_ignored_keys=frozenset(("latents_mean", "latents_std")),
        required_dtype=(torch.float16, torch.float32),
    )
    model.latents_mean = torch.tensor(metadata["latents_mean"], dtype=torch.float32, device=device)
    model.latents_std = torch.tensor(metadata["latents_std"], dtype=torch.float32, device=device)
    return model


def load_audio_vae_encoder(source: Path, device: str | torch.device) -> MiniMaxH3AudioEncoderModel:
    path = resolve_audio_vae_checkpoint(source)
    metadata = _metadata(path, "minimax_h3_audio_vae")
    kwargs = metadata.get("kwargs", {})
    expected = {
        "encoder_dim": 64,
        "encoder_rates": [2, 4, 4, 5, 5],
        "latent_dim": 2048,
        "vae_latent_channels": 32,
        "sample_rate": 32000,
        "attn_proj": True,
    }
    disagreements = {key: (kwargs.get(key), value) for key, value in expected.items() if kwargs.get(key) != value}
    if disagreements:
        raise ValueError(f"unsupported MiniMax H3 audio VAE config: {disagreements}")
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3AudioEncoderModel(
            tuple(metadata["latents_mean"]),
            tuple(metadata["latents_std"]),
        )

    def select(source_key: str) -> str | None:
        if source_key.startswith(("encoder.", "pre_block.", "mean_proj.", "logs_proj.")):
            return source_key
        return None

    model = _load_subset(
        model,
        path,
        device=torch.device(device),
        select_key=select,
        allowed_ignored_prefixes=("decoder.", "dec_in_proj."),
        allowed_ignored_keys=frozenset(("latents_mean", "latents_std")),
        required_dtype=torch.float32,
    )
    model.latents_mean = torch.tensor(metadata["latents_mean"], dtype=torch.float32, device=device)
    model.latents_std = torch.tensor(metadata["latents_std"], dtype=torch.float32, device=device)
    return model


def load_video_vae_decoder(source: Path, device: str | torch.device) -> MiniMaxH3VideoDecoderModel:
    """Load only the released video VAE decoder for sequential inference."""
    path = resolve_video_vae_checkpoint(source)
    metadata = _metadata(path, "minimax_h3_video_vae")
    if metadata.get("vae_clip_length") != 17 or metadata.get("vae_token_drop") != 3:
        raise ValueError("unsupported MiniMax H3 video VAE temporal geometry")
    # Keep derived, non-persistent rotary buffers materialized. They have no
    # checkpoint key and would otherwise remain meta tensors after an
    # assign-based strict load.
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3VideoDecoderModel(tuple(metadata["latents_mean"]), tuple(metadata["latents_std"]))

    def select(source_key: str) -> str | None:
        if source_key.startswith(("decoder.", "post_quant_conv.")) and source_key != "decoder.mask_token":
            return source_key
        return None

    model = _load_subset(
        model,
        path,
        device=torch.device(device),
        select_key=select,
        allowed_ignored_prefixes=("encoder.", "quant_conv."),
        allowed_ignored_keys=frozenset(("decoder.mask_token", "latents_mean", "latents_std")),
        required_dtype=(torch.float16, torch.float32),
    )
    model.latents_mean = torch.tensor(metadata["latents_mean"], dtype=torch.float32, device=device)
    model.latents_std = torch.tensor(metadata["latents_std"], dtype=torch.float32, device=device)
    return model


def load_audio_vae_decoder(source: Path, device: str | torch.device) -> MiniMaxH3AudioDecoderModel:
    """Load only the released FP32 audio VAE decoder for sequential inference."""
    path = resolve_audio_vae_checkpoint(source)
    metadata = _metadata(path, "minimax_h3_audio_vae")
    kwargs = metadata.get("kwargs", {})
    if kwargs.get("decoder_type") not in (None, "bigvgan") or kwargs.get("sample_rate") != 32_000:
        raise ValueError("unsupported MiniMax H3 audio decoder configuration")
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3AudioDecoderModel(tuple(metadata["latents_mean"]), tuple(metadata["latents_std"]))

    def select(source_key: str) -> str | None:
        if source_key.startswith(("decoder.", "dec_in_proj.")):
            return source_key
        return None

    model = _load_subset(
        model,
        path,
        device=torch.device(device),
        select_key=select,
        allowed_ignored_prefixes=("encoder.", "pre_block.", "mean_proj.", "logs_proj."),
        allowed_ignored_keys=frozenset(("latents_mean", "latents_std")),
        required_dtype=torch.float32,
    )
    model.latents_mean = torch.tensor(metadata["latents_mean"], dtype=torch.float32, device=device)
    model.latents_std = torch.tensor(metadata["latents_std"], dtype=torch.float32, device=device)
    return model


def text_encoder_metadata(source: Path) -> tuple[Path, dict]:
    path = resolve_text_encoder_checkpoint(source)
    metadata = _metadata(path, "minimax_h3_te")
    if metadata != {"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}:
        raise ValueError(f"unsupported MiniMax H3 text encoder metadata: {metadata}")
    return path, metadata
