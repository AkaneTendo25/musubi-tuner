import json
import math
import os
import re
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer
from tqdm import tqdm
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from musubi_tuner.cosmos3.cosmos_framework.model.vfm.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler

from musubi_tuner.cosmos3.cosmos_framework.data.vfm.sequence_packing import ModalityData, PackedSequence, SequencePlan, pack_input_sequence
from musubi_tuner.cosmos3.cosmos_framework.model.vfm.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from musubi_tuner.cosmos3.cosmos_framework.model.vfm.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
from musubi_tuner.cosmos3.cosmos_framework.model.vfm.utils.data_and_condition import GenerationDataClean
from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."

DEFAULT_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
DEFAULT_IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."
DEFAULT_VIDEO_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
DEFAULT_INVERSE_DURATION_TEMPLATE = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
DEFAULT_INVERSE_IMAGE_RESOLUTION_TEMPLATE = "This image is not of {height}x{width} resolution."
DEFAULT_INVERSE_VIDEO_RESOLUTION_TEMPLATE = "This video is not of {height}x{width} resolution."
DEFAULT_NEGATIVE_METADATA_MODE = "same"
DEFAULT_NEGATIVE_PROMPT_PATH = Path(__file__).resolve().parent / "defaults" / "neg_prompts.json"

COSMOS3_FP8_TARGET_KEYS = ["language_model.model.layers"]
COSMOS3_FP8_EXCLUDE_KEYS = [
    "norm",
    "embed_tokens",
    "lm_head",
    "time_embedder",
    "rotary_emb",
    # Keep small modality bridge projections in bf16, matching the conservative
    # patch/head exclusion pattern used by other Musubi video models.
    "vae2llm",
    "llm2vae",
    "sound2llm",
    "llm2sound",
]

_ROOT_INDEX = "model.safetensors.index.json"
_TRANSFORMER_INDEX = "diffusion_pytorch_model.safetensors.index.json"


def load_default_negative_prompt() -> str:
    if not DEFAULT_NEGATIVE_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing Cosmos3 negative prompt defaults: {DEFAULT_NEGATIVE_PROMPT_PATH}")
    return json.dumps(json.loads(DEFAULT_NEGATIVE_PROMPT_PATH.read_text(encoding="utf-8")))

_HF_EXPORT_DROP_KEY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:feature_extractor|image_processor|scheduler|sound_tokenizer|text_encoder|tokenizer|vae)\."),
)
_HF_EXPORT_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^transformer\."), ""),
    (re.compile(r"^vision_encoder\."), ""),
    (re.compile(r"^model\.net\."), ""),
    (re.compile(r"^action_proj_in\."), "action2llm."),
    (re.compile(r"^action_proj_out\."), "llm2action."),
    (re.compile(r"^audio_proj_in\."), "sound2llm."),
    (re.compile(r"^audio_proj_out\."), "llm2sound."),
    (re.compile(r"^audio_modality_embed$"), "sound_modality_embed"),
    (re.compile(r"^proj_in\."), "vae2llm."),
    (re.compile(r"^proj_out\."), "llm2vae."),
    (re.compile(r"^time_embedder\.linear_1\."), "time_embedder.mlp.0."),
    (re.compile(r"^time_embedder\.linear_2\."), "time_embedder.mlp.2."),
    (re.compile(r"\.self_attn\.to_q\."), ".self_attn.q_proj."),
    (re.compile(r"\.self_attn\.to_k\."), ".self_attn.k_proj."),
    (re.compile(r"\.self_attn\.to_v\."), ".self_attn.v_proj."),
    (re.compile(r"\.self_attn\.to_out\."), ".self_attn.o_proj."),
    (re.compile(r"\.self_attn\.norm_q\."), ".self_attn.q_norm."),
    (re.compile(r"\.self_attn\.norm_k\."), ".self_attn.k_norm."),
    (re.compile(r"\.self_attn\.add_q_proj\."), ".self_attn.q_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_k_proj\."), ".self_attn.k_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_v_proj\."), ".self_attn.v_proj_moe_gen."),
    (re.compile(r"\.self_attn\.to_add_out\."), ".self_attn.o_proj_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_q\."), ".self_attn.q_norm_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_k\."), ".self_attn.k_norm_moe_gen."),
    (re.compile(r"^model\.lm_head\."), "language_model.lm_head."),
    (re.compile(r"^lm_head\."), "language_model.lm_head."),
    (re.compile(r"^model\.visual\."), "language_model.visual."),
    (re.compile(r"^visual\."), "language_model.visual."),
    (
        re.compile(r"^(blocks\.|deepstack_merger_list\.|merger\.|patch_embed\.|pos_embed\.)(.*)$"),
        r"language_model.visual.\1\2",
    ),
    (
        re.compile(
            r"^language_model\.(?!model\.|lm_head\.|visual\.)(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"
        ),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^model\.(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
)
_NATIVE_NET_KEY_PREFIXES: tuple[str, ...] = (
    "action2llm.",
    "action_pos_embed.",
    "language_model.",
    "latent_pos_embed.",
    "llm2action.",
    "llm2sound.",
    "llm2vae.",
    "sound2llm.",
    "time_embedder.",
    "vae2llm.",
)
_NATIVE_NET_KEYS: frozenset[str] = frozenset(
    {
        "action_modality_embed",
        "latent_pos_embed",
        "sound_modality_embed",
    }
)


def _component_path(model_path: str, subfolder: Optional[str]) -> str:
    if subfolder is None or subfolder == "":
        return model_path
    path = os.path.join(model_path, subfolder)
    return path if os.path.isdir(path) else model_path


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(model_path: str, tokenizer_subfolder: Optional[str] = "text_tokenizer"):
    tokenizer_path = _component_path(model_path, tokenizer_subfolder)
    logger.info(f"Loading Cosmos3 tokenizer from {tokenizer_path}")
    return AutoTokenizer.from_pretrained(tokenizer_path)


def _find_vae_weight(model_path: str, vae_subfolder: Optional[str]) -> str:
    path = _component_path(model_path, vae_subfolder)
    if os.path.isfile(path):
        return path

    candidates = []
    for suffix in (".pth", ".pt", ".safetensors"):
        candidates.extend(str(p) for p in Path(path).glob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(
            f"No native Cosmos3/Wan2.2 VAE weight file found under {path}. "
            "Pass --vae with a Wan2.2_VAE.pth path or a directory containing compatible VAE weights."
        )
    if len(candidates) > 1:
        preferred = [p for p in candidates if Path(p).name in {"Wan2.2_VAE.pth", "diffusion_pytorch_model.safetensors"}]
        if len(preferred) == 1:
            return preferred[0]
        raise ValueError(f"Expected one VAE weight file under {path}, found {len(candidates)}: {candidates}")
    return candidates[0]


def load_vae(model_path: str, vae_subfolder: Optional[str] = "vae", dtype: Optional[torch.dtype] = None, device="cpu"):
    from musubi_tuner.cosmos3.cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import WanVAE as CosmosWan2VAE

    vae_path = _find_vae_weight(model_path, vae_subfolder)
    dtype = torch.bfloat16 if dtype is None else dtype
    logger.info(f"Loading native Cosmos3 Wan2.2 VAE from {vae_path}")
    vae = CosmosWan2VAE(vae_pth=vae_path, dtype=dtype, device=torch.device(device), is_amp=(torch.device(device).type == "cuda"))
    vae.eval()
    return vae


def _find_sound_tokenizer_paths(model_path: str, sound_tokenizer_subfolder: Optional[str]) -> tuple[str, str]:
    path = _component_path(model_path, sound_tokenizer_subfolder)
    if os.path.isfile(path):
        weight_path = path
        config_path = str(Path(path).with_suffix(".json"))
        return weight_path, config_path if os.path.exists(config_path) else ""

    config_path = os.path.join(path, "config.json")
    candidates = []
    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            candidates.append(candidate)
    for suffix in (".ckpt", ".pth", ".pt", ".safetensors"):
        candidates.extend(str(p) for p in Path(path).glob(f"*{suffix}") if str(p) not in candidates)
    if not candidates:
        raise FileNotFoundError(f"No Cosmos3 AVAE sound tokenizer weight file found under {path}.")
    if len(candidates) > 1:
        preferred = [p for p in candidates if Path(p).name == "diffusion_pytorch_model.safetensors"]
        if len(preferred) == 1:
            candidates = preferred
        else:
            raise ValueError(f"Expected one sound tokenizer weight file under {path}, found {len(candidates)}: {candidates}")
    return candidates[0], config_path if os.path.exists(config_path) else ""


def load_sound_tokenizer(
    model_path: str,
    sound_tokenizer_subfolder: Optional[str] = "sound_tokenizer",
    dtype: Optional[torch.dtype] = None,
    device="cpu",
):
    from musubi_tuner.cosmos3.cosmos_framework.model.vfm.tokenizers.audio.avae import AVAEModel

    weight_path, config_path = _find_sound_tokenizer_paths(model_path, sound_tokenizer_subfolder)
    config = _read_json(config_path) if config_path else {}
    dtype = torch.bfloat16 if dtype is None else dtype
    sample_rate = int(config.get("sampling_rate", config.get("sample_rate", 48000)))
    audio_channels = 2 if bool(config.get("stereo", True)) else int(config.get("audio_channels", 1))
    io_channels = int(config.get("vocoder_input_dim", config.get("io_channels", 64)))
    hop_size = int(config.get("hop_size", 1920))
    normalize_volume = bool(config.get("normalize_volume", True))
    logger.info(f"Loading native Cosmos3 AVAE sound tokenizer from {weight_path}")
    tokenizer = AVAEModel(
        vae_pth=weight_path,
        config_path=config_path,
        sample_rate=sample_rate,
        audio_channels=audio_channels,
        io_channels=io_channels,
        hop_size=hop_size,
        dtype=dtype,
        device=device,
        normalize_volume=normalize_volume,
    )
    tokenizer.model.eval()
    return tokenizer


def encode_audio_to_latents(sound_tokenizer: Any, audio: torch.Tensor) -> torch.Tensor:
    """Encode B,C,T audio in [-1,1] to AVAE latents B,C,T."""
    return sound_tokenizer.encode(audio, force_pad=True)


def decode_sound_latents_to_audio(sound_tokenizer: Any, latents: torch.Tensor) -> torch.Tensor:
    return sound_tokenizer.decode(latents).to(torch.float32).clamp_(-1.0, 1.0)


def _load_root_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if config_path.exists():
        return _read_json(config_path)
    return {}


def _load_transformer_config(model_path: Path, transformer_subfolder: Optional[str]) -> dict[str, Any]:
    transformer_path = Path(_component_path(str(model_path), transformer_subfolder))
    config_path = transformer_path / "config.json"
    if config_path.exists():
        return _read_json(config_path)
    root_config = model_path / "config.json"
    if root_config.exists():
        return _read_json(root_config)
    raise FileNotFoundError(f"Cosmos3 transformer config not found under {model_path}")


def _build_text_config_dict(root_config: dict[str, Any], transformer_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(root_config.get("text_config"), dict):
        text_config = dict(root_config["text_config"])
    else:
        keys = (
            "attention_bias",
            "attention_dropout",
            "bos_token_id",
            "eos_token_id",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "initializer_range",
            "intermediate_size",
            "max_position_embeddings",
            "model_type",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_scaling",
            "rope_theta",
            "use_cache",
            "vocab_size",
        )
        text_config = {key: transformer_config[key] for key in keys if key in transformer_config}
    text_config.setdefault("model_type", "qwen3_vl_text")
    text_config.setdefault("dtype", transformer_config.get("dtype", "bfloat16"))
    return text_config


def _build_mot_config(root_config: dict[str, Any], transformer_config: dict[str, Any]) -> Qwen3VLMoTConfig:
    config_dict: dict[str, Any] = {
        "text_config": _build_text_config_dict(root_config, transformer_config),
        "tie_word_embeddings": bool(root_config.get("tie_word_embeddings", False)),
    }
    if isinstance(root_config.get("vision_config"), dict):
        config_dict["vision_config"] = root_config["vision_config"]
    for token_key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        if token_key in root_config:
            config_dict[token_key] = root_config[token_key]

    return Qwen3VLMoTConfig(
        config_dict=config_dict,
        qk_norm_for_text=bool(transformer_config.get("qk_norm_for_text", True)),
        qk_norm_for_diffusion=bool(transformer_config.get("qk_norm_for_diffusion", True)),
        include_visual=False,
    )


def _build_network_config(root_config: dict[str, Any], transformer_config: dict[str, Any], vlm_config) -> Cosmos3VFMNetworkConfig:
    model_config = root_config.get("model", {}).get("config", {}) if isinstance(root_config.get("model"), dict) else {}
    diffusion_config = model_config.get("diffusion_expert_config", {}) if isinstance(model_config, dict) else {}

    latent_patch_size = int(transformer_config.get("latent_patch_size", diffusion_config.get("patch_spatial", 2)))
    max_latent_side = int(diffusion_config.get("max_vae_latent_side_after_patchify", 20))
    state_t = int(model_config.get("state_t", 300)) if isinstance(model_config, dict) else 300

    return Cosmos3VFMNetworkConfig(
        vlm_config=vlm_config,
        latent_patch_size=latent_patch_size,
        latent_downsample_factor=int(model_config.get("latent_downsample_factor", 16)) if isinstance(model_config, dict) else 16,
        latent_channel_size=int(transformer_config.get("latent_channel", model_config.get("state_ch", 48))),
        max_latent_h=max_latent_side,
        max_latent_w=max_latent_side,
        max_latent_t=state_t,
        rope_h_extrapolation_ratio=float(diffusion_config.get("rope_h_extrapolation_ratio", 1.0)),
        rope_w_extrapolation_ratio=float(diffusion_config.get("rope_w_extrapolation_ratio", 1.0)),
        rope_t_extrapolation_ratio=float(diffusion_config.get("rope_t_extrapolation_ratio", 1.0)),
        enable_fps_modulation=bool(transformer_config.get("enable_fps_modulation", True)),
        base_fps=float(transformer_config.get("base_fps", diffusion_config.get("base_fps", 24))),
        vision_gen=True,
        action_gen=bool(transformer_config.get("action_gen", model_config.get("action_gen", True))),
        sound_gen=bool(transformer_config.get("sound_gen", model_config.get("sound_gen", True))),
        position_embedding_type=transformer_config.get("position_embedding_type", "unified_3d_mrope"),
        joint_attn_implementation=transformer_config.get("joint_attn_implementation", "two_way"),
        timestep_scale=float(transformer_config.get("timestep_scale", 0.001)),
        action_dim=int(transformer_config.get("max_action_dim", transformer_config.get("action_dim", 64))),
        num_embodiment_domains=int(transformer_config.get("num_embodiment_domains", 32)),
        temporal_compression_factor_vision=int(model_config.get("tokenizer", {}).get("temporal_compression_factor", 4))
        if isinstance(model_config.get("tokenizer"), dict)
        else 4,
        natten_parameter_list=model_config.get("natten_parameter_list") if isinstance(model_config, dict) else None,
        video_temporal_causal=bool(transformer_config.get("video_temporal_causal", False)),
        sound_dim=int(transformer_config["sound_dim"]) if transformer_config.get("sound_dim") is not None else None,
        sound_latent_fps=int(transformer_config.get("sound_latent_fps", 25)),
    )


def build_native_transformer(model_path: str, transformer_subfolder: Optional[str] = "transformer") -> Cosmos3VFMNetwork:
    model_root = Path(model_path)
    root_config = _load_root_config(model_root)
    transformer_config = _load_transformer_config(model_root, transformer_subfolder)
    mot_config = _build_mot_config(root_config, transformer_config)
    language_model = Qwen3VLTextForCausalLM(mot_config)
    network_config = _build_network_config(root_config, transformer_config, language_model.config)
    model = Cosmos3VFMNetwork(language_model=language_model, config=network_config)
    model.pad_for_cuda_graphs = False
    return model


def _read_safetensors_index(index_path: Path) -> dict[str, str]:
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a safetensors weight_map.")
    return {str(key): str(value) for key, value in weight_map.items()}


def _find_transformer_weight_map(model_path: Path, transformer_subfolder: Optional[str]) -> tuple[Path, dict[str, str]]:
    transformer_path = Path(_component_path(str(model_path), transformer_subfolder))
    transformer_index = transformer_path / _TRANSFORMER_INDEX
    if transformer_index.exists():
        return transformer_path, _read_safetensors_index(transformer_index)

    single = transformer_path / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return transformer_path, {key: single.name for key in safe_open(str(single), framework="pt").keys()}

    root_index = model_path / _ROOT_INDEX
    if root_index.exists():
        return model_path, _read_safetensors_index(root_index)

    raise FileNotFoundError(f"Cosmos3 transformer safetensors index not found under {model_path}")


def _resolve_transformer_shard_path(model_root: Path, weight_root: Path, rel_path: str) -> Path:
    shard_path = weight_root / rel_path
    if shard_path.exists():
        return shard_path

    fallback_path = model_root / rel_path
    if fallback_path.exists():
        return fallback_path

    return shard_path


def _validate_transformer_checkpoint_files(model_root: Path, weight_root: Path, weight_map: dict[str, str]) -> None:
    expected_shards = sorted({rel_path for rel_path in weight_map.values() if _is_transformer_weight_path(weight_root, rel_path)})
    missing: list[str] = []
    partial: list[str] = []

    for rel_path in expected_shards:
        shard_path = _resolve_transformer_shard_path(model_root, weight_root, rel_path)
        if shard_path.exists():
            continue

        part_path = shard_path.with_name(f"{shard_path.name}.part")
        fallback_part_path = (model_root / rel_path).with_name(f"{Path(rel_path).name}.part")
        if part_path.exists():
            partial.append(str(part_path))
        elif fallback_part_path.exists():
            partial.append(str(fallback_part_path))
        else:
            missing.append(str(shard_path))

    if missing or partial:
        parts = ["Cosmos3 transformer checkpoint is incomplete."]
        if partial:
            parts.append(f"Partial shards: {partial[:20]}")
        if missing:
            parts.append(f"Missing shards: {missing[:20]}")
        raise FileNotFoundError(" ".join(parts))


def validate_transformer_checkpoint_files(model_path: str, transformer_subfolder: Optional[str] = "transformer") -> None:
    model_root = Path(model_path)
    weight_root, weight_map = _find_transformer_weight_map(model_root, transformer_subfolder)
    _validate_transformer_checkpoint_files(model_root, weight_root, weight_map)


def _should_drop_hf_export_key(name: str) -> bool:
    return any(pattern.search(name) is not None for pattern in _HF_EXPORT_DROP_KEY_RES)


def _is_transformer_weight_path(weight_root: Path, rel_path: str) -> bool:
    rel_path = rel_path.replace("\\", "/")
    if weight_root.name == "transformer":
        return True
    return rel_path.startswith("transformer/") or rel_path.startswith("vision_encoder/")


def _hf_export_to_native_key(name: str, rel_path: str, weight_root: Path) -> str | None:
    if not _is_transformer_weight_path(weight_root, rel_path) or _should_drop_hf_export_key(name):
        return None

    for pattern, replacement in _HF_EXPORT_KEY_MAPPING_RES:
        name = pattern.sub(replacement, name)
    if _should_drop_hf_export_key(name):
        return None
    if name in _NATIVE_NET_KEYS or name.startswith(_NATIVE_NET_KEY_PREFIXES):
        return name
    return None


def _assign_tensor_to_module(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype],
    device: str | torch.device,
) -> None:
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    tensor = tensor.to(device=device)

    module_name, tensor_name = name.rsplit(".", 1) if "." in name else ("", name)
    module = model.get_submodule(module_name) if module_name else model

    old_param = module._parameters.get(tensor_name)
    if old_param is not None:
        module._parameters[tensor_name] = nn.Parameter(tensor, requires_grad=old_param.requires_grad)
        return

    if tensor_name in module._buffers:
        module._buffers[tensor_name] = tensor
        return

    raise KeyError(f"Could not assign tensor {name}; target module has no parameter or buffer named {tensor_name}.")


def _materialize_remaining_meta_tensors(model: nn.Module, device: str | torch.device) -> list[str]:
    materialized = []
    for module_prefix, module in model.named_modules():
        for name, param in list(module._parameters.items()):
            if param is not None and param.device.type == "meta":
                full_name = f"{module_prefix}.{name}" if module_prefix else name
                empty = torch.empty_like(param, device=device)
                module._parameters[name] = nn.Parameter(empty, requires_grad=param.requires_grad)
                materialized.append(full_name)
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                full_name = f"{module_prefix}.{name}" if module_prefix else name
                module._buffers[name] = torch.empty_like(buffer, device=device)
                materialized.append(full_name)
    return materialized


def load_native_transformer_weights(
    model: Cosmos3VFMNetwork,
    model_path: str,
    transformer_subfolder: Optional[str],
    dtype: Optional[torch.dtype],
    loading_device: str | torch.device,
) -> None:
    model_root = Path(model_path)
    weight_root, weight_map = _find_transformer_weight_map(model_root, transformer_subfolder)
    _validate_transformer_checkpoint_files(model_root, weight_root, weight_map)
    target_state = model.state_dict()
    target_keys = set(target_state.keys())
    loaded_keys: set[str] = set()
    skipped = 0

    files_to_keys: dict[str, list[str]] = {}
    for checkpoint_key, rel_path in weight_map.items():
        native_key = _hf_export_to_native_key(checkpoint_key, rel_path, weight_root)
        if native_key is None or native_key not in target_keys:
            skipped += 1
            continue
        files_to_keys.setdefault(rel_path, []).append(checkpoint_key)

    logger.info(f"Loading native Cosmos3 transformer weights from {weight_root} ({len(files_to_keys)} shard files)")
    for rel_path, checkpoint_keys in sorted(files_to_keys.items()):
        shard_path = _resolve_transformer_shard_path(model_root, weight_root, rel_path)

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for checkpoint_key in checkpoint_keys:
                native_key = _hf_export_to_native_key(checkpoint_key, rel_path, weight_root)
                if native_key is None or native_key not in target_keys:
                    continue
                tensor = f.get_tensor(checkpoint_key)
                expected_shape = tuple(target_state[native_key].shape)
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for {native_key} loaded from {checkpoint_key}: "
                        f"checkpoint {tuple(tensor.shape)} != model {expected_shape}"
                    )
                _assign_tensor_to_module(model, native_key, tensor, dtype=dtype, device=loading_device)
                loaded_keys.add(native_key)

    materialized = _materialize_remaining_meta_tensors(model, loading_device)
    missing = sorted(target_keys - loaded_keys)
    if missing:
        materialized_set = set(materialized)
        missing_not_materialized = [key for key in missing if key not in materialized_set]
        if missing_not_materialized:
            raise RuntimeError(f"Missing {len(missing_not_materialized)} Cosmos3 native weights. First keys: {missing_not_materialized[:20]}")
        logger.warning(f"Materialized {len(materialized)} Cosmos3 tensors not present in checkpoint. First keys: {materialized[:20]}")
    logger.info(f"Loaded {len(loaded_keys)} Cosmos3 tensors; skipped {skipped} non-native or unused checkpoint keys.")


def load_transformer(
    model_path: str,
    transformer_subfolder: Optional[str],
    dtype: Optional[torch.dtype],
    loading_device: str | torch.device,
):
    validate_transformer_checkpoint_files(model_path, transformer_subfolder)
    logger.info(f"Building native Cosmos3 transformer from {model_path}")
    with torch.device("meta"):
        model = build_native_transformer(model_path, transformer_subfolder)
    load_native_transformer_weights(model, model_path, transformer_subfolder, dtype, loading_device)
    _reinitialize_rotary_buffers(model, loading_device)
    patch_transformer_for_training(model)
    model.eval()
    return model


def _reinitialize_rotary_buffers(model: nn.Module, device: str | torch.device | None = None) -> None:
    """Restore non-persistent RoPE buffers after meta-device construction.

    The Cosmos3 checkpoint does not carry ``rotary_emb.inv_freq``. Rebuild it
    after loading weights when the model was constructed on ``meta``.
    """
    rotary_emb = getattr(getattr(model.language_model, "model", None), "rotary_emb", None)
    if rotary_emb is None or not hasattr(rotary_emb, "init_weights"):
        return
    buffer_device = torch.device(device) if device is not None else None
    rotary_emb.init_weights(buffer_device=buffer_device)
    logger.info("Reinitialized Cosmos3 rotary embedding buffers on %s.", buffer_device)


def set_attention_backend(model: nn.Module, attn_mode: str):
    if attn_mode not in {"torch", "flash", "flash3", "xformers", "sageattn"}:
        return
    logger.info(f"Cosmos3 native attention backend is selected by local dispatch; requested Musubi mode={attn_mode}.")


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    try:
        return model.language_model.model.layers
    except AttributeError as e:
        raise AttributeError("Expected native Cosmos3 model with language_model.model.layers.") from e


def apply_scaled_fp8(model: nn.Module, calc_device: torch.device, move_to_device: bool = False):
    logger.info(
        "Applying scaled FP8 optimization to native Cosmos3 transformer weights "
        f"(calc_device={calc_device}, move_to_device={move_to_device})."
    )
    state_dict = model.state_dict()
    state_dict = optimize_state_dict_with_fp8(
        state_dict,
        calc_device=calc_device,
        target_layer_keys=COSMOS3_FP8_TARGET_KEYS,
        exclude_layer_keys=COSMOS3_FP8_EXCLUDE_KEYS,
        move_to_device=move_to_device,
    )
    apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Applied scaled FP8 to Cosmos3 transformer: {info}")


def patch_transformer_for_training(model: nn.Module):
    model.blocks_to_swap = getattr(model, "blocks_to_swap", None)
    model.activation_cpu_offloading = False
    model.gradient_checkpointing = False
    text_model = model.language_model.model
    text_model.blocks_to_swap = getattr(text_model, "blocks_to_swap", None)

    def cleanup_block_swap_state(self):
        old_offloaders = [getattr(self, "offloader", None), getattr(self.language_model.model, "offloader", None)]
        seen_offloaders = set()
        for offloader in old_offloaders:
            if offloader is None or id(offloader) in seen_offloaders:
                continue
            seen_offloaders.add(id(offloader))
            for handle in getattr(offloader, "remove_handles", []):
                handle.remove()
        for handle in getattr(self, "_cosmos3_block_swap_handles", []):
            handle.remove()
        self._cosmos3_block_swap_handles = []

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        layers = get_transformer_layers(self)
        for layer in layers:
            if hasattr(layer, "enable_gradient_checkpointing"):
                layer.enable_gradient_checkpointing(activation_cpu_offloading)
            else:
                layer.gradient_checkpointing = True
                layer.activation_cpu_offloading = activation_cpu_offloading
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        print(f"Cosmos3 native: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        for layer in get_transformer_layers(self):
            if hasattr(layer, "disable_gradient_checkpointing"):
                layer.disable_gradient_checkpointing()
            else:
                layer.gradient_checkpointing = False
                layer.activation_cpu_offloading = False
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_block_swap(
        self,
        num_blocks: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
    ):
        layers = get_transformer_layers(self)
        self.blocks_to_swap = num_blocks
        self._cosmos3_block_swap_device = device
        self._cosmos3_block_swap_supports_backward = supports_backward
        self._cosmos3_block_swap_use_pinned_memory = use_pinned_memory
        self.num_blocks = len(layers)
        assert self.blocks_to_swap <= self.num_blocks - 1, (
            f"Cannot swap more than {self.num_blocks - 1} Cosmos3 blocks. Requested {self.blocks_to_swap}."
        )

        cleanup_block_swap_state(self)

        offloader = ModelOffloader(
            "cosmos3-native-block",
            layers,
            self.num_blocks,
            self.blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        # Cosmos3 decoder blocks return FactoredSequencePack dictionaries; the
        # decoder loop uses tensor-level hooks for backward block swapping.
        for handle in getattr(offloader, "remove_handles", []):
            handle.remove()
        offloader.remove_handles = []
        self.offloader = offloader
        text_model = self.language_model.model
        text_model.blocks_to_swap = self.blocks_to_swap
        text_model.num_blocks = self.num_blocks
        text_model.offloader = offloader

        print(
            f"Cosmos3 native: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {self.num_blocks}. "
            f"Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.language_model.model.offloader = self.offloader
            self.prepare_block_swap_before_forward()
            print("Cosmos3 native: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.language_model.model.offloader = self.offloader
            self.prepare_block_swap_before_forward()
            print("Cosmos3 native: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            saved_layers = self.language_model.model.layers
            self.language_model.model.layers = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.language_model.model.layers = saved_layers

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(get_transformer_layers(self))
        self.language_model.model.blocks_to_swap = self.blocks_to_swap
        self.language_model.model.offloader = self.offloader

    model.enable_gradient_checkpointing = types.MethodType(enable_gradient_checkpointing, model)
    model.disable_gradient_checkpointing = types.MethodType(disable_gradient_checkpointing, model)
    model.enable_block_swap = types.MethodType(enable_block_swap, model)
    model.switch_block_swap_for_inference = types.MethodType(switch_block_swap_for_inference, model)
    model.switch_block_swap_for_training = types.MethodType(switch_block_swap_for_training, model)
    model.move_to_device_except_swap_blocks = types.MethodType(move_to_device_except_swap_blocks, model)
    model.prepare_block_swap_before_forward = types.MethodType(prepare_block_swap_before_forward, model)


def encode_video_to_latents(vae: Any, video: torch.Tensor) -> torch.Tensor:
    """Encode B,C,F,H,W video in [-1,1] to normalized native Cosmos3/Wan2.2 latents."""
    return vae.encode(video)


def decode_latents_to_video(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    video = vae.decode(latents).to(torch.float32)
    return (video / 2 + 0.5).clamp_(0, 1)


def normalize_sample_dimensions(
    width: int,
    height: int,
    frame_count: int,
    vae_scale_factor_temporal: int = 4,
) -> tuple[int, int, int]:
    if vae_scale_factor_temporal < 1:
        raise ValueError("vae_scale_factor_temporal must be positive.")
    width = max(16, (int(width) // 16) * 16)
    height = max(16, (int(height) // 16) * 16)
    frame_count = max(1, int(frame_count))
    frame_count = (frame_count - 1) // vae_scale_factor_temporal * vae_scale_factor_temporal + 1
    return width, height, frame_count


def make_flow_timesteps(
    sample_steps: int,
    device: torch.device | str,
    discrete_flow_shift: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sample_steps <= 0:
        raise ValueError("sample_steps must be positive.")
    if discrete_flow_shift <= 0:
        raise ValueError("discrete_flow_shift must be positive.")

    sigmas = torch.linspace(1.0, 0.0, sample_steps + 1, device=device, dtype=torch.float32)
    if discrete_flow_shift != 1.0:
        sigmas = discrete_flow_shift * sigmas / (1.0 + (discrete_flow_shift - 1.0) * sigmas)
    timesteps = sigmas * 1000.0
    return timesteps, sigmas


def load_scheduler(
    model_path: str,
    subfolder: str = "scheduler",
    flow_shift: float = 10.0,
) -> UniPCMultistepScheduler:
    scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder=subfolder)
    return UniPCMultistepScheduler.from_config(scheduler.config, flow_shift=float(flow_shift))


def _clone_scheduler(scheduler: UniPCMultistepScheduler | FlowUniPCMultistepScheduler):
    if isinstance(scheduler, UniPCMultistepScheduler):
        return UniPCMultistepScheduler.from_config(scheduler.config)
    return FlowUniPCMultistepScheduler(
        num_train_timesteps=int(getattr(scheduler.config, "num_train_timesteps", 1000)),
        shift=float(getattr(scheduler.config, "shift", 1.0)),
        use_dynamic_shifting=bool(getattr(scheduler.config, "use_dynamic_shifting", False)),
    )


def encode_image_to_condition_latent(
    vae: Any,
    image_path: str,
    width: int,
    height: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    array = np.asarray(image).copy()
    image_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    image_tensor = image_tensor.to(device=vae.device, dtype=vae.dtype)
    image_tensor = image_tensor / 127.5 - 1.0
    with torch.no_grad():
        condition_latent = encode_video_to_latents(vae, image_tensor)
    return condition_latent[:, :, :1].to(device=device, dtype=dtype)


def generate_latents(
    transformer: nn.Module,
    tokenizer,
    scheduler: UniPCMultistepScheduler | FlowUniPCMultistepScheduler,
    prompt: str,
    negative_prompt: Optional[str],
    width: int,
    height: int,
    frame_count: int,
    fps: float,
    sample_steps: int,
    flow_shift: float,
    guidance_scale: Optional[float],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    vae_scale_factor_temporal: int = 4,
    image_condition_latent: Optional[torch.Tensor] = None,
    use_system_prompt: bool = False,
    add_resolution_template: bool = False,
    add_duration_template: bool = False,
    negative_metadata_mode: str = DEFAULT_NEGATIVE_METADATA_MODE,
    sound_latent_length: Optional[int] = None,
    sound_fps: float = 25.0,
    progress: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if scheduler is None:
        raise ValueError("Cosmos3 sampling requires a scheduler instance.")

    cond_ids, uncond_ids = tokenize_prompt(
        tokenizer,
        prompt,
        negative_prompt,
        frame_count,
        height,
        width,
        fps,
        use_system_prompt=use_system_prompt,
        add_resolution_template=add_resolution_template,
        add_duration_template=add_duration_template,
        negative_metadata_mode=negative_metadata_mode,
    )

    latent_t = max(1, (frame_count + vae_scale_factor_temporal - 1) // vae_scale_factor_temporal)
    latent_h = height // 16
    latent_w = width // 16
    latents = torch.randn(
        (1, transformer.config.latent_channel_size, latent_t, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    has_image_condition = image_condition_latent is not None
    if has_image_condition:
        expected_shape = (1, transformer.config.latent_channel_size, 1, latent_h, latent_w)
        if tuple(image_condition_latent.shape) != expected_shape:
            raise ValueError(
                f"image condition latent shape {tuple(image_condition_latent.shape)} does not match expected {expected_shape}"
            )
        latents[:, :, :1] = image_condition_latent

    sound_latents = None
    if sound_latent_length is not None:
        sound_dim = int(getattr(transformer.config, "sound_dim", 64))
        sound_latents = torch.randn(
            (1, sound_dim, sound_latent_length),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    cond_ids = torch.tensor(cond_ids, dtype=torch.long, device=device)
    uncond_ids = torch.tensor(uncond_ids, dtype=torch.long, device=device)
    vision_scheduler = _clone_scheduler(scheduler)
    sound_scheduler = _clone_scheduler(scheduler) if sound_latents is not None else None
    if isinstance(vision_scheduler, FlowUniPCMultistepScheduler):
        vision_scheduler.set_timesteps(sample_steps, device=device, shift=float(flow_shift))
    else:
        vision_scheduler = UniPCMultistepScheduler.from_config(vision_scheduler.config, flow_shift=float(flow_shift))
        vision_scheduler.set_timesteps(sample_steps, device=device)
    if sound_scheduler is not None:
        if isinstance(sound_scheduler, FlowUniPCMultistepScheduler):
            sound_scheduler.set_timesteps(sample_steps, device=device, shift=float(flow_shift))
        else:
            sound_scheduler = UniPCMultistepScheduler.from_config(sound_scheduler.config, flow_shift=float(flow_shift))
            sound_scheduler.set_timesteps(sample_steps, device=device)
    use_cfg = guidance_scale is not None and guidance_scale != 1.0

    def normalize_vision_prediction(prediction: torch.Tensor) -> torch.Tensor:
        if prediction.dim() == latents.dim() - 1:
            prediction = prediction.unsqueeze(0)
        if prediction.dim() != latents.dim():
            raise ValueError(
                f"Cosmos3 vision prediction rank {prediction.dim()} does not match latent rank {latents.dim()} "
                f"(prediction shape={tuple(prediction.shape)}, latent shape={tuple(latents.shape)})"
            )
        return prediction

    step_iter = range(sample_steps)
    if progress:
        step_iter = tqdm(step_iter, desc="Cosmos3 sampling")

    for step in step_iter:
        t = vision_scheduler.timesteps[step]
        model_input = vision_scheduler.scale_model_input(latents, t)
        pred = run_transformer_for_sample(
            transformer,
            input_ids=cond_ids,
            vision_tokens=model_input,
            timestep=t,
            has_image_condition=has_image_condition,
            fps=fps,
            device=device,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            sound_tokens=sound_latents[0] if sound_latents is not None else None,
            sound_fps=sound_fps,
            return_dict=sound_latents is not None,
        )
        sound_pred = None
        if sound_latents is not None:
            assert isinstance(pred, dict)
            sound_pred = pred["preds_sound"][0].unsqueeze(0)
            pred = normalize_vision_prediction(pred["preds_vision"][0])
        else:
            pred = normalize_vision_prediction(pred)
        if use_cfg:
            uncond_pred = run_transformer_for_sample(
                transformer,
                input_ids=uncond_ids,
                vision_tokens=model_input,
                timestep=t,
                has_image_condition=has_image_condition,
                fps=fps,
                device=device,
                vae_scale_factor_temporal=vae_scale_factor_temporal,
                sound_tokens=sound_latents[0] if sound_latents is not None else None,
                sound_fps=sound_fps,
                return_dict=sound_latents is not None,
            )
            if sound_latents is not None:
                assert isinstance(uncond_pred, dict)
                uncond_sound_pred = uncond_pred["preds_sound"][0].unsqueeze(0)
                uncond_pred = normalize_vision_prediction(uncond_pred["preds_vision"][0])
                assert sound_pred is not None
                sound_pred = uncond_sound_pred + guidance_scale * (sound_pred - uncond_sound_pred)
            else:
                uncond_pred = normalize_vision_prediction(uncond_pred)
            pred = uncond_pred + guidance_scale * (pred - uncond_pred)

        if has_image_condition:
            pred = pred.clone()
            pred[:, :, :1] = 0
        latents = vision_scheduler.step(pred, t, latents, return_dict=False)[0]
        if has_image_condition:
            latents[:, :, :1] = image_condition_latent
        if sound_latents is not None:
            assert sound_pred is not None
            sound_latents = (
                sound_scheduler.step(sound_pred, t, sound_latents, return_dict=False)[0]
                if sound_scheduler is not None
                else sound_latents
            )

    if sound_latents is not None:
        return latents, sound_latents
    return latents


def generate_video(
    transformer: nn.Module,
    vae: Any,
    tokenizer,
    scheduler: UniPCMultistepScheduler | FlowUniPCMultistepScheduler,
    prompt: str,
    negative_prompt: Optional[str],
    width: int,
    height: int,
    frame_count: int,
    fps: float,
    sample_steps: int,
    flow_shift: float,
    guidance_scale: Optional[float],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    discrete_flow_shift: float = 1.0,
    vae_scale_factor_temporal: int = 4,
    image_path: Optional[str] = None,
    use_system_prompt: bool = False,
    add_resolution_template: bool = False,
    add_duration_template: bool = False,
    negative_metadata_mode: str = DEFAULT_NEGATIVE_METADATA_MODE,
    progress: bool = False,
) -> torch.Tensor:
    image_condition_latent = None
    if image_path is not None:
        image_condition_latent = encode_image_to_condition_latent(vae, image_path, width, height, device, dtype)

    latents = generate_latents(
        transformer,
        tokenizer,
        scheduler,
        prompt,
        negative_prompt,
        width,
        height,
        frame_count,
        fps,
        sample_steps,
        flow_shift,
        guidance_scale,
        generator,
        device,
        dtype,
        vae_scale_factor_temporal=vae_scale_factor_temporal,
        image_condition_latent=image_condition_latent,
        use_system_prompt=use_system_prompt,
        add_resolution_template=add_resolution_template,
        add_duration_template=add_duration_template,
        negative_metadata_mode=negative_metadata_mode,
        progress=progress,
    )
    return decode_latents_to_video(vae, latents.detach().to(vae.device, dtype=torch.float32))


def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: int | float,
    use_float_positions: bool = False,
) -> tuple[torch.Tensor, int | float]:
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
) -> tuple[torch.Tensor, int | float]:
    fps_modulation_enabled = fps is not None and grid_t > 1

    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / temporal_compression_factor
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = frame_indices / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    return mrope_ids, math.ceil(mrope_ids.max().item()) + 1


def tokenize_prompt(
    tokenizer,
    prompt: str,
    negative_prompt: Optional[str],
    num_frames: int,
    height: int,
    width: int,
    fps: float,
    use_system_prompt: bool = True,
    add_resolution_template: bool = True,
    add_duration_template: bool = True,
    negative_metadata_mode: str = DEFAULT_NEGATIVE_METADATA_MODE,
) -> tuple[list[int], list[int]]:
    is_image = num_frames == 1
    negative_prompt = "" if negative_prompt is None else negative_prompt
    negative_metadata_mode = negative_metadata_mode.lower()
    if negative_metadata_mode not in {"same", "inverse", "none"}:
        raise ValueError("--negative_metadata_mode must be one of: same, inverse, none")

    resolution_template = DEFAULT_IMAGE_RESOLUTION_TEMPLATE if is_image else DEFAULT_VIDEO_RESOLUTION_TEMPLATE
    inverse_resolution_template = (
        DEFAULT_INVERSE_IMAGE_RESOLUTION_TEMPLATE if is_image else DEFAULT_INVERSE_VIDEO_RESOLUTION_TEMPLATE
    )

    def append_sentence(base: str, addition: str) -> str:
        base = base.rstrip(".")
        return f"{base}. {addition}" if base else addition

    def format_json_prompt_with_metadata(text: str) -> Optional[str]:
        try:
            prompt_obj = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(prompt_obj, dict):
            return None

        if not is_image and add_duration_template:
            duration_seconds = int(num_frames / fps) if fps > 0 else 0
            prompt_obj["duration"] = f"{duration_seconds}s"
            prompt_obj["fps"] = float(fps)
        if add_resolution_template:
            prompt_obj["resolution"] = {"H": int(height), "W": int(width)}
            divisor = math.gcd(int(width), int(height))
            if divisor > 0:
                prompt_obj["aspect_ratio"] = f"{int(width) // divisor},{int(height) // divisor}"
        return json.dumps(prompt_obj)

    def apply_templates(text: str, is_negative: bool = False) -> str:
        text = text.strip()
        if not is_negative:
            json_prompt = format_json_prompt_with_metadata(text)
            if json_prompt is not None:
                return json_prompt

        if is_negative and negative_metadata_mode == "none":
            return text
        use_inverse_metadata = is_negative and negative_metadata_mode == "inverse"

        if not is_image and add_duration_template:
            template = DEFAULT_INVERSE_DURATION_TEMPLATE if use_inverse_metadata else DEFAULT_DURATION_TEMPLATE
            text = append_sentence(text, template.format(duration=num_frames / fps, fps=fps))
        if add_resolution_template:
            template = inverse_resolution_template if use_inverse_metadata else resolution_template
            text = append_sentence(text, template.format(height=height, width=width))
        return text

    def tokenize(text: str):
        conversations = []
        if use_system_prompt:
            system_prompt = SYSTEM_PROMPT_IMAGE if is_image else SYSTEM_PROMPT_VIDEO
            conversations.append({"role": "system", "content": system_prompt})
        conversations.append({"role": "user", "content": text})
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                add_vision_id=False,
                return_dict=True,
            )
        logger.warning("Cosmos3 tokenizer has no chat template; falling back to raw tokenization.")
        if use_system_prompt:
            system_prompt = SYSTEM_PROMPT_IMAGE if is_image else SYSTEM_PROMPT_VIDEO
            text = f"{system_prompt} {text}".strip()
        return tokenizer(text, add_special_tokens=False, return_tensors=None)

    start_of_generation_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    eos_token_id = tokenizer.eos_token_id
    if start_of_generation_id is None or eos_token_id is None:
        raise ValueError("Cosmos3 tokenizer must define eos_token_id and <|vision_start|>.")

    def add_special_tokens(input_ids: list[int]) -> list[int]:
        return list(input_ids) + [eos_token_id, start_of_generation_id]

    cond = add_special_tokens(tokenize(apply_templates(prompt)).input_ids)
    uncond = add_special_tokens(tokenize(apply_templates(negative_prompt, is_negative=True)).input_ids)
    return cond, uncond


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]
    return int(value)


def _packing_special_tokens(transformer: nn.Module, input_ids: torch.Tensor) -> tuple[list[int], dict[str, int]]:
    ids = [int(x) for x in input_ids.detach().cpu().flatten().tolist()]
    if len(ids) < 2:
        raise ValueError("Cosmos3 text ids must include at least text plus generation suffix tokens.")

    config = transformer.config
    vlm_config = getattr(config, "vlm_config", None)
    text_config = getattr(vlm_config, "text_config", None)

    eos_token_id = _as_optional_int(getattr(text_config, "eos_token_id", None))
    start_of_generation_id = _as_optional_int(getattr(vlm_config, "vision_start_token_id", None))
    end_of_generation_id = _as_optional_int(getattr(vlm_config, "vision_end_token_id", None))
    if eos_token_id is None:
        eos_token_id = 151645
    if start_of_generation_id is None:
        start_of_generation_id = 151652
    if end_of_generation_id is None:
        end_of_generation_id = 151653

    if len(ids) >= 2 and ids[-2] == eos_token_id and ids[-1] == start_of_generation_id:
        ids = ids[:-2]

    special_tokens = {
        "eos_token_id": eos_token_id,
        "start_of_generation": start_of_generation_id,
        "end_of_generation": end_of_generation_id,
    }
    return ids, special_tokens


def build_packed_sequence(
    transformer: nn.Module,
    input_ids: torch.Tensor,
    vision_tokens: torch.Tensor,
    timestep: torch.Tensor,
    has_image_condition: bool,
    fps: float,
    device: torch.device | str,
    vae_scale_factor_temporal: int = 4,
    sound_tokens: Optional[torch.Tensor] = None,
    sound_condition_indexes: Optional[list[int]] = None,
    sound_fps: float = 25.0,
) -> PackedSequence:
    config = transformer.config
    input_ids = input_ids.to(device=device, dtype=torch.long).flatten()
    base_text_ids, special_tokens = _packing_special_tokens(transformer, input_ids)
    latent_patch_size = int(config.latent_patch_size)
    _, _, latent_t, _, _ = vision_tokens.shape
    has_sound = sound_tokens is not None
    sound_batch = None
    if sound_tokens is not None:
        if sound_tokens.dim() == 2:
            sound_batch = sound_tokens.unsqueeze(0)
        elif sound_tokens.dim() == 3:
            sound_batch = sound_tokens
        else:
            raise ValueError(f"sound_tokens must have shape C,T or B,C,T, got {tuple(sound_tokens.shape)}")

    packed = pack_input_sequence(
        sequence_plans=[
            SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[0] if has_image_condition else [],
                has_sound=has_sound,
                condition_frame_indexes_sound=sound_condition_indexes or [],
            )
        ],
        input_text_indexes=[base_text_ids],
        gen_data_clean=GenerationDataClean(
            batch_size=1,
            is_image_batch=latent_t == 1,
            x0_tokens_vision=[vision_tokens.to(device)],
            fps_vision=torch.tensor([fps], device=device, dtype=torch.float32),
            num_vision_items_per_sample=[1],
            x0_tokens_sound=sound_batch.to(device) if sound_batch is not None else None,
            fps_sound=torch.tensor([sound_fps], device=device, dtype=torch.float32) if has_sound else None,
        ),
        input_timesteps=torch.tensor([[float(timestep.detach().float().item())]], dtype=torch.float32),
        special_tokens=special_tokens,
        latent_patch_size=latent_patch_size,
        include_end_of_generation_token=False,
        position_embedding_type=getattr(config, "position_embedding_type", "unified_3d_mrope"),
        unified_3d_mrope_reset_spatial_ids=bool(getattr(config, "unified_3d_mrope_reset_spatial_ids", True)),
        unified_3d_mrope_temporal_modality_margin=int(
            getattr(config, "unified_3d_mrope_temporal_modality_margin", 15000)
        ),
        enable_fps_modulation=bool(getattr(config, "enable_fps_modulation", True)),
        base_fps=float(getattr(config, "base_fps", 24.0)),
        temporal_compression_factor=vae_scale_factor_temporal,
        video_temporal_causal=bool(getattr(config, "video_temporal_causal", False)),
        action_dim=int(getattr(config, "action_dim", 64)),
    )
    if torch.device(device).type == "cuda":
        packed.to_cuda()
    return packed

    text_len = int(input_ids.numel())

    text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
        num_tokens=text_len,
        temporal_offset=0,
        use_float_positions=bool(config.enable_fps_modulation),
    )

    latent_patch_size = int(config.latent_patch_size)
    _, _, latent_t, latent_h, latent_w = vision_tokens.shape
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    num_vision_tokens = latent_t * patch_h * patch_w

    noisy_start = 1 if has_image_condition else 0
    noisy_frame_indexes = torch.arange(noisy_start, latent_t, device=device, dtype=torch.long)
    frame_token_stride = patch_h * patch_w
    mse_loss_indexes: list[int] = []
    text_offset = text_len
    for frame_idx in range(noisy_start, latent_t):
        frame_start = text_offset + frame_idx * frame_token_stride
        mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))

    effective_fps = fps if bool(config.enable_fps_modulation) else None
    mrope_margin = int(getattr(config, "unified_3d_mrope_temporal_modality_margin", 15000))
    reset_spatial = bool(getattr(config, "unified_3d_mrope_reset_spatial_ids", True))
    vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=latent_t,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=next_mrope_offset + mrope_margin,
        reset_spatial_indices=reset_spatial,
        fps=effective_fps,
        base_fps=float(config.base_fps),
        temporal_compression_factor=vae_scale_factor_temporal,
    )

    num_noisy_patches = len(mse_loss_indexes)
    timestep_value = float(timestep.detach().float().item())
    vision = ModalityData(
        sequence_indexes=torch.arange(text_offset, text_offset + num_vision_tokens, dtype=torch.long, device=device),
        timesteps=torch.full((num_noisy_patches,), timestep_value, device=device, dtype=torch.float32),
        mse_loss_indexes=torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
        token_shapes=[(latent_t, patch_h, patch_w)],
        tokens=[vision_tokens.to(device)],
        condition_mask=[
            torch.cat(
                [
                    torch.ones((noisy_start, 1, 1), device=device, dtype=vision_tokens.dtype),
                    torch.zeros((latent_t - noisy_start, 1, 1), device=device, dtype=vision_tokens.dtype),
                ],
                dim=0,
            )
        ],
        noisy_frame_indexes=[noisy_frame_indexes],
    )

    sound = None
    num_sound_tokens = 0
    sound_position_ids = None
    if sound_tokens is not None:
        if sound_tokens.dim() == 3:
            if sound_tokens.shape[0] != 1:
                raise ValueError(f"Expected one sample of sound tokens, got shape {tuple(sound_tokens.shape)}")
            sound_tokens = sound_tokens[0]
        if sound_tokens.dim() != 2:
            raise ValueError(f"sound_tokens must have shape C,T or 1,C,T, got {tuple(sound_tokens.shape)}")

        sound_tokens = sound_tokens.to(device)
        _, sound_t = sound_tokens.shape
        sound_offset = text_offset + num_vision_tokens
        sound_condition_set = {
            idx for idx in (sound_condition_indexes or []) if 0 <= idx < sound_t
        }
        sound_noisy_indexes = torch.tensor(
            [idx for idx in range(sound_t) if idx not in sound_condition_set],
            dtype=torch.long,
            device=device,
        )
        sound_mse_loss_indexes = [
            sound_offset + frame_idx for frame_idx in range(sound_t) if frame_idx not in sound_condition_set
        ]
        sound_condition_mask = torch.zeros((sound_t, 1), device=device, dtype=sound_tokens.dtype)
        for frame_idx in sound_condition_set:
            sound_condition_mask[frame_idx, 0] = 1.0

        sound = ModalityData(
            sequence_indexes=torch.arange(sound_offset, sound_offset + sound_t, dtype=torch.long, device=device),
            timesteps=torch.full((len(sound_mse_loss_indexes),), timestep_value, device=device, dtype=torch.float32),
            mse_loss_indexes=torch.tensor(sound_mse_loss_indexes, dtype=torch.long, device=device),
            token_shapes=[(sound_t, 1, 1)],
            tokens=[sound_tokens],
            condition_mask=[sound_condition_mask],
            noisy_frame_indexes=[sound_noisy_indexes],
        )
        effective_sound_fps = sound_fps if bool(config.enable_fps_modulation) else None
        sound_position_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=sound_t,
            grid_h=1,
            grid_w=1,
            temporal_offset=next_mrope_offset + mrope_margin,
            reset_spatial_indices=reset_spatial,
            fps=effective_sound_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=1,
        )
        num_sound_tokens = sound_t

    position_ids = [text_mrope_ids, vision_mrope_ids]
    if sound_position_ids is not None:
        position_ids.append(sound_position_ids)
    generation_tokens = num_vision_tokens + num_sound_tokens

    return PackedSequence(
        sequence_length=text_len + generation_tokens,
        sample_lens=[text_len + generation_tokens],
        split_lens=[text_len, generation_tokens],
        attn_modes=["causal", "full"],
        is_image_batch=latent_t == 1,
        text_ids=input_ids,
        text_indexes=torch.arange(text_len, dtype=torch.long, device=device),
        position_ids=torch.cat(position_ids, dim=1).to(device),
        label_ids=None,
        ce_loss_indexes=None,
        ce_loss_weights=None,
        vision=vision,
        action=None,
        sound=sound,
    )


def run_transformer_for_sample(
    transformer: nn.Module,
    input_ids: torch.Tensor,
    vision_tokens: torch.Tensor,
    timestep: torch.Tensor,
    has_image_condition: bool,
    fps: float,
    device: torch.device,
    vae_scale_factor_temporal: int,
    sound_tokens: Optional[torch.Tensor] = None,
    sound_condition_indexes: Optional[list[int]] = None,
    sound_fps: float = 25.0,
    return_dict: bool = False,
) -> torch.Tensor | dict[str, Any]:
    packed = build_packed_sequence(
        transformer,
        input_ids=input_ids,
        vision_tokens=vision_tokens,
        timestep=timestep,
        has_image_condition=has_image_condition,
        fps=fps,
        device=device,
        vae_scale_factor_temporal=vae_scale_factor_temporal,
        sound_tokens=sound_tokens,
        sound_condition_indexes=sound_condition_indexes,
        sound_fps=sound_fps,
    )
    outputs = transformer(
        packed,
        fps_vision=torch.tensor([fps], device=device, dtype=torch.float32),
        fps_sound=torch.tensor([sound_fps], device=device, dtype=torch.float32) if sound_tokens is not None else None,
    )
    if return_dict:
        return outputs
    return outputs["preds_vision"][0]


def apply_i2v_conditioning(latents: torch.Tensor, noisy_model_input: torch.Tensor, noise: torch.Tensor):
    mask = torch.zeros((latents.shape[0], 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype)
    mask[:, :, 0] = 1.0
    noisy_model_input = mask * latents + (1.0 - mask) * noisy_model_input
    target = (noise - latents) * (1.0 - mask)
    return noisy_model_input, target


def find_single_file(directory_or_file: str, suffix: str = ".safetensors") -> str:
    if os.path.isfile(directory_or_file):
        return directory_or_file
    matches = []
    for root, _dirs, files in os.walk(directory_or_file):
        for filename in files:
            if filename.endswith(suffix):
                matches.append(os.path.join(root, filename))
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix} file under {directory_or_file}, found {len(matches)}.")
    return matches[0]
